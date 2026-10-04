//! Rendering a frame, in the stages `Renderer::submit` runs them.

use super::{DrawList, Renderer};
use crate::{
    cull::{DRAW_STRIDE, Phase},
    image,
    uniforms::MvpMatrices,
};
use ash::{prelude::VkResult, vk};
use glam::Mat4;
use std::slice;

/// What `Renderer::write_frame_data` prepared for recording.
pub(super) struct FrameData {
    /// `MeshPool::info_buffer` indices of the frame's meshes.
    pub candidates: Vec<u32>,
    pub dynamic_index_count: u32,
    pub view_proj: Mat4,
}

impl FrameData {
    #[inline]
    fn candidate_count(&self) -> u32 {
        self.candidates.len() as u32
    }
}

impl Renderer {
    /// Waits for the frame's last use to finish, then acquires a swapchain image.
    /// `None` when the swapchain was out of date, so the frame is skipped.
    pub(super) fn acquire_image(
        &mut self,
        frame: usize,
        surface_extent: vk::Extent2D,
    ) -> VkResult<Option<u32>> {
        let frame_in_flight = self.in_flight_frames[frame];
        unsafe {
            self.render_device
                .wait_for_fences(&[frame_in_flight], true, u64::MAX)?;
            self.mesh_pool.release_frame(frame);

            // NOTE: A suboptimal image can still be presented, so it's rebuilt next frame.
            let image_index = match self.swapchain_device.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.available_frames[frame],
                vk::Fence::null(),
            ) {
                Ok((image_index, suboptimal)) => {
                    self.swapchain_outdated |= suboptimal;
                    image_index
                }
                // Nothing was acquired so skip this frame, its fence is still signaled.
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.rebuild_swapchain(surface_extent)?;
                    return Ok(None);
                }
                Err(err) => return Err(err),
            };

            // Only reset once work that signals the fence is certain to be submitted.
            self.render_device.reset_fences(&[frame_in_flight])?;
            Ok(Some(image_index))
        }
    }

    /// Writes the frame's host visible buffers, which the fence wait freed up.
    pub(super) fn write_frame_data(
        &mut self,
        frame: usize,
        draws: &DrawList,
    ) -> VkResult<FrameData> {
        unsafe {
            self.frame_index_buffers[frame].write(&self.adapter, &self.render_device, draws.indices)?;
            self.frame_vertex_buffers[frame].write(&self.adapter, &self.render_device, draws.vertices)?;
        }

        let mvp = MvpMatrices {
            model: Mat4::IDENTITY,
            view: draws.view,
            proj: draws.proj,
        };
        self.uniform_buffers[frame].mem_write(slice::from_ref(&mvp));

        Ok(FrameData {
            candidates: draws
                .meshes
                .iter()
                .map(|id| self.mesh_pool.info_index(id))
                .collect(),
            dynamic_index_count: draws.indices.len() as u32,
            view_proj: draws.proj * draws.view,
        })
    }

    /// Records the frame's command buffer.
    ///
    /// With occlusion culling: phase 1 culling, pass 1, the depth pyramid,
    /// phase 2 culling and pass 2. Without, frustum culling and one pass.
    pub(super) fn record_frame(
        &mut self,
        frame: usize,
        image_index: u32,
        data: &FrameData,
    ) -> VkResult<()> {
        let cmd_buffer = self.frame_cmd_buffers[frame];
        let candidate_count = data.candidate_count();
        let culling = candidate_count > 0;
        let occlusion = culling && self.occlusion_culling;
        let compact = self.compacts(candidate_count);
        let first_phase = match occlusion {
            true => Phase::LastVisible,
            false => Phase::Frustum,
        };
        let color_view = self.swapchain.present_image_views[image_index as usize];

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.render_device
                .begin_command_buffer(cmd_buffer, &begin_info)?;

            self.mesh_pool.cmd_upload(
                &self.adapter,
                &self.render_device,
                &mut self.frame_staging_buffers[frame],
                cmd_buffer,
            )?;

            if culling {
                self.culler.prepare(
                    &self.adapter,
                    &self.render_device,
                    &self.mesh_pool,
                    &self.depth_pyramid,
                    frame,
                    cmd_buffer,
                    &data.candidates,
                )?;
                self.cmd_cull(cmd_buffer, frame, data, first_phase, compact);
            }

            self.cmd_begin_attachments(cmd_buffer, image_index);

            // Pass 1, with all dynamic geometry
            self.cmd_begin_rendering(cmd_buffer, color_view, vk::AttachmentLoadOp::CLEAR);
            self.cmd_bind_graphics(cmd_buffer, frame);
            if culling {
                self.cmd_draw_meshes(cmd_buffer, frame, first_phase, candidate_count, compact);
            }
            if data.dynamic_index_count > 0 {
                self.cmd_bind_geometry(
                    cmd_buffer,
                    self.frame_index_buffers[frame].buffer.handle,
                    self.frame_vertex_buffers[frame].buffer.handle,
                );
                self.render_device
                    .cmd_draw_indexed(cmd_buffer, data.dynamic_index_count, 1, 0, 0, 0);
            }
            self.render_device.cmd_end_rendering(cmd_buffer);

            if occlusion {
                self.cmd_build_depth_pyramid(cmd_buffer);
                self.cmd_cull(cmd_buffer, frame, data, Phase::Occlusion, compact);
                self.cmd_resume_attachments(cmd_buffer);

                // Pass 2, what phase 2 found newly visible
                self.cmd_begin_rendering(cmd_buffer, color_view, vk::AttachmentLoadOp::LOAD);
                self.cmd_bind_graphics(cmd_buffer, frame);
                self.cmd_draw_meshes(cmd_buffer, frame, Phase::Occlusion, candidate_count, compact);
                self.render_device.cmd_end_rendering(cmd_buffer);
            }

            image::transition_layout(
                &self.render_device,
                self.swapchain.present_images[image_index as usize],
                cmd_buffer,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::ImageAspectFlags::COLOR,
            );

            self.render_device.end_command_buffer(cmd_buffer)
        }
    }

    /// Submits the recorded frame and queues its image for presentation.
    pub(super) fn submit_and_present(&mut self, frame: usize, image_index: u32) -> VkResult<()> {
        let image_rendered = self.rendered_images[image_index as usize];
        let cmd_buffer_submit_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(self.frame_cmd_buffers[frame]);
        let wait_sem_submit_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.available_frames[frame])
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        let signal_sem_submit_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(image_rendered)
            // Includes the transition to `PRESENT_SRC_KHR`, which finishes after color output.
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(slice::from_ref(&cmd_buffer_submit_info))
            .wait_semaphore_infos(slice::from_ref(&wait_sem_submit_info))
            .signal_semaphore_infos(slice::from_ref(&signal_sem_submit_info));

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&image_rendered))
            .swapchains(slice::from_ref(&self.swapchain.handle))
            .image_indices(slice::from_ref(&image_index));

        unsafe {
            self.render_device.queue_submit2(
                self.graphics_queue.handle,
                slice::from_ref(&submit_info),
                self.in_flight_frames[frame],
            )?;
            match self
                .swapchain_device
                .queue_present(self.graphics_queue.handle, &present_info)
            {
                Ok(suboptimal) => self.swapchain_outdated |= suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.swapchain_outdated = true,
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    #[inline]
    fn cmd_cull(
        &self,
        cmd_buffer: vk::CommandBuffer,
        frame: usize,
        data: &FrameData,
        phase: Phase,
        compact: bool,
    ) {
        unsafe {
            self.culler.cmd_cull(
                &self.render_device,
                &self.depth_pyramid,
                frame,
                cmd_buffer,
                data.candidate_count(),
                data.view_proj,
                phase,
                compact,
            )
        };
    }

    /// Readies the swapchain image and the depth buffer for pass 1, which clears both.
    fn cmd_begin_attachments(&self, cmd_buffer: vk::CommandBuffer, image_index: u32) {
        image::transition_layout(
            &self.render_device,
            self.swapchain.present_images[image_index as usize],
            cmd_buffer,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::empty(),
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            // Chains onto the acquire semaphore wait, which happens at this stage.
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::ImageAspectFlags::COLOR,
        );
        // Every frame shares the depth buffer, last frame may still be using it.
        image::transition_layout(
            &self.render_device,
            self.swapchain.depth_image.handle,
            cmd_buffer,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::ImageAspectFlags::DEPTH,
        );
    }

    /// Builds the depth pyramid from pass 1's depth.
    fn cmd_build_depth_pyramid(&self, cmd_buffer: vk::CommandBuffer) {
        image::transition_layout(
            &self.render_device,
            self.swapchain.depth_image.handle,
            cmd_buffer,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::ImageAspectFlags::DEPTH,
        );
        unsafe {
            self.depth_pyramid
                .cmd_build(&self.render_device, cmd_buffer)
        };
    }

    /// Readies pass 1's attachments for pass 2, which depth tests against
    /// pass 1's depth and draws over its color.
    fn cmd_resume_attachments(&self, cmd_buffer: vk::CommandBuffer) {
        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image(self.swapchain.depth_image.handle)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1),
            );
        let color_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
        let dependency_info = vk::DependencyInfo::default()
            .memory_barriers(slice::from_ref(&color_barrier))
            .image_memory_barriers(slice::from_ref(&depth_barrier));
        unsafe {
            self.render_device
                .cmd_pipeline_barrier2(cmd_buffer, &dependency_info)
        };
    }

    /// Clears or keeps the frame's color and depth, depth is always stored for later passes.
    fn cmd_begin_rendering(
        &self,
        cmd_buffer: vk::CommandBuffer,
        color_view: vk::ImageView,
        load_op: vk::AttachmentLoadOp,
    ) {
        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(color_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.1; 4] },
            });
        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.depth_image_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.swapchain.extent.into())
            .color_attachments(slice::from_ref(&color_attachment_info))
            .depth_attachment(&depth_attachment_info)
            .layer_count(1);
        unsafe {
            self.render_device
                .cmd_begin_rendering(cmd_buffer, &rendering_info)
        };
    }

    /// The graphics pipeline, its dynamic state and this frame's uniforms.
    fn cmd_bind_graphics(&self, cmd_buffer: vk::CommandBuffer, frame: usize) {
        let extent = self.swapchain.extent;
        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .max_depth(1.0);
        unsafe {
            self.render_device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
            self.render_device
                .cmd_set_viewport(cmd_buffer, 0, slice::from_ref(&viewport));
            self.render_device
                .cmd_set_scissor(cmd_buffer, 0, slice::from_ref(&extent.into()));
            self.render_device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline_layout,
                0,
                slice::from_ref(&self.ubo_desc_sets[frame]),
                &[],
            );
        }
    }

    #[inline]
    fn cmd_bind_geometry(
        &self,
        cmd_buffer: vk::CommandBuffer,
        index_buffer: vk::Buffer,
        vertex_buffer: vk::Buffer,
    ) {
        unsafe {
            self.render_device
                .cmd_bind_index_buffer(cmd_buffer, index_buffer, 0, vk::IndexType::UINT32);
            self.render_device
                .cmd_bind_vertex_buffers(cmd_buffer, 0, &[vertex_buffer], &[0]);
        }
    }

    /// Draws the meshes `phase` found visible, which all share the mesh pool's buffers.
    fn cmd_draw_meshes(
        &self,
        cmd_buffer: vk::CommandBuffer,
        frame: usize,
        phase: Phase,
        candidate_count: u32,
        compact: bool,
    ) {
        self.cmd_bind_geometry(
            cmd_buffer,
            self.mesh_pool.index_buffer.handle,
            self.mesh_pool.vertex_buffer.handle,
        );
        let draw_buffer = self.culler.frame_draw_buffers[frame].buffer.handle;
        let region = phase.draw_region();
        let first_draw_offset =
            region as vk::DeviceSize * candidate_count as vk::DeviceSize * DRAW_STRIDE;
        let stride = DRAW_STRIDE as u32;
        unsafe {
            if compact {
                self.render_device.cmd_draw_indexed_indirect_count(
                    cmd_buffer,
                    draw_buffer,
                    first_draw_offset,
                    self.culler.frame_count_buffers[frame].handle,
                    region as vk::DeviceSize * size_of::<u32>() as vk::DeviceSize,
                    candidate_count,
                    stride,
                );
            } else {
                // Culled meshes still have a draw, of zero instances.
                for first_draw in (0..candidate_count).step_by(self.indirect_draw_batch as usize) {
                    self.render_device.cmd_draw_indexed_indirect(
                        cmd_buffer,
                        draw_buffer,
                        first_draw_offset + first_draw as vk::DeviceSize * DRAW_STRIDE,
                        self.indirect_draw_batch.min(candidate_count - first_draw),
                        stride,
                    );
                }
            }
        }
    }
}
