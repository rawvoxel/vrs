//! Recording a frame's command buffer.

use super::{Renderer, frame::FrameData};
use crate::{
    cull::{DRAW_STRIDE, Phase},
    image,
};
use ash::{prelude::VkResult, vk};
use std::slice;

impl Renderer {
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

            self.mesh_pool
                .cmd_upload(&self.render_device, frame, cmd_buffer);

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
                self.cmd_draw_dynamic(cmd_buffer, frame, data.dynamic_index_count);
            }
            self.render_device.cmd_end_rendering(cmd_buffer);

            if occlusion {
                self.cmd_build_depth_pyramid(cmd_buffer);
                self.cmd_cull(cmd_buffer, frame, data, Phase::Occlusion, compact);
                self.cmd_resume_attachments(cmd_buffer);

                // Pass 2, what phase 2 found newly visible
                self.cmd_begin_rendering(cmd_buffer, color_view, vk::AttachmentLoadOp::LOAD);
                self.cmd_bind_graphics(cmd_buffer, frame);
                self.cmd_draw_meshes(
                    cmd_buffer,
                    frame,
                    Phase::Occlusion,
                    candidate_count,
                    compact,
                );
                self.render_device.cmd_end_rendering(cmd_buffer);
            }

            let image = self.swapchain.present_images[image_index as usize];
            let mut last_use = (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            );
            if self.capture_requested {
                self.cmd_capture(cmd_buffer, frame, image)?;
                last_use = (
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags2::empty(),
                    vk::PipelineStageFlags2::COPY,
                );
            }
            // Headless, the next frame to use the image starts from `UNDEFINED`.
            if !self.is_headless() {
                let (layout, access, stage) = last_use;
                image::transition_layout(
                    &self.render_device,
                    image,
                    cmd_buffer,
                    layout,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    access,
                    vk::AccessFlags2::empty(),
                    stage,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    vk::ImageAspectFlags::COLOR,
                );
            }

            self.render_device.end_command_buffer(cmd_buffer)
        }
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
                color: vk::ClearColorValue {
                    float32: self.clear_color,
                },
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

    /// The pass's dynamic state and this frame's descriptors, which both graphics pipelines share.
    fn cmd_bind_graphics(&self, cmd_buffer: vk::CommandBuffer, frame: usize) {
        let extent = self.swapchain.extent;
        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .max_depth(1.0);
        unsafe {
            self.render_device
                .cmd_set_viewport(cmd_buffer, 0, slice::from_ref(&viewport));
            self.render_device
                .cmd_set_scissor(cmd_buffer, 0, slice::from_ref(&extent.into()));
            let cull_mode = match self.backface_culling {
                true => vk::CullModeFlags::BACK,
                false => vk::CullModeFlags::NONE,
            };
            self.render_device.cmd_set_cull_mode(cmd_buffer, cull_mode);
            self.render_device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline_layout,
                0,
                slice::from_ref(&self.frame_desc_sets[frame]),
                &[],
            );
        }
    }

    #[inline]
    fn cmd_bind_pipeline(&self, cmd_buffer: vk::CommandBuffer, pipeline: vk::Pipeline) {
        unsafe {
            self.render_device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            )
        };
    }

    /// Draws the frame's own geometry, from its host visible buffers.
    fn cmd_draw_dynamic(&self, cmd_buffer: vk::CommandBuffer, frame: usize, index_count: u32) {
        self.cmd_bind_pipeline(cmd_buffer, self.dynamic_pipeline);
        unsafe {
            self.render_device.cmd_bind_index_buffer(
                cmd_buffer,
                self.frame_index_buffers[frame].buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            self.render_device.cmd_bind_vertex_buffers(
                cmd_buffer,
                0,
                &[self.frame_vertex_buffers[frame].buffer.handle],
                &[0],
            );
            self.render_device
                .cmd_draw_indexed(cmd_buffer, index_count, 1, 0, 0, 0);
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
        // The mesh shaders pull vertices themselves, there's no vertex buffer.
        self.cmd_bind_pipeline(cmd_buffer, self.mesh_pipeline);
        unsafe {
            self.render_device.cmd_bind_index_buffer(
                cmd_buffer,
                self.mesh_pool.index_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
        }
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
