//! The stages `Renderer::submit` runs a frame in, around recording it.

use super::{DrawList, Renderer};
use crate::uniforms::{FrameUniforms, MvpMatrices};
use ash::{prelude::VkResult, vk};
use glam::Mat4;
use std::slice;

/// What `Renderer::write_frame_data` prepared for recording.
pub(super) struct FrameData {
    /// `MeshPool::info_buffer` indices of the frame's meshes.
    pub candidates: Vec<u32>,
    pub dynamic_index_count: u32,
    pub view_proj: Mat4,
    /// Whether the light probes are updated and sampled.
    pub gi: bool,
    /// Whether the frame's depth is drawn first, for the shadow pass, see
    /// `Renderer::uses_shadow_pass`.
    pub shadow_pass: bool,
    /// Whether the light probes are drawn over the frame, see `DebugView::Probes`.
    pub draw_probes: bool,
}

impl FrameData {
    #[inline]
    pub(super) fn candidate_count(&self) -> u32 {
        self.candidates.len() as u32
    }
}

impl Renderer {
    /// Points `frame`'s descriptors at the mesh pool's buffers, which repacking replaces, its
    /// palette and shadow table, which growing replaces, and the depth buffer and shadow image,
    /// which resizing replaces. The uniforms and light probes never move.
    ///
    /// # Safety
    /// The device must not be using `frame`'s descriptor set.
    unsafe fn write_frame_descriptors(&self, frame: usize) {
        let buffer_infos = [
            self.mesh_pool.vertex_buffer.handle,
            self.mesh_pool.info_buffer.handle,
            self.mesh_pool.type_buffer.handle,
            self.frame_palette_buffers[frame].buffer.handle,
            self.frame_shadow_tables[frame].buffer.handle,
        ]
        .map(|buffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(vk::WHOLE_SIZE)
        });
        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, buffer_info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.frame_desc_sets[frame])
                    .dst_binding(1 + i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(buffer_info))
            })
            .collect();
        let shadow_storage = vk::DescriptorImageInfo::default()
            .image_view(self.shadow_pass.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let depth_sampled = vk::DescriptorImageInfo::default()
            .image_view(self.swapchain.depth_image_view)
            .sampler(self.shadow_pass.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let shadow_sampled = shadow_storage.sampler(self.shadow_pass.sampler);
        let late_buffers = [
            (16, self.frame_changed_chunks[frame].buffer.handle),
            (17, self.frame_light_buffers[frame].buffer.handle),
            (18, self.shadow_pass.tiles.handle),
        ]
        .map(|(binding, buffer)| {
            (
                binding,
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(vk::WHOLE_SIZE),
            )
        });
        let late_writes = late_buffers.each_ref().map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.frame_desc_sets[frame])
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(slice::from_ref(info))
        });
        let light_shadow = vk::DescriptorImageInfo::default()
            .image_view(self.shadow_pass.light_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let reflection = vk::DescriptorImageInfo::default()
            .image_view(self.shadow_pass.reflection_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let image_writes = [
            (13, vk::DescriptorType::STORAGE_IMAGE, &shadow_storage),
            (
                14,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &depth_sampled,
            ),
            (
                15,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &shadow_sampled,
            ),
            (19, vk::DescriptorType::STORAGE_IMAGE, &light_shadow),
            (20, vk::DescriptorType::SAMPLED_IMAGE, &light_shadow),
            (21, vk::DescriptorType::STORAGE_IMAGE, &reflection),
            (22, vk::DescriptorType::SAMPLED_IMAGE, &reflection),
        ]
        .map(|(binding, ty, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.frame_desc_sets[frame])
                .dst_binding(binding)
                .descriptor_type(ty)
                .image_info(slice::from_ref(info))
        });
        unsafe {
            self.render_device.update_descriptor_sets(&writes, &[]);
            self.render_device.update_descriptor_sets(&late_writes, &[]);
            self.render_device
                .update_descriptor_sets(&image_writes, &[]);
        }
    }

    /// Waits for the device to finish `frame`'s last use, freeing up what it used.
    pub(super) fn wait_for_frame(&mut self, frame: usize) -> VkResult<()> {
        unsafe {
            self.render_device
                .wait_for_fences(&[self.in_flight_frames[frame]], true, u64::MAX)?;
        }
        self.mesh_pool.release_frame(frame);
        Ok(())
    }

    /// Waits for the frame's last use to finish, then acquires a swapchain image.
    /// `None` when the swapchain was out of date, so the frame is skipped.
    ///
    /// Headless, each frame has an image of its own, which the wait freed up.
    pub(super) fn acquire_image(
        &mut self,
        frame: usize,
        surface_extent: vk::Extent2D,
    ) -> VkResult<Option<u32>> {
        let frame_in_flight = self.in_flight_frames[frame];
        self.wait_for_frame(frame)?;
        if self.is_headless() {
            unsafe { self.render_device.reset_fences(&[frame_in_flight])? };
            return Ok(Some(frame as u32));
        }
        unsafe {
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
        let lighting = unsafe {
            self.write_palette(frame)?;
            self.write_lights(frame)?;
            let lighting = self.write_lighting(frame, draws.view)?;
            self.write_frame_descriptors(frame);
            self.frame_index_buffers[frame].write(
                &self.adapter,
                &self.render_device,
                draws.indices,
            )?;
            self.frame_vertex_buffers[frame].write(
                &self.adapter,
                &self.render_device,
                draws.vertices,
            )?;
            lighting
        };

        let uniforms = FrameUniforms {
            matrices: MvpMatrices {
                model: Mat4::IDENTITY,
                view: draws.view,
                proj: draws.proj,
            },
            lighting,
            gi: self.gi_uniforms(draws.view),
            shadow: self.shadow_uniforms(draws.proj * draws.view),
        };
        self.uniform_buffers[frame].mem_write(slice::from_ref(&uniforms));

        Ok(FrameData {
            candidates: draws
                .meshes
                .iter()
                .map(|id| self.mesh_pool.info_index(id))
                .collect(),
            dynamic_index_count: draws.indices.len() as u32,
            view_proj: draws.proj * draws.view,
            gi: self.gi_enabled(),
            shadow_pass: self.uses_shadow_pass(),
            draw_probes: self.lighting.debug_view == super::DebugView::Probes && self.gi_enabled(),
        })
    }

    /// Submits the recorded frame and queues its image for presentation, unless headless.
    pub(super) fn submit_and_present(&mut self, frame: usize, image_index: u32) -> VkResult<()> {
        let cmd_buffer_submit_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(self.frame_cmd_buffers[frame]);
        if self.is_headless() {
            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(slice::from_ref(&cmd_buffer_submit_info));
            return unsafe {
                self.render_device.queue_submit2(
                    self.graphics_queue.handle,
                    slice::from_ref(&submit_info),
                    self.in_flight_frames[frame],
                )
            };
        }

        let image_rendered = self.rendered_images[image_index as usize];
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
}
