//! The stages `Renderer::submit` runs a frame in, around recording it.

use super::{DrawList, Renderer};
use crate::renderer::shader::{bindings, uniforms::FrameUniforms};
use ash::{prelude::VkResult, vk};
use glam::Mat4;
use std::slice;

/// What `Renderer::write_frame_data` prepared for recording.
pub(super) struct FrameData {
    /// `MeshPool::info_buffer` indices of the frame's meshes.
    pub candidates: Vec<u32>,
    pub view_proj: Mat4,
    /// Whether the light probes are updated and sampled.
    pub probes: bool,
    /// Whether the frame's depth is drawn first, for the trace pass, see
    /// `Renderer::uses_trace_pass`.
    pub trace_pass: bool,
    /// Whether the light probes are drawn over the frame, see `DebugView::Probes`.
    pub probe_overlay: bool,
    /// Whether color is drawn with the light from everywhere but the sun the trace pass lit.
    pub upscaled_indirect: bool,
    /// Whether the `Lighting::sky` is drawn behind the meshes.
    pub sky: bool,
}

impl FrameData {
    #[inline]
    pub(super) fn candidate_count(&self) -> u32 {
        self.candidates.len() as u32
    }
}

impl Renderer {
    /// Points `frame`'s descriptors at the mesh pool's buffers, which repacking replaces, its
    /// palette, chunk table, changed chunks and lights, which growing replaces, and the depth
    /// buffer and the trace pass's images and tiles, which resizing replaces. The uniforms
    /// and light probes never move.
    ///
    /// # Safety
    /// The device must not be using `frame`'s descriptor set.
    unsafe fn write_frame_descriptors(&self, frame: usize) {
        let buffers = [
            (bindings::VERTEX_WORDS, self.mesh_pool.vertex_buffer.handle),
            (bindings::MESHES, self.mesh_pool.info_buffer.handle),
            (bindings::CHUNK_WORDS, self.mesh_pool.chunk_buffer.handle),
            (
                bindings::PALETTE,
                self.frame_palette_buffers[frame].buffer.handle,
            ),
            (
                bindings::CHUNK_TABLE,
                self.frame_chunk_tables[frame].buffer.handle,
            ),
            (
                bindings::CHANGED_CHUNKS,
                self.frame_changed_chunks[frame].buffer.handle,
            ),
            (
                bindings::LIGHTS,
                self.frame_light_buffers[frame].buffer.handle,
            ),
            (bindings::LIGHT_TILES, self.trace_pass.tiles.handle),
        ]
        .map(|(binding, buffer)| {
            let info = vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(vk::WHOLE_SIZE);
            (binding, info)
        });
        let pass = &self.trace_pass;
        let general = |view| {
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
        };
        let depth = vk::DescriptorImageInfo::default()
            .image_view(self.swapchain.depth_image_view)
            .sampler(pass.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let images = [
            (bindings::SHADOWS_STORAGE, general(pass.shadows.view)),
            (bindings::DEPTH_BUFFER, depth),
            (
                bindings::SHADOWS,
                general(pass.shadows.view).sampler(pass.sampler),
            ),
            (
                bindings::LIGHT_SHADOWS_STORAGE,
                general(pass.light_shadows.view),
            ),
            (bindings::LIGHT_SHADOWS, general(pass.light_shadows.view)),
            (
                bindings::REFLECTIONS_STORAGE,
                general(pass.reflections.view),
            ),
            (bindings::REFLECTIONS, general(pass.reflections.view)),
            (bindings::INDIRECT_STORAGE, general(pass.indirect.view)),
            (bindings::INDIRECT, general(pass.indirect.view)),
        ];
        let write = |binding: u32| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.frame_desc_sets[frame])
                .dst_binding(binding)
                .descriptor_type(bindings::TYPES_BY_BINDING[binding as usize])
        };
        let writes: Vec<vk::WriteDescriptorSet> = buffers
            .iter()
            .map(|(binding, info)| write(*binding).buffer_info(slice::from_ref(info)))
            .chain(
                images
                    .iter()
                    .map(|(binding, info)| write(*binding).image_info(slice::from_ref(info))),
            )
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Waits for the device to finish `frame`'s last use, freeing up what it used.
    pub(super) fn wait_for_frame(&mut self, frame: usize) -> VkResult<()> {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_frames[frame]], true, u64::MAX)?;
        }
        self.mesh_pool.release_frame(frame);
        let timings = self.frame_timestamps[frame].read(
            &self.device,
            self.adapter.properties.limits.timestamp_period,
            self.graphics_queue_family.properties.timestamp_valid_bits,
        );
        if timings.is_some() {
            self.gpu_timings = timings;
        }
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
            unsafe { self.device.reset_fences(&[frame_in_flight])? };
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
            self.device.reset_fences(&[frame_in_flight])?;
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
            lighting
        };

        let frame_size = self.swapchain.extent;
        let uniforms = FrameUniforms {
            view: draws.view,
            proj: draws.proj,
            camera_offset_from_clip: (draws.proj * without_translation(draws.view)).inverse(),
            camera_position: draws.view.inverse().w_axis.truncate().to_array(),
            padding: 0.0,
            frame_size: [frame_size.width, frame_size.height],
            uses_trace_pass: self.uses_trace_pass() as u32,
            traced_indirect: self.trace_pass_finds_indirect() as u32,
            lighting,
            probe_volume: self.probe_uniforms(draws.view),
            trace: self.trace_uniforms(),
        };
        self.uniform_buffers[frame].mem_write(slice::from_ref(&uniforms));

        Ok(FrameData {
            candidates: draws
                .meshes
                .iter()
                .map(|id| self.mesh_pool.info_index(id))
                .collect(),
            view_proj: draws.proj * draws.view,
            probes: self.probes_enabled(),
            trace_pass: self.uses_trace_pass(),
            probe_overlay: self.lighting.debug_view == super::DebugView::Probes
                && self.probes_enabled(),
            // The ambient occlusion's own view needs each pixel's.
            upscaled_indirect: self.trace_pass_finds_indirect()
                && self.lighting.debug_view != super::DebugView::AmbientOcclusion,
            sky: self.lighting.sky.is_some(),
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
                self.device.queue_submit2(
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
            self.device.queue_submit2(
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

/// `view` turning the world around the camera but not moving it, so the camera stays at the
/// origin.
fn without_translation(view: Mat4) -> Mat4 {
    Mat4 {
        w_axis: glam::Vec4::W,
        ..view
    }
}
