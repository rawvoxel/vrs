//! Hooks for tests and tools. They stall the device, so keep them out of normal frames.

use super::Renderer;
use crate::{buffer::Buffer, commands::record_submit_commandbuffer};
use ash::{prelude::VkResult, vk};

/// The last submitted frame, for `Renderer::debug_frame_stats`.
pub(super) struct LastFrame {
    pub frame: usize,
    pub meshes: u32,
    pub compact: bool,
}

/// How many of a frame's meshes were drawn, read back from the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStats {
    /// Meshes in the frame's `DrawList`.
    pub meshes: u32,
    /// Meshes each pass drew: what was visible last frame, then what occlusion culling
    /// found newly visible. Without occlusion culling there's only the first pass.
    ///
    /// `None` when the frame's draws weren't counted on the device, which needs the
    /// `drawIndirectCount` feature and at most `maxDrawIndirectCount` meshes.
    pub drawn: Option<[u32; 2]>,
}

impl FrameStats {
    /// Meshes drawn across both passes.
    #[inline]
    pub fn total_drawn(&self) -> Option<u32> {
        self.drawn.map(|[first, second]| first + second)
    }
}

impl Renderer {
    /// The last submitted frame's draw counts, `None` before the first frame.
    ///
    /// For tests and tools: waits for the device to go idle, then reads the counts back.
    pub fn debug_frame_stats(&mut self) -> VkResult<Option<FrameStats>> {
        let Some(last) = &self.last_frame else {
            return Ok(None);
        };
        let (frame, meshes, compact) = (last.frame, last.meshes, last.compact);
        // Culling didn't run, so the counts were never reset.
        if meshes == 0 {
            return Ok(Some(FrameStats {
                meshes,
                drawn: Some([0, 0]),
            }));
        }
        if !compact {
            return Ok(Some(FrameStats {
                meshes,
                drawn: None,
            }));
        }

        let size = size_of::<[u32; 2]>() as vk::DeviceSize;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut readback = Buffer::new(&self.adapter, &self.render_device, info, flags, 0)?;

        let read = readback.mem_map(&self.render_device).and_then(|()| unsafe {
            self.render_device.device_wait_idle()?;
            let count_buffer = self.culler.frame_count_buffers[frame].handle;
            record_submit_commandbuffer(
                &self.render_device,
                self.init_cmd_buffer,
                self.init_cmd_fence,
                self.graphics_queue.handle,
                &[],
                &[],
                &[],
                |cmd_buffer| {
                    let region = vk::BufferCopy::default().size(size);
                    self.render_device.cmd_copy_buffer(
                        cmd_buffer,
                        count_buffer,
                        readback.handle,
                        &[region],
                    );
                },
            )?;
            self.render_device
                .wait_for_fences(&[self.init_cmd_fence], true, u64::MAX)?;

            let mut drawn = [0u32; 2];
            readback.mem_read_at(0, &mut drawn);
            Ok(drawn)
        });
        unsafe { readback.destroy(&self.render_device) };

        Ok(Some(FrameStats {
            meshes,
            drawn: Some(read?),
        }))
    }

    /// Draws as if the device lacked features, to test the fallbacks on any device.
    ///
    /// `compact` off stops counting draws on the device, and `max_batch` limits the draws
    /// per indirect call, 1 being what a device without `multiDrawIndirect` does.
    /// Neither can enable more than the device supports.
    pub fn debug_limit_draw_paths(&mut self, compact: bool, max_batch: u32) {
        let (device_compact, device_batch) = self.device_draw_paths;
        self.culler.can_compact = compact && device_compact;
        self.indirect_draw_batch = max_batch.clamp(1, device_batch);
    }
}
