//! Hooks for tests and tools. They stall the device, so keep them out of normal frames.

mod capture;
mod probes;

pub use capture::Capture;
pub(super) use capture::RecordedCapture;
pub use probes::ProbeRecord;

use super::Renderer;
use crate::vulkan::{buffer::Buffer, commands::record_submit_command_buffer};
use ash::{prelude::VkResult, vk};

/// The last submitted frame, for `Renderer::debug_frame_stats`.
pub(super) struct LastFrame {
    pub frame: usize,
    pub meshes: u32,
}

/// How many of a frame's meshes were drawn, read back from the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStats {
    /// Meshes in the frame's `DrawList`.
    pub meshes: u32,
    /// Meshes each pass drew: what was visible last frame, then what occlusion culling
    /// found newly visible. Without occlusion culling there's only the first pass.
    pub drawn: [u32; 2],
}

impl FrameStats {
    /// Meshes drawn across both passes.
    #[inline]
    pub fn total_drawn(&self) -> u32 {
        self.drawn[0] + self.drawn[1]
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
        let (frame, meshes) = (last.frame, last.meshes);
        // Culling didn't run, so the counts were never reset.
        if meshes == 0 {
            return Ok(Some(FrameStats {
                meshes,
                drawn: [0, 0],
            }));
        }

        let words = self.debug_read_words(self.culler.frame_count_buffers[frame].handle, 2)?;
        let drawn = [words[0], words[1]];
        Ok(Some(FrameStats { meshes, drawn }))
    }

    /// Bytes the last frame's chunk table took, written every frame, see `VoxelGrid`.
    pub fn debug_chunk_table_bytes(&self) -> usize {
        size_of_val(self.chunk_table.as_slice())
    }

    /// The first `count` words of device buffer `buffer`.
    ///
    /// For tests and tools: waits for the device to go idle, then copies them back.
    fn debug_read_words(&mut self, buffer: vk::Buffer, count: usize) -> VkResult<Vec<u32>> {
        let size = (count * size_of::<u32>()) as vk::DeviceSize;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut readback = Buffer::new(&self.adapter, &self.device, info, flags, 0)?;

        let read = readback.mem_map(&self.device).and_then(|()| unsafe {
            self.device.device_wait_idle()?;
            record_submit_command_buffer(
                &self.device,
                self.init_cmd_buffer,
                self.init_cmd_fence,
                self.graphics_queue.handle,
                &[],
                &[],
                &[],
                |cmd_buffer| {
                    let region = vk::BufferCopy::default().size(size);
                    self.device
                        .cmd_copy_buffer(cmd_buffer, buffer, readback.handle, &[region]);
                },
            )?;
            self.device
                .wait_for_fences(&[self.init_cmd_fence], true, u64::MAX)?;

            let mut words = vec![0u32; count];
            readback.mem_read_at(0, &mut words);
            Ok(words)
        });
        unsafe { readback.destroy(&self.device) };
        read
    }
}
