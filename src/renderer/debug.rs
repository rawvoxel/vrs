//! Hooks for tests and tools. They stall the device, so keep them out of normal frames.

use super::Renderer;
use crate::{buffer::Buffer, commands::record_submit_commandbuffer, mesh::MeshCapacity};
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

/// The mesh pool's size and how often it's been repacked.
#[derive(Clone, Copy, Debug)]
pub struct MeshPoolStats {
    pub capacity: MeshCapacity,
    /// Meshes with an id, uploaded or not.
    pub meshes: u32,
    /// Repacked at the same capacity, to join scattered free space.
    pub defrags: u32,
    /// Repacked into a larger capacity.
    pub growths: u32,
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
            drawn: read?,
        }))
    }

    /// For tests and tools, without stalling.
    pub fn debug_mesh_pool_stats(&self) -> MeshPoolStats {
        let repacks = self.mesh_pool.repack_counts();
        MeshPoolStats {
            capacity: self.mesh_pool.capacity,
            meshes: self.mesh_pool.mesh_count(),
            defrags: repacks.defrags,
            growths: repacks.growths,
        }
    }
}
