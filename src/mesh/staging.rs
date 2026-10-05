use crate::vulkan::{adapter::Adapter, buffer::DynamicBuffer};
use ash::{prelude::VkResult, vk};

/// Starting size of each frame's staging buffer, they grow to fit.
const INITIAL_BYTES: vk::DeviceSize = 64 * 1024;

/// Where mesh data waits for its frame to copy it into the mesh pool, one buffer per frame.
///
/// Data is written as soon as a mesh is created, so it's never copied on the host first.
/// A frame's buffer is written until that frame records its copies, then left alone
/// until its fence has been waited on.
pub(super) struct UploadStaging {
    frame_buffers: Vec<DynamicBuffer>,
    /// Bytes written to each frame's buffer.
    frame_lens: Vec<vk::DeviceSize>,
    /// Whether each frame's buffer has been recorded, so the device may still be reading it.
    frame_in_flight: Vec<bool>,
}

impl UploadStaging {
    /// Owns no buffers, destroying them does nothing.
    pub fn null() -> Self {
        Self {
            frame_buffers: Vec::new(),
            frame_lens: Vec::new(),
            frame_in_flight: Vec::new(),
        }
    }

    /// Fills in a null staging, leaving whatever it created behind if it fails.
    pub fn create(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        frames_in_flight: usize,
    ) -> VkResult<()> {
        for _ in 0..frames_in_flight {
            self.frame_buffers.push(DynamicBuffer::new(
                adapter,
                logical_device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                INITIAL_BYTES,
            )?);
            self.frame_lens.push(0);
            self.frame_in_flight.push(false);
        }
        Ok(())
    }

    /// Call once `frame`'s fence has been waited on, before writing to it.
    pub fn begin_writes(&mut self, frame: usize) {
        if self.frame_in_flight[frame] {
            self.frame_in_flight[frame] = false;
            self.frame_lens[frame] = 0;
        }
    }

    /// Appends `pieces` to `frame`'s buffer one after another, growing it while keeping
    /// what's already there. Returns where they start.
    ///
    /// # Safety
    /// `begin_writes` must have been called for `frame` since it was last recorded.
    pub unsafe fn push(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        frame: usize,
        pieces: &[&[u8]],
    ) -> VkResult<vk::DeviceSize> {
        debug_assert!(
            !self.frame_in_flight[frame],
            "Staging written while in flight."
        );
        let start = self.frame_lens[frame];
        let len: usize = pieces.iter().map(|piece| piece.len()).sum();
        let end = start + len as vk::DeviceSize;
        let buffer = &mut self.frame_buffers[frame];
        unsafe { buffer.reserve_preserving(adapter, logical_device, end, start)? };
        let mut offset = start;
        for piece in pieces {
            buffer.buffer.mem_write_at(offset, piece);
            offset += piece.len() as vk::DeviceSize;
        }
        self.frame_lens[frame] = end;
        Ok(start)
    }

    /// The buffer copies read `frame`'s data from.
    #[inline]
    pub fn buffer(&self, frame: usize) -> vk::Buffer {
        self.frame_buffers[frame].buffer.handle
    }

    /// Call once `frame`'s copies are recorded, the device owns its buffer until its fence.
    #[inline]
    pub fn mark_in_flight(&mut self, frame: usize) {
        self.frame_in_flight[frame] = true;
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        for buffer in self.frame_buffers.iter() {
            unsafe { buffer.destroy(logical_device) };
        }
    }
}
