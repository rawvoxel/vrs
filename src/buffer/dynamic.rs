use super::Buffer;
use crate::adapter::Adapter;
use ash::{prelude::VkResult, vk};

/// A buffer which grows whenever it's too small. By default a persistently mapped
/// `HOST_VISIBLE` buffer the device reads directly.
pub struct DynamicBuffer {
    pub buffer: Buffer,
    pub usage: vk::BufferUsageFlags,
    pub memory_flags: vk::MemoryPropertyFlags,
}

impl DynamicBuffer {
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            buffer: Buffer::null(),
            usage: vk::BufferUsageFlags::empty(),
            memory_flags: vk::MemoryPropertyFlags::empty(),
        }
    }

    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        usage: vk::BufferUsageFlags,
        capacity: vk::DeviceSize,
    ) -> VkResult<Self> {
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        Self::with_memory(adapter, logical_device, usage, flags, capacity)
    }

    /// Only `HOST_VISIBLE` memory is mapped, and can be written.
    pub fn with_memory(
        adapter: &Adapter,
        logical_device: &ash::Device,
        usage: vk::BufferUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
        capacity: vk::DeviceSize,
    ) -> VkResult<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut buffer = Buffer::new(adapter, logical_device, info, memory_flags, 0)?;
        if memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && let Err(err) = buffer.mem_map(logical_device)
        {
            unsafe { buffer.destroy(logical_device) };
            return Err(err);
        }

        Ok(Self {
            buffer,
            usage,
            memory_flags,
        })
    }

    #[inline]
    pub fn capacity(&self) -> vk::DeviceSize {
        self.buffer.size
    }

    /// Writes `data` to the start of the buffer, first growing it to the next power of two when it doesn't fit.
    ///
    /// # Safety
    /// The device must not be using the buffer, and a grown buffer has a new handle.
    pub unsafe fn write<T: Copy>(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        data: &[T],
    ) -> VkResult<()> {
        unsafe { self.reserve(adapter, logical_device, size_of_val(data) as vk::DeviceSize)? };
        self.buffer.mem_write(data);
        Ok(())
    }

    /// Grows the buffer to the next power of two when it's smaller than `byte_count`, discarding its contents.
    ///
    /// # Safety
    /// The device must not be using the buffer, and a grown buffer has a new handle.
    pub unsafe fn reserve(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        byte_count: vk::DeviceSize,
    ) -> VkResult<()> {
        if byte_count > self.capacity() {
            let grown = Self::with_memory(
                adapter,
                logical_device,
                self.usage,
                self.memory_flags,
                byte_count.next_power_of_two(),
            )?;
            unsafe { std::mem::replace(self, grown).destroy(logical_device) };
        }
        Ok(())
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe { self.buffer.destroy(logical_device) }
    }
}
