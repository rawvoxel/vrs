use crate::{adapter::Adapter, alloc};
use ash::{prelude::VkResult, util, vk};

#[derive(Clone)]
pub struct Buffer {
    pub memory: vk::DeviceMemory,
    pub inner: vk::Buffer,
}

impl Buffer {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        memory_property_flags: vk::MemoryPropertyFlags,
        buffer_info: vk::BufferCreateInfo,
    ) -> VkResult<Self> {
        let (buffer, memory) = create_buffer(
            adapter,
            logical_device,
            buffer_info,
            memory_property_flags,
            0,
        )?;

        Ok(Self {
            inner: buffer,
            memory,
        })
    }

    #[rustfmt::skip]
    pub fn map_memory<T: Copy>(&self, logical_device: &ash::Device, offset: u64, data: &[T]) -> VkResult<()> {
        map_buffer_memory(logical_device, self.inner, self.memory, offset, data)
    }

    #[rustfmt::skip]
    pub fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.device_wait_idle().unwrap();
            logical_device.free_memory(self.memory, None);
            logical_device.destroy_buffer(self.inner, None);
        }
    }
}

#[inline]
#[rustfmt::skip]
fn create_buffer(
    adapter: &Adapter,
    logical_device: &ash::Device,
    buffer_info: vk::BufferCreateInfo,
    memory_property_flags: vk::MemoryPropertyFlags,
    offset: u64,
) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
    unsafe {
        let buffer = logical_device.create_buffer(&buffer_info, None)?;
        let memory_requirements = logical_device.get_buffer_memory_requirements(buffer);
        let memory = alloc::vk_alloc(adapter,logical_device, memory_property_flags, memory_requirements)?;

        logical_device.bind_buffer_memory(buffer, memory, offset)?;

        Ok((buffer, memory))
    }
}

#[inline]
#[rustfmt::skip]
fn map_buffer_memory<T: Copy>(
    logical_device: &ash::Device,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    offset: u64,
    data: &[T],
) -> VkResult<()> {
    unsafe {
        let memory_req = logical_device.get_buffer_memory_requirements(buffer);
        let buffer_ptr = logical_device.map_memory(
            buffer_memory,
            offset,
            memory_req.size,
            vk::MemoryMapFlags::empty(),
        )?;

        util::Align::new(buffer_ptr, align_of::<T>() as u64, memory_req.size).copy_from_slice(&data);

        logical_device.unmap_memory(buffer_memory);

        Ok(())
    }
}
