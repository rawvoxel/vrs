use crate::alloc;
use ash::{prelude::VkResult, util, vk};

#[derive(Clone)]
pub struct Buffer {
    pub inner: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

impl Buffer {
    pub fn new(
        vk_device: &ash::Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        memory_property_flags: vk::MemoryPropertyFlags,
        buffer_info: vk::BufferCreateInfo,
    ) -> VkResult<Self> {
        let (buffer, memory) = create_buffer(
            vk_device,
            buffer_info,
            memory_properties,
            memory_property_flags,
            0,
        )?;

        Ok(Self {
            inner: buffer,
            memory,
        })
    }

    #[rustfmt::skip]
    pub fn map_memory<T: Copy>(&self, vk_device: &ash::Device, offset: u64, data: &[T]) -> VkResult<()> {
        map_buffer_memory(vk_device, self.inner, self.memory, offset, data)
    }

    #[rustfmt::skip]
    pub fn destroy(&self, vk_device: &ash::Device) {
        unsafe {
            vk_device.device_wait_idle().unwrap();
            vk_device.free_memory(self.memory, None);
            vk_device.destroy_buffer(self.inner, None);
        }
    }
}

#[inline]
fn create_buffer(
    vk_device: &ash::Device,
    buffer_info: vk::BufferCreateInfo,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    memory_property_flags: vk::MemoryPropertyFlags,
    offset: u64,
) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
    unsafe {
        let buffer = vk_device.create_buffer(&buffer_info, None)?;
        let memory_requirements = vk_device.get_buffer_memory_requirements(buffer);
        let memory = alloc::vk_alloc(
            &vk_device,
            memory_properties,
            memory_property_flags,
            memory_requirements,
        )?;
        vk_device.bind_buffer_memory(buffer, memory, offset)?;
        Ok((buffer, memory))
    }
}

#[inline]
fn map_buffer_memory<T: Copy>(
    vk_device: &ash::Device,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    offset: u64,
    data: &[T],
) -> VkResult<()> {
    unsafe {
        let memory_req = vk_device.get_buffer_memory_requirements(buffer);
        let buffer_ptr = vk_device.map_memory(
            buffer_memory,
            offset,
            memory_req.size,
            vk::MemoryMapFlags::empty(),
        )?;

        util::Align::new(buffer_ptr, align_of::<T>() as u64, memory_req.size)
            .copy_from_slice(&data);

        vk_device.unmap_memory(buffer_memory);

        Ok(())
    }
}
