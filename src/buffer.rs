use crate::{adapter::Adapter, alloc};
use ash::{prelude::VkResult, util, vk};

#[derive(Clone)]
pub struct Buffer {
    pub memory: vk::DeviceMemory,
    pub handle: vk::Buffer,
}

impl Buffer {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        buffer_info: vk::BufferCreateInfo,
        memory_property_flags: vk::MemoryPropertyFlags,
        buffer_offset: u64,
    ) -> VkResult<Self> {
        unsafe {
            let buffer = logical_device.create_buffer(&buffer_info, None)?;
            let memory_req = logical_device.get_buffer_memory_requirements(buffer);
            let memory =
                alloc::vk_alloc(adapter, logical_device, memory_req, memory_property_flags)?;
            logical_device.bind_buffer_memory(buffer, memory, buffer_offset)?;

            Ok(Self {
                handle: buffer,
                memory,
            })
        }
    }

    pub fn map_memory<T: Copy>(
        &self,
        logical_device: &ash::Device,
        data: &[T],
        offset: u64,
    ) -> VkResult<()> {
        unsafe {
            let memory_req = logical_device.get_buffer_memory_requirements(self.handle);
            let buffer_ptr = logical_device.map_memory(
                self.memory,
                offset,
                memory_req.size,
                vk::MemoryMapFlags::empty(),
            )?;

            util::Align::new(buffer_ptr, align_of::<T>() as u64, memory_req.size)
                .copy_from_slice(&data);

            Ok(logical_device.unmap_memory(self.memory))
        }
    }

    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.free_memory(self.memory, None);
            logical_device.destroy_buffer(self.handle, None);
        }
    }
}

#[inline]
pub fn create_buffer_view(
    logical_device: &ash::Device,
    buffer: vk::Buffer,
    format: vk::Format,
    offset: vk::DeviceSize,
    range: vk::DeviceSize,
) -> VkResult<vk::BufferView> {
    let info = vk::BufferViewCreateInfo::default()
        .buffer(buffer)
        .format(format)
        .offset(offset)
        .range(range);

    unsafe { logical_device.create_buffer_view(&info, None) }
}
