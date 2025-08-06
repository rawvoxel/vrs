use crate::{adapter::Adapter, alloc};
use ash::{prelude::VkResult, util, vk};
use std::ffi::c_void;

#[derive(Clone)]
pub struct Buffer {
    pub memory_req: vk::MemoryRequirements,
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
                memory_req,
            })
        }
    }

    pub fn mem_map(
        &self,
        logical_device: &ash::Device,
        buffer_offset: u64,
    ) -> VkResult<*mut c_void> {
        unsafe {
            logical_device.map_memory(
                self.memory,
                buffer_offset,
                self.memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
        }
    }

    pub fn mem_write<T: Copy>(&self, buffer_ptr: *mut c_void, data: &[T]) {
        unsafe {
            util::Align::new(buffer_ptr, align_of::<T>() as u64, self.memory_req.size)
                .copy_from_slice(&data);
        }
    }

    pub fn mem_unmap(&self, logical_device: &ash::Device) {
        unsafe { logical_device.unmap_memory(self.memory) }
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
