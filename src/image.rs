use crate::{adapter::Adapter, alloc, surface::Surface};
use ash::{prelude::VkResult, vk};

#[derive(Clone)]
pub struct Image {
    pub memory: vk::DeviceMemory,
    pub inner: vk::Image,
}

impl Image {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        image_info: vk::ImageCreateInfo,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> VkResult<Self> {
        unsafe {
            let image = logical_device.create_image(&image_info, None)?;
            let memory_requirements = logical_device.get_image_memory_requirements(image);
            let memory = alloc::vk_alloc(
                adapter,
                logical_device,
                memory_property_flags,
                memory_requirements,
            )?;
            logical_device.bind_image_memory(image, memory, 0)?;

            Ok(Self {
                inner: image,
                memory,
            })
        }
    }

    pub fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.device_wait_idle().unwrap();
            logical_device.free_memory(self.memory, None);
            logical_device.destroy_image(self.inner, None);
        }
    }
}
