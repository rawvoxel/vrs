use crate::{adapter::Adapter, alloc};
use ash::{prelude::VkResult, vk};

#[derive(Clone, Copy)]
pub struct Image {
    pub memory: vk::DeviceMemory,
    pub handle: vk::Image,
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
                handle: image,
                memory,
            })
        }
    }

    pub fn destroy(&self, logical_device: &ash::Device) -> VkResult<()> {
        unsafe {
            logical_device.device_wait_idle()?;
            logical_device.free_memory(self.memory, None);
            logical_device.destroy_image(self.handle, None);

            Ok(())
        }
    }
}

#[inline]
pub fn create_image_view(
    logical_device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect_mask: vk::ImageAspectFlags,
    components: vk::ComponentMapping,
) -> VkResult<vk::ImageView> {
    let image_view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .format(format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .components(components)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .level_count(1)
                .layer_count(1),
        );

    unsafe { logical_device.create_image_view(&image_view_info, None) }
}
