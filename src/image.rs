use crate::{alloc, context::Context, surface::SurfaceInfo};
use ash::{prelude::VkResult, vk};

pub struct Image {
    pub inner: vk::Image,
    pub memory: vk::DeviceMemory,
}

impl Image {
    pub fn new(
        context: &Context,
        memory_property_flags: vk::MemoryPropertyFlags,
        image_info: vk::ImageCreateInfo,
    ) -> VkResult<Self> {
        let (image, memory) = create_image(
            &context.vk_device,
            context.physical_device_memory_properties,
            memory_property_flags,
            image_info,
        )?;
        Ok(Self {
            inner: image,
            memory,
        })
    }

    pub fn destroy(&self, vk_device: &ash::Device) {
        unsafe {
            vk_device.device_wait_idle().unwrap();
            vk_device.free_memory(self.memory, None);
            vk_device.destroy_image(self.inner, None);
        }
    }
}

#[inline]
pub fn create_image(
    vk_device: &ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    memory_property_flags: vk::MemoryPropertyFlags,
    image_info: vk::ImageCreateInfo,
) -> VkResult<(vk::Image, vk::DeviceMemory)> {
    unsafe {
        let image = vk_device.create_image(&image_info, None)?;
        let memory_requirements = vk_device.get_image_memory_requirements(image);
        let memory = alloc::vk_alloc(
            &vk_device,
            memory_properties,
            memory_property_flags,
            memory_requirements,
        )?;
        vk_device.bind_image_memory(image, memory, 0)?;
        Ok((image, memory))
    }
}

#[inline]
pub fn create_present_image_views(
    context: &Context,
    surface_info: SurfaceInfo,
    images: &Vec<vk::Image>,
) -> VkResult<Vec<vk::ImageView>> {
    let image_views = images
        .iter()
        .map(|&image| {
            let image_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .format(surface_info.format.format)
                .view_type(vk::ImageViewType::TYPE_2D)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );

            unsafe {
                context
                    .vk_device
                    .create_image_view(&image_view_info, None)
                    .expect("Could not create vk::ImageView")
            }
        })
        .collect();

    Ok(image_views)
}

#[inline]
pub fn create_depth_image_view(
    context: &Context,
    image: vk::Image,
    image_info: vk::ImageCreateInfo,
) -> VkResult<vk::ImageView> {
    let image_view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .format(image_info.format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .level_count(1)
                .layer_count(1),
        );

    unsafe { context.vk_device.create_image_view(&image_view_info, None) }
}
