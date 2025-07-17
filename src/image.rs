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

#[inline]
pub fn create_present_image_views(
    surface: &Surface,
    logical_device: &ash::Device,
    images: &Vec<vk::Image>,
) -> VkResult<Vec<vk::ImageView>> {
    let image_views = images
        .iter()
        .map(|&image| {
            let image_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .format(surface.format.format)
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
                logical_device
                    .create_image_view(&image_view_info, None)
                    .expect("Could not create vk::ImageView")
            }
        })
        .collect();

    Ok(image_views)
}

#[inline]
pub fn create_depth_image_view(
    logical_device: &ash::Device,
    image: &Image,
    image_info: vk::ImageCreateInfo,
) -> VkResult<vk::ImageView> {
    let image_view_info = vk::ImageViewCreateInfo::default()
        .image(image.inner)
        .format(image_info.format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .level_count(1)
                .layer_count(1),
        );

    unsafe { logical_device.create_image_view(&image_view_info, None) }
}
