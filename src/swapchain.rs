use std::slice;

use crate::{
    adapter::Adapter,
    image::{Image, create_image_view},
    queue::QueueFamily,
    surface::Surface,
};
use ash::{khr, prelude::VkResult, vk};

#[derive(Clone)]
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,
    pub depth_image: Image,
    pub depth_image_view: vk::ImageView,
}

impl Swapchain {
    pub fn new(
        adapter: &Adapter,
        surface: &Surface,
        surface_extent: vk::Extent2D,
        queue_family: &QueueFamily,
        logical_device: &ash::Device,
        swapchain_device: &khr::swapchain::Device,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> VkResult<Self> {
        unsafe {
            let mut swapchain_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface.handle)
                .min_image_count(surface.min_image_count)
                .image_format(surface.format)
                .image_color_space(surface.color_space)
                .image_extent(surface_extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(slice::from_ref(&queue_family.index))
                .pre_transform(surface.transform_flags)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(surface.present_mode)
                .clipped(true);

            if let Some(old_swapchain) = old_swapchain {
                swapchain_info.old_swapchain = old_swapchain;
            }

            let swapchain = swapchain_device.create_swapchain(&swapchain_info, None)?;
            let present_images = swapchain_device.get_swapchain_images(swapchain)?;
            let present_image_views = present_images
                .iter()
                .map(|&image| {
                    create_image_view(
                        logical_device,
                        image,
                        surface.format,
                        vk::ImageAspectFlags::COLOR,
                        vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        },
                    )
                    .expect("Could not create vk::ImageView")
                })
                .collect();
            let depth_image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(surface_extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let depth_image = Image::new(
                adapter,
                logical_device,
                depth_image_info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let depth_image_view = create_image_view(
                logical_device,
                depth_image.handle,
                depth_image_info.format,
                vk::ImageAspectFlags::DEPTH,
                vk::ComponentMapping::default(),
            )?;

            Ok(Swapchain {
                handle: swapchain,
                extent: surface_extent,
                present_images,
                present_image_views,
                depth_image,
                depth_image_view,
            })
        }
    }

    pub fn rebuild(
        &mut self,
        adapter: &Adapter,
        surface: &Surface,
        surface_extent: vk::Extent2D,
        queue_family: &QueueFamily,
        logical_device: &ash::Device,
        swapchain_device: &khr::swapchain::Device,
    ) -> VkResult<()> {
        unsafe {
            logical_device.device_wait_idle().unwrap();

            let old_handle = self.handle;
            self.depth_image.destroy(logical_device)?;
            logical_device.destroy_image_view(self.depth_image_view, None);
            for &image_view in self.present_image_views.iter() {
                logical_device.destroy_image_view(image_view, None);
            }

            *self = Self::new(
                adapter,
                surface,
                surface_extent,
                queue_family,
                logical_device,
                swapchain_device,
                Some(self.handle),
            )?;

            swapchain_device.destroy_swapchain(old_handle, None);
            Ok(())
        }
    }

    #[rustfmt::skip]
    pub fn destroy(&self, logical_device: &ash::Device, swapchain_device: &khr::swapchain::Device) -> VkResult<()>{
        unsafe {
            logical_device.device_wait_idle().unwrap();

            self.depth_image.destroy(logical_device)?;
            logical_device.destroy_image_view(self.depth_image_view, None);
            for &image_view in self.present_image_views.iter() {
                logical_device.destroy_image_view(image_view, None);
            }

            swapchain_device.destroy_swapchain(self.handle, None);
            Ok(())
        }
    }
}
