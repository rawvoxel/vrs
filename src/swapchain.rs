use crate::{
    adapter::Adapter,
    image::{Image, create_image_view},
    queue::QueueFamily,
    surface::Surface,
};
use ash::{khr, prelude::VkResult, vk};
use std::slice;

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
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            handle: vk::SwapchainKHR::null(),
            extent: vk::Extent2D::default(),
            present_images: Vec::new(),
            present_image_views: Vec::new(),
            depth_image: Image::null(),
            depth_image_view: vk::ImageView::null(),
        }
    }

    pub fn new(
        adapter: &Adapter,
        surface: &Surface,
        extent: vk::Extent2D,
        queue_family: &QueueFamily,
        logical_device: &ash::Device,
        swapchain_device: &khr::swapchain::Device,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> VkResult<Self> {
        let mut this = Self::null();
        this.extent = extent;
        let created = this.create(
            adapter,
            surface,
            queue_family,
            logical_device,
            swapchain_device,
            old_swapchain,
        );
        if let Err(err) = created {
            unsafe { this.destroy(logical_device, swapchain_device) };
            return Err(err);
        }
        Ok(this)
    }

    /// Fills in a null swapchain, leaving whatever it created behind if it fails.
    fn create(
        &mut self,
        adapter: &Adapter,
        surface: &Surface,
        queue_family: &QueueFamily,
        logical_device: &ash::Device,
        swapchain_device: &khr::swapchain::Device,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> VkResult<()> {
        unsafe {
            let swapchain_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface.handle)
                .min_image_count(surface.image_count)
                .image_format(surface.format)
                .image_color_space(surface.color_space)
                .image_extent(self.extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(slice::from_ref(&queue_family.index))
                .pre_transform(surface.transform_flags)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(surface.present_mode)
                .clipped(true)
                .old_swapchain(old_swapchain.unwrap_or_default());

            self.handle = swapchain_device.create_swapchain(&swapchain_info, None)?;
            self.present_images = swapchain_device.get_swapchain_images(self.handle)?;
            for &image in self.present_images.iter() {
                let view = create_image_view(
                    logical_device,
                    image,
                    surface.format,
                    vk::ComponentMapping::default(),
                    vk::ImageAspectFlags::COLOR,
                )?;
                self.present_image_views.push(view);
            }

            let depth_image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(surface.depth_format)
                .extent(self.extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                // Sampled to build the depth pyramid for occlusion culling.
                .usage(
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            self.depth_image = Image::new(
                adapter,
                logical_device,
                depth_image_info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            self.depth_image_view = create_image_view(
                logical_device,
                self.depth_image.handle,
                depth_image_info.format,
                vk::ComponentMapping::default(),
                vk::ImageAspectFlags::DEPTH,
            )?;

            Ok(())
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
            logical_device.device_wait_idle()?;

            let old_swapchain = self.clone();

            *self = Self::new(
                adapter,
                surface,
                surface_extent,
                queue_family,
                logical_device,
                swapchain_device,
                Some(self.handle),
            )?;

            old_swapchain.destroy(logical_device, swapchain_device);
            Ok(())
        }
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy(
        &self,
        logical_device: &ash::Device,
        swapchain_device: &khr::swapchain::Device,
    ) {
        unsafe {
            logical_device.destroy_image_view(self.depth_image_view, None);
            self.depth_image.destroy(logical_device);
            for &image_view in self.present_image_views.iter() {
                logical_device.destroy_image_view(image_view, None);
            }

            swapchain_device.destroy_swapchain(self.handle, None);
        }
    }
}
