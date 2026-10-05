use crate::vulkan::{
    adapter::Adapter,
    image::{Image, create_image_view},
    queue::QueueFamily,
    surface::Surface,
};
use ash::{khr, prelude::VkResult, vk};
use std::slice;

/// The images frames are drawn into, with a depth buffer they share.
///
/// For an offscreen `Surface` there's no `VkSwapchainKHR`, the images are its own.
#[derive(Clone)]
pub struct Swapchain {
    /// Null when offscreen.
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) extent: vk::Extent2D,
    pub(crate) present_images: Vec<vk::Image>,
    pub(crate) present_image_views: Vec<vk::ImageView>,
    /// The images behind `present_images` when offscreen, otherwise empty.
    pub(crate) owned_images: Vec<Image>,
    pub(crate) depth_image: Image,
    pub(crate) depth_image_view: vk::ImageView,
}

impl Swapchain {
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            handle: vk::SwapchainKHR::null(),
            extent: vk::Extent2D::default(),
            present_images: Vec::new(),
            present_image_views: Vec::new(),
            owned_images: Vec::new(),
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

    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    #[inline]
    pub fn images(&self) -> &[vk::Image] {
        &self.present_images
    }

    #[inline]
    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.present_image_views
    }

    #[inline]
    pub fn depth_image(&self) -> &Image {
        &self.depth_image
    }

    #[inline]
    pub fn depth_image_view(&self) -> vk::ImageView {
        self.depth_image_view
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
            match surface.is_offscreen() {
                true => self.create_offscreen_images(adapter, surface, logical_device)?,
                false => {
                    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
                        .surface(surface.handle)
                        .min_image_count(surface.image_count)
                        .image_format(surface.format)
                        .image_color_space(surface.color_space)
                        .image_extent(self.extent)
                        .image_array_layers(1)
                        .image_usage(surface.image_usage)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .queue_family_indices(slice::from_ref(&queue_family.index))
                        .pre_transform(surface.transform_flags)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(surface.present_mode)
                        .clipped(true)
                        .old_swapchain(old_swapchain.unwrap_or_default());
                    self.handle = swapchain_device.create_swapchain(&swapchain_info, None)?;
                    self.present_images = swapchain_device.get_swapchain_images(self.handle)?;
                }
            }
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
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
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

    fn create_offscreen_images(
        &mut self,
        adapter: &Adapter,
        surface: &Surface,
        logical_device: &ash::Device,
    ) -> VkResult<()> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(surface.format)
            .extent(self.extent.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(surface.image_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        for _ in 0..surface.image_count {
            let image = Image::new(
                adapter,
                logical_device,
                image_info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            self.owned_images.push(image);
            self.present_images.push(image.handle);
        }
        Ok(())
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
            for image in self.owned_images.iter() {
                image.destroy(logical_device);
            }

            // A headless device never loads the swapchain functions.
            if self.handle != vk::SwapchainKHR::null() {
                swapchain_device.destroy_swapchain(self.handle, None);
            }
        }
    }
}
