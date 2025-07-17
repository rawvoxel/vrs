use crate::{
    adapter::Adapter,
    image::{self, Image},
    renderer::Renderer,
    surface::Surface,
};
use ash::{khr, prelude::VkResult, vk};

#[derive(Clone)]
pub struct Swapchain {
    pub inner: vk::SwapchainKHR,

    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,

    pub depth_image: Image,
    pub depth_image_view: vk::ImageView,

    pub frames_in_flight: usize,
    pub current_frame: usize,
}

impl Swapchain {
    pub fn new(
        adapter: &Adapter,
        surface: &Surface,
        logical_device: &ash::Device,
        swapchain_device: &khr::swapchain::Device,
        present_mode: vk::PresentModeKHR,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> VkResult<Self> {
        unsafe {
            let swapchain =
                create_swapchain(swapchain_device, surface, present_mode, old_swapchain)?;

            let present_images = swapchain_device.get_swapchain_images(swapchain)?;
            let present_image_views =
                image::create_present_image_views(surface, logical_device, &present_images)?;

            let depth_image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(surface.extent.into())
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
            let depth_image_view =
                image::create_depth_image_view(logical_device, &depth_image, depth_image_info)?;

            Ok(Swapchain {
                inner: swapchain,
                present_images,
                present_image_views,
                depth_image,
                depth_image_view,
                frames_in_flight: 2,
                current_frame: 0,
            })
        }
    }

    #[inline]
    pub fn advance_current_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
    }

    #[rustfmt::skip]
    pub fn destroy(&self, renderer: &Renderer) {
        unsafe {
            renderer.render_device.device_wait_idle().unwrap();
            self.depth_image.destroy(&renderer.render_device);
            for &image_view in self.present_image_views.iter() {
                renderer.render_device.destroy_image_view(image_view, None);
            }
            renderer.render_device.destroy_image_view(self.depth_image_view, None);
            renderer.swapchain_device.destroy_swapchain(self.inner, None);
        }
    }
}

#[inline]
pub fn create_swapchain(
    swapchain_device: &khr::swapchain::Device,
    surface: &Surface,
    present_mode: vk::PresentModeKHR,
    old_swapchain: Option<vk::SwapchainKHR>,
) -> VkResult<vk::SwapchainKHR> {
    let mut swapchain_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface.inner)
        .image_format(surface.format.format)
        .image_color_space(surface.format.color_space)
        .image_extent(surface.extent)
        .pre_transform(surface.transform_flags)
        .min_image_count(surface.min_image_count)
        .present_mode(present_mode)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .clipped(true)
        .image_array_layers(1);

    if let Some(old_swapchain) = old_swapchain {
        swapchain_info.old_swapchain = old_swapchain;
    }

    unsafe { swapchain_device.create_swapchain(&swapchain_info, None) }
}
