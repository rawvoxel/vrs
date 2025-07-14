use crate::{context::find_memorytype_index, surface::SurfaceInfo, sync};
use ash::{khr, prelude::VkResult, vk};

#[derive(Clone)]
pub struct PresentQueue {
    pub queue: vk::Queue,
    pub mode: vk::PresentModeKHR,
    pub present_semaphores: Vec<vk::Semaphore>,
    pub render_semaphores: Vec<vk::Semaphore>,
}

impl PresentQueue {
    pub fn new(
        vk_device: &ash::Device,
        surface_instance: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> VkResult<Self> {
        unsafe {
            let queue = vk_device.get_device_queue(queue_family_index, 0);
            let modes = surface_instance
                .get_physical_device_surface_present_modes(physical_device, surface)?;
            let mode = modes
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let present_semaphores = sync::create_semaphores(&vk_device, 2)?;
            let render_semaphores = sync::create_semaphores(&vk_device, 2)?;

            Ok(Self {
                queue,
                mode,
                present_semaphores,
                render_semaphores,
            })
        }
    }

    pub fn destroy(&self, vk_device: &ash::Device) {
        unsafe {
            vk_device.device_wait_idle().unwrap();
            for &sem in self.present_semaphores.iter() {
                vk_device.destroy_semaphore(sem, None);
            }
            for &sem in self.render_semaphores.iter() {
                vk_device.destroy_semaphore(sem, None);
            }
        }
    }
}

#[derive(Clone)]
pub struct Swapchain {
    pub inner: vk::SwapchainKHR,

    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,

    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_device_memory: vk::DeviceMemory,

    pub frames_in_flight: usize,
    pub current_frame: usize,
}

impl Swapchain {
    pub fn new(
        vk_device: &ash::Device,
        swapchain_device: &khr::swapchain::Device,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        surface_info: SurfaceInfo,
        present_mode: vk::PresentModeKHR,
        physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> VkResult<Self> {
        unsafe {
            let swapchain = create_swapchain(
                &swapchain_device,
                surface,
                surface_format,
                surface_info,
                present_mode,
                old_swapchain,
            )?;

            let present_images = swapchain_device.get_swapchain_images(swapchain)?;
            let present_image_views =
                create_present_image_views(&vk_device, &present_images, surface_format.format)?;

            let depth_image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(surface_info.extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let (depth_image, depth_image_device_memory) = create_image(
                &vk_device,
                physical_device_memory_properties,
                depth_image_info,
            )?;
            let depth_image_view =
                create_depth_image_view(&vk_device, &depth_image, depth_image_info.format)?;

            Ok(Swapchain {
                inner: swapchain,
                present_images,
                present_image_views,
                depth_image,
                depth_image_view,
                depth_image_device_memory,
                frames_in_flight: 2,
                current_frame: 0,
            })
        }
    }

    #[inline]
    pub fn advance_current_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
    }

    pub fn rebuild(&self, vk_device: &ash::Device, swapchain_device: &khr::swapchain::Device) {
        self.destroy(vk_device, swapchain_device);
    }

    pub fn destroy(&self, vk_device: &ash::Device, swapchain_device: &khr::swapchain::Device) {
        unsafe {
            vk_device.device_wait_idle().unwrap();
            for &image_view in self.present_image_views.iter() {
                vk_device.destroy_image_view(image_view, None);
            }
            vk_device.free_memory(self.depth_image_device_memory, None);
            vk_device.destroy_image(self.depth_image, None);
            vk_device.destroy_image_view(self.depth_image_view, None);
            swapchain_device.destroy_swapchain(self.inner, None);
        }
    }
}

#[inline]
pub fn create_swapchain(
    swapchain_device: &khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_info: SurfaceInfo,
    present_mode: vk::PresentModeKHR,
    old_swapchain: Option<vk::SwapchainKHR>,
) -> VkResult<vk::SwapchainKHR> {
    let mut swapchain_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(surface_info.extent)
        .pre_transform(surface_info.pre_transform)
        .min_image_count(surface_info.min_image_count)
        .present_mode(present_mode)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .clipped(true)
        .image_array_layers(1);

    if let Some(old_swapchain) = old_swapchain {
        swapchain_info.old_swapchain = old_swapchain;
    }

    unsafe { Ok(swapchain_device.create_swapchain(&swapchain_info, None)?) }
}

#[inline]
pub fn create_image(
    device: &ash::Device,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    image_info: vk::ImageCreateInfo,
) -> VkResult<(vk::Image, vk::DeviceMemory)> {
    unsafe {
        let image = device.create_image(&image_info, None)?;
        let memory_req = device.get_image_memory_requirements(image);
        let memory_index = find_memorytype_index(
            &memory_req,
            &physical_device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("Unable to find suitable memory index for image.");
        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_req.size)
            .memory_type_index(memory_index);
        let memory = device.allocate_memory(&memory_allocate_info, None)?;
        device.bind_image_memory(image, memory, 0)?;
        Ok((image, memory))
    }
}

#[inline]
pub fn create_present_image_views(
    device: &ash::Device,
    images: &Vec<vk::Image>,
    format: vk::Format,
) -> VkResult<Vec<vk::ImageView>> {
    let image_views = images
        .iter()
        .map(|&image| {
            let image_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .format(format)
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
                device
                    .create_image_view(&image_view_info, None)
                    .expect("Could not create vk::ImageView")
            }
        })
        .collect();

    Ok(image_views)
}

#[inline]
pub fn create_depth_image_view(
    device: &ash::Device,
    image: &vk::Image,
    format: vk::Format,
) -> VkResult<vk::ImageView> {
    let image_view_info = vk::ImageViewCreateInfo::default()
        .image(*image)
        .format(format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .level_count(1)
                .layer_count(1),
        );

    unsafe { Ok(device.create_image_view(&image_view_info, None)?) }
}
