use anyhow::Result;
use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    prelude::VkResult,
    vk,
};
use ash_window;
use std::{
    borrow::Cow,
    ffi::{self, c_char},
};
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

#[derive(Clone)]
pub struct Context {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,

    pub debug_utils_instance: debug_utils::Instance,
    pub debug_utils_messenger: vk::DebugUtilsMessengerEXT,

    pub physical_device: vk::PhysicalDevice,
    pub physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,

    pub surface_instance: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_extent: vk::Extent2D,

    pub swapchain_device: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,

    pub present_queue: vk::Queue,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,

    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_device_memory: vk::DeviceMemory,

    pub command_pool: vk::CommandPool,
    pub init_command_buffer: vk::CommandBuffer,
    pub draw_command_buffer: vk::CommandBuffer,

    pub present_complete_semaphore: vk::Semaphore,
    pub render_complete_semaphore: vk::Semaphore,

    pub init_commands_fence: vk::Fence,
    pub draw_commands_fence: vk::Fence,
}

impl Context {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_width: u32,
        window_height: u32,
    ) -> Result<Self> {
        unsafe {
            let entry = Entry::load()?;
            let instance = create_instance(&entry, display_handle)?;

            let debug_utils_instance = debug_utils::Instance::new(&entry, &instance);
            let debug_utils_messenger = create_debug_utils_messenger(&debug_utils_instance)?;

            let surface_instance = surface::Instance::new(&entry, &instance);
            let surface =
                ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?;

            let (physical_device, physical_device_memory_properties, queue_family_index) =
                get_physical_device(&instance, &surface_instance, &surface)?;
            let device = create_logical_device(&instance, physical_device, queue_family_index)?;

            let (surface_format, surface_extent, pre_transform, desired_image_count) =
                get_surface_info(
                    &physical_device,
                    &surface_instance,
                    &surface,
                    window_width,
                    window_height,
                )?;

            let present_queue = device.get_device_queue(queue_family_index, 0);
            let present_modes = surface_instance
                .get_physical_device_surface_present_modes(physical_device, surface)?;
            let present_mode = present_modes
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_device = swapchain::Device::new(&instance, &device);
            let swapchain = create_swapchain(
                &swapchain_device,
                surface,
                surface_format,
                surface_extent,
                pre_transform,
                desired_image_count,
                present_mode,
            )?;

            let command_pool = create_command_pool(&device, queue_family_index)?;
            let command_buffers = create_command_buffers(&device, command_pool)?;
            let init_command_buffer = command_buffers[0];
            let draw_command_buffer = command_buffers[1];

            let present_images = swapchain_device.get_swapchain_images(swapchain)?;
            let present_image_views =
                create_present_image_views(&device, &present_images, surface_format.format)?;

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
            let (depth_image, depth_image_device_memory) =
                create_image(&device, depth_image_info, physical_device_memory_properties)?;
            let depth_image_view =
                create_depth_image_view(&device, &depth_image, depth_image_info.format)?;

            let present_complete_semaphore = create_semaphore(&device)?;
            let render_complete_semaphore = create_semaphore(&device)?;
            let init_commands_fence = create_fence(&device)?;
            let draw_commands_fence = create_fence(&device)?;

            // NOTE: WHAT IS THIS
            let image_memory_barriers = vk::ImageMemoryBarrier::default()
                .image(depth_image)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .layer_count(1)
                        .level_count(1),
                );

            record_submit_commandbuffer(
                &device,
                init_command_buffer,
                init_commands_fence,
                present_queue,
                &[],
                &[],
                &[],
                |setup_command_buffer| {
                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[image_memory_barriers],
                    );
                },
            )?;

            Ok(Self {
                entry,
                instance,
                device,

                debug_utils_instance,
                debug_utils_messenger,

                physical_device,
                physical_device_memory_properties,
                queue_family_index,

                surface_instance,
                surface,
                surface_format,
                surface_extent,

                swapchain_device,
                swapchain,

                present_queue,
                present_images,
                present_image_views,

                depth_image,
                depth_image_view,
                depth_image_device_memory,

                command_pool,
                init_command_buffer,
                draw_command_buffer,

                present_complete_semaphore,
                render_complete_semaphore,

                init_commands_fence,
                draw_commands_fence,
            })
        }
    }
}

impl Drop for Context {
    #[rustfmt::skip]
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_semaphore(self.present_complete_semaphore, None);
            self.device.destroy_semaphore(self.render_complete_semaphore, None);
            self.device.destroy_fence(self.draw_commands_fence, None);
            self.device.destroy_fence(self.init_commands_fence, None);
            self.device.free_memory(self.depth_image_device_memory, None);
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            for &image_view in self.present_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain_device.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            self.debug_utils_instance.destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

#[inline]
fn create_instance(entry: &Entry, raw_display_handle: RawDisplayHandle) -> VkResult<Instance> {
    let app_name = c"vxl 0.1.0";
    let engine_name = c"raw_engine";
    let version = vk::make_api_version(0, 0, 1, 0);

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(version)
        .engine_name(&engine_name)
        .engine_version(version)
        .api_version(vk::API_VERSION_1_3);

    let layers_names_raw: Vec<*const c_char> = [c"VK_LAYER_KHRONOS_validation"]
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let mut extension_names =
        ash_window::enumerate_required_extensions(raw_display_handle)?.to_vec();
    extension_names.push(debug_utils::NAME.as_ptr());
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
    }

    let instance_create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names)
        .flags(instance_create_flags);

    unsafe { Ok(entry.create_instance(&instance_create_info, None)?) }
}

#[inline]
fn create_debug_utils_messenger(
    debug_utils_instance: &debug_utils::Instance,
) -> VkResult<vk::DebugUtilsMessengerEXT> {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    unsafe { Ok(debug_utils_instance.create_debug_utils_messenger(&debug_info, None)?) }
}

#[inline]
fn get_physical_device(
    instance: &Instance,
    surface_instance: &surface::Instance,
    surface: &vk::SurfaceKHR,
) -> VkResult<(vk::PhysicalDevice, vk::PhysicalDeviceMemoryProperties, u32)> {
    unsafe {
        let physical_devices = instance.enumerate_physical_devices()?;
        let (physical_device, queue_family_index) = physical_devices
            .iter()
            .find_map(|physical_device| {
                instance
                    .get_physical_device_queue_family_properties(*physical_device)
                    .iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        let supports_graphic_and_surface =
                            info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && surface_instance
                                    .get_physical_device_surface_support(
                                        *physical_device,
                                        index as u32,
                                        *surface,
                                    )
                                    .expect("Could not get physical device surface support.");
                        if supports_graphic_and_surface {
                            Some((*physical_device, index as u32))
                        } else {
                            None
                        }
                    })
            })
            .expect("Could not find suitable device.");

        let memory_properties = instance.get_physical_device_memory_properties(physical_device);
        Ok((physical_device, memory_properties, queue_family_index))
    }
}

#[inline]
fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> VkResult<Device> {
    unsafe {
        let queue_priorities = [1.0];
        let device_queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let extension_names_raw = [
            swapchain::NAME.as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ash::khr::portability_subset::NAME.as_ptr(),
        ];
        let features = vk::PhysicalDeviceFeatures::default().shader_clip_distance(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&device_queue_info))
            .enabled_extension_names(&extension_names_raw)
            .enabled_features(&features);

        Ok(instance.create_device(physical_device, &device_info, None)?)
    }
}

#[inline]
fn get_surface_info(
    physical_device: &vk::PhysicalDevice,
    surface_instance: &surface::Instance,
    surface: &vk::SurfaceKHR,
    window_width: u32,
    window_height: u32,
) -> VkResult<(
    vk::SurfaceFormatKHR,
    vk::Extent2D,
    vk::SurfaceTransformFlagsKHR,
    u32,
)> {
    unsafe {
        let surface_format =
            surface_instance.get_physical_device_surface_formats(*physical_device, *surface)?[0];

        let surface_capabilities = surface_instance
            .get_physical_device_surface_capabilities(*physical_device, *surface)?;

        let mut desired_image_count = surface_capabilities.min_image_count + 1;

        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }

        let surface_extent = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: window_width,
                height: window_height,
            },
            _ => surface_capabilities.current_extent,
        };

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        Ok((
            surface_format,
            surface_extent,
            pre_transform,
            desired_image_count,
        ))
    }
}

#[inline]
fn create_swapchain(
    swapchain_device: &swapchain::Device,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_extent: vk::Extent2D,
    pre_transform: vk::SurfaceTransformFlagsKHR,
    desired_image_count: u32,
    present_mode: vk::PresentModeKHR,
) -> VkResult<vk::SwapchainKHR> {
    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(surface_extent)
        .pre_transform(pre_transform)
        .min_image_count(desired_image_count)
        .present_mode(present_mode)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .clipped(true)
        .image_array_layers(1);

    unsafe { Ok(swapchain_device.create_swapchain(&swapchain_info, None)?) }
}

#[inline]
fn create_command_pool(device: &Device, queue_family_index: u32) -> VkResult<vk::CommandPool> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);

    unsafe { Ok(device.create_command_pool(&command_pool_info, None)?) }
}

#[inline]
fn create_command_buffers(
    device: &Device,
    command_pool: vk::CommandPool,
) -> VkResult<Vec<vk::CommandBuffer>> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(2)
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    unsafe { Ok(device.allocate_command_buffers(&command_buffer_allocate_info)?) }
}

#[inline]
fn create_present_image_views(
    device: &Device,
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

fn create_image(
    device: &Device,
    image_info: vk::ImageCreateInfo,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
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
fn create_depth_image_view(
    device: &Device,
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

#[inline]
fn create_semaphore(device: &Device) -> VkResult<vk::Semaphore> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    unsafe { Ok(device.create_semaphore(&semaphore_info, None)?) }
}

#[inline]
fn create_fence(device: &Device) -> VkResult<vk::Fence> {
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    unsafe { Ok(device.create_fence(&fence_info, None)?) }
}

pub fn record_submit_commandbuffer<F: FnOnce(vk::CommandBuffer)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    op: F,
) -> VkResult<()> {
    let command_buffers = vec![command_buffer];
    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(wait_semaphores)
        .wait_dst_stage_mask(wait_mask)
        .command_buffers(&command_buffers)
        .signal_semaphores(signal_semaphores);

    unsafe {
        device.wait_for_fences(&[command_buffer_fence], true, u64::MAX)?;
        device.reset_fences(&[command_buffer_fence])?;
        device.reset_command_buffer(
            command_buffer,
            vk::CommandBufferResetFlags::RELEASE_RESOURCES,
        )?;

        device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
        op(command_buffer);
        device.end_command_buffer(command_buffer)?;

        device.queue_submit(submit_queue, &[submit_info], command_buffer_fence)
    }
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _)| index as u32)
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    unsafe {
        let callback_data = *p_callback_data;
        let message_id_number = callback_data.message_id_number;
        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };
        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        println!(
            "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );

        vk::FALSE
    }
}
