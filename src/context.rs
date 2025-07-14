use crate::{
    commands::{Commands, record_submit_commandbuffer},
    debug::DebugUtils,
    surface::SurfaceInfo,
    swapchain::{PresentQueue, Swapchain},
};
use anyhow::Result;
use ash::{ext::debug_utils, khr, prelude::VkResult, vk};
use ash_window::{self, create_surface};
use std::ffi;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

#[derive(Clone)]
pub struct Context {
    pub vk_entry: ash::Entry,
    pub vk_instance: ash::Instance,
    pub vk_device: ash::Device,

    pub debug_utils_instance: debug_utils::Instance,
    pub debug_utils: DebugUtils,

    pub surface_instance: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_info: SurfaceInfo,

    pub physical_device: vk::PhysicalDevice,
    pub physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,

    pub swapchain_device: khr::swapchain::Device,
    pub present_queue: PresentQueue,
    pub swapchain: Swapchain,
    pub commands: Commands,
}

impl Context {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_width: u32,
        window_height: u32,
    ) -> Result<Self> {
        unsafe {
            let vk_entry = ash::Entry::load()?;
            let vk_instance = create_instance(&vk_entry, display_handle)?;

            let debug_utils_instance = debug_utils::Instance::new(&vk_entry, &vk_instance);
            let debug_utils = DebugUtils::new(&debug_utils_instance)?;

            let surface_instance = khr::surface::Instance::new(&vk_entry, &vk_instance);
            let surface =
                create_surface(&vk_entry, &vk_instance, display_handle, window_handle, None)?;

            let (physical_device, physical_device_memory_properties, queue_family_index) =
                get_physical_device(&vk_instance, &surface_instance, surface)?;
            let vk_device = create_device(&vk_instance, physical_device, queue_family_index)?;

            let surface_format =
                surface_instance.get_physical_device_surface_formats(physical_device, surface)?[0];
            let surface_capabilities = surface_instance
                .get_physical_device_surface_capabilities(physical_device, surface)?;
            let surface_info = SurfaceInfo::new(surface_capabilities, window_width, window_height)?;

            let present_queue = PresentQueue::new(
                &vk_device,
                &surface_instance,
                surface,
                physical_device,
                queue_family_index,
            )?;

            let swapchain_device = khr::swapchain::Device::new(&vk_instance, &vk_device);
            let swapchain = Swapchain::new(
                &vk_device,
                &swapchain_device,
                surface,
                surface_format,
                surface_info,
                present_queue.mode,
                physical_device_memory_properties,
                None,
            )?;

            let commands = Commands::new(&vk_device, queue_family_index)?;

            // NOTE: WHAT IS THIS
            let image_memory_barrier = vk::ImageMemoryBarrier::default()
                .image(swapchain.depth_image)
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
                &vk_device,
                commands.init_buffers[0],
                commands.init_fences[0],
                present_queue.queue,
                &[],
                &[],
                &[],
                |setup_command_buffer| {
                    vk_device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[image_memory_barrier],
                    );
                },
            )?;

            Ok(Self {
                vk_entry,
                vk_instance,
                vk_device,

                debug_utils_instance,
                debug_utils,

                surface_instance,
                surface,
                surface_format,
                surface_capabilities,
                surface_info,

                physical_device,
                physical_device_memory_properties,
                queue_family_index,

                swapchain_device,
                present_queue,
                swapchain,
                commands,
            })
        }
    }
}

impl Drop for Context {
    #[rustfmt::skip]
    fn drop(&mut self) {
        unsafe {
            self.vk_device.device_wait_idle().unwrap();

            self.present_queue.destroy(&self.vk_device);
            self.swapchain.destroy(&self.vk_device, &self.swapchain_device);
            self.commands.destroy(&self.vk_device);
            self.vk_device.destroy_device(None);

            self.surface_instance.destroy_surface(self.surface, None);
            self.debug_utils.destroy(&self.debug_utils_instance);
            self.vk_instance.destroy_instance(None);
        }
    }
}

#[inline]
fn create_instance(
    vk_entry: &ash::Entry,
    raw_display_handle: RawDisplayHandle,
) -> VkResult<ash::Instance> {
    let app_name = c"vxl 0.1.0";
    let engine_name = c"raw_engine";
    let version = vk::make_api_version(0, 0, 1, 0);

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(version)
        .engine_name(&engine_name)
        .engine_version(version)
        .api_version(vk::API_VERSION_1_3);

    let layers_names: Vec<*const ffi::c_char> = [c"VK_LAYER_KHRONOS_validation"]
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
        .enabled_layer_names(&layers_names)
        .enabled_extension_names(&extension_names)
        .flags(instance_create_flags);

    unsafe { Ok(vk_entry.create_instance(&instance_create_info, None)?) }
}

#[inline]
fn get_physical_device(
    vk_instance: &ash::Instance,
    surface_instance: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> VkResult<(vk::PhysicalDevice, vk::PhysicalDeviceMemoryProperties, u32)> {
    unsafe {
        let physical_devices = vk_instance.enumerate_physical_devices()?;
        let (physical_device, queue_family_index) = physical_devices
            .into_iter()
            .find_map(|physical_device| {
                vk_instance
                    .get_physical_device_queue_family_properties(physical_device)
                    .into_iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        let supports_graphic_and_surface =
                            info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && surface_instance
                                    .get_physical_device_surface_support(
                                        physical_device,
                                        index as u32,
                                        surface,
                                    )
                                    .expect("Could not get physical device surface support.");
                        if supports_graphic_and_surface {
                            Some((physical_device, index as u32))
                        } else {
                            None
                        }
                    })
            })
            .expect("Could not find suitable device.");

        let memory_properties = vk_instance.get_physical_device_memory_properties(physical_device);
        Ok((physical_device, memory_properties, queue_family_index))
    }
}

#[inline]
fn create_device(
    vk_instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> VkResult<ash::Device> {
    unsafe {
        let queue_priorities = [1.0];
        let device_queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let extension_names_raw = [
            khr::swapchain::NAME.as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            khr::portability_subset::NAME.as_ptr(),
        ];

        let features = vk::PhysicalDeviceFeatures::default().shader_clip_distance(true);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&device_queue_info))
            .enabled_extension_names(&extension_names_raw)
            .enabled_features(&features);

        Ok(vk_instance.create_device(physical_device, &device_info, None)?)
    }
}

#[inline]
pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    physical_device_memory_properties.memory_types
        [..physical_device_memory_properties.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _)| index as _)
}
