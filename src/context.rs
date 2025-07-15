use crate::{debug::DebugUtils, surface::Surface};
use anyhow::Result;
use ash::{ext, khr, prelude::VkResult, vk};
use std::ffi;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

#[derive(Clone)]
pub struct Context {
    pub vk_entry: ash::Entry,
    pub vk_instance: ash::Instance,

    pub debug_utils_instance: ext::debug_utils::Instance,
    pub debug_utils: DebugUtils,

    pub surface_instance: khr::surface::Instance,
    pub surface: Surface,

    pub physical_device: vk::PhysicalDevice,
    pub physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,

    pub vk_device: ash::Device,
    pub command_pool: vk::CommandPool,
}

impl Context {
    #[rustfmt::skip]
    pub fn new(display_handle: RawDisplayHandle, window_handle: RawWindowHandle) -> Result<Self> {
        unsafe {
            let vk_entry = ash::Entry::load()?;
            let vk_instance = create_instance(&vk_entry, display_handle)?;

            let debug_utils_instance = ext::debug_utils::Instance::new(&vk_entry, &vk_instance);
            let debug_utils = DebugUtils::new(&debug_utils_instance)?;

            let surface_instance = khr::surface::Instance::new(&vk_entry, &vk_instance);
            let surface = Surface::new(&vk_entry, &vk_instance, display_handle, window_handle)?;

            let (physical_device, physical_device_memory_properties, queue_family_index) =
                try_find_physical_device_graphics(&vk_instance, &surface_instance, surface.inner)?;

            let vk_device = create_device(&vk_instance, physical_device, queue_family_index)?;
            let command_pool = create_command_pool(&vk_device, queue_family_index)?;

            Ok(Self {
                vk_entry,
                vk_instance,

                debug_utils_instance,
                debug_utils,

                surface_instance,
                surface,

                physical_device,
                physical_device_memory_properties,
                queue_family_index,

                vk_device,
                command_pool,
            })
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.vk_device.device_wait_idle().unwrap();
            self.vk_device.destroy_command_pool(self.command_pool, None);
            self.vk_device.destroy_device(None);

            self.surface.destroy(&self.surface_instance);
            self.debug_utils.destroy(&self.debug_utils_instance);
            self.vk_instance.destroy_instance(None);
        }
    }
}

#[inline]
fn create_instance(
    vk_entry: &ash::Entry,
    display_handle: RawDisplayHandle,
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

    let mut extension_names = ash_window::enumerate_required_extensions(display_handle)?.to_vec();

    extension_names.push(ext::debug_utils::NAME.as_ptr());

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

    unsafe { vk_entry.create_instance(&instance_create_info, None) }
}

#[inline]
fn try_find_physical_device_graphics(
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
                    .find_map(|(queue_family_index, queue_family_properties)| {
                        let graphics_queue = queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS);
                        let supports_surface = surface_instance
                            .get_physical_device_surface_support(
                                physical_device,
                                queue_family_index as u32,
                                surface,
                            )
                            .expect("Could not get physical device surface support.");
                        if graphics_queue && supports_surface {
                            Some((physical_device, queue_family_index as u32))
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
pub fn create_device(
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

        vk_instance.create_device(physical_device, &device_info, None)
    }
}

#[inline]
pub fn create_command_pool(
    vk_device: &ash::Device,
    queue_family_index: u32,
) -> VkResult<vk::CommandPool> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);

    unsafe { vk_device.create_command_pool(&command_pool_info, None) }
}

pub fn record_submit_commandbuffer<F: FnOnce(vk::CommandBuffer)>(
    vk_device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_dst_stage_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    op: F,
) -> VkResult<()> {
    let command_buffers = vec![command_buffer];
    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&command_buffers)
        .wait_dst_stage_mask(wait_dst_stage_mask)
        .wait_semaphores(wait_semaphores)
        .signal_semaphores(signal_semaphores);

    let reset_flags = vk::CommandBufferResetFlags::RELEASE_RESOURCES;

    unsafe {
        vk_device.wait_for_fences(&[command_buffer_fence], true, u64::MAX)?;
        vk_device.reset_fences(&[command_buffer_fence])?;
        vk_device.reset_command_buffer(command_buffer, reset_flags)?;
        vk_device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

        op(command_buffer);

        vk_device.end_command_buffer(command_buffer)?;
        vk_device.queue_submit(submit_queue, &[submit_info], command_buffer_fence)
    }
}
