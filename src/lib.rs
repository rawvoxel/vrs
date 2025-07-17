#![warn(unused_qualifications)]

use adapter::Adapter;
use anyhow::Result;
use ash::{ext, khr, prelude::VkResult, vk};
use queue::QueueFamily;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi;

pub mod adapter;
pub mod alloc;
pub mod buffer;
pub mod commands;
pub mod debug;
pub mod image;
pub mod queue;
pub mod renderer;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod vertex;

use debug::Debug;

pub struct App {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub display_handle: RawDisplayHandle,
    pub window_handle: RawWindowHandle,
    pub debug_instance: ext::debug_utils::Instance,
    pub debug: Debug,
    pub adapters: Vec<Adapter>,
}

unsafe impl Send for App {}
unsafe impl Sync for App {}

impl App {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        app_info: vk::ApplicationInfo,
    ) -> Result<App> {
        let entry = create_entry()?;
        let instance = create_instance(&entry, display_handle, app_info)?;
        let debug_instance = ext::debug_utils::Instance::new(&entry, &instance);
        let debug = Debug::new(&debug_instance)?;
        let adapters: Vec<Adapter> = unsafe {
            instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|physical_device| Adapter::new(&instance, physical_device).unwrap())
                .collect()
        };

        Ok(Self {
            entry,
            instance,
            display_handle,
            window_handle,
            debug_instance,
            debug,
            adapters,
        })
    }

    #[inline]
    pub fn request_physical_device_type(
        &self,
        device_type: vk::PhysicalDeviceType,
    ) -> Option<Adapter> {
        self.adapters.iter().find_map(|adapter| {
            if adapter.properties.device_type == device_type {
                Some(adapter.clone())
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn create_logical_device(
        &self,
        adapter: &Adapter,
        queue_family: &QueueFamily,
    ) -> VkResult<ash::Device> {
        unsafe {
            let queue_priorities = [1.0];
            let device_queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family.index)
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

            self.instance
                .create_device(adapter.inner, &device_info, None)
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.debug.destroy(&self.debug_instance);
            self.instance.destroy_instance(None);
        }
    }
}

#[inline]
fn create_entry() -> Result<ash::Entry, ash::LoadingError> {
    unsafe { ash::Entry::load() }
}

#[inline]
fn create_instance(
    entry: &ash::Entry,
    display_handle: RawDisplayHandle,
    application_info: vk::ApplicationInfo,
) -> VkResult<ash::Instance> {
    let layers: Vec<*const ffi::c_char> = [c"VK_LAYER_KHRONOS_validation"]
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let mut extensions = ash_window::enumerate_required_extensions(display_handle)?.to_vec();

    extensions.push(ext::debug_utils::NAME.as_ptr());

    if cfg!(any(target_os = "macos", target_os = "ios")) {
        extensions.push(khr::portability_enumeration::NAME.as_ptr());
        extensions.push(khr::get_physical_device_properties2::NAME.as_ptr());
    }

    let instance_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(instance_flags);

    unsafe { entry.create_instance(&instance_info, None) }
}

#[inline]
pub fn create_command_pool(
    logical_device: &ash::Device,
    queue_family: &QueueFamily,
) -> VkResult<vk::CommandPool> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family.index);

    unsafe { logical_device.create_command_pool(&command_pool_info, None) }
}

#[inline]
pub fn create_command_buffers(
    logical_device: &ash::Device,
    command_pool: vk::CommandPool,
    command_buffer_count: u32,
) -> VkResult<Vec<vk::CommandBuffer>> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(command_buffer_count)
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    unsafe { logical_device.allocate_command_buffers(&command_buffer_allocate_info) }
}

pub fn record_submit_commandbuffer<F: FnOnce(vk::CommandBuffer)>(
    logical_device: &ash::Device,
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
        logical_device.wait_for_fences(&[command_buffer_fence], true, u64::MAX)?;
        logical_device.reset_fences(&[command_buffer_fence])?;
        logical_device.reset_command_buffer(command_buffer, reset_flags)?;
        logical_device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

        op(command_buffer);

        logical_device.end_command_buffer(command_buffer)?;
        logical_device.queue_submit(submit_queue, &[submit_info], command_buffer_fence)
    }
}
