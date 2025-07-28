#![warn(unused_qualifications)]

use adapter::Adapter;
use anyhow::Result;
use ash::{ext, khr, prelude::VkResult, vk};
use queue::QueueFamily;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::{ffi, slice};

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
        let entry = unsafe { ash::Entry::load()? };
        let instance = create_instance(&entry, display_handle, app_info)?;
        let debug_instance = ext::debug_utils::Instance::new(&entry, &instance);
        let debug = Debug::new(&debug_instance)?;
        let adapters = unsafe {
            instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|physical_device| Adapter::new(&instance, physical_device))
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
    pub fn request_adapter_type(
        &self,
        physical_device_type: vk::PhysicalDeviceType,
    ) -> Option<Adapter> {
        self.adapters.iter().find_map(|adapter| {
            if adapter.properties.device_type == physical_device_type {
                Some(adapter.clone())
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn destroy(&mut self) {
        unsafe {
            self.debug.destroy(&self.debug_instance);
            self.instance.destroy_instance(None);
        }
    }
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
