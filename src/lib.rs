use adapter::Adapter;
use anyhow::Result;
use ash::{ext, khr, prelude::VkResult, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::{self, CStr};

pub mod adapter;
pub mod alloc;
pub mod buffer;
pub mod commands;
pub mod debug;
pub mod descriptors;
pub mod image;
pub mod queue;
pub mod renderer;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod uniforms;
pub mod vertex;

use debug::Debug;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

pub struct App {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub display_handle: RawDisplayHandle,
    pub window_handle: RawWindowHandle,
    pub debug_instance: ext::debug_utils::Instance,
    /// Only present in debug builds, when the validation layer is installed.
    pub debug: Option<Debug>,
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
        let validation = cfg!(debug_assertions) && validation_available(&entry)?;
        let instance = create_instance(&entry, display_handle, app_info, validation)?;
        let debug_instance = ext::debug_utils::Instance::new(&entry, &instance);
        let debug = match validation {
            true => Some(Debug::new(&debug_instance)?),
            false => None,
        };
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

    /// The most capable adapter: discrete, then integrated, then virtual, then anything else.
    #[inline]
    pub fn request_adapter(&self) -> Option<Adapter> {
        [
            vk::PhysicalDeviceType::DISCRETE_GPU,
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            vk::PhysicalDeviceType::VIRTUAL_GPU,
        ]
        .into_iter()
        .find_map(|device_type| self.request_adapter_type(device_type))
        .or_else(|| self.adapters.first().cloned())
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

}

/// Anything created from this `App`, such as a `Renderer`, must be dropped first.
impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug) = &self.debug {
                debug.destroy(&self.debug_instance);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Whether both the validation layer and the debug utils extension are installed.
#[inline]
fn validation_available(entry: &ash::Entry) -> VkResult<bool> {
    unsafe {
        let has_layer = entry
            .enumerate_instance_layer_properties()?
            .iter()
            .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
        let has_extension = entry
            .enumerate_instance_extension_properties(None)?
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(ext::debug_utils::NAME));
        Ok(has_layer && has_extension)
    }
}

#[inline]
fn create_instance(
    entry: &ash::Entry,
    display_handle: RawDisplayHandle,
    application_info: vk::ApplicationInfo,
    validation: bool,
) -> VkResult<ash::Instance> {
    let mut layers: Vec<*const ffi::c_char> = Vec::new();
    let mut extensions = ash_window::enumerate_required_extensions(display_handle)?.to_vec();
    if validation {
        layers.push(VALIDATION_LAYER.as_ptr());
        extensions.push(ext::debug_utils::NAME.as_ptr());
    }
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
