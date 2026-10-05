use crate::vulkan::{adapter::Adapter, validation::Messenger};
use anyhow::Result;
use ash::{ext, khr, prelude::VkResult, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::{self, CStr};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The Vulkan instance, and the window surface everything renders to unless it's headless.
pub struct App {
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) debug_instance: ext::debug_utils::Instance,
    /// Only present in debug builds, when the validation layer is installed.
    pub(crate) messenger: Option<Messenger>,
    pub(crate) surface_instance: khr::surface::Instance,
    /// The window's surface, owned by the `App`. Null when headless.
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) adapters: Vec<Adapter>,
}

impl App {
    /// Renders to the window.
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        app_info: vk::ApplicationInfo,
    ) -> Result<App> {
        let extensions = ash_window::enumerate_required_extensions(display_handle)?;
        let mut app = Self::create(app_info, extensions)?;
        app.surface = unsafe {
            ash_window::create_surface(
                &app.entry,
                &app.instance,
                display_handle,
                window_handle,
                None,
            )?
        };
        Ok(app)
    }

    /// Renders to images of its own instead of a window, such as for tests on machines without a display.
    pub fn headless(app_info: vk::ApplicationInfo) -> Result<App> {
        Self::create(app_info, &[])
    }

    fn create(app_info: vk::ApplicationInfo, extensions: &[*const ffi::c_char]) -> Result<App> {
        let entry = unsafe { ash::Entry::load()? };
        let validation = cfg!(debug_assertions) && validation_available(&entry)?;
        let instance = create_instance(&entry, extensions, app_info, validation)?;
        // Dropping the `App` destroys whatever was created, if anything below fails.
        let mut app = Self {
            debug_instance: ext::debug_utils::Instance::new(&entry, &instance),
            messenger: None,
            surface_instance: khr::surface::Instance::new(&entry, &instance),
            surface: vk::SurfaceKHR::null(),
            adapters: Vec::new(),
            entry,
            instance,
        };
        if validation {
            app.messenger = Some(Messenger::new(&app.debug_instance)?);
        }
        app.adapters = unsafe {
            app.instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|physical_device| Adapter::new(&app.instance, physical_device))
                .collect()
        };
        Ok(app)
    }

    #[inline]
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    #[inline]
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// The window's surface, `None` when headless.
    #[inline]
    pub fn surface(&self) -> Option<vk::SurfaceKHR> {
        (self.surface != vk::SurfaceKHR::null()).then_some(self.surface)
    }

    #[inline]
    pub fn is_headless(&self) -> bool {
        self.surface().is_none()
    }

    /// Only in debug builds, when the validation layer is installed.
    #[inline]
    pub fn validation_enabled(&self) -> bool {
        self.messenger.is_some()
    }

    #[inline]
    pub fn adapters(&self) -> &[Adapter] {
        &self.adapters
    }

    /// Every adapter, discrete first, then integrated, then virtual, then anything else.
    pub fn adapters_by_preference(&self) -> Vec<&Adapter> {
        let rank = |adapter: &&Adapter| match adapter.properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            _ => 3,
        };
        let mut adapters: Vec<&Adapter> = self.adapters.iter().collect();
        adapters.sort_by_key(rank);
        adapters
    }
}

/// Anything created from this `App`, such as a `Renderer`, must be dropped first.
impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            // A headless instance never loads the surface functions.
            if let Some(surface) = self.surface() {
                self.surface_instance.destroy_surface(surface, None);
            }
            if let Some(messenger) = &self.messenger {
                messenger.destroy(&self.debug_instance);
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
    surface_extensions: &[*const ffi::c_char],
    application_info: vk::ApplicationInfo,
    validation: bool,
) -> VkResult<ash::Instance> {
    let mut layers: Vec<*const ffi::c_char> = Vec::new();
    let mut extensions = surface_extensions.to_vec();
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
