use crate::vulkan::{
    adapter::Adapter,
    gpu::VulkanDevice,
    validation::{self, Messenger},
};
use anyhow::Result;
use ash::{ext, khr, prelude::VkResult, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::{self, CStr};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The Vulkan instance, the window surface everything renders to unless it's headless, and the
/// devices made from it with [`create_device`](crate::vulkan::gpu::create_device), which it
/// destroys.
pub struct App {
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) debug_instance: ext::debug_utils::Instance,
    /// Only with validation turned on ([`validation::VARIABLE`]), when the layer is installed.
    pub(crate) messenger: Option<Messenger>,
    pub(crate) surface_instance: khr::surface::Instance,
    /// The window's surface, owned by the `App`. Null when headless.
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) adapters: Vec<Adapter>,
    /// The devices made from it and not yet destroyed, destroyed when it's dropped.
    pub(crate) devices: Vec<ash::Device>,
}

impl App {
    /// Renders to the window. The instance is Vulkan 1.3 or later, whatever `app_info` asks for.
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

    /// Renders to images of its own instead of a window, such as for tests on machines without a
    /// display. The instance is Vulkan 1.3 or later, whatever `app_info` asks for.
    pub fn headless(app_info: vk::ApplicationInfo) -> Result<App> {
        Self::create(app_info, &[])
    }

    /// The instance takes `app_info`'s API version, raised to Vulkan 1.3 when it asks for less:
    /// every device made from it needs 1.3.
    fn create(app_info: vk::ApplicationInfo, extensions: &[*const ffi::c_char]) -> Result<App> {
        let app_info = app_info.api_version(app_info.api_version.max(vk::API_VERSION_1_3));
        let entry = unsafe { ash::Entry::load()? };
        let wanted = std::env::var(validation::VARIABLE).is_ok_and(|value| value != "0");
        let validation = wanted && validation_available(&entry)?;
        if wanted && !validation {
            log::warn!(
                "{} is set, but the validation layer isn't installed",
                validation::VARIABLE
            );
        }
        let instance = create_instance(&entry, extensions, app_info, validation)?;
        // Dropping the `App` destroys whatever was created, if anything below fails.
        let mut app = Self {
            debug_instance: ext::debug_utils::Instance::new(&entry, &instance),
            messenger: None,
            surface_instance: khr::surface::Instance::new(&entry, &instance),
            surface: vk::SurfaceKHR::null(),
            adapters: Vec::new(),
            devices: Vec::new(),
            entry,
            instance,
        };
        if validation {
            app.messenger = Some(Messenger::new(&app.debug_instance)?);
        }
        app.adapters = unsafe { app.instance.enumerate_physical_devices()? }
            .into_iter()
            .map(|physical_device| Adapter::new(&app.instance, physical_device))
            .collect::<VkResult<_>>()?;
        Ok(app)
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

    /// Whether validation is on: turned on ([`validation::VARIABLE`]), with the layer installed.
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

    /// Destroys `device`, one made from this `App`, once the device has finished all its work:
    /// its wait's error, such as `ERROR_DEVICE_LOST`, is returned, the device destroyed anyway.
    ///
    /// # Panics
    /// When `device` wasn't made from this `App`, or was destroyed already.
    ///
    /// # Safety
    /// Everything made with the device is destroyed, and nothing uses it any more.
    pub unsafe fn destroy_device(&mut self, device: &VulkanDevice) -> VkResult<()> {
        let index = self
            .devices
            .iter()
            .position(|recorded| recorded.handle() == device.device.handle())
            .expect("the device was made from this App, and not destroyed yet");
        let recorded = self.devices.swap_remove(index);
        unsafe {
            let waited = recorded.device_wait_idle();
            recorded.destroy_device(None);
            waited
        }
    }
}

/// Every device made from this `App` is destroyed, then the instance. Anything made with them
/// must be dropped first.
impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            for device in self.devices.drain(..) {
                // Nothing can be destroyed while the device might still use it.
                if let Err(error) = device.device_wait_idle() {
                    log::warn!("waiting for a device to finish before destroying it: {error}");
                }
                device.destroy_device(None);
            }
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
        let has_layer = enumerated(|| entry.enumerate_instance_layer_properties())?
            .iter()
            .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
        let has_extension = enumerated(|| entry.enumerate_instance_extension_properties(None))?
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(ext::debug_utils::NAME));
        Ok(has_layer && has_extension)
    }
}

/// `enumerate`'s list, asked again while the loader answers `INCOMPLETE`: the list changed
/// between its count and its read (Vulkan's enumeration pattern; the loader can return it on the
/// count query itself, as it did with validation runs in parallel), so the answer is asked anew.
fn enumerated<T>(enumerate: impl Fn() -> VkResult<Vec<T>>) -> VkResult<Vec<T>> {
    loop {
        match enumerate() {
            Err(vk::Result::INCOMPLETE) => continue,
            answered => return answered,
        }
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
