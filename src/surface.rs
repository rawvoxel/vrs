use ash::{khr, prelude::VkResult, vk};
use ash_window::create_surface;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

#[derive(Clone)]
pub struct Surface {
    pub inner: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(
        vk_entry: &ash::Entry,
        vk_instance: &ash::Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> VkResult<Self> {
        let surface =
            unsafe { create_surface(vk_entry, vk_instance, display_handle, window_handle, None)? };
        Ok(Self { inner: surface })
    }

    pub fn destroy(&self, surface_instance: &khr::surface::Instance) {
        unsafe {
            surface_instance.destroy_surface(self.inner, None);
        }
    }
}

#[derive(Copy, Clone)]
pub struct SurfaceInfo {
    pub format: vk::SurfaceFormatKHR,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub extent: vk::Extent2D,
    pub transform_flags: vk::SurfaceTransformFlagsKHR,
    pub min_image_count: u32,
}

impl SurfaceInfo {
    pub fn new(
        surface_instance: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        window_width: u32,
        window_height: u32,
    ) -> VkResult<Self> {
        unsafe {
            let surface_format =
                surface_instance.get_physical_device_surface_formats(physical_device, surface)?[0];
            let surface_capabilities = surface_instance
                .get_physical_device_surface_capabilities(physical_device, surface)?;
            let extent = match surface_capabilities.current_extent.width {
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

            let min_image_count = (surface_capabilities.min_image_count + 1)
                .min(surface_capabilities.max_image_count);

            Ok(Self {
                format: surface_format,
                capabilities: surface_capabilities,
                extent,
                transform_flags: pre_transform,
                min_image_count,
            })
        }
    }
}
