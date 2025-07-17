use crate::{App, adapter::Adapter};
use ash::{khr, prelude::VkResult, vk};
use ash_window::create_surface;

#[derive(Clone)]
pub struct Surface {
    pub inner: vk::SurfaceKHR,
    pub format: vk::SurfaceFormatKHR,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub transform_flags: vk::SurfaceTransformFlagsKHR,
    pub extent: vk::Extent2D,
    pub min_image_count: u32,
}

impl Surface {
    pub fn new(
        app: &App,
        adapter: &Adapter,
        surface_instance: &khr::surface::Instance,
        window_width: u32,
        window_height: u32,
    ) -> VkResult<Self> {
        unsafe {
            let surface = create_surface(
                &app.entry,
                &app.instance,
                app.display_handle,
                app.window_handle,
                None,
            )?;
            let format =
                surface_instance.get_physical_device_surface_formats(adapter.inner, surface)?[0];
            let capabilities = surface_instance
                .get_physical_device_surface_capabilities(adapter.inner, surface)?;
            let extent = match capabilities.current_extent.width {
                u32::MAX => vk::Extent2D {
                    width: window_width,
                    height: window_height,
                },
                _ => capabilities.current_extent,
            };

            let transform_flags = if capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                capabilities.current_transform
            };

            let min_image_count =
                (capabilities.min_image_count + 1).min(capabilities.max_image_count);

            Ok(Self {
                inner: surface,
                format,
                capabilities,
                transform_flags,
                extent,
                min_image_count,
            })
        }
    }

    pub fn destroy(&self, surface_instance: &khr::surface::Instance) {
        unsafe {
            surface_instance.destroy_surface(self.inner, None);
        }
    }
}
