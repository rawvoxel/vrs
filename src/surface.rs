use crate::{App, adapter::Adapter, context::Context};
use ash::{khr, prelude::VkResult, vk};
use ash_window::create_surface;

#[derive(Clone)]
pub struct Surface {
    pub inner: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(app: &App) -> VkResult<Self> {
        let surface = unsafe {
            create_surface(
                &app.entry,
                &app.instance,
                app.display_handle,
                app.window_handle,
                None,
            )?
        };
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
        app: &App,
        context: &Context,
        adapter: &Adapter,
        window_width: u32,
        window_height: u32,
    ) -> VkResult<Self> {
        unsafe {
            let format = app
                .surface_instance
                .get_physical_device_surface_formats(adapter.inner, context.surface.inner)?[0];
            let capabilities = app
                .surface_instance
                .get_physical_device_surface_capabilities(adapter.inner, context.surface.inner)?;
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
                format,
                capabilities,
                extent,
                transform_flags,
                min_image_count,
            })
        }
    }
}
