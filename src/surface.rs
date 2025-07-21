use crate::{App, adapter::Adapter};
use ash::{khr, prelude::VkResult, vk};
use ash_window::create_surface;

#[derive(Clone)]
pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub transform_flags: vk::SurfaceTransformFlagsKHR,
    pub min_image_count: u32,
}

impl Surface {
    pub fn new(
        app: &App,
        adapter: &Adapter,
        surface_instance: &khr::surface::Instance,
    ) -> VkResult<Self> {
        unsafe {
            let surface = create_surface(
                &app.entry,
                &app.instance,
                app.display_handle,
                app.window_handle,
                None,
            )?;
            let modes = surface_instance
                .get_physical_device_surface_present_modes(adapter.handle, surface)?;
            let present_mode = modes
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let format =
                surface_instance.get_physical_device_surface_formats(adapter.handle, surface)?[0];
            let capabilities = surface_instance
                .get_physical_device_surface_capabilities(adapter.handle, surface)?;
            let transform_flags = if capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                capabilities.current_transform
            };

            // FIXME: Investigate why this makes or breaks swapchain recreation.
            // let min_image_count = capabilities.min_image_count + 1;
            let min_image_count = 2;

            Ok(Self {
                handle: surface,
                present_mode,
                format: format.format,
                color_space: format.color_space,
                transform_flags,
                min_image_count,
            })
        }
    }

    pub fn extent(
        &self,
        window_width: u32,
        window_height: u32,
        adapter: &Adapter,
        surface_instance: &khr::surface::Instance,
    ) -> VkResult<vk::Extent2D> {
        unsafe {
            let capabilities = surface_instance
                .get_physical_device_surface_capabilities(adapter.handle, self.handle)?;
            Ok(match capabilities.current_extent.width {
                u32::MAX => vk::Extent2D {
                    width: window_width,
                    height: window_height,
                },
                _ => capabilities.current_extent,
            })
        }
    }
}
