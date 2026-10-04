use crate::{App, adapter::Adapter};
use ash::{khr, prelude::VkResult, vk};

#[derive(Clone)]
pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub format: vk::Format,
    /// A sampleable depth format, for the depth pyramid.
    pub depth_format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub transform_flags: vk::SurfaceTransformFlagsKHR,
    pub image_count: u32,
}

impl Surface {
    pub fn new(
        app: &App,
        adapter: &Adapter,
        surface_instance: &khr::surface::Instance,
    ) -> VkResult<Self> {
        unsafe {
            // NOTE: The `App` owns the surface.
            let surface = app.surface;
            let depth_format = pick_depth_format(&app.instance, adapter);
            let present_mode = surface_instance
                .get_physical_device_surface_present_modes(adapter.handle, surface)?
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let format = pick_color_format(
                &surface_instance.get_physical_device_surface_formats(adapter.handle, surface)?,
            );
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

            // One more than the minimum, so acquiring never waits on the presentation engine.
            // A maximum of zero means there's no limit.
            let mut image_count = capabilities.min_image_count + 1;
            if capabilities.max_image_count > 0 {
                image_count = image_count.min(capabilities.max_image_count);
            }

            Ok(Self {
                handle: surface,
                present_mode,
                format: format.format,
                depth_format,
                color_space: format.color_space,
                transform_flags,
                image_count,
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

/// `D32_SFLOAT` when it can be rendered to and sampled, otherwise `D16_UNORM`, which always can.
fn pick_depth_format(instance: &ash::Instance, adapter: &Adapter) -> vk::Format {
    let required =
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE;
    let d32_features = unsafe {
        instance
            .get_physical_device_format_properties(adapter.handle, vk::Format::D32_SFLOAT)
            .optimal_tiling_features
    };
    match d32_features.contains(required) {
        true => vk::Format::D32_SFLOAT,
        false => vk::Format::D16_UNORM,
    }
}

/// An 8 bit sRGB format when there is one, so shaders can work in linear color.
fn pick_color_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .find(|format| {
            matches!(
                format.format,
                vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
            ) && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(&formats[0])
        .to_owned()
}
