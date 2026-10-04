use crate::{App, adapter::Adapter};
use ash::{khr, prelude::VkResult, vk};

/// What the renderer draws into: the window's surface, or images of its own when headless.
#[derive(Clone)]
pub struct Surface {
    /// Null when offscreen.
    pub(crate) handle: vk::SurfaceKHR,
    pub(crate) present_mode: vk::PresentModeKHR,
    /// Every present mode the surface supports, for changing vsync.
    pub(crate) present_modes: Vec<vk::PresentModeKHR>,
    pub(crate) format: vk::Format,
    /// A sampleable depth format, for the depth pyramid.
    pub(crate) depth_format: vk::Format,
    pub(crate) color_space: vk::ColorSpaceKHR,
    pub(crate) transform_flags: vk::SurfaceTransformFlagsKHR,
    pub(crate) image_count: u32,
    /// Includes `TRANSFER_SRC` when the images can be copied, for capturing frames.
    pub(crate) image_usage: vk::ImageUsageFlags,
}

impl Surface {
    /// The `App`'s window surface, which must not be headless.
    pub fn new(
        app: &App,
        adapter: &Adapter,
        surface_instance: &khr::surface::Instance,
        vsync: bool,
    ) -> VkResult<Self> {
        unsafe {
            // NOTE: The `App` owns the surface.
            let surface = app.surface;
            let present_modes = surface_instance
                .get_physical_device_surface_present_modes(adapter.handle, surface)?;
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

            let mut image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
            if capabilities
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::TRANSFER_SRC)
            {
                image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
            }

            Ok(Self {
                handle: surface,
                present_mode: pick_present_mode(&present_modes, vsync),
                present_modes,
                format: format.format,
                depth_format: pick_depth_format(&app.instance, adapter),
                color_space: format.color_space,
                transform_flags,
                image_count,
                image_usage,
            })
        }
    }

    /// `image_count` images of the renderer's own, for a headless `App`.
    pub fn offscreen(app: &App, adapter: &Adapter, image_count: u32) -> Self {
        Self {
            handle: vk::SurfaceKHR::null(),
            present_mode: vk::PresentModeKHR::FIFO,
            present_modes: Vec::new(),
            // Every device can render to and copy from it.
            format: vk::Format::R8G8B8A8_SRGB,
            depth_format: pick_depth_format(&app.instance, adapter),
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            transform_flags: vk::SurfaceTransformFlagsKHR::IDENTITY,
            image_count,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        }
    }

    #[inline]
    pub fn is_offscreen(&self) -> bool {
        self.handle == vk::SurfaceKHR::null()
    }

    #[inline]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    #[inline]
    pub fn depth_format(&self) -> vk::Format {
        self.depth_format
    }

    #[inline]
    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_mode
    }

    /// Whether frames can be copied out of the images.
    #[inline]
    pub fn can_capture(&self) -> bool {
        self.image_usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)
    }

    /// Takes effect when the swapchain is next built. Offscreen there's nothing to sync to.
    pub fn set_vsync(&mut self, vsync: bool) {
        if !self.is_offscreen() {
            self.present_mode = pick_present_mode(&self.present_modes, vsync);
        }
    }

    /// The size to render at, which offscreen is the requested size.
    pub fn extent(
        &self,
        window_width: u32,
        window_height: u32,
        adapter: &Adapter,
        surface_instance: &khr::surface::Instance,
    ) -> VkResult<vk::Extent2D> {
        let requested = vk::Extent2D {
            width: window_width,
            height: window_height,
        };
        if self.is_offscreen() {
            return Ok(requested);
        }
        unsafe {
            let capabilities = surface_instance
                .get_physical_device_surface_capabilities(adapter.handle, self.handle)?;
            Ok(match capabilities.current_extent.width {
                u32::MAX => requested,
                _ => capabilities.current_extent,
            })
        }
    }
}

/// With vsync, `FIFO`, which every surface supports. Without, `MAILBOX` doesn't tear,
/// then `IMMEDIATE` does, and `FIFO` when neither is supported.
fn pick_present_mode(present_modes: &[vk::PresentModeKHR], vsync: bool) -> vk::PresentModeKHR {
    let preferred: &[vk::PresentModeKHR] = match vsync {
        true => &[],
        false => &[vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE],
    };
    preferred
        .iter()
        .copied()
        .find(|mode| present_modes.contains(mode))
        .unwrap_or(vk::PresentModeKHR::FIFO)
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
