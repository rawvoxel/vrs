use ash::{prelude::VkResult, vk};

#[derive(Copy, Clone)]
pub struct SurfaceInfo {
    pub extent: vk::Extent2D,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub min_image_count: u32,
}

impl SurfaceInfo {
    pub fn new(
        surface_capabilities: vk::SurfaceCapabilitiesKHR,
        window_width: u32,
        window_height: u32,
    ) -> VkResult<Self> {
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

        let min_image_count =
            (surface_capabilities.min_image_count + 1).min(surface_capabilities.max_image_count);

        Ok(Self {
            extent,
            pre_transform,
            min_image_count,
        })
    }
}
