//! When frames draw their depth first for the shadow pass, at what resolution it traces, and
//! its uniforms.

use crate::{
    lighting::shadows::ShadowUniforms,
    renderer::{DebugView, Renderer},
};
use ash::prelude::VkResult;
use glam::Mat4;

/// The most of each pixel's lights that are shadowed, see `Lighting::shadowed_lights`. Each
/// shadow texel keeps that many, 8 bits each. Matches `MAX_SHADOWED_LIGHTS` in
/// `common/lights.slang`.
pub const MAX_SHADOWED_LIGHTS: u32 = 4;

/// How finely the shadow pass traces shadows and reflections, and finds the light from
/// everywhere but the sun, see `Lighting::shadow_resolution`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowResolution {
    /// A shadow ray for every pixel.
    #[default]
    Full,
    /// One for every 2x2 pixels.
    Half,
    /// One for every 4x4 pixels.
    Quarter,
}

impl ShadowResolution {
    /// Frame pixels along each axis that share a shadow ray.
    #[inline]
    pub fn pixels_per_ray(self) -> u32 {
        match self {
            ShadowResolution::Full => 1,
            ShadowResolution::Half => 2,
            ShadowResolution::Quarter => 4,
        }
    }
}

impl Renderer {
    /// Frame pixels along each axis that share a shadow ray, 1 without shadows.
    pub(in crate::renderer) fn shadow_scale(&self) -> u32 {
        match self.lighting.shadow_grid {
            Some(_) => self.lighting.shadow_resolution.pixels_per_ray(),
            None => 1,
        }
    }

    /// Whether frames draw their depth first, for the shadow pass to trace shadows and cull
    /// lights from, then their color over it. Even at full resolution, that's faster than
    /// tracing shadows as pixels are shaded, which wastes rays on pixels drawn over.
    pub(in crate::renderer) fn uses_shadow_pass(&self) -> bool {
        self.lighting.shadow_grid.is_some() || !self.lights.is_empty()
    }

    /// Makes the shadow images fit the frame and `shadow_scale`, waiting for the device to go
    /// idle when they have to be replaced.
    pub(in crate::renderer) fn prepare_shadow_pass(&mut self) -> VkResult<()> {
        let scale = self.shadow_scale();
        let frame = self.swapchain.extent;
        let fits = self.shadow_pass.scale == scale
            && self.shadow_pass.extent.width == frame.width.div_ceil(scale)
            && self.shadow_pass.extent.height == frame.height.div_ceil(scale);
        if !self.uses_shadow_pass() || fits {
            return Ok(());
        }
        unsafe {
            self.render_device.device_wait_idle()?;
            self.shadow_pass
                .resize(&self.adapter, &self.render_device, frame, scale)?;
        }
        self.init_shadow_pass()
    }

    /// The shadow pass's uniforms for a frame seen through `view_proj`.
    pub(in crate::renderer) fn shadow_uniforms(&self, view_proj: Mat4) -> ShadowUniforms {
        let extent = self.shadow_pass.extent;
        let frame = self.swapchain.extent;
        let shadowed = match self.lighting.shadow_grid {
            Some(_) => self.lighting.shadowed_lights.min(MAX_SHADOWED_LIGHTS),
            None => 0,
        };
        let [tiles_x, tiles_y] = self.shadow_pass.tile_counts;
        ShadowUniforms {
            inverse_view_proj: view_proj.inverse(),
            params: [
                self.shadow_scale(),
                extent.width,
                extent.height,
                self.glossy_palette as u32,
            ],
            frame_size: [
                frame.width,
                frame.height,
                self.uses_shadow_pass() as u32,
                self.lights_indirect_in_shadow_pass() as u32,
            ],
            lights: [self.light_count(), tiles_x, tiles_y, shadowed],
        }
    }

    /// Whether the shadow pass lights faces with the light from everywhere but the sun,
    /// probes and ambient occlusion, for shading to upscale, which below full resolution is
    /// cheaper than every pixel finding it. Not for `DebugView::AmbientOcclusion`, which shows
    /// each pixel's.
    pub(in crate::renderer) fn lights_indirect_in_shadow_pass(&self) -> bool {
        let indirect_work = self.gi_enabled() || self.lighting.ambient_occlusion > 0.0;
        self.uses_shadow_pass()
            && self.shadow_scale() > 1
            && indirect_work
            && self.lighting.debug_view != DebugView::AmbientOcclusion
    }
}
