//! Shadows traced through the voxels: the chunks they're traced through, and the pass tracing
//! them, with reflections, from each frame's depth.

mod grid;
mod pass;

pub(in crate::renderer) use grid::SHADOW_TABLE_CHUNKS;
pub use grid::{MAX_SHADOW_LEVELS, ShadowGrid};
pub(crate) use pass::{ShadowPass, ShadowUniforms};

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
