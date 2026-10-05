//! The trace pass: from each frame's depth, rays traced through the voxel grid for the sun's and
//! the lights' shadows, glossy voxels' reflections, and the light from everywhere but the sun,
//! at the `TraceResolution`, for shading to upscale.

mod pass;

pub(crate) use pass::{TracePass, TraceUniforms};

/// The most of each pixel's lights that are shadowed, see `Lighting::shadowed_lights`. Each
/// trace texel keeps that many, 8 bits each. Matches `MAX_SHADOWED_LIGHTS` in
/// `common/lights.slang`.
pub const MAX_SHADOWED_LIGHTS: u32 = 4;

/// How finely the trace pass traces shadows and reflections, and finds the light from
/// everywhere but the sun, see `Lighting::trace_resolution`. Each traces once for a block of
/// pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TraceResolution {
    /// A shadow ray for every pixel.
    #[default]
    Full,
    /// One for every 2x2 pixels.
    Half,
    /// One for every 4x4 pixels.
    Quarter,
}

impl TraceResolution {
    /// Frame pixels along each axis that share a shadow ray.
    #[inline]
    pub fn pixels_per_ray(self) -> u32 {
        match self {
            TraceResolution::Full => 1,
            TraceResolution::Half => 2,
            TraceResolution::Quarter => 4,
        }
    }
}
