//! What's drawn where nothing else is.

/// A sky drawn behind everything, see `Lighting::sky`: a gradient from the horizon up, the
/// `Lighting::ground_color` below it, and the sun's disc, in the `Lighting::sun_color`. Its
/// colors are what's seen, while the `Lighting::sky_color` is the light it casts, so they
/// needn't match.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sky {
    /// Straight up, in linear color, as is the horizon's.
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    /// The sun's angular radius, in degrees, 0 for no disc.
    pub sun_radius: f32,
    /// How much brighter the disc is than the light the sun casts.
    pub sun_brightness: f32,
    /// How far from the camera half of what's seen is the sky's color, in world units, so
    /// distant chunks fade into the horizon. 0 for no haze.
    pub haze_distance: f32,
}

impl Default for Sky {
    /// A clear blue day, with a sun a little larger than the real one's, and no haze.
    fn default() -> Self {
        Self {
            zenith: [0.22, 0.4, 0.8],
            horizon: [0.62, 0.72, 0.86],
            sun_radius: 0.6,
            sun_brightness: 20.0,
            haze_distance: 0.0,
        }
    }
}

impl Sky {
    /// As the shaders read it, see `LightingUniforms`.
    pub(in crate::renderer) fn uniforms(&self) -> [[f32; 4]; 3] {
        let [zr, zg, zb] = self.zenith;
        let [hr, hg, hb] = self.horizon;
        [
            [zr, zg, zb, 0.0],
            [hr, hg, hb, 0.0],
            [
                self.sun_radius.to_radians().cos(),
                self.sun_brightness.max(0.0),
                self.haze_distance.max(0.0),
                1.0,
            ],
        ]
    }
}
