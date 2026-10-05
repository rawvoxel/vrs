//! Point and spot lights, see `Renderer::lights`, and each frame's as the shaders read them.

use crate::renderer::Renderer;
use ash::prelude::VkResult;
use glam::Vec3;

/// The most lights a frame lights with, the rest are left out. Light indices fit 8 bits in
/// `trace_pass.slang`'s light shadows.
pub const MAX_LIGHTS: usize = 255;

/// A point light, or a spot light with a `spot`, lighting chunk meshes within its `range`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// In the world.
    pub position: Vec3,
    /// Linear, its brightness at its center. It fades with the square of the distance, to
    /// nothing at `range`.
    pub color: [f32; 3],
    /// How far its light reaches, in world units.
    pub range: f32,
    /// The cone a spot light shines in, `None` for a point light.
    pub spot: Option<Spot>,
}

impl Light {
    /// A point light at `position` of `color` reaching `range` world units.
    pub fn point(position: Vec3, color: [f32; 3], range: f32) -> Self {
        Self {
            position,
            color,
            range,
            spot: None,
        }
    }

    /// The same light shining in `spot`'s cone.
    pub fn with_spot(self, spot: Spot) -> Self {
        Self {
            spot: Some(spot),
            ..self
        }
    }
}

/// The cone of a spot light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spot {
    /// Where the cone points, in the world. Needn't be normalized.
    pub direction: Vec3,
    /// The angle from its center, in radians, within which it's at full brightness.
    pub inner_angle: f32,
    /// The angle from its center, in radians, past which it gives no light.
    pub outer_angle: f32,
}

/// A `Light` as the shaders read it. Matches `Light` in `common/layout.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuLight {
    /// The position, then the range.
    position_range: [f32; 4],
    /// The color, then 1 for a spot light.
    color_spot: [f32; 4],
    /// The spot's direction, normalized, then the cosine of its outer angle.
    direction_outer: [f32; 4],
    /// The cosine of the spot's inner angle, the rest unused.
    inner: [f32; 4],
}

impl GpuLight {
    /// No light, which the shaders read in place of none.
    pub const NONE: Self = Self {
        position_range: [0.0, 0.0, 0.0, 1.0],
        color_spot: [0.0; 4],
        direction_outer: [0.0, -1.0, 0.0, -1.0],
        inner: [-1.0, 0.0, 0.0, 0.0],
    };
}

impl From<&Light> for GpuLight {
    fn from(light: &Light) -> Self {
        let [r, g, b] = light.color;
        let (spot, direction, outer, inner) = match light.spot {
            Some(spot) => {
                let outer = spot.outer_angle.max(1e-3);
                let inner = spot.inner_angle.clamp(0.0, outer - 1e-4);
                (
                    1.0,
                    spot.direction.normalize_or(Vec3::NEG_Y),
                    outer.cos(),
                    inner.cos(),
                )
            }
            None => (0.0, Vec3::NEG_Y, -1.0, -1.0),
        };
        Self {
            position_range: light.position.extend(light.range.max(1e-3)).to_array(),
            color_spot: [r, g, b, spot],
            direction_outer: direction.extend(outer).to_array(),
            inner: [inner, 0.0, 0.0, 0.0],
        }
    }
}

impl Renderer {
    /// How many of `lights` light the frame.
    pub(in crate::renderer) fn light_count(&self) -> u32 {
        self.lights.len().min(MAX_LIGHTS) as u32
    }

    /// Writes the frame's lights.
    ///
    /// # Safety
    /// The device must not be using `frame`'s light buffer.
    pub(in crate::renderer) unsafe fn write_lights(&mut self, frame: usize) -> VkResult<()> {
        self.gpu_lights.clear();
        self.gpu_lights.extend(
            self.lights[..self.light_count() as usize]
                .iter()
                .map(GpuLight::from),
        );
        // An empty buffer can't be bound, a placeholder entry stands in.
        if self.gpu_lights.is_empty() {
            self.gpu_lights.push(GpuLight::NONE);
        }
        unsafe {
            self.frame_light_buffers[frame].write(&self.adapter, &self.device, &self.gpu_lights)
        }
    }
}
