//! Converging the probes faster for a while after the light changes [Majercik et al. 2021].

use crate::renderer::Renderer;
use glam::Vec3;

/// A while of faster converging probes, see `Renderer::track_lighting_changes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::renderer) struct Convergence {
    /// What the probes' hysteresis for light is multiplied by.
    pub hysteresis_scale: f32,
    /// What their hysteresis for distances is multiplied by.
    pub distance_hysteresis_scale: f32,
    /// Frames left.
    pub frames: u32,
}

impl Convergence {
    /// None, the probes' usual hysteresis.
    pub const NONE: Self = Self {
        hysteresis_scale: 1.0,
        distance_hysteresis_scale: 1.0,
        frames: 0,
    };
    /// After a small change in the light, such as the sun moving a little.
    const SMALL: Self = Self {
        hysteresis_scale: 0.85,
        distance_hysteresis_scale: 1.0,
        frames: 4,
    };
    /// After a big change in the light, such as a sudden time of day.
    const LARGE: Self = Self {
        hysteresis_scale: 0.5,
        distance_hysteresis_scale: 1.0,
        frames: 10,
    };
    /// After a big change in everything, see `Renderer::converge_lighting`.
    const LARGE_WITH_DISTANCES: Self = Self {
        hysteresis_scale: 0.5,
        distance_hysteresis_scale: 0.5,
        frames: 10,
    };
}

impl Renderer {
    /// Makes the light probes converge faster for a while, as after a big change in what's
    /// lit, such as teleporting or a cut to another scene. vrs does this on its own when the
    /// sun or sky change, or chunks near probes are added, removed or remeshed.
    pub fn converge_lighting(&mut self) {
        self.convergence = Convergence::LARGE_WITH_DISTANCES;
    }

    /// Notices how much the sun and sky changed since last frame, and sets this frame's
    /// probe hysteresis scale for it [Majercik et al. 2021].
    pub(in crate::renderer) fn track_lighting_changes(&mut self) {
        let now = self.lighting;
        if let Some(before) = self.previous_lighting {
            let angle = before
                .sun_direction
                .normalize_or(Vec3::Y)
                .angle_between(now.sun_direction.normalize_or(Vec3::Y))
                .to_degrees();
            let relative = |a: [f32; 3], b: [f32; 3]| {
                (0..3)
                    .map(|i| (a[i] - b[i]).abs() / a[i].abs().max(0.01))
                    .fold(0.0, f32::max)
            };
            let change = relative(before.sun_color, now.sun_color)
                .max(relative(before.sky_color, now.sky_color))
                .max(relative(before.ground_color, now.ground_color));
            let large = angle > 10.0 || change > 0.25;
            let small = angle > 0.25 || change > 0.005;
            let stronger = |candidate: Convergence| {
                candidate.hysteresis_scale < self.convergence.hysteresis_scale
                    || self.convergence.frames == 0
            };
            if large && stronger(Convergence::LARGE) {
                self.convergence = Convergence::LARGE;
            } else if small && stronger(Convergence::SMALL) {
                self.convergence = Convergence::SMALL;
            }
        }
        self.previous_lighting = Some(now);
        self.hysteresis_scale = match self.convergence.frames {
            0 => [1.0, 1.0],
            _ => {
                self.convergence.frames -= 1;
                [
                    self.convergence.hysteresis_scale,
                    self.convergence.distance_hysteresis_scale,
                ]
            }
        };
    }
}
