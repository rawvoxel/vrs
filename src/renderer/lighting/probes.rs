//! Keeping the light probes up to date: their uniforms each frame, and converging them faster
//! after the light changes.

use crate::{
    lighting::gi::{GiUniforms, PROBE_CELL},
    renderer::Renderer,
};
use glam::{IVec3, Mat4, Quat, Vec3};

/// A while of faster converging probes, see `Renderer::track_lighting_changes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::renderer) struct Convergence {
    /// What the probes' hysteresis for light is multiplied by.
    pub scale: f32,
    /// What their hysteresis for distances is multiplied by.
    pub depth_scale: f32,
    /// Frames left.
    pub frames: u32,
}

impl Convergence {
    /// None, the probes' usual hysteresis.
    pub const NONE: Self = Self {
        scale: 1.0,
        depth_scale: 1.0,
        frames: 0,
    };
    /// After a small change in the light, such as the sun moving a little.
    const SMALL: Self = Self {
        scale: 0.85,
        depth_scale: 1.0,
        frames: 4,
    };
    /// After a big change in the light, such as a sudden time of day.
    const LARGE: Self = Self {
        scale: 0.5,
        depth_scale: 1.0,
        frames: 10,
    };
    /// After a big change in everything, see `Renderer::converge_lighting`.
    const LARGE_WITH_DISTANCES: Self = Self {
        scale: 0.5,
        depth_scale: 0.5,
        frames: 10,
    };
}

/// The longest `ProbeTuning::update_interval`.
const MAX_UPDATE_INTERVAL: u32 = 16;

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
                candidate.scale < self.convergence.scale || self.convergence.frames == 0
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
                [self.convergence.scale, self.convergence.depth_scale]
            }
        };
    }

    /// Whether this frame updates and samples the light probes, see `Lighting::global_illumination`.
    pub(in crate::renderer) fn gi_enabled(&self) -> bool {
        self.gi_configured
            && self.lighting.global_illumination
            && self
                .lighting
                .shadow_grid
                .is_some_and(|grid| grid.voxels_per_edge.is_multiple_of(PROBE_CELL))
    }

    /// The light probes' uniforms, their volume centered on the camera.
    pub(in crate::renderer) fn gi_uniforms(&self, view: Mat4) -> GiUniforms {
        let camera = view.inverse().w_axis.truncate();
        let counts = self.probe_volume.counts;
        let cascades = self.probe_volume.cascades;
        let tuning = self.lighting.probe_tuning;
        let mut uniforms = GiUniforms {
            volume_counts: [
                counts[0],
                counts[1],
                counts[2],
                self.probe_volume.atlas_columns,
            ],
            ray_rotation: random_rotation(self.frame_number).to_array(),
            camera_position: camera.extend(1.0).to_array(),
            tuning: [
                tuning.hysteresis.clamp(0.0, 1.0),
                tuning.depth_hysteresis.clamp(0.0, 1.0),
                tuning.depth_sharpness.max(1.0),
                tuning.self_shadow_bias.max(0.0),
            ],
            rays: [
                tuning.ray_distance.max(0.0),
                self.hysteresis_scale[0],
                self.hysteresis_scale[1],
                0.0,
            ],
            frame: [
                self.frame_number as u32,
                self.probe_volume.rays_per_frame,
                cascades,
                tuning
                    .update_interval
                    .clamp(1, MAX_UPDATE_INTERVAL)
                    .next_power_of_two(),
            ],
            ..Default::default()
        };
        if self.gi_enabled()
            && let Some(grid) = self.lighting.shadow_grid
        {
            for cascade in 0..cascades {
                let cell_size = grid.voxel_size * (PROBE_CELL << cascade) as f32;
                let center = (camera / cell_size).floor().as_ivec3();
                let half = IVec3::from_array(counts.map(|count| count as i32 / 2));
                uniforms.volume_mins[cascade as usize] = (center - half).extend(1).to_array();
            }
        }
        uniforms
    }
}

/// A rotation uniformly random over every rotation, the same for the same `seed` [Shoemake 1992].
fn random_rotation(seed: u64) -> Quat {
    // SplitMix64, for three well mixed numbers from consecutive seeds.
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f32 / (1u64 << 53) as f32
    };
    let (u1, u2, u3) = (next(), next(), next());
    let tau = std::f32::consts::TAU;
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    Quat::from_xyzw(
        a * (tau * u2).sin(),
        a * (tau * u2).cos(),
        b * (tau * u3).sin(),
        b * (tau * u3).cos(),
    )
}
