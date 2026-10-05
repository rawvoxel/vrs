//! Dynamic diffuse global illumination: cascades of light probes around the camera, each
//! storing the light reaching it from every direction and how far away the nearest voxels
//! are, those near voxels traced through them in turns and sampled when shading. See
//! `gi.slang`.

mod cells;
mod convergence;
mod volume;

use crate::renderer::Renderer;
use glam::{IVec3, Mat4, Quat};

pub(crate) use cells::bake_probe_cells;
pub(in crate::renderer) use convergence::Convergence;
pub(crate) use volume::ProbeVolume;

/// Voxels along each edge of a probe's cell, and between neighboring probes, in the finest
/// cascade. Matches `PROBE_CELL` in `common/probe_volume.slang`.
pub const PROBE_CELL: u32 = 16;
/// The most cascades of probes. Matches `MAX_CASCADES` in `common/layout.slang`.
pub const MAX_CASCADES: u32 = 4;

/// The light probe volumes, see `RendererConfig::global_illumination`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GiConfig {
    /// Probes along each axis of each cascade, which is centered on the camera with probes
    /// 16 of the `ShadowGrid`'s finest voxels apart in the first, and twice as far apart in
    /// each next. Each probe takes about 2.9 KB.
    pub probe_counts: [u32; 3],
    /// Cascades of probes, from 1 to `MAX_CASCADES`. Each reaches twice as far as the last,
    /// with probes twice as far apart, for chunks of coarser `ShadowGrid::levels`. Surfaces
    /// are lit by the finest one around them.
    pub cascades: u32,
    /// Rays traced each frame, shared between every cascade's probes near voxels so the cost
    /// stays steady: each traces a power of two of them from 32 to 256. Past
    /// `rays_per_frame / 32` such probes, each still traces 32.
    pub rays_per_frame: u32,
}

impl Default for GiConfig {
    /// One cascade, 512 by 256 by 512 voxels, with 64 Ki rays a frame.
    fn default() -> Self {
        Self {
            probe_counts: [32, 16, 32],
            cascades: 1,
            rays_per_frame: 1 << 16,
        }
    }
}

/// `Lighting::global_illumination` as the shaders read it. Matches `GiUniforms` in
/// `common/layout.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GiUniforms {
    /// Each cascade's probe cell at its low corner, in its own cells, w is 1 when the probes
    /// light the frame.
    pub volume_mins: [[i32; 4]; MAX_CASCADES as usize],
    /// Probes along each axis of each cascade, w is probes per row of the atlases.
    pub volume_counts: [u32; 4],
    /// This frame's random rotation of the probes' rays, a quaternion.
    pub ray_rotation: [f32; 4],
    /// The camera's world position, w unused.
    pub camera_position: [f32; 4],
    /// x is the frame's number, y the rays traced each frame, z how many cascades there are, w
    /// the frames between the finest cascade's probes' updates.
    pub frame: [u32; 4],
    /// `ProbeTuning`'s hysteresis, depth hysteresis, depth sharpness and self-shadow bias.
    pub tuning: [f32; 4],
    /// `ProbeTuning`'s ray distance, then this frame's multipliers of the hysteresis for
    /// light and for distances, below 1 while the probes converge faster after a change.
    pub rays: [f32; 4],
}

/// How the light probes blend what they see and are sampled, see `Lighting::probe_tuning`.
/// The hysteresis, depth sharpness and self-shadow bias default to the papers' [Majercik et
/// al. 2019 and 2021].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeTuning {
    /// How much of a probe's light each frame keeps, from 0 to 1. Higher is steadier but
    /// slower to follow changes.
    pub hysteresis: f32,
    /// How much of a probe's distances each frame keeps, from 0 to 1.
    pub depth_hysteresis: f32,
    /// How narrowly each distance texel averages its rays. Higher follows voxels' edges more
    /// closely but is noisier.
    pub depth_sharpness: f32,
    /// How far off a surface probes' visibility is tested, in probe spacings. Higher leaks
    /// less shadow into lit areas, too high leaks light through thin walls.
    pub self_shadow_bias: f32,
    /// How far the finest cascade's probes' rays reach, in the `ShadowGrid`'s finest voxels,
    /// twice as far in each coarser one. Further finds more distant light, at more cost.
    pub ray_distance: f32,
    /// Frames between each probe's updates in the finest cascade, rounded up to a power of
    /// two from 1 to 16, twice as many in each coarser one. The probes take turns, and each
    /// update blends in as much more, so light follows changes as fast, a little noisier, for
    /// a fraction of the cost. Probes near chunks that just changed update every frame.
    pub update_interval: u32,
}

impl Default for ProbeTuning {
    fn default() -> Self {
        Self {
            hysteresis: 0.97,
            depth_hysteresis: 0.97,
            depth_sharpness: 50.0,
            self_shadow_bias: 0.3,
            ray_distance: 512.0,
            update_interval: 4,
        }
    }
}

/// The longest `ProbeTuning::update_interval`.
const MAX_UPDATE_INTERVAL: u32 = 16;

impl Renderer {
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
