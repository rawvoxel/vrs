//! Dynamic diffuse global illumination: cascades of light probes around the camera, each
//! storing the light reaching it from every direction and how far away the nearest voxels
//! are, those near voxels traced through them in turns and sampled when shading. See
//! `probe_update.slang`.

mod cells;
mod convergence;
mod volume;

use super::VoxelGrid;
use crate::renderer::Renderer;
use ash::prelude::VkResult;
use glam::{IVec3, IVec4, Mat4, Quat, Vec3};

pub(crate) use cells::bake_probe_cells;
pub(in crate::renderer) use convergence::Convergence;
pub(crate) use volume::ProbeVolume;

/// Voxels along each edge of a probe's cell, and between neighboring probes, in the finest
/// cascade. Matches `PROBE_CELL` in `common/probes.slang`.
pub const PROBE_CELL: u32 = 16;
/// The most cascades of probes. Matches `MAX_CASCADES` in `common/layout.slang`.
pub const MAX_CASCADES: u32 = 4;

/// The light probe volumes, see `RendererConfig::global_illumination`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeConfig {
    /// Probes along each axis of each cascade, which is centered on the camera with probes
    /// `spacing` apart in the first, and twice as far apart in each next. Each probe takes
    /// about 3.4 KB, see `device_bytes`.
    pub probe_counts: [u32; 3],
    /// The `VoxelGrid`'s finest voxels between the first cascade's neighboring probes:
    /// `PROBE_CELL`, or that times a power of two, rounded down to one. Probes `PROBE_CELL`
    /// voxels of a level apart are placed by that level's chunks, so further apart they're
    /// placed by a coarser level's, and their rays reach further in proportion.
    pub spacing: u32,
    /// Cascades of probes, from 1 to `MAX_CASCADES`. Each reaches twice as far as the last,
    /// with probes twice as far apart, for chunks of coarser `VoxelGrid::levels`. Surfaces
    /// are lit by the finest one around them.
    pub cascades: u32,
    /// Rays traced each frame, shared between every cascade's probes near voxels so the cost
    /// stays steady: each traces a power of two of them from 32 to 256. Past
    /// `rays_per_frame / 32` such probes, each still traces 32.
    pub rays_per_frame: u32,
}

impl Default for ProbeConfig {
    /// One cascade, 512 by 256 by 512 voxels, with 64 Ki rays a frame.
    fn default() -> Self {
        Self {
            probe_counts: [32, 16, 32],
            spacing: PROBE_CELL,
            cascades: 1,
            rays_per_frame: 1 << 16,
        }
    }
}

impl ProbeConfig {
    /// Probes in every cascade, as the renderer makes them: at least one along each axis, in
    /// 1 to `MAX_CASCADES` cascades.
    pub fn probe_count(&self) -> u32 {
        let counts = self.probe_counts.map(|count| count.max(1));
        counts.iter().product::<u32>() * self.cascades.clamp(1, MAX_CASCADES)
    }

    /// Bytes of device memory the probes take: their light and distance atlases, and their
    /// buffers and rays.
    pub fn device_bytes(&self) -> u64 {
        volume::device_bytes(self.probe_count(), self.rays_per_frame)
    }
}

/// `Lighting::global_illumination` as the shaders read it. Matches `ProbeUniforms` in
/// `common/layout.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ProbeUniforms {
    /// Each cascade's probe cell at its low corner, in its own cells, then the level of the
    /// voxel grid whose `PROBE_CELL` voxels its cells are, see `ProbeConfig::spacing`.
    pub volume_mins: [[i32; 4]; MAX_CASCADES as usize],
    /// Probes along each axis of each cascade.
    pub volume_counts: [u32; 3],
    pub cascades: u32,
    /// Probes per row of the atlases.
    pub atlas_columns: u32,
    pub frame_number: u32,
    pub rays_per_frame: u32,
    /// Frames between the finest cascade's probes' updates.
    pub update_interval: u32,
    /// This frame's random rotation of the probes' rays, a quaternion.
    pub ray_rotation: [f32; 4],
    /// `ProbeTuning`'s.
    pub hysteresis: f32,
    pub distance_hysteresis: f32,
    pub distance_sharpness: f32,
    pub self_shadow_bias: f32,
    pub ray_distance: f32,
    /// This frame's multipliers of the hysteresis for light and for distances, below 1 while
    /// the probes converge faster after a change.
    pub hysteresis_scale: f32,
    pub distance_hysteresis_scale: f32,
    /// 1 when the probes light the frame.
    pub enabled: u32,
}

/// How the light probes blend what they see and are sampled, see `Lighting::probe_tuning`.
/// The hysteresis, distance sharpness and self-shadow bias default to the papers' [Majercik et
/// al. 2019 and 2021].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeTuning {
    /// How much of a probe's light each frame keeps, from 0 to 1. Higher is steadier but
    /// slower to follow changes.
    pub hysteresis: f32,
    /// How much of a probe's distances each frame keeps, from 0 to 1.
    pub distance_hysteresis: f32,
    /// How narrowly each distance texel averages its rays. Higher follows voxels' edges more
    /// closely but is noisier.
    pub distance_sharpness: f32,
    /// How far off a surface probes' visibility is tested, in probe spacings. Higher leaks
    /// less shadow into lit areas, too high leaks light through thin walls.
    pub self_shadow_bias: f32,
    /// How far probes' rays reach, in the `VoxelGrid`'s finest voxels, with probes
    /// `PROBE_CELL` of them apart, and in proportion with probes further apart, as in coarser
    /// cascades. Further finds more distant light, at more cost.
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
            distance_hysteresis: 0.97,
            distance_sharpness: 50.0,
            self_shadow_bias: 0.3,
            ray_distance: 512.0,
            update_interval: 4,
        }
    }
}

/// The longest `ProbeTuning::update_interval`.
const MAX_UPDATE_INTERVAL: u32 = 16;
/// Past this many chunks changing in a frame, as when a world is loaded, every probe converges
/// faster, rather than finding those next to each.
const MANY_CHANGED_CHUNKS: usize = 1024;

impl Renderer {
    /// The light probes', `None` without any, see `set_probe_config`.
    pub fn probe_config(&self) -> Option<ProbeConfig> {
        self.probe_config
    }

    /// Replaces the light probes with `config`'s, `None` for none, as when what they need to
    /// reach changes. Waits for the device to go idle. The new probes start over, so for a few
    /// frames, as on the first ones, until they've seen enough, surfaces get the sky's light
    /// instead of theirs. Nothing changes when `config` is the same.
    pub fn set_probe_config(&mut self, config: Option<ProbeConfig>) -> VkResult<()> {
        if config == self.probe_config {
            return Ok(());
        }
        unsafe { self.device.device_wait_idle()? };
        // Made first, so the old ones are kept should that fail.
        let volume = ProbeVolume::new(
            &self.adapter,
            &self.device,
            config,
            self.graphics_pipeline_layout,
        )?;
        let mut old = std::mem::replace(&mut self.probe_volume, volume);
        unsafe { old.destroy(&self.device) };
        self.probe_volume
            .write_descriptors(&self.device, &self.frame_desc_sets);
        self.probe_config = config;
        self.init_probe_volume()
    }

    /// Whether this frame updates and samples the light probes, see `Lighting::global_illumination`.
    pub(in crate::renderer) fn probes_enabled(&self) -> bool {
        self.probe_config.is_some()
            && self.lighting.global_illumination
            && self
                .lighting
                .voxel_grid
                .is_some_and(|grid| grid.voxels_per_edge.is_multiple_of(PROBE_CELL))
    }

    /// The light probes' uniforms, their volume centered on the camera.
    pub(in crate::renderer) fn probe_uniforms(&self, view: Mat4) -> ProbeUniforms {
        let camera = view.inverse().w_axis.truncate();
        let counts = self.probe_volume.counts;
        let cascades = self.probe_volume.cascades;
        let tuning = self.lighting.probe_tuning;
        let mut uniforms = ProbeUniforms {
            volume_counts: counts,
            cascades,
            atlas_columns: self.probe_volume.atlas_columns,
            frame_number: self.frame_number as u32,
            rays_per_frame: self.probe_volume.rays_per_frame,
            update_interval: tuning
                .update_interval
                .clamp(1, MAX_UPDATE_INTERVAL)
                .next_power_of_two(),
            ray_rotation: random_rotation(self.frame_number).to_array(),
            hysteresis: tuning.hysteresis.clamp(0.0, 1.0),
            distance_hysteresis: tuning.distance_hysteresis.clamp(0.0, 1.0),
            distance_sharpness: tuning.distance_sharpness.max(1.0),
            self_shadow_bias: tuning.self_shadow_bias.max(0.0),
            ray_distance: tuning.ray_distance.max(0.0),
            hysteresis_scale: self.hysteresis_scale[0],
            distance_hysteresis_scale: self.hysteresis_scale[1],
            ..Default::default()
        };
        if self.probes_enabled()
            && let Some(grid) = self.lighting.voxel_grid
        {
            uniforms.enabled = 1;
            for cascade in 0..cascades {
                let (volume_min, level) = self.cascade_volume(grid, camera, cascade);
                uniforms.volume_mins[cascade as usize] = volume_min.extend(level as i32).to_array();
            }
        }
        uniforms
    }

    /// Cascade `cascade`'s probe cell at its low corner, in its own cells, its volume centered
    /// on `camera`, and the level of `grid` whose `PROBE_CELL` voxels its cells are.
    fn cascade_volume(&self, grid: VoxelGrid, camera: Vec3, cascade: u32) -> (IVec3, u32) {
        let level = self.probe_volume.first_level + cascade;
        let cell_size = grid.voxel_size * (PROBE_CELL << level) as f32;
        let center = (camera / cell_size).floor().as_ivec3();
        let half = IVec3::from_array(self.probe_volume.counts.map(|count| count as i32 / 2));
        (center - half, level)
    }

    /// Sets a bit for each light probe over whose cell or a neighboring one a chunk in
    /// `changed` is, by chunk coordinate, then level, so it converges faster. By probe slot,
    /// as `classifyProbes` reads them, with the volumes centered on `camera`.
    pub(in crate::renderer) fn mark_changed_probes(
        &mut self,
        grid: VoxelGrid,
        camera: Vec3,
        changed: &[IVec4],
    ) {
        let probes = self.probe_volume.probe_count() as usize;
        self.changed_probes.clear();
        self.changed_probes.resize(probes.div_ceil(32), 0);
        if !self.probes_enabled() || changed.is_empty() {
            return;
        }
        if changed.len() > MANY_CHANGED_CHUNKS {
            self.changed_probes.fill(u32::MAX);
            return;
        }
        let counts = IVec3::from_array(self.probe_volume.counts.map(|count| count as i32));
        let per_cascade = counts.element_product();
        for cascade in 0..self.probe_volume.cascades {
            let (volume_min, level) = self.cascade_volume(grid, camera, cascade);
            let volume_max = volume_min + counts - 1;
            let spacing = IVec3::splat((PROBE_CELL << level) as i32);
            for chunk in changed {
                // In the finest voxels, then the cells a cell away or nearer.
                let size = (grid.voxels_per_edge << chunk.w) as i32;
                let chunk_min = chunk.truncate() * size;
                let low = (chunk_min.div_euclid(spacing) - 1).max(volume_min);
                let high = ((chunk_min + (size - 1)).div_euclid(spacing) + 1).min(volume_max);
                for z in low.z..=high.z {
                    for y in low.y..=high.y {
                        for x in low.x..=high.x {
                            let local = IVec3::new(x, y, z).rem_euclid(counts);
                            let slot = cascade as i32 * per_cascade
                                + local.x
                                + counts.x * (local.y + counts.y * local.z);
                            self.changed_probes[slot as usize / 32] |= 1 << (slot % 32);
                        }
                    }
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_probes_device_bytes() {
        // 16 Ki probes in 128 by 128 tiles of 10 and 16 texels a side, 8 bytes each, 32 bytes
        // each, 512 Ki rays of 16 bytes, 32 of them each, and an active list of 8 words and
        // one each.
        let atlases = 1280 * 1280 * 8 + 2048 * 2048 * 8;
        let buffers = 16384 * 32 + 524_288 * 16 + (8 + 16384) * 4;
        assert_eq!(ProbeConfig::default().device_bytes(), atlases + buffers);
    }
}
