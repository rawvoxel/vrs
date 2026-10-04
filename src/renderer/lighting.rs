//! How chunk meshes are lit: light from the sky and the ground, and a sun whose shadows are
//! traced through the voxels.

use super::Renderer;
use crate::{
    gi::{GiUniforms, PROBE_CELL, ProbeTuning},
    lights::{GpuLight, Light},
    mesh::MeshKind,
    shadows::ShadowUniforms,
};
use ash::prelude::VkResult;
use glam::{IVec3, IVec4, Mat4, Quat, Vec3};

/// A while of faster converging probes, see `Renderer::track_lighting_changes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Convergence {
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

/// Chunks per side of each level's table shadow rays find chunks in, centered on the camera.
const SHADOW_TABLE_CHUNKS: i32 = 32;
/// The most levels a `ShadowGrid` has. Matches `MAX_SHADOW_LEVELS` in `common.slang`.
pub const MAX_SHADOW_LEVELS: u32 = 4;
/// A shadow table entry without a chunk. Matches `NO_CHUNK` in `mesh.slang`.
const NO_CHUNK: u32 = u32::MAX;

/// The chunks shadow rays are traced through: those `voxels_per_edge` voxels wide of
/// `voxel_size`, placed on the grid of chunks that size whose corner is the world's origin,
/// and with `levels` over 1, coarser levels of detail of them, each of voxels twice as big as
/// the last, as many to a chunk. Other chunks are lit, but cast no shadows.
///
/// Each level's chunks are found in a table 32 chunks wide around the camera, so coarser
/// levels reach further. Rays go through the finest level's chunk wherever there is one,
/// so where a level's chunks overlap a coarser one's, the finer ones count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowGrid {
    pub voxel_size: f32,
    pub voxels_per_edge: u32,
    /// Levels of detail, from 1 to `MAX_SHADOW_LEVELS`.
    pub levels: u32,
}

impl ShadowGrid {
    /// Chunks of one size, `voxels_per_edge` voxels of `voxel_size` wide.
    pub fn new(voxel_size: f32, voxels_per_edge: u32) -> Self {
        Self {
            voxel_size,
            voxels_per_edge,
            levels: 1,
        }
    }

    /// Levels the shaders trace through.
    fn level_count(self) -> u32 {
        self.levels.clamp(1, MAX_SHADOW_LEVELS)
    }
}

/// How finely sun shadows are traced, see `Lighting::shadow_resolution`.
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

/// What chunk meshes show instead of their colors, for seeing how they're lit, see
/// `Lighting::debug_view`. Matches the `DEBUG_` constants in `mesh.slang`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugView {
    /// Their colors, lit, as usual.
    #[default]
    Off,
    /// All the light reaching them, as if they were white.
    Light,
    /// Only the sun's.
    Direct,
    /// Only the light from everywhere else: the probes', or the sky and ground's.
    Indirect,
    /// How much voxel ambient occlusion leaves of the indirect light.
    AmbientOcclusion,
    /// As usual, with the light probes drawn as small cubes, each face colored with the
    /// light the probe holds from its direction.
    Probes,
}

/// How chunk meshes are lit, see `Renderer::lighting`. Colored meshes keep their own colors.
///
/// A face's light is the sky's and the ground's, blended by how far it faces up or down,
/// plus the sun's when it faces the sun and no voxel is in the way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// Toward the sun, in the world. Needn't be normalized.
    pub sun_direction: Vec3,
    /// In linear color, as are the others.
    pub sun_color: [f32; 3],
    /// What faces facing straight up get from the sky.
    pub sky_color: [f32; 3],
    /// What faces facing straight down get from the ground.
    pub ground_color: [f32; 3],
    /// The chunks that cast shadows, `None` for no shadows.
    pub shadow_grid: Option<ShadowGrid>,
    /// How far shadow rays reach, in the grid's voxels. Light probes' rays reach as far.
    pub shadow_distance: f32,
    /// Whether the light probes from `RendererConfig::global_illumination` light chunks, in
    /// place of the sky and ground colors, with light bounced off the voxels around them.
    /// Their rays are traced through the `shadow_grid`, which needs `voxels_per_edge` to be a
    /// multiple of 16. Outside the probes, chunks are lit by the sky and ground colors.
    pub global_illumination: bool,
    /// How the light probes blend what they see and are sampled.
    pub probe_tuning: ProbeTuning,
    /// How much greedy chunks' corners and crevices darken the light from everywhere but the
    /// sun, from 0, not at all, to 1, a corner closed on two sides gets none. Read from the
    /// voxels around each face, in neighboring chunks too when they're in the `shadow_grid`.
    pub ambient_occlusion: f32,
    /// What chunk meshes show, to see how they're lit.
    pub debug_view: DebugView,
    /// How finely shadows are traced. With shadows, frames draw their depth first, trace a
    /// shadow ray for each block of pixels and shadowed light from it, then draw their color
    /// over it, upscaling the shadows. Below full resolution, shadows' edges are a little
    /// softer.
    pub shadow_resolution: ShadowResolution,
    /// How many of the `Renderer::lights` reaching each pixel are shadowed, its brightest, at
    /// most 4. The rest light through voxels.
    pub shadowed_lights: u32,
}

impl Default for Lighting {
    /// A high afternoon sun under a pale sky, without shadows.
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(0.45, 1.0, 0.3),
            sun_color: [0.85, 0.8, 0.72],
            sky_color: [0.32, 0.36, 0.44],
            ground_color: [0.16, 0.15, 0.14],
            shadow_grid: None,
            shadow_distance: 512.0,
            global_illumination: true,
            shadow_resolution: ShadowResolution::Full,
            probe_tuning: ProbeTuning::default(),
            ambient_occlusion: 0.6,
            debug_view: DebugView::Off,
            shadowed_lights: 4,
        }
    }
}

/// `Lighting` as `mesh.slang` reads it. Matches `LightingUniforms` there.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LightingUniforms {
    /// Normalized, w unused, as is every color's.
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_color: [f32; 4],
    pub ground_color: [f32; 4],
    /// The chunk coordinate of the shadow table's corner, and chunks per side, 0 without shadows.
    pub shadow_table: [i32; 4],
    /// The grid's voxel size, voxels per edge and the shadow distance in voxels, then how many
    /// levels it has.
    pub shadow_grid: [f32; 4],
    /// The ambient occlusion's strength, then the debug view's number, the rest unused.
    pub options: [f32; 4],
    /// The chunk coordinate of each level's shadow table's corner, in that level's chunks, w
    /// unused.
    pub shadow_levels: [[i32; 4]; MAX_SHADOW_LEVELS as usize],
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
    pub(super) fn track_lighting_changes(&mut self) {
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

    /// Frame pixels along each axis that share a shadow ray, 1 without shadows.
    pub(super) fn shadow_scale(&self) -> u32 {
        match self.lighting.shadow_grid {
            Some(_) => self.lighting.shadow_resolution.pixels_per_ray(),
            None => 1,
        }
    }

    /// Whether frames draw their depth first, for the shadow pass to trace shadows and cull
    /// lights from, then their color over it. Even at full resolution, that's faster than
    /// tracing shadows as pixels are shaded, which wastes rays on pixels drawn over.
    pub(super) fn uses_shadow_pass(&self) -> bool {
        self.lighting.shadow_grid.is_some() || !self.lights.is_empty()
    }

    /// Makes the shadow images fit the frame and `shadow_scale`, waiting for the device to go
    /// idle when they have to be replaced.
    pub(super) fn prepare_shadow_pass(&mut self) -> VkResult<()> {
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
    pub(super) fn shadow_uniforms(&self, view_proj: Mat4) -> ShadowUniforms {
        let extent = self.shadow_pass.extent;
        let frame = self.swapchain.extent;
        let shadowed = match self.lighting.shadow_grid {
            Some(_) => self.lighting.shadowed_lights.min(4),
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
            frame_size: [frame.width, frame.height, self.uses_shadow_pass() as u32, 0],
            lights: [self.light_count(), tiles_x, tiles_y, shadowed],
        }
    }

    /// How many of `lights` light the frame.
    pub(super) fn light_count(&self) -> u32 {
        self.lights.len().min(crate::lights::MAX_LIGHTS) as u32
    }

    /// Writes the frame's lights.
    ///
    /// # Safety
    /// The device must not be using `frame`'s light buffer.
    pub(super) unsafe fn write_lights(&mut self, frame: usize) -> VkResult<()> {
        // An empty buffer can't be bound, a placeholder entry stands in.
        let mut lights: Vec<GpuLight> = self.lights[..self.light_count() as usize]
            .iter()
            .map(GpuLight::from)
            .collect();
        if lights.is_empty() {
            lights.push(GpuLight::from(&Light {
                position: Vec3::ZERO,
                color: [0.0; 3],
                range: 1.0,
                spot: None,
            }));
        }
        unsafe {
            self.frame_light_buffers[frame].write(&self.adapter, &self.render_device, &lights)
        }
    }

    /// Whether this frame updates and samples the light probes, see `Lighting::global_illumination`.
    pub(super) fn gi_enabled(&self) -> bool {
        self.gi_configured
            && self.lighting.global_illumination
            && self
                .lighting
                .shadow_grid
                .is_some_and(|grid| grid.voxels_per_edge.is_multiple_of(PROBE_CELL))
    }

    /// The light probes' uniforms, their volume centered on the camera.
    pub(super) fn gi_uniforms(&self, view: Mat4) -> GiUniforms {
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
                0,
            ],
            ..Default::default()
        };
        if let (true, Some(grid)) = (self.gi_enabled(), self.lighting.shadow_grid) {
            for cascade in 0..cascades {
                let cell_size = grid.voxel_size * (PROBE_CELL << cascade) as f32;
                let center = (camera / cell_size).floor().as_ivec3();
                let half = IVec3::from_array(counts.map(|count| count as i32 / 2));
                uniforms.volume_mins[cascade as usize] = (center - half).extend(1).to_array();
            }
        }
        uniforms
    }

    /// Fills in `frame`'s shadow table around the camera, from the chunks in the mesh pool,
    /// and returns the lighting for its uniforms.
    ///
    /// # Safety
    /// The device must not be using `frame`'s shadow table.
    pub(super) unsafe fn write_lighting(
        &mut self,
        frame: usize,
        view: Mat4,
    ) -> VkResult<LightingUniforms> {
        let lighting = self.lighting;
        let color = |[r, g, b]: [f32; 3]| [r, g, b, 0.0];
        let mut uniforms = LightingUniforms {
            sun_direction: lighting
                .sun_direction
                .normalize_or(Vec3::Y)
                .extend(0.0)
                .into(),
            sun_color: color(lighting.sun_color),
            sky_color: color(lighting.sky_color),
            ground_color: color(lighting.ground_color),
            options: [
                lighting.ambient_occlusion.clamp(0.0, 1.0),
                lighting.debug_view as u32 as f32,
                0.0,
                0.0,
            ],
            ..Default::default()
        };
        let Some(grid) = lighting.shadow_grid else {
            return Ok(uniforms);
        };

        let levels = grid.level_count();
        let camera = view.inverse().w_axis.truncate();
        // Each level's table's corner, in that level's chunks.
        let corners: Vec<IVec3> = (0..levels)
            .map(|level| {
                let chunk_size = grid.voxel_size * (grid.voxels_per_edge << level) as f32;
                (camera / chunk_size).floor().as_ivec3() - IVec3::splat(SHADOW_TABLE_CHUNKS / 2)
            })
            .collect();
        let side = SHADOW_TABLE_CHUNKS as usize;
        // A table entry's index, from its level and chunk, when the table holds it.
        let entry = |chunk: IVec4| {
            let relative = chunk.truncate() - corners[chunk.w as usize];
            let inside =
                relative.min_element() >= 0 && relative.max_element() < SHADOW_TABLE_CHUNKS;
            inside.then(|| {
                let [x, y, z] = relative.to_array().map(|axis| axis as usize);
                ((chunk.w as usize * side + z) * side + y) * side + x
            })
        };
        self.shadow_table.clear();
        self.shadow_table
            .resize(levels as usize * side.pow(3), NO_CHUNK);
        for (slot, mesh) in self.mesh_pool.meshes() {
            if mesh.kind == MeshKind::Colored || mesh.voxels_per_edge != grid.voxels_per_edge {
                continue;
            }
            // The level whose voxels are the mesh's size.
            let ratio = mesh.scale / grid.voxel_size;
            let level = ratio.log2().round();
            let level_size = level.exp2();
            if !(0.0..levels as f32).contains(&level)
                || (ratio - level_size).abs() > level_size * 1e-5
            {
                continue;
            }
            let chunk_size = grid.voxel_size * level_size * grid.voxels_per_edge as f32;
            let position = mesh.origin / chunk_size;
            let chunk = position.round();
            if (position - chunk).abs().max_element() > 1e-3 {
                continue;
            }
            let chunk = chunk.as_ivec3().extend(level as i32);
            if let Some(index) = entry(chunk) {
                self.shadow_table[index] = slot;
                self.chunk_serials_next.insert(chunk, mesh.serial);
            }
        }
        // Chunks added, removed or remeshed since last frame, a bit for each table entry, so
        // the probes around them converge faster.
        let previous = std::mem::take(&mut self.chunk_serials);
        self.changed_chunks.clear();
        self.changed_chunks
            .resize((levels as usize * side.pow(3)).div_ceil(32), 0);
        let mut changed = |chunk: IVec4| {
            if (chunk.w as u32) < levels
                && let Some(index) = entry(chunk)
            {
                self.changed_chunks[index / 32] |= 1 << (index % 32);
            }
        };
        for (&chunk, serial) in &self.chunk_serials_next {
            if previous.get(&chunk) != Some(serial) {
                changed(chunk);
            }
        }
        for chunk in previous.keys() {
            if !self.chunk_serials_next.contains_key(chunk) {
                changed(*chunk);
            }
        }
        self.chunk_serials = std::mem::replace(&mut self.chunk_serials_next, previous);
        self.chunk_serials_next.clear();
        unsafe {
            self.frame_shadow_tables[frame].write(
                &self.adapter,
                &self.render_device,
                &self.shadow_table,
            )?;
            self.frame_changed_chunks[frame].write(
                &self.adapter,
                &self.render_device,
                &self.changed_chunks,
            )?;
        }

        uniforms.shadow_table = corners[0].extend(SHADOW_TABLE_CHUNKS).to_array();
        uniforms.shadow_grid = [
            grid.voxel_size,
            grid.voxels_per_edge as f32,
            lighting.shadow_distance,
            levels as f32,
        ];
        for (level, corner) in corners.iter().enumerate() {
            uniforms.shadow_levels[level] = corner.extend(0).to_array();
        }
        Ok(uniforms)
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
