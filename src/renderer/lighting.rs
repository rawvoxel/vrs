//! How chunk meshes are lit: the sky and the ground, or the light probes in their place, a
//! sun and the `Renderer::lights`, whose shadows are traced through the voxels, ambient
//! occlusion, and reflections off glossy voxels.

use super::Renderer;
use crate::{
    gi::{GiUniforms, PROBE_CELL, ProbeTuning},
    lights::GpuLight,
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
/// The most of each pixel's lights that are shadowed, see `Lighting::shadowed_lights`. Each
/// shadow texel keeps that many, 8 bits each. Matches `MAX_SHADOWED_LIGHTS` in
/// `common.slang`.
pub const MAX_SHADOWED_LIGHTS: u32 = 4;
/// The longest `ProbeTuning::update_interval`.
const MAX_UPDATE_INTERVAL: u32 = 16;
/// A shadow table entry without a chunk. Matches `NO_CHUNK` in `common.slang`.
const NO_CHUNK: u32 = u32::MAX;

/// The chunks shadow rays are traced through: those `voxels_per_edge` voxels wide of
/// `voxel_size`, placed on the grid of chunks that size whose corner is the world's origin,
/// and with `levels` over 1, coarser levels of detail of them, each of voxels twice as big as
/// the last, as many to a chunk. Other chunks are lit, but cast no shadows.
///
/// Each level's chunks are found in a table 32 chunks wide around the camera, so coarser
/// levels reach further. Rays go through the finest level's chunk wherever there is one, so
/// where a level's chunks overlap a coarser one's, the finer ones count.
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

    /// With `levels` levels of detail.
    pub fn with_levels(self, levels: u32) -> Self {
        Self { levels, ..self }
    }

    /// Levels the shaders trace through.
    fn level_count(self) -> u32 {
        self.levels.clamp(1, MAX_SHADOW_LEVELS)
    }
}

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

/// What chunk meshes show instead of their colors, for seeing how they're lit, see
/// `Lighting::debug_view`. Matches the `DEBUG_` constants in `mesh.slang`, but for `Probes`,
/// which shades chunks as usual.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugView {
    /// Their colors, lit, as usual.
    #[default]
    Off,
    /// All the light reaching them, as if they were white.
    Light,
    /// Only the sun's and the lights'.
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
/// A face's light is the sky's and the ground's, blended by how far it faces up or down, or
/// with `global_illumination`, the light probes' in their place, darkened in corners by
/// ambient occlusion. Then the sun's and the `Renderer::lights`' where they face it and no
/// voxel is in the way, and with a glossy `Material`, what it reflects.
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
    /// How far the sun's shadow rays and reflection rays reach, in the grid's finest voxels.
    /// The light probes' reach `ProbeTuning::ray_distance`.
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
    /// How finely shadows are traced. With shadows or lights, frames draw their depth first,
    /// trace a shadow ray for each block of pixels and shadowed light from it, and reflection
    /// rays for glossy voxels, then draw their color over it, upscaling them. Below full
    /// resolution, the light from everywhere but the sun is found for each block too, and
    /// edges are a little softer.
    pub shadow_resolution: ShadowResolution,
    /// How many of the `Renderer::lights` reaching each pixel are shadowed, its brightest, at
    /// most `MAX_SHADOWED_LIGHTS`. The rest light through voxels.
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

/// `Lighting` as the shaders read it. Matches `LightingUniforms` in `common.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LightingUniforms {
    /// Normalized, w unused, as is every color's.
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_color: [f32; 4],
    pub ground_color: [f32; 4],
    /// The finest level's table's corner, which the shaders read from `shadow_levels`, and
    /// chunks per side of each level's, 0 without shadows.
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
    pub(super) fn lights_indirect_in_shadow_pass(&self) -> bool {
        let indirect_work = self.gi_enabled() || self.lighting.ambient_occlusion > 0.0;
        self.uses_shadow_pass()
            && self.shadow_scale() > 1
            && indirect_work
            && self.lighting.debug_view != DebugView::AmbientOcclusion
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
            self.frame_light_buffers[frame].write(
                &self.adapter,
                &self.render_device,
                &self.gpu_lights,
            )
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

    /// The lighting for `frame`'s uniforms, filling in its shadow table around the camera and
    /// which chunks changed since last frame.
    ///
    /// # Safety
    /// The device must not be using `frame`'s shadow table or changed chunks.
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

        let table = TableLayout::around(grid, view.inverse().w_axis.truncate());
        self.fill_shadow_table(grid, &table);
        self.mark_changed_chunks(&table);
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

        uniforms.shadow_table = table.corners[0].extend(SHADOW_TABLE_CHUNKS).to_array();
        uniforms.shadow_grid = [
            grid.voxel_size,
            grid.voxels_per_edge as f32,
            lighting.shadow_distance,
            table.levels as f32,
        ];
        for (level, corner) in table.corners[..table.levels as usize].iter().enumerate() {
            uniforms.shadow_levels[level] = corner.extend(0).to_array();
        }
        Ok(uniforms)
    }

    /// Puts each chunk of `grid`'s levels in the mesh pool in its `table` entry, noting its
    /// serial for `mark_changed_chunks`.
    fn fill_shadow_table(&mut self, grid: ShadowGrid, table: &TableLayout) {
        self.shadow_table.clear();
        self.shadow_table.resize(table.entries(), NO_CHUNK);
        for (slot, mesh) in self.mesh_pool.meshes() {
            if mesh.kind == MeshKind::Colored || mesh.voxels_per_edge != grid.voxels_per_edge {
                continue;
            }
            // The level whose voxels are the mesh's size.
            let ratio = mesh.scale / grid.voxel_size;
            let level = ratio.log2().round();
            let level_size = level.exp2();
            if !(0.0..table.levels as f32).contains(&level)
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
            if let Some(index) = table.entry(chunk) {
                self.shadow_table[index] = slot;
                self.chunk_serials_next.insert(chunk, mesh.serial);
            }
        }
    }

    /// Sets a bit for each `table` entry whose chunk was added, removed or remeshed since last
    /// frame, so the probes around it converge faster.
    fn mark_changed_chunks(&mut self, table: &TableLayout) {
        let previous = std::mem::take(&mut self.chunk_serials);
        self.changed_chunks.clear();
        self.changed_chunks.resize(table.entries().div_ceil(32), 0);
        let mut changed = |chunk: IVec4| {
            if let Some(index) = table.entry(chunk) {
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
    }
}

/// Where each level's shadow table is, `SHADOW_TABLE_CHUNKS` chunks wide around the camera.
struct TableLayout {
    levels: u32,
    /// Each level's corner, in that level's chunks.
    corners: [IVec3; MAX_SHADOW_LEVELS as usize],
}

impl TableLayout {
    fn around(grid: ShadowGrid, camera: Vec3) -> Self {
        Self {
            levels: grid.level_count(),
            corners: std::array::from_fn(|level| {
                let chunk_size = grid.voxel_size * (grid.voxels_per_edge << level) as f32;
                (camera / chunk_size).floor().as_ivec3() - IVec3::splat(SHADOW_TABLE_CHUNKS / 2)
            }),
        }
    }

    /// Every level's entries.
    fn entries(&self) -> usize {
        self.levels as usize * (SHADOW_TABLE_CHUNKS as usize).pow(3)
    }

    /// The entry of a chunk, by its coordinate and then level, when the table holds it.
    fn entry(&self, chunk: IVec4) -> Option<usize> {
        if !(0..self.levels as i32).contains(&chunk.w) {
            return None;
        }
        let relative = chunk.truncate() - self.corners[chunk.w as usize];
        let inside = relative.min_element() >= 0 && relative.max_element() < SHADOW_TABLE_CHUNKS;
        let side = SHADOW_TABLE_CHUNKS as usize;
        inside.then(|| {
            let [x, y, z] = relative.to_array().map(|axis| axis as usize);
            ((chunk.w as usize * side + z) * side + y) * side + x
        })
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
