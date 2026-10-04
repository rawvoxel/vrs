//! How chunk meshes are lit: light from the sky and the ground, and a sun whose shadows are
//! traced through the voxels.

use super::Renderer;
use crate::{
    gi::{GiUniforms, PROBE_CELL},
    mesh::MeshKind,
    shadows::ShadowUniforms,
};
use ash::prelude::VkResult;
use glam::{IVec3, Mat4, Quat, Vec3};

/// Chunks per side of the table shadow rays find chunks in, centered on the camera.
const SHADOW_TABLE_CHUNKS: i32 = 32;
/// A shadow table entry without a chunk. Matches `NO_CHUNK` in `mesh.slang`.
const NO_CHUNK: u32 = u32::MAX;

/// The chunks shadow rays are traced through: those `voxels_per_edge` voxels wide of
/// `voxel_size`, placed on the grid of chunks that size whose corner is the world's origin.
/// Other chunks are lit, but cast no shadows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowGrid {
    pub voxel_size: f32,
    pub voxels_per_edge: u32,
}

/// How finely sun shadows are traced, see `Lighting::shadow_resolution`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowResolution {
    /// A shadow ray for every pixel, as it's shaded.
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
    /// How finely sun shadows are traced. Below full resolution, frames draw their depth
    /// first, trace a shadow ray for each block of pixels from it, then draw their color over
    /// it, upscaling the shadows. That costs drawing every mesh twice, and makes shadows'
    /// edges a little softer.
    pub shadow_resolution: ShadowResolution,
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
    /// The grid's voxel size, voxels per edge and the shadow distance in voxels, w unused.
    pub shadow_grid: [f32; 4],
}

impl Renderer {
    /// Frame pixels along each axis that share a shadow ray, 1 when shading traces its own
    /// or there are no shadows.
    pub(super) fn shadow_scale(&self) -> u32 {
        match self.lighting.shadow_grid {
            Some(_) => self.lighting.shadow_resolution.pixels_per_ray(),
            None => 1,
        }
    }

    /// Makes the shadow image fit the frame and `shadow_scale`, waiting for the device to go
    /// idle when it has to be replaced.
    pub(super) fn prepare_shadow_pass(&mut self) -> VkResult<()> {
        let scale = self.shadow_scale();
        let frame = self.swapchain.extent;
        let fits = self.shadow_pass.scale == scale
            && self.shadow_pass.extent.width == frame.width.div_ceil(scale)
            && self.shadow_pass.extent.height == frame.height.div_ceil(scale);
        if scale == 1 || fits {
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
        ShadowUniforms {
            inverse_view_proj: view_proj.inverse(),
            params: [self.shadow_scale(), extent.width, extent.height, 0],
            frame_size: [frame.width, frame.height, 0, 0],
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
        let mut uniforms = GiUniforms {
            volume_counts: [
                counts[0],
                counts[1],
                counts[2],
                self.probe_volume.atlas_columns,
            ],
            ray_rotation: random_rotation(self.frame_number).to_array(),
            camera_position: camera.extend(1.0).to_array(),
            frame: [
                self.frame_number as u32,
                self.probe_volume.rays_per_frame,
                0,
                0,
            ],
            ..Default::default()
        };
        if let (true, Some(grid)) = (self.gi_enabled(), self.lighting.shadow_grid) {
            let cell_size = grid.voxel_size * PROBE_CELL as f32;
            let center = (camera / cell_size).floor().as_ivec3();
            let half = IVec3::from_array(counts.map(|count| count as i32 / 2));
            uniforms.volume_min = (center - half).extend(1).to_array();
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
            ..Default::default()
        };
        let Some(grid) = lighting.shadow_grid else {
            return Ok(uniforms);
        };

        let chunk_size = grid.voxel_size * grid.voxels_per_edge as f32;
        let camera = view.inverse().w_axis.truncate();
        let corner =
            (camera / chunk_size).floor().as_ivec3() - IVec3::splat(SHADOW_TABLE_CHUNKS / 2);
        let side = SHADOW_TABLE_CHUNKS as usize;
        self.shadow_table.clear();
        self.shadow_table.resize(side.pow(3), NO_CHUNK);
        for (slot, mesh) in self.mesh_pool.meshes() {
            let same_size = mesh.voxels_per_edge == grid.voxels_per_edge
                && (mesh.scale - grid.voxel_size).abs() <= grid.voxel_size * 1e-5;
            if mesh.kind == MeshKind::Colored || !same_size {
                continue;
            }
            let position = mesh.origin / chunk_size;
            let chunk = position.round();
            if (position - chunk).abs().max_element() > 1e-3 {
                continue;
            }
            let relative = chunk.as_ivec3() - corner;
            if relative.min_element() < 0 || relative.max_element() >= SHADOW_TABLE_CHUNKS {
                continue;
            }
            let [x, y, z] = relative.to_array().map(|axis| axis as usize);
            self.shadow_table[(z * side + y) * side + x] = slot;
        }
        unsafe {
            self.frame_shadow_tables[frame].write(
                &self.adapter,
                &self.render_device,
                &self.shadow_table,
            )?;
        }

        uniforms.shadow_table = corner.extend(SHADOW_TABLE_CHUNKS).to_array();
        uniforms.shadow_grid = [
            grid.voxel_size,
            grid.voxels_per_edge as f32,
            lighting.shadow_distance,
            0.0,
        ];
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
