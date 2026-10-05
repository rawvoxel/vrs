//! How chunk meshes are lit: the sky and the ground, or the light probes in their place, a
//! sun and the `Renderer::lights`, whose shadows are traced through the voxels, ambient
//! occlusion, and reflections off glossy voxels, then exposed and tonemapped.

mod debug_view;
pub(crate) mod lights;
pub(crate) mod probes;
mod sky;
mod tonemapping;
pub(crate) mod trace;
pub(crate) mod voxel_grid;

pub use debug_view::DebugView;
pub use lights::{Light, MAX_LIGHTS, Spot};
pub use probes::{MAX_CASCADES, ProbeConfig, ProbeTuning};
pub use sky::Sky;
pub use tonemapping::Tonemapping;
pub use trace::{MAX_SHADOWED_LIGHTS, TraceResolution};
pub use voxel_grid::{MAX_GRID_LEVELS, VoxelGrid};

use super::Renderer;
use ash::prelude::VkResult;
use glam::{Mat4, Vec3};

/// How meshes are lit, see `Renderer::lighting`. Colored meshes are lit as rough surfaces of
/// their own colors.
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
    /// The chunks rays are traced through, `None` for no shadows, reflections or probes.
    pub voxel_grid: Option<VoxelGrid>,
    /// How far the sun's shadow rays and reflection rays reach, in the grid's finest voxels.
    /// The light probes' reach `ProbeTuning::ray_distance`.
    pub trace_distance: f32,
    /// Whether the light probes from `RendererConfig::global_illumination` light chunks, in
    /// place of the sky and ground colors, with light bounced off the voxels around them.
    /// Their rays are traced through the `voxel_grid`, which needs `voxels_per_edge` to be a
    /// multiple of 16. Outside the probes, chunks are lit by the sky and ground colors.
    pub global_illumination: bool,
    /// How the light probes blend what they see and are sampled.
    pub probe_tuning: ProbeTuning,
    /// How much axis aligned chunks' corners and crevices darken the light from everywhere
    /// but the sun, from 0, not at all, to 1, a corner closed on two sides gets none. Read from the
    /// voxels around each face, in neighboring chunks too when they're in the `voxel_grid`.
    pub ambient_occlusion: f32,
    /// What chunk meshes show, to see how they're lit.
    pub debug_view: DebugView,
    /// How finely the trace pass traces. With a voxel grid or lights, frames draw their depth
    /// first, trace a shadow ray for each block of pixels and shadowed light from it, and
    /// reflection rays for glossy voxels, then draw their color over it, upscaling them. Below full
    /// resolution, the light from everywhere but the sun is found for each block too, and
    /// edges are a little softer.
    pub trace_resolution: TraceResolution,
    /// How many of the `Renderer::lights` reaching each pixel are shadowed, its brightest, at
    /// most `MAX_SHADOWED_LIGHTS`. The rest light through voxels.
    pub shadowed_lights: u32,
    /// What's drawn where no mesh is, `None` for the `Renderer::clear_color`.
    pub sky: Option<Sky>,
    /// What shaded colors, and the sky's, are multiplied by before `tonemapping`.
    pub exposure: f32,
    /// How shaded colors brighter than the display shows are brought into its range.
    pub tonemapping: Tonemapping,
}

impl Default for Lighting {
    /// A high afternoon sun under a pale sky, without shadows, drawn over the clear color as
    /// lit.
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(0.45, 1.0, 0.3),
            sun_color: [0.85, 0.8, 0.72],
            sky_color: [0.32, 0.36, 0.44],
            ground_color: [0.16, 0.15, 0.14],
            voxel_grid: None,
            trace_distance: 512.0,
            global_illumination: true,
            trace_resolution: TraceResolution::Full,
            probe_tuning: ProbeTuning::default(),
            ambient_occlusion: 0.6,
            debug_view: DebugView::Off,
            shadowed_lights: 4,
            sky: None,
            exposure: 1.0,
            tonemapping: Tonemapping::None,
        }
    }
}

/// `Lighting` as the shaders read it. Matches `LightingUniforms` in `common/layout.slang`, field
/// for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LightingUniforms {
    /// Normalized, toward the sun.
    pub sun_direction: [f32; 3],
    pub ambient_occlusion: f32,
    pub sun_color: [f32; 3],
    pub debug_view: u32,
    pub sky_color: [f32; 3],
    pub exposure: f32,
    pub ground_color: [f32; 3],
    pub tonemapping: u32,
    /// The voxel grid's finest voxels' size, voxels along each chunk's edge, and levels, 0
    /// without a grid.
    pub grid_voxel_size: f32,
    pub grid_voxels_per_edge: u32,
    pub grid_levels: u32,
    pub trace_distance: f32,
    /// The chunk coordinate of each level's chunk table's low corner, in that level's
    /// chunks, w unused.
    pub chunk_table_corners: [[i32; 4]; MAX_GRID_LEVELS as usize],
    /// The sky, all 0 without one.
    pub sky_zenith: [f32; 3],
    pub sun_disc_cosine: f32,
    pub sky_horizon: [f32; 3],
    pub sun_brightness: f32,
    pub haze_distance: f32,
    pub has_sky: u32,
    pub padding: [f32; 2],
}

impl Renderer {
    /// The lighting for `frame`'s uniforms, filling in its chunk table around the camera and
    /// which chunks changed since last frame.
    ///
    /// # Safety
    /// The device must not be using `frame`'s chunk table or changed chunks.
    pub(super) unsafe fn write_lighting(
        &mut self,
        frame: usize,
        view: Mat4,
    ) -> VkResult<LightingUniforms> {
        let lighting = self.lighting;
        let mut uniforms = LightingUniforms {
            sun_direction: lighting.sun_direction.normalize_or(Vec3::Y).to_array(),
            ambient_occlusion: lighting.ambient_occlusion.clamp(0.0, 1.0),
            sun_color: lighting.sun_color,
            debug_view: lighting.debug_view as u32,
            sky_color: lighting.sky_color,
            exposure: lighting.exposure.max(0.0),
            ground_color: lighting.ground_color,
            tonemapping: lighting.tonemapping as u32,
            trace_distance: lighting.trace_distance,
            ..Default::default()
        };
        if let Some(sky) = lighting.sky {
            sky.write_uniforms(&mut uniforms);
        }
        let Some(grid) = lighting.voxel_grid else {
            return Ok(uniforms);
        };

        let camera = view.inverse().w_axis.truncate();
        let table = unsafe { self.write_chunk_table(frame, grid, camera)? };
        uniforms.grid_voxel_size = grid.voxel_size;
        uniforms.grid_voxels_per_edge = grid.voxels_per_edge;
        uniforms.grid_levels = table.levels;
        for (level, corner) in table.corners[..table.levels as usize].iter().enumerate() {
            uniforms.chunk_table_corners[level] = corner.extend(0).to_array();
        }
        Ok(uniforms)
    }
}
