//! How chunk meshes are lit: the sky and the ground, or the light probes in their place, a
//! sun and the `Renderer::lights`, whose shadows are traced through the voxels, ambient
//! occlusion, and reflections off glossy voxels, then exposed and tonemapped.

mod debug_view;
mod lights;
mod probes;
mod shadow_grid;
mod shadow_pass;
mod sky;
mod tonemapping;

pub use debug_view::DebugView;
pub(super) use probes::Convergence;
pub use shadow_grid::{MAX_SHADOW_LEVELS, ShadowGrid};
pub use shadow_pass::{MAX_SHADOWED_LIGHTS, ShadowResolution};
pub use sky::Sky;
pub use tonemapping::Tonemapping;

use super::Renderer;
use crate::lighting::gi::ProbeTuning;
use ash::prelude::VkResult;
use glam::{Mat4, Vec3};
use shadow_grid::SHADOW_TABLE_CHUNKS;

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
            shadow_grid: None,
            shadow_distance: 512.0,
            global_illumination: true,
            shadow_resolution: ShadowResolution::Full,
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

/// `Lighting` as the shaders read it. Matches `LightingUniforms` in `common/layout.slang`.
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
    /// The ambient occlusion's strength, then the debug view's number, the exposure and the
    /// tonemapping's number.
    pub options: [f32; 4],
    /// The chunk coordinate of each level's shadow table's corner, in that level's chunks, w
    /// unused.
    pub shadow_levels: [[i32; 4]; MAX_SHADOW_LEVELS as usize],
    /// The sky's zenith and horizon colors, then the cosine of the sun's radius, its
    /// brightness, the haze distance, and 1 when there's a sky. All 0 without.
    pub sky: [[f32; 4]; 3],
}

impl Renderer {
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
                lighting.exposure.max(0.0),
                lighting.tonemapping as u32 as f32,
            ],
            sky: lighting.sky.map(|sky| sky.uniforms()).unwrap_or_default(),
            ..Default::default()
        };
        let Some(grid) = lighting.shadow_grid else {
            return Ok(uniforms);
        };

        let camera = view.inverse().w_axis.truncate();
        let table = unsafe { self.write_shadow_table(frame, grid, camera)? };

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
}
