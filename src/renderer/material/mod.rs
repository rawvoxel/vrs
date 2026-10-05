//! How voxels of a type look and light.

mod palette;

/// Materials below this roughness reflect sharply. Matches `GLOSSY_ROUGHNESS` in
/// `common/shading.slang`.
pub(super) const GLOSSY_ROUGHNESS: f32 = 0.5;

/// How voxels of a type look and light, see `Renderer::set_materials`. Colors are linear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// The surface's color, which light reflects off and, for see-through voxels, passes
    /// through tinted by.
    pub color: [f32; 3],
    /// The light it gives off itself, whatever its color, lighting what's around it through
    /// the light probes. Over 1 glows brighter.
    pub emission: [f32; 3],
    /// How rough it is, from 0, a mirror, to 1, matte. Below 0.5 it's glossy: it reflects
    /// what's around it, more sharply the smoother it is, and the sun's and lights' highlights.
    pub roughness: f32,
    /// How metallic it is, from 0 to 1. Metals reflect in their own color, and reflect
    /// rather than scatter their light.
    pub metalness: f32,
    /// Whether light passes through it, tinted by its color, such as glass. It's still drawn
    /// as solid faces.
    pub see_through: bool,
}

impl Material {
    /// A matte, opaque material of `color`, giving off no light.
    pub const fn color(color: [f32; 3]) -> Self {
        Self {
            color,
            emission: [0.0; 3],
            roughness: 1.0,
            metalness: 0.0,
            see_through: false,
        }
    }
}

impl Default for Material {
    /// Matte, opaque light gray.
    fn default() -> Self {
        Self::color([0.8; 3])
    }
}

/// A `Material` as the shaders read it. Matches `Material` in `common/layout.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct GpuMaterial {
    /// The color, then the roughness.
    color_roughness: [f32; 4],
    /// The emission, then the metalness, plus 2 when it's see-through.
    emission_metalness: [f32; 4],
}

impl From<&Material> for GpuMaterial {
    fn from(material: &Material) -> Self {
        let [r, g, b] = material.color;
        let [er, eg, eb] = material.emission;
        // Flagged by adding 2 to the metalness. Matches `seeThrough` in `common/materials.slang`.
        let see_through = if material.see_through { 2.0 } else { 0.0 };
        Self {
            color_roughness: [r, g, b, material.roughness.clamp(0.0, 1.0)],
            emission_metalness: [er, eg, eb, material.metalness.clamp(0.0, 1.0) + see_through],
        }
    }
}
