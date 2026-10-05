//! The frame descriptor set's bindings, which every shader shares, by number. Each matches
//! the `[[vk::binding]]` of the same resource in the shaders.

use ash::vk;

/// `FrameUniforms`.
pub(crate) const UNIFORMS: u32 = 0;
/// The mesh pool's vertices, mesh infos and voxel types.
pub(crate) const VERTICES: u32 = 1;
pub(crate) const MESHES: u32 = 2;
pub(crate) const TYPES: u32 = 3;
/// The materials, by voxel type.
pub(crate) const PALETTE: u32 = 4;
/// Which chunk is where around the camera, for rays.
pub(crate) const SHADOW_TABLE: u32 = 5;
/// The light probes, their atlases sampled, then written, their rays, and those tracing.
pub(crate) const PROBES: u32 = 6;
pub(crate) const PROBE_IRRADIANCE: u32 = 7;
pub(crate) const PROBE_DEPTH: u32 = 8;
pub(crate) const PROBE_IRRADIANCE_STORAGE: u32 = 9;
pub(crate) const PROBE_DEPTH_STORAGE: u32 = 10;
pub(crate) const PROBE_RAYS: u32 = 11;
pub(crate) const ACTIVE_PROBES: u32 = 12;
/// The shadow pass's sun shadows written, the depth they're traced from, and them sampled.
pub(crate) const SHADOWS_STORAGE: u32 = 13;
pub(crate) const DEPTH: u32 = 14;
pub(crate) const SHADOWS: u32 = 15;
/// Which shadow table entries' chunks changed since last frame.
pub(crate) const CHANGED_CHUNKS: u32 = 16;
/// The lights, and those reaching each tile of pixels.
pub(crate) const LIGHTS: u32 = 17;
pub(crate) const TILE_LIGHTS: u32 = 18;
/// The shadow pass's light shadows, reflections, and light from everywhere but the sun,
/// each written, then read.
pub(crate) const LIGHT_SHADOWS_STORAGE: u32 = 19;
pub(crate) const LIGHT_SHADOWS: u32 = 20;
pub(crate) const REFLECTIONS_STORAGE: u32 = 21;
pub(crate) const REFLECTIONS: u32 = 22;
pub(crate) const INDIRECT_STORAGE: u32 = 23;
pub(crate) const INDIRECT: u32 = 24;

/// Each binding's type, by number.
pub(crate) const TYPES_BY_BINDING: [vk::DescriptorType; 25] = {
    use vk::DescriptorType as Type;
    let mut types = [Type::STORAGE_BUFFER; 25];
    types[UNIFORMS as usize] = Type::UNIFORM_BUFFER;
    types[PROBE_IRRADIANCE as usize] = Type::COMBINED_IMAGE_SAMPLER;
    types[PROBE_DEPTH as usize] = Type::COMBINED_IMAGE_SAMPLER;
    types[PROBE_IRRADIANCE_STORAGE as usize] = Type::STORAGE_IMAGE;
    types[PROBE_DEPTH_STORAGE as usize] = Type::STORAGE_IMAGE;
    types[SHADOWS_STORAGE as usize] = Type::STORAGE_IMAGE;
    types[DEPTH as usize] = Type::COMBINED_IMAGE_SAMPLER;
    types[SHADOWS as usize] = Type::COMBINED_IMAGE_SAMPLER;
    types[LIGHT_SHADOWS_STORAGE as usize] = Type::STORAGE_IMAGE;
    types[LIGHT_SHADOWS as usize] = Type::SAMPLED_IMAGE;
    types[REFLECTIONS_STORAGE as usize] = Type::STORAGE_IMAGE;
    types[REFLECTIONS as usize] = Type::SAMPLED_IMAGE;
    types[INDIRECT_STORAGE as usize] = Type::STORAGE_IMAGE;
    types[INDIRECT as usize] = Type::SAMPLED_IMAGE;
    types
};
