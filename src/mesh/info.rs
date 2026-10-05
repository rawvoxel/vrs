//! A mesh as the GPU reads it.

use super::Mesh;

impl Mesh {
    #[inline]
    pub(crate) fn info(&self) -> MeshInfo {
        MeshInfo {
            bounds_min: self.bounds_min.to_array(),
            index_count: self.index_count(),
            bounds_max: self.bounds_max.to_array(),
            first_index: self.indices.start,
            origin: self.origin.to_array(),
            scale: self.scale,
            vertex_base: self.vertex_words.start,
            types_base: self.type_words.start,
            occupancy_base: self.type_words.start + self.occupancy_offset,
            probes_base: self.type_words.start + self.probes_offset,
            kind: self.kind as u32,
            visible: 0,
            voxels_per_edge: self.voxels_per_edge,
            position_steps: match self.voxels_per_edge {
                0 => 0.0,
                edge => crate::voxel::position_steps(edge) as f32,
            },
        }
    }
}

/// A mesh's record in `MeshPool::info_buffer`, which culling reads to build its draw and the
/// mesh shaders read to draw it. Matches `MeshInfo` in `cull.slang` and `common/layout.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshInfo {
    pub bounds_min: [f32; 3],
    pub index_count: u32,
    pub bounds_max: [f32; 3],
    pub first_index: u32,
    /// World position of the mesh's local origin, and world units per local unit.
    pub origin: [f32; 3],
    pub scale: f32,
    /// Where the mesh's vertices and voxel types start, in words.
    pub vertex_base: u32,
    pub types_base: u32,
    pub kind: u32,
    /// Written by occlusion culling, non-zero when the mesh was last found visible.
    pub visible: u32,
    /// Zero but for chunks.
    pub voxels_per_edge: u32,
    pub position_steps: f32,
    /// Where the chunk's occupancy and probe cells start in the type buffer, in words.
    pub occupancy_base: u32,
    pub probes_base: u32,
}

impl MeshInfo {
    /// For a slot without a mesh, which draws nothing.
    pub const EMPTY: Self = Self {
        bounds_min: [0.0; 3],
        index_count: 0,
        bounds_max: [0.0; 3],
        first_index: 0,
        origin: [0.0; 3],
        scale: 0.0,
        vertex_base: 0,
        types_base: 0,
        kind: 0,
        visible: 0,
        voxels_per_edge: 0,
        position_steps: 0.0,
        occupancy_base: 0,
        probes_base: 0,
    };
}
