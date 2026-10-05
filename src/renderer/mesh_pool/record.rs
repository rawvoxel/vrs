//! The pool's record of each mesh: how its vertices are stored and where it lives, and the
//! same as the GPU reads it.

use glam::Vec3;
use std::ops::Range;

/// How a mesh's vertices are stored, and so how it's colored.
/// Matches the `KIND_` constants in `common/layout.slang`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshKind {
    /// `mesh::Vertex`es, in the world with colors and normals of their own.
    Colored = 0,
    /// `mesh::AxisAlignedVertex`es, colored by voxel.
    AxisAligned = 1,
    /// `mesh::PackedVertex`es, colored by voxel.
    Packed = 2,
}

impl MeshKind {
    /// Words a vertex takes in the pool's vertex buffer.
    #[inline]
    pub fn words_per_vertex(self) -> u32 {
        match self {
            MeshKind::Colored => 5,
            MeshKind::AxisAligned => 2,
            MeshKind::Packed => 3,
        }
    }

    /// Whether it's a chunk, colored by voxel, with voxel types of its own.
    #[inline]
    pub fn is_chunk(self) -> bool {
        matches!(self, MeshKind::AxisAligned | MeshKind::Packed)
    }
}

/// Where a mesh lives in its pool's shared buffers, in words or elements, and where it is
/// in the world.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshRecord {
    pub kind: MeshKind,
    pub vertex_words: Range<u32>,
    pub indices: Range<u32>,
    /// A chunk's voxel types, its occupancy, then its probe cells, empty but for chunks.
    pub chunk_words: Range<u32>,
    /// Where the occupancy starts in `chunk_words`, see `mesh::chunk::VoxelTypes::occupancy`.
    pub occupancy_offset: u32,
    /// Where the probe cells start in `chunk_words`, see `lighting::probes::bake_probe_cells`.
    pub probes_offset: u32,
    /// World position of the mesh's local origin, a chunk's voxel `(0, 0, 0)` corner.
    pub origin: Vec3,
    /// World units per local unit, a chunk's voxel size.
    pub scale: f32,
    /// Zero but for chunks.
    pub voxels_per_edge: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    /// Unique to this mesh for its pool's life, so a chunk remeshed in place of another,
    /// even in the same slot, can be told from it.
    pub serial: u64,
}

impl MeshRecord {
    #[inline]
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    #[inline]
    pub fn vertex_count(&self) -> u32 {
        self.vertex_words.len() as u32 / self.kind.words_per_vertex()
    }

    /// As the GPU reads it.
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
            types_base: self.chunk_words.start,
            occupancy_base: self.chunk_words.start + self.occupancy_offset,
            probes_base: self.chunk_words.start + self.probes_offset,
            kind: self.kind as u32,
            visible: 0,
            voxels_per_edge: self.voxels_per_edge,
            position_steps: match self.voxels_per_edge {
                0 => 0.0,
                edge => crate::mesh::position_steps(edge) as f32,
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
