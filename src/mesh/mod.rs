//! Meshes: their vertex formats, ids, and the pool every mesh shares buffers in.

mod info;
pub mod octahedral;
mod pool;
mod range_allocator;
mod repack;
mod staging;
mod vertex;

pub(crate) use pool::{MeshPool, NewMesh};
pub use vertex::{Index, LitVertex, Vertex};

use glam::Vec3;
use std::ops::Range;

/// Refers to a mesh in a `MeshPool`.
///
/// Not `Clone` or `Copy`, `MeshPool::destroy` takes it by value
/// so a destroyed mesh can't be drawn or destroyed again.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MeshId(usize);

/// How much a `MeshPool` holds when created. It grows when `RendererConfig::grow_mesh_pool` allows.
#[derive(Clone, Copy, Debug)]
pub struct MeshCapacity {
    /// Every kind of vertex shares the buffer, see `MeshKind::words_per_vertex`.
    pub vertex_words: u32,
    pub indices: u32,
    /// Chunks' packed voxel types, see `voxel::VoxelTypes`, each chunk's occupancy, one bit
    /// per block of `voxel::OCCUPANCY_BLOCK`³ voxels, and for chunks a multiple of 16 voxels
    /// wide, a word per 16³ voxels of where its light probe goes and a bit of whether it has
    /// voxels.
    pub type_words: u32,
    pub meshes: u32,
}

impl Default for MeshCapacity {
    /// 32 MiB of vertices, 16 MiB each of indices and voxel types, and 5 MiB of mesh records.
    fn default() -> Self {
        Self {
            vertex_words: 1 << 23,
            indices: 1 << 22,
            type_words: 1 << 22,
            meshes: 1 << 16,
        }
    }
}

/// How a mesh's vertices are stored, and so how it's colored.
/// Matches the `KIND_` constants in `common/layout.slang`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshKind {
    /// `Vertex`es, world positions with their own colors.
    Colored = 0,
    /// `voxel::GreedyVertex`es, colored by voxel.
    Greedy = 1,
    /// `voxel::PackedVertex`es, colored by voxel.
    Packed = 2,
    /// `LitVertex`es, world positions with their own colors and normals, lit like
    /// chunks.
    Lit = 3,
}

impl MeshKind {
    /// Words a vertex takes in the pool's vertex buffer.
    #[inline]
    pub fn words_per_vertex(self) -> u32 {
        match self {
            MeshKind::Colored => 4,
            MeshKind::Greedy => 2,
            MeshKind::Packed => 3,
            MeshKind::Lit => 5,
        }
    }

    /// Whether it's a chunk, colored by voxel, with voxel types of its own.
    #[inline]
    pub fn is_chunk(self) -> bool {
        matches!(self, MeshKind::Greedy | MeshKind::Packed)
    }
}

/// Where a mesh lives in its pool's shared buffers, in words or elements, and where it is
/// in the world.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub kind: MeshKind,
    pub vertex_words: Range<u32>,
    pub indices: Range<u32>,
    /// A chunk's voxel types, its occupancy, then its probe cells, empty but for chunks.
    pub type_words: Range<u32>,
    /// Where the occupancy starts in `type_words`, see `voxel::VoxelTypes::occupancy`.
    pub occupancy_offset: u32,
    /// Where the probe cells start in `type_words`, see `lighting::gi::bake_probe_cells`.
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

impl Mesh {
    #[inline]
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    #[inline]
    pub fn vertex_count(&self) -> u32 {
        self.vertex_words.len() as u32 / self.kind.words_per_vertex()
    }
}
