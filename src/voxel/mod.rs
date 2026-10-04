//! Chunk meshes in compact vertex formats, colored per voxel from a palette.

mod types;
mod vertex;

pub use types::VoxelTypes;
pub use vertex::{
    Face, GreedyVertex, POSITION_BIAS, PackedVertex, SEE_THROUGH_FLAG, octahedral_decode,
    octahedral_encode, position_steps, unpack_position,
};

use crate::vertex::Index;
use glam::Vec3;

/// A chunk's vertices, in whichever format its mesher suits.
#[derive(Clone, Copy, Debug)]
pub enum ChunkVertices<'a> {
    /// Axis aligned faces only, such as from greedy meshing, 8 bytes a vertex.
    Greedy(&'a [GreedyVertex]),
    /// Any surface, such as from surface nets or marching cubes, 12 bytes a vertex.
    Packed(&'a [PackedVertex]),
}

impl ChunkVertices<'_> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ChunkVertices::Greedy(vertices) => vertices.len(),
            ChunkVertices::Packed(vertices) => vertices.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A chunk to draw, see `Renderer::create_chunk_mesh`.
#[derive(Clone, Copy, Debug)]
pub struct ChunkMesh<'a> {
    /// Relative to the chunk's first vertex.
    pub indices: &'a [Index],
    pub vertices: ChunkVertices<'a>,
    /// Where the chunk's voxel `(0, 0, 0)` corner is in the world.
    pub origin: Vec3,
    /// How big one of the chunk's voxels is in the world, which includes its LOD.
    pub voxel_size: f32,
    /// What each voxel is, which colors the faces through the renderer's palette.
    /// Also sets the chunk's width, which its vertices were packed for.
    pub types: &'a VoxelTypes,
}
