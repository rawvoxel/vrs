//! Chunk meshes, in `PackedVertex` or `AxisAlignedVertex` formats, colored per voxel from their
//! palette compressed `VoxelTypes`.

mod types;

pub use types::{OCCUPANCY_BLOCK, VoxelTypes};

use crate::mesh::{AxisAlignedVertex, Index, PackedVertex};
use glam::Vec3;

/// A chunk's vertices, in whichever format its mesher suits.
#[derive(Clone, Copy, Debug)]
pub enum ChunkVertices<'a> {
    /// Axis aligned faces only, such as from greedy meshing, 8 bytes a vertex. Each quad is 4
    /// vertices in a row, which the renderer overlaps a little with its neighbors so no
    /// cracks open where one quad's corner meets the middle of another's edge.
    AxisAligned(&'a [AxisAlignedVertex]),
    /// Any surface, such as from surface nets or marching cubes, 12 bytes a vertex.
    Packed(&'a [PackedVertex]),
}

impl ChunkVertices<'_> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ChunkVertices::AxisAligned(vertices) => vertices.len(),
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
