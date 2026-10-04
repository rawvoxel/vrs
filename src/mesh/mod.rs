mod pool;
mod range_allocator;
mod staging;

pub(crate) use pool::MeshPool;

use glam::Vec3;
use std::ops::Range;

/// Refers to a mesh in a `MeshPool`.
///
/// Not `Clone` or `Copy`, `MeshPool::destroy` takes it by value
/// so a destroyed mesh can't be drawn or destroyed again.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MeshId(usize);

/// How much a `MeshPool` holds. Its buffers are allocated up front and don't grow.
#[derive(Clone, Copy, Debug)]
pub struct MeshCapacity {
    pub vertices: u32,
    pub indices: u32,
    pub meshes: u32,
}

impl Default for MeshCapacity {
    /// 48 MiB of vertices, 16 MiB of indices and 3 MiB of mesh records.
    fn default() -> Self {
        Self {
            vertices: 1 << 21,
            indices: 1 << 22,
            meshes: 1 << 16,
        }
    }
}

/// Where a mesh lives in its pool's shared buffers, in elements, and its bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub vertices: Range<u32>,
    pub indices: Range<u32>,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

impl Mesh {
    #[inline]
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }
}
