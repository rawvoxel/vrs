//! The pool's size and how often it's been repacked, read without stalling.

use super::MeshCapacity;
use crate::renderer::Renderer;

/// The mesh pool's size and how often it's been repacked.
#[derive(Clone, Copy, Debug)]
pub struct MeshPoolStats {
    pub capacity: MeshCapacity,
    /// Meshes with an id, uploaded or not.
    pub meshes: u32,
    /// Repacked at the same capacity, to join scattered free space.
    pub defrags: u32,
    /// Repacked into a larger capacity.
    pub growths: u32,
}

impl Renderer {
    /// The mesh pool's size and how often it's been repacked, without stalling.
    pub fn mesh_pool_stats(&self) -> MeshPoolStats {
        let repacks = self.mesh_pool.repack_counts();
        MeshPoolStats {
            capacity: self.mesh_pool.capacity,
            meshes: self.mesh_pool.mesh_count(),
            defrags: repacks.defrags,
            growths: repacks.growths,
        }
    }
}
