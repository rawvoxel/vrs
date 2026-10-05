//! Meshes as they're handed to the renderer: their ids, and their vertex formats, with normals
//! packed in a word. Chunk meshes' are `voxel`'s.

pub mod octahedral;
mod vertex;

pub use vertex::{Index, LitVertex, Vertex};

/// Refers to a mesh in a `MeshPool`.
///
/// Not `Clone` or `Copy`, `MeshPool::destroy` takes it by value
/// so a destroyed mesh can't be drawn or destroyed again.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MeshId(pub(crate) usize);
