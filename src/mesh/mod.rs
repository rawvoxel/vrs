//! Meshes as they're handed to the renderer: their ids, every mesh's vertex formats, normals
//! packed in a word, and chunk meshes, see `chunk`.

pub mod chunk;
pub mod octahedral;
mod vertex;

pub use vertex::{
    AxisAlignedVertex, Face, Index, LOOSE_FLAG, POSITION_BIAS, PackedVertex, SEE_THROUGH_FLAG,
    Vertex, position_steps, unpack_position,
};

/// Refers to a mesh in a `MeshPool`.
///
/// Not `Clone` or `Copy`, `MeshPool::destroy` takes it by value
/// so a destroyed mesh can't be drawn or destroyed again.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MeshId(pub(crate) usize);
