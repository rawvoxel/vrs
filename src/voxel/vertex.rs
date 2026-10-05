//! Chunk vertices, packed in the words the mesh shader decodes.
//!
//! A position is chunk-local in voxels of the chunk's own LOD, biased by [`POSITION_BIAS`]
//! and stored in steps of `1 / position_steps(edge)`. Both are powers of two, so the decode
//! is exact and neighboring chunks decode a shared edge to the same point.

use crate::mesh::octahedral;
use glam::Vec3;

/// Voxels of margin on each side of a chunk's interior a vertex may sit, for skirts and the shell.
pub const POSITION_BIAS: f32 = 16.0;
/// The flag bit marking a vertex of a see-through face, which the renderer doesn't treat
/// differently yet.
pub const SEE_THROUGH_FLAG: u32 = 1;
/// Where a `GreedyVertex` keeps its `Face`, in its flags.
const FACE_SHIFT: u32 = 1;

/// Quantization steps per voxel for a chunk `voxels_per_edge` voxels wide: the largest power
/// of two whose range still fits the interior plus a [`POSITION_BIAS`] margin on each side.
/// 1024 for 32 voxels, 512 for 64.
pub fn position_steps(voxels_per_edge: u32) -> u32 {
    let range = (voxels_per_edge + 2 * POSITION_BIAS as u32).next_power_of_two();
    65536 / range
}

/// A face's direction, which is all a greedy mesh's normals can be.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    NegX = 0,
    PosX = 1,
    NegY = 2,
    PosY = 3,
    NegZ = 4,
    PosZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::NegX,
        Face::PosX,
        Face::NegY,
        Face::PosY,
        Face::NegZ,
        Face::PosZ,
    ];

    /// The outward normal. Matches `FACE_NORMALS` in `mesh.slang`.
    pub fn normal(self) -> Vec3 {
        match self {
            Face::NegX => Vec3::NEG_X,
            Face::PosX => Vec3::X,
            Face::NegY => Vec3::NEG_Y,
            Face::PosY => Vec3::Y,
            Face::NegZ => Vec3::NEG_Z,
            Face::PosZ => Vec3::Z,
        }
    }

    /// The face pointing most nearly along `normal`.
    pub fn from_normal(normal: Vec3) -> Self {
        let abs = normal.abs();
        match () {
            _ if abs.x >= abs.y && abs.x >= abs.z => match normal.x < 0.0 {
                true => Face::NegX,
                false => Face::PosX,
            },
            _ if abs.y >= abs.z => match normal.y < 0.0 {
                true => Face::NegY,
                false => Face::PosY,
            },
            _ => match normal.z < 0.0 {
                true => Face::NegZ,
                false => Face::PosZ,
            },
        }
    }
}

/// A vertex of any chunk mesh, 12 bytes:
/// - word 0: position x (16 bits) | position y (16 bits)
/// - word 1: position z (16 bits) | flags (16 bits, bit 0 marks a see-through face)
/// - word 2: octahedral normal u (16 bits) | v (16 bits)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedVertex {
    pub position_xy: u32,
    pub position_z_flags: u32,
    pub normal: u32,
}

impl PackedVertex {
    /// `voxel_position` is chunk-local in the chunk's voxels. `None` when it's outside the
    /// [`POSITION_BIAS`] margin, which no mesher should produce.
    pub fn new(voxel_position: Vec3, normal: Vec3, voxels_per_edge: u32) -> Option<Self> {
        let [position_xy, position_z_flags] = pack_position(voxel_position, voxels_per_edge)?;
        let [u, v] = octahedral::encode(normal);
        Some(Self {
            position_xy,
            position_z_flags,
            normal: u as u32 | ((v as u32) << 16),
        })
    }

    #[inline]
    pub fn flags(&self) -> u32 {
        self.position_z_flags >> 16
    }
}

/// A vertex of a greedy (axis aligned) chunk mesh, 8 bytes: a `PackedVertex` without its
/// normal word. Its normal is one of six directions, kept as a `Face` in flag bits 1 to 3.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GreedyVertex {
    pub position_xy: u32,
    pub position_z_flags: u32,
}

impl GreedyVertex {
    /// `voxel_position` is chunk-local in the chunk's voxels. `None` when it's outside the
    /// [`POSITION_BIAS`] margin, which no mesher should produce.
    pub fn new(voxel_position: Vec3, face: Face, voxels_per_edge: u32) -> Option<Self> {
        let [position_xy, position_z] = pack_position(voxel_position, voxels_per_edge)?;
        Some(Self {
            position_xy,
            position_z_flags: position_z | ((face as u32) << (FACE_SHIFT + 16)),
        })
    }

    /// A packed vertex of a greedy mesh, its normal turned into the nearest `Face`.
    pub fn from_packed(vertex: PackedVertex) -> Self {
        let face = Face::from_normal(octahedral::decode(vertex.normal));
        let flags = (vertex.flags() & SEE_THROUGH_FLAG) | ((face as u32) << FACE_SHIFT);
        Self {
            position_xy: vertex.position_xy,
            position_z_flags: (vertex.position_z_flags & 0xffff) | (flags << 16),
        }
    }

    #[inline]
    pub fn face(&self) -> Face {
        Face::ALL[((self.position_z_flags >> (16 + FACE_SHIFT)) & 7) as usize]
    }
}

/// The two position words, with zeroed flags.
fn pack_position(voxel_position: Vec3, voxels_per_edge: u32) -> Option<[u32; 2]> {
    let steps = position_steps(voxels_per_edge) as f32;
    let quantized = ((voxel_position + Vec3::splat(POSITION_BIAS)) * steps).round();
    if quantized.min_element() < 0.0 || quantized.max_element() > 65535.0 {
        return None;
    }
    Some([
        quantized.x as u32 | ((quantized.y as u32) << 16),
        quantized.z as u32,
    ])
}

/// The chunk-local voxel position of packed position words, as the shader decodes it.
pub fn unpack_position(position_xy: u32, position_z_flags: u32, voxels_per_edge: u32) -> Vec3 {
    let steps = position_steps(voxels_per_edge) as f32;
    let quantized = Vec3::new(
        (position_xy & 0xffff) as f32,
        (position_xy >> 16) as f32,
        (position_z_flags & 0xffff) as f32,
    );
    quantized / steps - Vec3::splat(POSITION_BIAS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_are_powers_of_two_fitting_the_margin() {
        assert_eq!(position_steps(32), 1024);
        assert_eq!(position_steps(64), 512);
    }

    #[test]
    fn positions_round_trip_exactly_on_voxel_fractions() {
        for position in [
            Vec3::ZERO,
            Vec3::new(32.0, 0.5, 31.25),
            Vec3::new(-16.0, 47.0, 3.0),
        ] {
            let vertex = PackedVertex::new(position, Vec3::Y, 32).unwrap();
            let decoded = unpack_position(vertex.position_xy, vertex.position_z_flags, 32);
            assert_eq!(decoded, position);
        }
        assert!(PackedVertex::new(Vec3::new(-16.5, 0.0, 0.0), Vec3::Y, 32).is_none());
        assert!(PackedVertex::new(Vec3::new(48.0, 0.0, 0.0), Vec3::Y, 32).is_none());
    }

    #[test]
    fn greedy_vertices_keep_position_and_face() {
        let packed = PackedVertex::new(Vec3::new(4.0, 2.0, 7.0), Vec3::NEG_Z, 32).unwrap();
        let greedy = GreedyVertex::from_packed(packed);
        assert_eq!(greedy.face(), Face::NegZ);
        assert_eq!(greedy.position_xy, packed.position_xy);
        assert_eq!(
            greedy.position_z_flags & 0xffff,
            packed.position_z_flags & 0xffff
        );
        assert_eq!(
            GreedyVertex::new(Vec3::new(4.0, 2.0, 7.0), Face::NegZ, 32),
            Some(greedy)
        );
    }
}
