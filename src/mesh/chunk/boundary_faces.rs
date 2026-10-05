//! Which of a chunk's voxels along its sides have faces out of it, from its mesh.

use crate::mesh::{AxisAlignedVertex, Index, unpack_position};

/// For each side of a chunk `edge` voxels wide, in `Face` order, a bit for each voxel along it,
/// set when the voxel has a face out of that side: `edge²` bits a side, 32 to a word, the
/// voxels in order of the side's two other axes, the lower one fastest. Meshers leave out the
/// faces between solid voxels, so a ray reaching a solid voxel through a side of its chunk where
/// it has no face comes from solid voxels beyond, as when the chunks beyond were left out for
/// being solid throughout: it hits the voxel's back [Majercik et al. 2021].
///
/// From the triangles lying in each side, each voxel's square marked when its center is in one.
pub(crate) fn boundary_faces(
    edge: u32,
    vertices: &[AxisAlignedVertex],
    indices: &[Index],
) -> Vec<u32> {
    let side_bits = (edge * edge) as usize;
    let mut words = vec![0u32; (6 * side_bits).div_ceil(32)];
    if edge == 0 {
        return words;
    }
    for triangle in indices.as_chunks::<3>().0 {
        let [Some(a), Some(b), Some(c)] = triangle.map(|index| vertices.get(index as usize)) else {
            continue;
        };
        let face = a.face();
        if b.face() != face || c.face() != face {
            continue;
        }
        let axis = face as usize / 2;
        let plane = if (face as usize).is_multiple_of(2) {
            0.0
        } else {
            edge as f32
        };
        let corners = [a, b, c]
            .map(|corner| unpack_position(corner.position_xy, corner.position_z_flags, edge));
        if corners
            .iter()
            .any(|corner| (corner[axis] - plane).abs() > 1e-3)
        {
            continue;
        }
        // The side's two other axes, the lower one first.
        let (u, v) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let flat = corners.map(|corner| [corner[u], corner[v]]);
        let low = flat.iter().fold([f32::MAX; 2], |low, point| {
            [low[0].min(point[0]), low[1].min(point[1])]
        });
        let high = flat.iter().fold([f32::MIN; 2], |high, point| {
            [high[0].max(point[0]), high[1].max(point[1])]
        });
        let range = |low: f32, high: f32| {
            (low.floor().max(0.0) as u32)..(high.ceil().min(edge as f32) as u32)
        };
        for j in range(low[1], high[1]) {
            for i in range(low[0], high[0]) {
                if inside(&flat, [i as f32 + 0.5, j as f32 + 0.5]) {
                    let bit = face as usize * side_bits + (i + edge * j) as usize;
                    words[bit / 32] |= 1 << (bit % 32);
                }
            }
        }
    }
    words
}

/// Whether `point` is in the triangle `corners`, either way around.
fn inside(corners: &[[f32; 2]; 3], point: [f32; 2]) -> bool {
    let side = |a: [f32; 2], b: [f32; 2]| {
        (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    };
    let sides = [
        side(corners[0], corners[1]),
        side(corners[1], corners[2]),
        side(corners[2], corners[0]),
    ];
    sides.iter().all(|&s| s >= 0.0) || sides.iter().all(|&s| s <= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Face;
    use glam::Vec3;

    const EDGE: u32 = 4;

    fn bit(words: &[u32], face: Face, i: u32, j: u32) -> bool {
        let bit = face as usize * (EDGE * EDGE) as usize + (i + EDGE * j) as usize;
        words[bit / 32] & (1 << (bit % 32)) != 0
    }

    /// One quad over the +x side's lower half, `y < 2`, as two triangles.
    fn half_of_positive_x() -> (Vec<AxisAlignedVertex>, Vec<Index>) {
        let x = EDGE as f32;
        let corners = [(0.0, 0.0), (0.0, x), (2.0, x), (2.0, 0.0)];
        let vertices = corners
            .map(|(y, z)| AxisAlignedVertex::new(Vec3::new(x, y, z), Face::PosX, EDGE).unwrap())
            .to_vec();
        (vertices, vec![0, 1, 2, 0, 2, 3])
    }

    #[test]
    fn marks_the_voxels_a_side_s_faces_cover() {
        let (vertices, indices) = half_of_positive_x();
        let words = boundary_faces(EDGE, &vertices, &indices);
        for z in 0..EDGE {
            for y in 0..EDGE {
                assert_eq!(bit(&words, Face::PosX, y, z), y < 2, "y {y} z {z}");
                assert!(!bit(&words, Face::NegX, y, z));
            }
        }
    }

    #[test]
    fn faces_inside_the_chunk_mark_nothing() {
        let corners = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)];
        let vertices = corners
            .map(|(y, z)| AxisAlignedVertex::new(Vec3::new(2.0, y, z), Face::PosX, EDGE).unwrap())
            .to_vec();
        let words = boundary_faces(EDGE, &vertices, &[0, 1, 2, 0, 2, 3]);
        assert!(words.iter().all(|&word| word == 0));
    }
}
