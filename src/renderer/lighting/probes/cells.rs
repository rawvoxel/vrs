//! Where each probe cell of a chunk puts its probe, baked once as the chunk is added.

use super::PROBE_CELL;
use crate::mesh::chunk::VoxelTypes;

/// Set in a cell's baked word when it has room for a probe. Matches `CELL_PLACED` in
/// `probe_update.slang`.
const CELL_PLACED: u32 = 1 << 31;

/// Where each probe cell of a chunk puts its probe, and which cells have voxels, for chunks a
/// multiple of `PROBE_CELL` voxels wide, else nothing. Cells are x-fastest.
///
/// First a word per cell: the probe's position in the cell in half voxels, 6 bits per axis
/// from bit 0, and `CELL_PLACED`. Each probe goes in the largest empty part of its cell, the
/// octree node of the cell nearest its center at the largest size with an empty one, so it's
/// as far from the voxels as can be and can see around. An empty cell's probe is at its
/// center, a solid cell has none.
///
/// Then a bit per cell, 32 to a word, set when any of its voxels isn't empty, so the probes
/// near voxels can be found from a few words per chunk.
pub(crate) fn bake_probe_cells(types: &VoxelTypes) -> Vec<u32> {
    let edge = types.voxels_per_edge();
    if edge == 0 || !edge.is_multiple_of(PROBE_CELL) {
        return Vec::new();
    }
    let solids = types.solids();
    let cells = edge / PROBE_CELL;
    let solid_in = |min: [u32; 3], size: u32| {
        (0..size).any(|z| {
            (0..size).any(|y| {
                let row = (min[0] + edge * (min[1] + y + edge * (min[2] + z))) as usize;
                solids[row..row + size as usize].contains(&true)
            })
        })
    };
    let cell_min = |cell: u32| {
        [cell % cells, cell / cells % cells, cell / (cells * cells)].map(|axis| axis * PROBE_CELL)
    };
    let mut words: Vec<u32> = (0..cells.pow(3))
        .map(|cell| {
            let cell_min = cell_min(cell);
            // Breadth first, each size's nodes nearest the center first.
            let mut size = PROBE_CELL;
            while size > 0 {
                let nodes = PROBE_CELL / size;
                let mut nearest: Option<(u32, [u32; 3])> = None;
                for node in 0..nodes.pow(3) {
                    let node = [node % nodes, node / nodes % nodes, node / (nodes * nodes)];
                    // In half voxels, so a single voxel's center is whole.
                    let center = node.map(|axis| axis * size * 2 + size);
                    let distance: u32 = center
                        .iter()
                        .map(|&axis| axis.abs_diff(PROBE_CELL).pow(2))
                        .sum();
                    if nearest.is_some_and(|(best, _)| distance >= best) {
                        continue;
                    }
                    let min = [0, 1, 2].map(|axis| cell_min[axis] + node[axis] * size);
                    if !solid_in(min, size) {
                        nearest = Some((distance, center));
                    }
                }
                if let Some((_, [x, y, z])) = nearest {
                    return x | (y << 6) | (z << 12) | CELL_PLACED;
                }
                size /= 2;
            }
            0
        })
        .collect();
    let mut has_voxels = vec![0; cells.pow(3).div_ceil(32) as usize];
    for cell in 0..cells.pow(3) {
        if solid_in(cell_min(cell), PROBE_CELL) {
            has_voxels[(cell / 32) as usize] |= 1 << (cell % 32);
        }
    }
    words.extend(has_voxels);
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk one probe cell wide, solid where `solid` says: its probe word, and whether its
    /// mask says it has voxels.
    fn cell(solid: impl Fn(u32, u32, u32) -> bool) -> (u32, bool) {
        let edge = PROBE_CELL;
        let types: Vec<u16> = (0..edge.pow(3))
            .map(|i| solid(i % edge, i / edge % edge, i / (edge * edge)) as u16)
            .collect();
        let words = bake_probe_cells(&VoxelTypes::pack(edge, &types));
        assert_eq!(words.len(), 2);
        (words[0], words[1] & 1 != 0)
    }

    /// The probe's position in its cell, in voxels, when it has one.
    fn position(word: u32) -> Option<[f32; 3]> {
        (word & CELL_PLACED != 0)
            .then(|| [0, 6, 12].map(|shift| ((word >> shift) & 63) as f32 * 0.5))
    }

    #[test]
    fn empty_cells_center_their_probe() {
        let (word, has_voxels) = cell(|_, _, _| false);
        assert_eq!(position(word), Some([8.0; 3]));
        assert!(!has_voxels);
    }

    #[test]
    fn solid_cells_have_no_probe() {
        let (word, has_voxels) = cell(|_, _, _| true);
        assert_eq!(position(word), None);
        assert!(has_voxels);
    }

    #[test]
    fn probes_take_the_largest_empty_node_nearest_the_center() {
        // The bottom half is solid, so an 8³ node of the top half, in the open, not a smaller
        // one hugging the ground.
        let (word, has_voxels) = cell(|_, y, _| y < 8);
        let [x, y, z] = position(word).unwrap();
        assert_eq!(y, 12.0);
        assert!([x, z].iter().all(|&axis| axis == 4.0 || axis == 12.0));
        assert!(has_voxels);

        // Only one voxel is empty, so the probe is at its center.
        assert_eq!(
            position(cell(|x, y, z| (x, y, z) != (3, 9, 14)).0),
            Some([3.5, 9.5, 14.5])
        );
    }

    #[test]
    fn masks_mark_each_cell_with_voxels() {
        // Two cells wide, a voxel only in cell (1, 0, 1).
        let edge = 2 * PROBE_CELL;
        let types: Vec<u16> = (0..edge.pow(3))
            .map(|i| ((i % edge, i / edge % edge, i / (edge * edge)) == (20, 3, 17)) as u16)
            .collect();
        let words = bake_probe_cells(&VoxelTypes::pack(edge, &types));
        assert_eq!(words.len(), 8 + 1);
        // Cell (1, 0, 1) of 2 per edge, x-fastest, is cell 5.
        assert_eq!(words[8], 1 << 5);
    }

    #[test]
    fn chunks_not_a_multiple_of_a_cell_wide_have_no_cells() {
        assert!(bake_probe_cells(&VoxelTypes::pack(8, &[0; 512])).is_empty());
    }
}
