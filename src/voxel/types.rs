//! A chunk's voxel types, palette compressed in the words the shader decodes:
//!
//! - word 0: the header, `bits | (palette_words << 8)`;
//! - words `1 ..= palette_words`: the chunk's palette of type ids, two 16-bit ids per word,
//!   the even entry in the low half;
//! - then one `bits` wide palette index per voxel, x-fastest, `32 / bits` per word.
//!
//! `bits` is the smallest of 1, 2, 4, 8 or 16 that indexes the palette, so no index straddles
//! a word.

/// Voxels along each edge of a block of `VoxelTypes::occupancy`. Matches `OCCUPANCY_BLOCK` in
/// `common.slang`.
pub const OCCUPANCY_BLOCK: u32 = 8;

/// One chunk's voxel types, see the module docs. Type 0 is empty space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoxelTypes {
    voxels_per_edge: u32,
    words: Vec<u32>,
}

impl VoxelTypes {
    /// Packs one type id per voxel of a chunk `voxels_per_edge` wide, x-fastest.
    ///
    /// # Panics
    /// If `types` isn't `voxels_per_edge³` long.
    pub fn pack(voxels_per_edge: u32, types: &[u16]) -> Self {
        assert_eq!(
            types.len(),
            voxels_per_edge.pow(3) as usize,
            "A chunk {voxels_per_edge} voxels wide has {} voxels.",
            voxels_per_edge.pow(3)
        );

        let mut palette: Vec<u16> = Vec::new();
        let mut indices: Vec<u16> = Vec::with_capacity(types.len());
        // Runs are long, so the last entry is almost always the one.
        let mut last = 0;
        for &id in types {
            if palette.get(last) != Some(&id) {
                last = match palette.iter().position(|&entry| entry == id) {
                    Some(index) => index,
                    None => {
                        palette.push(id);
                        palette.len() - 1
                    }
                };
            }
            indices.push(last as u16);
        }
        if palette.is_empty() {
            palette.push(0);
        }

        let bits = bits_for(palette.len());
        let per_word = 32 / bits;
        let palette_words = palette.len().div_ceil(2);
        let mut words =
            Vec::with_capacity(1 + palette_words + types.len().div_ceil(per_word as usize));
        words.push(bits | ((palette_words as u32) << 8));
        for pair in palette.chunks(2) {
            words.push(pair[0] as u32 | ((pair.get(1).copied().unwrap_or(0) as u32) << 16));
        }
        let mut word = 0;
        let mut filled = 0;
        for &index in &indices {
            word |= (index as u32) << (filled * bits);
            filled += 1;
            if filled == per_word {
                words.push(word);
                (word, filled) = (0, 0);
            }
        }
        if filled > 0 {
            words.push(word);
        }

        Self {
            voxels_per_edge,
            words,
        }
    }

    /// Words already packed in the module docs' layout. `None` when
    /// they don't describe a chunk `voxels_per_edge` wide.
    pub fn from_words(voxels_per_edge: u32, words: Vec<u32>) -> Option<Self> {
        let header = *words.first()?;
        let bits = header & 0xff;
        let palette_words = (header >> 8) as usize;
        let voxels = voxels_per_edge.pow(3) as usize;
        let expected = matches!(bits, 1 | 2 | 4 | 8 | 16)
            .then(|| 1 + palette_words + voxels.div_ceil((32 / bits) as usize))?;
        (words.len() == expected).then_some(Self {
            voxels_per_edge,
            words,
        })
    }

    #[inline]
    pub fn voxels_per_edge(&self) -> u32 {
        self.voxels_per_edge
    }

    #[inline]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// One bit per block of `OCCUPANCY_BLOCK`³ voxels, set when any of them isn't empty, so
    /// shadow rays can cross empty blocks in one step. Blocks are x-fastest,
    /// `voxels_per_edge.div_ceil(OCCUPANCY_BLOCK)` to an edge, 32 to a word, the first in the
    /// lowest bit.
    pub fn occupancy(&self) -> Vec<u32> {
        let edge = self.voxels_per_edge;
        let blocks = edge.div_ceil(OCCUPANCY_BLOCK);
        let mut occupancy = vec![0; blocks.pow(3).div_ceil(32) as usize];
        for (index, solid) in self.solids().into_iter().enumerate() {
            if !solid {
                continue;
            }
            let index = index as u32;
            let (x, y, z) = (index % edge, index / edge % edge, index / (edge * edge));
            let block = [x, y, z].map(|axis| axis / OCCUPANCY_BLOCK);
            let block = block[0] + blocks * (block[1] + blocks * block[2]);
            occupancy[(block / 32) as usize] |= 1 << (block % 32);
        }
        occupancy
    }

    /// Whether each voxel isn't empty, x-fastest.
    pub fn solids(&self) -> Vec<bool> {
        let header = self.words[0];
        let bits = header & 0xff;
        let palette_words = (header >> 8) as usize;
        let solid_entries: Vec<bool> = (0..2 * palette_words)
            .map(|entry| (self.words[1 + entry / 2] >> ((entry % 2) * 16)) & 0xffff != 0)
            .collect();
        let per_word = 32 / bits;
        let mask = (1u32 << bits) - 1;
        let indices = &self.words[1 + palette_words..];
        (0..self.voxels_per_edge.pow(3))
            .map(|index| {
                let entry =
                    (indices[(index / per_word) as usize] >> ((index % per_word) * bits)) & mask;
                solid_entries.get(entry as usize).copied().unwrap_or(false)
            })
            .collect()
    }

    /// The type id of voxel `(x, y, z)`, decoded the way `typeAt` in `common.slang` decodes it.
    pub fn type_at(&self, x: u32, y: u32, z: u32) -> u16 {
        let edge = self.voxels_per_edge;
        let index = x + edge * (y + edge * z);
        let header = self.words[0];
        let bits = header & 0xff;
        let per_word = 32 / bits;
        let indices = 1 + (header >> 8);
        let word = self.words[(indices + index / per_word) as usize];
        let entry = (word >> ((index % per_word) * bits)) & ((1 << bits) - 1);
        let palette_word = self.words[1 + (entry >> 1) as usize];
        ((palette_word >> ((entry & 1) * 16)) & 0xffff) as u16
    }
}

/// The smallest power of two width that indexes `palette_len` entries.
fn bits_for(palette_len: usize) -> u32 {
    match palette_len {
        0..=2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        17..=256 => 8,
        _ => 16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trips(edge: u32, types: &[u16]) -> VoxelTypes {
        let packed = VoxelTypes::pack(edge, types);
        for (index, &id) in types.iter().enumerate() {
            let index = index as u32;
            let (x, y, z) = (index % edge, (index / edge) % edge, index / (edge * edge));
            assert_eq!(packed.type_at(x, y, z), id, "voxel {index}");
        }
        assert_eq!(
            VoxelTypes::from_words(edge, packed.words().to_vec()).as_ref(),
            Some(&packed)
        );
        packed
    }

    #[test]
    fn packs_at_the_narrowest_width() {
        for (distinct, bits) in [(1, 1), (2, 1), (3, 2), (5, 4), (17, 8), (300, 16)] {
            let types: Vec<u16> = (0..16u32.pow(3))
                .map(|i| (i as usize * distinct / 4096) as u16)
                .collect();
            let packed = round_trips(16, &types);
            assert_eq!(packed.words()[0] & 0xff, bits, "{distinct} distinct types");
        }
    }

    #[test]
    fn occupancy_marks_blocks_with_any_solid_voxel() {
        let edge = 20;
        let mut types = vec![0u16; (edge * edge * edge) as usize];
        // One voxel in block (1, 0, 2) and one in the last, partial, block (2, 2, 2).
        for (x, y, z) in [(9, 0, 17), (19, 19, 19)] {
            types[(x + edge * (y + edge * z)) as usize] = 4;
        }
        let occupancy = VoxelTypes::pack(edge, &types).occupancy();
        let blocks = edge.div_ceil(OCCUPANCY_BLOCK);
        assert_eq!(occupancy.len(), 1);
        let expected = [(1, 0, 2), (2, 2, 2)]
            .map(|(x, y, z)| 1u32 << (x + blocks * (y + blocks * z)))
            .iter()
            .fold(0, |bits, bit| bits | bit);
        assert_eq!(occupancy[0], expected);
        assert_eq!(
            VoxelTypes::pack(edge, &vec![0; types.len()]).occupancy(),
            [0]
        );
    }

    #[test]
    fn rejects_words_of_the_wrong_size() {
        let packed = VoxelTypes::pack(4, &[3; 64]);
        assert!(VoxelTypes::from_words(8, packed.words().to_vec()).is_none());
        assert!(VoxelTypes::from_words(4, Vec::new()).is_none());
    }
}
