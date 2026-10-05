//! The chunks rays are traced through, and the table each frame finds them in.

use crate::renderer::{MeshRecord, Renderer};
use ash::prelude::VkResult;
use glam::{IVec3, IVec4, Vec3};

/// The most levels a `VoxelGrid` has. Matches `MAX_GRID_LEVELS` in `common/layout.slang`.
pub const MAX_GRID_LEVELS: u32 = 4;
/// The most chunks along each side of a level's box, so a chunk's place in it takes 10 bits an
/// axis of its key, see `TableLayout::key`.
const MAX_BOX_SIDE: i32 = 1 << 10;
/// The mesh of a chunk table slot without a chunk. Matches `NO_CHUNK` in `common/voxels.slang`.
const NO_CHUNK: u32 = u32::MAX;
/// Slots in the chunk table for each chunk, rounded up to a power of two, and the fewest. At
/// most a quarter full, a lookup rarely looks past a slot or two, even for a chunk it doesn't
/// hold, as rays mostly look up.
const SLOTS_PER_CHUNK: usize = 4;
const MIN_SLOTS: usize = 64;

/// Each level's lowest and highest chunk coordinate, in that level's chunks, `None` for a level
/// without chunks.
type LevelBounds = [Option<[IVec3; 2]>; MAX_GRID_LEVELS as usize];

/// The chunks every ray is traced through, for shadows, reflections, the light probes and
/// ambient occlusion: those `voxels_per_edge` voxels wide of `voxel_size`, placed on the grid
/// of chunks that size whose corner is the world's origin, and with `levels` over 1, coarser
/// levels of detail of them, each of voxels twice as big as the last, as many to a chunk.
/// Other chunks are lit, but no ray sees them: they cast no shadows, aren't reflected, and
/// the probes don't light them or see them.
///
/// Each frame, rays find the chunks in a hashed table, which takes 32 to 64 bytes a chunk
/// however far apart they are. Each level's chunks are in a box around them all, which rays cross
/// that level's chunks at a time where it has none, up to 1024 chunks wide: past that, it's
/// the part nearest the camera, and chunks outside it are left out. Rays go through the
/// finest level's chunk wherever there is one, so where a level's chunks overlap a coarser
/// one's, the finer ones count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGrid {
    pub voxel_size: f32,
    pub voxels_per_edge: u32,
    /// Levels of detail, from 1 to `MAX_GRID_LEVELS`.
    pub levels: u32,
}

impl VoxelGrid {
    /// Chunks of one size, `voxels_per_edge` voxels of `voxel_size` wide.
    pub fn new(voxel_size: f32, voxels_per_edge: u32) -> Self {
        Self {
            voxel_size,
            voxels_per_edge,
            levels: 1,
        }
    }

    /// With `levels` levels of detail.
    pub fn with_levels(self, levels: u32) -> Self {
        Self { levels, ..self }
    }

    /// Levels the shaders trace through.
    fn level_count(self) -> u32 {
        self.levels.clamp(1, MAX_GRID_LEVELS)
    }

    /// `mesh`'s chunk coordinate, in its level's chunks, then its level, when it's one of the
    /// grid's chunks.
    fn chunk_of(self, mesh: &MeshRecord) -> Option<IVec4> {
        if !mesh.kind.is_chunk() || mesh.voxels_per_edge != self.voxels_per_edge {
            return None;
        }
        // The level whose voxels are the mesh's size.
        let ratio = mesh.scale / self.voxel_size;
        let level = ratio.log2().round();
        let level_size = level.exp2();
        if !(0.0..self.level_count() as f32).contains(&level)
            || (ratio - level_size).abs() > level_size * 1e-5
        {
            return None;
        }
        let chunk_size = self.voxel_size * level_size * self.voxels_per_edge as f32;
        let position = mesh.origin / chunk_size;
        let chunk = position.round();
        if (position - chunk).abs().max_element() > 1e-3 {
            return None;
        }
        Some(chunk.as_ivec3().extend(level as i32))
    }
}

/// One of the `VoxelGrid`'s chunks in the mesh pool.
#[derive(Clone, Copy, Debug)]
pub(in crate::renderer) struct GridChunk {
    /// Its chunk coordinate, in its level's chunks, then its level.
    pub chunk: IVec4,
    /// Its slot in the mesh pool.
    pub mesh: u32,
    pub serial: u64,
}

impl Renderer {
    /// Fills in `frame`'s chunk table around `camera`, and which light probes are next to
    /// chunks that changed since last frame.
    ///
    /// # Safety
    /// The device must not be using `frame`'s chunk table or changed probes.
    pub(in crate::renderer) unsafe fn write_chunk_table(
        &mut self,
        frame: usize,
        grid: VoxelGrid,
        camera: Vec3,
    ) -> VkResult<TableLayout> {
        self.grid_chunks.clear();
        for (slot, mesh) in self.mesh_pool.meshes() {
            if let Some(chunk) = grid.chunk_of(mesh) {
                self.grid_chunks.push(GridChunk {
                    chunk,
                    mesh: slot,
                    serial: mesh.serial,
                });
            }
        }
        let table = TableLayout::around(grid, camera, &self.grid_chunks);
        self.fill_chunk_table(&table);
        let changed = self.changed_chunks();
        self.mark_changed_probes(grid, camera, &changed);
        unsafe {
            self.frame_chunk_tables[frame].write(&self.adapter, &self.device, &self.chunk_table)?;
            self.frame_changed_probes[frame].write(
                &self.adapter,
                &self.device,
                &self.changed_probes,
            )?;
        }
        Ok(table)
    }

    /// Puts each of the grid's chunks in its box in the `table` it lays out, noting its serial
    /// for `changed_chunks`.
    fn fill_chunk_table(&mut self, table: &TableLayout) {
        let serials = &mut self.chunk_serials_next;
        table.fill(&self.grid_chunks, &mut self.chunk_table, |grid_chunk| {
            serials.insert(grid_chunk.chunk, grid_chunk.serial);
        });
    }

    /// The chunks in the table added, removed or remeshed since last frame's, by chunk
    /// coordinate, then level.
    fn changed_chunks(&mut self) -> Vec<IVec4> {
        let previous = std::mem::take(&mut self.chunk_serials);
        let mut changed: Vec<IVec4> = self
            .chunk_serials_next
            .iter()
            .filter(|&(chunk, serial)| previous.get(chunk) != Some(serial))
            .map(|(&chunk, _)| chunk)
            .collect();
        changed.extend(
            previous
                .keys()
                .filter(|chunk| !self.chunk_serials_next.contains_key(chunk)),
        );
        self.chunk_serials = std::mem::replace(&mut self.chunk_serials_next, previous);
        self.chunk_serials_next.clear();
        changed
    }
}

/// The box each level's chunks are found in this frame, and the table's size, see `VoxelGrid`.
pub(in crate::renderer) struct TableLayout {
    pub levels: u32,
    /// Each level's box's low corner, in that level's chunks.
    pub corners: [IVec3; MAX_GRID_LEVELS as usize],
    /// Chunks along each side of each level's box, none for a level without chunks.
    pub sizes: [IVec3; MAX_GRID_LEVELS as usize],
    /// Slots in the table, a power of two.
    pub slots: u32,
}

impl TableLayout {
    /// Each level's box around all of `chunks` of that level, of whole chunks of the grid's
    /// coarsest level, so a ray crossing a coarser level's chunk where a finer one has none
    /// never crosses a finer level's box. Too wide, it's the `MAX_BOX_SIDE` nearest `camera`.
    fn around(grid: VoxelGrid, camera: Vec3, chunks: &[GridChunk]) -> Self {
        let levels = grid.level_count();
        let mut bounds = LevelBounds::default();
        for grid_chunk in chunks {
            let level = &mut bounds[grid_chunk.chunk.w as usize];
            let chunk = grid_chunk.chunk.truncate();
            *level = Some(match *level {
                Some([low, high]) => [low.min(chunk), high.max(chunk)],
                None => [chunk, chunk],
            });
        }
        let mut corners = [IVec3::ZERO; MAX_GRID_LEVELS as usize];
        let mut sizes = [IVec3::ZERO; MAX_GRID_LEVELS as usize];
        for level in 0..levels as usize {
            let Some([low, high]) = bounds[level] else {
                continue;
            };
            let align = IVec3::splat(1 << (levels as usize - 1 - level));
            let low = low.div_euclid(align) * align;
            let end = (high.div_euclid(align) + 1) * align;
            let chunk_size = grid.voxel_size * (grid.voxels_per_edge << level) as f32;
            let camera_chunk = (camera / chunk_size).floor().as_ivec3();
            let centered = camera_chunk.div_euclid(align) * align - IVec3::splat(MAX_BOX_SIDE / 2);
            corners[level] = centered.clamp(low, (end - MAX_BOX_SIDE).max(low));
            sizes[level] = (end - corners[level]).min(IVec3::splat(MAX_BOX_SIDE));
        }
        let slots = (chunks.len() * SLOTS_PER_CHUNK)
            .next_power_of_two()
            .max(MIN_SLOTS);
        Self {
            levels,
            corners,
            sizes,
            slots: slots as u32,
        }
    }

    /// The box around every level's, from its lowest corner to past its highest, in `grid`'s
    /// finest voxels: empty at the origin without chunks. Of whole chunks of the coarsest
    /// level, as theirs are, so rays crossing one in none of theirs cross none of theirs.
    pub fn grid_box(&self, grid: VoxelGrid) -> [IVec3; 2] {
        let mut grid_box: Option<[IVec3; 2]> = None;
        for level in 0..self.levels as usize {
            if self.sizes[level].cmple(IVec3::ZERO).any() {
                continue;
            }
            let chunk_size = (grid.voxels_per_edge << level) as i32;
            let low = self.corners[level] * chunk_size;
            let high = (self.corners[level] + self.sizes[level]) * chunk_size;
            grid_box = Some(match grid_box {
                Some([min, max]) => [min.min(low), max.max(high)],
                None => [low, high],
            });
        }
        grid_box.unwrap_or([IVec3::ZERO; 2])
    }

    /// Fills `table` with the `chunks` in their levels' boxes, two words a slot: a chunk's key,
    /// then its mesh, each in the first slot from its key's hash on with its key or no chunk.
    /// Of chunks with the same key, the last counts. Calls `placed` with each.
    fn fill(&self, chunks: &[GridChunk], table: &mut Vec<u32>, mut placed: impl FnMut(&GridChunk)) {
        table.clear();
        table.resize(2 * self.slots as usize, NO_CHUNK);
        let mask = self.slots - 1;
        for grid_chunk in chunks {
            let Some(key) = self.key(grid_chunk.chunk) else {
                continue;
            };
            let mut slot = (chunk_hash(key) & mask) as usize;
            while table[2 * slot + 1] != NO_CHUNK && table[2 * slot] != key {
                slot = (slot + 1) & mask as usize;
            }
            table[2 * slot] = key;
            table[2 * slot + 1] = grid_chunk.mesh;
            placed(grid_chunk);
        }
    }

    /// The table key of a chunk, by its coordinate and then level, when it's in its level's
    /// box: the level, then its place in the box, 10 bits an axis. Matches `chunkKey` in
    /// `common/voxels.slang`.
    fn key(&self, chunk: IVec4) -> Option<u32> {
        if !(0..self.levels as i32).contains(&chunk.w) {
            return None;
        }
        let level = chunk.w as usize;
        let relative = chunk.truncate() - self.corners[level];
        let inside = relative.cmpge(IVec3::ZERO).all() && relative.cmplt(self.sizes[level]).all();
        inside.then(|| {
            let [x, y, z] = relative.to_array().map(|axis| axis as u32);
            ((level as u32) << 30) | (z << 20) | (y << 10) | x
        })
    }
}

/// A chunk key's bits mixed for its first slot, lowbias32 [Wellons 2018]. Matches `chunkHash`
/// in `common/voxels.slang`.
fn chunk_hash(key: u32) -> u32 {
    let mut x = key;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chunks one voxel 1 wide, so chunk coordinates are positions.
    fn grid(levels: u32) -> VoxelGrid {
        VoxelGrid::new(1.0, 1).with_levels(levels)
    }

    fn chunks(coordinates: &[IVec4]) -> Vec<GridChunk> {
        let chunk = |(mesh, &chunk): (usize, &IVec4)| GridChunk {
            chunk,
            mesh: mesh as u32,
            serial: mesh as u64,
        };
        coordinates.iter().enumerate().map(chunk).collect()
    }

    /// The mesh `levelChunk` in `common/voxels.slang` finds for `chunk` in `table`, laid out
    /// by `layout`, and whether it's outside its level's box.
    fn look_up(table: &[u32], layout: &TableLayout, chunk: IVec4) -> (Option<u32>, bool) {
        let Some(key) = layout.key(chunk) else {
            return (None, true);
        };
        let mask = layout.slots - 1;
        let mut slot = chunk_hash(key) & mask;
        loop {
            let (at, mesh) = (table[2 * slot as usize], table[2 * slot as usize + 1]);
            if mesh == NO_CHUNK {
                return (None, false);
            }
            if at == key {
                return (Some(mesh), false);
            }
            slot = (slot + 1) & mask;
        }
    }

    /// `coordinates`' table and its layout, each chunk's mesh its index.
    fn build(grid: VoxelGrid, camera: Vec3, coordinates: &[IVec4]) -> (Vec<u32>, TableLayout) {
        let chunks = chunks(coordinates);
        let layout = TableLayout::around(grid, camera, &chunks);
        let mut table = Vec::new();
        layout.fill(&chunks, &mut table, |_| ());
        (table, layout)
    }

    #[test]
    fn finds_every_chunk_however_far_apart_and_no_others() {
        // Nearly as far apart as a box allows, along every axis, at every level.
        let mut coordinates = Vec::new();
        for level in 0..4 {
            for step in [0, 1, 7, 200, 690] {
                for chunk in [
                    IVec4::new(step, -step / 3, step / 2, level),
                    IVec4::new(-step / 3, step / 2, 3, level),
                ] {
                    if !coordinates.contains(&chunk) {
                        coordinates.push(chunk);
                    }
                }
            }
        }
        let (table, layout) = build(grid(4), Vec3::ZERO, &coordinates);
        for (mesh, &chunk) in coordinates.iter().enumerate() {
            assert_eq!(
                look_up(&table, &layout, chunk).0,
                Some(mesh as u32),
                "{chunk}"
            );
        }
        // Inside the boxes, where there's no chunk.
        for chunk in [IVec4::new(2, 2, 2, 0), IVec4::new(-1, 5, 3, 1)] {
            assert_eq!(look_up(&table, &layout, chunk), (None, false), "{chunk}");
        }
        assert!(layout.slots as usize >= SLOTS_PER_CHUNK * coordinates.len());
    }

    #[test]
    fn boxes_are_whole_chunks_of_the_coarsest_level() {
        let coordinates = [IVec4::new(3, 5, -9, 0), IVec4::new(1, 1, 1, 2)];
        let (_, layout) = build(grid(3), Vec3::ZERO, &coordinates);
        // Level 0's box is whole level 2 chunks, 4 of its own, around its one chunk.
        assert_eq!(layout.corners[0], IVec3::new(0, 4, -12));
        assert_eq!(layout.sizes[0], IVec3::splat(4));
        assert_eq!(layout.sizes[1], IVec3::ZERO);
        assert_eq!(layout.corners[2], IVec3::ONE);
        assert_eq!(layout.sizes[2], IVec3::ONE);
    }

    #[test]
    fn too_wide_a_level_keeps_the_part_nearest_the_camera() {
        let coordinates = [IVec4::new(0, 0, 0, 0), IVec4::new(5000, 0, 0, 0)];
        let (_, layout) = build(grid(1), Vec3::new(4800.5, 0.0, 0.0), &coordinates);
        assert_eq!(layout.corners[0].x, 5001 - MAX_BOX_SIDE);
        assert_eq!(layout.sizes[0].x, MAX_BOX_SIDE);
        assert_eq!(layout.key(IVec4::new(0, 0, 0, 0)), None);
        assert!(layout.key(IVec4::new(5000, 0, 0, 0)).is_some());
    }

    #[test]
    fn the_grid_box_holds_every_level_s() {
        let coordinates = [IVec4::new(3, 5, -9, 0), IVec4::new(1, 1, 1, 2)];
        let (_, layout) = build(grid(3), Vec3::ZERO, &coordinates);
        // Level 0's box from (0, 4, -12) to (4, 8, -8), level 2's, of chunks 4 wide, from
        // (4, 4, 4) to (8, 8, 8).
        assert_eq!(
            layout.grid_box(grid(3)),
            [IVec3::new(0, 4, -12), IVec3::splat(8)]
        );
        let (_, empty) = build(grid(3), Vec3::ZERO, &[]);
        assert_eq!(empty.grid_box(grid(3)), [IVec3::ZERO; 2]);
    }

    #[test]
    fn of_the_same_chunk_the_last_counts() {
        let (table, layout) = build(grid(1), Vec3::ZERO, &[IVec4::ZERO, IVec4::ZERO]);
        assert_eq!(look_up(&table, &layout, IVec4::ZERO).0, Some(1));
    }
}
