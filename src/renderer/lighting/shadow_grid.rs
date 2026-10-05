//! The chunks shadow rays are traced through, and the table each frame finds them in around the
//! camera.

use crate::renderer::Renderer;
use ash::prelude::VkResult;
use glam::{IVec3, IVec4, Vec3};

/// Chunks per side of each level's table shadow rays find chunks in, centered on the camera.
pub(in crate::renderer) const SHADOW_TABLE_CHUNKS: i32 = 32;
/// The most levels a `ShadowGrid` has. Matches `MAX_SHADOW_LEVELS` in `common/layout.slang`.
pub const MAX_SHADOW_LEVELS: u32 = 4;
/// A shadow table entry without a chunk. Matches `NO_CHUNK` in `common/voxels.slang`.
const NO_CHUNK: u32 = u32::MAX;

/// The chunks shadow rays are traced through: those `voxels_per_edge` voxels wide of
/// `voxel_size`, placed on the grid of chunks that size whose corner is the world's origin,
/// and with `levels` over 1, coarser levels of detail of them, each of voxels twice as big as
/// the last, as many to a chunk. Other chunks are lit, but cast no shadows.
///
/// Each level's chunks are found in a table 32 chunks wide around the camera, so coarser
/// levels reach further. Rays go through the finest level's chunk wherever there is one, so
/// where a level's chunks overlap a coarser one's, the finer ones count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowGrid {
    pub voxel_size: f32,
    pub voxels_per_edge: u32,
    /// Levels of detail, from 1 to `MAX_SHADOW_LEVELS`.
    pub levels: u32,
}

impl ShadowGrid {
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
        self.levels.clamp(1, MAX_SHADOW_LEVELS)
    }
}

impl Renderer {
    /// Fills in `frame`'s shadow table around `camera`, and which of its chunks changed since
    /// last frame.
    ///
    /// # Safety
    /// The device must not be using `frame`'s shadow table or changed chunks.
    pub(in crate::renderer) unsafe fn write_shadow_table(
        &mut self,
        frame: usize,
        grid: ShadowGrid,
        camera: Vec3,
    ) -> VkResult<TableLayout> {
        let table = TableLayout::around(grid, camera);
        self.fill_shadow_table(grid, &table);
        self.mark_changed_chunks(&table);
        unsafe {
            self.frame_shadow_tables[frame].write(
                &self.adapter,
                &self.render_device,
                &self.shadow_table,
            )?;
            self.frame_changed_chunks[frame].write(
                &self.adapter,
                &self.render_device,
                &self.changed_chunks,
            )?;
        }
        Ok(table)
    }

    /// Puts each chunk of `grid`'s levels in the mesh pool in its `table` entry, noting its
    /// serial for `mark_changed_chunks`.
    fn fill_shadow_table(&mut self, grid: ShadowGrid, table: &TableLayout) {
        self.shadow_table.clear();
        self.shadow_table.resize(table.entries(), NO_CHUNK);
        for (slot, mesh) in self.mesh_pool.meshes() {
            if !mesh.kind.is_chunk() || mesh.voxels_per_edge != grid.voxels_per_edge {
                continue;
            }
            // The level whose voxels are the mesh's size.
            let ratio = mesh.scale / grid.voxel_size;
            let level = ratio.log2().round();
            let level_size = level.exp2();
            if !(0.0..table.levels as f32).contains(&level)
                || (ratio - level_size).abs() > level_size * 1e-5
            {
                continue;
            }
            let chunk_size = grid.voxel_size * level_size * grid.voxels_per_edge as f32;
            let position = mesh.origin / chunk_size;
            let chunk = position.round();
            if (position - chunk).abs().max_element() > 1e-3 {
                continue;
            }
            let chunk = chunk.as_ivec3().extend(level as i32);
            if let Some(index) = table.entry(chunk) {
                self.shadow_table[index] = slot;
                self.chunk_serials_next.insert(chunk, mesh.serial);
            }
        }
    }

    /// Sets a bit for each `table` entry whose chunk was added, removed or remeshed since last
    /// frame, so the probes around it converge faster.
    fn mark_changed_chunks(&mut self, table: &TableLayout) {
        let previous = std::mem::take(&mut self.chunk_serials);
        self.changed_chunks.clear();
        self.changed_chunks.resize(table.entries().div_ceil(32), 0);
        let mut changed = |chunk: IVec4| {
            if let Some(index) = table.entry(chunk) {
                self.changed_chunks[index / 32] |= 1 << (index % 32);
            }
        };
        for (&chunk, serial) in &self.chunk_serials_next {
            if previous.get(&chunk) != Some(serial) {
                changed(chunk);
            }
        }
        for chunk in previous.keys() {
            if !self.chunk_serials_next.contains_key(chunk) {
                changed(*chunk);
            }
        }
        self.chunk_serials = std::mem::replace(&mut self.chunk_serials_next, previous);
        self.chunk_serials_next.clear();
    }
}

/// Where each level's shadow table is, `SHADOW_TABLE_CHUNKS` chunks wide around the camera.
pub(in crate::renderer) struct TableLayout {
    pub levels: u32,
    /// Each level's corner, in that level's chunks.
    pub corners: [IVec3; MAX_SHADOW_LEVELS as usize],
}

impl TableLayout {
    fn around(grid: ShadowGrid, camera: Vec3) -> Self {
        Self {
            levels: grid.level_count(),
            corners: std::array::from_fn(|level| {
                let chunk_size = grid.voxel_size * (grid.voxels_per_edge << level) as f32;
                (camera / chunk_size).floor().as_ivec3() - IVec3::splat(SHADOW_TABLE_CHUNKS / 2)
            }),
        }
    }

    /// Every level's entries.
    fn entries(&self) -> usize {
        self.levels as usize * (SHADOW_TABLE_CHUNKS as usize).pow(3)
    }

    /// The entry of a chunk, by its coordinate and then level, when the table holds it.
    fn entry(&self, chunk: IVec4) -> Option<usize> {
        if !(0..self.levels as i32).contains(&chunk.w) {
            return None;
        }
        let relative = chunk.truncate() - self.corners[chunk.w as usize];
        let inside = relative.min_element() >= 0 && relative.max_element() < SHADOW_TABLE_CHUNKS;
        let side = SHADOW_TABLE_CHUNKS as usize;
        inside.then(|| {
            let [x, y, z] = relative.to_array().map(|axis| axis as usize);
            ((chunk.w as usize * side + z) * side + y) * side + x
        })
    }
}
