//! Reading the light probes back, to see where they are and which are in use.

use crate::renderer::Renderer;
use ash::prelude::VkResult;
use glam::Vec3;

/// Words a probe takes, matching `Probe` in `common/layout.slang`: its cell, its state, its
/// position in the voxel grid's finest voxels and how many times it's been updated.
const PROBE_WORDS: usize = 8;
/// Matches `PROBE_PLACED` and `PROBE_OFF` in `common/layout.slang`.
const PROBE_PLACED: u32 = 1;
const PROBE_OFF: u32 = 2;

/// A light probe as the device last left it, see `Renderer::debug_probes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeRecord {
    /// The cascade it's in, the finest first.
    pub cascade: u32,
    /// Where it is in the world.
    pub position: Vec3,
    /// Whether its cell has room for it.
    pub placed: bool,
    /// Whether it's found itself inside voxels, so it's neither traced nor sampled.
    pub off: bool,
    /// Times it's been updated since it was placed.
    pub updates: u32,
}

impl ProbeRecord {
    /// Whether it lights anything: placed, updated and not off.
    #[inline]
    pub fn in_use(&self) -> bool {
        self.placed && !self.off && self.updates > 0
    }
}

impl Renderer {
    /// Every light probe of every cascade, empty without them or a `VoxelGrid`.
    ///
    /// For tests and tools: waits for the device to go idle, then reads them back.
    pub fn debug_probes(&mut self) -> VkResult<Vec<ProbeRecord>> {
        let Some(grid) = self.lighting.voxel_grid else {
            return Ok(Vec::new());
        };
        if self.probe_config.is_none() {
            return Ok(Vec::new());
        }
        let count = self.probe_volume.probe_count() as usize;
        let per_cascade = (count / self.probe_volume.cascades.max(1) as usize).max(1);
        let buffer = self.probe_volume.probe_buffer.handle;
        let words = self.debug_read_words(buffer, count * PROBE_WORDS)?;
        Ok(words
            .as_chunks::<PROBE_WORDS>()
            .0
            .iter()
            .enumerate()
            .map(|(slot, probe)| {
                let position = Vec3::new(
                    f32::from_bits(probe[4]),
                    f32::from_bits(probe[5]),
                    f32::from_bits(probe[6]),
                );
                ProbeRecord {
                    cascade: (slot / per_cascade) as u32,
                    position: position * grid.voxel_size,
                    placed: probe[3] & PROBE_PLACED != 0,
                    off: probe[3] & PROBE_OFF != 0,
                    updates: probe[7],
                }
            })
            .collect())
    }
}
