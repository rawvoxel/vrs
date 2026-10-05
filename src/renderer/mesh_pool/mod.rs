//! The buffers every mesh shares: vertices, indices, voxel types and the records culling and
//! drawing read, which meshes are suballocated in, staged into, and repacked to defragment or
//! grow.

mod info;
mod mesh;
mod range_allocator;
mod repack;
mod staging;

pub(crate) use info::MeshInfo;
pub use mesh::{Mesh, MeshKind};

use range_allocator::RangeAllocator;
use staging::UploadStaging;

use crate::{
    mesh::{Index, MeshId},
    vulkan::{
        adapter::Adapter,
        buffer::{Buffer, create_device_local_buffer},
        bytes::slice_as_bytes,
    },
};
use ash::{prelude::VkResult, vk};
use glam::Vec3;
use std::slice;

/// How much a `MeshPool` holds when created. It grows when `RendererConfig::grow_mesh_pool` allows.
#[derive(Clone, Copy, Debug)]
pub struct MeshCapacity {
    /// Every kind of vertex shares the buffer, see `MeshKind::words_per_vertex`.
    pub vertex_words: u32,
    pub indices: u32,
    /// Chunks' packed voxel types, see `voxel::VoxelTypes`, each chunk's occupancy, one bit
    /// per block of `voxel::OCCUPANCY_BLOCK`³ voxels, and for chunks a multiple of 16 voxels
    /// wide, a word per 16³ voxels of where its light probe goes and a bit of whether it has
    /// voxels.
    pub type_words: u32,
    pub meshes: u32,
}

impl Default for MeshCapacity {
    /// 32 MiB of vertices, 16 MiB each of indices and voxel types, and 5 MiB of mesh records.
    fn default() -> Self {
        Self {
            vertex_words: 1 << 23,
            indices: 1 << 22,
            type_words: 1 << 22,
            meshes: 1 << 16,
        }
    }
}

/// Bytes in a word of the vertex and type buffers.
pub(super) const WORD_SIZE: vk::DeviceSize = size_of::<u32>() as vk::DeviceSize;
pub(super) const INDEX_SIZE: vk::DeviceSize = size_of::<Index>() as vk::DeviceSize;

/// A mesh for `MeshPool::create`, whatever its kind.
pub(crate) struct NewMesh<'a> {
    pub kind: MeshKind,
    /// `kind.words_per_vertex()` words per vertex.
    pub vertex_words: &'a [u32],
    pub indices: &'a [Index],
    /// Packed voxel types, empty but for chunks.
    pub type_words: &'a [u32],
    /// The types' occupancy, stored right after them. Empty but for chunks.
    pub occupancy: &'a [u32],
    /// Where the chunk's light probes go, stored after the occupancy. Empty but for chunks.
    pub probes: &'a [u32],
    pub origin: Vec3,
    pub scale: f32,
    /// Zero but for chunks.
    pub voxels_per_edge: u32,
    /// In the world.
    pub bounds: (Vec3, Vec3),
}

impl NewMesh<'_> {
    /// Words the mesh takes in the type buffer, its types, occupancy and probe cells.
    #[inline]
    pub fn type_word_count(&self) -> u32 {
        (self.type_words.len() + self.occupancy.len() + self.probes.len()) as u32
    }
}

/// Which of the pool's buffers an upload copies into.
const VERTICES: usize = 0;
const INDICES: usize = 1;
const INFOS: usize = 2;
const TYPES: usize = 3;

/// A mesh waiting for the next frame to copy its staged data into the shared buffers.
pub(super) struct Upload {
    pub slot: usize,
    /// Into the vertex, index, info and type buffers, `None` for empty data.
    pub copies: [Option<vk::BufferCopy2<'static>>; 4],
}

/// How often a pool has been repacked, see `MeshPool::repack`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RepackCounts {
    /// Repacked at the same capacity, to join scattered free space.
    pub defrags: u32,
    /// Repacked into a larger capacity.
    pub growths: u32,
}

/// Every mesh shares one `DEVICE_LOCAL` buffer each of vertices, indices and voxel types,
/// so drawing many meshes needs no rebinding. Vertices are pulled by the mesh shader from
/// a buffer of words, so meshes of every `MeshKind` share it. Each mesh's `MeshInfo` lives
/// in `info_buffer` at its `MeshId`, for culling and drawing on the device.
///
/// Creating a mesh writes its data to the current frame's staging buffer, which that frame
/// copies in before drawing. Destroying a mesh only frees its ranges once no frame in flight
/// can still draw it. When there's no room, `repack` defragments or grows the buffers.
pub(crate) struct MeshPool {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub info_buffer: Buffer,
    pub type_buffer: Buffer,
    pub capacity: MeshCapacity,
    pub(super) vertex_ranges: RangeAllocator,
    pub(super) index_ranges: RangeAllocator,
    pub(super) type_ranges: RangeAllocator,
    /// Indexed by `MeshId`, `None` slots are free for reuse.
    pub(super) meshes: Vec<Option<Mesh>>,
    pub(super) uploads: Vec<Upload>,
    staging: UploadStaging,
    /// Meshes to free once each frame's fence has been waited on.
    pub(super) frame_releases: Vec<Vec<Mesh>>,
    pub(super) repacks: RepackCounts,
    /// The next mesh's `Mesh::serial`.
    next_serial: u64,
}

impl MeshPool {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        capacity: MeshCapacity,
        frames_in_flight: usize,
    ) -> VkResult<Self> {
        let mut pool = Self::null(capacity, frames_in_flight);
        [
            pool.vertex_buffer,
            pool.index_buffer,
            pool.info_buffer,
            pool.type_buffer,
        ] = create_pool_buffers(adapter, logical_device, capacity)?;
        if let Err(err) = pool
            .staging
            .create(adapter, logical_device, frames_in_flight)
        {
            unsafe { pool.destroy_buffers(logical_device) };
            return Err(err);
        }
        Ok(pool)
    }

    /// Owns no buffers, destroying them does nothing.
    pub(crate) fn null(capacity: MeshCapacity, frames_in_flight: usize) -> Self {
        Self {
            vertex_buffer: Buffer::null(),
            index_buffer: Buffer::null(),
            info_buffer: Buffer::null(),
            type_buffer: Buffer::null(),
            capacity,
            vertex_ranges: RangeAllocator::new(capacity.vertex_words),
            index_ranges: RangeAllocator::new(capacity.indices),
            type_ranges: RangeAllocator::new(capacity.type_words),
            meshes: Vec::new(),
            uploads: Vec::new(),
            staging: UploadStaging::null(),
            frame_releases: (0..frames_in_flight).map(|_| Vec::new()).collect(),
            repacks: RepackCounts::default(),
            next_serial: 0,
        }
    }

    #[inline]
    pub fn repack_counts(&self) -> RepackCounts {
        self.repacks
    }

    /// Meshes with an id, uploaded or not.
    #[inline]
    pub fn mesh_count(&self) -> u32 {
        self.meshes.iter().flatten().count() as u32
    }

    /// Reserves space and stages the data for `frame` to upload.
    ///
    /// Fails with `ERROR_OUT_OF_DEVICE_MEMORY` when the pool is full.
    ///
    /// # Safety
    /// `begin_writes` must have been called for `frame` since it last recorded uploads.
    pub unsafe fn create(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        frame: usize,
        new: &NewMesh,
    ) -> VkResult<MeshId> {
        let out_of_memory = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        let free_slot = self.meshes.iter().position(Option::is_none);
        if free_slot.is_none() && self.meshes.len() >= self.capacity.meshes as usize {
            return Err(out_of_memory);
        }

        let vertex_words = self
            .vertex_ranges
            .alloc(new.vertex_words.len() as u32)
            .ok_or(out_of_memory)?;
        let Some(indices) = self.index_ranges.alloc(new.indices.len() as u32) else {
            self.vertex_ranges.free(vertex_words);
            return Err(out_of_memory);
        };
        let Some(type_words) = self.type_ranges.alloc(new.type_word_count()) else {
            self.vertex_ranges.free(vertex_words);
            self.index_ranges.free(indices);
            return Err(out_of_memory);
        };

        let mesh = Mesh {
            kind: new.kind,
            vertex_words,
            indices,
            type_words,
            occupancy_offset: new.type_words.len() as u32,
            probes_offset: (new.type_words.len() + new.occupancy.len()) as u32,
            origin: new.origin,
            scale: new.scale,
            voxels_per_edge: new.voxels_per_edge,
            bounds_min: new.bounds.0,
            bounds_max: new.bounds.1,
            serial: self.next_serial,
        };
        self.next_serial += 1;
        let slot = free_slot.unwrap_or(self.meshes.len());

        // NOTE: Even empty meshes upload their `MeshInfo`, the slot's last one may still be there.
        let info = mesh.info();
        let mut uploads: [(Vec<&[u8]>, vk::DeviceSize); 4] = Default::default();
        uploads[VERTICES] = (
            vec![slice_as_bytes(new.vertex_words)],
            mesh.vertex_words.start as vk::DeviceSize * WORD_SIZE,
        );
        uploads[INDICES] = (
            vec![slice_as_bytes(new.indices)],
            mesh.indices.start as vk::DeviceSize * INDEX_SIZE,
        );
        uploads[INFOS] = (
            vec![slice_as_bytes(slice::from_ref(&info))],
            (slot * size_of::<MeshInfo>()) as vk::DeviceSize,
        );
        uploads[TYPES] = (
            vec![
                slice_as_bytes(new.type_words),
                slice_as_bytes(new.occupancy),
                slice_as_bytes(new.probes),
            ],
            mesh.type_words.start as vk::DeviceSize * WORD_SIZE,
        );
        let mut copies = [None; 4];
        for (copy, (pieces, dst_offset)) in copies.iter_mut().zip(uploads) {
            let size: usize = pieces.iter().map(|piece| piece.len()).sum();
            if size == 0 {
                continue;
            }
            match unsafe { self.staging.push(adapter, logical_device, frame, &pieces) } {
                Ok(src_offset) => {
                    *copy = Some(
                        vk::BufferCopy2::default()
                            .src_offset(src_offset)
                            .dst_offset(dst_offset)
                            .size(size as vk::DeviceSize),
                    );
                }
                Err(err) => {
                    self.release(mesh);
                    return Err(err);
                }
            }
        }

        match free_slot {
            Some(slot) => self.meshes[slot] = Some(mesh),
            None => self.meshes.push(Some(mesh)),
        };
        self.uploads.push(Upload { slot, copies });

        Ok(MeshId(slot))
    }

    /// Frees the mesh once `last_frame`, the frame slot most recently submitted, has been waited on.
    pub fn destroy(&mut self, id: MeshId, last_frame: usize) {
        let mesh = self.meshes[id.0]
            .take()
            .expect("MeshId must come from this MeshPool.");

        // Never uploaded, so no frame can be using it.
        if let Some(upload) = self.uploads.iter().position(|upload| upload.slot == id.0) {
            self.uploads.swap_remove(upload);
            self.release(mesh);
        } else {
            self.frame_releases[last_frame].push(mesh);
        }
    }

    /// # Panics
    /// If `id` came from another pool.
    #[inline]
    pub fn get(&self, id: &MeshId) -> &Mesh {
        self.meshes[id.0]
            .as_ref()
            .expect("MeshId must come from this MeshPool.")
    }

    /// Every mesh with an id, uploaded or not, by its index into `info_buffer`.
    pub fn meshes(&self) -> impl Iterator<Item = (u32, &Mesh)> {
        self.meshes
            .iter()
            .enumerate()
            .filter_map(|(slot, mesh)| Some((slot as u32, mesh.as_ref()?)))
    }

    /// The mesh's index into `info_buffer`.
    ///
    /// # Panics
    /// If `id` came from another pool.
    #[inline]
    pub fn info_index(&self, id: &MeshId) -> u32 {
        assert!(
            self.meshes.get(id.0).is_some_and(Option::is_some),
            "MeshId must come from this MeshPool."
        );
        id.0 as u32
    }

    /// Frees meshes waiting on `frame` and readies its staging buffer for writes.
    /// Call once its fence has been waited on.
    pub fn release_frame(&mut self, frame: usize) {
        for mesh in std::mem::take(&mut self.frame_releases[frame]) {
            self.release(mesh);
        }
        self.staging.begin_writes(frame);
    }

    /// Whether destroyed meshes are waiting on frames in flight to free their space.
    pub fn has_pending_releases(&self) -> bool {
        self.frame_releases.iter().any(|meshes| !meshes.is_empty())
    }

    /// Frees every destroyed mesh's space, call once the device is idle.
    pub fn release_all(&mut self) {
        for frame in 0..self.frame_releases.len() {
            for mesh in std::mem::take(&mut self.frame_releases[frame]) {
                self.release(mesh);
            }
        }
    }

    #[inline]
    fn release(&mut self, mesh: Mesh) {
        self.vertex_ranges.free(mesh.vertex_words);
        self.index_ranges.free(mesh.indices);
        self.type_ranges.free(mesh.type_words);
    }

    /// Records copying `frame`'s staged uploads, followed by a barrier making them
    /// visible to drawing and culling.
    ///
    /// # Safety
    /// Every upload must have been staged for `frame`.
    pub unsafe fn cmd_upload(
        &mut self,
        logical_device: &ash::Device,
        frame: usize,
        cmd_buffer: vk::CommandBuffer,
    ) {
        if self.uploads.is_empty() {
            return;
        }

        let mut regions: [Vec<vk::BufferCopy2>; 4] = Default::default();
        for upload in self.uploads.drain(..) {
            for (regions, copy) in regions.iter_mut().zip(upload.copies) {
                regions.extend(copy);
            }
        }
        let staging_buffer = self.staging.buffer(frame);
        self.staging.mark_in_flight(frame);

        unsafe {
            // Freed ranges are only reused once their last frame's fence was waited on, but a
            // draw could read any of its buffers, so validation sees frames still in flight
            // reading them whole. Mesh slots are reused right away though, so frames in flight
            // may still cull with, upload or update the visibility of a destroyed mesh's
            // `MeshInfo`. Wait for all of these before copying.
            let reuse_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::INDEX_INPUT
                        | vk::PipelineStageFlags2::VERTEX_SHADER
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER
                        | vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::COPY,
                )
                .src_access_mask(
                    vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
            let reuse_dependency =
                vk::DependencyInfo::default().memory_barriers(slice::from_ref(&reuse_barrier));
            logical_device.cmd_pipeline_barrier2(cmd_buffer, &reuse_dependency);

            for (dst, regions) in self.buffers().into_iter().zip(&regions) {
                if regions.is_empty() {
                    continue;
                }
                let copy_info = vk::CopyBufferInfo2::default()
                    .src_buffer(staging_buffer)
                    .dst_buffer(dst.handle)
                    .regions(regions);
                logical_device.cmd_copy_buffer2(cmd_buffer, &copy_info);
            }

            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(DRAW_AND_CULL_STAGES)
                .dst_access_mask(DRAW_AND_CULL_ACCESS);
            let dependency_info =
                vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
            logical_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        }
    }

    /// The vertex, index, info and type buffers, in the order uploads copy into them.
    #[inline]
    pub(super) fn buffers(&self) -> [&Buffer; 4] {
        [
            &self.vertex_buffer,
            &self.index_buffer,
            &self.info_buffer,
            &self.type_buffer,
        ]
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy_buffers(&self, logical_device: &ash::Device) {
        unsafe {
            for buffer in self.buffers() {
                buffer.destroy(logical_device);
            }
            self.staging.destroy(logical_device);
        }
    }
}

/// Where the pool's buffers are read: indices, the mesh shaders pulling vertices and voxel
/// types, and culling reading and updating `MeshInfo`s.
pub(super) const DRAW_AND_CULL_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::INDEX_INPUT.as_raw()
        | vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
);
pub(super) const DRAW_AND_CULL_ACCESS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
    vk::AccessFlags2::INDEX_READ.as_raw()
        | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
        | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
);

/// The vertex, index, info and type buffers for `capacity`. If one fails, those already
/// created are destroyed.
pub(super) fn create_pool_buffers(
    adapter: &Adapter,
    logical_device: &ash::Device,
    capacity: MeshCapacity,
) -> VkResult<[Buffer; 4]> {
    let mut sizes = [(vk::BufferUsageFlags::STORAGE_BUFFER, 0); 4];
    sizes[VERTICES].1 = capacity.vertex_words as vk::DeviceSize * WORD_SIZE;
    sizes[INDICES] = (
        vk::BufferUsageFlags::INDEX_BUFFER,
        capacity.indices as vk::DeviceSize * INDEX_SIZE,
    );
    sizes[INFOS].1 = capacity.meshes as vk::DeviceSize * size_of::<MeshInfo>() as vk::DeviceSize;
    sizes[TYPES].1 = capacity.type_words as vk::DeviceSize * WORD_SIZE;

    let mut buffers = [
        Buffer::null(),
        Buffer::null(),
        Buffer::null(),
        Buffer::null(),
    ];
    for (buffer, (usage, size)) in buffers.iter_mut().zip(sizes) {
        // Repacking copies out of them. An empty buffer isn't allowed.
        let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC;
        match create_device_local_buffer(adapter, logical_device, usage, size.max(WORD_SIZE)) {
            Ok(created) => *buffer = created,
            Err(err) => {
                for buffer in buffers.iter() {
                    unsafe { buffer.destroy(logical_device) };
                }
                return Err(err);
            }
        }
    }
    Ok(buffers)
}
