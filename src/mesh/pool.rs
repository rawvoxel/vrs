use super::{Mesh, MeshCapacity, MeshId, range_allocator::RangeAllocator, staging::UploadStaging};
use crate::{
    adapter::Adapter,
    buffer::{Buffer, create_device_local_buffer},
    bytes::slice_as_bytes,
    vertex::{Index, Vertex},
};
use ash::{prelude::VkResult, vk};
use glam::Vec3;
use std::slice;

impl Mesh {
    #[inline]
    pub(crate) fn info(&self) -> MeshInfo {
        MeshInfo {
            bounds_min: self.bounds_min.to_array(),
            index_count: self.index_count(),
            bounds_max: self.bounds_max.to_array(),
            first_index: self.indices.start,
            vertex_offset: self.vertices.start as i32,
            visible: 0,
            _pad: [0; 2],
        }
    }
}

/// A mesh's record in `MeshPool::info_buffer`, which culling reads to build its draw.
/// Matches `MeshInfo` in `cull.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshInfo {
    pub bounds_min: [f32; 3],
    pub index_count: u32,
    pub bounds_max: [f32; 3],
    pub first_index: u32,
    pub vertex_offset: i32,
    /// Written by occlusion culling, non-zero when the mesh was last found visible.
    pub visible: u32,
    pub _pad: [u32; 2],
}

/// A mesh waiting for the next frame to copy its staged data into the shared buffers.
struct Upload {
    slot: usize,
    /// Into the vertex, index and info buffers, `None` for empty data.
    copies: [Option<vk::BufferCopy2<'static>>; 3],
}

/// Every mesh shares one `DEVICE_LOCAL` vertex buffer and one index buffer,
/// so drawing many meshes needs no rebinding. Each mesh's `MeshInfo` lives in
/// `info_buffer` at its `MeshId`, for culling on the device.
///
/// Creating a mesh writes its data to the current frame's staging buffer, which that frame
/// copies in before drawing. Destroying a mesh only frees its ranges once no frame in flight
/// can still draw it.
pub(crate) struct MeshPool {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub info_buffer: Buffer,
    pub capacity: MeshCapacity,
    vertex_ranges: RangeAllocator,
    index_ranges: RangeAllocator,
    /// Indexed by `MeshId`, `None` slots are free for reuse.
    meshes: Vec<Option<Mesh>>,
    uploads: Vec<Upload>,
    staging: UploadStaging,
    /// Meshes to free once each frame's fence has been waited on.
    frame_releases: Vec<Vec<Mesh>>,
}

impl MeshPool {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        capacity: MeshCapacity,
        frames_in_flight: usize,
    ) -> VkResult<Self> {
        let mut pool = Self::null(capacity, frames_in_flight);
        let sizes = [
            (
                vk::BufferUsageFlags::VERTEX_BUFFER,
                capacity.vertices as vk::DeviceSize * size_of::<Vertex>() as vk::DeviceSize,
            ),
            (
                vk::BufferUsageFlags::INDEX_BUFFER,
                capacity.indices as vk::DeviceSize * size_of::<Index>() as vk::DeviceSize,
            ),
            (
                vk::BufferUsageFlags::STORAGE_BUFFER,
                capacity.meshes as vk::DeviceSize * size_of::<MeshInfo>() as vk::DeviceSize,
            ),
        ];
        let created = sizes
            .into_iter()
            .enumerate()
            .try_for_each(|(i, (usage, size))| {
                let buffer = create_device_local_buffer(adapter, logical_device, usage, size)?;
                match i {
                    0 => pool.vertex_buffer = buffer,
                    1 => pool.index_buffer = buffer,
                    _ => pool.info_buffer = buffer,
                }
                Ok(())
            })
            .and_then(|()| {
                pool.staging
                    .create(adapter, logical_device, frames_in_flight)
            });
        if let Err(err) = created {
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
            capacity,
            vertex_ranges: RangeAllocator::new(capacity.vertices),
            index_ranges: RangeAllocator::new(capacity.indices),
            meshes: Vec::new(),
            uploads: Vec::new(),
            staging: UploadStaging::null(),
            frame_releases: (0..frames_in_flight).map(|_| Vec::new()).collect(),
        }
    }

    /// Reserves space and stages the data for `frame` to upload.
    /// Indices are relative to the mesh's own first vertex.
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
        indices: &[Index],
        vertices: &[Vertex],
    ) -> VkResult<MeshId> {
        let out_of_memory = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        let free_slot = self.meshes.iter().position(Option::is_none);
        if free_slot.is_none() && self.meshes.len() >= self.capacity.meshes as usize {
            return Err(out_of_memory);
        }

        let vertex_range = self
            .vertex_ranges
            .alloc(vertices.len() as u32)
            .ok_or(out_of_memory)?;
        let Some(index_range) = self.index_ranges.alloc(indices.len() as u32) else {
            self.vertex_ranges.free(vertex_range);
            return Err(out_of_memory);
        };

        let (bounds_min, bounds_max) = match vertices.first() {
            Some(first) => vertices.iter().fold(
                (Vec3::from(first.pos), Vec3::from(first.pos)),
                |(min, max), vertex| {
                    let pos = Vec3::from(vertex.pos);
                    (min.min(pos), max.max(pos))
                },
            ),
            None => (Vec3::ZERO, Vec3::ZERO),
        };
        let mesh = Mesh {
            vertices: vertex_range,
            indices: index_range,
            bounds_min,
            bounds_max,
        };
        let slot = free_slot.unwrap_or(self.meshes.len());

        // NOTE: Even empty meshes upload their `MeshInfo`, the slot's last one may still be there.
        let info = mesh.info();
        let pieces = [
            (
                slice_as_bytes(vertices),
                mesh.vertices.start as usize * size_of::<Vertex>(),
            ),
            (
                slice_as_bytes(indices),
                mesh.indices.start as usize * size_of::<Index>(),
            ),
            (
                slice_as_bytes(slice::from_ref(&info)),
                slot * size_of::<MeshInfo>(),
            ),
        ];
        let mut copies = [None; 3];
        for (copy, (bytes, dst_offset)) in copies.iter_mut().zip(pieces) {
            if bytes.is_empty() {
                continue;
            }
            match unsafe { self.staging.push(adapter, logical_device, frame, bytes) } {
                Ok(src_offset) => {
                    *copy = Some(
                        vk::BufferCopy2::default()
                            .src_offset(src_offset)
                            .dst_offset(dst_offset as vk::DeviceSize)
                            .size(bytes.len() as vk::DeviceSize),
                    );
                }
                Err(err) => {
                    self.release(mesh);
                    return Err(err);
                }
            }
        }

        match free_slot {
            Some(slot) => {
                self.meshes[slot] = Some(mesh);
                slot
            }
            None => {
                self.meshes.push(Some(mesh));
                self.meshes.len() - 1
            }
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

    #[inline]
    fn release(&mut self, mesh: Mesh) {
        self.vertex_ranges.free(mesh.vertices);
        self.index_ranges.free(mesh.indices);
    }

    /// Records copying `frame`'s staged uploads, followed by a barrier making them
    /// visible to vertex input and culling.
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

        let mut regions: [Vec<vk::BufferCopy2>; 3] = Default::default();
        for upload in self.uploads.drain(..) {
            for (regions, copy) in regions.iter_mut().zip(upload.copies) {
                regions.extend(copy);
            }
        }
        let [vertex_regions, index_regions, info_regions] = regions;
        let staging_buffer = self.staging.buffer(frame);
        self.staging.mark_in_flight(frame);

        unsafe {
            // Freed ranges are only reused once their last frame's fence was waited on, but an
            // indexed draw could read any vertex, so validation sees frames still in flight reading
            // the whole vertex buffer. Mesh slots are reused right away though, so frames in flight
            // may still cull with, upload or update the visibility of a destroyed mesh's `MeshInfo`.
            // Wait for all of these before copying.
            let reuse_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::INDEX_INPUT
                        | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
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

            for (dst, regions) in [
                (&self.vertex_buffer, &vertex_regions),
                (&self.index_buffer, &index_regions),
                (&self.info_buffer, &info_regions),
            ] {
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
                .dst_stage_mask(
                    vk::PipelineStageFlags2::INDEX_INPUT
                        | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                        | vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::INDEX_READ
                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                        | vk::AccessFlags2::SHADER_STORAGE_READ
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                );
            let dependency_info =
                vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
            logical_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        }
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy_buffers(&self, logical_device: &ash::Device) {
        unsafe {
            self.vertex_buffer.destroy(logical_device);
            self.index_buffer.destroy(logical_device);
            self.info_buffer.destroy(logical_device);
            self.staging.destroy(logical_device);
        }
    }
}
