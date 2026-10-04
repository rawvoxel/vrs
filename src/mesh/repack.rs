//! Moving every mesh into new buffers, packed together, to defragment or grow a pool.

use super::{
    Mesh, MeshCapacity,
    pool::{MeshInfo, MeshPool, create_pool_buffers},
    range_allocator::RangeAllocator,
};
use crate::{
    adapter::Adapter,
    buffer::Buffer,
    commands::record_submit_commandbuffer,
    vertex::{Index, Vertex},
};
use ash::{prelude::VkResult, vk};
use std::{ops::Range, slice};

const VERTEX_SIZE: vk::DeviceSize = size_of::<Vertex>() as vk::DeviceSize;
const INDEX_SIZE: vk::DeviceSize = size_of::<Index>() as vk::DeviceSize;

impl MeshPool {
    /// The capacity to repack into so a mesh of `vertices` and `indices` fits, which is the
    /// current one when packing alone makes room. `None` when no repack would help, or it
    /// would need to grow without `allow_growth`.
    ///
    /// Destroyed meshes still waiting on frames in flight count as used, see `release_all`.
    pub fn plan_repack(
        &self,
        vertices: u32,
        indices: u32,
        allow_growth: bool,
    ) -> Option<MeshCapacity> {
        let mut capacity = self.capacity;
        let mut repack = false;
        let mut grow = false;

        let has_free_slot = self.meshes.iter().any(Option::is_none);
        if !has_free_slot && self.meshes.len() >= capacity.meshes as usize {
            capacity.meshes = grown(capacity.meshes, self.meshes.len() as u64 + 1);
            repack = true;
            grow = true;
        }
        for (capacity, ranges, len) in [
            (&mut capacity.vertices, &self.vertex_ranges, vertices),
            (&mut capacity.indices, &self.index_ranges, indices),
        ] {
            if ranges.can_alloc(len) {
                continue;
            }
            repack = true;
            // Packed, the free space joins up, but there may not be enough of it.
            if ranges.free_len() < len {
                let used = (ranges.capacity() - ranges.free_len()) as u64;
                *capacity = grown(*capacity, used + len as u64);
                grow = true;
            }
        }

        (repack && (allow_growth || !grow)).then_some(capacity)
    }

    /// Copies every mesh into new buffers of `capacity`, packed together from the start, and
    /// rewrites every `MeshInfo` for where they moved. Blocks until the copies finish. If it
    /// fails, the pool is left as it was.
    ///
    /// Meshes still waiting to upload are pointed at their new place instead of copied.
    ///
    /// # Safety
    /// The device must be idle, with `release_all` called since, and `capacity` must fit
    /// every mesh.
    pub unsafe fn repack(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        queue: vk::Queue,
        cmd_buffer: vk::CommandBuffer,
        cmd_fence: vk::Fence,
        capacity: MeshCapacity,
    ) -> VkResult<()> {
        debug_assert!(!self.has_pending_releases());

        // Where each mesh goes, without touching the pool until the copies succeed.
        let mut vertex_ranges = RangeAllocator::new(capacity.vertices);
        let mut index_ranges = RangeAllocator::new(capacity.indices);
        let mut uploading = vec![false; self.meshes.len()];
        for upload in self.uploads.iter() {
            uploading[upload.slot] = true;
        }
        let mut moved: Vec<Option<Mesh>> = Vec::with_capacity(self.meshes.len());
        let mut vertex_regions = Vec::new();
        let mut index_regions = Vec::new();
        for (slot, mesh) in self.meshes.iter().enumerate() {
            let Some(mesh) = mesh else {
                moved.push(None);
                continue;
            };
            let vertices = vertex_ranges
                .alloc(mesh.vertices.len() as u32)
                .expect("The repacked capacity fits every mesh.");
            let indices = index_ranges
                .alloc(mesh.indices.len() as u32)
                .expect("The repacked capacity fits every mesh.");
            // Data still waiting to upload isn't in the old buffers yet.
            if !uploading[slot] {
                push_copy(&mut vertex_regions, &mesh.vertices, &vertices, VERTEX_SIZE);
                push_copy(&mut index_regions, &mesh.indices, &indices, INDEX_SIZE);
            }
            moved.push(Some(Mesh {
                vertices,
                indices,
                ..mesh.clone()
            }));
        }
        let infos: Vec<MeshInfo> = moved
            .iter()
            .map(|mesh| mesh.as_ref().map_or(MeshInfo::EMPTY, Mesh::info))
            .collect();

        let [vertex_buffer, index_buffer, info_buffer] =
            create_pool_buffers(adapter, logical_device, capacity)?;
        let new_buffers = [vertex_buffer, index_buffer, info_buffer];
        let copied = unsafe {
            self.copy_into(
                adapter,
                logical_device,
                queue,
                cmd_buffer,
                cmd_fence,
                &new_buffers,
                &vertex_regions,
                &index_regions,
                &infos,
            )
        };
        if let Err(err) = copied {
            for buffer in new_buffers.iter() {
                unsafe { buffer.destroy(logical_device) };
            }
            return Err(err);
        }

        // Commit
        let grew = capacity.vertices > self.capacity.vertices
            || capacity.indices > self.capacity.indices
            || capacity.meshes > self.capacity.meshes;
        match grew {
            true => self.repacks.growths += 1,
            false => self.repacks.defrags += 1,
        }
        unsafe {
            self.vertex_buffer.destroy(logical_device);
            self.index_buffer.destroy(logical_device);
            self.info_buffer.destroy(logical_device);
        }
        [self.vertex_buffer, self.index_buffer, self.info_buffer] = new_buffers;
        self.capacity = capacity;
        self.vertex_ranges = vertex_ranges;
        self.index_ranges = index_ranges;
        self.meshes = moved;
        for upload in self.uploads.iter_mut() {
            let mesh = self.meshes[upload.slot]
                .as_ref()
                .expect("Destroyed meshes have no upload.");
            let [vertex_copy, index_copy, info_copy] = &mut upload.copies;
            if let Some(copy) = vertex_copy {
                copy.dst_offset = mesh.vertices.start as vk::DeviceSize * VERTEX_SIZE;
            }
            if let Some(copy) = index_copy {
                copy.dst_offset = mesh.indices.start as vk::DeviceSize * INDEX_SIZE;
            }
            // Already written with the rest.
            *info_copy = None;
        }

        Ok(())
    }

    /// Copies the old buffers' meshes and every `MeshInfo` into `new_buffers`, waiting for it.
    #[allow(clippy::too_many_arguments)]
    unsafe fn copy_into(
        &self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        queue: vk::Queue,
        cmd_buffer: vk::CommandBuffer,
        cmd_fence: vk::Fence,
        new_buffers: &[Buffer; 3],
        vertex_regions: &[vk::BufferCopy2],
        index_regions: &[vk::BufferCopy2],
        infos: &[MeshInfo],
    ) -> VkResult<()> {
        let info_bytes = size_of_val(infos) as vk::DeviceSize;
        let info_staging_info = vk::BufferCreateInfo::default()
            .size(info_bytes.max(1))
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut info_staging = Buffer::new(adapter, logical_device, info_staging_info, flags, 0)?;

        let copied = info_staging.mem_map(logical_device).and_then(|()| {
            info_staging.mem_write(infos);
            let info_region = vk::BufferCopy2::default().size(info_bytes);
            let copies = [
                (
                    self.vertex_buffer.handle,
                    new_buffers[0].handle,
                    vertex_regions,
                ),
                (
                    self.index_buffer.handle,
                    new_buffers[1].handle,
                    index_regions,
                ),
                (
                    info_staging.handle,
                    new_buffers[2].handle,
                    match infos.is_empty() {
                        true => &[][..],
                        false => slice::from_ref(&info_region),
                    },
                ),
            ];
            record_submit_commandbuffer(
                logical_device,
                cmd_buffer,
                cmd_fence,
                queue,
                &[],
                &[],
                &[],
                |cmd_buffer| unsafe {
                    for (src, dst, regions) in copies {
                        if regions.is_empty() {
                            continue;
                        }
                        let copy_info = vk::CopyBufferInfo2::default()
                            .src_buffer(src)
                            .dst_buffer(dst)
                            .regions(regions);
                        logical_device.cmd_copy_buffer2(cmd_buffer, &copy_info);
                    }

                    // Every later frame draws, culls with and uploads into the new buffers.
                    let barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COPY)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::INDEX_INPUT
                                | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                                | vk::PipelineStageFlags2::COMPUTE_SHADER
                                | vk::PipelineStageFlags2::COPY,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2::INDEX_READ
                                | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                                | vk::AccessFlags2::SHADER_STORAGE_READ
                                | vk::AccessFlags2::SHADER_STORAGE_WRITE
                                | vk::AccessFlags2::TRANSFER_WRITE,
                        );
                    let dependency_info =
                        vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
                    logical_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
                },
            )?;
            unsafe { logical_device.wait_for_fences(&[cmd_fence], true, u64::MAX) }
        });

        unsafe { info_staging.destroy(logical_device) };
        copied
    }
}

/// Doubles `capacity` until it holds `needed`.
fn grown(capacity: u32, needed: u64) -> u32 {
    let mut grown = (capacity as u64).max(1);
    while grown < needed {
        grown *= 2;
    }
    grown.min(u32::MAX as u64) as u32
}

/// A copy moving `from` to `to`, in elements of `element_size` bytes, unless it's empty.
fn push_copy(
    regions: &mut Vec<vk::BufferCopy2<'static>>,
    from: &Range<u32>,
    to: &Range<u32>,
    element_size: vk::DeviceSize,
) {
    if from.is_empty() {
        return;
    }
    regions.push(
        vk::BufferCopy2::default()
            .src_offset(from.start as vk::DeviceSize * element_size)
            .dst_offset(to.start as vk::DeviceSize * element_size)
            .size(from.len() as vk::DeviceSize * element_size),
    );
}

#[cfg(test)]
mod tests {
    use super::grown;

    #[test]
    fn grows_by_doubling_until_it_fits() {
        assert_eq!(grown(100, 100), 100);
        assert_eq!(grown(100, 101), 200);
        assert_eq!(grown(100, 350), 400);
        assert_eq!(grown(0, 3), 4);
    }
}
