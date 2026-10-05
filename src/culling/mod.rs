//! Culling meshes on the GPU: against the view, then against a depth pyramid of what was
//! drawn, see `Culler`.

pub(crate) mod depth_pyramid;

use crate::{
    culling::depth_pyramid::DepthPyramid,
    mesh::MeshPool,
    vulkan::{
        adapter::Adapter,
        buffer::{Buffer, DynamicBuffer},
        bytes::as_bytes,
        descriptors::{
            allocate_descriptor_sets, create_descriptor_pool, create_descriptor_set_layout,
        },
        pipeline::create_compute_pipeline,
    },
};
use ash::{prelude::VkResult, vk};
use glam::Mat4;
#[cfg(test)]
use glam::Vec4;
use std::slice;

const CULL_SPV: &[u8] = include_bytes!("../../assets/shader/cull.spv");
/// Matches `numthreads` in `cull.slang`.
const WORKGROUP_SIZE: u32 = 64;
/// Starting number of meshes each frame's buffers fit, they grow to fit.
const INITIAL_CANDIDATES: vk::DeviceSize = 1024;
pub const DRAW_STRIDE: vk::DeviceSize =
    size_of::<vk::DrawIndexedIndirectCommand>() as vk::DeviceSize;

/// Which culling pass to run, matches the `PHASE_` constants in `cull.slang`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Frustum culling only, without occlusion.
    Frustum = 0,
    /// Draws meshes visible last frame, to build the depth pyramid from.
    LastVisible = 1,
    /// Tests meshes against the depth pyramid, drawing newly visible ones after
    /// phase 1's draws and remembering what's visible for next frame.
    Occlusion = 2,
}

impl Phase {
    /// Which half of the frame's draws, and which draw count, this phase writes.
    #[inline]
    pub fn draw_region(self) -> u32 {
        (self == Phase::Occlusion) as u32
    }
}

/// Matches `CullParams` in `cull.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CullParams {
    pub view_proj_rows: [[f32; 4]; 4],
    pub depth_size: [u32; 2],
    pub candidate_count: u32,
    pub phase: u32,
    pub pyramid_levels: u32,
}

/// Culls a frame's meshes on the device, writing indirect draws for the visible ones.
///
/// Each frame's draw buffer has room for two phases of draws, see `Phase`. Visible draws
/// are packed together and counted on the device, for `cmd_draw_indexed_indirect_count`.
pub struct Culler {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub frame_desc_sets: Vec<vk::DescriptorSet>,
    /// `MeshPool::info_buffer` indices to cull, written by the host.
    pub frame_candidate_buffers: Vec<DynamicBuffer>,
    pub frame_draw_buffers: Vec<DynamicBuffer>,
    /// One draw count per phase region.
    pub frame_count_buffers: Vec<Buffer>,
}

impl Culler {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        frames_in_flight: u32,
    ) -> VkResult<Self> {
        let mut culler = Self::null();
        if let Err(err) = culler.create(adapter, logical_device, frames_in_flight) {
            unsafe { culler.destroy(logical_device) };
            return Err(err);
        }
        Ok(culler)
    }

    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            desc_set_layout: vk::DescriptorSetLayout::null(),
            desc_pool: vk::DescriptorPool::null(),
            frame_desc_sets: Vec::new(),
            frame_candidate_buffers: Vec::new(),
            frame_draw_buffers: Vec::new(),
            frame_count_buffers: Vec::new(),
        }
    }

    /// Fills in a null culler, leaving whatever it created behind if it fails.
    fn create(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        frames_in_flight: u32,
    ) -> VkResult<()> {
        // Mesh infos, candidates, draws, draw counts and the depth pyramid.
        self.desc_set_layout = create_descriptor_set_layout(
            logical_device,
            &[
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::SAMPLED_IMAGE,
            ],
            vk::ShaderStageFlags::COMPUTE,
        )?;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4 * frames_in_flight),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(frames_in_flight),
        ];
        self.desc_pool = create_descriptor_pool(logical_device, &pool_sizes, frames_in_flight)?;
        self.frame_desc_sets = allocate_descriptor_sets(
            logical_device,
            self.desc_pool,
            self.desc_set_layout,
            frames_in_flight,
        )?;

        (self.pipeline, self.pipeline_layout) = create_compute_pipeline(
            logical_device,
            CULL_SPV,
            c"cullMain",
            self.desc_set_layout,
            size_of::<CullParams>() as u32,
        )?;

        let count_info = vk::BufferCreateInfo::default()
            .size(2 * size_of::<u32>() as vk::DeviceSize)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    // Lets the counts be read back when debugging.
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        for _ in 0..frames_in_flight {
            self.frame_candidate_buffers.push(DynamicBuffer::new(
                adapter,
                logical_device,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                INITIAL_CANDIDATES * size_of::<u32>() as vk::DeviceSize,
            )?);
            self.frame_draw_buffers.push(DynamicBuffer::with_memory(
                adapter,
                logical_device,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                2 * INITIAL_CANDIDATES * DRAW_STRIDE,
            )?);
            self.frame_count_buffers.push(Buffer::new(
                adapter,
                logical_device,
                count_info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                0,
            )?);
        }

        Ok(())
    }

    /// Writes this frame's `candidates`, indices into `mesh_pool.info_buffer`, binds its
    /// buffers and records resetting its draw counts. Call once a frame before `cmd_cull`.
    ///
    /// # Safety
    /// The device must not be using this frame's buffers or descriptor set.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        mesh_pool: &MeshPool,
        depth_pyramid: &DepthPyramid,
        frame: usize,
        cmd_buffer: vk::CommandBuffer,
        candidates: &[u32],
    ) -> VkResult<()> {
        let draws_size = 2 * candidates.len() as vk::DeviceSize * DRAW_STRIDE;
        unsafe {
            self.frame_candidate_buffers[frame].write(adapter, logical_device, candidates)?;
            self.frame_draw_buffers[frame].reserve(adapter, logical_device, draws_size)?;
        }
        let count_buffer = self.frame_count_buffers[frame].handle;

        // Buffers may have grown since last frame, and the device isn't using this frame's set.
        let buffer_infos = [
            mesh_pool.info_buffer.handle,
            self.frame_candidate_buffers[frame].buffer.handle,
            self.frame_draw_buffers[frame].buffer.handle,
            count_buffer,
        ]
        .map(|buffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(vk::WHOLE_SIZE)
        });
        let pyramid_info = vk::DescriptorImageInfo::default()
            .image_view(depth_pyramid.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let desc_set = self.frame_desc_sets[frame];
        let mut writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(binding, buffer_info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(buffer_info))
            })
            .collect();
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(slice::from_ref(&pyramid_info)),
        );

        unsafe {
            logical_device.update_descriptor_sets(&writes, &[]);

            // Last frame's phase 2 updated mesh visibility, which this frame reads.
            // The draw counts are reset.
            logical_device.cmd_fill_buffer(cmd_buffer, count_buffer, 0, vk::WHOLE_SIZE, 0);
            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::CLEAR,
                )
                .src_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
                )
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                );
            let dependency_info =
                vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
            logical_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        }

        Ok(())
    }

    /// Records culling this frame's candidates against `view_proj`, after which the
    /// phase's draws are ready for `DRAW_INDIRECT`. `Phase::Occlusion` needs the depth
    /// pyramid built and ready for compute reads.
    ///
    /// # Safety
    /// `prepare` must have been called this frame.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cmd_cull(
        &self,
        logical_device: &ash::Device,
        depth_pyramid: &DepthPyramid,
        frame: usize,
        cmd_buffer: vk::CommandBuffer,
        candidate_count: u32,
        view_proj: Mat4,
        phase: Phase,
    ) {
        if candidate_count == 0 {
            return;
        }

        let params = CullParams {
            view_proj_rows: [0, 1, 2, 3].map(|row| view_proj.row(row).to_array()),
            depth_size: [
                depth_pyramid.depth_extent.width,
                depth_pyramid.depth_extent.height,
            ],
            candidate_count,
            phase: phase as u32,
            pyramid_levels: depth_pyramid.levels,
        };

        unsafe {
            logical_device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            logical_device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                slice::from_ref(&self.frame_desc_sets[frame]),
                &[],
            );
            logical_device.cmd_push_constants(
                cmd_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                as_bytes(&params),
            );
            logical_device.cmd_dispatch(cmd_buffer, candidate_count.div_ceil(WORKGROUP_SIZE), 1, 1);

            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ);
            let dependency_info =
                vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
            logical_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        }
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            for buffer in self
                .frame_candidate_buffers
                .iter()
                .chain(&self.frame_draw_buffers)
            {
                buffer.destroy(logical_device);
            }
            for buffer in self.frame_count_buffers.iter() {
                buffer.destroy(logical_device);
            }
            logical_device.destroy_descriptor_pool(self.desc_pool, None);
            logical_device.destroy_descriptor_set_layout(self.desc_set_layout, None);
            logical_device.destroy_pipeline(self.pipeline, None);
            logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

/// The inward facing planes of `view_proj`'s frustum, for Vulkan's `0..1` depth range.
/// A point `p` is inside a plane when `plane.xyz.dot(p) + plane.w >= 0`.
/// Matches `inFrustum` in `cull.slang`, which tests check it against.
#[cfg(test)]
pub fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
    let [x, y, z, w] = [0, 1, 2, 3].map(|row| view_proj.row(row));
    [w + x, w - x, w + y, w - y, z, w - z]
}

#[cfg(test)]
mod tests {
    use super::frustum_planes;
    use glam::{Mat4, Vec3};

    fn inside(view_proj: Mat4, point: Vec3) -> bool {
        frustum_planes(view_proj)
            .iter()
            .all(|plane| plane.truncate().dot(point) + plane.w >= 0.0)
    }

    #[test]
    fn identity_frustum_is_the_clip_volume() {
        let view_proj = Mat4::IDENTITY;
        assert!(inside(view_proj, Vec3::new(0.0, 0.0, 0.5)));
        assert!(inside(view_proj, Vec3::new(1.0, -1.0, 0.0)));
        assert!(!inside(view_proj, Vec3::new(1.1, 0.0, 0.5)));
        assert!(!inside(view_proj, Vec3::new(0.0, -1.1, 0.5)));
        assert!(!inside(view_proj, Vec3::new(0.0, 0.0, -0.1)));
        assert!(!inside(view_proj, Vec3::new(0.0, 0.0, 1.1)));
    }

    #[test]
    fn perspective_frustum_sees_ahead_not_behind() {
        let proj = Mat4::perspective_rh(90f32.to_radians(), 1.0, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let view_proj = proj * view;
        assert!(inside(view_proj, Vec3::new(0.0, 0.0, -5.0)));
        assert!(inside(view_proj, Vec3::new(4.9, 0.0, -5.0)));
        assert!(!inside(view_proj, Vec3::new(5.1, 0.0, -5.0)));
        assert!(!inside(view_proj, Vec3::new(0.0, 0.0, 5.0)));
        assert!(!inside(view_proj, Vec3::new(0.0, 0.0, -0.05)));
        assert!(!inside(view_proj, Vec3::new(0.0, 0.0, -101.0)));
    }
}
