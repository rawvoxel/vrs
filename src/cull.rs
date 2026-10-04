use crate::{
    adapter::Adapter,
    buffer::{Buffer, DynamicBuffer},
    descriptors::create_descriptor_pool,
    mesh::MeshPool,
    renderer::create_shader_module,
};
use ash::{prelude::VkResult, vk};
use glam::{Mat4, Vec4};
use std::slice;

const CULL_SPV: &[u8] = include_bytes!("../assets/shader/cull.spv");
/// Matches `numthreads` in `cull.slang`.
const WORKGROUP_SIZE: u32 = 64;
/// Starting number of meshes each frame's buffers fit, they grow to fit.
const INITIAL_CANDIDATES: vk::DeviceSize = 1024;
const DRAW_STRIDE: vk::DeviceSize = size_of::<vk::DrawIndexedIndirectCommand>() as vk::DeviceSize;

/// Matches `CullParams` in `cull.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CullParams {
    pub planes: [[f32; 4]; 6],
    pub candidate_count: u32,
    pub compact: u32,
}

/// Frustum culls a frame's meshes on the device, writing indirect draws for the visible ones.
///
/// Compacted draws are counted on the device for `cmd_draw_indexed_indirect_count`, which
/// needs the `drawIndirectCount` feature. Otherwise every mesh keeps a draw, with
/// culled ones drawing zero instances.
pub struct Culler {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub frame_desc_sets: Vec<vk::DescriptorSet>,
    /// `MeshPool::info_buffer` indices to cull, written by the host.
    pub frame_candidate_buffers: Vec<DynamicBuffer>,
    pub frame_draw_buffers: Vec<DynamicBuffer>,
    pub frame_count_buffers: Vec<Buffer>,
    /// Whether the device supports counting draws, see `Culler`.
    pub can_compact: bool,
}

impl Culler {
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        frames_in_flight: u32,
        can_compact: bool,
    ) -> VkResult<Self> {
        let desc_set_layout = create_desc_set_layout(logical_device)?;
        let desc_pool = create_descriptor_pool(
            logical_device,
            vk::DescriptorType::STORAGE_BUFFER,
            4 * frames_in_flight,
            frames_in_flight,
        )?;
        let set_layouts = vec![desc_set_layout; frames_in_flight as usize];
        let desc_set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);
        let frame_desc_sets = unsafe { logical_device.allocate_descriptor_sets(&desc_set_info)? };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<CullParams>() as u32);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(slice::from_ref(&desc_set_layout))
            .push_constant_ranges(slice::from_ref(&push_constant_range));
        let pipeline_layout =
            unsafe { logical_device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let shader_module = create_shader_module(logical_device, CULL_SPV)?;
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .module(shader_module)
            .name(c"cullMain")
            .stage(vk::ShaderStageFlags::COMPUTE);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            let pipelines = logical_device.create_compute_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&pipeline_info),
                None,
            );
            logical_device.destroy_shader_module(shader_module, None);
            pipelines.map_err(|(_, err)| err)?[0]
        };

        let frame_candidate_buffers = (0..frames_in_flight)
            .map(|_| {
                DynamicBuffer::new(
                    adapter,
                    logical_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    INITIAL_CANDIDATES * size_of::<u32>() as vk::DeviceSize,
                )
            })
            .collect::<VkResult<Vec<_>>>()?;
        let frame_draw_buffers = (0..frames_in_flight)
            .map(|_| {
                DynamicBuffer::with_memory(
                    adapter,
                    logical_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    INITIAL_CANDIDATES * DRAW_STRIDE,
                )
            })
            .collect::<VkResult<Vec<_>>>()?;
        let count_info = vk::BufferCreateInfo::default()
            .size(size_of::<u32>() as vk::DeviceSize)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    // Lets the count be read back when debugging.
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let frame_count_buffers = (0..frames_in_flight)
            .map(|_| {
                Buffer::new(
                    adapter,
                    logical_device,
                    count_info,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    0,
                )
            })
            .collect::<VkResult<Vec<_>>>()?;

        Ok(Self {
            pipeline,
            pipeline_layout,
            desc_set_layout,
            desc_pool,
            frame_desc_sets,
            frame_candidate_buffers,
            frame_draw_buffers,
            frame_count_buffers,
            can_compact,
        })
    }

    /// Records culling `candidates`, indices into `mesh_pool.info_buffer`, against the
    /// frustum of `view_proj`. Draws are ready for `DRAW_INDIRECT` afterwards, compacted
    /// and counted in `frame_count_buffers[frame]` when `compact`.
    ///
    /// # Safety
    /// The device must not be using this frame's buffers, and `compact` needs `can_compact`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cmd_cull(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        mesh_pool: &MeshPool,
        frame: usize,
        cmd_buffer: vk::CommandBuffer,
        candidates: &[u32],
        view_proj: Mat4,
        compact: bool,
    ) -> VkResult<()> {
        debug_assert!(!compact || self.can_compact);
        let candidate_count = candidates.len() as u32;
        if candidate_count == 0 {
            return Ok(());
        }

        let draw_buffer = &mut self.frame_draw_buffers[frame];
        unsafe {
            self.frame_candidate_buffers[frame].write(adapter, logical_device, candidates)?;
            draw_buffer.reserve(adapter, logical_device, candidates.len() as u64 * DRAW_STRIDE)?;
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
        let desc_set = self.frame_desc_sets[frame];
        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
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

        let params = CullParams {
            planes: frustum_planes(view_proj).map(|plane| plane.to_array()),
            candidate_count,
            compact: compact as u32,
        };

        unsafe {
            logical_device.update_descriptor_sets(&writes, &[]);

            if compact {
                logical_device.cmd_fill_buffer(cmd_buffer, count_buffer, 0, vk::WHOLE_SIZE, 0);
                let clear_barrier = vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_STORAGE_READ
                            | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    );
                let clear_dependency =
                    vk::DependencyInfo::default().memory_barriers(slice::from_ref(&clear_barrier));
                logical_device.cmd_pipeline_barrier2(cmd_buffer, &clear_dependency);
            }

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
                slice::from_ref(&desc_set),
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

            let draw_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ);
            let draw_dependency =
                vk::DependencyInfo::default().memory_barriers(slice::from_ref(&draw_barrier));
            logical_device.cmd_pipeline_barrier2(cmd_buffer, &draw_dependency);
        }

        Ok(())
    }

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
pub fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
    let [x, y, z, w] = [0, 1, 2, 3].map(|row| view_proj.row(row));
    [w + x, w - x, w + y, w - y, z, w - z]
}

#[inline]
fn as_bytes<T: Copy>(data: &T) -> &[u8] {
    // SAFETY: Only used for plain data without implicit padding.
    unsafe { slice::from_raw_parts((data as *const T).cast(), size_of::<T>()) }
}

#[inline]
fn create_desc_set_layout(logical_device: &ash::Device) -> VkResult<vk::DescriptorSetLayout> {
    // Mesh infos, candidates, draws and the draw count.
    let bindings = [0, 1, 2, 3].map(|binding| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
    });
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe { logical_device.create_descriptor_set_layout(&info, None) }
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
