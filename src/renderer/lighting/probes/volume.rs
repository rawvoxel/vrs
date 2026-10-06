//! The probes' buffers, atlases and compute passes, updated each frame.

use super::{MAX_CASCADES, PROBE_CELL, ProbeConfig};
use crate::{
    renderer::{
        shader::bindings,
        timings::{FrameTimestamps, GpuStage},
    },
    vulkan::{
        adapter::Adapter, buffer::Buffer, image::StorageImage, pipeline::create_shader_module,
        sync::cmd_memory_barrier,
    },
};
use ash::{prelude::VkResult, vk};
use std::slice;

const PROBE_UPDATE_SPV: &[u8] = include_bytes!("../../../../assets/shader/probe_update.spv");
/// Interior texels along a probe's irradiance and distance maps, each with a one texel border.
/// Match `IRRADIANCE_TEXELS` and `DISTANCE_TEXELS` in `common/probes.slang`.
const IRRADIANCE_TEXELS: u32 = 8;
const DISTANCE_TEXELS: u32 = 14;
/// Base storage formats every device supports, and its bytes a texel.
const ATLAS_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const ATLAS_TEXEL_BYTES: vk::DeviceSize = 8;
/// Matches `Probe` in `common/layout.slang`.
const PROBE_SIZE: vk::DeviceSize = 32;
/// The fewest rays a probe traces in a frame, however many probes share the budget. Matches
/// `MIN_RAYS_PER_PROBE` in `probe_update.slang`.
const MIN_RAYS_PER_PROBE: vk::DeviceSize = 32;
/// Matches `numthreads` of `classifyProbes` in `probe_update.slang`.
const CLASSIFY_WORKGROUP: u32 = 64;
/// Words before the active probe list in its buffer: the tracing dispatch's group counts, the
/// updates' group counts, how many probes are active and how many rays each traces. Matches
/// `ACTIVE_LIST` in `probe_update.slang`.
const ACTIVE_LIST: vk::DeviceSize = 8;
/// Where the updates' group counts are in the active probe list's buffer, in bytes.
const UPDATE_GROUPS: vk::DeviceSize = 3 * size_of::<u32>() as vk::DeviceSize;
/// A probe cell that no probe holds, so every probe is placed on the first frame.
const UNPLACED: u32 = 0x8000_0000;

/// Probes along and down the atlases: nearly square, so they fit the smallest image limits
/// for longer.
fn atlas_layout(probes: u32) -> (u32, u32) {
    let columns = (probes as f64).sqrt().ceil() as u32;
    (columns, probes.div_ceil(columns))
}

/// Rays the ray buffer holds for `probes` probes: the budget, or the fewest rays each when
/// there are more active probes than it covers.
fn ray_capacity(probes: vk::DeviceSize, rays_per_frame: u32) -> vk::DeviceSize {
    (rays_per_frame as vk::DeviceSize).max(probes * MIN_RAYS_PER_PROBE)
}

/// Bytes of device memory `probes` probes take, tracing `rays_per_frame` rays a frame: their
/// buffers and atlases, as `ProbeVolume::new` makes them, see `ProbeConfig::device_bytes`.
pub(super) fn device_bytes(probes: u32, rays_per_frame: u32) -> vk::DeviceSize {
    let (columns, rows) = atlas_layout(probes);
    let atlas = |texels: u32| {
        let tile = (texels + 2) as vk::DeviceSize;
        columns as vk::DeviceSize * tile * rows as vk::DeviceSize * tile * ATLAS_TEXEL_BYTES
    };
    let probes = probes as vk::DeviceSize;
    probes * PROBE_SIZE
        + ray_capacity(probes, rays_per_frame) * size_of::<[f32; 4]>() as vk::DeviceSize
        + (ACTIVE_LIST + probes) * size_of::<u32>() as vk::DeviceSize
        + atlas(IRRADIANCE_TEXELS)
        + atlas(DISTANCE_TEXELS)
}

/// A probe volume's buffers, atlases and compute pipelines.
///
/// Probes are stored by their cell modulo the volume's size, so as the volume follows the
/// camera, probes leaving it are reused by those entering it, which `classifyProbes` notices
/// and resets. Both atlases stay in `GENERAL` layout, written as storage images and sampled.
pub(crate) struct ProbeVolume {
    /// Probes along each axis of each cascade.
    pub counts: [u32; 3],
    pub cascades: u32,
    /// The level of the voxel grid whose `PROBE_CELL` voxels the first cascade's cells are,
    /// see `ProbeConfig::spacing`.
    pub first_level: u32,
    pub rays_per_frame: u32,
    pub atlas_columns: u32,
    pub probe_buffer: Buffer,
    pub ray_buffer: Buffer,
    pub active_buffer: Buffer,
    pub irradiance: StorageImage,
    pub distances: StorageImage,
    pub sampler: vk::Sampler,
    pub classify: vk::Pipeline,
    pub trace: vk::Pipeline,
    pub update_irradiance: vk::Pipeline,
    pub update_distances: vk::Pipeline,
    /// Shares out the frame's rays once the active probes are counted.
    pub plan: vk::Pipeline,
}

impl ProbeVolume {
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            counts: [0; 3],
            cascades: 0,
            first_level: 0,
            rays_per_frame: 0,
            atlas_columns: 0,
            probe_buffer: Buffer::null(),
            ray_buffer: Buffer::null(),
            active_buffer: Buffer::null(),
            irradiance: StorageImage::null(),
            distances: StorageImage::null(),
            sampler: vk::Sampler::null(),
            classify: vk::Pipeline::null(),
            trace: vk::Pipeline::null(),
            update_irradiance: vk::Pipeline::null(),
            plan: vk::Pipeline::null(),
            update_distances: vk::Pipeline::null(),
        }
    }

    /// Probes for `config`, or a single one for the descriptors when there's none, run with
    /// `layout`, the frame descriptor set's pipeline layout. Needs `cmd_init` before use.
    pub fn new(
        adapter: &Adapter,
        device: &ash::Device,
        config: Option<ProbeConfig>,
        layout: vk::PipelineLayout,
    ) -> VkResult<Self> {
        let mut volume = Self::null();
        volume.counts = config.map_or([1; 3], |config| config.probe_counts.map(|n| n.max(1)));
        volume.cascades = config.map_or(1, |config| config.cascades.clamp(1, MAX_CASCADES));
        volume.first_level =
            config.map_or(0, |config| (config.spacing / PROBE_CELL).max(1).ilog2());
        volume.rays_per_frame = config.map_or(0, |config| config.rays_per_frame);
        if let Err(err) = volume.create(adapter, device, layout) {
            unsafe { volume.destroy(device) };
            return Err(err);
        }
        Ok(volume)
    }

    /// Every cascade's.
    #[inline]
    pub fn probe_count(&self) -> u32 {
        self.counts.iter().product::<u32>() * self.cascades
    }

    /// Fills in a null volume, leaving whatever it created behind if it fails.
    fn create(
        &mut self,
        adapter: &Adapter,
        device: &ash::Device,
        layout: vk::PipelineLayout,
    ) -> VkResult<()> {
        let probes = self.probe_count() as vk::DeviceSize;
        let storage_buffer = |size: vk::DeviceSize, usage: vk::BufferUsageFlags| {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            Buffer::new(
                adapter,
                device,
                info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                0,
            )
        };
        // Read back too, see `Renderer::debug_probes`.
        self.probe_buffer = storage_buffer(
            probes * PROBE_SIZE,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        self.ray_buffer = storage_buffer(
            ray_capacity(probes, self.rays_per_frame) * size_of::<[f32; 4]>() as vk::DeviceSize,
            vk::BufferUsageFlags::empty(),
        )?;
        self.active_buffer = storage_buffer(
            (ACTIVE_LIST + probes) * size_of::<u32>() as vk::DeviceSize,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let rows;
        (self.atlas_columns, rows) = atlas_layout(probes as u32);
        let max_dimension = adapter.properties.limits.max_image_dimension2_d;
        if self.atlas_columns.max(rows) * (DISTANCE_TEXELS + 2) > max_dimension {
            return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
        }
        for (atlas, texels) in [
            (&mut self.irradiance, IRRADIANCE_TEXELS),
            (&mut self.distances, DISTANCE_TEXELS),
        ] {
            let tile = texels + 2;
            let extent = vk::Extent2D {
                width: self.atlas_columns * tile,
                height: rows * tile,
            };
            // Cleared by `cmd_init`.
            *atlas = StorageImage::new(
                adapter,
                device,
                extent,
                ATLAS_FORMAT,
                vk::ImageUsageFlags::TRANSFER_DST,
            )?;
        }

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        self.sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        let module = create_shader_module(device, PROBE_UPDATE_SPV)?;
        let entries = [
            c"classifyProbes",
            c"planRays",
            c"traceProbeRays",
            c"updateIrradiance",
            c"updateDistances",
        ];
        let stages = entries.map(|entry| {
            vk::PipelineShaderStageCreateInfo::default()
                .module(module)
                .name(entry)
                .stage(vk::ShaderStageFlags::COMPUTE)
        });
        let infos = stages.map(|stage| {
            vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)
        });
        let created =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None) };
        unsafe { device.destroy_shader_module(module, None) };
        let pipelines = match created {
            Ok(pipelines) => pipelines,
            Err((pipelines, err)) => {
                for pipeline in pipelines {
                    unsafe { device.destroy_pipeline(pipeline, None) };
                }
                return Err(err);
            }
        };
        [
            self.classify,
            self.plan,
            self.trace,
            self.update_irradiance,
            self.update_distances,
        ] = [
            pipelines[0],
            pipelines[1],
            pipelines[2],
            pipelines[3],
            pipelines[4],
        ];
        Ok(())
    }

    /// Records readying the atlases, cleared, and marking every probe unplaced.
    ///
    /// # Safety
    /// Nothing may be using the volume.
    pub unsafe fn cmd_init(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let atlases = [self.irradiance.image.handle, self.distances.image.handle];
        let to_general = atlases.map(|image| {
            vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(range)
        });
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&to_general),
            );
            for image in atlases {
                device.cmd_clear_color_image(
                    cmd_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue::default(),
                    slice::from_ref(&range),
                );
            }
            device.cmd_fill_buffer(
                cmd_buffer,
                self.probe_buffer.handle,
                0,
                vk::WHOLE_SIZE,
                UNPLACED,
            );
            let ready = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                )
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
            device.cmd_pipeline_barrier2(
                cmd_buffer,
                &vk::DependencyInfo::default().memory_barriers(slice::from_ref(&ready)),
            );
        }
    }

    /// Points the probes' bindings of every frame's descriptor set at the volume. They never
    /// move.
    pub fn write_descriptors(&self, device: &ash::Device, sets: &[vk::DescriptorSet]) {
        let buffer = |buffer: &Buffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer.handle)
                .range(vk::WHOLE_SIZE)
        };
        let image = |view: vk::ImageView, sampler: vk::Sampler| {
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .sampler(sampler)
                .image_layout(vk::ImageLayout::GENERAL)
        };
        let probes = buffer(&self.probe_buffer);
        let rays = buffer(&self.ray_buffer);
        let active = buffer(&self.active_buffer);
        let irradiance_sampled = image(self.irradiance.view, self.sampler);
        let distances_sampled = image(self.distances.view, self.sampler);
        let irradiance_storage = image(self.irradiance.view, vk::Sampler::null());
        let distances_storage = image(self.distances.view, vk::Sampler::null());
        for &set in sets {
            let write = |binding: u32| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(bindings::TYPES_BY_BINDING[binding as usize])
            };
            let writes = [
                write(bindings::PROBES).buffer_info(slice::from_ref(&probes)),
                write(bindings::PROBE_IRRADIANCE).image_info(slice::from_ref(&irradiance_sampled)),
                write(bindings::PROBE_DISTANCES).image_info(slice::from_ref(&distances_sampled)),
                write(bindings::PROBE_IRRADIANCE_STORAGE)
                    .image_info(slice::from_ref(&irradiance_storage)),
                write(bindings::PROBE_DISTANCES_STORAGE)
                    .image_info(slice::from_ref(&distances_storage)),
                write(bindings::PROBE_RAYS).buffer_info(slice::from_ref(&rays)),
                write(bindings::ACTIVE_PROBES).buffer_info(slice::from_ref(&active)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    /// Records this frame's probe update: placing and choosing the probes near voxels, sharing
    /// out the frame's rays, tracing them, then blending the results into the atlases, which
    /// the frame's shading then samples. `set` is the frame's descriptor set, bound with
    /// `layout`.
    ///
    /// # Safety
    /// The frame's uniforms must be written, and the mesh pool's uploads recorded before.
    pub unsafe fn cmd_update(
        &self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        set: vk::DescriptorSet,
        layout: vk::PipelineLayout,
        timestamps: &FrameTimestamps,
    ) {
        let barrier = |src, dst| unsafe { cmd_memory_barrier(device, cmd_buffer, src, dst) };
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let write = vk::AccessFlags2::SHADER_WRITE;
        let read_write = vk::AccessFlags2::SHADER_READ | write;
        let dispatch = |pipeline: vk::Pipeline, offset: vk::DeviceSize| unsafe {
            device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_dispatch_indirect(cmd_buffer, self.active_buffer.handle, offset);
        };

        unsafe {
            // Last frame's shading and updates are done with the volume, and its dispatches
            // with the active count.
            barrier(
                (
                    compute
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT,
                    write,
                ),
                (
                    compute | vk::PipelineStageFlags2::CLEAR,
                    read_write | vk::AccessFlags2::TRANSFER_WRITE,
                ),
            );
            // No active probes yet: dispatches one group high and deep, none wide.
            let groups: [u32; ACTIVE_LIST as usize] = [0, 1, 1, 0, 1, 1, 0, 0];
            device.cmd_update_buffer(
                cmd_buffer,
                self.active_buffer.handle,
                0,
                crate::vulkan::bytes::slice_as_bytes(&groups),
            );
            barrier(
                (
                    vk::PipelineStageFlags2::CLEAR,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                (compute, read_write),
            );

            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                slice::from_ref(&set),
                &[],
            );
            device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, self.classify);
            device.cmd_dispatch(
                cmd_buffer,
                self.probe_count().div_ceil(CLASSIFY_WORKGROUP),
                1,
                1,
            );
            barrier((compute, write), (compute, read_write));
            device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, self.plan);
            device.cmd_dispatch(cmd_buffer, 1, 1, 1);
            timestamps.cmd_mark(device, cmd_buffer, GpuStage::ProbeClassification);
            barrier(
                (compute, write),
                (
                    compute | vk::PipelineStageFlags2::DRAW_INDIRECT,
                    read_write | vk::AccessFlags2::INDIRECT_COMMAND_READ,
                ),
            );

            // One group per 32 rays, then one per active probe.
            dispatch(self.trace, 0);
            timestamps.cmd_mark(device, cmd_buffer, GpuStage::ProbeRays);
            // The rays are written, and tracing is done sampling the atlases.
            barrier((compute, write), (compute, read_write));
            dispatch(self.update_irradiance, UPDATE_GROUPS);
            dispatch(self.update_distances, UPDATE_GROUPS);
            timestamps.cmd_mark(device, cmd_buffer, GpuStage::ProbeUpdates);
            barrier(
                (compute, write),
                (
                    vk::PipelineStageFlags2::FRAGMENT_SHADER | compute,
                    vk::AccessFlags2::SHADER_READ,
                ),
            );
        }
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for pipeline in [
                self.classify,
                self.plan,
                self.trace,
                self.update_irradiance,
                self.update_distances,
            ] {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_sampler(self.sampler, None);
            self.irradiance.destroy(device);
            self.distances.destroy(device);
            for buffer in [&self.probe_buffer, &self.ray_buffer, &self.active_buffer] {
                buffer.destroy(device);
            }
        }
    }
}
