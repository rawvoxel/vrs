//! Dynamic diffuse global illumination: a volume of light probes around the camera, each
//! storing the light reaching it from every direction and how far away the nearest voxels
//! are, traced through the voxels every frame and sampled when shading. See `gi.slang`.

use crate::{
    adapter::Adapter,
    buffer::Buffer,
    image::{Image, create_image_view},
    pipeline::create_shader_module,
    timings::{FrameTimestamps, Mark},
    voxel::VoxelTypes,
};
use ash::{prelude::VkResult, vk};
use std::slice;

const GI_SPV: &[u8] = include_bytes!("../assets/shader/gi.spv");
/// Interior texels along a probe's irradiance and distance maps, each with a one texel border.
/// Match `IRRADIANCE_TEXELS` and `DEPTH_TEXELS` in `common.slang`.
const IRRADIANCE_TEXELS: u32 = 8;
const DEPTH_TEXELS: u32 = 14;
/// Base storage formats every device supports.
const ATLAS_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Matches `Probe` in `common.slang`.
const PROBE_SIZE: vk::DeviceSize = 32;
/// The fewest rays a probe traces in a frame, however many probes share the budget. Matches
/// `MIN_RAYS_PER_PROBE` in `gi.slang`.
const MIN_RAYS_PER_PROBE: vk::DeviceSize = 32;
/// Matches `numthreads` of `classifyProbes` in `gi.slang`.
const CLASSIFY_WORKGROUP: u32 = 64;
/// Words before the active probe list in its buffer: the tracing dispatch's group counts, the
/// updates' group counts, how many probes are active and how many rays each traces. Matches
/// `ACTIVE_LIST` in `gi.slang`.
const ACTIVE_LIST: vk::DeviceSize = 8;
/// Where the updates' group counts are in the active probe list's buffer, in bytes.
const UPDATE_GROUPS: vk::DeviceSize = 3 * size_of::<u32>() as vk::DeviceSize;
/// A probe cell that no probe holds, so every probe is placed on the first frame.
const UNPLACED: u32 = 0x8000_0000;

/// Voxels along each edge of a probe's cell, and between neighboring probes, in the finest
/// cascade. Matches `PROBE_SPACING` in `common.slang`.
pub const PROBE_CELL: u32 = 16;
/// The most cascades of probes. Matches `MAX_CASCADES` in `common.slang`.
pub const MAX_CASCADES: u32 = 4;
/// Set in a cell's baked word when it has room for a probe. Matches `CELL_PLACED` in `gi.slang`.
const CELL_PLACED: u32 = 1 << 31;

/// Where each probe cell of a chunk puts its probe, and which cells have voxels, for chunks a
/// multiple of `PROBE_CELL` voxels wide, else nothing. Cells are x-fastest.
///
/// First a word per cell: the probe's position in the cell in half voxels, 6 bits per axis
/// from bit 0, and `CELL_PLACED`. Each probe goes in the largest empty part of its cell, the
/// octree node of the cell nearest its center at the largest size with an empty one, so it's
/// as far from the voxels as can be and can see around. An empty cell's probe is at its
/// center, a solid cell has none.
///
/// Then a bit per cell, 32 to a word, set when any of its voxels isn't empty, so the probes
/// near voxels can be found from a few words per chunk.
pub(crate) fn bake_probe_cells(types: &VoxelTypes) -> Vec<u32> {
    let edge = types.voxels_per_edge();
    if edge == 0 || !edge.is_multiple_of(PROBE_CELL) {
        return Vec::new();
    }
    let solids = types.solids();
    let cells = edge / PROBE_CELL;
    let solid_in = |min: [u32; 3], size: u32| {
        (0..size).any(|z| {
            (0..size).any(|y| {
                let row = (min[0] + edge * (min[1] + y + edge * (min[2] + z))) as usize;
                solids[row..row + size as usize].contains(&true)
            })
        })
    };
    let cell_min = |cell: u32| {
        [cell % cells, cell / cells % cells, cell / (cells * cells)].map(|axis| axis * PROBE_CELL)
    };
    let mut words: Vec<u32> = (0..cells.pow(3))
        .map(|cell| {
            let cell_min = cell_min(cell);
            // Breadth first, each size's nodes nearest the center first.
            let mut size = PROBE_CELL;
            while size > 0 {
                let nodes = PROBE_CELL / size;
                let mut nearest: Option<(u32, [u32; 3])> = None;
                for node in 0..nodes.pow(3) {
                    let node = [node % nodes, node / nodes % nodes, node / (nodes * nodes)];
                    // In half voxels, so a single voxel's center is whole.
                    let center = node.map(|axis| axis * size * 2 + size);
                    let distance: u32 = center
                        .iter()
                        .map(|&axis| axis.abs_diff(PROBE_CELL).pow(2))
                        .sum();
                    if nearest.is_some_and(|(best, _)| distance >= best) {
                        continue;
                    }
                    let min = [0, 1, 2].map(|axis| cell_min[axis] + node[axis] * size);
                    if !solid_in(min, size) {
                        nearest = Some((distance, center));
                    }
                }
                if let Some((_, [x, y, z])) = nearest {
                    return x | (y << 6) | (z << 12) | CELL_PLACED;
                }
                size /= 2;
            }
            0
        })
        .collect();
    let mut has_voxels = vec![0; cells.pow(3).div_ceil(32) as usize];
    for cell in 0..cells.pow(3) {
        if solid_in(cell_min(cell), PROBE_CELL) {
            has_voxels[(cell / 32) as usize] |= 1 << (cell % 32);
        }
    }
    words.extend(has_voxels);
    words
}

/// The light probe volumes, see `RendererConfig::global_illumination`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GiConfig {
    /// Probes along each axis of each cascade, which is centered on the camera with probes
    /// 16 of the `ShadowGrid`'s finest voxels apart in the first, and twice as far apart in
    /// each next. Each probe takes about 2.9 KB.
    pub probe_counts: [u32; 3],
    /// Cascades of probes, from 1 to `MAX_CASCADES`. Each reaches twice as far as the last,
    /// with probes twice as far apart, for chunks of coarser `ShadowGrid::levels`. Surfaces
    /// are lit by the finest one around them.
    pub cascades: u32,
    /// Rays traced each frame, shared between every cascade's probes near voxels so the cost
    /// stays steady: each traces a power of two of them from 32 to 256. Past
    /// `rays_per_frame / 32` such probes, each still traces 32.
    pub rays_per_frame: u32,
}

impl Default for GiConfig {
    /// One cascade, 512 by 256 by 512 voxels, with 64 Ki rays a frame.
    fn default() -> Self {
        Self {
            probe_counts: [32, 16, 32],
            cascades: 1,
            rays_per_frame: 1 << 16,
        }
    }
}

/// `Lighting::global_illumination` as `common.slang` reads it. Matches `GiUniforms` there.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GiUniforms {
    /// Each cascade's probe cell at its low corner, in its own cells, w is 1 when the probes
    /// light the frame.
    pub volume_mins: [[i32; 4]; MAX_CASCADES as usize],
    /// Probes along each axis of each cascade, w is probes per row of the atlases.
    pub volume_counts: [u32; 4],
    /// This frame's random rotation of the probes' rays, a quaternion.
    pub ray_rotation: [f32; 4],
    /// The camera's world position, w unused.
    pub camera_position: [f32; 4],
    /// x is the frame's number, y the rays traced each frame, z how many cascades there are, w
    /// the frames between the finest cascade's probes' updates.
    pub frame: [u32; 4],
    /// `ProbeTuning`'s hysteresis, depth hysteresis, depth sharpness and self-shadow bias.
    pub tuning: [f32; 4],
    /// `ProbeTuning`'s ray distance, the rest unused.
    pub rays: [f32; 4],
}

/// How the light probes blend what they see and are sampled, see `Lighting::probe_tuning`.
/// The defaults are the papers' [Majercik et al. 2019 and 2021].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeTuning {
    /// How much of a probe's light each frame keeps, from 0 to 1. Higher is steadier but
    /// slower to follow changes.
    pub hysteresis: f32,
    /// How much of a probe's distances each frame keeps, from 0 to 1.
    pub depth_hysteresis: f32,
    /// How narrowly each distance texel averages its rays. Higher follows voxels' edges more
    /// closely but is noisier.
    pub depth_sharpness: f32,
    /// How far off a surface probes' visibility is tested, in probe spacings. Higher leaks
    /// less shadow into lit areas, too high leaks light through thin walls.
    pub self_shadow_bias: f32,
    /// How far probes' rays reach, in the `ShadowGrid`'s voxels. Further finds more distant
    /// light, at more cost.
    pub ray_distance: f32,
    /// Frames between each probe's updates in the finest cascade, rounded up to a power of
    /// two from 1 to 16, twice as many in each coarser one. The probes take turns, and each
    /// update blends in as much more, so light follows changes as fast, a little noisier, for
    /// a fraction of the cost. Probes near chunks that just changed update every frame.
    pub update_interval: u32,
}

impl Default for ProbeTuning {
    fn default() -> Self {
        Self {
            hysteresis: 0.97,
            depth_hysteresis: 0.97,
            depth_sharpness: 50.0,
            self_shadow_bias: 0.3,
            ray_distance: 512.0,
            update_interval: 4,
        }
    }
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
    pub rays_per_frame: u32,
    pub atlas_columns: u32,
    pub probe_buffer: Buffer,
    pub ray_buffer: Buffer,
    pub active_buffer: Buffer,
    pub irradiance: Image,
    pub irradiance_view: vk::ImageView,
    pub depth: Image,
    pub depth_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub classify: vk::Pipeline,
    pub trace: vk::Pipeline,
    pub update_irradiance: vk::Pipeline,
    pub update_depth: vk::Pipeline,
    /// Shares out the frame's rays once the active probes are counted.
    pub plan: vk::Pipeline,
}

impl ProbeVolume {
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            counts: [0; 3],
            cascades: 0,
            rays_per_frame: 0,
            atlas_columns: 0,
            probe_buffer: Buffer::null(),
            ray_buffer: Buffer::null(),
            active_buffer: Buffer::null(),
            irradiance: Image::null(),
            irradiance_view: vk::ImageView::null(),
            depth: Image::null(),
            depth_view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            classify: vk::Pipeline::null(),
            trace: vk::Pipeline::null(),
            update_irradiance: vk::Pipeline::null(),
            plan: vk::Pipeline::null(),
            update_depth: vk::Pipeline::null(),
        }
    }

    /// Probes for `config`, or a single one for the descriptors when there's none, run with
    /// `layout`, the frame descriptor set's pipeline layout. Needs `cmd_init` before use.
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        config: Option<GiConfig>,
        layout: vk::PipelineLayout,
    ) -> VkResult<Self> {
        let mut volume = Self::null();
        volume.counts = config.map_or([1; 3], |config| config.probe_counts.map(|n| n.max(1)));
        volume.cascades = config.map_or(1, |config| config.cascades.clamp(1, MAX_CASCADES));
        volume.rays_per_frame = config.map_or(0, |config| config.rays_per_frame);
        if let Err(err) = volume.create(adapter, logical_device, layout) {
            unsafe { volume.destroy(logical_device) };
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
        logical_device: &ash::Device,
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
                logical_device,
                info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                0,
            )
        };
        self.probe_buffer =
            storage_buffer(probes * PROBE_SIZE, vk::BufferUsageFlags::TRANSFER_DST)?;
        // The budget, or the fewest rays each when there are more active probes than it covers.
        let rays = (self.rays_per_frame as vk::DeviceSize).max(probes * MIN_RAYS_PER_PROBE);
        self.ray_buffer = storage_buffer(
            rays * size_of::<[f32; 4]>() as vk::DeviceSize,
            vk::BufferUsageFlags::empty(),
        )?;
        self.active_buffer = storage_buffer(
            (ACTIVE_LIST + probes) * size_of::<u32>() as vk::DeviceSize,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        // Nearly square atlases, so they fit the smallest image limits for longer.
        self.atlas_columns = (probes as f64).sqrt().ceil() as u32;
        let rows = (probes as u32).div_ceil(self.atlas_columns);
        let max_dimension = adapter.properties.limits.max_image_dimension2_d;
        if self.atlas_columns.max(rows) * (DEPTH_TEXELS + 2) > max_dimension {
            return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
        }
        for (image, view, texels) in [
            (
                &mut self.irradiance,
                &mut self.irradiance_view,
                IRRADIANCE_TEXELS,
            ),
            (&mut self.depth, &mut self.depth_view, DEPTH_TEXELS),
        ] {
            let tile = texels + 2;
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(ATLAS_FORMAT)
                .extent(vk::Extent3D {
                    width: self.atlas_columns * tile,
                    height: rows * tile,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            *image = Image::new(
                adapter,
                logical_device,
                info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            *view = create_image_view(
                logical_device,
                image.handle,
                ATLAS_FORMAT,
                vk::ComponentMapping::default(),
                vk::ImageAspectFlags::COLOR,
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
        self.sampler = unsafe { logical_device.create_sampler(&sampler_info, None)? };

        let module = create_shader_module(logical_device, GI_SPV)?;
        let entries = [
            c"classifyProbes",
            c"planRays",
            c"traceProbeRays",
            c"updateIrradiance",
            c"updateDepth",
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
        let created = unsafe {
            logical_device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
        };
        unsafe { logical_device.destroy_shader_module(module, None) };
        let pipelines = match created {
            Ok(pipelines) => pipelines,
            Err((pipelines, err)) => {
                for pipeline in pipelines {
                    unsafe { logical_device.destroy_pipeline(pipeline, None) };
                }
                return Err(err);
            }
        };
        [
            self.classify,
            self.plan,
            self.trace,
            self.update_irradiance,
            self.update_depth,
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
    pub unsafe fn cmd_init(&self, logical_device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let to_general = [self.irradiance.handle, self.depth.handle].map(|image| {
            vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(range)
        });
        unsafe {
            logical_device.cmd_pipeline_barrier2(
                cmd_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&to_general),
            );
            for image in [self.irradiance.handle, self.depth.handle] {
                logical_device.cmd_clear_color_image(
                    cmd_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue::default(),
                    slice::from_ref(&range),
                );
            }
            logical_device.cmd_fill_buffer(
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
            logical_device.cmd_pipeline_barrier2(
                cmd_buffer,
                &vk::DependencyInfo::default().memory_barriers(slice::from_ref(&ready)),
            );
        }
    }

    /// Points bindings 6 to 12 of every frame's descriptor set at the volume, matching
    /// `common.slang`. They never move.
    pub fn write_descriptors(&self, logical_device: &ash::Device, sets: &[vk::DescriptorSet]) {
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
        let irradiance_sampled = image(self.irradiance_view, self.sampler);
        let depth_sampled = image(self.depth_view, self.sampler);
        let irradiance_storage = image(self.irradiance_view, vk::Sampler::null());
        let depth_storage = image(self.depth_view, vk::Sampler::null());
        for &set in sets {
            let write = |binding: u32, ty: vk::DescriptorType| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(ty)
            };
            let writes = [
                write(6, vk::DescriptorType::STORAGE_BUFFER).buffer_info(slice::from_ref(&probes)),
                write(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&irradiance_sampled)),
                write(8, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&depth_sampled)),
                write(9, vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(slice::from_ref(&irradiance_storage)),
                write(10, vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(slice::from_ref(&depth_storage)),
                write(11, vk::DescriptorType::STORAGE_BUFFER).buffer_info(slice::from_ref(&rays)),
                write(12, vk::DescriptorType::STORAGE_BUFFER).buffer_info(slice::from_ref(&active)),
            ];
            unsafe { logical_device.update_descriptor_sets(&writes, &[]) };
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
        logical_device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        set: vk::DescriptorSet,
        layout: vk::PipelineLayout,
        timestamps: &FrameTimestamps,
    ) {
        let barrier = |src_stage, src_access, dst_stage, dst_access| {
            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access);
            unsafe {
                logical_device.cmd_pipeline_barrier2(
                    cmd_buffer,
                    &vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier)),
                )
            };
        };
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let read_write = vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
        let dispatch = |pipeline: vk::Pipeline, offset: vk::DeviceSize| unsafe {
            logical_device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            logical_device.cmd_dispatch_indirect(cmd_buffer, self.active_buffer.handle, offset);
        };

        unsafe {
            // Last frame's shading and updates are done with the volume, and its dispatches
            // with the active count.
            barrier(
                compute
                    | vk::PipelineStageFlags2::FRAGMENT_SHADER
                    | vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::SHADER_WRITE,
                compute | vk::PipelineStageFlags2::CLEAR,
                read_write | vk::AccessFlags2::TRANSFER_WRITE,
            );
            // No active probes yet: dispatches one group high and deep, none wide.
            let groups = [0u32, 1, 1, 0, 1, 1, 0, 0];
            logical_device.cmd_update_buffer(
                cmd_buffer,
                self.active_buffer.handle,
                0,
                crate::bytes::slice_as_bytes(&groups),
            );
            barrier(
                vk::PipelineStageFlags2::CLEAR,
                vk::AccessFlags2::TRANSFER_WRITE,
                compute,
                read_write,
            );

            logical_device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                slice::from_ref(&set),
                &[],
            );
            logical_device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.classify,
            );
            logical_device.cmd_dispatch(
                cmd_buffer,
                self.probe_count().div_ceil(CLASSIFY_WORKGROUP),
                1,
                1,
            );
            barrier(compute, vk::AccessFlags2::SHADER_WRITE, compute, read_write);
            logical_device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, self.plan);
            logical_device.cmd_dispatch(cmd_buffer, 1, 1, 1);
            timestamps.mark(logical_device, cmd_buffer, Mark::GiClassify);
            barrier(
                compute,
                vk::AccessFlags2::SHADER_WRITE,
                compute | vk::PipelineStageFlags2::DRAW_INDIRECT,
                read_write | vk::AccessFlags2::INDIRECT_COMMAND_READ,
            );

            // One group per 32 rays, then one per active probe.
            dispatch(self.trace, 0);
            timestamps.mark(logical_device, cmd_buffer, Mark::GiTrace);
            // The rays are written, and tracing is done sampling the atlases.
            barrier(compute, vk::AccessFlags2::SHADER_WRITE, compute, read_write);
            dispatch(self.update_irradiance, UPDATE_GROUPS);
            dispatch(self.update_depth, UPDATE_GROUPS);
            timestamps.mark(logical_device, cmd_buffer, Mark::GiUpdate);
            barrier(
                compute,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER | compute,
                vk::AccessFlags2::SHADER_READ,
            );
        }
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            for pipeline in [
                self.classify,
                self.plan,
                self.trace,
                self.update_irradiance,
                self.update_depth,
            ] {
                logical_device.destroy_pipeline(pipeline, None);
            }
            logical_device.destroy_sampler(self.sampler, None);
            logical_device.destroy_image_view(self.irradiance_view, None);
            logical_device.destroy_image_view(self.depth_view, None);
            self.irradiance.destroy(logical_device);
            self.depth.destroy(logical_device);
            for buffer in [&self.probe_buffer, &self.ray_buffer, &self.active_buffer] {
                buffer.destroy(logical_device);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk one probe cell wide, solid where `solid` says: its probe word, and whether its
    /// mask says it has voxels.
    fn cell(solid: impl Fn(u32, u32, u32) -> bool) -> (u32, bool) {
        let edge = PROBE_CELL;
        let types: Vec<u16> = (0..edge.pow(3))
            .map(|i| solid(i % edge, i / edge % edge, i / (edge * edge)) as u16)
            .collect();
        let words = bake_probe_cells(&VoxelTypes::pack(edge, &types));
        assert_eq!(words.len(), 2);
        (words[0], words[1] & 1 != 0)
    }

    /// The probe's position in its cell, in voxels, when it has one.
    fn position(word: u32) -> Option<[f32; 3]> {
        (word & CELL_PLACED != 0)
            .then(|| [0, 6, 12].map(|shift| ((word >> shift) & 63) as f32 * 0.5))
    }

    #[test]
    fn empty_cells_center_their_probe() {
        let (word, has_voxels) = cell(|_, _, _| false);
        assert_eq!(position(word), Some([8.0; 3]));
        assert!(!has_voxels);
    }

    #[test]
    fn solid_cells_have_no_probe() {
        let (word, has_voxels) = cell(|_, _, _| true);
        assert_eq!(position(word), None);
        assert!(has_voxels);
    }

    #[test]
    fn probes_take_the_largest_empty_node_nearest_the_center() {
        // The bottom half is solid, so an 8³ node of the top half, in the open, not a smaller
        // one hugging the ground.
        let (word, has_voxels) = cell(|_, y, _| y < 8);
        let [x, y, z] = position(word).unwrap();
        assert_eq!(y, 12.0);
        assert!([x, z].iter().all(|&axis| axis == 4.0 || axis == 12.0));
        assert!(has_voxels);

        // Only one voxel is empty, so the probe is at its center.
        assert_eq!(
            position(cell(|x, y, z| (x, y, z) != (3, 9, 14)).0),
            Some([3.5, 9.5, 14.5])
        );
    }

    #[test]
    fn masks_mark_each_cell_with_voxels() {
        // Two cells wide, a voxel only in cell (1, 0, 1).
        let edge = 2 * PROBE_CELL;
        let types: Vec<u16> = (0..edge.pow(3))
            .map(|i| ((i % edge, i / edge % edge, i / (edge * edge)) == (20, 3, 17)) as u16)
            .collect();
        let words = bake_probe_cells(&VoxelTypes::pack(edge, &types));
        assert_eq!(words.len(), 8 + 1);
        // Cell (1, 0, 1) of 2 per edge, x-fastest, is cell 5.
        assert_eq!(words[8], 1 << 5);
    }

    #[test]
    fn chunks_not_a_multiple_of_a_cell_wide_have_no_cells() {
        assert!(bake_probe_cells(&VoxelTypes::pack(8, &[0; 512])).is_empty());
    }
}
