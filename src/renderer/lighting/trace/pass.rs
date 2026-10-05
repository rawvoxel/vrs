//! The trace pass: from the frame's depth, the sun's and the lights' shadows, glossy voxels'
//! reflections, and below full resolution the light from everywhere but the sun, at a
//! fraction of the frame's resolution, for shading to upscale. See `trace_pass.slang`,
//! `Lighting::trace_resolution`, `Renderer::lights` and `Material::roughness`.
//!
//! Frames draw their depth first for it whenever there's a voxel grid or a light, see
//! `Renderer::uses_trace_pass`.

use crate::{
    renderer::{
        DebugView, Renderer,
        lighting::trace::MAX_SHADOWED_LIGHTS,
        timings::{FrameTimestamps, GpuStage},
    },
    vulkan::{
        adapter::Adapter, buffer::Buffer, image::StorageImage, pipeline::create_shader_module,
        sync::cmd_memory_barrier,
    },
};
use ash::{prelude::VkResult, vk};
use std::slice;

const TRACE_PASS_SPV: &[u8] = include_bytes!("../../../../assets/shader/trace_pass.spv");
/// Each texel's sun visibility, tinted by see-through voxels, and its surface's distance from
/// the camera, which upscaling compares. A base storage format every device supports.
const SHADOW_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
/// Each texel's shadowed lights and how much of each gets through, 8 bits each.
const LIGHT_SHADOW_FORMAT: vk::Format = vk::Format::R32G32_UINT;
/// Each texel's reflection, and which face it's of.
const REFLECTION_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Each texel's light from everywhere but the sun, and which face it's of.
const INDIRECT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Matches `numthreads` of `traceSurfaces` in `trace_pass.slang`.
const WORKGROUP_SIZE: u32 = 8;
/// Pixels along each edge of a tile lights are culled for. Matches `LIGHT_TILE` in
/// `common/lights.slang`.
const LIGHT_TILE: u32 = 16;
/// The most lights a tile keeps. Matches `MAX_TILE_LIGHTS` in `common/lights.slang`.
const MAX_TILE_LIGHTS: u32 = 63;
/// Words a tile's light list takes: its count, then its lights.
const TILE_WORDS: u32 = MAX_TILE_LIGHTS + 1;

/// The trace pass's uniforms. Matches `TraceUniforms` in `common/layout.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TraceUniforms {
    /// Frame pixels per trace texel along each axis.
    pub pixels_per_texel: u32,
    /// 1 when any material is glossy, for reflections to be traced.
    pub trace_reflections: u32,
    /// The trace images' width and height.
    pub image_size: [u32; 2],
    pub light_count: u32,
    /// Light tiles across and down the frame.
    pub tiles_across: u32,
    pub tiles_down: u32,
    /// How many of each pixel's lights are shadowed.
    pub shadowed_lights: u32,
}

/// The trace pass's images, the lights reaching each tile, and the pipelines making them.
/// The images stay in `GENERAL` layout, written as storage images and read by shading.
pub(crate) struct TracePass {
    pub shadows: StorageImage,
    pub light_shadows: StorageImage,
    pub reflections: StorageImage,
    /// Each texel's light from everywhere but the sun, with its ambient occlusion.
    pub indirect: StorageImage,
    /// Each tile's lights, `TILE_WORDS` words a tile.
    pub tiles: Buffer,
    pub extent: vk::Extent2D,
    /// Light tiles across and down the frame.
    pub tile_counts: [u32; 2],
    /// Frame pixels per texel the images were made for.
    pub scale: u32,
    /// Reads the depth buffer, and the shadows when shading.
    pub sampler: vk::Sampler,
    pub cull_pipeline: vk::Pipeline,
    pub pipeline: vk::Pipeline,
}

impl TracePass {
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            shadows: StorageImage::null(),
            light_shadows: StorageImage::null(),
            reflections: StorageImage::null(),
            indirect: StorageImage::null(),
            tiles: Buffer::null(),
            extent: vk::Extent2D::default(),
            tile_counts: [0; 2],
            scale: 0,
            sampler: vk::Sampler::null(),
            cull_pipeline: vk::Pipeline::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    /// The pipelines, run with `layout`, the frame descriptor set's pipeline layout, and
    /// images for a 1x1 frame for the descriptors until `resize`. Needs `cmd_init` before use.
    pub fn new(
        adapter: &Adapter,
        device: &ash::Device,
        layout: vk::PipelineLayout,
    ) -> VkResult<Self> {
        let mut pass = Self::null();
        let one = vk::Extent2D {
            width: 1,
            height: 1,
        };
        let created = pass
            .create_pipelines(device, layout)
            .and_then(|()| pass.create_images(adapter, device, one, 1));
        if let Err(err) = created {
            unsafe { pass.destroy(device) };
            return Err(err);
        }
        Ok(pass)
    }

    fn create_pipelines(
        &mut self,
        device: &ash::Device,
        layout: vk::PipelineLayout,
    ) -> VkResult<()> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        self.sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        let module = create_shader_module(device, TRACE_PASS_SPV)?;
        let infos = [c"cullLights", c"traceSurfaces"].map(|entry| {
            vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .module(module)
                        .name(entry)
                        .stage(vk::ShaderStageFlags::COMPUTE),
                )
                .layout(layout)
        });
        let created =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None) };
        unsafe { device.destroy_shader_module(module, None) };
        match created {
            Ok(pipelines) => {
                [self.cull_pipeline, self.pipeline] = [pipelines[0], pipelines[1]];
                Ok(())
            }
            Err((pipelines, err)) => {
                for pipeline in pipelines {
                    unsafe { device.destroy_pipeline(pipeline, None) };
                }
                Err(err)
            }
        }
    }

    /// Replaces the images and tiles with ones for a `frame` sized frame at `scale` pixels per
    /// texel. Needs `cmd_init` before use.
    ///
    /// # Safety
    /// Nothing may be using them.
    pub unsafe fn resize(
        &mut self,
        adapter: &Adapter,
        device: &ash::Device,
        frame: vk::Extent2D,
        scale: u32,
    ) -> VkResult<()> {
        unsafe { self.destroy_images(device) };
        self.create_images(adapter, device, frame, scale)
    }

    fn images_mut(&mut self) -> [&mut StorageImage; 4] {
        [
            &mut self.shadows,
            &mut self.light_shadows,
            &mut self.reflections,
            &mut self.indirect,
        ]
    }

    fn create_images(
        &mut self,
        adapter: &Adapter,
        device: &ash::Device,
        frame: vk::Extent2D,
        scale: u32,
    ) -> VkResult<()> {
        let extent = vk::Extent2D {
            width: frame.width.div_ceil(scale),
            height: frame.height.div_ceil(scale),
        };
        let formats = [
            SHADOW_FORMAT,
            LIGHT_SHADOW_FORMAT,
            REFLECTION_FORMAT,
            INDIRECT_FORMAT,
        ];
        for (image, format) in self.images_mut().into_iter().zip(formats) {
            *image = StorageImage::new(
                adapter,
                device,
                extent,
                format,
                vk::ImageUsageFlags::empty(),
            )?;
        }

        self.tile_counts = [
            frame.width.div_ceil(LIGHT_TILE),
            frame.height.div_ceil(LIGHT_TILE),
        ];
        let tile_words = self.tile_counts[0] * self.tile_counts[1] * TILE_WORDS;
        let tiles_info = vk::BufferCreateInfo::default()
            .size(tile_words as vk::DeviceSize * size_of::<u32>() as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        self.tiles = Buffer::new(
            adapter,
            device,
            tiles_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            0,
        )?;
        self.extent = extent;
        self.scale = scale;
        Ok(())
    }

    /// Records readying new images.
    ///
    /// # Safety
    /// Nothing may be using the images.
    pub unsafe fn cmd_init(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        for image in [
            &self.shadows,
            &self.light_shadows,
            &self.reflections,
            &self.indirect,
        ] {
            crate::vulkan::image::transition_layout(
                device,
                image.image.handle,
                cmd_buffer,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::ImageAspectFlags::COLOR,
            );
        }
    }

    /// Records listing the lights reaching each tile, then tracing the frame's shadows and
    /// reflections and finding the light from everywhere but the sun, from its depth, which
    /// must be in `SHADER_READ_ONLY_OPTIMAL` layout and ready for compute reads. Leaves them
    /// ready for shading. `set` is the frame's descriptor set, bound with `layout`.
    ///
    /// # Safety
    /// The frame's uniforms and descriptors must be written.
    pub unsafe fn cmd_trace(
        &self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        set: vk::DescriptorSet,
        layout: vk::PipelineLayout,
        timestamps: &FrameTimestamps,
    ) {
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let fragment = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let barrier = |src, dst| unsafe { cmd_memory_barrier(device, cmd_buffer, src, dst) };
        let dispatch = |pipeline: vk::Pipeline, groups: [u32; 2]| unsafe {
            device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_dispatch(cmd_buffer, groups[0], groups[1], 1);
        };
        let write = vk::AccessFlags2::SHADER_STORAGE_WRITE;
        unsafe {
            // Last frame's shading is done reading the images and tiles.
            barrier((fragment, vk::AccessFlags2::empty()), (compute, write));
            device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                slice::from_ref(&set),
                &[],
            );
            dispatch(self.cull_pipeline, self.tile_counts);
            timestamps.cmd_mark(device, cmd_buffer, GpuStage::LightCulling);
            barrier(
                (compute, write),
                (compute, vk::AccessFlags2::SHADER_STORAGE_READ),
            );
            let groups = [self.extent.width, self.extent.height]
                .map(|texels| texels.div_ceil(WORKGROUP_SIZE));
            dispatch(self.pipeline, groups);
            timestamps.cmd_mark(device, cmd_buffer, GpuStage::TracePass);
            barrier(
                (compute, write),
                (
                    fragment,
                    vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
            );
        }
    }

    /// # Safety
    /// Nothing may be using the images or tiles.
    unsafe fn destroy_images(&mut self, device: &ash::Device) {
        unsafe {
            for image in self.images_mut() {
                image.destroy(device);
            }
            self.tiles.destroy(device);
        }
        self.tiles = Buffer::null();
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.destroy_images(device);
            device.destroy_pipeline(self.cull_pipeline, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_sampler(self.sampler, None);
        }
    }
}

impl Renderer {
    /// Frame pixels along each axis that share a shadow ray, 1 without shadows.
    pub(in crate::renderer) fn trace_scale(&self) -> u32 {
        match self.lighting.voxel_grid {
            Some(_) => self.lighting.trace_resolution.pixels_per_ray(),
            None => 1,
        }
    }

    /// Whether frames draw their depth first, for the trace pass to trace shadows and cull
    /// lights from, then their color over it. Even at full resolution, that's faster than
    /// tracing shadows as pixels are shaded, which wastes rays on pixels drawn over.
    pub(in crate::renderer) fn uses_trace_pass(&self) -> bool {
        self.lighting.voxel_grid.is_some() || !self.lights.is_empty()
    }

    /// Makes the trace images fit the frame and `trace_scale`, waiting for the device to go
    /// idle when they have to be replaced.
    pub(in crate::renderer) fn prepare_trace_pass(&mut self) -> VkResult<()> {
        let scale = self.trace_scale();
        let frame = self.swapchain.extent;
        let fits = self.trace_pass.scale == scale
            && self.trace_pass.extent.width == frame.width.div_ceil(scale)
            && self.trace_pass.extent.height == frame.height.div_ceil(scale);
        if !self.uses_trace_pass() || fits {
            return Ok(());
        }
        unsafe {
            self.device.device_wait_idle()?;
            self.trace_pass
                .resize(&self.adapter, &self.device, frame, scale)?;
        }
        self.init_trace_pass()
    }

    /// The trace pass's uniforms for a frame seen through `view_proj`.
    pub(in crate::renderer) fn trace_uniforms(&self) -> TraceUniforms {
        let extent = self.trace_pass.extent;
        let shadowed = match self.lighting.voxel_grid {
            Some(_) => self.lighting.shadowed_lights.min(MAX_SHADOWED_LIGHTS),
            None => 0,
        };
        let [tiles_across, tiles_down] = self.trace_pass.tile_counts;
        TraceUniforms {
            pixels_per_texel: self.trace_scale(),
            trace_reflections: self.glossy_palette as u32,
            image_size: [extent.width, extent.height],
            light_count: self.light_count(),
            tiles_across,
            tiles_down,
            shadowed_lights: shadowed,
        }
    }

    /// Whether the trace pass lights faces with the light from everywhere but the sun,
    /// probes and ambient occlusion, for shading to upscale, which below full resolution is
    /// cheaper than every pixel finding it. Not for `DebugView::AmbientOcclusion`, which shows
    /// each pixel's.
    pub(in crate::renderer) fn trace_pass_finds_indirect(&self) -> bool {
        let indirect_work = self.probes_enabled() || self.lighting.ambient_occlusion > 0.0;
        self.uses_trace_pass()
            && self.trace_scale() > 1
            && indirect_work
            && self.lighting.debug_view != DebugView::AmbientOcclusion
    }
}
