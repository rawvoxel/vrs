//! The shadow pass: from the frame's depth, the sun's and the lights' shadows, glossy voxels'
//! reflections, and below full resolution the light from everywhere but the sun, at a
//! fraction of the frame's resolution, for shading to upscale. See `shadow.slang`,
//! `Lighting::shadow_resolution`, `Renderer::lights` and `Material::roughness`.

use crate::{
    timings::{FrameTimestamps, GpuStage},
    vulkan::{
        adapter::Adapter, buffer::Buffer, image::StorageImage, pipeline::create_shader_module,
        sync::cmd_memory_barrier,
    },
};
use ash::{prelude::VkResult, vk};
use glam::Mat4;
use std::slice;

const SHADOW_SPV: &[u8] = include_bytes!("../../assets/shader/shadow.spv");
/// Each texel's sun visibility, tinted by see-through voxels, and its surface's distance from
/// the camera, which upscaling compares. A base storage format every device supports.
const SHADOW_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
/// Each texel's shadowed lights and how much of each gets through, 8 bits each.
const LIGHT_SHADOW_FORMAT: vk::Format = vk::Format::R32G32_UINT;
/// Each texel's reflection, and which face it's of.
const REFLECTION_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Each texel's light from everywhere but the sun, and which face it's of.
const INDIRECT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Matches `numthreads` of `traceShadows` in `shadow.slang`.
const WORKGROUP_SIZE: u32 = 8;
/// Pixels along each edge of a tile lights are culled for. Matches `LIGHT_TILE` in
/// `common/lights.slang`.
pub const LIGHT_TILE: u32 = 16;
/// The most lights a tile keeps. Matches `TILE_LIGHTS` in `common/lights.slang`.
const TILE_LIGHTS: u32 = 63;
/// Words a tile's light list takes: its count, then its lights.
const TILE_WORDS: u32 = TILE_LIGHTS + 1;

/// The shadow pass's uniforms. Matches `ShadowUniforms` in `common/layout.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShadowUniforms {
    /// From clip space back to the world, to find each pixel's position from its depth.
    pub inverse_view_proj: Mat4,
    /// Frame pixels per shadow texel along each axis, then the shadow image's width and
    /// height, then 1 when any material is glossy, for reflections to be traced.
    pub params: [u32; 4],
    /// The frame's width and height, then 1 when the shadow pass runs this frame, then 1 when
    /// it also lights faces with the light from everywhere but the sun.
    pub frame_size: [u32; 4],
    /// How many lights there are, the light tiles across and down the frame, and how many of
    /// each pixel's lights are shadowed.
    pub lights: [u32; 4],
}

/// The shadow pass's images, the lights reaching each tile, and the pipelines making them.
/// The images stay in `GENERAL` layout, written as storage images and read by shading.
pub(crate) struct ShadowPass {
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

impl ShadowPass {
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
        logical_device: &ash::Device,
        layout: vk::PipelineLayout,
    ) -> VkResult<Self> {
        let mut pass = Self::null();
        let one = vk::Extent2D {
            width: 1,
            height: 1,
        };
        let created = pass
            .create_pipelines(logical_device, layout)
            .and_then(|()| pass.create_images(adapter, logical_device, one, 1));
        if let Err(err) = created {
            unsafe { pass.destroy(logical_device) };
            return Err(err);
        }
        Ok(pass)
    }

    fn create_pipelines(
        &mut self,
        logical_device: &ash::Device,
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
        self.sampler = unsafe { logical_device.create_sampler(&sampler_info, None)? };

        let module = create_shader_module(logical_device, SHADOW_SPV)?;
        let infos = [c"cullLights", c"traceShadows"].map(|entry| {
            vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .module(module)
                        .name(entry)
                        .stage(vk::ShaderStageFlags::COMPUTE),
                )
                .layout(layout)
        });
        let created = unsafe {
            logical_device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
        };
        unsafe { logical_device.destroy_shader_module(module, None) };
        match created {
            Ok(pipelines) => {
                [self.cull_pipeline, self.pipeline] = [pipelines[0], pipelines[1]];
                Ok(())
            }
            Err((pipelines, err)) => {
                for pipeline in pipelines {
                    unsafe { logical_device.destroy_pipeline(pipeline, None) };
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
        logical_device: &ash::Device,
        frame: vk::Extent2D,
        scale: u32,
    ) -> VkResult<()> {
        unsafe { self.destroy_images(logical_device) };
        self.create_images(adapter, logical_device, frame, scale)
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
        logical_device: &ash::Device,
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
                logical_device,
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
            logical_device,
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
    pub unsafe fn cmd_init(&self, logical_device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        for image in [
            &self.shadows,
            &self.light_shadows,
            &self.reflections,
            &self.indirect,
        ] {
            crate::vulkan::image::transition_layout(
                logical_device,
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
        logical_device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        set: vk::DescriptorSet,
        layout: vk::PipelineLayout,
        timestamps: &FrameTimestamps,
    ) {
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let fragment = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let barrier =
            |src, dst| unsafe { cmd_memory_barrier(logical_device, cmd_buffer, src, dst) };
        let dispatch = |pipeline: vk::Pipeline, groups: [u32; 2]| unsafe {
            logical_device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            logical_device.cmd_dispatch(cmd_buffer, groups[0], groups[1], 1);
        };
        let write = vk::AccessFlags2::SHADER_STORAGE_WRITE;
        unsafe {
            // Last frame's shading is done reading the images and tiles.
            barrier((fragment, vk::AccessFlags2::empty()), (compute, write));
            logical_device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                slice::from_ref(&set),
                &[],
            );
            dispatch(self.cull_pipeline, self.tile_counts);
            timestamps.cmd_mark(logical_device, cmd_buffer, GpuStage::LightCulling);
            barrier(
                (compute, write),
                (compute, vk::AccessFlags2::SHADER_STORAGE_READ),
            );
            let groups = [self.extent.width, self.extent.height]
                .map(|texels| texels.div_ceil(WORKGROUP_SIZE));
            dispatch(self.pipeline, groups);
            timestamps.cmd_mark(logical_device, cmd_buffer, GpuStage::ShadowPass);
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
    unsafe fn destroy_images(&mut self, logical_device: &ash::Device) {
        unsafe {
            for image in self.images_mut() {
                image.destroy(logical_device);
            }
            self.tiles.destroy(logical_device);
        }
        self.tiles = Buffer::null();
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        unsafe {
            self.destroy_images(logical_device);
            logical_device.destroy_pipeline(self.cull_pipeline, None);
            logical_device.destroy_pipeline(self.pipeline, None);
            logical_device.destroy_sampler(self.sampler, None);
        }
    }
}
