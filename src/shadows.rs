//! Shadows traced from the frame's depth, the sun's and the lights', and glossy voxels'
//! reflections, at a fraction of the frame's resolution, for shading to upscale. See
//! `shadow.slang`, `Lighting::shadow_resolution`, `Renderer::lights` and `Material::roughness`.

use crate::{
    adapter::Adapter,
    buffer::Buffer,
    image::{Image, create_image_view},
    pipeline::create_shader_module,
};
use ash::{prelude::VkResult, vk};
use glam::Mat4;
use std::slice;

const SHADOW_SPV: &[u8] = include_bytes!("../assets/shader/shadow.spv");
/// Each texel's sun visibility, tinted by see-through voxels, and its surface's distance from
/// the camera, which upscaling compares. A base storage format every device supports.
const SHADOW_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
/// Each texel's shadowed lights and how much of each gets through, 8 bits each.
const LIGHT_SHADOW_FORMAT: vk::Format = vk::Format::R32G32_UINT;
/// Each texel's reflection, and which face it's of.
const REFLECTION_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Matches `numthreads` of `traceShadows` in `shadow.slang`.
const WORKGROUP_SIZE: u32 = 8;
/// Pixels along each edge of a tile lights are culled for. Matches `LIGHT_TILE` in
/// `common.slang`.
pub const LIGHT_TILE: u32 = 16;
/// Words a tile's light list takes: its count, then `TILE_LIGHTS` lights.
const TILE_WORDS: u32 = 64;

/// The shadow pass's uniforms. Matches `ShadowUniforms` in `common.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShadowUniforms {
    /// From clip space back to the world, to find each pixel's position from its depth.
    pub inverse_view_proj: Mat4,
    /// Frame pixels per shadow texel along each axis, then the shadow image's width and
    /// height, then 1 when any material is glossy, for reflections to be traced.
    pub params: [u32; 4],
    /// The frame's width and height, then 1 when the shadow pass runs this frame, then unused.
    pub frame_size: [u32; 4],
    /// How many lights there are, the light tiles across and down the frame, and how many of
    /// each pixel's lights are shadowed.
    pub lights: [u32; 4],
}

/// The low resolution shadow images, the lights reaching each tile, and the pipelines making
/// them. The images stay in `GENERAL` layout, written as storage images and read by shading.
pub(crate) struct ShadowPass {
    pub image: Image,
    pub view: vk::ImageView,
    pub light_image: Image,
    pub light_view: vk::ImageView,
    pub reflection_image: Image,
    pub reflection_view: vk::ImageView,
    /// Each tile's lights, `TILE_WORDS` words a tile.
    pub tiles: Buffer,
    pub extent: vk::Extent2D,
    /// Light tiles across and down the frame.
    pub tile_counts: [u32; 2],
    /// Frame pixels per texel the images were made for.
    pub scale: u32,
    /// Reads the depth buffer, and the shadow image when shading.
    pub sampler: vk::Sampler,
    pub cull_pipeline: vk::Pipeline,
    pub pipeline: vk::Pipeline,
}

impl ShadowPass {
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            image: Image::null(),
            view: vk::ImageView::null(),
            light_image: Image::null(),
            light_view: vk::ImageView::null(),
            reflection_image: Image::null(),
            reflection_view: vk::ImageView::null(),
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
        for (image, view, format) in [
            (&mut self.image, &mut self.view, SHADOW_FORMAT),
            (
                &mut self.light_image,
                &mut self.light_view,
                LIGHT_SHADOW_FORMAT,
            ),
            (
                &mut self.reflection_image,
                &mut self.reflection_view,
                REFLECTION_FORMAT,
            ),
        ] {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
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
                format,
                vk::ComponentMapping::default(),
                vk::ImageAspectFlags::COLOR,
            )?;
        }

        self.tile_counts = [
            frame.width.div_ceil(LIGHT_TILE),
            frame.height.div_ceil(LIGHT_TILE),
        ];
        let tiles_info = vk::BufferCreateInfo::default()
            .size(
                (self.tile_counts[0] * self.tile_counts[1] * TILE_WORDS) as vk::DeviceSize
                    * size_of::<u32>() as vk::DeviceSize,
            )
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
            self.image.handle,
            self.light_image.handle,
            self.reflection_image.handle,
        ] {
            crate::image::transition_layout(
                logical_device,
                image,
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

    /// Records listing the lights reaching each tile, then tracing the frame's shadows, from
    /// its depth, which must be in `SHADER_READ_ONLY_OPTIMAL` layout and ready for compute
    /// reads. Leaves them ready for shading. `set` is the frame's descriptor set, bound with
    /// `layout`.
    ///
    /// # Safety
    /// The frame's uniforms and descriptors must be written.
    pub unsafe fn cmd_trace(
        &self,
        logical_device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        set: vk::DescriptorSet,
        layout: vk::PipelineLayout,
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
        let dispatch = |pipeline: vk::Pipeline, groups: [u32; 2]| unsafe {
            logical_device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            logical_device.cmd_dispatch(cmd_buffer, groups[0], groups[1], 1);
        };
        unsafe {
            // Last frame's shading is done reading the images and tiles.
            barrier(
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::empty(),
                compute,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
            logical_device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                slice::from_ref(&set),
                &[],
            );
            dispatch(self.cull_pipeline, self.tile_counts);
            barrier(
                compute,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                compute,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            );
            dispatch(
                self.pipeline,
                [
                    self.extent.width.div_ceil(WORKGROUP_SIZE),
                    self.extent.height.div_ceil(WORKGROUP_SIZE),
                ],
            );
            barrier(
                compute,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
            );
        }
    }

    /// # Safety
    /// Nothing may be using the images or tiles.
    unsafe fn destroy_images(&mut self, logical_device: &ash::Device) {
        unsafe {
            for view in [self.view, self.light_view, self.reflection_view] {
                logical_device.destroy_image_view(view, None);
            }
            self.image.destroy(logical_device);
            self.light_image.destroy(logical_device);
            self.reflection_image.destroy(logical_device);
            self.tiles.destroy(logical_device);
        }
        self.view = vk::ImageView::null();
        self.light_view = vk::ImageView::null();
        self.reflection_view = vk::ImageView::null();
        self.image = Image::null();
        self.light_image = Image::null();
        self.reflection_image = Image::null();
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
