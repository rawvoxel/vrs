//! Sun shadows traced at a fraction of the frame's resolution, from its depth, for shading to
//! upscale. See `shadow.slang` and `Lighting::shadow_resolution`.

use crate::{
    adapter::Adapter,
    image::{Image, create_image_view},
    pipeline::create_shader_module,
};
use ash::{prelude::VkResult, vk};
use glam::Mat4;
use std::slice;

const SHADOW_SPV: &[u8] = include_bytes!("../assets/shader/shadow.spv");
/// Each texel's sun visibility and its surface's distance from the camera, which upscaling
/// compares. A base storage format every device supports.
const SHADOW_FORMAT: vk::Format = vk::Format::R32G32_SFLOAT;
/// Matches `numthreads` of `traceShadows` in `shadow.slang`.
const WORKGROUP_SIZE: u32 = 8;

/// The shadow pass's uniforms. Matches `ShadowUniforms` in `common.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShadowUniforms {
    /// From clip space back to the world, to find each pixel's position from its depth.
    pub inverse_view_proj: Mat4,
    /// Frame pixels per shadow texel along each axis, 1 when shading traces its own, then
    /// the shadow image's width and height, then unused.
    pub params: [u32; 4],
    /// The frame's width and height, then unused.
    pub frame_size: [u32; 4],
}

/// The low resolution shadow image and the pipeline tracing it. The image stays in `GENERAL`
/// layout, written as a storage image and read by shading.
pub(crate) struct ShadowPass {
    pub image: Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    /// Frame pixels per texel the image was made for.
    pub scale: u32,
    /// Reads the depth buffer, and the shadow image when shading.
    pub sampler: vk::Sampler,
    pub pipeline: vk::Pipeline,
}

impl ShadowPass {
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            image: Image::null(),
            view: vk::ImageView::null(),
            extent: vk::Extent2D::default(),
            scale: 0,
            sampler: vk::Sampler::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    /// The pipeline, run with `layout`, the frame descriptor set's pipeline layout, and a 1x1
    /// image for the descriptors until `resize`. Needs `cmd_init` before use.
    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        layout: vk::PipelineLayout,
    ) -> VkResult<Self> {
        let mut pass = Self::null();
        let created = pass.create_pipeline(logical_device, layout).and_then(|()| {
            pass.create_image(
                adapter,
                logical_device,
                vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                1,
            )
        });
        if let Err(err) = created {
            unsafe { pass.destroy(logical_device) };
            return Err(err);
        }
        Ok(pass)
    }

    fn create_pipeline(
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
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .module(module)
            .name(c"traceShadows")
            .stage(vk::ShaderStageFlags::COMPUTE);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let created = unsafe {
            logical_device.create_compute_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&info),
                None,
            )
        };
        unsafe { logical_device.destroy_shader_module(module, None) };
        self.pipeline = created.map_err(|(_, err)| err)?[0];
        Ok(())
    }

    /// Replaces the image with one for a `frame` sized frame at `scale` pixels per texel.
    /// Needs `cmd_init` before use.
    ///
    /// # Safety
    /// Nothing may be using the image.
    pub unsafe fn resize(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        frame: vk::Extent2D,
        scale: u32,
    ) -> VkResult<()> {
        unsafe { self.destroy_image(logical_device) };
        let extent = vk::Extent2D {
            width: frame.width.div_ceil(scale),
            height: frame.height.div_ceil(scale),
        };
        self.create_image(adapter, logical_device, extent, scale)
    }

    fn create_image(
        &mut self,
        adapter: &Adapter,
        logical_device: &ash::Device,
        extent: vk::Extent2D,
        scale: u32,
    ) -> VkResult<()> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(SHADOW_FORMAT)
            .extent(extent.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        self.image = Image::new(
            adapter,
            logical_device,
            info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.view = create_image_view(
            logical_device,
            self.image.handle,
            SHADOW_FORMAT,
            vk::ComponentMapping::default(),
            vk::ImageAspectFlags::COLOR,
        )?;
        self.extent = extent;
        self.scale = scale;
        Ok(())
    }

    /// Records readying a new image.
    ///
    /// # Safety
    /// Nothing may be using the image.
    pub unsafe fn cmd_init(&self, logical_device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        crate::image::transition_layout(
            logical_device,
            self.image.handle,
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

    /// Records tracing the frame's shadows from its depth, which must be in
    /// `SHADER_READ_ONLY_OPTIMAL` layout and ready for compute reads. Leaves them ready for
    /// shading. `set` is the frame's descriptor set, bound with `layout`.
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
        unsafe {
            // Last frame's shading is done reading the image.
            barrier(
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COMPUTE_SHADER,
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
            logical_device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            logical_device.cmd_dispatch(
                cmd_buffer,
                self.extent.width.div_ceil(WORKGROUP_SIZE),
                self.extent.height.div_ceil(WORKGROUP_SIZE),
                1,
            );
            barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
        }
    }

    /// # Safety
    /// Nothing may be using the image.
    unsafe fn destroy_image(&mut self, logical_device: &ash::Device) {
        unsafe {
            logical_device.destroy_image_view(self.view, None);
            self.image.destroy(logical_device);
        }
        self.view = vk::ImageView::null();
        self.image = Image::null();
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        unsafe {
            self.destroy_image(logical_device);
            logical_device.destroy_pipeline(self.pipeline, None);
            logical_device.destroy_sampler(self.sampler, None);
        }
    }
}
