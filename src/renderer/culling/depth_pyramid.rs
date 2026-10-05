use crate::vulkan::{
    adapter::Adapter,
    bytes::as_bytes,
    descriptors::{allocate_descriptor_sets, create_descriptor_pool, create_descriptor_set_layout},
    image::Image,
    pipeline::create_compute_pipeline,
};
use ash::{prelude::VkResult, vk};
use std::slice;

const DEPTH_PYRAMID_SPV: &[u8] = include_bytes!("../../../assets/shader/depth_pyramid.spv");
/// Matches `numthreads` in `depth_pyramid.slang`.
const WORKGROUP_SIZE: u32 = 8;
const PYRAMID_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

/// Matches `ReduceParams` in `depth_pyramid.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReduceParams {
    pub src_size: [u32; 2],
    pub dst_size: [u32; 2],
}

/// Mip levels of a depth buffer where each texel keeps the farthest depth it covers,
/// so occlusion culling can test a box against a few texels instead of every pixel.
///
/// Level 0 is half the depth buffer's size rounded up to powers of two, halving down to 1x1.
/// Every level stays in `GENERAL` layout.
pub struct DepthPyramid {
    pub image: Image,
    /// Every level, for culling.
    pub view: vk::ImageView,
    pub level_views: Vec<vk::ImageView>,
    pub levels: u32,
    pub extent: vk::Extent2D,
    pub depth_extent: vk::Extent2D,

    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    /// One per level, reading the level below it, or the depth buffer for level 0.
    pub level_desc_sets: Vec<vk::DescriptorSet>,
}

impl DepthPyramid {
    /// Built from `depth_view`, which must be sampled in `SHADER_READ_ONLY_OPTIMAL` layout.
    pub fn new(
        adapter: &Adapter,
        device: &ash::Device,
        depth_view: vk::ImageView,
        depth_extent: vk::Extent2D,
    ) -> VkResult<Self> {
        let mut pyramid = Self::null();
        let created = pyramid
            .create_pipeline(device)
            .and_then(|()| pyramid.create_levels(adapter, device, depth_view, depth_extent));
        if let Err(err) = created {
            unsafe { pyramid.destroy(device) };
            return Err(err);
        }
        Ok(pyramid)
    }

    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            image: Image::null(),
            view: vk::ImageView::null(),
            level_views: Vec::new(),
            levels: 0,
            extent: vk::Extent2D::default(),
            depth_extent: vk::Extent2D::default(),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            desc_set_layout: vk::DescriptorSetLayout::null(),
            desc_pool: vk::DescriptorPool::null(),
            level_desc_sets: Vec::new(),
        }
    }

    /// Rebuilds the levels for a new depth buffer. If it fails, the pyramid has no levels.
    ///
    /// # Safety
    /// The device must not be using the pyramid.
    pub unsafe fn resize(
        &mut self,
        adapter: &Adapter,
        device: &ash::Device,
        depth_view: vk::ImageView,
        depth_extent: vk::Extent2D,
    ) -> VkResult<()> {
        unsafe { self.destroy_levels(device) };
        let created = self.create_levels(adapter, device, depth_view, depth_extent);
        if created.is_err() {
            unsafe { self.destroy_levels(device) };
        }
        created
    }

    fn create_pipeline(&mut self, device: &ash::Device) -> VkResult<()> {
        self.desc_set_layout = create_descriptor_set_layout(
            device,
            &[
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::STORAGE_IMAGE,
            ],
            vk::ShaderStageFlags::COMPUTE,
        )?;
        (self.pipeline, self.pipeline_layout) = create_compute_pipeline(
            device,
            DEPTH_PYRAMID_SPV,
            c"reduceDepth",
            self.desc_set_layout,
            size_of::<ReduceParams>() as u32,
        )?;
        Ok(())
    }

    /// Fills in levels after `destroy_levels`, leaving whatever it created behind if it fails.
    fn create_levels(
        &mut self,
        adapter: &Adapter,
        device: &ash::Device,
        depth_view: vk::ImageView,
        depth_extent: vk::Extent2D,
    ) -> VkResult<()> {
        // Powers of two, so halving never rounds and every level fits Vulkan's mip chain.
        // Texels past the depth buffer's edge repeat it.
        self.depth_extent = depth_extent;
        self.extent = vk::Extent2D {
            width: depth_extent.width.div_ceil(2).next_power_of_two(),
            height: depth_extent.height.div_ceil(2).next_power_of_two(),
        };
        let mut levels = 1;
        let mut level_extent = self.extent;
        while level_extent.width > 1 || level_extent.height > 1 {
            level_extent = half_extent(level_extent);
            levels += 1;
        }
        self.levels = levels;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(PYRAMID_FORMAT)
            .extent(self.extent.into())
            .mip_levels(levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        self.image = Image::new(
            adapter,
            device,
            image_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.view = create_view(device, self.image.handle, 0, levels)?;
        for level in 0..levels {
            let view = create_view(device, self.image.handle, level, 1)?;
            self.level_views.push(view);
        }

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(levels),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(levels),
        ];
        self.desc_pool = create_descriptor_pool(device, &pool_sizes, levels)?;
        self.level_desc_sets =
            allocate_descriptor_sets(device, self.desc_pool, self.desc_set_layout, levels)?;

        for (level, &desc_set) in self.level_desc_sets.iter().enumerate() {
            let src_info = match level {
                0 => vk::DescriptorImageInfo::default()
                    .image_view(depth_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                _ => vk::DescriptorImageInfo::default()
                    .image_view(self.level_views[level - 1])
                    .image_layout(vk::ImageLayout::GENERAL),
            };
            let dst_info = vk::DescriptorImageInfo::default()
                .image_view(self.level_views[level])
                .image_layout(vk::ImageLayout::GENERAL);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(slice::from_ref(&src_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(slice::from_ref(&dst_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        Ok(())
    }

    /// Records moving every level to `GENERAL`, once after creating or resizing.
    /// Culling's descriptor set refers to the pyramid in every phase, even before it's built.
    ///
    /// # Safety
    /// The device must not be using the pyramid.
    pub unsafe fn cmd_init_layout(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        let image_barrier = vk::ImageMemoryBarrier2::default()
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(self.image.handle)
            .subresource_range(color_range(0, self.levels));
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_barrier));
        unsafe { device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info) };
    }

    /// Records building every level from the depth buffer, after which the pyramid is
    /// ready for compute reads.
    ///
    /// # Safety
    /// The depth buffer must be in `SHADER_READ_ONLY_OPTIMAL` and visible to compute reads.
    pub unsafe fn cmd_build(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        unsafe {
            // Every level is rebuilt, but last frame's culling may still be reading them.
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.image.handle)
                .subresource_range(color_range(0, self.levels));
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(slice::from_ref(&image_barrier));
            device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);

            device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);

            let mut src_extent = self.depth_extent;
            let mut dst_extent = self.extent;
            for (level, &desc_set) in self.level_desc_sets.iter().enumerate() {
                let params = ReduceParams {
                    src_size: [src_extent.width, src_extent.height],
                    dst_size: [dst_extent.width, dst_extent.height],
                };
                device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    slice::from_ref(&desc_set),
                    &[],
                );
                device.cmd_push_constants(
                    cmd_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    as_bytes(&params),
                );
                device.cmd_dispatch(
                    cmd_buffer,
                    dst_extent.width.div_ceil(WORKGROUP_SIZE),
                    dst_extent.height.div_ceil(WORKGROUP_SIZE),
                    1,
                );

                // The next level reads this one, and culling reads them all.
                let level_barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.image.handle)
                    .subresource_range(color_range(level as u32, 1));
                let dependency_info = vk::DependencyInfo::default()
                    .image_memory_barriers(slice::from_ref(&level_barrier));
                device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);

                src_extent = dst_extent;
                dst_extent = half_extent(dst_extent);
            }
        }
    }

    /// Leaves the levels null, so destroying them again does nothing.
    unsafe fn destroy_levels(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.desc_pool, None);
            for view in self.level_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            device.destroy_image_view(self.view, None);
            self.image.destroy(device);
        }
        self.desc_pool = vk::DescriptorPool::null();
        self.level_desc_sets.clear();
        self.view = vk::ImageView::null();
        self.image = Image::null();
        self.levels = 0;
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.destroy_levels(device);
            device.destroy_descriptor_set_layout(self.desc_set_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

#[inline]
fn half_extent(extent: vk::Extent2D) -> vk::Extent2D {
    vk::Extent2D {
        width: extent.width.div_ceil(2).max(1),
        height: extent.height.div_ceil(2).max(1),
    }
}

#[inline]
fn color_range(base_level: u32, level_count: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(base_level)
        .level_count(level_count)
        .layer_count(1)
}

#[inline]
fn create_view(
    device: &ash::Device,
    image: vk::Image,
    base_level: u32,
    level_count: u32,
) -> VkResult<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .format(PYRAMID_FORMAT)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(color_range(base_level, level_count));

    unsafe { device.create_image_view(&info, None) }
}
