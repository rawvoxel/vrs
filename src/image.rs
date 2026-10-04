use crate::{adapter::Adapter, alloc};
use ash::{prelude::VkResult, vk};
use std::slice;

#[derive(Clone, Copy)]
pub struct Image {
    pub memory: vk::DeviceMemory,
    pub handle: vk::Image,
}

impl Image {
    /// Owns nothing, destroying it does nothing.
    pub fn null() -> Self {
        Self {
            memory: vk::DeviceMemory::null(),
            handle: vk::Image::null(),
        }
    }

    pub fn new(
        adapter: &Adapter,
        logical_device: &ash::Device,
        image_info: vk::ImageCreateInfo,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> VkResult<Self> {
        unsafe {
            let mut this = Self::null();
            this.handle = logical_device.create_image(&image_info, None)?;
            let memory_requirements = logical_device.get_image_memory_requirements(this.handle);
            let bound = alloc::vk_alloc(adapter, logical_device, memory_requirements, memory_property_flags)
                .and_then(|memory| {
                    this.memory = memory;
                    logical_device.bind_image_memory(this.handle, memory, 0)
                });
            if let Err(err) = bound {
                this.destroy(logical_device);
                return Err(err);
            }
            Ok(this)
        }
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.free_memory(self.memory, None);
            logical_device.destroy_image(self.handle, None);
        }
    }
}

#[inline]
pub fn create_image_view(
    logical_device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    components: vk::ComponentMapping,
    aspect_mask: vk::ImageAspectFlags,
) -> VkResult<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .format(format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .components(components)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .level_count(1)
                .layer_count(1),
        );

    unsafe { logical_device.create_image_view(&info, None) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn transition_layout(
    render_device: &ash::Device,
    image: vk::Image,
    cmd_buffer: vk::CommandBuffer,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access_mask: vk::AccessFlags2,
    dst_access_mask: vk::AccessFlags2,
    src_stage_mask: vk::PipelineStageFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    aspect_mask: vk::ImageAspectFlags,
) {
    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .layer_count(1)
                .level_count(1),
        );

    let dependency_info =
        vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_memory_barrier));

    unsafe {
        render_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
    }
}
