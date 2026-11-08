use crate::vulkan::{adapter::Adapter, alloc};
use ash::{prelude::VkResult, vk};
use std::slice;

pub struct Image {
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) handle: vk::Image,
}

impl Image {
    #[inline]
    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    #[inline]
    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn new(
        adapter: &Adapter,
        device: &ash::Device,
        image_info: vk::ImageCreateInfo,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> VkResult<Self> {
        unsafe {
            let handle = device.create_image(&image_info, None)?;
            let memory_requirements = device.get_image_memory_requirements(handle);
            let memory = match alloc::vk_alloc(
                adapter,
                device,
                memory_requirements,
                memory_property_flags,
            ) {
                Ok(memory) => memory,
                Err(error) => {
                    device.destroy_image(handle, None);
                    return Err(error);
                }
            };
            if let Err(error) = device.bind_image_memory(handle, memory, 0) {
                device.destroy_image(handle, None);
                device.free_memory(memory, None);
                return Err(error);
            }
            Ok(Self { memory, handle })
        }
    }

    /// # Safety
    /// Before freeing an object, an application must ensure the object is no longer in use by the device.
    #[inline]
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_image(self.handle, None);
        }
    }
}

#[inline]
pub fn create_image_view(
    device: &ash::Device,
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

    unsafe { device.create_image_view(&info, None) }
}

/// Records a barrier moving `image` from `old_layout` to `new_layout`, after `src`'s accesses in
/// its stages and before `dst`'s: each a `(stage, access)` pair, as [`cmd_memory_barrier`] takes.
///
/// [`cmd_memory_barrier`]: crate::vulkan::sync::cmd_memory_barrier
///
/// # Safety
/// `cmd_buffer` is recording, outside any rendering, and `image` is in `old_layout` (or it's
/// `UNDEFINED`, its contents then lost).
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn transition_layout(
    device: &ash::Device,
    image: vk::Image,
    cmd_buffer: vk::CommandBuffer,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
    aspect_mask: vk::ImageAspectFlags,
) {
    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)
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
        device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
    }
}
