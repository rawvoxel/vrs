use ash::{prelude::VkResult, vk};

/// Records a barrier making `src_access` in `src_stage` available to `dst_access` in
/// `dst_stage`, for every resource.
///
/// # Safety
/// `cmd_buffer` must be recording.
pub(crate) unsafe fn cmd_memory_barrier(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1);
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd_buffer,
            &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier)),
        )
    };
}

/// Starts signaled, so the first wait on it returns straight away.
#[inline]
pub fn create_fence(device: &ash::Device) -> VkResult<vk::Fence> {
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    unsafe { device.create_fence(&fence_info, None) }
}

/// See `create_fence`. If one fails, those already created are destroyed.
#[inline]
pub fn create_fences(device: &ash::Device, count: u32) -> VkResult<Vec<vk::Fence>> {
    let mut fences = Vec::with_capacity(count as usize);
    for _ in 0..count {
        match create_fence(device) {
            Ok(fence) => fences.push(fence),
            Err(err) => {
                for fence in fences {
                    unsafe { device.destroy_fence(fence, None) };
                }
                return Err(err);
            }
        }
    }
    Ok(fences)
}

#[inline]
pub fn create_semaphore(device: &ash::Device) -> VkResult<vk::Semaphore> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    unsafe { device.create_semaphore(&semaphore_info, None) }
}

/// If one fails, those already created are destroyed.
#[inline]
pub fn create_semaphores(device: &ash::Device, count: u32) -> VkResult<Vec<vk::Semaphore>> {
    let mut semaphores = Vec::with_capacity(count as usize);
    for _ in 0..count {
        match create_semaphore(device) {
            Ok(semaphore) => semaphores.push(semaphore),
            Err(err) => {
                for semaphore in semaphores {
                    unsafe { device.destroy_semaphore(semaphore, None) };
                }
                return Err(err);
            }
        }
    }
    Ok(semaphores)
}
