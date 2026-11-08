use ash::{prelude::VkResult, vk};

/// Records a barrier making `src`'s accesses in its stages available and visible to `dst`'s
/// accesses in its stages, for every resource: each a `(stage, access)` pair.
///
/// # Safety
/// `cmd_buffer` must be recording.
#[inline]
pub unsafe fn cmd_memory_barrier(
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

/// Records a barrier after which every write recorded before it is done and visible to every
/// read and write recorded after it, in any stage.
///
/// # Safety
/// `command_buffer` must be recording.
#[inline]
pub unsafe fn cmd_pipeline_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    unsafe {
        cmd_memory_barrier(
            device,
            command_buffer,
            (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
            (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        )
    };
}

/// Waits until `fence` is signaled: the driver's own wait, as long as the work takes. A lost
/// device returns `ERROR_DEVICE_LOST`.
///
/// # Safety
/// `fence` is signaled, or a submission still to finish signals it.
#[inline]
pub unsafe fn wait_for_fence(device: &ash::Device, fence: vk::Fence) -> VkResult<()> {
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX) }
}

/// Waits until `device` runs nothing: every queue's work done, a shared device's other user's
/// included. On a lost device the driver's wait returns `ERROR_DEVICE_LOST` in finite time, and
/// nothing is pending after it (the spec's Lost Device section).
///
/// # Safety
/// No other thread submits to `device`'s queues during the wait.
#[inline]
pub unsafe fn wait_for_device_idle(device: &ash::Device) -> VkResult<()> {
    unsafe { device.device_wait_idle() }
}

/// Whether `fence` is signaled, without waiting. A lost device returns `ERROR_DEVICE_LOST`.
///
/// # Safety
/// `fence` is a fence of `device`'s.
#[inline]
pub unsafe fn fence_signaled(device: &ash::Device, fence: vk::Fence) -> VkResult<bool> {
    unsafe { device.get_fence_status(fence) }
}

/// Starts signaled, so the first wait on it returns straight away.
#[inline]
pub fn create_fence(device: &ash::Device) -> VkResult<vk::Fence> {
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    unsafe { device.create_fence(&fence_info, None) }
}

/// Destroys `fence`.
///
/// # Safety
///
/// `fence` was made on `device`, and no submission still pending signals it.
#[inline]
pub unsafe fn destroy_fence(device: &ash::Device, fence: vk::Fence) {
    unsafe { device.destroy_fence(fence, None) };
}
