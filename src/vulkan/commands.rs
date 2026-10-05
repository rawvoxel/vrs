use crate::vulkan::queue::QueueFamily;
use ash::{prelude::VkResult, vk};
use std::slice;

#[inline]
pub fn create_command_pool(
    logical_device: &ash::Device,
    queue_family: &QueueFamily,
) -> VkResult<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family.index);

    unsafe { logical_device.create_command_pool(&info, None) }
}

/// If one fails, those already created are destroyed.
#[inline]
pub fn create_command_pools(
    logical_device: &ash::Device,
    queue_family: &QueueFamily,
    pool_count: u32,
) -> VkResult<Vec<vk::CommandPool>> {
    let mut pools = Vec::with_capacity(pool_count as usize);
    for _ in 0..pool_count {
        match create_command_pool(logical_device, queue_family) {
            Ok(pool) => pools.push(pool),
            Err(err) => {
                for pool in pools {
                    unsafe { logical_device.destroy_command_pool(pool, None) };
                }
                return Err(err);
            }
        }
    }
    Ok(pools)
}

#[inline]
pub fn create_command_buffer(
    logical_device: &ash::Device,
    command_pool: vk::CommandPool,
    level: vk::CommandBufferLevel,
) -> VkResult<vk::CommandBuffer> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .command_buffer_count(1)
        .level(level);

    unsafe { Ok(logical_device.allocate_command_buffers(&allocate_info)?[0]) }
}

/// One from each pool, they're freed with their pools.
#[inline]
pub fn create_command_buffers(
    logical_device: &ash::Device,
    command_pools: &[vk::CommandPool],
    level: vk::CommandBufferLevel,
) -> VkResult<Vec<vk::CommandBuffer>> {
    command_pools
        .iter()
        .map(|&command_pool| create_command_buffer(logical_device, command_pool, level))
        .collect()
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn record_submit_commandbuffer<F: FnOnce(vk::CommandBuffer)>(
    logical_device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_dst_stage_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    op: F,
) -> VkResult<()> {
    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    let submit_info = [vk::SubmitInfo::default()
        .command_buffers(slice::from_ref(&command_buffer))
        .wait_dst_stage_mask(wait_dst_stage_mask)
        .wait_semaphores(wait_semaphores)
        .signal_semaphores(signal_semaphores)];

    unsafe {
        logical_device.wait_for_fences(&[command_buffer_fence], true, u64::MAX)?;
        logical_device.reset_fences(&[command_buffer_fence])?;
        logical_device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

        op(command_buffer);

        logical_device.end_command_buffer(command_buffer)?;
        logical_device.queue_submit(submit_queue, &submit_info, command_buffer_fence)
    }
}
