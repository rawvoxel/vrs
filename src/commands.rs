use crate::queue::QueueFamily;
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

#[inline]
pub fn create_command_pools(
    logical_device: &ash::Device,
    queue_family: &QueueFamily,
    pool_count: u32,
) -> Vec<vk::CommandPool> {
    (0..pool_count)
        .map(|_| {
            create_command_pool(logical_device, queue_family)
                .expect("Could not create vk::CommandPool.")
        })
        .collect()
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

#[inline]
pub fn create_command_buffers(
    logical_device: &ash::Device,
    command_pools: &Vec<vk::CommandPool>,
    level: vk::CommandBufferLevel,
) -> Vec<vk::CommandBuffer> {
    command_pools
        .iter()
        .map(|&command_pool| {
            create_command_buffer(logical_device, command_pool, level)
                .expect("Could not create vk::CommandBuffer.")
        })
        .collect()
}

#[inline]
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
