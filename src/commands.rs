use crate::sync;
use ash::{prelude::VkResult, vk};

#[derive(Clone)]
pub struct Commands {
    pub pool: vk::CommandPool,
    pub init_buffers: Vec<vk::CommandBuffer>,
    pub draw_buffers: Vec<vk::CommandBuffer>,
    pub init_fences: Vec<vk::Fence>,
    pub draw_fences: Vec<vk::Fence>,
}

impl Commands {
    pub fn new(vk_device: &ash::Device, queue_family_index: u32) -> VkResult<Self> {
        let pool = create_command_pool(&vk_device, queue_family_index)?;
        Ok(Self {
            pool,
            init_buffers: create_command_buffers(&vk_device, pool, 2)?,
            draw_buffers: create_command_buffers(&vk_device, pool, 2)?,
            init_fences: sync::create_fences(&vk_device, 2)?,
            draw_fences: sync::create_fences(&vk_device, 2)?,
        })
    }

    pub fn destroy(&self, vk_device: &ash::Device) {
        unsafe {
            vk_device.device_wait_idle().unwrap();
            for &fence in self.init_fences.iter() {
                vk_device.destroy_fence(fence, None);
            }
            for &fence in self.draw_fences.iter() {
                vk_device.destroy_fence(fence, None);
            }
            vk_device.destroy_command_pool(self.pool, None);
        };
    }
}

#[inline]
pub fn create_command_pool(
    vk_device: &ash::Device,
    queue_family_index: u32,
) -> VkResult<vk::CommandPool> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);

    unsafe { Ok(vk_device.create_command_pool(&command_pool_info, None)?) }
}

#[inline]
pub fn create_command_buffers(
    vk_device: &ash::Device,
    command_pool: vk::CommandPool,
    command_buffer_count: u32,
) -> VkResult<Vec<vk::CommandBuffer>> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(command_buffer_count)
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    unsafe { Ok(vk_device.allocate_command_buffers(&command_buffer_allocate_info)?) }
}

pub fn record_submit_commandbuffer<F: FnOnce(vk::CommandBuffer)>(
    vk_device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_dst_stage_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    op: F,
) -> VkResult<()> {
    let command_buffers = vec![command_buffer];
    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&command_buffers)
        .wait_dst_stage_mask(wait_dst_stage_mask)
        .wait_semaphores(wait_semaphores)
        .signal_semaphores(signal_semaphores);

    let reset_flags = vk::CommandBufferResetFlags::RELEASE_RESOURCES;

    unsafe {
        vk_device.wait_for_fences(&[command_buffer_fence], true, u64::MAX)?;
        vk_device.reset_fences(&[command_buffer_fence])?;
        vk_device.reset_command_buffer(command_buffer, reset_flags)?;
        vk_device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

        op(command_buffer);

        vk_device.end_command_buffer(command_buffer)?;
        vk_device.queue_submit(submit_queue, &[submit_info], command_buffer_fence)
    }
}
