use crate::sync::create_fences;
use ash::{prelude::VkResult, vk};

#[derive(Clone)]
pub struct Commands {
    pub buffers: Vec<vk::CommandBuffer>,
    pub fences: Vec<vk::Fence>,
}

impl Commands {
    pub fn new(vk_device: &ash::Device, command_pool: vk::CommandPool) -> VkResult<Self> {
        Ok(Self {
            buffers: create_command_buffers(&vk_device, command_pool, 2)?,
            fences: create_fences(&vk_device, 2)?,
        })
    }

    pub fn destroy(&self, vk_device: &ash::Device) {
        unsafe {
            vk_device.device_wait_idle().unwrap();
            for &fence in self.fences.iter() {
                vk_device.destroy_fence(fence, None);
            }
        };
    }
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
