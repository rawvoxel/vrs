use crate::{create_command_buffers, sync::create_fences};
use ash::{prelude::VkResult, vk};

#[derive(Clone)]
pub struct Commands {
    pub fences: Vec<vk::Fence>,
    pub buffers: Vec<vk::CommandBuffer>,
}

impl Commands {
    pub fn new(logical_device: &ash::Device, command_pool: vk::CommandPool) -> VkResult<Self> {
        Ok(Self {
            buffers: create_command_buffers(logical_device, command_pool, 2)?,
            fences: create_fences(logical_device, 2),
        })
    }

    pub fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.device_wait_idle().unwrap();
            for &fence in self.fences.iter() {
                logical_device.destroy_fence(fence, None);
            }
        };
    }
}
