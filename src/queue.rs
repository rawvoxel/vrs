use crate::{context::Context, sync};
use ash::{prelude::VkResult, vk};

#[derive(Clone)]
pub struct PresentQueue {
    pub inner: vk::Queue,
    pub mode: vk::PresentModeKHR,
    pub presented_semaphores: Vec<vk::Semaphore>,
    pub rendered_semaphores: Vec<vk::Semaphore>,
}

impl PresentQueue {
    pub fn new(context: &Context) -> VkResult<Self> {
        unsafe {
            let queue = context
                .vk_device
                .get_device_queue(context.queue_family_index, 0);
            let modes = context
                .surface_instance
                .get_physical_device_surface_present_modes(
                    context.physical_device,
                    context.surface.inner,
                )?;
            let mode = modes
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let presented_semaphores = sync::create_semaphores(&context.vk_device, 2)?;
            let rendered_semaphores = sync::create_semaphores(&context.vk_device, 2)?;

            Ok(Self {
                inner: queue,
                mode,
                presented_semaphores,
                rendered_semaphores,
            })
        }
    }

    pub fn destroy(&self, vk_device: &ash::Device) {
        unsafe {
            vk_device.device_wait_idle().unwrap();
            for &sem in self.presented_semaphores.iter() {
                vk_device.destroy_semaphore(sem, None);
            }
            for &sem in self.rendered_semaphores.iter() {
                vk_device.destroy_semaphore(sem, None);
            }
        }
    }
}
