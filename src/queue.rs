use crate::{
    App,
    adapter::{Adapter, QueueFamily},
    context::Context,
    sync,
};
use ash::{prelude::VkResult, vk};

#[derive(Clone)]
pub struct PresentQueue {
    pub inner: vk::Queue,
    pub mode: vk::PresentModeKHR,
    pub presented_semaphores: Vec<vk::Semaphore>,
    pub rendered_semaphores: Vec<vk::Semaphore>,
}

impl PresentQueue {
    pub fn new(
        app: &App,
        context: &Context,
        adapter: &Adapter,
        queue_family: &QueueFamily,
        logical_device: &ash::Device,
        queue_index: u32,
    ) -> VkResult<Self> {
        unsafe {
            let queue = logical_device.get_device_queue(queue_family.queue_index, queue_index);
            let modes = app
                .surface_instance
                .get_physical_device_surface_present_modes(adapter.inner, context.surface.inner)?;
            let mode = modes
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let presented_semaphores = sync::create_semaphores(logical_device, 2)?;
            let rendered_semaphores = sync::create_semaphores(logical_device, 2)?;

            Ok(Self {
                inner: queue,
                mode,
                presented_semaphores,
                rendered_semaphores,
            })
        }
    }

    #[rustfmt::skip]
    pub fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.device_wait_idle().unwrap();
            for &sem in self.presented_semaphores.iter() {
                logical_device.destroy_semaphore(sem, None);
            }
            for &sem in self.rendered_semaphores.iter() {
                logical_device.destroy_semaphore(sem, None);
            }
        }
    }
}
