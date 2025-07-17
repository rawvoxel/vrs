use crate::{
    App,
    adapter::{Adapter, QueueFamily},
    debug::Debug,
    surface::Surface,
};
use anyhow::Result;
use ash::{khr, prelude::VkResult, vk};

#[derive(Clone)]
pub struct Context {
    pub debug: Debug,
    pub surface: Surface,
    pub adapters: Vec<Adapter>,
}

impl Context {
    pub fn new(app: &App) -> Result<Self> {
        unsafe {
            let debug = Debug::new(&app.debug_instance)?;
            let surface = Surface::new(app)?;
            let adapters: Vec<Adapter> = app
                .instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|physical_device| Adapter::new(app, physical_device).unwrap())
                .collect();

            Ok(Self {
                debug,
                surface,
                adapters,
            })
        }
    }

    pub fn destroy(&mut self, app: &App) {
        self.surface.destroy(&app.surface_instance);
        self.debug.destroy(&app.debug_instance);
    }
}

#[inline]
pub fn create_logical_device(
    app: &App,
    adapter: &Adapter,
    queue_family: &QueueFamily,
) -> VkResult<ash::Device> {
    unsafe {
        let queue_priorities = [1.0];
        let device_queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family.queue_index)
            .queue_priorities(&queue_priorities);

        let extension_names_raw = [
            khr::swapchain::NAME.as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            khr::portability_subset::NAME.as_ptr(),
        ];

        let features = vk::PhysicalDeviceFeatures::default().shader_clip_distance(true);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&device_queue_info))
            .enabled_extension_names(&extension_names_raw)
            .enabled_features(&features);

        app.instance
            .create_device(adapter.inner, &device_info, None)
    }
}

#[inline]
pub fn create_command_pool(
    logical_device: &ash::Device,
    queue_family: &QueueFamily,
) -> VkResult<vk::CommandPool> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family.queue_index);

    unsafe { logical_device.create_command_pool(&command_pool_info, None) }
}

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
        logical_device.wait_for_fences(&[command_buffer_fence], true, u64::MAX)?;
        logical_device.reset_fences(&[command_buffer_fence])?;
        logical_device.reset_command_buffer(command_buffer, reset_flags)?;
        logical_device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

        op(command_buffer);

        logical_device.end_command_buffer(command_buffer)?;
        logical_device.queue_submit(submit_queue, &[submit_info], command_buffer_fence)
    }
}
