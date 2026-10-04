use crate::{App, adapter::Adapter, queue::QueueFamily};
use ash::{khr, prelude::VkResult, vk};
use std::slice;

pub(super) fn create_render_device(
    app: &App,
    adapter: &Adapter,
    queue_family: &QueueFamily,
    multi_draw_indirect: bool,
    draw_indirect_count: bool,
    swapchain: bool,
) -> VkResult<ash::Device> {
    let queue_priorities = [1.0];
    let device_queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family.index)
        .queue_priorities(&queue_priorities);

    // Headless devices have no swapchain to present with.
    let mut extension_names_raw = Vec::new();
    if swapchain {
        extension_names_raw.push(khr::swapchain::NAME.as_ptr());
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    extension_names_raw.push(khr::portability_subset::NAME.as_ptr());

    // NOTE: Keep in mind proper ordering when adding features.
    // Indirect draws name their mesh by their first instance, which the mesh shaders read.
    let features = vk::PhysicalDeviceFeatures::default()
        .multi_draw_indirect(multi_draw_indirect)
        .draw_indirect_first_instance(true);
    let mut vulkan_11_features =
        vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
    let mut synchronization2 =
        vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    let mut dynamic_rendering =
        vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
    let mut vulkan_12_features =
        vk::PhysicalDeviceVulkan12Features::default().draw_indirect_count(draw_indirect_count);

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(slice::from_ref(&device_queue_info))
        .enabled_extension_names(&extension_names_raw)
        .enabled_features(&features)
        .push_next(&mut synchronization2)
        .push_next(&mut dynamic_rendering)
        .push_next(&mut vulkan_11_features)
        .push_next(&mut vulkan_12_features);

    unsafe {
        app.instance
            .create_device(adapter.handle, &device_info, None)
    }
}
