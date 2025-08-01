use ash::{prelude::VkResult, vk};
use std::slice;

pub fn create_ubo_descriptor_pool(
    logical_device: &ash::Device,
    descriptor_count: u32,
) -> VkResult<vk::DescriptorPool> {
    let descriptor_pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(descriptor_count);

    let info =
        vk::DescriptorPoolCreateInfo::default().pool_sizes(slice::from_ref(&descriptor_pool_size));

    unsafe { logical_device.create_descriptor_pool(&info, None) }
}
