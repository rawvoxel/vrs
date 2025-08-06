use ash::{prelude::VkResult, vk};
use std::slice;

#[inline]
pub fn create_descriptor_pool(
    logical_device: &ash::Device,
    descriptor_type: vk::DescriptorType,
    descriptor_count: u32,
    max_sets: u32,
) -> VkResult<vk::DescriptorPool> {
    let size = vk::DescriptorPoolSize::default()
        .ty(descriptor_type)
        .descriptor_count(descriptor_count);
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(max_sets)
        .pool_sizes(slice::from_ref(&size));

    unsafe { logical_device.create_descriptor_pool(&info, None) }
}

#[inline]
pub fn create_descriptor_pools(
    logical_device: &ash::Device,
    descriptor_type: vk::DescriptorType,
    descriptor_count: u32,
    max_sets: u32,
    pool_count: u32,
) -> Vec<vk::DescriptorPool> {
    (0..pool_count)
        .map(|_| {
            create_descriptor_pool(logical_device, descriptor_type, descriptor_count, max_sets)
                .expect("Could not create vk::DescriptorPool.")
        })
        .collect()
}

#[inline]
pub fn create_descriptor_set_layout(
    render_device: &ash::Device,
    descriptor_type: vk::DescriptorType,
    stage_flags: vk::ShaderStageFlags,
) -> VkResult<vk::DescriptorSetLayout> {
    let layout_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(descriptor_type)
        .descriptor_count(1)
        .stage_flags(stage_flags);

    let info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(slice::from_ref(&layout_binding));

    unsafe { render_device.create_descriptor_set_layout(&info, None) }
}

#[inline]
pub fn create_descriptor_set(
    logical_device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    set_layouts: vk::DescriptorSetLayout,
) -> VkResult<vk::DescriptorSet> {
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(slice::from_ref(&set_layouts));

    unsafe { Ok(logical_device.allocate_descriptor_sets(&info)?[0]) }
}

#[inline]
pub fn create_descriptor_sets(
    logical_device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    set_layouts: &[vk::DescriptorSetLayout],
) -> VkResult<Vec<vk::DescriptorSet>> {
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(set_layouts);

    unsafe { logical_device.allocate_descriptor_sets(&info) }
}
