use ash::{prelude::VkResult, vk};

#[inline]
pub fn create_descriptor_pool(
    device: &ash::Device,
    pool_sizes: &[vk::DescriptorPoolSize],
    max_sets: u32,
) -> VkResult<vk::DescriptorPool> {
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(max_sets)
        .pool_sizes(pool_sizes);

    unsafe { device.create_descriptor_pool(&info, None) }
}

/// One descriptor per binding, numbered in the order of `types`.
#[inline]
pub fn create_descriptor_set_layout(
    device: &ash::Device,
    types: &[vk::DescriptorType],
    stage_flags: vk::ShaderStageFlags,
) -> VkResult<vk::DescriptorSetLayout> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = types
        .iter()
        .enumerate()
        .map(|(binding, &ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding as u32)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(stage_flags)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe { device.create_descriptor_set_layout(&info, None) }
}

/// `count` sets of the same layout.
#[inline]
pub fn allocate_descriptor_sets(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    count: u32,
) -> VkResult<Vec<vk::DescriptorSet>> {
    let set_layouts = vec![set_layout; count as usize];
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);

    unsafe { device.allocate_descriptor_sets(&info) }
}
