use crate::adapter::Adapter;
use ash::{prelude::VkResult, vk};

#[inline]
pub fn vk_alloc(
    adapter: &Adapter,
    logical_device: &ash::Device,
    memory_property_flags: vk::MemoryPropertyFlags,
    memory_requirements: vk::MemoryRequirements,
) -> VkResult<vk::DeviceMemory> {
    let memory_type_index =
        find_memory_type_index(adapter, memory_property_flags, memory_requirements);

    let memory_allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type_index);

    unsafe { logical_device.allocate_memory(&memory_allocate_info, None) }
}

#[inline]
fn find_memory_type_index(
    adapter: &Adapter,
    memory_property_flags: vk::MemoryPropertyFlags,
    memory_requirements: vk::MemoryRequirements,
) -> u32 {
    adapter.memory_properties.memory_types[..adapter.memory_properties.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_requirements.memory_type_bits != 0
                && memory_type.property_flags & memory_property_flags == memory_property_flags
        })
        .map(|(index, _)| index as _)
        .expect("Could not find suitable memory type index.")
}
