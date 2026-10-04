use crate::adapter::Adapter;
use ash::{prelude::VkResult, vk};

#[inline]
pub fn vk_alloc(
    adapter: &Adapter,
    logical_device: &ash::Device,
    memory_requirements: vk::MemoryRequirements,
    memory_property_flags: vk::MemoryPropertyFlags,
) -> VkResult<vk::DeviceMemory> {
    // No memory of the right type is as good as none at all.
    let memory_type_index =
        find_memory_type_index(adapter, memory_requirements, memory_property_flags)
            .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
    // Asking for more than the whole heap breaks valid usage, rather than just failing.
    let memory_properties = &adapter.memory_properties;
    let heap_index = memory_properties.memory_types[memory_type_index as usize].heap_index;
    if memory_requirements.size > memory_properties.memory_heaps[heap_index as usize].size {
        return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    }
    let memory_allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type_index);

    unsafe { logical_device.allocate_memory(&memory_allocate_info, None) }
}

#[inline]
fn find_memory_type_index(
    adapter: &Adapter,
    memory_requirements: vk::MemoryRequirements,
    memory_property_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    adapter.memory_properties.memory_types[..adapter.memory_properties.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_requirements.memory_type_bits != 0
                && memory_type.property_flags & memory_property_flags == memory_property_flags
        })
        .map(|(index, _)| index as _)
}
