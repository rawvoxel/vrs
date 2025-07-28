use ash::vk;

#[derive(Clone, Copy)]
pub struct QueueFamily {
    pub properties: vk::QueueFamilyProperties,
    pub index: u32,
}

#[derive(Clone, Copy)]
pub struct Queue {
    pub handle: vk::Queue,
    pub index: u32,
}

impl Queue {
    pub fn new(queue_family: &QueueFamily, logical_device: &ash::Device, queue_index: u32) -> Self {
        unsafe {
            let queue = logical_device.get_device_queue(queue_family.index, queue_index);
            Self {
                handle: queue,
                index: queue_index,
            }
        }
    }
}
