use ash::vk;

#[derive(Clone, Copy)]
pub struct QueueFamily {
    pub(crate) properties: vk::QueueFamilyProperties,
    pub(crate) index: u32,
}
impl QueueFamily {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub fn properties(&self) -> &vk::QueueFamilyProperties {
        &self.properties
    }
}

#[derive(Clone, Copy)]
pub struct Queue {
    pub(crate) handle: vk::Queue,
    pub(crate) index: u32,
}

impl Queue {
    #[inline]
    pub fn handle(&self) -> vk::Queue {
        self.handle
    }

    /// Its index within its family.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn new(queue_family: &QueueFamily, device: &ash::Device, queue_index: u32) -> Self {
        unsafe {
            let queue = device.get_device_queue(queue_family.index, queue_index);
            Self {
                handle: queue,
                index: queue_index,
            }
        }
    }
}
