use crate::{queue::QueueFamily, surface::Surface};
use ash::{khr, prelude::VkResult, vk};

#[derive(Clone)]
pub struct Adapter {
    pub inner: vk::PhysicalDevice,
    pub features: vk::PhysicalDeviceFeatures,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<QueueFamily>,
}

impl Adapter {
    pub fn new(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> VkResult<Self> {
        unsafe {
            let features = instance.get_physical_device_features(physical_device);
            let properties = instance.get_physical_device_properties(physical_device);
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let queue_families = instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .map(|(index, properties)| QueueFamily {
                    properties,
                    index: index as u32,
                })
                .collect();

            Ok(Self {
                inner: physical_device,
                features,
                properties,
                memory_properties,
                queue_families,
            })
        }
    }

    #[inline]
    pub fn get_queue_family(
        &self,
        surface_instance: &khr::surface::Instance,
        surface: &Surface,
        supports_surface: bool,
        queue_flags: vk::QueueFlags,
    ) -> VkResult<QueueFamily> {
        self.queue_families
            .iter()
            .find_map(|queue_family| unsafe {
                let contains_flags = queue_family.properties.queue_flags.contains(queue_flags);
                let supports_surface = supports_surface
                    == surface_instance
                        .get_physical_device_surface_support(
                            self.inner,
                            queue_family.index,
                            surface.inner,
                        )
                        .expect("Could not get physical device surface support.");

                if contains_flags && supports_surface {
                    Some(Ok(queue_family.clone()))
                } else {
                    None
                }
            })
            .expect("Could not find suitable queue family.")
    }
}
