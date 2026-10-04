use crate::{queue::QueueFamily, surface::Surface};
use ash::{khr, vk};

#[derive(Clone)]
pub struct Adapter {
    pub handle: vk::PhysicalDevice,
    pub features: vk::PhysicalDeviceFeatures,
    /// The Vulkan 1.2 `drawIndirectCount` feature.
    pub draw_indirect_count: bool,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<QueueFamily>,
}

impl Adapter {
    pub fn new(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        unsafe {
            let features = instance.get_physical_device_features(physical_device);
            let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
            let mut features2 =
                vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan_12_features);
            instance.get_physical_device_features2(physical_device, &mut features2);
            let draw_indirect_count = vulkan_12_features.draw_indirect_count == vk::TRUE;
            let properties = instance.get_physical_device_properties(physical_device);
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let queue_families = get_queue_families(&instance, physical_device);

            Self {
                handle: physical_device,
                features,
                draw_indirect_count,
                properties,
                memory_properties,
                queue_families,
            }
        }
    }

    #[inline]
    pub fn get_queue_family(
        &self,
        surface_instance: &khr::surface::Instance,
        surface: &Surface,
        supports_surface: bool,
        queue_flags: vk::QueueFlags,
    ) -> QueueFamily {
        self.queue_families
            .iter()
            .find_map(|queue_family| unsafe {
                let contains_flags = queue_family.properties.queue_flags.contains(queue_flags);
                let supports_surface = supports_surface
                    == surface_instance
                        .get_physical_device_surface_support(
                            self.handle,
                            queue_family.index,
                            surface.handle,
                        )
                        .expect("Could not get physical device surface support.");

                if contains_flags && supports_surface {
                    Some(queue_family.clone())
                } else {
                    None
                }
            })
            .expect("Could not find suitable queue family.")
    }
}

fn get_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Vec<QueueFamily> {
    unsafe {
        instance
            .get_physical_device_queue_family_properties(physical_device)
            .into_iter()
            .enumerate()
            .map(|(index, properties)| QueueFamily {
                properties,
                index: index as u32,
            })
            .collect()
    }
}
