use crate::vulkan::queue::QueueFamily;
use ash::{khr, prelude::VkResult, vk};

#[derive(Clone)]
pub struct Adapter {
    pub(crate) handle: vk::PhysicalDevice,
    pub(crate) features: vk::PhysicalDeviceFeatures,
    /// The Vulkan 1.2 `drawIndirectCount` feature.
    pub(crate) draw_indirect_count: bool,
    /// The Vulkan 1.1 `shaderDrawParameters` feature.
    pub(crate) shader_draw_parameters: bool,
    pub(crate) properties: vk::PhysicalDeviceProperties,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) queue_families: Vec<QueueFamily>,
    /// The queue families the buffers made with it are used on, when more than one (a
    /// [`GpuContext`](crate::vulkan::gpu::GpuContext)'s, with a compute queue of its own): they're
    /// made `CONCURRENT` across them, so each queue reads and writes them with no ownership
    /// transfer. Empty: `EXCLUSIVE`.
    pub(crate) shared_by: Vec<u32>,
}
impl Adapter {
    #[inline]
    pub fn handle(&self) -> vk::PhysicalDevice {
        self.handle
    }

    #[inline]
    pub fn features(&self) -> &vk::PhysicalDeviceFeatures {
        &self.features
    }

    #[inline]
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    #[inline]
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    #[inline]
    pub fn queue_families(&self) -> &[QueueFamily] {
        &self.queue_families
    }

    /// The first family of compute queues without graphics (an async compute queue's) that can
    /// write timestamps, if it has one.
    pub fn compute_only_family(&self) -> Option<QueueFamily> {
        self.queue_families.iter().copied().find(|family| {
            let properties = family.properties();
            properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && properties.queue_count >= 1
                && properties.timestamp_valid_bits > 0
        })
    }

    /// `info` shared across the queue families buffers are used on ([`Adapter::shared_by`]'s):
    /// `CONCURRENT` across them, or `EXCLUSIVE` with one.
    pub(crate) fn sharing<'a>(&'a self, info: vk::BufferCreateInfo<'a>) -> vk::BufferCreateInfo<'a> {
        if self.shared_by.len() > 1 {
            info.sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&self.shared_by)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        }
    }
}

impl Adapter {
    pub fn new(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        unsafe {
            let features = instance.get_physical_device_features(physical_device);
            let mut vulkan_11_features = vk::PhysicalDeviceVulkan11Features::default();
            let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut vulkan_11_features)
                .push_next(&mut vulkan_12_features);
            instance.get_physical_device_features2(physical_device, &mut features2);
            let draw_indirect_count = vulkan_12_features.draw_indirect_count == vk::TRUE;
            let shader_draw_parameters = vulkan_11_features.shader_draw_parameters == vk::TRUE;
            let properties = instance.get_physical_device_properties(physical_device);
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let queue_families = get_queue_families(instance, physical_device);

            Self {
                handle: physical_device,
                features,
                draw_indirect_count,
                shader_draw_parameters,
                properties,
                memory_properties,
                queue_families,
                shared_by: Vec::new(),
            }
        }
    }

    /// The first queue family with `queue_flags`, which can also present to `surface` when there is one.
    #[inline]
    pub fn get_queue_family(
        &self,
        surface_instance: &khr::surface::Instance,
        surface: Option<vk::SurfaceKHR>,
        queue_flags: vk::QueueFlags,
    ) -> VkResult<Option<QueueFamily>> {
        for queue_family in self.queue_families.iter() {
            if !queue_family.properties.queue_flags.contains(queue_flags) {
                continue;
            }
            let presents = match surface {
                Some(surface) => unsafe {
                    surface_instance.get_physical_device_surface_support(
                        self.handle,
                        queue_family.index,
                        surface,
                    )?
                },
                None => true,
            };
            if presents {
                return Ok(Some(*queue_family));
            }
        }
        Ok(None)
    }

    /// Whether it has every feature the renderer needs: Vulkan 1.3, for dynamic rendering and
    /// synchronization2, indirect draws that name their mesh by their first instance, and
    /// indirect draws whose count the device writes.
    #[inline]
    pub fn supports_renderer(&self) -> bool {
        self.supports_vulkan_1_3()
            && self.features.draw_indirect_first_instance == vk::TRUE
            && self.features.multi_draw_indirect == vk::TRUE
            && self.draw_indirect_count
            && self.shader_draw_parameters
    }

    /// Dynamic rendering and synchronization2 are core in Vulkan 1.3.
    #[inline]
    pub fn supports_vulkan_1_3(&self) -> bool {
        vk::api_version_major(self.properties.api_version) > 1
            || vk::api_version_minor(self.properties.api_version) >= 3
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
