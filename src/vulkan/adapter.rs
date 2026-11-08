use crate::vulkan::queue::QueueFamily;
use ash::{prelude::VkResult, vk};
use std::ffi::CStr;

/// The sizes compute shaders' subgroups can run at on an adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubgroupProperties {
    /// The size compute shaders run at unless a pipeline requires another.
    pub size: u32,
    /// The smallest size a compute pipeline can require.
    pub min_size: u32,
    /// The largest size a compute pipeline can require, and the most a subgroup ever holds.
    pub max_size: u32,
    /// Whether compute pipelines can require a size at all.
    pub requirable: bool,
}

#[derive(Clone)]
pub struct Adapter {
    pub(crate) handle: vk::PhysicalDevice,
    pub(crate) name: String,
    pub(crate) features: vk::PhysicalDeviceFeatures,
    pub(crate) properties: vk::PhysicalDeviceProperties,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) queue_families: Vec<QueueFamily>,
    /// The device extensions it has.
    pub(crate) extensions: Vec<vk::ExtensionProperties>,
    /// All zero before Vulkan 1.3, which can't be asked for them.
    pub(crate) subgroup: SubgroupProperties,
}
impl Adapter {
    #[inline]
    pub fn handle(&self) -> vk::PhysicalDevice {
        self.handle
    }

    /// The device's name, as its driver gives it.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
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

    /// The device extensions it has.
    #[inline]
    pub fn extensions(&self) -> &[vk::ExtensionProperties] {
        &self.extensions
    }

    /// The sizes its compute shaders' subgroups can run at.
    #[inline]
    pub fn subgroup(&self) -> SubgroupProperties {
        self.subgroup
    }
}

impl Adapter {
    pub fn new(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> VkResult<Self> {
        unsafe {
            let features = instance.get_physical_device_features(physical_device);
            let properties = instance.get_physical_device_properties(physical_device);
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let queue_families = get_queue_families(instance, physical_device);
            let extensions = instance.enumerate_device_extension_properties(physical_device)?;
            let name = properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();

            let mut adapter = Self {
                handle: physical_device,
                name,
                features,
                properties,
                memory_properties,
                queue_families,
                extensions,
                subgroup: SubgroupProperties {
                    size: 0,
                    min_size: 0,
                    max_size: 0,
                    requirable: false,
                },
            };
            if adapter.supports_vulkan_1_3() {
                let mut vulkan_11_properties = vk::PhysicalDeviceVulkan11Properties::default();
                let mut vulkan_13_properties = vk::PhysicalDeviceVulkan13Properties::default();
                let mut properties = vk::PhysicalDeviceProperties2::default()
                    .push_next(&mut vulkan_11_properties)
                    .push_next(&mut vulkan_13_properties);
                instance.get_physical_device_properties2(physical_device, &mut properties);
                adapter.subgroup = SubgroupProperties {
                    size: vulkan_11_properties.subgroup_size,
                    min_size: vulkan_13_properties.min_subgroup_size,
                    max_size: vulkan_13_properties.max_subgroup_size,
                    requirable: vulkan_13_properties
                        .required_subgroup_size_stages
                        .contains(vk::ShaderStageFlags::COMPUTE),
                };
            }
            Ok(adapter)
        }
    }

    /// Dynamic rendering and synchronization2 are core in Vulkan 1.3.
    #[inline]
    pub fn supports_vulkan_1_3(&self) -> bool {
        vk::api_version_major(self.properties.api_version) > 1
            || vk::api_version_minor(self.properties.api_version) >= 3
    }

    /// Whether it has the device extension `name`.
    pub(crate) fn has_extension(&self, name: &CStr) -> bool {
        self.extensions
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
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
