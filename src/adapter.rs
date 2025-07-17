use crate::App;
use ash::{prelude::VkResult, vk};

#[derive(Clone)]
pub struct Adapter {
    pub inner: vk::PhysicalDevice,
    pub features: vk::PhysicalDeviceFeatures,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Adapter {
    #[rustfmt::skip]
    pub fn new(app: &App, physical_device: vk::PhysicalDevice) -> VkResult<Self> {
        unsafe {
            let features = app.instance.get_physical_device_features(physical_device);
            let properties = app.instance.get_physical_device_properties(physical_device);
            let memory_properties = app.instance.get_physical_device_memory_properties(physical_device);

            Ok(Self {
                inner: physical_device,
                features,
                properties,
                memory_properties,
            })
        }
    }
}

#[derive(Clone)]
pub struct QueueFamily {
    pub properties: vk::QueueFamilyProperties,
    pub queue_index: u32,
}

#[inline]
pub fn supports_queue_family(
    app: &App,
    physical_device: vk::PhysicalDevice,
    queue_family_properties: vk::QueueFamilyProperties,
    queue_family_index: u32,
    surface: vk::SurfaceKHR,
    queue_flags: vk::QueueFlags,
    supports_surface: bool,
) -> VkResult<bool> {
    unsafe {
        let contains_flags = queue_family_properties.queue_flags.contains(queue_flags);
        let supports_surface = supports_surface
            == app.surface_instance.get_physical_device_surface_support(
                physical_device,
                queue_family_index,
                surface,
            )?;

        Ok(contains_flags && supports_surface)
    }
}

#[inline]
pub fn request_physical_device_type<'a>(
    adapters: &'a Vec<Adapter>,
    device_type: vk::PhysicalDeviceType,
) -> Option<&'a Adapter> {
    adapters.iter().find_map(|adapter| {
        if adapter.properties.device_type == device_type {
            Some(adapter)
        } else {
            None
        }
    })
}

#[inline]
pub fn request_queue_family(
    app: &App,
    adapter: &Adapter,
    surface: vk::SurfaceKHR,
    queue_flags: vk::QueueFlags,
    supports_surface: bool,
) -> VkResult<QueueFamily> {
    let queue_families = unsafe {
        app.instance
            .get_physical_device_queue_family_properties(adapter.inner)
    };
    queue_families
        .iter()
        .enumerate()
        .find_map(|(queue_family_index, &queue_family_properties)| {
            if supports_queue_family(
                app,
                adapter.inner,
                queue_family_properties,
                queue_family_index as u32,
                surface,
                queue_flags,
                supports_surface,
            )
            .unwrap()
            {
                let queue_family = QueueFamily {
                    properties: queue_family_properties,
                    queue_index: queue_family_index as u32,
                };
                Some(Ok(queue_family))
            } else {
                None
            }
        })
        .expect("Could not find suitable queue family.")
}
