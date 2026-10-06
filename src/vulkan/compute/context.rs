use crate::{
    App,
    vulkan::{
        adapter::Adapter,
        commands,
        queue::{Queue, QueueFamily},
        sync,
    },
};
use ash::{prelude::VkResult, vk};
use std::slice;

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

/// A device for compute alone, on any adapter, with one queue to submit to.
///
/// It has buffer device addresses, 64-bit integers, timeline semaphores, synchronization2 and
/// subgroup size control with full subgroups, and its compute shaders have basic and ballot
/// subgroup operations. Adapters without them are refused with
/// [`vk::Result::ERROR_FEATURE_NOT_PRESENT`]; [`ComputeContext::missing_features`] says which.
pub struct ComputeContext {
    adapter: Adapter,
    device: ash::Device,
    queue_family: QueueFamily,
    queue: Queue,
    /// For [`ComputeContext::submit_and_wait`].
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    subgroup: SubgroupProperties,
}

/// What an adapter offers, for deciding whether it can make a [`ComputeContext`].
struct Support {
    missing: Vec<&'static str>,
    subgroup: SubgroupProperties,
    /// Whether subgroup operations work on 64-bit and 16-bit types.
    subgroup_extended_types: bool,
}

impl ComputeContext {
    /// Makes the device on `adapter`. `app` must have been made with Vulkan 1.3 or later.
    pub fn new(app: &App, adapter: &Adapter) -> VkResult<Self> {
        let support = query_support(app, adapter);
        if !support.missing.is_empty() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let queue_family =
            pick_queue_family(adapter).ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family.index())
            .queue_priorities(&queue_priorities);
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .shader_subgroup_extended_types(support.subgroup_extended_types);
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(slice::from_ref(&queue_info))
            .enabled_features(&features)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);
        let device = unsafe {
            app.instance()
                .create_device(adapter.handle(), &device_info, None)?
        };

        // Dropping `context` destroys whatever was made, if anything below fails.
        let mut context = Self {
            adapter: adapter.clone(),
            queue: Queue::new(&queue_family, &device, 0),
            device,
            queue_family,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            subgroup: support.subgroup,
        };
        context.command_pool = commands::create_command_pool(&context.device, &queue_family)?;
        context.command_buffer = commands::create_command_buffer(
            &context.device,
            context.command_pool,
            vk::CommandBufferLevel::PRIMARY,
        )?;
        context.fence = sync::create_fence(&context.device)?;
        Ok(context)
    }

    /// Why `adapter` can't make a [`ComputeContext`], or nothing when it can.
    pub fn missing_features(app: &App, adapter: &Adapter) -> Vec<&'static str> {
        let mut missing = query_support(app, adapter).missing;
        if pick_queue_family(adapter).is_none() {
            missing.push("a compute queue");
        }
        missing
    }

    #[inline]
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    #[inline]
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    #[inline]
    pub fn queue(&self) -> Queue {
        self.queue
    }

    #[inline]
    pub fn queue_family(&self) -> QueueFamily {
        self.queue_family
    }

    #[inline]
    pub fn subgroup(&self) -> SubgroupProperties {
        self.subgroup
    }

    /// Nanoseconds per timestamp tick.
    #[inline]
    pub fn timestamp_period(&self) -> f32 {
        self.adapter.properties().limits.timestamp_period
    }

    /// How many bits of the queue's timestamps are meaningful; 0 when it has none.
    #[inline]
    pub fn timestamp_valid_bits(&self) -> u32 {
        self.queue_family.properties().timestamp_valid_bits
    }

    /// Records commands with `record`, submits them and waits until the device has run them.
    pub fn submit_and_wait<F: FnOnce(vk::CommandBuffer)>(&mut self, record: F) -> VkResult<()> {
        commands::record_submit_command_buffer(
            &self.device,
            self.command_buffer,
            self.fence,
            self.queue.handle(),
            &[],
            &[],
            &[],
            record,
        )?;
        unsafe {
            self.device
                .wait_for_fences(slice::from_ref(&self.fence), true, u64::MAX)
        }
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        unsafe {
            // Nothing can be destroyed while the device might still use it.
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

fn query_support(app: &App, adapter: &Adapter) -> Support {
    let instance = app.instance();
    let mut missing = Vec::new();

    let api_version = adapter.properties().api_version;
    if (
        vk::api_version_major(api_version),
        vk::api_version_minor(api_version),
    ) < (1, 3)
    {
        missing.push("Vulkan 1.3");
    }

    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default();
    let shader_int64 = {
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);
        unsafe { instance.get_physical_device_features2(adapter.handle(), &mut features) };
        features.features.shader_int64 == vk::TRUE
    };
    let required = [
        (shader_int64, "shaderInt64"),
        (
            vulkan_12_features.buffer_device_address == vk::TRUE,
            "bufferDeviceAddress",
        ),
        (
            vulkan_12_features.timeline_semaphore == vk::TRUE,
            "timelineSemaphore",
        ),
        (
            vulkan_13_features.synchronization2 == vk::TRUE,
            "synchronization2",
        ),
        (
            vulkan_13_features.subgroup_size_control == vk::TRUE,
            "subgroupSizeControl",
        ),
        (
            vulkan_13_features.compute_full_subgroups == vk::TRUE,
            "computeFullSubgroups",
        ),
    ];
    missing.extend(
        required
            .iter()
            .filter(|(present, _)| !present)
            .map(|(_, name)| *name),
    );

    let mut vulkan_11_properties = vk::PhysicalDeviceVulkan11Properties::default();
    let mut vulkan_13_properties = vk::PhysicalDeviceVulkan13Properties::default();
    {
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vulkan_11_properties)
            .push_next(&mut vulkan_13_properties);
        unsafe { instance.get_physical_device_properties2(adapter.handle(), &mut properties) };
    }
    if !vulkan_11_properties
        .subgroup_supported_stages
        .contains(vk::ShaderStageFlags::COMPUTE)
    {
        missing.push("subgroups in compute shaders");
    }
    if !vulkan_11_properties.subgroup_supported_operations.contains(
        vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::BALLOT
            | vk::SubgroupFeatureFlags::ARITHMETIC,
    ) {
        missing.push("basic, ballot and arithmetic subgroup operations");
    }

    Support {
        missing,
        subgroup: SubgroupProperties {
            size: vulkan_11_properties.subgroup_size,
            min_size: vulkan_13_properties.min_subgroup_size,
            max_size: vulkan_13_properties.max_subgroup_size,
            requirable: vulkan_13_properties
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE),
        },
        subgroup_extended_types: vulkan_12_features.shader_subgroup_extended_types == vk::TRUE,
    }
}

/// A compute queue family, preferring one without graphics (an async compute queue), then one
/// with timestamps.
fn pick_queue_family(adapter: &Adapter) -> Option<QueueFamily> {
    let families = adapter.queue_families();
    let compute = |family: &&QueueFamily| {
        family
            .properties()
            .queue_flags
            .contains(vk::QueueFlags::COMPUTE)
    };
    let without_graphics = |family: &&QueueFamily| {
        !family
            .properties()
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS)
    };
    let timed = |family: &&QueueFamily| family.properties().timestamp_valid_bits > 0;
    families
        .iter()
        .filter(compute)
        .find(|family| without_graphics(family) && timed(family))
        .or_else(|| families.iter().filter(compute).find(timed))
        .or_else(|| families.iter().find(compute))
        .copied()
}
