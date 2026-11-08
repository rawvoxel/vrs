use crate::{
    App,
    vulkan::{
        adapter::Adapter,
        queue::{Queue, QueueFamily},
    },
};
use ash::{ext, khr, prelude::VkResult, vk};
use std::slice;

mod shared;

pub use shared::create_shared_device;

/// A device for GPU-driven rendering on any adapter that has it: compute, and rasterization
/// through task and mesh shaders, on one queue, with buffers reached by address.
///
/// It has Vulkan 1.3's synchronization2, dynamic rendering, maintenance4, subgroup size control
/// with full subgroups, and demoting fragments to helpers (what `discard` compiles to); buffer
/// device addresses, timeline semaphores, 16-bit and 64-bit integers, 16-bit floats, 64-bit
/// buffer atomics, stores and atomics in fragment shaders, and task and mesh shaders; and the
/// shader clock and pipeline executables where the adapter has them. [`missing_features`] says
/// what an adapter lacks.
///
/// [`create_device`] makes one on an adapter, which an [`App`] destroys, and
/// [`create_shared_device`] makes one from another Vulkan user's create info, which its caller
/// destroys. It holds handles and function tables only: a copy is the same device.
#[derive(Clone)]
pub struct VulkanDevice {
    /// The device's functions.
    pub device: ash::Device,
    /// What it was made on: its limits, memory types, subgroup sizes and timestamp period.
    pub adapter: Adapter,
    /// Its one queue, for graphics and compute, with timestamps. Vulkan requires submits to it to
    /// be externally synchronized: one submitter at a time.
    pub queue: Queue,
    /// The queue's family, whose properties give its timestamps' valid bits.
    pub queue_family: QueueFamily,
    /// The `VK_EXT_mesh_shader` commands: `cmd_draw_mesh_tasks` and its indirect forms.
    pub mesh_shader: ext::mesh_shader::Device,
    /// The `VK_KHR_pipeline_executable_properties` commands, where the adapter has them: what the
    /// driver compiled a pipeline into (its registers, its shared memory), for pipelines made to
    /// capture it
    /// ([`ComputePipelines::capturing_statistics`](crate::vulkan::compute::ComputePipelines::capturing_statistics)).
    pub pipeline_executables: Option<khr::pipeline_executable_properties::Device>,
    /// The instance, kept for [`memory_budget`] where the adapter reports budgets
    /// (`VK_EXT_memory_budget`).
    pub budget: Option<ash::Instance>,
    /// Whether shaders may read the subgroup's and the device's clocks (`VK_KHR_shader_clock`,
    /// enabled where the adapter has it): for instrumented builds that time their own phases.
    pub shader_clock: bool,
}

/// Makes the device on `adapter`, one of `app`'s adapters. `app` records it and destroys it when
/// dropped, or earlier with [`App::destroy_device`]. Adapters that lack something are refused
/// with [`vk::Result::ERROR_FEATURE_NOT_PRESENT`]; [`missing_features`] says what.
///
/// # Safety
/// Everything made with the device is dropped before `app` is, or before
/// [`App::destroy_device`] destroys the device.
pub unsafe fn create_device(app: &mut App, adapter: &Adapter) -> VkResult<VulkanDevice> {
    let queue_family = pick_queue_family(adapter).ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
    let queue_priorities = [1.0];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family.index())
        .queue_priorities(&queue_priorities);
    let info = vk::DeviceCreateInfo::default().queue_create_infos(slice::from_ref(&queue_info));
    let device = unsafe {
        shared::create_merged(
            app.instance(),
            adapter.clone(),
            queue_family,
            &mut Vec::new(),
            &info,
        )?
    };
    app.devices.push(device.device.clone());
    Ok(device)
}

/// Why `adapter` can't make a [`VulkanDevice`], or nothing when it can.
pub fn missing_features(app: &App, adapter: &Adapter) -> Vec<&'static str> {
    let mut missing = query_support(app.instance(), adapter).missing;
    if pick_queue_family(adapter).is_none() {
        missing.push("a graphics and compute queue with timestamps");
    }
    missing
}

/// Each memory heap's budget and what this process uses of it, in bytes, as the driver says now
/// (`VK_EXT_memory_budget`: the budget allows for other processes); `None` where the adapter
/// doesn't report them.
pub fn memory_budget(device: &VulkanDevice) -> Option<Vec<(u64, u64)>> {
    let instance = device.budget.as_ref()?;
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let heaps = {
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            instance
                .get_physical_device_memory_properties2(device.adapter.handle(), &mut properties)
        };
        properties.memory_properties.memory_heap_count as usize
    };
    Some(
        (0..heaps)
            .map(|heap| (budget.heap_budget[heap], budget.heap_usage[heap]))
            .collect(),
    )
}

/// What an adapter offers, for deciding whether it can make a [`VulkanDevice`] and with which
/// optional features.
struct Support {
    missing: Vec<&'static str>,
    /// Whether subgroup operations work on 8-bit, 16-bit and 64-bit types.
    subgroup_extended_types: bool,
    /// Whether mesh shaders can be counted by pipeline statistics queries.
    mesh_shader_queries: bool,
    /// `VK_KHR_pipeline_executable_properties`, with `pipelineExecutableInfo`.
    pipeline_executables: bool,
    /// `VK_KHR_shader_clock`, with both of its features.
    shader_clock: bool,
}

fn query_support(instance: &ash::Instance, adapter: &Adapter) -> Support {
    // Older devices can't be asked for Vulkan 1.2's and 1.3's features at all.
    if !adapter.supports_vulkan_1_3() {
        return Support {
            missing: vec!["Vulkan 1.3"],
            subgroup_extended_types: false,
            mesh_shader_queries: false,
            pipeline_executables: false,
            shader_clock: false,
        };
    }
    let mut missing = Vec::new();
    let mesh_extension = adapter.has_extension(ext::mesh_shader::NAME);
    if !mesh_extension {
        missing.push("VK_EXT_mesh_shader");
    }
    let executables_extension = adapter.has_extension(khr::pipeline_executable_properties::NAME);
    let clock_extension = adapter.has_extension(khr::shader_clock::NAME);

    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default();
    let mut mesh_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
    let mut executable_features =
        vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
    let mut clock_features = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
    {
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);
        if mesh_extension {
            features = features.push_next(&mut mesh_features);
        }
        if executables_extension {
            features = features.push_next(&mut executable_features);
        }
        if clock_extension {
            features = features.push_next(&mut clock_features);
        }
        unsafe { instance.get_physical_device_features2(adapter.handle(), &mut features) };
    }
    let core = adapter.features;
    let required = [
        (core.shader_int16 == vk::TRUE, "shaderInt16"),
        (core.shader_int64 == vk::TRUE, "shaderInt64"),
        (
            core.fragment_stores_and_atomics == vk::TRUE,
            "fragmentStoresAndAtomics",
        ),
        (
            vulkan_12_features.shader_float16 == vk::TRUE,
            "shaderFloat16",
        ),
        (
            vulkan_12_features.buffer_device_address == vk::TRUE,
            "bufferDeviceAddress",
        ),
        (
            vulkan_12_features.timeline_semaphore == vk::TRUE,
            "timelineSemaphore",
        ),
        (
            vulkan_12_features.shader_buffer_int64_atomics == vk::TRUE,
            "shaderBufferInt64Atomics",
        ),
        (
            vulkan_13_features.synchronization2 == vk::TRUE,
            "synchronization2",
        ),
        (
            vulkan_13_features.dynamic_rendering == vk::TRUE,
            "dynamicRendering",
        ),
        (
            vulkan_13_features.subgroup_size_control == vk::TRUE,
            "subgroupSizeControl",
        ),
        (
            vulkan_13_features.compute_full_subgroups == vk::TRUE,
            "computeFullSubgroups",
        ),
        (vulkan_13_features.maintenance4 == vk::TRUE, "maintenance4"),
        (
            vulkan_13_features.shader_demote_to_helper_invocation == vk::TRUE,
            "shaderDemoteToHelperInvocation",
        ),
        (mesh_features.task_shader == vk::TRUE, "taskShader"),
        (mesh_features.mesh_shader == vk::TRUE, "meshShader"),
    ];
    missing.extend(
        required
            .iter()
            .filter(|(present, _)| !present)
            .map(|(_, name)| *name),
    );

    let mut vulkan_11_properties = vk::PhysicalDeviceVulkan11Properties::default();
    {
        let mut properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut vulkan_11_properties);
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
        subgroup_extended_types: vulkan_12_features.shader_subgroup_extended_types == vk::TRUE,
        mesh_shader_queries: mesh_features.mesh_shader_queries == vk::TRUE,
        pipeline_executables: executable_features.pipeline_executable_info == vk::TRUE,
        shader_clock: clock_features.shader_subgroup_clock == vk::TRUE
            && clock_features.shader_device_clock == vk::TRUE,
    }
}

/// Whether a queue family has graphics, compute and timestamps.
fn suits_device(family: &QueueFamily) -> bool {
    let properties = family.properties();
    properties
        .queue_flags
        .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        && properties.timestamp_valid_bits > 0
}

/// The first queue family with graphics, compute and timestamps.
fn pick_queue_family(adapter: &Adapter) -> Option<QueueFamily> {
    adapter
        .queue_families()
        .iter()
        .find(|family| suits_device(family))
        .copied()
}
