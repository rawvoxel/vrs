use crate::{
    App,
    vulkan::{
        adapter::Adapter,
        commands,
        compute::SubgroupProperties,
        queue::{Queue, QueueFamily},
    },
};
use ash::{ext, khr, prelude::VkResult, vk};
use std::{ffi::CStr, slice, sync::Mutex};

mod shared;

/// What a [`GpuContext`] is made for, beyond compute and mesh-shader rasterization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuOptions {
    /// Present to the app's window surface: the queue must be able to, and the swapchain
    /// extension is enabled. The app must have a window.
    pub present: bool,
}

/// The mesh-shader limits of a device, for sizing task and mesh workgroups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshShaderLimits {
    pub max_task_workgroup_invocations: u32,
    pub max_task_workgroup_total_count: u32,
    pub max_task_payload_size: u32,
    pub max_mesh_workgroup_invocations: u32,
    pub max_mesh_workgroup_total_count: u32,
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    pub max_preferred_task_workgroup_invocations: u32,
    pub max_preferred_mesh_workgroup_invocations: u32,
}

/// A device for GPU-driven rendering on any adapter that has it: compute, and rasterization
/// through task and mesh shaders, on one queue, with buffers reached by address.
///
/// It has everything a [`ComputeContext`](crate::vulkan::compute::ComputeContext) has (buffer
/// device addresses, 64-bit integers, timeline semaphores, synchronization2, subgroup size
/// control with full subgroups), plus task and mesh shaders, 64-bit buffer atomics, stores and
/// atomics in fragment shaders, dynamic rendering and maintenance4, and optionally a swapchain.
/// Adapters without them are refused with [`vk::Result::ERROR_FEATURE_NOT_PRESENT`];
/// [`GpuContext::missing_features`] says which.
///
/// The queue is shared: whoever submits goes through [`GpuContext::with_queue`], so submits from
/// different owners of the context never race.
pub struct GpuContext {
    adapter: Adapter,
    device: ash::Device,
    mesh_shader: ext::mesh_shader::Device,
    queue_family: QueueFamily,
    queue: Mutex<Queue>,
    /// For [`GpuContext::submit_and_wait`].
    immediate: Mutex<Immediate>,
    subgroup: SubgroupProperties,
    mesh_limits: MeshShaderLimits,
    options: GpuOptions,
    /// The `VK_KHR_pipeline_executable_properties` commands, where the adapter has them.
    pipeline_executables: Option<khr::pipeline_executable_properties::Device>,
    /// Whether shaders may read the subgroup's and the device's clocks (`VK_KHR_shader_clock`).
    shader_clock: bool,
    /// The instance, kept for [`GpuContext::memory_budget`] where the adapter reports budgets.
    budget: Option<ash::Instance>,
}

/// The instance to read `physical_device`'s memory budgets through, if it reports them
/// (`VK_EXT_memory_budget`).
pub(super) fn budget_instance(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<ash::Instance> {
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(physical_device) }.ok()?;
    extensions
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(ext::memory_budget::NAME))
        .then(|| instance.clone())
}

/// A command buffer and fence for one-off submissions.
struct Immediate {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

/// What an adapter offers, for deciding whether it can make a [`GpuContext`].
struct Support {
    missing: Vec<&'static str>,
    subgroup: SubgroupProperties,
    subgroup_extended_types: bool,
    mesh_shader_queries: bool,
    mesh_limits: MeshShaderLimits,
}

impl GpuContext {
    /// Makes the device on `adapter`. `app` must have been made with Vulkan 1.3 or later, and
    /// with a window when `options.present` is set.
    pub fn new(app: &App, adapter: &Adapter, options: GpuOptions) -> VkResult<Self> {
        let support = query_support(app.instance(), adapter);
        if !support.missing.is_empty() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let queue_family = pick_queue_family(app, adapter, options)
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family.index())
            .queue_priorities(&queue_priorities);
        let mut extensions = vec![ext::mesh_shader::NAME.as_ptr()];
        if options.present {
            extensions.push(khr::swapchain::NAME.as_ptr());
        }
        let pipeline_executables = pipeline_executables_supported(app.instance(), adapter);
        if pipeline_executables {
            extensions.push(khr::pipeline_executable_properties::NAME.as_ptr());
        }
        let shader_clock = shader_clock_supported(app.instance(), adapter);
        if shader_clock {
            extensions.push(khr::shader_clock::NAME.as_ptr());
        }
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .fragment_stores_and_atomics(true);
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .shader_buffer_int64_atomics(true)
            .shader_subgroup_extended_types(support.subgroup_extended_types);
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true)
            .maintenance4(true);
        let mut mesh_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .task_shader(true)
            .mesh_shader(true)
            .mesh_shader_queries(support.mesh_shader_queries);
        let mut executable_features =
            vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
                .pipeline_executable_info(true);
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(slice::from_ref(&queue_info))
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features)
            .push_next(&mut mesh_features);
        if pipeline_executables {
            device_info = device_info.push_next(&mut executable_features);
        }
        let mut clock_features = vk::PhysicalDeviceShaderClockFeaturesKHR::default()
            .shader_subgroup_clock(true)
            .shader_device_clock(true);
        if shader_clock {
            device_info = device_info.push_next(&mut clock_features);
        }
        let device = unsafe {
            app.instance()
                .create_device(adapter.handle(), &device_info, None)?
        };
        let immediate = match create_immediate(&device, &queue_family) {
            Ok(immediate) => immediate,
            Err(error) => {
                unsafe { device.destroy_device(None) };
                return Err(error);
            }
        };
        let pipeline_executables = pipeline_executables
            .then(|| khr::pipeline_executable_properties::Device::new(app.instance(), &device));
        Ok(Self {
            adapter: adapter.clone(),
            mesh_shader: ext::mesh_shader::Device::new(app.instance(), &device),
            queue: Mutex::new(Queue::new(&queue_family, &device, 0)),
            device,
            queue_family,
            immediate: Mutex::new(immediate),
            subgroup: support.subgroup,
            mesh_limits: support.mesh_limits,
            options,
            pipeline_executables,
            shader_clock,
            budget: budget_instance(app.instance(), adapter.handle()),
        })
    }

    /// Why `adapter` can't make a [`GpuContext`] with `options`, or nothing when it can.
    pub fn missing_features(
        app: &App,
        adapter: &Adapter,
        options: GpuOptions,
    ) -> Vec<&'static str> {
        let mut missing = query_support(app.instance(), adapter).missing;
        if pick_queue_family(app, adapter, options).is_none() {
            missing.push(if options.present {
                "a graphics and compute queue that can present to the window"
            } else {
                "a graphics and compute queue"
            });
        }
        if options.present && !has_extension(app.instance(), adapter, khr::swapchain::NAME) {
            missing.push("VK_KHR_swapchain");
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

    /// The `VK_KHR_pipeline_executable_properties` commands, where the adapter has them: what
    /// the driver compiled a pipeline into (its registers, its shared memory), for pipelines made
    /// to capture it ([`ComputePipelines::capturing_statistics`](crate::vulkan::compute::ComputePipelines::capturing_statistics)).
    #[inline]
    pub fn pipeline_executables(&self) -> Option<&khr::pipeline_executable_properties::Device> {
        self.pipeline_executables.as_ref()
    }

    /// Whether shaders may read the subgroup's and the device's clocks (`VK_KHR_shader_clock`,
    /// enabled where the adapter has it): for instrumented builds that time their own phases.
    #[inline]
    pub fn shader_clock(&self) -> bool {
        self.shader_clock
    }

    /// The `VK_EXT_mesh_shader` commands: `cmd_draw_mesh_tasks` and its indirect forms.
    #[inline]
    pub fn mesh_shader(&self) -> &ext::mesh_shader::Device {
        &self.mesh_shader
    }

    #[inline]
    pub fn queue_family(&self) -> QueueFamily {
        self.queue_family
    }

    #[inline]
    pub fn subgroup(&self) -> SubgroupProperties {
        self.subgroup
    }

    /// Each memory heap's budget and what this process uses of it, in bytes, as the driver says
    /// now (`VK_EXT_memory_budget`: the budget allows for other processes); `None` where the
    /// adapter doesn't report them.
    pub fn memory_budget(&self) -> Option<Vec<(u64, u64)>> {
        let instance = self.budget.as_ref()?;
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let heaps = {
            let mut properties =
                vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                instance
                    .get_physical_device_memory_properties2(self.adapter.handle(), &mut properties)
            };
            properties.memory_properties.memory_heap_count as usize
        };
        Some(
            (0..heaps)
                .map(|heap| (budget.heap_budget[heap], budget.heap_usage[heap]))
                .collect(),
        )
    }

    #[inline]
    pub fn mesh_limits(&self) -> MeshShaderLimits {
        self.mesh_limits
    }

    #[inline]
    pub fn options(&self) -> GpuOptions {
        self.options
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

    /// Runs `use_queue` with the queue, which no one else submits to meanwhile.
    pub fn with_queue<R>(&self, use_queue: impl FnOnce(vk::Queue) -> R) -> R {
        let queue = self
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        use_queue(queue.handle())
    }

    /// Records commands with `record`, submits them and waits until the device has run them.
    pub fn submit_and_wait<F: FnOnce(vk::CommandBuffer)>(&self, record: F) -> VkResult<()> {
        let immediate = self
            .immediate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .reset_fences(slice::from_ref(&immediate.fence))?;
            self.device
                .begin_command_buffer(immediate.command_buffer, &begin_info)?;
            record(immediate.command_buffer);
            self.device.end_command_buffer(immediate.command_buffer)?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(slice::from_ref(&immediate.command_buffer));
            self.with_queue(|queue| {
                self.device
                    .queue_submit(queue, slice::from_ref(&submit), immediate.fence)
            })?;
            self.device
                .wait_for_fences(slice::from_ref(&immediate.fence), true, u64::MAX)
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        let immediate = self
            .immediate
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            // Nothing can be destroyed while the device might still use it.
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(immediate.fence, None);
            self.device
                .destroy_command_pool(immediate.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

fn create_immediate(device: &ash::Device, queue_family: &QueueFamily) -> VkResult<Immediate> {
    let command_pool = commands::create_command_pool(device, queue_family)?;
    let command_buffer =
        commands::create_command_buffer(device, command_pool, vk::CommandBufferLevel::PRIMARY);
    let fence = command_buffer.and_then(|command_buffer| {
        // Made unsignaled: `submit_and_wait` resets it before every use anyway.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        Ok((command_buffer, fence))
    });
    match fence {
        Ok((command_buffer, fence)) => Ok(Immediate {
            command_pool,
            command_buffer,
            fence,
        }),
        Err(error) => {
            unsafe { device.destroy_command_pool(command_pool, None) };
            Err(error)
        }
    }
}

/// Whether `adapter`'s shaders can read the subgroup's and the device's clocks
/// (`VK_KHR_shader_clock`, both of its features).
fn shader_clock_supported(instance: &ash::Instance, adapter: &Adapter) -> bool {
    if !has_extension(instance, adapter, khr::shader_clock::NAME) {
        return false;
    }
    let mut clock_features = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
    {
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut clock_features);
        unsafe { instance.get_physical_device_features2(adapter.handle(), &mut features) };
    }
    clock_features.shader_subgroup_clock == vk::TRUE && clock_features.shader_device_clock == vk::TRUE
}

/// Whether `adapter`'s driver tells what it compiled a pipeline into
/// (`VK_KHR_pipeline_executable_properties`, with `pipelineExecutableInfo`).
fn pipeline_executables_supported(instance: &ash::Instance, adapter: &Adapter) -> bool {
    if !has_extension(instance, adapter, khr::pipeline_executable_properties::NAME) {
        return false;
    }
    let mut executable_features =
        vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
    {
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut executable_features);
        unsafe { instance.get_physical_device_features2(adapter.handle(), &mut features) };
    }
    executable_features.pipeline_executable_info == vk::TRUE
}

fn has_extension(instance: &ash::Instance, adapter: &Adapter, name: &CStr) -> bool {
    unsafe {
        instance
            .enumerate_device_extension_properties(adapter.handle())
            .map(|extensions| {
                extensions
                    .iter()
                    .any(|extension| extension.extension_name_as_c_str() == Ok(name))
            })
            .unwrap_or(false)
    }
}

fn query_support(instance: &ash::Instance, adapter: &Adapter) -> Support {
    let mut missing = Vec::new();

    let api_version = adapter.properties().api_version;
    if (
        vk::api_version_major(api_version),
        vk::api_version_minor(api_version),
    ) < (1, 3)
    {
        missing.push("Vulkan 1.3");
    }
    let mesh_extension = has_extension(instance, adapter, ext::mesh_shader::NAME);
    if !mesh_extension {
        missing.push("VK_EXT_mesh_shader");
    }

    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default();
    let mut mesh_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
    let core = {
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);
        if mesh_extension {
            features = features.push_next(&mut mesh_features);
        }
        unsafe { instance.get_physical_device_features2(adapter.handle(), &mut features) };
        features.features
    };
    let required = [
        (core.shader_int64 == vk::TRUE, "shaderInt64"),
        (
            core.fragment_stores_and_atomics == vk::TRUE,
            "fragmentStoresAndAtomics",
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
    let mut vulkan_13_properties = vk::PhysicalDeviceVulkan13Properties::default();
    let mut mesh_properties = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
    {
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vulkan_11_properties)
            .push_next(&mut vulkan_13_properties);
        if mesh_extension {
            properties = properties.push_next(&mut mesh_properties);
        }
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
        mesh_shader_queries: mesh_features.mesh_shader_queries == vk::TRUE,
        mesh_limits: MeshShaderLimits {
            max_task_workgroup_invocations: mesh_properties.max_task_work_group_invocations,
            max_task_workgroup_total_count: mesh_properties.max_task_work_group_total_count,
            max_task_payload_size: mesh_properties.max_task_payload_size,
            max_mesh_workgroup_invocations: mesh_properties.max_mesh_work_group_invocations,
            max_mesh_workgroup_total_count: mesh_properties.max_mesh_work_group_total_count,
            max_mesh_output_vertices: mesh_properties.max_mesh_output_vertices,
            max_mesh_output_primitives: mesh_properties.max_mesh_output_primitives,
            max_preferred_task_workgroup_invocations: mesh_properties
                .max_preferred_task_work_group_invocations,
            max_preferred_mesh_workgroup_invocations: mesh_properties
                .max_preferred_mesh_work_group_invocations,
        },
    }
}

/// A queue family with graphics, compute and timestamps, that can present to the app's window
/// when `options.present` asks for it.
fn pick_queue_family(app: &App, adapter: &Adapter, options: GpuOptions) -> Option<QueueFamily> {
    let presents = |family: &QueueFamily| {
        if !options.present {
            return true;
        }
        let Some(surface) = app.surface() else {
            return false;
        };
        unsafe {
            app.surface_instance
                .get_physical_device_surface_support(adapter.handle(), family.index(), surface)
                .unwrap_or(false)
        }
    };
    adapter
        .queue_families()
        .iter()
        .find(|family| {
            let properties = family.properties();
            properties
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                && properties.timestamp_valid_bits > 0
                && presents(family)
        })
        .copied()
}
