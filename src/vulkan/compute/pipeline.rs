use crate::vulkan::adapter::Adapter;
use ash::{khr, prelude::VkResult, vk};
use std::{ffi::CStr, fmt::Write, slice};

/// Compute pipelines for several entry points of one SPIR-V module, sharing one layout: no
/// descriptor sets, and push constants of a fixed size. Shaders reach memory through addresses.
pub struct ComputePipelines {
    layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
}

impl ComputePipelines {
    /// Nothing, until it's replaced; destroying it does nothing.
    pub fn null() -> Self {
        Self {
            layout: vk::PipelineLayout::null(),
            pipelines: Vec::new(),
        }
    }

    /// One pipeline per name in `entry_points`, in that order, from the SPIR-V `words`, made
    /// through `cache`. Every pipeline runs full subgroups, at `required_subgroup_size` when
    /// there is one, so a workgroup's width must be a multiple of the largest subgroup size.
    pub fn new(
        device: &ash::Device,
        words: &[u32],
        entry_points: &[&CStr],
        push_constant_size: u32,
        cache: &PipelineCache,
        required_subgroup_size: Option<u32>,
    ) -> VkResult<Self> {
        Self::specialized(
            device,
            words,
            entry_points,
            push_constant_size,
            cache,
            required_subgroup_size,
            &[],
        )
    }

    /// [`ComputePipelines::new`], every pipeline with the module's specialization constants
    /// `constants` set: each `(id, value)` a 32-bit constant (a `bool`'s `VkBool32`, a `uint`).
    pub fn specialized(
        device: &ash::Device,
        words: &[u32],
        entry_points: &[&CStr],
        push_constant_size: u32,
        cache: &PipelineCache,
        required_subgroup_size: Option<u32>,
        constants: &[(u32, u32)],
    ) -> VkResult<Self> {
        Self::build(
            device,
            words,
            entry_points,
            push_constant_size,
            cache.handle,
            required_subgroup_size,
            constants,
            vk::PipelineCreateFlags::empty(),
        )
    }

    /// [`ComputePipelines::new`] through no cache, each pipeline keeping what the driver compiled
    /// it into for [`ComputePipelines::statistics`] (`VK_KHR_pipeline_executable_properties`:
    /// the device must have it enabled).
    pub fn capturing_statistics(
        device: &ash::Device,
        words: &[u32],
        entry_points: &[&CStr],
        push_constant_size: u32,
        required_subgroup_size: Option<u32>,
    ) -> VkResult<Self> {
        Self::build(
            device,
            words,
            entry_points,
            push_constant_size,
            vk::PipelineCache::null(),
            required_subgroup_size,
            &[],
            vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR,
        )
    }

    /// What the driver says it compiled the `index`th pipeline into, each of its executables'
    /// statistics (registers, shared memory and the like, as the driver names them), for
    /// pipelines made by [`ComputePipelines::capturing_statistics`].
    pub fn statistics(
        &self,
        executables: &khr::pipeline_executable_properties::Device,
        index: usize,
    ) -> VkResult<Vec<PipelineStatistic>> {
        let pipeline_info = vk::PipelineInfoKHR::default().pipeline(self.pipelines[index]);
        let properties = unsafe { executables.get_pipeline_executable_properties(&pipeline_info)? };
        let mut statistics = Vec::new();
        for (executable_index, property) in properties.iter().enumerate() {
            let executable = property
                .name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            let executable_info = vk::PipelineExecutableInfoKHR::default()
                .pipeline(self.pipelines[index])
                .executable_index(executable_index as u32);
            for statistic in
                unsafe { executables.get_pipeline_executable_statistics(&executable_info)? }
            {
                let name = statistic
                    .name_as_c_str()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let value = unsafe {
                    match statistic.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                            (statistic.value.b32 == vk::TRUE).to_string()
                        }
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => {
                            statistic.value.i64.to_string()
                        }
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => {
                            statistic.value.u64.to_string()
                        }
                        _ => format!("{}", statistic.value.f64),
                    }
                };
                statistics.push(PipelineStatistic {
                    executable: executable.clone(),
                    name,
                    value,
                });
            }
        }
        Ok(statistics)
    }

    /// What the driver shows of the code it compiled the `index`th pipeline into, each of its
    /// executables' representations as text (AMD's: the ISA disassembly), for pipelines made by
    /// [`ComputePipelines::capturing_statistics`]. Empty where the driver shows none.
    pub fn representations(
        &self,
        executables: &khr::pipeline_executable_properties::Device,
        index: usize,
    ) -> VkResult<Vec<PipelineRepresentation>> {
        let pipeline_info = vk::PipelineInfoKHR::default().pipeline(self.pipelines[index]);
        let properties = unsafe { executables.get_pipeline_executable_properties(&pipeline_info)? };
        let mut representations = Vec::new();
        for executable_index in 0..properties.len() {
            let executable_info = vk::PipelineExecutableInfoKHR::default()
                .pipeline(self.pipelines[index])
                .executable_index(executable_index as u32);
            // Asked twice: first their names and sizes, then their text into buffers that big.
            let mut sized =
                unsafe { executables.get_pipeline_executable_internal_representations(&executable_info)? };
            let mut buffers: Vec<Vec<u8>> = sized.iter().map(|each| vec![0u8; each.data_size]).collect();
            for (each, buffer) in sized.iter_mut().zip(buffers.iter_mut()) {
                each.p_data = buffer.as_mut_ptr().cast();
            }
            let mut count = sized.len() as u32;
            unsafe {
                (executables.fp().get_pipeline_executable_internal_representations_khr)(
                    executables.device(),
                    &executable_info,
                    &mut count,
                    sized.as_mut_ptr(),
                )
            }
            .result()?;
            for (each, buffer) in sized.iter().zip(buffers) {
                let name = each
                    .name_as_c_str()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let length = buffer.iter().position(|&byte| byte == 0).unwrap_or(buffer.len());
                representations.push(PipelineRepresentation {
                    name,
                    text: each.is_text == vk::TRUE,
                    data: buffer[..length.min(each.data_size)].to_vec(),
                });
            }
        }
        Ok(representations)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        device: &ash::Device,
        words: &[u32],
        entry_points: &[&CStr],
        push_constant_size: u32,
        cache: vk::PipelineCache,
        required_subgroup_size: Option<u32>,
        constants: &[(u32, u32)],
        flags: vk::PipelineCreateFlags,
    ) -> VkResult<Self> {
        let entries: Vec<vk::SpecializationMapEntry> = constants
            .iter()
            .enumerate()
            .map(|(index, &(id, _))| {
                vk::SpecializationMapEntry::default()
                    .constant_id(id)
                    .offset(4 * index as u32)
                    .size(4)
            })
            .collect();
        let values: Vec<u32> = constants.iter().map(|&(_, value)| value).collect();
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(crate::vulkan::bytes::slice_as_bytes(&values));
        let module_info = vk::ShaderModuleCreateInfo::default().code(words);
        let module = unsafe { device.create_shader_module(&module_info, None)? };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size);
        let mut layout_info = vk::PipelineLayoutCreateInfo::default();
        if push_constant_size > 0 {
            layout_info = layout_info.push_constant_ranges(slice::from_ref(&push_constant_range));
        }
        let layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { device.destroy_shader_module(module, None) };
                return Err(error);
            }
        };

        let mut subgroup_sizes: Vec<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo> =
            entry_points
                .iter()
                .map(|_| {
                    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                        .required_subgroup_size(required_subgroup_size.unwrap_or(0))
                })
                .collect();
        let infos: Vec<vk::ComputePipelineCreateInfo> = entry_points
            .iter()
            .zip(subgroup_sizes.iter_mut())
            .map(|(name, subgroup_size)| {
                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(name);
                if !constants.is_empty() {
                    stage = stage.specialization_info(&specialization);
                }
                if required_subgroup_size.is_some() {
                    stage = stage.push_next(subgroup_size);
                } else {
                    stage = stage.flags(
                        vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
                            | vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE,
                    );
                }
                vk::ComputePipelineCreateInfo::default()
                    .flags(flags)
                    .stage(stage)
                    .layout(layout)
            })
            .collect();
        let created = unsafe { device.create_compute_pipelines(cache, &infos, None) };
        unsafe { device.destroy_shader_module(module, None) };
        match created {
            Ok(pipelines) => Ok(Self { layout, pipelines }),
            Err((pipelines, error)) => {
                unsafe {
                    for pipeline in pipelines {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_pipeline_layout(layout, None);
                }
                Err(error)
            }
        }
    }

    #[inline]
    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// The pipeline of the `index`th entry point.
    #[inline]
    pub fn pipeline(&self, index: usize) -> vk::Pipeline {
        self.pipelines[index]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// # Safety
    /// Once the device no longer uses them.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            for &pipeline in &self.pipelines {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// One statistic a driver gives of a pipeline it compiled ([`ComputePipelines::statistics`]): the
/// executable it's of (a compute pipeline's one, as the driver names it), its name and its value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineStatistic {
    pub executable: String,
    pub name: String,
    pub value: String,
}

/// What a driver shows of a pipeline's compiled code ([`ComputePipelines::representations`]): its
/// name, whether it's text, and its bytes (text without its terminating zero).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineRepresentation {
    pub name: String,
    pub text: bool,
    pub data: Vec<u8>,
}

/// A `VkPipelineCache`: pipelines made through it again, in this run or a later one whose cache
/// was loaded from [`PipelineCache::data`], skip most of their compiling.
pub struct PipelineCache {
    handle: vk::PipelineCache,
}

impl PipelineCache {
    /// Nothing; pipelines made through it aren't cached.
    pub fn null() -> Self {
        Self {
            handle: vk::PipelineCache::null(),
        }
    }

    /// A cache starting from `data`, which a driver ignores when it was saved by another device
    /// or driver. If the driver rejects it outright, the cache starts empty.
    pub fn new(device: &ash::Device, data: &[u8]) -> VkResult<Self> {
        let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
        let handle = match unsafe { device.create_pipeline_cache(&info, None) } {
            Ok(handle) => handle,
            Err(_) if !data.is_empty() => unsafe {
                device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?
            },
            Err(error) => return Err(error),
        };
        Ok(Self { handle })
    }

    #[inline]
    pub fn handle(&self) -> vk::PipelineCache {
        self.handle
    }

    /// What the cache holds, to save and load into a later run's cache.
    pub fn data(&self, device: &ash::Device) -> VkResult<Vec<u8>> {
        if self.handle == vk::PipelineCache::null() {
            return Ok(Vec::new());
        }
        unsafe { device.get_pipeline_cache_data(self.handle) }
    }

    /// # Safety
    /// Once no pipeline is being made through it.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_pipeline_cache(self.handle, None) };
    }
}

/// A name for `adapter`'s pipeline cache data, distinct per device, driver and cache format, for
/// a file name: a cache saved by one never meets another.
pub fn pipeline_cache_key(adapter: &Adapter) -> String {
    let properties = adapter.properties();
    let mut key = format!(
        "{:04x}-{:04x}-{:08x}-",
        properties.vendor_id, properties.device_id, properties.driver_version
    );
    for byte in properties.pipeline_cache_uuid {
        let _ = write!(key, "{byte:02x}");
    }
    key
}
