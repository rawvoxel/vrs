use crate::vulkan::adapter::Adapter;
use ash::{prelude::VkResult, vk};
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
                if required_subgroup_size.is_some() {
                    stage = stage.push_next(subgroup_size);
                } else {
                    stage = stage.flags(
                        vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
                            | vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE,
                    );
                }
                vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)
            })
            .collect();
        let created = unsafe { device.create_compute_pipelines(cache.handle, &infos, None) };
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
