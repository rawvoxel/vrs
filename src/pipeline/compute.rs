use super::create_shader_module;
use ash::{prelude::VkResult, vk};
use std::{ffi::CStr, slice};

/// A compute pipeline with one descriptor set and `push_constant_size` bytes of push constants.
pub fn create_compute_pipeline(
    logical_device: &ash::Device,
    spv: &[u8],
    entry: &CStr,
    desc_set_layout: vk::DescriptorSetLayout,
    push_constant_size: u32,
) -> VkResult<(vk::Pipeline, vk::PipelineLayout)> {
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(slice::from_ref(&desc_set_layout))
        .push_constant_ranges(slice::from_ref(&push_constant_range));

    unsafe {
        let pipeline_layout = logical_device.create_pipeline_layout(&pipeline_layout_info, None)?;
        let pipeline = create_shader_module(logical_device, spv).and_then(|shader_module| {
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .module(shader_module)
                .name(entry)
                .stage(vk::ShaderStageFlags::COMPUTE);
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);
            let pipelines = logical_device.create_compute_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&pipeline_info),
                None,
            );
            logical_device.destroy_shader_module(shader_module, None);
            pipelines
                .map(|pipelines| pipelines[0])
                .map_err(|(_, err)| err)
        });

        match pipeline {
            Ok(pipeline) => Ok((pipeline, pipeline_layout)),
            Err(err) => {
                logical_device.destroy_pipeline_layout(pipeline_layout, None);
                Err(err)
            }
        }
    }
}
