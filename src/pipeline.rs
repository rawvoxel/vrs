use ash::{prelude::VkResult, util, vk};
use std::{ffi::CStr, io::Cursor, slice};

#[inline]
pub fn create_shader_module(
    logical_device: &ash::Device,
    spv_bytes: &[u8],
) -> VkResult<vk::ShaderModule> {
    let shader_code =
        util::read_spv(&mut Cursor::new(spv_bytes)).map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    unsafe { logical_device.create_shader_module(&shader_info, None) }
}

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
            pipelines.map(|pipelines| pipelines[0]).map_err(|(_, err)| err)
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

/// Plain data as bytes, for push constants.
#[inline]
pub fn as_bytes<T: Copy>(data: &T) -> &[u8] {
    // SAFETY: Only used for plain data without implicit padding.
    unsafe { slice::from_raw_parts((data as *const T).cast(), size_of::<T>()) }
}
