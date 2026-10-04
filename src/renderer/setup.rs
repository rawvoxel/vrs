//! Creating the device and the graphics pipeline.

use crate::{
    App, adapter::Adapter, buffer::Buffer, queue::QueueFamily, surface::Surface,
    uniforms::MvpMatrices, vertex::Vertex,
};
use ash::{khr, prelude::VkResult, vk};
use std::slice;

pub(super) fn create_render_device(
    app: &App,
    adapter: &Adapter,
    queue_family: &QueueFamily,
    multi_draw_indirect: bool,
    draw_indirect_count: bool,
) -> VkResult<ash::Device> {
    let queue_priorities = [1.0];
    let device_queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family.index)
        .queue_priorities(&queue_priorities);

    let extension_names_raw = [
        khr::swapchain::NAME.as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        khr::portability_subset::NAME.as_ptr(),
    ];

    // NOTE: Keep in mind proper ordering when adding features.
    let features = vk::PhysicalDeviceFeatures::default().multi_draw_indirect(multi_draw_indirect);
    let mut synchronization2 =
        vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    let mut dynamic_rendering =
        vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
    let mut vulkan_12_features =
        vk::PhysicalDeviceVulkan12Features::default().draw_indirect_count(draw_indirect_count);

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(slice::from_ref(&device_queue_info))
        .enabled_extension_names(&extension_names_raw)
        .enabled_features(&features)
        .push_next(&mut synchronization2)
        .push_next(&mut dynamic_rendering)
        .push_next(&mut vulkan_12_features);

    unsafe { app.instance.create_device(adapter.handle, &device_info, None) }
}

pub(super) fn create_pipeline_layout(
    render_device: &ash::Device,
    set_layouts: &[vk::DescriptorSetLayout],
) -> VkResult<vk::PipelineLayout> {
    let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
    unsafe { render_device.create_pipeline_layout(&layout_create_info, None) }
}

pub(super) fn create_graphics_pipeline(
    surface: &Surface,
    render_device: &ash::Device,
    shader_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
) -> VkResult<vk::Pipeline> {
    let shader_stage_infos = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(shader_module)
            .name(c"vertMain")
            .stage(vk::ShaderStageFlags::VERTEX),
        vk::PipelineShaderStageCreateInfo::default()
            .module(shader_module)
            .name(c"fragMain")
            .stage(vk::ShaderStageFlags::FRAGMENT),
    ];

    let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&Vertex::VERTEX_ATTRIBUTE_DESCRIPTIONS)
        .vertex_binding_descriptions(&Vertex::VERTEX_BINDING_DESCRIPTIONS);

    let vertex_input_assembly_state_info =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(Vertex::VERTEX_TOPOLOGY);

    let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization_info = vk::PipelineRasterizationStateCreateInfo::default()
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0)
        .polygon_mode(vk::PolygonMode::FILL);

    let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::SRC_COLOR)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_DST_COLOR)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ZERO)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];

    let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op(vk::LogicOp::CLEAR)
        .attachments(&color_blend_attachment_states);

    let noop_stencil_state = vk::StencilOpState::default()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::KEEP)
        .depth_fail_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::ALWAYS);

    let depth_state_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .front(noop_stencil_state)
        .back(noop_stencil_state)
        .max_depth_bounds(1.0);

    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(slice::from_ref(&surface.format))
        .depth_attachment_format(surface.depth_format);

    let graphics_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .push_next(&mut pipeline_rendering_info)
        .stages(&shader_stage_infos)
        .vertex_input_state(&vertex_input_state_info)
        .input_assembly_state(&vertex_input_assembly_state_info)
        .viewport_state(&viewport_state_info)
        .rasterization_state(&rasterization_info)
        .multisample_state(&multisample_state_info)
        .color_blend_state(&color_blend_state_info)
        .depth_stencil_state(&depth_state_info)
        .dynamic_state(&dynamic_state_info)
        .layout(pipeline_layout);

    unsafe {
        render_device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&graphics_pipeline_info),
                None,
            )
            .map(|pipelines| pipelines[0])
            .map_err(|(_, err)| err)
    }
}

/// A mapped `HOST_VISIBLE` buffer the size of `MvpMatrices`.
pub(super) fn create_uniform_buffer(
    adapter: &Adapter,
    render_device: &ash::Device,
) -> VkResult<Buffer> {
    let info = vk::BufferCreateInfo::default()
        .size(size_of::<MvpMatrices>() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let mut buffer = Buffer::new(adapter, render_device, info, flags, 0)?;
    if let Err(err) = buffer.mem_map(render_device) {
        unsafe { buffer.destroy(render_device) };
        return Err(err);
    }
    Ok(buffer)
}

/// Points each frame's descriptor set at that frame's uniform buffer.
pub(super) fn write_uniform_descriptor_sets(
    render_device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    uniform_buffers: &[Buffer],
) {
    let buffer_infos: Vec<vk::DescriptorBufferInfo> = uniform_buffers
        .iter()
        .map(|buffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer.handle)
                .range(size_of::<MvpMatrices>() as vk::DeviceSize)
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = descriptor_sets
        .iter()
        .zip(buffer_infos.iter())
        .map(|(&set, buffer_info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(buffer_info))
        })
        .collect();

    unsafe { render_device.update_descriptor_sets(&writes, &[]) }
}
