//! The renderer's graphics pipeline.

use crate::{surface::Surface, vertex::Vertex};
use ash::{prelude::VkResult, vk};
use std::slice;

pub fn create_pipeline_layout(
    render_device: &ash::Device,
    set_layouts: &[vk::DescriptorSetLayout],
) -> VkResult<vk::PipelineLayout> {
    let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
    unsafe { render_device.create_pipeline_layout(&layout_create_info, None) }
}

pub fn create_graphics_pipeline(
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
