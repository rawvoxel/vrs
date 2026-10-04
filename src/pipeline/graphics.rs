//! The renderer's graphics pipeline.

use crate::{surface::Surface, vertex::Vertex};
use ash::{prelude::VkResult, vk};
use std::{ffi::CStr, slice};

pub fn create_pipeline_layout(
    render_device: &ash::Device,
    set_layouts: &[vk::DescriptorSetLayout],
) -> VkResult<vk::PipelineLayout> {
    let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
    unsafe { render_device.create_pipeline_layout(&layout_create_info, None) }
}

/// What a graphics pipeline draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelinePass {
    /// Color, and depth.
    Shaded,
    /// Depth alone, without a fragment shader or color attachment.
    DepthOnly,
    /// Color over depth already drawn, testing against it without writing it.
    ShadedOverDepth,
}

/// The renderer's pipeline drawing into `surface`, with the vertex and fragment shaders of
/// `entries`, the fragment shader unused for `PipelinePass::DepthOnly`. With `vertex_input`
/// it reads `Vertex`es from a vertex buffer, without, the vertex shader pulls its own.
pub fn create_graphics_pipeline(
    surface: &Surface,
    render_device: &ash::Device,
    shader_module: vk::ShaderModule,
    entries: (&CStr, &CStr),
    vertex_input: bool,
    pass: PipelinePass,
    pipeline_layout: vk::PipelineLayout,
) -> VkResult<vk::Pipeline> {
    let all_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(shader_module)
            .name(entries.0)
            .stage(vk::ShaderStageFlags::VERTEX),
        vk::PipelineShaderStageCreateInfo::default()
            .module(shader_module)
            .name(entries.1)
            .stage(vk::ShaderStageFlags::FRAGMENT),
    ];
    let shader_stage_infos = match pass {
        PipelinePass::DepthOnly => &all_stages[..1],
        PipelinePass::Shaded | PipelinePass::ShadedOverDepth => &all_stages[..],
    };

    let vertex_input_state_info = match vertex_input {
        true => vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&Vertex::VERTEX_ATTRIBUTE_DESCRIPTIONS)
            .vertex_binding_descriptions(&Vertex::VERTEX_BINDING_DESCRIPTIONS),
        false => vk::PipelineVertexInputStateCreateInfo::default(),
    };

    let vertex_input_assembly_state_info =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(Vertex::VERTEX_TOPOLOGY);

    let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    // Front faces wind counter-clockwise on screen, the cull mode is dynamic.
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

    let color_attachments = match pass {
        PipelinePass::DepthOnly => 0,
        PipelinePass::Shaded | PipelinePass::ShadedOverDepth => 1,
    };
    let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op(vk::LogicOp::CLEAR)
        .attachments(&color_blend_attachment_states[..color_attachments]);

    let noop_stencil_state = vk::StencilOpState::default()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::KEEP)
        .depth_fail_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::ALWAYS);

    let depth_state_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(pass != PipelinePass::ShadedOverDepth)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .front(noop_stencil_state)
        .back(noop_stencil_state)
        .max_depth_bounds(1.0);

    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::CULL_MODE,
    ]);

    let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&slice::from_ref(&surface.format)[..color_attachments])
        .depth_attachment_format(surface.depth_format);

    let graphics_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .push_next(&mut pipeline_rendering_info)
        .stages(shader_stage_infos)
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
