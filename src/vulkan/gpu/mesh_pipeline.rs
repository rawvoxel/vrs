use crate::vulkan::compute::PipelineCache;
use ash::{prelude::VkResult, vk};
use std::{ffi::CStr, slice};

/// The depth test and write of a [`MeshPipeline`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    pub test: bool,
    pub write: bool,
    pub compare: vk::CompareOp,
}

impl DepthState {
    /// No depth test and no depth writes.
    pub const NONE: Self = Self {
        test: false,
        write: false,
        compare: vk::CompareOp::ALWAYS,
    };

    /// Reverse-Z: nearer is larger, so a fragment passes when it's at least as near.
    pub const REVERSE_Z: Self = Self {
        test: true,
        write: true,
        compare: vk::CompareOp::GREATER_OR_EQUAL,
    };
}

/// One stage of a [`MeshPipeline`]: an entry point of a SPIR-V module. Stages may share a module
/// or each have their own.
#[derive(Clone, Copy, Debug)]
pub struct ShaderStage<'a> {
    pub words: &'a [u32],
    pub entry: &'a CStr,
}

/// What a [`MeshPipeline`] is made of.
#[derive(Clone, Copy, Debug)]
pub struct MeshPipelineDesc<'a> {
    pub task: Option<ShaderStage<'a>>,
    pub mesh: ShaderStage<'a>,
    pub fragment: Option<ShaderStage<'a>>,
    /// Push constants of this size, visible to every stage.
    pub push_constant_size: u32,
    /// Dynamic rendering: the color attachments' formats, in order, and the depth attachment's.
    pub color_formats: &'a [vk::Format],
    pub depth_format: Option<vk::Format>,
    pub depth: DepthState,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
}

/// A graphics pipeline of task (optional), mesh and fragment (optional) shaders, drawn with
/// `cmd_draw_mesh_tasks` under dynamic rendering. There are no vertex or index buffers and no
/// descriptor sets: shaders reach memory through addresses in the push constants. Viewport and
/// scissor are dynamic.
pub struct MeshPipeline {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl MeshPipeline {
    /// Nothing, until it's replaced; destroying it does nothing.
    pub fn null() -> Self {
        Self {
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    /// The stages every mesh pipeline's push constants are visible to.
    pub const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
        vk::ShaderStageFlags::TASK_EXT.as_raw()
            | vk::ShaderStageFlags::MESH_EXT.as_raw()
            | vk::ShaderStageFlags::FRAGMENT.as_raw(),
    );

    pub fn new(
        device: &ash::Device,
        desc: &MeshPipelineDesc,
        cache: &PipelineCache,
    ) -> VkResult<Self> {
        // One shader module per stage, destroyed once the pipeline is made.
        let stage_list: Vec<(vk::ShaderStageFlags, ShaderStage)> = [
            desc.task
                .map(|stage| (vk::ShaderStageFlags::TASK_EXT, stage)),
            Some((vk::ShaderStageFlags::MESH_EXT, desc.mesh)),
            desc.fragment
                .map(|stage| (vk::ShaderStageFlags::FRAGMENT, stage)),
        ]
        .into_iter()
        .flatten()
        .collect();
        let mut modules = Vec::new();
        let destroy_modules = |modules: &[vk::ShaderModule]| unsafe {
            for &module in modules {
                device.destroy_shader_module(module, None);
            }
        };
        for (_, stage) in &stage_list {
            let module_info = vk::ShaderModuleCreateInfo::default().code(stage.words);
            match unsafe { device.create_shader_module(&module_info, None) } {
                Ok(module) => modules.push(module),
                Err(error) => {
                    destroy_modules(&modules);
                    return Err(error);
                }
            }
        }

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(Self::PUSH_STAGES)
            .offset(0)
            .size(desc.push_constant_size);
        let mut layout_info = vk::PipelineLayoutCreateInfo::default();
        if desc.push_constant_size > 0 {
            layout_info = layout_info.push_constant_ranges(slice::from_ref(&push_constant_range));
        }
        let layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(error) => {
                destroy_modules(&modules);
                return Err(error);
            }
        };

        let stages: Vec<vk::PipelineShaderStageCreateInfo> = stage_list
            .iter()
            .zip(&modules)
            .map(|(&(flags, stage), &module)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(flags)
                    .module(module)
                    .name(stage.entry)
            })
            .collect();

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(desc.cull_mode)
            .front_face(desc.front_face)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth.test)
            .depth_write_enable(desc.depth.write)
            .depth_compare_op(desc.depth.compare);
        let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = desc
            .color_formats
            .iter()
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect();
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(desc.color_formats)
            .depth_attachment_format(desc.depth_format.unwrap_or(vk::Format::UNDEFINED));

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering);
        let created = unsafe {
            device.create_graphics_pipelines(cache.handle(), slice::from_ref(&info), None)
        };
        destroy_modules(&modules);
        match created {
            Ok(pipelines) => Ok(Self {
                layout,
                pipeline: pipelines[0],
            }),
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

    #[inline]
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// # Safety
    /// Once the device no longer uses it.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
