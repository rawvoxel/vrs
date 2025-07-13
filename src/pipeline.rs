use crate::context::{Context, find_memorytype_index, record_submit_commandbuffer};
use anyhow::Result;
use ash::prelude::*;
use ash::{util::*, vk};
use std::{default::Default, io::Cursor, mem::offset_of};

#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub col: [f32; 4],
}

const VERT_SPV: &[u8] = include_bytes!("../assets/shader/triangle/vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("../assets/shader/triangle/frag.spv");

pub struct Pipeline {
    pub viewports: Vec<vk::Viewport>,
    pub renderpass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub graphics_pipeline: vk::Pipeline,
}

#[rustfmt::skip]
impl Pipeline {
    pub fn new(context: &mut Context) -> VkResult<Self> {
        let viewports = vec![create_viewport(context)];
        let renderpass = create_renderpass(context)?;
        let framebuffers = create_framebuffers(context, &renderpass)?;
        let vert_shader_module = create_shader_module(context, VERT_SPV)?;
        let frag_shader_module = create_shader_module(context, FRAG_SPV)?;
        let pipeline_layout = create_pipeline_layout(context)?;
        let graphics_pipeline = create_graphics_pipeline(
            context,
            &viewports,
            &renderpass,
            &vert_shader_module,
            &frag_shader_module,
            &pipeline_layout,
        )?;

        unsafe {
            context.device.destroy_pipeline_layout(pipeline_layout, None);
            context.device.destroy_shader_module(frag_shader_module, None);
            context.device.destroy_shader_module(vert_shader_module, None);
        }

        Ok(Self {
            viewports,
            renderpass,
            framebuffers,
            graphics_pipeline,
        })
    }

    #[rustfmt::skip]
    pub fn destroy(&self, context: &Context) {
        unsafe {
            context.device.device_wait_idle().unwrap();
            context.device.destroy_pipeline(self.graphics_pipeline, None);
            for framebuffer in self.framebuffers.iter() {
                context.device.destroy_framebuffer(*framebuffer, None);
            }
            context.device.destroy_render_pass(self.renderpass, None);
        }
    }
}

#[inline]
pub fn create_viewport(context: &Context) -> vk::Viewport {
    vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: context.surface_resolution.width as f32,
        height: context.surface_resolution.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

#[inline]
pub fn create_renderpass(context: &mut Context) -> VkResult<vk::RenderPass> {
    let renderpass_attachments = [
        vk::AttachmentDescription::default()
            .format(context.surface_format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        vk::AttachmentDescription::default()
            .format(vk::Format::D16_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];

    let color_attachment_ref = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let depth_attachment_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass_dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)];

    let subpass = vk::SubpassDescription::default()
        .color_attachments(&color_attachment_ref)
        .depth_stencil_attachment(&depth_attachment_ref)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&renderpass_attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&subpass_dependencies);

    Ok(unsafe { context.device.create_render_pass(&create_info, None)? })
}

#[inline]
pub fn create_framebuffers(
    context: &Context,
    renderpass: &vk::RenderPass,
) -> VkResult<Vec<vk::Framebuffer>> {
    Ok(context
        .present_image_views
        .iter()
        .map(|&present_image_view| {
            let attachments = [present_image_view, context.depth_image_view];
            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(*renderpass)
                .attachments(&attachments)
                .width(context.surface_resolution.width)
                .height(context.surface_resolution.height)
                .layers(1);

            unsafe {
                context
                    .device
                    .create_framebuffer(&create_info, None)
                    .expect("Could not create vk::Framebuffer.")
            }
        })
        .collect())
}

#[inline]
pub fn create_shader_module(context: &Context, spv_bytes: &[u8]) -> VkResult<vk::ShaderModule> {
    let mut spv_file = Cursor::new(&spv_bytes);
    let shader_code = read_spv(&mut spv_file).expect("Failed to read shader spv file.");
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    unsafe { Ok(context.device.create_shader_module(&shader_info, None)?) }
}

#[inline]
#[rustfmt::skip]
pub fn create_pipeline_layout(context: &Context) -> VkResult<vk::PipelineLayout> {
    let layout_create_info = vk::PipelineLayoutCreateInfo::default();
    unsafe { Ok(context.device.create_pipeline_layout(&layout_create_info, None)?) }
}

#[inline]
pub fn create_graphics_pipeline(
    context: &Context,
    viewports: &Vec<vk::Viewport>,
    renderpass: &vk::RenderPass,
    vert_shader_module: &vk::ShaderModule,
    frag_shader_module: &vk::ShaderModule,
    pipeline_layout: &vk::PipelineLayout,
) -> VkResult<vk::Pipeline> {
    let shader_entry_name = c"main";
    let shader_stage_create_infos = [
        vk::PipelineShaderStageCreateInfo {
            module: *vert_shader_module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            module: *frag_shader_module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let vertex_input_attribute_descriptions = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: offset_of!(Vertex, pos) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: offset_of!(Vertex, col) as u32,
        },
    ];
    let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
        .vertex_binding_descriptions(&vertex_input_binding_descriptions);
    let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let scissors = [context.surface_resolution.into()];
    let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
        .scissors(&scissors)
        .viewports(&viewports);

    let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        ..Default::default()
    };
    let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    let noop_stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        ..Default::default()
    };
    let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable: 1,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        front: noop_stencil_state,
        back: noop_stencil_state,
        max_depth_bounds: 1.0,
        ..Default::default()
    };
    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: 0,
        src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ZERO,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    }];
    let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op(vk::LogicOp::CLEAR)
        .attachments(&color_blend_attachment_states);

    let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

    let graphics_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stage_create_infos)
        .vertex_input_state(&vertex_input_state_info)
        .input_assembly_state(&vertex_input_assembly_state_info)
        .viewport_state(&viewport_state_info)
        .rasterization_state(&rasterization_info)
        .multisample_state(&multisample_state_info)
        .depth_stencil_state(&depth_state_info)
        .color_blend_state(&color_blend_state_info)
        .dynamic_state(&dynamic_state_info)
        .layout(*pipeline_layout)
        .render_pass(*renderpass);

    unsafe {
        let graphics_pipelines = context
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_pipeline_info], None)
            .expect("Could not create vk::Pipeline.");

        Ok(graphics_pipelines[0])
    }
}

#[inline]
pub fn create_buffer<T: Copy>(
    context: &Context,
    buffer_info: &vk::BufferCreateInfo,
    data: &[T],
) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
    unsafe {
        let buffer = context.device.create_buffer(&buffer_info, None)?;

        let memory_req = context.device.get_buffer_memory_requirements(buffer);
        let memory_index = find_memorytype_index(
            &memory_req,
            &context.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for buffer.");

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_req.size)
            .memory_type_index(memory_index);

        let memory = context.device.allocate_memory(&allocate_info, None)?;

        // NOTE: What is this even for? Never used in example.
        let buffer_ptr =
            context
                .device
                .map_memory(memory, 0, memory_req.size, vk::MemoryMapFlags::empty())?;

        let mut align = Align::new(buffer_ptr, align_of::<T>() as u64, memory_req.size);
        align.copy_from_slice(&data);

        context.device.unmap_memory(memory);
        context.device.bind_buffer_memory(buffer, memory, 0)?;

        Ok((buffer, memory))
    }
}

#[inline]
pub fn create_index_buffer(
    context: &Context,
    data: &[u32],
) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size_of_val(&data) as u64)
        .usage(vk::BufferUsageFlags::INDEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    Ok(create_buffer(context, &buffer_info, data)?)
}

#[inline]
pub fn create_vertex_buffer(
    context: &Context,
    data: &[Vertex],
) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(3 * size_of::<Vertex>() as u64)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    Ok(create_buffer(context, &buffer_info, data)?)
}

pub fn test_submit(
    context: &Context,
    pipeline: &Pipeline,
    index_count: u32,
    index_buffer: &vk::Buffer,
    vertex_buffer: &vk::Buffer,
) -> Result<()> {
    let Pipeline {
        viewports,
        renderpass,
        framebuffers,
        graphics_pipeline,
        ..
    } = pipeline;

    let clear_values = [
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];

    unsafe {
        let (present_index, _suboptimal) = context.swapchain_device.acquire_next_image(
            context.swapchain,
            u64::MAX,
            context.present_complete_semaphore,
            vk::Fence::null(),
        )?;

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(*renderpass)
            .framebuffer(framebuffers[present_index as usize])
            .render_area(context.surface_resolution.into())
            .clear_values(&clear_values);

        let scissors = [context.surface_resolution.into()];

        record_submit_commandbuffer(
            &context.device,
            context.draw_command_buffer,
            context.draw_commands_fence,
            context.present_queue,
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[context.present_complete_semaphore],
            &[context.render_complete_semaphore],
            |device, draw_command_buffer| {
                device.cmd_begin_render_pass(
                    draw_command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    draw_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    *graphics_pipeline,
                );
                device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                device.cmd_bind_vertex_buffers(draw_command_buffer, 0, &[*vertex_buffer], &[0]);
                device.cmd_bind_index_buffer(
                    draw_command_buffer,
                    *index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(draw_command_buffer, index_count, 1, 0, 0, 1);
                device.cmd_end_render_pass(draw_command_buffer);
            },
        )?;

        let wait_semaphores = [context.render_complete_semaphore];
        let swapchains = [context.swapchain];
        let image_indices = [present_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        context
            .swapchain_device
            .queue_present(context.present_queue, &present_info)?;
    }

    Ok(())
}
