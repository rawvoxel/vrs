use crate::{
    buffer::Buffer,
    commands::Commands,
    context::{Context, record_submit_commandbuffer},
    queue::PresentQueue,
    surface::SurfaceInfo,
    swapchain::Swapchain,
    vertex::{Index, Vertex},
};
use anyhow::Result;
use ash::{khr, prelude::VkResult, util, vk};
use std::{default::Default, io::Cursor};

const VERT_SPV: &[u8] = include_bytes!("../assets/shader/triangle/vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("../assets/shader/triangle/frag.spv");

#[derive(Clone)]
pub struct GraphicsPipeline {
    pub surface_info: SurfaceInfo,
    pub scissors: Vec<vk::Rect2D>,
    pub viewports: Vec<vk::Viewport>,

    pub present_queue: PresentQueue,

    pub swapchain_device: khr::swapchain::Device,
    pub swapchain: Swapchain,

    pub renderpass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub graphics_pipeline: vk::Pipeline,

    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,

    pub init_commands: Commands,
    pub draw_commands: Commands,
}

impl GraphicsPipeline {
    pub fn new(context: &Context, window_width: u32, window_height: u32) -> VkResult<Self> {
        let surface_info = SurfaceInfo::new(
            &context.surface_instance,
            context.surface.inner,
            context.physical_device,
            window_width,
            window_height,
        )?;

        let scissors: Vec<vk::Rect2D> = vec![surface_info.extent.into()];
        let viewports: Vec<vk::Viewport> = vec![
            vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(surface_info.extent.width as f32)
                .height(surface_info.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0),
        ];

        let present_queue = PresentQueue::new(&context)?;
        let swapchain_device =
            khr::swapchain::Device::new(&context.vk_instance, &context.vk_device);
        let swapchain = Swapchain::new(
            &context,
            &swapchain_device,
            surface_info,
            present_queue.mode,
            None,
        )?;

        let renderpass = create_renderpass(&context.vk_device, surface_info.format)?;
        let framebuffers = create_framebuffers(
            &context,
            surface_info,
            swapchain.present_image_views.clone(),
            swapchain.depth_image_view,
            renderpass,
        )?;

        let vert_shader_module = create_shader_module(&context.vk_device, VERT_SPV)?;
        let frag_shader_module = create_shader_module(&context.vk_device, FRAG_SPV)?;

        let pipeline_layout = create_pipeline_layout(&context.vk_device)?;
        let graphics_pipeline = create_graphics_pipeline(
            &context.vk_device,
            &scissors,
            &viewports,
            renderpass,
            vert_shader_module,
            frag_shader_module,
            pipeline_layout,
        )?;

        let vertex_buffer = create_vertex_buffer(
            &context.vk_device,
            context.physical_device_memory_properties,
        )?;
        let index_buffer = create_index_buffer(
            &context.vk_device,
            context.physical_device_memory_properties,
        )?;

        let init_commands = Commands::new(&context.vk_device, context.command_pool)?;
        let draw_commands = Commands::new(&context.vk_device, context.command_pool)?;

        // NOTE: WHAT IS THIS
        let image_memory_barrier = vk::ImageMemoryBarrier::default()
            .image(swapchain.depth_image)
            .dst_access_mask(
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .layer_count(1)
                    .level_count(1),
            );

        record_submit_commandbuffer(
            &context.vk_device,
            init_commands.buffers[0],
            init_commands.fences[0],
            present_queue.inner,
            &[],
            &[],
            &[],
            |init_command_buffer| unsafe {
                context.vk_device.cmd_pipeline_barrier(
                    init_command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_memory_barrier],
                );
            },
        )?;

        unsafe {
            context
                .vk_device
                .destroy_shader_module(vert_shader_module, None);
            context
                .vk_device
                .destroy_shader_module(frag_shader_module, None);
            context
                .vk_device
                .destroy_pipeline_layout(pipeline_layout, None);
        }

        Ok(Self {
            surface_info,
            scissors,
            viewports,

            present_queue,

            swapchain_device,
            swapchain,

            renderpass,
            framebuffers,
            graphics_pipeline,

            vertex_buffer,
            index_buffer,

            init_commands,
            draw_commands,
        })
    }

    pub fn destroy(&self, vk_device: &ash::Device) {
        unsafe {
            vk_device.device_wait_idle().unwrap();

            self.present_queue.destroy(&vk_device);
            self.swapchain.destroy(&vk_device, &self.swapchain_device);

            vk_device.destroy_render_pass(self.renderpass, None);
            for framebuffer in self.framebuffers.iter() {
                vk_device.destroy_framebuffer(*framebuffer, None);
            }
            vk_device.destroy_pipeline(self.graphics_pipeline, None);

            self.index_buffer.destroy(&vk_device);
            self.vertex_buffer.destroy(&vk_device);
            self.init_commands.destroy(&vk_device);
            self.draw_commands.destroy(&vk_device);
        }
    }
}

#[inline]
pub fn create_renderpass(
    vk_device: &ash::Device,
    surface_format: vk::SurfaceFormatKHR,
) -> VkResult<vk::RenderPass> {
    let renderpass_attachments = [
        vk::AttachmentDescription::default()
            .format(surface_format.format)
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

    let renderpass_info = vk::RenderPassCreateInfo::default()
        .attachments(&renderpass_attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&subpass_dependencies);

    unsafe { vk_device.create_render_pass(&renderpass_info, None) }
}

#[inline]
pub fn create_framebuffers(
    context: &Context,
    surface_info: SurfaceInfo,
    present_image_views: Vec<vk::ImageView>,
    depth_image_view: vk::ImageView,
    renderpass: vk::RenderPass,
) -> VkResult<Vec<vk::Framebuffer>> {
    Ok(present_image_views
        .iter()
        .map(|&present_image_view| {
            let attachments = [present_image_view, depth_image_view];
            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(renderpass)
                .attachments(&attachments)
                .width(surface_info.extent.width)
                .height(surface_info.extent.height)
                .layers(1);

            unsafe {
                context
                    .vk_device
                    .create_framebuffer(&create_info, None)
                    .expect("Could not create vk::Framebuffer.")
            }
        })
        .collect())
}

#[inline]
pub fn create_shader_module(
    vk_device: &ash::Device,
    spv_bytes: &[u8],
) -> VkResult<vk::ShaderModule> {
    let mut spv_file = Cursor::new(&spv_bytes);
    let shader_code = util::read_spv(&mut spv_file).expect("Failed to read shader spv file.");
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    unsafe { vk_device.create_shader_module(&shader_info, None) }
}

#[inline]
#[rustfmt::skip]
pub fn create_pipeline_layout(vk_device: &ash::Device) -> VkResult<vk::PipelineLayout> {
    let layout_create_info = vk::PipelineLayoutCreateInfo::default();
    unsafe { vk_device.create_pipeline_layout(&layout_create_info, None) }
}

#[inline]
pub fn create_graphics_pipeline(
    vk_device: &ash::Device,
    scissors: &Vec<vk::Rect2D>,
    viewports: &Vec<vk::Viewport>,
    renderpass: vk::RenderPass,
    vert_shader_module: vk::ShaderModule,
    frag_shader_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
) -> VkResult<vk::Pipeline> {
    let shader_entry_name = c"main";
    let shader_stage_create_infos = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(vert_shader_module)
            .name(shader_entry_name)
            .stage(vk::ShaderStageFlags::VERTEX),
        vk::PipelineShaderStageCreateInfo::default()
            .module(frag_shader_module)
            .name(shader_entry_name)
            .stage(vk::ShaderStageFlags::FRAGMENT),
    ];

    let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&Vertex::VERTEX_ATTRIBUTE_DESCRIPTIONS)
        .vertex_binding_descriptions(&Vertex::VERTEX_BINDING_DESCRIPTIONS);

    let vertex_input_assembly_state_info =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(Vertex::VERTEX_TOPOLOGY);

    let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
        .scissors(scissors)
        .viewports(viewports);

    let rasterization_info = vk::PipelineRasterizationStateCreateInfo::default()
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0)
        .polygon_mode(vk::PolygonMode::FILL);

    let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let noop_stencil_state = vk::StencilOpState::default()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::KEEP)
        .depth_fail_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::ALWAYS);

    let depth_state_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .front(noop_stencil_state)
        .back(noop_stencil_state)
        .max_depth_bounds(1.0);

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

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

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
        .layout(pipeline_layout)
        .render_pass(renderpass);

    unsafe {
        let graphics_pipelines = vk_device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_pipeline_info], None)
            .expect("Could not create vk::Pipeline.");

        Ok(graphics_pipelines[0])
    }
}

#[inline]
pub fn create_index_buffer(
    vk_device: &ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
) -> VkResult<Buffer> {
    let memory_property_flags =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let buffer_info = vk::BufferCreateInfo::default()
        .size(3 * size_of::<Index>() as u64)
        .usage(vk::BufferUsageFlags::INDEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    Buffer::new(
        vk_device,
        memory_properties,
        memory_property_flags,
        buffer_info,
    )
}

#[inline]
pub fn create_vertex_buffer(
    vk_device: &ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
) -> VkResult<Buffer> {
    let memory_property_flags =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let buffer_info = vk::BufferCreateInfo::default()
        .size(3 * size_of::<Vertex>() as u64)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    Buffer::new(
        vk_device,
        memory_properties,
        memory_property_flags,
        buffer_info,
    )
}

pub fn submit(
    context: &Context,
    pipeline: &mut GraphicsPipeline,
    indices: &[Index],
    vertices: &[Vertex],
) -> Result<()> {
    let Context { vk_device, .. } = context;
    let GraphicsPipeline {
        surface_info,
        scissors,
        viewports,
        present_queue,
        swapchain_device,
        swapchain,
        renderpass,
        framebuffers,
        graphics_pipeline,
        vertex_buffer,
        index_buffer,
        draw_commands,
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

    let buffer_offset = 0;
    index_buffer.map_memory(vk_device, buffer_offset, indices)?;
    vertex_buffer.map_memory(vk_device, buffer_offset, vertices)?;

    let frame = swapchain.current_frame;

    unsafe {
        let (image_index, _suboptimal) = swapchain_device.acquire_next_image(
            swapchain.inner,
            u64::MAX,
            present_queue.presented_semaphores[frame],
            vk::Fence::null(),
        )?;

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(*renderpass)
            .framebuffer(framebuffers[image_index as usize])
            .render_area(surface_info.extent.into())
            .clear_values(&clear_values);

        record_submit_commandbuffer(
            &vk_device,
            draw_commands.buffers[frame],
            draw_commands.fences[frame],
            present_queue.inner,
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[present_queue.presented_semaphores[frame]],
            &[present_queue.rendered_semaphores[frame]],
            |draw_command_buffer| {
                vk_device.cmd_begin_render_pass(
                    draw_command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                vk_device.cmd_bind_pipeline(
                    draw_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    *graphics_pipeline,
                );
                vk_device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                vk_device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                vk_device.cmd_bind_vertex_buffers(
                    draw_command_buffer,
                    0,
                    &[vertex_buffer.inner],
                    &[0],
                );
                vk_device.cmd_bind_index_buffer(
                    draw_command_buffer,
                    index_buffer.inner,
                    0,
                    vk::IndexType::UINT32,
                );
                vk_device.cmd_draw_indexed(draw_command_buffer, indices.len() as _, 1, 0, 0, 1);
                vk_device.cmd_end_render_pass(draw_command_buffer);
            },
        )?;

        let wait_semaphores = [present_queue.rendered_semaphores[frame]];
        let swapchains = [swapchain.inner];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        swapchain_device.queue_present(present_queue.inner, &present_info)?;
        swapchain.advance_current_frame();
    }

    Ok(())
}
