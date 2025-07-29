use crate::{
    adapter::Adapter, buffer::Buffer, commands::{create_command_buffer, create_command_buffers, create_command_pool, create_command_pools, record_submit_commandbuffer}, image, queue::{Queue, QueueFamily},  surface::Surface, swapchain::Swapchain, sync::{create_fence, create_fences, create_semaphores}, vertex::{Index, Vertex}, App
};
use ash::{khr, prelude::VkResult, util, vk};
use std::{default::Default, io::Cursor, slice};

const VERT_SPV: &[u8] = include_bytes!("../assets/shader/triangle/vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("../assets/shader/triangle/frag.spv");
const FRAMES_IN_FLIGHT: u32 = 2;

#[derive(Clone)]
pub struct Renderer {
    pub vertex_src_buffer: Buffer,
    pub vertex_dst_buffer: Buffer,
    pub index_src_buffer: Buffer,
    pub index_dst_buffer: Buffer,

    pub current_frame: usize,
    pub in_flight: Vec<vk::Fence>,
    pub available: Vec<vk::Semaphore>,
    pub rendered: Vec<vk::Semaphore>,
    pub cmd_buffers: Vec<vk::CommandBuffer>,
    pub cmd_pools: Vec<vk::CommandPool>,

    pub init_cmd_fence: vk::Fence,
    pub init_cmd_buffer: vk::CommandBuffer,
    pub init_cmd_pool: vk::CommandPool,

    pub swapchain: Swapchain,
    pub swapchain_device: khr::swapchain::Device,

    pub graphics_pipeline: vk::Pipeline,
    pub graphics_queue: Queue,
    pub render_device: ash::Device,
    pub surface: Surface,
    pub surface_instance: khr::surface::Instance,

    pub graphics_queue_family: QueueFamily,
    pub discrete_gpu: Adapter,
}

impl Renderer {
    pub fn new(app: &App, window_width: u32, window_height: u32) -> VkResult<Self> {
        // FIXME: Create fallback for integrated GPUs.
        let discrete_gpu = app
            .request_adapter_type(vk::PhysicalDeviceType::DISCRETE_GPU)
            .expect("No discrete gpu was found!");

        let surface_instance = khr::surface::Instance::new(&app.entry, &app.instance);
        let surface = Surface::new(app, &discrete_gpu, &surface_instance)?;
        let surface_extent = surface.extent(
            window_width,
            window_height,
            &discrete_gpu,
            &surface_instance,
        )?;

        let graphics_queue_family = discrete_gpu.get_queue_family(
            &surface_instance,
            &surface,
            true,
            vk::QueueFlags::GRAPHICS,
        );

        let render_device = create_render_device(app, &discrete_gpu, &graphics_queue_family)?;
        let graphics_queue = Queue::new(&graphics_queue_family, &render_device, 0);

        let vert_shader_module = create_shader_module(&render_device, VERT_SPV)?;
        let frag_shader_module = create_shader_module(&render_device, FRAG_SPV)?;
        let pipeline_layout = create_pipeline_layout(&render_device)?;
        let graphics_pipeline = create_graphics_pipeline(
            &surface,
            &render_device,
            vert_shader_module,
            frag_shader_module,
            pipeline_layout,
        )?;

        let swapchain_device = khr::swapchain::Device::new(&app.instance, &render_device);
        let swapchain = Swapchain::new(
            &discrete_gpu,
            &surface,
            surface_extent,
            &graphics_queue_family,
            &render_device,
            &swapchain_device,
            None,
        )?;

        let init_cmd_pool = create_command_pool(&render_device, &graphics_queue_family)?;
        let init_cmd_buffer = create_command_buffer(&render_device, init_cmd_pool)?;
        let init_cmd_fence = create_fence(&render_device)?;

        let cmd_pools =
            create_command_pools(&render_device, &graphics_queue_family, FRAMES_IN_FLIGHT);
        let cmd_buffers = create_command_buffers(&render_device, &cmd_pools);
        let in_flight = create_fences(&render_device, FRAMES_IN_FLIGHT);
        let available = create_semaphores(&render_device, FRAMES_IN_FLIGHT);
        let rendered = create_semaphores(&render_device, FRAMES_IN_FLIGHT);

        let vertex_src_buffer = create_vertex_src_buffer(&discrete_gpu, &render_device, 3)?;
        let vertex_dst_buffer = create_vertex_dst_buffer(&discrete_gpu, &render_device, 3)?;
        let index_src_buffer = create_index_src_buffer(&discrete_gpu, &render_device, 3)?;
        let index_dst_buffer = create_index_dst_buffer(&discrete_gpu, &render_device, 3)?;

        record_submit_commandbuffer(
            &render_device,
            init_cmd_buffer,
            init_cmd_fence,
            graphics_queue.handle,
            &[],
            &[],
            &[],
            |command_buffer| {
                image::transition_layout(
                    &render_device,
                    swapchain.depth_image.handle,
                    command_buffer,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags2::empty(),
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    vk::ImageAspectFlags::DEPTH,
                );
            },
        )?;

        unsafe {
            render_device.destroy_shader_module(vert_shader_module, None);
            render_device.destroy_shader_module(frag_shader_module, None);
            render_device.destroy_pipeline_layout(pipeline_layout, None);
        }

        Ok(Self {
            vertex_src_buffer,
            vertex_dst_buffer,
            index_src_buffer,
            index_dst_buffer,

            current_frame: 0,
            in_flight,
            available,
            rendered,
            cmd_buffers,
            cmd_pools,

            init_cmd_fence,
            init_cmd_buffer,
            init_cmd_pool,

            swapchain,
            swapchain_device,

            graphics_pipeline,
            graphics_queue,
            render_device,
            surface,
            surface_instance,

            graphics_queue_family,
            discrete_gpu,
        })
    }

    #[inline]
    pub fn advance_current_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.in_flight.len();
    }
    
    pub fn transfer_mesh_buffers(
        &self,
        indices: &[Index],
        vertices: &[Vertex],
        cmd_buffer: vk::CommandBuffer
    ) -> VkResult<()> {
        unsafe {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.render_device.begin_command_buffer(cmd_buffer, &command_buffer_begin_info)?;

            // Indices
            let size = (indices.len() * size_of::<Index>()) as vk::DeviceSize;
            let regions = vk::BufferCopy2::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);
            let copy_buffer_info = vk::CopyBufferInfo2::default()
                .src_buffer(self.index_src_buffer.handle)
                .dst_buffer(self.index_dst_buffer.handle)
                .regions(slice::from_ref(&regions));
            self.render_device.cmd_copy_buffer2(cmd_buffer, &copy_buffer_info);

            // Vertices
            let size = (vertices.len() * size_of::<Vertex>()) as vk::DeviceSize;
            let regions = vk::BufferCopy2::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);
            let copy_buffer_info = vk::CopyBufferInfo2::default()
                .src_buffer(self.vertex_src_buffer.handle)
                .dst_buffer(self.vertex_dst_buffer.handle)
                .regions(slice::from_ref(&regions));
            self.render_device.cmd_copy_buffer2(cmd_buffer, &copy_buffer_info);

            self.render_device.end_command_buffer(cmd_buffer)?;

            let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::default()
                .command_buffer(cmd_buffer);
            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(slice::from_ref(&cmd_buffer_submit_info));
            self.render_device.queue_submit2(
                self.graphics_queue.handle,
                slice::from_ref(&submit_info),
                vk::Fence::null(),
            )?;
        
            // NOTE: Is this wait really necessary?
            // NOTE: It may be wise to use a fence here for multiple simultaneous transfers.
            self.render_device.queue_wait_idle(self.graphics_queue.handle)?;

            Ok(())
        }
    }

    #[rustfmt::skip]
    pub fn submit(
        &mut self,
        window_width: u32,
        window_height: u32,
        indices: &[Index],
        vertices: &[Vertex],
    ) -> VkResult<()> {

        let surface_extent = self.surface
            .extent(window_width, window_height, &self.discrete_gpu, &self.surface_instance)?;
        if self.swapchain.extent != surface_extent {
            self.rebuild_swapchain(surface_extent)?;
        }
        
        let extent = self.swapchain.extent;
        let scissors = [extent.into()];
        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        };

        let in_flight = self.in_flight[self.current_frame];
        let available = self.available[self.current_frame];
        let rendered = self.rendered[self.current_frame];
        let cmd_buffer = self.cmd_buffers[self.current_frame];

        unsafe {
            self.render_device.wait_for_fences(&[in_flight], true, u64::MAX)?;
            self.render_device.reset_fences(&[in_flight])?;

            let (image_index, _suboptimal) = self.swapchain_device.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                available,
                vk::Fence::null(),
            )?;

            // Buffer transfer
            // FIXME: THIS IS BAD, MAP MEMORY ONCE.
            self.index_src_buffer.map_memory(&self.render_device, indices, 0)?;
            self.vertex_src_buffer.map_memory(&self.render_device, vertices, 0)?;
            // NOTE: It may be wise to create a dedicated command buffer for transfers.
            self.transfer_mesh_buffers(indices, vertices, cmd_buffer)?;

            // Commands
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.render_device.begin_command_buffer(cmd_buffer, &command_buffer_begin_info)?;

            let present_image = self.swapchain.present_images[image_index as usize];
            let present_image_view = self.swapchain.present_image_views[image_index as usize];

            // Color image transition
            image::transition_layout(
                &self.render_device,
                present_image,
                cmd_buffer,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageAspectFlags::COLOR,
            );

            // Render commands
            let color_attachment_info = vk::RenderingAttachmentInfo::default()
                .image_view(present_image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value);
            let rendering_info = vk::RenderingInfo::default()
                .render_area(extent.into())
                .color_attachments(slice::from_ref(&color_attachment_info))
                .layer_count(1);
            self.render_device
                .cmd_begin_rendering(cmd_buffer, &rendering_info);

            self.render_device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            self.render_device.cmd_set_viewport(cmd_buffer, 0, &viewports);
            self.render_device.cmd_set_scissor(cmd_buffer, 0, &scissors);

            self.render_device.cmd_bind_index_buffer(
                cmd_buffer,
                self.index_dst_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            self.render_device.cmd_bind_vertex_buffers(
                cmd_buffer,
                0,
                &[self.vertex_dst_buffer.handle],
                &[0],
            );

            self.render_device.cmd_draw_indexed(cmd_buffer, indices.len() as u32, 1, 0, 0, 1);
            self.render_device.cmd_end_rendering(cmd_buffer);

            // Present image transition
            image::transition_layout(
                &self.render_device,
                present_image,
                cmd_buffer,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::ImageAspectFlags::COLOR,
            );

            self.render_device.end_command_buffer(cmd_buffer)?;

            // Queue submit
            // NOTE: This may require `device_mask` to be filled at a later date.
            // NOTE: Not sure it's necessary since we're only using one device.
            let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::default()
                .command_buffer(cmd_buffer);
            let wait_sem_submit_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(available)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
            let signal_sem_submit_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(rendered)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(slice::from_ref(&cmd_buffer_submit_info))
                .wait_semaphore_infos(slice::from_ref(&wait_sem_submit_info))
                .signal_semaphore_infos(slice::from_ref(&signal_sem_submit_info));

            self.render_device.queue_submit2(
                self.graphics_queue.handle,
                slice::from_ref(&submit_info),
                in_flight,
            )?;

            // Queue present
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(slice::from_ref(&rendered))
                .swapchains(slice::from_ref(&self.swapchain.handle))
                .image_indices(slice::from_ref(&image_index));

            self.swapchain_device
                .queue_present(self.graphics_queue.handle, &present_info)?;
            
            Ok(self.advance_current_frame())
        }
    }

    #[rustfmt::skip]
    pub fn rebuild_swapchain(&mut self, extent: vk::Extent2D) -> VkResult<()> {
        unsafe {
            self.render_device.device_wait_idle()?;

            self.swapchain.rebuild(
                &self.discrete_gpu,
                &self.surface,
                extent,
                &self.graphics_queue_family,
                &self.render_device,
                &self.swapchain_device,
            )?;

            record_submit_commandbuffer(
                &self.render_device,
                self.init_cmd_buffer,
                self.init_cmd_fence,
                self.graphics_queue.handle,
                &[],
                &[],
                &[],
                |command_buffer| {
                    image::transition_layout(
                        &self.render_device,
                        self.swapchain.depth_image.handle,
                        command_buffer,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        vk::AccessFlags2::empty(),
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::ImageAspectFlags::DEPTH,
                    );
                },
            )
        }
    }

    #[inline]
    pub fn destroy(&mut self) -> VkResult<()> {
        unsafe {
            // NOTE: This call is the only reason this destroy
            // NOTE: method is safe where all others are not.
            self.render_device.device_wait_idle()?;

            self.index_src_buffer.destroy(&self.render_device);
            self.index_dst_buffer.destroy(&self.render_device);
            self.vertex_src_buffer.destroy(&self.render_device);
            self.vertex_dst_buffer.destroy(&self.render_device);

            for f in self.in_flight.iter() {
                self.render_device.destroy_fence(*f, None);
            }
            for s in self.available.iter() {
                self.render_device.destroy_semaphore(*s, None);
            }
            for s in self.rendered.iter() {
                self.render_device.destroy_semaphore(*s, None);
            }
            for p in self.cmd_pools.iter() {
                self.render_device.destroy_command_pool(*p, None);
            }

            self.swapchain.destroy(&self.render_device, &self.swapchain_device);

            self.render_device.destroy_fence(self.init_cmd_fence, None);
            self.render_device.destroy_command_pool(self.init_cmd_pool, None);

            self.render_device.destroy_pipeline(self.graphics_pipeline, None);
            self.render_device.destroy_device(None);

            self.surface_instance.destroy_surface(self.surface.handle, None);

            Ok(())
        }
    }
}

#[inline]
fn create_render_device(
    app: &App,
    adapter: &Adapter,
    queue_family: &QueueFamily,
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
    let features = vk::PhysicalDeviceFeatures::default().shader_clip_distance(true);
    let mut synchronization2 =
        vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    let mut dynamic_rendering =
        vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(slice::from_ref(&device_queue_info))
        .enabled_extension_names(&extension_names_raw)
        .enabled_features(&features)
        .push_next(&mut synchronization2)
        .push_next(&mut dynamic_rendering);

    unsafe {
        app.instance
            .create_device(adapter.handle, &device_info, None)
    }
}

#[inline]
fn create_shader_module(render_device: &ash::Device, spv_bytes: &[u8]) -> VkResult<vk::ShaderModule> {
    let mut spv_file = Cursor::new(&spv_bytes);
    let shader_code = util::read_spv(&mut spv_file).expect("Failed to read shader spv file.");
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    unsafe { render_device.create_shader_module(&shader_info, None) }
}

#[inline]
fn create_pipeline_layout(render_device: &ash::Device) -> VkResult<vk::PipelineLayout> {
    let layout_create_info = vk::PipelineLayoutCreateInfo::default();
    unsafe { render_device.create_pipeline_layout(&layout_create_info, None) }
}

#[inline]
fn create_graphics_pipeline(
    surface: &Surface,
    render_device: &ash::Device,
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
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .front(noop_stencil_state)
        .back(noop_stencil_state)
        .max_depth_bounds(1.0);

    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(slice::from_ref(&surface.format));

    let graphics_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .push_next(&mut pipeline_rendering_info)
        .stages(&shader_stage_create_infos)
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
        let graphics_pipelines = render_device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_pipeline_info], None)
            .expect("Could not create vk::Pipeline.");

        Ok(graphics_pipelines[0])
    }
}

#[inline]
fn create_index_src_buffer(
    adapter: &Adapter,
    render_device: &ash::Device,
    element_count: vk::DeviceSize,
) -> VkResult<Buffer> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(element_count * size_of::<Index>() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let offset = 0;

    Buffer::new(
        adapter,
        render_device,
        buffer_info,
        flags,
        offset,
    )
}

#[inline]
fn create_index_dst_buffer(
    adapter: &Adapter,
    render_device: &ash::Device,
    element_count: vk::DeviceSize,
) -> VkResult<Buffer> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(element_count * size_of::<Index>() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let offset = 0;

    Buffer::new(
        adapter,
        render_device,
        buffer_info,
        flags,
        offset,
    )
}

#[inline]
fn create_vertex_src_buffer(
    adapter: &Adapter,
    render_device: &ash::Device,
    element_count: vk::DeviceSize,
) -> VkResult<Buffer> {
    let info = vk::BufferCreateInfo::default()
        .size(element_count * size_of::<Vertex>() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let offset = 0;

    Buffer::new(
        adapter,
        render_device,
        info,
        flags,
        offset,
    )
}

#[inline]
fn create_vertex_dst_buffer(
    adapter: &Adapter,
    render_device: &ash::Device,
    element_count: vk::DeviceSize
) -> VkResult<Buffer> {
    let info = vk::BufferCreateInfo::default()
        .size(element_count * size_of::<Vertex>() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let offset = 0;

    Buffer::new(
        adapter,
        render_device,
        info,
        flags,
        offset,
    )
}

