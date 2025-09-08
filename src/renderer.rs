use crate::{
    adapter::Adapter, buffer::Buffer, descriptors::{create_descriptor_set, create_descriptor_set_layout}, image, queue::{Queue, QueueFamily}, surface::Surface, swapchain::Swapchain, uniforms::MvpMatrices, vertex::{Index, Vertex}, App
};
use crate::sync::{create_fence, create_fences, create_semaphores};
use crate::descriptors::create_descriptor_pool;
use crate::commands::{create_command_buffer, create_command_buffers, create_command_pool, create_command_pools, record_submit_commandbuffer};
use ash::{khr, prelude::VkResult, util, vk};
use std::{default::Default, ffi::c_void, io::Cursor, slice};

const SLANG_SPV: &[u8] = include_bytes!("../assets/shader/slang.spv");
const FRAMES_IN_FLIGHT: u32 = 2;

#[derive(Clone)]
pub struct Renderer {
    pub index_src_buffer_ptr: *mut c_void,
    pub index_src_buffer: Buffer,
    pub index_dst_buffer: Buffer,
    pub vertex_src_buffer_ptr: *mut c_void,
    pub vertex_src_buffer: Buffer,
    pub vertex_dst_buffer: Buffer,

    pub current_frame: usize,
    pub in_flight_frames: Vec<vk::Fence>,
    pub available_frames: Vec<vk::Semaphore>,
    pub rendered_frames: Vec<vk::Semaphore>,
    pub frame_cmd_buffers: Vec<vk::CommandBuffer>,
    pub frame_cmd_pools: Vec<vk::CommandPool>,
    
    pub init_cmd_fence: vk::Fence,
    pub init_cmd_buffer: vk::CommandBuffer,
    pub init_cmd_pool: vk::CommandPool,

    pub swapchain: Swapchain,
    pub swapchain_device: khr::swapchain::Device,

    pub ubo_desc_pool: vk::DescriptorPool,
    pub ubo_desc_set_layout: vk::DescriptorSetLayout,
    pub ubo_desc_set: vk::DescriptorSet,
    pub uniform_buffers: Vec<Buffer>,
    pub uniform_buffer_ptrs: Vec<*mut c_void>,

    pub graphics_pipeline: vk::Pipeline,
    pub graphics_pipeline_layout: vk::PipelineLayout,
    pub graphics_queue: Queue,
    pub graphics_queue_family: QueueFamily,
    pub render_device: ash::Device,

    pub surface: Surface,
    pub surface_instance: khr::surface::Instance,
    pub discrete_gpu: Adapter,
}

unsafe impl Send for Renderer {}
unsafe impl Sync for Renderer {}

impl Renderer {
    pub fn new(app: &App, window_width: u32, window_height: u32) -> VkResult<Self> {
        // FIXME: Create fallback for integrated GPUs.
        let discrete_gpu = app
            .request_adapter_type(vk::PhysicalDeviceType::DISCRETE_GPU)
            .expect("No discrete gpu was found!");

        // Surface
        let surface_instance = khr::surface::Instance::new(&app.entry, &app.instance);
        let surface = Surface::new(app, &discrete_gpu, &surface_instance)?;
        let surface_extent = surface.extent(
            window_width,
            window_height,
            &discrete_gpu,
            &surface_instance,
        )?;

        // Render Device
        let graphics_queue_family = discrete_gpu.get_queue_family(
            &surface_instance,
            &surface,
            true,
            vk::QueueFlags::GRAPHICS,
        );
        let render_device = create_render_device(app, &discrete_gpu, &graphics_queue_family)?;
        let graphics_queue = Queue::new(&graphics_queue_family, &render_device, 0);

        // Uniform Buffers
        let ubo_desc_pool = create_descriptor_pool(&render_device, vk::DescriptorType::UNIFORM_BUFFER, 1, 1)?;
        let ubo_desc_set_layout = create_descriptor_set_layout(
            &render_device,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::VERTEX,
        )?;
        let ubo_desc_set = create_descriptor_set(&render_device, ubo_desc_pool, ubo_desc_set_layout)?;
        let uniform_buffers = create_uniform_buffers(&discrete_gpu, &render_device, FRAMES_IN_FLIGHT);
        let uniform_buffer_ptrs = uniform_buffers.iter()
            .map(|b| {
                b.mem_map(&render_device, 0).expect("Could not memory map buffers.")
            })
            .collect();

        // Graphics Pipeline
        let shader_module = create_shader_module(&render_device, SLANG_SPV)?;
        let graphics_pipeline_layout = create_pipeline_layout(&render_device, slice::from_ref(&ubo_desc_set_layout))?;
        let graphics_pipeline = create_graphics_pipeline(
            &surface,
            &render_device,
            shader_module,
            graphics_pipeline_layout,
        )?;

        // Swapchain
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

        // Global Commands
        let init_cmd_pool = create_command_pool(&render_device, &graphics_queue_family)?;
        let init_cmd_buffer = create_command_buffer(&render_device, init_cmd_pool, vk::CommandBufferLevel::PRIMARY)?;
        let init_cmd_fence = create_fence(&render_device)?;

        // Frames in Flight
        let frame_cmd_pools = create_command_pools(
            &render_device,
            &graphics_queue_family,
            FRAMES_IN_FLIGHT
        );
        let frame_cmd_buffers = create_command_buffers(
            &render_device,
            &frame_cmd_pools,
            vk::CommandBufferLevel::PRIMARY
        );
        let in_flight_frames = create_fences(&render_device, FRAMES_IN_FLIGHT);
        let available_frames = create_semaphores(&render_device, FRAMES_IN_FLIGHT);
        let rendered_frames = create_semaphores(&render_device, FRAMES_IN_FLIGHT);

        // Mesh Buffers
        // FIXME: Figure out how to initialize an upper limit for size.
        let index_src_buffer = create_transfer_src_buffer(
            &discrete_gpu,
            &render_device,
            12 * size_of::<Index>() as vk::DeviceSize
        )?;
        let vertex_src_buffer = create_transfer_src_buffer(
            &discrete_gpu,
            &render_device,
            12 * size_of::<Vertex>() as vk::DeviceSize
        )?;
        let index_dst_buffer = create_index_dst_buffer(&discrete_gpu, &render_device, 12)?;
        let vertex_dst_buffer = create_vertex_dst_buffer(&discrete_gpu, &render_device, 12)?;
        let index_src_buffer_ptr = index_src_buffer.mem_map(&render_device, 0)?;
        let vertex_src_buffer_ptr = vertex_src_buffer.mem_map(&render_device, 0)?;

        // Transition Depth Image Layout
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
            render_device.destroy_shader_module(shader_module, None);
        }

        Ok(Self {
            index_src_buffer_ptr,
            index_src_buffer,
            index_dst_buffer,
            vertex_src_buffer_ptr,
            vertex_src_buffer,
            vertex_dst_buffer,

            current_frame: 0,
            in_flight_frames,
            available_frames,
            rendered_frames,
            frame_cmd_buffers,
            frame_cmd_pools,

            ubo_desc_pool,
            ubo_desc_set_layout,
            ubo_desc_set,
            uniform_buffers,
            uniform_buffer_ptrs,

            init_cmd_fence,
            init_cmd_buffer,
            init_cmd_pool,

            swapchain,
            swapchain_device,

            graphics_pipeline,
            graphics_pipeline_layout,
            graphics_queue,
            graphics_queue_family,
            render_device,

            surface,
            surface_instance,
            discrete_gpu,
        })
    }

    #[inline]
    pub fn advance_current_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.in_flight_frames.len();
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
            // TODO: It may be wise to use a fence here for multiple simultaneous transfers.
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
            color: vk::ClearColorValue { float32: [0.1; 4] },
        };

        let frame_in_flight = self.in_flight_frames[self.current_frame];
        let frame_available = self.available_frames[self.current_frame];
        let frame_rendered = self.rendered_frames[self.current_frame];
        let frame_cmd_buffer = self.frame_cmd_buffers[self.current_frame];

        unsafe {
            self.render_device.wait_for_fences(&[frame_in_flight], true, u64::MAX)?;
            self.render_device.reset_fences(&[frame_in_flight])?;

            let (image_index, _suboptimal) = self.swapchain_device.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                frame_available,
                vk::Fence::null(),
            )?;

            // Mesh buffer transfers
            self.index_src_buffer.mem_write(self.index_src_buffer_ptr, indices);
            self.vertex_src_buffer.mem_write(self.vertex_src_buffer_ptr, vertices);
            // NOTE: It may be wise to create a dedicated command buffer for transfers.
            self.transfer_mesh_buffers(indices, vertices, frame_cmd_buffer)?;

            // Uniform buffers
            let mvp_identity = MvpMatrices::default();
            let u_ptr = self.uniform_buffer_ptrs[self.current_frame];
            self.uniform_buffers[self.current_frame].mem_write(u_ptr, slice::from_ref(&mvp_identity));
            // TODO
            // self.render_device.update_descriptor_sets(descriptor_writes, descriptor_copies);

            // Commands
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.render_device.begin_command_buffer(frame_cmd_buffer, &command_buffer_begin_info)?;
            

            let present_image = self.swapchain.present_images[image_index as usize];
            let present_image_view = self.swapchain.present_image_views[image_index as usize];

            // Color image transition
            image::transition_layout(
                &self.render_device,
                present_image,
                frame_cmd_buffer,
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
                .cmd_begin_rendering(frame_cmd_buffer, &rendering_info);

            self.render_device.cmd_bind_pipeline(
                frame_cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            self.render_device.cmd_set_viewport(frame_cmd_buffer, 0, &viewports);
            self.render_device.cmd_set_scissor(frame_cmd_buffer, 0, &scissors);

            // Buffer Bindings
            self.render_device.cmd_bind_index_buffer(
                frame_cmd_buffer,
                self.index_dst_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            self.render_device.cmd_bind_vertex_buffers(
                frame_cmd_buffer,
                0,
                &[self.vertex_dst_buffer.handle],
                &[0],
            );
            self.render_device.cmd_bind_descriptor_sets(
                frame_cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline_layout,
                0,
                &[self.ubo_desc_set],
                &[1]
            );

            self.render_device.cmd_draw_indexed(frame_cmd_buffer, indices.len() as u32, 1, 0, 0, 0);
            self.render_device.cmd_end_rendering(frame_cmd_buffer);

            // Present image transition
            image::transition_layout(
                &self.render_device,
                present_image,
                frame_cmd_buffer,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::ImageAspectFlags::COLOR,
            );

            self.render_device.end_command_buffer(frame_cmd_buffer)?;

            // Queue submit
            // NOTE: This may require `device_mask` to be filled at a later date.
            // NOTE: Not sure it's necessary since we're only using one device.
            let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::default()
                .command_buffer(frame_cmd_buffer);
            let wait_sem_submit_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(frame_available)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
            let signal_sem_submit_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(frame_rendered)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(slice::from_ref(&cmd_buffer_submit_info))
                .wait_semaphore_infos(slice::from_ref(&wait_sem_submit_info))
                .signal_semaphore_infos(slice::from_ref(&signal_sem_submit_info));

            self.render_device.queue_submit2(
                self.graphics_queue.handle,
                slice::from_ref(&submit_info),
                frame_in_flight,
            )?;

            // Queue present
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(slice::from_ref(&frame_rendered))
                .swapchains(slice::from_ref(&self.swapchain.handle))
                .image_indices(slice::from_ref(&image_index));

            self.swapchain_device
                .queue_present(self.graphics_queue.handle, &present_info)?;
            
            Ok(self.advance_current_frame())
        }
    }

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

            // Depth Image Layout Transition
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
            // NOTE: method is not unsafe where all others are.
            self.render_device.device_wait_idle()?;

            self.index_src_buffer.mem_unmap(&self.render_device);
            self.index_src_buffer.destroy(&self.render_device);
            self.index_dst_buffer.destroy(&self.render_device);
            self.vertex_src_buffer.mem_unmap(&self.render_device);
            self.vertex_src_buffer.destroy(&self.render_device);
            self.vertex_dst_buffer.destroy(&self.render_device);

            for ubo in self.uniform_buffers.iter() {
                ubo.destroy(&self.render_device);
            }
            
            self.render_device.destroy_descriptor_set_layout(self.ubo_desc_set_layout, None);
            self.render_device.destroy_descriptor_pool(self.ubo_desc_pool, None);

            for f in self.in_flight_frames.iter() {
                self.render_device.destroy_fence(*f, None);
            }
            for s in self.available_frames.iter() {
                self.render_device.destroy_semaphore(*s, None);
            }
            for s in self.rendered_frames.iter() {
                self.render_device.destroy_semaphore(*s, None);
            }
            for p in self.frame_cmd_pools.iter() {
                self.render_device.destroy_command_pool(*p, None);
            }

            self.swapchain.destroy(&self.render_device, &self.swapchain_device);

            self.render_device.destroy_fence(self.init_cmd_fence, None);
            self.render_device.destroy_command_pool(self.init_cmd_pool, None);

            self.render_device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
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
fn create_pipeline_layout(render_device: &ash::Device, set_layouts: & [vk::DescriptorSetLayout]) -> VkResult<vk::PipelineLayout> {
    let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
    unsafe { render_device.create_pipeline_layout(&layout_create_info, None) }
}

#[inline]
fn create_graphics_pipeline(
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
        let graphics_pipelines = render_device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_pipeline_info], None)
            .expect("Could not create vk::Pipeline.");

        Ok(graphics_pipelines[0])
    }
}

#[inline]
fn create_transfer_src_buffer(
    adapter: &Adapter,
    render_device: &ash::Device,
    size: vk::DeviceSize,
) -> VkResult<Buffer> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
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

#[inline]
fn create_uniform_buffers(
    adapter: &Adapter,
    render_device: &ash::Device,
    buffer_count: u32
) -> Vec<Buffer> {
    let info = vk::BufferCreateInfo::default()
        .size(size_of::<MvpMatrices>() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    (0..buffer_count)
        .map(|_| {
            Buffer::new(adapter, render_device, info, flags, 0)
                .expect("Could not create uniform buffer.")
    })
    .collect()
}
