//! Building a `Renderer`.

use super::{Renderer, device::create_render_device};
use crate::{
    App,
    buffer::DynamicBuffer,
    commands::{create_command_buffer, create_command_pool},
    cull::Culler,
    descriptors::{allocate_descriptor_sets, create_descriptor_pool, create_descriptor_set_layout},
    hiz::DepthPyramid,
    mesh::{MeshCapacity, MeshPool},
    pipeline::{create_graphics_pipeline, create_pipeline_layout, create_shader_module},
    queue::Queue,
    surface::Surface,
    swapchain::Swapchain,
    sync::{create_fence, create_semaphore},
    uniforms::{create_uniform_buffer, write_uniform_descriptor_sets},
    vertex::{Index, Vertex},
};
use ash::{khr, prelude::VkResult, vk};
use std::slice;

const SLANG_SPV: &[u8] = include_bytes!("../../assets/shader/slang.spv");
pub(super) const FRAMES_IN_FLIGHT: u32 = 2;
/// Starting size of each frame's dynamic geometry buffers, they grow to fit.
const INITIAL_DYNAMIC_ELEMENTS: vk::DeviceSize = 1024;

impl Renderer {
    pub fn new(app: &App, window_width: u32, window_height: u32) -> VkResult<Self> {
        Self::with_mesh_capacity(app, window_width, window_height, MeshCapacity::default())
    }

    /// Sizes the shared mesh buffers, which don't grow.
    pub fn with_mesh_capacity(
        app: &App,
        window_width: u32,
        window_height: u32,
        mesh_capacity: MeshCapacity,
    ) -> VkResult<Self> {
        let adapter = app
            .request_adapter()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;

        // Surface
        let surface_instance = app.surface_instance.clone();
        let surface = Surface::new(app, &adapter, &surface_instance)?;
        let surface_extent =
            surface.extent(window_width, window_height, &adapter, &surface_instance)?;

        // Render Device
        // NOTE: Culling runs on the graphics queue, so it must also support compute.
        let graphics_queue_family = adapter
            .get_queue_family(
                &surface_instance,
                &surface,
                true,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )?
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let multi_draw_indirect = adapter.features.multi_draw_indirect == vk::TRUE;
        // Compacting needs to draw every visible mesh in one counted call.
        let draw_indirect_count = adapter.draw_indirect_count && multi_draw_indirect;
        let render_device = create_render_device(
            app,
            &adapter,
            &graphics_queue_family,
            multi_draw_indirect,
            draw_indirect_count,
        )?;

        // Everything else starts null, so dropping the renderer if `create` fails
        // destroys exactly what was created.
        let mut renderer = Self {
            occlusion_culling: true,

            mesh_pool: MeshPool::null(mesh_capacity, FRAMES_IN_FLIGHT as usize),
            frame_staging_buffers: Vec::new(),
            culler: Culler::null(draw_indirect_count),
            depth_pyramid: DepthPyramid::null(),
            indirect_draw_batch: match multi_draw_indirect {
                true => adapter.properties.limits.max_draw_indirect_count,
                false => 1,
            },
            device_draw_paths: (draw_indirect_count, 1),
            last_frame: None,
            frame_index_buffers: Vec::new(),
            frame_vertex_buffers: Vec::new(),

            current_frame: 0,
            in_flight_frames: Vec::new(),
            available_frames: Vec::new(),
            rendered_images: Vec::new(),
            frame_cmd_buffers: Vec::new(),
            frame_cmd_pools: Vec::new(),

            init_cmd_fence: vk::Fence::null(),
            init_cmd_buffer: vk::CommandBuffer::null(),
            init_cmd_pool: vk::CommandPool::null(),

            swapchain: Swapchain::null(),
            swapchain_device: khr::swapchain::Device::new(&app.instance, &render_device),
            swapchain_outdated: false,

            ubo_desc_pool: vk::DescriptorPool::null(),
            ubo_desc_set_layout: vk::DescriptorSetLayout::null(),
            ubo_desc_sets: Vec::new(),
            uniform_buffers: Vec::new(),

            graphics_pipeline: vk::Pipeline::null(),
            graphics_pipeline_layout: vk::PipelineLayout::null(),
            graphics_queue: Queue::new(&graphics_queue_family, &render_device, 0),
            graphics_queue_family,
            render_device,

            surface,
            surface_instance,
            adapter,
        };
        renderer.device_draw_paths.1 = renderer.indirect_draw_batch;
        renderer.create(surface_extent, mesh_capacity)?;
        Ok(renderer)
    }

    /// Fills in everything the device makes, leaving it for `Drop` if it fails.
    fn create(
        &mut self,
        surface_extent: vk::Extent2D,
        mesh_capacity: MeshCapacity,
    ) -> VkResult<()> {
        let device = self.render_device.clone();
        let adapter = self.adapter.clone();

        // Uniform Buffers
        let ubo_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(FRAMES_IN_FLIGHT);
        self.ubo_desc_pool =
            create_descriptor_pool(&device, slice::from_ref(&ubo_pool_size), FRAMES_IN_FLIGHT)?;
        self.ubo_desc_set_layout = create_descriptor_set_layout(
            &device,
            &[vk::DescriptorType::UNIFORM_BUFFER],
            vk::ShaderStageFlags::VERTEX,
        )?;
        self.ubo_desc_sets = allocate_descriptor_sets(
            &device,
            self.ubo_desc_pool,
            self.ubo_desc_set_layout,
            FRAMES_IN_FLIGHT,
        )?;
        for _ in 0..FRAMES_IN_FLIGHT {
            self.uniform_buffers
                .push(create_uniform_buffer(&adapter, &device)?);
        }
        write_uniform_descriptor_sets(&device, &self.ubo_desc_sets, &self.uniform_buffers);

        // Graphics Pipeline
        self.graphics_pipeline_layout =
            create_pipeline_layout(&device, slice::from_ref(&self.ubo_desc_set_layout))?;
        let shader_module = create_shader_module(&device, SLANG_SPV)?;
        let pipeline = create_graphics_pipeline(
            &self.surface,
            &device,
            shader_module,
            self.graphics_pipeline_layout,
        );
        unsafe { device.destroy_shader_module(shader_module, None) };
        self.graphics_pipeline = pipeline?;

        // Swapchain
        self.swapchain = Swapchain::new(
            &adapter,
            &self.surface,
            surface_extent,
            &self.graphics_queue_family,
            &device,
            &self.swapchain_device,
            None,
        )?;

        // Global Commands
        self.init_cmd_pool = create_command_pool(&device, &self.graphics_queue_family)?;
        self.init_cmd_buffer =
            create_command_buffer(&device, self.init_cmd_pool, vk::CommandBufferLevel::PRIMARY)?;
        self.init_cmd_fence = create_fence(&device)?;

        // Frames in Flight
        for _ in 0..FRAMES_IN_FLIGHT {
            let pool = create_command_pool(&device, &self.graphics_queue_family)?;
            self.frame_cmd_pools.push(pool);
            self.frame_cmd_buffers.push(create_command_buffer(
                &device,
                pool,
                vk::CommandBufferLevel::PRIMARY,
            )?);
            self.in_flight_frames.push(create_fence(&device)?);
            self.available_frames.push(create_semaphore(&device)?);
        }
        for _ in 0..self.swapchain.present_images.len() {
            self.rendered_images.push(create_semaphore(&device)?);
        }

        // Meshes
        self.mesh_pool =
            MeshPool::new(&adapter, &device, mesh_capacity, FRAMES_IN_FLIGHT as usize)?;
        self.culler = Culler::new(&adapter, &device, FRAMES_IN_FLIGHT, self.culler.can_compact)?;
        self.depth_pyramid = DepthPyramid::new(
            &adapter,
            &device,
            self.swapchain.depth_image_view,
            self.swapchain.extent,
        )?;
        self.init_depth_pyramid_layout()?;

        // Per Frame Buffers
        let dynamic_buffer = |usage, element_size: usize| {
            DynamicBuffer::new(
                &adapter,
                &device,
                usage,
                INITIAL_DYNAMIC_ELEMENTS * element_size as vk::DeviceSize,
            )
        };
        for _ in 0..FRAMES_IN_FLIGHT {
            self.frame_staging_buffers.push(dynamic_buffer(
                vk::BufferUsageFlags::TRANSFER_SRC,
                size_of::<Vertex>(),
            )?);
            self.frame_index_buffers.push(dynamic_buffer(
                vk::BufferUsageFlags::INDEX_BUFFER,
                size_of::<Index>(),
            )?);
            self.frame_vertex_buffers.push(dynamic_buffer(
                vk::BufferUsageFlags::VERTEX_BUFFER,
                size_of::<Vertex>(),
            )?);
        }

        Ok(())
    }
}
