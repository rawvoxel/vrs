mod frame;
mod setup;

use crate::{
    App,
    adapter::Adapter,
    buffer::{Buffer, DynamicBuffer},
    commands::{create_command_buffer, create_command_pool, record_submit_commandbuffer},
    cull::Culler,
    descriptors::{allocate_descriptor_sets, create_descriptor_pool, create_descriptor_set_layout},
    hiz::DepthPyramid,
    mesh::{Mesh, MeshCapacity, MeshId, MeshPool},
    pipeline::create_shader_module,
    queue::{Queue, QueueFamily},
    surface::Surface,
    swapchain::Swapchain,
    sync::{create_fence, create_semaphore},
    vertex::{Index, Vertex},
};
use ash::{khr, prelude::VkResult, vk};
use glam::Mat4;
use setup::{
    create_graphics_pipeline, create_pipeline_layout, create_render_device, create_uniform_buffer,
    write_uniform_descriptor_sets,
};
use std::slice;

const SLANG_SPV: &[u8] = include_bytes!("../../assets/shader/slang.spv");
const FRAMES_IN_FLIGHT: u32 = 2;
/// Starting size of each frame's dynamic geometry buffers, they grow to fit.
const INITIAL_DYNAMIC_ELEMENTS: vk::DeviceSize = 1024;

/// Everything drawn by one `Renderer::submit`.
#[derive(Clone, Copy, Default)]
pub struct DrawList<'a> {
    /// Meshes added with `Renderer::create_mesh`. References, so ids can stay
    /// wherever they're kept, such as a map of chunks.
    pub meshes: &'a [&'a MeshId],
    /// Geometry for this frame only, read by the device straight from host visible memory.
    pub indices: &'a [Index],
    pub vertices: &'a [Vertex],
    /// Meshes outside this camera's frustum, or hidden behind others, are culled on the device.
    pub view: Mat4,
    pub proj: Mat4,
}

/// Draws meshes into the `App`'s window, culling them on the device.
pub struct Renderer {
    /// Whether meshes hidden behind others are culled, on by default.
    /// Off, meshes are only frustum culled and drawn in one pass.
    pub occlusion_culling: bool,

    mesh_pool: MeshPool,
    /// Mesh uploads pass through these on their way into `mesh_pool`.
    frame_staging_buffers: Vec<DynamicBuffer>,
    /// Turns each frame's meshes into indirect draws for the visible ones.
    culler: Culler,
    /// Built from each frame's depth between its two culling phases.
    depth_pyramid: DepthPyramid,
    /// Most draws one indirect call can make, 1 without the `multiDrawIndirect` feature.
    indirect_draw_batch: u32,
    frame_index_buffers: Vec<DynamicBuffer>,
    frame_vertex_buffers: Vec<DynamicBuffer>,

    current_frame: usize,
    in_flight_frames: Vec<vk::Fence>,
    available_frames: Vec<vk::Semaphore>,
    /// One per swapchain image, the presentation engine may hold onto it until that image is reacquired.
    rendered_images: Vec<vk::Semaphore>,
    frame_cmd_buffers: Vec<vk::CommandBuffer>,
    frame_cmd_pools: Vec<vk::CommandPool>,

    init_cmd_fence: vk::Fence,
    init_cmd_buffer: vk::CommandBuffer,
    init_cmd_pool: vk::CommandPool,

    swapchain: Swapchain,
    swapchain_device: khr::swapchain::Device,
    /// Set when the swapchain no longer matches the surface and must be rebuilt.
    swapchain_outdated: bool,

    ubo_desc_pool: vk::DescriptorPool,
    ubo_desc_set_layout: vk::DescriptorSetLayout,
    ubo_desc_sets: Vec<vk::DescriptorSet>,
    uniform_buffers: Vec<Buffer>,

    graphics_pipeline: vk::Pipeline,
    graphics_pipeline_layout: vk::PipelineLayout,
    graphics_queue: Queue,
    graphics_queue_family: QueueFamily,
    render_device: ash::Device,

    surface: Surface,
    surface_instance: khr::surface::Instance,
    adapter: Adapter,
}

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
        renderer.create(surface_extent, mesh_capacity)?;
        Ok(renderer)
    }

    /// Fills in everything the device makes, leaving it for `Drop` if it fails.
    fn create(&mut self, surface_extent: vk::Extent2D, mesh_capacity: MeshCapacity) -> VkResult<()> {
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
            self.uniform_buffers.push(create_uniform_buffer(&adapter, &device)?);
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
        self.mesh_pool = MeshPool::new(&adapter, &device, mesh_capacity, FRAMES_IN_FLIGHT as usize)?;
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

    /// The adapter the renderer chose.
    #[inline]
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    /// Adds a mesh to the shared mesh buffers, uploaded by the next `submit`.
    /// Indices are relative to the mesh's own first vertex, and an empty mesh draws nothing.
    ///
    /// Fails with `ERROR_OUT_OF_DEVICE_MEMORY` when the shared buffers have no room left.
    #[inline]
    pub fn create_mesh(&mut self, indices: &[Index], vertices: &[Vertex]) -> VkResult<MeshId> {
        self.mesh_pool.create(indices, vertices)
    }

    /// Its space is reused once no frame in flight can still draw it.
    #[inline]
    pub fn destroy_mesh(&mut self, id: MeshId) {
        let frame_count = self.in_flight_frames.len();
        let last_frame = (self.current_frame + frame_count - 1) % frame_count;
        self.mesh_pool.destroy(id, last_frame);
    }

    /// Where the mesh lives in the shared buffers, and its bounds.
    ///
    /// # Panics
    /// If `id` came from another renderer.
    #[inline]
    pub fn mesh(&self, id: &MeshId) -> &Mesh {
        self.mesh_pool.get(id)
    }

    /// Renders and presents a frame, skipping it while the window is minimized.
    pub fn submit(&mut self, window_width: u32, window_height: u32, draws: &DrawList) -> VkResult<()> {
        let surface_extent = self.surface.extent(
            window_width,
            window_height,
            &self.adapter,
            &self.surface_instance,
        )?;
        // A minimized window has no surface to present to.
        if surface_extent.width == 0 || surface_extent.height == 0 {
            return Ok(());
        }
        if self.swapchain_outdated || self.swapchain.extent != surface_extent {
            self.rebuild_swapchain(surface_extent)?;
        }

        let frame = self.current_frame;
        let Some(image_index) = self.acquire_image(frame, surface_extent)? else {
            return Ok(());
        };
        let data = self.write_frame_data(frame, draws)?;
        self.record_frame(frame, image_index, &data)?;
        self.submit_and_present(frame, image_index)?;

        self.current_frame = (self.current_frame + 1) % self.in_flight_frames.len();
        Ok(())
    }

    fn rebuild_swapchain(&mut self, extent: vk::Extent2D) -> VkResult<()> {
        unsafe {
            self.render_device.device_wait_idle()?;

            self.swapchain.rebuild(
                &self.adapter,
                &self.surface,
                extent,
                &self.graphics_queue_family,
                &self.render_device,
                &self.swapchain_device,
            )?;
            self.swapchain_outdated = false;

            // The device is idle, so no semaphore is still waiting to be signaled or presented.
            let image_count = self.swapchain.present_images.len();
            if self.rendered_images.len() != image_count {
                for semaphore in self.rendered_images.drain(..) {
                    self.render_device.destroy_semaphore(semaphore, None);
                }
                for _ in 0..image_count {
                    let semaphore = create_semaphore(&self.render_device)?;
                    self.rendered_images.push(semaphore);
                }
            }

            self.depth_pyramid.resize(
                &self.adapter,
                &self.render_device,
                self.swapchain.depth_image_view,
                self.swapchain.extent,
            )?;
        }
        self.init_depth_pyramid_layout()
    }

    fn init_depth_pyramid_layout(&self) -> VkResult<()> {
        record_submit_commandbuffer(
            &self.render_device,
            self.init_cmd_buffer,
            self.init_cmd_fence,
            self.graphics_queue.handle,
            &[],
            &[],
            &[],
            |cmd_buffer| unsafe {
                self.depth_pyramid
                    .cmd_init_layout(&self.render_device, cmd_buffer)
            },
        )
    }
}

/// Must be dropped before the `App` it was created from.
/// Anything `create` didn't get to is still null, which Vulkan destroys as a no-op.
impl Drop for Renderer {
    fn drop(&mut self) {
        let device = &self.render_device;
        unsafe {
            // NOTE: Nothing can be done about a lost device here, so destroy regardless.
            let _ = device.device_wait_idle();

            self.mesh_pool.destroy_buffers(device);
            self.culler.destroy(device);
            self.depth_pyramid.destroy(device);
            for buffer in self
                .frame_staging_buffers
                .iter()
                .chain(&self.frame_index_buffers)
                .chain(&self.frame_vertex_buffers)
            {
                buffer.destroy(device);
            }

            for buffer in self.uniform_buffers.iter() {
                buffer.destroy(device);
            }
            device.destroy_descriptor_set_layout(self.ubo_desc_set_layout, None);
            device.destroy_descriptor_pool(self.ubo_desc_pool, None);

            for &fence in self.in_flight_frames.iter() {
                device.destroy_fence(fence, None);
            }
            for &semaphore in self.available_frames.iter().chain(&self.rendered_images) {
                device.destroy_semaphore(semaphore, None);
            }
            for &pool in self.frame_cmd_pools.iter() {
                device.destroy_command_pool(pool, None);
            }

            self.swapchain.destroy(device, &self.swapchain_device);

            device.destroy_fence(self.init_cmd_fence, None);
            device.destroy_command_pool(self.init_cmd_pool, None);

            device.destroy_pipeline(self.graphics_pipeline, None);
            device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            device.destroy_device(None);
        }
    }
}
