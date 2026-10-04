//! Building a `Renderer`.

use super::{Lighting, Renderer, RendererConfig, device::create_render_device};
use crate::{
    App,
    adapter::Adapter,
    buffer::DynamicBuffer,
    commands::{create_command_buffer, create_command_pool},
    cull::Culler,
    descriptors::{allocate_descriptor_sets, create_descriptor_pool, create_descriptor_set_layout},
    gi::ProbeVolume,
    hiz::DepthPyramid,
    mesh::MeshPool,
    pipeline::{create_graphics_pipeline, create_pipeline_layout, create_shader_module},
    queue::{Queue, QueueFamily},
    surface::Surface,
    swapchain::Swapchain,
    sync::{create_fence, create_semaphore},
    uniforms::{create_uniform_buffer, write_uniform_descriptor_sets},
    vertex::{Index, Vertex},
};
use ash::{khr, prelude::VkResult, vk};
use std::slice;

const DYNAMIC_SPV: &[u8] = include_bytes!("../../assets/shader/slang.spv");
const MESH_SPV: &[u8] = include_bytes!("../../assets/shader/mesh.spv");
pub(super) const FRAMES_IN_FLIGHT: u32 = 2;
/// Starting size of each frame's dynamic geometry buffers, they grow to fit.
const INITIAL_DYNAMIC_ELEMENTS: vk::DeviceSize = 1024;
/// Frames start as the near black the renderer used before it drew in sRGB.
const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.01, 0.01, 0.01, 1.0];

impl Renderer {
    /// With the default `RendererConfig`.
    pub fn new(app: &App, window_width: u32, window_height: u32) -> VkResult<Self> {
        Self::with_config(app, window_width, window_height, RendererConfig::default())
    }

    /// For a headless `App`, the size is what frames render at.
    pub fn with_config(
        app: &App,
        window_width: u32,
        window_height: u32,
        config: RendererConfig,
    ) -> VkResult<Self> {
        let surface_instance = app.surface_instance.clone();
        let (adapter, graphics_queue_family) = pick_adapter(app, &surface_instance)?;

        // Surface
        let surface = match app.surface() {
            Some(_) => Surface::new(app, &adapter, &surface_instance, config.vsync)?,
            None => Surface::offscreen(app, &adapter, FRAMES_IN_FLIGHT),
        };
        let surface_extent =
            surface.extent(window_width, window_height, &adapter, &surface_instance)?;

        // Render Device
        let render_device = create_render_device(
            app,
            &adapter,
            &graphics_queue_family,
            !surface.is_offscreen(),
        )?;

        // Everything else starts null, so dropping the renderer if `create` fails
        // destroys exactly what was created.
        let mut renderer = Self {
            occlusion_culling: config.occlusion_culling,
            backface_culling: config.backface_culling,
            grow_mesh_pool: config.grow_mesh_pool,
            clear_color: DEFAULT_CLEAR_COLOR,
            lighting: Lighting::default(),

            mesh_pool: MeshPool::null(config.mesh_capacity, FRAMES_IN_FLIGHT as usize),
            culler: Culler::null(),
            depth_pyramid: DepthPyramid::null(),
            probe_volume: ProbeVolume::null(),
            gi_configured: config.global_illumination.is_some(),
            frame_number: 0,
            last_frame: None,
            capture_requested: false,
            capture_buffers: Vec::new(),
            recorded_capture: None,
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

            frame_desc_pool: vk::DescriptorPool::null(),
            frame_desc_set_layout: vk::DescriptorSetLayout::null(),
            frame_desc_sets: Vec::new(),
            uniform_buffers: Vec::new(),
            palette: Vec::new(),
            palette_version: 0,
            frame_palette_buffers: Vec::new(),
            frame_palette_versions: Vec::new(),
            frame_shadow_tables: Vec::new(),
            shadow_table: Vec::new(),

            mesh_pipeline: vk::Pipeline::null(),
            dynamic_pipeline: vk::Pipeline::null(),
            graphics_pipeline_layout: vk::PipelineLayout::null(),
            graphics_queue: Queue::new(&graphics_queue_family, &render_device, 0),
            graphics_queue_family,
            render_device,

            surface,
            surface_instance,
            adapter,
        };
        renderer.create(surface_extent, config)?;
        Ok(renderer)
    }

    /// Fills in everything the device makes, leaving it for `Drop` if it fails.
    fn create(&mut self, surface_extent: vk::Extent2D, config: RendererConfig) -> VkResult<()> {
        let device = self.render_device.clone();
        let adapter = self.adapter.clone();

        // Frame Descriptors, matching the bindings in `common.slang`: the uniforms, the mesh
        // pool's vertices, mesh infos and voxel types, the palette and the shadow table, then
        // the light probes, their atlases sampled, their atlases written, their rays and the
        // probes they're tracing.
        let bindings = [
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
        ];
        let pool_sizes = [
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::DescriptorType::STORAGE_IMAGE,
        ]
        .map(|ty| {
            let count = bindings.iter().filter(|&&binding| binding == ty).count() as u32;
            vk::DescriptorPoolSize::default()
                .ty(ty)
                .descriptor_count(count * FRAMES_IN_FLIGHT)
        });
        self.frame_desc_pool = create_descriptor_pool(&device, &pool_sizes, FRAMES_IN_FLIGHT)?;
        self.frame_desc_set_layout = create_descriptor_set_layout(
            &device,
            &bindings,
            vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT
                | vk::ShaderStageFlags::COMPUTE,
        )?;
        self.frame_desc_sets = allocate_descriptor_sets(
            &device,
            self.frame_desc_pool,
            self.frame_desc_set_layout,
            FRAMES_IN_FLIGHT,
        )?;
        for _ in 0..FRAMES_IN_FLIGHT {
            self.uniform_buffers
                .push(create_uniform_buffer(&adapter, &device)?);
        }
        write_uniform_descriptor_sets(&device, &self.frame_desc_sets, &self.uniform_buffers);

        // Graphics Pipelines
        self.graphics_pipeline_layout =
            create_pipeline_layout(&device, slice::from_ref(&self.frame_desc_set_layout))?;
        for (pipeline, spv, entries, vertex_input) in [
            (
                &mut self.mesh_pipeline,
                MESH_SPV,
                (c"meshVert", c"meshFrag"),
                false,
            ),
            (
                &mut self.dynamic_pipeline,
                DYNAMIC_SPV,
                (c"vertMain", c"fragMain"),
                true,
            ),
        ] {
            let shader_module = create_shader_module(&device, spv)?;
            let created = create_graphics_pipeline(
                &self.surface,
                &device,
                shader_module,
                entries,
                vertex_input,
                self.graphics_pipeline_layout,
            );
            unsafe { device.destroy_shader_module(shader_module, None) };
            *pipeline = created?;
        }

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
        if !self.surface.is_offscreen() {
            for _ in 0..self.swapchain.present_images.len() {
                self.rendered_images.push(create_semaphore(&device)?);
            }
        }

        // Light Probes, updated with the graphics pipelines' layout, it's only the frame's set.
        self.probe_volume = ProbeVolume::new(
            &adapter,
            &device,
            config.global_illumination,
            self.graphics_pipeline_layout,
        )?;
        self.probe_volume
            .write_descriptors(&device, &self.frame_desc_sets);
        self.init_probe_volume()?;

        // Meshes
        self.mesh_pool = MeshPool::new(
            &adapter,
            &device,
            config.mesh_capacity,
            FRAMES_IN_FLIGHT as usize,
        )?;
        self.culler = Culler::new(&adapter, &device, FRAMES_IN_FLIGHT)?;
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
            // Only grown when a frame is captured.
            self.capture_buffers.push(DynamicBuffer::new(
                &adapter,
                &device,
                vk::BufferUsageFlags::TRANSFER_DST,
                4,
            )?);
            self.frame_palette_buffers.push(DynamicBuffer::new(
                &adapter,
                &device,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                INITIAL_DYNAMIC_ELEMENTS * size_of::<[f32; 4]>() as vk::DeviceSize,
            )?);
            // Written by the first frame.
            self.frame_palette_versions.push(u64::MAX);
            // Grown to fit the first frame with shadows.
            self.frame_shadow_tables.push(DynamicBuffer::new(
                &adapter,
                &device,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                4,
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

/// The most preferred adapter with every feature the renderer needs that can render, cull, and present to
/// the `App`'s window unless it's headless. Culling runs on the graphics queue, so it must
/// also support compute.
fn pick_adapter(
    app: &App,
    surface_instance: &khr::surface::Instance,
) -> VkResult<(Adapter, QueueFamily)> {
    for adapter in app.adapters_by_preference() {
        if !adapter.supports_renderer() {
            continue;
        }
        let queue_family = adapter.get_queue_family(
            surface_instance,
            app.surface(),
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
        )?;
        if let Some(queue_family) = queue_family {
            return Ok((adapter.clone(), queue_family));
        }
    }
    Err(vk::Result::ERROR_INITIALIZATION_FAILED)
}
