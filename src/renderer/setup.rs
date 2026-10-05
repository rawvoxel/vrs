//! Building a `Renderer`.

use super::{Lighting, Renderer, RendererConfig, device::create_device, material::GpuMaterial};
use crate::{
    App,
    renderer::{
        culling::{Culler, depth_pyramid::DepthPyramid},
        lighting::{lights::GpuLight, probes::ProbeVolume, trace::TracePass},
        mesh_pool::MeshPool,
        shader::{
            bindings,
            uniforms::{create_uniform_buffer, write_uniform_descriptor_sets},
        },
        timings::FrameTimestamps,
    },
    vulkan::{
        adapter::Adapter,
        buffer::DynamicBuffer,
        commands::{create_command_buffer, create_command_pool},
        descriptors::{
            allocate_descriptor_sets, create_descriptor_pool, create_descriptor_set_layout,
        },
        pipeline::{
            PipelinePass, create_graphics_pipeline, create_pipeline_layout, create_shader_module,
        },
        queue::{Queue, QueueFamily},
        surface::Surface,
        swapchain::Swapchain,
        sync::{create_fence, create_semaphore},
    },
};
use ash::{khr, prelude::VkResult, vk};
use std::slice;

const MESH_SPV: &[u8] = include_bytes!("../../assets/shader/mesh.spv");
const PROBE_OVERLAY_SPV: &[u8] = include_bytes!("../../assets/shader/probe_overlay.spv");
const SKY_PASS_SPV: &[u8] = include_bytes!("../../assets/shader/sky_pass.spv");
pub(super) const FRAMES_IN_FLIGHT: u32 = 2;
/// Materials each frame's palette buffer starts with room for, it grows to fit.
const INITIAL_MATERIALS: vk::DeviceSize = 1024;
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
        let device = create_device(
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
            lights: Vec::new(),

            mesh_pool: MeshPool::null(config.mesh_capacity, FRAMES_IN_FLIGHT as usize),
            culler: Culler::null(),
            depth_pyramid: DepthPyramid::null(),
            probe_volume: ProbeVolume::null(),
            trace_pass: TracePass::null(),
            probe_config: config.global_illumination,
            frame_number: 0,
            last_frame: None,
            frame_timestamps: Vec::new(),
            gpu_timings: None,
            capture_requested: false,
            capture_buffers: Vec::new(),
            recorded_capture: None,

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
            swapchain_device: khr::swapchain::Device::new(&app.instance, &device),
            swapchain_outdated: false,

            frame_desc_pool: vk::DescriptorPool::null(),
            frame_desc_set_layout: vk::DescriptorSetLayout::null(),
            frame_desc_sets: Vec::new(),
            uniform_buffers: Vec::new(),
            palette: Vec::new(),
            palette_version: 0,
            glossy_palette: false,
            frame_palette_buffers: Vec::new(),
            frame_palette_versions: Vec::new(),
            frame_chunk_tables: Vec::new(),
            chunk_table: Vec::new(),
            grid_chunks: Vec::new(),
            frame_changed_probes: Vec::new(),
            changed_probes: Vec::new(),
            frame_light_buffers: Vec::new(),
            gpu_lights: Vec::new(),
            chunk_serials: Default::default(),
            chunk_serials_next: Default::default(),
            previous_lighting: None,
            convergence: super::lighting::probes::Convergence::NONE,
            hysteresis_scale: [1.0; 2],

            mesh_pipeline: vk::Pipeline::null(),
            mesh_depth_pipeline: vk::Pipeline::null(),
            mesh_over_depth_pipeline: vk::Pipeline::null(),
            mesh_upscaled_indirect_pipeline: vk::Pipeline::null(),
            probe_overlay_pipeline: vk::Pipeline::null(),
            sky_pipeline: vk::Pipeline::null(),
            graphics_pipeline_layout: vk::PipelineLayout::null(),
            graphics_queue: Queue::new(&graphics_queue_family, &device, 0),
            graphics_queue_family,
            device,

            surface,
            surface_instance,
            adapter,
        };
        renderer.create(surface_extent, config)?;
        Ok(renderer)
    }

    /// Fills in everything the device makes, leaving it for `Drop` if it fails.
    fn create(&mut self, surface_extent: vk::Extent2D, config: RendererConfig) -> VkResult<()> {
        let device = self.device.clone();
        let adapter = self.adapter.clone();

        // Frame Descriptors, see `bindings`.
        let bindings = bindings::TYPES_BY_BINDING;
        let pool_sizes = [
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::SAMPLED_IMAGE,
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
        let mesh_entries = (c"meshVert", c"meshFrag");
        for (pipeline, spv, entries, pass) in [
            (
                &mut self.mesh_pipeline,
                MESH_SPV,
                mesh_entries,
                PipelinePass::ColorAndDepth,
            ),
            (
                &mut self.mesh_depth_pipeline,
                MESH_SPV,
                mesh_entries,
                PipelinePass::DepthOnly,
            ),
            (
                &mut self.mesh_over_depth_pipeline,
                MESH_SPV,
                mesh_entries,
                PipelinePass::ColorOverDepth,
            ),
            (
                &mut self.mesh_upscaled_indirect_pipeline,
                MESH_SPV,
                mesh_entries,
                PipelinePass::ColorOverUpscaledIndirect,
            ),
            (
                &mut self.probe_overlay_pipeline,
                PROBE_OVERLAY_SPV,
                (c"probeVert", c"probeFrag"),
                PipelinePass::ColorAndDepth,
            ),
            (
                &mut self.sky_pipeline,
                SKY_PASS_SPV,
                (c"skyVert", c"skyFrag"),
                PipelinePass::Behind,
            ),
        ] {
            let shader_module = create_shader_module(&device, spv)?;
            let created = create_graphics_pipeline(
                &self.surface,
                &device,
                shader_module,
                entries,
                pass,
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
        let timestamps = self.graphics_queue_family.properties.timestamp_valid_bits > 0
            && adapter.properties.limits.timestamp_compute_and_graphics == vk::TRUE;
        for _ in 0..FRAMES_IN_FLIGHT {
            let pool = create_command_pool(&device, &self.graphics_queue_family)?;
            self.frame_cmd_pools.push(pool);
            self.frame_cmd_buffers.push(create_command_buffer(
                &device,
                pool,
                vk::CommandBufferLevel::PRIMARY,
            )?);
            self.in_flight_frames.push(create_fence(&device)?);
            self.frame_timestamps.push(if timestamps {
                FrameTimestamps::new(&device)?
            } else {
                FrameTimestamps::null()
            });
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

        // The trace pass, with the graphics pipelines' layout too.
        self.trace_pass = TracePass::new(&adapter, &device, self.graphics_pipeline_layout)?;
        self.init_trace_pass()?;

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

        // Per Frame Buffers, grown to fit what each frame writes.
        let storage_buffer = |size: usize| {
            DynamicBuffer::new(
                &adapter,
                &device,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                size as vk::DeviceSize,
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
                INITIAL_MATERIALS * size_of::<GpuMaterial>() as vk::DeviceSize,
            )?);
            // Written by the first frame.
            self.frame_palette_versions.push(u64::MAX);
            self.frame_chunk_tables.push(storage_buffer(4)?);
            self.frame_changed_probes.push(storage_buffer(4)?);
            self.frame_light_buffers
                .push(storage_buffer(size_of::<GpuLight>())?);
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
