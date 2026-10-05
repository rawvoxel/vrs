mod config;
pub(crate) mod culling;
mod debug;
mod device;
mod draw_list;
mod frame;
pub(crate) mod lighting;
mod material;
pub(crate) mod mesh_pool;
mod record;
mod setup;
pub(crate) mod shader;
pub(crate) mod timings;

pub use config::RendererConfig;
pub use debug::{Capture, FrameStats};
pub use draw_list::DrawList;
pub(crate) use lighting::LightingUniforms;
pub use lighting::{
    DebugView, Light, Lighting, MAX_CASCADES, MAX_GRID_LEVELS, MAX_LIGHTS, MAX_SHADOWED_LIGHTS,
    ProbeConfig, ProbeTuning, Sky, Spot, Tonemapping, TraceResolution, VoxelGrid,
};
pub use material::Material;
pub use mesh_pool::{MeshCapacity, MeshKind, MeshPoolStats, MeshRecord};
pub use timings::{GpuStage, GpuTimings};

use crate::{
    mesh::MeshId,
    renderer::{
        culling::{Culler, depth_pyramid::DepthPyramid},
        lighting::{lights::GpuLight, probes::ProbeVolume, trace::TracePass},
        mesh_pool::MeshPool,
        timings::FrameTimestamps,
    },
    vulkan::{
        adapter::Adapter,
        buffer::{Buffer, DynamicBuffer},
        commands::record_submit_command_buffer,
        queue::{Queue, QueueFamily},
        surface::Surface,
        swapchain::Swapchain,
        sync::create_semaphore,
    },
};
use ash::{khr, prelude::VkResult, vk};
use glam::IVec4;
use std::collections::HashMap;

/// Draws meshes into the `App`'s window, or images of its own when headless,
/// culling them on the device.
pub struct Renderer {
    /// Whether meshes hidden behind others are culled, starting as `RendererConfig::occlusion_culling`.
    /// Off, meshes are only frustum culled and drawn in one pass. Changes take effect next frame.
    pub occlusion_culling: bool,
    /// Whether triangles facing away are skipped, starting as `RendererConfig::backface_culling`.
    /// Front faces wind counter-clockwise on screen. Changes take effect next frame.
    pub backface_culling: bool,
    /// Whether the mesh pool grows when it's out of room, starting as `RendererConfig::grow_mesh_pool`.
    /// Either way it defragments when that makes room.
    pub grow_mesh_pool: bool,
    /// What each frame starts as, in linear color.
    pub clear_color: [f32; 4],
    /// How chunk meshes are lit. Changes take effect next frame.
    pub lighting: Lighting,
    /// The point and spot lights lighting chunk meshes, and the light probes, the first
    /// `MAX_LIGHTS` of them. Each pixel's brightest `Lighting::shadowed_lights` are shadowed,
    /// in the `Lighting::voxel_grid`, at the `Lighting::trace_resolution`. With any, frames
    /// draw their depth first, as with shadows. Changes take effect next frame.
    pub lights: Vec<Light>,

    mesh_pool: MeshPool,
    /// Turns each frame's meshes into indirect draws for the visible ones.
    culler: Culler,
    /// Built from each frame's depth between its two culling phases.
    depth_pyramid: DepthPyramid,
    /// The light probes, a single unused one when there's no `RendererConfig::global_illumination`.
    probe_volume: ProbeVolume,
    /// Traces shadows and reflections, and finds the light from everywhere but the sun, from
    /// each frame's depth, see `uses_trace_pass`.
    trace_pass: TracePass,
    /// Whether there's a `RendererConfig::global_illumination`.
    probes_configured: bool,
    /// Frames submitted, which seeds each frame's probe rays.
    frame_number: u64,
    /// What `debug_frame_stats` reads back.
    last_frame: Option<debug::LastFrame>,
    /// Timestamps between each frame in flight's stages, null pools when the queue has none.
    frame_timestamps: Vec<FrameTimestamps>,
    /// The last finished frame's, see `gpu_timings`.
    gpu_timings: Option<GpuTimings>,
    /// Whether the next frame is copied into `capture_buffers`.
    capture_requested: bool,
    capture_buffers: Vec<DynamicBuffer>,
    /// What `debug_read_capture` reads back.
    recorded_capture: Option<debug::RecordedCapture>,

    current_frame: usize,
    in_flight_frames: Vec<vk::Fence>,
    /// Signaled when the frame's swapchain image is acquired, unused when headless.
    available_frames: Vec<vk::Semaphore>,
    /// One per swapchain image, the presentation engine may hold onto it until that image
    /// is reacquired. Empty when headless.
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

    frame_desc_pool: vk::DescriptorPool,
    frame_desc_set_layout: vk::DescriptorSetLayout,
    frame_desc_sets: Vec<vk::DescriptorSet>,
    uniform_buffers: Vec<Buffer>,
    /// Materials by voxel type, see `set_materials`.
    palette: Vec<material::GpuMaterial>,
    palette_version: u64,
    /// Whether any material is glossy, for the trace pass to trace reflections.
    glossy_palette: bool,
    frame_palette_buffers: Vec<DynamicBuffer>,
    /// The palette version each frame's buffer holds.
    frame_palette_versions: Vec<u64>,
    /// Which chunk is where around the camera, for shadow rays, see `write_lighting`.
    frame_chunk_tables: Vec<DynamicBuffer>,
    /// Reused to build each frame's chunk table in.
    chunk_table: Vec<u32>,
    /// Which chunk table entries' chunks changed since last frame, a bit each, see
    /// `write_lighting`.
    frame_changed_chunks: Vec<DynamicBuffer>,
    /// Reused to build each frame's changed chunks in.
    changed_chunks: Vec<u32>,
    /// Each frame's `lights`.
    frame_light_buffers: Vec<DynamicBuffer>,
    /// Reused to build each frame's lights in.
    gpu_lights: Vec<GpuLight>,
    /// The serial of each chunk in last frame's chunk table, by chunk coordinate, then level.
    chunk_serials: HashMap<IVec4, u64>,
    /// This frame's, built while the chunk table is.
    chunk_serials_next: HashMap<IVec4, u64>,
    /// Last frame's lighting, to notice changes in it.
    previous_lighting: Option<Lighting>,
    /// The faster converging probes still under way.
    convergence: lighting::probes::Convergence,
    /// This frame's multipliers of the probes' hysteresis for light and distances.
    hysteresis_scale: [f32; 2],

    /// Draws the mesh pool's meshes, pulling their vertices.
    mesh_pipeline: vk::Pipeline,
    /// Draws the mesh pool's meshes' depth alone, for the trace pass.
    mesh_depth_pipeline: vk::Pipeline,
    /// Draws the mesh pool's meshes' color over their depth already drawn.
    mesh_over_depth_pipeline: vk::Pipeline,
    /// The same, when the trace pass lit the light from everywhere but the sun.
    mesh_upscaled_indirect_pipeline: vk::Pipeline,
    /// Draws the light probes, for `DebugView::Probes`.
    probe_overlay_pipeline: vk::Pipeline,
    /// Draws the `Lighting::sky` where no mesh is.
    sky_pipeline: vk::Pipeline,
    graphics_pipeline_layout: vk::PipelineLayout,
    graphics_queue: Queue,
    graphics_queue_family: QueueFamily,
    device: ash::Device,

    surface: Surface,
    surface_instance: khr::surface::Instance,
    adapter: Adapter,
}

impl Renderer {
    /// The adapter the renderer chose.
    #[inline]
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    /// Whether the renderer draws into images of its own, for a headless `App`.
    #[inline]
    pub fn is_headless(&self) -> bool {
        self.surface.is_offscreen()
    }

    /// Whether presenting waits for the display to refresh, never when headless.
    #[inline]
    pub fn vsync(&self) -> bool {
        !self.is_headless() && self.surface.present_mode == vk::PresentModeKHR::FIFO
    }

    /// Rebuilds the swapchain with or without vsync next frame. Headless, there's nothing to sync to.
    pub fn set_vsync(&mut self, vsync: bool) {
        let present_mode = self.surface.present_mode;
        self.surface.set_vsync(vsync);
        self.swapchain_outdated |= self.surface.present_mode != present_mode;
    }

    /// Renders and presents a frame, skipping it while the window is minimized.
    ///
    /// Fails with `ERROR_TOO_MANY_OBJECTS` when `draws` lists more meshes than the device's
    /// `maxDrawIndirectCount`, which is `u32::MAX` on desktop drivers.
    pub fn submit(
        &mut self,
        window_width: u32,
        window_height: u32,
        draws: &DrawList,
    ) -> VkResult<()> {
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
        self.prepare_trace_pass()?;
        self.track_lighting_changes();

        // Every visible mesh is drawn by one counted indirect call.
        if draws.meshes.len() > self.adapter.properties.limits.max_draw_indirect_count as usize {
            return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
        }

        let frame = self.current_frame;
        let Some(image_index) = self.acquire_image(frame, surface_extent)? else {
            return Ok(());
        };
        let data = self.write_frame_data(frame, draws)?;
        self.record_frame(frame, image_index, &data)?;
        self.submit_and_present(frame, image_index)?;
        self.last_frame = Some(debug::LastFrame {
            frame,
            meshes: data.candidates.len() as u32,
        });

        self.current_frame = (self.current_frame + 1) % self.in_flight_frames.len();
        self.frame_number += 1;
        Ok(())
    }

    fn rebuild_swapchain(&mut self, extent: vk::Extent2D) -> VkResult<()> {
        unsafe {
            self.device.device_wait_idle()?;

            self.swapchain.rebuild(
                &self.adapter,
                &self.surface,
                extent,
                &self.graphics_queue_family,
                &self.device,
                &self.swapchain_device,
            )?;
            self.swapchain_outdated = false;

            // The device is idle, so no semaphore is still waiting to be signaled or presented.
            let image_count = match self.is_headless() {
                true => 0,
                false => self.swapchain.present_images.len(),
            };
            if self.rendered_images.len() != image_count {
                for semaphore in self.rendered_images.drain(..) {
                    self.device.destroy_semaphore(semaphore, None);
                }
                for _ in 0..image_count {
                    let semaphore = create_semaphore(&self.device)?;
                    self.rendered_images.push(semaphore);
                }
            }

            self.depth_pyramid.resize(
                &self.adapter,
                &self.device,
                self.swapchain.depth_image_view,
                self.swapchain.extent,
            )?;
        }
        self.init_depth_pyramid_layout()
    }

    /// Readies a new depth pyramid, waiting for the device.
    fn init_depth_pyramid_layout(&self) -> VkResult<()> {
        self.run_init_commands(|cmd_buffer| unsafe {
            self.depth_pyramid.cmd_init_layout(&self.device, cmd_buffer)
        })
    }

    /// Readies new trace pass images, waiting for the device.
    fn init_trace_pass(&self) -> VkResult<()> {
        self.run_init_commands(|cmd_buffer| unsafe {
            self.trace_pass.cmd_init(&self.device, cmd_buffer)
        })
    }

    /// Readies the light probes, waiting for the device.
    fn init_probe_volume(&self) -> VkResult<()> {
        self.run_init_commands(|cmd_buffer| unsafe {
            self.probe_volume.cmd_init(&self.device, cmd_buffer)
        })
    }

    /// Records `record` into the init command buffer, submits it and waits for it to finish.
    fn run_init_commands(&self, record: impl FnOnce(vk::CommandBuffer)) -> VkResult<()> {
        record_submit_command_buffer(
            &self.device,
            self.init_cmd_buffer,
            self.init_cmd_fence,
            self.graphics_queue.handle,
            &[],
            &[],
            &[],
            record,
        )?;
        unsafe {
            self.device
                .wait_for_fences(&[self.init_cmd_fence], true, u64::MAX)
        }
    }
}

/// Must be dropped before the `App` it was created from.
/// Anything `create` didn't get to is still null, which Vulkan destroys as a no-op.
impl Drop for Renderer {
    fn drop(&mut self) {
        let device = &self.device;
        unsafe {
            // NOTE: Nothing can be done about a lost device here, so destroy regardless.
            let _ = device.device_wait_idle();

            self.mesh_pool.destroy_buffers(device);
            self.culler.destroy(device);
            self.depth_pyramid.destroy(device);
            self.probe_volume.destroy(device);
            self.trace_pass.destroy(device);
            for buffer in self
                .capture_buffers
                .iter()
                .chain(&self.frame_palette_buffers)
                .chain(&self.frame_chunk_tables)
                .chain(&self.frame_changed_chunks)
                .chain(&self.frame_light_buffers)
            {
                buffer.destroy(device);
            }

            for timestamps in &self.frame_timestamps {
                timestamps.destroy(device);
            }
            for buffer in self.uniform_buffers.iter() {
                buffer.destroy(device);
            }
            device.destroy_descriptor_set_layout(self.frame_desc_set_layout, None);
            device.destroy_descriptor_pool(self.frame_desc_pool, None);

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

            device.destroy_pipeline(self.mesh_pipeline, None);
            device.destroy_pipeline(self.mesh_depth_pipeline, None);
            device.destroy_pipeline(self.mesh_over_depth_pipeline, None);
            device.destroy_pipeline(self.mesh_upscaled_indirect_pipeline, None);
            device.destroy_pipeline(self.probe_overlay_pipeline, None);
            device.destroy_pipeline(self.sky_pipeline, None);
            device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            device.destroy_device(None);
        }
    }
}
