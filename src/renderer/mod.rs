mod capture;
mod config;
mod debug;
mod device;
mod frame;
mod meshes;
mod palette;
mod record;
mod setup;

pub use capture::Capture;
pub use config::RendererConfig;
pub use debug::{FrameStats, MeshPoolStats};

use crate::{
    adapter::Adapter,
    buffer::{Buffer, DynamicBuffer},
    commands::record_submit_commandbuffer,
    cull::Culler,
    hiz::DepthPyramid,
    mesh::{MeshId, MeshPool},
    queue::{Queue, QueueFamily},
    surface::Surface,
    swapchain::Swapchain,
    sync::create_semaphore,
    vertex::{Index, Vertex},
};
use ash::{khr, prelude::VkResult, vk};
use glam::Mat4;

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
    /// Into Vulkan's clip space, where y points down and depth runs from 0 to 1. An OpenGL style
    /// projection, such as glam's `perspective_rh`, needs its y flipped, otherwise the image
    /// is mirrored and backface culling skips the wrong faces.
    pub proj: Mat4,
}

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

    mesh_pool: MeshPool,
    /// Turns each frame's meshes into indirect draws for the visible ones.
    culler: Culler,
    /// Built from each frame's depth between its two culling phases.
    depth_pyramid: DepthPyramid,
    /// Most draws one indirect call can make, 1 without the `multiDrawIndirect` feature.
    indirect_draw_batch: u32,
    /// What the device supports, which `debug_limit_draw_paths` limits the above to.
    device_draw_paths: (bool, u32),
    /// What `debug_frame_stats` reads back.
    last_frame: Option<debug::LastFrame>,
    /// Whether the next frame is copied into `capture_buffers`.
    capture_requested: bool,
    capture_buffers: Vec<DynamicBuffer>,
    /// What `debug_read_capture` reads back.
    recorded_capture: Option<capture::RecordedCapture>,
    frame_index_buffers: Vec<DynamicBuffer>,
    frame_vertex_buffers: Vec<DynamicBuffer>,

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
    /// Linear colors by voxel type, see `set_palette`.
    palette: Vec<[f32; 4]>,
    palette_version: u64,
    frame_palette_buffers: Vec<DynamicBuffer>,
    /// The palette version each frame's buffer holds.
    frame_palette_versions: Vec<u64>,

    /// Draws the mesh pool's meshes, pulling their vertices.
    mesh_pipeline: vk::Pipeline,
    /// Draws each frame's own geometry from `frame_vertex_buffers`.
    dynamic_pipeline: vk::Pipeline,
    graphics_pipeline_layout: vk::PipelineLayout,
    graphics_queue: Queue,
    graphics_queue_family: QueueFamily,
    render_device: ash::Device,

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
            compact: self.compacts(data.candidates.len() as u32),
        });

        self.current_frame = (self.current_frame + 1) % self.in_flight_frames.len();
        Ok(())
    }

    /// Whether a frame with `candidate_count` meshes compacts and counts its draws.
    /// A counted draw can't be split, so too many meshes fall back to one draw each.
    #[inline]
    fn compacts(&self, candidate_count: u32) -> bool {
        self.culler.can_compact && candidate_count <= self.indirect_draw_batch
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
            let image_count = match self.is_headless() {
                true => 0,
                false => self.swapchain.present_images.len(),
            };
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
                .capture_buffers
                .iter()
                .chain(&self.frame_palette_buffers)
                .chain(&self.frame_index_buffers)
                .chain(&self.frame_vertex_buffers)
            {
                buffer.destroy(device);
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
            device.destroy_pipeline(self.dynamic_pipeline, None);
            device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            device.destroy_device(None);
        }
    }
}
