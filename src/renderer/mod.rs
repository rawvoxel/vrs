mod capture;
mod config;
mod debug;
mod device;
mod frame;
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
    mesh::{Mesh, MeshId, MeshPool},
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

    /// Adds a mesh to the shared mesh buffers, uploaded by the next `submit`.
    /// Indices are relative to the mesh's own first vertex, and an empty mesh draws nothing.
    ///
    /// Its data is staged right away, which first waits for the device to finish the frame
    /// that last used the next frame's staging buffer. That's normally long done.
    ///
    /// When the pool's out of room it waits for the device to go idle, freeing destroyed
    /// meshes' space, then defragments, or grows when `grow_mesh_pool` allows. Those stall,
    /// but are rare. Fails with `ERROR_OUT_OF_DEVICE_MEMORY` when none of it makes room.
    pub fn create_mesh(&mut self, indices: &[Index], vertices: &[Vertex]) -> VkResult<MeshId> {
        let frame = self.current_frame;
        self.wait_for_frame(frame)?;
        let out_of_room = |result: &VkResult<MeshId>| {
            matches!(result, Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))
        };

        let created = self.stage_mesh(frame, indices, vertices);
        if !out_of_room(&created) {
            return created;
        }
        // Frames in flight may still hold destroyed meshes' space.
        let mut created = created;
        if self.mesh_pool.has_pending_releases() {
            unsafe { self.render_device.device_wait_idle()? };
            self.mesh_pool.release_all();
            created = self.stage_mesh(frame, indices, vertices);
            if !out_of_room(&created) {
                return created;
            }
        }

        let plan = self.mesh_pool.plan_repack(
            vertices.len() as u32,
            indices.len() as u32,
            self.grow_mesh_pool,
        );
        let Some(capacity) = plan else {
            return created;
        };
        unsafe {
            self.render_device.device_wait_idle()?;
            self.mesh_pool.repack(
                &self.adapter,
                &self.render_device,
                self.graphics_queue.handle,
                self.init_cmd_buffer,
                self.init_cmd_fence,
                capacity,
            )?;
        }
        self.stage_mesh(frame, indices, vertices)
    }

    #[inline]
    fn stage_mesh(
        &mut self,
        frame: usize,
        indices: &[Index],
        vertices: &[Vertex],
    ) -> VkResult<MeshId> {
        // SAFETY: `create_mesh` waited for `frame`, which readies its staging buffer.
        unsafe {
            self.mesh_pool
                .create(&self.adapter, &self.render_device, frame, indices, vertices)
        }
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
