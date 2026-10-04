//! A cube of 8³ voxels, checkered in two voxel types, with the camera circling it.
//!
//! Run with `cargo run -p vrs --example cube`.
//!
//! The title shows frames a second and how long frames take on the GPU.
//!
//! Greedy meshing merges each side into one quad, yet each voxel keeps its own color:
//! the renderer looks up the voxel under every pixel in the chunk's voxel types.

#[path = "common/camera.rs"]
mod camera;
#[path = "common/greedy.rs"]
mod greedy;
#[path = "common/title.rs"]
mod title;

use anyhow::Result;
use ash::vk;
use glam::{Mat4, Vec3};
use std::time::Instant;
use vrs::{
    App,
    mesh::MeshId,
    renderer::{DrawList, Renderer},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

/// Voxels along each edge of the cube, which sits in the corner of its chunk.
const CUBE: usize = 8;
/// The cube is one unit wide.
const VOXEL_SIZE: f32 = 1.0 / CUBE as f32;
/// Linear colors by voxel type, 0 is empty space.
const PALETTE: [[f32; 3]; 3] = [[0.0; 3], [0.9, 0.35, 0.1], [0.1, 0.45, 0.9]];
/// Radians a second the camera circles the cube.
const ORBIT_SPEED: f32 = 0.6;

fn create_cube(renderer: &mut Renderer) -> Result<MeshId> {
    renderer.set_palette(&PALETTE);
    let inside = |x: i32, y: i32, z: i32| [x, y, z].iter().all(|&v| (0..CUBE as i32).contains(&v));
    let types = greedy::chunk_types(|x, y, z| match inside(x, y, z) {
        true => 1 + (x + y + z) as u16 % 2,
        false => 0,
    });
    // Centered on the origin.
    let origin = Vec3::splat(-0.5);
    let cube = greedy::create_chunk(renderer, origin, VOXEL_SIZE, &types, inside)?;
    Ok(cube.expect("The cube has voxels."))
}

struct Vulkan {
    cube: MeshId,
    // Fields drop in order, so the renderer drops before the app.
    renderer: Renderer,
    _app: App,
}

impl Vulkan {
    fn new(window: &Window) -> Result<Self> {
        let size = window.inner_size();
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"vrs cube")
            .api_version(vk::API_VERSION_1_3);
        let app = App::new(
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            app_info,
        )?;
        let mut renderer = Renderer::new(&app, size.width, size.height)?;
        let cube = create_cube(&mut renderer)?;

        Ok(Self {
            cube,
            renderer,
            _app: app,
        })
    }
}

struct Cube {
    started: Instant,
    title: title::FrameTitle,
    // Fields drop in order, so Vulkan drops before the window it renders to.
    vulkan: Option<Vulkan>,
    window: Option<Window>,
}

impl ApplicationHandler for Cube {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("vrs - cube")
            .with_inner_size(PhysicalSize::new(800, 600));
        let window = event_loop
            .create_window(attributes)
            .expect("Could not create window.");
        self.vulkan = Some(Vulkan::new(&window).expect("Could not initialize Vulkan."));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(vulkan)) = (&self.window, &mut self.vulkan) else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let size = window.inner_size();
                // A minimized window has no surface to present to.
                if size.width == 0 || size.height == 0 {
                    return;
                }
                let angle = self.started.elapsed().as_secs_f32() * ORBIT_SPEED;
                let eye = Vec3::new(2.2 * angle.sin(), 1.4, 2.2 * angle.cos());
                let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
                let aspect = size.width as f32 / size.height as f32;
                let proj = camera::perspective(60.0, aspect, 0.1, 100.0);

                let draws = DrawList {
                    meshes: &[&vulkan.cube],
                    view,
                    proj,
                    ..Default::default()
                };
                vulkan
                    .renderer
                    .submit(size.width, size.height, &draws)
                    .expect("Could not submit frame.");
                self.title.frame(window, &mut vulkan.renderer, |_| None);
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.vulkan = None;
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut Cube {
        started: Instant::now(),
        title: title::FrameTitle::new("cube"),
        vulkan: None,
        window: None,
    })?;
    Ok(())
}
