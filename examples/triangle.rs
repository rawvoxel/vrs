//! The classic Vulkan "hello triangle": one red, green and blue vertex.
//!
//! Run with `cargo run -p vrs --example triangle`.
//!
//! The title shows frames a second and how long frames take on the GPU.

#[path = "common/title.rs"]
mod title;

use anyhow::Result;
use ash::vk;
use vrs::{
    App,
    mesh::{Index, MeshId, Vertex},
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

/// Counter-clockwise on screen, so it faces the camera.
const INDICES: [Index; 3] = [0, 2, 1];

#[rustfmt::skip]
const VERTICES: [Vertex; 3] = [
    Vertex { pos: [ 0.0, -0.5, 0.0], normal: TOWARD_CAMERA, col: [1.0, 0.0, 0.0] },
    Vertex { pos: [ 0.5,  0.5, 0.0], normal: TOWARD_CAMERA, col: [0.0, 1.0, 0.0] },
    Vertex { pos: [-0.5,  0.5, 0.0], normal: TOWARD_CAMERA, col: [0.0, 0.0, 1.0] },
];
/// The default camera looks along +z.
const TOWARD_CAMERA: [f32; 3] = [0.0, 0.0, -1.0];

struct Vulkan {
    triangle: MeshId,
    // Fields drop in order, so the renderer drops before the app.
    renderer: Renderer,
    _app: App,
}

impl Vulkan {
    fn new(window: &Window) -> Result<Self> {
        let size = window.inner_size();
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"vrs triangle")
            .api_version(vk::API_VERSION_1_3);
        let app = App::new(
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            app_info,
        )?;
        let mut renderer = Renderer::new(&app, size.width, size.height)?;
        // The sun behind the camera, lighting the triangle head on.
        renderer.lighting.sun_direction = TOWARD_CAMERA.into();
        // Uploaded once, then drawn every frame.
        let triangle = renderer.create_mesh(&INDICES, &VERTICES)?;

        Ok(Self {
            triangle,
            renderer,
            _app: app,
        })
    }
}

struct Triangle {
    title: title::FrameTitle,
    // Fields drop in order, so Vulkan drops before the window it renders to.
    vulkan: Option<Vulkan>,
    window: Option<Window>,
}

impl ApplicationHandler for Triangle {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("vrs - triangle")
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
                let draws = DrawList {
                    meshes: &[&vulkan.triangle],
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
    event_loop.run_app(&mut Triangle {
        title: title::FrameTitle::new("triangle"),
        vulkan: None,
        window: None,
    })?;
    Ok(())
}
