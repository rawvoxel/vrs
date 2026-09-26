//! The classic Vulkan "hello triangle": one red, green and blue vertex.
//!
//! Run with `cargo run -p vrs --example triangle`.

use anyhow::Result;
use ash::vk;
use vrs::{
    App,
    renderer::Renderer,
    vertex::{Index, Vertex},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

const INDICES: [Index; 3] = [0, 1, 2];

#[rustfmt::skip]
const VERTICES: [Vertex; 3] = [
    Vertex { pos: [ 0.0, -0.5, 0.0], col: [1.0, 0.0, 0.0] },
    Vertex { pos: [ 0.5,  0.5, 0.0], col: [0.0, 1.0, 0.0] },
    Vertex { pos: [-0.5,  0.5, 0.0], col: [0.0, 0.0, 1.0] },
];

struct Vulkan {
    app: App,
    renderer: Renderer,
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
        let renderer = Renderer::new(&app, size.width, size.height)?;

        Ok(Self { app, renderer })
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        self.renderer
            .destroy()
            .expect("Could not destroy Renderer.");
        self.app.destroy();
    }
}

#[derive(Default)]
struct Triangle {
    // NOTE: Field order matters, Vulkan must be dropped before the window it renders to.
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
                vulkan
                    .renderer
                    .submit(size.width, size.height, &INDICES, &VERTICES)
                    .expect("Could not submit frame.");
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
    event_loop.run_app(&mut Triangle::default())?;
    Ok(())
}
