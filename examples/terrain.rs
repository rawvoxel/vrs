//! Rolling hills of greedy meshed voxel chunks to fly around, colored per voxel, ringed by
//! coarser chunks of the next level of detail.
//!
//! Run with `cargo run -p vrs --example terrain --release`, debug builds take a few seconds
//! to mesh the world. Click to look around with the mouse, Escape to let go of it. WASD
//! moves, Space and Ctrl go up and down, and Shift moves faster. L drops a light where the
//! camera is, Backspace takes them all away.
//!
//! The title shows how many chunks were drawn of those in the frame, chunks behind hills
//! are culled, then frames a second and how long frames take on the GPU: updating the light
//! probes, tracing shadows, and drawing everything else, averaged over the last second.

#[path = "common/camera.rs"]
mod camera;
#[path = "common/greedy.rs"]
mod greedy;
#[path = "common/terrain.rs"]
mod terrain;
#[path = "common/title.rs"]
mod title;

use anyhow::Result;
use ash::vk;
use glam::Vec3;
use std::{collections::HashSet, time::Instant};
use vrs::{
    App,
    mesh::MeshId,
    renderer::{DrawList, Light, Renderer},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{CursorGrabMode, Window, WindowId},
};

/// Voxels a second.
const SPEED: f32 = 40.0;
const FAST_SPEED: f32 = 160.0;
/// Radians per pixel of mouse movement.
const LOOK_SPEED: f32 = 0.003;
/// The colors dropped lights take turns at, in linear color.
const LIGHT_COLORS: [[f32; 3]; 4] = [
    [1.0, 0.55, 0.2],
    [0.3, 0.6, 1.0],
    [1.0, 0.3, 0.6],
    [0.5, 1.0, 0.4],
];
/// Dropped lights' brightness at their center, and how far their light reaches, in voxels.
const LIGHT_BRIGHTNESS: f32 = 300.0;
const LIGHT_RANGE: f32 = 48.0;

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn look(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * LOOK_SPEED;
        self.pitch = (self.pitch - dy * LOOK_SPEED).clamp(-1.5, 1.5);
    }

    fn fly(&mut self, keys: &HashSet<KeyCode>, seconds: f32) {
        let forward = terrain::forward(self.yaw, 0.0);
        let right = forward.cross(Vec3::Y);
        let mut direction = Vec3::ZERO;
        for (key, step) in [
            (KeyCode::KeyW, forward),
            (KeyCode::KeyS, -forward),
            (KeyCode::KeyD, right),
            (KeyCode::KeyA, -right),
            (KeyCode::Space, Vec3::Y),
            (KeyCode::ControlLeft, -Vec3::Y),
        ] {
            if keys.contains(&key) {
                direction += step;
            }
        }
        let speed = match keys.contains(&KeyCode::ShiftLeft) {
            true => FAST_SPEED,
            false => SPEED,
        };
        self.position += direction.normalize_or_zero() * speed * seconds;
    }
}

struct Vulkan {
    chunks: Vec<MeshId>,
    // Fields drop in order, so the renderer drops before the app.
    renderer: Renderer,
    _app: App,
}

impl Vulkan {
    fn new(window: &Window) -> Result<Self> {
        let size = window.inner_size();
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"vrs terrain")
            .api_version(vk::API_VERSION_1_3);
        let app = App::new(
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            app_info,
        )?;
        let mut renderer = Renderer::with_config(&app, size.width, size.height, terrain::config())?;
        let started = Instant::now();
        let chunks = terrain::create_world(&mut renderer)?;
        println!(
            "Meshed {} chunks in {:.1?}.",
            chunks.len(),
            started.elapsed()
        );

        Ok(Self {
            chunks,
            renderer,
            _app: app,
        })
    }
}

struct Terrain {
    camera: Camera,
    keys: HashSet<KeyCode>,
    mouse_captured: bool,
    last_frame: Instant,
    title: title::FrameTitle,
    // Fields drop in order, so Vulkan drops before the window it renders to.
    vulkan: Option<Vulkan>,
    window: Option<Window>,
}

impl Terrain {
    fn new() -> Self {
        Self {
            camera: Camera {
                position: terrain::START_POSITION,
                yaw: terrain::START_YAW,
                pitch: terrain::START_PITCH,
            },
            keys: HashSet::new(),
            mouse_captured: false,
            last_frame: Instant::now(),
            title: title::FrameTitle::new("terrain"),
            vulkan: None,
            window: None,
        }
    }

    fn capture_mouse(&mut self, capture: bool) {
        let Some(window) = &self.window else {
            return;
        };
        if capture {
            // Not every platform can lock the cursor in place.
            let grabbed = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            if grabbed.is_err() {
                return;
            }
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
        }
        window.set_cursor_visible(!capture);
        self.mouse_captured = capture;
    }

    fn drop_light(&mut self) {
        let Some(vulkan) = &mut self.vulkan else {
            return;
        };
        let lights = &mut vulkan.renderer.lights;
        let color = LIGHT_COLORS[lights.len() % LIGHT_COLORS.len()];
        lights.push(Light::point(
            self.camera.position,
            color.map(|channel| channel * LIGHT_BRIGHTNESS),
            LIGHT_RANGE,
        ));
    }

    fn redraw(&mut self) {
        let (Some(window), Some(vulkan)) = (&self.window, &mut self.vulkan) else {
            return;
        };
        let now = Instant::now();
        let seconds = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.camera.fly(&self.keys, seconds);

        let size = window.inner_size();
        // A minimized window has no surface to present to.
        if size.width == 0 || size.height == 0 {
            return;
        }
        let (view, proj) = terrain::view_proj(
            self.camera.position,
            terrain::forward(self.camera.yaw, self.camera.pitch),
            size.width as f32 / size.height as f32,
        );
        let meshes: Vec<&MeshId> = vulkan.chunks.iter().collect();
        let draws = DrawList {
            meshes: &meshes,
            view,
            proj,
            ..Default::default()
        };
        vulkan
            .renderer
            .submit(size.width, size.height, &draws)
            .expect("Could not submit frame.");

        // The draw counts are read back, stalling the device, only when the title changes.
        self.title.frame(window, &mut vulkan.renderer, |renderer| {
            let stats = renderer.debug_frame_stats().ok().flatten()?;
            Some(format!(
                "{} of {} chunks drawn",
                stats.total_drawn(),
                stats.meshes
            ))
        });
    }
}

impl ApplicationHandler for Terrain {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("vrs - terrain")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = event_loop
            .create_window(attributes)
            .expect("Could not create window.");
        self.vulkan = Some(Vulkan::new(&window).expect("Could not initialize Vulkan."));
        self.window = Some(window);
        self.last_frame = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => self.capture_mouse(true),
            WindowEvent::Focused(false) => {
                self.capture_mouse(false);
                self.keys.clear();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => match state {
                ElementState::Pressed if key == KeyCode::Escape => self.capture_mouse(false),
                ElementState::Pressed if key == KeyCode::KeyL => self.drop_light(),
                ElementState::Pressed if key == KeyCode::Backspace => {
                    if let Some(vulkan) = &mut self.vulkan {
                        vulkan.renderer.lights.clear();
                    }
                }
                ElementState::Pressed => {
                    self.keys.insert(key);
                }
                ElementState::Released => {
                    self.keys.remove(&key);
                }
            },
            _ => (),
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
            && self.mouse_captured
        {
            self.camera.look(dx as f32, dy as f32);
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
    event_loop.run_app(&mut Terrain::new())?;
    Ok(())
}
