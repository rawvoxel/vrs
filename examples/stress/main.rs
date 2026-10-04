//! Exercises the renderer the way a voxel game would, checking what it draws.
//!
//! Run with `cargo run -p vrs --example stress`, it exits with a failure code if a check fails.
//!
//! A grid of chunk meshes sits behind a wall mesh. Scenarios cull it with different cameras
//! and draw paths, remesh chunks every frame against a nearly full mesh pool, and resize and
//! minimize the window. Each one compares the draw counts read back from the device with the
//! meshes that are actually in view, and with occlusion culling, those the wall can't hide.
//!
//! Debug builds also fail on any Vulkan validation message, with synchronization validation
//! turned on, when the validation layer is installed. It can't see which vertices and indices
//! an indirect draw reads though, so a missing barrier before those reads goes unnoticed.

mod scenarios;
mod scene;

use anyhow::{Result, bail};
use ash::vk;
use glam::{Mat4, Vec2};
use scenarios::{Expect, SCENARIOS};
use scene::{CAPACITY, GRID, Scene, WALL_HALF_SIZE, WALL_Z, chunk, live, quad};
use std::time::Instant;
use vrs::{
    App, debug,
    mesh::MeshCapacity,
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

/// Frames after a scenario starts before its meshes are created, so the last
/// scenario's destroyed meshes are released first.
const SETTLE_FRAMES: u32 = 3;
/// Chunks remeshed per frame when remeshing.
const REMESH_PER_FRAME: u32 = 24;

#[derive(Default)]
struct Stress {
    window: Option<Window>,
    scene: Option<Scene>,
    scenario: usize,
    frame: u32,
    out_of_memory: u32,
    started: Option<Instant>,
    messages_at_start: debug::MessageCounts,
    failures: Vec<String>,
}

impl Stress {
    fn fail(&mut self, failure: String) {
        println!("  FAIL {failure}");
        self.failures.push(failure);
    }

    fn check_messages(&mut self, what: &str) {
        let counts = debug::message_counts();
        let errors = counts.errors - self.messages_at_start.errors;
        let warnings = counts.warnings - self.messages_at_start.warnings;
        if errors > 0 || warnings > 0 {
            self.fail(format!(
                "{what}: {errors} validation errors and {warnings} warnings"
            ));
        }
        self.messages_at_start = counts;
    }

    /// Building a renderer that fails partway must destroy whatever it created,
    /// which validation reports otherwise when the device and instance are destroyed.
    fn check_failed_construction(&mut self, app: &App, size: PhysicalSize<u32>) {
        println!("failed construction");
        for capacity in [
            MeshCapacity {
                vertices: u32::MAX,
                ..CAPACITY
            },
            MeshCapacity {
                indices: u32::MAX,
                ..CAPACITY
            },
        ] {
            match Renderer::with_mesh_capacity(app, size.width, size.height, capacity) {
                Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => (),
                Err(err) => self.fail(format!("impossible capacity failed with {err}")),
                Ok(_) => self.fail("impossible capacity was created".to_string()),
            }
        }
        self.check_messages("failed construction");
    }

    fn start_scenario(&mut self) {
        let scenario = &SCENARIOS[self.scenario];
        println!("{}", scenario.name);
        let scene = self.scene.as_mut().unwrap();
        scene.clear();
        scene.renderer.occlusion_culling = scenario.occlusion;
        let (compact, max_batch) = scenario.draw_paths;
        scene.renderer.debug_limit_draw_paths(compact, max_batch);
        self.frame = 0;
        self.out_of_memory = 0;
        self.started = Some(Instant::now());
    }

    fn fill_scene(&mut self) {
        let scenario = &SCENARIOS[self.scenario];
        let scene = self.scene.as_mut().unwrap();
        if scenario.wall {
            let min = Vec2::splat(-WALL_HALF_SIZE);
            scene.wall = Some(
                quad(
                    &mut scene.renderer,
                    min,
                    2.0 * WALL_HALF_SIZE,
                    WALL_Z,
                    [0.9, 0.8, 0.1],
                )
                .unwrap(),
            );
        }
        for cell in 0..GRID * GRID {
            scene.chunks[cell as usize] = Some(chunk(&mut scene.renderer, cell, 0).unwrap());
        }
    }

    fn remesh(&mut self, window: &Window) {
        let frame = self.frame;
        let scene = self.scene.as_mut().unwrap();
        for k in 0..REMESH_PER_FRAME {
            let cell = ((frame * 37 + k * 11) % (GRID * GRID)) as usize;
            if let Some(old) = scene.chunks[cell].take() {
                scene.renderer.destroy_mesh(old);
            }
            match chunk(&mut scene.renderer, cell as u32, frame) {
                Ok(id) => scene.chunks[cell] = Some(id),
                Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => self.out_of_memory += 1,
                Err(err) => panic!("Creating a mesh failed with {err}."),
            }
        }
        // A chunk destroyed before it's ever uploaded.
        if let Ok(id) = chunk(&mut scene.renderer, 1, frame + 1) {
            scene.renderer.destroy_mesh(id);
        }
        match frame % 120 {
            30 => drop(window.request_inner_size(PhysicalSize::new(500, 700))),
            60 => window.set_minimized(true),
            75 => window.set_minimized(false),
            90 => drop(window.request_inner_size(PhysicalSize::new(900, 400))),
            _ => (),
        }
    }

    fn finish_scenario(&mut self, view_proj: Mat4) {
        let scenario = &SCENARIOS[self.scenario];
        let scene = self.scene.as_mut().unwrap();
        let stats = scene.renderer.debug_frame_stats().unwrap().unwrap();
        let (in_view, must_draw) = scene.expected(view_proj);
        let elapsed = self.started.unwrap().elapsed();
        println!(
            "  {} frames in {:.1?}, {} meshes, {in_view} in view, {must_draw} the wall can't hide, drawn {:?}{}",
            scenario.frames,
            elapsed,
            stats.meshes,
            stats.drawn,
            match scenario.remesh {
                true => format!(", the pool was full {} times", self.out_of_memory),
                false => String::new(),
            },
        );

        let drawn = stats.total_drawn();
        let failure =
            match (scenario.expect, drawn) {
                (Expect::Uncounted, None) => None,
                (Expect::Uncounted, Some(_)) => {
                    Some("draws were counted on an uncounted draw path".to_string())
                }
                (_, None) => Some("draws weren't counted".to_string()),
                (Expect::FrustumOnly, Some(drawn)) => (drawn != in_view)
                    .then(|| format!("drew {drawn} meshes, {in_view} are in view")),
                (Expect::Occluded, Some(drawn)) => (drawn < must_draw || drawn > in_view)
                    .then(|| format!("drew {drawn} meshes, expected {must_draw} to {in_view}")),
                (Expect::OnlyWall, Some(drawn)) => (drawn != 1)
                    .then(|| format!("drew {drawn} meshes, the wall hides all but itself")),
            };
        if let Some(failure) = failure {
            self.fail(format!("{}: {failure}", scenario.name));
        }
        if scenario.remesh && self.out_of_memory == 0 {
            self.fail(format!("{}: the mesh pool never filled up", scenario.name));
        }
        self.check_messages(scenario.name);
    }
}

impl ApplicationHandler for Stress {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("vrs - stress")
            .with_inner_size(PhysicalSize::new(800, 600));
        let window = event_loop.create_window(attributes).unwrap();
        let size = window.inner_size();
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"vrs stress")
            .api_version(vk::API_VERSION_1_3);
        let app = App::new(
            window.display_handle().unwrap().as_raw(),
            window.window_handle().unwrap().as_raw(),
            app_info,
        )
        .unwrap();
        if app.debug.is_none() {
            println!("Validation is off, only draw counts are checked.");
        }

        self.messages_at_start = debug::message_counts();
        self.check_failed_construction(&app, size);

        let renderer =
            Renderer::with_mesh_capacity(&app, size.width, size.height, CAPACITY).unwrap();
        self.scene = Some(Scene {
            chunks: (0..GRID * GRID).map(|_| None).collect(),
            wall: None,
            renderer,
            app,
        });
        self.window = Some(window);
        self.start_scenario();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.fail("the window was closed early".to_string());
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let Some(window) = self.window.take() else {
                    return;
                };
                let scenario = &SCENARIOS[self.scenario];
                self.frame += 1;
                if self.frame == SETTLE_FRAMES {
                    self.fill_scene();
                } else if self.frame > SETTLE_FRAMES && scenario.remesh {
                    self.remesh(&window);
                }

                let size = window.inner_size();
                let (view, proj) = scenario
                    .camera
                    .view_proj(size.width as f32 / size.height.max(1) as f32);
                let scene = self.scene.as_mut().unwrap();
                let live = live(&scene.chunks, &scene.wall);
                let draws = DrawList {
                    meshes: &live,
                    view,
                    proj,
                    ..Default::default()
                };
                scene
                    .renderer
                    .submit(size.width, size.height, &draws)
                    .unwrap();

                if self.frame == SETTLE_FRAMES + scenario.frames {
                    self.finish_scenario(proj * view);
                    self.scenario += 1;
                    if self.scenario == SCENARIOS.len() {
                        event_loop.exit();
                    } else {
                        self.start_scenario();
                    }
                }
                self.window = Some(window);
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
        // The renderer must be dropped before the app.
        if let Some(Scene { renderer, app, .. }) = self.scene.take() {
            drop(renderer);
            drop(app);
            self.check_messages("shutting down");
        }
    }
}

fn main() -> Result<()> {
    // Synchronization validation catches missing barriers, which is most of what can go wrong.
    for (name, value) in [
        ("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "true"),
        (
            "VK_LAYER_ENABLES",
            "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT",
        ),
    ] {
        if std::env::var_os(name).is_none() {
            // SAFETY: No other threads exist yet.
            unsafe { std::env::set_var(name, value) };
        }
    }

    let mut stress = Stress::default();
    EventLoop::new()?.run_app(&mut stress)?;

    if stress.scenario < SCENARIOS.len() {
        stress.failures.push(format!(
            "stopped after {} of {} scenarios",
            stress.scenario,
            SCENARIOS.len()
        ));
    }
    match stress.failures.len() {
        0 => {
            println!("All {} scenarios passed.", SCENARIOS.len());
            Ok(())
        }
        failed => bail!("{failed} checks failed: {:#?}", stress.failures),
    }
}
