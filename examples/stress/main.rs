//! Exercises the renderer the way a voxel game would, checking what it draws.
//!
//! Run with `cargo run -p vrs --example stress`, it exits with a failure code if a check fails.
//! Debug builds take much longer, but check for validation messages, see below.
//! `-- --headless` renders without a window, such as on a CI machine with a software
//! Vulkan driver like lavapipe. With a window, its title shows the scenario running, frames a
//! second and how long frames take on the GPU.
//!
//! First, a renderer failing partway through being built must leak nothing, and the lighting
//! is checked against scenes with known answers, see `probes`.
//!
//! Then a grid of chunk meshes sits behind a wall mesh. Scenarios cull it with different cameras,
//! remesh chunks every frame against a nearly full mesh pool, and resize and
//! minimize the window. Each one compares the draw counts read back from the device with the
//! meshes that are actually in view, and with occlusion culling, those the wall can't hide.
//! Then it captures the last frame and checks each chunk shows the color it should, which
//! catches chunks drawn from stale or missing data.
//!
//! Debug builds also fail on any Vulkan validation message, with synchronization validation
//! turned on, when the validation layer is installed. It can't see which vertices and indices
//! an indirect draw reads though, and the pixel check only sees a race the device loses,
//! so a missing barrier before those reads can still go unnoticed.

#[path = "../common/camera.rs"]
mod camera;
#[path = "../common/greedy.rs"]
mod greedy;
mod pixels;
mod probes;
mod scenarios;
mod scene;
mod target;
#[path = "../common/title.rs"]
mod title;

use anyhow::{Result, bail};
use ash::vk;
use glam::Mat4;
use scenarios::{Expect, PoolTest, SCENARIOS};
use scene::{CAPACITY, CHUNK_Z, GRID, Scene, VOXEL_SIZE, chunk_min, live, voxel_slab};
use std::time::Instant;
use target::{START_SIZE, Target};
use vrs::{
    App,
    renderer::{DrawList, MeshCapacity, MeshPoolStats, Renderer, RendererConfig},
    vulkan::debug,
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
/// Frames after the scene is filled before a pool test, so the repack has uploaded meshes
/// to copy, not just ones still waiting to upload.
const POOL_TEST_DELAY: u32 = 2;

#[derive(Default)]
struct Stress {
    target: Option<Target>,
    scene: Option<Scene>,
    /// With a window.
    title: Option<title::FrameTitle>,
    scenario: usize,
    frame: u32,
    /// Chunks that couldn't be meshed, which never happens while every chunk fits the pool.
    out_of_room: u32,
    started: Option<Instant>,
    pool_at_start: Option<MeshPoolStats>,
    messages_at_start: debug::MessageCounts,
    failures: Vec<String>,
}

impl Stress {
    fn fail(&mut self, failure: impl Into<String>) {
        let failure = failure.into();
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

    /// Checks construction, builds the scene and starts the first scenario.
    fn start(&mut self, app: App, target: Target) {
        if !app.validation_enabled() {
            println!("Validation is off, only draws and pixels are checked.");
        }
        let size = target.size();
        self.messages_at_start = debug::message_counts();
        self.check_failed_construction(&app, size);
        self.check_global_illumination(&app);

        // Only the growth scenario grows the pool, the rest keep it nearly full.
        let config = RendererConfig {
            mesh_capacity: CAPACITY,
            grow_mesh_pool: false,
            ..Default::default()
        };
        let renderer = Renderer::with_config(&app, size.width, size.height, config).unwrap();
        println!(
            "{} on {:?}",
            match renderer.is_headless() {
                true => "Headless",
                false => "Windowed",
            },
            renderer
                .adapter()
                .properties()
                .device_name_as_c_str()
                .unwrap(),
        );
        self.scene = Some(Scene::new(renderer, app));
        self.target = Some(target);
        self.start_scenario();
    }

    /// Building a renderer that fails partway must destroy whatever it created,
    /// which validation reports otherwise when the device and instance are destroyed.
    fn check_failed_construction(&mut self, app: &App, size: PhysicalSize<u32>) {
        println!("failed construction");
        for mesh_capacity in [
            // Far past any device's memory, even one that overcommits, such as lavapipe's.
            // The first fails on the pool's first buffer, the second on its last.
            MeshCapacity {
                vertex_words: u32::MAX,
                ..CAPACITY
            },
            MeshCapacity {
                meshes: u32::MAX,
                ..CAPACITY
            },
        ] {
            let config = RendererConfig {
                mesh_capacity,
                ..Default::default()
            };
            match Renderer::with_config(app, size.width, size.height, config) {
                Err(
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                ) => (),
                Err(err) => self.fail(format!("impossible capacity failed with {err}")),
                Ok(_) => self.fail("impossible capacity was created"),
            }
        }
        self.check_messages("failed construction");
    }

    /// The light probes' lighting of scenes with known answers, see `probes`.
    fn check_global_illumination(&mut self, app: &App) {
        println!("global illumination");
        match probes::check(app) {
            Ok(Some(failures)) => {
                for failure in failures {
                    self.fail(format!("global illumination: {failure}"));
                }
            }
            Ok(None) => println!("  frames can't be captured, so it isn't checked"),
            Err(err) => self.fail(format!("global illumination failed with {err}")),
        }
        self.check_messages("global illumination");
    }

    fn start_scenario(&mut self) {
        let scenario = &SCENARIOS[self.scenario];
        println!("{}", scenario.name);
        let scene = self.scene.as_mut().unwrap();
        scene.clear();
        scene.renderer.occlusion_culling = scenario.occlusion;
        scene.renderer.grow_mesh_pool = scenario.pool == PoolTest::Grow;
        self.pool_at_start = Some(scene.renderer.debug_mesh_pool_stats());
        self.target.as_mut().unwrap().resize(START_SIZE);
        self.frame = 0;
        self.out_of_room = 0;
        self.started = Some(Instant::now());
    }

    fn remesh(&mut self) {
        let frame = self.frame;
        let scene = self.scene.as_mut().unwrap();
        for k in 0..REMESH_PER_FRAME {
            let cell = (frame * 37 + k * 11) % (GRID * GRID);
            match scene.remesh(cell, frame) {
                Ok(()) => (),
                Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => self.out_of_room += 1,
                Err(err) => panic!("Creating a mesh failed with {err}."),
            }
        }
        // A chunk destroyed before it's ever uploaded.
        if let Ok(id) = voxel_slab(
            &mut scene.renderer,
            chunk_min(1).extend(CHUNK_Z),
            VOXEL_SIZE,
            [1, 2],
            false,
            false,
        ) {
            scene.renderer.destroy_mesh(id);
        }

        let target = self.target.as_mut().unwrap();
        match frame % 120 {
            30 => target.resize(PhysicalSize::new(500, 700)),
            60 => target.set_minimized(true),
            75 => target.set_minimized(false),
            90 => target.resize(PhysicalSize::new(900, 400)),
            _ => (),
        }
    }

    /// Makes the pool repack for the scenario's pool test.
    fn test_pool(&mut self) {
        let scenario = &SCENARIOS[self.scenario];
        let scene = self.scene.as_mut().unwrap();
        let capacity = scene.renderer.debug_mesh_pool_stats().capacity.vertex_words;
        let ballast = match scenario.pool {
            PoolTest::None => return,
            PoolTest::Defragment => {
                scene.punch_holes();
                // Past the free space at the end, but not past all of it.
                capacity - scene.vertex_words_end() + 8
            }
            PoolTest::Grow => capacity,
        };
        if let Err(err) = scene.add_ballast(ballast) {
            self.fail(format!(
                "{}: creating the ballast failed with {err}",
                scenario.name
            ));
        }
    }

    /// Renders one frame. `false` once every scenario has run.
    fn frame(&mut self) -> bool {
        let scenario = &SCENARIOS[self.scenario];
        self.frame += 1;
        if self.frame == SETTLE_FRAMES {
            let scene = self.scene.as_mut().unwrap();
            scene.fill(scenario.wall, scenario.wall_faces_away).unwrap();
        } else if self.frame == SETTLE_FRAMES + POOL_TEST_DELAY {
            self.test_pool();
        } else if self.frame > SETTLE_FRAMES && scenario.remesh {
            self.remesh();
        }

        let last_frame = self.frame == SETTLE_FRAMES + scenario.frames;
        let size = self.target.as_ref().unwrap().size();
        let (view, proj) = scenario
            .camera
            .view_proj(size.width as f32 / size.height.max(1) as f32);
        let scene = self.scene.as_mut().unwrap();
        let captured = last_frame && scene.renderer.debug_capture_next_frame();
        let live = live(&scene.chunks, &scene.wall, &scene.ballast);
        let draws = DrawList {
            meshes: &live,
            view,
            proj,
        };
        scene
            .renderer
            .submit(size.width, size.height, &draws)
            .unwrap();
        if let (Some(Target::Window(window)), Some(title)) = (&self.target, &mut self.title) {
            let name = scenario.name;
            title.frame(window, &mut scene.renderer, |_| Some(name.to_string()));
        }

        if last_frame {
            self.finish_scenario(proj * view, captured);
            self.scenario += 1;
            if self.scenario == SCENARIOS.len() {
                return false;
            }
            self.start_scenario();
        }
        true
    }

    fn finish_scenario(&mut self, view_proj: Mat4, captured: bool) {
        let scenario = &SCENARIOS[self.scenario];
        let scene = self.scene.as_mut().unwrap();
        let stats = scene.renderer.debug_frame_stats().unwrap().unwrap();
        let (in_view, must_draw) = scene.expected(view_proj);
        let capture = match captured {
            true => scene.renderer.debug_read_capture().unwrap(),
            false => None,
        };
        let pixels = capture
            .as_ref()
            .map(|capture| pixels::check(scene, capture, view_proj));
        let elapsed = self.started.unwrap().elapsed();
        println!(
            "  {} frames in {:.1?}, {} meshes, {in_view} in view, {must_draw} the wall can't hide, drawn {:?}{}",
            scenario.frames,
            elapsed,
            stats.meshes,
            stats.drawn,
            match &pixels {
                Some(Ok(checked)) => format!(", {checked} pixels match"),
                Some(Err(_)) => String::new(),
                None => ", frames can't be captured".to_string(),
            },
        );

        let drawn = stats.total_drawn();
        let failure =
            match scenario.expect {
                Expect::FrustumOnly => (drawn != in_view)
                    .then(|| format!("drew {drawn} meshes, {in_view} are in view")),
                Expect::Occluded => (drawn < must_draw || drawn > in_view)
                    .then(|| format!("drew {drawn} meshes, expected {must_draw} to {in_view}")),
                Expect::OnlyWall => (drawn != 1)
                    .then(|| format!("drew {drawn} meshes, the wall hides all but itself")),
            };
        if let Some(failure) = failure {
            self.fail(format!("{}: {failure}", scenario.name));
        }
        if let Some(Err(failure)) = pixels {
            self.fail(format!("{}: {failure}", scenario.name));
        }
        let before = self.pool_at_start.unwrap();
        let after = self
            .scene
            .as_ref()
            .unwrap()
            .renderer
            .debug_mesh_pool_stats();
        let defrags = after.defrags - before.defrags;
        let growths = after.growths - before.growths;
        if defrags > 0 || growths > 0 {
            println!(
                "  the pool defragmented {defrags} and grew {growths} times, to {} vertex words",
                after.capacity.vertex_words
            );
        }
        let pool_failure = match scenario.pool {
            PoolTest::None => (growths > 0).then_some("the pool grew with growth off"),
            PoolTest::Defragment if defrags == 0 => Some("the pool never defragmented"),
            PoolTest::Defragment => (growths > 0).then_some("the pool grew with growth off"),
            PoolTest::Grow => (growths == 0
                || after.capacity.vertex_words <= before.capacity.vertex_words)
                .then_some("the pool never grew"),
        };
        if let Some(failure) = pool_failure {
            self.fail(format!("{}: {failure}", scenario.name));
        }
        // Destroyed chunks' space is released or packed together before running out.
        if self.out_of_room > 0 {
            self.fail(format!(
                "{}: the pool ran out of room {} times, though every chunk fits",
                scenario.name, self.out_of_room
            ));
        }
        self.check_messages(scenario.name);
    }

    fn stop(&mut self) {
        // The renderer must be dropped before the app.
        if let Some(Scene { renderer, app, .. }) = self.scene.take() {
            drop(renderer);
            drop(app);
            self.check_messages("shutting down");
        }
    }
}

impl ApplicationHandler for Stress {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.target.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("vrs - stress")
            .with_inner_size(START_SIZE);
        let window = event_loop.create_window(attributes).unwrap();
        let app = App::new(
            window.display_handle().unwrap().as_raw(),
            window.window_handle().unwrap().as_raw(),
            app_info(),
        )
        .unwrap();
        self.title = Some(title::FrameTitle::new("stress"));
        self.start(app, Target::Window(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.fail("the window was closed early");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let running = self.frame();
                if !running {
                    event_loop.exit();
                }
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(Target::Window(window)) = &self.target {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.stop();
    }
}

fn app_info() -> vk::ApplicationInfo<'static> {
    vk::ApplicationInfo::default()
        .application_name(c"vrs stress")
        .api_version(vk::API_VERSION_1_3)
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
    if std::env::args().any(|arg| arg == "--headless") {
        stress.start(App::headless(app_info())?, Target::headless());
        while stress.frame() {}
        stress.stop();
    } else {
        EventLoop::new()?.run_app(&mut stress)?;
    }

    if stress.scenario < SCENARIOS.len() {
        stress.failures.push(format!(
            "stopped after {} of {} scenarios",
            stress.scenario,
            SCENARIOS.len()
        ));
    }
    match stress.failures.len() {
        0 => {
            println!("All checks passed.");
            Ok(())
        }
        failed => bail!("{failed} checks failed: {:#?}", stress.failures),
    }
}
