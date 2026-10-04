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

use anyhow::{Result, bail};
use ash::vk;
use glam::{Mat4, Vec2, Vec3};
use std::time::Instant;
use vrs::{
    App, debug,
    mesh::{MeshCapacity, MeshId},
    renderer::{DrawList, Renderer},
    vertex::Vertex,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

const GRID: u32 = 16;
const CHUNK_SIZE: f32 = 2.0 / GRID as f32 * 0.9;
const CHUNK_Z: f32 = 0.5;
const WALL_Z: f32 = 0.2;
const WALL_HALF_SIZE: f32 = 0.5;
/// Room for half again the grid's geometry, so remeshing has to reuse freed space.
const CAPACITY: MeshCapacity = MeshCapacity {
    vertices: GRID * GRID * 4 * 3 / 2 + 4,
    indices: GRID * GRID * 6 * 3 / 2 + 6,
    meshes: GRID * GRID * 3 / 2,
};
/// Frames after a scenario starts before its meshes are created, so the last
/// scenario's destroyed meshes are released first.
const SETTLE_FRAMES: u32 = 3;
/// Chunks remeshed per frame when remeshing.
const REMESH_PER_FRAME: u32 = 24;

#[derive(Clone, Copy, Debug)]
enum Camera {
    Identity,
    /// Scales by two, so the wall fills the screen.
    Zoom,
    Perspective,
}

impl Camera {
    fn view_proj(self, aspect: f32) -> (Mat4, Mat4) {
        match self {
            Camera::Identity => (Mat4::IDENTITY, Mat4::IDENTITY),
            Camera::Zoom => (Mat4::IDENTITY, Mat4::from_scale(Vec3::new(2.0, 2.0, 1.0))),
            Camera::Perspective => (
                Mat4::look_at_rh(Vec3::new(0.0, 0.0, -1.5), Vec3::new(0.0, 0.0, 0.5), Vec3::NEG_Y),
                Mat4::perspective_rh(60f32.to_radians(), aspect, 0.1, 10.0),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    /// Draw counts must match the meshes in view exactly.
    FrustumOnly,
    /// At least every mesh the wall can't hide, at most every mesh in view.
    Occluded,
    /// Only the wall, which hides everything else.
    OnlyWall,
    /// The draw path doesn't count draws, so only validation is checked.
    Uncounted,
}

struct Scenario {
    name: &'static str,
    frames: u32,
    camera: Camera,
    wall: bool,
    occlusion: bool,
    /// Remesh chunks every frame, and resize and minimize the window.
    remesh: bool,
    /// See `Renderer::debug_limit_draw_paths`.
    draw_paths: (bool, u32),
    expect: Expect,
}

const ALL_PATHS: (bool, u32) = (true, u32::MAX);

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "frustum culling",
        frames: 60,
        camera: Camera::Identity,
        wall: false,
        occlusion: false,
        remesh: false,
        draw_paths: ALL_PATHS,
        expect: Expect::FrustumOnly,
    },
    Scenario {
        name: "frustum culling, zoomed in",
        frames: 60,
        camera: Camera::Zoom,
        wall: false,
        occlusion: false,
        remesh: false,
        draw_paths: ALL_PATHS,
        expect: Expect::FrustumOnly,
    },
    Scenario {
        name: "occlusion culling",
        frames: 60,
        camera: Camera::Identity,
        wall: true,
        occlusion: true,
        remesh: false,
        draw_paths: ALL_PATHS,
        expect: Expect::Occluded,
    },
    Scenario {
        name: "occlusion culling, perspective",
        frames: 60,
        camera: Camera::Perspective,
        wall: true,
        occlusion: true,
        remesh: false,
        draw_paths: ALL_PATHS,
        expect: Expect::Occluded,
    },
    Scenario {
        name: "occlusion culling, wall fills the screen",
        frames: 60,
        camera: Camera::Zoom,
        wall: true,
        occlusion: true,
        remesh: false,
        draw_paths: ALL_PATHS,
        expect: Expect::OnlyWall,
    },
    Scenario {
        name: "remeshing",
        frames: 600,
        camera: Camera::Perspective,
        wall: true,
        occlusion: true,
        remesh: true,
        draw_paths: ALL_PATHS,
        expect: Expect::Occluded,
    },
    Scenario {
        name: "remeshing, uncounted draws",
        frames: 240,
        camera: Camera::Perspective,
        wall: true,
        occlusion: true,
        remesh: true,
        draw_paths: (false, u32::MAX),
        expect: Expect::Uncounted,
    },
    Scenario {
        name: "remeshing, 7 draws per call",
        frames: 240,
        camera: Camera::Perspective,
        wall: true,
        occlusion: true,
        remesh: true,
        draw_paths: (true, 7),
        expect: Expect::Uncounted,
    },
];

fn quad(renderer: &mut Renderer, min: Vec2, size: f32, z: f32, col: [f32; 3]) -> Result<MeshId, vk::Result> {
    let corner = |x: f32, y: f32| Vertex {
        pos: [min.x + x * size, min.y + y * size, z],
        col,
    };
    renderer.create_mesh(
        &[0, 1, 2, 2, 1, 3],
        &[corner(0.0, 0.0), corner(1.0, 0.0), corner(0.0, 1.0), corner(1.0, 1.0)],
    )
}

/// Some chunks are empty, like air in a voxel world.
fn chunk(renderer: &mut Renderer, cell: u32, frame: u32) -> Result<MeshId, vk::Result> {
    if (cell + frame).is_multiple_of(13) {
        return renderer.create_mesh(&[], &[]);
    }
    let min = Vec2::new((cell % GRID) as f32, (cell / GRID) as f32) / GRID as f32 * 2.0 - 1.0;
    let shade = ((frame / 7 + cell) % 10) as f32 / 10.0;
    quad(renderer, min, CHUNK_SIZE, CHUNK_Z, [shade, 1.0 - shade, 0.5])
}

/// A box's screen rectangle in NDC, `None` when it's outside the frustum.
fn screen_rect(view_proj: Mat4, min: Vec3, max: Vec3) -> Option<(Vec2, Vec2)> {
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for corner in 0..8 {
        let p = Vec3::select(
            glam::BVec3::new(corner & 1 != 0, corner & 2 != 0, corner & 4 != 0),
            max,
            min,
        );
        let ndc = view_proj.project_point3(p);
        lo = lo.min(ndc.truncate());
        hi = hi.max(ndc.truncate());
    }
    let outside = hi.x < -1.0 || lo.x > 1.0 || hi.y < -1.0 || lo.y > 1.0;
    (!outside).then(|| (lo.max(Vec2::NEG_ONE), hi.min(Vec2::ONE)))
}

struct Scene {
    chunks: Vec<Option<MeshId>>,
    wall: Option<MeshId>,
    renderer: Renderer,
    app: App,
}

/// Every mesh in the scene, borrowing only its meshes so the renderer stays free.
fn live<'a>(chunks: &'a [Option<MeshId>], wall: &'a Option<MeshId>) -> Vec<&'a MeshId> {
    chunks.iter().flatten().chain(wall.as_ref()).collect()
}

impl Scene {

    fn clear(&mut self) {
        for id in self.chunks.iter_mut().filter_map(Option::take) {
            self.renderer.destroy_mesh(id);
        }
        if let Some(wall) = self.wall.take() {
            self.renderer.destroy_mesh(wall);
        }
    }

    /// Meshes in view, and of those, how many the wall can't hide.
    fn expected(&self, view_proj: Mat4) -> (u32, u32) {
        let wall = self.wall.as_ref().and_then(|wall| {
            let mesh = self.renderer.mesh(wall);
            screen_rect(view_proj, mesh.bounds_min, mesh.bounds_max)
        });
        let (mut in_view, mut must_draw) = (0, 0);
        for id in live(&self.chunks, &self.wall) {
            let mesh = self.renderer.mesh(id);
            if mesh.index_count() == 0 {
                continue;
            }
            let Some((lo, hi)) = screen_rect(view_proj, mesh.bounds_min, mesh.bounds_max) else {
                continue;
            };
            in_view += 1;
            let hidden = wall.is_some_and(|(wall_lo, wall_hi)| {
                mesh.bounds_min.z > WALL_Z && lo.cmpge(wall_lo).all() && hi.cmple(wall_hi).all()
            });
            if !hidden {
                must_draw += 1;
            }
        }
        (in_view, must_draw)
    }
}

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
            self.fail(format!("{what}: {errors} validation errors and {warnings} warnings"));
        }
        self.messages_at_start = counts;
    }

    /// Building a renderer that fails partway must destroy whatever it created,
    /// which validation reports otherwise when the device and instance are destroyed.
    fn check_failed_construction(&mut self, app: &App, size: PhysicalSize<u32>) {
        println!("failed construction");
        for capacity in [
            MeshCapacity { vertices: u32::MAX, ..CAPACITY },
            MeshCapacity { indices: u32::MAX, ..CAPACITY },
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
            scene.wall = Some(quad(&mut scene.renderer, min, 2.0 * WALL_HALF_SIZE, WALL_Z, [0.9, 0.8, 0.1]).unwrap());
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
        let failure = match (scenario.expect, drawn) {
            (Expect::Uncounted, None) => None,
            (Expect::Uncounted, Some(_)) => Some("draws were counted on an uncounted draw path".to_string()),
            (_, None) => Some("draws weren't counted".to_string()),
            (Expect::FrustumOnly, Some(drawn)) => {
                (drawn != in_view).then(|| format!("drew {drawn} meshes, {in_view} are in view"))
            }
            (Expect::Occluded, Some(drawn)) => (drawn < must_draw || drawn > in_view)
                .then(|| format!("drew {drawn} meshes, expected {must_draw} to {in_view}")),
            (Expect::OnlyWall, Some(drawn)) => {
                (drawn != 1).then(|| format!("drew {drawn} meshes, the wall hides all but itself"))
            }
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

        let renderer = Renderer::with_mesh_capacity(&app, size.width, size.height, CAPACITY).unwrap();
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
                let (view, proj) = scenario.camera.view_proj(size.width as f32 / size.height.max(1) as f32);
                let scene = self.scene.as_mut().unwrap();
                let live = live(&scene.chunks, &scene.wall);
                let draws = DrawList {
                    meshes: &live,
                    view,
                    proj,
                    ..Default::default()
                };
                scene.renderer.submit(size.width, size.height, &draws).unwrap();

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
        ("VK_LAYER_ENABLES", "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"),
    ] {
        if std::env::var_os(name).is_none() {
            // SAFETY: No other threads exist yet.
            unsafe { std::env::set_var(name, value) };
        }
    }

    let mut stress = Stress::default();
    EventLoop::new()?.run_app(&mut stress)?;

    if stress.scenario < SCENARIOS.len() {
        stress.failures.push(format!("stopped after {} of {} scenarios", stress.scenario, SCENARIOS.len()));
    }
    match stress.failures.len() {
        0 => {
            println!("All {} scenarios passed.", SCENARIOS.len());
            Ok(())
        }
        failed => bail!("{failed} checks failed: {:#?}", stress.failures),
    }
}
