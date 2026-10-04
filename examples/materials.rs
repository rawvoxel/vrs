//! Materials side by side, with the camera circling them: a row of red lacquer and a row of
//! gold, each block rougher than the last, with blocks glowing in different colors between
//! them, on a glossy checkered floor at dusk.
//!
//! Run with `cargo run -p vrs --example materials --release`. Space pauses the camera, and L
//! turns on and off a point light inside each glowing block. The title shows frames a second
//! and how long frames take on the GPU.
//!
//! Along +x, roughness goes 0, 0.1, 0.2, 0.3, 0.4 and 1. The smoothest blocks show sharp
//! reflections of the floor's checks, the glowing blocks and each other, blurring as they get rougher,
//! until past 0.5 they only reflect the probes' blurry light, and at 1 not at all. Gold
//! reflects in its own color, the lacquer mostly at grazing angles. The glowing blocks light
//! what's around them through the light probes, softly, and their reflections in the floor.
//! The point lights add sharp light and shadows close by.

#[path = "common/greedy.rs"]
mod greedy;
#[path = "common/title.rs"]
mod title;

use anyhow::Result;
use ash::vk;
use glam::{Mat4, Vec3};
use greedy::{ChunkMask, EDGE};
use std::time::Instant;
use vrs::{
    App,
    mesh::MeshId,
    renderer::{
        DrawList, GiConfig, Light, Lighting, Material, Renderer, RendererConfig, ShadowGrid,
    },
    voxel::{ChunkMesh, ChunkVertices, VoxelTypes},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

/// The roughness of each block along a row.
const ROUGHNESS: [f32; 6] = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0];
/// What each glowing block gives off, in linear color: red, orange, green, cyan, blue and
/// magenta. Over 1 lights more around it, but shows as 1, so each color's other channels are
/// kept low for it to show as that color.
const GLOWS: [[f32; 3]; 6] = [
    [3.0, 0.1, 0.05],
    [3.0, 0.6, 0.02],
    [0.1, 3.0, 0.15],
    [0.05, 2.5, 3.0],
    [0.15, 0.35, 4.0],
    [3.0, 0.1, 3.0],
];
const LACQUER: [f32; 3] = [0.4, 0.04, 0.04];
const GOLD: [f32; 3] = [1.0, 0.78, 0.34];
/// Voxel types: empty space, the floor's two checks, then each row's blocks, then the glowing
/// blocks.
const FLOOR: u16 = 1;
const FIRST_LACQUER: u16 = 3;
const FIRST_GOLD: u16 = FIRST_LACQUER + ROUGHNESS.len() as u16;
const FIRST_GLOW: u16 = FIRST_GOLD + ROUGHNESS.len() as u16;

/// The floor's top, and how wide its checks are, in voxels.
const FLOOR_TOP: i32 = 4;
const CHECK: i32 = 8;
/// Where each block along a row starts on x, its width, and how far apart they are.
const FIRST_BLOCK: i32 = 6;
const BLOCK: i32 = 5;
const BLOCK_SPACING: i32 = 9;
/// Where each row is on z, and how tall its blocks are.
const LACQUER_ROW: i32 = 20;
const GOLD_ROW: i32 = 39;
const BLOCK_HEIGHT: i32 = 8;
/// The glowing blocks between the rows, 2 voxels wide, raised on the floor.
const GLOW_ROW: i32 = 31;
const GLOW: i32 = 2;
const GLOW_BOTTOM: i32 = FLOOR_TOP + 2;

/// Radians a second the camera circles the materials, how far out and how high.
const ORBIT_SPEED: f32 = 0.25;
const ORBIT_RADIUS: f32 = 46.0;
const ORBIT_HEIGHT: f32 = 34.0;
/// How bright each glowing block's point light is, for its glow, and how far it reaches.
const LIGHT_BRIGHTNESS: f32 = 12.0;
const LIGHT_RANGE: f32 = 24.0;

/// Every voxel type's material, indexed by type.
fn materials() -> Vec<Material> {
    let mut materials = vec![Material::color([0.0; 3])];
    materials.extend([0.2, 0.06].map(|gray| Material {
        roughness: 0.3,
        ..Material::color([gray; 3])
    }));
    for color in [LACQUER, GOLD] {
        let metalness = if color == GOLD { 1.0 } else { 0.0 };
        materials.extend(ROUGHNESS.map(|roughness| Material {
            roughness,
            metalness,
            ..Material::color(color)
        }));
    }
    // Nearly black, so light falling on them doesn't wash out their own.
    materials.extend(GLOWS.map(|emission| Material {
        emission,
        ..Material::color([0.02; 3])
    }));
    materials
}

/// The voxel type at chunk voxel `x`, `y`, `z`.
fn voxel(x: i32, y: i32, z: i32) -> u16 {
    if y < FLOOR_TOP {
        return FLOOR + ((x / CHECK + z / CHECK) % 2) as u16;
    }
    let index = (x - FIRST_BLOCK).div_euclid(BLOCK_SPACING);
    let along = (x - FIRST_BLOCK).rem_euclid(BLOCK_SPACING);
    if !(0..ROUGHNESS.len() as i32).contains(&index) {
        return 0;
    }
    let index = index as u16;
    let block = along < BLOCK && y < FLOOR_TOP + BLOCK_HEIGHT;
    if block && (LACQUER_ROW..LACQUER_ROW + BLOCK).contains(&z) {
        return FIRST_LACQUER + index;
    }
    if block && (GOLD_ROW..GOLD_ROW + BLOCK).contains(&z) {
        return FIRST_GOLD + index;
    }
    // Centered under the blocks.
    let glow_x = (BLOCK - GLOW) / 2;
    let glowing = (glow_x..glow_x + GLOW).contains(&along)
        && (GLOW_BOTTOM..GLOW_BOTTOM + GLOW).contains(&y)
        && (GLOW_ROW..GLOW_ROW + GLOW).contains(&z);
    if glowing {
        return FIRST_GLOW + index;
    }
    0
}

/// The middle of the chunk's floor, which the camera circles. The chunk itself is at the
/// origin: the shadow grid only holds chunks on the grid of chunks from there, so shadows,
/// the light probes and reflections trace through it.
const CENTER: Vec3 = Vec3::new(EDGE as f32 / 2.0, 0.0, EDGE as f32 / 2.0);

/// A point light in the middle of each glowing block, of its color.
fn glow_lights() -> Vec<Light> {
    GLOWS
        .iter()
        .enumerate()
        .map(|(index, glow)| {
            let corner = Vec3::new(
                (FIRST_BLOCK + index as i32 * BLOCK_SPACING + (BLOCK - GLOW) / 2) as f32,
                GLOW_BOTTOM as f32,
                GLOW_ROW as f32,
            );
            let middle = corner + GLOW as f32 / 2.0;
            Light {
                position: middle,
                color: glow.map(|channel| channel * LIGHT_BRIGHTNESS),
                range: LIGHT_RANGE,
                spot: None,
            }
        })
        .collect()
}

/// Meshes the chunk, sets the materials and lights it at dusk, with shadows and light probes.
fn create_scene(renderer: &mut Renderer) -> Result<MeshId> {
    renderer.set_materials(&materials());
    renderer.lighting = Lighting {
        sun_direction: Vec3::new(0.7, 0.35, -0.4),
        sun_color: [0.8, 0.5, 0.3],
        sky_color: [0.16, 0.2, 0.3],
        ground_color: [0.05, 0.05, 0.05],
        shadow_grid: Some(ShadowGrid::new(1.0, EDGE as u32)),
        ..Lighting::default()
    };
    let solid = |x: i32, y: i32, z: i32| {
        let inside = [x, y, z].iter().all(|&v| (0..EDGE as i32).contains(&v));
        inside && voxel(x, y, z) != 0
    };
    let mesh = greedy::mesh(&ChunkMask::from_fn(solid));
    let types: Vec<u16> = (0..EDGE.pow(3))
        .map(|index| {
            let (x, y, z) = (index % EDGE, index / EDGE % EDGE, index / EDGE.pow(2));
            voxel(x as i32, y as i32, z as i32)
        })
        .collect();
    Ok(renderer.create_chunk_mesh(&ChunkMesh {
        indices: &mesh.indices,
        vertices: ChunkVertices::Greedy(&mesh.vertices),
        origin: Vec3::ZERO,
        voxel_size: 1.0,
        types: &VoxelTypes::pack(EDGE as u32, &types),
    })?)
}

/// The view and projection for the camera `angle` radians around the materials.
fn view_proj(angle: f32, aspect: f32) -> (Mat4, Mat4) {
    let eye = CENTER
        + Vec3::new(
            ORBIT_RADIUS * angle.sin(),
            ORBIT_HEIGHT,
            ORBIT_RADIUS * angle.cos(),
        );
    let view = Mat4::look_at_rh(eye, CENTER + Vec3::new(0.0, 4.0, 0.0), Vec3::Y);
    // glam's projection is OpenGL's, with y up, Vulkan's clip space has y down.
    let mut proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 0.1, 500.0);
    proj.y_axis.y = -proj.y_axis.y;
    (view, proj)
}

struct Vulkan {
    scene: MeshId,
    // NOTE: Field order matters, the renderer must be dropped before the app.
    renderer: Renderer,
    _app: App,
}

impl Vulkan {
    fn new(window: &Window) -> Result<Self> {
        let size = window.inner_size();
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"vrs materials")
            .api_version(vk::API_VERSION_1_3);
        let app = App::new(
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            app_info,
        )?;
        let config = RendererConfig {
            global_illumination: Some(GiConfig::default()),
            ..RendererConfig::default()
        };
        let mut renderer = Renderer::with_config(&app, size.width, size.height, config)?;
        let scene = create_scene(&mut renderer)?;
        renderer.lights = glow_lights();

        Ok(Self {
            scene,
            renderer,
            _app: app,
        })
    }
}

struct Materials {
    /// How far the camera has circled, in radians.
    angle: f32,
    paused: bool,
    last_frame: Instant,
    title: title::FrameTitle,
    // NOTE: Field order matters, Vulkan must be dropped before the window it renders to.
    vulkan: Option<Vulkan>,
    window: Option<Window>,
}

impl ApplicationHandler for Materials {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("vrs - materials")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = event_loop
            .create_window(attributes)
            .expect("Could not create window.");
        self.vulkan = Some(Vulkan::new(&window).expect("Could not initialize Vulkan."));
        self.window = Some(window);
        self.last_frame = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(vulkan)) = (&self.window, &mut self.vulkan) else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Space => self.paused = !self.paused,
                KeyCode::KeyL => {
                    let lights = &mut vulkan.renderer.lights;
                    *lights = match lights.is_empty() {
                        true => glow_lights(),
                        false => Vec::new(),
                    };
                }
                _ => (),
            },
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                if !self.paused {
                    self.angle += (now - self.last_frame).as_secs_f32() * ORBIT_SPEED;
                }
                self.last_frame = now;

                let size = window.inner_size();
                // A minimized window has no surface to present to.
                if size.width == 0 || size.height == 0 {
                    return;
                }
                let (view, proj) = view_proj(self.angle, size.width as f32 / size.height as f32);
                let draws = DrawList {
                    meshes: &[&vulkan.scene],
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
    event_loop.run_app(&mut Materials {
        angle: 0.6,
        paused: false,
        last_frame: Instant::now(),
        title: title::FrameTitle::new("materials"),
        vulkan: None,
        window: None,
    })?;
    Ok(())
}
