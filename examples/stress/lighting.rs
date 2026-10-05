//! Lighting checked against scenes with known answers, in the order they run:
//!
//! - A floor under a uniform white sky, without a sun, lit by its probes as by the sky alone,
//!   at full and half trace resolution, where the trace pass lights faces with the probes'
//!   light for shading to upscale.
//! - A sealed hollow box under a bright sun and sky, seen from inside: no light may leak in,
//!   at full and half trace resolution, and with 4 cascades of probes blending.
//! - A sealed hall seen from far across it, where the depth buffer is least precise: no light
//!   may leak onto its pyramids' steps or into its corners, at full and half trace resolution.
//! - The box with a hole in its roof, and the hall with skylights, in a still frame: their
//!   bounced light holds steady from frame to frame.
//! - The same box with a hole in its roof: sunlight bouncing in lights all of it. With glass
//!   over the hole, the glass dims it. A few frames after cutting the hole or the sun coming
//!   out, the light inside has mostly caught up.
//! - The sealed box in the dark with a glowing block inside, and then a point light: either
//!   lights the walls, the light's bouncing off them too.
//! - A point light behind a wall: the floor past the wall is in its shadow, at full and half
//!   trace resolution.
//! - A point light inside a glowing block over a floor: the block doesn't shadow it.
//! - A spot light over a floor: it lights only a pool under it.
//! - A glowing block high over a mirror floor, out of view: the floor shows it.
//! - A point light behind a wall of coarser chunks: the wall shadows it once the voxel grid
//!   has their level.
//! - The floor under a white sky, of coarser chunks only: lit as the fine one is. Then steps
//!   of coarser chunks, whose sides aren't darker than fine ones' where probes touch them.
//! - The box with a hole in its roof, with too few probes to cover it: coarser cascades
//!   bring bounced light to more of it.

use crate::{
    camera,
    greedy::{self, EDGE},
};
use ash::{prelude::VkResult, vk};
use glam::{Mat4, Vec3};
use vrs::{
    App,
    mesh::{MeshId, Vertex},
    renderer::{
        Capture, DebugView, DrawList, Light, Lighting, Material, ProbeConfig, Renderer,
        RendererConfig, Spot, TraceResolution, VoxelGrid,
    },
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// Enough for several bounces of light to settle.
const FRAMES: u32 = 40;
/// Frames after a change for the probes to mostly catch up with it.
const CHANGE_FRAMES: u32 = 8;

/// Voxel types: empty space, plain gray, glowing white, gray glass letting 70% through, then
/// a white metal mirror.
const GRAY: u16 = 1;
const GLOW: u16 = 2;
const GLASS: u16 = 3;
const MIRROR: u16 = 4;
const ALBEDO: f32 = 0.5;
const MATERIALS: [Material; 5] = [
    Material::matte([0.0; 3]),
    Material::matte([ALBEDO; 3]),
    Material {
        emission: [4.0; 3],
        ..Material::matte([1.0; 3])
    },
    Material {
        see_through: true,
        ..Material::matte([0.7; 3])
    },
    Material {
        roughness: 0.0,
        metalness: 1.0,
        ..Material::matte([0.9; 3])
    },
];

/// How far a floor under a white sky may be from its color, relative to it: the probes'
/// light is a blend of noisy rays.
const ALBEDO_TOLERANCE: f32 = 0.06;
/// Brightness under which a pixel counts as unlit in a sealed box, in linear color.
const UNLIT: f32 = 0.02;
/// Brightness over which a pixel counts as reached by light, in linear color: any byte above
/// black, sRGB or not.
const BOUNCED: f32 = 0.002;
/// How much lit pixels may change from frame to frame once the light has settled, relative to
/// their brightness, on average, and the fraction of them that may change by over a tenth: the
/// probes' rays are random, but blended over enough of them the light holds steady.
const FLICKER: f32 = 0.005;
const FLICKERING_PIXELS: f32 = 0.0005;
/// Pixels allowed to show through one pixel cracks between quads, as a fraction of all.
const CRACKS: f32 = 0.002;
/// What "most of" a surface is, as a fraction of its pixels.
const MOST: f32 = 0.5;
/// How much of a light a shadowed floor may get, as a fraction of the light it gets without
/// the wall, for the few pixels along the wall's foot.
const SHADOW_LEAK: f32 = 0.02;

/// Each pixel's red channel, in linear color: every material is gray or white, and every
/// light white, so it's how bright each pixel is. `None` when the frame can't be read.
fn linear_reds(capture: &Capture) -> Option<Vec<f32>> {
    let mut reds = Vec::with_capacity((capture.width * capture.height) as usize);
    for y in 0..capture.height {
        for x in 0..capture.width {
            let red = capture.rgba8(x, y)?[0] as f32 / 255.0;
            reds.push(match capture.is_srgb() {
                true if red <= 0.04045 => red / 12.92,
                true => ((red + 0.055) / 1.055).powf(2.4),
                false => red,
            });
        }
    }
    Some(reds)
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

/// The fraction of `values` over `threshold`.
fn fraction_over(values: &[f32], threshold: f32) -> f32 {
    values.iter().filter(|&&value| value > threshold).count() as f32 / values.len() as f32
}

/// A fraction as a percentage, to `decimals` places.
fn percent(fraction: f32, decimals: usize) -> String {
    format!("{:.decimals$}%", fraction * 100.0)
}

/// Voxels' types by world voxel.
type Voxels<'a> = &'a dyn Fn(i32, i32, i32) -> u16;

/// Where a check's camera is and what it looks at.
#[derive(Clone, Copy)]
struct View {
    eye: Vec3,
    target: Vec3,
}

/// Down at the floor's middle.
const FLOOR_VIEW: View = View {
    eye: Vec3::new(10.0, 20.0, 10.0),
    target: Vec3::new(10.0, 8.0, 10.5),
};
/// From inside the box, at its far wall.
const BOX_VIEW: View = View {
    eye: Vec3::new(0.5, 20.0, -10.0),
    target: Vec3::new(0.5, 18.0, 10.0),
};
/// From high inside the box, down at its far half's floor.
const HOLE_VIEW: View = View {
    eye: Vec3::new(0.5, 30.0, -14.0),
    target: Vec3::new(0.5, 6.0, 10.0),
};

/// A floor 8 voxels thick.
fn floor(_: i32, y: i32, _: i32) -> u16 {
    if y < 8 { GRAY } else { 0 }
}

/// A box 32 voxels wide inside, its walls 2 thick.
fn shell(x: i32, y: i32, z: i32) -> u16 {
    let inside = |v: i32, lo: i32, hi: i32| (lo..hi).contains(&v);
    let outer = inside(x, -18, 18) && inside(y, 4, 40) && inside(z, -18, 18);
    let hollow = inside(x, -16, 16) && inside(y, 6, 38) && inside(z, -16, 16);
    if outer && !hollow { GRAY } else { 0 }
}

/// A sealed hall nearly as wide as the scene, its walls and roof 2 thick, holding rows of
/// stepped pyramids, like roofs, whose every step is a concave edge.
fn hall(x: i32, y: i32, z: i32) -> u16 {
    let inside = |v: i32, lo: i32, hi: i32| (lo..hi).contains(&v);
    let outer = inside(x, -62, 126) && inside(y, 4, 60) && inside(z, -62, 126);
    let hollow = inside(x, -60, 124) && inside(y, 6, 58) && inside(z, -60, 124);
    if outer && !hollow {
        return GRAY;
    }
    // Pyramids 24 wide every 32, rising a voxel for each 2 in from their edges.
    let (px, pz) = ((x + 60).rem_euclid(32), (z + 60).rem_euclid(32));
    let from_edge = px.min(23 - px).min(pz).min(23 - pz);
    if hollow && px < 24 && pz < 24 && y >= 6 && y < 6 + from_edge / 2 + 1 {
        return GRAY;
    }
    0
}

/// The hall with a skylight in its roof over every other pyramid, the sun reaching some of
/// it and its light bouncing around the rest.
fn skylit_hall(x: i32, y: i32, z: i32) -> u16 {
    let skylight = y >= 58 && (x + 60).rem_euclid(64) < 12 && (z + 60).rem_euclid(64) < 12;
    if skylight { 0 } else { hall(x, y, z) }
}

/// Far enough from the world's origin that a voxel's coordinates there are only exact to a
/// few thousandths, as in a big world.
const FAR: Vec3 = Vec3::new(4096.0, 2816.0, -4096.0);

/// From one corner of the hall, across it to the far corner's floor.
const HALL_VIEW: View = View {
    eye: Vec3::new(-55.0, 45.0, -55.0),
    target: Vec3::new(110.0, 6.0, 110.0),
};

/// Where the box's roof has a hole.
fn hole(x: i32, y: i32, z: i32) -> bool {
    y >= 38 && (-4..4).contains(&x) && (-4..4).contains(&z)
}

/// The box with a hole in its roof.
fn holed(x: i32, y: i32, z: i32) -> u16 {
    if hole(x, y, z) { 0 } else { shell(x, y, z) }
}

/// A uniform white sky, without a sun.
fn sky() -> Lighting {
    Lighting {
        sun_color: [0.0; 3],
        sky_color: [1.0; 3],
        ground_color: [1.0; 3],
        ..Lighting::default()
    }
}

/// A bright sun and sky.
fn bright() -> Lighting {
    Lighting {
        sun_direction: Vec3::new(0.3, 1.0, 0.2),
        sun_color: [3.0; 3],
        sky_color: [1.0; 3],
        ground_color: [1.0; 3],
        ..Lighting::default()
    }
}

/// A bright sun almost straight overhead, without a sky, so only light through the box's
/// hole lights it.
fn sun_only() -> Lighting {
    Lighting {
        sun_direction: Vec3::new(0.05, 1.0, 0.1),
        sky_color: [0.0; 3],
        ground_color: [0.0; 3],
        ..bright()
    }
}

/// No sun or sky.
fn dark() -> Lighting {
    Lighting {
        sun_color: [0.0; 3],
        sky_color: [0.0; 3],
        ground_color: [0.0; 3],
        ..Lighting::default()
    }
}

/// No light at all but the lights', without probes.
fn lights_only() -> Lighting {
    Lighting {
        global_illumination: false,
        ..dark()
    }
}

/// `lighting` at half trace resolution.
fn half(lighting: Lighting) -> Lighting {
    Lighting {
        trace_resolution: TraceResolution::Half,
        ..lighting
    }
}

/// What a check renders, `FRAMES` frames of it.
#[derive(Clone, Copy)]
struct Scene<'a> {
    voxels: Voxels<'a>,
    /// Chunks of the next level of detail, of voxels twice as big.
    coarse: Option<Voxels<'a>>,
    /// The voxel grid's levels.
    levels: u32,
    probes: ProbeConfig,
    lighting: Lighting,
    lights: &'a [Light],
    /// Then `CHANGE_FRAMES` frames of these voxels and this lighting.
    change: Option<(Voxels<'a>, Lighting)>,
    /// Where in the world its voxel at the origin is, the camera and lights moved with it.
    origin: Vec3,
    /// Triangles of a colored mesh, outside the voxel grid.
    colored: &'a [Vertex],
}

impl<'a> Scene<'a> {
    fn new(voxels: Voxels<'a>, lighting: Lighting) -> Self {
        Self {
            voxels,
            coarse: None,
            levels: 1,
            probes: ProbeConfig::default(),
            lighting,
            lights: &[],
            change: None,
            origin: Vec3::ZERO,
            colored: &[],
        }
    }

    fn colored(self, colored: &'a [Vertex]) -> Self {
        Self { colored, ..self }
    }

    /// Moved to `origin`, a whole number of coarse chunks from the world's origin.
    fn at(self, origin: Vec3) -> Self {
        Self { origin, ..self }
    }

    /// Its lights, moved with it.
    fn placed_lights(&self) -> Vec<Light> {
        let moved = |light: &Light| Light {
            position: light.position + self.origin,
            ..*light
        };
        self.lights.iter().map(moved).collect()
    }

    /// The view matrix of `view`, moved with it.
    fn view_matrix(&self, view: View) -> Mat4 {
        Mat4::look_at_rh(view.eye + self.origin, view.target + self.origin, Vec3::Y)
    }

    fn lights(self, lights: &'a [Light]) -> Self {
        Self { lights, ..self }
    }

    /// With `coarse` chunks, and `levels` levels in the voxel grid.
    fn coarse(self, coarse: Voxels<'a>, levels: u32) -> Self {
        Self {
            coarse: Some(coarse),
            levels,
            ..self
        }
    }

    fn probes(self, probes: ProbeConfig) -> Self {
        Self { probes, ..self }
    }

    fn change(self, voxels: Voxels<'a>, lighting: Lighting) -> Self {
        Self {
            change: Some((voxels, lighting)),
            ..self
        }
    }
}

/// Each chunk of level `level` around the origin with any voxels, as meshes: 64 voxels wide,
/// each `2^level` world voxels wide, of the type of the world voxel at its low corner. Scenes
/// fit the 3 by 3 chunks around the origin's, one chunk high.
fn create_chunks(
    renderer: &mut Renderer,
    voxels: Voxels,
    level: u32,
    origin: Vec3,
) -> VkResult<Vec<MeshId>> {
    let scale = 1 << level;
    let mut ids = Vec::new();
    for chunk_x in -1..2 {
        for chunk_z in -1..2 {
            let corner = [chunk_x, 0, chunk_z].map(|chunk| chunk * EDGE as i32 * scale);
            let local = |x: i32, y: i32, z: i32| {
                voxels(
                    corner[0] + x * scale,
                    corner[1] + y * scale,
                    corner[2] + z * scale,
                )
            };
            let types = greedy::chunk_types(local);
            let solid = |x: i32, y: i32, z: i32| local(x, y, z) != 0;
            ids.extend(greedy::create_chunk(
                renderer,
                origin + Vec3::from(corner.map(|voxel| voxel as f32)),
                scale as f32,
                &types,
                solid,
            )?);
        }
    }
    Ok(ids)
}

/// Every chunk of `scene`'s voxels, `voxels` in place of its own.
fn create_scene(renderer: &mut Renderer, scene: &Scene, voxels: Voxels) -> VkResult<Vec<MeshId>> {
    let mut ids = create_chunks(renderer, voxels, 0, scene.origin)?;
    if let Some(coarse) = scene.coarse {
        ids.extend(create_chunks(renderer, coarse, 1, scene.origin)?);
    }
    if !scene.colored.is_empty() {
        let moved = |vertex: &Vertex| Vertex {
            position: (Vec3::from(vertex.position) + scene.origin).to_array(),
            ..*vertex
        };
        let vertices: Vec<Vertex> = scene.colored.iter().map(moved).collect();
        let indices: Vec<u32> = (0..vertices.len() as u32).collect();
        ids.push(renderer.create_colored_mesh(&indices, &vertices)?);
    }
    Ok(ids)
}

/// The frame after `scene`'s frames, seen from `view`, each pixel's brightness. `None` when
/// frames can't be captured.
fn render(app: &App, scene: Scene, view: View) -> VkResult<Option<Vec<f32>>> {
    let config = RendererConfig {
        global_illumination: Some(scene.probes),
        ..Default::default()
    };
    let mut renderer = Renderer::with_config(app, WIDTH, HEIGHT, config)?;
    renderer.set_materials(&MATERIALS);
    renderer.lights = scene.placed_lights();
    let grid = Some(VoxelGrid::new(1.0, EDGE as u32).with_levels(scene.levels));
    let light = |lighting: Lighting| Lighting {
        voxel_grid: grid,
        ..lighting
    };
    renderer.lighting = light(scene.lighting);
    let view_matrix = scene.view_matrix(view);
    let proj = camera::perspective(70.0, WIDTH as f32 / HEIGHT as f32, 0.1, 500.0);
    let frames = |renderer: &mut Renderer, ids: &[MeshId], count: u32| {
        let meshes: Vec<&MeshId> = ids.iter().collect();
        let draws = DrawList {
            meshes: &meshes,
            view: view_matrix,
            proj,
        };
        (0..count).try_for_each(|_| renderer.submit(WIDTH, HEIGHT, &draws))
    };

    let mut ids = create_scene(&mut renderer, &scene, scene.voxels)?;
    frames(&mut renderer, &ids, FRAMES)?;
    if let Some((voxels, lighting)) = scene.change {
        for id in ids.drain(..) {
            renderer.destroy_mesh(id);
        }
        ids = create_scene(&mut renderer, &scene, voxels)?;
        renderer.lighting = light(lighting);
        frames(&mut renderer, &ids, CHANGE_FRAMES)?;
    }
    if !renderer.debug_capture_next_frame() {
        return Ok(None);
    }
    frames(&mut renderer, &ids, 1)?;
    let capture = renderer.debug_read_capture()?;
    Ok(capture.as_ref().and_then(linear_reds))
}

/// `count` consecutive frames after `scene`'s frames, seen from `view`, each pixel's
/// brightness in each. Empty when frames can't be captured.
fn render_frames(app: &App, scene: Scene, view: View, count: u32) -> VkResult<Vec<Vec<f32>>> {
    let config = RendererConfig {
        global_illumination: Some(scene.probes),
        ..Default::default()
    };
    let mut renderer = Renderer::with_config(app, WIDTH, HEIGHT, config)?;
    renderer.set_materials(&MATERIALS);
    renderer.lights = scene.placed_lights();
    renderer.lighting = Lighting {
        voxel_grid: Some(VoxelGrid::new(1.0, EDGE as u32).with_levels(scene.levels)),
        ..scene.lighting
    };
    let ids = create_scene(&mut renderer, &scene, scene.voxels)?;
    let meshes: Vec<&MeshId> = ids.iter().collect();
    let draws = DrawList {
        meshes: &meshes,
        view: scene.view_matrix(view),
        proj: camera::perspective(70.0, WIDTH as f32 / HEIGHT as f32, 0.1, 500.0),
    };
    for _ in 0..FRAMES {
        renderer.submit(WIDTH, HEIGHT, &draws)?;
    }
    let mut frames = Vec::new();
    for _ in 0..count {
        if !renderer.debug_capture_next_frame() {
            return Ok(Vec::new());
        }
        renderer.submit(WIDTH, HEIGHT, &draws)?;
        match renderer
            .debug_read_capture()?
            .as_ref()
            .and_then(linear_reds)
        {
            Some(frame) => frames.push(frame),
            None => return Ok(Vec::new()),
        }
    }
    Ok(frames)
}

/// How much lit pixels change from frame to frame, relative to their brightness, on average,
/// and the fraction of them changing by over a tenth, over `frames`.
fn flicker(frames: &[Vec<f32>]) -> (f32, f32) {
    let (mut change, mut big, mut samples) = (0.0, 0usize, 0usize);
    for pair in frames.windows(2) {
        for (before, after) in pair[0].iter().zip(&pair[1]) {
            let brightness = before.max(*after);
            if brightness < BOUNCED * 10.0 {
                continue;
            }
            let relative = (after - before).abs() / brightness;
            change += relative;
            big += (relative > 0.1) as usize;
            samples += 1;
        }
    }
    let samples = samples.max(1) as f32;
    (change / samples, big as f32 / samples)
}

/// What `Checks::render` fails with when the device's frames can't be captured, which skips
/// the checks.
const CANT_CAPTURE: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

/// Rendering scenes and noting what was wrong.
struct Checks<'a> {
    app: &'a App,
    failures: Vec<String>,
}

impl Checks<'_> {
    /// `scene`'s last frame from `view`, see `render`, or `CANT_CAPTURE`.
    fn render(&self, scene: Scene, view: View) -> VkResult<Vec<f32>> {
        render(self.app, scene, view)?.ok_or(CANT_CAPTURE)
    }

    /// Prints what was `measured`, and fails with it and `why` unless it's `fine`.
    fn expect(&mut self, measured: String, fine: bool, why: &str) {
        println!("  {measured}");
        if !fine {
            self.failures.push(format!("{measured}, {why}"));
        }
    }
}

/// A floor under a white sky reflects as much light as its color, at full and half shadow
/// resolution.
fn sky_floor(checks: &mut Checks) -> VkResult<()> {
    for (resolution, lighting) in [("", sky()), (" at half resolution", half(sky()))] {
        let lit = mean(&checks.render(Scene::new(&floor, lighting), FLOOR_VIEW)?);
        checks.expect(
            format!("a floor under a white sky{resolution} is {lit:.3}, {ALBEDO} is exact"),
            (lit - ALBEDO).abs() <= ALBEDO * ALBEDO_TOLERANCE,
            "the probes should light it as the sky does",
        );
    }
    Ok(())
}

/// No light leaks into a sealed box, at full and half trace resolution, and with cascades.
fn sealed_box(checks: &mut Checks) -> VkResult<()> {
    let four_cascades = ProbeConfig {
        cascades: 4,
        ..ProbeConfig::default()
    };
    for (how, scene) in [
        ("", Scene::new(&shell, bright())),
        (" at half resolution", Scene::new(&shell, half(bright()))),
        (
            " with 4 cascades",
            Scene::new(&shell, bright()).probes(four_cascades),
        ),
    ] {
        let lit = fraction_over(&checks.render(scene, BOX_VIEW)?, UNLIT);
        checks.expect(
            format!("{} of a sealed box's inside is lit{how}", percent(lit, 2)),
            lit <= CRACKS,
            "light leaks in",
        );
    }
    Ok(())
}

/// No light leaks into a sealed hall, onto steps and into corners seen from far across it,
/// where the depth buffer is least precise, at full and half trace resolution, and far from
/// the world's origin, where positions are least precise.
fn sealed_hall(checks: &mut Checks) -> VkResult<()> {
    for (how, scene) in [
        ("", Scene::new(&hall, bright())),
        (" at half resolution", Scene::new(&hall, half(bright()))),
        (" far from the origin", Scene::new(&hall, bright()).at(FAR)),
        (
            " far from the origin at half resolution",
            Scene::new(&hall, half(bright())).at(FAR),
        ),
    ] {
        let lit = fraction_over(&checks.render(scene, HALL_VIEW)?, UNLIT);
        // No crack shows the sky, every surface seen is inside.
        checks.expect(
            format!("{} of a sealed hall's inside is lit{how}", percent(lit, 2)),
            lit == 0.0,
            "light leaks in along its steps and corners",
        );
    }
    Ok(())
}

/// A floor drawn as a colored mesh, outside the voxel grid as grass and leaves are, under a
/// sky brighter than the ground, gets the same light from everywhere all over, at full and
/// half trace resolution: none of it, such as the square under the camera, differs.
fn colored_floor(checks: &mut Checks) -> VkResult<()> {
    let (y, reach) = (10.0, 400.0);
    let corner = |x: f32, z: f32| Vertex {
        position: [x, y, z],
        normal: [0.0, 1.0, 0.0],
        color: [1.0; 3],
    };
    let [a, b, c, d] = [
        corner(-reach, -reach),
        corner(-reach, reach),
        corner(reach, reach),
        corner(reach, -reach),
    ];
    // Both windings, so it's seen whichever way is its front.
    let floor = [a, b, c, a, c, d, a, c, b, a, d, c];
    let lighting = Lighting {
        sun_color: [0.0; 3],
        sky_color: [1.0; 3],
        ground_color: [0.1; 3],
        debug_view: DebugView::Indirect,
        ..Lighting::default()
    };
    // Down past the camera's foot, to well beyond it.
    let view = View {
        eye: Vec3::new(0.0, 30.0, -10.0),
        target: Vec3::new(0.0, 10.0, 5.0),
    };
    for (how, lighting) in [("", lighting), (" at half resolution", half(lighting))] {
        let frame = checks.render(Scene::new(&|_, _, _| 0, lighting).colored(&floor), view)?;
        let mut sorted = frame.clone();
        sorted.sort_by(f32::total_cmp);
        let median = sorted[sorted.len() / 2];
        let off = frame
            .iter()
            .filter(|&&value| (value - median).abs() > median * 0.1)
            .count() as f32
            / frame.len() as f32;
        checks.expect(
            format!(
                "{} of a colored floor's light from everywhere is off by over 10%{how}",
                percent(off, 2)
            ),
            off < 0.001,
            "it depends on where it is from the camera",
        );
    }
    Ok(())
}

/// The bounced light in a box with a hole in its roof is steady from frame to frame once it
/// has settled, at full and half trace resolution, and in a hall busy with voxels near most of
/// its probes, which share the frame's rays, each tracing fewer.
fn steady_light(checks: &mut Checks) -> VkResult<()> {
    let scenes: [(&str, Voxels, View, Lighting); 3] = [
        (" in a box", &holed, HOLE_VIEW, sun_only()),
        (
            " in a box at half resolution",
            &holed,
            HOLE_VIEW,
            half(sun_only()),
        ),
        (
            " in a busy hall at half resolution",
            &skylit_hall,
            HALL_VIEW,
            half(sun_only()),
        ),
    ];
    for (how, voxels, view, lighting) in scenes {
        let frames = render_frames(checks.app, Scene::new(voxels, lighting), view, 16)?;
        if frames.is_empty() {
            return Err(CANT_CAPTURE);
        }
        let (mean_change, big) = flicker(&frames);
        checks.expect(
            format!(
                "bounced light changes {} a frame{how}, {} of pixels by over 10%",
                percent(mean_change, 2),
                percent(big, 2)
            ),
            mean_change <= FLICKER && big <= FLICKERING_PIXELS,
            "it flickers",
        );
    }
    Ok(())
}

/// Sunlight through a hole in a box's roof bounces around it, dimmed through glass, and
/// follows changes within a few frames.
fn holed_box(checks: &mut Checks) -> VkResult<()> {
    let open = checks.render(Scene::new(&holed, sun_only()), HOLE_VIEW)?;
    let direct = Lighting {
        global_illumination: false,
        ..sun_only()
    };
    let without = checks.render(Scene::new(&holed, direct), HOLE_VIEW)?;
    let (lit, lit_without) = (
        fraction_over(&open, BOUNCED),
        fraction_over(&without, BOUNCED),
    );
    checks.expect(
        format!(
            "{} of a box with a hole in its roof is lit with probes, {} without",
            percent(lit, 0),
            percent(lit_without, 0)
        ),
        lit >= MOST,
        "bounced light should reach most of it",
    );
    let brighter = mean(&open) / mean(&without);
    checks.expect(
        format!("bounced light makes the box {brighter:.2}x as bright"),
        brighter >= 1.1,
        "it should add to the sun's",
    );

    let glazed = |x: i32, y: i32, z: i32| match hole(x, y, z) && shell(x, y, z) != 0 {
        true => GLASS,
        false => shell(x, y, z),
    };
    let through = mean(&checks.render(Scene::new(&glazed, sun_only()), HOLE_VIEW)?) / mean(&open);
    // The glass lets 70% through, and bounces dim what it lets in less than that.
    checks.expect(
        format!(
            "a box with glass over its hole is {} as bright as with it open",
            percent(through, 0)
        ),
        (0.4..0.95).contains(&through),
        "the glass should dim it",
    );

    let dark_holed = Lighting {
        sun_color: [0.0; 3],
        ..sun_only()
    };
    for (name, scene) in [
        (
            "cutting a hole in a box",
            Scene::new(&shell, sun_only()).change(&holed, sun_only()),
        ),
        (
            "the sun coming out",
            Scene::new(&holed, dark_holed).change(&holed, sun_only()),
        ),
    ] {
        let caught_up = mean(&checks.render(scene, HOLE_VIEW)?) / mean(&open);
        checks.expect(
            format!(
                "{CHANGE_FRAMES} frames after {name}, the light inside is {} of the way there",
                percent(caught_up, 0)
            ),
            caught_up >= 0.9,
            "the probes should catch up faster",
        );
    }
    Ok(())
}

/// A glowing block and a point light light a dark box around them, through the probes, and
/// the light's bounces add to its own.
fn lit_box(checks: &mut Checks) -> VkResult<()> {
    let lamp = |x: i32, y: i32, z: i32| {
        let glowing = (-2..2).contains(&x) && (18..22).contains(&y) && (4..8).contains(&z);
        if glowing { GLOW } else { shell(x, y, z) }
    };
    let glow = fraction_over(
        &checks.render(Scene::new(&lamp, dark()), BOX_VIEW)?,
        BOUNCED,
    );
    let glow_alone = fraction_over(
        &checks.render(Scene::new(&lamp, lights_only()), BOX_VIEW)?,
        BOUNCED,
    );
    checks.expect(
        format!(
            "{} of a dark box with a glowing block inside is lit with probes, {} without",
            percent(glow, 0),
            percent(glow_alone, 0)
        ),
        glow >= MOST,
        "its light should reach most of the box",
    );

    let bulb = [Light::point(Vec3::new(0.0, 20.0, 6.0), [400.0; 3], 60.0)];
    let with = checks.render(Scene::new(&shell, dark()).lights(&bulb), BOX_VIEW)?;
    let direct = checks.render(Scene::new(&shell, lights_only()).lights(&bulb), BOX_VIEW)?;
    let none = checks.render(Scene::new(&shell, lights_only()), BOX_VIEW)?;
    let lit = fraction_over(&direct, BOUNCED);
    checks.expect(
        format!(
            "a point light in a dark box lights {} of it, {} without the light",
            percent(lit, 0),
            percent(fraction_over(&none, BOUNCED), 0)
        ),
        lit >= MOST,
        "it should reach most of the box",
    );
    let bounced = mean(&with) / mean(&direct);
    checks.expect(
        format!("a point light's bounced light makes a dark box {bounced:.2}x as bright"),
        bounced >= 1.05,
        "the probes should bounce it",
    );
    Ok(())
}

/// A floor behind a wall from a point light is in its shadow, at full and half shadow
/// resolution.
fn shadowed_light(checks: &mut Checks) -> VkResult<()> {
    let walled = |x: i32, y: i32, z: i32| {
        let wall = (0..2).contains(&x) && y < 24 && (-24..24).contains(&z);
        if y < 8 || wall { GRAY } else { 0 }
    };
    let lamp_post = [Light::point(Vec3::new(-6.0, 12.0, 0.5), [400.0; 3], 60.0)];
    let view = View {
        eye: Vec3::new(9.0, 14.0, 0.5),
        target: Vec3::new(9.0, 8.0, 0.6),
    };
    for resolution in [TraceResolution::Full, TraceResolution::Half] {
        let lighting = Lighting {
            trace_resolution: resolution,
            ..lights_only()
        };
        let open = mean(&checks.render(Scene::new(&floor, lighting).lights(&lamp_post), view)?);
        checks.expect(
            format!("a point light lights a floor to {open:.3}, at {resolution:?} resolution"),
            open >= BOUNCED,
            "it should reach the floor",
        );
        let shadowed = checks.render(Scene::new(&walled, lighting).lights(&lamp_post), view)?;
        let through = mean(&shadowed) / open;
        checks.expect(
            format!(
                "a floor behind a wall gets {} of a point light's light, at {resolution:?} resolution",
                percent(through, 1)
            ),
            through <= SHADOW_LEAK,
            "the wall should shadow it",
        );
    }
    Ok(())
}

/// A point light inside a glowing block lights the floor around it, unshadowed by the block.
fn lamp_in_block(checks: &mut Checks) -> VkResult<()> {
    let lamp_block = |x: i32, y: i32, z: i32| {
        let glowing = (0..2).contains(&x) && (10..12).contains(&y) && (0..2).contains(&z);
        if glowing { GLOW } else { floor(x, y, z) }
    };
    let inside = [Light::point(Vec3::new(1.0, 11.0, 1.0), [100.0; 3], 30.0)];
    let view = View {
        eye: Vec3::new(1.0, 24.0, 1.5),
        target: Vec3::new(1.0, 8.0, 1.0),
    };
    let lit = checks.render(Scene::new(&lamp_block, lights_only()).lights(&inside), view)?;
    let reached = fraction_over(&lit, BOUNCED);
    checks.expect(
        format!(
            "a point light inside a glowing block lights {} of the floor around it",
            percent(reached, 0)
        ),
        reached >= 0.9,
        "the block shouldn't shadow it",
    );
    Ok(())
}

/// A spot light lights only a pool under it, where a point light lights all the floor.
fn spot_light(checks: &mut Checks) -> VkResult<()> {
    let point = Light::point(Vec3::new(9.0, 14.0, 0.5), [100.0; 3], 30.0);
    let spot = point.with_spot(Spot {
        direction: Vec3::NEG_Y,
        inner_angle: 15f32.to_radians(),
        outer_angle: 20f32.to_radians(),
    });
    let view = View {
        eye: Vec3::new(9.0, 20.0, 0.5),
        target: Vec3::new(9.0, 8.0, 0.6),
    };
    let lit = |light: Light| -> VkResult<f32> {
        let frame = checks.render(Scene::new(&floor, lights_only()).lights(&[light]), view)?;
        Ok(fraction_over(&frame, BOUNCED))
    };
    let (lit_spot, lit_point) = (lit(spot)?, lit(point)?);
    checks.expect(
        format!(
            "a point light lights {} of the floor under it",
            percent(lit_point, 0)
        ),
        lit_point >= 0.9,
        "it should light all of it",
    );
    // The cone, 20 degrees wide, falls on a pool about 4 voxels across.
    checks.expect(
        format!(
            "a spot light lights {} of the floor under it",
            percent(lit_spot, 0)
        ),
        (0.01..0.3).contains(&lit_spot),
        "it should light only a pool",
    );
    Ok(())
}

/// A mirror floor shows a glowing block out of view, a matte one doesn't.
fn mirror(checks: &mut Checks) -> VkResult<()> {
    let over = |floor_type: u16| {
        move |x: i32, y: i32, z: i32| {
            let glowing = (-4..4).contains(&x) && (40..44).contains(&y) && (16..24).contains(&z);
            match (glowing, y < 8) {
                (true, _) => GLOW,
                (false, true) => floor_type,
                (false, false) => 0,
            }
        }
    };
    let view = View {
        eye: Vec3::new(0.5, 20.0, -10.0),
        target: Vec3::new(0.5, 8.0, 10.0),
    };
    // Pixels as bright as the block's reflection, which only a mirror shows.
    let showing = |floor_type| -> VkResult<f32> {
        let frame = checks.render(Scene::new(&over(floor_type), lights_only()), view)?;
        Ok(fraction_over(&frame, 0.5))
    };
    let (mirror, matte) = (showing(MIRROR)?, showing(GRAY)?);
    checks.expect(
        format!(
            "{} of a mirror floor shows a glowing block over it, {} of a matte one",
            percent(mirror, 1),
            percent(matte, 1)
        ),
        mirror >= 0.01 && matte <= 0.001,
        "only the mirror should reflect it",
    );
    Ok(())
}

/// A wall of coarser chunks shadows a point light once the voxel grid has their level.
fn coarse_wall(checks: &mut Checks) -> VkResult<()> {
    // Fine chunks' floor, then past them, coarser chunks' floor and wall, with a point light
    // on the fine side, looking down at the coarse floor behind the wall.
    let fine_floor = |x: i32, y: i32, _: i32| if y < 8 && x < 64 { GRAY } else { 0 };
    let coarse_walled = |x: i32, y: i32, z: i32| {
        let wall = (64..66).contains(&x) && y < 24 && (-24..24).contains(&z);
        if x >= 64 && (y < 8 || wall) { GRAY } else { 0 }
    };
    let coarse_floor = |x: i32, y: i32, _: i32| if x >= 64 && y < 8 { GRAY } else { 0 };
    let lamp = [Light::point(Vec3::new(50.0, 12.0, 0.5), [400.0; 3], 60.0)];
    let view = View {
        eye: Vec3::new(75.0, 14.0, 0.5),
        target: Vec3::new(75.0, 8.0, 0.6),
    };
    let lit = |coarse: Voxels, levels| -> VkResult<f32> {
        let scene = Scene::new(&fine_floor, lights_only())
            .lights(&lamp)
            .coarse(coarse, levels);
        Ok(mean(&checks.render(scene, view)?))
    };
    let open = lit(&coarse_floor, 2)?;
    let (shadowed, unshadowed) = (lit(&coarse_walled, 2)?, lit(&coarse_walled, 1)?);
    checks.expect(
        format!("a point light lights a coarser floor to {open:.3}"),
        open >= BOUNCED,
        "it should reach the floor",
    );
    let (through, unshadowed) = (shadowed / open, unshadowed / open);
    checks.expect(
        format!(
            "a floor behind a coarser wall gets {} of a point light's light",
            percent(through, 1)
        ),
        through <= SHADOW_LEAK,
        "the wall's level should shadow it",
    );
    checks.expect(
        format!(
            "{} gets past the wall without its level in the voxel grid",
            percent(unshadowed, 0)
        ),
        unshadowed >= MOST,
        "chunks of levels the grid doesn't have shouldn't shadow",
    );
    Ok(())
}

/// Coarser chunks are lit as fine ones are, and probes touching them don't darken them.
fn coarse_chunks(checks: &mut Checks) -> VkResult<()> {
    let air = |_: i32, _: i32, _: i32| 0;
    let lit = mean(&checks.render(Scene::new(&air, sky()).coarse(&floor, 2), FLOOR_VIEW)?);
    checks.expect(
        format!("a floor of coarser chunks under a white sky is {lit:.3}, {ALBEDO} is exact"),
        (lit - ALBEDO).abs() <= ALBEDO * ALBEDO_TOLERANCE,
        "it should be lit as a fine one is",
    );

    // Edges and tops where the probes' cells' centers are.
    let steps = |x: i32, y: i32, z: i32| {
        let step = ((x + 8).div_euclid(16) + (z + 8).div_euclid(16)).rem_euclid(3);
        if y < 8 + 16 * step { GRAY } else { 0 }
    };
    let view = View {
        eye: Vec3::new(-10.0, 70.0, -20.0),
        target: Vec3::new(10.0, 24.0, 10.0),
    };
    // Dim pixels, which inner corners have too, as many on fine steps.
    let dim =
        |scene| -> VkResult<f32> { Ok(1.0 - fraction_over(&checks.render(scene, view)?, 0.1)) };
    let (dim_coarse, dim_fine) = (
        dim(Scene::new(&air, sky()).coarse(&steps, 2))?,
        dim(Scene::new(&steps, sky()))?,
    );
    checks.expect(
        format!(
            "{} of steps of coarser chunks under a white sky are dim, {} of fine ones",
            percent(dim_coarse, 1),
            percent(dim_fine, 1)
        ),
        dim_coarse <= dim_fine + 0.01,
        "probes touching them shouldn't darken them",
    );
    Ok(())
}

/// A floor of fine chunks on one side and coarser ones on the other, as around a camera
/// whose fine chunks reach only so far, lit the same on either side of the seam as a floor
/// all of fine chunks, under a sun and sky, with two cascades, at half trace resolution.
fn level_seam(checks: &mut Checks) -> VkResult<()> {
    let fine_half = |x: i32, y: i32, z: i32| if x < 0 { floor(x, y, z) } else { 0 };
    let coarse_half = |x: i32, y: i32, z: i32| if x >= 0 { floor(x, y, z) } else { 0 };
    let two = ProbeConfig {
        cascades: 2,
        ..ProbeConfig::default()
    };
    let view = View {
        eye: Vec3::new(0.0, 30.0, -40.0),
        target: Vec3::new(0.0, 8.0, 20.0),
    };
    let lighting = half(bright());
    let mixed = checks.render(
        Scene::new(&fine_half, lighting)
            .coarse(&coarse_half, 2)
            .probes(two),
        view,
    )?;
    let fine = checks.render(Scene::new(&floor, lighting).probes(two), view)?;
    let side_mean = |frame: &[f32], right: bool| {
        let values: Vec<f32> = frame
            .iter()
            .enumerate()
            .filter(|(index, _)| ((index % WIDTH as usize) >= WIDTH as usize / 2) == right)
            .map(|(_, value)| *value)
            .collect();
        mean(&values)
    };
    for (side, right) in [("fine", false), ("coarse", true)] {
        let ratio = side_mean(&mixed, right) / side_mean(&fine, right).max(1e-6);
        checks.expect(
            format!("the {side} side of a floor where its chunks' level changes is {ratio:.3}x as bright as all fine"),
            (ratio - 1.0).abs() <= 0.05,
            "the seam between levels shows",
        );
    }
    Ok(())
}

/// Coarser chunks and probe cascades, see `coarse_chunks` and `cascades`.
fn coarse_chunks_and_cascades(checks: &mut Checks) -> VkResult<()> {
    coarse_chunks(checks)?;
    level_seam(checks)?;
    cascades(checks)
}

/// Coarser cascades bring bounced light to more of a box than too few probes cover.
fn cascades(checks: &mut Checks) -> VkResult<()> {
    let lit = |cascades| -> VkResult<f32> {
        let few = ProbeConfig {
            probe_counts: [4; 3],
            cascades,
            ..ProbeConfig::default()
        };
        let frame = checks.render(Scene::new(&holed, sun_only()).probes(few), HOLE_VIEW)?;
        Ok(fraction_over(&frame, BOUNCED))
    };
    let (one, four) = (lit(1)?, lit(4)?);
    checks.expect(
        format!(
            "{} of a box with a hole in its roof is lit by 4 cascades of 4 probes a side, {} by 1",
            percent(four, 0),
            percent(one, 0)
        ),
        four >= one + 0.1,
        "coarser cascades should reach further",
    );
    Ok(())
}

/// Every check, in the order they run.
const CHECKS: [fn(&mut Checks) -> VkResult<()>; 13] = [
    sky_floor,
    sealed_box,
    sealed_hall,
    colored_floor,
    steady_light,
    holed_box,
    lit_box,
    shadowed_light,
    lamp_in_block,
    spot_light,
    mirror,
    coarse_wall,
    coarse_chunks_and_cascades,
];

/// What was wrong with the lighting, if anything. Skipped, with `Ok(None)`, when frames can't
/// be captured.
pub fn check(app: &App) -> VkResult<Option<Vec<String>>> {
    let mut checks = Checks {
        app,
        failures: Vec::new(),
    };
    for check in CHECKS {
        match check(&mut checks) {
            Err(CANT_CAPTURE) => return Ok(None),
            result => result?,
        }
    }
    Ok(Some(checks.failures))
}
