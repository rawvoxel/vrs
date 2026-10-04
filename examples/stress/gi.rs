//! Global illumination checked against scenes with known answers.
//!
//! - A floor under a uniform white sky, without a sun, lit by its probes as by the sky alone.
//! - A sealed hollow box under a bright sun and sky, seen from inside: no light may leak in.
//! - The same box with a hole in its roof: sunlight bouncing in lights all of it.
//! - The sealed box with a glowing block inside, in the dark: its light reaches the walls.
//! - The box with glass over its hole: sunlight comes through, dimmed by the glass.
//! - The sealed box in the dark with a point light inside: it lights the walls, and bounced
//!   off them, more still.
//! - A point light behind a wall: the floor past the wall is in its shadow, at full and half
//!   shadow resolution.
//! - A spot light over a floor: it lights only a pool under it.
//! - A point light inside a glowing block over a floor: the block doesn't shadow it.
//! - A glowing block high over a mirror floor, out of view: the floor shows it.
//! - A point light behind a wall of coarser chunks: the wall shadows it once the shadow grid
//!   has their level.
//! - The floor under a white sky, of coarser chunks only: lit as the fine one is. Then steps
//!   of coarser chunks, whose sides aren't black where probes touch them.
//! - The box with a hole in its roof, with too few probes to cover it: coarser cascades
//!   bring bounced light to more of it.
//! - The sealed box with 4 cascades: no light leaks in where they blend.

use crate::greedy::{self, ChunkMask, EDGE};
use ash::vk;
use glam::{Mat4, Vec3};
use vrs::{
    App,
    mesh::MeshId,
    renderer::{
        Capture, DrawList, GiConfig, Light, Lighting, Material, Renderer, RendererConfig,
        ShadowGrid, ShadowResolution, Spot,
    },
    voxel::{ChunkMesh, ChunkVertices, VoxelTypes},
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// Enough for several bounces of light to settle.
const FRAMES: u32 = 40;
/// Frames after a change for the probes to mostly catch up with it.
const CHANGE_FRAMES: u32 = 8;
const ALBEDO: f32 = 0.5;

/// Brightness under which a pixel counts as unlit in a sealed box, in linear color.
const UNLIT: f32 = 0.02;
/// Brightness over which a pixel counts as reached by bounced light, in linear color, well
/// clear of black after 8 bit rounding.
const BOUNCED: f32 = 0.002;
/// Pixels allowed to show through one pixel cracks between quads, as a fraction of all.
const CRACKS: f64 = 0.002;

/// Each pixel's red channel, in linear color. `None` when the frame can't be read.
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
fn fraction_over(values: &[f32], threshold: f32) -> f64 {
    values.iter().filter(|&&value| value > threshold).count() as f64 / values.len() as f64
}

/// Plain gray, glowing white, gray glass letting 70% through, then a white metal mirror.
const MATERIALS: [Material; 5] = [
    Material::color([0.0; 3]),
    Material::color([ALBEDO; 3]),
    Material {
        emission: [4.0; 3],
        ..Material::color([1.0; 3])
    },
    Material {
        see_through: true,
        ..Material::color([0.7; 3])
    },
    Material {
        roughness: 0.0,
        metalness: 1.0,
        ..Material::color([0.9; 3])
    },
];

/// Voxels' types by world voxel.
type Voxels<'a> = &'a dyn Fn(i32, i32, i32) -> u16;

/// The frame after `FRAMES` frames of `voxels`' types lit by `lighting`, with the probes on
/// or off, seen from `eye` looking at `target`. `None` when frames can't be captured.
fn render(
    app: &App,
    voxels: impl Fn(i32, i32, i32) -> u16,
    lighting: Lighting,
    eye: Vec3,
    target: Vec3,
) -> Result<Option<Vec<f32>>, vk::Result> {
    render_changing(app, &voxels, lighting, &[], None, eye, target)
}

/// As `render`, lit by `lights` too.
fn render_lit(
    app: &App,
    voxels: impl Fn(i32, i32, i32) -> u16,
    lighting: Lighting,
    lights: &[Light],
    eye: Vec3,
    target: Vec3,
) -> Result<Option<Vec<f32>>, vk::Result> {
    render_changing(app, &voxels, lighting, lights, None, eye, target)
}

/// Each chunk of level `level` around the origin with any voxels, as meshes: 64 voxels wide,
/// each `2^level` world voxels wide, of the type of the world voxel at its low corner.
fn create_chunks(
    renderer: &mut Renderer,
    voxels: Voxels,
    level: u32,
) -> Result<Vec<MeshId>, vk::Result> {
    let scale = 1 << level;
    let mut ids = Vec::new();
    for chunk_x in -1..2 {
        for chunk_z in -1..2 {
            let corner = [chunk_x, 0, chunk_z].map(|chunk| chunk * EDGE as i32 * scale);
            let voxels = |x: i32, y: i32, z: i32| {
                voxels(
                    corner[0] + x * scale,
                    corner[1] + y * scale,
                    corner[2] + z * scale,
                )
            };
            let solid = |x: i32, y: i32, z: i32| voxels(x, y, z) != 0;
            let types: Vec<u16> = (0..EDGE.pow(3))
                .map(|i| {
                    let (x, y, z) = (i % EDGE, i / EDGE % EDGE, i / EDGE.pow(2));
                    voxels(x as i32, y as i32, z as i32)
                })
                .collect();
            if types.iter().all(|&id| id == 0) {
                continue;
            }
            let mesh = greedy::mesh(&ChunkMask::from_fn(solid));
            ids.push(renderer.create_chunk_mesh(&ChunkMesh {
                indices: &mesh.indices,
                vertices: ChunkVertices::Greedy(&mesh.vertices),
                origin: Vec3::from(corner.map(|voxel| voxel as f32)),
                voxel_size: scale as f32,
                types: &VoxelTypes::pack(EDGE as u32, &types),
            })?);
        }
    }
    Ok(ids)
}

/// What `render_scene` renders.
struct Scene<'a> {
    voxels: Voxels<'a>,
    /// Chunks of the next level of detail, of voxels twice as big, and how many levels the
    /// shadow grid has.
    coarse: Option<Voxels<'a>>,
    levels: u32,
    gi: GiConfig,
    lighting: Lighting,
    lights: &'a [Light],
}

impl<'a> Scene<'a> {
    fn new(voxels: Voxels<'a>, lighting: Lighting, lights: &'a [Light]) -> Self {
        Self {
            voxels,
            coarse: None,
            levels: 1,
            gi: GiConfig::default(),
            lighting,
            lights,
        }
    }
}

/// As `render_lit`, then, with `change`, `CHANGE_FRAMES` more frames of its voxels and
/// lighting.
fn render_changing(
    app: &App,
    voxels: Voxels,
    lighting: Lighting,
    lights: &[Light],
    change: Option<(Voxels, Lighting)>,
    eye: Vec3,
    target: Vec3,
) -> Result<Option<Vec<f32>>, vk::Result> {
    render_scene(
        app,
        Scene::new(voxels, lighting, lights),
        change,
        eye,
        target,
    )
}

/// As `render_changing`, for any `scene`.
fn render_scene(
    app: &App,
    scene: Scene,
    change: Option<(Voxels, Lighting)>,
    eye: Vec3,
    target: Vec3,
) -> Result<Option<Vec<f32>>, vk::Result> {
    let Scene {
        voxels,
        coarse,
        levels,
        gi,
        lighting,
        lights,
    } = scene;
    let config = RendererConfig {
        global_illumination: Some(gi),
        ..Default::default()
    };
    let mut renderer = Renderer::with_config(app, WIDTH, HEIGHT, config)?;
    renderer.set_materials(&MATERIALS);
    renderer.lights = lights.to_vec();
    let grid = Some(ShadowGrid {
        levels,
        ..ShadowGrid::new(1.0, EDGE as u32)
    });
    renderer.lighting = Lighting {
        shadow_grid: grid,
        ..lighting
    };
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let mut proj =
        Mat4::perspective_rh(70f32.to_radians(), WIDTH as f32 / HEIGHT as f32, 0.1, 500.0);
    proj.y_axis.y = -proj.y_axis.y;

    let frames = |renderer: &mut Renderer, ids: &[MeshId], count: u32| {
        let meshes: Vec<&MeshId> = ids.iter().collect();
        let draws = DrawList {
            meshes: &meshes,
            view,
            proj,
            ..Default::default()
        };
        (0..count).try_for_each(|_| renderer.submit(WIDTH, HEIGHT, &draws))
    };
    let mut ids = create_chunks(&mut renderer, voxels, 0)?;
    if let Some(coarse) = coarse {
        ids.extend(create_chunks(&mut renderer, coarse, 1)?);
    }
    frames(&mut renderer, &ids, FRAMES)?;
    if let Some((voxels, lighting)) = change {
        for id in ids.drain(..) {
            renderer.destroy_mesh(id);
        }
        ids = create_chunks(&mut renderer, voxels, 0)?;
        renderer.lighting = Lighting {
            shadow_grid: grid,
            ..lighting
        };
        frames(&mut renderer, &ids, CHANGE_FRAMES)?;
    }
    let meshes: Vec<&MeshId> = ids.iter().collect();
    let draws = DrawList {
        meshes: &meshes,
        view,
        proj,
        ..Default::default()
    };
    if !renderer.debug_capture_next_frame() {
        return Ok(None);
    }
    renderer.submit(WIDTH, HEIGHT, &draws)?;
    let capture = renderer.debug_read_capture()?;
    Ok(capture.as_ref().and_then(linear_reds))
}

/// What was wrong with the probes' lighting, if anything. Skipped, with `Ok(None)`, when
/// frames can't be captured.
pub fn check(app: &App) -> Result<Option<Vec<String>>, vk::Result> {
    let mut failures = Vec::new();

    // A uniform white sky, no sun, and a flat floor filling the view.
    let sky = Lighting {
        sun_color: [0.0; 3],
        sky_color: [1.0; 3],
        ground_color: [1.0; 3],
        ..Lighting::default()
    };
    let floor = |_: i32, y: i32, _: i32| (y < 8) as u16;
    let (eye, target) = (Vec3::new(10.0, 20.0, 10.0), Vec3::new(10.0, 8.0, 10.5));
    let Some(floor_lit) = render(app, floor, sky, eye, target)? else {
        return Ok(None);
    };
    let floor_mean = mean(&floor_lit);
    println!("  a floor under a white sky is {floor_mean:.3}, {ALBEDO} is exact");
    if (floor_mean - ALBEDO).abs() > ALBEDO * 0.06 {
        failures.push(format!(
            "a floor under a white sky is {floor_mean:.3}, not {ALBEDO}"
        ));
    }

    // A box 32 voxels wide inside, its walls 2 thick, under a bright sun and sky.
    let bright = Lighting {
        sun_direction: Vec3::new(0.3, 1.0, 0.2),
        sun_color: [3.0; 3],
        sky_color: [1.0; 3],
        ground_color: [1.0; 3],
        ..Lighting::default()
    };
    let shell = |x: i32, y: i32, z: i32| {
        let inside = |v: i32, lo: i32, hi: i32| (lo..hi).contains(&v);
        let outer = inside(x, -18, 18) && inside(y, 4, 40) && inside(z, -18, 18);
        let hollow = inside(x, -16, 16) && inside(y, 6, 38) && inside(z, -16, 16);
        (outer && !hollow) as u16
    };
    let (eye, target) = (Vec3::new(0.5, 20.0, -10.0), Vec3::new(0.5, 18.0, 10.0));
    if let Some(sealed) = render(app, shell, bright, eye, target)? {
        let lit = fraction_over(&sealed, UNLIT);
        println!("  {:.2}% of a sealed box's inside is lit", lit * 100.0);
        if lit > CRACKS {
            failures.push(format!(
                "{:.2}% of a sealed box's inside is lit, light leaks in",
                lit * 100.0
            ));
        }
    }

    // The same box with a hole in its roof, lit only by the sun through it.
    let holed = move |x: i32, y: i32, z: i32| {
        let hole = y >= 38 && (-4..4).contains(&x) && (-4..4).contains(&z);
        if hole { 0 } else { shell(x, y, z) }
    };
    let sun_only = Lighting {
        sun_direction: Vec3::new(0.05, 1.0, 0.1),
        sky_color: [0.0; 3],
        ground_color: [0.0; 3],
        ..bright
    };
    let (eye, target) = (Vec3::new(0.5, 30.0, -14.0), Vec3::new(0.5, 6.0, 10.0));
    let direct = Lighting {
        global_illumination: false,
        ..sun_only
    };
    if let (Some(with), Some(without)) = (
        render(app, holed, sun_only, eye, target)?,
        render(app, holed, direct, eye, target)?,
    ) {
        let (lit_with, lit_without) = (
            fraction_over(&with, BOUNCED),
            fraction_over(&without, BOUNCED),
        );
        println!(
            "  {:.0}% of a box with a hole in its roof is lit with probes, {:.0}% without",
            lit_with * 100.0,
            lit_without * 100.0
        );
        if lit_with < 0.5 || mean(&with) < mean(&without) * 1.1 {
            failures.push(format!(
                "sunlight through a hole lights {:.0}% of a box, bounced light should reach most of it",
                lit_with * 100.0
            ));
        }
    }

    // The same box with glass in its hole: the sun's light comes through, but dimmed.
    let glazed = move |x: i32, y: i32, z: i32| {
        let hole = y >= 38 && (-4..4).contains(&x) && (-4..4).contains(&z);
        if hole {
            shell(x, y, z) * 3
        } else {
            shell(x, y, z)
        }
    };
    if let (Some(open), Some(glass)) = (
        render(app, holed, sun_only, eye, target)?,
        render(app, glazed, sun_only, eye, target)?,
    ) {
        let through = mean(&glass) / mean(&open);
        println!(
            "  a box with glass over its hole is {:.0}% as bright as with it open",
            through * 100.0
        );
        if !(0.4..0.95).contains(&through) {
            failures.push(format!(
                "a box with glass over its hole is {:.0}% as bright as with it open, glass should dim it",
                through * 100.0
            ));
        }
    }

    // Cutting the hole into the sealed box, then turning the sun on over the holed box: a few
    // frames later, the light inside has mostly caught up.
    let dark_holed = Lighting {
        sun_color: [0.0; 3],
        ..sun_only
    };
    if let (Some(converged), Some(cut), Some(sunrise)) = (
        render(app, holed, sun_only, eye, target)?,
        render_changing(
            app,
            &shell,
            sun_only,
            &[],
            Some((&holed, sun_only)),
            eye,
            target,
        )?,
        render_changing(
            app,
            &holed,
            dark_holed,
            &[],
            Some((&holed, sun_only)),
            eye,
            target,
        )?,
    ) {
        for (name, changed) in [
            ("cutting a hole in a box", cut),
            ("the sun coming out", sunrise),
        ] {
            let caught_up = mean(&changed) / mean(&converged);
            println!(
                "  {CHANGE_FRAMES} frames after {name}, the light inside is {:.0}% of the way there",
                caught_up * 100.0
            );
            if caught_up < 0.9 {
                failures.push(format!(
                    "{CHANGE_FRAMES} frames after {name}, the light inside is only {:.0}% of the way there",
                    caught_up * 100.0
                ));
            }
        }
    }

    // The sealed box in the dark, with a block glowing inside.
    let dark = Lighting {
        sun_color: [0.0; 3],
        sky_color: [0.0; 3],
        ground_color: [0.0; 3],
        ..Lighting::default()
    };
    let lamp = move |x: i32, y: i32, z: i32| {
        let glowing = (-2..2).contains(&x) && (18..22).contains(&y) && (4..8).contains(&z);
        if glowing { 2 } else { shell(x, y, z) }
    };
    let (eye, target) = (Vec3::new(0.5, 20.0, -10.0), Vec3::new(0.5, 18.0, 10.0));
    let unlit = Lighting {
        global_illumination: false,
        ..dark
    };
    if let (Some(with), Some(without)) = (
        render(app, lamp, dark, eye, target)?,
        render(app, lamp, unlit, eye, target)?,
    ) {
        let (lit_with, lit_without) = (
            fraction_over(&with, BOUNCED),
            fraction_over(&without, BOUNCED),
        );
        println!(
            "  {:.0}% of a dark box with a glowing block inside is lit with probes, {:.0}% without",
            lit_with * 100.0,
            lit_without * 100.0
        );
        if lit_with < 0.5 {
            failures.push(format!(
                "a glowing block lights {:.0}% of the box around it, its light should reach most of it",
                lit_with * 100.0
            ));
        }
    }

    // The sealed box in the dark, with a point light inside.
    let bulb = [Light {
        position: Vec3::new(0.0, 20.0, 6.0),
        color: [400.0; 3],
        range: 60.0,
        spot: None,
    }];
    if let (Some(with), Some(direct), Some(none)) = (
        render_lit(app, shell, dark, &bulb, eye, target)?,
        render_lit(app, shell, unlit, &bulb, eye, target)?,
        render_lit(app, shell, unlit, &[], eye, target)?,
    ) {
        let lit = fraction_over(&direct, BOUNCED);
        let bounced = mean(&with) / mean(&direct);
        println!(
            "  a point light in a dark box lights {:.0}% of it, {:.2}x with bounced light, {:.0}% without the light",
            lit * 100.0,
            bounced,
            fraction_over(&none, BOUNCED) * 100.0
        );
        if lit < 0.5 {
            failures.push(format!(
                "a point light lights {:.0}% of the box around it, it should reach most of it",
                lit * 100.0
            ));
        }
        if bounced < 1.05 {
            failures.push(format!(
                "a point light's bounced light only makes a box {bounced:.2}x as bright"
            ));
        }
    }

    // A wall 16 voxels high on a floor, a point light low on one side, looking down at the
    // floor on the other.
    let walled = |x: i32, y: i32, z: i32| {
        let wall = (0..2).contains(&x) && y < 24 && (-24..24).contains(&z);
        (y < 8 || wall) as u16
    };
    let lamp_post = [Light {
        position: Vec3::new(-6.0, 12.0, 0.5),
        color: [400.0; 3],
        range: 60.0,
        spot: None,
    }];
    let (eye, target) = (Vec3::new(9.0, 14.0, 0.5), Vec3::new(9.0, 8.0, 0.6));
    for resolution in [ShadowResolution::Full, ShadowResolution::Half] {
        let lighting = Lighting {
            shadow_resolution: resolution,
            ..unlit
        };
        if let (Some(shadowed), Some(open)) = (
            render_lit(app, walled, lighting, &lamp_post, eye, target)?,
            render_lit(app, floor, lighting, &lamp_post, eye, target)?,
        ) {
            let through = mean(&shadowed) / mean(&open);
            println!(
                "  a floor behind a wall gets {:.1}% of a point light's light, at {resolution:?} resolution",
                through * 100.0
            );
            if mean(&open) < BOUNCED || through > 0.02 {
                failures.push(format!(
                    "a floor behind a wall gets {:.1}% of a point light's light at {resolution:?} resolution, the wall should shadow it",
                    through * 100.0
                ));
            }
        }
    }

    // A point light inside a glowing block 2 voxels wide, 2 voxels over the floor.
    let lamp_block = |x: i32, y: i32, z: i32| {
        let glowing = (0..2).contains(&x) && (10..12).contains(&y) && (0..2).contains(&z);
        if glowing { 2 } else { floor(x, y, z) }
    };
    let inside = [Light {
        position: Vec3::new(1.0, 11.0, 1.0),
        color: [100.0; 3],
        range: 30.0,
        spot: None,
    }];
    let (eye, target) = (Vec3::new(1.0, 24.0, 1.5), Vec3::new(1.0, 8.0, 1.0));
    if let Some(lit) = render_lit(app, lamp_block, unlit, &inside, eye, target)? {
        let reached = fraction_over(&lit, BOUNCED);
        println!(
            "  a point light inside a glowing block lights {:.0}% of the floor around it",
            reached * 100.0
        );
        if reached < 0.9 {
            failures.push(format!(
                "a point light inside a glowing block lights {:.0}% of the floor around it, the block shouldn't shadow it",
                reached * 100.0
            ));
        }
    }

    // A spot light 6 voxels over the floor, pointing down in a narrow cone.
    let spot = |spot| {
        [Light {
            position: Vec3::new(9.0, 14.0, 0.5),
            color: [100.0; 3],
            range: 30.0,
            spot,
        }]
    };
    let narrow = Some(Spot {
        direction: Vec3::NEG_Y,
        inner_angle: 15f32.to_radians(),
        outer_angle: 20f32.to_radians(),
    });
    let (eye, target) = (Vec3::new(9.0, 20.0, 0.5), Vec3::new(9.0, 8.0, 0.6));
    if let (Some(cone), Some(point)) = (
        render_lit(app, floor, unlit, &spot(narrow), eye, target)?,
        render_lit(app, floor, unlit, &spot(None), eye, target)?,
    ) {
        let (lit_cone, lit_point) = (
            fraction_over(&cone, BOUNCED),
            fraction_over(&point, BOUNCED),
        );
        println!(
            "  a spot light lights {:.0}% of the floor under it, a point light {:.0}%",
            lit_cone * 100.0,
            lit_point * 100.0
        );
        if !(0.01..0.3).contains(&lit_cone) || lit_point < 0.9 {
            failures.push(format!(
                "a spot light lights {:.0}% of the floor under it and a point light {:.0}%, the spot should light only a pool",
                lit_cone * 100.0,
                lit_point * 100.0
            ));
        }
    }

    // A glowing block high over a floor, above the view, reflected in the floor when it's a
    // mirror.
    let over = |floor_type: u16| {
        move |x: i32, y: i32, z: i32| {
            let glowing = (-4..4).contains(&x) && (40..44).contains(&y) && (16..24).contains(&z);
            if glowing {
                2
            } else {
                (y < 8) as u16 * floor_type
            }
        }
    };
    let (eye, target) = (Vec3::new(0.5, 20.0, -10.0), Vec3::new(0.5, 8.0, 10.0));
    if let (Some(mirror), Some(matte)) = (
        render(app, over(4), unlit, eye, target)?,
        render(app, over(1), unlit, eye, target)?,
    ) {
        let (bright_mirror, bright_matte) =
            (fraction_over(&mirror, 0.5), fraction_over(&matte, 0.5));
        println!(
            "  {:.1}% of a mirror floor shows a glowing block over it, {:.1}% of a matte one",
            bright_mirror * 100.0,
            bright_matte * 100.0
        );
        if bright_mirror < 0.01 || bright_matte > 0.001 {
            failures.push(format!(
                "{:.1}% of a mirror floor shows a glowing block over it and {:.1}% of a matte one, the mirror should reflect it",
                bright_mirror * 100.0,
                bright_matte * 100.0
            ));
        }
    }

    // A floor of fine chunks, then past them, a floor and a wall of coarser chunks, with a
    // point light on the fine side, looking down at the coarse floor behind the wall.
    let fine_floor = |x: i32, y: i32, _: i32| (y < 8 && x < 64) as u16;
    let coarse_walled = |x: i32, y: i32, z: i32| {
        let wall = (64..66).contains(&x) && y < 24 && (-24..24).contains(&z);
        (x >= 64 && (y < 8 || wall)) as u16
    };
    let coarse_floor = |x: i32, y: i32, _: i32| (x >= 64 && y < 8) as u16;
    let lamp = [Light {
        position: Vec3::new(50.0, 12.0, 0.5),
        color: [400.0; 3],
        range: 60.0,
        spot: None,
    }];
    let lod = |coarse, levels| Scene {
        coarse: Some(coarse),
        levels,
        ..Scene::new(&fine_floor, unlit, &lamp)
    };
    let (eye, target) = (Vec3::new(75.0, 14.0, 0.5), Vec3::new(75.0, 8.0, 0.6));
    if let (Some(two_levels), Some(one_level), Some(open)) = (
        render_scene(app, lod(&coarse_walled, 2), None, eye, target)?,
        render_scene(app, lod(&coarse_walled, 1), None, eye, target)?,
        render_scene(app, lod(&coarse_floor, 2), None, eye, target)?,
    ) {
        let (through, unshadowed) = (
            mean(&two_levels) / mean(&open),
            mean(&one_level) / mean(&open),
        );
        println!(
            "  a floor behind a coarser wall gets {:.1}% of a point light's light, {:.0}% without the wall's level",
            through * 100.0,
            unshadowed * 100.0
        );
        if mean(&open) < BOUNCED || through > 0.02 || unshadowed < 0.5 {
            failures.push(format!(
                "a floor behind a coarser wall gets {:.1}% of a point light's light, and {:.0}% without the wall's level, its level should shadow it",
                through * 100.0,
                unshadowed * 100.0
            ));
        }
    }

    // The floor under a white sky, of coarser chunks only, whose voxels' corners are where
    // the probes' cells' centers are.
    let air = |_: i32, _: i32, _: i32| 0;
    let coarse = Scene {
        coarse: Some(&floor),
        levels: 2,
        ..Scene::new(&air, sky, &[])
    };
    let (eye, target) = (Vec3::new(10.0, 20.0, 10.0), Vec3::new(10.0, 8.0, 10.5));
    if let Some(lit) = render_scene(app, coarse, None, eye, target)? {
        let coarse_mean = mean(&lit);
        println!(
            "  a floor of coarser chunks under a white sky is {coarse_mean:.3}, {ALBEDO} is exact"
        );
        if (coarse_mean - ALBEDO).abs() > ALBEDO * 0.06 {
            failures.push(format!(
                "a floor of coarser chunks under a white sky is {coarse_mean:.3}, not {ALBEDO}"
            ));
        }
    }
    let steps = |x: i32, y: i32, z: i32| {
        // Edges and tops where the probes' cells' centers are.
        let step = ((x + 8).div_euclid(16) + (z + 8).div_euclid(16)).rem_euclid(3);
        (y < 8 + 16 * step) as u16
    };
    let stepped = Scene {
        coarse: Some(&steps),
        levels: 2,
        ..Scene::new(&air, sky, &[])
    };
    let (eye, target) = (Vec3::new(-10.0, 70.0, -20.0), Vec3::new(10.0, 24.0, 10.0));
    if let (Some(coarse), Some(fine)) = (
        render_scene(app, stepped, None, eye, target)?,
        render(app, steps, sky, eye, target)?,
    ) {
        let (dim_coarse, dim_fine) = (
            1.0 - fraction_over(&coarse, 0.1),
            1.0 - fraction_over(&fine, 0.1),
        );
        println!(
            "  {:.1}% of steps of coarser chunks under a white sky are dim, {:.1}% of fine ones",
            dim_coarse * 100.0,
            dim_fine * 100.0
        );
        if dim_coarse > dim_fine + 0.01 {
            failures.push(format!(
                "{:.1}% of steps of coarser chunks under a white sky are dim and {:.1}% of fine ones, probes touching them shouldn't darken them",
                dim_coarse * 100.0,
                dim_fine * 100.0
            ));
        }
    }

    // The box with a hole in its roof, with too few probes in each cascade to cover it, then
    // the sealed box with 4 cascades.
    let (eye, target) = (Vec3::new(0.5, 30.0, -14.0), Vec3::new(0.5, 6.0, 10.0));
    let few = |cascades| Scene {
        gi: GiConfig {
            probe_counts: [4; 3],
            cascades,
            ..GiConfig::default()
        },
        ..Scene::new(&holed, sun_only, &[])
    };
    if let (Some(one), Some(four)) = (
        render_scene(app, few(1), None, eye, target)?,
        render_scene(app, few(4), None, eye, target)?,
    ) {
        let (lit_one, lit_four) = (fraction_over(&one, BOUNCED), fraction_over(&four, BOUNCED));
        println!(
            "  {:.0}% of a box with a hole in its roof is lit by 4 cascades of 4 probes a side, {:.0}% by 1",
            lit_four * 100.0,
            lit_one * 100.0
        );
        if lit_four < lit_one + 0.1 {
            failures.push(format!(
                "{:.0}% of a box with a hole in its roof is lit by 4 cascades of 4 probes a side and {:.0}% by 1, coarser cascades should reach further",
                lit_four * 100.0,
                lit_one * 100.0
            ));
        }
    }
    let (eye, target) = (Vec3::new(0.5, 20.0, -10.0), Vec3::new(0.5, 18.0, 10.0));
    let four = Scene {
        gi: GiConfig {
            cascades: 4,
            ..GiConfig::default()
        },
        ..Scene::new(&shell, bright, &[])
    };
    if let Some(sealed) = render_scene(app, four, None, eye, target)? {
        let lit = fraction_over(&sealed, UNLIT);
        println!(
            "  {:.2}% of a sealed box's inside is lit with 4 cascades",
            lit * 100.0
        );
        if lit > CRACKS {
            failures.push(format!(
                "{:.2}% of a sealed box's inside is lit with 4 cascades, light leaks in",
                lit * 100.0
            ));
        }
    }
    Ok(Some(failures))
}
