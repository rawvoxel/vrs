//! Rolling hills of voxel chunks from a heightmap, shared by the terrain and screenshot examples.

use crate::{
    camera,
    greedy::{self, EDGE},
};
use ash::prelude::VkResult;
use glam::{Mat4, Vec3};
use vrs::{
    mesh::MeshId,
    renderer::{
        MeshCapacity, ProbeConfig, Renderer, RendererConfig, ShadowGrid, ShadowResolution, Sky,
        Tonemapping,
    },
};

/// Chunks across the world along x and z, and stacked up y.
pub const CHUNKS: [usize; 3] = [8, 2, 8];
/// Chunks of the next level of detail, twice as wide, across the ring around the world.
pub const RING_CHUNKS: usize = 4;

/// Linear colors by voxel type, 0 is empty space, then `STONE` to `SNOW`.
const PALETTE: [[f32; 3]; 6] = [
    [0.0; 3],
    [0.30, 0.30, 0.32],
    [0.28, 0.17, 0.08],
    [0.13, 0.42, 0.08],
    [0.76, 0.66, 0.38],
    [0.92, 0.94, 0.97],
];
const STONE: u16 = 1;
const DIRT: u16 = 2;
const GRASS: u16 = 3;
const SAND: u16 = 4;
const SNOW: u16 = 5;
/// Ground at or below this height is sand, at or above `SNOW_LINE` it's snow.
const BEACH_LINE: i32 = 30;
const SNOW_LINE: i32 = 80;

/// Two cascades of light probes, the second for the ring, and a mesh pool with room for every
/// chunk's voxel types without growing: up to six types, 4 bits each, a few words of header and
/// palette, 16 of occupancy and 64 of light probe cells.
pub fn config() -> RendererConfig {
    let ring = (CHUNKS[0] / 2 + 2 * RING_CHUNKS) * (CHUNKS[2] / 2 + 2 * RING_CHUNKS);
    let chunks = (CHUNKS[0] * CHUNKS[1] * CHUNKS[2] + ring) as u32;
    RendererConfig {
        mesh_capacity: MeshCapacity {
            type_words: chunks * (EDGE.pow(3) as u32 / 8 + 96),
            ..MeshCapacity::default()
        },
        global_illumination: Some(ProbeConfig {
            cascades: 2,
            ..ProbeConfig::default()
        }),
        ..RendererConfig::default()
    }
}

/// The ground's height in voxels at a column, anywhere in the world.
fn height(x: i32, z: i32) -> i32 {
    let (x, z) = (x as f32, z as f32);
    let hills = 22.0 * (x * 0.021).sin() * (z * 0.017).cos();
    let ridges = 9.0 * ((x + z) * 0.047).sin() + 5.0 * ((x - 2.0 * z) * 0.083).cos();
    (52.0 + hills + ridges) as i32
}

/// What's at a voxel `depth` voxels below the surface of ground `height` voxels high.
fn voxel_type(height: i32, depth: i32) -> u16 {
    match depth {
        0 if height <= BEACH_LINE => SAND,
        0 if height >= SNOW_LINE => SNOW,
        0 => GRASS,
        1..4 if height <= BEACH_LINE => SAND,
        1..4 => DIRT,
        _ => STONE,
    }
}

/// Where the world's levels are, in voxels of the finest along x and z.
struct World {
    /// The finest chunks' end, from 0.
    fine: [i32; 2],
    /// The ring's start and end, walled off past them.
    min: i32,
    max: [i32; 2],
}

impl World {
    fn new() -> Self {
        let fine = [CHUNKS[0], CHUNKS[2]].map(|chunks| (chunks * EDGE) as i32);
        let ring = (RING_CHUNKS * EDGE * 2) as i32;
        Self {
            fine,
            min: -ring,
            max: fine.map(|end| end + ring),
        }
    }

    /// Whether column `x`, `z` has the finest chunks.
    fn in_fine(&self, x: i32, z: i32) -> bool {
        (0..self.fine[0]).contains(&x) && (0..self.fine[1]).contains(&z)
    }

    /// Whether voxel `x`, `y`, `z` is solid, as the level there has it, so faces meet across
    /// seams between levels. The world's edges are walled off, and its underside never seen.
    fn solid(&self, x: i32, y: i32, z: i32) -> bool {
        let inside = (self.min..self.max[0]).contains(&x) && (self.min..self.max[1]).contains(&z);
        if y < 0 || !inside {
            return y < 0;
        }
        if self.in_fine(x, z) {
            return y < height(x, z);
        }
        let coarse = |axis: i32| axis.div_euclid(2) * 2;
        coarse(y) < height(coarse(x), coarse(z))
    }
}

/// Sets `renderer`'s palette and lights it with a low sun whose shadows are traced through
/// the chunks at half resolution, then meshes every chunk with ground in it and adds it.
///
/// Around the world's `CHUNKS`, a ring `RING_CHUNKS` wide of chunks of the next level of
/// detail carries the hills on, each of voxels twice as big, as far again on every side.
pub fn create_world(renderer: &mut Renderer) -> VkResult<Vec<MeshId>> {
    renderer.set_palette(&PALETTE);
    renderer.lighting.sun_direction = Vec3::new(-0.6, 0.5, 0.25);
    renderer.lighting.shadow_grid = Some(ShadowGrid::new(1.0, EDGE as u32).with_levels(2));
    // One shadow ray per 2x2 pixels, which looks nearly the same for well under half the cost.
    renderer.lighting.shadow_resolution = ShadowResolution::Half;
    // A blue sky, the far hills fading into it, and highlights rolling off rather than clipping.
    renderer.lighting.sky = Some(Sky {
        haze_distance: 1500.0,
        ..Default::default()
    });
    renderer.lighting.tonemapping = Tonemapping::Aces;
    let world = World::new();
    let edge = EDGE as i32;
    let mut ids = Vec::new();
    for chunk_z in 0..CHUNKS[2] as i32 {
        for chunk_y in 0..CHUNKS[1] as i32 {
            for chunk_x in 0..CHUNKS[0] as i32 {
                let corner = [chunk_x, chunk_y, chunk_z].map(|chunk| chunk * edge);
                ids.extend(create_chunk(renderer, &world, corner, 0)?);
            }
        }
    }
    let coarse_edge = edge * 2;
    let coarse_chunks = world.max.map(|end| (end - world.min) / coarse_edge);
    for chunk_z in 0..coarse_chunks[1] {
        for chunk_x in 0..coarse_chunks[0] {
            let corner = [
                world.min + chunk_x * coarse_edge,
                0,
                world.min + chunk_z * coarse_edge,
            ];
            if !world.in_fine(corner[0], corner[2]) {
                ids.extend(create_chunk(renderer, &world, corner, 1)?);
            }
        }
    }
    Ok(ids)
}

/// The chunk of level `level` at world voxel `corner`, of voxels `2^level` wide, unless it's
/// all air.
fn create_chunk(
    renderer: &mut Renderer,
    world: &World,
    corner: [i32; 3],
    level: u32,
) -> VkResult<Option<MeshId>> {
    let edge = EDGE as i32;
    let scale = 1 << level;
    let to_world = |x: i32, y: i32, z: i32| {
        [
            corner[0] + x * scale,
            corner[1] + y * scale,
            corner[2] + z * scale,
        ]
    };
    // Heights of the chunk's columns and the ring around it, for the shell.
    let around = EDGE + 2;
    let mut heights = vec![0; around * around];
    for z in -1..=edge {
        for x in -1..=edge {
            let [wx, _, wz] = to_world(x, 0, z);
            heights[(z + 1) as usize * around + (x + 1) as usize] = height(wx, wz);
        }
    }
    let height_at = |x: i32, z: i32| heights[(z + 1) as usize * around + (x + 1) as usize];
    let solid = |x: i32, y: i32, z: i32| {
        let [wx, wy, wz] = to_world(x, y, z);
        let in_chunk = (0..edge).contains(&x) && (0..edge).contains(&z);
        match in_chunk && wy >= 0 {
            true => wy < height_at(x, z),
            false => world.solid(wx, wy, wz),
        }
    };
    let types = greedy::chunk_types(|x, y, z| {
        let ground = height_at(x, z);
        let wy = to_world(x, y, z)[1];
        match wy < ground {
            true => voxel_type(ground, (ground - 1 - wy) / scale),
            false => 0,
        }
    });
    // Only chunks of nothing but air are left out. Those under the ground have no faces but
    // are still created, so their voxels cast shadows.
    let origin = Vec3::from(corner.map(|voxel| voxel as f32));
    greedy::create_chunk(renderer, origin, scale as f32, &types, solid)
}

/// Where the camera starts, over a corner of the world looking across it.
pub const START_POSITION: Vec3 = Vec3::new(-40.0, 130.0, -40.0);
pub const START_YAW: f32 = 0.785;
pub const START_PITCH: f32 = -0.45;

/// The way a camera turned `yaw` radians from +z and `pitch` from level looks.
pub fn forward(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(
        yaw.sin() * pitch.cos(),
        pitch.sin(),
        yaw.cos() * pitch.cos(),
    )
}

/// A camera at `position` looking along `forward`, for a `DrawList`'s view and projection.
pub fn view_proj(position: Vec3, forward: Vec3, aspect: f32) -> (Mat4, Mat4) {
    let view = Mat4::look_to_rh(position, forward, Vec3::Y);
    (view, camera::perspective(70.0, aspect, 0.5, 2000.0))
}
