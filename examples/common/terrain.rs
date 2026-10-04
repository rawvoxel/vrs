//! Rolling hills of voxel chunks from a heightmap, shared by the terrain and screenshot examples.

use crate::greedy::{self, ChunkMask, EDGE};
use ash::prelude::VkResult;
use glam::{Mat4, Vec3};
use vrs::{
    mesh::{MeshCapacity, MeshId},
    renderer::{GiConfig, Renderer, RendererConfig, ShadowGrid, ShadowResolution},
    voxel::{ChunkMesh, ChunkVertices, VoxelTypes},
};

/// Chunks across the world along x and z, and stacked up y.
pub const CHUNKS: [usize; 3] = [8, 2, 8];
const STONE: u16 = 1;
const DIRT: u16 = 2;
const GRASS: u16 = 3;
const SAND: u16 = 4;
const SNOW: u16 = 5;
/// Ground at or below this height is sand, at or above `SNOW_LINE` it's snow.
const BEACH_LINE: i32 = 30;
const SNOW_LINE: i32 = 80;

/// Every chunk has up to six voxel types, 4 bits each, a few words of header and palette, 16 of
/// occupancy and 64 of light probe cells, so it's room for them all without growing.
pub fn config() -> RendererConfig {
    let ring = (CHUNKS[0] / 2 + 2 * RING_CHUNKS) * (CHUNKS[2] / 2 + 2 * RING_CHUNKS);
    let chunks = (CHUNKS[0] * CHUNKS[1] * CHUNKS[2] + ring) as u32;
    RendererConfig {
        mesh_capacity: MeshCapacity {
            type_words: chunks * (EDGE.pow(3) as u32 / 8 + 96),
            ..MeshCapacity::default()
        },
        global_illumination: Some(GiConfig {
            cascades: 2,
            ..GiConfig::default()
        }),
        ..RendererConfig::default()
    }
}

/// Linear colors by voxel type, 0 is empty space.
fn palette() -> [[f32; 3]; 6] {
    [
        [0.0; 3],
        [0.30, 0.30, 0.32],
        [0.28, 0.17, 0.08],
        [0.13, 0.42, 0.08],
        [0.76, 0.66, 0.38],
        [0.92, 0.94, 0.97],
    ]
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

/// Meshes every chunk with ground in it and adds it to `renderer`, setting its palette, and
/// lights it with a low sun whose shadows are traced through the chunks at half resolution.
///
/// Around the world's `CHUNKS`, a ring `RING_CHUNKS` wide of chunks of the next level of
/// detail carries the hills on, each of voxels twice as big, as far again on every side.
pub fn create(renderer: &mut Renderer) -> VkResult<Vec<MeshId>> {
    renderer.set_palette(&palette());
    renderer.lighting.sun_direction = Vec3::new(-0.6, 0.5, 0.25);
    renderer.lighting.shadow_grid = Some(ShadowGrid {
        levels: 2,
        ..ShadowGrid::new(1.0, EDGE as u32)
    });
    // One shadow ray per 2x2 pixels, which looks nearly the same for well under half the cost.
    renderer.lighting.shadow_resolution = ShadowResolution::Half;
    let edge = EDGE as i32;
    let fine = CHUNKS.map(|chunks| chunks as i32 * edge);
    // The ring's chunks, twice as wide, from one ring width before the world to one after.
    let coarse_edge = edge * 2;
    let ring = RING_CHUNKS as i32 * coarse_edge;
    let mut ids = Vec::new();
    for chunk_z in 0..CHUNKS[2] as i32 {
        for chunk_y in 0..CHUNKS[1] as i32 {
            for chunk_x in 0..CHUNKS[0] as i32 {
                let corner = [chunk_x, chunk_y, chunk_z].map(|chunk| chunk * edge);
                ids.extend(create_chunk(renderer, corner, 0, ring)?);
            }
        }
    }
    let coarse_chunks = [fine[0], fine[2]].map(|size| (size + 2 * ring) / coarse_edge);
    for chunk_z in 0..coarse_chunks[1] {
        for chunk_x in 0..coarse_chunks[0] {
            let corner = [
                chunk_x * coarse_edge - ring,
                0,
                chunk_z * coarse_edge - ring,
            ];
            let in_world = (0..fine[0]).contains(&corner[0]) && (0..fine[2]).contains(&corner[2]);
            if !in_world {
                ids.extend(create_chunk(renderer, corner, 1, ring)?);
            }
        }
    }
    Ok(ids)
}

/// Coarser chunks across the ring around the world.
pub const RING_CHUNKS: usize = 4;

/// The chunk of level `level` at world voxel `corner`, of voxels `2^level` wide, unless it's
/// empty. The world, with the ring `ring` voxels wide around it, is walled off at its edges.
fn create_chunk(
    renderer: &mut Renderer,
    corner: [i32; 3],
    level: u32,
    ring: i32,
) -> VkResult<Option<MeshId>> {
    let edge = EDGE as i32;
    let scale = 1 << level;
    let world_min = -ring;
    let world_max = [CHUNKS[0], CHUNKS[2]].map(|chunks| chunks as i32 * edge + ring);
    // Heights of the chunk's columns and the ring around it, for the shell.
    let around = EDGE + 2;
    let mut heights = vec![0; around * around];
    for z in -1..=edge {
        for x in -1..=edge {
            heights[(z + 1) as usize * around + (x + 1) as usize] =
                height(corner[0] + x * scale, corner[2] + z * scale);
        }
    }
    let height_at = |x: i32, z: i32| heights[(z + 1) as usize * around + (x + 1) as usize];
    // The world's edges are walled off, and its underside never seen. Past the chunk, voxels
    // are as the level there has them, so faces meet across seams between levels.
    let fine = [CHUNKS[0], CHUNKS[2]].map(|chunks| chunks as i32 * edge);
    let solid = |x: i32, y: i32, z: i32| {
        let [wx, wy, wz] = [
            corner[0] + x * scale,
            corner[1] + y * scale,
            corner[2] + z * scale,
        ];
        let inside =
            (world_min..world_max[0]).contains(&wx) && (world_min..world_max[1]).contains(&wz);
        if wy < 0 || !inside {
            return wy < 0;
        }
        if (0..edge).contains(&x) && (0..edge).contains(&z) {
            return wy < height_at(x, z);
        }
        let in_fine = (0..fine[0]).contains(&wx) && (0..fine[1]).contains(&wz);
        let coarse = |axis: i32| axis.div_euclid(2) * 2;
        match in_fine {
            true => wy < height(wx, wz),
            false => coarse(wy) < height(coarse(wx), coarse(wz)),
        }
    };

    let types: Vec<u16> = (0..EDGE.pow(3))
        .map(|index| {
            let (x, y, z) = (index % EDGE, index / EDGE % EDGE, index / EDGE.pow(2));
            let ground = height_at(x as i32, z as i32);
            let depth = (ground - 1 - (corner[1] + y as i32 * scale)) / scale;
            if ground - 1 < corner[1] + y as i32 * scale {
                0
            } else {
                voxel_type(ground, depth)
            }
        })
        .collect();
    // Chunks under the ground have no faces, but still cast shadows.
    if types.iter().all(|&id| id == 0) {
        return Ok(None);
    }
    let mesh = greedy::mesh(&ChunkMask::from_fn(solid));
    renderer
        .create_chunk_mesh(&ChunkMesh {
            indices: &mesh.indices,
            vertices: ChunkVertices::Greedy(&mesh.vertices),
            origin: Vec3::from(corner.map(|voxel| voxel as f32)),
            voxel_size: scale as f32,
            types: &VoxelTypes::pack(EDGE as u32, &types),
        })
        .map(Some)
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
    // glam's projection is OpenGL's, with y up, Vulkan's clip space has y down.
    let mut proj = Mat4::perspective_rh(70f32.to_radians(), aspect, 0.5, 2000.0);
    proj.y_axis.y = -proj.y_axis.y;
    (view, proj)
}
