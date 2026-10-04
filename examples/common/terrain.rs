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
    let chunks = (CHUNKS[0] * CHUNKS[1] * CHUNKS[2]) as u32;
    RendererConfig {
        mesh_capacity: MeshCapacity {
            type_words: chunks * (EDGE.pow(3) as u32 / 8 + 96),
            ..MeshCapacity::default()
        },
        global_illumination: Some(GiConfig::default()),
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
pub fn create(renderer: &mut Renderer) -> VkResult<Vec<MeshId>> {
    renderer.set_palette(&palette());
    renderer.lighting.sun_direction = Vec3::new(-0.6, 0.5, 0.25);
    renderer.lighting.shadow_grid = Some(ShadowGrid {
        voxel_size: 1.0,
        voxels_per_edge: EDGE as u32,
    });
    // One shadow ray per 2x2 pixels, which looks nearly the same for well under half the cost.
    renderer.lighting.shadow_resolution = ShadowResolution::Half;
    let edge = EDGE as i32;
    let world = CHUNKS.map(|chunks| chunks as i32 * edge);
    let mut ids = Vec::new();
    for chunk_z in 0..CHUNKS[2] as i32 {
        for chunk_y in 0..CHUNKS[1] as i32 {
            for chunk_x in 0..CHUNKS[0] as i32 {
                let corner = [chunk_x, chunk_y, chunk_z].map(|chunk| chunk * edge);
                // Heights of the chunk's columns and the ring around it, for the shell.
                let ring = EDGE + 2;
                let mut heights = vec![0; ring * ring];
                for z in -1..=edge {
                    for x in -1..=edge {
                        heights[(z + 1) as usize * ring + (x + 1) as usize] =
                            height(corner[0] + x, corner[2] + z);
                    }
                }
                let height_at =
                    |x: i32, z: i32| heights[(z + 1) as usize * ring + (x + 1) as usize];
                // The world's edges are walled off, and its underside never seen.
                let solid = |x: i32, y: i32, z: i32| {
                    let [wx, wy, wz] = [corner[0] + x, corner[1] + y, corner[2] + z];
                    let inside = (0..world[0]).contains(&wx) && (0..world[2]).contains(&wz);
                    wy < 0 || (inside && wy < height_at(x, z))
                };

                let types: Vec<u16> = (0..EDGE.pow(3))
                    .map(|index| {
                        let (x, y, z) = (index % EDGE, index / EDGE % EDGE, index / EDGE.pow(2));
                        let ground = height_at(x as i32, z as i32);
                        let depth = ground - 1 - (corner[1] + y as i32);
                        if depth < 0 {
                            0
                        } else {
                            voxel_type(ground, depth)
                        }
                    })
                    .collect();
                // Chunks under the ground have no faces, but still cast shadows.
                if types.iter().all(|&id| id == 0) {
                    continue;
                }
                let mesh = greedy::mesh(&ChunkMask::from_fn(solid));
                ids.push(renderer.create_chunk_mesh(&ChunkMesh {
                    indices: &mesh.indices,
                    vertices: ChunkVertices::Greedy(&mesh.vertices),
                    origin: Vec3::from(corner.map(|voxel| voxel as f32)),
                    voxel_size: 1.0,
                    types: &VoxelTypes::pack(EDGE as u32, &types),
                })?);
            }
        }
    }
    Ok(ids)
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
