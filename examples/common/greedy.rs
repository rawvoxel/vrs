//! Binary greedy meshing for cubed voxel chunks 64 voxels wide, shared by the examples.
//!
//! Every face of a solid voxel next to an empty one is found with bit operations, a row of
//! voxels at a time, then merged with its neighbors on the same plane into rectangles.

use ash::prelude::VkResult;
use glam::{Vec3, Vec3Swizzles};
use vrs::{
    mesh::{Index, MeshId},
    renderer::Renderer,
    voxel::{ChunkMesh, ChunkVertices, Face, GreedyVertex, VoxelTypes},
};

/// Voxels along each edge of a chunk, one per bit of a word.
pub const EDGE: usize = 64;
const LOG2: usize = 6;
const PLANE: usize = EDGE * EDGE;

/// A chunk's solid voxels, and those just outside it, whose faces against the chunk aren't drawn.
pub struct ChunkMask {
    /// Bit `x` of word `z * 64 + y`.
    pub chunk: Vec<u64>,
    /// The six layers of voxels just outside each face, `-x, +x, -y, +y, -z, +z`, 64 words each.
    pub shell: Vec<u64>,
}

impl ChunkMask {
    /// `solid` is asked about every voxel from -1 to 64 on each axis, the shell included.
    pub fn from_fn(solid: impl Fn(i32, i32, i32) -> bool) -> Self {
        let b = EDGE as i32;
        let mut mask = Self {
            chunk: vec![0; PLANE],
            shell: vec![0; 6 * EDGE],
        };
        for z in 0..b {
            for y in 0..b {
                for x in 0..b {
                    if solid(x, y, z) {
                        mask.chunk[(z * b + y) as usize] |= 1 << x;
                    }
                }
            }
        }
        for a in 0..EDGE {
            for c in 0..EDGE {
                let (ai, ci) = (a as i32, c as i32);
                let faces = [
                    solid(-1, ci, ai),
                    solid(b, ci, ai),
                    solid(ai, -1, ci),
                    solid(ai, b, ci),
                    solid(ci, ai, -1),
                    solid(ci, ai, b),
                ];
                for (face, solid) in faces.into_iter().enumerate() {
                    if solid {
                        mask.shell[face * EDGE + a] |= 1 << c;
                    }
                }
            }
        }
        mask
    }
}

/// A chunk's greedy mesh, in voxels of the chunk.
#[derive(Default)]
pub struct GreedyMesh {
    pub indices: Vec<Index>,
    pub vertices: Vec<GreedyVertex>,
}

/// Meshes the faces of `mask`'s voxels wherever the neighbor across the face is empty.
pub fn mesh(mask: &ChunkMask) -> GreedyMesh {
    let (chunk, slices) = (&mask.chunk, &mask.shell);
    let (xyzi, yzxi, zxyi) = (0, PLANE, PLANE << 1);
    // The voxels along each axis (3 planes), then each face's visible voxels (6 planes).
    let mut buffer = vec![0u64; 9 * PLANE];
    let (mask_data, cull_data) = buffer.split_at_mut(3 * PLANE);

    for z in 0..EDGE {
        let zi = z << LOG2;
        for y in 0..EDGE {
            let yi = y << LOG2;
            let mut x_bits = chunk[zi + y];
            while x_bits != 0 {
                let x = x_bits.trailing_zeros() as usize;
                let xi = x << LOG2;
                mask_data[xyzi + zi + y] |= 1 << x; // xyz
                mask_data[yzxi + xi + z] |= 1 << y; // yzx
                mask_data[zxyi + yi + x] |= 1 << z; // zxy
                x_bits &= !(1 << x);
            }
        }
    }

    let (min, max) = (0, EDGE - 1);

    for (axis, neg, pos) in [(0, 0, 1), (1, 2, 3), (2, 4, 5)] {
        let (ai, ni, pi) = (axis << (LOG2 << 1), neg << (LOG2 << 1), pos << (LOG2 << 1));

        for z in 0..EDGE {
            let zi = z << LOG2;
            let (azi, nzi, pzi) = (ai + zi, ni + zi, pi + zi);
            for y in 0..EDGE {
                let y_mask = 1 << y;

                let x_bits = mask_data[azi + y];
                let mut x_bits_neg = x_bits & !(x_bits << 1);
                let mut x_bits_pos = x_bits & !(x_bits >> 1);

                while x_bits_neg != 0 {
                    let x = x_bits_neg.trailing_zeros() as usize;
                    x_bits_neg &= !(1 << x);
                    if !(x == min && slices[neg * EDGE + z] & y_mask == y_mask) {
                        cull_data[nzi + x] |= 1 << y;
                    }
                }
                while x_bits_pos != 0 {
                    let x = x_bits_pos.trailing_zeros() as usize;
                    x_bits_pos &= !(1 << x);
                    if !(x == max && slices[pos * EDGE + z] & y_mask == y_mask) {
                        cull_data[pzi + x] |= 1 << y;
                    }
                }
            }
        }
    }

    let mut mesh = GreedyMesh::default();

    for n in 0..6 {
        let ni = n << (LOG2 << 1);
        for z in 0..EDGE {
            let nzi = ni + (z << LOG2);
            for x in 0..EDGE {
                let mut y_bits = cull_data[nzi + x];

                while y_bits != 0 {
                    let y = y_bits.trailing_zeros() as usize;
                    let h = (y_bits >> y).trailing_ones() as usize;
                    let y_mask = (!0 >> (EDGE - h)) << y;
                    y_bits &= !y_mask;

                    let mut d = 1;

                    for z_next in z + 1..EDGE {
                        let nzi = ni + (z_next << LOG2);
                        let y_bits_next = &mut cull_data[nzi + x];
                        if (*y_bits_next & y_mask) != y_mask {
                            break;
                        }
                        *y_bits_next &= !y_mask;
                        d += 1;
                    }

                    push_quad(
                        &mut mesh,
                        n,
                        (x + (n & 1)) as f32,
                        y as f32,
                        z as f32,
                        h as f32,
                        d as f32,
                    );
                }
            }
        }
    }

    mesh
}

/// Appends one greedy quad: face `n` (`-x, +x, -y, +y, -z, +z`), in the face's rotated
/// frame. `plane` is along the face's axis, its corner at `(b, c)` on the two in-plane
/// axes, `h` long along `b` and `d` along `c`, wound so the face looks along `n`.
fn push_quad(mesh: &mut GreedyMesh, n: usize, plane: f32, b: f32, c: f32, h: f32, d: f32) {
    let vi = Vec3::new(plane, b, c);
    let vo = Vec3::new(vi.x, vi.y + h, vi.z + d);

    // The corners, from the bottom left one, clockwise in the face's frame.
    let mut vertices = [
        vi,
        Vec3::new(vi.x, vi.y + h, vi.z),
        vo,
        Vec3::new(vo.x, vo.y - h, vo.z),
    ];

    // Into the chunk's frame, by the face's axis.
    match n >> 1 {
        1 => {
            for vertex in vertices.iter_mut() {
                *vertex = vertex.zxy();
            }
        }
        2 => {
            for vertex in vertices.iter_mut() {
                *vertex = vertex.yzx();
            }
        }
        _ => {}
    }

    // Faces looking along an axis's positive side wind the other way.
    if n & 1 == 1 {
        vertices.reverse();
    }

    // Two triangles, counter-clockwise on screen when the face looks at the camera.
    let l = mesh.vertices.len() as Index;
    mesh.indices.extend([l, l + 3, l + 1, l + 1, l + 3, l + 2]);

    let face = Face::ALL[n];
    mesh.vertices.extend(vertices.map(|vertex| {
        GreedyVertex::new(vertex, face, EDGE as u32).expect("Quads stay inside the chunk.")
    }));
}

/// The type of each of a chunk's voxels, x-fastest, from `voxel`, asked about each in it.
pub fn chunk_types(voxel: impl Fn(i32, i32, i32) -> u16) -> Vec<u16> {
    (0..EDGE.pow(3))
        .map(|index| {
            let (x, y, z) = (index % EDGE, index / EDGE % EDGE, index / EDGE.pow(2));
            voxel(x as i32, y as i32, z as i32)
        })
        .collect()
}

/// Meshes a chunk of voxels of `types`, see `chunk_types`, and adds it to `renderer` at
/// `origin`, of voxels `voxel_size` wide, unless it's all empty space. `solid` says which
/// voxels are solid, those just around the chunk too, see `ChunkMask::from_fn`.
pub fn create_chunk(
    renderer: &mut Renderer,
    origin: Vec3,
    voxel_size: f32,
    types: &[u16],
    solid: impl Fn(i32, i32, i32) -> bool,
) -> VkResult<Option<MeshId>> {
    if types.iter().all(|&id| id == 0) {
        return Ok(None);
    }
    let mesh = mesh(&ChunkMask::from_fn(solid));
    renderer
        .create_chunk_mesh(&ChunkMesh {
            indices: &mesh.indices,
            vertices: ChunkVertices::Greedy(&mesh.vertices),
            origin,
            voxel_size,
            types: &VoxelTypes::pack(EDGE as u32, types),
        })
        .map(Some)
}
