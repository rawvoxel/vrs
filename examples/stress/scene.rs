//! The chunks and the wall the stress test draws, and which of them should be drawn.
//!
//! Chunks are voxel chunks, colored per voxel through the palette. Each is the face of a slab
//! one voxel thick, half one type and half another, so its color has to come from the voxel
//! under each pixel, not the face. Every third chunk uses packed vertices, the rest greedy.

use ash::vk;
use glam::{Mat4, Vec2, Vec3};
use vrs::{
    App,
    mesh::{MeshCapacity, MeshId},
    renderer::Renderer,
    vertex::Vertex,
    voxel::{ChunkMesh, ChunkVertices, Face, GreedyVertex, PackedVertex, VoxelTypes},
};

pub const GRID: u32 = 16;
pub const CHUNK_SIZE: f32 = 2.0 / GRID as f32 * 0.9;
pub const CHUNK_Z: f32 = 0.5;
pub const WALL_Z: f32 = 0.2;
pub const WALL_HALF_SIZE: f32 = 0.5;
/// Voxels along each edge of a chunk, and of the wall.
pub const EDGE: u32 = 4;
pub const VOXEL_SIZE: f32 = CHUNK_SIZE / EDGE as f32;
/// Shades take palette entries 1 to 10, 0 is empty space.
const SHADES: u32 = 10;
const WALL_TYPE: u16 = SHADES as u16 + 1;
pub const WALL_COLOR: [f32; 3] = [0.9, 0.8, 0.1];
/// Room for half again the grid's geometry, so remeshing has to reuse freed space.
pub const CAPACITY: MeshCapacity = MeshCapacity {
    // Packed vertices are 3 words, greedy 2.
    vertex_words: GRID * GRID * 4 * 3 * 3 / 2 + 16,
    indices: GRID * GRID * 6 * 3 / 2 + 6,
    // A header, two palette words and four index words a chunk.
    type_words: GRID * GRID * 7 * 3 / 2 + 8,
    meshes: GRID * GRID * 3 / 2,
};

/// The colors voxel types draw with.
pub fn palette() -> Vec<[f32; 3]> {
    let shades = (0..SHADES).map(|shade| {
        let shade = shade as f32 / SHADES as f32;
        [shade, 1.0 - shade, 0.5]
    });
    [[0.0; 3]]
        .into_iter()
        .chain(shades)
        .chain([WALL_COLOR])
        .collect()
}

/// The corner of a cell's chunk nearest the origin.
pub fn chunk_min(cell: u32) -> Vec2 {
    Vec2::new((cell % GRID) as f32, (cell / GRID) as f32) / GRID as f32 * 2.0 - 1.0
}

/// A chunk's two voxel types when it's meshed on `frame`, `None` when it's empty, like air.
fn chunk_types(cell: u32, frame: u32) -> Option<[u16; 2]> {
    if (cell + frame).is_multiple_of(13) {
        return None;
    }
    let shade = (frame / 7 + cell) % SHADES;
    Some([1 + shade as u16, 1 + ((shade + SHADES / 2) % SHADES) as u16])
}

/// A chunk's two colors, its low x half and its high x half, when it's meshed on `frame`.
pub fn chunk_colors(cell: u32, frame: u32) -> Option<[[f32; 3]; 2]> {
    let palette = palette();
    chunk_types(cell, frame).map(|types| types.map(|id| palette[id as usize]))
}

/// A slab one voxel thick with its `-Z` face drawn at `face_corner`, facing the camera unless
/// `facing_away`, which backface culling skips. Its low x half is `types[0]`, the rest `types[1]`.
///
/// The slab is the chunk's second layer, so the cells in front of the face are empty and inside
/// the chunk: finding a face's voxel by going the wrong way lands there, not on the slab.
pub fn voxel_slab(
    renderer: &mut Renderer,
    face_corner: Vec3,
    voxel_size: f32,
    types: [u16; 2],
    packed: bool,
    facing_away: bool,
) -> Result<MeshId, vk::Result> {
    let voxels: Vec<u16> = (0..EDGE.pow(3))
        .map(|index| {
            let (x, z) = (index % EDGE, index / (EDGE * EDGE));
            match (z, x < EDGE / 2) {
                (1, true) => types[0],
                (1, false) => types[1],
                _ => 0,
            }
        })
        .collect();
    let types = VoxelTypes::pack(EDGE, &voxels);

    let edge = EDGE as f32;
    let corners = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(edge, 0.0, 1.0),
        Vec3::new(0.0, edge, 1.0),
        Vec3::new(edge, edge, 1.0),
    ];
    let greedy = corners.map(|corner| GreedyVertex::new(corner, Face::NegZ, EDGE).unwrap());
    let packed_vertices =
        corners.map(|corner| PackedVertex::new(corner, Vec3::NEG_Z, EDGE).unwrap());
    // Front faces wind counter-clockwise on screen.
    let indices = match facing_away {
        false => [0, 2, 1, 1, 2, 3],
        true => [0, 1, 2, 2, 1, 3],
    };
    renderer.create_chunk_mesh(&ChunkMesh {
        indices: &indices,
        vertices: match packed {
            true => ChunkVertices::Packed(&packed_vertices),
            false => ChunkVertices::Greedy(&greedy),
        },
        origin: face_corner - Vec3::Z * voxel_size,
        voxel_size,
        types: &types,
    })
}

/// A box's screen rectangle in NDC, `None` when it's outside the frustum.
pub fn screen_rect(view_proj: Mat4, min: Vec3, max: Vec3) -> Option<(Vec2, Vec2)> {
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

pub struct Scene {
    pub chunks: Vec<Option<MeshId>>,
    /// Each chunk's two colors, `None` when it's empty or wasn't meshed.
    pub colors: Vec<Option<[[f32; 3]; 2]>>,
    pub wall: Option<MeshId>,
    /// Backface culling skips the wall, so it hides nothing.
    pub wall_faces_away: bool,
    /// A big mesh out of view, which takes up pool space without being drawn.
    pub ballast: Option<MeshId>,
    pub renderer: Renderer,
    pub app: App,
}

/// Every mesh in the scene, borrowing only its meshes so the renderer stays free.
pub fn live<'a>(
    chunks: &'a [Option<MeshId>],
    wall: &'a Option<MeshId>,
    ballast: &'a Option<MeshId>,
) -> Vec<&'a MeshId> {
    chunks
        .iter()
        .flatten()
        .chain(wall.as_ref())
        .chain(ballast.as_ref())
        .collect()
}

impl Scene {
    pub fn new(mut renderer: Renderer, app: App) -> Self {
        renderer.set_palette(&palette());
        let cells = (GRID * GRID) as usize;
        Self {
            chunks: (0..cells).map(|_| None).collect(),
            colors: vec![None; cells],
            wall: None,
            wall_faces_away: false,
            ballast: None,
            renderer,
            app,
        }
    }

    pub fn clear(&mut self) {
        for id in self.chunks.iter_mut().filter_map(Option::take) {
            self.renderer.destroy_mesh(id);
        }
        self.colors.fill(None);
        if let Some(wall) = self.wall.take() {
            self.renderer.destroy_mesh(wall);
        }
        self.wall_faces_away = false;
        if let Some(ballast) = self.ballast.take() {
            self.renderer.destroy_mesh(ballast);
        }
    }

    /// Destroys every other chunk, scattering free space through the pool.
    pub fn punch_holes(&mut self) {
        for cell in (0..GRID * GRID).step_by(2) {
            if let Some(id) = self.chunks[cell as usize].take() {
                self.renderer.destroy_mesh(id);
            }
            self.colors[cell as usize] = None;
        }
    }

    /// Where the last mesh's vertices end in the pool, in words.
    pub fn vertex_words_end(&self) -> u32 {
        live(&self.chunks, &self.wall, &self.ballast)
            .into_iter()
            .map(|id| self.renderer.mesh(id).vertex_words.end)
            .max()
            .unwrap_or(0)
    }

    /// A colored mesh of at least `words` vertex words, far outside every camera's view,
    /// frustum culled every frame.
    pub fn add_ballast(&mut self, words: u32) -> Result<(), vk::Result> {
        let far = Vertex {
            pos: [1000.0, 1000.0, CHUNK_Z],
            col: [1.0; 3],
        };
        // Colored vertices are 4 words.
        let vertices = vec![far; words.div_ceil(4) as usize];
        self.ballast = Some(self.renderer.create_mesh(&[0, 1, 2], &vertices)?);
        Ok(())
    }

    /// Every chunk, and the wall in front of them when `wall`, facing away when `wall_faces_away`.
    pub fn fill(&mut self, wall: bool, wall_faces_away: bool) -> Result<(), vk::Result> {
        if wall {
            let size = 2.0 * WALL_HALF_SIZE;
            self.wall = Some(voxel_slab(
                &mut self.renderer,
                Vec3::new(-WALL_HALF_SIZE, -WALL_HALF_SIZE, WALL_Z),
                size / EDGE as f32,
                [WALL_TYPE; 2],
                false,
                wall_faces_away,
            )?);
            self.wall_faces_away = wall_faces_away;
        }
        for cell in 0..GRID * GRID {
            self.remesh(cell, 0)?;
        }
        Ok(())
    }

    /// Replaces a cell's chunk with the one meshed on `frame`. When the pool is full the
    /// cell is left without one.
    pub fn remesh(&mut self, cell: u32, frame: u32) -> Result<(), vk::Result> {
        let index = cell as usize;
        if let Some(old) = self.chunks[index].take() {
            self.renderer.destroy_mesh(old);
        }
        self.colors[index] = None;

        let id = match chunk_types(cell, frame) {
            Some(types) => voxel_slab(
                &mut self.renderer,
                chunk_min(cell).extend(CHUNK_Z),
                VOXEL_SIZE,
                types,
                cell.is_multiple_of(3),
                false,
            )?,
            None => self.renderer.create_mesh(&[], &[])?,
        };
        self.chunks[index] = Some(id);
        self.colors[index] = chunk_colors(cell, frame);
        Ok(())
    }

    /// Meshes in view, and of those, how many the wall can't hide.
    pub fn expected(&self, view_proj: Mat4) -> (u32, u32) {
        let wall = self.wall_rect(view_proj);
        let (mut in_view, mut must_draw) = (0, 0);
        for id in live(&self.chunks, &self.wall, &self.ballast) {
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

    /// The wall's screen rectangle in NDC, when there's one in view facing the camera.
    pub fn wall_rect(&self, view_proj: Mat4) -> Option<(Vec2, Vec2)> {
        if self.wall_faces_away {
            return None;
        }
        let mesh = self.renderer.mesh(self.wall.as_ref()?);
        screen_rect(view_proj, mesh.bounds_min, mesh.bounds_max)
    }
}
