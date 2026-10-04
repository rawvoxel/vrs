//! The chunks and the wall the stress test draws, and which of them should be drawn.

use ash::vk;
use glam::{Mat4, Vec2, Vec3};
use vrs::{
    App,
    mesh::{MeshCapacity, MeshId},
    renderer::Renderer,
    vertex::Vertex,
};

pub const GRID: u32 = 16;
pub const CHUNK_SIZE: f32 = 2.0 / GRID as f32 * 0.9;
pub const CHUNK_Z: f32 = 0.5;
pub const WALL_Z: f32 = 0.2;
pub const WALL_HALF_SIZE: f32 = 0.5;
/// Room for half again the grid's geometry, so remeshing has to reuse freed space.
pub const CAPACITY: MeshCapacity = MeshCapacity {
    vertices: GRID * GRID * 4 * 3 / 2 + 4,
    indices: GRID * GRID * 6 * 3 / 2 + 6,
    meshes: GRID * GRID * 3 / 2,
};

pub fn quad(
    renderer: &mut Renderer,
    min: Vec2,
    size: f32,
    z: f32,
    col: [f32; 3],
) -> Result<MeshId, vk::Result> {
    let corner = |x: f32, y: f32| Vertex {
        pos: [min.x + x * size, min.y + y * size, z],
        col,
    };
    renderer.create_mesh(
        &[0, 1, 2, 2, 1, 3],
        &[
            corner(0.0, 0.0),
            corner(1.0, 0.0),
            corner(0.0, 1.0),
            corner(1.0, 1.0),
        ],
    )
}

/// Some chunks are empty, like air in a voxel world.
pub fn chunk(renderer: &mut Renderer, cell: u32, frame: u32) -> Result<MeshId, vk::Result> {
    if (cell + frame).is_multiple_of(13) {
        return renderer.create_mesh(&[], &[]);
    }
    let min = Vec2::new((cell % GRID) as f32, (cell / GRID) as f32) / GRID as f32 * 2.0 - 1.0;
    let shade = ((frame / 7 + cell) % 10) as f32 / 10.0;
    quad(
        renderer,
        min,
        CHUNK_SIZE,
        CHUNK_Z,
        [shade, 1.0 - shade, 0.5],
    )
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

pub struct Scene {
    pub chunks: Vec<Option<MeshId>>,
    pub wall: Option<MeshId>,
    pub renderer: Renderer,
    pub app: App,
}

/// Every mesh in the scene, borrowing only its meshes so the renderer stays free.
pub fn live<'a>(chunks: &'a [Option<MeshId>], wall: &'a Option<MeshId>) -> Vec<&'a MeshId> {
    chunks.iter().flatten().chain(wall.as_ref()).collect()
}

impl Scene {
    pub fn clear(&mut self) {
        for id in self.chunks.iter_mut().filter_map(Option::take) {
            self.renderer.destroy_mesh(id);
        }
        if let Some(wall) = self.wall.take() {
            self.renderer.destroy_mesh(wall);
        }
    }

    /// Meshes in view, and of those, how many the wall can't hide.
    pub fn expected(&self, view_proj: Mat4) -> (u32, u32) {
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
