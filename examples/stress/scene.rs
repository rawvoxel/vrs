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
pub const WALL_COLOR: [f32; 3] = [0.9, 0.8, 0.1];
/// Room for half again the grid's geometry, so remeshing has to reuse freed space.
pub const CAPACITY: MeshCapacity = MeshCapacity {
    vertices: GRID * GRID * 4 * 3 / 2 + 4,
    indices: GRID * GRID * 6 * 3 / 2 + 6,
    meshes: GRID * GRID * 3 / 2,
};

/// Faces the camera unless `facing_away`, which backface culling skips.
pub fn quad(
    renderer: &mut Renderer,
    min: Vec2,
    size: f32,
    z: f32,
    col: [f32; 3],
    facing_away: bool,
) -> Result<MeshId, vk::Result> {
    let corner = |x: f32, y: f32| Vertex {
        pos: [min.x + x * size, min.y + y * size, z],
        col,
    };
    // Front faces wind counter-clockwise on screen.
    let indices = match facing_away {
        false => [0, 2, 1, 1, 2, 3],
        true => [0, 1, 2, 2, 1, 3],
    };
    renderer.create_mesh(
        &indices,
        &[
            corner(0.0, 0.0),
            corner(1.0, 0.0),
            corner(0.0, 1.0),
            corner(1.0, 1.0),
        ],
    )
}

/// The corner of a cell's chunk nearest the origin.
pub fn chunk_min(cell: u32) -> Vec2 {
    Vec2::new((cell % GRID) as f32, (cell / GRID) as f32) / GRID as f32 * 2.0 - 1.0
}

/// A chunk's color when it's meshed on `frame`, `None` when it's empty, like air in a voxel world.
pub fn chunk_color(cell: u32, frame: u32) -> Option<[f32; 3]> {
    if (cell + frame).is_multiple_of(13) {
        return None;
    }
    let shade = ((frame / 7 + cell) % 10) as f32 / 10.0;
    Some([shade, 1.0 - shade, 0.5])
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
    /// Each chunk's color, `None` when it's empty or wasn't meshed.
    pub colors: Vec<Option<[f32; 3]>>,
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
    pub fn new(renderer: Renderer, app: App) -> Self {
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

    /// Where the last mesh's vertices end in the pool.
    pub fn vertices_end(&self) -> u32 {
        live(&self.chunks, &self.wall, &self.ballast)
            .into_iter()
            .map(|id| self.renderer.mesh(id).vertices.end)
            .max()
            .unwrap_or(0)
    }

    /// A mesh of `vertices` far outside every camera's view, frustum culled every frame.
    pub fn add_ballast(&mut self, vertices: u32) -> Result<(), vk::Result> {
        let far = Vertex {
            pos: [1000.0, 1000.0, CHUNK_Z],
            col: [1.0; 3],
        };
        let vertices = vec![far; vertices as usize];
        self.ballast = Some(self.renderer.create_mesh(&[0, 1, 2], &vertices)?);
        Ok(())
    }

    /// Every chunk, and the wall in front of them when `wall`, facing away when `wall_faces_away`.
    pub fn fill(&mut self, wall: bool, wall_faces_away: bool) -> Result<(), vk::Result> {
        if wall {
            let min = Vec2::splat(-WALL_HALF_SIZE);
            let size = 2.0 * WALL_HALF_SIZE;
            self.wall = Some(quad(
                &mut self.renderer,
                min,
                size,
                WALL_Z,
                WALL_COLOR,
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

        let color = chunk_color(cell, frame);
        let id = match color {
            Some(color) => quad(
                &mut self.renderer,
                chunk_min(cell),
                CHUNK_SIZE,
                CHUNK_Z,
                color,
                false,
            )?,
            None => self.renderer.create_mesh(&[], &[])?,
        };
        self.chunks[index] = Some(id);
        self.colors[index] = color;
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
