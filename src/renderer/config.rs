use crate::mesh::MeshCapacity;

/// How a `Renderer` is built, see `Renderer::with_config`.
///
/// These are starting values, which can change at runtime. The mesh capacity by growing.
#[derive(Clone, Copy, Debug)]
pub struct RendererConfig {
    /// The shared mesh buffers' size to start with.
    pub mesh_capacity: MeshCapacity,
    /// Whether the mesh pool grows when it's out of room, see `Renderer::grow_mesh_pool`.
    pub grow_mesh_pool: bool,
    /// Whether presenting waits for the display to refresh, see `Renderer::set_vsync`.
    pub vsync: bool,
    /// Whether meshes hidden behind others are culled, see `Renderer::occlusion_culling`.
    pub occlusion_culling: bool,
    /// Whether triangles facing away are skipped, see `Renderer::backface_culling`.
    pub backface_culling: bool,
}

impl Default for RendererConfig {
    /// The default mesh capacity, which grows, without vsync, with occlusion and backface culling.
    fn default() -> Self {
        Self {
            mesh_capacity: MeshCapacity::default(),
            grow_mesh_pool: true,
            vsync: false,
            occlusion_culling: true,
            backface_culling: true,
        }
    }
}
