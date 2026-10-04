use crate::mesh::MeshCapacity;

/// How a `Renderer` is built, see `Renderer::with_config`.
///
/// Apart from the mesh capacity these are starting values, which can change at runtime.
#[derive(Clone, Copy, Debug)]
pub struct RendererConfig {
    /// Sizes the shared mesh buffers, which don't grow.
    pub mesh_capacity: MeshCapacity,
    /// Whether presenting waits for the display to refresh, see `Renderer::set_vsync`.
    pub vsync: bool,
    /// Whether meshes hidden behind others are culled, see `Renderer::occlusion_culling`.
    pub occlusion_culling: bool,
}

impl Default for RendererConfig {
    /// The default mesh capacity, without vsync, with occlusion culling.
    fn default() -> Self {
        Self {
            mesh_capacity: MeshCapacity::default(),
            vsync: false,
            occlusion_culling: true,
        }
    }
}
