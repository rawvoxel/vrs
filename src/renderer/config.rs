use crate::mesh::MeshCapacity;

/// How a `Renderer` is built, see `Renderer::with_config`.
#[derive(Clone, Copy, Debug)]
pub struct RendererConfig {
    /// Sizes the shared mesh buffers, which don't grow.
    pub mesh_capacity: MeshCapacity,
    /// Whether presenting waits for the display to refresh, see `Renderer::set_vsync`.
    pub vsync: bool,
}

impl Default for RendererConfig {
    /// The default mesh capacity, without vsync.
    fn default() -> Self {
        Self {
            mesh_capacity: MeshCapacity::default(),
            vsync: false,
        }
    }
}
