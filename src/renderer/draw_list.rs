//! What a frame draws, and from where.

use super::MeshId;
use glam::Mat4;

/// Everything drawn by one `Renderer::submit`.
#[derive(Clone, Copy, Default)]
pub struct DrawList<'a> {
    /// Meshes added with `Renderer::create_mesh`. References, so ids can stay
    /// wherever they're kept, such as a map of chunks.
    pub meshes: &'a [&'a MeshId],
    /// Meshes outside this camera's frustum, or hidden behind others, are culled on the device.
    pub view: Mat4,
    /// Into Vulkan's clip space, where y points down and depth runs from 0 to 1. An OpenGL style
    /// projection, such as glam's `perspective_rh`, needs its y flipped, otherwise the image
    /// is mirrored and backface culling skips the wrong faces.
    pub proj: Mat4,
}
