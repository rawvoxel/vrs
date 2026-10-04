//! Projections for the examples' cameras.

use glam::Mat4;

/// A right-handed perspective projection into Vulkan's clip space, `fov_y_degrees` high.
/// glam's projection is OpenGL's, with y up, while Vulkan's clip space has y down, so y is
/// flipped: without that, the image is mirrored and the wrong faces are culled.
pub fn perspective(fov_y_degrees: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let mut proj = Mat4::perspective_rh(fov_y_degrees.to_radians(), aspect, near, far);
    proj.y_axis.y = -proj.y_axis.y;
    proj
}
