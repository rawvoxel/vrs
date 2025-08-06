use glam::{Mat4, Vec4};

#[derive(Clone, Copy)]
pub struct MvpMatrices {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

impl Default for MvpMatrices {
    fn default() -> Self {
        Self::identity()
    }
}

impl MvpMatrices {
    pub fn identity() -> Self {
        Self {
            model: identity_matrix(),
            view: identity_matrix(),
            proj: identity_matrix(),
        }
    }
}

#[rustfmt::skip]
pub fn identity_matrix() -> Mat4 {
    Mat4 {
        x_axis:  Vec4::X,
        y_axis:  Vec4::Y,
        z_axis:  Vec4::Z,
        w_axis:  Vec4::W,
    }
}
