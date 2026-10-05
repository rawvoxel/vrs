//! What chunk meshes' voxels look like by type: their materials.

use crate::renderer::{
    Renderer,
    material::{GLOSSY_ROUGHNESS, GpuMaterial, Material},
};
use ash::prelude::VkResult;

impl Renderer {
    /// Materials indexed by voxel type, which chunk meshes are drawn with from the next
    /// `submit`. Type 0 is empty space, and types past the end show magenta.
    pub fn set_materials(&mut self, materials: &[Material]) {
        self.palette = materials.iter().map(GpuMaterial::from).collect();
        self.palette_version += 1;
        self.glossy_palette = materials
            .iter()
            .any(|material| material.roughness < GLOSSY_ROUGHNESS);
    }

    /// Matte, opaque materials of linear `colors`, indexed by voxel type, see `set_materials`.
    pub fn set_palette(&mut self, colors: &[[f32; 3]]) {
        let materials: Vec<Material> = colors.iter().map(|&color| Material::color(color)).collect();
        self.set_materials(&materials);
    }

    /// Copies the materials into `frame`'s buffer when they've changed since that frame last
    /// ran.
    ///
    /// # Safety
    /// The device must not be using `frame`'s palette buffer.
    pub(in crate::renderer) unsafe fn write_palette(&mut self, frame: usize) -> VkResult<()> {
        if self.frame_palette_versions[frame] == self.palette_version {
            return Ok(());
        }
        // An empty buffer can't be bound, a placeholder entry makes every type magenta.
        let magenta = [GpuMaterial::from(&Material::color([1.0, 0.0, 1.0]))];
        let materials = match self.palette.is_empty() {
            true => &magenta[..],
            false => &self.palette[..],
        };
        unsafe {
            self.frame_palette_buffers[frame].write(
                &self.adapter,
                &self.render_device,
                materials,
            )?;
        }
        self.frame_palette_versions[frame] = self.palette_version;
        Ok(())
    }
}
