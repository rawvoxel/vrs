//! The colors chunk meshes take from their voxels' types.

use super::Renderer;
use ash::prelude::VkResult;

impl Renderer {
    /// Linear colors indexed by voxel type, which chunk meshes are colored with from the
    /// next `submit`. Types past the end show magenta.
    pub fn set_palette(&mut self, colors: &[[f32; 3]]) {
        self.palette = colors.iter().map(|&[r, g, b]| [r, g, b, 1.0]).collect();
        self.palette_version += 1;
    }

    /// Copies the palette into `frame`'s buffer when it's changed since that frame last ran.
    ///
    /// # Safety
    /// The device must not be using `frame`'s palette buffer.
    pub(super) unsafe fn write_palette(&mut self, frame: usize) -> VkResult<()> {
        if self.frame_palette_versions[frame] == self.palette_version {
            return Ok(());
        }
        // An empty buffer can't be bound, a placeholder entry makes every type magenta.
        let magenta = [[1.0, 0.0, 1.0, 1.0]];
        let colors = match self.palette.is_empty() {
            true => &magenta[..],
            false => &self.palette[..],
        };
        unsafe {
            self.frame_palette_buffers[frame].write(&self.adapter, &self.render_device, colors)?;
        }
        self.frame_palette_versions[frame] = self.palette_version;
        Ok(())
    }
}
