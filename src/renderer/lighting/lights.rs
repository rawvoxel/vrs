//! Each frame's `Renderer::lights`, as the shaders read them.

use crate::{
    lighting::lights::{GpuLight, MAX_LIGHTS},
    renderer::Renderer,
};
use ash::prelude::VkResult;

impl Renderer {
    /// How many of `lights` light the frame.
    pub(in crate::renderer) fn light_count(&self) -> u32 {
        self.lights.len().min(MAX_LIGHTS) as u32
    }

    /// Writes the frame's lights.
    ///
    /// # Safety
    /// The device must not be using `frame`'s light buffer.
    pub(in crate::renderer) unsafe fn write_lights(&mut self, frame: usize) -> VkResult<()> {
        self.gpu_lights.clear();
        self.gpu_lights.extend(
            self.lights[..self.light_count() as usize]
                .iter()
                .map(GpuLight::from),
        );
        // An empty buffer can't be bound, a placeholder entry stands in.
        if self.gpu_lights.is_empty() {
            self.gpu_lights.push(GpuLight::NONE);
        }
        unsafe {
            self.frame_light_buffers[frame].write(
                &self.adapter,
                &self.render_device,
                &self.gpu_lights,
            )
        }
    }
}
