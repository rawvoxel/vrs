mod compute;
mod graphics;

pub use compute::create_compute_pipeline;
pub use graphics::{create_graphics_pipeline, create_pipeline_layout};

use ash::{prelude::VkResult, util, vk};
use std::io::Cursor;

#[inline]
pub fn create_shader_module(
    logical_device: &ash::Device,
    spv_bytes: &[u8],
) -> VkResult<vk::ShaderModule> {
    let shader_code = util::read_spv(&mut Cursor::new(spv_bytes))
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    unsafe { logical_device.create_shader_module(&shader_info, None) }
}
