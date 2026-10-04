use crate::{
    adapter::Adapter, bindings, buffer::Buffer, gi::GiUniforms, renderer::LightingUniforms,
    shadows::ShadowUniforms,
};
use ash::{prelude::VkResult, vk};
use glam::{Mat4, Vec4};
use std::slice;

/// Everything in a frame's uniform buffer. Starts with `MvpMatrices`, all the dynamic
/// geometry's shader reads, and matches `Uniforms` in `common.slang`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FrameUniforms {
    pub matrices: MvpMatrices,
    pub lighting: LightingUniforms,
    pub gi: GiUniforms,
    pub shadow: ShadowUniforms,
}

#[repr(C)]
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

/// A mapped `HOST_VISIBLE` buffer the size of `FrameUniforms`.
pub fn create_uniform_buffer(adapter: &Adapter, render_device: &ash::Device) -> VkResult<Buffer> {
    let info = vk::BufferCreateInfo::default()
        .size(size_of::<FrameUniforms>() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let mut buffer = Buffer::new(adapter, render_device, info, flags, 0)?;
    if let Err(err) = buffer.mem_map(render_device) {
        unsafe { buffer.destroy(render_device) };
        return Err(err);
    }
    Ok(buffer)
}

/// Points each frame's descriptor set at that frame's uniform buffer.
pub fn write_uniform_descriptor_sets(
    render_device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    uniform_buffers: &[Buffer],
) {
    let buffer_infos: Vec<vk::DescriptorBufferInfo> = uniform_buffers
        .iter()
        .map(|buffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer.handle)
                .range(size_of::<FrameUniforms>() as vk::DeviceSize)
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = descriptor_sets
        .iter()
        .zip(buffer_infos.iter())
        .map(|(&set, buffer_info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(bindings::UNIFORMS)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(buffer_info))
        })
        .collect();

    unsafe { render_device.update_descriptor_sets(&writes, &[]) }
}
