use crate::{
    renderer::{
        LightingUniforms,
        lighting::{probes::ProbeUniforms, trace::TraceUniforms},
        shader::bindings,
    },
    vulkan::{adapter::Adapter, buffer::Buffer},
};
use ash::{prelude::VkResult, vk};
use glam::Mat4;
use std::slice;

/// Everything in a frame's uniform buffer. Matches `Uniforms` in `common/layout.slang`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FrameUniforms {
    /// From the world to the camera, and the camera to Vulkan's clip space.
    pub view: Mat4,
    pub proj: Mat4,
    /// From clip space back to a point's offset from the camera, to find a pixel's position
    /// from its depth. Without the view's translation, which far from the world's origin
    /// would cost the position most of its precision.
    pub camera_offset_from_clip: Mat4,
    pub camera_position: [f32; 3],
    pub padding: f32,
    /// The frame's width and height in pixels.
    pub frame_size: [u32; 2],
    /// 1 when the trace pass runs, and when it also finds the light from everywhere but the
    /// sun, for shading to upscale.
    pub uses_trace_pass: u32,
    pub traced_indirect: u32,
    pub lighting: LightingUniforms,
    pub probe_volume: ProbeUniforms,
    pub trace: TraceUniforms,
}

// Each part fills whole 16 byte rows, as the shaders' std140 layout lays them out.
const _: () = {
    assert!(size_of::<LightingUniforms>().is_multiple_of(16));
    assert!(size_of::<ProbeUniforms>().is_multiple_of(16));
    assert!(size_of::<TraceUniforms>().is_multiple_of(16));
    assert!(size_of::<FrameUniforms>().is_multiple_of(16));
};

/// A mapped `HOST_VISIBLE` buffer the size of `FrameUniforms`.
pub fn create_uniform_buffer(adapter: &Adapter, device: &ash::Device) -> VkResult<Buffer> {
    let info = vk::BufferCreateInfo::default()
        .size(size_of::<FrameUniforms>() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let mut buffer = Buffer::new(adapter, device, info, flags, 0)?;
    if let Err(err) = buffer.mem_map(device) {
        unsafe { buffer.destroy(device) };
        return Err(err);
    }
    Ok(buffer)
}

/// Points each frame's descriptor set at that frame's uniform buffer.
pub fn write_uniform_descriptor_sets(
    device: &ash::Device,
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

    unsafe { device.update_descriptor_sets(&writes, &[]) }
}
