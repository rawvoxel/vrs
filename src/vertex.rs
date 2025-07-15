use ash::vk;
use std::mem::offset_of;

#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub col: [f32; 3],
}
pub type Index = u32;

impl Vertex {
    pub const VERTEX_TOPOLOGY: vk::PrimitiveTopology = vk::PrimitiveTopology::TRIANGLE_LIST;

    pub const VERTEX_BINDING_DESCRIPTIONS: [vk::VertexInputBindingDescription; 1] =
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Self>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }];

    pub const VERTEX_ATTRIBUTE_DESCRIPTIONS: [vk::VertexInputAttributeDescription; 2] = [
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Self, pos) as _,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Self, col) as _,
        },
    ];
}
