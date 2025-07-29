use mint::ColumnMatrix4;

pub struct TransformUniforms {
    pub model: ColumnMatrix4<u32>,
    pub view: ColumnMatrix4<u32>,
    pub proj: ColumnMatrix4<u32>,
}
