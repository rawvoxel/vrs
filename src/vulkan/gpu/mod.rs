//! GPU-driven rendering, beside the renderer rather than part of it: a device for compute and
//! mesh-shader rasterization on one queue, mesh-shader pipelines, and images to render into.
//!
//! - [`GpuContext`] makes a device with everything
//!   [`ComputeContext`](crate::vulkan::compute::ComputeContext) has, plus task and mesh shaders,
//!   64-bit buffer atomics, fragment stores and atomics and dynamic rendering. The compute
//!   module's buffers, pipelines and timestamps work on it too.
//! - [`MeshPipeline`] builds a task + mesh + fragment pipeline from SPIR-V modules, with push
//!   constants every stage sees and no descriptor sets.
//! - [`RenderImage`] is a 2D image with its view, to render into or copy from.
//!
//! Nothing here touches the renderer: it is a separate device, made and dropped on its own.

mod context;
mod mesh_pipeline;
mod target;

pub use context::GpuContext;
pub use mesh_pipeline::{DepthState, MeshPipeline, MeshPipelineDesc, ShaderStage};
pub use target::RenderImage;
