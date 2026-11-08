//! GPU-driven rendering: the device for compute and mesh-shader rasterization on one queue,
//! mesh-shader pipelines, and images to render into.
//!
//! - [`VulkanDevice`] is the device and its queue, made on an adapter by [`create_device`] for
//!   an [`App`](crate::App) that destroys it, or from another Vulkan user's create info (such as
//!   wgpu-hal's) by [`create_shared_device`].
//! - [`MeshPipeline`] builds a task + mesh + fragment pipeline from SPIR-V modules, with push
//!   constants every stage sees and no descriptor sets.
//! - [`RenderImage`] is a 2D image with its view, to render into or copy from.

mod device;
mod mesh_pipeline;
mod target;

pub use device::{
    VulkanDevice, create_device, create_shared_device, memory_budget, missing_features,
};
pub use mesh_pipeline::{DepthState, MeshPipeline, MeshPipelineDesc, ShaderStage};
pub use target::RenderImage;
