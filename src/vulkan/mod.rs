//! Vulkan building blocks: the GPU and its queues, memory, buffers and images, pipelines,
//! descriptors, commands and synchronization, and what frames are presented to.

pub mod adapter;
pub mod alloc;
pub mod buffer;
pub(crate) mod bytes;
pub mod commands;
pub mod compute;
pub mod descriptors;
pub mod image;
pub(crate) mod pipeline;
pub mod queue;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod validation;
