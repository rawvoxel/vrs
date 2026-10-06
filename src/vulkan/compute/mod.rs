//! Headless GPU compute, beside the renderer rather than part of it: a device with the features
//! GPU-driven compute needs, buffers that shaders reach by address, compute pipelines built from
//! SPIR-V words with a pipeline cache, and GPU timestamps for any number of passes.
//!
//! - [`ComputeContext`] makes a device on any [`Adapter`](crate::vulkan::adapter::Adapter) with
//!   Vulkan 1.3, buffer device addresses, 64-bit integers, timeline semaphores and subgroup size
//!   control, and one compute queue to submit to.
//! - [`AddressedBuffer`] is device memory a shader reaches through a `u64` address, and
//!   [`HostBuffer`] is memory the CPU writes to or reads back from.
//! - [`ComputePipelines`] builds several entry points of one SPIR-V module at once, through a
//!   [`PipelineCache`] that can be saved to disk and loaded again.
//! - [`TimestampPool`] times passes on the GPU, in milliseconds.
//!
//! Nothing here touches the renderer: it is a separate device, made and dropped on its own.

mod buffer;
mod context;
mod pipeline;
mod timestamps;

pub use buffer::{AddressedBuffer, HostAccess, HostBuffer};
pub use context::{ComputeContext, SubgroupProperties};
pub use pipeline::{ComputePipelines, PipelineCache, pipeline_cache_key};
pub use timestamps::TimestampPool;
