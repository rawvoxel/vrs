//! GPU compute: buffers that shaders reach by address, compute pipelines built from SPIR-V words
//! with a pipeline cache, and GPU timestamps for any number of passes.
//!
//! - [`AddressedBuffer`] is device memory a shader reaches through a `u64` address, and
//!   [`HostBuffer`] is memory the CPU writes to or reads back from.
//! - [`ComputePipelines`] builds several entry points of one SPIR-V module at once, through a
//!   [`PipelineCache`] that can be saved to disk and loaded again.
//! - [`TimestampPool`] times passes on the GPU, in milliseconds.

mod buffer;
mod pipeline;
mod timestamps;

pub use buffer::{AddressedBuffer, HostAccess, HostBuffer};
pub use pipeline::{
    ComputePipelines, PipelineCache, PipelineRepresentation, PipelineStatistic, pipeline_cache_key,
};
pub use timestamps::TimestampPool;
