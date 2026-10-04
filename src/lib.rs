#![doc = include_str!("../README.md")]

pub mod adapter;
pub mod alloc;
mod app;
pub(crate) mod bindings;
pub mod buffer;
pub(crate) mod bytes;
pub mod commands;
pub(crate) mod cull;
pub mod debug;
pub mod descriptors;
pub(crate) mod gi;
pub(crate) mod hiz;
pub mod image;
pub(crate) mod lights;
pub mod mesh;
pub(crate) mod pipeline;
pub mod queue;
pub mod renderer;
pub(crate) mod shadows;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub(crate) mod timings;
pub(crate) mod uniforms;
pub mod vertex;
pub mod voxel;

pub use app::App;

// Both can be shared with other threads, without any `unsafe impl`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<App>();
    assert_send_sync::<renderer::Renderer>();
};
