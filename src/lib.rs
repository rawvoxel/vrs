#![doc = include_str!("../README.md")]

pub mod adapter;
pub mod alloc;
mod app;
pub mod buffer;
pub(crate) mod bytes;
pub mod commands;
pub(crate) mod cull;
pub mod debug;
pub mod descriptors;
pub mod gi;
pub(crate) mod hiz;
pub mod image;
pub mod lights;
pub mod mesh;
pub(crate) mod pipeline;
pub mod queue;
pub mod renderer;
pub mod shadows;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod uniforms;
pub mod vertex;
pub mod voxel;

pub use app::App;

// Both are shared with other threads, such as Bevy resources, without any `unsafe impl`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<App>();
    assert_send_sync::<renderer::Renderer>();
};
