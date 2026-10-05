#![doc = include_str!("../README.md")]

mod app;
pub mod mesh;
pub mod renderer;
pub mod voxel;
pub mod vulkan;

pub use app::App;

// Both can be shared with other threads, without any `unsafe impl`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<App>();
    assert_send_sync::<renderer::Renderer>();
};
