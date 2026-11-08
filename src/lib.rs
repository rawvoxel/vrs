#![doc = include_str!("../README.md")]

mod app;
pub mod vulkan;

pub use app::App;
pub use ash;

// It can be shared with other threads, without any `unsafe impl`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<App>();
};
