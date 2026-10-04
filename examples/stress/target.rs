//! Where the stress test draws: a window, or images of the renderer's own when headless.

use winit::{dpi::PhysicalSize, window::Window};

pub const START_SIZE: PhysicalSize<u32> = PhysicalSize::new(800, 600);

pub enum Target {
    Window(Window),
    /// Resizing and minimizing take effect straight away.
    Headless {
        size: PhysicalSize<u32>,
        minimized: bool,
    },
}

impl Target {
    pub fn headless() -> Self {
        Target::Headless {
            size: START_SIZE,
            minimized: false,
        }
    }

    /// Zero while minimized.
    pub fn size(&self) -> PhysicalSize<u32> {
        match self {
            Target::Window(window) => window.inner_size(),
            Target::Headless {
                minimized: true, ..
            } => PhysicalSize::new(0, 0),
            Target::Headless { size, .. } => *size,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        match self {
            Target::Window(window) => drop(window.request_inner_size(new_size)),
            Target::Headless { size, .. } => *size = new_size,
        }
    }

    pub fn set_minimized(&mut self, minimize: bool) {
        match self {
            Target::Window(window) => window.set_minimized(minimize),
            Target::Headless { minimized, .. } => *minimized = minimize,
        }
    }
}
