//! The windowed examples' titles: frames a second and how long frames take on the GPU,
//! averaged over the last second.

use std::time::{Duration, Instant};
use vrs::renderer::{GpuStage, GpuTimings, Renderer};
use winit::window::Window;

/// How often the title changes.
const INTERVAL: Duration = Duration::from_secs(1);

/// GPU milliseconds, added up over frames.
#[derive(Clone, Copy, Default)]
struct GpuTime {
    total: f64,
    /// Updating the light probes.
    probes: f64,
    /// Tracing shadows and reflections.
    shadows: f64,
}

impl GpuTime {
    fn add(&mut self, timings: &GpuTimings) {
        self.total += timings.total();
        self.probes += timings.sum(&GpuStage::PROBES);
        self.shadows += timings.sum(&GpuStage::SHADOWS);
    }
}

/// A window's title, `vrs - <name>: <details>, <fps> fps, GPU <ms> ms: ...`.
pub struct FrameTitle {
    name: &'static str,
    since: Instant,
    frames: u32,
    /// Since the title last changed.
    gpu: GpuTime,
    /// Frames `gpu` is of.
    timed: u32,
}

impl FrameTitle {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            since: Instant::now(),
            frames: 0,
            gpu: GpuTime::default(),
            timed: 0,
        }
    }

    /// Counts a frame `renderer` just submitted, and every second sets `window`'s title,
    /// with what `details` says first, when it says anything.
    pub fn frame(
        &mut self,
        window: &Window,
        renderer: &mut Renderer,
        details: impl FnOnce(&mut Renderer) -> Option<String>,
    ) {
        self.frames += 1;
        // Without stalling, a frame from as many ago as are in flight.
        if let Some(timings) = renderer.debug_gpu_timings() {
            self.gpu.add(timings);
            self.timed += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < INTERVAL {
            return;
        }

        let mut parts: Vec<String> = details(renderer).into_iter().collect();
        parts.push(format!(
            "{:.0} fps",
            self.frames as f64 / elapsed.as_secs_f64()
        ));
        if self.timed > 0 {
            let frames = self.timed as f64;
            let (total, probes, shadows) = (
                self.gpu.total / frames,
                self.gpu.probes / frames,
                self.gpu.shadows / frames,
            );
            // Only what the frames did.
            let mut stages = Vec::new();
            for (name, time) in [("probes", probes), ("shadows", shadows)] {
                if time > 0.0 {
                    stages.push(format!("{name} {time:.2}"));
                }
            }
            stages.push(format!("drawing {:.2}", total - probes - shadows));
            parts.push(format!("GPU {total:.2} ms: {}", stages.join(", ")));
        }
        window.set_title(&format!("vrs - {}: {}", self.name, parts.join(", ")));

        self.since = Instant::now();
        self.frames = 0;
        self.gpu = GpuTime::default();
        self.timed = 0;
    }
}
