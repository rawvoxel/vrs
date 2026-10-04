//! The windowed examples' titles: frames a second and how long frames take on the GPU,
//! averaged over the last second.

use std::time::{Duration, Instant};
use vrs::renderer::Renderer;
use winit::window::Window;

/// How often the title changes.
const INTERVAL: Duration = Duration::from_secs(1);
/// The GPU stages of a frame that update the light probes, and that trace shadows, see
/// `Renderer::debug_gpu_timings`. The rest draw.
const PROBE_STAGES: [&str; 3] = ["probe classification", "probe rays", "probe updates"];
const SHADOW_STAGES: [&str; 2] = ["light culling", "shadows and reflections"];

/// A window's title, `vrs - <name>: <details>, <fps> fps, GPU <ms> ms: ...`.
pub struct FrameTitle {
    name: &'static str,
    since: Instant,
    frames: u32,
    /// GPU milliseconds since the title last changed: in total, the probes' and the shadows',
    /// and how many frames they're of.
    gpu: [f64; 3],
    timed: u32,
}

impl FrameTitle {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            since: Instant::now(),
            frames: 0,
            gpu: [0.0; 3],
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
            let sum = |stages: &[&str]| stages.iter().map(|&stage| timings.stage(stage)).sum();
            let times = [timings.total(), sum(&PROBE_STAGES), sum(&SHADOW_STAGES)];
            for (total, time) in self.gpu.iter_mut().zip(times) {
                *total += time;
            }
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
            let [total, probes, shadows] = self.gpu.map(|time| time / self.timed as f64);
            // Only the stages the frames ran.
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
        self.gpu = [0.0; 3];
        self.timed = 0;
    }
}
