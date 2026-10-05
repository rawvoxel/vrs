//! GPU timestamps between the stages of each frame, see `Renderer::gpu_timings`.

use ash::{prelude::VkResult, vk};

/// A stage of a frame on the GPU, in the order they run. Frames skip the stages they don't
/// need, and anything between two stages counts toward the later one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GpuStage {
    /// Copying new meshes into the mesh pool.
    Uploads,
    /// Placing the light probes and choosing those near voxels, which share the frame's rays.
    ProbeClassification,
    /// Tracing the probes' rays.
    ProbeRays,
    /// Blending the rays into the probes.
    ProbeUpdates,
    /// Readying the frame's meshes for culling, then culling them against the view: those
    /// visible last frame with occlusion culling, else all of them.
    Culling,
    /// Drawing what culling found, only depth with the trace pass.
    FirstPass,
    /// Building the depth pyramid, culling the rest against it, and drawing what was newly
    /// visible.
    OcclusionPass,
    /// Readying the depth for the trace pass, and listing the lights reaching each tile.
    LightCulling,
    /// Tracing shadows and reflections, and finding the light from everywhere but the sun.
    TracePass,
    /// Drawing color over the depth, after the trace pass.
    ColorPass,
    /// Drawing the light probes, for `DebugView::Probes`.
    ProbeOverlay,
}

impl GpuStage {
    /// Every stage, in the order they run.
    pub const ALL: [GpuStage; 11] = [
        GpuStage::Uploads,
        GpuStage::ProbeClassification,
        GpuStage::ProbeRays,
        GpuStage::ProbeUpdates,
        GpuStage::Culling,
        GpuStage::FirstPass,
        GpuStage::OcclusionPass,
        GpuStage::LightCulling,
        GpuStage::TracePass,
        GpuStage::ColorPass,
        GpuStage::ProbeOverlay,
    ];
    /// Those updating the light probes.
    pub const PROBES: [GpuStage; 3] = [
        GpuStage::ProbeClassification,
        GpuStage::ProbeRays,
        GpuStage::ProbeUpdates,
    ];
    /// The trace pass's: listing the lights reaching each tile, then tracing.
    pub const TRACE_PASS: [GpuStage; 2] = [GpuStage::LightCulling, GpuStage::TracePass];

    /// For people.
    pub fn name(self) -> &'static str {
        match self {
            GpuStage::Uploads => "uploads",
            GpuStage::ProbeClassification => "probe classification",
            GpuStage::ProbeRays => "probe rays",
            GpuStage::ProbeUpdates => "probe updates",
            GpuStage::Culling => "culling",
            GpuStage::FirstPass => "first pass",
            GpuStage::OcclusionPass => "occlusion pass",
            GpuStage::LightCulling => "light culling",
            GpuStage::TracePass => "trace pass",
            GpuStage::ColorPass => "color pass",
            GpuStage::ProbeOverlay => "probe overlay",
        }
    }

    /// Its timestamp's query, after the frame's start.
    fn query(self) -> u32 {
        self as u32 + 1
    }
}

/// Timestamps a frame writes: its start, then the end of each stage.
const QUERIES: u32 = GpuStage::ALL.len() as u32 + 1;

/// How long each stage of a frame took on the GPU, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuTimings {
    /// By stage, `None` for those the frame skipped.
    stages: [Option<f64>; GpuStage::ALL.len()],
}

impl GpuTimings {
    /// The stages the frame ran, in order, with their times.
    pub fn stages(&self) -> impl Iterator<Item = (GpuStage, f64)> + '_ {
        GpuStage::ALL
            .into_iter()
            .zip(self.stages)
            .filter_map(|(stage, time)| Some((stage, time?)))
    }

    /// `stage`'s time, 0 when the frame skipped it.
    pub fn stage(&self, stage: GpuStage) -> f64 {
        self.stages[stage as usize].unwrap_or(0.0)
    }

    /// The time of all of `stages`.
    pub fn sum(&self, stages: &[GpuStage]) -> f64 {
        stages.iter().map(|&stage| self.stage(stage)).sum()
    }

    /// From the frame's first stage starting to its last ending.
    pub fn total(&self) -> f64 {
        self.stages.iter().flatten().sum()
    }
}

/// One frame in flight's timestamps.
pub(crate) struct FrameTimestamps {
    pool: vk::QueryPool,
    /// Whether a frame has written them since they were made.
    used: bool,
}

impl FrameTimestamps {
    /// Records nothing, for a queue without timestamps.
    pub fn null() -> Self {
        Self {
            pool: vk::QueryPool::null(),
            used: false,
        }
    }

    pub fn new(device: &ash::Device) -> VkResult<Self> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(QUERIES);
        Ok(Self {
            pool: unsafe { device.create_query_pool(&info, None)? },
            used: false,
        })
    }

    /// Records clearing last time's timestamps, then the start's.
    ///
    /// # Safety
    /// Outside any rendering.
    pub unsafe fn cmd_start(&mut self, device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        if self.pool == vk::QueryPool::null() {
            return;
        }
        self.used = true;
        unsafe {
            device.cmd_reset_query_pool(cmd_buffer, self.pool, 0, QUERIES);
            self.cmd_write(device, cmd_buffer, 0);
        }
    }

    /// Records the end of `stage`, once everything recorded before it is done.
    ///
    /// # Safety
    /// After `cmd_start` in the same command buffer.
    pub unsafe fn cmd_mark(
        &self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        stage: GpuStage,
    ) {
        unsafe { self.cmd_write(device, cmd_buffer, stage.query()) };
    }

    unsafe fn cmd_write(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer, query: u32) {
        if self.pool == vk::QueryPool::null() {
            return;
        }
        unsafe {
            device.cmd_write_timestamp2(
                cmd_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                query,
            );
        }
    }

    /// The last frame's timings, once the device is done with it, with `period` nanoseconds a
    /// tick and `valid_bits` bits of each timestamp meaningful.
    pub fn read(&self, device: &ash::Device, period: f32, valid_bits: u32) -> Option<GpuTimings> {
        if !self.used {
            return None;
        }
        // Each timestamp, then whether it was written.
        let mut results = [[0u64; 2]; QUERIES as usize];
        // Stages the frame skipped were never written, which makes it not ready, but every
        // result is filled in with whether it was.
        let read = unsafe {
            device.get_query_pool_results(
                self.pool,
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        if let Err(err) = read
            && err != vk::Result::NOT_READY
        {
            return None;
        }
        let mask = match valid_bits {
            64.. => u64::MAX,
            bits => (1u64 << bits) - 1,
        };
        let [start, ends @ ..] = results;
        let mut last = (start[1] != 0).then_some(start[0] & mask)?;
        let mut timings = GpuTimings::default();
        for (time, [end, written]) in timings.stages.iter_mut().zip(ends) {
            if written == 0 {
                continue;
            }
            let end = end & mask;
            let ticks = end.wrapping_sub(last) & mask;
            *time = Some(ticks as f64 * period as f64 / 1e6);
            last = end;
        }
        Some(timings)
    }

    /// # Safety
    /// The device must be done with the pool.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_query_pool(self.pool, None) };
    }
}

impl super::Renderer {
    /// How long each stage of the last frame the device finished took, `None` before one has
    /// or when its queue can't time them. Without stalling: it's the frame from as many frames
    /// ago as are in flight.
    pub fn gpu_timings(&self) -> Option<&GpuTimings> {
        self.gpu_timings.as_ref()
    }
}
