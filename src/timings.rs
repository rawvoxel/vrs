//! GPU timestamps between the stages of each frame, see `Renderer::debug_gpu_timings`.

use ash::{prelude::VkResult, vk};

/// Where a frame writes a timestamp, each at the end of its stage, in the order they run.
/// Stages a frame skips write none, and their time goes to the next stage written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mark {
    Start,
    Uploads,
    GiClassify,
    GiTrace,
    GiUpdate,
    Culling,
    FirstPass,
    OcclusionPass,
    LightCulling,
    Shadows,
    ColorPass,
    ProbeOverlay,
}

impl Mark {
    const ALL: [Mark; 12] = [
        Mark::Start,
        Mark::Uploads,
        Mark::GiClassify,
        Mark::GiTrace,
        Mark::GiUpdate,
        Mark::Culling,
        Mark::FirstPass,
        Mark::OcclusionPass,
        Mark::LightCulling,
        Mark::Shadows,
        Mark::ColorPass,
        Mark::ProbeOverlay,
    ];

    /// The stage ending at the mark.
    fn stage(self) -> &'static str {
        match self {
            Mark::Start => "start",
            Mark::Uploads => "uploads",
            Mark::GiClassify => "probe classification",
            Mark::GiTrace => "probe rays",
            Mark::GiUpdate => "probe updates",
            Mark::Culling => "culling",
            Mark::FirstPass => "first pass",
            Mark::OcclusionPass => "occlusion pass",
            Mark::LightCulling => "light culling",
            Mark::Shadows => "shadows and reflections",
            Mark::ColorPass => "color pass",
            Mark::ProbeOverlay => "probe overlay",
        }
    }
}

/// How long each stage of a frame took on the GPU, in milliseconds, in the order they ran.
/// Only the stages the frame ran are listed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuTimings {
    pub stages: Vec<(&'static str, f64)>,
}

impl GpuTimings {
    /// From the frame's first stage starting to its last ending.
    pub fn total(&self) -> f64 {
        self.stages.iter().map(|(_, ms)| ms).sum()
    }

    /// The stage named `name`'s time, 0 when the frame didn't run it.
    pub fn stage(&self, name: &str) -> f64 {
        self.stages
            .iter()
            .find(|(stage, _)| *stage == name)
            .map_or(0.0, |(_, ms)| *ms)
    }
}

/// One frame in flight's timestamps.
pub(crate) struct FrameTimestamps {
    pool: vk::QueryPool,
    /// Whether a frame has written them since they were made.
    used: bool,
}

impl FrameTimestamps {
    pub fn null() -> Self {
        Self {
            pool: vk::QueryPool::null(),
            used: false,
        }
    }

    pub fn new(logical_device: &ash::Device) -> VkResult<Self> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(Mark::ALL.len() as u32);
        Ok(Self {
            pool: unsafe { logical_device.create_query_pool(&info, None)? },
            used: false,
        })
    }

    /// Records clearing last time's timestamps, then the start's. A null pool records nothing,
    /// for a queue without timestamps.
    ///
    /// # Safety
    /// Outside any rendering.
    pub unsafe fn cmd_start(
        &mut self,
        logical_device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
    ) {
        if self.pool == vk::QueryPool::null() {
            return;
        }
        unsafe {
            logical_device.cmd_reset_query_pool(cmd_buffer, self.pool, 0, Mark::ALL.len() as u32);
        }
        self.used = true;
        self.mark(logical_device, cmd_buffer, Mark::Start);
    }

    /// Records the timestamp of `mark`, once everything recorded before it is done.
    pub fn mark(&self, logical_device: &ash::Device, cmd_buffer: vk::CommandBuffer, mark: Mark) {
        if self.pool == vk::QueryPool::null() {
            return;
        }
        let index = Mark::ALL.iter().position(|&m| m == mark).unwrap_or(0) as u32;
        unsafe {
            logical_device.cmd_write_timestamp2(
                cmd_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                index,
            );
        }
    }

    /// The last frame's timings, once the device is done with it, with `period` nanoseconds a
    /// tick and `valid_bits` bits of each timestamp meaningful.
    pub fn read(
        &self,
        logical_device: &ash::Device,
        period: f32,
        valid_bits: u32,
    ) -> Option<GpuTimings> {
        if !self.used {
            return None;
        }
        // Each timestamp, then whether it was written.
        let mut results = [[0u64; 2]; Mark::ALL.len()];
        // Stages the frame skipped were never written, which makes it not ready, but every
        // result is filled in with whether it was.
        let read = unsafe {
            logical_device.get_query_pool_results(
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
        let mut stages = Vec::new();
        let mut last: Option<u64> = None;
        for (mark, [time, available]) in Mark::ALL.iter().zip(results) {
            if available == 0 {
                continue;
            }
            let time = time & mask;
            if let Some(last) = last {
                let ticks = time.wrapping_sub(last) & mask;
                stages.push((mark.stage(), ticks as f64 * period as f64 / 1e6));
            }
            last = Some(time);
        }
        Some(GpuTimings { stages })
    }

    /// # Safety
    /// The device must be done with the pool.
    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        unsafe { logical_device.destroy_query_pool(self.pool, None) };
    }
}
