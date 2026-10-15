use ash::{prelude::VkResult, vk};

/// GPU timestamps at any number of points in a command buffer, read back as ticks or as the
/// milliseconds between two of them.
pub struct TimestampPool {
    pool: vk::QueryPool,
    count: u32,
    /// Nanoseconds per tick.
    period: f32,
    /// Bits of each timestamp that are meaningful.
    valid_bits: u32,
}

impl TimestampPool {
    /// Records nothing, for a queue without timestamps.
    pub fn null() -> Self {
        Self {
            pool: vk::QueryPool::null(),
            count: 0,
            period: 0.0,
            valid_bits: 0,
        }
    }

    /// `count` timestamps, read with `period` nanoseconds a tick and `valid_bits` bits meaningful
    /// (from the queue family's properties). A queue without timestamps (`valid_bits == 0`) gets
    /// a pool that records nothing.
    pub fn new(device: &ash::Device, count: u32, period: f32, valid_bits: u32) -> VkResult<Self> {
        if valid_bits == 0 || count == 0 {
            return Ok(Self::null());
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        Ok(Self {
            pool: unsafe { device.create_query_pool(&info, None)? },
            count,
            period,
            valid_bits,
        })
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Whether timestamps are recorded at all.
    #[inline]
    pub fn is_recording(&self) -> bool {
        self.pool != vk::QueryPool::null()
    }

    /// Records clearing every timestamp, so the ones written after it are this submission's.
    ///
    /// # Safety
    /// In a command buffer that is recording, outside any rendering.
    pub unsafe fn cmd_reset(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        if self.is_recording() {
            unsafe { device.cmd_reset_query_pool(cmd_buffer, self.pool, 0, self.count) };
        }
    }

    /// Records timestamp `index`, written once everything recorded before it is done.
    ///
    /// # Safety
    /// After [`TimestampPool::cmd_reset`] in the same command buffer, and at most once per reset.
    pub unsafe fn cmd_write(
        &self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        index: u32,
    ) {
        if self.is_recording() && index < self.count {
            unsafe {
                device.cmd_write_timestamp2(
                    cmd_buffer,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    self.pool,
                    index,
                );
            }
        }
    }

    /// Each timestamp's ticks, or `None` for one that wasn't written. Call once the device is
    /// done with the command buffer that wrote them.
    pub fn read_ticks(&self, device: &ash::Device) -> Vec<Option<u64>> {
        self.read_first_ticks(device, self.count)
    }

    /// The first `count` timestamps' ticks (at most all of them), as [`TimestampPool::read_ticks`]:
    /// those a submission wrote, when it wrote them in order from the first.
    pub fn read_first_ticks(&self, device: &ash::Device, count: u32) -> Vec<Option<u64>> {
        let count = count.min(self.count);
        if !self.is_recording() || count == 0 {
            return Vec::new();
        }
        // Each timestamp, then whether it was written.
        let mut results = vec![[0u64; 2]; count as usize];
        // Timestamps that weren't written make it not ready, but every result is still filled in
        // with whether it was.
        let read = unsafe {
            device.get_query_pool_results(
                self.pool,
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        if !matches!(read, Ok(()) | Err(vk::Result::NOT_READY)) {
            return vec![None; count as usize];
        }
        let mask = if self.valid_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << self.valid_bits) - 1
        };
        results
            .iter()
            .map(|&[ticks, available]| (available != 0).then_some(ticks & mask))
            .collect()
    }

    /// Milliseconds from tick `start` to tick `end`, across a wrap of the valid bits.
    pub fn ms_between(&self, start: u64, end: u64) -> f64 {
        let mask = if self.valid_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << self.valid_bits) - 1
        };
        (end.wrapping_sub(start) & mask) as f64 * self.period as f64 / 1e6
    }

    /// # Safety
    /// Once the device no longer uses it.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_query_pool(self.pool, None) };
    }
}
