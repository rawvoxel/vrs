use ash::{prelude::VkResult, vk};

/// GPU timestamps at any number of points in a command buffer, read back as ticks or as the
/// milliseconds between two of them.
pub struct TimestampPool {
    pool: vk::QueryPool,
    count: u32,
    /// Nanoseconds per tick.
    period: f32,
    /// The bits of each timestamp that are meaningful.
    mask: u64,
}

impl TimestampPool {
    /// `count` timestamps (not none), read with `period` nanoseconds a tick and `valid_bits` bits
    /// meaningful (from the adapter's limits and the queue family's properties). A queue without
    /// timestamps (`valid_bits == 0`) is refused with
    /// [`vk::Result::ERROR_FEATURE_NOT_PRESENT`].
    pub fn new(device: &ash::Device, count: u32, period: f32, valid_bits: u32) -> VkResult<Self> {
        debug_assert!(count > 0, "a timestamp pool holds some timestamps");
        if valid_bits == 0 {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        Ok(Self {
            pool: unsafe { device.create_query_pool(&info, None)? },
            count,
            period,
            mask: valid_mask(valid_bits),
        })
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Records clearing every timestamp, so the ones written after it are this submission's.
    ///
    /// # Safety
    /// In a command buffer that is recording, outside any rendering.
    pub unsafe fn cmd_reset(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer) {
        unsafe { device.cmd_reset_query_pool(cmd_buffer, self.pool, 0, self.count) };
    }

    /// Records timestamp `index` (one of the pool's), written once everything recorded before it
    /// is done.
    ///
    /// # Safety
    /// After [`TimestampPool::cmd_reset`] in the same command buffer, and at most once per reset.
    pub unsafe fn cmd_write(
        &self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        index: u32,
    ) {
        debug_assert!(
            index < self.count,
            "timestamp {index} is past the pool's end"
        );
        unsafe {
            device.cmd_write_timestamp2(
                cmd_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                index,
            );
        }
    }

    /// Each timestamp's ticks, or `None` for one that wasn't written. Call once the device is
    /// done with the command buffer that wrote them.
    pub fn read_ticks(&self, device: &ash::Device) -> VkResult<Vec<Option<u64>>> {
        self.read_first_ticks(device, self.count)
    }

    /// The first `count` timestamps' ticks (at most all of them), as [`TimestampPool::read_ticks`]:
    /// those a submission wrote, when it wrote them in order from the first.
    pub fn read_first_ticks(&self, device: &ash::Device, count: u32) -> VkResult<Vec<Option<u64>>> {
        let count = count.min(self.count);
        if count == 0 {
            return Ok(Vec::new());
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
        match read {
            Ok(()) | Err(vk::Result::NOT_READY) => Ok(results
                .iter()
                .map(|&[ticks, available]| (available != 0).then_some(ticks & self.mask))
                .collect()),
            Err(error) => Err(error),
        }
    }

    /// Milliseconds from tick `start` to tick `end`, across a wrap of the valid bits.
    pub fn ms_between(&self, start: u64, end: u64) -> f64 {
        (end.wrapping_sub(start) & self.mask) as f64 * self.period as f64 / 1e6
    }

    /// # Safety
    /// Once the device no longer uses it.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_query_pool(self.pool, None) };
    }
}

/// The bits of a timestamp that are meaningful when `valid_bits` of them are (1 to 64).
fn valid_mask(valid_bits: u32) -> u64 {
    if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_keeps_exactly_the_valid_bits() {
        assert_eq!(valid_mask(1), 1);
        assert_eq!(valid_mask(36), (1 << 36) - 1);
        assert_eq!(valid_mask(63), u64::MAX >> 1);
        assert_eq!(valid_mask(64), u64::MAX);
    }

    /// The milliseconds between two timestamps count across the valid bits' wrap.
    #[test]
    fn the_time_between_two_timestamps_counts_across_their_wrap() {
        let pool = TimestampPool {
            pool: vk::QueryPool::null(),
            count: 2,
            period: 1.0,
            mask: valid_mask(36),
        };
        assert_eq!(pool.ms_between(1_000_000, 3_000_000), 2.0);
        // 1 ms before the 36 bits wrap to 1 ms after.
        assert_eq!(pool.ms_between((1 << 36) - 1_000_000, 1_000_000), 2.0);
    }
}
