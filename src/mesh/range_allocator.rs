use std::ops::Range;

/// Hands out first fit ranges of `0..capacity`, merging neighbouring ranges as they're freed.
#[derive(Debug)]
pub(crate) struct RangeAllocator {
    capacity: u32,
    /// Sorted, with no two ranges touching.
    free: Vec<Range<u32>>,
}

impl RangeAllocator {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            // One free range covering everything.
            #[allow(clippy::single_range_in_vec_init)]
            free: vec![0..capacity],
        }
    }

    /// Total free space, which may be split across several ranges.
    #[cfg(test)]
    pub fn free_len(&self) -> u32 {
        self.free.iter().map(|range| range.len() as u32).sum()
    }

    /// An empty range never fails and takes no space.
    pub fn alloc(&mut self, len: u32) -> Option<Range<u32>> {
        if len == 0 {
            return Some(0..0);
        }

        let index = self
            .free
            .iter()
            .position(|range| range.len() as u32 >= len)?;
        let range = &mut self.free[index];
        let allocated = range.start..range.start + len;
        range.start += len;
        if range.start == range.end {
            self.free.remove(index);
        }
        Some(allocated)
    }

    pub fn free(&mut self, range: Range<u32>) {
        if range.is_empty() {
            return;
        }
        debug_assert!(range.end <= self.capacity, "Freed range is out of bounds.");

        let index = self.free.partition_point(|free| free.start < range.start);
        debug_assert!(
            self.free
                .get(index)
                .is_none_or(|next| range.end <= next.start)
                && (index == 0 || self.free[index - 1].end <= range.start),
            "Freed range overlaps free space."
        );

        let joins_prev = index > 0 && self.free[index - 1].end == range.start;
        let joins_next = self
            .free
            .get(index)
            .is_some_and(|next| next.start == range.end);
        match (joins_prev, joins_next) {
            (true, true) => {
                self.free[index - 1].end = self.free[index].end;
                self.free.remove(index);
            }
            (true, false) => self.free[index - 1].end = range.end,
            (false, true) => self.free[index].start = range.start,
            (false, false) => self.free.insert(index, range),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RangeAllocator;

    #[test]
    fn allocates_in_order_until_full() {
        let mut ranges = RangeAllocator::new(10);
        assert_eq!(ranges.alloc(4), Some(0..4));
        assert_eq!(ranges.alloc(6), Some(4..10));
        assert_eq!(ranges.alloc(1), None);
        assert_eq!(ranges.free_len(), 0);
    }

    #[test]
    fn empty_ranges_take_no_space() {
        let mut ranges = RangeAllocator::new(4);
        assert_eq!(ranges.alloc(0), Some(0..0));
        ranges.free(0..0);
        assert_eq!(ranges.alloc(4), Some(0..4));
        assert_eq!(ranges.alloc(0), Some(0..0));
    }

    #[test]
    fn reuses_the_first_hole_that_fits() {
        let mut ranges = RangeAllocator::new(10);
        let a = ranges.alloc(2).unwrap();
        let b = ranges.alloc(3).unwrap();
        let _c = ranges.alloc(2).unwrap();
        ranges.free(a);
        ranges.free(b);
        // `a` and `b` merged into one hole at the front.
        assert_eq!(ranges.alloc(5), Some(0..5));
    }

    #[test]
    fn merges_with_both_neighbours() {
        let mut ranges = RangeAllocator::new(9);
        let a = ranges.alloc(3).unwrap();
        let b = ranges.alloc(3).unwrap();
        let c = ranges.alloc(3).unwrap();
        ranges.free(a);
        ranges.free(c);
        ranges.free(b);
        assert_eq!(ranges.free, vec![0..9]);
    }

    #[test]
    fn fails_when_space_is_fragmented() {
        let mut ranges = RangeAllocator::new(6);
        let a = ranges.alloc(2).unwrap();
        let _b = ranges.alloc(2).unwrap();
        let c = ranges.alloc(2).unwrap();
        ranges.free(a);
        ranges.free(c);
        assert_eq!(ranges.free_len(), 4);
        assert_eq!(ranges.alloc(3), None);
    }
}
