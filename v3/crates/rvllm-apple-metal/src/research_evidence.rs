//! Encoded-dispatch evidence, not GPU completion or numerical qualification.
#![forbid(unsafe_code)]

#[cfg(any(target_os = "macos", target_os = "ios", test))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Fixed order is part of the receipt schema, independent of hash-map order.
pub const RESEARCH_KERNEL_NAMES: [&str; 5] = [
    "research_gemm_mma16x64",
    "research_qkv_mma16x64",
    "research_rounded_gate32",
    "research_gqa_kv8_d256",
    "research_gqa_kv8_d512",
];

#[cfg(any(target_os = "macos", target_os = "ios", test))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum ResearchKernel {
    ShortGemm,
    ShortQkv,
    RoundedGate,
    Gqa256,
    Gqa512,
}

/// Sample only at a quiescent owner boundary. These atomics do not synchronize
/// Metal resources or establish that a command buffer completed successfully.
#[cfg(any(target_os = "macos", target_os = "ios", test))]
#[derive(Debug, Default)]
pub(crate) struct ResearchDispatchCounters {
    counts: [AtomicU64; 5],
    overflowed: AtomicBool,
}

#[cfg(any(target_os = "macos", target_os = "ios", test))]
impl ResearchDispatchCounters {
    /// Called after a real encode/endEncoding, never from a routing predicate.
    pub(crate) fn record(&self, kernel: ResearchKernel) {
        if self.counts[kernel as usize]
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .is_err()
        {
            self.overflowed.store(true, Ordering::Relaxed);
        }
    }

    pub(crate) fn snapshot(&self) -> ResearchDispatchSnapshot {
        ResearchDispatchSnapshot {
            counts: std::array::from_fn(|i| self.counts[i].load(Ordering::Relaxed)),
            overflowed: self.overflowed.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResearchDispatchSnapshot {
    pub counts: [u64; 5],
    pub overflowed: bool,
}

impl ResearchDispatchSnapshot {
    pub fn checked_since(self, earlier: Self) -> Result<Self, &'static str> {
        if self.overflowed || earlier.overflowed {
            return Err("research dispatch counter overflow");
        }
        let mut counts = [0; 5];
        for (i, count) in counts.iter_mut().enumerate() {
            *count = self.counts[i]
                .checked_sub(earlier.counts[i])
                .ok_or("research dispatch counters reset within a request")?;
        }
        Ok(Self {
            counts,
            overflowed: false,
        })
    }

    /// Positive evidence requires actual encoded work of the selected family.
    /// This does not assert all eligible layers used the candidate, any tensor
    /// accuracy, or any speedup. A valid fallback is deliberately `Ok(false)`.
    pub fn selection_exercised(self, requested: &str) -> Result<bool, &'static str> {
        if self.overflowed {
            return Err("research dispatch counter overflow");
        }
        let allowed = match requested {
            "off" => [false; 5],
            "metal-short-mma16x64" => [true, true, false, false, false],
            "metal-rounded-gate32" => [false, false, true, false, false],
            "metal-gqa-kv8" => [false, false, false, true, true],
            _ => return Err("unknown research candidate in receipt"),
        };
        if self
            .counts
            .iter()
            .zip(allowed)
            .any(|(&n, ok)| n != 0 && !ok)
        {
            return Err("receipt contains a different candidate's dispatches");
        }
        Ok(requested == "off" || self.counts.iter().any(|&n| n != 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_without_dispatch_is_not_evidence() {
        let empty = ResearchDispatchSnapshot::default();
        assert_eq!(empty.selection_exercised("off"), Ok(true));
        assert_eq!(empty.selection_exercised("metal-gqa-kv8"), Ok(false));
        assert!(empty.selection_exercised("auto").is_err());
    }

    #[test]
    fn counters_distinguish_kernel_families_and_request_boundaries() {
        let counters = ResearchDispatchCounters::default();
        counters.record(ResearchKernel::ShortGemm);
        let before = counters.snapshot();
        counters.record(ResearchKernel::Gqa256);
        counters.record(ResearchKernel::Gqa512);
        let delta = counters.snapshot().checked_since(before).unwrap();
        assert_eq!(delta.counts, [0, 0, 0, 1, 1]);
        assert_eq!(delta.selection_exercised("metal-gqa-kv8"), Ok(true));
        assert!(delta.selection_exercised("off").is_err());
        assert!(delta.selection_exercised("metal-rounded-gate32").is_err());
        assert!(counters
            .snapshot()
            .selection_exercised("metal-gqa-kv8")
            .is_err());
    }

    #[test]
    fn counter_reset_and_overflow_never_become_zero_work() {
        let mut before = ResearchDispatchSnapshot::default();
        before.counts[0] = 1;
        assert!(ResearchDispatchSnapshot::default()
            .checked_since(before)
            .is_err());
        let counters = ResearchDispatchCounters::default();
        counters.counts[0].store(u64::MAX, Ordering::Relaxed);
        counters.record(ResearchKernel::ShortGemm);
        let after = counters.snapshot();
        assert_eq!(after.counts[0], u64::MAX);
        assert!(after
            .checked_since(ResearchDispatchSnapshot::default())
            .is_err());
    }

    #[test]
    fn all_stable_slots_match_expected_kernel_names() {
        let counters = ResearchDispatchCounters::default();
        for kernel in [
            ResearchKernel::ShortGemm,
            ResearchKernel::ShortQkv,
            ResearchKernel::RoundedGate,
            ResearchKernel::Gqa256,
            ResearchKernel::Gqa512,
        ] {
            counters.record(kernel);
        }
        assert_eq!(counters.snapshot().counts, [1, 1, 1, 1, 1]);
        assert_eq!(
            RESEARCH_KERNEL_NAMES,
            [
                "research_gemm_mma16x64",
                "research_qkv_mma16x64",
                "research_rounded_gate32",
                "research_gqa_kv8_d256",
                "research_gqa_kv8_d512",
            ]
        );
    }
}
