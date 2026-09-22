//! Encoded-dispatch evidence, not GPU completion or numerical qualification.
#![forbid(unsafe_code)]

#[cfg(any(target_os = "macos", target_os = "ios", test))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Append-only diagnostic slots; the first five retain their original indices.
/// Consumers must bind the registry and executable used by a receipt.
pub const RESEARCH_KERNEL_COUNT: usize = 10;
pub const RESEARCH_KERNEL_NAMES: [&str; RESEARCH_KERNEL_COUNT] = [
    "research_gemm_mma16x64",
    "research_qkv_mma16x64",
    "research_rounded_gate32",
    "research_gqa_kv8_d256",
    "research_gqa_kv8_d512",
    "research_gemm_mma32_prefetch",
    "research_qkv_mma32_prefetch",
    "research_attn_q4_d256",
    "research_attn_q4_d512",
    "research_rms_simd32",
];

#[cfg(any(target_os = "macos", target_os = "ios", test))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum ResearchKernel {
    ShortGemm,
    ShortQkv,
    RoundedGate,
    Gqa256,
    Gqa512,
    PrefetchGemm,
    PrefetchQkv,
    Temporal256,
    Temporal512,
    Rms32,
}

/// Sample only at a quiescent owner boundary. These atomics do not synchronize
/// Metal resources or establish that a command buffer completed successfully.
#[cfg(any(target_os = "macos", target_os = "ios", test))]
#[derive(Debug, Default)]
pub(crate) struct ResearchDispatchCounters {
    counts: [AtomicU64; RESEARCH_KERNEL_COUNT],
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
    pub counts: [u64; RESEARCH_KERNEL_COUNT],
    pub overflowed: bool,
}

impl ResearchDispatchSnapshot {
    pub fn checked_since(self, earlier: Self) -> Result<Self, &'static str> {
        if self.overflowed || earlier.overflowed {
            return Err("research dispatch counter overflow");
        }
        let mut counts = [0; RESEARCH_KERNEL_COUNT];
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
            "off" => 0..0,
            "metal-short-mma16x64" => 0..2,
            "metal-rounded-gate32" => 2..3,
            "metal-gqa-kv8" => 3..5,
            "metal-mma32-prefetch" => 5..7,
            "metal-attn-q4" => 7..9,
            "metal-rms-simd32" => 9..10,
            _ => return Err("unknown research candidate in receipt"),
        };
        if self
            .counts
            .iter()
            .enumerate()
            .any(|(i, &n)| n != 0 && !allowed.contains(&i))
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
        assert_eq!(&delta.counts[..5], &[0, 0, 0, 1, 1]);
        assert!(delta.counts[5..].iter().all(|&n| n == 0));
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
        assert_eq!(&counters.snapshot().counts[..5], &[1, 1, 1, 1, 1]);
        assert!(counters.snapshot().counts[5..].iter().all(|&n| n == 0));
        assert_eq!(
            &RESEARCH_KERNEL_NAMES[..5],
            &[
                "research_gemm_mma16x64",
                "research_qkv_mma16x64",
                "research_rounded_gate32",
                "research_gqa_kv8_d256",
                "research_gqa_kv8_d512",
            ]
        );
    }
    #[test]
    fn prefetchgemm_slots_are_selected_individually_and_never_foreign() {
        for kind in [ResearchKernel::PrefetchGemm, ResearchKernel::PrefetchQkv] {
            let counters = ResearchDispatchCounters::default();
            counters.record(kind);
            let got = counters.snapshot();
            assert_eq!(got.selection_exercised("metal-mma32-prefetch"), Ok(true));
            assert!(got.selection_exercised("off").is_err());
            assert!(got.selection_exercised("metal-gqa-kv8").is_err());
        }
    }

    #[test]
    fn temporal256_slots_are_selected_individually_and_never_foreign() {
        for kind in [ResearchKernel::Temporal256, ResearchKernel::Temporal512] {
            let counters = ResearchDispatchCounters::default();
            counters.record(kind);
            let got = counters.snapshot();
            assert_eq!(got.selection_exercised("metal-attn-q4"), Ok(true));
            assert!(got.selection_exercised("off").is_err());
            assert!(got.selection_exercised("metal-gqa-kv8").is_err());
        }
    }

    #[test]
    fn rms32_slots_are_selected_individually_and_never_foreign() {
        for kind in [ResearchKernel::Rms32] {
            let counters = ResearchDispatchCounters::default();
            counters.record(kind);
            let got = counters.snapshot();
            assert_eq!(got.selection_exercised("metal-rms-simd32"), Ok(true));
            assert!(got.selection_exercised("off").is_err());
            assert!(got.selection_exercised("metal-gqa-kv8").is_err());
        }
    }
}
