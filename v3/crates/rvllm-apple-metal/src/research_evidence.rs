//! Encoded-dispatch evidence, not GPU completion or numerical qualification.
#![forbid(unsafe_code)]

#[cfg(any(target_os = "macos", target_os = "ios", test))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Append-only diagnostic slots; the first five retain their original indices.
/// Consumers must bind the registry and executable used by a receipt.
pub const RESEARCH_DISPATCH_SCHEMA: &str = "rvllm.metal.research-dispatch.v3";
pub const RESEARCH_KERNEL_COUNT: usize = 46;
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
    "wave2_gemm_mma32_f32",
    "wave2_qkv_mma32_f32",
    "wave2_gemm_mma32x64",
    "wave2_qkv_mma32x64",
    "wave2_gemm_mma32_load4",
    "wave2_qkv_mma32_load4",
    "wave2_rmsnorm_simd256",
    "research_gemm_load4_m16n32k64",
    "research_qkv_load4_m16n32k64",
    "research_gemm_load4_m16n64k64",
    "research_qkv_load4_m16n64k64",
    "research_gemm_load4_m32n32k64",
    "research_qkv_load4_m32n32k64",
    "research_gemm_load4_m32n64k32",
    "research_qkv_load4_m32n64k32",
    "research_gemm_load4_m32n64k64",
    "research_qkv_load4_m32n64k64",
    "research_gemm_load4_m32n64k128",
    "research_qkv_load4_m32n64k128",
    "research_gemm_load4_m64n64k64",
    "research_qkv_load4_m64n64k64",
    "research_global_d512_r8p64t64",
    "research_global_d512_r8p64t128",
    "research_global_d512_r8p128t64",
    "research_global_d512_r8p128t128",
    "research_global_d512_r16p64t64",
    "research_global_d512_r16p64t128",
    "research_global_d512_r16p128t64",
    "research_global_d512_r16p128t128",
    "research_global_d512_r1p128t32",
    "research_global_d512_split_r8s256t128_partial",
    "research_global_d512_split_r8s256t128_merge",
    "research_global_d512_atlas_r16k16p64t128",
    "research_global_d512_atlas_r16k32p64t128",
    "research_global_d512_atlas_tile_r16k16p64t128",
    "research_global_d512_atlas_tile_r16k32p64t128",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum ResearchKernel {
    ShortGemm = 0,
    ShortQkv = 1,
    RoundedGate = 2,
    Gqa256 = 3,
    Gqa512 = 4,
    PrefetchGemm = 5,
    PrefetchQkv = 6,
    Temporal256 = 7,
    Temporal512 = 8,
    Rms32 = 9,
    F32Gemm = 10,
    F32Qkv = 11,
    LongGemm = 12,
    LongQkv = 13,
    Load4Gemm = 14,
    Load4Qkv = 15,
    Rms256 = 16,
    Tile16x32K64Gemm = 17,
    Tile16x32K64Qkv = 18,
    Tile16x64K64Gemm = 19,
    Tile16x64K64Qkv = 20,
    Tile32x32K64Gemm = 21,
    Tile32x32K64Qkv = 22,
    Tile32x64K32Gemm = 23,
    Tile32x64K32Qkv = 24,
    Tile32x64K64Gemm = 25,
    Tile32x64K64Qkv = 26,
    Tile32x64K128Gemm = 27,
    Tile32x64K128Qkv = 28,
    Tile64x64K64Gemm = 29,
    Tile64x64K64Qkv = 30,
    GlobalD512R8P64T64 = 31,
    GlobalD512R8P64T128 = 32,
    GlobalD512R8P128T64 = 33,
    GlobalD512R8P128T128 = 34,
    GlobalD512R16P64T64 = 35,
    GlobalD512R16P64T128 = 36,
    GlobalD512R16P128T64 = 37,
    GlobalD512R16P128T128 = 38,
    GlobalD512R1P128T32 = 39,
    GlobalD512SplitR8S256T128Partial = 40,
    GlobalD512SplitR8S256T128Merge = 41,
    GlobalD512AtlasR16K16P64T128 = 42,
    GlobalD512AtlasR16K32P64T128 = 43,
    GlobalD512AtlasTileR16K16P64T128 = 44,
    GlobalD512AtlasTileR16K32P64T128 = 45,
}

impl ResearchKernel {
    pub const fn name(self) -> &'static str {
        RESEARCH_KERNEL_NAMES[self as usize]
    }
    /// Source budgets, checked in addition to queried PSO/device limits.
    pub const fn limits(self) -> (usize, usize) {
        match self {
            Self::GlobalD512R8P64T64 => (64, 10016),
            Self::GlobalD512R8P64T128 => (128, 10016),
            Self::GlobalD512R8P128T64 => (64, 11040),
            Self::GlobalD512R8P128T128 => (128, 11040),
            Self::GlobalD512R16P64T64 => (64, 18976),
            Self::GlobalD512R16P64T128 => (128, 18976),
            Self::GlobalD512R16P128T64 => (64, 20000),
            Self::GlobalD512R16P128T128 => (128, 20000),
            Self::GlobalD512R1P128T32 => (32, 3200),
            Self::GlobalD512SplitR8S256T128Partial => (128, 10016),
            Self::GlobalD512SplitR8S256T128Merge => (32, 0),
            Self::GlobalD512AtlasR16K16P64T128 | Self::GlobalD512AtlasTileR16K16P64T128 => {
                (128, 21568)
            }
            Self::GlobalD512AtlasR16K32P64T128 | Self::GlobalD512AtlasTileR16K32P64T128 => {
                (128, 26752)
            }
            Self::ShortGemm => (128, 10496),
            Self::ShortQkv => (128, 10496),
            Self::RoundedGate => (128, 14336),
            Self::Gqa256 => (64, 8224),
            Self::Gqa512 => (128, 16416),
            Self::PrefetchGemm => (128, 8192),
            Self::PrefetchQkv => (128, 8192),
            Self::Temporal256 => (128, 16448),
            Self::Temporal512 => (128, 16416),
            Self::Rms32 => (32, 0),
            Self::F32Gemm => (128, 12288),
            Self::F32Qkv => (128, 12288),
            Self::LongGemm => (128, 14336),
            Self::LongQkv => (128, 14336),
            Self::Load4Gemm => (128, 8192),
            Self::Load4Qkv => (128, 8192),
            Self::Rms256 => (256, 32),
            Self::Tile16x32K64Gemm | Self::Tile16x32K64Qkv => (64, 6144),
            Self::Tile16x64K64Gemm | Self::Tile16x64K64Qkv => (128, 10240),
            Self::Tile32x32K64Gemm | Self::Tile32x32K64Qkv => (128, 8192),
            Self::Tile32x64K32Gemm | Self::Tile32x64K32Qkv => (128, 8192),
            Self::Tile32x64K64Gemm | Self::Tile32x64K64Qkv => (128, 12288),
            Self::Tile32x64K128Gemm | Self::Tile32x64K128Qkv => (128, 24576),
            Self::Tile64x64K64Gemm | Self::Tile64x64K64Qkv => (128, 16384),
        }
    }
    pub const fn owner(self) -> crate::research::MetalResearchCandidate {
        use crate::research::MetalResearchCandidate;
        match self {
            Self::GlobalD512R8P64T64 => MetalResearchCandidate::GlobalD512R8P64T64,
            Self::GlobalD512R8P64T128 => MetalResearchCandidate::GlobalD512R8P64T128,
            Self::GlobalD512R8P128T64 => MetalResearchCandidate::GlobalD512R8P128T64,
            Self::GlobalD512R8P128T128 => MetalResearchCandidate::GlobalD512R8P128T128,
            Self::GlobalD512R16P64T64 => MetalResearchCandidate::GlobalD512R16P64T64,
            Self::GlobalD512R16P64T128 => MetalResearchCandidate::GlobalD512R16P64T128,
            Self::GlobalD512R16P128T64 => MetalResearchCandidate::GlobalD512R16P128T64,
            Self::GlobalD512R16P128T128 => MetalResearchCandidate::GlobalD512R16P128T128,
            Self::GlobalD512R1P128T32 => MetalResearchCandidate::GlobalD512R1P128T32,
            Self::GlobalD512SplitR8S256T128Partial | Self::GlobalD512SplitR8S256T128Merge => {
                MetalResearchCandidate::GlobalD512SplitR8S256T128
            }
            Self::GlobalD512AtlasR16K16P64T128 => {
                MetalResearchCandidate::GlobalD512AtlasR16K16P64T128
            }
            Self::GlobalD512AtlasR16K32P64T128 => {
                MetalResearchCandidate::GlobalD512AtlasR16K32P64T128
            }
            Self::GlobalD512AtlasTileR16K16P64T128 => {
                MetalResearchCandidate::GlobalD512AtlasTileR16K16P64T128
            }
            Self::GlobalD512AtlasTileR16K32P64T128 => {
                MetalResearchCandidate::GlobalD512AtlasTileR16K32P64T128
            }
            Self::ShortGemm | Self::ShortQkv => MetalResearchCandidate::ShortMma16x64,
            Self::RoundedGate => MetalResearchCandidate::RoundedGate32,
            Self::Gqa256 | Self::Gqa512 => MetalResearchCandidate::GqaKv8,
            Self::PrefetchGemm | Self::PrefetchQkv => MetalResearchCandidate::Mma32Prefetch,
            Self::Temporal256 | Self::Temporal512 => MetalResearchCandidate::AttentionQ4,
            Self::Rms32 => MetalResearchCandidate::RmsSimd32,
            Self::F32Gemm | Self::F32Qkv => MetalResearchCandidate::Mma32F32,
            Self::LongGemm | Self::LongQkv => MetalResearchCandidate::LongMma32x64,
            Self::Load4Gemm | Self::Load4Qkv => MetalResearchCandidate::Mma32Load4,
            Self::Rms256 => MetalResearchCandidate::RmsnormSimd256,
            Self::Tile16x32K64Gemm | Self::Tile16x32K64Qkv => {
                MetalResearchCandidate::Load4M16N32K64
            }
            Self::Tile16x64K64Gemm | Self::Tile16x64K64Qkv => {
                MetalResearchCandidate::Load4M16N64K64
            }
            Self::Tile32x32K64Gemm | Self::Tile32x32K64Qkv => {
                MetalResearchCandidate::Load4M32N32K64
            }
            Self::Tile32x64K32Gemm | Self::Tile32x64K32Qkv => {
                MetalResearchCandidate::Load4M32N64K32
            }
            Self::Tile32x64K64Gemm | Self::Tile32x64K64Qkv => {
                MetalResearchCandidate::Load4M32N64K64
            }
            Self::Tile32x64K128Gemm | Self::Tile32x64K128Qkv => {
                MetalResearchCandidate::Load4M32N64K128
            }
            Self::Tile64x64K64Gemm | Self::Tile64x64K64Qkv => {
                MetalResearchCandidate::Load4M64N64K64
            }
        }
    }
}

/// Sample only at a quiescent owner boundary. These atomics do not synchronize
/// Metal resources or establish that a command buffer completed successfully.
#[cfg(any(target_os = "macos", target_os = "ios", test))]
#[derive(Debug)]
pub(crate) struct ResearchDispatchCounters {
    counts: [AtomicU64; RESEARCH_KERNEL_COUNT],
    overflowed: AtomicBool,
}

#[cfg(any(target_os = "macos", target_os = "ios", test))]
impl Default for ResearchDispatchCounters {
    fn default() -> Self {
        Self {
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
            overflowed: AtomicBool::new(false),
        }
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResearchDispatchSnapshot {
    pub counts: [u64; RESEARCH_KERNEL_COUNT],
    pub overflowed: bool,
}

impl Default for ResearchDispatchSnapshot {
    fn default() -> Self {
        Self {
            counts: [0; RESEARCH_KERNEL_COUNT],
            overflowed: false,
        }
    }
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
        let candidate: crate::research::MetalResearchCandidate = requested.parse()?;
        let allowed = candidate.kernels();
        if self
            .counts
            .iter()
            .enumerate()
            .any(|(i, &n)| n != 0 && !allowed.iter().any(|kernel| *kernel as usize == i))
        {
            return Err("receipt contains a different candidate's dispatches");
        }
        Ok(requested == "off" || self.counts.iter().any(|&n| n != 0))
    }
    /// Require every entry point in the explicitly selected family. This is a
    /// coverage check, NOT tensor accuracy or complete per-layer work accounting.
    pub fn complete_family_exercised(self, requested: &str) -> Result<bool, &'static str> {
        if !self.selection_exercised(requested)? {
            return Ok(false);
        }
        let candidate: crate::research::MetalResearchCandidate = requested.parse()?;
        Ok(candidate
            .kernels()
            .iter()
            .all(|k| self.counts[*k as usize] != 0))
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
    #[test]
    fn incomplete_family_is_not_full_coverage_and_new_slots_do_not_alias() {
        for candidate in crate::research_catalog::ALL_CANDIDATES {
            let mut snapshot = ResearchDispatchSnapshot::default();
            for kernel in candidate.kernels() {
                snapshot.counts[*kernel as usize] = 1;
            }
            assert_eq!(
                snapshot.complete_family_exercised(candidate.name()),
                Ok(true)
            );
            if candidate.kernels().len() > 1 {
                snapshot.counts[candidate.kernels()[0] as usize] = 0;
                assert_eq!(snapshot.selection_exercised(candidate.name()), Ok(true));
                assert_eq!(
                    snapshot.complete_family_exercised(candidate.name()),
                    Ok(false)
                );
            }
            if candidate != crate::research::MetalResearchCandidate::Off {
                assert!(snapshot.complete_family_exercised("off").is_err());
            }
        }
    }
}
