//! One checked launch description for all explicit research projections/norms.
//! This module never acquires a device, allocates scratch, or records dispatch.
#![forbid(unsafe_code)]

use crate::research::{buffer_span, disjoint_writes, matrix_bytes, projection_buffers_fit};
use crate::research::MetalResearchCandidate;
use crate::research_evidence::ResearchKernel;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallbackReason {
    NotThisOperation,
    ModelPhaseOrTrace,
    StoragePrecision,
    ProjectionScale,
    Shape,
    Alignment,
    BufferOrAlias,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionPlan {
    pub kernel: ResearchKernel,
    pub tile_m: usize,
    pub tile_n: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectionRequest {
    pub candidate: MetalResearchCandidate,
    /// Supplied only by the incumbent full-model/prefill and trace predicate.
    /// Matrix dimensions alone do not establish the model/phase contract.
    pub full_prefill: bool,
    pub native_bf16: bool,
    pub alpha: f32,
    pub beta: f32,
    pub shape: [u32; 3],
    pub output_f32: bool,
    pub offsets: [usize; 3],
    pub arena_bytes: usize,
}

impl ProjectionRequest {
    pub fn plan(self) -> Result<ProjectionPlan, FallbackReason> {
        use MetalResearchCandidate::*;
        let pair = match self.candidate {
            ShortMma16x64 => [ResearchKernel::ShortGemm, ResearchKernel::ShortQkv],
            Mma32Prefetch => [ResearchKernel::PrefetchGemm, ResearchKernel::PrefetchQkv],
            Mma32F32 => [ResearchKernel::F32Gemm, ResearchKernel::F32Qkv],
            LongMma32x64 => [ResearchKernel::LongGemm, ResearchKernel::LongQkv],
            Mma32Load4 => [ResearchKernel::Load4Gemm, ResearchKernel::Load4Qkv],
            _ => return Err(FallbackReason::NotThisOperation),
        };
        if !self.full_prefill { return Err(FallbackReason::ModelPhaseOrTrace); }
        if !self.native_bf16 { return Err(FallbackReason::StoragePrecision); }
        if self.alpha != 1.0 || self.beta != 0.0 {
            return Err(FallbackReason::ProjectionScale);
        }
        let [m, n, k] = self.shape;
        let spec = self.candidate.spec();
        if !(spec.min_tokens..=spec.max_tokens).contains(&m)
            || !crate::research_next::prefetch_projection_shape(m, n, k, self.output_f32)
        {
            return Err(FallbackReason::Shape);
        }
        if self.candidate == Mma32Load4 && (self.offsets[0] % 8 != 0 || self.offsets[1] % 8 != 0) {
            return Err(FallbackReason::Alignment);
        }
        if !projection_buffers_fit(self.offsets, self.shape,
            if self.output_f32 { 4 } else { 2 }, self.arena_bytes)
        {
            return Err(FallbackReason::BufferOrAlias);
        }
        let (tile_m, tile_n) = projection_tile(self.candidate).ok_or(FallbackReason::NotThisOperation)?;
        Ok(ProjectionPlan { kernel: pair[usize::from(self.output_f32)], tile_m, tile_n })
    }
}

/// Intrinsic output tile, not runtime shape admission. Component oracles use
/// this geometry with their own explicitly guarded synthetic tail fixtures.
pub fn projection_tile(candidate: MetalResearchCandidate) -> Option<(usize, usize)> {
    use MetalResearchCandidate::*;
    match candidate {
        ShortMma16x64 => Some((16, 64)),
        LongMma32x64 => Some((32, 64)),
        Mma32Prefetch | Mma32F32 | Mma32Load4 => Some((32, 32)),
        _ => None,
    }
}

/// Preserve exact in-place normalization, never partial aliases or a gamma
/// overlap. Explicit bounds are checked before a device can observe the plan.
pub fn postnorm_buffers_fit(offsets: [usize; 3], tokens: u32, hidden: u32, arena_bytes: usize) -> bool {
    if !(6..=1024).contains(&tokens) || hidden != 3840 { return false; }
    let checked = || {
        let bytes = matrix_bytes(tokens, hidden, 2)?;
        let input = buffer_span(offsets[0], bytes, arena_bytes)?;
        let output = buffer_span(offsets[1], bytes, arena_bytes)?;
        let gamma = buffer_span(offsets[2], matrix_bytes(1, hidden, 2)?, arena_bytes)?;
        let input_ok = input == output || disjoint_writes(&[input], &[output.clone()]);
        Some(input_ok && disjoint_writes(&[gamma], &[output]))
    };
    checked().unwrap_or(false)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PostnormPlan {
    pub kernel: ResearchKernel,
    pub binds_token_count: bool,
}

pub fn postnorm_plan(
    candidate: MetalResearchCandidate,
    full_prefill: bool,
    native_bf16: bool,
    offsets: [usize; 3],
    tokens: u32,
    hidden: u32,
    eps: f32,
    arena_bytes: usize,
) -> Result<PostnormPlan, FallbackReason> {
    let (kernel, binds_token_count) = match candidate {
        MetalResearchCandidate::RmsSimd32 => (ResearchKernel::Rms32, true),
        MetalResearchCandidate::RmsnormSimd256 => (ResearchKernel::Rms256, false),
        _ => return Err(FallbackReason::NotThisOperation),
    };
    if !full_prefill { return Err(FallbackReason::ModelPhaseOrTrace); }
    if !native_bf16 { return Err(FallbackReason::StoragePrecision); }
    if eps != 1.0e-6 || !(6..=1024).contains(&tokens) || hidden != 3840 {
        return Err(FallbackReason::Shape);
    }
    if !postnorm_buffers_fit(offsets, tokens, hidden, arena_bytes) {
        return Err(FallbackReason::BufferOrAlias);
    }
    Ok(PostnormPlan { kernel, binds_token_count })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(candidate: MetalResearchCandidate, m: u32, n: u32, k: u32, fp32: bool) -> ProjectionRequest {
        let a = m as usize * k as usize * 2;
        let b = n as usize * k as usize * 2;
        ProjectionRequest { candidate, full_prefill: true, native_bf16: true,
            alpha: 1.0, beta: 0.0, shape: [m, n, k], output_f32: fp32,
            offsets: [0, a, a + b], arena_bytes: a + b + m as usize * n as usize * if fp32 { 4 } else { 2 } }
    }

    #[test]
    fn old_projection_routes_retain_their_exact_shape_sets() {
        for candidate in [MetalResearchCandidate::ShortMma16x64, MetalResearchCandidate::Mma32Prefetch] {
            for m in [0, 1, 5, 6, 15, 16, 17, 32, 63, 64, 65, 84, 1024, 1025] {
                for (n, k, fp32) in [(8192, 3840, true), (9216, 3840, true),
                    (30720, 3840, false), (3840, 4096, false), (3840, 8192, false), (3840, 15360, false)] {
                    let expected = match candidate {
                        MetalResearchCandidate::ShortMma16x64 => crate::research::short_mma_shape(m, n, k, fp32),
                        _ => crate::research_next::prefetch_projection_shape(m, n, k, fp32),
                    };
                    assert_eq!(request(candidate, m, n, k, fp32).plan().is_ok(), expected);
                }
            }
        }
    }

    #[test]
    fn every_projection_requires_identity_precision_scale_and_nonaliasing() {
        for candidate in [MetalResearchCandidate::ShortMma16x64, MetalResearchCandidate::Mma32Prefetch,
            MetalResearchCandidate::Mma32F32, MetalResearchCandidate::LongMma32x64, MetalResearchCandidate::Mma32Load4] {
            let good = request(candidate, 64, 8192, 3840, true);
            let plan = good.plan().unwrap();
            assert_eq!(plan.kernel.owner(), candidate);
            assert_eq!(ProjectionRequest { full_prefill: false, ..good }.plan(), Err(FallbackReason::ModelPhaseOrTrace));
            assert_eq!(ProjectionRequest { native_bf16: false, ..good }.plan(), Err(FallbackReason::StoragePrecision));
            for alpha in [0.0, 0.5, f32::NAN, f32::INFINITY] {
                assert_eq!(ProjectionRequest { alpha, ..good }.plan(), Err(FallbackReason::ProjectionScale));
            }
            assert_eq!(ProjectionRequest { beta: 1.0, ..good }.plan(), Err(FallbackReason::ProjectionScale));
            assert!(ProjectionRequest { output_f32: false, ..good }.plan().is_err());
            assert!(ProjectionRequest { arena_bytes: good.arena_bytes - 1, ..good }.plan().is_err());
            assert!(ProjectionRequest { offsets: [0, good.offsets[1], 0], ..good }.plan().is_err());
            assert!(ProjectionRequest { shape: [u32::MAX, 8192, 3840], ..good }.plan().is_err());
        }
    }

    #[test]
    fn long_tile_and_vector_alignment_are_independent_admission_axes() {
        let long = request(MetalResearchCandidate::LongMma32x64, 63, 30720, 3840, false);
        assert_eq!(long.plan(), Err(FallbackReason::Shape));
        assert!(request(MetalResearchCandidate::Mma32Load4, 6, 30720, 3840, false).plan().is_ok());
        let mut v = request(MetalResearchCandidate::Mma32Load4, 64, 8192, 3840, true);
        v.offsets[0] = 2;
        assert_eq!(v.plan(), Err(FallbackReason::Alignment));
        let long = request(MetalResearchCandidate::LongMma32x64, 84, 8192, 3840, true).plan().unwrap();
        assert_eq!((long.tile_m, long.tile_n, long.kernel.limits()), (32, 64, (128, 14336)));
    }

    #[test]
    fn rms_alias_eps_and_full_model_gates_are_shared_by_both_reductions() {
        let bytes = 6 * 3840 * 2;
        let end = bytes + 3840 * 2;
        for candidate in [MetalResearchCandidate::RmsSimd32, MetalResearchCandidate::RmsnormSimd256] {
            assert!(postnorm_plan(candidate, true, true, [0, 0, bytes], 6, 3840, 1e-6, end).is_ok());
            for offsets in [[0, 2, bytes], [0, 0, 0], [usize::MAX, 0, bytes]] {
                assert!(postnorm_plan(candidate, true, true, offsets, 6, 3840, 1e-6, end).is_err());
            }
            for eps in [0.0, -1.0, 1e-5, f32::NAN, f32::INFINITY] {
                assert!(postnorm_plan(candidate, true, true, [0, 0, bytes], 6, 3840, eps, end).is_err());
            }
            assert!(postnorm_plan(candidate, false, true, [0, 0, bytes], 6, 3840, 1e-6, end).is_err());
            assert!(postnorm_plan(candidate, true, false, [0, 0, bytes], 6, 3840, 1e-6, end).is_err());
        }
    }
}
