//! Source-defined PR27 geometry. No environment access, allocation, or device calls.
//! Resource counts below are source budgets, never measured physical registers.
#![forbid(unsafe_code)]
use crate::research::MetalResearchCandidate as C;
use crate::research_catalog::CandidateSpec;
use crate::research_evidence::ResearchKernel as K;

pub const BASE: &str = "c009497bee8125d456e4aee26f0b35836d79c231";
pub const FAMILY: [C; 4] = [
    C::PrefillWide64,
    C::PrefillWide128,
    C::PrefillMma8K32,
    C::PrefillWide64Mma8K32,
];

pub const fn projection_tile(c: C) -> Option<[usize; 2]> {
    match c {
        C::PrefillWide64 | C::PrefillWide64Mma8K32 => Some([64, 64]),
        C::PrefillWide128 => Some([32, 128]),
        _ => None,
    }
}
pub const fn query_tile(c: C) -> Option<usize> {
    match c {
        C::PrefillMma8K32 | C::PrefillWide64Mma8K32 => Some(8),
        _ => None,
    }
}
/// Independent host version of the MLX coordinate formula, tested bijectively.
/// The Metal probe is still mandatory: host algebra cannot establish device ABI.
pub const fn fragment_coord(lane: usize) -> Option<[usize; 2]> {
    if lane >= 32 {
        return None;
    }
    let qid = lane / 4;
    Some([(qid & 4) + ((lane / 2) % 4), (qid & 2) * 2 + (lane % 2) * 2])
}

pub const fn projection_kernels(c: C) -> Option<[K; 2]> {
    match c {
        C::PrefillWide64 => Some([K::Wide64Gemm, K::Wide64Qkv]),
        C::PrefillWide128 => Some([K::Wide128Gemm, K::Wide128Qkv]),
        C::PrefillWide64Mma8K32 => Some([K::Wide64Mma8K32Gemm, K::Wide64Mma8K32Qkv]),
        _ => None,
    }
}

pub const fn postnorm_kernels(c: C) -> Option<[K; 2]> {
    match c {
        C::PrefillWide64 => Some([K::Wide64Raw, K::Wide64Norm]),
        C::PrefillWide128 => Some([K::Wide128Raw, K::Wide128Norm]),
        C::PrefillWide64Mma8K32 => Some([K::Wide64Mma8K32Raw, K::Wide64Mma8K32Norm]),
        _ => None,
    }
}

pub const fn attention_kernel(c: C, d: u32) -> Option<K> {
    match (c, d) {
        (C::PrefillMma8K32, 256) => Some(K::Mma8K32D256),
        (C::PrefillMma8K32, 512) => Some(K::Mma8K32D512),
        (C::PrefillWide64Mma8K32, 256) => Some(K::Wide64Mma8K32D256),
        (C::PrefillWide64Mma8K32, 512) => Some(K::Wide64Mma8K32D512),
        _ => None,
    }
}

pub const fn spec(c: C) -> CandidateSpec {
    match c {
        C::PrefillWide64 => CandidateSpec {
            name: "metal-prefill-wide64",
            kernels: &[K::Wide64Gemm, K::Wide64Qkv, K::Wide64Raw, K::Wide64Norm],
            source_file: Some(
                "crates/rvllm-apple-metal/src/research_shaders/prefill27_wide64.metal",
            ),
            min_tokens: 6,
            max_tokens: 2048,
            window_independent: true,
            numerical_contract: "bf16-k8-wide-fragment-store",
            source: concat!(
                include_str!("research_shaders/prefill27_fragments.metal"),
                include_str!("research_shaders/prefill27_projection.metal"),
                include_str!("research_shaders/prefill_postnorm_common.metal"),
                include_str!("research_shaders/prefill27_wide64.metal"),
            ),
        },
        C::PrefillWide128 => CandidateSpec {
            name: "metal-prefill-wide128",
            kernels: &[K::Wide128Gemm, K::Wide128Qkv, K::Wide128Raw, K::Wide128Norm],
            source_file: Some(
                "crates/rvllm-apple-metal/src/research_shaders/prefill27_wide128.metal",
            ),
            min_tokens: 6,
            max_tokens: 2048,
            window_independent: true,
            numerical_contract: "bf16-k8-wide-fragment-store",
            source: concat!(
                include_str!("research_shaders/prefill27_fragments.metal"),
                include_str!("research_shaders/prefill27_projection.metal"),
                include_str!("research_shaders/prefill_postnorm_common.metal"),
                include_str!("research_shaders/prefill27_wide128.metal"),
            ),
        },
        C::PrefillMma8K32 => CandidateSpec {
            name: "metal-prefill-mma8k32",
            kernels: &[K::Mma8K32D256, K::Mma8K32D512],
            source_file: Some(
                "crates/rvllm-apple-metal/src/research_shaders/prefill27_mma8k32.metal",
            ),
            min_tokens: 6,
            max_tokens: 2048,
            window_independent: false,
            numerical_contract: "bf16-fp32-mma-q8k32-paged",
            source: concat!(
                include_str!("research_shaders/prefill27_fragments.metal"),
                include_str!("research_shaders/prefill27_attention.metal"),
                include_str!("research_shaders/prefill27_mma8k32.metal"),
            ),
        },
        C::PrefillWide64Mma8K32 => CandidateSpec {
            name: "metal-prefill-wide64-mma8k32",
            kernels: &[
                K::Wide64Mma8K32Gemm,
                K::Wide64Mma8K32Qkv,
                K::Wide64Mma8K32Raw,
                K::Wide64Mma8K32Norm,
                K::Wide64Mma8K32D256,
                K::Wide64Mma8K32D512,
            ],
            source_file: Some(
                "crates/rvllm-apple-metal/src/research_shaders/prefill27_wide64_mma8k32.metal",
            ),
            min_tokens: 6,
            max_tokens: 2048,
            window_independent: false,
            numerical_contract: "bf16-fp32-mma-q8k32-paged",
            source: concat!(
                include_str!("research_shaders/prefill27_fragments.metal"),
                include_str!("research_shaders/prefill27_projection.metal"),
                include_str!("research_shaders/prefill_postnorm_common.metal"),
                include_str!("research_shaders/prefill27_attention.metal"),
                include_str!("research_shaders/prefill27_wide64_mma8k32.metal"),
            ),
        },
        _ => panic!("not a PR27 candidate"),
    }
}

#[cfg(test)]
#[path = "prefill_next_tests.rs"]
mod tests;
