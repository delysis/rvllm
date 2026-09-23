//! Explicit research identities and source budgets, never measured GPU limits.
#![forbid(unsafe_code)]

use crate::research::MetalResearchCandidate;
use crate::research_evidence::ResearchKernel;

#[derive(Clone, Copy, Debug)]
pub struct CandidateSpec {
    pub name: &'static str,
    pub kernels: &'static [ResearchKernel],
    pub source_file: Option<&'static str>,
    pub min_tokens: u32,
    pub max_tokens: u32,
    pub window_independent: bool,
    pub numerical_contract: &'static str,
    pub(crate) source: &'static str,
}

pub const ALL_CANDIDATES: [MetalResearchCandidate; 11] = [
    MetalResearchCandidate::Off,
    MetalResearchCandidate::ShortMma16x64,
    MetalResearchCandidate::RoundedGate32,
    MetalResearchCandidate::GqaKv8,
    MetalResearchCandidate::Mma32Prefetch,
    MetalResearchCandidate::AttentionQ4,
    MetalResearchCandidate::RmsSimd32,
    MetalResearchCandidate::Mma32F32,
    MetalResearchCandidate::LongMma32x64,
    MetalResearchCandidate::Mma32Load4,
    MetalResearchCandidate::RmsnormSimd256,
];

impl MetalResearchCandidate {
    pub const fn spec(self) -> CandidateSpec {
        use ResearchKernel::*;
        match self {
            Self::Off => CandidateSpec {
                name: "off",
                kernels: &[],
                source_file: None,
                min_tokens: 0, max_tokens: 0,
                window_independent: false,
                numerical_contract: "baseline",
                source: "",
            },
            Self::ShortMma16x64 => CandidateSpec {
                name: "metal-short-mma16x64",
                kernels: &[ShortGemm, ShortQkv],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/short_mma16x64.metal"),
                min_tokens: 6, max_tokens: 64,
                window_independent: true,
                numerical_contract: "storage-boundaries-preserved",
                source: include_str!("research_shaders/short_mma16x64.metal"),
            },
            Self::RoundedGate32 => CandidateSpec {
                name: "metal-rounded-gate32",
                kernels: &[RoundedGate],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/rounded_gate32.metal"),
                min_tokens: 6, max_tokens: 1024,
                window_independent: false,
                numerical_contract: "storage-boundaries-preserved",
                source: include_str!("research_shaders/rounded_gate32.metal"),
            },
            Self::GqaKv8 => CandidateSpec {
                name: "metal-gqa-kv8",
                kernels: &[Gqa256, Gqa512],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/gqa_kv8.metal"),
                min_tokens: 64, max_tokens: 1024,
                window_independent: false,
                numerical_contract: "fp32-online-softmax",
                source: include_str!("research_shaders/gqa_kv8.metal"),
            },
            Self::Mma32Prefetch => CandidateSpec {
                name: "metal-mma32-prefetch",
                kernels: &[PrefetchGemm, PrefetchQkv],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/mma32_prefetch.metal"),
                min_tokens: 6, max_tokens: 1024,
                window_independent: true,
                numerical_contract: "same-contraction-order",
                source: include_str!("research_shaders/mma32_prefetch.metal"),
            },
            Self::AttentionQ4 => CandidateSpec {
                name: "metal-attn-q4",
                kernels: &[Temporal256, Temporal512],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/attn_q4.metal"),
                min_tokens: 64, max_tokens: 1024,
                window_independent: false,
                numerical_contract: "fp32-online-softmax",
                source: include_str!("research_shaders/attn_q4.metal"),
            },
            Self::RmsSimd32 => CandidateSpec {
                name: "metal-rms-simd32",
                kernels: &[Rms32],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/rms_simd32.metal"),
                min_tokens: 6, max_tokens: 1024,
                window_independent: true,
                numerical_contract: "reduction-order-change",
                source: include_str!("research_shaders/rms_simd32.metal"),
            },
            Self::Mma32F32 => CandidateSpec {
                name: "metal-mma32-f32",
                kernels: &[F32Gemm, F32Qkv],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/mma32_f32.metal"),
                min_tokens: 6, max_tokens: 1024,
                window_independent: true,
                numerical_contract: "operand-lowering-change",
                source: include_str!("research_shaders/mma32_f32.metal"),
            },
            Self::LongMma32x64 => CandidateSpec {
                name: "metal-long-mma32x64",
                kernels: &[LongGemm, LongQkv],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/long_mma32x64.metal"),
                min_tokens: 64, max_tokens: 1024,
                window_independent: true,
                numerical_contract: "storage-boundaries-preserved",
                source: include_str!("research_shaders/long_mma32x64.metal"),
            },
            Self::Mma32Load4 => CandidateSpec {
                name: "metal-mma32-load4",
                kernels: &[Load4Gemm, Load4Qkv],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/mma32_load4.metal"),
                min_tokens: 6, max_tokens: 1024,
                window_independent: true,
                numerical_contract: "layout-only-bitwise-fp32-gate",
                source: include_str!("research_shaders/mma32_load4.metal"),
            },
            Self::RmsnormSimd256 => CandidateSpec {
                name: "metal-rmsnorm-simd256",
                kernels: &[Rms256],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/rmsnorm_simd256.metal"),
                min_tokens: 6, max_tokens: 1024,
                window_independent: true,
                numerical_contract: "reduction-order-change",
                source: include_str!("research_shaders/rmsnorm_simd256.metal"),
            },
        }
    }
}

/// A source catalog only. Queried PSO/device limits and actual dispatch must
/// still be checked by the existing owner. Exporting this touches no device.
pub fn catalog_json() -> serde_json::Value {
    let candidates: Vec<_> = ALL_CANDIDATES.into_iter().map(|candidate| {
        let spec = candidate.spec();
        let kernels: Vec<_> = spec.kernels.iter().map(|k| k.name()).collect();
        let budgets: Vec<_> = spec.kernels.iter().map(|k| {
            let (threads, shared) = k.limits();
            serde_json::json!({"kernel": k.name(), "threads": threads,
                "source_shared_bytes": shared})
        }).collect();
        serde_json::json!({"name": spec.name, "kernels": kernels,
            "source_file": spec.source_file, "min_tokens": spec.min_tokens,
            "max_tokens": spec.max_tokens, "window_independent": spec.window_independent,
            "numerical_contract": spec.numerical_contract, "budgets": budgets})
    }).collect();
    serde_json::json!({"schema": "rvllm.metal.research-catalog.v1",
        "dispatch_schema": "rvllm.metal.research-dispatch.v3",
        "default": "off", "device_qualified": false, "candidates": candidates})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalFloatType, MetalKernelOptions};
    use crate::research_evidence::{RESEARCH_KERNEL_COUNT, RESEARCH_KERNEL_NAMES};
    use std::collections::BTreeSet;

    #[test]
    fn catalog_is_bijective_and_preserves_the_existing_ten_slots() {
        let mut names = BTreeSet::new();
        let mut slots = BTreeSet::new();
        for candidate in ALL_CANDIDATES {
            let spec = candidate.spec();
            assert!(names.insert(spec.name));
            assert_eq!(spec.name.parse(), Ok(candidate));
            for kernel in spec.kernels {
                assert!(slots.insert(*kernel as usize));
                assert_eq!(RESEARCH_KERNEL_NAMES[*kernel as usize], kernel.name());
                assert_eq!(kernel.owner(), candidate);
            }
        }
        assert_eq!(slots, (0..RESEARCH_KERNEL_COUNT).collect::<std::collections::BTreeSet<_>>());
        assert_eq!(&RESEARCH_KERNEL_NAMES[..10], &[
            "research_gemm_mma16x64", "research_qkv_mma16x64", "research_rounded_gate32",
            "research_gqa_kv8_d256", "research_gqa_kv8_d512", "research_gemm_mma32_prefetch",
            "research_qkv_mma32_prefetch", "research_attn_q4_d256", "research_attn_q4_d512",
            "research_rms_simd32"]);
    }

    #[test]
    fn reviewed_catalog_matches_runtime_and_every_exported_entry() {
        let reviewed: serde_json::Value = serde_json::from_str(
            include_str!("../../../tools/gemma4_metal_catalog.json")).unwrap();
        assert_eq!(reviewed, catalog_json());
        for candidate in ALL_CANDIDATES {
            for dtype in [MetalFloatType::Bf16, MetalFloatType::F16] {
                let source = crate::kernels::kernel_source_with_options(dtype,
                    MetalKernelOptions { research: candidate, ..MetalKernelOptions::default() });
                for kernel in candidate.spec().kernels {
                    let marker = format!("kernel void {}(", kernel.name());
                    assert_eq!(source.matches(&marker).count(), 1, "{marker}");
                }
            }
        }
        let off = MetalKernelOptions::default();
        assert_eq!(off.research, MetalResearchCandidate::Off);
        assert_eq!(crate::kernels::kernel_source_with_options(MetalFloatType::F16, off),
            crate::kernels::KERNEL_SOURCE);
    }

    #[test]
    fn all_candidates_reject_near_miss_model_identities() {
        use crate::research::Gemma12bResearchShape;
        let good = Gemma12bResearchShape { tokens: 84, hidden: 3840, intermediate: 15360,
            layers: 48, heads: 16, kv_heads: 8, head_dim: 256, attention_window: 1024,
            moe_experts: 0, moe_top_k: 0, moe_intermediate: 0, ple: 0 };
        for candidate in ALL_CANDIDATES {
            let good = Gemma12bResearchShape { tokens: candidate.spec().min_tokens, ..good };
            assert_eq!(good.supports(candidate), candidate != MetalResearchCandidate::Off);
            for bad in [Gemma12bResearchShape { hidden: 5376, ..good },
                Gemma12bResearchShape { layers: 47, ..good },
                Gemma12bResearchShape { head_dim: 128, ..good },
                Gemma12bResearchShape { moe_experts: 1, ..good },
                Gemma12bResearchShape { tokens: 0, ..good },
                Gemma12bResearchShape { tokens: 1025, ..good }] {
                assert!(!bad.supports(candidate));
            }
        }
    }
}
