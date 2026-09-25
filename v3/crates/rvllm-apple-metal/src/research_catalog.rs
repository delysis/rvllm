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

pub const ALL_CANDIDATES: [MetalResearchCandidate; 27] = [
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
    MetalResearchCandidate::Load4M16N32K64,
    MetalResearchCandidate::Load4M16N64K64,
    MetalResearchCandidate::Load4M32N32K64,
    MetalResearchCandidate::Load4M32N64K32,
    MetalResearchCandidate::Load4M32N64K64,
    MetalResearchCandidate::Load4M32N64K128,
    MetalResearchCandidate::Load4M64N64K64,
    MetalResearchCandidate::GlobalD512R8P64T64,
    MetalResearchCandidate::GlobalD512R8P64T128,
    MetalResearchCandidate::GlobalD512R8P128T64,
    MetalResearchCandidate::GlobalD512R8P128T128,
    MetalResearchCandidate::GlobalD512R16P64T64,
    MetalResearchCandidate::GlobalD512R16P64T128,
    MetalResearchCandidate::GlobalD512R16P128T64,
    MetalResearchCandidate::GlobalD512R16P128T128,
    MetalResearchCandidate::GlobalD512R1P128T32,
];

// Compile exactly one specialization pair with the shared implementation.
// The delivery source manifest also hashes load4_tiled_common.metal.
macro_rules! load4_tile_spec {
    ($suffix:literal, $gemm:ident, $qkv:ident, $min:literal) => {
        CandidateSpec {
            name: concat!("metal-load4-", $suffix),
            kernels: &[ResearchKernel::$gemm, ResearchKernel::$qkv],
            source_file: Some(concat!(
                "crates/rvllm-apple-metal/src/research_shaders/load4_",
                $suffix,
                ".metal"
            )),
            min_tokens: $min,
            max_tokens: 1024,
            window_independent: true,
            numerical_contract: "layout-only-bitwise-fp32-gate",
            source: concat!(
                include_str!("research_shaders/load4_tiled_common.metal"),
                include_str!(concat!("research_shaders/load4_", $suffix, ".metal"))
            ),
        }
    };
}

// One named entry point per opt-in family member; common source is included
// in the generated-MSL receipt, not loaded or concatenated during encoding.
macro_rules! global_decode_spec {
    ($suffix:literal, $kernel:ident) => {
        CandidateSpec {
            name: concat!("metal-global-d512-", $suffix),
            kernels: &[ResearchKernel::$kernel],
            source_file: Some(concat!(
                "crates/rvllm-apple-metal/src/research_shaders/global_decode_",
                $suffix,
                ".metal"
            )),
            min_tokens: 1,
            max_tokens: 1,
            window_independent: false,
            numerical_contract: "bf16-fp32-fixed64-tree-online-once-rounded",
            source: concat!(
                include_str!("research_shaders/global_decode_common.metal"),
                include_str!(concat!(
                    "research_shaders/global_decode_",
                    $suffix,
                    ".metal"
                ))
            ),
        }
    };
}

impl MetalResearchCandidate {
    pub const fn spec(self) -> CandidateSpec {
        use ResearchKernel::*;
        match self {
            Self::GlobalD512R8P64T64 => global_decode_spec!("r8p64t64", GlobalD512R8P64T64),
            Self::GlobalD512R8P64T128 => global_decode_spec!("r8p64t128", GlobalD512R8P64T128),
            Self::GlobalD512R8P128T64 => global_decode_spec!("r8p128t64", GlobalD512R8P128T64),
            Self::GlobalD512R8P128T128 => global_decode_spec!("r8p128t128", GlobalD512R8P128T128),
            Self::GlobalD512R16P64T64 => global_decode_spec!("r16p64t64", GlobalD512R16P64T64),
            Self::GlobalD512R16P64T128 => global_decode_spec!("r16p64t128", GlobalD512R16P64T128),
            Self::GlobalD512R16P128T64 => global_decode_spec!("r16p128t64", GlobalD512R16P128T64),
            Self::GlobalD512R16P128T128 => {
                global_decode_spec!("r16p128t128", GlobalD512R16P128T128)
            }
            Self::GlobalD512R1P128T32 => {
                global_decode_spec!("r1p128t32", GlobalD512R1P128T32)
            }
            Self::Off => CandidateSpec {
                name: "off",
                kernels: &[],
                source_file: None,
                min_tokens: 0,
                max_tokens: 0,
                window_independent: false,
                numerical_contract: "baseline",
                source: "",
            },
            Self::ShortMma16x64 => CandidateSpec {
                name: "metal-short-mma16x64",
                kernels: &[ShortGemm, ShortQkv],
                source_file: Some(
                    "crates/rvllm-apple-metal/src/research_shaders/short_mma16x64.metal",
                ),
                min_tokens: 6,
                max_tokens: 64,
                window_independent: true,
                numerical_contract: "storage-boundaries-preserved",
                source: include_str!("research_shaders/short_mma16x64.metal"),
            },
            Self::RoundedGate32 => CandidateSpec {
                name: "metal-rounded-gate32",
                kernels: &[RoundedGate],
                source_file: Some(
                    "crates/rvllm-apple-metal/src/research_shaders/rounded_gate32.metal",
                ),
                min_tokens: 6,
                max_tokens: 1024,
                window_independent: false,
                numerical_contract: "storage-boundaries-preserved",
                source: include_str!("research_shaders/rounded_gate32.metal"),
            },
            Self::GqaKv8 => CandidateSpec {
                name: "metal-gqa-kv8",
                kernels: &[Gqa256, Gqa512],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/gqa_kv8.metal"),
                min_tokens: 64,
                max_tokens: 1024,
                window_independent: false,
                numerical_contract: "fp32-online-softmax",
                source: include_str!("research_shaders/gqa_kv8.metal"),
            },
            Self::Mma32Prefetch => CandidateSpec {
                name: "metal-mma32-prefetch",
                kernels: &[PrefetchGemm, PrefetchQkv],
                source_file: Some(
                    "crates/rvllm-apple-metal/src/research_shaders/mma32_prefetch.metal",
                ),
                min_tokens: 6,
                max_tokens: 1024,
                window_independent: true,
                numerical_contract: "same-contraction-order",
                source: include_str!("research_shaders/mma32_prefetch.metal"),
            },
            Self::AttentionQ4 => CandidateSpec {
                name: "metal-attn-q4",
                kernels: &[Temporal256, Temporal512],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/attn_q4.metal"),
                min_tokens: 64,
                max_tokens: 1024,
                window_independent: false,
                numerical_contract: "fp32-online-softmax",
                source: include_str!("research_shaders/attn_q4.metal"),
            },
            Self::RmsSimd32 => CandidateSpec {
                name: "metal-rms-simd32",
                kernels: &[Rms32],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/rms_simd32.metal"),
                min_tokens: 6,
                max_tokens: 1024,
                window_independent: true,
                numerical_contract: "reduction-order-change",
                source: include_str!("research_shaders/rms_simd32.metal"),
            },
            Self::Mma32F32 => CandidateSpec {
                name: "metal-mma32-f32",
                kernels: &[F32Gemm, F32Qkv],
                source_file: Some("crates/rvllm-apple-metal/src/research_shaders/mma32_f32.metal"),
                min_tokens: 6,
                max_tokens: 1024,
                window_independent: true,
                numerical_contract: "operand-lowering-change",
                source: include_str!("research_shaders/mma32_f32.metal"),
            },
            Self::LongMma32x64 => CandidateSpec {
                name: "metal-long-mma32x64",
                kernels: &[LongGemm, LongQkv],
                source_file: Some(
                    "crates/rvllm-apple-metal/src/research_shaders/long_mma32x64.metal",
                ),
                min_tokens: 64,
                max_tokens: 1024,
                window_independent: true,
                numerical_contract: "storage-boundaries-preserved",
                source: include_str!("research_shaders/long_mma32x64.metal"),
            },
            Self::Mma32Load4 => CandidateSpec {
                name: "metal-mma32-load4",
                kernels: &[Load4Gemm, Load4Qkv],
                source_file: Some(
                    "crates/rvllm-apple-metal/src/research_shaders/mma32_load4.metal",
                ),
                min_tokens: 6,
                max_tokens: 1024,
                window_independent: true,
                numerical_contract: "layout-only-bitwise-fp32-gate",
                source: include_str!("research_shaders/mma32_load4.metal"),
            },
            Self::RmsnormSimd256 => CandidateSpec {
                name: "metal-rmsnorm-simd256",
                kernels: &[Rms256],
                source_file: Some(
                    "crates/rvllm-apple-metal/src/research_shaders/rmsnorm_simd256.metal",
                ),
                min_tokens: 6,
                max_tokens: 1024,
                window_independent: true,
                numerical_contract: "reduction-order-change",
                source: include_str!("research_shaders/rmsnorm_simd256.metal"),
            },
            Self::Load4M16N32K64 => {
                load4_tile_spec!("m16n32k64", Tile16x32K64Gemm, Tile16x32K64Qkv, 6)
            }
            Self::Load4M16N64K64 => {
                load4_tile_spec!("m16n64k64", Tile16x64K64Gemm, Tile16x64K64Qkv, 6)
            }
            Self::Load4M32N32K64 => {
                load4_tile_spec!("m32n32k64", Tile32x32K64Gemm, Tile32x32K64Qkv, 6)
            }
            Self::Load4M32N64K32 => {
                load4_tile_spec!("m32n64k32", Tile32x64K32Gemm, Tile32x64K32Qkv, 6)
            }
            Self::Load4M32N64K64 => {
                load4_tile_spec!("m32n64k64", Tile32x64K64Gemm, Tile32x64K64Qkv, 6)
            }
            Self::Load4M32N64K128 => {
                load4_tile_spec!("m32n64k128", Tile32x64K128Gemm, Tile32x64K128Qkv, 6)
            }
            Self::Load4M64N64K64 => {
                load4_tile_spec!("m64n64k64", Tile64x64K64Gemm, Tile64x64K64Qkv, 64)
            }
        }
    }
}

/// A source catalog only. Queried PSO/device limits and actual dispatch must
/// still be checked by the existing owner. Exporting this touches no device.
pub fn catalog_json() -> serde_json::Value {
    let candidates: Vec<_> = ALL_CANDIDATES
        .into_iter()
        .map(|candidate| {
            let spec = candidate.spec();
            let kernels: Vec<_> = spec.kernels.iter().map(|k| k.name()).collect();
            let budgets: Vec<_> = spec
                .kernels
                .iter()
                .map(|k| {
                    let (threads, shared) = k.limits();
                    serde_json::json!({"kernel": k.name(), "threads": threads,
                "source_shared_bytes": shared})
                })
                .collect();
            serde_json::json!({"name": spec.name, "kernels": kernels,
            "source_file": spec.source_file, "min_tokens": spec.min_tokens,
            "max_tokens": spec.max_tokens, "window_independent": spec.window_independent,
            "numerical_contract": spec.numerical_contract, "budgets": budgets})
        })
        .collect();
    serde_json::json!({"schema": "rvllm.metal.research-catalog.v1",
        "dispatch_schema": "rvllm.metal.research-dispatch.v3",
        "default": "off", "device_qualified": false, "candidates": candidates})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research_evidence::{RESEARCH_KERNEL_COUNT, RESEARCH_KERNEL_NAMES};
    use crate::{MetalFloatType, MetalKernelOptions};
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
        assert_eq!(
            slots,
            (0..RESEARCH_KERNEL_COUNT).collect::<std::collections::BTreeSet<_>>()
        );
        assert_eq!(
            &RESEARCH_KERNEL_NAMES[..10],
            &[
                "research_gemm_mma16x64",
                "research_qkv_mma16x64",
                "research_rounded_gate32",
                "research_gqa_kv8_d256",
                "research_gqa_kv8_d512",
                "research_gemm_mma32_prefetch",
                "research_qkv_mma32_prefetch",
                "research_attn_q4_d256",
                "research_attn_q4_d512",
                "research_rms_simd32"
            ]
        );
    }

    #[test]
    fn reviewed_catalog_matches_runtime_and_every_exported_entry() {
        let reviewed: serde_json::Value =
            serde_json::from_str(include_str!("../../../tools/gemma4_metal_catalog.json")).unwrap();
        let mut legacy = catalog_json();
        let all = legacy["candidates"].as_array_mut().unwrap();
        assert_eq!(all.len(), 27);
        let additions = all.split_off(18);
        let reviewed_global: serde_json::Value =
            serde_json::from_str(include_str!("../../../tools/global-decode/family.json")).unwrap();
        assert_eq!(reviewed_global["candidates"], serde_json::json!(additions));
        assert_eq!(reviewed, legacy);
        // The additive family must have all source-defined specializations.
        assert_eq!(
            ALL_CANDIDATES[18..].len(),
            crate::attention_global_decode::DECODE_TILES.len()
        );
        for (candidate, tile) in ALL_CANDIDATES[18..]
            .iter()
            .zip(crate::attention_global_decode::DECODE_TILES)
        {
            assert_eq!(candidate.global_decode_tile(), Some(tile));
            assert_eq!(candidate.kernels().len(), 1);
            assert_eq!(
                candidate.kernels()[0].limits(),
                (tile.threads as usize, tile.threadgroup_bytes())
            );
        }
        for candidate in ALL_CANDIDATES {
            for dtype in [MetalFloatType::Bf16, MetalFloatType::F16] {
                let source = crate::kernels::kernel_source_with_options(
                    dtype,
                    MetalKernelOptions {
                        research: candidate,
                        ..MetalKernelOptions::default()
                    },
                );
                for kernel in candidate.spec().kernels {
                    let marker = format!("kernel void {}(", kernel.name());
                    assert_eq!(source.matches(&marker).count(), 1, "{marker}");
                }
            }
        }
        let off = MetalKernelOptions::default();
        assert_eq!(off.research, MetalResearchCandidate::Off);
        assert_eq!(
            crate::kernels::kernel_source_with_options(MetalFloatType::F16, off),
            crate::kernels::KERNEL_SOURCE
        );
    }

    #[test]
    fn all_candidates_reject_near_miss_model_identities() {
        use crate::research::Gemma12bResearchShape;
        let good = Gemma12bResearchShape {
            tokens: 84,
            hidden: 3840,
            intermediate: 15360,
            layers: 48,
            heads: 16,
            kv_heads: 8,
            head_dim: 256,
            attention_window: 1024,
            moe_experts: 0,
            moe_top_k: 0,
            moe_intermediate: 0,
            ple: 0,
        };
        for candidate in ALL_CANDIDATES {
            let good = Gemma12bResearchShape {
                tokens: candidate.spec().min_tokens,
                kv_heads: if candidate.global_decode_tile().is_some() {
                    1
                } else {
                    8
                },
                head_dim: if candidate.global_decode_tile().is_some() {
                    512
                } else {
                    256
                },
                attention_window: if candidate.global_decode_tile().is_some() {
                    0
                } else {
                    1024
                },
                ..good
            };
            assert_eq!(
                good.supports(candidate),
                candidate != MetalResearchCandidate::Off
            );
            for bad in [
                Gemma12bResearchShape {
                    hidden: 5376,
                    ..good
                },
                Gemma12bResearchShape { layers: 47, ..good },
                Gemma12bResearchShape {
                    head_dim: 128,
                    ..good
                },
                Gemma12bResearchShape {
                    moe_experts: 1,
                    ..good
                },
                Gemma12bResearchShape { tokens: 0, ..good },
                Gemma12bResearchShape {
                    tokens: 1025,
                    ..good
                },
            ] {
                assert!(!bad.supports(candidate));
            }
        }
    }
}
