#![forbid(unsafe_code)]
use super::*;
use crate::research_projection::{FallbackReason, ProjectionRequest};

fn model(m: u32, d: u32) -> Gemma12bResearchShape {
    Gemma12bResearchShape {
        tokens: m,
        hidden: 3840,
        intermediate: 15360,
        layers: 48,
        heads: 16,
        kv_heads: if d == 256 { 8 } else { 1 },
        head_dim: d,
        attention_window: if d == 256 { 1024 } else { 0 },
        moe_experts: 0,
        moe_top_k: 0,
        moe_intermediate: 0,
        ple: 0,
    }
}
fn projection(c: MetalResearchCandidate, m: u32, n: u32, k: u32, fp32: bool) -> ProjectionRequest {
    let a = m as usize * k as usize * 2;
    let b = n as usize * k as usize * 2;
    ProjectionRequest {
        candidate: c,
        full_prefill: true,
        native_bf16: true,
        alpha: 1.0,
        beta: 0.0,
        shape: [m, n, k],
        output_f32: fp32,
        offsets: [0, a, a + b],
        arena_bytes: a + b + m as usize * n as usize * if fp32 { 4 } else { 2 },
    }
}
fn attention(c: MetalResearchCandidate, m: u32, d: u32) -> AttentionRequest {
    let model = model(m, d);
    let q = m as usize * 16 * d as usize * 2;
    let cache = 128 * 32 * model.kv_heads as usize * d as usize * 2;
    let offsets = [
        0,
        q,
        q + cache,
        q + 2 * cache,
        2 * q + 2 * cache,
        2 * q + 2 * cache + 512,
        2 * q + 2 * cache + 516,
        2 * q + 2 * cache + 524,
    ];
    AttentionRequest {
        candidate: c,
        model,
        native_bf16: true,
        apple9_or_10: true,
        prefill_single_sequence: true,
        trace: false,
        block_size: 32,
        max_blocks: 128,
        num_blocks: 128,
        scale: 1.0,
        offsets,
        arena_bytes: offsets[7] + m as usize * 4,
    }
}

#[test]
fn prefill_round_is_default_off_and_append_only() {
    assert_eq!(
        MetalResearchCandidate::default(),
        MetalResearchCandidate::Off
    );
    assert_eq!(crate::research_evidence::RESEARCH_KERNEL_COUNT, 102);
    assert_eq!(ResearchKernel::DonorSg4GlobalAttention as usize, 85);
    assert_eq!(ResearchKernel::PrefillControlGemm as usize, 86);
    assert_eq!(ResearchKernel::PrefillCombinedD512 as usize, 95);
    for c in FAMILY {
        assert_eq!(c.name().parse::<MetalResearchCandidate>(), Ok(c));
        assert!(c.kernels().iter().all(|k| (*k as usize) >= 86));
        assert!(c.global_decode_tile().is_none());
        assert!(c.split_global_decode_tile().is_none());
        assert!(!model(1, 256).supports(c));
    }
}
#[test]
fn prefill_round_accepts_all_six_roles_and_tail_rows_through_2048() {
    for c in FAMILY
        .into_iter()
        .filter(|c| projection_kernels(*c).is_some())
    {
        for m in [
            6, 17, 31, 32, 33, 64, 255, 256, 257, 512, 1024, 1025, 2047, 2048,
        ] {
            for (n, k, fp32) in [
                (8192, 3840, true),
                (9216, 3840, true),
                (30720, 3840, false),
                (3840, 4096, false),
                (3840, 8192, false),
                (3840, 15360, false),
            ] {
                let r = projection(c, m, n, k, fp32);
                let p = projection_plan(r).unwrap();
                assert_eq!((p.tile_m, p.tile_n), (32, 64));
                assert_eq!(p.kernel.owner(), c);
                if n != 3840 {
                    assert!(projection_plan(ProjectionRequest {
                        output_f32: !fp32,
                        ..r
                    })
                    .is_err());
                }
            }
        }
        for m in [0, 1, 5, 2049, u32::MAX] {
            assert!(projection_plan(projection(c, m, 8192, 3840, true)).is_err());
        }
    }
}
#[test]
fn prefill_round_projection_permissions_and_extent_checks_fail_closed() {
    let r = projection(
        MetalResearchCandidate::PrefillPipeline32x64,
        2048,
        9216,
        3840,
        true,
    );
    assert_eq!(
        projection_plan(ProjectionRequest {
            full_prefill: false,
            ..r
        }),
        Err(FallbackReason::ModelPhaseOrTrace)
    );
    assert_eq!(
        projection_plan(ProjectionRequest {
            native_bf16: false,
            ..r
        }),
        Err(FallbackReason::StoragePrecision)
    );
    for alpha in [0.0, 0.5, f32::NAN, f32::INFINITY] {
        assert_eq!(
            projection_plan(ProjectionRequest { alpha, ..r }),
            Err(FallbackReason::ProjectionScale)
        );
    }
    for beta in [1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            projection_plan(ProjectionRequest { beta, ..r }),
            Err(FallbackReason::ProjectionScale)
        );
    }
    assert_eq!(
        projection_plan(ProjectionRequest {
            offsets: [2, r.offsets[1], r.offsets[2]],
            ..r
        }),
        Err(FallbackReason::Alignment)
    );
    for offsets in [
        [0, r.offsets[1], 0],
        [0, r.offsets[1], r.offsets[2] + 2],
        [0, usize::MAX - 7, r.offsets[2]],
    ] {
        assert!(projection_plan(ProjectionRequest { offsets, ..r }).is_err());
    }
    assert!(projection_plan(ProjectionRequest {
        arena_bytes: r.arena_bytes - 1,
        ..r
    })
    .is_err());
    assert!(projection_plan(ProjectionRequest {
        shape: [2048, 9215, 3840],
        ..r
    })
    .is_err());
}
#[test]
fn prefill_round_rejects_near_miss_models_dtypes_and_devices() {
    let c = MetalResearchCandidate::PrefillPipeline32x64;
    let good = model(256, 256);
    assert!(projection_layer_supported(c, good, true, true));
    assert!(!projection_layer_supported(c, good, false, true));
    assert!(!projection_layer_supported(c, good, true, false));
    for bad in [
        Gemma12bResearchShape {
            hidden: 4096,
            ..good
        },
        Gemma12bResearchShape { layers: 47, ..good },
        Gemma12bResearchShape { heads: 32, ..good },
        Gemma12bResearchShape {
            kv_heads: 4,
            ..good
        },
        Gemma12bResearchShape {
            intermediate: 16384,
            ..good
        },
        Gemma12bResearchShape {
            moe_experts: 1,
            ..good
        },
        Gemma12bResearchShape {
            moe_top_k: 1,
            ..good
        },
        Gemma12bResearchShape {
            moe_intermediate: 1,
            ..good
        },
        Gemma12bResearchShape { ple: 1, ..good },
    ] {
        assert!(!projection_layer_supported(c, bad, true, true));
    }
}
#[test]
fn prefill_round_attention_geometry_permissions_and_aliases() {
    for c in [
        MetalResearchCandidate::PrefillQ4K16,
        MetalResearchCandidate::PrefillPipeline32x64Q4K16,
    ] {
        for d in [256, 512] {
            for m in RUNGS {
                let r = attention(c, m, d);
                let p = r.plan().unwrap();
                assert_eq!(p.grid, [m.div_ceil(4) as usize, 16, 1]);
                assert_eq!(
                    p.kernel.limits(),
                    (128, 16 * d as usize * 2 + 64 * 4 + 16 * 4)
                );
                for bad in [
                    AttentionRequest { trace: true, ..r },
                    AttentionRequest {
                        prefill_single_sequence: false,
                        ..r
                    },
                    AttentionRequest {
                        native_bf16: false,
                        ..r
                    },
                    AttentionRequest {
                        apple9_or_10: false,
                        ..r
                    },
                    AttentionRequest {
                        scale: f32::NAN,
                        ..r
                    },
                    AttentionRequest { block_size: 0, ..r },
                    AttentionRequest {
                        max_blocks: 129,
                        ..r
                    },
                    AttentionRequest { num_blocks: 0, ..r },
                    AttentionRequest {
                        arena_bytes: r.arena_bytes - 1,
                        ..r
                    },
                ] {
                    assert!(bad.plan().is_err());
                }
                let mut bad = r;
                bad.offsets[2] = bad.offsets[1];
                assert!(bad.plan().is_err());
                let mut bad = r;
                bad.offsets[3] = 0;
                assert!(bad.plan().is_err());
                let mut bad = r;
                bad.offsets[4] += 2;
                assert!(bad.plan().is_err());
                let mut bad = r;
                bad.model.attention_window = 512;
                assert!(bad.plan().is_err());
            }
        }
    }
}
#[test]
fn prefill_round_visibility_is_absolute_inclusive_and_hole_aware() {
    assert_eq!(visible_range(2047, 2050, 4096, 1024).unwrap(), 1024..2048);
    assert_eq!(visible_range(96, 400, 4096, 32).unwrap(), 65..97);
    assert_eq!(visible_range(31, 400, 4096, 0).unwrap(), 0..32);
    for (p, c) in [(-1, 40), (40, 40), (0, 0), (0, 4097)] {
        assert!(visible_range(p, c, 4096, 1024).is_err());
    }
    assert_eq!(physical_token(&[2, -1, 0], 32, 3, 31), Ok(Some(95)));
    assert_eq!(physical_token(&[2, -1, 0], 32, 3, 32), Ok(None));
    assert_eq!(physical_token(&[2, -1, 0], 32, 3, 64), Ok(Some(0)));
    assert!(physical_token(&[3], 32, 3, 0).is_err());
    assert!(physical_token(&[0], 0, 1, 0).is_err());
    assert!(physical_token(&[0], 32, 1, 32).is_err());
}
#[test]
fn prefill_round_vector_loads_and_matrix_stores_have_exact_coverage() {
    let mut a = vec![0u8; 32 * 32];
    let mut b = vec![0u8; 64 * 32];
    let mut c = vec![0u8; 32 * 64];
    for tid in 0..128 {
        for j in 0..2 {
            for x in 0..4 {
                a[(tid + j * 128) * 4 + x] += 1;
            }
        }
        for j in 0..4 {
            for x in 0..4 {
                b[(tid + j * 128) * 4 + x] += 1;
            }
        }
    }
    for sg in 0..4 {
        for i in 0..16 {
            for j in 0..32 {
                c[((sg / 2) * 16 + i) * 64 + (sg % 2) * 32 + j] += 1;
            }
        }
    }
    assert!(a.into_iter().chain(b).chain(c).all(|n| n == 1));
    assert_eq!(32 * 32 * 2 + 64 * 32 * 2, 6144);
    assert_eq!(32 * 64 * 4, 8192);
}
#[test]
fn prefill_round_prefetch_never_loads_a_panel_after_k() {
    for k in [3840, 4096, 8192, 15360] {
        let mut loads = vec![0u32; k / 32];
        loads[0] += 1;
        for kb in (0..k).step_by(32) {
            if kb + 32 < k {
                loads[(kb + 32) / 32] += 1;
            }
        }
        assert!(loads.iter().all(|&n| n == 1));
    }
}
#[test]
fn prefill_round_sources_keep_boundaries_and_live_metadata_guards() {
    let s = MetalResearchCandidate::PrefillPipeline32x64Q4K16.source();
    assert!(s.contains("device float *C [[buffer(2)]]"));
    assert!(s.contains("pr26_store(device half *C"));
    assert!(s.contains("kb + 32u < K"));
    assert!(s.contains("uint(page) < num_blocks"));
    assert!(s.contains("uint(positions[row]) + 1u"));
    assert!(s.contains("poisoned ? half(NAN) : f16_sat"));
    assert!(!s.contains("half weight"));
    assert!(!s.contains("half score"));
}
#[test]
fn prefill_round_prefill_coverage_cannot_be_decode_evidence() {
    let c = MetalResearchCandidate::PrefillPipeline32x64Q4K16;
    let counts = expected_prefill_dispatches(c).unwrap();
    assert_eq!(
        counts.iter().map(|p| p.1).collect::<Vec<_>>(),
        [48, 48, 96, 96, 40, 8]
    );
    assert!(expected_prefill_dispatches(MetalResearchCandidate::Donor12bSg8).is_none());
    assert_eq!(projection_flops(256, 30720, 3840), Some(60_397_977_600));
    assert_eq!(projection_flops(u32::MAX, u32::MAX, u32::MAX), None);
}

#[test]
fn prefill_round_raw_norm_preserves_fused_precision_and_checks_scratch() {
    let c = MetalResearchCandidate::PrefillPipeline32x64;
    let m = 256;
    let n = 3840;
    let raw = m as usize * n as usize * 4;
    let out = m as usize * n as usize * 2;
    let bytes = raw + out + n as usize * 2;
    let p = postnorm_plan(c, true, true, [0, raw, raw + out], m, n, 1e-6, bytes).unwrap();
    assert_eq!(p.kernel, ResearchKernel::PrefillPipelineNorm);
    assert!(p.binds_token_count);
    assert!(postnorm_plan(c, true, true, [0, 0, raw + out], m, n, 1e-6, bytes).is_err());
    assert!(postnorm_plan(c, true, true, [0, raw, 0], m, n, 1e-6, bytes).is_err());
    assert!(postnorm_plan(c, true, true, [0, raw, raw + out], m, n, 1e-5, bytes).is_err());
    assert!(postnorm_plan(c, true, true, [0, raw, raw + out], m, n, 1e-6, bytes - 1).is_err());
    let p = projection_plan(projection(c, m, n, 15360, true)).unwrap();
    assert_eq!(p.kernel, ResearchKernel::PrefillPipelineRaw);
    // A witness that rounding the projection BEFORE RMS is observably different.
    let x = [1.003f32, 0.203, 0.717];
    let round = |v: f32| half::bf16::from_f32(v).to_f32();
    let raw_rms = (x.iter().map(|v| v * v).sum::<f32>() / 3.0 + 1e-6).sqrt();
    let rounded = x.map(round);
    let rounded_rms = (rounded.iter().map(|v| v * v).sum::<f32>() / 3.0 + 1e-6).sqrt();
    assert!(x
        .iter()
        .zip(rounded)
        .any(|(&a, b)| round(a / raw_rms) != round(b / rounded_rms)));
}

#[test]
fn prefill_round_short_norm_retains_the_existing_bf16_boundary() {
    for k in [4096, 8192, 15360] {
        for m in [6, 17, 19, 20, 31, 32] {
            assert!(!raw_norm_route(m, 3840, k));
            assert!(raw_norm_projection_shape(m, 3840, k)); // component oracle only
        }
        for m in [33, 64, 256, 512, 1024, 2048] {
            assert!(raw_norm_route(m, 3840, k));
        }
    }
    assert!(!raw_norm_route(2049, 3840, 15360));
    assert!(!raw_norm_route(256, 8192, 3840));
}
