#![forbid(unsafe_code)]
use super::*;
use crate::prefill_round::{self as old, AttentionRequest};
use crate::research::Gemma12bResearchShape;
use crate::research_projection::ProjectionRequest;

fn model(tokens: u32, head_dim: u32) -> Gemma12bResearchShape {
    Gemma12bResearchShape {
        tokens,
        head_dim,
        hidden: 3840,
        intermediate: 15360,
        layers: 48,
        heads: 16,
        kv_heads: if head_dim == 256 { 8 } else { 1 },
        attention_window: if head_dim == 256 { 1024 } else { 0 },
        moe_experts: 0,
        moe_top_k: 0,
        moe_intermediate: 0,
        ple: 0,
    }
}
fn request(candidate: C, m: u32, n: u32, k: u32, fp32: bool) -> ProjectionRequest {
    let a = m as usize * k as usize * 2;
    let b = n as usize * k as usize * 2;
    ProjectionRequest {
        candidate,
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
fn attention(c: C, m: u32, d: u32) -> AttentionRequest {
    let model = model(m, d);
    let q = m as usize * 16 * d as usize * 2;
    let cache = 128 * 32 * model.kv_heads as usize * d as usize * 2;
    let end = 2 * q + 2 * cache;
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
        offsets: [
            0,
            q,
            q + cache,
            q + 2 * cache,
            end,
            end + 512,
            end + 516,
            end + 524,
        ],
        arena_bytes: end + 524 + m as usize * 4,
    }
}

#[test]
fn old_dispatch_prefix_and_defaults_are_unchanged() {
    assert_eq!(C::default(), C::Off);
    assert_eq!(K::PrefillControlGemm as usize, 86);
    assert_eq!(K::PrefillCombinedNorm as usize, 101);
    assert_eq!(K::Wide64Gemm as usize, 102);
    assert_eq!(crate::research_evidence::RESEARCH_KERNEL_COUNT, 118);
    for c in FAMILY {
        assert_eq!(c.name().parse(), Ok(c));
        assert!(c.kernels().iter().all(|k| *k as usize >= 102));
        assert!(!model(1, 256).supports(c));
    }
}
#[test]
fn fragment_coordinates_cover_each_element_once() {
    let mut count = [0; 64];
    for lane in 0..32 {
        let [r, c] = fragment_coord(lane).unwrap();
        for e in 0..2 {
            count[r * 8 + c + e] += 1;
        }
    }
    assert!(count.iter().all(|x| *x == 1));
    assert_eq!(fragment_coord(32), None);
}
#[test]
fn wide_padded_loads_and_direct_outputs_cover_tiles_without_aliases() {
    for (bm, bn, gm, gn) in [(64, 64, 2, 2), (32, 128, 1, 4)] {
        let mut loads = vec![0; (bm + bn) * 40];
        let mut output = vec![0; bm * bn];
        for tid in 0..128 {
            for j in 0..(bm + bn) / 16 {
                let v = tid + j * 128;
                for e in 0..4 {
                    loads[(v / 8) * 40 + (v % 8) * 4 + e] += 1;
                }
            }
        }
        for row in loads.chunks_exact(40) {
            assert!(row[..32].iter().all(|n| *n == 1));
            assert!(row[32..].iter().all(|n| *n == 0));
        }
        assert_eq!(gm * gn, 4);
        for sg in 0..4 {
            for lane in 0..32 {
                let [r, c] = fragment_coord(lane).unwrap();
                for i in 0..4 {
                    for j in 0..4 {
                        for e in 0..2 {
                            output[((sg / gn) * 32 + i * 8 + r) * bn
                                + (sg % gn) * 32
                                + j * 8
                                + c
                                + e] += 1;
                        }
                    }
                }
            }
        }
        assert!(output.iter().all(|n| *n == 1));
    }
}
#[test]
fn projections_use_requested_geometry_with_exact_role_storage() {
    for c in [C::PrefillWide64, C::PrefillWide128, C::PrefillWide64Mma8K32] {
        for m in [
            6, 17, 31, 32, 33, 63, 64, 65, 101, 255, 256, 257, 304, 512, 1024, 2048,
        ] {
            for (n, k, fp32) in [
                (8192, 3840, true),
                (9216, 3840, true),
                (30720, 3840, false),
                (3840, 4096, false),
                (3840, 8192, false),
                (3840, 15360, false),
                (3840, 4096, true),
                (3840, 8192, true),
                (3840, 15360, true),
            ] {
                let p = old::projection_plan(request(c, m, n, k, fp32)).unwrap();
                assert_eq!([p.tile_m, p.tile_n], projection_tile(c).unwrap());
                assert_eq!(p.kernel.owner(), c);
                assert_eq!(p.kernel.limits().0, 128);
            }
            assert!(old::projection_plan(request(c, m, 8192, 3840, false)).is_err());
            assert!(old::projection_plan(request(c, m, 30720, 3840, true)).is_err());
        }
        for m in [0, 1, 5, 2049, u32::MAX] {
            assert!(old::projection_plan(request(c, m, 8192, 3840, true)).is_err());
        }
    }
}
#[test]
fn projection_identity_scale_precision_and_extents_fail_closed() {
    let r = request(C::PrefillWide64, 256, 8192, 3840, true);
    for bad in [
        ProjectionRequest {
            full_prefill: false,
            ..r
        },
        ProjectionRequest {
            native_bf16: false,
            ..r
        },
        ProjectionRequest {
            alpha: f32::NAN,
            ..r
        },
        ProjectionRequest {
            beta: f32::INFINITY,
            ..r
        },
        ProjectionRequest {
            offsets: [2, r.offsets[1], r.offsets[2]],
            ..r
        },
        ProjectionRequest {
            offsets: [0, r.offsets[1], 0],
            ..r
        },
        ProjectionRequest {
            arena_bytes: r.arena_bytes - 1,
            ..r
        },
        ProjectionRequest {
            shape: [256, 8191, 3840],
            ..r
        },
    ] {
        assert!(old::projection_plan(bad).is_err());
    }
    for c in FAMILY {
        let good = model(256, 256);
        assert!(!Gemma12bResearchShape { layers: 47, ..good }.supports(c));
        assert!(!Gemma12bResearchShape {
            moe_experts: 1,
            ..good
        }
        .supports(c));
        assert!(!Gemma12bResearchShape {
            hidden: 4096,
            ..good
        }
        .supports(c));
    }
}
#[test]
fn new_attention_geometry_and_resources_are_bounded() {
    for c in [C::PrefillMma8K32, C::PrefillWide64Mma8K32] {
        for d in [256, 512] {
            for m in [6, 17, 33, 63, 65, 101, 256, 304, 512, 1024, 2048] {
                let r = attention(c, m, d);
                let p = r.plan().unwrap();
                assert_eq!(p.grid, [m.div_ceil(8) as usize, 16, 1]);
                assert_eq!(
                    p.kernel.limits(),
                    (
                        128,
                        8 * d as usize * 2 + 8 * 32 * 4 + 32 * 4 + 24 * 4 + 8 * 4 + 4 * 4
                    )
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
                    AttentionRequest {
                        max_blocks: 129,
                        ..r
                    },
                    AttentionRequest {
                        arena_bytes: r.arena_bytes - 1,
                        ..r
                    },
                ] {
                    assert!(bad.plan().is_err());
                }
                let mut bad = r;
                bad.offsets[0] = 2;
                assert!(bad.plan().is_err());
                let mut bad = r;
                bad.offsets[2] = bad.offsets[1];
                assert!(bad.plan().is_err());
                let mut bad = r;
                bad.offsets[3] = 0;
                assert!(bad.plan().is_err());
            }
        }
    }
}
#[test]
fn old_q4_geometry_and_short_bf16_norm_boundary_survive() {
    assert_eq!(
        attention(C::PrefillQ4K16, 101, 512).plan().unwrap().grid,
        [26, 16, 1]
    );
    for k in [4096, 8192, 15360] {
        assert!(!old::raw_norm_route(32, 3840, k));
        assert!(old::raw_norm_route(33, 3840, k));
    }
    for c in [C::PrefillWide64, C::PrefillWide128] {
        let p = old::postnorm_plan(c, true, true, [0, 15360, 23040], 1, 3840, 1e-6, 40000);
        assert!(p.is_err()); // M1 cannot make a decode norm entry.
    }
}
#[test]
fn matrix_attention_and_projection_have_prefill_only_distinct_ledgers() {
    let counts = old::expected_prefill_dispatches(C::PrefillWide64Mma8K32).unwrap();
    assert_eq!(
        counts.iter().map(|v| v.1).collect::<Vec<_>>(),
        [48, 48, 96, 96, 40, 8]
    );
    assert!(counts
        .iter()
        .all(|v| v.0.starts_with("research_prefill27_")));
    assert!(old::expected_prefill_dispatches(C::Donor12bSg8).is_none());
}
#[test]
fn sources_retain_fp32_probabilities_range_and_guard_contracts() {
    let s = C::PrefillWide64Mma8K32.source();
    assert_eq!(
        s.matches("kernel void pr27_fragment_layout_probe(").count(),
        1
    );
    assert!(s.contains("device float *C [[buffer(2)]]"));
    assert!(s.contains("simdgroup_float8x8 p;"));
    assert!(s.contains("uint(page) < num_blocks"));
    assert!(s.contains("stats[16u + rr]"));
    assert!(s.contains("scores[r * 32u + uint(lane)] != 0.0f"));
    assert!(s.contains("pr26_raw_norm"));
    assert!(!s.contains("half probability"));
    assert!(!s.contains("threadgroup float *ct"));
}
