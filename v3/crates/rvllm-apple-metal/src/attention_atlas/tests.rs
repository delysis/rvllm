use super::*;
use super::{experiment as e, plan::*, reference::*};
use std::collections::BTreeSet;

fn shape(local: bool, q: u32, s: u32) -> Shape {
    Shape {
        queries: q,
        live_keys: s,
        kv_heads: if local { 8 } else { 1 },
        dim: if local { 256 } else { 512 },
        window: if local { 1024 } else { 0 },
        page_size: 7,
        max_blocks: s.div_ceil(7) + 2,
        physical_blocks: s.div_ceil(7) + 2,
    }
}
fn plan(local: bool, q: u32, s: u32) -> Plan {
    Plan::new(
        catalog()[0],
        shape(local, q, s),
        CacheFormat::SeparateBf16,
        Output::F32,
    )
    .unwrap()
}
fn fixture(p: Plan, pattern: Pattern) -> Data {
    generate(
        p,
        &Fixture {
            seed: 9,
            first_position: p.shape.live_keys - p.shape.queries,
            pattern,
        },
        64 * 1024 * 1024,
    )
    .unwrap()
}

#[test]
fn atlas_abi_is_exactly_sixteen_words() {
    assert_eq!(std::mem::size_of::<Params>(), 64);
    assert_eq!(std::mem::align_of::<Params>(), 4);
}
#[test]
fn atlas_catalog_unique_bounded_and_fail_closed() {
    let cs = catalog();
    assert_eq!(cs.len(), 158);
    let mut names = BTreeSet::new();
    for c in cs {
        c.validate().unwrap();
        assert!(names.insert(c.name()));
        assert_eq!(parse_candidate(&c.name()).unwrap(), c);
        assert!(c.source_threadgroup_bytes() <= 32768);
    }
    assert!(parse_candidate("baseline").is_err());
    assert!(parse_candidate("auto").is_err());
}
#[test]
fn atlas_shape_rejects_generic_gemma_and_overflow() {
    for s in [
        Shape {
            kv_heads: 4,
            ..shape(false, 1, 1)
        },
        Shape {
            dim: 240,
            ..shape(true, 1, 1)
        },
        Shape {
            queries: 0,
            ..shape(true, 1, 1)
        },
        Shape {
            queries: 2,
            ..shape(false, 1, 1)
        },
        Shape {
            physical_blocks: 0,
            ..shape(false, 1, 1)
        },
        Shape {
            max_blocks: u32::MAX,
            ..shape(false, 1, 1)
        },
    ] {
        assert!(s.validate().is_err());
    }
}
#[test]
fn atlas_window_includes_self_and_exactly_1024_keys() {
    let s = shape(true, 1, 2050);
    for (p, start) in [(0, 0), (1023, 0), (1024, 1), (2048, 1025)] {
        assert_eq!(s.key_start(p), start);
    }
}
#[test]
fn atlas_packing_never_uses_packed_row_as_position() {
    for local in [true, false] {
        let s = shape(local, 7, 21);
        let mut seen = BTreeSet::new();
        for h in 0..s.kv_heads {
            for r in 0..s.queries * s.gqa() {
                let pair = packed_query_head(s, r, h).unwrap();
                assert!(seen.insert(pair));
            }
        }
        assert_eq!(seen.len(), 7 * 16);
        assert!(packed_query_head(s, 7 * s.gqa(), 0).is_none());
    }
}
#[test]
fn atlas_partitions_cover_without_overlap_even_with_empty_splits() {
    for n in [0, 1, 7, 31, 33, 129] {
        for splits in [1, 2, 4, 8, 16, 32] {
            let mut all = vec![];
            for i in 0..splits {
                all.extend(partition(3, 3 + n, i, splits).unwrap());
            }
            assert_eq!(all, (3..3 + n).collect::<Vec<_>>());
        }
    }
    assert!(partition(5, 3, 0, 1).is_none());
    assert!(partition(0, 7, 32, 32).is_none());
}
#[test]
fn atlas_split_prefill_and_raw_local_are_refused() {
    let c = catalog().into_iter().find(|c| c.splits == 4).unwrap();
    assert!(Plan::new(
        c,
        shape(false, 9, 21),
        CacheFormat::SeparateBf16,
        Output::Bf16
    )
    .is_err());
    assert!(Plan::new(
        c,
        shape(true, 1, 21),
        CacheFormat::BaseBf16FactorF32RopeV1,
        Output::Bf16
    )
    .is_err());
}
#[test]
fn atlas_metadata_accepts_holes_and_ignores_future_poison() {
    let s = shape(false, 2, 21);
    let mut pages = vec![i32::MAX; s.max_blocks as usize];
    pages[0] = -1;
    pages[1] = 0;
    assert_eq!(metadata_status(s, &[7, 8], &pages), 0);
    pages[1] = s.physical_blocks as i32;
    assert_eq!(metadata_status(s, &[7, 8], &pages), 3);
    assert_eq!(metadata_status(s, &[8, 7], &pages), 2);
    assert_eq!(metadata_status(s, &[-1, 0], &pages), 2);
}
#[test]
fn atlas_layout_guards_and_no_aliasing() {
    let p = plan(false, 1, 33);
    let l = Layout::new(p, 64 * 1024 * 1024).unwrap();
    l.validate(p, l.capacity).unwrap();
    let mut bad = l.clone();
    bad.spans[1] = bad.spans[0].clone();
    assert!(bad.validate(p, l.capacity).is_err());
    assert!(Layout::new(p, l.capacity - 1).is_err());
    assert!(!p.pso_fits(16, 128, 0, 32768));
    assert!(!p.pso_fits(32, 16, 0, 32768));
    assert!(!p.pso_fits(32, 128, 65536, 32768));
}
#[test]
fn atlas_bf16_round_to_even_not_truncation() {
    assert_eq!(round(f32::from_bits(0x3f808000)), 0x3f80);
    assert_eq!(round(f32::from_bits(0x3f818000)), 0x3f82);
    assert_eq!(round(f32::INFINITY), 0x7f80);
    assert_eq!(round(-0.0), 0x8000);
    assert!(widen(round(f32::NAN)).is_nan());
}
#[test]
fn atlas_holes_do_not_eagerly_cast_negative_pages() {
    let p = plan(false, 1, 9);
    let d = fixture(p, Pattern::AllHoles);
    assert_eq!(d.physical_token(p.shape, 0), None);
    let out = oracle(p, &d, 1_000_000).unwrap();
    assert!(out.iter().all(|v| *v == 0.0));
}
#[test]
fn atlas_zero_query_is_mean_of_visible_values() {
    for local in [true, false] {
        let p = plan(local, 2, 9);
        let d = fixture(p, Pattern::ZeroQueries);
        let out = oracle(p, &d, 10_000_000).unwrap();
        for t in 0..2 {
            let count = d.positions[t] as u32 + 1;
            for h in 0..16 {
                for dim in 0..p.shape.dim {
                    let mean = (0..count)
                        .map(|k| f64::from(widen(d.kv(p, k, h / p.shape.gqa(), dim).unwrap().1)))
                        .sum::<f64>()
                        / f64::from(count);
                    assert!(
                        (out[(t * 16 + h as usize) * p.shape.dim as usize + dim as usize] - mean)
                            .abs()
                            < 1e-12
                    );
                }
            }
        }
    }
}
#[test]
fn atlas_full_oracle_never_silently_samples() {
    let p = plan(false, 1, 9);
    let d = fixture(p, Pattern::Mixed);
    assert!(oracle(p, &d, 1).is_err());
}
#[test]
fn atlas_raw_abi_materializes_exact_same_logical_kv() {
    let mut p = plan(false, 1, 9);
    p.cache = CacheFormat::BaseBf16FactorF32RopeV1;
    let d = fixture(p, Pattern::Mixed);
    let m = d.materialized(p).unwrap();
    let mut mp = p;
    mp.cache = CacheFormat::SeparateBf16;
    let mut different = false;
    for t in 0..9 {
        for dim in 0..512 {
            let kv = d.kv(p, t, 0, dim).unwrap();
            assert_eq!(Some(kv), m.kv(mp, t, 0, dim));
            different |= kv.0 != kv.1;
        }
    }
    assert!(different, "final K must not alias V");
    assert_eq!(
        oracle(p, &d, 10_000_000).unwrap(),
        oracle(mp, &m, 10_000_000).unwrap()
    );
}
#[test]
fn atlas_ane_packing_mask_is_original_position_causal() {
    let p = plan(false, 2, 9);
    let d = fixture(p, Pattern::Mixed);
    let layout = ane::PackedLayout::new(p, 1_000_000).unwrap();
    let q = layout.pack_query(p, &d).unwrap();
    assert_eq!(q, d.q);
    let mask = layout.mask(p, &d).unwrap();
    for row in 0..16 {
        assert_eq!(mask[row * 9 + 8], f32::NEG_INFINITY);
    }
    for row in 16..32 {
        assert_eq!(mask[row * 9 + 8], 0.0);
    }
    assert!(!layout.execution_available);
    assert!(!layout.native_attention_admitted);
}
#[test]
fn atlas_ane_mask_budget_refuses_quadratic_allocation() {
    assert!(ane::PackedLayout::new(plan(false, 33, 2048), 100).is_err());
}
#[test]
fn atlas_predeclared_accuracy_rejects_nonfinite_and_empty() {
    let tol = Tolerance {
        max_abs: 1e-4,
        rel_l2: 1e-4,
    };
    assert!(compare(&[], &[], tol).is_err());
    assert!(compare(&[f32::NAN], &[0.0], tol).is_err());
    assert!(compare(&[1.0], &[1.0], tol).unwrap().passed);
    assert!(!compare(&[2.0], &[1.0], tol).unwrap().passed);
    assert!(Tolerance {
        max_abs: f64::NAN,
        ..tol
    }
    .validate()
    .is_err());
}
fn samples() -> Vec<e::Sample> {
    (0..8)
        .map(|i| e::Sample {
            index: i,
            arm: if i % 4 == 0 || i % 4 == 3 {
                "control".into()
            } else {
                "candidate".into()
            },
            host_ns: 1000,
            gpu_ns: if i % 4 == 0 || i % 4 == 3 { 100 } else { 80 },
            encoded_dispatches: 2,
            completed_dispatches: 2,
        })
        .collect()
}
#[test]
fn atlas_timing_rejects_reordering_drops_zero_and_drift() {
    let timing = e::Timing {
        warmups: 2,
        blocks: 2,
        max_drift_fraction: 0.05,
    };
    let mut s = samples();
    let score = e::score(&s, &timing).unwrap();
    assert!(score.drift_passed);
    assert_eq!(score.descriptive_ratio, 1.25);
    assert!(!score.promotion_eligible);
    s[1].arm = "control".into();
    assert!(e::score(&s, &timing).is_err());
    s = samples();
    s[0].gpu_ns = 0;
    assert!(e::score(&s, &timing).is_err());
    s = samples();
    assert!(e::score(&s[..7], &timing).is_err());
    s[0].gpu_ns = 150;
    assert!(!e::score(&s, &timing).unwrap().drift_passed);
}
#[test]
fn atlas_spec_duplicates_and_unknown_fields_refused() {
    assert!(
        serde_json::from_str::<Candidate>(r#"{"strategy":"vector","strategy":"matrix"}"#).is_err()
    );
    let mut c = serde_json::to_value(catalog()[0]).unwrap();
    c["waive_oracle"] = true.into();
    assert!(serde_json::from_value::<Candidate>(c).is_err());
}
#[test]
fn atlas_source_identity_varies_with_actual_candidate() {
    let a = source(catalog()[0], 512).unwrap();
    let b = source(catalog()[1], 512).unwrap();
    assert_ne!(sha256(a.as_bytes()), sha256(b.as_bytes()));
    assert!(a.contains("ATLAS_COOP_ENTRY(atlas_vector,512,1,1,64,32,true,1)"));
    assert!(a.contains("attention_prefill_simdgroup_f16"));
    assert!(COMMON.contains("state+2u+d"));
    assert!(MATRIX.contains("simdgroup_multiply_accumulate"));
}

#[test]
fn atlas_small_local_tiles_refuse_global_and_prefill() {
    let c = catalog()
        .into_iter()
        .find(|c| c.strategy == Strategy::Cooperative && c.rows == 2)
        .unwrap();
    assert!(Plan::new(c, shape(true, 1, 9), CacheFormat::SeparateBf16, Output::F32).is_ok());
    assert!(Plan::new(
        c,
        shape(false, 1, 9),
        CacheFormat::SeparateBf16,
        Output::F32
    )
    .is_err());
    assert!(Plan::new(
        c,
        shape(true, 33, 35),
        CacheFormat::SeparateBf16,
        Output::F32
    )
    .is_err());
}
