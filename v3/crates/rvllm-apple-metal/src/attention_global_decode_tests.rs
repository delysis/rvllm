use super::reference::{output_f32, output_f64, sampled_dots, Fixture};
use super::*;

#[test]
fn split32_scratch_is_explicit_and_default_footprint_is_unchanged() {
    assert_eq!(SPLIT_SCRATCH_BYTES, 526_336);
    assert_eq!(SPLIT32_SCRATCH_BYTES, 1_052_672);
    assert_eq!(
        model_scratch_bytes(crate::MetalResearchCandidate::Off),
        SPLIT_SCRATCH_BYTES
    );
    assert_eq!(
        model_scratch_bytes(crate::MetalResearchCandidate::GlobalD512SplitMmaR8K32S256T128),
        SPLIT_SCRATCH_BYTES
    );
    assert_eq!(
        model_scratch_bytes(crate::MetalResearchCandidate::GlobalD512SplitCoopKeyR8K8P64T128S32),
        SPLIT32_SCRATCH_BYTES
    );
}

fn shape() -> DecodeShape {
    Fixture::new(33, 32).shape
}
fn layout(plan: DecodePlan) -> (DecodeBuffers, usize) {
    let q = 32;
    let k = q + 16384 + 64;
    let v = k + plan.cache_bytes + 64;
    let output = v + plan.cache_bytes + 64;
    let block_tables = output + 32768 + 64;
    let context_lens = block_tables + plan.shape.max_blocks as usize * 4 + 64;
    let positions = context_lens + 64;
    (
        DecodeBuffers {
            q,
            k,
            v,
            output,
            block_tables,
            context_lens,
            positions,
        },
        positions + 128,
    )
}

#[test]
fn geometry_and_scratch_are_exact_for_all_tiles() {
    for tile in DECODE_TILES {
        let plan = DecodePlan::new(tile, shape(), DecodeOutput::Bf16).unwrap();
        assert_eq!(plan.grid, [(16 / tile.rows) as usize, 1, 1]);
        assert_eq!(plan.threads, [tile.threads as usize, 1, 1]);
        assert_eq!(plan.scratch_bytes, 0);
        let expected = if tile.simd_matrix {
            (4 * (tile.rows * tile.panel + tile.keys * tile.panel)
                + 8 * tile.rows * tile.keys
                + 12 * tile.rows
                + 4 * tile.keys) as usize
        } else {
            (tile.rows * DIM * 2
                + tile.keys * tile.panel * 2
                + 3 * tile.rows * tile.keys * 4
                + tile.keys * 4) as usize
        };
        assert_eq!(plan.threadgroup_bytes, expected);
        assert!(tile.output_floats_per_thread() <= 128);
        let (buffers, capacity) = layout(plan);
        assert!(plan.buffers_fit(buffers, capacity));
    }
    assert_eq!(std::mem::size_of::<DecodeParams>(), 40);
    assert_eq!(std::mem::align_of::<DecodeParams>(), 4);
}

#[test]
fn bounded_split_plan_has_exact_two_stage_geometry_and_disjoint_scratch() {
    let plan = SplitDecodePlan::new(SPLIT_R8S256T128, shape(), DecodeOutput::Bf16).unwrap();
    assert_eq!(plan.partial_grid, [2, 16, 1]);
    assert_eq!(plan.partial_threads, [128, 1, 1]);
    assert_eq!(plan.merge_grid, [16, 1, 1]);
    assert_eq!(plan.merge_threads, [32, 1, 1]);
    assert_eq!(plan.partial_threadgroup_bytes, 10016);
    assert_eq!(plan.merge_threadgroup_bytes, 0);
    assert_eq!(plan.partial_count, 16);
    assert_eq!(plan.scratch_bytes, 16 * 16 * 514 * 4);
    let matrix =
        SplitDecodePlan::new(SPLIT_MATRIX_R8K32S256T128, shape(), DecodeOutput::Bf16).unwrap();
    assert_eq!(matrix.partial_grid, plan.partial_grid);
    assert_eq!(matrix.merge_grid, plan.merge_grid);
    assert_eq!(matrix.scratch_bytes, plan.scratch_bytes);
    assert_eq!(matrix.partial_threadgroup_bytes, 12512);
    assert!(matrix.tile.simd_matrix);
    assert_eq!(matrix.tile.keys, 32);

    let unsplit = DecodePlan::new(DECODE_TILES[0], shape(), DecodeOutput::Bf16).unwrap();
    let (common, mut capacity) = layout(unsplit);
    capacity = capacity.next_multiple_of(16);
    let buffers = SplitDecodeBuffers {
        common,
        partials: capacity,
    };
    capacity += plan.scratch_bytes;
    assert!(plan.buffers_fit(buffers, capacity));
    assert!(!plan.buffers_fit(
        SplitDecodeBuffers {
            partials: buffers.common.output,
            ..buffers
        },
        capacity
    ));
    assert!(SplitDecodePlan::new(
        SPLIT_R8S256T128,
        DecodeShape {
            max_blocks: 129,
            block_size: 32,
            ..shape()
        },
        DecodeOutput::Bf16
    )
    .is_none());
}

#[test]
fn every_near_miss_shape_and_overflow_is_rejected() {
    let good = shape();
    for bad in [
        DecodeShape {
            sequences: 0,
            ..good
        },
        DecodeShape {
            sequences: 2,
            ..good
        },
        DecodeShape { heads: 8, ..good },
        DecodeShape {
            kv_heads: 2,
            ..good
        },
        DecodeShape {
            head_dim: 256,
            ..good
        },
        DecodeShape {
            window: 1024,
            ..good
        },
        DecodeShape {
            scale: 1.0 / 512.0_f32.sqrt(),
            ..good
        },
        DecodeShape {
            scale: f32::NAN,
            ..good
        },
        DecodeShape {
            scale: f32::INFINITY,
            ..good
        },
        DecodeShape {
            block_size: 0,
            ..good
        },
        DecodeShape {
            max_blocks: 0,
            ..good
        },
        DecodeShape {
            num_blocks: 0,
            ..good
        },
        DecodeShape {
            block_size: u32::MAX,
            max_blocks: 2,
            ..good
        },
        DecodeShape {
            block_size: 1,
            max_blocks: i32::MAX as u32 + 1,
            ..good
        },
    ] {
        for tile in DECODE_TILES {
            assert!(DecodePlan::new(tile, bad, DecodeOutput::Bf16).is_none());
        }
    }
    for tile in [
        DecodeTile {
            rows: 4,
            ..DECODE_TILES[0]
        },
        DecodeTile {
            panel: 256,
            ..DECODE_TILES[0]
        },
        DecodeTile {
            threads: 32,
            ..DECODE_TILES[0]
        },
        DecodeTile {
            rows: 16,
            threads: 32,
            ..DECODE_TILES[0]
        },
        DecodeTile {
            rows: 1,
            keys: 8,
            panel: 64,
            threads: 32,
            per_tile_softmax: false,
            simd_matrix: false,
        },
        DecodeTile {
            rows: 1,
            keys: 8,
            panel: 128,
            threads: 64,
            per_tile_softmax: false,
            simd_matrix: false,
        },
    ] {
        assert!(DecodePlan::new(tile, good, DecodeOutput::Bf16).is_none());
    }
}

#[test]
fn spans_alignment_aliasing_and_rejected_output_are_fail_closed() {
    for output in [DecodeOutput::Bf16, DecodeOutput::F32] {
        let plan = DecodePlan::new(DECODE_TILES[0], shape(), output).unwrap();
        let (good, capacity) = layout(plan);
        for bad in [
            DecodeBuffers {
                q: usize::MAX - 1,
                ..good
            },
            DecodeBuffers {
                output: usize::MAX - 3,
                ..good
            },
            DecodeBuffers {
                q: good.q + 1,
                ..good
            },
            DecodeBuffers {
                context_lens: good.context_lens + 2,
                ..good
            },
            DecodeBuffers {
                positions: capacity,
                ..good
            },
            DecodeBuffers {
                output: good.k,
                ..good
            },
            DecodeBuffers {
                output: good.q,
                ..good
            },
            DecodeBuffers {
                output: good.v,
                ..good
            },
            DecodeBuffers {
                output: good.block_tables,
                ..good
            },
            DecodeBuffers { v: good.k, ..good },
        ] {
            let mut sentinel = vec![0xa5_u8; 64];
            if plan.buffers_fit(bad, capacity) {
                sentinel.fill(0);
            }
            assert_eq!(sentinel, vec![0xa5; 64]);
        }
        assert!(!plan.buffers_fit(good, good.positions + 3));
    }
}

#[test]
fn packed_mapping_is_bijective_and_does_not_change_query_position() {
    for tile in DECODE_TILES {
        let mut heads = Vec::new();
        for group in 0..16 / tile.rows {
            for sg in 0..tile.threads / 32 {
                for row in 0..tile.rows / (tile.threads / 32) {
                    heads.push(packed_head(tile, group, sg, row).unwrap());
                }
            }
        }
        heads.sort_unstable();
        assert_eq!(heads, (0..16).collect::<Vec<_>>());
        assert!(packed_head(tile, 16 / tile.rows, 0, 0).is_none());
        assert!(packed_head(tile, 0, tile.threads / 32, 0).is_none());
    }
    let f = Fixture::new(33, 7);
    assert_eq!(
        visible_end(f.shape, &f.table, f.context, f.position),
        Some(33)
    );
    assert_eq!(visible_end(f.shape, &f.table, f.context, 5), Some(6));
    // A packed triangular row mask would incorrectly expose only 1..16 keys.
    assert!((1..=16).all(|wrong_end| wrong_end != 33));
}

#[test]
fn page_ownership_negative_ids_and_invalid_metadata_are_explicit() {
    let mut f = Fixture::new(33, 7);
    let base = physical_base(f.shape, &f.table, 8).unwrap();
    assert_eq!(base, (f.table[1] as usize * 7 + 1) * 512);
    f.table[1] = -123;
    assert_eq!(physical_base(f.shape, &f.table, 8), None);
    assert_eq!(
        visible_end(f.shape, &f.table, f.context, f.position),
        Some(33)
    );
    let p = DecodePlan::new(DECODE_TILES[0], f.shape, DecodeOutput::F32).unwrap();
    let result = output_f32(&f, p).unwrap();
    assert!(result.iter().all(|x| x.is_finite()));
    f.table[0] = f.shape.num_blocks as i32;
    assert!(visible_end(f.shape, &f.table, f.context, f.position).is_none());
    assert!(output_f32(&f, p).is_err());
    for (context, position) in [(0, 0), (33, -1), (33, 33), (i32::MAX, 0)] {
        assert!(visible_end(f.shape, &f.table, context, position).is_none());
    }
}

#[test]
fn scalar_schedule_matches_independent_dense_fp64_on_tails() {
    for length in [1, 7, 8, 9, 31, 32, 33, 257] {
        let mut f = Fixture::new(length, 7);
        if length > 8 {
            f.table[1] = -1;
        }
        for panel in [64, 128] {
            let plan = DecodePlan::new(
                DecodeTile {
                    panel,
                    ..DECODE_TILES[0]
                },
                f.shape,
                DecodeOutput::F32,
            )
            .unwrap();
            let got = output_f32(&f, plan).unwrap();
            let expected = output_f64(&f, plan).unwrap();
            for (&a, &b) in got.iter().zip(&expected) {
                assert!(
                    (a as f64 - b).abs() <= 2e-5,
                    "S={length}, panel={panel}: {a} vs {b}"
                );
            }
            assert!(!sampled_dots(&f, plan).unwrap().is_empty());
        }
    }
}

#[test]
fn empty_visible_history_is_zero_without_nonfinite_arithmetic() {
    let mut f = Fixture::new(9, 7);
    f.table.fill(-1);
    for tile in DECODE_TILES {
        let plan = DecodePlan::new(tile, f.shape, DecodeOutput::F32).unwrap();
        let out = output_f32(&f, plan).unwrap();
        assert!(out.iter().all(|x| x.to_bits() == 0));
    }
}

#[test]
fn restored_prefix_and_speculative_rollback_are_read_only_and_repeatable() {
    let mut f = Fixture::new(65, 7);
    f.position = 30;
    let plan = DecodePlan::new(DECODE_TILES[0], f.shape, DecodeOutput::F32).unwrap();
    let before = output_f32(&f, plan).unwrap();
    for t in 31..65 {
        let base = physical_base(f.shape, &f.table, t).unwrap();
        f.k[base..base + 512].fill(0x7fc1);
        f.v[base..base + 512].fill(0x7fc1);
    }
    let tables = f.table.clone();
    assert_eq!(before, output_f32(&f, plan).unwrap());
    // Roll back logical length without reclaiming shared prefix pages.
    f.context = 31;
    assert_eq!(before, output_f32(&f, plan).unwrap());
    assert_eq!(f.table, tables);
    assert_eq!(before, output_f32(&f.clone(), plan).unwrap());
}

#[test]
fn panel_softmax_kv_alias_and_wrong_scale_controls_are_detected() {
    let mut f = Fixture::new(2, 7);
    f.q.fill(round_bf16(1.0));
    f.k.fill(0);
    f.v.fill(0);
    let a = physical_base(f.shape, &f.table, 0).unwrap();
    let b = physical_base(f.shape, &f.table, 1).unwrap();
    f.k[a] = round_bf16(8.0);
    f.k[b + 511] = round_bf16(4.0);
    f.v[b..b + 512].fill(round_bf16(1.0));
    let plan = DecodePlan::new(DECODE_TILES[0], f.shape, DecodeOutput::F32).unwrap();
    let right = output_f32(&f, plan).unwrap()[0];
    let expected = 1.0_f32 / (1.0 + 4.0_f32.exp());
    assert!((right - expected).abs() < 1e-7);
    let per_panel_wrong = 0.5 * (1.0 / (1.0 + 8.0_f32.exp()) + 1.0 / (1.0 + (-4.0_f32).exp()));
    assert!((right - per_panel_wrong).abs() > 0.4);
    let mut aliased = f.clone();
    aliased.v.clone_from(&f.k);
    assert!((output_f32(&aliased, plan).unwrap()[0] - right).abs() > 1.0);
    let mut wrong = f.shape;
    wrong.scale = 1.0 / 512.0_f32.sqrt();
    assert!(DecodePlan::new(DECODE_TILES[0], wrong, DecodeOutput::Bf16).is_none());
}

#[test]
fn bf16_is_rounded_once_with_even_ties_and_preserves_sign() {
    for (x, want) in [
        (1.0 + 2.0_f32.powi(-8), 0x3f80),
        (1.0 + 3.0 * 2.0_f32.powi(-8), 0x3f82),
        (-0.0, 0x8000),
        (0.0, 0),
    ] {
        assert_eq!(round_bf16(x), want);
    }
    assert!(widen_bf16(round_bf16(f32::NAN)).is_nan());
    let f = Fixture::new(9, 7);
    let plan = DecodePlan::new(DECODE_TILES[0], f.shape, DecodeOutput::F32).unwrap();
    let out = output_f32(&f, plan).unwrap();
    let bf16: Vec<_> = out.iter().copied().map(round_bf16).collect();
    assert_eq!(bf16.len(), (HEADS * DIM) as usize);
    assert!(out.iter().zip(bf16).all(|(&x, b)| round_bf16(x) == b));
}

#[test]
fn nonfinite_live_inputs_fail_but_poisoned_padding_does_not() {
    let mut f = Fixture::new(9, 7);
    let plan = DecodePlan::new(DECODE_TILES[0], f.shape, DecodeOutput::F32).unwrap();
    assert!(output_f32(&f, plan).is_ok());
    f.q[0] = 0x7fc1;
    assert!(output_f32(&f, plan).is_err());
}

#[test]
fn staging_panel_size_does_not_change_fp32_association() {
    let f = Fixture::new(33, 7);
    let first = DecodePlan::new(DECODE_TILES[0], f.shape, DecodeOutput::F32).unwrap();
    let expected = output_f32(&f, first).unwrap();
    for tile in DECODE_TILES {
        let plan = DecodePlan::new(tile, f.shape, DecodeOutput::F32).unwrap();
        let actual = output_f32(&f, plan).unwrap();
        assert!(actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}
