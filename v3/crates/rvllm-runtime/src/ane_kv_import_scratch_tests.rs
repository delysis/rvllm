//! Opt-in byte qualification, host packing measurements and device continuation.
#![forbid(unsafe_code)]

use super::live_tests::{load_snapshot, record, signature};
use super::*;
use crate::ane_prefill::{AnePrefillSnapshot, PrefillLayerShape};
use crate::apple_measurement::PowerMonitor;
use crate::gemma_ane_decode::AneWeightPlan;
use rvllm_apple::ane_attention_layout::PackedAttentionLayout;
use serde_json::json;
use std::fs::File;
use std::hint::black_box;
use std::path::Path;

fn layout(shape: PrefillLayerShape) -> PackedAttentionLayout {
    match shape.sliding_window {
        Some(window) => PackedAttentionLayout::sliding(
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            window,
        ),
        None => PackedAttentionLayout::new(shape.query_heads, shape.kv_heads, shape.head_dim, 1024),
    }
    .unwrap()
}

fn pack_request(snapshot: &AnePrefillSnapshot, reuse: bool) {
    // Match the candidate decoder's ownership: one scratch vector per complete
    // import, reused between its layers, then dropped before the next request.
    let mut scratch = Vec::new();
    for layer in &snapshot.layers {
        let layout = layout(layer.shape);
        if reuse {
            if scratch.len() < layout.input_bytes() {
                scratch.resize(layout.input_bytes(), 0);
            }
            let packed = &mut scratch[..layout.input_bytes()];
            layout
                .import_cache_into(&layer.keys, &layer.values, snapshot.tokens, packed)
                .unwrap();
            black_box(packed);
        } else {
            let packed = layout
                .import_cache(&layer.keys, &layer.values, snapshot.tokens)
                .unwrap();
            black_box(packed.as_slice());
        }
    }
}

#[test]
#[ignore = "hash-checked captured KV; CPU only, optional bounded packing measurements"]
fn kv_import_scratch_captured_bytes_and_host_measurement() {
    assert!(std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_none());
    let snapshot_path = std::env::var("RVLLM_TWO_TOKEN_SNAPSHOT").unwrap();
    let output = std::env::var("RVLLM_KV_SCRATCH_RECEIPT").unwrap();
    let mut receipt = File::create_new(output).unwrap();
    let (snapshot, _, snapshot_hash) = load_snapshot(Path::new(&snapshot_path));
    let mut scratch = Vec::new();
    let mut bytes_compared = 0_usize;
    let mut growths = 0;
    for layer in &snapshot.layers {
        let layout = layout(layer.shape);
        let expected = layout
            .import_cache(&layer.keys, &layer.values, snapshot.tokens)
            .unwrap();
        if scratch.len() < layout.input_bytes() {
            scratch.resize(layout.input_bytes(), 0);
            growths += 1;
        }
        scratch.fill(0xa5);
        let packed = &mut scratch[..layout.input_bytes()];
        layout
            .import_cache_into(&layer.keys, &layer.values, snapshot.tokens, packed)
            .unwrap();
        assert_eq!(packed, expected);
        bytes_compared += packed.len();
    }
    assert_eq!(growths, 1);
    record(
        &mut receipt,
        json!({"stage":"qualification","snapshot_report_sha256":snapshot_hash,
        "layers":48,"tokens":84,"bytes_compared":bytes_compared,"all_bytes_identical":true,
        "scratch_growths":growths,"maximum_scratch_bytes":scratch.len(),"device_calls":0}),
    );
    drop(scratch);
    if std::env::var("RVLLM_KV_SCRATCH_TIMING").as_deref() == Ok("true") {
        let monitor = PowerMonitor::start(None).unwrap();
        // Fixed, unmeasured warmups for both variants.
        pack_request(&snapshot, false);
        pack_request(&snapshot, true);
        for block in 0..3 {
            let order = if block == 1 {
                [true, false, false, true]
            } else {
                [false, true, true, false]
            };
            for (slot, reuse) in order.into_iter().enumerate() {
                let phase = monitor.begin();
                for _ in 0..4 {
                    pack_request(&snapshot, reuse);
                }
                let measurement = phase.finish(4 * LAYERS);
                record(
                    &mut receipt,
                    json!({"stage":"timing","block":block,"slot":slot,
                    "reuse_scratch":reuse,"requests":4,"layers_per_request":48,
                    "measurement":measurement,"device_calls":0,
                    "claim":"Raw CPU packing measurement; power, contention and repeat drift require independent analysis. Excludes surface copy and complete handoff."}),
                );
            }
        }
    }
    assert_eq!(rvllm_apple::ane_linear::compile_budget_used(), 0);
    record(
        &mut receipt,
        json!({"stage":"complete","compiler_calls":0,"device_calls":0}),
    );
}

#[test]
#[ignore = "Gemma 4 12B cached ANE import/continuation qualification; no new graphs or timing"]
fn kv_import_scratch_device_continuation() {
    assert!(std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some());
    let model = std::env::var("RVLLM_GEMMA4_MODEL_DIR").unwrap();
    let snapshot_path = std::env::var("RVLLM_TWO_TOKEN_SNAPSHOT").unwrap();
    let output = std::env::var("RVLLM_KV_SCRATCH_RECEIPT").unwrap();
    let mut receipt = File::create_new(output).unwrap();
    let (mut snapshot, anchor, snapshot_hash) = load_snapshot(Path::new(&snapshot_path));
    record(
        &mut receipt,
        json!({"stage":"fixture","snapshot_report_sha256":snapshot_hash,"timing_claim":false}),
    );
    let mut decoder = GemmaAneDecode::load_with_weight_plan(
        Path::new(&model),
        1024,
        AneWeightPlan::StaticInt8FfnCached,
    )
    .unwrap();
    for tokens in [84, 21] {
        snapshot.tokens = tokens;
        for layer in &mut snapshot.layers {
            let count = tokens * layer.shape.kv_heads * layer.shape.head_dim;
            layer.keys.truncate(count);
            layer.values.truncate(count);
        }
        decoder.import_prefill(&snapshot).unwrap();
        let mut token = anchor;
        let mut expected = Vec::new();
        for _ in 0..3 {
            let output = decoder.decode(token).unwrap();
            token = output.token;
            expected.push(signature(&output));
        }
        decoder.import_prefill_reusing_scratch(&snapshot).unwrap();
        token = anchor;
        for expected in &expected {
            let output = decoder.decode(token).unwrap();
            assert_eq!(signature(&output), *expected);
            token = output.token;
        }
        // Invalid complete snapshots must be rejected before any state change.
        let last = snapshot.layers[0].keys.pop().unwrap();
        assert!(decoder.import_prefill_reusing_scratch(&snapshot).is_err());
        assert_eq!(decoder.next_position, Some(tokens + 3));
        assert!(decoder
            .layers
            .iter()
            .all(|layer| layer.attention.tokens_seen() == tokens + 3));
        snapshot.layers[0].keys.push(last);
        record(
            &mut receipt,
            json!({"stage":"continuation","tokens":tokens,
            "predictions":expected,"candidate_bit_identical":true,"invalid_snapshot_preserved_frontiers":true}),
        );
    }
    assert_eq!(rvllm_apple::ane_linear::compile_budget_used(), 0);
    drop(decoder);
    record(
        &mut receipt,
        json!({"stage":"complete","compiler_calls":0,"timing_claim":false}),
    );
}
