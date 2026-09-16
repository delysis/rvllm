//! Bounded, opt-in qualification against an identified captured Metal prefill.
#![forbid(unsafe_code)]

use super::*;
use crate::ane_prefill::{
    AnePrefillSnapshot, PrefillLayerKv, PrefillLayerShape, PrefillScalarType,
};
use crate::gemma_ane_decode::AneWeightPlan;
use half::f16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn load_snapshot(path: &Path) -> (AnePrefillSnapshot, TokenId, String) {
    let bytes = std::fs::read(path).unwrap();
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let report: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(report["schema"], "rvllm.metal_ane_prefill_handoff.v2");
    assert_eq!(report["source_dtype"], "bfloat16");
    assert_eq!(report["destination_dtype"], "float16");
    assert_eq!(report["cache_capture_after_collection"], true);
    // This is intentionally one bounded fixture, not an arbitrary snapshot API.
    assert_eq!(report["ane_next_position"], 84);
    assert_eq!(report["prompt_token_ids"].as_array().unwrap().len(), 84);
    let abi_text = report["metal_numeric_abi"].as_str().unwrap();
    assert_eq!(abi_text.len(), 64);
    let mut abi = [0; 32];
    for (i, byte) in abi.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&abi_text[2 * i..2 * i + 2], 16).unwrap();
    }
    let records = report["layers"].as_array().unwrap();
    assert_eq!(records.len(), LAYERS);
    let directory = path.parent().unwrap();
    let mut layers = Vec::with_capacity(LAYERS);
    for (index, record) in records.iter().enumerate() {
        let global = (index + 1) % 6 == 0;
        let shape = PrefillLayerShape {
            query_heads: 16,
            kv_heads: if global { 1 } else { 8 },
            head_dim: if global { 512 } else { 256 },
            sliding_window: if global { None } else { Some(1024) },
        };
        assert_eq!(record["layer"], index);
        assert_eq!(record["kv_heads"], shape.kv_heads);
        assert_eq!(record["head_dim"], shape.head_dim);
        let elements = 84 * shape.kv_heads * shape.head_dim;
        assert_eq!(record["kv_elements_each"], elements);
        let read = |kind: &str| {
            let filename = format!("layer-{index:02}-{kind}.fp16");
            assert_eq!(record[format!("{kind}_file")], filename);
            let bytes = std::fs::read(directory.join(filename)).unwrap();
            assert_eq!(bytes.len(), 2 * elements);
            assert_eq!(
                record[format!("{kind}_sha256")],
                format!("{:x}", Sha256::digest(&bytes))
            );
            let values: Vec<_> = bytes
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();
            assert!(values.iter().all(|v| v.is_finite()));
            values
        };
        layers.push(PrefillLayerKv {
            shape,
            keys: read("key"),
            values: read("value"),
        });
    }
    let anchor = TokenId(u32::try_from(report["first_generated_token"].as_u64().unwrap()).unwrap());
    assert!((anchor.raw() as usize) < VOCAB);
    (
        AnePrefillSnapshot {
            tokens: 84,
            source_dtype: PrefillScalarType::Bf16,
            metal_numeric_abi: abi,
            layers,
        },
        anchor,
        hash,
    )
}

fn signature(token: &AneDecodedToken) -> Value {
    json!({"token":token.token.raw(), "position":token.position,
        "top_five_bits":token.top_five.iter().map(|&(id, value)| (id, value.to_bits())).collect::<Vec<_>>()})
}

fn record(file: &mut File, value: Value) {
    writeln!(file, "{value}").unwrap();
    file.flush().unwrap();
}

fn assert_poisoned(decoder: &mut GemmaAneDecode, anchor: TokenId) {
    assert!(decoder.next_position.is_none());
    assert!(
        matches!(decoder.decode(anchor), Err(error) if error.contains("complete prefill import"))
    );
}

#[test]
#[ignore = "Gemma 4 12B single-I/O reference transaction; 162 cache-only loads, no compilation, no timing"]
fn two_token_reference_accept_reject_and_poisoning() {
    assert!(std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some());
    let model = std::env::var("RVLLM_GEMMA4_MODEL_DIR").unwrap();
    let snapshot_path = std::env::var("RVLLM_TWO_TOKEN_SNAPSHOT").unwrap();
    let output = std::env::var("RVLLM_TWO_TOKEN_RECEIPT").unwrap();
    let mut receipt = File::create_new(output).unwrap();
    let (snapshot, anchor, snapshot_hash) = load_snapshot(Path::new(&snapshot_path));
    record(
        &mut receipt,
        json!({"stage":"fixture", "snapshot_report_sha256":snapshot_hash,
        "prompt_tokens":84,"weight_plan":"static-int8-ffn-cached","timing_claim":false}),
    );
    let mut decoder = GemmaAneDecode::load_with_weight_plan(
        Path::new(&model),
        1024,
        AneWeightPlan::StaticInt8FfnCached,
    )
    .unwrap();
    assert_eq!(rvllm_apple::ane_linear::compile_budget_used(), 0);
    decoder.import_prefill(&snapshot).unwrap();
    let first = decoder.decode(anchor).unwrap();
    let second = decoder.decode(first.token).unwrap();
    let third = decoder.decode(second.token).unwrap();
    let expected = [signature(&first), signature(&second), signature(&third)];
    record(
        &mut receipt,
        json!({"stage":"serial", "predictions":expected}),
    );

    decoder.import_prefill(&snapshot).unwrap();
    let pending = decoder
        .begin_two_token_reference(anchor, first.token)
        .unwrap();
    assert_eq!(signature(&pending.predictions()[0]), expected[0]);
    assert_eq!(signature(&pending.predictions()[1]), expected[1]);
    let accepted = pending.resolve(true).unwrap();
    assert!(accepted.draft_accepted);
    assert_eq!(accepted.next_position, 86);
    assert_eq!(
        accepted
            .predictions
            .iter()
            .map(signature)
            .collect::<Vec<_>>(),
        expected[..2]
    );
    assert_eq!(
        signature(&decoder.decode(second.token).unwrap()),
        expected[2]
    );
    record(
        &mut receipt,
        json!({"stage":"accepted-and-continuation", "passed":true}),
    );

    decoder.import_prefill(&snapshot).unwrap();
    let wrong = TokenId((first.token.raw() + 1) % VOCAB as u32);
    let rejected = decoder
        .begin_two_token_reference(anchor, wrong)
        .unwrap()
        .resolve(true)
        .unwrap();
    assert!(!rejected.draft_accepted);
    assert_eq!(rejected.next_position, 85);
    assert_eq!(
        rejected
            .predictions
            .iter()
            .map(signature)
            .collect::<Vec<_>>(),
        expected[..1]
    );
    assert_eq!(
        signature(&decoder.decode(first.token).unwrap()),
        expected[1]
    );
    assert_eq!(
        signature(&decoder.decode(second.token).unwrap()),
        expected[2]
    );
    record(
        &mut receipt,
        json!({"stage":"rejected-replaced-and-continuation", "passed":true}),
    );

    decoder.import_prefill(&snapshot).unwrap();
    let stopped = decoder
        .begin_two_token_reference(anchor, first.token)
        .unwrap()
        .resolve(false)
        .unwrap();
    assert!(!stopped.draft_accepted);
    assert_eq!(stopped.next_position, 85);
    assert_eq!(
        stopped
            .predictions
            .iter()
            .map(signature)
            .collect::<Vec<_>>(),
        expected[..1]
    );
    assert_eq!(
        signature(&decoder.decode(first.token).unwrap()),
        expected[1]
    );
    record(
        &mut receipt,
        json!({"stage":"one-output-boundary", "passed":true}),
    );

    decoder.import_prefill(&snapshot).unwrap();
    drop(
        decoder
            .begin_two_token_reference(anchor, first.token)
            .unwrap(),
    );
    assert_poisoned(&mut decoder, anchor);
    record(
        &mut receipt,
        json!({"stage":"unresolved-drop-poisoned", "passed":true}),
    );

    for fail_after in [1, LAYERS + 1] {
        decoder.import_prefill(&snapshot).unwrap();
        let mut observed = 0;
        let result = decoder.begin_reference_observed(anchor, first.token, &mut |_, _| {
            observed += 1;
            if observed == fail_after {
                Err("injected observer failure".into())
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(error) if error == "injected observer failure"));
        assert_eq!(observed, fail_after);
        assert_poisoned(&mut decoder, anchor);
        record(
            &mut receipt,
            json!({"stage":"execution-error-poisoned", "failed_after_layer_callbacks":fail_after,"passed":true}),
        );
    }

    decoder.import_prefill(&snapshot).unwrap();
    let pending = decoder.begin_two_token_reference(anchor, wrong).unwrap();
    let mut rolled_back = 0;
    let result = pending.resolve_with(true, |attention, retained| {
        if rolled_back == 1 {
            return Err("injected between-layer rollback failure".into());
        }
        attention.discard_last_unwrapped(retained)?;
        rolled_back += 1;
        Ok(())
    });
    assert!(matches!(result, Err(error) if error == "injected between-layer rollback failure"));
    assert_eq!(decoder.layers[0].attention.tokens_seen(), 85);
    assert!(decoder.layers[1..]
        .iter()
        .all(|layer| layer.attention.tokens_seen() == 86));
    assert_poisoned(&mut decoder, anchor);
    record(
        &mut receipt,
        json!({"stage":"partial-rollback-poisoned", "successful_layer_rollbacks":rolled_back,"passed":true}),
    );

    decoder.import_prefill(&snapshot).unwrap();
    assert_eq!(signature(&decoder.decode(anchor).unwrap()), expected[0]);
    assert_eq!(rvllm_apple::ane_linear::compile_budget_used(), 0);
    drop(decoder);
    record(
        &mut receipt,
        json!({"stage":"complete", "compiler_calls":0,
        "reimport_recovered":true,"timing_claim":false,
        "claim":"Reference S1 transaction at captured Metal position84; token and top-five bit parity, rejection/replacement, bounded host-injected failures. Driver journal must separately verify unloads. No batched arithmetic or speed claim."}),
    );
}
