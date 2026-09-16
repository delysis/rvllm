//! Explicit full-checkpoint owner/cancellation qualification; never a host gate.
#![forbid(unsafe_code)]

use super::*;
use crate::request_api::{RequestHandle, TokenEvent};
use serde_json::{json, Value};
use std::io::Write;

fn reference(path: &str) -> (Vec<TokenId>, Vec<u32>) {
    let value: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let prompt: Vec<u32> = serde_json::from_value(value["prompt_token_ids"].clone()).unwrap();
    let expected: Vec<u32> = serde_json::from_value(value["generated_tokens"].clone()).unwrap();
    assert!(!prompt.is_empty() && expected.len() >= 2);
    (prompt.into_iter().map(TokenId).collect(), expected)
}

fn request(prompt: &[TokenId], count: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(prompt.to_vec(), u32::try_from(count).unwrap());
    request.cache_policy = CachePolicy::Disabled;
    request
}

fn finish(mut stream: RequestHandle, expected: &[u32]) -> Value {
    let mut generated = Vec::new();
    loop {
        match stream.recv().unwrap() {
            TokenEvent::Token { token_id, .. } => generated.push(token_id.raw()),
            TokenEvent::Finished {
                finish_reason,
                report,
                ..
            } => {
                assert_eq!(generated, expected);
                assert_eq!(report.selected_backend, BackendKind::MetalPrefillAneDecode);
                return json!({"generated_tokens":generated,"matches_reference":true,
                    "finish_reason":format!("{finish_reason:?}"),"measurement":report.measurement});
            }
        }
    }
}

#[test]
#[ignore = "Gemma 4 12B resident Metal/ANE A-B-A and cancel-then-fresh; strict zero compiles"]
fn full_checkpoint_successive_requests_and_cancellation() {
    let model = PathBuf::from(std::env::var("RVLLM_GEMMA_WORKER_MODEL").unwrap());
    let metallib = PathBuf::from(std::env::var("RVLLM_GEMMA_WORKER_METALLIB").unwrap());
    let output = PathBuf::from(std::env::var("RVLLM_GEMMA_WORKER_RECEIPT").unwrap());
    let mut receipt = std::fs::File::create_new(output).unwrap();
    let (a, expected_a) = reference(&std::env::var("RVLLM_GEMMA_WORKER_REFERENCE_A").unwrap());
    let (b, expected_b) = reference(&std::env::var("RVLLM_GEMMA_WORKER_REFERENCE_B").unwrap());
    assert_ne!(a, b);
    let config = AppleEngineConfig {
        backend_policy: BackendPolicy::MetalPrefillAneDecode,
        cache_policy: CachePolicy::Disabled,
        maximum_concurrency: 1,
        ingress_queue_capacity: 2,
        event_queue_capacity: 1,
        ..AppleEngineConfig::default()
    };
    let (engine, health) =
        spawn_gemma_disaggregated_worker(config, GemmaDisaggregatedConfig::new(model, metallib))
            .unwrap();
    assert_eq!(health.state(), GemmaWorkerState::Ready);
    let prepared = health.preparation().unwrap();
    assert_eq!(prepared.compiler_calls, 0);
    writeln!(receipt, "{}", json!({"stage":"prepared","measurement":prepared.measurement,"compiler_calls":prepared.compiler_calls})).unwrap();
    receipt.sync_all().unwrap();
    for (label, prompt, expected) in [
        ("A", &a, &expected_a),
        ("B", &b, &expected_b),
        ("A-repeat", &a, &expected_a),
    ] {
        let result = finish(
            engine.submit(request(prompt, expected.len())).unwrap(),
            expected,
        );
        writeln!(receipt, "{}", json!({"stage":label,"result":result})).unwrap();
        receipt.sync_all().unwrap();
    }

    let mut abandoned = engine.submit(request(&b, expected_b.len())).unwrap();
    // Observe Metal's token plus an actual ANE token before cancelling. The
    // bounded queue may then backpressure another token; dropping releases it.
    for expected in expected_b.iter().take(2) {
        assert!(
            matches!(abandoned.recv().unwrap(), TokenEvent::Token { token_id, .. } if token_id.raw() == *expected)
        );
    }
    let queued = engine.submit(request(&a, expected_a.len())).unwrap();
    drop(abandoned);
    let result = finish(queued, &expected_a);
    writeln!(
        receipt,
        "{}",
        json!({"stage":"cancel-B-then-queued-A","result":result})
    )
    .unwrap();
    receipt.sync_all().unwrap();
    let result = finish(
        engine.submit(request(&b, expected_b.len())).unwrap(),
        &expected_b,
    );
    writeln!(
        receipt,
        "{}",
        json!({"stage":"fresh-B-after-cancellation","result":result})
    )
    .unwrap();
    assert_eq!(health.state(), GemmaWorkerState::Ready);
    engine
        .handle_memory_pressure(MemoryPressure::Critical)
        .unwrap();
    assert_eq!(health.state(), GemmaWorkerState::ReleasedForMemoryPressure);
    assert_eq!(rvllm_apple::ane_linear::compile_budget_used(), 0);
    writeln!(
        receipt,
        "{}",
        json!({"stage":"released","compiler_calls":0,"state":"ReleasedForMemoryPressure"})
    )
    .unwrap();
    receipt.sync_all().unwrap();
}
