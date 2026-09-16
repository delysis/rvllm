//! Development-only load generator for the production continuous request path.

use rvllm_runtime::{BackendReport, CacheTier, EngineHandle, GenerateRequest, TokenEvent};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

const SCHEMA: &str = "rvllm.apple_continuous_bench.v1";

#[derive(Clone, Debug)]
pub struct ContinuousBenchConfig {
    pub request: GenerateRequest,
    pub batch: usize,
    pub iters: usize,
    pub warmup: usize,
}

#[derive(Debug)]
struct RequestObservation {
    ttft: Option<Duration>,
    inter_token: Vec<Duration>,
    token_received_at: Vec<Instant>,
    service_time: Duration,
    generated_tokens: usize,
    generated_token_ids: Vec<u32>,
    report: BackendReport,
}

/// Submit a complete cohort before receiving any events, then drain every
/// response concurrently. This exercises bounded ingress, admission,
/// scheduling, batching, and response backpressure through [`EngineHandle`].
pub fn run_engine_benchmark(
    engine: &EngineHandle,
    config: ContinuousBenchConfig,
) -> Result<Value, String> {
    validate(&config)?;

    for round in 0..config.warmup {
        run_round(engine, &config.request, config.batch)
            .map_err(|error| format!("warmup round {round}: {error}"))?;
    }

    let measured_start = Instant::now();
    let mut observations = Vec::with_capacity(config.batch.saturating_mul(config.iters));
    for round in 0..config.iters {
        observations.extend(
            run_round(engine, &config.request, config.batch)
                .map_err(|error| format!("measured round {round}: {error}"))?,
        );
    }
    let measured_elapsed = measured_start.elapsed();

    Ok(summarize(&config, &observations, measured_elapsed))
}

fn validate(config: &ContinuousBenchConfig) -> Result<(), String> {
    if config.batch == 0 {
        return Err("batch must be positive".to_owned());
    }
    if config.iters == 0 {
        return Err("iters must be positive".to_owned());
    }
    config
        .request
        .validate()
        .map_err(|error| format!("invalid benchmark request: {error}"))
}

fn run_round(
    engine: &EngineHandle,
    request: &GenerateRequest,
    batch: usize,
) -> Result<Vec<RequestObservation>, String> {
    let mut submitted = Vec::with_capacity(batch);
    for request_index in 0..batch {
        let submitted_at = Instant::now();
        let handle = engine
            .submit(request.clone())
            .map_err(|error| format!("submit request {request_index}: {error}"))?;
        submitted.push((submitted_at, handle));
    }

    std::thread::scope(|scope| {
        let joins = submitted
            .into_iter()
            .map(|(submitted_at, mut handle)| {
                scope.spawn(move || {
                    let mut first_token_at = None;
                    let mut previous_token_at = None;
                    let mut inter_token = Vec::new();
                    let mut token_received_at = Vec::new();
                    let mut generated_tokens = 0usize;
                    let mut generated_token_ids = Vec::new();
                    loop {
                        match handle
                            .recv()
                            .map_err(|error| format!("request {}: {error}", handle.id().raw()))?
                        {
                            TokenEvent::Token { token_id, .. } => {
                                let received_at = Instant::now();
                                first_token_at.get_or_insert(received_at);
                                if let Some(previous) = previous_token_at.replace(received_at) {
                                    inter_token
                                        .push(received_at.saturating_duration_since(previous));
                                }
                                token_received_at.push(received_at);
                                generated_tokens += 1;
                                generated_token_ids.push(token_id.raw());
                            }
                            TokenEvent::Finished { report, .. } => {
                                return Ok(RequestObservation {
                                    ttft: first_token_at
                                        .map(|first| first.saturating_duration_since(submitted_at)),
                                    inter_token,
                                    token_received_at,
                                    service_time: Instant::now()
                                        .saturating_duration_since(submitted_at),
                                    generated_tokens,
                                    generated_token_ids,
                                    report,
                                });
                            }
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        joins
            .into_iter()
            .map(|join| {
                join.join()
                    .map_err(|_| "benchmark receiver thread panicked".to_owned())?
            })
            .collect()
    })
}

fn summarize(
    config: &ContinuousBenchConfig,
    observations: &[RequestObservation],
    elapsed: Duration,
) -> Value {
    let ttft = observations
        .iter()
        .filter_map(|observation| observation.ttft)
        .collect::<Vec<_>>();
    let per_request_inter_token = observations
        .iter()
        .flat_map(|observation| observation.inter_token.iter().copied())
        .collect::<Vec<_>>();
    let inter_token = independent_cohort_intervals(observations, config.batch);
    let queue = observations
        .iter()
        .map(|observation| observation.report.queue_time)
        .collect::<Vec<_>>();
    let service = observations
        .iter()
        .map(|observation| observation.service_time)
        .collect::<Vec<_>>();
    let total_tokens = observations
        .iter()
        .map(|observation| observation.generated_tokens)
        .sum::<usize>();
    let aggregate_tok_per_sec = if elapsed.is_zero() {
        0.0
    } else {
        total_tokens as f64 / elapsed.as_secs_f64()
    };
    let ms_per_step = duration_ms(elapsed) / config.iters as f64;

    let mut cache_tiers = BTreeMap::<String, u64>::new();
    let mut reported_batch_sizes = BTreeMap::<String, u64>::new();
    let mut generated_sequences = BTreeMap::<String, u64>::new();
    for observation in observations {
        *cache_tiers
            .entry(cache_tier_label(observation.report.cache_tier).to_owned())
            .or_default() += 1;
        *reported_batch_sizes
            .entry(observation.report.batch_size.to_string())
            .or_default() += 1;
        let sequence = observation
            .generated_token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        *generated_sequences.entry(sequence).or_default() += 1;
    }
    let cache_tier = if cache_tiers.len() == 1 {
        cache_tiers
            .first_key_value()
            .map(|(tier, _)| tier.clone())
            .unwrap_or_else(|| "none".to_owned())
    } else {
        "mixed".to_owned()
    };

    json!({
        "schema": SCHEMA,
        "path": "engine_handle_continuous_worker",
        "batch": config.batch,
        "iters": config.iters,
        "warmup": config.warmup,
        "request_count": observations.len(),
        "prompt_tokens": config.request.prompt_tokens.len(),
        "max_new_tokens": config.request.max_output_tokens,
        "generated_tokens": total_tokens,
        "elapsed_ms": duration_ms(elapsed),
        "ms_per_step": ms_per_step,
        "aggregate_tok_per_sec": aggregate_tok_per_sec,
        "tok_per_sec": aggregate_tok_per_sec,
        "ttft_ms": percentile_summary(&ttft),
        "inter_token_latency_ms": percentile_summary(&inter_token),
        "per_request_inter_token_latency_ms": percentile_summary(&per_request_inter_token),
        "queue_time_ms": percentile_summary(&queue),
        "service_time_ms": percentile_summary(&service),
        "cache_tier": cache_tier,
        "cache_tier_histogram": cache_tiers,
        "batch_histogram": reported_batch_sizes,
        "generated_token_sequence_histogram": generated_sequences,
        "batch_histogram_semantics": "terminal_report_max_batch_size_per_request",
        "percentile_method": "nearest_rank",
        "limitations": [
            "event timestamps are host-observed at RequestHandle receipt, not GPU timestamps",
            "inter-token percentiles use one independent earliest-receipt timestamp per cohort step; per-request duplicated observations are reported separately",
            "batch_histogram counts each request's maximum reported batch size; per-step batch telemetry is not exposed",
            "aggregate throughput includes queue, prefill, decode, and host response-drain time",
        ],
    })
}

fn independent_cohort_intervals(
    observations: &[RequestObservation],
    batch: usize,
) -> Vec<Duration> {
    if batch == 0 {
        return Vec::new();
    }

    let mut intervals = Vec::new();
    for cohort in observations.chunks_exact(batch) {
        let common_tokens = cohort
            .iter()
            .map(|observation| observation.token_received_at.len())
            .min()
            .unwrap_or(0);
        let mut previous = None;
        for token_index in 0..common_tokens {
            let representative = cohort
                .iter()
                .map(|observation| observation.token_received_at[token_index])
                .min();
            if let Some(received_at) = representative {
                if let Some(previous_at) = previous.replace(received_at) {
                    intervals.push(received_at.saturating_duration_since(previous_at));
                }
            }
        }
    }
    intervals
}

fn percentile_summary(values: &[Duration]) -> Value {
    let mut millis = values.iter().copied().map(duration_ms).collect::<Vec<_>>();
    millis.sort_by(f64::total_cmp);
    let mut object = Map::new();
    object.insert("samples".to_owned(), json!(millis.len()));
    object.insert(
        "p50".to_owned(),
        optional_number(nearest_rank(&millis, 0.50)),
    );
    object.insert(
        "p95".to_owned(),
        optional_number(nearest_rank(&millis, 0.95)),
    );
    object.insert(
        "p99".to_owned(),
        optional_number(nearest_rank(&millis, 0.99)),
    );
    Value::Object(object)
}

fn nearest_rank(sorted: &[f64], percentile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    sorted.get(rank.saturating_sub(1)).copied()
}

fn optional_number(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn cache_tier_label(tier: CacheTier) -> &'static str {
    match tier {
        CacheTier::None => "none",
        CacheTier::Active => "active",
        CacheTier::Hot => "hot",
        CacheTier::Warm => "warm",
        CacheTier::Persistent => "persistent",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::{ReqId, TokenId};
    use rvllm_runtime::{
        ActiveRequest, AppleEngineConfig, BackendKind, ContinuousInferenceWorker,
        ContinuousStepOutput, FinishReason, GenerationOutcome, InferenceError, MemoryPressure,
    };
    use std::sync::atomic::AtomicBool;

    struct DeterministicContinuousWorker;

    impl ContinuousInferenceWorker for DeterministicContinuousWorker {
        fn admit(
            &mut self,
            _request_id: ReqId,
            _request: &GenerateRequest,
            _cancellation: &AtomicBool,
        ) -> Result<(), InferenceError> {
            Ok(())
        }

        fn step(
            &mut self,
            active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            let batch_size = u32::try_from(active.len()).unwrap_or(u32::MAX);
            let mut outputs = Vec::new();
            for request in active {
                if request.emitted_tokens < request.request.max_output_tokens {
                    outputs.push(ContinuousStepOutput::Token {
                        request_id: request.request_id,
                        token_id: TokenId(7),
                        text: None,
                    });
                } else {
                    let mut report = BackendReport::new(BackendKind::Metal);
                    report.batch_size = batch_size;
                    report.queue_time = request.queue_time;
                    outputs.push(ContinuousStepOutput::Finished {
                        request_id: request.request_id,
                        outcome: GenerationOutcome {
                            finish_reason: FinishReason::Length,
                            report,
                        },
                    });
                }
            }
            Ok(outputs)
        }

        fn abort(&mut self, _request_id: ReqId) {}

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn benchmark_uses_continuous_engine_and_emits_machine_readable_metrics() {
        let mut engine_config = AppleEngineConfig::default();
        engine_config.maximum_concurrency = 4;
        let engine = EngineHandle::spawn_continuous_with_factory(engine_config, |_| {
            Ok(DeterministicContinuousWorker)
        })
        .unwrap_or_else(|error| panic!("start continuous worker: {error}"));
        let record = run_engine_benchmark(
            &engine,
            ContinuousBenchConfig {
                request: GenerateRequest::new(vec![TokenId(1), TokenId(2)], 3),
                batch: 4,
                iters: 2,
                warmup: 1,
            },
        )
        .unwrap_or_else(|error| panic!("run benchmark: {error}"));

        assert_eq!(record["schema"], SCHEMA);
        assert_eq!(record["request_count"], 8);
        assert_eq!(record["generated_tokens"], 24);
        assert_eq!(record["ttft_ms"]["samples"], 8);
        assert_eq!(record["inter_token_latency_ms"]["samples"], 4);
        assert_eq!(record["per_request_inter_token_latency_ms"]["samples"], 16);
        assert_eq!(record["queue_time_ms"]["samples"], 8);
        assert!(record["aggregate_tok_per_sec"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(record["cache_tier_histogram"]["none"].as_u64().unwrap_or(0) == 8);
        assert!(record["batch_histogram"].is_object());
    }

    #[test]
    fn empty_percentiles_are_null_instead_of_synthetic_zeroes() {
        let summary = percentile_summary(&[]);
        assert_eq!(summary["samples"], 0);
        assert!(summary["p50"].is_null());
        assert!(summary["p95"].is_null());
        assert!(summary["p99"].is_null());
    }

    #[test]
    fn nearest_rank_is_deterministic_for_small_samples() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(nearest_rank(&values, 0.50), Some(2.0));
        assert_eq!(nearest_rank(&values, 0.95), Some(4.0));
        assert_eq!(nearest_rank(&values, 0.99), Some(4.0));
    }
}
