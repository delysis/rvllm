use rvllm_bench::harness::{BenchHarness, BenchHarnessConfig, StubBenchHarness};
use rvllm_bench::schema::{
    BenchDeviceMetadata, BenchMetrics, BenchReport, BenchWorkload, BENCH_SCHEMA_VERSION,
};
use rvllm_core::{ConfigError, RvllmError};
use serde_json::json;

fn sample_device() -> BenchDeviceMetadata {
    BenchDeviceMetadata {
        backend: "apple".to_owned(),
        device_name: "Apple M4 Max".to_owned(),
        architecture: "arm64".to_owned(),
        accelerator: "metal+ane".to_owned(),
        memory_bytes: Some(128 * 1024 * 1024 * 1024),
        metadata: [
            ("gpu_family".to_owned(), json!("Apple9")),
            ("npu_generation".to_owned(), json!("M4")),
            ("ane_cores".to_owned(), json!(16)),
        ]
        .into_iter()
        .collect(),
    }
}

fn sample_workload() -> BenchWorkload {
    BenchWorkload {
        model: "gemma-4-E2B-it-Q8_0".to_owned(),
        batch: 8,
        prompt_tokens: 128,
        generated_tokens: 256,
        warmup_iters: 3,
        measured_iters: 10,
    }
}

fn sample_metrics() -> BenchMetrics {
    BenchMetrics {
        tok_per_sec: 42_000.0,
        watts: 28.5,
        joules_per_token: 0.000_678_571_4,
        p50_ms: 3.25,
        p95_ms: 5.75,
    }
}

#[test]
fn energy_bench_report_serializes_required_fields() {
    let report = BenchReport::new(sample_workload(), sample_metrics(), sample_device());

    let value = serde_json::to_value(&report).expect("serialize report");
    assert_eq!(value["schema_version"], json!(BENCH_SCHEMA_VERSION));
    assert_eq!(value["metrics"]["tok_per_sec"], json!(42_000.0));
    assert_eq!(value["metrics"]["watts"], json!(28.5));
    assert_eq!(value["metrics"]["joules_per_token"], json!(0.000_678_571_4));
    assert_eq!(value["metrics"]["p50_ms"], json!(3.25));
    assert_eq!(value["metrics"]["p95_ms"], json!(5.75));
    assert_eq!(value["device"]["device_name"], json!("Apple M4 Max"));
    assert_eq!(value["device"]["metadata"]["ane_cores"], json!(16));

    let round_trip = BenchReport::from_json_value(value).expect("deserialize report");
    assert_eq!(round_trip.metrics, sample_metrics());
    assert_eq!(round_trip.device, sample_device());
}

#[test]
fn energy_bench_report_rejects_missing_required_metric() {
    let value = json!({
        "schema_version": BENCH_SCHEMA_VERSION,
        "workload": sample_workload(),
        "metrics": {
            "tok_per_sec": 42_000.0,
            "watts": 28.5,
            "joules_per_token": 0.000_678_571_4,
            "p50_ms": 3.25
        },
        "device": sample_device()
    });

    let err = BenchReport::from_json_value(value).expect_err("missing p95_ms should fail");
    match err {
        RvllmError::Config {
            err: ConfigError::MissingField { name },
            field,
        } => {
            assert_eq!(name, "metrics.p95_ms");
            assert_eq!(field, "metrics.p95_ms");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn energy_bench_report_rejects_missing_device_metadata() {
    let mut device = serde_json::to_value(sample_device()).expect("serialize device");
    device
        .as_object_mut()
        .expect("device object")
        .remove("accelerator");

    let value = json!({
        "schema_version": BENCH_SCHEMA_VERSION,
        "workload": sample_workload(),
        "metrics": sample_metrics(),
        "device": device
    });

    let err = BenchReport::from_json_value(value).expect_err("missing accelerator should fail");
    match err {
        RvllmError::Config {
            err: ConfigError::MissingField { name },
            field,
        } => {
            assert_eq!(name, "device.accelerator");
            assert_eq!(field, "device.accelerator");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn stub_harness_emits_schema_report_from_explicit_metrics() {
    let config = BenchHarnessConfig {
        workload: sample_workload(),
        device: sample_device(),
    };
    let mut harness = StubBenchHarness::with_metrics(config, sample_metrics());

    let report = harness.run().expect("stub report");
    assert_eq!(report.schema_version, BENCH_SCHEMA_VERSION);
    assert_eq!(report.metrics.tok_per_sec, 42_000.0);
    assert_eq!(report.metrics.watts, 28.5);
    assert_eq!(report.metrics.p50_ms, 3.25);
    assert_eq!(report.metrics.p95_ms, 5.75);
}
