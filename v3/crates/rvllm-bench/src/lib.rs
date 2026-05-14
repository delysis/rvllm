// rvllm-bench — scaffold only.
//   pub mod gates;    // regression gate output (JSON)
//   pub mod profile;  // nsys/ncu hooks

pub mod harness;
pub mod schema;

#[cfg(test)]
mod tests {
    use super::harness::{EnergyBenchHarness, HarnessMeasurement};
    use super::schema::{
        BenchDeviceMetadata, BenchEnergyMetrics, BenchHarnessStub, BenchLatencyStats, BenchReport,
        BenchRunMetrics,
    };
    use serde_json::Value;

    #[test]
    fn bench_report_json_has_required_energy_latency_and_device_fields() {
        let report = BenchReport::new(
            BenchHarnessStub::new("rvllm-bench", "stub-energy-v1"),
            BenchDeviceMetadata::new("apple", "Apple M4 Max", "metal+ane"),
            BenchRunMetrics::new(
                16,
                128,
                2048,
                4096.5,
                BenchEnergyMetrics::new(38.25, 0.00934),
                BenchLatencyStats::new(3.5, 5.75),
            ),
        );

        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["harness"]["name"], "rvllm-bench");
        assert_eq!(value["device"]["backend"], "apple");
        assert_eq!(value["device"]["name"], "Apple M4 Max");
        assert_eq!(value["run"]["tok_per_sec"], 4096.5);
        assert_eq!(value["run"]["energy"]["watts"], 38.25);
        assert_eq!(value["run"]["energy"]["joules_per_token"], 0.00934);
        assert_eq!(value["run"]["latency_ms"]["p50"], 3.5);
        assert_eq!(value["run"]["latency_ms"]["p95"], 5.75);

        let roundtrip: BenchReport = serde_json::from_value(value).unwrap();
        assert_eq!(roundtrip.run.generated_tokens, 2048);
        assert_eq!(roundtrip.device.kind, "metal+ane");
    }

    #[test]
    fn bench_report_rejects_missing_required_fields() {
        let required_paths = [
            (&["schema_version"][..], "schema_version"),
            (&["harness", "name"][..], "name"),
            (&["harness", "kind"][..], "kind"),
            (&["device", "backend"][..], "backend"),
            (&["device", "name"][..], "name"),
            (&["device", "kind"][..], "kind"),
            (&["run", "batch"][..], "batch"),
            (&["run", "iters"][..], "iters"),
            (&["run", "generated_tokens"][..], "generated_tokens"),
            (&["run", "tok_per_sec"][..], "tok_per_sec"),
            (&["run", "energy", "watts"][..], "watts"),
            (
                &["run", "energy", "joules_per_token"][..],
                "joules_per_token",
            ),
            (&["run", "latency_ms", "p50"][..], "p50"),
            (&["run", "latency_ms", "p95"][..], "p95"),
        ];

        for (path, expected) in required_paths {
            let mut value = complete_report_json();
            remove_path(&mut value, path);

            let err = serde_json::from_value::<BenchReport>(value).unwrap_err();
            assert!(
                err.to_string().contains(expected),
                "missing path {path:?} returned {err}"
            );
        }
    }

    #[test]
    fn harness_stub_builds_report_from_explicit_measurement() {
        let harness = EnergyBenchHarness::new(
            BenchHarnessStub::new("rvllm-bench", "stub-energy-v1"),
            BenchDeviceMetadata::new("cuda", "NVIDIA H100", "sm90"),
        );

        let report = harness
            .report_from_measurement(HarnessMeasurement::new(
                2,
                10,
                20,
                1_000_000_000,
                40.0,
                BenchLatencyStats::new(1.25, 2.5),
            ))
            .unwrap();

        assert_eq!(report.run.tok_per_sec, 20.0);
        assert_eq!(report.run.energy.joules_per_token, 2.0);
        assert_eq!(report.run.latency_ms.p95, 2.5);
    }

    fn complete_report_json() -> Value {
        serde_json::json!({
            "schema_version": 1,
            "harness": {
                "name": "rvllm-bench",
                "kind": "stub-energy-v1"
            },
            "device": {
                "backend": "cuda",
                "name": "NVIDIA H100",
                "kind": "sm90"
            },
            "run": {
                "batch": 1,
                "iters": 4,
                "generated_tokens": 4,
                "tok_per_sec": 512.0,
                "energy": {
                    "watts": 700.0,
                    "joules_per_token": 1.367
                },
                "latency_ms": {
                    "p50": 1.25,
                    "p95": 2.5
                }
            }
        })
    }

    fn remove_path(value: &mut Value, path: &[&str]) {
        let (last, parents) = path.split_last().unwrap();
        let mut cursor = value;
        for key in parents {
            cursor = cursor.get_mut(*key).unwrap();
        }
        cursor.as_object_mut().unwrap().remove(*last).unwrap();
    }
}
