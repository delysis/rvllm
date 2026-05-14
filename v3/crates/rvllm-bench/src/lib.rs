// rvllm-bench - scaffold only.
//   pub mod gates;    // regression gate output (JSON)
//   pub mod profile;  // nsys/ncu hooks

pub mod harness;
pub mod schema;

use rvllm_core::{AppleCtx, AppleError, ConfigError, Result, RvllmError};

#[cfg(feature = "apple")]
use rvllm_runtime::AppleBackendMode;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchConfig {
    pub apple: Option<AppleBenchConfig>,
}

#[cfg(feature = "apple")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleBenchConfig {
    pub mode: AppleBackendMode,
    pub rollout_tokens: Option<u32>,
    pub private_ane_opt_in: bool,
}

#[cfg(not(feature = "apple"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleBenchConfig {
    _private: (),
}

pub fn parse_bench_args<I, S>(args: I) -> Result<BenchConfig>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut parser = ArgParser::new(args);
    let mut apple = AppleParseState::new();

    while let Some(arg) = parser.next() {
        match split_arg(&arg) {
            ("--apple-mode", Some(value)) => apple.set_mode(value)?,
            ("--apple-mode", None) => apple.set_mode(&parser.value("--apple-mode")?)?,
            ("--apple-rollout-tokens", Some(value)) => apple.set_rollout_tokens(value)?,
            ("--apple-rollout-tokens", None) => {
                apple.set_rollout_tokens(&parser.value("--apple-rollout-tokens")?)?;
            }
            ("--apple-private-ane-opt-in", None) => apple.set_private_ane_opt_in()?,
            ("--apple-private-ane-opt-in", Some(_)) => {
                return Err(invalid(
                    "--apple-private-ane-opt-in",
                    "flag does not take a value",
                ));
            }
            (flag, _) if flag.starts_with("--apple-") => return Err(unknown_apple_flag(flag)),
            (flag, _) => return Err(invalid("cli", format!("unknown bench flag {flag:?}"))),
        }
    }

    Ok(BenchConfig {
        apple: apple.finish()?,
    })
}

pub fn apple_execution_unavailable() -> RvllmError {
    RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "rvllm-bench",
            op: "bench",
        },
        apple_ctx("bench"),
    )
}

struct ArgParser {
    args: std::vec::IntoIter<String>,
}

impl ArgParser {
    fn new<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut args: Vec<String> = args.into_iter().map(Into::into).collect();
        if !args.is_empty() {
            args.remove(0);
        }
        Self {
            args: args.into_iter(),
        }
    }

    fn next(&mut self) -> Option<String> {
        self.args.next()
    }

    fn value(&mut self, flag: &'static str) -> Result<String> {
        match self.args.next() {
            Some(value) if !value.starts_with("--") => Ok(value),
            Some(value) => Err(invalid(flag, format!("expected value, got flag {value:?}"))),
            None => Err(invalid(flag, "missing value")),
        }
    }
}

fn split_arg(arg: &str) -> (&str, Option<&str>) {
    match arg.split_once('=') {
        Some((flag, value)) => (flag, Some(value)),
        None => (arg, None),
    }
}

#[cfg(feature = "apple")]
fn parse_u32(flag: &'static str, value: &str) -> Result<u32> {
    value
        .parse()
        .map_err(|_| invalid(flag, format!("expected u32, got {value:?}")))
}

fn invalid(field: &'static str, reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.into(),
        },
        field,
    )
}

fn apple_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "rvllm-bench",
        op,
        device: "apple-silicon",
    }
}

#[cfg(not(feature = "apple"))]
fn apple_feature_unavailable(op: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "rvllm-apple",
            op,
        },
        apple_ctx(op),
    )
}

#[cfg(feature = "apple")]
fn unknown_apple_flag(flag: &str) -> RvllmError {
    invalid("cli", format!("unknown Apple bench flag {flag:?}"))
}

#[cfg(not(feature = "apple"))]
fn unknown_apple_flag(_flag: &str) -> RvllmError {
    apple_feature_unavailable("cli")
}

#[cfg(feature = "apple")]
struct AppleParseState {
    mode: Option<AppleBackendMode>,
    rollout_tokens: Option<u32>,
    private_ane_opt_in: bool,
}

#[cfg(feature = "apple")]
impl AppleParseState {
    const fn new() -> Self {
        Self {
            mode: None,
            rollout_tokens: None,
            private_ane_opt_in: false,
        }
    }

    fn set_mode(&mut self, value: &str) -> Result<()> {
        self.mode = Some(AppleBackendMode::from_flag(value).ok_or_else(|| {
            invalid(
                "--apple-mode",
                format!("unknown Apple backend mode {value:?}"),
            )
        })?);
        Ok(())
    }

    fn set_rollout_tokens(&mut self, value: &str) -> Result<()> {
        let tokens = parse_u32("--apple-rollout-tokens", value)?;
        if tokens == 0 {
            return Err(invalid(
                "--apple-rollout-tokens",
                "must be greater than zero",
            ));
        }
        self.rollout_tokens = Some(tokens);
        Ok(())
    }

    fn set_private_ane_opt_in(&mut self) -> Result<()> {
        self.private_ane_opt_in = true;
        Ok(())
    }

    fn finish(self) -> Result<Option<AppleBenchConfig>> {
        match self.mode {
            None if self.rollout_tokens.is_none() && !self.private_ane_opt_in => Ok(None),
            None => Err(invalid("--apple-mode", "required when Apple flags are set")),
            Some(mode) if mode.requires_private_ane() && self.rollout_tokens.is_none() => {
                Err(invalid(
                    "--apple-rollout-tokens",
                    "required for private ANE Apple modes",
                ))
            }
            Some(mode) => Ok(Some(AppleBenchConfig {
                mode,
                rollout_tokens: self.rollout_tokens,
                private_ane_opt_in: self.private_ane_opt_in,
            })),
        }
    }
}

#[cfg(not(feature = "apple"))]
struct AppleParseState;

#[cfg(not(feature = "apple"))]
impl AppleParseState {
    const fn new() -> Self {
        Self
    }

    fn set_mode(&mut self, _value: &str) -> Result<()> {
        Err(apple_feature_unavailable("cli"))
    }

    fn set_rollout_tokens(&mut self, _value: &str) -> Result<()> {
        Err(apple_feature_unavailable("cli"))
    }

    fn set_private_ane_opt_in(&mut self) -> Result<()> {
        Err(apple_feature_unavailable("cli"))
    }

    fn finish(self) -> Result<Option<AppleBenchConfig>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::harness::{EnergyBenchHarness, HarnessMeasurement};
    use super::schema::{
        BenchDeviceMetadata, BenchEnergyMetrics, BenchHarnessStub, BenchLatencyStats, BenchReport,
        BenchRunMetrics,
    };
    use super::*;
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

    #[cfg(feature = "apple")]
    #[test]
    fn parses_apple_bench_mode_without_touching_hardware() {
        use rvllm_runtime::AppleBackendMode;

        let cfg = match parse_bench_args([
            "rvllm-bench",
            "--apple-mode",
            "metal-prefill-metal-decode",
            "--apple-rollout-tokens",
            "1",
        ]) {
            Ok(cfg) => cfg,
            Err(e) => panic!("unexpected parse error: {e}"),
        };

        let apple = match cfg.apple {
            Some(apple) => apple,
            None => panic!("expected apple config"),
        };
        assert_eq!(apple.mode, AppleBackendMode::MetalPrefillMetalDecode);
        assert_eq!(apple.rollout_tokens, Some(1));
        assert!(!apple.private_ane_opt_in);
    }

    #[cfg(feature = "apple")]
    #[test]
    fn metal_only_bench_mode_does_not_require_rollout_tokens() {
        use rvllm_runtime::AppleBackendMode;

        let cfg = match parse_bench_args(["rvllm-bench", "--apple-mode", "metal-only"]) {
            Ok(cfg) => cfg,
            Err(e) => panic!("unexpected parse error: {e}"),
        };
        let apple = match cfg.apple {
            Some(apple) => apple,
            None => panic!("expected apple config"),
        };
        assert_eq!(apple.mode, AppleBackendMode::MetalOnly);
        assert_eq!(apple.rollout_tokens, None);
    }

    #[cfg(not(feature = "apple"))]
    #[test]
    fn apple_bench_flags_require_apple_feature() {
        let err = match parse_bench_args(["rvllm-bench", "--apple-mode", "metal-only"]) {
            Ok(_) => panic!("apple flags should be rejected when feature is disabled"),
            Err(err) => err,
        };

        assert!(format!("{err}").contains("FeatureNotAvailable"));
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
