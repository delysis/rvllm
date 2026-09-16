//! Rust suite runner for Gemma 4 Metal text inference cases.
//!
//! This intentionally shells out to `rvllm_metal_infer` so the suite records
//! the exact public command line, stdout, stderr, exit status, and duration for
//! every model/prompt case without adding another inference implementation.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

const SUITE_SCHEMA: &str = "rvllm.apple_metal_model_suite.v1";
const MANIFEST_SCHEMA: &str = "rvllm.apple_metal_model_suite_manifest.v1";
const CLAIM: &str =
    "Apple Metal model suite evidence; not production-ready until acceptance gates pass";
const DEFAULT_MAX_TOTAL_TOKENS: usize = 2048;
const DEFAULT_CASE_TIMEOUT_SECONDS: u64 = 900;

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    report: Option<PathBuf>,
    json_output: bool,
}

#[derive(Debug)]
struct SuiteCase {
    name: String,
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    max_total_tokens: usize,
    required: bool,
    large_model_opt_in: bool,
    no_bos: bool,
    hf_reference: Option<PathBuf>,
    timeout_seconds: u64,
    min_tok_per_s: Option<f64>,
    max_prefill_ms: Option<f64>,
    max_decode_ms: Option<f64>,
}

fn main() -> ExitCode {
    match run() {
        Ok(status) => {
            if status == "pass" {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(err) => {
            eprintln!("rvllm-metal-model-suite: {err}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<&'static str, String> {
    let args = parse_args(std::env::args().skip(1))?;
    let manifest = std::fs::read_to_string(&args.manifest)
        .map_err(|err| format!("read manifest {}: {err}", args.manifest.display()))?;
    let cases = parse_manifest(&manifest)?;
    let infer_bin = infer_binary_path()?;
    let suite_start = Instant::now();
    let mut reports = Vec::with_capacity(cases.len());

    for case in &cases {
        reports.push(run_case(&infer_bin, case)?);
    }

    let passed = reports
        .iter()
        .filter(|case| case.get("status").and_then(serde_json::Value::as_str) == Some("pass"))
        .count();
    let skipped = reports
        .iter()
        .filter(|case| {
            case.get("status")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|status| status.starts_with("skip_"))
        })
        .count();
    let status = if reports.iter().all(|case| {
        case.get("status")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|status| status == "pass" || status.starts_with("skip_"))
    }) {
        "pass"
    } else {
        "fail"
    };
    let report = serde_json::json!({
        "schema": SUITE_SCHEMA,
        "claim": CLAIM,
        "status": status,
        "manifest": args.manifest,
        "case_count": reports.len(),
        "passed": passed,
        "skipped": skipped,
        "duration_ms": ms(suite_start.elapsed()),
        "cases": reports,
    });
    if let Some(path) = &args.report {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|err| format!("create report parent {}: {err}", parent.display()))?;
            }
        }
        std::fs::write(
            path,
            serde_json::to_string_pretty(&report)
                .map_err(|err| format!("serialize report: {err}"))?,
        )
        .map_err(|err| format!("write report {}: {err}", path.display()))?;
    }
    if args.json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|err| format!("serialize json: {err}"))?
        );
    } else {
        println!("status: {status}");
        println!("claim: {CLAIM}");
        for case in report["cases"].as_array().unwrap_or(&Vec::new()) {
            println!(
                "{}: {}",
                case["name"].as_str().unwrap_or("<unnamed>"),
                case["status"].as_str().unwrap_or("unknown")
            );
        }
    }
    Ok(status)
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut manifest = None;
    let mut report = None;
    let mut json_output = false;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--manifest" => {
                manifest = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--manifest requires a value".to_owned())?,
                ));
            }
            "--report" => {
                report = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--report requires a value".to_owned())?,
                ));
            }
            "--json" => json_output = true,
            "-h" | "--help" => return Err(usage()),
            other if other.starts_with('-') => return Err(format!("unknown argument: {other}")),
            other => return Err(format!("unexpected positional argument: {other}")),
        }
    }
    Ok(Args {
        manifest: manifest.ok_or_else(|| "--manifest is required".to_owned())?,
        report,
        json_output,
    })
}

fn usage() -> String {
    "usage: rvllm_metal_model_suite --manifest <JSON> [--report <JSON>] [--json]".to_owned()
}

fn parse_manifest(raw: &str) -> Result<Vec<SuiteCase>, String> {
    let value: serde_json::Value =
        serde_json::from_str(raw).map_err(|err| format!("parse manifest json: {err}"))?;
    let schema = value
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "manifest missing schema".to_owned())?;
    if schema != MANIFEST_SCHEMA {
        return Err(format!(
            "manifest schema must be {MANIFEST_SCHEMA:?}; got {schema:?}"
        ));
    }
    let cases = value
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "manifest cases must be an array".to_owned())?;
    if cases.is_empty() {
        return Err("manifest must contain at least one case".to_owned());
    }
    let mut names = std::collections::HashSet::new();
    cases
        .iter()
        .enumerate()
        .map(|(idx, case)| parse_case(idx, case, &mut names))
        .collect()
}

fn parse_case(
    idx: usize,
    value: &serde_json::Value,
    names: &mut std::collections::HashSet<String>,
) -> Result<SuiteCase, String> {
    let name = required_string(value, "name", idx)?.to_owned();
    if !names.insert(name.clone()) {
        return Err(format!("duplicate case name {name:?}"));
    }
    let max_new_tokens = optional_usize(value, "max_new_tokens")?.unwrap_or(1);
    let max_total_tokens =
        optional_usize(value, "max_total_tokens")?.unwrap_or(DEFAULT_MAX_TOTAL_TOKENS);
    if max_new_tokens == 0 {
        return Err(format!("case {name} max_new_tokens must be positive"));
    }
    if max_total_tokens == 0 {
        return Err(format!("case {name} max_total_tokens must be positive"));
    }
    let timeout_seconds = optional_usize(value, "timeout_seconds")?
        .map(u64::try_from)
        .transpose()
        .map_err(|_| format!("case {name} timeout_seconds exceeds u64"))?
        .unwrap_or(DEFAULT_CASE_TIMEOUT_SECONDS);
    if timeout_seconds == 0 {
        return Err(format!("case {name} timeout_seconds must be positive"));
    }
    if value.get("expected_error_contains").is_some() {
        return Err(format!(
            "case {name} uses expected_error_contains; model-suite cases must pass, fail, or skip missing optional models"
        ));
    }
    let min_tok_per_s = optional_positive_f64(value, "min_tok_per_s", &name)?;
    let max_prefill_ms = optional_positive_f64(value, "max_prefill_ms", &name)?;
    let max_decode_ms = optional_positive_f64(value, "max_decode_ms", &name)?;
    Ok(SuiteCase {
        name,
        model_dir: PathBuf::from(required_string(value, "model_dir", idx)?),
        prompt: required_string(value, "prompt", idx)?.to_owned(),
        max_new_tokens,
        max_total_tokens,
        required: optional_bool(value, "required").unwrap_or(true),
        large_model_opt_in: optional_bool(value, "large_model_opt_in").unwrap_or(false),
        no_bos: optional_bool(value, "no_bos").unwrap_or(false),
        hf_reference: value
            .get("hf_reference")
            .and_then(serde_json::Value::as_str)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from),
        timeout_seconds,
        min_tok_per_s,
        max_prefill_ms,
        max_decode_ms,
    })
}

fn required_string<'a>(
    value: &'a serde_json::Value,
    field: &str,
    idx: usize,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|item| !item.is_empty())
        .ok_or_else(|| format!("case {idx} missing non-empty {field:?}"))
}

fn optional_bool(value: &serde_json::Value, field: &str) -> Option<bool> {
    value.get(field).and_then(serde_json::Value::as_bool)
}

fn optional_usize(value: &serde_json::Value, field: &str) -> Result<Option<usize>, String> {
    value
        .get(field)
        .map(|raw| {
            raw.as_u64()
                .and_then(|item| usize::try_from(item).ok())
                .ok_or_else(|| format!("{field} must be a positive integer"))
        })
        .transpose()
}

fn optional_positive_f64(
    value: &serde_json::Value,
    field: &str,
    case_name: &str,
) -> Result<Option<f64>, String> {
    value
        .get(field)
        .map(|raw| {
            let number = raw
                .as_f64()
                .ok_or_else(|| format!("case {case_name} {field} must be a number"))?;
            if !number.is_finite() || number <= 0.0 {
                return Err(format!(
                    "case {case_name} {field} must be finite and positive"
                ));
            }
            Ok(number)
        })
        .transpose()
}

fn infer_binary_path() -> Result<PathBuf, String> {
    let current = std::env::current_exe().map_err(|err| format!("current_exe: {err}"))?;
    let dir = current
        .parent()
        .ok_or_else(|| format!("current_exe has no parent: {}", current.display()))?;
    let candidate = dir.join("rvllm_metal_infer");
    if candidate.is_file() {
        Ok(candidate)
    } else {
        Err(format!(
            "rvllm_metal_infer binary not found next to {}",
            current.display()
        ))
    }
}

fn run_case(infer_bin: &Path, case: &SuiteCase) -> Result<serde_json::Value, String> {
    let mut command = vec![
        infer_bin.display().to_string(),
        "--model-dir".to_owned(),
        case.model_dir.display().to_string(),
        "--prompt".to_owned(),
        case.prompt.clone(),
        "--max-new-tokens".to_owned(),
        case.max_new_tokens.to_string(),
        "--max-total-tokens".to_owned(),
        case.max_total_tokens.to_string(),
        "--json".to_owned(),
    ];
    if case.large_model_opt_in {
        command.push("--large-model-opt-in".to_owned());
    }
    if case.no_bos {
        command.push("--no-bos".to_owned());
    }
    if let Some(path) = &case.hf_reference {
        command.push("--hf-reference".to_owned());
        command.push(path.display().to_string());
    }

    if !case.model_dir.is_dir() {
        let status = if case.required {
            "missing_model"
        } else {
            "skip_missing_model"
        };
        return Ok(serde_json::json!({
            "name": case.name,
            "status": status,
            "model_dir": case.model_dir,
            "prompt": case.prompt,
            "required": case.required,
            "reason": "model_dir does not exist",
            "command": command,
            "claim": CLAIM,
        }));
    }
    if let Some(path) = &case.hf_reference {
        if !path.is_file() {
            return Ok(serde_json::json!({
                "name": case.name,
                "status": "missing_reference",
                "model_dir": case.model_dir,
                "prompt": case.prompt,
                "hf_reference": path,
                "command": command,
                "claim": CLAIM,
            }));
        }
    }

    let start = Instant::now();
    let mut child = Command::new(infer_bin);
    child
        .arg("--model-dir")
        .arg(&case.model_dir)
        .arg("--prompt")
        .arg(&case.prompt)
        .arg("--max-new-tokens")
        .arg(case.max_new_tokens.to_string())
        .arg("--max-total-tokens")
        .arg(case.max_total_tokens.to_string())
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if case.large_model_opt_in {
        child.arg("--large-model-opt-in");
    }
    if case.no_bos {
        child.arg("--no-bos");
    }
    if let Some(path) = &case.hf_reference {
        child.arg("--hf-reference").arg(path);
    }
    let output = run_child_with_timeout(child, Duration::from_secs(case.timeout_seconds))
        .map_err(|err| format!("run case {}: {err}", case.name))?;
    let duration_ms = ms(start.elapsed());
    let stdout = String::from_utf8_lossy(&output.output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.output.stderr).into_owned();
    let parsed = parse_json_from_stdout(&stdout);
    let matched = parsed
        .as_ref()
        .ok()
        .and_then(|json| json.get("hf_reference"))
        .and_then(|reference| reference.get("matched"))
        .and_then(serde_json::Value::as_bool);
    let performance_failures = parsed.as_ref().map_or_else(
        |_| Vec::new(),
        |report| performance_gate_failures(report, case),
    );
    let status = if output.timed_out || !output.output.status.success() {
        "fail"
    } else if matched == Some(false) {
        "fail"
    } else if parsed.is_err() {
        "fail"
    } else if !performance_failures.is_empty() {
        "fail"
    } else {
        "pass"
    };
    Ok(serde_json::json!({
        "name": case.name,
        "status": status,
        "model_dir": case.model_dir,
        "prompt": case.prompt,
        "command": command,
        "returncode": output.output.status.code(),
        "duration_ms": duration_ms,
        "timeout_seconds": case.timeout_seconds,
        "timed_out": output.timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "report": parsed.ok(),
        "performance_gates": {
            "min_tok_per_s": case.min_tok_per_s,
            "max_prefill_ms": case.max_prefill_ms,
            "max_decode_ms": case.max_decode_ms,
            "failures": performance_failures,
        },
        "claim": CLAIM,
    }))
}

fn performance_gate_failures(report: &serde_json::Value, case: &SuiteCase) -> Vec<String> {
    let mut failures = Vec::new();
    if let Some(minimum) = case.min_tok_per_s {
        match report.get("tok_per_s").and_then(serde_json::Value::as_f64) {
            Some(actual) if actual.is_finite() && actual >= minimum => {}
            Some(actual) => failures.push(format!(
                "tok_per_s {actual:.6} is below required minimum {minimum:.6}"
            )),
            None => failures.push("report missing numeric tok_per_s".to_owned()),
        }
    }
    if let Some(maximum) = case.max_prefill_ms {
        match report.get("prefill_ms").and_then(serde_json::Value::as_f64) {
            Some(actual) if actual.is_finite() && actual <= maximum => {}
            Some(actual) => failures.push(format!(
                "prefill_ms {actual:.6} exceeds required maximum {maximum:.6}"
            )),
            None => failures.push("report missing numeric prefill_ms".to_owned()),
        }
    }
    if let Some(maximum) = case.max_decode_ms {
        match report.get("decode_ms").and_then(serde_json::Value::as_f64) {
            Some(actual) if actual.is_finite() && actual <= maximum => {}
            Some(actual) => failures.push(format!(
                "decode_ms {actual:.6} exceeds required maximum {maximum:.6}"
            )),
            None => failures.push("report missing numeric decode_ms".to_owned()),
        }
    }
    failures
}

struct TimedOutput {
    output: std::process::Output,
    timed_out: bool,
}

fn run_child_with_timeout(mut child: Command, timeout: Duration) -> Result<TimedOutput, String> {
    let mut child = child
        .spawn()
        .map_err(|err| format!("spawn rvllm_metal_infer: {err}"))?;
    let start = Instant::now();
    loop {
        if child
            .try_wait()
            .map_err(|err| format!("poll rvllm_metal_infer: {err}"))?
            .is_some()
        {
            let output = child
                .wait_with_output()
                .map_err(|err| format!("collect rvllm_metal_infer output: {err}"))?;
            return Ok(TimedOutput {
                output,
                timed_out: false,
            });
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let output = child
                .wait_with_output()
                .map_err(|err| format!("collect timed-out rvllm_metal_infer output: {err}"))?;
            return Ok(TimedOutput {
                output,
                timed_out: true,
            });
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn parse_json_from_stdout(stdout: &str) -> Result<serde_json::Value, String> {
    let start = stdout
        .find('{')
        .ok_or_else(|| "stdout did not contain a JSON object".to_owned())?;
    serde_json::from_str(&stdout[start..]).map_err(|err| format!("parse stdout json: {err}"))
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_manifest_rejects_duplicate_names() {
        let raw = r#"{
            "schema": "rvllm.apple_metal_model_suite_manifest.v1",
            "cases": [
                {"name":"same","model_dir":"/tmp/a","prompt":"Hello"},
                {"name":"same","model_dir":"/tmp/b","prompt":"Hello"}
            ]
        }"#;
        let err = parse_manifest(raw).expect_err("duplicate names must fail");
        assert!(err.contains("duplicate case name"));
    }

    #[test]
    fn parse_manifest_accepts_minimal_case() {
        let raw = r#"{
            "schema": "rvllm.apple_metal_model_suite_manifest.v1",
            "cases": [
                {"name":"e2b","model_dir":"/tmp/model","prompt":"Hello"}
            ]
        }"#;
        let cases = parse_manifest(raw).expect("manifest should parse");
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].max_new_tokens, 1);
        assert_eq!(cases[0].max_total_tokens, DEFAULT_MAX_TOTAL_TOKENS);
        assert!(cases[0].required);
        assert_eq!(cases[0].min_tok_per_s, None);
        assert_eq!(cases[0].max_prefill_ms, None);
        assert_eq!(cases[0].max_decode_ms, None);
    }

    #[test]
    fn parse_manifest_accepts_optional_missing_case_marker() {
        let raw = r#"{
            "schema": "rvllm.apple_metal_model_suite_manifest.v1",
            "cases": [
                {"name":"missing","model_dir":"/tmp/model","prompt":"Hello","required":false}
            ]
        }"#;
        let cases = parse_manifest(raw).expect("manifest should parse");
        assert!(!cases[0].required);
    }

    #[test]
    fn parse_manifest_rejects_expected_error_contract() {
        let raw = r#"{
            "schema": "rvllm.apple_metal_model_suite_manifest.v1",
            "cases": [
                {
                    "name":"moe",
                    "model_dir":"/tmp/model",
                    "prompt":"Hello",
                    "expected_error_contains":"unsupported MoE marker"
                }
            ]
        }"#;
        let err = parse_manifest(raw).expect_err("expected_error_contains must be rejected");
        assert!(err.contains("expected_error_contains"));
    }

    #[test]
    fn parse_manifest_accepts_performance_gates() {
        let raw = r#"{
            "schema": "rvllm.apple_metal_model_suite_manifest.v1",
            "cases": [
                {
                    "name":"e2b",
                    "model_dir":"/tmp/model",
                    "prompt":"Hello",
                    "min_tok_per_s":20.0,
                    "max_prefill_ms":500.0,
                    "max_decode_ms":250.0
                }
            ]
        }"#;
        let cases = parse_manifest(raw).expect("performance gates should parse");
        assert_eq!(cases[0].min_tok_per_s, Some(20.0));
        assert_eq!(cases[0].max_prefill_ms, Some(500.0));
        assert_eq!(cases[0].max_decode_ms, Some(250.0));
    }

    #[test]
    fn parse_manifest_rejects_nonpositive_performance_gate() {
        for field in ["min_tok_per_s", "max_prefill_ms", "max_decode_ms"] {
            let raw = format!(
                r#"{{
                    "schema": "rvllm.apple_metal_model_suite_manifest.v1",
                    "cases": [
                        {{"name":"e2b","model_dir":"/tmp/model","prompt":"Hello","{field}":0}}
                    ]
                }}"#
            );
            let err = parse_manifest(&raw).expect_err("zero performance gate must fail");
            assert!(err.contains(field), "{err}");
            assert!(err.contains("finite and positive"), "{err}");
        }
    }

    #[test]
    fn performance_gates_fail_closed_on_regression_or_missing_metric() {
        let case = SuiteCase {
            name: "e2b".to_owned(),
            model_dir: PathBuf::from("/tmp/model"),
            prompt: "Hello".to_owned(),
            max_new_tokens: 4,
            max_total_tokens: 16,
            required: true,
            large_model_opt_in: false,
            no_bos: false,
            hf_reference: None,
            timeout_seconds: 60,
            min_tok_per_s: Some(20.0),
            max_prefill_ms: Some(500.0),
            max_decode_ms: Some(250.0),
        };

        assert!(performance_gate_failures(
            &serde_json::json!({"tok_per_s": 25.0, "prefill_ms": 400.0, "decode_ms": 160.0}),
            &case,
        )
        .is_empty());
        let slow = performance_gate_failures(
            &serde_json::json!({"tok_per_s": 12.0, "prefill_ms": 600.0, "decode_ms": 300.0}),
            &case,
        );
        assert_eq!(slow.len(), 3);
        let missing = performance_gate_failures(&serde_json::json!({}), &case);
        assert_eq!(missing.len(), 3);
    }
}
