//! A fixed, fail-closed full-route timing collection. Queue conditions are
//! adjudicated separately; this program never labels its own result promotable.
#![forbid(unsafe_code)]

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    Control,
    Candidate,
}

// Two warmups per arm, then one ABBA block and its reversed BAAB mate.
const ORDER: [Arm; 12] = [
    Arm::Control,
    Arm::Candidate,
    Arm::Control,
    Arm::Candidate,
    Arm::Control,
    Arm::Candidate,
    Arm::Candidate,
    Arm::Control,
    Arm::Candidate,
    Arm::Control,
    Arm::Control,
    Arm::Candidate,
];

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Candidate => "candidate",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    infer_executable: Pin,
    model_dir: PathBuf,
    prompts_jsonl: PathBuf,
    control_metallib: Pin,
    candidate_metallib: Pin,
    candidate_name: String,
    #[serde(default = "default_prompt_lengths")]
    expected_prompt_lengths: Vec<usize>,
    output_dir: PathBuf,
}

fn default_prompt_lengths() -> Vec<usize> {
    vec![101, 304]
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut chunk).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        hash.update(&chunk[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn verify_pin(pin: &Pin) -> Result<()> {
    if pin.sha256.len() != 64 || !pin.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("invalid SHA-256 pin for {}", pin.path.display()));
    }
    if sha256(&pin.path)? != pin.sha256.to_ascii_lowercase() {
        return Err(format!("SHA-256 mismatch for {}", pin.path.display()));
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn positive(value: &Value, key: &str) -> Result<f64> {
    let number = value[key]
        .as_f64()
        .ok_or_else(|| format!("missing numeric {key}"))?;
    if !number.is_finite() || number <= 0.0 {
        return Err(format!("invalid {key}"));
    }
    Ok(number)
}

fn expected_dispatch(candidate_name: &str) -> &'static [(&'static str, u64)] {
    const PIPELINE: &[(&str, u64)] = &[
        ("research_prefill_pipeline_gemm", 48),
        ("research_prefill_pipeline_qkv", 48),
        ("research_prefill_pipeline_raw_norm_projection", 96),
        ("research_prefill_pipeline_raw_norm", 96),
    ];
    const COMBINED: &[(&str, u64)] = &[
        ("research_prefill_combined_gemm", 48),
        ("research_prefill_combined_qkv", 48),
        ("research_prefill_combined_raw_norm_projection", 96),
        ("research_prefill_combined_raw_norm", 96),
        ("research_prefill_combined_d256", 40),
        ("research_prefill_combined_d512", 8),
    ];
    match candidate_name {
        "metal-prefill-pipeline32x64" => PIPELINE,
        "metal-prefill-pipeline32x64-q4k16" => COMBINED,
        _ => &[],
    }
}

fn check_output(
    report: &Value,
    arm: Arm,
    candidate_name: &str,
    expected_prompt_lengths: &[usize],
) -> Result<Value> {
    if report["schema"] != "rvllm.apple_metal_text_session.v1"
        || report["status"] != "pass"
        || report["backend"] != "direct"
        || report["metal_compute_dtype"] != "bfloat16"
    {
        return Err("wrong route, dtype, or session status".into());
    }
    let cases = report["cases"].as_array().ok_or("missing cases")?;
    if cases.len() != expected_prompt_lengths.len() {
        return Err("wrong number of predeclared prompts".into());
    }
    let mut compact = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let expected_prompt_len = expected_prompt_lengths[index];
        let prompt = case["prompt_token_ids"]
            .as_array()
            .ok_or("missing prompt IDs")?;
        let generated = case["generated_token_ids"]
            .as_array()
            .ok_or("missing generated IDs")?;
        if prompt.len() != expected_prompt_len
            || generated.len() != 64
            || case["library_compiles"] != 0
            || case["pipeline_state_compiles"] != 0
            || case["research_dispatch"]["overflowed"] != false
        {
            return Err(format!(
                "case {index}: changed work, compile, or dispatch overflow"
            ));
        }
        let counts = case["research_dispatch"]["counts"]
            .as_object()
            .ok_or("missing dispatch counts")?;
        let expected = if arm == Arm::Control {
            &[][..]
        } else {
            expected_dispatch(candidate_name)
        };
        if counts.len() != expected.len()
            || expected
                .iter()
                .any(|(name, count)| counts.get(*name).and_then(Value::as_u64) != Some(*count))
        {
            return Err(format!("case {index}: fallback or wrong research dispatch"));
        }
        compact.push(json!({
            "name": case["name"],
            "prompt_token_ids": prompt,
            "generated_token_ids": generated,
            "prefill_ms": positive(case, "prefill_ms")?,
            "decode_ms": positive(case, "decode_ms")?,
            "research_dispatch": counts,
        }));
    }
    Ok(json!(compact))
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    (sorted[middle - 1] + sorted[middle]) / 2.0
}

fn drift(values: &[f64]) -> f64 {
    let anchor = values[0];
    values
        .iter()
        .map(|v| (v - anchor).abs() / anchor)
        .fold(0.0, f64::max)
}

fn run_child(config: &Config, arm: Arm, ordinal: usize) -> Result<Value> {
    let metallib = if arm == Arm::Control {
        &config.control_metallib.path
    } else {
        &config.candidate_metallib.path
    };
    let selector = if arm == Arm::Control {
        "off"
    } else {
        config.candidate_name.as_str()
    };
    let output = Command::new(&config.infer_executable.path)
        .args([
            "--model-dir",
            config.model_dir.to_str().ok_or("model path is not UTF-8")?,
            "--prompts-jsonl",
            config
                .prompts_jsonl
                .to_str()
                .ok_or("prompt path is not UTF-8")?,
            "--session-backend",
            "direct",
            "--max-new-tokens",
            "64",
            "--max-total-tokens",
            "1024",
            "--large-model-opt-in",
            "--json",
        ])
        .env("RVLLM_METAL_RESEARCH", selector)
        .env("RVLLM_METAL_METALLIB_BF16", metallib)
        .env_remove("RVLLM_METAL_PREFILL_GEMM")
        .env_remove("RVLLM_METAL_PREFILL_ATTENTION")
        .env_remove("RVLLM_METAL_QKV_PREFILL")
        .output()
        .map_err(|e| format!("child {ordinal}: {e}"))?;
    let prefix = format!("sample-{ordinal:02}-{}", arm.name());
    write_new(
        &config.output_dir.join(format!("{prefix}.stdout")),
        &output.stdout,
    )?;
    write_new(
        &config.output_dir.join(format!("{prefix}.stderr")),
        &output.stderr,
    )?;
    if !output.status.success() {
        return Err(format!(
            "child {ordinal} exited {:?}; raw output retained",
            output.status.code()
        ));
    }
    let report: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("child {ordinal} report: {e}"))?;
    let compact = check_output(
        &report,
        arm,
        &config.candidate_name,
        &config.expected_prompt_lengths,
    )?;
    write_new(
        &config.output_dir.join(format!("{prefix}.validated.json")),
        &serde_json::to_vec_pretty(&compact).map_err(|e| e.to_string())?,
    )?;
    Ok(compact)
}

fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let config_path = PathBuf::from(
        args.next()
            .ok_or("usage: rvllm-prefill-route-abba CONFIG.json")?,
    );
    if args.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    let config_bytes = fs::read(&config_path).map_err(|e| e.to_string())?;
    let config: Config = serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
    if config.schema != "rvllm.prefill_route_abba_config.v1"
        || expected_dispatch(&config.candidate_name).is_empty()
        || !matches!(
            config.expected_prompt_lengths.as_slice(),
            [101] | [304] | [101, 304] | [304, 101]
        )
        || !config.model_dir.is_dir()
        || !config.prompts_jsonl.is_file()
    {
        return Err("invalid predeclared route ABBA configuration".into());
    }
    for pin in [
        &config.infer_executable,
        &config.control_metallib,
        &config.candidate_metallib,
    ] {
        verify_pin(pin)?;
    }
    fs::create_dir(&config.output_dir).map_err(|e| {
        format!(
            "fresh output directory {}: {e}",
            config.output_dir.display()
        )
    })?;
    write_new(&config.output_dir.join("config.json"), &config_bytes)?;

    let mut samples = Vec::new();
    let mut baseline: Option<Value> = None;
    for (ordinal, arm) in ORDER.into_iter().enumerate() {
        let cases = run_child(&config, arm, ordinal)?;
        let identity: Value = json!(cases
            .as_array()
            .ok_or("cases not an array")?
            .iter()
            .map(|case| {
                json!({
                    "name": case["name"],
                    "prompt_token_ids": case["prompt_token_ids"],
                    "generated_token_ids": case["generated_token_ids"],
                })
            })
            .collect::<Vec<_>>());
        if baseline
            .as_ref()
            .is_some_and(|original| original != &identity)
        {
            return Err(format!(
                "child {ordinal}: output trajectory differs; raw output retained"
            ));
        }
        baseline.get_or_insert(identity);
        samples.push(
            json!({"ordinal": ordinal, "arm": arm.name(), "warmup": ordinal < 4, "cases": cases}),
        );
    }
    let mut scores = Vec::new();
    for case_index in 0..config.expected_prompt_lengths.len() {
        let prefill = |ordinal: usize| -> Result<f64> {
            positive(&samples[ordinal]["cases"][case_index], "prefill_ms")
        };
        let controls = [prefill(4)?, prefill(7)?, prefill(9)?, prefill(10)?];
        let candidates = [prefill(5)?, prefill(6)?, prefill(8)?, prefill(11)?];
        let abba = median(&controls[..2]) / median(&candidates[..2]);
        let baab = median(&controls[2..]) / median(&candidates[2..]);
        scores.push(json!({
            "case": case_index,
            "control_ms": controls,
            "candidate_ms": candidates,
            "abba_ratio": abba,
            "baab_ratio": baab,
            "control_drift": drift(&controls),
            "candidate_drift": drift(&candidates),
            "five_percent_drift_pass": drift(&controls) <= 0.05 && drift(&candidates) <= 0.05,
        }));
    }
    let summary = json!({
        "schema": "rvllm.prefill_route_abba.v1",
        "claim": "counterbalanced full-route speed screen only; queue conditions, independent confirmation and numerical reference remain separate",
        "config_sha256": format!("{:x}", Sha256::digest(&config_bytes)),
        "candidate": config.candidate_name,
        "expected_prompt_lengths": config.expected_prompt_lengths,
        "samples": samples,
        "scores": scores,
    });
    write_new(
        &config.output_dir.join("summary.json"),
        &serde_json::to_vec_pretty(&summary).map_err(|e| e.to_string())?,
    )?;
    println!(
        "{}",
        serde_json::to_string(&summary).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rvllm-prefill-route-abba: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_order_is_balanced_and_reversed() {
        assert_eq!(
            &ORDER[..4],
            &[Arm::Control, Arm::Candidate, Arm::Control, Arm::Candidate]
        );
        assert_eq!(
            &ORDER[4..8],
            &[Arm::Control, Arm::Candidate, Arm::Candidate, Arm::Control]
        );
        assert_eq!(
            &ORDER[8..],
            &[Arm::Candidate, Arm::Control, Arm::Control, Arm::Candidate]
        );
    }

    #[test]
    fn dispatch_selector_is_closed() {
        assert_eq!(expected_dispatch("metal-prefill-pipeline32x64").len(), 4);
        assert_eq!(
            expected_dispatch("metal-prefill-pipeline32x64-q4k16").len(),
            6
        );
        assert!(expected_dispatch("off").is_empty());
    }

    #[test]
    fn receipt_rejects_fallback_and_inference_compilation() {
        let mut cases = Vec::new();
        for prompt_len in [101, 304] {
            cases.push(json!({
                "name": format!("M{prompt_len}"),
                "prompt_token_ids": vec![2_u64; prompt_len],
                "generated_token_ids": vec![3_u64; 64],
                "library_compiles": 0,
                "pipeline_state_compiles": 0,
                "research_dispatch": {
                    "overflowed": false,
                    "counts": expected_dispatch("metal-prefill-pipeline32x64")
                        .iter().copied().collect::<std::collections::BTreeMap<_, _>>(),
                },
                "prefill_ms": 10.0,
                "decode_ms": 20.0,
            }));
        }
        let mut report = json!({
            "schema": "rvllm.apple_metal_text_session.v1",
            "status": "pass",
            "backend": "direct",
            "metal_compute_dtype": "bfloat16",
            "cases": cases,
        });
        assert!(check_output(
            &report,
            Arm::Candidate,
            "metal-prefill-pipeline32x64",
            &[101, 304]
        )
        .is_ok());
        let mut reversed = report.clone();
        reversed["cases"].as_array_mut().unwrap().reverse();
        assert!(check_output(
            &reversed,
            Arm::Candidate,
            "metal-prefill-pipeline32x64",
            &[304, 101]
        )
        .is_ok());
        assert!(check_output(
            &reversed,
            Arm::Candidate,
            "metal-prefill-pipeline32x64",
            &[101, 304]
        )
        .is_err());
        let mut single = report.clone();
        single["cases"].as_array_mut().unwrap().remove(0);
        assert!(check_output(
            &single,
            Arm::Candidate,
            "metal-prefill-pipeline32x64",
            &[304]
        )
        .is_ok());
        assert!(check_output(
            &single,
            Arm::Candidate,
            "metal-prefill-pipeline32x64",
            &[101]
        )
        .is_err());
        assert!(check_output(
            &report,
            Arm::Candidate,
            "metal-prefill-pipeline32x64",
            &[304]
        )
        .is_err());
        report["cases"][0]["research_dispatch"]["counts"] = json!({});
        assert!(check_output(
            &report,
            Arm::Candidate,
            "metal-prefill-pipeline32x64",
            &[101, 304]
        )
        .unwrap_err()
        .contains("fallback"));
        report["cases"][0]["research_dispatch"]["counts"] =
            json!(expected_dispatch("metal-prefill-pipeline32x64")
                .iter()
                .copied()
                .collect::<std::collections::BTreeMap<_, _>>());
        report["cases"][0]["pipeline_state_compiles"] = json!(1);
        assert!(check_output(
            &report,
            Arm::Candidate,
            "metal-prefill-pipeline32x64",
            &[101, 304]
        )
        .unwrap_err()
        .contains("compile"));
    }
}
