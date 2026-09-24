//! Strict summarizer for completed MLX Gemma 4 stage-microbenchmark queue jobs.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::parse_strict_json;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str = "rvllm.mlx_gemma4_stage_microbenchmark.v1";
const OUTPUT_SCHEMA: &str = "rvllm.mlx_gemma4_stage_queue_summary.v1";
const MEASURED_STATUS: &str = "measured_microbenchmark_not_normal_route";
const PINNED_MLX_COMMIT: &str = "c215b6f88cf0fee0b0895623e4046cda797ef397";
const PINNED_MLX_LM_COMMIT: &str = "87b7b583a697537aa68f47130b40884700b5f55f";
const STAGE_TOOL: &str = "tools/mlx_gemma4_stage_bench.py";
const MODES: [&str; 2] = ["prefill", "decode"];
const STAGES: [(&str, Option<&str>); 11] = [
    ("embedding", None),
    ("qkv", Some("sliding_attention")),
    ("qkv", Some("full_attention")),
    ("attention_core_sdpa", Some("sliding_attention")),
    ("attention_core_sdpa", Some("full_attention")),
    ("o_projection", Some("sliding_attention")),
    ("o_projection", Some("full_attention")),
    ("ffn_gate_up_activation", None),
    ("ffn_down", None),
    ("rmsnorm_residual", None),
    ("lm_head", None),
];

type ByVariant = BTreeMap<String, Timing>;
type ByCategory = BTreeMap<String, ByVariant>;
type ByMode = BTreeMap<String, ByCategory>;
type ByLength = BTreeMap<String, ByMode>;
type ByBits = BTreeMap<String, ByLength>;

#[derive(Debug, Serialize)]
struct Summary {
    schema: &'static str,
    claim: &'static str,
    results_directory: String,
    evidence: Vec<QueueEvidence>,
    grouped: ByBits,
}

#[derive(Debug, Serialize)]
struct QueueEvidence {
    id: String,
    directory: String,
    weight_bits: u64,
    model: String,
    lengths: Vec<u64>,
    sampled_conditions_eligible: bool,
    sampled_controls_eligible: bool,
    comparison_stratum: Option<Value>,
    observed_strata: Vec<Value>,
    condition_samples: usize,
    condition_samples_with_observed_processes: usize,
    observed_processes: Vec<ObservedProcess>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct ObservedProcess {
    name: String,
    sample_count: u64,
    max_instances: u64,
}

#[derive(Clone, Debug, Serialize)]
struct Timing {
    total_ms: f64,
    mean_ms: f64,
    iterations: u64,
    job_id: String,
}

#[derive(Debug)]
struct JobIdentity {
    id: String,
    model: String,
    bits: u64,
    lengths: Vec<u64>,
    mlx_source: String,
    mlx_lm_source: String,
    input_hashes: BTreeMap<String, String>,
}

#[derive(Debug)]
struct ParsedCase {
    length: u64,
    mode: String,
    category: String,
    variant: Option<String>,
    total_ms: f64,
    mean_ms: f64,
}

fn main() {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let Some(directory) = args.next() else {
        eprintln!(
            "usage: {} RESULTS_DIRECTORY",
            PathBuf::from(program).display()
        );
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one RESULTS_DIRECTORY");
        std::process::exit(2);
    }
    match summarize(Path::new(&directory))
        .and_then(|summary| serde_json::to_string_pretty(&summary).map_err(|e| e.to_string()))
    {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("rvllm_mlx_stage_queue_summary: {error}");
            std::process::exit(1);
        }
    }
}

fn summarize(results_directory: &Path) -> Result<Summary, String> {
    let mut directories = fs::read_dir(results_directory)
        .map_err(|e| format!("read {}: {e}", results_directory.display()))?
        .map(|entry| entry.map(|entry| entry.path()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    directories.retain(|path| path.is_dir());
    directories.sort();
    if directories.is_empty() {
        return Err("results directory contains no job directories".to_owned());
    }

    let mut evidence = Vec::with_capacity(directories.len());
    let mut grouped = ByBits::new();
    for directory in directories {
        let (job_evidence, cases, iterations) = summarize_job(&directory)?;
        let bits = job_evidence.weight_bits.to_string();
        for case in cases {
            let variant = case.variant.unwrap_or_else(|| "none".to_owned());
            let slot = grouped
                .entry(bits.clone())
                .or_default()
                .entry(case.length.to_string())
                .or_default()
                .entry(case.mode)
                .or_default()
                .entry(case.category)
                .or_default()
                .entry(variant);
            if let std::collections::btree_map::Entry::Occupied(_) = slot {
                return Err(format!(
                    "duplicate grouped case across jobs for {}",
                    job_evidence.id
                ));
            }
            slot.or_insert(Timing {
                total_ms: case.total_ms,
                mean_ms: case.mean_ms,
                iterations,
                job_id: job_evidence.id.clone(),
            });
        }
        evidence.push(job_evidence);
    }
    Ok(Summary {
        schema: OUTPUT_SCHEMA,
        claim: "Validated isolated MLX operator timings only; not end-to-end inference, framework comparison, or promotion evidence.",
        results_directory: results_directory.display().to_string(),
        evidence,
        grouped,
    })
}

fn summarize_job(directory: &Path) -> Result<(QueueEvidence, Vec<ParsedCase>, u64), String> {
    let job = read_strict(&directory.join("job.json"))?;
    let identity = parse_job_identity(&job)?;
    let directory_id = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("non-UTF-8 job directory: {}", directory.display()))?;
    if directory_id != identity.id {
        return Err(format!(
            "directory/job identity mismatch: {directory_id} != {}",
            identity.id
        ));
    }

    let report = read_strict(&directory.join("report.json"))?;
    let report_id = string(&report, "id")?;
    if report_id != identity.id {
        return Err(format!(
            "report/job identity mismatch: {report_id} != {}",
            identity.id
        ));
    }
    if string(&report, "status")? != "succeeded"
        || report.get("exit_code").and_then(Value::as_i64) != Some(0)
    {
        return Err(format!(
            "{} is not a terminal successful queue result",
            identity.id
        ));
    }

    let stdout = fs::read(directory.join("trial.stdout"))
        .map_err(|e| format!("read {}/trial.stdout: {e}", directory.display()))?;
    let receipt: Value = parse_strict_json(&stdout).map_err(|e| {
        format!(
            "parse {}/trial.stdout as exactly one strict JSON document: {e}",
            directory.display()
        )
    })?;
    let (cases, iterations) = validate_receipt(&receipt, &identity)?;
    let activity = parse_conditions(&directory.join("conditions.jsonl"))?;
    let measurement = object_field(&report, "measurement")?;
    let mut observed_strata = Vec::new();
    for value in [
        measurement.get("comparison_stratum"),
        measurement.get("start_host"),
        measurement.get("end_host"),
    ] {
        push_unique(&mut observed_strata, value);
    }
    if let Some(samples) = measurement.get("power_samples").and_then(Value::as_array) {
        for sample in samples {
            push_unique(&mut observed_strata, sample.get("controls"));
        }
    }
    let comparison_stratum = measurement
        .get("comparison_stratum")
        .filter(|value| !value.is_null())
        .cloned();
    let evidence = QueueEvidence {
        id: identity.id,
        directory: directory.display().to_string(),
        weight_bits: identity.bits,
        model: identity.model,
        lengths: identity.lengths,
        sampled_conditions_eligible: boolean(&report, "sampled_conditions_eligible")?,
        sampled_controls_eligible: boolean(measurement, "sampled_controls_eligible")?,
        comparison_stratum,
        observed_strata,
        condition_samples: activity.samples,
        condition_samples_with_observed_processes: activity.samples_with_processes,
        observed_processes: activity.processes,
    };
    Ok((evidence, cases, iterations))
}

fn parse_job_identity(job: &Value) -> Result<JobIdentity, String> {
    let id = string(job, "id")?.to_owned();
    let args = job
        .pointer("/command/args")
        .and_then(Value::as_array)
        .ok_or("job command.args is missing or invalid")?
        .iter()
        .map(|value| value.as_str().ok_or("job command argument is not a string"))
        .collect::<Result<Vec<_>, _>>()?;
    if args.first().copied() != Some(STAGE_TOOL) {
        return Err(format!(
            "job does not invoke pinned stage tool {STAGE_TOOL}"
        ));
    }
    let model = argument(&args, "--model")?.to_owned();
    let bits = parse_u64_argument(&args, "--weight-bits")?;
    if !matches!(bits, 4 | 8 | 16) {
        return Err(format!("unsupported weight bits {bits}"));
    }
    let lengths = argument(&args, "--lengths")?
        .split(',')
        .map(|raw| {
            raw.parse::<u64>()
                .map_err(|e| format!("invalid --lengths item {raw:?}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if lengths.is_empty() || lengths.iter().any(|length| *length == 0) {
        return Err("--lengths must contain positive integers".to_owned());
    }
    if lengths.iter().copied().collect::<BTreeSet<_>>().len() != lengths.len() {
        return Err("--lengths contains duplicates".to_owned());
    }
    let mlx_source = argument(&args, "--mlx-source")?.to_owned();
    let mlx_lm_source = argument(&args, "--mlx-lm-source")?.to_owned();
    let mut input_hashes = BTreeMap::new();
    for input in array_field(job, "inputs")? {
        let path = string(input, "path")?.to_owned();
        let sha256 = string(input, "sha256")?.to_owned();
        if input_hashes.insert(path.clone(), sha256).is_some() {
            return Err(format!("job inputs duplicate path {path}"));
        }
    }
    let cwd = job
        .pointer("/command/cwd")
        .and_then(Value::as_str)
        .ok_or("job command.cwd is missing or invalid")?;
    let tool_path = Path::new(cwd).join(STAGE_TOOL);
    let tool_path = tool_path.to_str().ok_or("stage tool path is not UTF-8")?;
    if !input_hashes.contains_key(tool_path) {
        return Err(format!("job inputs do not seal stage tool {tool_path}"));
    }
    Ok(JobIdentity {
        id,
        model,
        bits,
        lengths,
        mlx_source,
        mlx_lm_source,
        input_hashes,
    })
}

fn validate_receipt(receipt: &Value, job: &JobIdentity) -> Result<(Vec<ParsedCase>, u64), String> {
    if string(receipt, "schema")? != RECEIPT_SCHEMA || string(receipt, "status")? != MEASURED_STATUS
    {
        return Err("trial receipt has wrong schema or non-measured status".to_owned());
    }
    let model = object_field(receipt, "model")?;
    if string(model, "path")? != job.model
        || unsigned(model, "requested_weight_bits")? != job.bits
        || unsigned(model, "config_exposed_weight_bits")? != job.bits
    {
        return Err("trial receipt model identity does not match job arguments".to_owned());
    }
    require_input_hash(
        job,
        &format!("{}/config.json", job.model),
        string(model, "config_sha256")?,
    )?;
    require_input_hash(
        job,
        &format!("{}/model.safetensors.index.json", job.model),
        string(model, "model_index_sha256")?,
    )?;
    let mlx_lm = object_field(receipt, "mlx_lm_source")?;
    if string(mlx_lm, "source_root")? != job.mlx_lm_source
        || string(mlx_lm, "commit")? != PINNED_MLX_LM_COMMIT
        || mlx_lm
            .get("working_tree_matches_commit")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("MLX-LM source root does not match job arguments".to_owned());
    }
    require_input_hash(
        job,
        &format!("{}/mlx_lm/models/gemma4_text.py", job.mlx_lm_source),
        string(mlx_lm, "referenced_path_sha256")?,
    )?;
    let mlx = object_field(receipt, "mlx_protocol_source")?;
    if string(mlx, "source_root")? != job.mlx_source
        || string(mlx, "commit")? != PINNED_MLX_COMMIT
        || mlx
            .get("working_tree_matches_commit")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("MLX source root does not match job arguments".to_owned());
    }
    require_input_hash(
        job,
        &format!("{}/benchmarks/python/time_utils.py", job.mlx_source),
        string(mlx, "referenced_path_sha256")?,
    )?;
    let protocol = object_field(receipt, "protocol")?;
    let warmups = unsigned(protocol, "warmups")?;
    let iterations = unsigned(protocol, "iterations")?;
    if warmups != 5 || iterations != 100 {
        return Err(format!(
            "unexpected timing protocol: {warmups} warmups, {iterations} iterations"
        ));
    }

    let expected = expected_keys(&job.lengths);
    let planned = case_keys(array_field(receipt, "planned_cases")?, false)?;
    if planned != expected {
        return Err(coverage_error("planned_cases", &expected, &planned));
    }
    let raw_cases = array_field(receipt, "cases")?;
    let measured = case_keys(raw_cases, true)?;
    if measured != expected {
        return Err(coverage_error("cases", &expected, &measured));
    }
    let mut cases = Vec::with_capacity(raw_cases.len());
    for value in raw_cases {
        let timing = object_field(value, "timing")?;
        let total_ms = finite_nonnegative(timing, "total_ms")?;
        let mean_ms = finite_nonnegative(timing, "mean_ms")?;
        let expected_mean = total_ms / iterations as f64;
        let tolerance = expected_mean.abs().max(1.0) * 1e-9;
        if (mean_ms - expected_mean).abs() > tolerance {
            return Err("case mean_ms does not match total_ms / iterations".to_owned());
        }
        cases.push(ParsedCase {
            length: unsigned(value, "prompt_or_context_tokens")?,
            mode: string(value, "mode")?.to_owned(),
            category: string(value, "category")?.to_owned(),
            variant: optional_string(value, "variant")?,
            total_ms,
            mean_ms,
        });
    }
    Ok((cases, iterations))
}

fn require_input_hash(job: &JobIdentity, path: &str, actual: &str) -> Result<(), String> {
    match job.input_hashes.get(path) {
        Some(expected) if expected == actual => Ok(()),
        Some(expected) => Err(format!(
            "input identity mismatch for {path}: {actual} != {expected}"
        )),
        None => Err(format!("job inputs do not seal {path}")),
    }
}

type CaseKey = (u64, String, String, Option<String>);

fn expected_keys(lengths: &[u64]) -> BTreeSet<CaseKey> {
    lengths
        .iter()
        .flat_map(|length| {
            MODES.into_iter().flat_map(move |mode| {
                STAGES.into_iter().map(move |(category, variant)| {
                    (
                        *length,
                        mode.to_owned(),
                        category.to_owned(),
                        variant.map(str::to_owned),
                    )
                })
            })
        })
        .collect()
}

fn case_keys(values: &[Value], require_timing: bool) -> Result<BTreeSet<CaseKey>, String> {
    let mut keys = BTreeSet::new();
    for value in values {
        if require_timing && value.get("timing").is_none() {
            return Err("measured case is missing timing".to_owned());
        }
        let key = (
            unsigned(value, "prompt_or_context_tokens")?,
            string(value, "mode")?.to_owned(),
            string(value, "category")?.to_owned(),
            optional_string(value, "variant")?,
        );
        if !keys.insert(key) {
            return Err("duplicate case identity".to_owned());
        }
    }
    Ok(keys)
}

fn coverage_error(label: &str, expected: &BTreeSet<CaseKey>, actual: &BTreeSet<CaseKey>) -> String {
    format!(
        "{label} coverage mismatch: missing={:?}, unexpected={:?}",
        expected.difference(actual).collect::<Vec<_>>(),
        actual.difference(expected).collect::<Vec<_>>()
    )
}

struct Activity {
    samples: usize,
    samples_with_processes: usize,
    processes: Vec<ObservedProcess>,
}

fn parse_conditions(path: &Path) -> Result<Activity, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut samples = 0;
    let mut samples_with_processes = 0;
    let mut aggregate = BTreeMap::<String, (u64, u64)>::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("{} line {} is blank", path.display(), index + 1));
        }
        let value: Value = parse_strict_json(line.as_bytes())
            .map_err(|e| format!("{} line {}: {e}", path.display(), index + 1))?;
        let observed = value
            .get("observed_processes")
            .map(|value| value.as_array().ok_or("observed_processes is not an array"))
            .transpose()?
            .cloned()
            .unwrap_or_default();
        let mut within_sample = BTreeMap::<String, u64>::new();
        for process in observed {
            let name = string(&process, "name")?;
            if name.is_empty() || unsigned(&process, "pid")? == 0 {
                return Err("invalid observed process identity".to_owned());
            }
            *within_sample.entry(name.to_owned()).or_default() += 1;
        }
        if !within_sample.is_empty() {
            samples_with_processes += 1;
        }
        for (name, instances) in within_sample {
            let entry = aggregate.entry(name).or_default();
            entry.0 += 1;
            entry.1 = entry.1.max(instances);
        }
        samples += 1;
    }
    if samples == 0 {
        return Err("conditions journal contains no samples".to_owned());
    }
    Ok(Activity {
        samples,
        samples_with_processes,
        processes: aggregate
            .into_iter()
            .map(|(name, (sample_count, max_instances))| ObservedProcess {
                name,
                sample_count,
                max_instances,
            })
            .collect(),
    })
}

fn read_strict(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_strict_json(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn argument<'a>(args: &[&'a str], name: &str) -> Result<&'a str, String> {
    let matches = args
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            value.strip_prefix(&format!("{name}=")).or_else(|| {
                (*value == name)
                    .then(|| args.get(index + 1).copied())
                    .flatten()
            })
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [value] => Ok(*value),
        [] => Err(format!("job arguments missing {name}")),
        _ => Err(format!("job arguments duplicate {name}")),
    }
}

fn parse_u64_argument(args: &[&str], name: &str) -> Result<u64, String> {
    let raw = argument(args, name)?;
    raw.parse()
        .map_err(|e| format!("invalid {name} value {raw:?}: {e}"))
}

fn object_field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, String> {
    value
        .get(name)
        .filter(|value| value.is_object())
        .ok_or_else(|| format!("missing or invalid object {name}"))
}

fn array_field<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], String> {
    value
        .get(name)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing or invalid array {name}"))
}

fn string<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid string {name}"))
}

fn optional_string(value: &Value, name: &str) -> Result<Option<String>, String> {
    match value.get(name) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        _ => Err(format!("missing or invalid nullable string {name}")),
    }
}

fn unsigned(value: &Value, name: &str) -> Result<u64, String> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid unsigned integer {name}"))
}

fn boolean(value: &Value, name: &str) -> Result<bool, String> {
    value
        .get(name)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or invalid boolean {name}"))
}

fn finite_nonnegative(value: &Value, name: &str) -> Result<f64, String> {
    let number = value
        .get(name)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing or invalid number {name}"))?;
    if !number.is_finite() || number < 0.0 {
        return Err(format!("{name} must be finite and non-negative"));
    }
    Ok(number)
}

fn push_unique(values: &mut Vec<Value>, candidate: Option<&Value>) {
    if let Some(candidate) = candidate.filter(|value| !value.is_null()) {
        if !values.contains(candidate) {
            values.push(candidate.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn receipt(lengths: &[u64]) -> Value {
        let planned = expected_keys(lengths)
            .iter()
            .map(|(length, mode, category, variant)| {
                json!({
                    "prompt_or_context_tokens": length, "mode": mode,
                    "category": category, "variant": variant
                })
            })
            .collect::<Vec<_>>();
        let cases = planned
            .iter()
            .cloned()
            .map(|mut value| {
                value.as_object_mut().unwrap().insert(
                    "timing".to_owned(),
                    json!({"total_ms": 10.0, "mean_ms": 0.1}),
                );
                value
            })
            .collect::<Vec<_>>();
        json!({
            "schema": RECEIPT_SCHEMA, "status": MEASURED_STATUS,
            "model":{"path":"/model/q4","requested_weight_bits":4,"config_exposed_weight_bits":4,
                "config_sha256":"config-hash","model_index_sha256":"index-hash"},
            "mlx_lm_source":{"source_root":"/mlx-lm","commit":PINNED_MLX_LM_COMMIT,
                "working_tree_matches_commit":true,"referenced_path_sha256":"gemma-hash"},
            "mlx_protocol_source":{"source_root":"/mlx","commit":PINNED_MLX_COMMIT,
                "working_tree_matches_commit":true,"referenced_path_sha256":"time-hash"},
            "protocol":{"warmups":5,"iterations":100},
            "planned_cases":planned, "cases":cases
        })
    }

    fn job() -> JobIdentity {
        JobIdentity {
            id: "job".into(),
            model: "/model/q4".into(),
            bits: 4,
            lengths: vec![256],
            mlx_source: "/mlx".into(),
            mlx_lm_source: "/mlx-lm".into(),
            input_hashes: BTreeMap::from([
                ("/model/q4/config.json".into(), "config-hash".into()),
                (
                    "/model/q4/model.safetensors.index.json".into(),
                    "index-hash".into(),
                ),
                (
                    "/mlx-lm/mlx_lm/models/gemma4_text.py".into(),
                    "gemma-hash".into(),
                ),
                (
                    "/mlx/benchmarks/python/time_utils.py".into(),
                    "time-hash".into(),
                ),
            ]),
        }
    }

    #[test]
    fn validates_complete_stage_matrix() {
        let (cases, iterations) = validate_receipt(&receipt(&[256]), &job()).unwrap();
        assert_eq!(cases.len(), 22);
        assert_eq!(iterations, 100);
    }

    #[test]
    fn rejects_missing_duplicate_and_negative_cases() {
        let mut missing = receipt(&[256]);
        missing["cases"].as_array_mut().unwrap().pop();
        assert!(validate_receipt(&missing, &job())
            .unwrap_err()
            .contains("coverage mismatch"));

        let mut duplicate = receipt(&[256]);
        let first = duplicate["cases"][0].clone();
        duplicate["cases"].as_array_mut().unwrap().push(first);
        assert!(validate_receipt(&duplicate, &job())
            .unwrap_err()
            .contains("duplicate case"));

        let mut negative = receipt(&[256]);
        negative["cases"][0]["timing"]["mean_ms"] = json!(-1.0);
        assert!(validate_receipt(&negative, &job())
            .unwrap_err()
            .contains("non-negative"));
    }

    #[test]
    fn rejects_model_identity_and_strict_json_duplicates() {
        let mut wrong = receipt(&[256]);
        wrong["model"]["path"] = json!("/different");
        assert!(validate_receipt(&wrong, &job())
            .unwrap_err()
            .contains("model identity"));

        let mut wrong_commit = receipt(&[256]);
        wrong_commit["mlx_protocol_source"]["commit"] = json!("untrusted");
        assert!(validate_receipt(&wrong_commit, &job())
            .unwrap_err()
            .contains("MLX source root"));

        let duplicate = br#"{"schema":"a","schema":"b"}"#;
        assert!(parse_strict_json::<Value>(duplicate).is_err());
        assert!(parse_strict_json::<Value>(br#"{"x":1} {"x":2}"#).is_err());
        assert!(parse_strict_json::<Value>(br#"{"mean_ms":1e400}"#).is_err());
    }

    #[test]
    fn requires_the_invoked_stage_tool_to_be_sealed() {
        let mut value = json!({
            "id":"job", "command":{
                "cwd":"/workspace/v3",
                "args":[STAGE_TOOL,"--model","/model/q4","--weight-bits","4",
                    "--lengths","256","--mlx-source","/mlx","--mlx-lm-source","/mlx-lm"]
            },
            "inputs":[]
        });
        assert!(parse_job_identity(&value)
            .unwrap_err()
            .contains("do not seal stage tool"));
        value["inputs"] = json!([{
            "path":"/workspace/v3/tools/mlx_gemma4_stage_bench.py",
            "sha256":"sealed"
        }]);
        assert_eq!(parse_job_identity(&value).unwrap().bits, 4);
    }

    #[test]
    fn preserves_observed_process_counts() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("conditions.jsonl");
        fs::write(&path, concat!(
            "{\"observed_processes\":[]}\n",
            "{\"observed_processes\":[{\"pid\":7,\"name\":\"cargo\"}]}\n",
            "{\"observed_processes\":[{\"pid\":8,\"name\":\"cargo\"},{\"pid\":9,\"name\":\"cargo\"}]}\n"
        )).unwrap();
        let activity = parse_conditions(&path).unwrap();
        assert_eq!(activity.samples, 3);
        assert_eq!(activity.samples_with_processes, 2);
        assert_eq!(
            activity.processes,
            vec![ObservedProcess {
                name: "cargo".into(),
                sample_count: 2,
                max_instances: 2
            }]
        );
    }
}
