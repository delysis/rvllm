//! Offline, fail-closed analysis of a paired or ABBA accelerator campaign.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_AGE_MS: f64 = 2500.0;

fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(reason.into())
    }
}
fn eq(v: &Value, key: &str, expected: Value) -> Result<()> {
    require(
        v.get(key) == Some(&expected),
        &format!("invalid or missing {key}; expected {expected}"),
    )
}
fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or_else(|| format!("invalid or missing nonnegative {key}"))
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("missing array {key}"))
}
fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    require(
        bytes.len() as u64 <= MAX_BYTES,
        "input exceeds 64 MiB bound",
    )?;
    Ok(bytes)
}
fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&read_bytes(path)?).map_err(|e| format!("{}: {e}", path.display()))
}

fn actual_stratum(phase: &Value) -> Result<Value> {
    eq(phase, "schema", json!("rvllm.apple_phase_measurement.v1"))?;
    eq(phase, "observer_journal_error", Value::Null)?;
    let start = number(phase, "start_ms")?;
    let end = number(phase, "end_ms")?;
    let wall = number(phase, "wall_ms")?;
    require(
        end > start && wall > 0.0 && ((end - start) - wall).abs() <= 0.1,
        "invalid phase interval",
    )?;
    let samples = array(phase, "power_samples")?;
    let first = samples.first().ok_or("missing power samples")?;
    let controls = &first["controls"];
    require(
        matches!(controls["power_source"].as_str(), Some("ac" | "battery"))
            && controls["low_power_mode"].as_bool().is_some()
            && matches!(controls["pmset_power_mode"].as_u64(), Some(0..=2))
            && matches!(controls["thermal_state"].as_u64(), Some(0..=1)),
        "unknown or unsupported power controls",
    )?;
    for key in ["cpu_speed_limit_percent", "scheduler_limit_percent"] {
        require(
            matches!(controls.get(key), Some(Value::Null)) || controls[key].as_u64() == Some(100),
            "missing/restricted CPU or scheduler limit",
        )?;
    }
    require(
        matches!(controls.get("available_cpus"), Some(Value::Null))
            || controls["available_cpus"].as_u64().is_some_and(|n| n > 0),
        "missing/invalid CPU count",
    )?;
    let nominal = controls["thermal_state"] == 0;
    eq(phase, "sampled_controls_eligible", json!(nominal))?;
    eq(
        phase,
        "comparison_stratum",
        if nominal {
            controls.clone()
        } else {
            Value::Null
        },
    )?;
    for endpoint in ["start_host", "end_host"] {
        require(
            phase[endpoint]["thermal_state"] == controls["thermal_state"]
                && phase[endpoint]["low_power_mode"] == controls["low_power_mode"],
            "endpoint control transition",
        )?;
    }
    let first_end = number(first, "end_ms")?;
    require(
        first_end <= start && start - first_end <= MAX_AGE_MS,
        "stale/missing start observation",
    )?;
    let mut previous = None;
    for sample in samples {
        let s = number(sample, "start_ms")?;
        let e = number(sample, "end_ms")?;
        require(
            s <= e
                && s <= end
                && e <= end + MAX_AGE_MS
                && previous.is_none_or(|p| s >= p && e - p <= MAX_AGE_MS)
                && sample["controls"] == *controls,
            "changed/unordered/gapped power observations",
        )?;
        previous = Some(e);
    }
    require(
        end - previous.ok_or("missing last observation")? <= MAX_AGE_MS,
        "stale ending observation",
    )?;
    Ok(controls.clone())
}

fn ambient(conditions: &Value) -> Result<Value> {
    let mut servers = array(conditions, "idle_llama_servers")?.clone();
    servers.sort_by_key(Value::to_string);
    let mut quiet = array(conditions, "quiet_process_names")?.clone();
    quiet.sort_by_key(Value::to_string);
    Ok(json!({"idle_llama_servers":servers,"quiet_process_names":quiet}))
}

fn validate_queue(report: &Value, job: &Value, observations: &[Value]) -> Result<(Value, Value)> {
    eq(report, "schema", json!("rvllm.experiment_result.v1"))?;
    eq(job, "schema", json!("rvllm.experiment_job.v1"))?;
    require(
        job["id"]
            .as_str()
            .is_some_and(|s| !s.is_empty() && s.len() <= 96),
        "missing/invalid queue job ID",
    )?;
    eq(report, "id", job["id"].clone())?;
    for doc in [report, job] {
        eq(doc, "purpose", json!("timing"))?;
    }
    for (key, value) in [
        ("status", json!("succeeded")),
        ("exit_code", json!(0)),
        ("signal_or_missing_exit_code", json!(false)),
        ("sampled_conditions_eligible", json!(true)),
        ("overdue", json!(false)),
        ("files_unchanged", json!(true)),
        ("file_error", Value::Null),
    ] {
        eq(report, key, value)?;
    }
    require(
        array(report, "violations")?.is_empty(),
        "queue condition violations",
    )?;
    require(
        report.get("validation") == Some(&Value::Null)
            || (report["validation"]["success"] == true && report["validation"]["exit_code"] == 0),
        "validator failure/missing validation field",
    )?;
    let stratum = actual_stratum(&report["measurement"])?;
    let c = &job["conditions"];
    for key in [
        "power_source",
        "low_power_mode",
        "pmset_power_mode",
        "thermal_state",
    ] {
        require(
            c.get(key) == stratum.get(key),
            "actual stratum differs from requested conditions",
        )?;
    }
    let ambient = ambient(c)?;
    require(
        !observations.is_empty(),
        "missing execution conditions observations",
    )?;
    for observation in observations {
        eq(observation, "ready", json!(true))?;
        require(
            observation["power"]["sample"]["controls"] == stratum,
            "condition observation stratum differs",
        )?;
        eq(&observation["power"], "observer_journal_error", Value::Null)?;
        require(
            number(&observation["power"], "age_ms")? <= MAX_AGE_MS,
            "stale condition power sample",
        )?;
        require(
            array(observation, "competing_processes")?.is_empty(),
            "competing process observed",
        )?;
        require(
            number(observation, "probe_ms")? <= MAX_AGE_MS,
            "overdue condition probe",
        )?;
        require(
            number(observation, "free_bytes")? >= number(c, "minimum_free_bytes")?,
            "disk floor violated",
        )?;
        let checks = array(observation, "idle_server_checks")?;
        let expected = array(c, "idle_llama_servers")?;
        require(
            checks.len() == expected.len(),
            "idle-server check count mismatch",
        )?;
        for server in expected {
            require(
                checks
                    .iter()
                    .filter(|check| {
                        check["pid"] == server["pid"]
                            && check["port"] == server["port"]
                            && check["idle"] == true
                    })
                    .count()
                    == 1,
                "idle-server identity/status mismatch",
            )?;
        }
    }
    Ok((stratum, ambient))
}

fn ids(v: &Value, key: &str, count: usize) -> Result<Vec<u32>> {
    let values = array(v, key)?;
    require(values.len() == count, &format!("wrong {key} count"))?;
    values
        .iter()
        .map(|n| {
            n.as_u64()
                .filter(|n| *n < 262_144)
                .map(|n| n as u32)
                .ok_or_else(|| format!("invalid {key} token"))
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Ane,
    Native,
}
impl Kind {
    fn label(self) -> &'static str {
        match self {
            Self::Ane => "metal_ane",
            Self::Native => "native_kit",
        }
    }
}

#[derive(Clone)]
struct Trial {
    kind: Kind,
    stratum: Value,
    ambient: Value,
    identity: Value,
    prompt: Vec<u32>,
    output: Vec<u32>,
    summary: Value,
}

fn median(values: &[f64]) -> Result<f64> {
    require(
        !values.is_empty() && values.iter().all(|v| v.is_finite() && *v >= 0.0),
        "invalid median input",
    )?;
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    Ok(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    })
}

fn backend(kind: Kind, report: &Value) -> Result<(Vec<u32>, Vec<u32>, Value)> {
    let rows = match kind {
        Kind::Ane => {
            if let Some(capture) = report.get("ffn_input_capture_enabled") {
                require(
                    capture == &json!(false),
                    "FFN input capture enabled in timing result",
                )?;
            }
            for (key, value) in [
                ("ane_cache_policy", json!("require-existing")),
                ("ane_compile_budget", json!(0)),
                ("ane_compile_budget_used", json!(0)),
                ("ane_execution_verified", json!(true)),
                ("cpu_or_gpu_decode_fallback", json!(false)),
                ("diagnostic_journal_enabled", json!(false)),
                ("layer_state_capture_enabled", json!(false)),
                ("loaded_ane_programs", json!(162)),
                ("inference_complete", json!(true)),
                ("requests_completed", json!(9)),
                ("global_context_capacity", json!(1024)),
                ("case_history", json!("all-requests")),
                ("execution_order", json!("prefill-decode-per-request")),
                ("metal_residency", json!("retained-through-ane-decode")),
            ] {
                eq(report, key, value)?;
            }
            require(
                matches!(
                    report["ane_weight_plan"].as_str(),
                    Some("static-int8-ffn-cached" | "static-int8-stacked-ffn-cached")
                ),
                "unsupported ANE precision/plan",
            )?;
            require(report["model_dir"].as_str().is_some_and(|s|s.ends_with("/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")),"unexpected standard-checkpoint identity")?;
            array(report, "cases")?
        }
        Kind::Native => {
            for (key, value) in [
                ("schema", json!("rvllm_llama_native_kit_baseline_v1")),
                ("status", json!("complete")),
                ("backend", json!("llama_cpp_native_kit_pin")),
                ("failure", Value::Null),
                ("warmups_requested", json!(2)),
                ("measured_repetitions_requested", json!(7)),
            ] {
                eq(report, key, value)?;
            }
            for (key, value) in [
                ("warmups", json!(2)),
                ("measured_repetitions", json!(7)),
                ("context_tokens", json!(1024)),
                ("model_vocab", json!(262144)),
            ] {
                eq(&report["configuration"], key, value)?;
            }
            eq(
                &report["preparation"],
                "expected_model_sha256",
                json!("93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"),
            )?;
            eq(
                &report["preparation"],
                "bindings_commit",
                json!("a3cf95eb1d4fa748480eb780e6fcbfc1a5c1c391"),
            )?;
            eq(
                &report["preparation"],
                "llama_cpp_commit",
                json!("5f55650a78f92aff4d48d671423e888fac0469ff"),
            )?;
            array(report, "repetitions")?
        }
    };
    require(
        rows.len() == 9,
        "need exactly two warmups and seven measured requests",
    )?;
    let mut shared: Option<(Vec<u32>, Vec<u32>)> = None;
    let (mut pp, mut imports, mut dec, mut sums, mut inner_pp, mut inner_dec) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for (index, row) in rows.iter().enumerate() {
        let prompt = ids(row, "prompt_token_ids", 84)?;
        require(prompt.first() == Some(&2), "missing BOS")?;
        let output = ids(
            row,
            if kind == Kind::Ane {
                "generated_tokens"
            } else {
                "output_token_ids"
            },
            10,
        )?;
        if let Some((p, o)) = &shared {
            require(*p == prompt && *o == output, "repetition token IDs differ")?;
        } else {
            shared = Some((prompt, output.clone()));
        }
        let (p, i, d) = match kind {
            Kind::Ane => {
                eq(row, "ane_decode_steps", json!(9))?;
                eq(row, "metal_decode_steps", json!(0))?;
                eq(row, "prefill_command_buffers", json!(1))?;
                let steps = array(row, "steps")?;
                require(steps.len() == 9, "wrong ANE step count")?;
                let mut d = 0.0;
                for (j, step) in steps.iter().enumerate() {
                    eq(step, "position", json!(84 + j))?;
                    eq(step, "input_token", json!(output[j]))?;
                    eq(step, "next_token", json!(output[j + 1]))?;
                    require(
                        array(step, "layer_states")?.is_empty(),
                        "layer capture in timing result",
                    )?;
                    if step.get("ffn_inputs").is_some() {
                        require(
                            array(step, "ffn_inputs")?.is_empty(),
                            "FFN input capture in timing step",
                        )?;
                    }
                    let ms = number(step, "total_ms")?;
                    require(ms > 0.0, "nonpositive ANE step")?;
                    d += ms;
                }
                (
                    number(row, "prefill_sample_capture_ms")?,
                    number(row, "ane_cache_import_ms")?,
                    d,
                )
            }
            Kind::Native => {
                for (key, value) in [
                    ("index", json!(index)),
                    ("warmup", json!(index < 2)),
                    ("n_generated", json!(10)),
                    ("actual_decode_steps", json!(9)),
                    ("requested_decode_steps", json!(9)),
                    ("requested_output_tokens", json!(10)),
                    ("completed_decode_steps", json!(9)),
                    ("output_token_count", json!(10)),
                    ("prompt_tokens", json!(84)),
                    ("internal_counters_match", json!(true)),
                    ("matches_first_repetition", json!(true)),
                    ("work_matches_84_prefill_9_decode", json!(true)),
                    ("internal_prompt_tokens", json!(84)),
                    ("internal_decode_runs_raw", json!(9)),
                ] {
                    eq(row, key, value)?;
                }
                require(
                    row["last_token_is_eog"].as_bool().is_some(),
                    "missing EOG result",
                )?;
                require(
                    (row["last_token_is_eog"] == true && row["finish_reason"] == "eog")
                        || (row["last_token_is_eog"] == false
                            && row["finish_reason"] == "max_tokens"),
                    "inconsistent finish reason",
                )?;
                let each = array(row, "decode_eval_wall_ms_each")?;
                require(
                    each.len() == 9
                        && each
                            .iter()
                            .all(|v| v.as_f64().is_some_and(|n| n.is_finite() && n > 0.0)),
                    "invalid native eval times",
                )?;
                let each_sum = each.iter().filter_map(Value::as_f64).sum::<f64>();
                require(
                    (each_sum - number(row, "decode_eval_wall_sum_ms")?).abs() <= 0.01,
                    "native eval sum mismatch",
                )?;
                let d = number(row, "decode_wall_ms")?;
                require(d + 0.01 >= each_sum, "decode wall shorter than evaluations")?;
                if index >= 2 {
                    inner_pp.push(number(row, "internal_prompt_ms")?);
                    inner_dec.push(number(row, "internal_decode_ms")?);
                }
                (number(row, "prefill_wall_ms")?, 0.0, d)
            }
        };
        require(
            p > 0.0 && d > 0.0 && (kind != Kind::Ane || i > 0.0),
            "nonpositive timing phase",
        )?;
        if index >= 2 {
            pp.push(p);
            imports.push(i);
            dec.push(d);
            sums.push(p + i + d);
        }
    }
    let prefill_import: Vec<_> = pp.iter().zip(&imports).map(|(p, i)| p + i).collect();
    let decode = median(&dec)?;
    require(
        (9000.0 / decode).is_finite(),
        "nonfinite derived decode rate",
    )?;
    let (prompt, output) = shared.ok_or("missing requests")?;
    let summary = json!({"backend":kind.label(),"warmups_discarded":2,"measured_requests":7,
        "prompt_tokens":84,"output_tokens":10,"actual_decode_steps_per_request":9,
        "median_prefill_ms":median(&pp)?,"median_ane_import_ms":median(&imports)?,
        "median_prefill_plus_import_ms":median(&prefill_import)?,"median_decode_ms":decode,
        "median_phase_sum_ms":median(&sums)?,"actual_decode_steps_per_second":9000.0/decode,
        "median_internal_prompt_ms":if inner_pp.is_empty(){None}else{Some(median(&inner_pp)?)},
        "median_internal_decode_ms":if inner_dec.is_empty(){None}else{Some(median(&inner_dec)?)},
        "per_request_phase_sum_ms":sums,
        "precision":if kind==Kind::Ane {"standard BF16 checkpoint; Metal BF16 prefill; INT8 FFNs and other FP16 ANE weights"}else{"Google native-QAT checkpoint; Q4_0 projections and Q6_K embedding/head"},
        "ane_weight_plan":report.get("ane_weight_plan"),
        "scope":if kind==Kind::Ane {"PP includes first sampling and KV export; import separate; decode sums step total_ms, excluding outer streaming/report overhead."}else{"PP ends at synchronized logits before first sampling; decode_wall_ms includes subsequent sampling/loop overhead; reset and first sampling are outside phase sum."}});
    Ok((prompt, output, summary))
}

fn load(directory: &Path) -> Result<Trial> {
    let report = read_json(&directory.join("report.json"))?;
    let job = read_json(&directory.join("job.json"))?;
    let observations =
        serde_json::Deserializer::from_slice(&read_bytes(&directory.join("conditions.jsonl"))?)
            .into_iter::<Value>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
    let (stratum, ambient) = validate_queue(&report, &job, &observations)?;
    let ane = directory.join("backend/report.json");
    let native = directory.join("backend.json");
    require(
        ane.is_file() != native.is_file(),
        "need exactly one supported backend report",
    )?;
    let kind = if ane.is_file() {
        Kind::Ane
    } else {
        Kind::Native
    };
    let data = read_json(if kind == Kind::Ane { &ane } else { &native })?;
    let (prompt, output, mut summary) = backend(kind, &data)?;
    summary["queue_directory"] = json!(directory);
    summary["job_id"] = job["id"].clone();
    summary["executable"] = job["command"]["executable"].clone();
    let identity = json!({"command":job["command"],"inputs":job["inputs"],"config_sha256":data.get("config_sha256")});
    Ok(Trial {
        kind,
        stratum,
        ambient,
        identity,
        prompt,
        output,
        summary,
    })
}

fn compare(runs: &[Trial]) -> Result<Value> {
    require(
        matches!(runs.len(), 2 | 4),
        "need two directories or ABBA four directories",
    )?;
    let first = &runs[0];
    let mut seen_jobs = BTreeSet::new();
    let mut seen_directories = BTreeSet::new();
    for run in runs {
        // These fields are supplied by load(), never by the backend receipt.
        if let Some(id) = run.summary.get("job_id") {
            require(
                seen_jobs.insert(id.to_string()),
                "reused job identity in paired campaign",
            )?;
        }
        if let Some(path) = run.summary.get("queue_directory") {
            require(
                seen_directories.insert(path.to_string()),
                "reused result directory in paired campaign",
            )?;
        }
    }
    require(runs[1].kind != first.kind, "paired backends must differ")?;
    if runs.len() == 4 {
        require(
            first.kind == Kind::Ane,
            "predeclared ABBA gate requires outer ANE processes",
        )?;
        require(
            runs[2].kind == runs[1].kind && runs[3].kind == first.kind,
            "four directories must be ABBA order",
        )?;
        require(
            runs[0].identity == runs[3].identity && runs[1].identity == runs[2].identity,
            "within-backend ABBA configuration/input identity drift",
        )?;
    }
    for run in runs.iter().skip(1) {
        require(
            run.stratum == first.stratum,
            "actual power/processor strata differ",
        )?;
        require(
            run.ambient == first.ambient,
            "ambient idle-server/quiet-process policies differ",
        )?;
        require(
            run.prompt == first.prompt && run.output == first.output,
            "prompt/output IDs differ across backends; matched-work ratio refused",
        )?;
    }
    let mut drift_details = Value::Null;
    if runs.len() == 4 {
        let mut details = serde_json::Map::new();
        for key in [
            "median_prefill_plus_import_ms",
            "median_decode_ms",
            "median_phase_sum_ms",
        ] {
            let a = number(&runs[0].summary, key)?;
            let b = number(&runs[3].summary, key)?;
            require(a > 0.0, "zero drift denominator")?;
            let value = (b - a).abs() / a;
            require(value.is_finite(), "nonfinite outer ANE drift")?;
            details.insert(key.into(), json!(value));
        }
        drift_details = Value::Object(details);
    }
    let aggregate = |kind: Kind, key: &str| -> Result<f64> {
        median(
            &runs
                .iter()
                .filter(|r| r.kind == kind)
                .map(|r| number(&r.summary, key))
                .collect::<Result<Vec<_>>>()?,
        )
    };
    let mut ratios = serde_json::Map::new();
    let mut eligibility = serde_json::Map::new();
    for key in [
        "median_prefill_plus_import_ms",
        "median_decode_ms",
        "median_phase_sum_ms",
    ] {
        let drift = drift_details.get(key).and_then(Value::as_f64);
        let passed = drift.map(|value| value <= 0.05);
        // An unbracketed pair may report an exploratory ratio, but never claims
        // to have assessed or passed the predeclared drift gate.
        let eligible = passed.unwrap_or(true);
        eligibility.insert(key.into(), json!({
            "ratio_eligible":eligible,"drift_assessed":drift.is_some(),
            "outer_ane_absolute_relative_drift":drift,"drift_gate_passed":passed,
            "reason":match passed {Some(true)=>"within_5_percent",Some(false)=>"exceeds_5_percent",None=>"single_pair_drift_not_assessed"}
        }));
        if eligible {
            let a = aggregate(Kind::Ane, key)?;
            let b = aggregate(Kind::Native, key)?;
            require(a > 0.0, "zero ratio denominator")?;
            require((b / a).is_finite(), "nonfinite latency ratio")?;
            ratios.insert(key.into(), json!(b / a));
        } else {
            ratios.insert(key.into(), Value::Null);
        }
    }
    let primary_passed = eligibility["median_decode_ms"]["drift_gate_passed"].as_bool();
    let all_eligible = eligibility
        .values()
        .all(|value| value["ratio_eligible"] == true);
    Ok(json!({"schema":"rvllm.paired_native_ane_analysis.v2",
        "status":if runs.len()==2{"exploratory_single_process_pair"}else if primary_passed==Some(false){"inconclusive_primary_decode_drift"}else if all_eligible{"exploratory_abba"}else{"exploratory_abba_partial_metric_eligibility"},
        "actual_stratum":first.stratum,"ambient_policy":first.ambient,
        "thermal_class":if first.stratum["thermal_state"]==0{"nominal"}else{"exploratory_stable_fair"},
        "prompt_token_ids":first.prompt,"output_token_ids":first.output,
        "processes":runs.iter().map(|r|&r.summary).collect::<Vec<_>>(),
        "native_over_ane_process_median_latency_ratios":ratios,
        "per_metric_eligibility":eligibility,
        "primary_gate":{"metric":"median_decode_ms","backend":"metal_ane","outer_process_indices":if runs.len()==4{Some([0,3])}else{None},"assessed":runs.len()==4,"passed":primary_passed,"threshold":0.05},
        "ratio_definition":"native latency / ANE-path latency; >1 means the ANE configuration took less time for the stated phase scope",
        "outer_a_absolute_relative_drift":drift_details,"drift_threshold":0.05,
        "interpretation":"Outer ANE decode drift is the predeclared primary gate; each secondary latency ratio has its own drift gate. Different checkpoints, coefficient precision, implementations and devices. No hardware-only or quantization-only claim. Seven repeated requests are not seven independent process trials. Phase sums are not complete end-to-end wall time; see each backend scope. Sampled stable controls do not prove fixed clocks, and CPU counters cannot normalize GPU/ANE time."}))
}

fn main() {
    let directories: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let result = if matches!(directories.len(), 2 | 4) {
        directories
            .iter()
            .map(|p| {
                p.canonicalize()
                    .map_err(|e| e.to_string())
                    .and_then(|p| load(&p))
                    .map_err(|e| format!("{}: {e}", p.display()))
            })
            .collect::<Result<Vec<_>>>()
            .and_then(|runs| compare(&runs))
    } else {
        Err("usage: analyzer DIR_A DIR_B [DIR_B DIR_A] > NEW_ANALYSIS.json".into())
    };
    let failed = result.is_err();
    let report=result.unwrap_or_else(|reason|json!({"schema":"rvllm.paired_native_ane_analysis.v2","status":"refused","reason":reason,"ratios":null}));
    let mut out = std::io::stdout().lock();
    if serde_json::to_writer_pretty(&mut out, &report)
        .and_then(|_| out.write_all(b"\n").map_err(serde_json::Error::io))
        .is_err()
        || failed
    {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn phase() -> Value {
        let c = json!({"power_source":"ac","low_power_mode":true,"pmset_power_mode":1,"thermal_state":0,
            "cpu_speed_limit_percent":null,"scheduler_limit_percent":null,"available_cpus":null});
        json!({"schema":"rvllm.apple_phase_measurement.v1","observer_journal_error":null,
            "sampled_controls_eligible":true,"comparison_stratum":c,"start_ms":20.0,"end_ms":1500.0,"wall_ms":1480.0,
            "start_host":{"thermal_state":0,"low_power_mode":true},"end_host":{"thermal_state":0,"low_power_mode":true},
            "power_samples":[{"start_ms":0.0,"end_ms":10.0,"controls":c},{"start_ms":1000.0,"end_ms":1010.0,"controls":c}]})
    }
    fn queue() -> (Value, Value, Vec<Value>) {
        let p = phase();
        let c = p["comparison_stratum"].clone();
        let report = json!({"schema":"rvllm.experiment_result.v1","id":"fixture","purpose":"timing","status":"succeeded",
            "exit_code":0,"signal_or_missing_exit_code":false,"sampled_conditions_eligible":true,"overdue":false,
            "files_unchanged":true,"file_error":null,"validation":null,"violations":[],"measurement":p});
        let mut conditions = c.clone();
        conditions["idle_llama_servers"] = json!([]);
        conditions["quiet_process_names"] = json!([]);
        conditions["minimum_free_bytes"] = json!(100);
        let job = json!({"schema":"rvllm.experiment_job.v1","id":"fixture","purpose":"timing","conditions":conditions});
        let observations = vec![
            json!({"ready":true,"power":{"sample":{"controls":c},"age_ms":1.0,"observer_journal_error":null},"competing_processes":[],"probe_ms":10.0,"free_bytes":1000,"idle_server_checks":[]}),
        ];
        (report, job, observations)
    }
    fn native() -> Value {
        let rows:Vec<_>=(0..9).map(|i|json!({"index":i,"warmup":i<2,"prompt_token_ids":vec![2;84],"output_token_ids":vec![8;10],
            "n_generated":10,"actual_decode_steps":9,"requested_decode_steps":9,"requested_output_tokens":10,"completed_decode_steps":9,
            "output_token_count":10,"prompt_tokens":84,"internal_counters_match":true,"matches_first_repetition":true,
            "work_matches_84_prefill_9_decode":true,"internal_prompt_tokens":84,"internal_decode_runs_raw":9,
            "last_token_is_eog":false,"finish_reason":"max_tokens","decode_eval_wall_ms_each":vec![8.0;9],
            "decode_eval_wall_sum_ms":72.0,"decode_wall_ms":90.0,"prefill_wall_ms":20.0,"internal_prompt_ms":18.0,"internal_decode_ms":70.0})).collect();
        json!({"schema":"rvllm_llama_native_kit_baseline_v1","status":"complete","backend":"llama_cpp_native_kit_pin","failure":null,
            "warmups_requested":2,"measured_repetitions_requested":7,
            "configuration":{"warmups":2,"measured_repetitions":7,"context_tokens":1024,"model_vocab":262144},
            "preparation":{"expected_model_sha256":"93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b",
                "bindings_commit":"a3cf95eb1d4fa748480eb780e6fcbfc1a5c1c391","llama_cpp_commit":"5f55650a78f92aff4d48d671423e888fac0469ff"},"repetitions":rows})
    }
    fn trial(kind: Kind) -> Trial {
        let (prompt, output, summary) = backend(Kind::Native, &native()).unwrap();
        Trial {
            kind,
            stratum: phase()["comparison_stratum"].clone(),
            ambient: json!({}),
            identity: json!({}),
            prompt,
            output,
            summary,
        }
    }
    #[test]
    fn false_eligibility_and_file_drift_are_rejected() {
        let (r, j, o) = queue();
        assert!(validate_queue(&r, &j, &o).is_ok());
        for key in ["sampled_conditions_eligible", "files_unchanged"] {
            let mut bad = r.clone();
            bad[key] = json!(false);
            assert!(validate_queue(&bad, &j, &o).is_err());
        }
        let mut bad = r;
        bad["measurement"]["sampled_controls_eligible"] = json!(false);
        assert!(validate_queue(&bad, &j, &o).is_err());
    }
    #[test]
    fn stale_and_gapped_samples_are_rejected() {
        let mut p = phase();
        p["start_ms"] = json!(3000.0);
        p["end_ms"] = json!(4500.0);
        p["wall_ms"] = json!(1500.0);
        assert!(actual_stratum(&p).is_err());
        let mut p = phase();
        p["end_ms"] = json!(5000.0);
        p["wall_ms"] = json!(4980.0);
        p["power_samples"][1]["start_ms"] = json!(4000.0);
        p["power_samples"][1]["end_ms"] = json!(4010.0);
        assert!(actual_stratum(&p).is_err());
    }
    #[test]
    fn wrong_counts_and_repeated_ids_are_rejected() {
        assert!(backend(Kind::Native, &native()).is_ok());
        for key in [
            "n_generated",
            "actual_decode_steps",
            "internal_decode_runs_raw",
        ] {
            let mut n = native();
            n["repetitions"][3][key] = json!(8);
            assert!(backend(Kind::Native, &n).is_err());
        }
        let mut n = native();
        n["repetitions"][4]["output_token_ids"][0] = json!(9);
        assert!(backend(Kind::Native, &n).is_err());
    }
    #[test]
    fn differing_strata_or_cross_backend_ids_refuse_ratios() {
        let a = trial(Kind::Ane);
        let b = trial(Kind::Native);
        assert!(compare(&[a.clone(), b.clone()]).is_ok());
        let mut bad = b.clone();
        bad.stratum["thermal_state"] = json!(1);
        assert!(compare(&[a.clone(), bad]).is_err());
        let mut bad = b;
        bad.output[0] = 9;
        assert!(compare(&[a, bad]).is_err());
    }
    #[test]
    fn primary_decode_drift_does_not_suppress_stable_secondary_ratio() {
        let a = trial(Kind::Ane);
        let b = trial(Kind::Native);
        let mut end = a.clone();
        end.summary["median_decode_ms"] = json!(100.0);
        end.summary["median_phase_sum_ms"] = json!(120.0);
        let value = compare(&[a, b.clone(), b, end]).unwrap();
        assert_eq!(value["status"], "inconclusive_primary_decode_drift");
        assert_eq!(value["primary_gate"]["passed"], false);
        assert!(
            value["native_over_ane_process_median_latency_ratios"]["median_decode_ms"].is_null()
        );
        assert!(value["native_over_ane_process_median_latency_ratios"]
            ["median_prefill_plus_import_ms"]
            .is_number());
    }

    #[test]
    fn isolated_prefill_drift_preserves_stable_decode_ratio() {
        let a = trial(Kind::Ane);
        let b = trial(Kind::Native);
        let mut end = a.clone();
        end.summary["median_prefill_plus_import_ms"] = json!(40.0);
        end.summary["median_phase_sum_ms"] = json!(130.0);
        let value = compare(&[a, b.clone(), b, end]).unwrap();
        assert_eq!(
            value["status"],
            "exploratory_abba_partial_metric_eligibility"
        );
        assert_eq!(value["primary_gate"]["passed"], true);
        assert_eq!(
            value["per_metric_eligibility"]["median_decode_ms"]
                ["outer_ane_absolute_relative_drift"],
            0.0
        );
        assert_eq!(
            value["per_metric_eligibility"]["median_decode_ms"]["ratio_eligible"],
            true
        );
        assert_eq!(
            value["native_over_ane_process_median_latency_ratios"]["median_decode_ms"],
            1.0
        );
        for key in ["median_prefill_plus_import_ms", "median_phase_sum_ms"] {
            assert_eq!(
                value["per_metric_eligibility"][key]["ratio_eligible"],
                false
            );
            assert!(value["native_over_ane_process_median_latency_ratios"][key].is_null());
        }
    }

    #[test]
    fn primary_gate_requires_outer_ane_and_unbracketed_pair_is_unassessed() {
        let a = trial(Kind::Ane);
        let b = trial(Kind::Native);
        assert!(compare(&[b.clone(), a.clone(), a.clone(), b.clone()]).is_err());
        let pair = compare(&[a, b]).unwrap();
        assert_eq!(pair["primary_gate"]["assessed"], false);
        assert!(pair["primary_gate"]["passed"].is_null());
    }
    #[test]
    fn stable_fair_is_explicit_and_not_nominal() {
        let mut p = phase();
        p["sampled_controls_eligible"] = json!(false);
        p["comparison_stratum"] = Value::Null;
        p["start_host"]["thermal_state"] = json!(1);
        p["end_host"]["thermal_state"] = json!(1);
        for s in p["power_samples"].as_array_mut().unwrap() {
            s["controls"]["thermal_state"] = json!(1);
        }
        assert_eq!(actual_stratum(&p).unwrap()["thermal_state"], 1);
    }

    #[test]
    fn reused_receipts_are_not_independent_trials() {
        let mut a = trial(Kind::Ane);
        let mut b = trial(Kind::Native);
        a.summary["job_id"] = json!("same");
        b.summary["job_id"] = json!("same");
        assert!(compare(&[a, b]).is_err());
    }

    #[test]
    fn ane_step_work_and_zero_compile_contract_are_enforced() {
        let steps: Vec<_> = (0..9)
            .map(|i| {
                json!({"position":84+i,"input_token":8,"next_token":8,
            "layer_states":[],"total_ms":10.0})
            })
            .collect();
        let case = json!({"prompt_token_ids":vec![2;84],"generated_tokens":vec![8;10],
            "ane_decode_steps":9,"metal_decode_steps":0,"prefill_command_buffers":1,
            "prefill_sample_capture_ms":60.0,"ane_cache_import_ms":5.0,"steps":steps});
        let a = json!({"ane_cache_policy":"require-existing","ane_compile_budget":0,"ane_compile_budget_used":0,
            "ane_execution_verified":true,"cpu_or_gpu_decode_fallback":false,"diagnostic_journal_enabled":false,
            "layer_state_capture_enabled":false,"loaded_ane_programs":162,"inference_complete":true,
            "requests_completed":9,"global_context_capacity":1024,"case_history":"all-requests",
            "execution_order":"prefill-decode-per-request","metal_residency":"retained-through-ane-decode",
            "ane_weight_plan":"static-int8-ffn-cached",
            "model_dir":"/fixture/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            "cases":vec![case;9]});
        let (_, _, summary) = backend(Kind::Ane, &a).unwrap();
        assert_eq!(summary["median_phase_sum_ms"], 155.0);
        for (key, value) in [
            ("ane_compile_budget_used", json!(1)),
            ("diagnostic_journal_enabled", json!(true)),
            ("loaded_ane_programs", json!(161)),
            ("ffn_input_capture_enabled", json!(true)),
        ] {
            let mut bad = a.clone();
            bad[key] = value;
            assert!(backend(Kind::Ane, &bad).is_err());
        }
        let mut bad = a.clone();
        bad["cases"][5]["steps"][0]["ffn_inputs"] = json!([{"file":"captured.fp16"}]);
        assert!(backend(Kind::Ane, &bad).is_err());
        let mut uncaptured = a.clone();
        uncaptured["ffn_input_capture_enabled"] = json!(false);
        assert!(backend(Kind::Ane, &uncaptured).is_ok());
        let mut bad = a;
        bad["cases"][5]["steps"][0]["position"] = json!(85);
        assert!(backend(Kind::Ane, &bad).is_err());
    }
}
