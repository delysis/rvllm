//! Fail-closed numerical-only comparison of six new, immutable prefill jobs.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "2f212a4b1c316e49b52b78fc80e33a62c35772f0028e4da7d52ee7268c3d008a";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const ARMS: [(&str, &str, &str, &str); 6] = [
    (
        "observatory",
        "hf",
        "observatory-hf-job.json",
        "e39d87c30533067c35065ae745469ce48aa49a53a2b550fab8d89193fbec8382",
    ),
    (
        "observatory",
        "off",
        "observatory-off-job.json",
        "c30da8b58dcc01c377707475985a74f3f732576dc8b55b887efe2f255b628079",
    ),
    (
        "observatory",
        "combined",
        "observatory-combined-job.json",
        "d85cc3b0f6fdf8a06a194dd683060bc897bba94b774bde61a2fc6da2d8589b4e",
    ),
    (
        "library",
        "hf",
        "library-hf-job.json",
        "e5addab1cab7c4bac3cbab4b2df467cc58a44332545a2fb382967af5d4564390",
    ),
    (
        "library",
        "off",
        "library-off-job.json",
        "79b6d2e5fd858058e520c55a70d7ec668f79f2f535d193166e771a85dc1c491c",
    ),
    (
        "library",
        "combined",
        "library-combined-job.json",
        "361c7bd1b9b3c0caf7a0f470aa3249f9225319f24cfc5e4b8231d734a5020ccf",
    ),
];

struct Arm {
    score: Value,
    timing_eligible: bool,
    conditions: Value,
    receipts: BTreeMap<&'static str, String>,
}

fn main() {
    let args = env::args_os().map(PathBuf::from).collect::<Vec<_>>();
    if args.len() != 11 {
        eprintln!("usage: rvllm_gemma4_distinct_reference_summary OUTPUT.json TOKENIZER.json SOURCE.json MANIFEST_DIR OBS_HF_DIR OBS_OFF_DIR OBS_COMBINED_DIR LIB_HF_DIR LIB_OFF_DIR LIB_COMBINED_DIR");
        std::process::exit(2);
    }
    let paths = &args[1..];
    let result = run(paths).and_then(|summary| {
        let mut bytes = serde_json::to_vec_pretty(&summary).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&paths[0])
            .and_then(|mut file| file.write_all(&bytes))
            .map_err(|error| format!("{}: {error}", paths[0].display()))
    });
    if let Err(error) = result {
        eprintln!("rvllm_gemma4_distinct_reference_summary: {error}");
        std::process::exit(1);
    }
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|sha| sha.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn read(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse_strict_json(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value.get(key).ok_or_else(|| format!("missing {key}"))
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not text"))
}

fn number(value: &Value, key: &str) -> Result<f64> {
    field(value, key)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{key} is not finite numeric"))
}

fn ids(value: &Value, key: &str) -> Result<Vec<u32>> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not an array"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("{key} contains an invalid token"))
        })
        .collect()
}

fn arg<'a>(job: &'a Value, flag: &str) -> Result<&'a str> {
    let args = job["command"]["args"]
        .as_array()
        .ok_or("command args missing")?;
    let matches = args
        .windows(2)
        .filter(|pair| pair[0].as_str() == Some(flag))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected one {flag}"));
    }
    matches[0][1]
        .as_str()
        .ok_or_else(|| format!("{flag} is not text"))
}

fn expected_case(
    tokenizer: &tokenizers::Tokenizer,
    source: &Value,
    id: &str,
) -> Result<(Vec<u32>, u32)> {
    let cases = field(source, "cases")?.as_array().ok_or("cases missing")?;
    let found = cases
        .iter()
        .filter(|case| case["id"].as_str() == Some(id))
        .collect::<Vec<_>>();
    if found.len() != 1 {
        return Err(format!("expected one source case {id}"));
    }
    let prompt = text(found[0], "prompt")?;
    let full = format!("{}{}", prompt, text(found[0], "continuation")?);
    let encode = |s: &str| -> Result<Vec<u32>> {
        let mut tokens = vec![2];
        tokens.extend_from_slice(
            tokenizer
                .encode(s, false)
                .map_err(|e| e.to_string())?
                .get_ids(),
        );
        Ok(tokens)
    };
    let prompt_ids = encode(prompt)?;
    let full_ids = encode(&full)?;
    let suffix = full_ids
        .strip_prefix(prompt_ids.as_slice())
        .filter(|s| !s.is_empty())
        .ok_or("continuation retokenized at boundary")?;
    Ok((prompt_ids, suffix[0]))
}

// Only this specific monitoring defect is admissible for numerical-only work.
// The queue's timing eligibility verdict remains unchanged and is reported.
fn stale_power_only(observation: &Value, job: &Value) -> bool {
    let power = &observation["power"];
    let controls = &power["sample"]["controls"];
    let conditions = &job["conditions"];
    let Some(age) = power["age_ms"].as_f64() else {
        return false;
    };
    let Some(free) = observation["free_bytes"].as_u64() else {
        return false;
    };
    observation["ready"] == false
        && age.is_finite()
        && age > 2500.0
        && power.get("observer_journal_error") == Some(&Value::Null)
        && free
            >= conditions["minimum_free_bytes"]
                .as_u64()
                .unwrap_or(u64::MAX)
        && controls["power_source"] == conditions["power_source"]
        && controls["low_power_mode"].as_bool().is_some_and(|value| {
            conditions["low_power_mode"]
                .as_bool()
                .is_none_or(|required| value == required)
        })
        && controls["pmset_power_mode"].as_u64().is_some_and(|mode| {
            mode <= 2
                && conditions["pmset_power_mode"]
                    .as_u64()
                    .is_none_or(|required| mode == required)
        })
        && controls["thermal_state"].as_u64().is_some_and(|state| {
            state <= 3
                && conditions["thermal_state"]
                    .as_u64()
                    .is_none_or(|required| state == required)
        })
        && ["cpu_speed_limit_percent", "scheduler_limit_percent"]
            .iter()
            .all(|key| {
                controls.get(*key) == Some(&Value::Null) || controls[*key].as_u64() == Some(100)
            })
        && (controls.get("available_cpus") == Some(&Value::Null)
            || controls["available_cpus"]
                .as_u64()
                .is_some_and(|count| count > 0))
        && observation["competing_processes"]
            .as_array()
            .is_some_and(Vec::is_empty)
        && observation["observed_processes"].as_array().is_some()
        && observation["idle_server_checks"]
            .as_array()
            .is_some_and(Vec::is_empty)
        && observation["probe_ms"]
            .as_f64()
            .is_some_and(|ms| ms.is_finite() && (0.0..=2500.0).contains(&ms))
        && (observation["activity_sampled"] == false
            || (observation["activity_sampled"] == true
                && observation["raw_sample_age_ms"]
                    .as_f64()
                    .is_some_and(|ms| ms.is_finite() && (0.0..=2500.0).contains(&ms))))
}

fn conditions(dir: &Path, job: &Value, report: &Value) -> Result<Value> {
    let journal = fs::read_to_string(dir.join("conditions.jsonl")).map_err(|e| e.to_string())?;
    let mut nonready = Vec::new();
    let mut observed = 0usize;
    for (line_no, line) in journal.lines().enumerate() {
        let value: Value = parse_strict_json(line.as_bytes())
            .map_err(|e| format!("condition line {}: {e}", line_no + 1))?;
        observed += 1;
        match value["ready"].as_bool() {
            Some(false) if stale_power_only(&value, job) => nonready.push(value),
            Some(false) => {
                return Err(format!(
                    "condition line {} is not stale-power-only",
                    line_no + 1
                ))
            }
            Some(true) => {}
            None => return Err(format!("condition line {} lacks readiness", line_no + 1)),
        }
    }
    if observed == 0 {
        return Err("empty condition journal".into());
    }
    let violations = report["violations"]
        .as_array()
        .ok_or("violations missing")?;
    if violations != &nonready.iter().take(32).cloned().collect::<Vec<_>>() {
        return Err("report violations differ from condition journal".into());
    }
    let eligible = report["sampled_conditions_eligible"]
        .as_bool()
        .ok_or("eligibility missing")?;
    if eligible != nonready.is_empty() {
        return Err("condition eligibility disagrees with journal".into());
    }
    Ok(json!({"timing_eligible":eligible,"observations":observed,
        "stale_power_observations":nonready.len(),"activity_unsampled":nonready.iter().filter(|v| v["activity_sampled"] == false).count(),
        "violations":violations}))
}

fn receipt(
    dir: &Path,
    manifest: &Path,
    sha: &str,
    id: &str,
) -> Result<(Value, Value, Value, BTreeMap<&'static str, String>)> {
    if digest(manifest)? != sha {
        return Err(format!("{id}: frozen manifest changed"));
    }
    let mut expected = read(manifest)?;
    expected["kernel_game_submission"] = Value::Null;
    expected["validator"] = Value::Null;
    let job_path = dir.join("job.json");
    let job = read(&job_path)?;
    if job != expected || text(&job, "id")? != id || text(&job, "purpose")? != "correctness" {
        return Err(format!("{id}: queue job differs from frozen manifest"));
    }
    let report_path = dir.join("report.json");
    let report = read(&report_path)?;
    if text(&report, "id")? != id
        || text(&report, "status")? != "succeeded"
        || text(&report, "purpose")? != "correctness"
        || report["exit_code"].as_i64() != Some(0)
        || report["files_unchanged"] != true
        || report["overdue"] != false
        || report["signal_or_missing_exit_code"] != false
    {
        return Err(format!(
            "{id}: queue receipt is not successful and unchanged"
        ));
    }
    let condition_status = conditions(dir, &job, &report)?;
    let receipts = [
        ("job", job_path),
        ("report", report_path),
        ("trial_stdout", dir.join("trial.stdout")),
        ("trial_stderr", dir.join("trial.stderr")),
        ("conditions", dir.join("conditions.jsonl")),
    ]
    .into_iter()
    .map(|(key, path)| digest(&path).map(|sha| (key, sha)))
    .collect::<Result<BTreeMap<_, _>>>()?;
    Ok((job, report, condition_status, receipts))
}

fn hf_score(reference: &Value, prompt: &[u32], target: u32) -> Result<Value> {
    if text(reference, "schema")? != "rvllm.gemma4_hf_reference_logits.v1"
        || ids(reference, "prompt_token_ids")? != prompt
        || ids(reference, "selected_token_ids")? != [target]
        || reference["decode_steps"].as_u64() != Some(1)
        || reference["full_logits"] != true
    {
        return Err("HF identity or shape differs".into());
    }
    let steps = reference["steps"].as_array().ok_or("HF steps missing")?;
    if steps.len() != 1 {
        return Err("HF must emit one step".into());
    }
    let step = &steps[0];
    let logits = step["logits"].as_array().ok_or("HF full logits missing")?;
    if logits.len() != 262_144 || target as usize >= logits.len() {
        return Err("HF vocabulary or target differs".into());
    }
    let values = logits
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| "HF contains nonfinite or nonnumeric logit".into())
        })
        .collect::<Result<Vec<f64>>>()?;
    let target_logit = values[target as usize];
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum = values.iter().map(|v| (v - max).exp()).sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("HF logsumexp invalid".into());
    }
    let nll = max + sum.ln() - target_logit;
    let rank = 1 + values
        .iter()
        .enumerate()
        .filter(|(i, v)| **v > target_logit || (**v == target_logit && *i < target as usize))
        .count();
    let greedy = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(i, _)| i as u32)
        .ok_or("HF logits empty")?;
    if step["next_token"].as_u64() != Some(u64::from(greedy))
        || ids(reference, "generated_tokens")? != [greedy]
    {
        return Err("HF generated ID disagrees with full vector".into());
    }
    let selected = step["selected_logits"]
        .as_array()
        .ok_or("HF selected logits missing")?;
    if selected.len() != 1
        || selected[0]["token_id"].as_u64() != Some(u64::from(target))
        || number(&selected[0], "logit")? != target_logit
    {
        return Err("HF selected logit disagrees with full vector".into());
    }
    Ok(json!({"target_token_id":target,"target_logit":target_logit,
        "target_rank":rank,"negative_log_likelihood":nll,"sampled_token_id":greedy}))
}

fn metal_score(trial: &Value, prompt: &[u32], target: u32, route: &str) -> Result<Value> {
    if text(trial, "schema")? != "rvllm.apple_metal_text_infer.v1"
        || ids(trial, "prompt_token_ids")? != prompt
        || ids(trial, "generated_token_ids")? != [target]
        || text(trial, "metal_compute_dtype")? != "bfloat16"
        || text(trial, "metal_weight_dtype")? != "bfloat16"
    {
        return Err("Metal model/prompt/target identity differs".into());
    }
    let dispatch = field(trial, "research_dispatch")?;
    if text(dispatch, "schema")? != "rvllm.metal.research-dispatch.v6"
        || dispatch["overflowed"] != false
    {
        return Err("Metal dispatch invalid".into());
    }
    let counts = dispatch["counts"]
        .as_object()
        .ok_or("dispatch counts missing")?;
    match route {
        "off" if counts.is_empty() => {}
        "combined" => {
            for (name, expected) in [
                ("research_prefill_combined_gemm", 48),
                ("research_prefill_combined_qkv", 48),
                ("research_prefill_combined_raw_norm_projection", 96),
                ("research_prefill_combined_raw_norm", 96),
                ("research_prefill_combined_d256", 40),
                ("research_prefill_combined_d512", 8),
            ] {
                if counts.get(name).and_then(Value::as_u64) != Some(expected) {
                    return Err(format!("candidate missing exact {name} dispatch"));
                }
            }
            if counts.len() != 6 {
                return Err("candidate had extra research dispatch".into());
            }
        }
        _ => return Err("control route or route name invalid".into()),
    }
    let teacher = field(trial, "teacher_forced")?;
    if text(teacher, "schema")? != "rvllm.metal_teacher_forced_quality.v1" {
        return Err("Metal teacher schema differs".into());
    }
    let steps = teacher["steps"].as_array().ok_or("Metal steps missing")?;
    if steps.len() != 1 || steps[0]["target_token_id"].as_u64() != Some(u64::from(target)) {
        return Err("Metal postdecode step count or target differs".into());
    }
    let score = field(teacher, "prefill_last_step")?;
    if score["target_token_id"].as_u64() != Some(u64::from(target))
        || score["target_rank"]
            .as_u64()
            .is_none_or(|rank| rank == 0 || rank > 262_144)
        || score["sampled_token_id"]
            .as_u64()
            .is_none_or(|id| id >= 262_144)
        || number(score, "negative_log_likelihood")? < 0.0
    {
        return Err("Metal prefill-final score invalid".into());
    }
    number(score, "target_logit")?;
    Ok(score.clone())
}

fn read_arm(
    dir: &Path,
    manifest: &Path,
    sha: &str,
    id: &str,
    route: &str,
    prompt: &[u32],
    target: u32,
) -> Result<Arm> {
    let (job, _report, conditions, mut receipts) = receipt(dir, manifest, sha, id)?;
    let score = if route == "hf" {
        let output = PathBuf::from(arg(&job, "--output")?);
        let reference = read(&output)?;
        let model = job["command"]["args"]
            .as_array()
            .and_then(|a| a.get(1))
            .and_then(Value::as_str)
            .ok_or("HF model argument missing")?;
        if text(&reference, "model_dir")? != model {
            return Err("HF output model differs from command".into());
        }
        let score = hf_score(&reference, prompt, target)?;
        receipts.insert("hf_full_reference", digest(&output)?);
        score
    } else {
        let trial = read(&dir.join("trial.stdout"))?;
        if text(&trial, "model_dir")? != arg(&job, "--model-dir")? {
            return Err("Metal output model differs from command".into());
        }
        metal_score(&trial, prompt, target, route)?
    };
    let timing_eligible = conditions["timing_eligible"] == true;
    Ok(Arm {
        score,
        timing_eligible,
        conditions,
        receipts,
    })
}

fn run(paths: &[PathBuf]) -> Result<Value> {
    if digest(&paths[1])? != TOKENIZER_SHA || digest(&paths[2])? != SOURCE_SHA {
        return Err("tokenizer or source changed".into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&paths[1]).map_err(|e| e.to_string())?;
    let source = read(&paths[2])?;
    if text(&source, "schema")? != "rvllm.gemma4_heldout_text.v1"
        || source["cases"]
            .as_array()
            .is_none_or(|cases| cases.len() != 2)
    {
        return Err("source schema/case count differs".into());
    }
    let mut cases = Vec::new();
    for (index, (slug, case_id, expected_len, expected_target)) in [
        ("observatory", "observatory-clock-v2", 213, 80444),
        ("library", "library-ledger-v2", 203, 506),
    ]
    .into_iter()
    .enumerate()
    {
        let (prompt, target) = expected_case(&tokenizer, &source, case_id)?;
        if prompt.len() != expected_len || target != expected_target {
            return Err(format!("{slug}: source tokens differ from preregistration"));
        }
        let arms = ARMS[index * 3..index * 3 + 3]
            .iter()
            .zip(&paths[4 + index * 3..7 + index * 3])
            .map(|((frozen_slug, route, file, sha), dir)| {
                if *frozen_slug != slug {
                    return Err("frozen arm order differs".into());
                }
                let id = format!("prefill26-prefill-ref-distinct-v2-{slug}-{route}-20260927");
                read_arm(dir, &paths[3].join(file), sha, &id, route, &prompt, target)
            })
            .collect::<Result<Vec<_>>>()?;
        let comparison = |i: usize| -> Result<Value> {
            let hf = &arms[0].score;
            let metal = &arms[i].score;
            Ok(
                json!({"target_logit_minus_hf":number(metal,"target_logit")?-number(hf,"target_logit")?,
                "nll_minus_hf":number(metal,"negative_log_likelihood")?-number(hf,"negative_log_likelihood")?,
                "rank_matches_hf":metal["target_rank"] == hf["target_rank"],
                "greedy_matches_hf":metal["sampled_token_id"] == hf["sampled_token_id"]}),
            )
        };
        cases.push(
            json!({"id":case_id,"prompt_tokens":prompt.len(),"target_token_id":target,
            "arms":arms.iter().zip(["hf","off","combined"]).map(|(arm,route)| json!({
                "route":route,"score":arm.score,"timing_eligible":arm.timing_eligible,
                "conditions":arm.conditions,"receipt_sha256":arm.receipts,
            })).collect::<Vec<_>>(),
            "off_vs_hf":comparison(1)?,"combined_vs_hf":comparison(2)?}),
        );
    }
    Ok(
        json!({"schema":"rvllm.gemma4_distinct_prefill_reference_summary.v2",
        "claim":"Numerical-only comparison at two synthetic full-prompt next-token positions; stale-power-only queue conditions are retained but never timing evidence; no full Metal vector, continuation quality, speed or promotion",
        "source_sha256":SOURCE_SHA,"tokenizer_sha256":TOKENIZER_SHA,"cases":cases}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_manifest_hashes_match_checkout() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../reports/prefill-round-fullroute-short-20260927/numerical-reference-distinct-v2-queue");
        for (_, _, file, sha) in ARMS {
            assert_eq!(digest(&root.join(file)).unwrap(), sha, "{file}");
        }
    }

    #[test]
    fn stale_power_is_numerical_only_not_timing_eligible() {
        let job = json!({"conditions":{"minimum_free_bytes":100,"power_source":"ac"}});
        let sample = json!({"ready":false,"power":{"age_ms":2634.0,"observer_journal_error":null,
            "sample":{"controls":{"power_source":"ac","low_power_mode":false,
                "pmset_power_mode":2,"thermal_state":0,"cpu_speed_limit_percent":null,
                "scheduler_limit_percent":null,"available_cpus":null}}},"free_bytes":200,
            "competing_processes":[],"observed_processes":[],"idle_server_checks":[],
            "probe_ms":0.2,"activity_sampled":false});
        assert!(stale_power_only(&sample, &job));
        let mut competitor = sample.clone();
        competitor["competing_processes"] = json!([{"pid":42}]);
        assert!(!stale_power_only(&competitor, &job));
        let mut fresh = sample;
        fresh["power"]["age_ms"] = json!(2000.0);
        assert!(!stale_power_only(&fresh, &job));
    }

    #[test]
    fn candidate_must_dispatch_every_named_prefill_kernel() {
        let trial = json!({"schema":"rvllm.apple_metal_text_infer.v1",
            "prompt_token_ids":[2],"generated_token_ids":[3],
            "metal_compute_dtype":"bfloat16","metal_weight_dtype":"bfloat16",
            "research_dispatch":{"schema":"rvllm.metal.research-dispatch.v6",
                "overflowed":false,"counts":{}},
            "teacher_forced":{"schema":"rvllm.metal_teacher_forced_quality.v1",
                "steps":[{"target_token_id":3}],
                "prefill_last_step":{"target_token_id":3,"target_logit":1.0,
                    "target_rank":1,"negative_log_likelihood":0.1,"sampled_token_id":3}}});
        assert!(metal_score(&trial, &[2], 3, "combined").is_err());
    }

    #[test]
    fn hf_ties_rank_and_choose_lowest_token_id() {
        let mut logits = vec![0.0; 262_144];
        logits[3] = 2.0;
        logits[8] = 2.0;
        let reference = json!({"schema":"rvllm.gemma4_hf_reference_logits.v1",
        "prompt_token_ids":[2,10],"decode_steps":1,"full_logits":true,
        "selected_token_ids":[8],"generated_tokens":[3],"steps":[{
            "next_token":3,"selected_logits":[{"token_id":8,"logit":2.0}],"logits":logits
        }]});
        let score = hf_score(&reference, &[2, 10], 8).unwrap();
        assert_eq!(score["target_rank"], 2);
        assert_eq!(score["sampled_token_id"], 3);
    }
}
