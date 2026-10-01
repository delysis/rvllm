//! Fail-closed full-vocabulary comparison of six distinct prefill jobs.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "f6d06d159a16472c4dba9496f2cbdc0b6f3b3b2498b0e05dfae64cc347f603bd";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const ARMS: [(&str, &str, &str, &str); 6] = [
    (
        "bridge",
        "hf",
        "bridge-hf-job.json",
        "194fda0dc5807ff6d02adac0b0b66b7a8f2fc6012bb31e7d6f58ebce22604a89",
    ),
    (
        "bridge",
        "off",
        "bridge-off-job.json",
        "ec54f48d0b303643a3c8f5c35a84467faf1d9d4c38b77c08d2431738ead37a85",
    ),
    (
        "bridge",
        "combined",
        "bridge-combined-job.json",
        "767d4bf67d0c0fa80c18806ccbf39ca7a12d44c431ed567f4ac2fab6aadc0744",
    ),
    (
        "seed",
        "hf",
        "seed-hf-job.json",
        "02fcb7a07c521e3f1fb064226aa39ba481d76c5a6de22d53a5c3dd203df877f8",
    ),
    (
        "seed",
        "off",
        "seed-off-job.json",
        "2ac12e6e661796a16929a264d81c2e541ddbb38ec06948967f7199d09129e1d4",
    ),
    (
        "seed",
        "combined",
        "seed-combined-job.json",
        "83b1fd6fb074edc2b090d7014e555170ea2555142cfb74a4069398018eab76b1",
    ),
];

struct Arm {
    score: Value,
    logits: Vec<f32>,
    timing_eligible: bool,
    conditions: Value,
    receipts: BTreeMap<&'static str, String>,
}

fn main() {
    let args = env::args_os().map(PathBuf::from).collect::<Vec<_>>();
    if args.len() != 11 {
        eprintln!("usage: rvllm_gemma4_full_vector_summary OUTPUT.json TOKENIZER.json SOURCE.json MANIFEST_DIR BRIDGE_HF_DIR BRIDGE_OFF_DIR BRIDGE_COMBINED_DIR SEED_HF_DIR SEED_OFF_DIR SEED_COMBINED_DIR");
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
        eprintln!("rvllm_gemma4_full_vector_summary: {error}");
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
                .map_or(true, |required| value == required)
        })
        && controls["pmset_power_mode"].as_u64().is_some_and(|mode| {
            mode <= 2
                && conditions["pmset_power_mode"]
                    .as_u64()
                    .map_or(true, |required| mode == required)
        })
        && controls["thermal_state"].as_u64().is_some_and(|state| {
            state <= 3
                && conditions["thermal_state"]
                    .as_u64()
                    .map_or(true, |required| state == required)
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

fn hf_score(reference: &Value, prompt: &[u32], target: u32) -> Result<(Value, Vec<f32>)> {
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
    let floats = values
        .into_iter()
        .map(|value| {
            let float = value as f32;
            float
                .is_finite()
                .then_some(float)
                .ok_or_else(|| "HF logit is not representable as finite f32".to_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        json!({"target_token_id":target,"target_logit":target_logit,
        "target_rank":rank,"negative_log_likelihood":nll,"sampled_token_id":greedy}),
        floats,
    ))
}

fn score_vector(logits: &[f32], target: u32) -> Result<Value> {
    if logits.len() != 262_144 || target as usize >= logits.len() {
        return Err("full vector vocabulary or target differs".into());
    }
    if logits
        .iter()
        .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        return Err("full vector contains NaN or positive infinity".into());
    }
    let selected = logits[target as usize];
    if !selected.is_finite() {
        return Err("full vector target is nonfinite".into());
    }
    let (greedy, max) = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .ok_or("full vector has no finite logit")?;
    let max = f64::from(max);
    let sum = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (f64::from(value) - max).exp())
        .sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("full vector log-sum-exp invalid".into());
    }
    let nll = max + sum.ln() - f64::from(selected);
    if !nll.is_finite() || nll < -1e-9 {
        return Err("full vector NLL invalid".into());
    }
    let rank = 1 + logits
        .iter()
        .enumerate()
        .filter(|(index, value)| {
            **value > selected || (**value == selected && *index < target as usize)
        })
        .count();
    Ok(json!({"target_token_id":target,"target_logit":selected,
        "target_rank":rank,"negative_log_likelihood":nll.max(0.0),
        "sampled_token_id":greedy}))
}

fn metal_vector(
    trial: &Value,
    job: &Value,
    prompt_len: usize,
    target: u32,
) -> Result<(Vec<f32>, String)> {
    let path = PathBuf::from(arg(job, "--teacher-prefill-full-logits-output")?);
    let receipt = field(field(trial, "teacher_forced")?, "prefill_full_logits")?;
    if text(receipt, "path")? != path.to_string_lossy()
        || receipt["vocab_size"].as_u64() != Some(262_144)
    {
        return Err("Metal full-vector receipt path or vocabulary differs".into());
    }
    let bytes = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let sha = Sha256Digest::bytes(&bytes).as_str().to_owned();
    if text(receipt, "sha256")? != sha || receipt["bytes"].as_u64() != Some(bytes.len() as u64) {
        return Err("Metal full-vector receipt hash or size differs".into());
    }
    let vector: Value = parse_strict_json(&bytes).map_err(|error| error.to_string())?;
    if text(&vector, "schema")? != "rvllm.metal_prefill_final_full_logits_bits.v1"
        || vector["prompt_tokens"].as_u64() != Some(prompt_len as u64)
        || vector["target_token_id"].as_u64() != Some(u64::from(target))
        || vector["vocab_size"].as_u64() != Some(262_144)
        || text(&vector, "representation")? != "f32::to_bits as unsigned decimal integers"
    {
        return Err("Metal full-vector identity differs".into());
    }
    let bits = vector["logit_bits"]
        .as_array()
        .ok_or("Metal logit bits missing")?;
    if bits.len() != 262_144 {
        return Err("Metal full-vector length differs".into());
    }
    let logits = bits
        .iter()
        .map(|bit| {
            bit.as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .map(f32::from_bits)
                .ok_or_else(|| "invalid Metal f32 bit pattern".to_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    let negative_infinity_count = logits
        .iter()
        .filter(|value| **value == f32::NEG_INFINITY)
        .count();
    if receipt["negative_infinity_count"].as_u64() != Some(negative_infinity_count as u64) {
        return Err("Metal negative-infinity count differs".into());
    }
    let reconstructed = score_vector(&logits, target)?;
    let reported = field(field(trial, "teacher_forced")?, "prefill_last_step")?;
    if reconstructed["target_token_id"] != reported["target_token_id"]
        || reconstructed["target_rank"] != reported["target_rank"]
        || reconstructed["sampled_token_id"] != reported["sampled_token_id"]
        || number(&reconstructed, "target_logit")? != number(reported, "target_logit")?
        || (number(&reconstructed, "negative_log_likelihood")?
            - number(reported, "negative_log_likelihood")?)
        .abs()
            > 1e-7
    {
        return Err("Metal scalar score disagrees with full-vector reconstruction".into());
    }
    Ok((logits, sha))
}

fn top_ids(logits: &[f32], count: usize) -> Vec<usize> {
    let mut ids = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    ids.sort_unstable_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    ids.truncate(count);
    ids
}

fn compare_vectors(left: &[f32], right: &[f32]) -> Result<Value> {
    if left.len() != 262_144 || right.len() != left.len() {
        return Err("compared full-vector lengths differ".into());
    }
    let mut exact_bits = 0usize;
    let mut both_finite = 0usize;
    let mut both_negative_infinity = 0usize;
    let mut one_sided_negative_infinity = 0usize;
    let (mut max_abs, mut sum_abs, mut sum_squares) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&a, &b) in left.iter().zip(right) {
        exact_bits += usize::from(a.to_bits() == b.to_bits());
        if a == f32::NEG_INFINITY && b == f32::NEG_INFINITY {
            both_negative_infinity += 1;
        } else if a == f32::NEG_INFINITY || b == f32::NEG_INFINITY {
            one_sided_negative_infinity += 1;
        } else if a.is_finite() && b.is_finite() {
            both_finite += 1;
            let diff = (f64::from(a) - f64::from(b)).abs();
            max_abs = max_abs.max(diff);
            sum_abs += diff;
            sum_squares += diff * diff;
        } else {
            return Err("unsupported nonfinite vector comparison".into());
        }
    }
    let right_top = top_ids(right, 256)
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let overlap = top_ids(left, 256)
        .into_iter()
        .filter(|id| right_top.contains(id))
        .count();
    Ok(
        json!({"vocab_size":left.len(),"exact_bit_matches":exact_bits,
        "both_finite":both_finite,"both_negative_infinity":both_negative_infinity,
        "one_sided_negative_infinity":one_sided_negative_infinity,
        "finite_max_absolute_difference":max_abs,
        "finite_mean_absolute_difference":(both_finite > 0).then(|| sum_abs / both_finite as f64),
        "finite_rms_difference":(both_finite > 0).then(|| (sum_squares / both_finite as f64).sqrt()),
        "top_256_id_overlap":overlap}),
    )
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
            .map_or(true, |rank| rank == 0 || rank > 262_144)
        || score["sampled_token_id"]
            .as_u64()
            .map_or(true, |id| id >= 262_144)
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
    let (score, logits) = if route == "hf" {
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
        let (score, logits) = hf_score(&reference, prompt, target)?;
        let reconstructed = score_vector(&logits, target)?;
        if score["target_rank"] != reconstructed["target_rank"]
            || score["sampled_token_id"] != reconstructed["sampled_token_id"]
            || number(&score, "target_logit")? != number(&reconstructed, "target_logit")?
            || (number(&score, "negative_log_likelihood")?
                - number(&reconstructed, "negative_log_likelihood")?)
            .abs()
                > 1e-7
        {
            return Err("HF scalar score differs after f32 vector reconstruction".into());
        }
        receipts.insert("hf_full_reference", digest(&output)?);
        (score, logits)
    } else {
        let trial = read(&dir.join("trial.stdout"))?;
        if text(&trial, "model_dir")? != arg(&job, "--model-dir")? {
            return Err("Metal output model differs from command".into());
        }
        let score = metal_score(&trial, prompt, target, route)?;
        let (logits, sha) = metal_vector(&trial, &job, prompt.len(), target)?;
        receipts.insert("metal_full_vector", sha);
        (score, logits)
    };
    let timing_eligible = conditions["timing_eligible"] == true;
    Ok(Arm {
        score,
        logits,
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
            .map_or(true, |cases| cases.len() != 2)
    {
        return Err("source schema/case count differs".into());
    }
    let mut cases = Vec::new();
    for (index, (slug, case_id, expected_len, expected_target)) in [
        ("bridge", "bridge-cable-v3", 229, 56896),
        ("seed", "seed-bank-v3", 246, 52102),
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
                let id = format!("prefill26-fullvec-v3-{slug}-{route}-20260927");
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
            "off_vs_hf":comparison(1)?,"combined_vs_hf":comparison(2)?,
            "off_vs_hf_vector":compare_vectors(&arms[0].logits,&arms[1].logits)?,
            "combined_vs_hf_vector":compare_vectors(&arms[0].logits,&arms[2].logits)?,
            "combined_vs_off_vector":compare_vectors(&arms[1].logits,&arms[2].logits)?}),
        );
    }
    Ok(
        json!({"schema":"rvllm.gemma4_full_vector_prefill_reference_summary.v3",
        "claim":"Numerical-only full-vocabulary comparison at two synthetic full-prompt next-token positions; stale-power-only queue conditions are retained but never timing evidence; no continuation quality, first arithmetic cause, speed or promotion",
        "source_sha256":SOURCE_SHA,"tokenizer_sha256":TOKENIZER_SHA,"cases":cases}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_manifest_hashes_match_checkout() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../reports/prefill-round-fullroute-short-20260927/numerical-reference-full-vector-v3-queue");
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
        let (score, _) = hf_score(&reference, &[2, 10], 8).unwrap();
        assert_eq!(score["target_rank"], 2);
        assert_eq!(score["sampled_token_id"], 3);
    }

    #[test]
    fn vector_metrics_preserve_bits_ties_and_negative_infinity() {
        let mut left = vec![0.0_f32; 262_144];
        let mut right = left.clone();
        left[0] = 2.0;
        left[1] = 2.0;
        right[0] = 2.0;
        right[1] = 2.0;
        left[9] = f32::NEG_INFINITY;
        right[9] = f32::NEG_INFINITY;
        right[10] = -0.0;
        right[11] = 1.0;
        assert_eq!(score_vector(&left, 1).unwrap()["sampled_token_id"], 0);
        assert_eq!(score_vector(&left, 1).unwrap()["target_rank"], 2);
        let metrics = compare_vectors(&left, &right).unwrap();
        assert_eq!(metrics["exact_bit_matches"], 262_142);
        assert_eq!(metrics["both_negative_infinity"], 1);
        assert_eq!(metrics["one_sided_negative_infinity"], 0);
        assert_eq!(metrics["finite_max_absolute_difference"], 1.0);
        right[12] = f32::NEG_INFINITY;
        assert_eq!(
            compare_vectors(&left, &right).unwrap()["one_sided_negative_infinity"],
            1
        );
        right[12] = f32::NAN;
        assert!(compare_vectors(&left, &right).is_err());
    }

    #[test]
    fn metal_vector_refuses_receipt_hash_drift() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full.json");
        let mut logits = vec![0.0_f32; 262_144];
        logits[8] = 2.0;
        logits[9] = f32::NEG_INFINITY;
        let vector = json!({"schema":"rvllm.metal_prefill_final_full_logits_bits.v1",
            "prompt_tokens":229,"target_token_id":8,"vocab_size":262_144,
            "representation":"f32::to_bits as unsigned decimal integers",
            "logit_bits":logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>()});
        let bytes = serde_json::to_vec(&vector).unwrap();
        fs::write(&path, &bytes).unwrap();
        let sha = Sha256Digest::bytes(&bytes).as_str().to_owned();
        let score = score_vector(&logits, 8).unwrap();
        let mut trial = json!({"teacher_forced":{"prefill_last_step":score,
            "prefill_full_logits":{"path":path,"sha256":sha,"bytes":bytes.len(),
                "vocab_size":262_144,"negative_infinity_count":1}}});
        let job = json!({"command":{"args":["--teacher-prefill-full-logits-output",path]}});
        assert_eq!(metal_vector(&trial, &job, 229, 8).unwrap().0.len(), 262_144);
        trial["teacher_forced"]["prefill_last_step"]["target_rank"] = json!(2);
        assert!(metal_vector(&trial, &job, 229, 8).is_err());
        trial["teacher_forced"]["prefill_last_step"]["target_rank"] = json!(1);
        trial["teacher_forced"]["prefill_full_logits"]["sha256"] = json!("wrong");
        assert!(metal_vector(&trial, &job, 229, 8).is_err());
    }
}
