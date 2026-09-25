//! Strict, non-selective summary of the Gemma 4 native-BF16 low-bit campaign.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

const PREFIX: &str = "g4metalbf16lowbit-";
const SUFFIX: &str = "-m1m4-";
const DATE: &str = "-20260925";
const ROLES: [&str; 7] = ["q", "k", "v", "o", "gate", "up", "down"];
const FORMATS: [&str; 2] = ["w4a16", "w8a16"];
const M_VALUES: [u64; 2] = [1, 4];
const MAX_RELATIVE_DRIFT: f64 = 0.20;
const MIN_REPEAT_SPEEDUP: f64 = 1.05;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn fail<T>(s: impl Into<String>) -> Result<T> {
    Err(s.into().into())
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_json(path: &Path) -> Result<(Value, String)> {
    let b = fs::read(path)?;
    Ok((serde_json::from_slice(&b)?, hash(&b)))
}
fn field<'a>(v: &'a Value, k: &str) -> Result<&'a Value> {
    v.get(k).ok_or_else(|| format!("missing {k}").into())
}
fn num(v: &Value, k: &str) -> Result<f64> {
    let x = field(v, k)?
        .as_f64()
        .ok_or_else(|| format!("{k} is not numeric"))?;
    if !x.is_finite() || x <= 0.0 {
        return fail(format!("invalid {k}"));
    }
    Ok(x)
}

fn expected_ids() -> Vec<String> {
    ROLES
        .iter()
        .flat_map(|r| ["screen", "confirm"].map(|p| format!("{PREFIX}{r}{SUFFIX}{p}{DATE}")))
        .collect()
}
fn role_name(short: &str) -> &'static str {
    match short {
        "q" => "query_projection",
        "k" => "key_projection",
        "v" => "value_projection",
        "o" => "output_projection",
        "gate" => "dense_gate_projection",
        "up" => "dense_up_projection",
        "down" => "dense_down_projection",
        _ => unreachable!("sealed role list"),
    }
}
fn relative_drift(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.min(b)
}
fn stable(a: f64, b: f64) -> bool {
    relative_drift(a, b) <= MAX_RELATIVE_DRIFT
}
fn upper_median(xs: &[f64]) -> Result<f64> {
    if xs.is_empty() {
        return fail("empty timing samples");
    }
    let mut x = xs.to_vec();
    x.sort_by(f64::total_cmp);
    Ok(x[x.len() / 2])
}
fn samples(v: &Value, k: &str) -> Result<Vec<f64>> {
    field(v, k)?
        .as_array()
        .ok_or_else(|| format!("{k} not array"))?
        .iter()
        .map(|x| {
            x.as_f64()
                .filter(|x| x.is_finite() && *x > 0.0)
                .ok_or_else(|| format!("invalid {k} sample").into())
        })
        .collect()
}
fn distribution(xs: &[f64]) -> Result<Value> {
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        return fail("empty distribution");
    }
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    Ok(
        json!({"count":s.len(),"min":s[0],"median":s[s.len()/2],"mean":mean,"max":s[s.len()-1],"samples":xs}),
    )
}

fn conditions(path: &Path) -> Result<Value> {
    let bytes = fs::read(path)?;
    let text = std::str::from_utf8(&bytes)?;
    let mut observations = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        observations.push(
            serde_json::from_str::<Value>(line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?,
        )
    }
    if observations.is_empty() {
        return fail(format!("empty conditions: {}", path.display()));
    }
    Ok(json!({"sha256":hash(&bytes),"gating":false,"observations":observations}))
}

fn validate_outer(id: &str, phase: &str, job: &Value, queue: &Value) -> Result<()> {
    if job.get("id").and_then(Value::as_str) != Some(id)
        || queue.get("id").and_then(Value::as_str) != Some(id)
    {
        return fail(format!("outer identity mismatch: {id}"));
    }
    let purpose = if phase == "screen" {
        "correctness"
    } else {
        "exploratory_timing"
    };
    if queue.get("status").and_then(Value::as_str) != Some("succeeded")
        || queue.get("exit_code").and_then(Value::as_i64) != Some(0)
        || queue
            .pointer("/validation/success")
            .and_then(Value::as_bool)
            != Some(true)
        || queue.get("purpose").and_then(Value::as_str) != Some(purpose)
        || queue.get("files_unchanged").and_then(Value::as_bool) != Some(true)
        || queue
            .get("violations")
            .and_then(Value::as_array)
            .is_none_or(|v| !v.is_empty())
    {
        return fail(format!("unclean queue result: {id}"));
    }
    Ok(())
}

fn summarize(root: &Path) -> Result<Value> {
    let results = root.join("results");
    let expected = expected_ids();
    let mut observed = Vec::new();
    for e in fs::read_dir(&results)? {
        let n = e?.file_name().to_string_lossy().into_owned();
        if n.starts_with(PREFIX) {
            observed.push(n)
        }
    }
    observed.sort();
    let mut sorted = expected.clone();
    sorted.sort();
    if observed != sorted {
        return fail(format!(
            "campaign membership mismatch; expected {sorted:?}, observed {observed:?}"
        ));
    }
    let mut runs = BTreeMap::<String, Value>::new();
    let mut canonical_config = None;
    let mut canonical_exe = None;
    let mut canonical_msl = None;
    for role in ROLES {
        for phase in ["screen", "confirm"] {
            let id = format!("{PREFIX}{role}{SUFFIX}{phase}{DATE}");
            let dir = results.join(&id);
            let (job, job_hash) = read_json(&dir.join("job.json"))?;
            let (queue, queue_hash) = read_json(&dir.join("report.json"))?;
            let (trial, trial_hash) = read_json(&dir.join("trial.stdout"))?;
            validate_outer(&id, phase, &job, &queue)?;
            if trial.get("schema").and_then(Value::as_str)
                != Some("rvllm.metal_low_bit_real_weight_bf16.v1")
                || trial.get("source_dtype").and_then(Value::as_str) != Some("Bf16")
                || trial.pointer("/abi/activation").and_then(Value::as_str) != Some("BF16")
                || trial.pointer("/abi/output").and_then(Value::as_str) != Some("BF16")
                || trial.pointer("/abi/accumulation").and_then(Value::as_str) != Some("F32")
            {
                return fail(format!("ABI/schema mismatch: {id}"));
            }
            if trial.get("tensor_role").and_then(Value::as_str) != Some(role_name(role)) {
                return fail(format!("tensor role mismatch: {id}"));
            }
            for (slot, key, label) in [
                (&mut canonical_config, "config_sha256", "config"),
                (&mut canonical_exe, "executable_sha256", "executable"),
                (&mut canonical_msl, "generated_msl_sha256", "generated MSL"),
            ] {
                let actual = field(&trial, key)?.clone();
                if slot.as_ref().is_some_and(|v| v != &actual) {
                    return fail(format!("cross-run {label} mismatch: {id}"));
                }
                if slot.is_none() {
                    *slot = Some(actual)
                }
            }
            let cases = field(&trial, "cases")?
                .as_array()
                .ok_or("cases not array")?;
            if cases.len() != 4 {
                return fail(format!("wrong case count: {id}"));
            }
            let mut keyed = BTreeMap::new();
            for case in cases {
                let format = case
                    .pointer("/dispatch/format")
                    .and_then(Value::as_str)
                    .ok_or("missing format")?;
                let m = case.get("m").and_then(Value::as_u64).ok_or("missing m")?;
                if !FORMATS.contains(&format) || !M_VALUES.contains(&m) {
                    return fail(format!("unexpected case {format}/M{m}: {id}"));
                }
                let key = format!("{format}-m{m}");
                if keyed.contains_key(&key) {
                    return fail(format!("duplicate case {key}: {id}"));
                }
                if case.get("guard_unchanged").and_then(Value::as_bool) != Some(true)
                    || case.get("repeatable_output_bits").and_then(Value::as_bool) != Some(true)
                    || case
                        .pointer("/dispatch/exact_correctness_dispatches_verified")
                        .and_then(Value::as_u64)
                        != Some(2)
                    || case
                        .pointer("/dispatch/exact_timing_dispatch_count_verified")
                        .and_then(Value::as_bool)
                        != Some(true)
                {
                    return fail(format!("correctness/dispatch invariant failed: {id}/{key}"));
                }
                let t = field(case, "timing")?;
                let candidate = samples(t, "candidate_ms")?;
                let native = samples(t, "native_ms")?;
                let expected_samples = if phase == "screen" { 6 } else { 18 };
                if candidate.len() != expected_samples
                    || native.len() != expected_samples
                    || t.get("samples_per_arm").and_then(Value::as_u64)
                        != Some(expected_samples as u64)
                {
                    return fail(format!("wrong sample count: {id}/{key}"));
                }
                let cm = num(t, "candidate_median_ms")?;
                let nm = num(t, "native_median_ms")?;
                let speed = num(t, "speedup")?;
                if relative_drift(cm, upper_median(&candidate)?) > 1e-9
                    || relative_drift(nm, upper_median(&native)?) > 1e-9
                    || relative_drift(speed, nm / cm) > 1e-9
                {
                    return fail(format!("derived timing mismatch: {id}/{key}"));
                }
                keyed.insert(key,json!({"format":format,"m":m,"shape":[case.get("n"),case.get("k")],"accuracy":case.get("accuracy"),"native_accuracy":case.get("native_accuracy"),"identities":case.get("identity"),"candidate_ms":distribution(&candidate)?,"native_ms":distribution(&native)?,"speedup":speed}));
            }
            runs.insert(format!("{role}-{phase}"),json!({"id":id,"role":trial.get("tensor_role"),"tensor":trial.get("tensor"),"source_tensor_sha256":trial.get("source_tensor_sha256"),"source_file_sha256":trial.get("source_file_sha256"),"shape":trial.get("shape"),"hashes":{"job":job_hash,"queue_report":queue_hash,"trial_stdout":trial_hash},"conditions":conditions(&dir.join("conditions.jsonl"))?,"cases":keyed}));
        }
    }
    let mut decisions = Vec::new();
    let mut promotable = 0;
    for role in ROLES {
        let screen_run = &runs[&format!("{role}-screen")];
        let confirm_run = &runs[&format!("{role}-confirm")];
        for key in ["role", "tensor", "source_tensor_sha256", "shape"] {
            if screen_run[key] != confirm_run[key] {
                return fail(format!("screen/confirm {key} identity mismatch: {role}"));
            }
        }
        for format in FORMATS {
            for m in M_VALUES {
                let key = format!("{format}-m{m}");
                let a = &screen_run["cases"][&key];
                let b = &confirm_run["cases"][&key];
                for identity_key in ["shape", "accuracy", "native_accuracy", "identities"] {
                    if a[identity_key] != b[identity_key] {
                        return fail(format!(
                            "screen/confirm {identity_key} mismatch: {role}/{key}"
                        ));
                    }
                }
                let cs = a["candidate_ms"]["median"].as_f64().unwrap();
                let cc = b["candidate_ms"]["median"].as_f64().unwrap();
                let ns = a["native_ms"]["median"].as_f64().unwrap();
                let nc = b["native_ms"]["median"].as_f64().unwrap();
                let ss = a["speedup"].as_f64().unwrap();
                let sc = b["speedup"].as_f64().unwrap();
                let stable_candidate = stable(cs, cc);
                let stable_native = stable(ns, nc);
                let stable_speedup = stable(ss, sc);
                let repeat_win = ss >= MIN_REPEAT_SPEEDUP && sc >= MIN_REPEAT_SPEEDUP;
                let eligible = stable_candidate && stable_native && stable_speedup && repeat_win;
                if eligible {
                    promotable += 1
                }
                let disposition = if eligible {
                    "stable_operator_win"
                } else if !stable_candidate || !stable_native || !stable_speedup {
                    "inconclusive_repeat_instability"
                } else {
                    "stable_not_faster"
                };
                decisions.push(json!({"role":role,"format":format,"m":m,"screen":{"candidate_median_ms":cs,"native_median_ms":ns,"speedup":ss},"confirm":{"candidate_median_ms":cc,"native_median_ms":nc,"speedup":sc},"relative_drift":{"candidate":relative_drift(cs,cc),"native":relative_drift(ns,nc),"speedup":relative_drift(ss,sc)},"thresholds":{"max_relative_drift":MAX_RELATIVE_DRIFT,"minimum_speedup_each_repeat":MIN_REPEAT_SPEEDUP},"disposition":disposition,"operator_promotion_eligible":eligible}));
            }
        }
    }
    Ok(
        json!({"schema":"rvllm.gemma4_metal_low_bit_bf16_campaign_summary.v1","complete":true,"promotion_authority":false,"claim_boundary":"real-checkpoint projection operator evidence only; no full-route, model-quality, or MLX comparison claim","all_conditions_retained":true,"conditions_are_not_a_gate":true,"common_identities":{"config_sha256":canonical_config,"executable_sha256":canonical_exe,"generated_msl_sha256":canonical_msl},"policy":{"screen_confirm_max_relative_drift":MAX_RELATIVE_DRIFT,"minimum_speedup_in_both_repeats":MIN_REPEAT_SPEEDUP,"all_three_candidate_native_speedup_must_be_stable":true},"stable_operator_win_count":promotable,"case_count":decisions.len(),"campaign_disposition":if promotable==decisions.len(){"all_operator_cases_stable_wins_not_full_route_promotable"}else{"not_promotable_repeat_instability_or_regression"},"decisions":decisions,"runs":runs}),
    )
}

fn markdown(v: &Value) -> String {
    let mut s=String::from("# Gemma 4 Metal native-BF16 low-bit campaign\n\nThis is real-checkpoint projection-operator evidence, not full-route, model-quality, or MLX-comparison evidence. Changing host conditions are retained and never used as a wait gate.\n\n");
    s.push_str(&format!("Campaign disposition: **{}**. Stable operator wins: **{}/{}**.\n\n| Role | Format | M | Screen | Confirm | Candidate drift | Native drift | Speedup drift | Disposition |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n",v["campaign_disposition"].as_str().unwrap(),v["stable_operator_win_count"],v["case_count"]));
    for d in v["decisions"].as_array().unwrap() {
        s.push_str(&format!(
            "| {} | {} | {} | {:.3}x | {:.3}x | {:.1}% | {:.1}% | {:.1}% | {} |\n",
            d["role"].as_str().unwrap(),
            d["format"].as_str().unwrap(),
            d["m"],
            d["screen"]["speedup"].as_f64().unwrap(),
            d["confirm"]["speedup"].as_f64().unwrap(),
            100.0 * d["relative_drift"]["candidate"].as_f64().unwrap(),
            100.0 * d["relative_drift"]["native"].as_f64().unwrap(),
            100.0 * d["relative_drift"]["speedup"].as_f64().unwrap(),
            d["disposition"].as_str().unwrap()
        ));
    }
    s.push_str("\nPromotion is refused unless candidate median, native median, and speedup agree within 20%, and both repetitions beat native by at least 5%. Even passing rows require full-route and checkpoint-quality gates before shipping promotion. The companion JSON retains every timing sample, identity hash, accuracy result, and condition observation.\n");
    s
}

fn main() -> Result<()> {
    let mut a = env::args_os().skip(1);
    let root = PathBuf::from(
        a.next()
            .ok_or("usage: rvllm-low-bit-campaign-summary QUEUE_ROOT OUT.json OUT.md")?,
    );
    let out = PathBuf::from(a.next().ok_or("missing OUT.json")?);
    let md = PathBuf::from(a.next().ok_or("missing OUT.md")?);
    if a.next().is_some() {
        return fail("too many arguments");
    }
    let v = summarize(&root)?;
    fs::write(out, serde_json::to_vec_pretty(&v)?)?;
    fs::write(md, markdown(&v))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ids_are_complete() {
        assert_eq!(expected_ids().len(), 14)
    }
    #[test]
    fn drift_is_symmetric() {
        assert!((relative_drift(8.0, 10.0) - 0.25).abs() < 1e-12);
        assert_eq!(relative_drift(8.0, 10.0), relative_drift(10.0, 8.0))
    }
    #[test]
    fn stability_boundary_is_strict() {
        assert!(stable(1.0, 1.2));
        assert!(!stable(1.0, 1.21))
    }
    #[test]
    fn distribution_retains_ordered_raw_samples() {
        let v = distribution(&[3.0, 1.0, 2.0]).unwrap();
        assert_eq!(v["samples"], json!([3.0, 1.0, 2.0]));
        assert_eq!(v["median"], 2.0)
    }
    #[test]
    fn empty_samples_fail() {
        assert!(upper_median(&[]).is_err())
    }
}
