//! Strict, non-selective summary for the Gemma 4 ANE stacked-FFN tournament.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

const PREFIX: &str = "gemma4-ane-stacked-baseline-exact-v2-";
const SUFFIX: &str = "-20260924";
const ARMS: [(&str, &str); 12] = [
    ("01-a1-stacked", "stacked"),
    ("02-b1-baseline", "baseline"),
    ("03-b2-baseline", "baseline"),
    ("04-a2-stacked", "stacked"),
    ("05-b3-baseline", "baseline"),
    ("06-a3-stacked", "stacked"),
    ("07-a4-stacked", "stacked"),
    ("08-b4-baseline", "baseline"),
    ("09-a5-stacked", "stacked"),
    ("10-b5-baseline", "baseline"),
    ("11-b6-baseline", "baseline"),
    ("12-a6-stacked", "stacked"),
];
const METRICS: [&str; 4] = ["attention_ms", "ffn_ms", "host_ms", "total_ms"];

fn fail<T>(message: impl Into<String>) -> Result<T, Box<dyn Error>> {
    Err(message.into().into())
}

fn read_json(path: &Path) -> Result<(Value, String), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    Ok((serde_json::from_slice(&bytes)?, hash))
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Box<dyn Error>> {
    value
        .get(key)
        .ok_or_else(|| format!("missing field {key}").into())
}

fn queue_execution_usable(queue: &Value) -> bool {
    if queue.get("status").and_then(Value::as_str) == Some("succeeded") {
        return queue.get("exit_code").and_then(Value::as_i64) == Some(0);
    }
    // Retain an otherwise clean timing observation when only changing host
    // conditions denied promotion eligibility. This does not upgrade it to a
    // promotion-qualified queue success.
    queue.get("status").and_then(Value::as_str) == Some("rejected")
        && queue.get("purpose").and_then(Value::as_str) == Some("timing")
        && queue.get("exit_code").and_then(Value::as_i64) == Some(0)
        && queue
            .get("sampled_conditions_eligible")
            .and_then(Value::as_bool)
            == Some(false)
        && queue.get("overdue").and_then(Value::as_bool) == Some(false)
        && queue.get("files_unchanged").and_then(Value::as_bool) == Some(true)
        && queue
            .get("signal_or_missing_exit_code")
            .and_then(Value::as_bool)
            == Some(false)
        && queue.get("stop_requested").and_then(Value::as_bool) == Some(false)
        && queue.get("file_error").is_some_and(Value::is_null)
        && queue.get("validation").is_some_and(Value::is_null)
        && queue
            .get("violations")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
}

fn number(value: &Value, key: &str) -> Result<f64, Box<dyn Error>> {
    let n = required(value, key)?
        .as_f64()
        .ok_or_else(|| format!("{key} is not numeric"))?;
    if !n.is_finite() || n < 0.0 {
        return fail(format!("{key} is invalid"));
    }
    Ok(n)
}

fn distribution(samples: &[f64]) -> Result<Value, Box<dyn Error>> {
    if samples.is_empty() {
        return fail("empty distribution");
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
    let quantile = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
    Ok(json!({
        "count": sorted.len(), "min": sorted[0], "p05": quantile(0.05),
        "median": quantile(0.5), "mean": mean, "p95": quantile(0.95),
        "max": sorted[sorted.len()-1], "population_stddev": variance.sqrt(),
        "samples": samples,
    }))
}

fn condition_strata(path: &Path) -> Result<Value, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut counts = BTreeMap::<String, usize>::new();
    for (line_no, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), line_no + 1))?;
        let controls = v
            .pointer("/power/sample/controls")
            .ok_or("condition sample lacks controls")?;
        let key = serde_json::to_string(&json!({
            "power_source": controls.get("power_source"),
            "low_power_mode": controls.get("low_power_mode"),
            "pmset_power_mode": controls.get("pmset_power_mode"),
            "thermal_state": controls.get("thermal_state"),
            "cpu_speed_limit_percent": controls.get("cpu_speed_limit_percent"),
            "scheduler_limit_percent": controls.get("scheduler_limit_percent"),
        }))?;
        *counts.entry(key).or_default() += 1;
    }
    if counts.is_empty() {
        return fail("empty condition journal");
    }
    let strata = counts
        .into_iter()
        .map(|(key, count)| -> Result<Value, Box<dyn Error>> {
            Ok(json!({"controls": serde_json::from_str::<Value>(&key)?, "samples": count}))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"gating": false, "strata": strata}))
}

fn validate_observed_sequence(
    expected: &[String],
    observed: &[String],
) -> Result<(), Box<dyn Error>> {
    if observed != expected {
        return fail(format!(
            "incomplete or mislabeled sequence: expected {expected:?}, observed {observed:?}"
        ));
    }
    Ok(())
}

fn summarize(root: &Path) -> Result<Value, Box<dyn Error>> {
    let results = root.join("results");
    let expected_ids: Vec<String> = ARMS
        .iter()
        .map(|(name, _)| format!("{PREFIX}{name}{SUFFIX}"))
        .collect();
    let mut observed = Vec::new();
    for entry in fs::read_dir(&results)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name.starts_with(PREFIX) {
            observed.push(name);
        }
    }
    observed.sort();
    validate_observed_sequence(&expected_ids, &observed)?;

    let mut arms = Vec::new();
    let mut route_values = BTreeMap::<String, BTreeMap<String, Vec<f64>>>::new();
    let mut canonical_tokens: Option<Value> = None;
    let mut canonical_text: Option<Value> = None;
    let mut canonical_executable: Option<Value> = None;
    let mut canonical_inputs: Option<Value> = None;
    let mut canonical_model_config: Option<Value> = None;
    let mut canonical_metallib: Option<Value> = None;
    let mut canonical_generated_msl: Option<Value> = None;
    let mut run_means = Vec::<BTreeMap<String, f64>>::new();

    for ((short, route), id) in ARMS.iter().zip(expected_ids.iter()) {
        let directory = results.join(id);
        let (job, job_sha256) = read_json(&directory.join("job.json"))?;
        let (queue, queue_sha256) = read_json(&directory.join("report.json"))?;
        let (report, inference_sha256) = read_json(&directory.join("inference/report.json"))?;
        if required(&job, "id")? != id || required(&queue, "id")? != id {
            return fail(format!("identity mismatch for {id}"));
        }
        let executable = required(required(&job, "command")?, "executable")?;
        let inputs = required(&job, "inputs")?;
        for (canonical, actual, label) in [
            (&mut canonical_executable, executable, "executable"),
            (&mut canonical_inputs, inputs, "job inputs"),
        ] {
            if let Some(want) = canonical.as_ref() {
                if want != actual {
                    return fail(format!("cross-arm {label} identity mismatch: {id}"));
                }
            } else {
                *canonical = Some(actual.clone());
            }
        }
        if !queue_execution_usable(&queue) {
            return fail(format!("queue arm did not complete cleanly: {id}"));
        }
        let plan = if *route == "stacked" {
            "static-int8-stacked-ffn-cached"
        } else {
            "static-int8-ffn-cached"
        };
        if report.get("ane_weight_plan").and_then(Value::as_str) != Some(plan) {
            return fail(format!("weight plan mismatch: {id}"));
        }
        if report.get("ane_compile_budget").and_then(Value::as_u64) != Some(0)
            || report
                .get("ane_compile_budget_used")
                .and_then(Value::as_u64)
                != Some(0)
            || report.get("requests_completed").and_then(Value::as_u64) != Some(9)
            || report.get("inference_complete").and_then(Value::as_bool) != Some(true)
            || report
                .get("ane_execution_verified")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return fail(format!("execution/count/compile invariant failed: {id}"));
        }
        let cases = required(&report, "cases")?
            .as_array()
            .ok_or("cases is not an array")?;
        if cases.len() != 9 {
            return fail(format!("wrong request count: {id}"));
        }
        let mut values = BTreeMap::<String, Vec<f64>>::new();
        for case in cases {
            if case.get("ane_decode_steps").and_then(Value::as_u64) != Some(9) {
                return fail(format!("wrong ANE step count: {id}"));
            }
            let steps = required(case, "steps")?
                .as_array()
                .ok_or("steps is not an array")?;
            if steps.len() != 9 {
                return fail(format!("wrong measured step count: {id}"));
            }
            let tokens = required(case, "generated_tokens")?;
            let text = required(case, "generated_text")?;
            if tokens.as_array().map(Vec::len) != Some(10) {
                return fail(format!("wrong generated token count: {id}"));
            }
            if let Some(want) = &canonical_tokens {
                if want != tokens {
                    return fail(format!("generated token mismatch: {id}"));
                }
            } else {
                canonical_tokens = Some(tokens.clone());
            }
            if let Some(want) = &canonical_text {
                if want != text {
                    return fail(format!("generated text mismatch: {id}"));
                }
            } else {
                canonical_text = Some(text.clone());
            }
            let observation = required(case, "kernel_game_route_observation")?;
            if observation
                .get("compiler_calls_this_request")
                .and_then(Value::as_u64)
                != Some(0)
                || observation
                    .get("compile_free_request")
                    .and_then(Value::as_bool)
                    != Some(true)
                || observation.get("ane_decode_steps").and_then(Value::as_u64) != Some(9)
                || observation.get("output_tokens").and_then(Value::as_u64) != Some(10)
                || observation
                    .get("command_completed")
                    .and_then(Value::as_bool)
                    != Some(true)
            {
                return fail(format!("per-request route invariant failed: {id}"));
            }
            let job_exe = job.pointer("/command/executable/sha256");
            if observation.pointer("/executable/sha256") != job_exe {
                return fail(format!("executable identity mismatch: {id}"));
            }
            for (canonical, actual, label) in [
                (
                    &mut canonical_model_config,
                    required(observation, "model_config_sha256")?,
                    "model config",
                ),
                (
                    &mut canonical_metallib,
                    required(observation, "metallib")?,
                    "metallib",
                ),
                (
                    &mut canonical_generated_msl,
                    required(observation, "generated_msl_sha256")?,
                    "generated MSL",
                ),
            ] {
                if let Some(want) = canonical.as_ref() {
                    if want != actual {
                        return fail(format!("cross-request {label} identity mismatch: {id}"));
                    }
                } else {
                    *canonical = Some(actual.clone());
                }
            }
            for step in steps {
                for metric in METRICS {
                    values
                        .entry(metric.into())
                        .or_default()
                        .push(number(step, metric)?);
                }
            }
        }
        let mut means = BTreeMap::new();
        let mut distributions = serde_json::Map::new();
        for metric in METRICS {
            let samples = &values[metric];
            means.insert(
                metric.into(),
                samples.iter().sum::<f64>() / samples.len() as f64,
            );
            distributions.insert(metric.into(), distribution(samples)?);
            route_values
                .entry((*route).into())
                .or_default()
                .entry(metric.into())
                .or_default()
                .extend(samples);
        }
        run_means.push(means);
        arms.push(json!({
            "ordinal": arms.len()+1, "id": id, "sealed_label": short, "route": route,
            "weight_plan": plan, "request_count": 9, "steps_per_request": 9,
            "identities": {"job_sha256":job_sha256,"queue_report_sha256":queue_sha256,"inference_report_sha256":inference_sha256,
                "executable_sha256":job.pointer("/command/executable/sha256"), "inputs":job.get("inputs")},
            "conditions": condition_strata(&directory.join("conditions.jsonl"))?,
            "step_distributions_ms": distributions,
        }));
    }

    let mut route_distributions = serde_json::Map::new();
    for (route, metrics) in &route_values {
        let mut out = serde_json::Map::new();
        for metric in METRICS {
            out.insert(metric.into(), distribution(&metrics[metric])?);
        }
        route_distributions.insert(route.clone(), Value::Object(out));
    }
    let blocks = [(0, 4, "ABBA"), (4, 8, "BAAB"), (8, 12, "ABBA")].into_iter().enumerate().map(|(index, (start, end, order))| {
        let mut effects = serde_json::Map::new();
        for metric in METRICS {
            let mut stacked = Vec::new(); let mut baseline = Vec::new();
            for i in start..end { if ARMS[i].1 == "stacked" { stacked.push(run_means[i][metric]); } else { baseline.push(run_means[i][metric]); } }
            let a = stacked.iter().sum::<f64>() / stacked.len() as f64;
            let b = baseline.iter().sum::<f64>() / baseline.len() as f64;
            effects.insert(metric.into(), json!({"stacked_mean_ms":a,"baseline_mean_ms":b,"stacked_minus_baseline_ms":a-b,"baseline_over_stacked_ratio":b/a}));
        }
        json!({"block":index+1,"order":order,"arm_ordinals":((start+1)..=end).collect::<Vec<_>>(),"effects":effects})
    }).collect::<Vec<_>>();
    let mut paired_effects = serde_json::Map::new();
    for metric in METRICS {
        let deltas = blocks
            .iter()
            .map(|block| {
                block["effects"][metric]["stacked_minus_baseline_ms"]
                    .as_f64()
                    .expect("internally generated finite delta")
            })
            .collect::<Vec<_>>();
        let ratios = blocks
            .iter()
            .map(|block| {
                block["effects"][metric]["baseline_over_stacked_ratio"]
                    .as_f64()
                    .expect("internally generated finite ratio")
            })
            .collect::<Vec<_>>();
        paired_effects.insert(
            metric.into(),
            json!({
                "experimental_unit":"one complete four-process counterbalanced block",
                "block_count":3,
                "stacked_minus_baseline_ms":distribution(&deltas)?,
                "baseline_over_stacked_ratio":distribution(&ratios)?,
            }),
        );
    }

    Ok(json!({
        "schema":"rvllm.gemma4_ane_stacked_tournament_summary.v1",
        "complete":true, "promotion_authority":false,
        "claim_boundary":"complete retained exploratory process-arm comparison; conditions are reported, never used as a stability gate",
        "sequence":"ABBA/BAAB/ABBA", "arms":arms,
        "generated_tokens":canonical_tokens, "generated_text":canonical_text,
        "common_identities":{"executable":canonical_executable,"inputs":canonical_inputs,
            "model_config_sha256":canonical_model_config,"metallib":canonical_metallib,
            "generated_msl_sha256":canonical_generated_msl},
        "route_step_distributions_ms":route_distributions, "paired_blocks":blocks,
        "paired_effect_distributions":paired_effects,
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: rvllm_ane_tournament_summary QUEUE_ROOT OUTPUT_JSON")?,
    );
    let output = PathBuf::from(args.next().ok_or("missing OUTPUT_JSON")?);
    if args.next().is_some() {
        return fail("too many arguments");
    }
    let summary = summarize(&root)?;
    fs::write(&output, serde_json::to_vec_pretty(&summary)?)?;
    println!("{}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_retains_every_sample() {
        let v = distribution(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(v["count"], 4);
        assert_eq!(v["samples"], json!([4.0, 1.0, 3.0, 2.0]));
        assert_eq!(v["mean"], 2.5);
    }

    #[test]
    fn sealed_sequence_is_counterbalanced() {
        assert_eq!(
            ARMS.map(|(_, route)| route),
            [
                "stacked", "baseline", "baseline", "stacked", "baseline", "stacked", "stacked",
                "baseline", "stacked", "baseline", "baseline", "stacked"
            ]
        );
    }

    #[test]
    fn empty_distribution_is_rejected() {
        assert!(distribution(&[]).unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn incomplete_or_mislabeled_sequence_is_rejected() {
        let expected = vec!["one".into(), "two".into()];
        assert!(validate_observed_sequence(&expected, &["one".into()])
            .unwrap_err()
            .to_string()
            .contains("incomplete or mislabeled"));
        assert!(validate_observed_sequence(&expected, &["one".into(), "wrong".into()]).is_err());
    }

    #[test]
    fn conditions_only_timing_rejection_is_retained_but_failures_are_not() {
        let retained = json!({
            "status":"rejected", "purpose":"timing", "exit_code":0,
            "sampled_conditions_eligible":false, "overdue":false,
            "files_unchanged":true, "signal_or_missing_exit_code":false,
            "stop_requested":false, "file_error":null, "validation":null,
            "violations":[]
        });
        assert!(queue_execution_usable(&retained));
        let mut failed = retained.clone();
        failed["violations"] = json!([{"observation_error":"lost observer"}]);
        assert!(!queue_execution_usable(&failed));
        failed = retained;
        failed["exit_code"] = json!(1);
        assert!(!queue_execution_usable(&failed));
    }
}
