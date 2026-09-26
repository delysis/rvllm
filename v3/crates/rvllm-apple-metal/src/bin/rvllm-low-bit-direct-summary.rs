//! Strict, non-selective adjudication of the direct Gemma 4 N4-vs-N8 campaign.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

const PREFIX: &str = "g4metalbf16lowbit-direct-n4n8-";
const DATE: &str = "-20260925";
const ORDERS: [&str; 2] = ["abba", "baab"];
const CELLS: [(&str, &str, u64); 11] = [
    ("down", "w4a16", 1),
    ("down", "w4a16", 4),
    ("down", "w8a16", 1),
    ("down", "w8a16", 4),
    ("gate", "w8a16", 4),
    ("k", "w8a16", 4),
    ("o", "w4a16", 1),
    ("o", "w4a16", 4),
    ("up", "w8a16", 1),
    ("up", "w8a16", 4),
    ("v", "w4a16", 4),
];
const SAMPLES: usize = 18;
const MAX_DRIFT: f64 = 0.20;
const MIN_WIN: f64 = 1.05;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(message.into().into())
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_json(path: &Path) -> Result<(Value, String)> {
    let bytes = fs::read(path)?;
    Ok((serde_json::from_slice(&bytes)?, digest(&bytes)))
}
fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value
        .get(key)
        .ok_or_else(|| format!("missing {key}").into())
}
fn positive(value: &Value, key: &str) -> Result<f64> {
    let number = field(value, key)?
        .as_f64()
        .ok_or_else(|| format!("{key} is not numeric"))?;
    if !number.is_finite() || number <= 0.0 {
        return fail(format!("invalid {key}"));
    }
    Ok(number)
}
fn values(value: &Value, key: &str) -> Result<Vec<f64>> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not an array"))?
        .iter()
        .map(|item| {
            item.as_f64()
                .filter(|x| x.is_finite() && *x > 0.0)
                .ok_or_else(|| format!("invalid {key} sample").into())
        })
        .collect()
}
fn upper_median(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return fail("empty timing samples");
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(sorted[sorted.len() / 2])
}
fn relative_drift(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.min(b)
}
fn close(a: f64, b: f64) -> bool {
    relative_drift(a, b) <= 1e-9
}
fn distribution(raw: &[f64]) -> Result<Value> {
    let mut sorted = raw.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = upper_median(raw)?;
    Ok(
        json!({"count":raw.len(),"min":sorted[0],"upper_median":median,
        "mean":raw.iter().sum::<f64>() / raw.len() as f64,"max":sorted[sorted.len()-1],
        "raw_samples_in_dispatch_order":raw}),
    )
}
fn expected_ids() -> Vec<String> {
    CELLS
        .iter()
        .flat_map(|(role, format, m)| {
            ORDERS.map(|order| format!("{PREFIX}{role}-{format}-m{m}-{order}{DATE}"))
        })
        .collect()
}
fn role_name(short: &str) -> &'static str {
    match short {
        "k" => "key_projection",
        "v" => "value_projection",
        "o" => "output_projection",
        "gate" => "dense_gate_projection",
        "up" => "dense_up_projection",
        "down" => "dense_down_projection",
        _ => unreachable!("sealed cell list"),
    }
}
fn retained_conditions(path: &Path) -> Result<Value> {
    let bytes = fs::read(path)?;
    let text = std::str::from_utf8(&bytes)?;
    let observations = text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line_number, line)| {
            serde_json::from_str::<Value>(line)
                .map_err(|error| format!("{}:{}: {error}", path.display(), line_number + 1).into())
        })
        .collect::<Result<Vec<_>>>()?;
    if observations.is_empty() {
        return fail(format!("empty conditions: {}", path.display()));
    }
    Ok(json!({"sha256":digest(&bytes),"gating":false,"observations":observations}))
}
fn validate_queue(id: &str, job: &Value, report: &Value) -> Result<()> {
    if job.get("id").and_then(Value::as_str) != Some(id)
        || report.get("id").and_then(Value::as_str) != Some(id)
    {
        return fail(format!("outer identity mismatch: {id}"));
    }
    if report.get("status").and_then(Value::as_str) != Some("succeeded")
        || report.get("exit_code").and_then(Value::as_i64) != Some(0)
        || report
            .pointer("/validation/success")
            .and_then(Value::as_bool)
            != Some(true)
        || report.get("purpose").and_then(Value::as_str) != Some("exploratory_timing")
        || report.get("files_unchanged").and_then(Value::as_bool) != Some(true)
        || report
            .get("violations")
            .and_then(Value::as_array)
            .map_or(true, |v| !v.is_empty())
    {
        return fail(format!("unclean queue result: {id}"));
    }
    Ok(())
}
fn require_same(slot: &mut Option<Value>, actual: &Value, label: &str, id: &str) -> Result<()> {
    if slot.as_ref().is_some_and(|expected| expected != actual) {
        return fail(format!("cross-run {label} mismatch: {id}"));
    }
    if slot.is_none() {
        *slot = Some(actual.clone());
    }
    Ok(())
}

fn summarize(queue_root: &Path) -> Result<Value> {
    let results = queue_root.join("results");
    let mut observed = fs::read_dir(&results)?
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().to_string_lossy().into_owned();
            name.starts_with(PREFIX).then_some(name)
        })
        .collect::<Vec<_>>();
    let mut expected = expected_ids();
    observed.sort();
    expected.sort();
    if observed != expected {
        return fail(format!(
            "campaign membership mismatch; expected {expected:?}, observed {observed:?}"
        ));
    }

    let mut runs = BTreeMap::new();
    let (mut common_config, mut common_executable, mut common_msl) = (None, None, None);
    for (role, format, m) in CELLS {
        for order in ORDERS {
            let id = format!("{PREFIX}{role}-{format}-m{m}-{order}{DATE}");
            let dir = results.join(&id);
            let (job, job_sha256) = read_json(&dir.join("job.json"))?;
            let (report, queue_report_sha256) = read_json(&dir.join("report.json"))?;
            let (trial, trial_stdout_sha256) = read_json(&dir.join("trial.stdout"))?;
            validate_queue(&id, &job, &report)?;
            if trial.get("schema").and_then(Value::as_str)
                != Some("rvllm.metal_low_bit_real_weight_bf16.v1")
                || trial.get("candidate_schedule").and_then(Value::as_str) != Some("n4-vs-n8")
                || trial.get("direct_order").and_then(Value::as_str)
                    != Some(&order.to_ascii_uppercase())
                || trial.get("source_dtype").and_then(Value::as_str) != Some("Bf16")
                || trial.get("tensor_role").and_then(Value::as_str) != Some(role_name(role))
            {
                return fail(format!("trial identity/order mismatch: {id}"));
            }
            require_same(
                &mut common_config,
                field(&trial, "config_sha256")?,
                "config",
                &id,
            )?;
            require_same(
                &mut common_executable,
                field(&trial, "executable_sha256")?,
                "executable",
                &id,
            )?;
            require_same(
                &mut common_msl,
                field(&trial, "generated_msl_sha256")?,
                "generated MSL",
                &id,
            )?;
            let cases = field(&trial, "cases")?
                .as_array()
                .ok_or("cases is not an array")?;
            if cases.len() != 1 {
                return fail(format!("expected exactly one case: {id}"));
            }
            let case = &cases[0];
            if case.get("m").and_then(Value::as_u64) != Some(m)
                || case.pointer("/dispatch/format").and_then(Value::as_str) != Some(format)
                || case.pointer("/dispatch/role").and_then(Value::as_str) != Some(role_name(role))
                || case.get("guard_unchanged").and_then(Value::as_bool) != Some(true)
                || case.get("repeatable_output_bits").and_then(Value::as_bool) != Some(true)
                || case
                    .get("cross_schedule_output_bits_equal")
                    .and_then(Value::as_bool)
                    != Some(true)
                || case
                    .pointer("/dispatch/exact_correctness_dispatches_verified")
                    .and_then(Value::as_u64)
                    != Some(4)
                || case
                    .pointer("/dispatch/exact_timing_dispatch_count_verified")
                    .and_then(Value::as_bool)
                    != Some(true)
            {
                return fail(format!("case correctness/dispatch mismatch: {id}"));
            }
            let timing = field(case, "timing")?;
            let n4 = values(timing, "n4_ms")?;
            let n8 = values(timing, "n8_ms")?;
            if n4.len() != SAMPLES
                || n8.len() != SAMPLES
                || timing.get("samples_per_arm").and_then(Value::as_u64) != Some(SAMPLES as u64)
                || timing.get("blocks").and_then(Value::as_u64) != Some(9)
            {
                return fail(format!("timing cardinality mismatch: {id}"));
            }
            let n4_median = upper_median(&n4)?;
            let n8_median = upper_median(&n8)?;
            let n4_over_n8 = n8_median / n4_median;
            let n8_over_n4 = n4_median / n8_median;
            if !close(n4_median, positive(timing, "n4_median_ms")?)
                || !close(n8_median, positive(timing, "n8_median_ms")?)
                || !close(n4_over_n8, positive(timing, "n4_over_n8_speedup")?)
                || !close(n8_over_n4, positive(timing, "n8_over_n4_speedup")?)
                || !close(n4_over_n8 * n8_over_n4, 1.0)
            {
                return fail(format!("derived timing mismatch: {id}"));
            }
            let key = format!("{role}/{format}/M{m}/{order}");
            runs.insert(key, json!({"id":id,"order":order.to_ascii_uppercase(),"role":role,
                "tensor":trial.get("tensor"),"source_tensor_sha256":trial.get("source_tensor_sha256"),
                "source_file":trial.get("source_file"),"source_file_offset":trial.get("source_file_offset"),
                "source_tensor_bytes":trial.get("source_tensor_bytes"),"shape":trial.get("shape"),
                "accuracy":case.get("accuracy"),"identity":case.get("identity"),
                "kernel_identities":{"n4":timing.get("n4_kernel"),"n8":timing.get("n8_kernel")},
                "timing":{"method":timing.get("method"),"n4_ms":distribution(&n4)?,"n8_ms":distribution(&n8)?,
                    "n4_over_n8_speedup":n4_over_n8,"n8_over_n4_speedup":n8_over_n4},
                "queue":{"sampled_conditions_eligible":report.get("sampled_conditions_eligible"),
                    "measurement":report.get("measurement"),"claim":report.get("claim")},
                "conditions":retained_conditions(&dir.join("conditions.jsonl"))?,
                "evidence_sha256":{"job_json":job_sha256,"queue_report_json":queue_report_sha256,
                    "trial_stdout":trial_stdout_sha256}}));
        }
    }

    let mut decisions = Vec::new();
    let mut n4_wins = 0;
    let mut n8_wins = 0;
    for (role, format, m) in CELLS {
        let a = &runs[&format!("{role}/{format}/M{m}/abba")];
        let b = &runs[&format!("{role}/{format}/M{m}/baab")];
        for key in [
            "role",
            "tensor",
            "source_tensor_sha256",
            "source_file",
            "source_file_offset",
            "source_tensor_bytes",
            "shape",
            "accuracy",
            "identity",
            "kernel_identities",
        ] {
            if a[key] != b[key] {
                return fail(format!("ABBA/BAAB {key} mismatch: {role}/{format}/M{m}"));
            }
        }
        let a4 = a["timing"]["n4_ms"]["upper_median"].as_f64().unwrap();
        let b4 = b["timing"]["n4_ms"]["upper_median"].as_f64().unwrap();
        let a8 = a["timing"]["n8_ms"]["upper_median"].as_f64().unwrap();
        let b8 = b["timing"]["n8_ms"]["upper_median"].as_f64().unwrap();
        let ar = a8 / a4;
        let br = b8 / b4;
        let drifts = (
            relative_drift(a4, b4),
            relative_drift(a8, b8),
            relative_drift(ar, br),
        );
        let stable = drifts.0 <= MAX_DRIFT && drifts.1 <= MAX_DRIFT && drifts.2 <= MAX_DRIFT;
        let winner = if stable && ar >= MIN_WIN && br >= MIN_WIN {
            n4_wins += 1;
            Some("n4")
        } else if stable && 1.0 / ar >= MIN_WIN && 1.0 / br >= MIN_WIN {
            n8_wins += 1;
            Some("n8")
        } else {
            None
        };
        let disposition = if !stable {
            "inconclusive_cross_order_drift"
        } else if winner.is_some() {
            "stable_operator_winner"
        } else {
            "stable_no_5_percent_winner"
        };
        decisions.push(json!({"role":role,"format":format,"m":m,
            "abba":{"n4_median_ms":a4,"n8_median_ms":a8,"n4_over_n8_speedup":ar,"n8_over_n4_speedup":1.0/ar},
            "baab":{"n4_median_ms":b4,"n8_median_ms":b8,"n4_over_n8_speedup":br,"n8_over_n4_speedup":1.0/br},
            "relative_drift":{"n4_median":drifts.0,"n8_median":drifts.1,"ratio":drifts.2},
            "winner":winner,"disposition":disposition}));
    }
    Ok(
        json!({"schema":"rvllm.gemma4_metal_low_bit_n4_vs_n8_direct_summary.v1",
        "complete":true,"promotion_authority":false,
        "claim_boundary":"real-checkpoint projection-operator evidence only; no full-route, model-quality, shipping-selector, or promotion claim",
        "all_raw_timing_and_conditions_retained":true,"conditions_are_not_a_gate":true,
        "policy":{"minimum_winner_speedup_in_both_orders":MIN_WIN,"maximum_abba_baab_relative_drift":MAX_DRIFT,
            "drift_fields":["n4_median","n8_median","ratio"],"median":"upper median"},
        "common_identities":{"config_sha256":common_config,"executable_sha256":common_executable,
            "generated_msl_sha256":common_msl},"job_count":runs.len(),"cell_count":decisions.len(),
        "stable_winner_counts":{"n4":n4_wins,"n8":n8_wins},
        "campaign_disposition":if n4_wins+n8_wins == CELLS.len() {"all_cells_have_stable_operator_winners_not_promotion"}
            else {"mixed_or_inconclusive_operator_results_not_promotion"},"decisions":decisions,"runs":runs}),
    )
}

fn markdown(summary: &Value) -> String {
    let mut out = format!(
        "# Gemma 4 direct N4-vs-N8 Metal results\n\nAll 22 jobs and 11 cells were validated. This is projection-operator evidence only: it is not a full-route, model-quality, shipping-selector, or promotion claim. Raw dispatch-order timings, queue measurements, condition observations, and evidence hashes remain in `summary.json`.\n\nCampaign disposition: **{}**.\n\n| Role | Format | M | ABBA N4/N8 | BAAB N4/N8 | N4 drift | N8 drift | Ratio drift | Verdict |\n|---|---|---:|---:|---:|---:|---:|---:|---|\n",
        summary["campaign_disposition"].as_str().unwrap()
    );
    for d in summary["decisions"].as_array().unwrap() {
        let verdict = d["winner"]
            .as_str()
            .map(|w| format!("{w} stable winner"))
            .unwrap_or_else(|| d["disposition"].as_str().unwrap().to_owned());
        out.push_str(&format!(
            "| {} | {} | {} | {:.3}x | {:.3}x | {:.1}% | {:.1}% | {:.1}% | {} |\n",
            d["role"].as_str().unwrap(),
            d["format"].as_str().unwrap(),
            d["m"],
            d["abba"]["n4_over_n8_speedup"].as_f64().unwrap(),
            d["baab"]["n4_over_n8_speedup"].as_f64().unwrap(),
            100.0 * d["relative_drift"]["n4_median"].as_f64().unwrap(),
            100.0 * d["relative_drift"]["n8_median"].as_f64().unwrap(),
            100.0 * d["relative_drift"]["ratio"].as_f64().unwrap(),
            verdict
        ));
    }
    out.push_str("\nA winner requires at least 1.05x in both ABBA and BAAB and no more than 20% ABBA/BAAB drift in the N4 median, N8 median, or reciprocal timing ratio. Failed and inconclusive cells remain represented; no favorable subset is selected.\n");
    out
}

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: rvllm-low-bit-direct-summary QUEUE_ROOT OUT.json OUT.md")?,
    );
    let json_path = PathBuf::from(args.next().ok_or("missing OUT.json")?);
    let readme_path = PathBuf::from(args.next().ok_or("missing OUT.md")?);
    if args.next().is_some() {
        return fail("too many arguments");
    }
    let summary = summarize(&root)?;
    fs::write(json_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(readme_path, markdown(&summary))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn campaign_is_exactly_twenty_two_jobs() {
        assert_eq!(expected_ids().len(), 22);
    }
    #[test]
    fn upper_median_is_used_for_even_samples() {
        assert_eq!(upper_median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 3.0);
    }
    #[test]
    fn drift_is_symmetric_and_uses_smaller_denominator() {
        assert_eq!(relative_drift(8.0, 10.0), relative_drift(10.0, 8.0));
        assert!((relative_drift(8.0, 10.0) - 0.25).abs() < 1e-12);
    }
    #[test]
    fn distribution_retains_dispatch_order() {
        assert_eq!(
            distribution(&[3.0, 1.0, 2.0]).unwrap()["raw_samples_in_dispatch_order"],
            json!([3.0, 1.0, 2.0])
        );
    }
    #[test]
    fn reciprocal_ratios_identify_each_winner() {
        let n4_over_n8 = 1.05_f64;
        assert!(n4_over_n8 >= MIN_WIN);
        assert!(!(1.0 / n4_over_n8 >= MIN_WIN));
    }
}
