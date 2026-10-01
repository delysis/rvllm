//! Fail-closed offline referee for the predeclared full-route profile sequence.
//! This does not turn independent-process profiles into ABBA qualification.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const LABELS: [&str; 5] = ["off-a", "pipeline", "off-b", "combined", "off-c"];
const PIPELINE: [(&str, u64); 4] = [
    ("research_prefill_pipeline_gemm", 48),
    ("research_prefill_pipeline_qkv", 48),
    ("research_prefill_pipeline_raw_norm_projection", 96),
    ("research_prefill_pipeline_raw_norm", 96),
];
const COMBINED: [(&str, u64); 6] = [
    ("research_prefill_combined_gemm", 48),
    ("research_prefill_combined_qkv", 48),
    ("research_prefill_combined_raw_norm_projection", 96),
    ("research_prefill_combined_raw_norm", 96),
    ("research_prefill_combined_d256", 40),
    ("research_prefill_combined_d512", 8),
];

type Result<T> = std::result::Result<T, String>;

#[derive(Debug)]
struct Case {
    name: String,
    prompt_ids: Vec<u64>,
    generated_ids: Vec<u64>,
    prefill_ms: Vec<f64>,
    decode_ms: Vec<f64>,
}

#[derive(Debug)]
struct Arm {
    label: &'static str,
    stratum: Value,
    sampled_conditions_eligible: bool,
    executable_sha256: String,
    common_pins: Vec<(String, String)>,
    receipt_sha256: Value,
    cases: Vec<Case>,
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value.get(key).ok_or_else(|| format!("missing field {key}"))
}

fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not text"))
}

fn number_field(value: &Value, key: &str) -> Result<f64> {
    let number = field(value, key)?
        .as_f64()
        .ok_or_else(|| format!("{key} is not numeric"))?;
    if !number.is_finite() || number <= 0.0 {
        return Err(format!("{key} must be finite and positive"));
    }
    Ok(number)
}

fn ids(value: &Value, key: &str) -> Result<Vec<u64>> {
    let values = field(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not an array"))?;
    if values.is_empty() {
        return Err(format!("{key} is empty"));
    }
    values
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| format!("{key} contains a non-ID")))
        .collect()
}

fn read_json(path: &Path) -> Result<(Value, String)> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let sha = format!("{:x}", Sha256::digest(&bytes));
    let value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((value, sha))
}

fn expected_dispatch(label: &str) -> &'static [(&'static str, u64)] {
    match label {
        "pipeline" => &PIPELINE,
        "combined" => &COMBINED,
        _ => &[],
    }
}

fn common_pins(job: &Value) -> Result<Vec<(String, String)>> {
    let inputs = field(job, "inputs")?
        .as_array()
        .ok_or("job inputs are not an array")?;
    let mut pins = Vec::new();
    for input in inputs {
        let path = text_field(input, "path")?;
        if [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "varied-prompts.jsonl",
        ]
        .iter()
        .any(|suffix| path.ends_with(suffix))
        {
            let suffix = path.rsplit('/').next().ok_or("input has no basename")?;
            pins.push((suffix.to_owned(), text_field(input, "sha256")?.to_owned()));
        }
    }
    pins.sort();
    pins.dedup();
    if pins.len() != 4 {
        return Err("expected exactly four common model/prompt input pins".into());
    }
    Ok(pins)
}

fn read_arm(dir: &Path, label: &'static str) -> Result<Arm> {
    let (queue, queue_sha) = read_json(&dir.join("report.json"))?;
    if text_field(&queue, "status")? != "succeeded"
        || field(&queue, "exit_code")?.as_i64() != Some(0)
        || !field(&queue, "violations")?
            .as_array()
            .is_some_and(Vec::is_empty)
    {
        return Err(format!("{label}: queue job not clean and eligible"));
    }
    let sampled_conditions_eligible = field(&queue, "sampled_conditions_eligible")?
        .as_bool()
        .ok_or("sampled_conditions_eligible is not a boolean")?;
    let stratum = field(field(&queue, "measurement")?, "comparison_stratum")?.clone();
    let (job, job_sha) = read_json(&dir.join("job.json"))?;
    let executable_sha256 =
        text_field(field(field(&job, "command")?, "executable")?, "sha256")?.to_owned();
    let pins = common_pins(&job)?;
    let (profile, profile_sha) = read_json(&dir.join("trial.stdout"))?;
    if text_field(&profile, "schema")? != "rvllm.apple_metal_text_session_profile.v1"
        || text_field(&profile, "backend")? != "direct"
        || text_field(&profile, "metal_compute_dtype")? != "bfloat16"
        || field(&profile, "sample_count")?.as_u64() != Some(3)
    {
        return Err(format!("{label}: wrong profile shape, backend, or dtype"));
    }
    let samples = field(&profile, "samples")?
        .as_array()
        .ok_or("profile samples are not an array")?;
    if samples.len() != 3 {
        return Err(format!("{label}: expected all three samples"));
    }
    let mut cases: Vec<Case> = Vec::new();
    for (sample_index, sample) in samples.iter().enumerate() {
        if text_field(sample, "status")? != "pass"
            || field(sample, "case_count")?.as_u64() != Some(2)
        {
            return Err(format!("{label}: sample {sample_index} failed"));
        }
        let sample_cases = field(sample, "cases")?
            .as_array()
            .ok_or("sample cases are not an array")?;
        if sample_cases.len() != 2 {
            return Err(format!(
                "{label}: sample {sample_index} has wrong case count"
            ));
        }
        for (case_index, raw) in sample_cases.iter().enumerate() {
            if text_field(raw, "status")? != "pass"
                || field(raw, "library_compiles")?.as_u64() != Some(0)
                || field(raw, "pipeline_state_compiles")?.as_u64() != Some(0)
            {
                return Err(format!(
                    "{label}: sample {sample_index} case {case_index} failed or compiled"
                ));
            }
            let dispatch = field(field(raw, "research_dispatch")?, "counts")?
                .as_object()
                .ok_or("dispatch counts are not an object")?;
            let expected = expected_dispatch(label);
            if dispatch.len() != expected.len()
                || expected.iter().any(|(name, count)| {
                    dispatch.get(*name).and_then(Value::as_u64) != Some(*count)
                })
                || field(field(raw, "research_dispatch")?, "overflowed")?.as_bool() != Some(false)
            {
                return Err(format!(
                    "{label}: sample {sample_index} case {case_index} route mismatch"
                ));
            }
            let name = text_field(raw, "name")?.to_owned();
            let prompt_ids = ids(raw, "prompt_token_ids")?;
            let generated_ids = ids(raw, "generated_token_ids")?;
            let prefill_ms = number_field(raw, "prefill_ms")?;
            let decode_ms = number_field(raw, "decode_ms")?;
            if sample_index == 0 {
                cases.push(Case {
                    name,
                    prompt_ids,
                    generated_ids,
                    prefill_ms: vec![prefill_ms],
                    decode_ms: vec![decode_ms],
                });
            } else {
                let saved = cases.get_mut(case_index).ok_or("case order changed")?;
                if saved.name != name
                    || saved.prompt_ids != prompt_ids
                    || saved.generated_ids != generated_ids
                {
                    return Err(format!(
                        "{label}: sample {sample_index} changed work or output"
                    ));
                }
                saved.prefill_ms.push(prefill_ms);
                saved.decode_ms.push(decode_ms);
            }
        }
    }
    if cases[0].name == cases[1].name {
        return Err(format!("{label}: duplicate case names"));
    }
    Ok(Arm {
        label,
        stratum,
        sampled_conditions_eligible,
        executable_sha256,
        common_pins: pins,
        receipt_sha256: json!({
            "queue_report": queue_sha,
            "job": job_sha,
            "profile_stdout": profile_sha,
        }),
        cases,
    })
}

fn median(values: &[f64]) -> Result<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("invalid median inputs".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

fn drift(values: &[f64]) -> Result<f64> {
    let anchor = *values.first().ok_or("empty drift inputs")?;
    if !anchor.is_finite() || anchor <= 0.0 {
        return Err("invalid drift anchor".into());
    }
    Ok(values
        .iter()
        .map(|v| (v - anchor).abs() / anchor)
        .fold(0.0_f64, f64::max))
}

fn ids_sha256(ids: &[u64]) -> Result<String> {
    let encoded = serde_json::to_vec(ids).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn summarize(arms: &[Arm]) -> Result<Value> {
    if arms.len() != LABELS.len() {
        return Err("wrong arm count".into());
    }
    let control = &arms[0];
    for (index, arm) in arms.iter().enumerate() {
        if arm.label != LABELS[index]
            || arm.executable_sha256 != control.executable_sha256
            || arm.common_pins != control.common_pins
        {
            return Err(format!(
                "{}: order, stratum, executable, or input identity mismatch",
                arm.label
            ));
        }
        for case_index in 0..2 {
            let case = &arm.cases[case_index];
            let original = &control.cases[case_index];
            if case.name != original.name
                || case.prompt_ids != original.prompt_ids
                || case.generated_ids != original.generated_ids
            {
                return Err(format!(
                    "{}: case {case_index} differs from control work or tokens",
                    arm.label
                ));
            }
        }
    }
    let mut cases = Vec::new();
    for case_index in 0..2 {
        let mut arm_data = Vec::new();
        for arm in arms {
            let case = &arm.cases[case_index];
            arm_data.push(json!({
                "arm": arm.label,
                "prefill_ms": case.prefill_ms,
                "decode_ms": case.decode_ms,
                "prefill_median_ms": median(&case.prefill_ms)?,
                "prefill_within_arm_drift": drift(&case.prefill_ms)?,
            }));
        }
        let off_a = &arms[0].cases[case_index].prefill_ms;
        let off_b = &arms[2].cases[case_index].prefill_ms;
        let off_c = &arms[4].cases[case_index].prefill_ms;
        let mut pipeline_controls = off_a.clone();
        pipeline_controls.extend_from_slice(off_b);
        let mut combined_controls = off_b.clone();
        combined_controls.extend_from_slice(off_c);
        let pipeline_median = median(&arms[1].cases[case_index].prefill_ms)?;
        let combined_median = median(&arms[3].cases[case_index].prefill_ms)?;
        let pipeline_control_drift = (median(off_b)? - median(off_a)?).abs() / median(off_a)?;
        let combined_control_drift = (median(off_c)? - median(off_b)?).abs() / median(off_b)?;
        let pipeline_within_drift = [0, 1, 2]
            .iter()
            .map(|i| drift(&arms[*i].cases[case_index].prefill_ms))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(0.0_f64, f64::max);
        let combined_within_drift = [2, 3, 4]
            .iter()
            .map(|i| drift(&arms[*i].cases[case_index].prefill_ms))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(0.0_f64, f64::max);
        let pipeline_conditions_eligible = [0, 1, 2]
            .iter()
            .all(|i| arms[*i].sampled_conditions_eligible && arms[*i].stratum == arms[0].stratum);
        let combined_conditions_eligible = [2, 3, 4]
            .iter()
            .all(|i| arms[*i].sampled_conditions_eligible && arms[*i].stratum == arms[2].stratum);
        cases.push(json!({
            "name": control.cases[case_index].name,
            "prompt_tokens": control.cases[case_index].prompt_ids.len(),
            "generated_tokens": control.cases[case_index].generated_ids.len(),
            "prompt_ids_sha256": ids_sha256(&control.cases[case_index].prompt_ids)?,
            "generated_ids_sha256": ids_sha256(&control.cases[case_index].generated_ids)?,
            "arms": arm_data,
            "pipeline": {
                "control_median_ms": median(&pipeline_controls)?,
                "candidate_median_ms": pipeline_median,
                "descriptive_ratio": median(&pipeline_controls)? / pipeline_median,
                "bracket_control_drift": pipeline_control_drift,
                "maximum_within_arm_drift": pipeline_within_drift,
                "conditions_eligible": pipeline_conditions_eligible,
                "five_percent_drift_pass": pipeline_conditions_eligible && pipeline_control_drift <= 0.05 && pipeline_within_drift <= 0.05,
            },
            "combined": {
                "control_median_ms": median(&combined_controls)?,
                "candidate_median_ms": combined_median,
                "descriptive_ratio": median(&combined_controls)? / combined_median,
                "bracket_control_drift": combined_control_drift,
                "maximum_within_arm_drift": combined_within_drift,
                "conditions_eligible": combined_conditions_eligible,
                "five_percent_drift_pass": combined_conditions_eligible && combined_control_drift <= 0.05 && combined_within_drift <= 0.05,
            },
        }));
    }
    Ok(json!({
        "schema": "rvllm.prefill_route_profile_summary.v1",
        "claim": "diagnostic independent-process profile only; not ABBA/BAAB or promotion",
        "observed_conditions": arms.iter().map(|arm| json!({
            "arm": arm.label,
            "stratum": arm.stratum,
            "sampled_conditions_eligible": arm.sampled_conditions_eligible,
        })).collect::<Vec<_>>(),
        "executable_sha256": control.executable_sha256,
        "common_input_pins": control.common_pins,
        "receipts": arms.iter().map(|arm| json!({"arm":arm.label,"sha256":arm.receipt_sha256})).collect::<Vec<_>>(),
        "cases": cases,
    }))
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 7 {
        return Err(
            "usage: rvllm-prefill-route-profile-summary OFF_A_DIR PIPELINE_DIR OFF_B_DIR COMBINED_DIR OFF_C_DIR OUT.json"
                .into(),
        );
    }
    let dirs: Vec<PathBuf> = args[1..6].iter().map(PathBuf::from).collect();
    let mut arms = Vec::new();
    for (dir, label) in dirs.iter().zip(LABELS) {
        arms.push(read_arm(dir, label)?);
    }
    let summary = summarize(&arms)?;
    let output = PathBuf::from(&args[6]);
    let bytes = serde_json::to_vec_pretty(&summary).map_err(|e| e.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|e| format!("{}: {e}", output.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("{}: {e}", output.display()))?;
    println!("{}", output.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{drift, median, summarize, Arm, Case, LABELS};
    use serde_json::json;

    fn arms() -> Vec<Arm> {
        LABELS
            .iter()
            .enumerate()
            .map(|(index, label)| Arm {
                label,
                stratum: json!({"power_source":"ac","pmset_power_mode":2}),
                sampled_conditions_eligible: true,
                executable_sha256: "exe".into(),
                common_pins: vec![("model.safetensors".into(), "model".into())],
                receipt_sha256: json!({"queue_report":"receipt"}),
                cases: ["first", "second"]
                    .iter()
                    .map(|name| Case {
                        name: (*name).into(),
                        prompt_ids: vec![1, 2],
                        generated_ids: vec![3, 4],
                        prefill_ms: vec![if index % 2 == 0 { 100.0 } else { 20.0 }; 3],
                        decode_ms: vec![10.0; 3],
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn median_retains_every_sample() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
        assert!(median(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn drift_is_relative_to_first_observation() {
        assert!((drift(&[100.0, 104.0, 95.0]).unwrap() - 0.05).abs() < 1e-12);
        assert!(drift(&[0.0, 1.0]).is_err());
    }

    #[test]
    fn summary_keeps_all_arms_rejects_output_drift_and_marks_mixed_conditions() {
        let mut receipts = arms();
        let summary = summarize(&receipts).unwrap();
        assert_eq!(summary["cases"][0]["pipeline"]["descriptive_ratio"], 5.0);
        assert_eq!(summary["cases"][1]["combined"]["descriptive_ratio"], 5.0);
        receipts[3].cases[0].generated_ids[1] = 5;
        assert!(summarize(&receipts).is_err());
        receipts[3].cases[0].generated_ids[1] = 4;
        receipts[4].stratum["pmset_power_mode"] = json!(1);
        let mixed = summarize(&receipts).unwrap();
        assert_eq!(mixed["cases"][0]["combined"]["conditions_eligible"], false);
        assert_eq!(
            mixed["cases"][0]["combined"]["five_percent_drift_pass"],
            false
        );
    }
}
