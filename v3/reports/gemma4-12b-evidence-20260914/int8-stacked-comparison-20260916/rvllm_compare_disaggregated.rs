//! Compare identical work under matching sampled controls.
#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
use rvllm_runtime::apple_measurement::compare_phase_measurements;
#[cfg(target_os = "macos")]
use serde_json::{json, Value};

#[cfg(target_os = "macos")]
const IDENTITY: [&str; 6] = [
    "model_dir",
    "config_sha256",
    "measurement_environment",
    "global_context_capacity",
    "execution_order",
    "metal_residency",
];

#[cfg(target_os = "macos")]
fn same_fields(a: &Value, b: &Value, keys: &[&str]) -> Result<(), String> {
    for key in keys {
        if a[key].is_null() || a[key] != b[key] {
            return Err(format!("unmatched or missing {key}"));
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn cases(report: &Value) -> Result<&[Value], String> {
    report["cases"]
        .as_array()
        .filter(|v| !v.is_empty())
        .map(Vec::as_slice)
        .ok_or_else(|| "missing or empty cases".into())
}

#[cfg(target_os = "macos")]
fn validate_stacked_qualification(
    baseline: &Value,
    candidate: &Value,
    qualification: &Value,
) -> Result<(), String> {
    let plans = (
        baseline["ane_weight_plan"].as_str(),
        candidate["ane_weight_plan"].as_str(),
    );
    if !matches!(
        plans,
        (
            Some("static-int8-ffn-cached"),
            Some("static-int8-stacked-ffn-cached")
        ) | (
            Some("static-int8-stacked-ffn-cached"),
            Some("static-int8-ffn-cached")
        )
    ) {
        return Err("qualification permits only original versus stacked INT8 FFN layouts".into());
    }
    same_fields(baseline, qualification, &IDENTITY)?;
    if qualification["schema"] != "rvllm.metal_prefill_ane_decode.v1"
        || qualification["ane_weight_plan"] != "static-int8-stacked-ffn-checked"
        || qualification["qualification_complete"] != true
        || qualification["inference_complete"] != true
        || qualification["cpu_or_gpu_decode_fallback"] != false
        || qualification["diagnostic_journal_enabled"] != true
        || qualification["loaded_ane_programs"] != 210
        || qualification["ane_compile_budget_used"] != 0
    {
        return Err("incomplete or unexpected stacked FFN qualification".into());
    }
    let qualified = cases(qualification)?;
    if qualification["references_requested"].as_u64() != Some(qualified.len() as u64)
        || qualification["references_completed"].as_u64() != Some(qualified.len() as u64)
    {
        return Err("qualification reference counts disagree".into());
    }
    let mut total_steps = 0_u64;
    for case in qualified {
        if case["matches_reference"] != true {
            return Err("qualification output does not match its reference".into());
        }
        total_steps = total_steps
            .checked_add(
                case["ane_decode_steps"]
                    .as_u64()
                    .ok_or("qualification decode count missing")?,
            )
            .ok_or("qualification step count overflow")?;
    }
    let checks = qualification["stacked_ffn_checks_per_layer"]
        .as_array()
        .ok_or("qualification comparison counts missing")?;
    if total_steps == 0
        || checks.len() != 48
        || checks
            .iter()
            .any(|count| count.as_u64() != Some(total_steps))
    {
        return Err("qualification did not compare every FFN on every decode step".into());
    }
    // Scope the exception to workloads covered by the bit-parity run.
    for report in [baseline, candidate] {
        for timed in cases(report)? {
            if !qualified.iter().any(|case| {
                same_fields(
                    timed,
                    case,
                    &["prompt_token_ids", "generated_tokens", "ane_decode_steps"],
                )
                .is_ok()
            }) {
                return Err("timed workload is outside the stacked qualification".into());
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn compare_reports(
    baseline: &Value,
    candidate: &Value,
    qualification: Option<&Value>,
) -> Result<Value, String> {
    for report in [baseline, candidate] {
        if report["schema"] != "rvllm.metal_prefill_ane_decode.v1"
            || report["inference_complete"] != true
            || report["layer_state_capture_enabled"] != false
            || report["diagnostic_journal_enabled"] != false
            || report["cpu_or_gpu_decode_fallback"] != false
            || report["ane_compile_budget_used"] != 0
        {
            return Err("comparison requires completed zero-compile inference without fallback, layer capture or driver journal".into());
        }
    }
    same_fields(baseline, candidate, &IDENTITY)?;
    let weight_comparison = if let Some(qualification) = qualification {
        validate_stacked_qualification(baseline, candidate, qualification)?;
        "qualified-stacked-int8-layout"
    } else {
        same_fields(baseline, candidate, &["ane_weight_plan"])?;
        "same-plan"
    };
    let before = cases(baseline)?;
    let after = cases(candidate)?;
    if before.len() != after.len() {
        return Err("case counts differ".into());
    }
    let mut comparisons = Vec::with_capacity(before.len());
    for (before, after) in before.iter().zip(after) {
        same_fields(
            before,
            after,
            &[
                "prompt_token_ids",
                "generated_tokens",
                "ane_decode_steps",
                "metal_decode_steps",
                "prefill_command_buffers",
            ],
        )?;
        let prefill = compare_phase_measurements(
            &before["prefill_measurement"],
            &after["prefill_measurement"],
        )?;
        let first = before["steps"].as_array().ok_or("missing baseline steps")?;
        let second = after["steps"].as_array().ok_or("missing candidate steps")?;
        if first.len() != second.len()
            || before["ane_decode_steps"].as_u64() != Some(first.len() as u64)
        {
            return Err("decode work counts disagree".into());
        }
        let decode = first
            .iter()
            .zip(second)
            .map(|(before, after)| {
                same_fields(before, after, &["position", "input_token", "next_token"])?;
                compare_phase_measurements(&before["measurement"], &after["measurement"])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let gpu_ratio = before["metal_gpu_execution_ms"]
            .as_f64()
            .zip(after["metal_gpu_execution_ms"].as_f64())
            .filter(|(before, after)| {
                before.is_finite() && after.is_finite() && *before > 0.0 && *after > 0.0
            })
            .map(|(before, after)| before / after);
        comparisons.push(json!({"prefill":prefill,"decode_steps":decode,
            "baseline_over_candidate_gpu_interval":gpu_ratio}));
    }
    Ok(json!({"schema":"rvllm.apple_paired_comparison.v1",
        "weight_comparison":weight_comparison,
        "baseline_weight_plan":baseline["ane_weight_plan"],
        "candidate_weight_plan":candidate["ane_weight_plan"],
        "cases":comparisons,
        "claim":"One paired observation, not a speedup conclusion. Repeat in alternating order, preserve power strata and check baseline drift. CPU cycles do not normalize device time."}))
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use sha2::{Digest, Sha256};
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.len() != 2 && !(paths.len() == 4 && paths[2] == "--stacked-qualification") {
        return Err("usage: rvllm_compare_disaggregated BASELINE_REPORT CANDIDATE_REPORT [--stacked-qualification CHECKED_REPORT]".into());
    }
    let baseline: Value = serde_json::from_slice(&std::fs::read(&paths[0])?)?;
    let candidate: Value = serde_json::from_slice(&std::fs::read(&paths[1])?)?;
    let qualification = if paths.len() == 4 {
        let bytes = std::fs::read(&paths[3])?;
        Some((
            serde_json::from_slice::<Value>(&bytes)?,
            format!("{:x}", Sha256::digest(bytes)),
        ))
    } else {
        None
    };
    let mut comparison = compare_reports(
        &baseline,
        &candidate,
        qualification.as_ref().map(|(report, _)| report),
    )?;
    comparison["baseline"] = json!(paths[0]);
    comparison["candidate"] = json!(paths[1]);
    if let Some((_, digest)) = qualification {
        comparison["stacked_qualification"] = json!({"path":paths[3],"sha256":digest});
    }
    println!("{}", serde_json::to_string_pretty(&comparison)?);
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Apple measurement comparison requires macOS");
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn fixtures() -> (Value, Value, Value) {
        let phase = |units, ms| {
            json!({
                "schema":"rvllm.apple_phase_measurement.v1",
                "sampled_controls_eligible":true,
                "comparison_stratum":{"power_source":"ac","low_power_mode":false,"thermal_state":0},
                "units":units,"wall_ms":ms
            })
        };
        let baseline = json!({
            "schema":"rvllm.metal_prefill_ane_decode.v1",
            "inference_complete":true, "layer_state_capture_enabled":false,
            "diagnostic_journal_enabled":false,"cpu_or_gpu_decode_fallback":false,
            "ane_compile_budget_used":0,"model_dir":"/pinned-model","config_sha256":"config",
            "measurement_environment":{"device":"fixture"},"global_context_capacity":1024,
            "execution_order":"prefill-decode-per-request","metal_residency":"retained-through-ane-decode",
            "ane_weight_plan":"static-int8-ffn-cached",
            "cases":[{
                "prompt_token_ids":[2,3],"generated_tokens":[10,20],
                "ane_decode_steps":1,"metal_decode_steps":0,"prefill_command_buffers":1,
                "matches_reference":true,"prefill_measurement":phase(2,100),
                "steps":[{"position":2,"input_token":10,"next_token":20,"measurement":phase(1,100)}]
            }]
        });
        let mut candidate = baseline.clone();
        candidate["ane_weight_plan"] = json!("static-int8-stacked-ffn-cached");
        candidate["cases"][0]["steps"][0]["measurement"]["wall_ms"] = json!(80);
        let mut qualification = baseline.clone();
        qualification["ane_weight_plan"] = json!("static-int8-stacked-ffn-checked");
        qualification["qualification_complete"] = json!(true);
        qualification["diagnostic_journal_enabled"] = json!(true);
        qualification["loaded_ane_programs"] = json!(210);
        qualification["references_requested"] = json!(1);
        qualification["references_completed"] = json!(1);
        qualification["stacked_ffn_checks_per_layer"] = json!(vec![1; 48]);
        (baseline, candidate, qualification)
    }

    #[test]
    fn qualified_layout_change_preserves_work_and_power_gates() {
        let (a, b, q) = fixtures();
        assert!(compare_reports(&a, &b, None).is_err());
        let comparison = compare_reports(&a, &b, Some(&q)).unwrap();
        assert_eq!(
            comparison["weight_comparison"],
            "qualified-stacked-int8-layout"
        );
        assert_eq!(
            comparison["cases"][0]["decode_steps"][0]["baseline_over_candidate_wall"],
            1.25
        );
        assert!(compare_reports(&b, &a, Some(&q)).is_ok());
        let mut bad = b.clone();
        bad["cases"][0]["steps"][0]["measurement"]["sampled_controls_eligible"] = json!(false);
        assert!(compare_reports(&a, &bad, Some(&q))
            .unwrap_err()
            .contains("thermally"));
        bad = b.clone();
        bad["cases"][0]["prefill_measurement"]["comparison_stratum"]["power_source"] =
            json!("battery");
        assert!(compare_reports(&a, &bad, Some(&q))
            .unwrap_err()
            .contains("strata differ"));
        bad = b.clone();
        bad["cases"][0]["generated_tokens"][1] = json!(21);
        assert!(compare_reports(&a, &bad, Some(&q)).is_err());
        bad = b.clone();
        bad["cases"][0]["steps"][0]["input_token"] = json!(11);
        assert!(compare_reports(&a, &bad, Some(&q)).is_err());
    }

    #[test]
    fn incomplete_or_different_qualification_cannot_waive_plan_identity() {
        let (a, b, q) = fixtures();
        for (key, value) in [
            ("model_dir", json!("/other-model")),
            ("qualification_complete", json!(false)),
            ("ane_weight_plan", json!("static-int8-ffn-cached")),
            ("loaded_ane_programs", json!(162)),
            ("references_completed", json!(0)),
            ("ane_compile_budget_used", json!(1)),
            ("stacked_ffn_checks_per_layer", json!(vec![1; 47])),
            ("stacked_ffn_checks_per_layer", json!(vec![0; 48])),
        ] {
            let mut bad = q.clone();
            bad[key] = value;
            assert!(
                compare_reports(&a, &b, Some(&bad)).is_err(),
                "accepted {key}"
            );
        }
        let mut bad = q.clone();
        bad["cases"][0]["matches_reference"] = json!(false);
        assert!(compare_reports(&a, &b, Some(&bad)).is_err());
        let mut bad = b;
        bad["ane_weight_plan"] = json!("static-lut4-ffn-cached");
        assert!(compare_reports(&a, &bad, Some(&q)).is_err());
    }

    #[test]
    fn same_plan_still_requires_identity_and_clean_timing() {
        let (a, _, _) = fixtures();
        assert!(compare_reports(&a, &a, None).is_ok());
        for (key, value) in [
            ("model_dir", Value::Null),
            ("diagnostic_journal_enabled", json!(true)),
            ("layer_state_capture_enabled", json!(true)),
            ("cpu_or_gpu_decode_fallback", json!(true)),
            ("ane_compile_budget_used", json!(1)),
        ] {
            let mut bad = a.clone();
            bad[key] = value;
            assert!(compare_reports(&a, &bad, None).is_err(), "accepted {key}");
        }
    }
}
