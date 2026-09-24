//! Compare identical work under matching sampled controls.
#![forbid(unsafe_code)]

#[cfg(target_os = "macos")]
use rvllm_runtime::apple_measurement::compare_phase_measurements;
#[cfg(target_os = "macos")]
use serde_json::{json, Value};

#[cfg(target_os = "macos")]
#[path = "rvllm_compare_disaggregated/fair.rs"]
mod fair;

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
fn validate_interleaved_qualification(
    baseline: &Value,
    candidate: &Value,
    route: &Value,
    component_events: &[Value],
) -> Result<(), String> {
    let plans = (
        baseline["ane_weight_plan"].as_str(),
        candidate["ane_weight_plan"].as_str(),
    );
    if !matches!(
        plans,
        (
            Some("static-int8-ffn-cached"),
            Some("static-int8-interleaved-ffn-cached")
        ) | (
            Some("static-int8-interleaved-ffn-cached"),
            Some("static-int8-ffn-cached")
        )
    ) {
        return Err(
            "qualification permits only original versus interleaved INT8 FFN layouts".into(),
        );
    }
    same_fields(
        candidate,
        route,
        &["model_dir", "config_sha256", "global_context_capacity"],
    )?;
    if route["schema"] != "rvllm.metal_prefill_ane_decode.v1"
        || route["ane_weight_plan"] != "static-int8-interleaved-ffn-cached"
        || route["qualification_complete"] != true
        || route["inference_complete"] != true
        || route["all_references_match"] != true
        || route["ane_execution_verified"] != true
        || route["cpu_or_gpu_decode_fallback"] != false
        || route["diagnostic_journal_enabled"] != true
        || route["loaded_ane_programs"] != 162
        || route["ane_compile_budget_used"] != 0
    {
        return Err("incomplete or unexpected interleaved full-route qualification".into());
    }
    let qualified = cases(route)?;
    if qualified.len() != 1
        || route["references_requested"] != 1
        || route["references_completed"] != 1
        || qualified[0]["matches_reference"] != true
        || qualified[0]["ane_decode_steps"].as_u64() == Some(0)
    {
        return Err("interleaved qualification lacks one complete reference route".into());
    }
    for report in [baseline, candidate] {
        for timed in cases(report)? {
            same_fields(
                timed,
                &qualified[0],
                &["prompt_token_ids", "generated_tokens", "ane_decode_steps"],
            )?;
        }
    }
    let begin = component_events
        .first()
        .ok_or("missing interleaved component begin event")?;
    let end = component_events
        .last()
        .ok_or("missing interleaved component end event")?;
    let comparisons: Vec<_> = component_events
        .iter()
        .filter(|event| event["event"] == "comparison")
        .collect();
    if begin["event"] != "begin"
        || begin["schema"] != "rvllm.ane.ffn-component-input.v1"
        || begin["candidate"] != "ane-int8-ffn-interleaved"
        || begin["control"] != "stacked-int8"
        || begin["layer"] != 0
        || begin["samples"] != 3
        || comparisons.len() != 3
        || comparisons
            .iter()
            .any(|event| event["bit_exact_finite"] != true)
        || end["event"] != "end"
        || end["matched_all_inputs"] != true
        || end["error"] != Value::Null
        || end["compiler_calls"]
            .as_u64()
            .map_or(true, |calls| calls > 2)
        || end["promotion"] != false
        || end["performance_qualified"] != false
    {
        return Err("interleaved component qualification is incomplete or inexact".into());
    }
    Ok(())
}

#[cfg(all(target_os = "macos", test))]
fn compare_reports(
    baseline: &Value,
    candidate: &Value,
    qualification: Option<&Value>,
) -> Result<Value, String> {
    compare_reports_with_policy(baseline, candidate, qualification, None, false)
}

#[cfg(target_os = "macos")]
fn compare_reports_with_policy(
    baseline: &Value,
    candidate: &Value,
    qualification: Option<&Value>,
    interleaved_qualification: Option<(&Value, &[Value])>,
    explore_fair: bool,
) -> Result<Value, String> {
    let compare_phase: fn(&Value, &Value) -> Result<Value, String> = if explore_fair {
        fair::compare
    } else {
        compare_phase_measurements
    };
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
    let weight_comparison = if qualification.is_some() && interleaved_qualification.is_some() {
        return Err("multiple layout qualifications are not allowed".into());
    } else if let Some(qualification) = qualification {
        validate_stacked_qualification(baseline, candidate, qualification)?;
        "qualified-stacked-int8-layout"
    } else if let Some((route, component_events)) = interleaved_qualification {
        validate_interleaved_qualification(baseline, candidate, route, component_events)?;
        "qualified-interleaved-int8-layout"
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
        let prefill = compare_phase(
            &before["prefill_measurement"],
            &after["prefill_measurement"],
        )?;
        let import = compare_phase(
            &before["ane_import_measurement"],
            &after["ane_import_measurement"],
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
                compare_phase(&before["measurement"], &after["measurement"])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let gpu_ratio = before["metal_gpu_execution_ms"]
            .as_f64()
            .zip(after["metal_gpu_execution_ms"].as_f64())
            .filter(|(before, after)| {
                before.is_finite() && after.is_finite() && *before > 0.0 && *after > 0.0
            })
            .map(|(before, after)| before / after);
        comparisons.push(
            json!({"prefill":prefill,"ane_import":import,"decode_steps":decode,
            "baseline_over_candidate_gpu_interval":gpu_ratio}),
        );
    }
    Ok(json!({"schema":"rvllm.apple_paired_comparison.v1",
        "thermal_comparison":if explore_fair {"exploratory-stable-fair"}else{"nominal-only"},
        "weight_comparison":weight_comparison,
        "baseline_weight_plan":baseline["ane_weight_plan"],
        "candidate_weight_plan":candidate["ane_weight_plan"],
        "cases":comparisons,
        "claim":"One paired observation, not a speedup conclusion. Repeat in alternating order, preserve power strata and check baseline drift. CPU cycles do not normalize device time."}))
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use sha2::{Digest, Sha256};
    let mut paths: Vec<_> = std::env::args_os().skip(1).collect();
    let explore_fair = paths
        .last()
        .is_some_and(|flag| flag == "--stable-fair-exploratory");
    if explore_fair {
        paths.pop();
    }
    let stacked = paths.len() == 4 && paths[2] == "--stacked-qualification";
    let interleaved = paths.len() == 5 && paths[2] == "--interleaved-qualification";
    if paths.len() != 2 && !stacked && !interleaved {
        return Err("usage: rvllm_compare_disaggregated BASELINE_REPORT CANDIDATE_REPORT [--stacked-qualification CHECKED_REPORT | --interleaved-qualification ROUTE_REPORT COMPONENT_EVENTS_JSONL] [--stable-fair-exploratory]".into());
    }
    let baseline: Value = serde_json::from_slice(&std::fs::read(&paths[0])?)?;
    let candidate: Value = serde_json::from_slice(&std::fs::read(&paths[1])?)?;
    let qualification = if stacked {
        let bytes = std::fs::read(&paths[3])?;
        Some((
            serde_json::from_slice::<Value>(&bytes)?,
            format!("{:x}", Sha256::digest(bytes)),
        ))
    } else {
        None
    };
    let interleaved_qualification = if interleaved {
        let route_bytes = std::fs::read(&paths[3])?;
        let events_bytes = std::fs::read(&paths[4])?;
        let route = serde_json::from_slice::<Value>(&route_bytes)?;
        let events = events_bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(serde_json::from_slice::<Value>)
            .collect::<Result<Vec<_>, _>>()?;
        Some((
            route,
            events,
            format!("{:x}", Sha256::digest(route_bytes)),
            format!("{:x}", Sha256::digest(events_bytes)),
        ))
    } else {
        None
    };
    let qualified = qualification.as_ref().map(|(report, _)| report);
    let interleaved_qualified = interleaved_qualification
        .as_ref()
        .map(|(route, events, _, _)| (route, events.as_slice()));
    let mut comparison = if explore_fair {
        compare_reports_with_policy(
            &baseline,
            &candidate,
            qualified,
            interleaved_qualified,
            true,
        )?
    } else {
        compare_reports_with_policy(
            &baseline,
            &candidate,
            qualified,
            interleaved_qualified,
            false,
        )?
    };
    comparison["baseline"] = json!(paths[0]);
    comparison["candidate"] = json!(paths[1]);
    if let Some((_, digest)) = qualification {
        comparison["stacked_qualification"] = json!({"path":paths[3],"sha256":digest});
    }
    if let Some((_, _, route_digest, events_digest)) = interleaved_qualification {
        comparison["interleaved_qualification"] = json!({
            "route":{"path":paths[3],"sha256":route_digest},
            "component_events":{"path":paths[4],"sha256":events_digest}
        });
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
                "ane_import_measurement":phase(1,5),
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
    fn interleaved_exception_requires_matching_route_and_bit_exact_component_events() {
        let (a, mut b, _) = fixtures();
        b["ane_weight_plan"] = json!("static-int8-interleaved-ffn-cached");
        let mut route = b.clone();
        route["qualification_complete"] = json!(true);
        route["all_references_match"] = json!(true);
        route["ane_execution_verified"] = json!(true);
        route["diagnostic_journal_enabled"] = json!(true);
        route["loaded_ane_programs"] = json!(162);
        route["references_requested"] = json!(1);
        route["references_completed"] = json!(1);
        let begin = json!({"event":"begin","schema":"rvllm.ane.ffn-component-input.v1",
            "candidate":"ane-int8-ffn-interleaved","control":"stacked-int8","layer":0,
            "samples":3});
        let comparison = |sample| {
            json!({"event":"comparison","sample":sample,
            "bit_exact_finite":true})
        };
        let end = json!({"event":"end","matched_all_inputs":true,"error":null,
            "compiler_calls":2,"promotion":false,"performance_qualified":false});
        let events = vec![begin, comparison(0), comparison(1), comparison(2), end];
        let result =
            compare_reports_with_policy(&a, &b, None, Some((&route, &events)), false).unwrap();
        assert_eq!(
            result["weight_comparison"],
            "qualified-interleaved-int8-layout"
        );

        let mut bad_route = route.clone();
        bad_route["all_references_match"] = json!(false);
        assert!(
            compare_reports_with_policy(&a, &b, None, Some((&bad_route, &events)), false).is_err()
        );
        let mut bad_events = events.clone();
        bad_events[2]["bit_exact_finite"] = json!(false);
        assert!(
            compare_reports_with_policy(&a, &b, None, Some((&route, &bad_events)), false).is_err()
        );
        let mut bad_workload = b.clone();
        bad_workload["cases"][0]["generated_tokens"] = json!([10, 21]);
        assert!(compare_reports_with_policy(
            &a,
            &bad_workload,
            None,
            Some((&route, &events)),
            false
        )
        .is_err());
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
