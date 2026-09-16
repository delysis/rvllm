//! Offline exploratory comparison within an observed stable Fair stratum.
//! Does not alter raw reports or nominal-only measurement eligibility.
#![forbid(unsafe_code)]

use serde_json::{json, Value};

const MAX_AGE_MS: f64 = 2500.0;

fn number(value: &Value, key: &str) -> Result<f64, String> {
    value[key]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or_else(|| format!("missing or invalid {key}"))
}

pub(super) fn validate(measurement: &Value) -> Result<&Value, String> {
    if measurement["schema"] != "rvllm.apple_phase_measurement.v1"
        || measurement["sampled_controls_eligible"] != false
        || measurement.get("observer_journal_error") != Some(&Value::Null)
    {
        return Err(
            "Fair exploration requires intact raw observations, separate from nominal eligibility"
                .into(),
        );
    }
    let start = number(measurement, "start_ms")?;
    let end = number(measurement, "end_ms")?;
    let wall = number(measurement, "wall_ms")?;
    if end <= start || wall <= 0.0 || ((end - start) - wall).abs() > 0.1 {
        return Err("invalid phase time interval".into());
    }
    let samples = measurement["power_samples"]
        .as_array()
        .filter(|samples| !samples.is_empty())
        .ok_or("missing power samples")?;
    let first = &samples[0];
    let controls = &first["controls"];
    if !matches!(controls["power_source"].as_str(), Some("ac" | "battery"))
        || controls["low_power_mode"].as_bool().is_none()
        || !matches!(controls["pmset_power_mode"].as_u64(), Some(0..=2))
        || controls["thermal_state"] != 1
    {
        return Err("Fair exploration requires known power controls and thermal state 1".into());
    }
    for key in ["cpu_speed_limit_percent", "scheduler_limit_percent"] {
        if !matches!(controls.get(key), Some(Value::Null)) && controls[key].as_u64() != Some(100) {
            return Err(format!("restricted or missing {key}"));
        }
    }
    match controls.get("available_cpus") {
        Some(Value::Null) => {}
        Some(value) if value.as_u64().is_some_and(|n| n > 0) => {}
        _ => return Err("missing or invalid available CPU count".into()),
    }
    for endpoint in ["start_host", "end_host"] {
        if measurement[endpoint]["thermal_state"] != 1
            || measurement[endpoint]["low_power_mode"] != controls["low_power_mode"]
        {
            return Err("thermal or low-power transition at phase endpoint".into());
        }
    }
    let first_end = number(first, "end_ms")?;
    if first_end > start || start - first_end > MAX_AGE_MS {
        return Err("stale or missing starting power observation".into());
    }
    let mut previous_end = None;
    for sample in samples {
        let sample_start = number(sample, "start_ms")?;
        let sample_end = number(sample, "end_ms")?;
        if sample_start > sample_end
            || sample_start > end
            || sample_end > end + MAX_AGE_MS
            || previous_end.is_some_and(|previous| {
                sample_start < previous || sample_end - previous > MAX_AGE_MS
            })
            || sample["controls"] != *controls
        {
            return Err("changing, unordered or poorly covered power observations".into());
        }
        previous_end = Some(sample_end);
    }
    if end - previous_end.ok_or("missing ending power observation")? > MAX_AGE_MS {
        return Err("stale ending power observation".into());
    }
    Ok(controls)
}

pub(super) fn compare(baseline: &Value, candidate: &Value) -> Result<Value, String> {
    let a = validate(baseline)?;
    let b = validate(candidate)?;
    if a != b {
        return Err("Fair power/processor strata differ".into());
    }
    if baseline["units"].as_u64().unwrap_or(0) == 0 || baseline["units"] != candidate["units"] {
        return Err("work counts differ or are zero".into());
    }
    let ratio = |field: &str| -> Option<f64> {
        let a = baseline[field].as_f64()?;
        let b = candidate[field].as_f64()?;
        let ratio = a / b;
        (a > 0.0 && b > 0.0 && ratio.is_finite()).then_some(ratio)
    };
    Ok(json!({
        "comparison_kind":"exploratory-stable-fair",
        "baseline_over_candidate_wall":ratio("wall_ms").ok_or("invalid wall times")?,
        "baseline_over_candidate_cpu_cycles":ratio("process_cpu_cycles"),
        "baseline_over_candidate_cpu_instructions":ratio("process_cpu_instructions"),
        "comparison_stratum":a,
        "nominal_state_eligible":false,
        "claim":"Conditional observation within sampled stable Fair controls. Requires repeated bracketing baselines and drift analysis. Does not identify or normalize device clocks, and must not be pooled with nominal-state results."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn phase() -> Value {
        let controls = json!({"power_source":"battery","low_power_mode":false,"pmset_power_mode":0,
            "thermal_state":1,"cpu_speed_limit_percent":null,"scheduler_limit_percent":null,"available_cpus":null});
        json!({"schema":"rvllm.apple_phase_measurement.v1","start_ms":20.0,"end_ms":1220.0,
            "wall_ms":1200.0,"units":1,"sampled_controls_eligible":false,"observer_journal_error":null,
            "start_host":{"low_power_mode":false,"thermal_state":1},
            "end_host":{"low_power_mode":false,"thermal_state":1},
            "power_samples":[{"start_ms":0.0,"end_ms":10.0,"controls":controls},
                {"start_ms":1000.0,"end_ms":1010.0,"controls":controls}]})
    }

    #[test]
    fn fair_is_separate_from_nominal_and_never_clock_normalized() {
        let a = phase();
        let mut b = a.clone();
        b["end_ms"] = json!(980.0);
        b["wall_ms"] = json!(960.0);
        b["power_samples"].as_array_mut().unwrap().pop();
        let result = compare(&a, &b).unwrap();
        assert_eq!(result["baseline_over_candidate_wall"], 1.25);
        assert_eq!(result["nominal_state_eligible"], false);
        assert!(result["baseline_over_candidate_cpu_cycles"].is_null());
        assert!(rvllm_runtime::apple_measurement::compare_phase_measurements(&a, &b).is_err());
    }

    #[test]
    fn fair_does_not_waive_transitions_gaps_or_restrictions() {
        let a = phase();
        let paths = [
            ("/power_samples/1/controls/power_source", json!("ac")),
            ("/power_samples/1/controls/thermal_state", json!(0)),
            ("/start_host/thermal_state", json!(2)),
            ("/end_host/low_power_mode", json!(true)),
            (
                "/power_samples/0/controls/cpu_speed_limit_percent",
                json!(80),
            ),
            (
                "/power_samples/0/controls/scheduler_limit_percent",
                json!(90),
            ),
            ("/power_samples/0/controls/available_cpus", json!(0)),
            ("/power_samples/0/controls/pmset_power_mode", Value::Null),
            ("/power_samples/0/end_ms", json!(21.0)),
            ("/power_samples/1/start_ms", json!(5.0)),
            ("/observer_journal_error", json!("disk error")),
            ("/sampled_controls_eligible", json!(true)),
        ];
        for (path, value) in paths {
            let mut bad = a.clone();
            *bad.pointer_mut(path).unwrap() = value;
            assert!(validate(&bad).is_err(), "accepted {path}");
        }
        let mut bad = a.clone();
        bad["end_ms"] = json!(4010.0);
        bad["wall_ms"] = json!(3990.0);
        assert!(validate(&bad).is_err());
        let mut bad = a.clone();
        for sample in bad["power_samples"].as_array_mut().unwrap() {
            sample["controls"]["power_source"] = json!("ac");
        }
        assert!(compare(&a, &bad).unwrap_err().contains("strata differ"));
        for sample in bad["power_samples"].as_array_mut().unwrap() {
            sample["controls"]["thermal_state"] = json!(2);
        }
        bad["start_host"]["thermal_state"] = json!(2);
        bad["end_host"]["thermal_state"] = json!(2);
        assert!(validate(&bad).is_err());
    }
}
