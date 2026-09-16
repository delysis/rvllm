//! Explicit Fair-state exploration. The nominal measurement contract is unchanged.
#![forbid(unsafe_code)]

use rvllm_runtime::apple_measurement::compare_phase_measurements;
use serde_json::{json, Value};

const MAX_AGE_MS: f64 = 2500.0;

fn require(ok: bool, reason: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(reason.into())
    }
}

fn number(value: &Value, key: &str) -> Result<f64, String> {
    value[key]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or_else(|| format!("missing or invalid {key}"))
}

pub(super) fn stratum(phase: &Value, exploratory_fair: bool) -> Result<Value, String> {
    if !exploratory_fair {
        compare_phase_measurements(phase, phase)?;
        return Ok(phase["comparison_stratum"].clone());
    }
    require(
        phase["schema"] == "rvllm.apple_phase_measurement.v1"
            && phase.get("observer_journal_error") == Some(&Value::Null),
        "missing phase identity or failed observer",
    )?;
    let start = number(phase, "start_ms")?;
    let end = number(phase, "end_ms")?;
    let wall = number(phase, "wall_ms")?;
    require(
        end > start && wall > 0.0 && ((end - start) - wall).abs() <= 0.1,
        "invalid phase interval",
    )?;
    let samples = phase["power_samples"]
        .as_array()
        .ok_or("missing power samples")?;
    let first = samples.first().ok_or("empty power samples")?;
    let controls = &first["controls"];
    require(
        matches!(controls["power_source"].as_str(), Some("ac" | "battery"))
            && controls["low_power_mode"].as_bool().is_some()
            && matches!(controls["pmset_power_mode"].as_u64(), Some(0..=2))
            && controls["thermal_state"] == 1,
        "exploration requires known controls and exactly Fair thermal state",
    )?;
    for key in ["cpu_speed_limit_percent", "scheduler_limit_percent"] {
        require(
            controls.get(key) == Some(&Value::Null) || controls[key].as_u64() == Some(100),
            "missing or restricted processor limit",
        )?;
    }
    require(
        controls.get("available_cpus") == Some(&Value::Null)
            || controls["available_cpus"].as_u64().is_some_and(|n| n > 0),
        "missing or invalid processor count",
    )?;
    require(
        phase["sampled_controls_eligible"] == false
            && phase.get("comparison_stratum") == Some(&Value::Null),
        "Fair exploration must not claim nominal eligibility",
    )?;
    for endpoint in ["start_host", "end_host"] {
        require(
            phase[endpoint]["thermal_state"] == controls["thermal_state"]
                && phase[endpoint]["low_power_mode"] == controls["low_power_mode"],
            "endpoint controls changed",
        )?;
    }
    let first_end = number(first, "end_ms")?;
    require(
        first_end <= start && start - first_end <= MAX_AGE_MS,
        "missing or stale starting observation",
    )?;
    let mut previous = None;
    for sample in samples {
        let s = number(sample, "start_ms")?;
        let e = number(sample, "end_ms")?;
        require(
            s <= e
                && s <= end
                && e <= end + MAX_AGE_MS
                && previous.map_or(true, |p| s >= p && e - p <= MAX_AGE_MS)
                && sample["controls"] == *controls,
            "changed, unordered or gapped observations",
        )?;
        previous = Some(e);
    }
    require(
        end - previous.ok_or("missing ending observation")? <= MAX_AGE_MS,
        "stale ending observation",
    )?;
    Ok(controls.clone())
}

pub(super) fn compare(
    baseline: &Value,
    candidate: &Value,
    exploratory_fair: bool,
) -> Result<Value, String> {
    if !exploratory_fair {
        return compare_phase_measurements(baseline, candidate);
    }
    let controls = stratum(baseline, true)?;
    require(
        controls == stratum(candidate, true)?,
        "power/processor strata differ",
    )?;
    require(
        baseline["units"].as_u64().is_some_and(|n| n > 0)
            && baseline["units"] == candidate["units"],
        "work counts differ or are zero",
    )?;
    Ok(
        json!({"baseline_over_candidate_wall":number(baseline,"wall_ms")?/number(candidate,"wall_ms")?,
        "comparison_stratum":controls,"thermal_class":"exploratory_stable_fair",
        "claim":"Exploratory paired wall time within stable Fair controls. Not nominal eligibility or clock normalization; baseline drift and repeated blocks must be checked."}),
    )
}

/// Three complete ABBA/BAAB/ABBA blocks with a predeclared 5% repeat drift
/// ceiling for both variants. This does not estimate device clocks.
pub(super) fn summarize(trials: &[Value], exploratory_fair: bool) -> Result<Value, String> {
    require(trials.len() == 12, "three complete timing blocks required")?;
    let common = stratum(&trials[0]["measurement"], exploratory_fair)?;
    let mut blocks = Vec::new();
    let mut ratios = Vec::new();
    let mut drift_passed = true;
    for (block, trials) in trials.chunks_exact(4).enumerate() {
        let order = if block == 1 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        let mut serial = Vec::new();
        let mut batch = Vec::new();
        for (trial, expected_batch) in trials.iter().zip(order) {
            require(
                trial["batch"] == expected_batch
                    && trial["repetitions"] == 128
                    && trial["useful_outputs_per_repetition"] == 2
                    && trial["measurement"]["units"] == 256
                    && trial["timing_controls_eligible"] == true
                    && trial["output_bit_identical_to_serial"] == true,
                "incorrect work, order or output check",
            )?;
            require(
                stratum(&trial["measurement"], exploratory_fair)? == common,
                "controls changed between timing blocks",
            )?;
            let elapsed = number(&trial["measurement"], "wall_ms")?;
            if expected_batch {
                batch.push(elapsed);
            } else {
                serial.push(elapsed);
            }
        }
        let drift = |times: &[f64]| times[0].max(times[1]) / times[0].min(times[1]) - 1.0;
        let serial_drift = drift(&serial);
        let batch_drift = drift(&batch);
        let block_passed = serial_drift <= 0.05 && batch_drift <= 0.05;
        drift_passed &= block_passed;
        let serial_mean = (serial[0] + serial[1]) / 2.0;
        let batch_mean = (batch[0] + batch[1]) / 2.0;
        let ratio = serial_mean / batch_mean;
        ratios.push(ratio);
        blocks.push(json!({"block":block,"baseline_over_candidate_wall":ratio,
            "serial_ms_per_useful_output":serial_mean/256.0,
            "batch_ms_per_useful_output":batch_mean/256.0,
            "serial_repeat_drift_fraction":serial_drift,"batch_repeat_drift_fraction":batch_drift,
            "repeat_drift_passed":block_passed}));
    }
    ratios.sort_by(f64::total_cmp);
    Ok(
        json!({"status":if drift_passed {"complete_component_comparison"} else {"inconclusive_repeat_drift"},
        "repeat_drift_limit_fraction":0.05,"repeat_drift_passed":drift_passed,
        "comparison_stratum":common,
        "thermal_class":if exploratory_fair {"exploratory_stable_fair"} else {"nominal"},
        "blocks":blocks,"median_baseline_over_candidate_wall":ratios[1],
        "all_blocks_favor_batch":ratios[0]>1.0,
        "claim":"Component timing of two useful FFN outputs. No accepted speculative tokens, full-model speedup, or DVFS normalization."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fair() -> Value {
        let controls = json!({"power_source":"ac","low_power_mode":true,"pmset_power_mode":1,
            "thermal_state":1,"cpu_speed_limit_percent":null,"scheduler_limit_percent":null,"available_cpus":null});
        json!({"schema":"rvllm.apple_phase_measurement.v1","observer_journal_error":null,
            "start_ms":20.0,"end_ms":1200.0,"wall_ms":1180.0,"units":256,
            "start_host":{"low_power_mode":true,"thermal_state":1},
            "end_host":{"low_power_mode":true,"thermal_state":1},
            "sampled_controls_eligible":false,"comparison_stratum":null,
            "power_samples":[{"start_ms":0.0,"end_ms":10.0,"controls":controls},
                {"start_ms":1000.0,"end_ms":1010.0,"controls":controls}]})
    }

    #[test]
    fn fair_is_explicit_and_never_nominal() {
        let phase = fair();
        assert!(stratum(&phase, false).is_err());
        assert_eq!(stratum(&phase, true).unwrap()["thermal_state"], 1);
        let mut faster = phase.clone();
        faster["start_ms"] = json!(256.0);
        faster["wall_ms"] = json!(944.0);
        assert_eq!(
            compare(&phase, &faster, true).unwrap()["baseline_over_candidate_wall"],
            1.25
        );
        for thermal in [-1, 0, 2, 3] {
            let mut invalid = phase.clone();
            for sample in invalid["power_samples"].as_array_mut().unwrap() {
                sample["controls"]["thermal_state"] = json!(thermal);
            }
            assert!(stratum(&invalid, true).is_err());
        }
    }

    #[test]
    fn missing_transient_or_restricted_controls_reject() {
        let phase = fair();
        for (pointer, replacement) in [
            ("/observer_journal_error", json!("write failed")),
            ("/power_samples", json!([])),
            ("/sampled_controls_eligible", json!(true)),
            ("/end_host/low_power_mode", json!(false)),
            ("/end_host/thermal_state", json!(0)),
            ("/power_samples/1/controls/power_source", json!("battery")),
            ("/power_samples/1/controls/pmset_power_mode", json!(0)),
            (
                "/power_samples/1/controls/cpu_speed_limit_percent",
                json!(80),
            ),
            ("/power_samples/1/start_ms", json!(5.0)),
            ("/power_samples/1/end_ms", json!(999.0)),
            ("/power_samples/0/end_ms", json!(21.0)),
            ("/wall_ms", json!(0.0)),
        ] {
            let mut invalid = phase.clone();
            *invalid.pointer_mut(pointer).unwrap() = replacement;
            assert!(stratum(&invalid, true).is_err(), "{pointer}");
        }
        for key in [
            "power_source",
            "pmset_power_mode",
            "low_power_mode",
            "available_cpus",
            "cpu_speed_limit_percent",
            "scheduler_limit_percent",
        ] {
            let mut invalid = phase.clone();
            invalid["power_samples"][0]["controls"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert!(stratum(&invalid, true).is_err(), "{key}");
        }
    }

    #[test]
    fn gaps_and_different_work_or_stable_strata_reject() {
        let phase = fair();
        let mut invalid = phase.clone();
        invalid["end_ms"] = json!(4000.0);
        invalid["wall_ms"] = json!(3980.0);
        assert!(stratum(&invalid, true).is_err());
        invalid["power_samples"][1]["start_ms"] = json!(3000.0);
        invalid["power_samples"][1]["end_ms"] = json!(3010.0);
        assert!(stratum(&invalid, true).is_err());
        for units in [0, 255] {
            invalid = phase.clone();
            invalid["units"] = json!(units);
            assert!(compare(&phase, &invalid, true).is_err());
        }
        invalid = phase.clone();
        for sample in invalid["power_samples"].as_array_mut().unwrap() {
            sample["controls"]["power_source"] = json!("battery");
        }
        assert!(stratum(&invalid, true).is_ok());
        assert!(compare(&phase, &invalid, true).is_err());
    }

    fn trials() -> Vec<Value> {
        (0..3)
            .flat_map(|block| {
                let order = if block == 1 {
                    [true, false, false, true]
                } else {
                    [false, true, true, false]
                };
                order.into_iter().map(|batch| {
                    let mut measurement = fair();
                    if batch {
                        measurement["start_ms"] = json!(256.0);
                        measurement["wall_ms"] = json!(944.0);
                    }
                    json!({"batch":batch,"repetitions":128,"useful_outputs_per_repetition":2,
                    "measurement":measurement,"timing_controls_eligible":true,
                    "output_bit_identical_to_serial":true})
                })
            })
            .collect()
    }

    #[test]
    fn incomplete_changed_work_or_cross_block_power_does_not_compare() {
        let trials = trials();
        assert_eq!(
            summarize(&trials, true).unwrap()["median_baseline_over_candidate_wall"],
            1.25
        );
        assert!(summarize(&trials[..11], true).is_err());
        for (pointer, value) in [
            ("/batch", json!(true)),
            ("/repetitions", json!(127)),
            ("/output_bit_identical_to_serial", json!(false)),
        ] {
            let mut bad = trials.clone();
            *bad[0].pointer_mut(pointer).unwrap() = value;
            assert!(summarize(&bad, true).is_err());
        }
        let mut bad = trials.clone();
        for sample in bad[4]["measurement"]["power_samples"]
            .as_array_mut()
            .unwrap()
        {
            sample["controls"]["power_source"] = json!("battery");
        }
        assert!(summarize(&bad, true).is_err());
    }

    #[test]
    fn repeated_baseline_or_candidate_drift_is_inconclusive() {
        for index in [2, 3] {
            let mut trials = trials();
            trials[index]["measurement"]["end_ms"] = json!(1400.0);
            trials[index]["measurement"]["wall_ms"] =
                json!(1400.0 - trials[index]["measurement"]["start_ms"].as_f64().unwrap());
            let result = summarize(&trials, true).unwrap();
            assert_eq!(result["status"], "inconclusive_repeat_drift");
            assert_eq!(result["repeat_drift_passed"], false);
        }
    }
}
