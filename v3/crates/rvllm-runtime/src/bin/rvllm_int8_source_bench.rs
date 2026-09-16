//! Host-only source preparation at the actual Gemma 4 12B FFN dimensions.
#![forbid(unsafe_code)]

use half::f16;
use rvllm_apple::ane_int8_ffn_weights::AneInt8FfnWeights;
use rvllm_runtime::apple_measurement::{compare_phase_measurements, PowerMonitor};
use serde_json::{json, Value};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: rvllm_int8_source_bench NEW_OUTPUT_DIRECTORY")?,
    );
    std::fs::create_dir(&path)?;
    let (hidden, intermediate) = (3840_usize, 15360_usize);
    let count = hidden * intermediate;
    let mut random = 7_u64;
    let weights: Vec<f16> = (0..count)
        .map(|_| {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            f16::from_f32(((random >> 32) as i32) as f32 * (0.02 / i32::MAX as f32))
        })
        .collect();
    let monitor = PowerMonitor::start(Some(&path.join("power-observations.jsonl")))?;
    // Keep a reference for exact output checks outside every timed interval.
    let reference = AneInt8FfnWeights::quantize_scalar_reference(
        &weights,
        &weights,
        &weights,
        hidden,
        intermediate,
    )?;
    let warmup = AneInt8FfnWeights::quantize(&weights, &weights, &weights, hidden, intermediate)?;
    if reference != warmup {
        return Err("candidate source differs from scalar reference".into());
    }
    drop(warmup);
    let mut trials = Vec::new();
    let mut pairs = Vec::new();
    // Three ABBA blocks provide six pairs. Raw observations survive even when
    // thermal state or source changes make every pair ineligible.
    for block in 0..3 {
        let order = if block % 2 == 0 {
            [false, true, true, false]
        } else {
            [true, false, false, true]
        };
        let start = trials.len();
        for candidate in order {
            let measured = monitor.begin();
            let output = if candidate {
                AneInt8FfnWeights::quantize(&weights, &weights, &weights, hidden, intermediate)?
            } else {
                AneInt8FfnWeights::quantize_scalar_reference(
                    &weights,
                    &weights,
                    &weights,
                    hidden,
                    intermediate,
                )?
            };
            let measurement = measured.finish(3 * count);
            if output != reference {
                return Err("trial source differs from scalar reference".into());
            }
            drop(output);
            trials.push(
                json!({"candidate":candidate,"source_matches":true,"measurement":measurement}),
            );
            std::fs::write(
                path.join("trials.json"),
                serde_json::to_vec_pretty(&trials)?,
            )?;
        }
        for (left, right) in [(start, start + 1), (start + 3, start + 2)] {
            let (baseline, candidate) = if trials[left]["candidate"] == false {
                (left, right)
            } else {
                (right, left)
            };
            let comparison = compare_phase_measurements(
                &trials[baseline]["measurement"],
                &trials[candidate]["measurement"],
            );
            pairs.push(match comparison {
                Ok(value) => json!({"baseline_trial":baseline,"candidate_trial":candidate,"eligible":true,"comparison":value}),
                Err(reason) => json!({"baseline_trial":baseline,"candidate_trial":candidate,"eligible":false,"reason":reason}),
            });
        }
    }
    let mut valid_ratios: Vec<f64> = pairs
        .iter()
        .filter_map(|pair| pair["comparison"]["baseline_over_candidate_wall"].as_f64())
        .collect();
    valid_ratios.sort_by(f64::total_cmp);
    let median: Option<f64> = if valid_ratios.len() >= 5 {
        let middle = valid_ratios.len() / 2;
        Some(if valid_ratios.len() % 2 == 0 {
            (valid_ratios[middle - 1] + valid_ratios[middle]) / 2.0
        } else {
            valid_ratios[middle]
        })
    } else {
        None
    };
    let report: Value = json!({"schema":"rvllm.gemma12b_int8_host_quantizer_comparison.v1",
        "hidden":hidden,"intermediate":intermediate,"coefficients_per_trial":3*count,
        "workload":"Synthetic deterministic FP16 weights; one input allocation reused as gate/up/down; real Gemma 4 12B matrix dimensions; conversion/packing/ANE excluded",
        "ane_calls":0,"all_source_values_match":true,"trials":trials,"pairs":pairs,
        "eligible_pairs":valid_ratios.len(),"median_paired_wall_ratio_if_at_least_five_valid_pairs":median,
        "claim":"Host quantization only. Same-power paired observations do not establish full-model startup or decode speedup."});
    std::fs::write(
        path.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", path.join("report.json").display());
    Ok(())
}
