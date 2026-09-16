//! Two useful token columns through one ordinary INT8 FFN graph.
#![forbid(unsafe_code)]

use super::*;
use rvllm_apple::ane_linear::compile_budget_used;
use rvllm_runtime::apple_measurement::PowerMonitor;
use serde_json::Value;

fn bit_equal(left: &[f16], right: &[f16]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn check_output(actual: &[f16], reference: &[f32]) -> Result<Value, String> {
    if actual.len() != reference.len() || actual.is_empty() {
        return Err("batch probe output/reference shape mismatch".into());
    }
    let mut error = ErrorStats::default();
    for (index, (actual, expected)) in actual.iter().zip(reference).enumerate() {
        error.observe(actual.to_f32(), *expected)?;
        if (actual.to_f32() - expected).abs() > 0.01 + 0.02 * expected.abs() {
            return Err(format!(
                "batch FFN output {index}: {actual} versus CPU {expected}"
            ));
        }
    }
    Ok(error.report())
}

fn pair_input(inputs: &[Vec<f16>], lanes: [usize; 2]) -> Vec<f16> {
    inputs[lanes[0]]
        .iter()
        .chain(&inputs[lanes[1]])
        .copied()
        .collect()
}

pub(super) fn run(
    weights: &AneInt8FfnWeights,
    dense: &[Vec<f16>; 3],
    inputs: &[Vec<f16>],
    policy: AneProgramCachePolicy,
    layer: usize,
    reconstructed_hash: &str,
    input_sources: &[Value],
    exploratory_fair: bool,
    interleaved: bool,
) -> Result<Value, String> {
    if inputs.len() < 2
        || inputs.len() != input_sources.len()
        || inputs.iter().any(|input| input.len() != HIDDEN)
    {
        return Err("batch probe requires at least two identified H3840 inputs".into());
    }
    let mut all_inputs = inputs.to_vec();
    let zero = all_inputs.len();
    all_inputs.push(vec![f16::ZERO; HIDDEN]);
    // The independent oracle runs before any accelerator program is loaded.
    let references: Vec<_> = all_inputs
        .iter()
        .map(|input| cpu_ffn(dense, input).output)
        .collect();
    // A missing baseline must stop before a candidate compilation is attempted.
    let mut serial = AneGatedFfn::compile_int8_with_cache_policy(
        weights,
        AneProgramCachePolicy::RequireExisting,
    )?;
    let mut batch = AneGatedFfn::compile_int8_batch_with_cache_policy(weights, 2, policy)?;
    let mut serial_outputs = vec![vec![f16::ZERO; HIDDEN]; all_inputs.len()];
    let mut serial_errors = Vec::new();
    for ((input, reference), output) in all_inputs.iter().zip(&references).zip(&mut serial_outputs)
    {
        serial.project(input, output)?;
        serial_errors.push(check_output(output, reference)?);
    }
    let mut cases: Vec<([usize; 2], Option<usize>)> = Vec::new();
    // Each source occupies both lanes. Adjacent reversed cases check token isolation.
    for i in 0..inputs.len() {
        let previous = cases.len();
        cases.push(([i, (i + 1) % inputs.len()], None));
        cases.push(([(i + 1) % inputs.len(), i], Some(previous)));
    }
    cases.push(([inputs.len() - 1, zero], None));
    cases.push(([zero, inputs.len() - 1], Some(cases.len() - 1)));
    // Return to the first input after zero and high-amplitude lanes have been used.
    cases.push((cases[0].0, None));
    let mut outputs = Vec::<Vec<f16>>::new();
    let mut case_reports = Vec::new();
    let mut all_match_serial = true;
    for (case_index, (lanes, swapped_from)) in cases.iter().enumerate() {
        let input = pair_input(&all_inputs, *lanes);
        let mut output = vec![f16::ZERO; 2 * HIDDEN];
        batch.project_batch(&input, &mut output)?;
        let mut errors = Vec::new();
        let mut serial_differences = Vec::new();
        for (lane, source) in lanes.iter().enumerate() {
            let actual = &output[lane * HIDDEN..(lane + 1) * HIDDEN];
            if *source == zero && actual.iter().any(|value| value.to_f32() != 0.0) {
                return Err(format!(
                    "nonzero output in zero-isolation case {case_index} lane {lane}"
                ));
            }
            errors.push(check_output(actual, &references[*source])?);
            let serial_output = &serial_outputs[*source];
            all_match_serial &= bit_equal(actual, serial_output);
            let mut difference = ErrorStats::default();
            for (a, b) in actual.iter().zip(serial_output) {
                difference.observe(a.to_f32(), b.to_f32())?;
            }
            serial_differences.push(difference.report());
            if let Some(previous) = swapped_from {
                let expected = &outputs[*previous][(1 - lane) * HIDDEN..(2 - lane) * HIDDEN];
                if !bit_equal(actual, expected) {
                    return Err(format!(
                        "lane-swap mismatch in case {case_index} lane {lane}"
                    ));
                }
            }
        }
        if case_index == cases.len() - 1 && !bit_equal(&output, &outputs[0]) {
            return Err("batch FFN repeated-use mismatch after changing columns".into());
        }
        let input_bytes: Vec<_> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
        case_reports.push(json!({"source_indices":lanes,"swapped_from":swapped_from,
            "input_fp16_sha256":format!("{:x}", Sha256::digest(&input_bytes)),
            "cpu_errors_by_lane":errors,"serial_differences_by_lane":serial_differences}));
        outputs.push(output);
    }
    if !all_match_serial {
        return Err("batch FFN differs from serial; timing is not qualified".into());
    }
    let journal_enabled = std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some();
    let mut timing_preflight = None;
    let mut trials: Vec<Value> = Vec::new();
    let mut pairs = Vec::new();
    let mut timing_evaluations = 0;
    let mut timing_preflight_rejection = None;
    let mut interleaved_result = None;
    if !journal_enabled {
        if compile_budget_used() != 0 {
            return Err("batched FFN timing requires zero compiler calls".into());
        }
        let monitor = PowerMonitor::start(None).map_err(|error| error.to_string())?;
        let preflight = monitor.begin().finish(1);
        let controls = batch_timing::stratum(&preflight, exploratory_fair);
        let can_time = controls.is_ok();
        timing_preflight_rejection = controls.err();
        timing_preflight = Some(preflight);
        if can_time {
            let lanes = [inputs.len() - 1, 0];
            let input = pair_input(&all_inputs, lanes);
            let expected = pair_input(&serial_outputs, lanes);
            let mut output = vec![f16::ZERO; 2 * HIDDEN];
            for _ in 0..8 {
                serial.project(&all_inputs[lanes[0]], &mut output[..HIDDEN])?;
                serial.project(&all_inputs[lanes[1]], &mut output[HIDDEN..])?;
                batch.project_batch(&input, &mut output)?;
                timing_evaluations += 3;
            }
            if interleaved {
                interleaved_result = Some(interleaved_timing::run(
                    &monitor,
                    exploratory_fair,
                    |use_batch, repetitions| {
                        let start = Instant::now();
                        for _ in 0..repetitions {
                            if use_batch {
                                batch.project_batch(&input, &mut output)?;
                            } else {
                                serial.project(&all_inputs[lanes[0]], &mut output[..HIDDEN])?;
                                serial.project(&all_inputs[lanes[1]], &mut output[HIDDEN..])?;
                            }
                        }
                        let elapsed = start.elapsed();
                        timing_evaluations += repetitions * if use_batch { 1 } else { 2 };
                        if !bit_equal(&output, &expected) {
                            return Err("interleaved FFN output differs from qualification".into());
                        }
                        Ok((start, elapsed))
                    },
                )?);
            } else {
                let repetitions = 128;
                for block in 0..3 {
                    let start = trials.len();
                    let order = if block % 2 == 0 {
                        [false, true, true, false]
                    } else {
                        [true, false, false, true]
                    };
                    for use_batch in order {
                        let phase = monitor.begin();
                        for _ in 0..repetitions {
                            if use_batch {
                                batch.project_batch(&input, &mut output)?;
                            } else {
                                serial.project(&all_inputs[lanes[0]], &mut output[..HIDDEN])?;
                                serial.project(&all_inputs[lanes[1]], &mut output[HIDDEN..])?;
                            }
                        }
                        let measurement = phase.finish(2 * repetitions);
                        timing_evaluations += repetitions * if use_batch { 1 } else { 2 };
                        if !bit_equal(&output, &expected) {
                            return Err(
                                "batch FFN timing output differs from serial qualification".into(),
                            );
                        }
                        let controls = batch_timing::stratum(&measurement, exploratory_fair);
                        let eligible = controls.is_ok();
                        trials.push(json!({"batch":use_batch,"repetitions":repetitions,
                        "useful_outputs_per_repetition":2,"source_indices":lanes,"measurement":measurement,
                        "timing_controls_eligible":eligible,"timing_rejection":controls.err(),
                        "output_bit_identical_to_serial":true}));
                        if !eligible {
                            break;
                        }
                    }
                    if trials.len() - start != 4 {
                        break;
                    }
                    for (a, b) in [(start, start + 1), (start + 3, start + 2)] {
                        let (baseline, candidate) = if trials[a]["batch"] == false {
                            (a, b)
                        } else {
                            (b, a)
                        };
                        pairs.push(match batch_timing::compare(
                        &trials[baseline]["measurement"], &trials[candidate]["measurement"],
                        exploratory_fair,
                    ) {
                        Ok(comparison) => json!({"baseline":baseline,"candidate":candidate,"eligible":true,"comparison":comparison}),
                        Err(reason) => json!({"baseline":baseline,"candidate":candidate,"eligible":false,"reason":reason}),
                    });
                    }
                }
            }
        }
    }
    drop(batch);
    drop(serial);
    let timing_summary = if trials.len() == 12 {
        Some(batch_timing::summarize(&trials, exploratory_fair)?)
    } else {
        None
    };
    Ok(
        json!({"schema":"rvllm.int8_ffn_logical_batch.v1","layer":layer,"logical_tokens":2,
        "reconstructed_fp16_sha256":reconstructed_hash,"identical_int8_weights":true,
        "source_bytes_per_graph":weights.source_blob_bytes(),"input_sources":input_sources,
        "zero_source_index":zero,"serial_cpu_errors":serial_errors,"cases":case_reports,
        "lane_swap_bit_parity":true,"repeated_use_bit_parity":true,"zero_isolation":true,
        "batch_outputs_bit_identical_to_serial":all_match_serial,
        "qualification_evaluations":all_inputs.len()+cases.len(),"timing_evaluations":timing_evaluations,
        "compiler_calls":compile_budget_used(),"driver_journal_enabled":journal_enabled,
        "timing_preflight":timing_preflight,"trials":trials,"pairs":pairs,"models_dropped":2,
        "timing_preflight_rejection":timing_preflight_rejection,
        "timing_summary":timing_summary,
        "timing_protocol":if interleaved {"interleaved_32x4x4_v1"} else {"burst_3x4x128_v1"},
        "interleaved_timing":interleaved_result,
        "thermal_class":if exploratory_fair {"exploratory_stable_fair"} else {"nominal"},
        "claim":"One FFN, identical reconstructed INT8 weights, two logical tokens and one external input/output. Timing includes packing, evaluate and readback; serial executes twice for the same useful work. No accepted draft tokens, full-model speedup or physical bandwidth claim. CPU cycles exclude ANE."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_gate_rejects_wrong_shapes_nonfinite_and_large_errors() {
        assert!(check_output(&[f16::ONE], &[1.0]).is_ok());
        assert!(check_output(&[], &[]).is_err());
        assert!(check_output(&[f16::ONE], &[]).is_err());
        assert!(check_output(&[f16::NAN], &[1.0]).is_err());
        assert!(check_output(&[f16::ONE], &[f32::INFINITY]).is_err());
        assert!(check_output(&[f16::ONE], &[2.0]).is_err());
        assert!(!bit_equal(&[f16::ONE], &[]));
        assert!(!bit_equal(&[f16::ONE], &[f16::ZERO]));
    }

    #[test]
    fn token_major_pair_preserves_lane_order_and_zero_columns() {
        let inputs = vec![
            vec![f16::ONE; 3],
            vec![f16::from_f32(2.0); 3],
            vec![f16::ZERO; 3],
        ];
        for lanes in [[0, 1], [1, 0], [0, 2], [2, 0]] {
            let paired = pair_input(&inputs, lanes);
            assert!(bit_equal(&paired[..3], &inputs[lanes[0]]));
            assert!(bit_equal(&paired[3..], &inputs[lanes[1]]));
        }
    }
}
