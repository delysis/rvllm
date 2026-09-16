//! Preserve complete S1/S2 evidence when the existing CPU reference rejects.
//! Successful collection never establishes numerical or timing qualification.
#![forbid(unsafe_code)]

use super::*;

fn checkpoint(
    journal: &mut capture_journal::CaptureJournal,
    value: serde_json::Value,
) -> Result<(), String> {
    let bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
    journal.append(&bytes).map_err(|error| error.to_string())
}

fn output_record(output: &[f16], reference: &[f32]) -> Result<serde_json::Value, String> {
    let comparison = ffn_capture::compare_all(output, reference)?;
    let bytes: Vec<_> = output
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    Ok(json!({
        "ane_fp16_bits":output.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        "ane_fp16_sha256":format!("{:x}", Sha256::digest(&bytes)),
        "legacy_comparison":comparison,
    }))
}

fn bit_equal(left: &[f16], right: &[f16]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn serial_comparison(actual: &[f16], serial: &[f16]) -> Result<serde_json::Value, String> {
    if actual.is_empty() || actual.len() != serial.len() {
        return Err("batch capture serial comparison shape mismatch".into());
    }
    let mut errors = ErrorStats::default();
    let mut mismatches = Vec::new();
    for (index, (a, b)) in actual.iter().zip(serial).enumerate() {
        errors.observe(a.to_f32(), b.to_f32())?;
        if a.to_bits() != b.to_bits() {
            mismatches.push(json!({"index":index,"s1_bits":b.to_bits(),"s2_bits":a.to_bits()}));
        }
    }
    Ok(
        json!({"bit_identical":mismatches.is_empty(),"mismatches":mismatches,"errors":errors.report()}),
    )
}

pub(super) fn run(
    weights: &AneInt8FfnWeights,
    dense: &[Vec<f16>; 3],
    inputs: &[Vec<f16>],
    sources: &[serde_json::Value],
    layer: usize,
    reconstructed_hash: &str,
    report_path: &std::path::Path,
) -> Result<serde_json::Value, String> {
    if inputs.len() < 2
        || inputs.len() != sources.len()
        || inputs.iter().any(|input| input.len() != HIDDEN)
    {
        return Err("batch capture requires at least two identified H3840 inputs".into());
    }
    let mut inputs = inputs.to_vec();
    let zero = inputs.len();
    inputs.push(vec![f16::ZERO; HIDDEN]);
    let mut sources = sources.to_vec();
    sources.push(json!({"diagnostic_zero":true,"sha256":format!("{:x}", Sha256::digest(vec![0_u8; 2*HIDDEN]))}));
    let references: Vec<_> = inputs.iter().map(|x| cpu_ffn(dense, x).output).collect();
    let affine_hash = ffn_oracle::affine_hash(weights);
    let partial_path = report_path.with_extension("partial.jsonl");
    let mut journal = capture_journal::CaptureJournal::create(&partial_path)
        .map_err(|error| error.to_string())?;
    checkpoint(
        &mut journal,
        json!({"schema":"rvllm.int8_ffn_batch_capture_partial.v1",
        "collection_complete":false,"phase":"identity","layer":layer,
        "affine_coefficients_sha256":affine_hash,"reconstructed_fp16_sha256":reconstructed_hash,
        "input_sources":sources,"zero_source_index":zero,"expected_serial_evaluations":inputs.len(),
        "expected_batch_evaluations":2*zero+3,"numerical_qualification_claim":false,"timing_claim":false}),
    )?;
    let mut serial = AneGatedFfn::compile_int8_with_cache_policy(
        weights,
        AneProgramCachePolicy::RequireExisting,
    )?;
    let mut serial_outputs = vec![vec![f16::ZERO; HIDDEN]; inputs.len()];
    let mut serial_records = Vec::new();
    for (source, ((input, output), reference)) in inputs
        .iter()
        .zip(&mut serial_outputs)
        .zip(&references)
        .enumerate()
    {
        serial.project(input, output)?;
        let record = output_record(output, reference)?;
        checkpoint(
            &mut journal,
            json!({"collection_complete":false,"phase":"serial","source_index":source,"output":record}),
        )?;
        serial_records.push(record);
    }
    drop(serial);

    let mut cases = Vec::new();
    for source in 0..zero {
        let prior = cases.len();
        cases.push(([source, (source + 1) % zero], None));
        cases.push(([(source + 1) % zero, source], Some(prior)));
    }
    cases.push(([zero - 1, zero], None));
    cases.push(([zero, zero - 1], Some(cases.len() - 1)));
    cases.push((cases[0].0, None));

    let mut batch = AneGatedFfn::compile_int8_batch_with_cache_policy(
        weights,
        2,
        AneProgramCachePolicy::RequireExisting,
    )?;
    let mut outputs = Vec::<Vec<f16>>::new();
    let mut case_records = Vec::new();
    let mut all_match_serial = true;
    let mut all_swap = true;
    let mut all_zero = true;
    let mut all_legacy = serial_records
        .iter()
        .all(|r| r["legacy_comparison"]["tolerance_passed"] == true);
    for (case, (lanes, swapped_from)) in cases.iter().enumerate() {
        let input: Vec<_> = inputs[lanes[0]]
            .iter()
            .chain(&inputs[lanes[1]])
            .copied()
            .collect();
        let mut output = vec![f16::ZERO; 2 * HIDDEN];
        batch.project_batch(&input, &mut output)?;
        let mut records = Vec::new();
        for (lane, source) in lanes.iter().enumerate() {
            let actual = &output[lane * HIDDEN..(lane + 1) * HIDDEN];
            let mut record = output_record(actual, &references[*source])?;
            record["serial_comparison"] = serial_comparison(actual, &serial_outputs[*source])?;
            all_match_serial &= record["serial_comparison"]["bit_identical"] == true;
            all_legacy &= record["legacy_comparison"]["tolerance_passed"] == true;
            if *source == zero {
                let passed = actual.iter().all(|v| v.to_f32() == 0.0);
                record["zero_isolation_passed"] = json!(passed);
                all_zero &= passed;
            }
            if let Some(prior) = swapped_from {
                let passed = bit_equal(
                    actual,
                    &outputs[*prior][(1 - lane) * HIDDEN..(2 - lane) * HIDDEN],
                );
                record["swap_bit_parity_passed"] = json!(passed);
                all_swap &= passed;
            }
            records.push(record);
        }
        let record =
            json!({"case":case,"source_indices":lanes,"swapped_from":swapped_from,"lanes":records});
        checkpoint(
            &mut journal,
            json!({"collection_complete":false,"phase":"batch","output":record}),
        )?;
        case_records.push(record);
        outputs.push(output);
    }
    drop(batch);
    if rvllm_apple::ane_linear::compile_budget_used() != 0 {
        return Err("batch data capture requires zero compiler calls".into());
    }
    let repeated = bit_equal(&outputs[0], outputs.last().ok_or("no batch outputs")?);
    Ok(
        json!({"schema":"rvllm.int8_ffn_batch_capture.v1","collection_complete":true,
        "layer":layer,"logical_tokens":2,"input_sources":sources,"zero_source_index":zero,
        "affine_coefficients_sha256":affine_hash,"reconstructed_fp16_sha256":reconstructed_hash,
        "partial_capture_journal":partial_path,
        "serial_outputs":serial_records,"cases":case_records,"batch_outputs_bit_identical_to_serial":all_match_serial,
        "lane_swap_bit_parity":all_swap,"zero_isolation":all_zero,"repeated_use_bit_parity":repeated,
        "legacy_gate_passed":all_legacy,"numerical_qualification_claim":false,"timing_claim":false,
        "compiler_calls":0,"serial_evaluations":inputs.len(),"batch_evaluations":cases.len(),"models_dropped":2,
        "driver_journal_enabled":true,
        "claim":"Finite data collection only, with every legacy tolerance failure and S1/S2 bit mismatch retained. No oracle, tolerance or qualification gate changed. No timing, accepted draft token, full-model or bandwidth claim."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_preserves_numerical_rejection_and_signed_zero_difference() {
        let record = output_record(&[f16::ONE], &[2.0]).unwrap();
        assert_eq!(record["legacy_comparison"]["tolerance_passed"], false);
        assert_eq!(
            record["legacy_comparison"]["failures"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        let comparison = serial_comparison(&[f16::ZERO], &[f16::NEG_ZERO]).unwrap();
        assert_eq!(comparison["bit_identical"], false);
        assert_eq!(comparison["mismatches"][0]["index"], 0);
        assert_eq!(comparison["errors"]["max_abs_error"], 0.0);
        assert!(!bit_equal(&[f16::ZERO], &[f16::NEG_ZERO]));
    }

    #[test]
    fn capture_refuses_incomplete_or_nonfinite_comparisons() {
        assert!(serial_comparison(&[], &[]).is_err());
        assert!(serial_comparison(&[f16::ZERO], &[]).is_err());
        assert!(serial_comparison(&[f16::NAN], &[f16::ZERO]).is_err());
        assert!(output_record(&[f16::ONE], &[f32::INFINITY]).is_err());
    }
}
