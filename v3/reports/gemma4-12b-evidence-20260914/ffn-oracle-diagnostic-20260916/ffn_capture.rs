//! Collect baseline outputs without discarding numerical rejection evidence.
//! A completed capture is not a successful numerical qualification.
#![forbid(unsafe_code)]

use super::*;

pub(super) fn compare_all(actual: &[f16], reference: &[f32]) -> Result<serde_json::Value, String> {
    if actual.is_empty() || actual.len() != reference.len() {
        return Err("S1 capture output/reference shape mismatch".into());
    }
    let mut errors = ErrorStats::default();
    let mut failures = Vec::new();
    for (index, (&actual, &expected)) in actual.iter().zip(reference).enumerate() {
        let actual = actual.to_f32();
        errors.observe(actual, expected)?;
        let tolerance = 0.01 + 0.02 * expected.abs();
        if (actual - expected).abs() > tolerance {
            failures.push(json!({"index":index,"actual":actual,"reference":expected,
                "absolute_error":(actual-expected).abs(),"tolerance":tolerance}));
        }
    }
    Ok(json!({"tolerance_passed":failures.is_empty(),"failures":failures,"errors":errors.report()}))
}

pub(super) fn run(
    weights: &AneInt8FfnWeights,
    dense: &[Vec<f16>; 3],
    inputs: &[Vec<f16>],
    sources: &[serde_json::Value],
    layer: usize,
    hash: &str,
) -> Result<serde_json::Value, String> {
    if inputs.is_empty() || inputs.len() != sources.len() {
        return Err("S1 capture requires identified inputs".into());
    }
    let references: Vec<_> = inputs
        .iter()
        .map(|input| cpu_ffn(dense, input).output)
        .collect();
    let mut serial = AneGatedFfn::compile_int8_with_cache_policy(
        weights,
        AneProgramCachePolicy::RequireExisting,
    )?;
    let mut outputs = vec![vec![f16::ZERO; HIDDEN]; inputs.len()];
    for (input, output) in inputs.iter().zip(&mut outputs) {
        serial.project(input, output)?;
    }
    // End device ownership before report construction or offline analysis.
    drop(serial);
    let mut per_input = Vec::new();
    let mut all_pass = true;
    for (output, reference) in outputs.iter().zip(&references) {
        let comparison = compare_all(output, reference)?;
        all_pass &= comparison["tolerance_passed"] == true;
        let bytes: Vec<_> = output
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        per_input.push(
            json!({"ane_fp16_output":output.iter().map(|v|v.to_f32()).collect::<Vec<_>>(),
            "ane_fp16_bits":output.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
            "ane_fp16_sha256":format!("{:x}",Sha256::digest(&bytes)),
            "legacy_f32_output":reference,"legacy_comparison":comparison}),
        );
    }
    Ok(
        json!({"schema":"rvllm.int8_ffn_s1_capture.v1","collection_complete":true,
        "layer":layer,"reconstructed_fp16_sha256":hash,"input_sources":sources,
        "per_input":per_input,"legacy_gate_passed":all_pass,
        "compiler_calls":rvllm_apple::ane_linear::compile_budget_used(),
        "accelerator_evaluations":inputs.len(),"models_dropped":1,
        "driver_journal_enabled":true,"timing_claim":false,
        "claim":"Baseline S1 data collection only. Every legacy tolerance failure is retained. Exit success means complete finite output collection, not numerical qualification. No candidate evaluation, tolerance change, oracle promotion or speed claim."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_retains_every_rejection_without_claiming_qualification() {
        let result =
            compare_all(&[f16::ONE, f16::ZERO, f16::from_f32(4.0)], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(result["tolerance_passed"], false);
        assert_eq!(result["failures"].as_array().unwrap().len(), 2);
        assert_eq!(result["failures"][0]["index"], 1);
        assert_eq!(result["failures"][1]["index"], 2);
        assert_eq!(result["errors"]["elements"], 3);
        assert_eq!(
            compare_all(&[f16::ONE], &[1.0]).unwrap()["tolerance_passed"],
            true
        );
    }

    #[test]
    fn capture_rejects_nonfinite_and_incomplete_outputs() {
        assert!(compare_all(&[], &[]).is_err());
        assert!(compare_all(&[f16::ONE], &[]).is_err());
        assert!(compare_all(&[f16::NAN], &[1.0]).is_err());
        assert!(compare_all(&[f16::ONE], &[f32::INFINITY]).is_err());
    }
}
