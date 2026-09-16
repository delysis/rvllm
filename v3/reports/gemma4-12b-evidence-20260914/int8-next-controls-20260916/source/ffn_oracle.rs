//! Independent arithmetic variants for diagnosing a baseline/oracle mismatch.
//! These results never waive a device qualification tolerance.
#![forbid(unsafe_code)]

use super::*;
use rvllm_apple::ane_int8_ffn_weights::AneInt8MatrixView;

pub(super) fn affine_hash(weights: &AneInt8FfnWeights) -> String {
    let mut hash = Sha256::new();
    hash.update(b"rvllm.int8-affine-coefficients.v1\0");
    let mut bytes = [0_u8; 4096];
    for matrix in weights.matrices() {
        hash.update((matrix.scales.len() as u64).to_le_bytes());
        hash.update((matrix.columns as u64).to_le_bytes());
        for chunk in matrix.values.chunks(bytes.len()) {
            for (slot, value) in bytes.iter_mut().zip(chunk) {
                *slot = value.to_le_bytes()[0];
            }
            hash.update(&bytes[..chunk.len()]);
        }
        for scale in matrix.scales {
            hash.update(scale.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn captured_outputs(
    path: &std::path::Path,
    sources: &[serde_json::Value],
    hash: &str,
    affine_hash: &str,
    allow_legacy: bool,
) -> Result<(Vec<Vec<f16>>, String, bool), String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let report: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let exact_affine_identity = report["schema"] == "rvllm.int8_ffn_s1_capture.v2";
    let mut expected_sources = sources.to_vec();
    if !exact_affine_identity {
        if !allow_legacy || report["schema"] != "rvllm.int8_ffn_s1_capture.v1" {
            return Err("capture lacks exact affine/input identity; explicit legacy diagnostic opt-in required".into());
        }
        // Old captures identified their synthetic generator but not its bytes.
        // Preserve that limitation explicitly; these are diagnostic inputs.
        for source in &mut expected_sources {
            if source.get("synthetic_pattern").is_some() {
                source
                    .as_object_mut()
                    .ok_or("invalid input source")?
                    .remove("sha256");
            }
        }
    }
    if (exact_affine_identity && report["affine_coefficients_sha256"] != affine_hash)
        || report["collection_complete"] != true
        || report["reconstructed_fp16_sha256"] != hash
        || report["input_sources"] != json!(expected_sources)
        || report["compiler_calls"] != 0
        || report["accelerator_evaluations"] != sources.len()
    {
        return Err("S1 capture identity, weights, work count or inputs differ".into());
    }
    let rows = report["per_input"]
        .as_array()
        .ok_or("missing S1 capture outputs")?;
    if rows.len() != sources.len() {
        return Err("S1 capture output count differs".into());
    }
    let mut outputs = Vec::new();
    for row in rows {
        let bits = row["ane_fp16_bits"]
            .as_array()
            .ok_or("missing S1 output bits")?;
        if bits.len() != HIDDEN {
            return Err("S1 output width differs".into());
        }
        let output: Vec<_> = bits
            .iter()
            .map(|value| {
                let bits = value
                    .as_u64()
                    .and_then(|n| u16::try_from(n).ok())
                    .ok_or("invalid S1 output bits")?;
                let value = f16::from_bits(bits);
                if !value.is_finite() {
                    return Err("nonfinite S1 output");
                }
                Ok(value)
            })
            .collect::<Result<_, _>>()?;
        let bytes: Vec<_> = output
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        if row["ane_fp16_sha256"] != format!("{:x}", Sha256::digest(&bytes)) {
            return Err("S1 output bits and hash differ".into());
        }
        // JSON numbers are parsed as f64, whose last bit may depend on the
        // parser's decimal conversion. This mirror is specified as f32; check
        // exact f32 bits after deserializing to that type. The original FP16
        // bits and their byte hash remain independently authoritative.
        let values: Vec<f32> = serde_json::from_value(row["ane_fp16_output"].clone())
            .map_err(|error| error.to_string())?;
        if values.len() != output.len()
            || values
                .iter()
                .zip(&output)
                .any(|(a, b)| a.to_bits() != b.to_f32().to_bits())
        {
            return Err("S1 output bits and f32 values differ".into());
        }
        outputs.push(output);
    }
    Ok((
        outputs,
        format!("{:x}", Sha256::digest(&bytes)),
        exact_affine_identity,
    ))
}

fn rounded_dot(weights: &[f16], input: &[f16], wide: bool) -> Vec<f16> {
    if wide {
        weights
            .chunks_exact(input.len())
            .map(|row| {
                f16::from_f64(
                    row.iter()
                        .zip(input)
                        .map(|(w, x)| f64::from(w.to_f32()) * f64::from(x.to_f32()))
                        .sum(),
                )
            })
            .collect()
    } else {
        cpu_project(weights, input)
            .into_iter()
            .map(f16::from_f32)
            .collect()
    }
}

fn mil_gated(gate: f16, up: f16) -> f16 {
    let mul = |a: f16, b: f16| f16::from_f32(a.to_f32() * b.to_f32());
    let add = |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32());
    let squared = mul(gate, gate);
    let cubed = mul(squared, gate);
    let cubic_scaled = mul(cubed, f16::from_f32(0.044715));
    let sum = add(gate, cubic_scaled);
    let argument = mul(sum, f16::from_f32(0.797_884_6));
    let tanh = f16::from_f32(argument.to_f32().tanh());
    let factor = add(tanh, f16::ONE);
    let halved = mul(gate, f16::from_f32(0.5));
    let activated = mul(halved, factor);
    mul(activated, up)
}

fn explicit_ffn(weights: &[Vec<f16>; 3], input: &[f16], wide: bool) -> CpuFfn {
    let gate = rounded_dot(&weights[0], input, wide);
    let up = rounded_dot(&weights[1], input, wide);
    let gated: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| mil_gated(g, u))
        .collect();
    let output = rounded_dot(&weights[2], &gated, wide)
        .into_iter()
        .map(f16::to_f32)
        .collect();
    CpuFfn {
        gate,
        up,
        gated,
        output,
    }
}

// Diagnostic hypothesis: keep INT8 times its stored FP16 scale unrounded
// through dot accumulation. This does not establish the ANE lowering.
fn unrounded_quantized_dot(matrix: &AneInt8MatrixView<'_>, input: &[f16]) -> Vec<f16> {
    assert_eq!(matrix.columns, input.len());
    matrix
        .values
        .chunks_exact(input.len())
        .zip(matrix.scales)
        .map(|(row, scale)| {
            let dot: f64 = row
                .iter()
                .zip(input)
                .map(|(&q, x)| f64::from(q) * f64::from(scale.to_f32()) * f64::from(x.to_f32()))
                .sum();
            f16::from_f64(dot)
        })
        .collect()
}

fn unrounded_ffn(affine: &AneInt8FfnWeights, input: &[f16]) -> CpuFfn {
    let matrices = affine.matrices();
    let gate = unrounded_quantized_dot(&matrices[0], input);
    let up = unrounded_quantized_dot(&matrices[1], input);
    let gated: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| mil_gated(g, u))
        .collect();
    let output = unrounded_quantized_dot(&matrices[2], &gated)
        .into_iter()
        .map(f16::to_f32)
        .collect();
    CpuFfn {
        gate,
        up,
        gated,
        output,
    }
}

fn differences(actual: &CpuFfn, reference: &CpuFfn) -> Result<serde_json::Value, String> {
    let mut stages: [ErrorStats; 4] = std::array::from_fn(|_| ErrorStats::default());
    for (stats, (a, b)) in
        stages[..3]
            .iter_mut()
            .zip([&actual.gate, &actual.up, &actual.gated].into_iter().zip([
                &reference.gate,
                &reference.up,
                &reference.gated,
            ]))
    {
        for (a, b) in a.iter().zip(b) {
            stats.observe(a.to_f32(), b.to_f32())?;
        }
    }
    for (a, b) in actual.output.iter().zip(&reference.output) {
        stages[3].observe(*a, *b)?;
    }
    Ok(json!(stages.map(|s| s.report())))
}

pub(super) fn diagnose(
    affine: &AneInt8FfnWeights,
    weights: &[Vec<f16>; 3],
    inputs: &[Vec<f16>],
    sources: &[serde_json::Value],
    hash: &str,
    capture: Option<&std::path::Path>,
    allow_legacy: bool,
) -> Result<serde_json::Value, String> {
    if inputs.is_empty() || inputs.len() != sources.len() {
        return Err("CPU diagnosis requires identified inputs".into());
    }
    let affine_hash = affine_hash(affine);
    let captured = capture
        .map(|path| captured_outputs(path, sources, hash, &affine_hash, allow_legacy))
        .transpose()?;
    let mut per_input = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let legacy = cpu_ffn(weights, input);
        let explicit_f32 = explicit_ffn(weights, input, false);
        let explicit_f64 = explicit_ffn(weights, input, true);
        let unrounded = unrounded_ffn(affine, input);
        let unrounded_gate_up: Vec<_> = rounded_dot(&weights[2], &unrounded.gated, true)
            .into_iter()
            .map(f16::to_f32)
            .collect();
        let unrounded_down: Vec<_> =
            unrounded_quantized_dot(&affine.matrices()[2], &explicit_f64.gated)
                .into_iter()
                .map(f16::to_f32)
                .collect();
        let ane_comparisons = captured.as_ref().map(|(outputs, _, _)| {
            let actual = &outputs[index];
            Ok::<_, String>(json!({
                "legacy":ffn_capture::compare_all(actual, &legacy.output)?,
                "mil_fp16_f32_dot":ffn_capture::compare_all(actual, &explicit_f32.output)?,
                "mil_fp16_f64_dot":ffn_capture::compare_all(actual, &explicit_f64.output)?,
                "unrounded_int8_times_scale_f64":ffn_capture::compare_all(actual, &unrounded.output)?,
                "unrounded_gate_up_rounded_down":ffn_capture::compare_all(actual, &unrounded_gate_up)?,
                "rounded_gate_up_unrounded_down":ffn_capture::compare_all(actual, &unrounded_down)?,
            }))
        }).transpose()?;
        per_input.push(json!({
            "ane_comparisons":ane_comparisons,
            "mil_f32_vs_legacy_gate_up_gated_down":differences(&explicit_f32, &legacy)?,
            "mil_f64_vs_mil_f32_gate_up_gated_down":differences(&explicit_f64, &explicit_f32)?,
            "legacy_f32_output":legacy.output,
            "mil_fp16_f32_dot_output":explicit_f32.output,
            "mil_fp16_f64_dot_output":explicit_f64.output,
            "unrounded_int8_times_scale_f64_output":unrounded.output,
            "unrounded_gate_up_rounded_down_output":unrounded_gate_up,
            "rounded_gate_up_unrounded_down_output":unrounded_down,
            "unrounded_vs_mil_f64_gate_up_gated_down":differences(&unrounded, &explicit_f64)?,
        }));
    }
    Ok(
        json!({"schema":"rvllm.ffn_cpu_oracle_diagnostic.v2","accelerator_calls":0,
        "compiler_calls":0,"reconstructed_fp16_sha256":hash,"input_sources":sources,"per_input":per_input,
        "affine_coefficients_sha256":affine_hash,
        "capture_exact_affine_and_input_identity":captured.as_ref().map(|(_,_,exact)|exact),
        "legacy_capture_explicitly_allowed":allow_legacy,
        "s1_capture_file":capture,"s1_capture_sha256":captured.as_ref().map(|(_,hash,_)|hash),
        "claim":"Offline arithmetic diagnosis only. Legacy reference is retained. Explicit variants round each declared MIL FP16 operation and output; dot accumulation uses sequential FP32 or FP64. The unrounded variant tests an INT8-times-scale accumulation hypothesis. ANE compiler fusion/reduction/dequantization order is not established. No numerical gate or tolerance is changed."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_outputs_require_matching_identity_and_bitwise_integrity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("capture.json");
        let sources = vec![json!({"synthetic_pattern":0,"sha256":"input-bytes"})];
        let values: Vec<_> = (0..HIDDEN)
            .map(|i| {
                let value = f16::from_bits((17 * i) as u16);
                if value.is_finite() {
                    value
                } else {
                    f16::ZERO
                }
            })
            .collect();
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut report = json!({"schema":"rvllm.int8_ffn_s1_capture.v2","collection_complete":true,
            "affine_coefficients_sha256":"affine",
            "reconstructed_fp16_sha256":"weights","input_sources":sources,
            "compiler_calls":0,"accelerator_evaluations":1,"per_input":[{
                "ane_fp16_bits":values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                "ane_fp16_output":values.iter().map(|v|v.to_f32()).collect::<Vec<_>>(),
                "ane_fp16_sha256":format!("{:x}",Sha256::digest(&bytes))}]});
        let write = |report: &serde_json::Value| {
            std::fs::write(&path, serde_json::to_vec(report).unwrap()).unwrap()
        };
        write(&report);
        assert_eq!(
            captured_outputs(&path, &sources, "weights", "affine", false)
                .unwrap()
                .0[0]
                .len(),
            HIDDEN
        );
        assert!(captured_outputs(&path, &sources, "different-weights", "affine", false).is_err());
        assert!(captured_outputs(&path, &sources, "weights", "different-scales", false).is_err());
        assert!(captured_outputs(
            &path,
            &[json!({"synthetic_pattern":0,"sha256":"different-input"})],
            "weights",
            "affine",
            false
        )
        .is_err());
        let mut legacy = report.clone();
        legacy["schema"] = json!("rvllm.int8_ffn_s1_capture.v1");
        legacy
            .as_object_mut()
            .unwrap()
            .remove("affine_coefficients_sha256");
        legacy["input_sources"][0]
            .as_object_mut()
            .unwrap()
            .remove("sha256");
        write(&legacy);
        assert!(captured_outputs(&path, &sources, "weights", "affine", false).is_err());
        assert!(
            !captured_outputs(&path, &sources, "weights", "affine", true)
                .unwrap()
                .2
        );
        report["per_input"][0]["ane_fp16_bits"][0] = json!(f16::ONE.to_bits());
        write(&report);
        assert!(captured_outputs(&path, &sources, "weights", "affine", false).is_err());
        report["per_input"][0]["ane_fp16_bits"][0] = json!(0);
        report["per_input"][0]["ane_fp16_output"][0] = json!(1.0);
        write(&report);
        assert!(captured_outputs(&path, &sources, "weights", "affine", false).is_err());
    }

    #[test]
    fn unrounded_dot_consumes_stored_integer_and_scale_rows() {
        let values = [1, 3, -2, 4];
        let scales = [f16::from_f32(0.5), f16::from_f32(0.25)];
        let view = AneInt8MatrixView {
            columns: 2,
            values: &values,
            scales: &scales,
        };
        let output = unrounded_quantized_dot(&view, &[f16::from_f32(2.0), f16::from_f32(-1.0)]);
        assert_eq!(output, vec![f16::from_f32(-0.5), f16::from_f32(-2.0)]);
    }
}
