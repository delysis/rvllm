//! Independent arithmetic variants for diagnosing a baseline/oracle mismatch.
//! These results never waive a device qualification tolerance.
#![forbid(unsafe_code)]

use super::*;

fn captured_outputs(
    path: &std::path::Path,
    sources: &[serde_json::Value],
    hash: &str,
) -> Result<(Vec<Vec<f16>>, String), String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let report: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if report["schema"] != "rvllm.int8_ffn_s1_capture.v1"
        || report["collection_complete"] != true
        || report["reconstructed_fp16_sha256"] != hash
        || report["input_sources"] != json!(sources)
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
        if row["ane_fp16_sha256"] != format!("{:x}", Sha256::digest(&bytes))
            || row["ane_fp16_output"]
                != json!(output.iter().map(|v| v.to_f32()).collect::<Vec<_>>())
        {
            return Err("S1 output bits, hash or values differ".into());
        }
        outputs.push(output);
    }
    Ok((outputs, format!("{:x}", Sha256::digest(&bytes))))
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
fn unrounded_quantized_dot(original: &[f16], input: &[f16]) -> Vec<f16> {
    original
        .chunks_exact(input.len())
        .map(|row| {
            let maximum = row.iter().map(|w| w.to_f32().abs()).fold(0.0_f32, f32::max);
            let scale = if maximum == 0.0 {
                f16::ONE
            } else {
                f16::from_f32(maximum / 127.0)
            }
            .to_f32();
            let dot: f64 = row
                .iter()
                .zip(input)
                .map(|(w, x)| {
                    let q = (w.to_f32() / scale).round_ties_even().clamp(-127.0, 127.0);
                    f64::from(q) * f64::from(scale) * f64::from(x.to_f32())
                })
                .sum();
            f16::from_f64(dot)
        })
        .collect()
}

fn unrounded_ffn(original: &[Vec<f16>; 3], input: &[f16]) -> CpuFfn {
    let gate = unrounded_quantized_dot(&original[0], input);
    let up = unrounded_quantized_dot(&original[1], input);
    let gated: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| mil_gated(g, u))
        .collect();
    let output = unrounded_quantized_dot(&original[2], &gated)
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
    original: &[Vec<f16>; 3],
    weights: &[Vec<f16>; 3],
    inputs: &[Vec<f16>],
    sources: &[serde_json::Value],
    hash: &str,
    capture: Option<&std::path::Path>,
) -> Result<serde_json::Value, String> {
    if inputs.is_empty() || inputs.len() != sources.len() {
        return Err("CPU diagnosis requires identified inputs".into());
    }
    let captured = capture
        .map(|path| captured_outputs(path, sources, hash))
        .transpose()?;
    let mut per_input = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let legacy = cpu_ffn(weights, input);
        let explicit_f32 = explicit_ffn(weights, input, false);
        let explicit_f64 = explicit_ffn(weights, input, true);
        let unrounded = unrounded_ffn(original, input);
        let ane_comparisons = captured.as_ref().map(|(outputs, _)| {
            let actual = &outputs[index];
            Ok::<_, String>(json!({
                "legacy":ffn_capture::compare_all(actual, &legacy.output)?,
                "mil_fp16_f32_dot":ffn_capture::compare_all(actual, &explicit_f32.output)?,
                "mil_fp16_f64_dot":ffn_capture::compare_all(actual, &explicit_f64.output)?,
                "unrounded_int8_times_scale_f64":ffn_capture::compare_all(actual, &unrounded.output)?,
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
            "unrounded_vs_mil_f64_gate_up_gated_down":differences(&unrounded, &explicit_f64)?,
        }));
    }
    Ok(
        json!({"schema":"rvllm.ffn_cpu_oracle_diagnostic.v1","accelerator_calls":0,
        "compiler_calls":0,"reconstructed_fp16_sha256":hash,"input_sources":sources,"per_input":per_input,
        "s1_capture_file":capture,"s1_capture_sha256":captured.as_ref().map(|(_,hash)|hash),
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
        let sources = vec![json!({"synthetic_pattern":0})];
        let mut report = json!({"schema":"rvllm.int8_ffn_s1_capture.v1","collection_complete":true,
            "reconstructed_fp16_sha256":"weights","input_sources":sources,
            "compiler_calls":0,"accelerator_evaluations":1,"per_input":[{
                "ane_fp16_bits":vec![0_u16; HIDDEN],"ane_fp16_output":vec![0.0_f32; HIDDEN],
                "ane_fp16_sha256":format!("{:x}",Sha256::digest(vec![0_u8;2*HIDDEN]))}]});
        let write = |report: &serde_json::Value| {
            std::fs::write(&path, serde_json::to_vec(report).unwrap()).unwrap()
        };
        write(&report);
        assert_eq!(
            captured_outputs(&path, &sources, "weights").unwrap().0[0].len(),
            HIDDEN
        );
        assert!(captured_outputs(&path, &sources, "different-weights").is_err());
        assert!(captured_outputs(&path, &[json!({"synthetic_pattern":1})], "weights").is_err());
        report["per_input"][0]["ane_fp16_bits"][0] = json!(f16::ONE.to_bits());
        write(&report);
        assert!(captured_outputs(&path, &sources, "weights").is_err());
        report["per_input"][0]["ane_fp16_bits"][0] = json!(0);
        report["per_input"][0]["ane_fp16_output"][0] = json!(1.0);
        write(&report);
        assert!(captured_outputs(&path, &sources, "weights").is_err());
    }
}
