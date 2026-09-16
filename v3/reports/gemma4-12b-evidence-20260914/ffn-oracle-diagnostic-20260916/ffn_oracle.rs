//! Independent arithmetic variants for diagnosing a baseline/oracle mismatch.
//! These results never waive a device qualification tolerance.
#![forbid(unsafe_code)]

use super::*;

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
    weights: &[Vec<f16>; 3],
    inputs: &[Vec<f16>],
    sources: &[serde_json::Value],
    hash: &str,
) -> Result<serde_json::Value, String> {
    let mut per_input = Vec::new();
    for input in inputs {
        let legacy = cpu_ffn(weights, input);
        let explicit_f32 = explicit_ffn(weights, input, false);
        let explicit_f64 = explicit_ffn(weights, input, true);
        per_input.push(json!({
            "mil_f32_vs_legacy_gate_up_gated_down":differences(&explicit_f32, &legacy)?,
            "mil_f64_vs_mil_f32_gate_up_gated_down":differences(&explicit_f64, &explicit_f32)?,
            "legacy_f32_output":legacy.output,
            "mil_fp16_f32_dot_output":explicit_f32.output,
            "mil_fp16_f64_dot_output":explicit_f64.output,
        }));
    }
    Ok(
        json!({"schema":"rvllm.ffn_cpu_oracle_diagnostic.v1","accelerator_calls":0,
        "compiler_calls":0,"reconstructed_fp16_sha256":hash,"input_sources":sources,"per_input":per_input,
        "claim":"Offline arithmetic diagnosis only. Legacy reference is retained. Explicit variants round each declared MIL FP16 operation and output; dot accumulation uses sequential FP32 or FP64. ANE compiler fusion/reduction order is not established. No numerical gate or tolerance is changed."}),
    )
}
