//! Default-off source contract for fusing an output projection and one FFN.
//!
//! This module deliberately exposes source construction and a CPU boundary
//! oracle only. Device RMS reduction/rounding must pass component correctness
//! before a decoder route or evaluation API is added.

use crate::ane_int8_ffn_weights::AneInt8FfnWeights;
use crate::gemma_decode_math::rms_norm_f16_in_place;
use half::f16;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AneOutputFfnIdentity {
    pub mil_sha256: String,
    pub weight_blob_sha256: String,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub external_inputs: usize,
    pub external_outputs: usize,
}

pub struct AneOutputFfnSource {
    pub mil: String,
    pub blob: Vec<u8>,
    pub identity: AneOutputFfnIdentity,
}

impl AneOutputFfnSource {
    pub fn build(
        output_weights: &[f16],
        attention_width: usize,
        ffn: &AneInt8FfnWeights,
        post_attention_gamma: &[f16],
        pre_ffn_gamma: &[f16],
        epsilon: f32,
    ) -> Result<Self, String> {
        let (hidden, intermediate) = ffn.shape();
        if hidden == 0
            || attention_width == 0
            || post_attention_gamma.len() != hidden
            || pre_ffn_gamma.len() != hidden
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err("ANE output/FFN norm shape or epsilon is invalid".into());
        }
        let input_channels = attention_width
            .checked_add(hidden)
            .ok_or("ANE output/FFN input width overflow")?;
        let input_bytes = io_bytes(input_channels)?;
        let output_bytes = io_bytes(hidden)?;
        let (blob, constants) = ffn.output_ffn_blob_and_constants(
            output_weights,
            attention_width,
            post_attention_gamma,
            pre_ffn_gamma,
        )?;
        let mil = output_ffn_mil(hidden, attention_width, intermediate, epsilon, &constants);
        let identity = AneOutputFfnIdentity {
            mil_sha256: hex_sha256(mil.as_bytes()),
            weight_blob_sha256: hex_sha256(&blob),
            input_bytes,
            output_bytes,
            external_inputs: 1,
            external_outputs: 1,
        };
        Ok(Self {
            mil,
            blob,
            identity,
        })
    }
}

/// Host-side specification for the boundary crossed by the candidate graph.
/// `output_branch` is the already projected attention branch. The returned
/// pair is `(residual_after_attention, raw_ffn_branch)`. It is not evidence
/// that ANE reduction order or rounding matches the host.
pub fn output_ffn_cpu_boundary_oracle(
    output_branch: &[f16],
    residual: &[f16],
    post_attention_gamma: &[f16],
    pre_ffn_gamma: &[f16],
    ffn: &AneInt8FfnWeights,
    epsilon: f32,
) -> Result<(Vec<f16>, Vec<f16>), String> {
    let (hidden, intermediate) = ffn.shape();
    if output_branch.len() != hidden || residual.len() != hidden {
        return Err("ANE output/FFN CPU oracle hidden shape mismatch".into());
    }
    let mut branch = output_branch.to_vec();
    rms_norm_f16_in_place(&mut branch, hidden, Some(post_attention_gamma), epsilon)?;
    let mut merged = Vec::with_capacity(hidden);
    merged.extend(
        branch
            .iter()
            .zip(residual)
            .map(|(&a, &b)| f16::from_f32(a.to_f32() + b.to_f32())),
    );
    let mut normalized = merged.clone();
    rms_norm_f16_in_place(&mut normalized, hidden, Some(pre_ffn_gamma), epsilon)?;

    let [gate, up, down] = ffn.dequantized();
    let project = |weights: &[f16], rows: usize, columns: usize, x: &[f16]| {
        weights
            .chunks_exact(columns)
            .take(rows)
            .map(|row| {
                f16::from_f32(
                    row.iter()
                        .zip(x)
                        .map(|(&w, &v)| w.to_f32() * v.to_f32())
                        .sum(),
                )
            })
            .collect::<Vec<_>>()
    };
    let gate = project(&gate, intermediate, hidden, &normalized);
    let up = project(&up, intermediate, hidden, &normalized);
    let gated: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let x = g.to_f32();
            let gelu = 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
            f16::from_f32(gelu * u.to_f32())
        })
        .collect();
    Ok((merged, project(&down, hidden, intermediate, &gated)))
}

fn io_bytes(channels: usize) -> Result<usize, String> {
    channels
        .checked_mul(64)
        .filter(|&bytes| bytes > 0 && bytes <= u32::MAX as usize)
        .ok_or_else(|| "ANE output/FFN I/O shape is empty or exceeds 4 GiB".into())
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn output_ffn_mil(
    hidden: usize,
    attention: usize,
    intermediate: usize,
    epsilon: f32,
    weights: &str,
) -> String {
    let input = hidden + attention;
    format!(
        r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, {input}, 1, 1]> packed) {{
        tensor<int32, [4]> attended_begin = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [4]> residual_begin = const()[val = tensor<int32, [4]>([0, {attention}, 0, 0])];
        tensor<int32, [4]> attended_size = const()[val = tensor<int32, [4]>([1, {attention}, 1, 1])];
        tensor<int32, [4]> hidden_size = const()[val = tensor<int32, [4]>([1, {hidden}, 1, 1])];
        tensor<fp16, [1, {attention}, 1, 1]> attended = slice_by_size(x = packed, begin = attended_begin, size = attended_size);
        tensor<fp16, [1, {hidden}, 1, 1]> residual = slice_by_size(x = packed, begin = residual_begin, size = hidden_size);
        string valid = const()[val = string("valid")];
        tensor<int32, [2]> ones2 = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> zeros4 = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
        int32 groups = const()[val = int32(1)];
{weights}        tensor<int32, [3]> norm_axes = const()[val = tensor<int32, [3]>([1, 2, 3])];
        bool keep_dims = const()[val = bool(true)];
        fp16 norm_mean = const()[val = fp16({norm_mean})];
        fp16 norm_epsilon = const()[val = fp16({epsilon})];
        fp16 negative_half = const()[val = fp16(-0.5)];
        tensor<fp16, [1, {hidden}, 1, 1]> projected = conv(dilations = ones2, groups = groups, pad = zeros4, pad_type = valid, strides = ones2, weight = Wo, x = attended);
        tensor<fp16, [1, {hidden}, 1, 1]> post_sq = mul(x = projected, y = projected);
        tensor<fp16, [1, 1, 1, 1]> post_sum = reduce_sum(axes = norm_axes, keep_dims = keep_dims, x = post_sq);
        tensor<fp16, [1, 1, 1, 1]> post_mean = mul(x = post_sum, y = norm_mean);
        tensor<fp16, [1, 1, 1, 1]> post_variance = add(x = post_mean, y = norm_epsilon);
        tensor<fp16, [1, 1, 1, 1]> post_inv = pow(x = post_variance, y = negative_half);
        tensor<fp16, [1, {hidden}, 1, 1]> post_unit = mul(x = projected, y = post_inv);
        tensor<fp16, [1, {hidden}, 1, 1]> post_norm = mul(x = post_unit, y = post_gamma);
        tensor<fp16, [1, {hidden}, 1, 1]> merged = add(x = post_norm, y = residual);
        tensor<fp16, [1, {hidden}, 1, 1]> pre_sq = mul(x = merged, y = merged);
        tensor<fp16, [1, 1, 1, 1]> pre_sum = reduce_sum(axes = norm_axes, keep_dims = keep_dims, x = pre_sq);
        tensor<fp16, [1, 1, 1, 1]> pre_mean = mul(x = pre_sum, y = norm_mean);
        tensor<fp16, [1, 1, 1, 1]> pre_variance = add(x = pre_mean, y = norm_epsilon);
        tensor<fp16, [1, 1, 1, 1]> pre_inv = pow(x = pre_variance, y = negative_half);
        tensor<fp16, [1, {hidden}, 1, 1]> pre_unit = mul(x = merged, y = pre_inv);
        tensor<fp16, [1, {hidden}, 1, 1]> x = mul(x = pre_unit, y = pre_gamma);
        fp16 half = const()[val = fp16(0.5)];
        fp16 one = const()[val = fp16(1.0)];
        fp16 cubic = const()[val = fp16(0.044715)];
        fp16 root = const()[val = fp16(0.7978845608)];
        tensor<fp16, [1, {intermediate}, 1, 1]> gate = conv(dilations = ones2, groups = groups, pad = zeros4, pad_type = valid, strides = ones2, weight = Wg, x = x);
        tensor<fp16, [1, {intermediate}, 1, 1]> up = conv(dilations = ones2, groups = groups, pad = zeros4, pad_type = valid, strides = ones2, weight = Wu, x = x);
        tensor<fp16, [1, {intermediate}, 1, 1]> g2 = mul(x = gate, y = gate);
        tensor<fp16, [1, {intermediate}, 1, 1]> g3 = mul(x = g2, y = gate);
        tensor<fp16, [1, {intermediate}, 1, 1]> cubic_term = mul(x = g3, y = cubic);
        tensor<fp16, [1, {intermediate}, 1, 1]> gelu_inner = add(x = gate, y = cubic_term);
        tensor<fp16, [1, {intermediate}, 1, 1]> ga = mul(x = gelu_inner, y = root);
        tensor<fp16, [1, {intermediate}, 1, 1]> tanh_ga = tanh(x = ga);
        tensor<fp16, [1, {intermediate}, 1, 1]> tanh_plus_one = add(x = tanh_ga, y = one);
        tensor<fp16, [1, {intermediate}, 1, 1]> half_gate = mul(x = gate, y = half);
        tensor<fp16, [1, {intermediate}, 1, 1]> activated = mul(x = half_gate, y = tanh_plus_one);
        tensor<fp16, [1, {intermediate}, 1, 1]> gated = mul(x = activated, y = up);
        tensor<fp16, [1, {hidden}, 1, 1]> y = conv(dilations = ones2, groups = groups, pad = zeros4, pad_type = valid, strides = ones2, weight = Wd, x = gated);
    }} -> (y);
}}
"#,
        norm_mean = 1.0 / hidden as f32
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asymmetric_fixture() -> (
        usize,
        usize,
        usize,
        Vec<f16>,
        AneInt8FfnWeights,
        Vec<f16>,
        Vec<f16>,
    ) {
        let (hidden, intermediate, attention) = (32, 64, 32);
        let values = |count: usize, multiplier: usize, offset: usize, scale: f32| {
            (0..count)
                .map(|index| {
                    f16::from_f32((((index * multiplier + offset) % 61) as f32 - 30.0) * scale)
                })
                .collect::<Vec<_>>()
        };
        let gate = values(intermediate * hidden, 17, 3, 0.0015);
        let up = values(intermediate * hidden, 23, 7, 0.0013);
        let down = values(hidden * intermediate, 29, 11, 0.0011);
        let output = values(hidden * attention, 31, 13, 0.0017);
        let post_gamma = (0..hidden)
            .map(|index| f16::from_f32(0.8 + index as f32 / 100.0))
            .collect();
        let pre_gamma = (0..hidden)
            .map(|index| f16::from_f32(1.1 - index as f32 / 160.0))
            .collect();
        (
            hidden,
            intermediate,
            attention,
            output,
            AneInt8FfnWeights::quantize(&gate, &up, &down, hidden, intermediate).unwrap(),
            post_gamma,
            pre_gamma,
        )
    }

    #[test]
    fn source_is_single_io_stable_and_fail_closed() {
        let h = 32;
        let m = 64;
        let a = 48;
        let dense = vec![f16::from_f32(0.03125); h * m];
        let ffn = AneInt8FfnWeights::quantize(&dense, &dense, &dense, h, m).unwrap();
        let source = AneOutputFfnSource::build(
            &vec![f16::from_f32(0.0625); h * a],
            a,
            &ffn,
            &vec![f16::ONE; h],
            &vec![f16::ONE; h],
            1e-6,
        )
        .unwrap();
        assert_eq!(
            (
                source.identity.external_inputs,
                source.identity.external_outputs
            ),
            (1, 1)
        );
        assert_eq!(source.identity.input_bytes, (h + a) * 64);
        assert_eq!(source.identity.output_bytes, h * 64);
        assert_eq!(source.mil.matches("func main<ios18>").count(), 1);
        assert_eq!(source.mil.matches("reduce_sum(").count(), 2);
        assert_eq!(source.mil.matches("pow(").count(), 2);
        assert!(!source.mil.contains("rsqrt("));
        assert!(!source.mil.contains("layer_norm("));
        assert_eq!(source.mil.matches("constexpr_affine_dequantize").count(), 3);
        assert_eq!(source.blob[..4], 9_u32.to_le_bytes());
        assert!(AneOutputFfnSource::build(
            &[],
            a,
            &ffn,
            &vec![f16::ONE; h],
            &vec![f16::ONE; h],
            1e-6
        )
        .is_err());
    }

    #[test]
    fn cpu_oracle_specifies_residual_and_raw_ffn_boundary() {
        let h = 32;
        let m = 32;
        let dense = vec![f16::from_f32(0.01); h * m];
        let ffn = AneInt8FfnWeights::quantize(&dense, &dense, &dense, h, m).unwrap();
        let (merged, branch) = output_ffn_cpu_boundary_oracle(
            &vec![f16::from_f32(0.25); h],
            &vec![f16::from_f32(-0.125); h],
            &vec![f16::ONE; h],
            &vec![f16::ONE; h],
            &ffn,
            1e-6,
        )
        .unwrap();
        assert_eq!(merged.len(), h);
        assert_eq!(branch.len(), h);
        assert!(branch.iter().all(|value| value.is_finite()));
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE compile only; exactly one compile and zero evaluations"]
    fn hardware_output_ffn_compile_only_probe() {
        use crate::ane_linear::{compile_budget_used, AneOutputFfnCompile, AneProgramCachePolicy};

        let h = 32;
        let m = 32;
        let a = 32;
        let dense = vec![f16::from_f32(0.01); h * m];
        let ffn = AneInt8FfnWeights::quantize(&dense, &dense, &dense, h, m).unwrap();
        let before = compile_budget_used();
        let compiled = AneOutputFfnCompile::compile_only(
            &vec![f16::from_f32(0.02); h * a],
            a,
            &ffn,
            &vec![f16::ONE; h],
            &vec![f16::ONE; h],
            1e-6,
            AneProgramCachePolicy::Compile,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before + 1);
        assert_eq!(
            (
                compiled.identity().external_inputs,
                compiled.identity().external_outputs
            ),
            (1, 1)
        );
        println!("identity={:?}", compiled.identity());
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE strict cache reload only; zero compiles and zero evaluations"]
    fn hardware_output_ffn_strict_cache_reload_probe() {
        use crate::ane_linear::{compile_budget_used, AneOutputFfnCompile, AneProgramCachePolicy};

        let h = 32;
        let m = 32;
        let a = 32;
        let dense = vec![f16::from_f32(0.01); h * m];
        let ffn = AneInt8FfnWeights::quantize(&dense, &dense, &dense, h, m).unwrap();
        let before = compile_budget_used();
        let cached = AneOutputFfnCompile::compile_only(
            &vec![f16::from_f32(0.02); h * a],
            a,
            &ffn,
            &vec![f16::ONE; h],
            &vec![f16::ONE; h],
            1e-6,
            AneProgramCachePolicy::RequireExisting,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before);
        assert_eq!(
            (
                cached.identity().external_inputs,
                cached.identity().external_outputs
            ),
            (1, 1)
        );
        println!("identity={:?}", cached.identity());
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE component oracle; zero compiles and four evaluations"]
    fn hardware_output_ffn_component_correctness_probe() {
        use crate::ane_linear::{compile_budget_used, AneProgramCachePolicy};
        use rvllm_apple_ane_sys::AneInMemoryProgram;

        let h = 32;
        let m = 32;
        let a = 32;
        let dense = vec![f16::from_f32(0.01); h * m];
        let ffn = AneInt8FfnWeights::quantize(&dense, &dense, &dense, h, m).unwrap();
        let output_weights = vec![f16::from_f32(0.02); h * a];
        let gamma = vec![f16::ONE; h];
        let source =
            AneOutputFfnSource::build(&output_weights, a, &ffn, &gamma, &gamma, 1e-6).unwrap();
        let before = compile_budget_used();
        let program = AneInMemoryProgram::compile_with_cache_policy(
            &source.mil,
            &source.blob,
            source.identity.input_bytes,
            source.identity.output_bytes,
            AneProgramCachePolicy::RequireExisting,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before);
        let mut kernel = program.create_request().unwrap();
        let mut previous = None;
        for case in [0_usize, 0, 1, 2] {
            let sample =
                |i: usize| f16::from_f32(((i * 17 + case * 13 + 3) % 37) as f32 / 256.0 - 0.07);
            let attended: Vec<_> = (0..a).map(sample).collect();
            let residual: Vec<_> = (0..h).map(|i| sample(i * 3 + 5)).collect();
            let projected: Vec<_> = output_weights
                .chunks_exact(a)
                .map(|row| {
                    f16::from_f32(
                        row.iter()
                            .zip(&attended)
                            .map(|(&weight, &value)| weight.to_f32() * value.to_f32())
                            .sum(),
                    )
                })
                .collect();
            let (_, expected) =
                output_ffn_cpu_boundary_oracle(&projected, &residual, &gamma, &gamma, &ffn, 1e-6)
                    .unwrap();
            let mut input = vec![0_u8; source.identity.input_bytes];
            for (channel, value) in attended.iter().chain(&residual).enumerate() {
                let offset = channel * 64;
                input[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            }
            kernel.write_input(&input).unwrap();
            kernel.evaluate().unwrap();
            let mut bytes = vec![0_u8; source.identity.output_bytes];
            kernel.read_output(&mut bytes).unwrap();
            let actual: Vec<_> = bytes
                .chunks_exact(64)
                .map(|row| f16::from_le_bytes([row[0], row[1]]))
                .collect();
            assert!(actual.iter().all(|value| value.is_finite()));
            let max_absolute_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual.to_f32() - expected.to_f32()).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_absolute_error <= 0.02,
                "case {case}: {max_absolute_error}"
            );
            if case == 0 {
                if let Some(first) = &previous {
                    assert_eq!(first, &actual, "repeated execution changed output bits");
                }
                previous = Some(actual.clone());
            }
            eprintln!(
                "{}",
                serde_json::json!({
                    "case": case,
                    "maximum_absolute_error": max_absolute_error,
                    "output_bits": actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                })
            );
        }
        assert_eq!(compile_budget_used(), before);
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded asymmetric private-ANE compile only; exactly one compile and zero evaluations"]
    fn hardware_output_ffn_asymmetric_compile_probe() {
        use crate::ane_linear::{compile_budget_used, AneOutputFfnCompile, AneProgramCachePolicy};

        let (_, _, attention, output, ffn, post_gamma, pre_gamma) = asymmetric_fixture();
        let before = compile_budget_used();
        let compiled = AneOutputFfnCompile::compile_only(
            &output,
            attention,
            &ffn,
            &post_gamma,
            &pre_gamma,
            1e-6,
            AneProgramCachePolicy::Compile,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before + 1);
        println!("identity={:?}", compiled.identity());
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded asymmetric private-ANE component oracle; zero compiles and four evaluations"]
    fn hardware_output_ffn_asymmetric_correctness_probe() {
        use crate::ane_linear::{compile_budget_used, AneProgramCachePolicy};
        use rvllm_apple_ane_sys::AneInMemoryProgram;

        let (hidden, _, attention, output, ffn, post_gamma, pre_gamma) = asymmetric_fixture();
        let source =
            AneOutputFfnSource::build(&output, attention, &ffn, &post_gamma, &pre_gamma, 1e-6)
                .unwrap();
        let before = compile_budget_used();
        let program = AneInMemoryProgram::compile_with_cache_policy(
            &source.mil,
            &source.blob,
            source.identity.input_bytes,
            source.identity.output_bytes,
            AneProgramCachePolicy::RequireExisting,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before);
        let mut kernel = program.create_request().unwrap();
        let mut repeated = None;
        for case in [0_usize, 0, 1, 2] {
            let sample = |index: usize| {
                f16::from_f32((((index * 19 + case * 11 + 5) % 53) as f32 - 26.0) * 0.003)
            };
            let attended: Vec<_> = (0..attention).map(sample).collect();
            let residual: Vec<_> = (0..hidden).map(|index| sample(index * 5 + 7)).collect();
            let projected: Vec<_> = output
                .chunks_exact(attention)
                .map(|row| {
                    f16::from_f32(
                        row.iter()
                            .zip(&attended)
                            .map(|(&weight, &value)| weight.to_f32() * value.to_f32())
                            .sum(),
                    )
                })
                .collect();
            let (_, expected) = output_ffn_cpu_boundary_oracle(
                &projected,
                &residual,
                &post_gamma,
                &pre_gamma,
                &ffn,
                1e-6,
            )
            .unwrap();
            let mut input = vec![0_u8; source.identity.input_bytes];
            for (channel, value) in attended.iter().chain(&residual).enumerate() {
                let offset = channel * 64;
                input[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            }
            kernel.write_input(&input).unwrap();
            kernel.evaluate().unwrap();
            let mut bytes = vec![0_u8; source.identity.output_bytes];
            kernel.read_output(&mut bytes).unwrap();
            let actual: Vec<_> = bytes
                .chunks_exact(64)
                .map(|row| f16::from_le_bytes([row[0], row[1]]))
                .collect();
            assert!(actual.iter().all(|value| value.is_finite()));
            assert!(
                actual.windows(2).any(|pair| pair[0] != pair[1]),
                "asymmetric fixture collapsed every output channel"
            );
            let max_absolute_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual.to_f32() - expected.to_f32()).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_absolute_error <= 0.02,
                "case {case}: {max_absolute_error}"
            );
            if case == 0 {
                if let Some(first) = &repeated {
                    assert_eq!(first, &actual, "repeated execution changed output bits");
                }
                repeated = Some(actual.clone());
            }
            eprintln!(
                "{}",
                serde_json::json!({
                    "case": case,
                    "maximum_absolute_error": max_absolute_error,
                    "distinct_output_bits": actual.iter().map(|value| value.to_bits()).collect::<std::collections::BTreeSet<_>>().len(),
                    "output_bits": actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                })
            );
        }
        assert_eq!(compile_budget_used(), before);
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "one actual Gemma 4 sliding-layer-shape compile and zero evaluations"]
    fn hardware_output_ffn_gemma_shape_compile_probe() {
        use crate::ane_linear::{compile_budget_used, AneOutputFfnCompile, AneProgramCachePolicy};

        let (hidden, intermediate, attention) = (3840, 15360, 4096);
        let dense: Vec<_> = (0..hidden * intermediate)
            .map(|index| f16::from_f32(((index * 17 + 3) % 31) as f32 / 512.0 - 0.03))
            .collect();
        let ffn =
            AneInt8FfnWeights::quantize(&dense, &dense, &dense, hidden, intermediate).unwrap();
        drop(dense);
        let output: Vec<_> = (0..hidden * attention)
            .map(|index| f16::from_f32(((index * 19 + 5) % 37) as f32 / 512.0 - 0.035))
            .collect();
        let post_gamma: Vec<_> = (0..hidden)
            .map(|index| f16::from_f32(0.9 + (index % 97) as f32 / 1000.0))
            .collect();
        let pre_gamma: Vec<_> = (0..hidden)
            .map(|index| f16::from_f32(1.05 - (index % 89) as f32 / 1200.0))
            .collect();
        let before = compile_budget_used();
        let compiled = AneOutputFfnCompile::compile_only(
            &output,
            attention,
            &ffn,
            &post_gamma,
            &pre_gamma,
            1e-6,
            AneProgramCachePolicy::Compile,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before + 1);
        assert_eq!(compiled.identity().input_bytes, (hidden + attention) * 64);
        assert_eq!(compiled.identity().output_bytes, hidden * 64);
        println!("identity={:?}", compiled.identity());
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "actual Gemma 4 sliding-layer-shape strict reload; zero compiles and evaluations"]
    fn hardware_output_ffn_gemma_shape_strict_reload_probe() {
        use crate::ane_linear::{compile_budget_used, AneOutputFfnCompile, AneProgramCachePolicy};

        let (hidden, intermediate, attention) = (3840, 15360, 4096);
        let dense: Vec<_> = (0..hidden * intermediate)
            .map(|index| f16::from_f32(((index * 17 + 3) % 31) as f32 / 512.0 - 0.03))
            .collect();
        let ffn =
            AneInt8FfnWeights::quantize(&dense, &dense, &dense, hidden, intermediate).unwrap();
        drop(dense);
        let output: Vec<_> = (0..hidden * attention)
            .map(|index| f16::from_f32(((index * 19 + 5) % 37) as f32 / 512.0 - 0.035))
            .collect();
        let post_gamma: Vec<_> = (0..hidden)
            .map(|index| f16::from_f32(0.9 + (index % 97) as f32 / 1000.0))
            .collect();
        let pre_gamma: Vec<_> = (0..hidden)
            .map(|index| f16::from_f32(1.05 - (index % 89) as f32 / 1200.0))
            .collect();
        let before = compile_budget_used();
        let cached = AneOutputFfnCompile::compile_only(
            &output,
            attention,
            &ffn,
            &post_gamma,
            &pre_gamma,
            1e-6,
            AneProgramCachePolicy::RequireExisting,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before);
        assert_eq!(cached.identity().input_bytes, (hidden + attention) * 64);
        assert_eq!(cached.identity().output_bytes, hidden * 64);
        println!("identity={:?}", cached.identity());
    }

    #[cfg(feature = "macos-private-ane-research")]
    fn compile_dialect_probe(mil: &str, input_channels: usize, output_channels: usize) {
        use crate::ane_linear::{compile_budget_used, AneProgramCachePolicy};
        use rvllm_apple_ane_sys::AneInMemoryProgram;

        let before = compile_budget_used();
        let _program = AneInMemoryProgram::compile_with_cache_policy(
            mil,
            &[],
            input_channels * 64,
            output_channels * 64,
            AneProgramCachePolicy::Compile,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before + 1);
    }

    #[cfg(feature = "macos-private-ane-research")]
    fn compile_blob_probe(mil: &str, blob: &[u8], input_channels: usize, output_channels: usize) {
        use crate::ane_linear::{compile_budget_used, AneProgramCachePolicy};
        use rvllm_apple_ane_sys::AneInMemoryProgram;

        let before = compile_budget_used();
        let _program = AneInMemoryProgram::compile_with_cache_policy(
            mil,
            blob,
            input_channels * 64,
            output_channels * 64,
            AneProgramCachePolicy::Compile,
        )
        .unwrap();
        assert_eq!(compile_budget_used(), before + 1);
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE reduce_sum dialect compile probe"]
    fn hardware_reduce_sum_dialect_compile_probe() {
        compile_dialect_probe(
            r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x) {
        tensor<int32, [3]> axes = const()[val = tensor<int32, [3]>([1, 2, 3])];
        bool keep_dims = const()[val = bool(true)];
        tensor<fp16, [1, 1, 1, 1]> y = reduce_sum(axes = axes, keep_dims = keep_dims, x = x);
    } -> (y);
}
"#,
            32,
            1,
        );
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE rsqrt dialect compile probe"]
    fn hardware_rsqrt_dialect_compile_probe() {
        compile_dialect_probe(
            r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x) {
        tensor<fp16, [1, 32, 1, 1]> y = rsqrt(x = x);
    } -> (y);
}
"#,
            32,
            32,
        );
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE pow inverse-square-root dialect compile probe"]
    fn hardware_pow_negative_half_dialect_compile_probe() {
        compile_dialect_probe(
            r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x) {
        fp16 exponent = const()[val = fp16(-0.5)];
        tensor<fp16, [1, 32, 1, 1]> y = pow(x = x, y = exponent);
    } -> (y);
}
"#,
            32,
            32,
        );
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE mixed output-projection plus INT8 FFN core compile probe"]
    fn hardware_output_ffn_core_without_norm_compile_probe() {
        let hidden = 32;
        let intermediate = 32;
        let attention = 32;
        let dense = vec![f16::from_f32(0.01); hidden * intermediate];
        let ffn =
            AneInt8FfnWeights::quantize(&dense, &dense, &dense, hidden, intermediate).unwrap();
        let output = vec![f16::from_f32(0.02); hidden * attention];
        let gamma = vec![f16::ONE; hidden];
        let (blob, weights) = ffn
            .output_ffn_blob_and_constants(&output, attention, &gamma, &gamma)
            .unwrap();
        let input = hidden + attention;
        let mil = format!(
            r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, {input}, 1, 1]> packed) {{
        tensor<int32, [4]> attended_begin = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [4]> residual_begin = const()[val = tensor<int32, [4]>([0, {attention}, 0, 0])];
        tensor<int32, [4]> attended_size = const()[val = tensor<int32, [4]>([1, {attention}, 1, 1])];
        tensor<int32, [4]> hidden_size = const()[val = tensor<int32, [4]>([1, {hidden}, 1, 1])];
        tensor<fp16, [1, {attention}, 1, 1]> attended = slice_by_size(x = packed, begin = attended_begin, size = attended_size);
        tensor<fp16, [1, {hidden}, 1, 1]> residual = slice_by_size(x = packed, begin = residual_begin, size = hidden_size);
        string valid = const()[val = string("valid")];
        tensor<int32, [2]> ones2 = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> zeros4 = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
        int32 groups = const()[val = int32(1)];
{weights}        fp16 half = const()[val = fp16(0.5)];
        fp16 one = const()[val = fp16(1.0)];
        fp16 cubic = const()[val = fp16(0.044715)];
        fp16 root = const()[val = fp16(0.7978845608)];
        tensor<fp16, [1, {hidden}, 1, 1]> projected = conv(dilations = ones2, groups = groups, pad = zeros4, pad_type = valid, strides = ones2, weight = Wo, x = attended);
        tensor<fp16, [1, {hidden}, 1, 1]> x = add(x = projected, y = residual);
        tensor<fp16, [1, {intermediate}, 1, 1]> gate = conv(dilations = ones2, groups = groups, pad = zeros4, pad_type = valid, strides = ones2, weight = Wg, x = x);
        tensor<fp16, [1, {intermediate}, 1, 1]> up = conv(dilations = ones2, groups = groups, pad = zeros4, pad_type = valid, strides = ones2, weight = Wu, x = x);
        tensor<fp16, [1, {intermediate}, 1, 1]> g2 = mul(x = gate, y = gate);
        tensor<fp16, [1, {intermediate}, 1, 1]> g3 = mul(x = g2, y = gate);
        tensor<fp16, [1, {intermediate}, 1, 1]> cubic_term = mul(x = g3, y = cubic);
        tensor<fp16, [1, {intermediate}, 1, 1]> gelu_inner = add(x = gate, y = cubic_term);
        tensor<fp16, [1, {intermediate}, 1, 1]> ga = mul(x = gelu_inner, y = root);
        tensor<fp16, [1, {intermediate}, 1, 1]> tanh_ga = tanh(x = ga);
        tensor<fp16, [1, {intermediate}, 1, 1]> tanh_plus_one = add(x = tanh_ga, y = one);
        tensor<fp16, [1, {intermediate}, 1, 1]> half_gate = mul(x = gate, y = half);
        tensor<fp16, [1, {intermediate}, 1, 1]> activated = mul(x = half_gate, y = tanh_plus_one);
        tensor<fp16, [1, {intermediate}, 1, 1]> gated = mul(x = activated, y = up);
        tensor<fp16, [1, {hidden}, 1, 1]> y = conv(dilations = ones2, groups = groups, pad = zeros4, pad_type = valid, strides = ones2, weight = Wd, x = gated);
    }} -> (y);
}}
"#
        );
        compile_blob_probe(&mil, &blob, input, hidden);
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE reduce_sum to pow composition compile probe"]
    fn hardware_reduce_sum_pow_compile_probe() {
        compile_dialect_probe(
            r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x) {
        tensor<int32, [3]> axes = const()[val = tensor<int32, [3]>([1, 2, 3])];
        bool keep_dims = const()[val = bool(true)];
        fp16 exponent = const()[val = fp16(-0.5)];
        tensor<fp16, [1, 32, 1, 1]> sq = mul(x = x, y = x);
        tensor<fp16, [1, 1, 1, 1]> sum = reduce_sum(axes = axes, keep_dims = keep_dims, x = sq);
        tensor<fp16, [1, 1, 1, 1]> y = pow(x = sum, y = exponent);
    } -> (y);
}
"#,
            32,
            1,
        );
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE RMS scalar broadcast compile probe"]
    fn hardware_rms_scalar_broadcast_compile_probe() {
        compile_dialect_probe(
            r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x) {
        tensor<int32, [3]> axes = const()[val = tensor<int32, [3]>([1, 2, 3])];
        bool keep_dims = const()[val = bool(true)];
        fp16 exponent = const()[val = fp16(-0.5)];
        tensor<fp16, [1, 32, 1, 1]> sq = mul(x = x, y = x);
        tensor<fp16, [1, 1, 1, 1]> sum = reduce_sum(axes = axes, keep_dims = keep_dims, x = sq);
        tensor<fp16, [1, 1, 1, 1]> inv = pow(x = sum, y = exponent);
        tensor<fp16, [1, 32, 1, 1]> y = mul(x = x, y = inv);
    } -> (y);
}
"#,
            32,
            32,
        );
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE two sequential RMS blocks compile probe"]
    fn hardware_two_rms_blocks_compile_probe() {
        compile_dialect_probe(
            r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x) {
        tensor<int32, [3]> axes = const()[val = tensor<int32, [3]>([1, 2, 3])];
        bool keep_dims = const()[val = bool(true)];
        fp16 exponent = const()[val = fp16(-0.5)];
        fp16 mean = const()[val = fp16(0.03125)];
        fp16 epsilon = const()[val = fp16(0.000001)];
        tensor<fp16, [1, 32, 1, 1]> sq0 = mul(x = x, y = x);
        tensor<fp16, [1, 1, 1, 1]> sum0 = reduce_sum(axes = axes, keep_dims = keep_dims, x = sq0);
        tensor<fp16, [1, 1, 1, 1]> mean0 = mul(x = sum0, y = mean);
        tensor<fp16, [1, 1, 1, 1]> variance0 = add(x = mean0, y = epsilon);
        tensor<fp16, [1, 1, 1, 1]> inv0 = pow(x = variance0, y = exponent);
        tensor<fp16, [1, 32, 1, 1]> normalized0 = mul(x = x, y = inv0);
        tensor<fp16, [1, 32, 1, 1]> sq1 = mul(x = normalized0, y = normalized0);
        tensor<fp16, [1, 1, 1, 1]> sum1 = reduce_sum(axes = axes, keep_dims = keep_dims, x = sq1);
        tensor<fp16, [1, 1, 1, 1]> mean1 = mul(x = sum1, y = mean);
        tensor<fp16, [1, 1, 1, 1]> variance1 = add(x = mean1, y = epsilon);
        tensor<fp16, [1, 1, 1, 1]> inv1 = pow(x = variance1, y = exponent);
        tensor<fp16, [1, 32, 1, 1]> y = mul(x = normalized0, y = inv1);
    } -> (y);
}
"#,
            32,
            32,
        );
    }

    #[cfg(feature = "macos-private-ane-research")]
    #[test]
    #[ignore = "bounded private-ANE dual RMS plus learned gamma compile probe"]
    fn hardware_two_rms_gamma_blocks_compile_probe() {
        compile_dialect_probe(
            r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 32, 1, 1]> x) {
        tensor<int32, [3]> axes = const()[val = tensor<int32, [3]>([1, 2, 3])];
        bool keep_dims = const()[val = bool(true)];
        fp16 exponent = const()[val = fp16(-0.5)];
        fp16 mean = const()[val = fp16(0.03125)];
        fp16 epsilon = const()[val = fp16(0.000001)];
        tensor<fp16, [1, 32, 1, 1]> gamma0 = const()[val = tensor<fp16, [1, 32, 1, 1]>([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])];
        tensor<fp16, [1, 32, 1, 1]> gamma1 = const()[val = tensor<fp16, [1, 32, 1, 1]>([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])];
        tensor<fp16, [1, 32, 1, 1]> sq0 = mul(x = x, y = x);
        tensor<fp16, [1, 1, 1, 1]> sum0 = reduce_sum(axes = axes, keep_dims = keep_dims, x = sq0);
        tensor<fp16, [1, 1, 1, 1]> mean0 = mul(x = sum0, y = mean);
        tensor<fp16, [1, 1, 1, 1]> variance0 = add(x = mean0, y = epsilon);
        tensor<fp16, [1, 1, 1, 1]> inv0 = pow(x = variance0, y = exponent);
        tensor<fp16, [1, 32, 1, 1]> unit0 = mul(x = x, y = inv0);
        tensor<fp16, [1, 32, 1, 1]> normalized0 = mul(x = unit0, y = gamma0);
        tensor<fp16, [1, 32, 1, 1]> sq1 = mul(x = normalized0, y = normalized0);
        tensor<fp16, [1, 1, 1, 1]> sum1 = reduce_sum(axes = axes, keep_dims = keep_dims, x = sq1);
        tensor<fp16, [1, 1, 1, 1]> mean1 = mul(x = sum1, y = mean);
        tensor<fp16, [1, 1, 1, 1]> variance1 = add(x = mean1, y = epsilon);
        tensor<fp16, [1, 1, 1, 1]> inv1 = pow(x = variance1, y = exponent);
        tensor<fp16, [1, 32, 1, 1]> unit1 = mul(x = normalized0, y = inv1);
        tensor<fp16, [1, 32, 1, 1]> y = mul(x = unit1, y = gamma1);
    } -> (y);
}
"#,
            32,
            32,
        );
    }
}
