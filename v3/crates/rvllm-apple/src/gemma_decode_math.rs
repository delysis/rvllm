//! Small host-side operations around ANE projections. These functions express
//! Gemma 4's FP16 tensor boundaries explicitly; they do not emulate BF16 or
//! establish full-model equivalence to the checkpoint's BF16 execution.

use half::f16;

/// Normalize independent heads using FP32 statistics and the actual learned
/// gamma (Gemma 4 does not use 1 + gamma). None is Gemma's unit-scale V norm.
pub fn rms_norm_f16_in_place(
    values: &mut [f16],
    width: usize,
    gamma: Option<&[f16]>,
    epsilon: f32,
) -> Result<(), String> {
    if width == 0
        || values.is_empty()
        || values.len() % width != 0
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || gamma.is_some_and(|g| g.len() != width || g.iter().any(|v| !v.is_finite()))
        || values.iter().any(|v| !v.is_finite())
    {
        return Err("Gemma RMSNorm shape, epsilon, or finite-value requirement failed".into());
    }
    for head in values.chunks_exact_mut(width) {
        let squared_sum: f32 = head
            .iter()
            .map(|v| {
                let x = v.to_f32();
                x * x
            })
            .sum();
        let inverse_rms = (squared_sum / width as f32 + epsilon).powf(-0.5);
        for (i, value) in head.iter_mut().enumerate() {
            let scale = gamma.map_or(1.0, |g| g[i].to_f32());
            *value = f16::from_f32(value.to_f32() * inverse_rms * scale);
            if !value.is_finite() {
                return Err("Gemma RMSNorm output exceeds FP16 range".into());
            }
        }
    }
    Ok(())
}

/// Precomputed frequencies and reusable one-position trig values. Proportional
/// RoPE rotates selected pairs across the *full* head's midpoint, and keeps the
/// full head dimension in the frequency exponent even when only 25% rotates.
pub struct GemmaRope {
    head_dim: usize,
    frequencies: Vec<f32>,
    cos: Vec<f16>,
    sin: Vec<f16>,
    position: Option<u32>,
}

impl GemmaRope {
    pub fn new(head_dim: usize, rotary_dim: usize, theta: f32) -> Result<Self, String> {
        if head_dim == 0
            || head_dim > 65536
            || head_dim % 2 != 0
            || rotary_dim == 0
            || rotary_dim % 2 != 0
            || rotary_dim > head_dim
            || !theta.is_finite()
            || theta <= 1.0
        {
            return Err("Gemma RoPE dimensions or theta invalid".into());
        }
        let pairs = rotary_dim / 2;
        let frequencies = (0..pairs)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        Ok(Self {
            head_dim,
            frequencies,
            cos: vec![f16::ZERO; pairs],
            sin: vec![f16::ZERO; pairs],
            position: None,
        })
    }

    pub fn apply_f16(&mut self, values: &mut [f16], position: u32) -> Result<(), String> {
        if values.is_empty()
            || values.len() % self.head_dim != 0
            || values.iter().any(|v| !v.is_finite())
        {
            return Err("Gemma RoPE requires finite complete heads".into());
        }
        if self.position != Some(position) {
            for ((&frequency, cos), sin) in self
                .frequencies
                .iter()
                .zip(&mut self.cos)
                .zip(&mut self.sin)
            {
                let angle = position as f32 * frequency;
                *cos = f16::from_f32(angle.cos());
                *sin = f16::from_f32(angle.sin());
            }
            self.position = Some(position);
        }
        let midpoint = self.head_dim / 2;
        for head in values.chunks_exact_mut(self.head_dim) {
            for i in 0..self.frequencies.len() {
                let left = head[i].to_f32();
                let right = head[midpoint + i].to_f32();
                let cos = self.cos[i].to_f32();
                let sin = self.sin[i].to_f32();
                // HF multiplies tensors before adding, so each product has an
                // FP16 boundary. A fused FP32 FMA changes that behavior.
                let lc = f16::from_f32(left * cos).to_f32();
                let rs = f16::from_f32(right * sin).to_f32();
                let rc = f16::from_f32(right * cos).to_f32();
                let ls = f16::from_f32(left * sin).to_f32();
                head[i] = f16::from_f32(lc - rs);
                head[midpoint + i] = f16::from_f32(rc + ls);
                if !head[i].is_finite() || !head[midpoint + i].is_finite() {
                    return Err("Gemma RoPE output exceeds FP16 range".into());
                }
            }
        }
        Ok(())
    }
}

/// Each residual sum is materialized in FP16 before a later norm or layer scale.
pub fn add_residual_f16(residual: &mut [f16], branch: &[f16]) -> Result<(), String> {
    if residual.len() != branch.len() || residual.is_empty() {
        return Err("Gemma residual shape mismatch".into());
    }
    for (value, branch) in residual.iter_mut().zip(branch) {
        *value = f16::from_f32(value.to_f32() + branch.to_f32());
        if !value.is_finite() {
            return Err("Gemma residual is nonfinite or exceeds FP16 range".into());
        }
    }
    Ok(())
}

/// Applied once, after both attention and FFN residual additions in a layer.
pub fn scale_layer_f16(values: &mut [f16], scalar: f32) -> Result<(), String> {
    if values.is_empty() || !scalar.is_finite() {
        return Err("Gemma layer scale invalid".into());
    }
    for value in values {
        *value = f16::from_f32(value.to_f32() * scalar);
        if !value.is_finite() {
            return Err("Gemma layer scale output exceeds FP16 range".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(value: &serde_json::Value) -> Vec<f16> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f16::from_bits(v.as_u64().unwrap() as u16))
            .collect()
    }

    #[test]
    fn norms_and_rotary_match_pinned_transformers_fp16() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/reference/gemma4-fp16-norm-rope.json"
        ))
        .unwrap();
        for case in reference["fixtures"].as_array().unwrap() {
            let dim = case["head_dim"].as_u64().unwrap() as usize;
            let rotary_dim = case["rotary_dim"].as_u64().unwrap() as usize;
            let position = case["position"].as_u64().unwrap() as u32;
            let mut values = tensor(&case["input"]);
            let gamma = tensor(&case["gamma"]);
            rms_norm_f16_in_place(&mut values, dim, Some(&gamma), 1e-6).unwrap();
            for (&actual, expected) in values.iter().zip(tensor(&case["normalized"])) {
                assert!(
                    (actual.to_f32() - expected.to_f32()).abs()
                        <= 0.001 * expected.to_f32().abs().max(1.0)
                );
            }
            let normalized = values.clone();
            let mut rope =
                GemmaRope::new(dim, rotary_dim, case["theta"].as_f64().unwrap() as f32).unwrap();
            rope.apply_f16(&mut values, position).unwrap();
            for (index, (&actual, expected)) in
                values.iter().zip(tensor(&case["rotated"])).enumerate()
            {
                assert!((actual.to_f32() - expected.to_f32()).abs() <= 0.002 * expected.to_f32().abs().max(1.0),
                    "dim={dim} position={position} index={index} actual={actual} expected={expected}");
                let within_half = index % (dim / 2);
                if within_half >= rotary_dim / 2 {
                    assert_eq!(actual, normalized[index]);
                }
            }
        }
        let unit = &reference["unit_norm"];
        let mut values = tensor(&unit["input"]);
        rms_norm_f16_in_place(&mut values, 8, None, 1e-6).unwrap();
        assert_eq!(values, tensor(&unit["output"]));
    }

    #[test]
    fn malformed_inputs_and_fp16_overflow_fail() {
        assert!(GemmaRope::new(512, 513, 1e6).is_err());
        assert!(GemmaRope::new(512, 127, 1e6).is_err());
        assert!(GemmaRope::new(512, 128, f32::NAN).is_err());
        assert!(rms_norm_f16_in_place(&mut [f16::ONE; 8], 3, None, 1e-6).is_err());
        assert!(rms_norm_f16_in_place(&mut [f16::INFINITY; 8], 8, None, 1e-6).is_err());
        assert!(rms_norm_f16_in_place(&mut [f16::ONE; 8], 8, None, 0.0).is_err());
        assert!(add_residual_f16(&mut [f16::MAX], &[f16::MAX]).is_err());
        assert!(scale_layer_f16(&mut [f16::MAX], 2.0).is_err());
    }
}
