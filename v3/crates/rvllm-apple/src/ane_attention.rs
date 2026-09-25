//! Experimental single-input ANE decode attention for qualified Gemma 4 shapes.
//!
//! The replaced four-input program triggered an AppleH16ANEInterface kernel
//! panic. This replacement follows the upstream single-input contract, but it
//! passed shared-request and full sliding-window numerical checks after scaling
//! the probability/value product. Other geometries remain blocked before any
//! private API call. See reports/ane-panic-20260914.md for the evidence boundary.

use crate::ane_attention_layout::PackedAttentionLayout;
use crate::ane_linear::fp16_linear_weight_blob;
use half::f16;
use rvllm_apple_ane_sys::{AneInMemoryKernel, AneInMemoryProgram, AneProgramCachePolicy};
use sha2::{Digest, Sha256};

// No runtime override: a new shape requires an explicit qualification change.
fn qualified_layout(layout: PackedAttentionLayout) -> bool {
    layout.query_heads() == 16
        && matches!(
            (
                layout.kv_heads(),
                layout.head_dim(),
                layout.capacity(),
                layout.window()
            ),
            (8, 256, 1024, Some(1024)) | (1, 512, 64 | 1024, None)
        )
}

/// One attention graph per head geometry and context capacity. Each layer
/// creates an independent request, so sharing code never shares KV state.
pub struct AneAttentionProgram {
    layout: PackedAttentionLayout,
    program: AneInMemoryProgram,
}

/// Default-off program owner for one layer's fused attention and output
/// projection. Production routing remains unchanged until the component and
/// full-route gates qualify this exact graph.
pub struct AneAttentionOutputCompile {
    layout: PackedAttentionLayout,
    output_channels: usize,
    program: AneInMemoryProgram,
}

pub struct AneAttentionOutput {
    kernel: AneInMemoryKernel,
    input_bytes: usize,
    output_channels: usize,
    output: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AneAttentionOutputIdentity {
    pub mil_sha256: String,
    pub weight_blob_sha256: String,
    pub input_bytes: usize,
    pub output_bytes: usize,
}

impl AneAttentionOutputCompile {
    pub fn source_identity(
        layout: PackedAttentionLayout,
        output_weights: &[f16],
        output_channels: usize,
    ) -> Result<AneAttentionOutputIdentity, String> {
        validate_fused_shape(layout, output_weights, output_channels)?;
        let blob = fp16_linear_weight_blob(output_weights)?;
        let mil = layout.mil_with_output_projection(output_channels, 64)?;
        Ok(AneAttentionOutputIdentity {
            mil_sha256: hex_sha256(mil.as_bytes()),
            weight_blob_sha256: hex_sha256(&blob),
            input_bytes: layout.input_bytes(),
            output_bytes: layout.projected_output_bytes(output_channels)?,
        })
    }

    pub fn compile_layer(
        layout: PackedAttentionLayout,
        output_weights: &[f16],
        output_channels: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        validate_fused_shape(layout, output_weights, output_channels)?;
        let blob = fp16_linear_weight_blob(output_weights)?;
        let mil = layout.mil_with_output_projection(output_channels, 64)?;
        let program = AneInMemoryProgram::compile_with_cache_policy(
            &mil,
            &blob,
            layout.input_bytes(),
            layout.projected_output_bytes(output_channels)?,
            policy,
        )?;
        Ok(Self {
            layout,
            output_channels,
            program,
        })
    }

    pub fn create_request(&self) -> Result<AneAttentionOutput, String> {
        Ok(AneAttentionOutput {
            kernel: self.program.create_request()?,
            input_bytes: self.layout.input_bytes(),
            output_channels: self.output_channels,
            output: vec![0; self.layout.projected_output_bytes(self.output_channels)?],
        })
    }
}

impl AneAttentionOutput {
    /// Evaluate one completely packed attention surface. The sole logical
    /// output column is decoded from ANE's 64-byte channel rows.
    pub fn evaluate_packed(&mut self, input: &[u8], output: &mut [f16]) -> Result<(), String> {
        if input.len() != self.input_bytes || output.len() != self.output_channels {
            return Err("fused attention/output projection I/O shape mismatch".into());
        }
        self.kernel.write_input(input)?;
        self.kernel.evaluate()?;
        self.kernel.read_output(&mut self.output)?;
        for (channel, value) in output.iter_mut().enumerate() {
            let offset = channel * 64;
            *value = f16::from_le_bytes([self.output[offset], self.output[offset + 1]]);
        }
        Ok(())
    }
}

fn validate_fused_shape(
    layout: PackedAttentionLayout,
    output_weights: &[f16],
    output_channels: usize,
) -> Result<(), String> {
    if !qualified_layout(layout)
        || output_weights.len() != output_channels.saturating_mul(layout.query_width())
    {
        return Err(
            "fused attention output projection shape is outside the qualified single-I/O boundary"
                .into(),
        );
    }
    Ok(())
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub struct AneAttention {
    kernel: AneInMemoryKernel,
    layout: PackedAttentionLayout,
    tokens_seen: usize,
    query: Vec<u8>,
    key: Vec<u8>,
    value: Vec<u8>,
    mask: Vec<u8>,
    output: Vec<u8>,
}

impl AneAttentionProgram {
    pub fn compile(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self, String> {
        Self::compile_with_cache_policy(
            query_heads,
            kv_heads,
            head_dim,
            capacity,
            AneProgramCachePolicy::Compile,
        )
    }

    pub fn compile_with_cache_policy(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let layout = PackedAttentionLayout::new(query_heads, kv_heads, head_dim, capacity)?;
        Self::compile_layout(layout, policy)
    }

    pub fn compile_sliding(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: usize,
    ) -> Result<Self, String> {
        Self::compile_sliding_with_cache_policy(
            query_heads,
            kv_heads,
            head_dim,
            window,
            AneProgramCachePolicy::Compile,
        )
    }

    pub fn compile_sliding_with_cache_policy(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        Self::compile_layout(
            PackedAttentionLayout::sliding(query_heads, kv_heads, head_dim, window)?,
            policy,
        )
    }

    fn compile_layout(
        layout: PackedAttentionLayout,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        if !qualified_layout(layout) {
            return Err("ANE attention shape is unqualified; only Gemma 4 12B sliding-1024 and global-64/global-1024 have shared-request hardware qualification; see v3/reports/ane-panic-20260914.md".into());
        }
        let program = AneInMemoryProgram::compile_with_cache_policy(
            &layout.mil(),
            &[],
            layout.input_bytes(),
            layout.output_bytes(),
            policy,
        )?;
        Ok(Self { layout, program })
    }

    /// Explicit graph permutation candidate; never changes the baseline source
    /// or single-I/O sizes. Shared program, separate KV surfaces per layer.
    pub fn compile_transpose_flags_with_cache_policy(
        layout: PackedAttentionLayout,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        if !qualified_layout(layout) {
            return Err("transpose attention shape is outside the single-I/O boundary".into());
        }
        let mil = layout.mil_transpose_flags()?;
        tracing::debug!(
            candidate = "ane-attention-transpose-flags",
            input_bytes = layout.input_bytes(),
            output_bytes = layout.output_bytes(),
            "ANE graph permutation candidate; compiler scheduling unmeasured"
        );
        let program = AneInMemoryProgram::compile_with_cache_policy(
            &mil,
            &[],
            layout.input_bytes(),
            layout.output_bytes(),
            policy,
        )?;
        Ok(Self { layout, program })
    }

    pub fn create_request(&self) -> Result<AneAttention, String> {
        let layout = self.layout;
        let mut kernel = self.program.create_request()?;
        kernel.write_input(&layout.import_cache(&[], &[], 0)?)?;
        Ok(AneAttention {
            kernel,
            layout,
            tokens_seen: 0,
            query: vec![0; layout.query_width() * 2],
            key: vec![0; layout.kv_width() * 2],
            value: vec![0; layout.kv_width() * 2],
            mask: vec![0; layout.capacity() * 2],
            output: vec![0; layout.output_bytes()],
        })
    }
}

impl AneAttention {
    pub fn compile(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self, String> {
        AneAttentionProgram::compile(query_heads, kv_heads, head_dim, capacity)?.create_request()
    }

    pub fn compile_sliding(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: usize,
    ) -> Result<Self, String> {
        AneAttentionProgram::compile_sliding(query_heads, kv_heads, head_dim, window)?
            .create_request()
    }

    /// Import K/V in [token, KV head, dimension] order after Metal completion.
    pub fn import_cache(
        &mut self,
        keys: &[f16],
        values: &[f16],
        tokens: usize,
    ) -> Result<(), String> {
        let packed = self.layout.import_cache(keys, values, tokens)?;
        self.kernel.write_input(&packed)?;
        self.tokens_seen = tokens;
        Ok(())
    }

    /// Experimental import using caller-owned scratch shared across layers.
    /// The vector only grows; smaller layouts use a fully cleared prefix.
    /// The resident surface and frontier change only after packing succeeds.
    pub fn import_cache_with_scratch(
        &mut self,
        keys: &[f16],
        values: &[f16],
        tokens: usize,
        scratch: &mut Vec<u8>,
    ) -> Result<(), String> {
        let bytes = self.layout.input_bytes();
        if scratch.len() < bytes {
            scratch.resize(bytes, 0);
        }
        let packed = &mut scratch[..bytes];
        self.layout
            .import_cache_into(keys, values, tokens, packed)?;
        self.kernel.write_input(packed)?;
        self.tokens_seen = tokens;
        Ok(())
    }

    /// Separately measurable CPU transpose; reuses the same scratch and the
    /// same synchronous surface write/frontier publication as the old candidate.
    pub fn import_cache_blocked32_with_scratch(
        &mut self,
        keys: &[f16],
        values: &[f16],
        tokens: usize,
        scratch: &mut Vec<u8>,
    ) -> Result<(), String> {
        self.layout.retained_tokens(tokens)?;
        if tokens.checked_mul(self.layout.kv_width()) != Some(keys.len())
            || values.len() != keys.len()
        {
            return Err("packed attention prefill shape mismatch".into());
        }
        let bytes = self.layout.input_bytes();
        if scratch.len() < bytes {
            scratch
                .try_reserve(bytes - scratch.len())
                .map_err(|error| format!("KV scratch allocation: {error}"))?;
            scratch.resize(bytes, 0);
        }
        let packed = &mut scratch[..bytes];
        let routed = self
            .layout
            .import_cache_blocked32_into(keys, values, tokens, packed)?;
        tracing::debug!(
            candidate = if routed {
                "cpu-kv-blocked32"
            } else {
                "reuse-scratch-fallback"
            },
            tokens,
            bytes,
            "CPU KV pack complete; surface write follows"
        );
        self.kernel.write_input(packed)?;
        self.tokens_seen = tokens;
        Ok(())
    }

    /// Append one position. Query and key must already be normalized/rotated;
    /// value must be normalized. All attention math is in the ANE program.
    /// The existing KV bytes stay on the surface; only the new slot is written.
    /// The caller must use `tokens_seen()` as the absolute RoPE position, even
    /// after a sliding cache wraps. The window is fixed at construction.
    pub fn decode(
        &mut self,
        query: &[f16],
        key: &[f16],
        value: &[f16],
        output: &mut [f16],
    ) -> Result<(), String> {
        if key.len() != self.layout.kv_width()
            || value.len() != key.len()
            || output.len() != self.layout.query_width()
        {
            return Err("ANE attention decode shape mismatch".into());
        }
        let position = self.tokens_seen;
        let next_tokens = position.checked_add(1).ok_or("ANE token count overflow")?;
        let key_offset = self
            .layout
            .key_offset(position)
            .ok_or("ANE attention capacity exceeded")?;
        let value_offset = self
            .layout
            .value_offset(position)
            .ok_or("ANE attention capacity exceeded")?;
        self.layout.encode_query(query, &mut self.query)?;
        self.layout.encode_mask(next_tokens, &mut self.mask)?;
        encode(key, &mut self.key);
        encode(value, &mut self.value);
        let stride = self.layout.row_bytes();
        self.kernel
            .write_tensor_strided(0, 0, stride, self.layout.groups() * 2, &self.query)?;
        self.kernel
            .write_tensor_strided(0, key_offset, stride, 2, &self.key)?;
        self.kernel
            .write_tensor_strided(0, value_offset, stride, 2, &self.value)?;
        self.kernel.write_tensor_strided(
            0,
            self.layout.mask_offset(),
            self.mask.len(),
            self.mask.len(),
            &self.mask,
        )?;
        self.kernel.evaluate()?;
        self.kernel.read_output(&mut self.output)?;
        self.layout.decode_output(&self.output, output)?;
        self.tokens_seen = next_tokens;
        Ok(())
    }

    pub fn tokens_seen(&self) -> usize {
        self.tokens_seen
    }

    /// Diagnostic transaction support: discard exactly one unwrapped append.
    /// No graph is evaluated or compiled. A failed surface write requires the
    /// owner to discard this state and perform a complete successful import.
    pub fn discard_last_unwrapped(&mut self, retained: usize) -> Result<(), String> {
        let (key_offset, value_offset) = self
            .layout
            .unwrapped_tail_offsets(self.tokens_seen, retained)?;
        // Rebuild the mask from the frontier. import_cache does not synchronize
        // the host scratch mask, so it must never be treated as saved state.
        self.layout.encode_mask(retained, &mut self.mask)?;
        self.key.fill(0);
        self.value.fill(0);
        let stride = self.layout.row_bytes();
        self.kernel
            .write_tensor_strided(0, key_offset, stride, 2, &self.key)?;
        self.kernel
            .write_tensor_strided(0, value_offset, stride, 2, &self.value)?;
        self.kernel.write_tensor_strided(
            0,
            self.layout.mask_offset(),
            self.mask.len(),
            self.mask.len(),
            &self.mask,
        )?;
        self.tokens_seen = retained;
        Ok(())
    }
}

fn encode(values: &[f16], bytes: &mut [u8]) {
    for (value, slot) in values.iter().zip(bytes.chunks_exact_mut(2)) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unqualified_attention_is_rejected_before_private_api_access() {
        let result = AneAttention::compile(16, 8, 256, 64);
        assert!(matches!(result, Err(error) if error.contains("unqualified")));
        let result = AneAttention::compile(16, 1, 512, 2048);
        assert!(matches!(result, Err(error) if error.contains("unqualified")));
        let result = AneAttention::compile_sliding(16, 8, 256, 31);
        assert!(matches!(result, Err(error) if error.contains("unqualified")));
    }

    #[test]
    #[ignore = "qualification only: one single-input global-1024 graph, two requests, sixteen evaluations"]
    fn hardware_qualifies_global_1024_request_isolation_and_capacity() {
        // Keep this bounded fixture independent of model construction. Its
        // recorded first pass qualified this exact shape; it never bypasses
        // the sys crate's single-input/output quarantine.
        let layout = PackedAttentionLayout::new(16, 1, 512, 1024).unwrap();
        let program = AneAttentionProgram {
            layout,
            program: AneInMemoryProgram::compile_with_cache_policy(
                &layout.mil(),
                &[],
                layout.input_bytes(),
                layout.output_bytes(),
                AneProgramCachePolicy::ReuseOrCompileUpTo(1),
            )
            .unwrap(),
        };
        let mut requests = [
            program.create_request().unwrap(),
            program.create_request().unwrap(),
        ];
        let sample = |i: usize, salt: usize| {
            f16::from_f32(((i * 17 + salt * 113) % 197) as f32 / 256.0 - 0.375)
        };
        let mut maximum_error = 0.0_f32;
        for imported in [5, 63, 511, 1022] {
            let mut caches = Vec::new();
            for (request, salt) in requests.iter_mut().zip([3, 19]) {
                let keys: Vec<_> = (0..imported * 512).map(|i| sample(i, salt)).collect();
                let values: Vec<_> = (0..keys.len()).map(|i| sample(i * 3, salt + 7)).collect();
                request.import_cache(&keys, &values, imported).unwrap();
                caches.push((keys, values));
            }
            for step in 0..2 {
                for index in [1, 0] {
                    let request = &mut requests[index];
                    let (keys, values) = &mut caches[index];
                    let q: Vec<_> = (0..8192)
                        .map(|i| sample(i + step * 13, index * 11 + 5))
                        .collect();
                    let k: Vec<_> = (0..512)
                        .map(|i| sample(i + step * 7, index * 11 + 3))
                        .collect();
                    let v: Vec<_> = (0..512)
                        .map(|i| sample(i + step * 19, index * 11 + 17))
                        .collect();
                    let mut actual = vec![f16::ZERO; 8192];
                    request.decode(&q, &k, &v, &mut actual).unwrap();
                    keys.extend(k);
                    values.extend(v);
                    let expected = reference(layout, &q, keys, values);
                    for (got, expected) in actual.iter().zip(expected) {
                        let error = (got.to_f32() - expected).abs();
                        assert!(got.is_finite() && error < 0.003, "global-1024 imported={imported} step={step} request={index}: {} vs {expected}", got.to_f32());
                        maximum_error = maximum_error.max(error);
                    }
                    assert_eq!(request.tokens_seen(), imported + step + 1);
                }
            }
        }
        for request in &mut requests {
            assert_eq!(request.tokens_seen(), 1024);
            assert!(request
                .decode(
                    &vec![f16::ZERO; 8192],
                    &vec![f16::ZERO; 512],
                    &vec![f16::ZERO; 512],
                    &mut vec![f16::ZERO; 8192]
                )
                .is_err());
            assert_eq!(request.tokens_seen(), 1024);
        }
        println!("global-1024 maximum absolute error: {maximum_error}");
    }

    fn reference(
        layout: PackedAttentionLayout,
        query: &[f16],
        keys: &[f16],
        values: &[f16],
    ) -> Vec<f32> {
        let tokens = keys.len() / layout.kv_width();
        let first = layout.window().map_or(0, |n| tokens.saturating_sub(n));
        let dim = layout.head_dim();
        let kv = layout.kv_heads();
        let mut output = vec![0.0; layout.query_width()];
        for head in 0..layout.query_heads() {
            let kv_head = head / layout.groups();
            let scores: Vec<f32> = (first..tokens)
                .map(|token| {
                    (0..dim)
                        .map(|d| {
                            query[head * dim + d].to_f32()
                                * keys[(token * kv + kv_head) * dim + d].to_f32()
                        })
                        .sum()
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|s| (s - max).exp()).sum();
            for (t, score) in scores.iter().enumerate() {
                let probability = (score - max).exp() / denominator;
                for d in 0..dim {
                    output[head * dim + d] +=
                        probability * values[((first + t) * kv + kv_head) * dim + d].to_f32();
                }
            }
        }
        output
    }

    #[test]
    #[ignore = "diagnostic only: one full-window probability graph, one evaluation"]
    fn hardware_reports_attention_probability_sums() {
        let layout = PackedAttentionLayout::sliding(16, 8, 256, 1024).unwrap();
        assert!(qualified_layout(layout));
        let sample = |i: usize| ((i * 17 + 3) % 257) as f32 / 512.0 - 0.25;
        let keys: Vec<_> = (0..1024 * layout.kv_width())
            .map(|i| f16::from_f32(sample(i + 19)))
            .collect();
        let values: Vec<_> = (0..keys.len())
            .map(|i| f16::from_f32(sample(i * 3 + 7) + 0.125))
            .collect();
        let query: Vec<_> = (0..layout.query_width())
            .map(|i| f16::from_f32(sample(i)))
            .collect();
        let mut packed = layout.import_cache(&keys, &values, 1024).unwrap();
        let mut encoded = vec![0; query.len() * 2];
        layout.encode_query(&query, &mut encoded).unwrap();
        for (row, data) in packed
            .chunks_exact_mut(layout.row_bytes())
            .zip(encoded.chunks_exact(layout.groups() * 2))
        {
            row[..data.len()].copy_from_slice(data);
        }
        let original = "tensor<fp16, [1, 2048, 1, 2]> y = reshape(x = transposed, shape = sq)[name = string(\"y\")];";
        let replacement = "tensor<int32, [4]> diagnostic_shape = const()[name = string(\"diagnostic_shape\"), val = tensor<int32, [4]>([1, 16, 1, 1024])];\n        tensor<fp16, [1, 16, 1, 1024]> y = reshape(x = probabilities, shape = diagnostic_shape)[name = string(\"y\")];";
        let mil = layout.mil();
        assert_eq!(mil.matches(original).count(), 1);
        let expected = reference(layout, &query, &keys, &values);
        let mut kernel = AneInMemoryKernel::compile(
            &mil.replace(original, replacement),
            &[],
            layout.input_bytes(),
            16 * 1024 * 2,
        )
        .unwrap();
        kernel.write_input(&packed).unwrap();
        kernel.evaluate().unwrap();
        let mut bytes = vec![0; 16 * 1024 * 2];
        kernel.read_output(&mut bytes).unwrap();
        let probabilities: Vec<_> = bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert!(probabilities
            .iter()
            .all(|p| p.is_finite() && *p >= f16::ZERO));
        for (head, probabilities) in probabilities.chunks_exact(1024).enumerate() {
            let mut wide = 0.0_f32;
            let mut product_f16 = 0.0_f32;
            let mut product_flush_subnormal = 0.0_f32;
            for (t, p) in probabilities.iter().enumerate() {
                let v = values[(t * 8 + head / 2) * 256 + 87].to_f32();
                wide += p.to_f32() * v;
                let product = f16::from_f32(p.to_f32() * v).to_f32();
                product_f16 += product;
                if product.abs() >= f16::MIN_POSITIVE.to_f32() {
                    product_flush_subnormal += product;
                }
            }
            eprintln!(
                "{}",
                serde_json::json!({
                    "head": head, "probability_sum": probabilities.iter().map(|p| p.to_f32()).sum::<f32>(),
                    "value_dim": 87, "fp32_reference": expected[head * 256 + 87],
                    "ane_probabilities_wide_pv": wide, "product_f16_pv": product_f16,
                    "product_flush_subnormal_pv": product_flush_subnormal,
                })
            );
        }
    }

    #[test]
    #[ignore = "executes private ANE API; two graphs, four requests, fourteen evaluations"]
    fn hardware_shared_attention_preserves_caches_across_window_wrap() {
        for (layout, imported) in [
            (
                PackedAttentionLayout::sliding(16, 8, 256, 1024).unwrap(),
                1023,
            ),
            (PackedAttentionLayout::new(16, 1, 512, 64).unwrap(), 60),
        ] {
            let sample = |i: usize| ((i * 17 + 3) % 257) as f32 / 512.0 - 0.25;
            let keys: Vec<Vec<_>> = (0..2)
                .map(|request| {
                    (0..(imported + 4) * layout.kv_width())
                        .map(|i| f16::from_f32(sample(i + request * 19)))
                        .collect()
                })
                .collect();
            let values: Vec<Vec<_>> = (0..2)
                .map(|request| {
                    (0..keys[request].len())
                        .map(|i| f16::from_f32(sample(i * 3 + 7) + request as f32 * 0.125))
                        .collect()
                })
                .collect();
            let queries: Vec<Vec<_>> = (0..4)
                .map(|step| {
                    (0..layout.query_width())
                        .map(|i| f16::from_f32(sample(i + step * 11)))
                        .collect()
                })
                .collect();
            // Prepare and check the independent FP32 oracle before compiling.
            // Distinct values make accidental request/cache aliasing observable.
            let expected: Vec<Vec<_>> = (0..2)
                .map(|request| {
                    (0..4)
                        .map(|step| {
                            let end = (imported + step + 1) * layout.kv_width();
                            reference(
                                layout,
                                &queries[step],
                                &keys[request][..end],
                                &values[request][..end],
                            )
                        })
                        .collect()
                })
                .collect();
            assert!(expected.iter().flatten().flatten().all(|v| v.is_finite()));
            assert!(expected[0][0]
                .iter()
                .zip(&expected[1][0])
                .any(|(a, b)| (a - b).abs() > 0.05));
            let program =
                AneAttentionProgram::compile_layout(layout, AneProgramCachePolicy::Compile)
                    .unwrap();
            let mut first = program.create_request().unwrap();
            let mut second = program.create_request().unwrap();
            let end = imported * layout.kv_width();
            first
                .import_cache(&keys[0][..end], &values[0][..end], imported)
                .unwrap();
            second
                .import_cache(&keys[1][..end], &values[1][..end], imported)
                .unwrap();
            drop(program);
            let mut actual = vec![f16::ZERO; layout.query_width()];
            let mut evaluate = |attention: &mut AneAttention, request: usize, step: usize| {
                let start = (imported + step) * layout.kv_width();
                let end = start + layout.kv_width();
                attention
                    .decode(
                        &queries[step],
                        &keys[request][start..end],
                        &values[request][start..end],
                        &mut actual,
                    )
                    .unwrap();
                assert_eq!(attention.tokens_seen(), imported + step + 1);
                for (i, (actual, expected)) in
                    actual.iter().zip(&expected[request][step]).enumerate()
                {
                    assert!((actual.to_f32() - expected).abs() < 0.003,
                        "layout={layout:?} request={request} step={step} element={i} actual={actual} expected={expected}");
                }
            };
            for step in 0..3 {
                evaluate(&mut first, 0, step);
                evaluate(&mut second, 1, step);
            }
            drop(first);
            evaluate(&mut second, 1, 3);
        }
    }

    #[test]
    #[ignore = "executes private ANE API; two qualified shapes, six short-context evaluations"]
    fn hardware_attention_matches_imported_prefill_and_appended_decode() {
        check_attention_matches_imported_prefill_and_appended_decode(
            AneProgramCachePolicy::Compile,
        );
    }

    #[test]
    #[ignore = "two qualified graphs; four loads, twelve evaluations, including load-only reuse"]
    fn hardware_cached_attention_matches_imported_prefill_and_appended_decode() {
        check_attention_matches_imported_prefill_and_appended_decode(
            AneProgramCachePolicy::ReuseOrCompile,
        );
        check_attention_matches_imported_prefill_and_appended_decode(
            AneProgramCachePolicy::RequireExisting,
        );
    }

    fn check_attention_matches_imported_prefill_and_appended_decode(policy: AneProgramCachePolicy) {
        for (heads, kv, dim, window) in [(16, 8, 256, Some(1024)), (16, 1, 512, None)] {
            let mut attention = match window {
                Some(window) => AneAttentionProgram::compile_sliding_with_cache_policy(
                    heads, kv, dim, window, policy,
                ),
                None => AneAttentionProgram::compile_with_cache_policy(heads, kv, dim, 64, policy),
            }
            .unwrap()
            .create_request()
            .unwrap();
            let sample = |i: usize| f16::from_f32(((i * 17 + 3) % 71) as f32 / 128.0 - 0.25);
            let mut keys: Vec<_> = (0..5 * kv * dim).map(sample).collect();
            let mut values: Vec<_> = (0..keys.len()).map(|i| sample(i * 3 + 7)).collect();
            attention.import_cache(&keys, &values, 5).unwrap();
            for step in 0..3 {
                let q: Vec<_> = (0..heads * dim).map(|i| sample(i + step * 11)).collect();
                let k: Vec<_> = (0..kv * dim).map(|i| sample(i + step * 7)).collect();
                let v: Vec<_> = (0..kv * dim).map(|i| sample(i + step * 13)).collect();
                let mut actual = vec![f16::ZERO; heads * dim];
                attention.decode(&q, &k, &v, &mut actual).unwrap();
                keys.extend(k);
                values.extend(v);
                let tokens = 6 + step;
                assert_eq!(attention.tokens_seen(), tokens);
                let first = window.map_or(0, |n| tokens.saturating_sub(n));
                for head in 0..heads {
                    let kv_head = head / (heads / kv);
                    let mut scores = Vec::new();
                    for token in first..tokens {
                        let score: f32 = (0..dim)
                            .map(|d| {
                                q[head * dim + d].to_f32()
                                    * keys[(token * kv + kv_head) * dim + d].to_f32()
                            })
                            .sum();
                        scores.push(score);
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denominator: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                    for d in 0..dim {
                        let expected: f32 = scores
                            .iter()
                            .enumerate()
                            .map(|(t, s)| {
                                (s - max).exp() / denominator
                                    * values[((first + t) * kv + kv_head) * dim + d].to_f32()
                            })
                            .sum();
                        let got = actual[head * dim + d].to_f32();
                        assert!((got - expected).abs() < 0.003, "heads={heads} kv={kv} step={step} head={head} dim={d} actual={got} expected={expected}");
                    }
                }
            }
            assert!(attention.import_cache(&[], &[], 65).is_err());
        }
    }
}
