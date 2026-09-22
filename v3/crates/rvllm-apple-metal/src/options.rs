//! Immutable choices for one Metal owner. Environment translation is explicit
//! and never runs on the layer-encoding path.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalKernelOptions {
    pub qkv_prefill_batch8: bool,
    pub prefill_mma32: bool,
    pub prefill_simd_attention: bool,
    pub quantized_bf16_accumulation: bool,
    /// Explicit experiment, captured once; never inferred from model/device.
    pub research: crate::research::MetalResearchCandidate,
}

impl MetalKernelOptions {
    /// Capture the macOS diagnostic controls once. Embedded callers should
    /// construct this value directly; iOS has no environment configuration.
    #[must_use]
    pub fn from_development_environment() -> Self {
        #[cfg(target_os = "macos")]
        {
            let selected = |name, value| std::env::var(name).ok().as_deref() == Some(value);
            let research = match std::env::var("RVLLM_METAL_RESEARCH") {
                Ok(value) => value.parse().unwrap_or_else(|reason| {
                    tracing::warn!(%reason, requested = %value, "Metal research disabled");
                    crate::research::MetalResearchCandidate::Off
                }),
                Err(_) => crate::research::MetalResearchCandidate::Off,
            };
            Self {
                research,
                qkv_prefill_batch8: selected("RVLLM_METAL_QKV_PREFILL", "batch8"),
                prefill_mma32: selected("RVLLM_METAL_PREFILL_GEMM", "mma32"),
                prefill_simd_attention: selected("RVLLM_METAL_PREFILL_ATTENTION", "simdgroup"),
                quantized_bf16_accumulation: selected("RVLLM_METAL_BF16_ACCUM", "quantized"),
            }
        }
        #[cfg(not(target_os = "macos"))]
        Self::default()
    }
}

/// Per-owner limits. Validation precedes tensor scanning and arena allocation.
/// Trace buffers are deliberately absent from this embedded configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalModelLimits {
    pub max_context_tokens: usize,
    pub max_batch_tokens: usize,
    pub max_batch_sequences: usize,
}

impl MetalModelLimits {
    pub fn validate(self, model_max_context: usize) -> rvllm_core::Result<()> {
        if self.max_context_tokens == 0
            || self.max_context_tokens > model_max_context
            || self.max_batch_tokens == 0
            || self.max_batch_sequences == 0
            || self.max_context_tokens > u32::MAX as usize
            || self.max_batch_tokens > u32::MAX as usize
            || self.max_batch_sequences > u32::MAX as usize
            || self
                .max_context_tokens
                .checked_mul(self.max_batch_sequences)
                .is_none()
        {
            return Err(rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::InvalidWeightBlob {
                    reason: "invalid explicit Metal model limits",
                },
                rvllm_core::AppleCtx {
                    backend: "model-metal-backend",
                    op: "validate_model_limits",
                    device: "apple-silicon",
                },
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_limits_reject_zero_excess_context_and_integer_overflow() {
        let valid = MetalModelLimits {
            max_context_tokens: 1024,
            max_batch_tokens: 1024,
            max_batch_sequences: 1,
        };
        assert!(valid.validate(1024).is_ok());
        assert!(valid.validate(1023).is_err());
        for bad in [
            MetalModelLimits {
                max_context_tokens: 0,
                ..valid
            },
            MetalModelLimits {
                max_batch_tokens: 0,
                ..valid
            },
            MetalModelLimits {
                max_batch_sequences: 0,
                ..valid
            },
            MetalModelLimits {
                max_batch_tokens: usize::MAX,
                ..valid
            },
            MetalModelLimits {
                max_context_tokens: usize::MAX,
                max_batch_sequences: 2,
                ..valid
            },
        ] {
            assert!(bad.validate(usize::MAX).is_err());
        }
    }
}
