//! Immutable choices for one Metal owner. Environment translation is explicit
//! and never runs on the layer-encoding path.

#[cfg(feature = "donor-route-attribution")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DonorRouteMask(u8);

#[cfg(feature = "donor-route-attribution")]
impl DonorRouteMask {
    pub const PROJECTION: u8 = 1;
    pub const GATE: u8 = 2;
    pub const LOCAL_ATTENTION: u8 = 4;
    pub const GLOBAL_ATTENTION: u8 = 8;
    pub const ALL: u8 =
        Self::PROJECTION | Self::GATE | Self::LOCAL_ATTENTION | Self::GLOBAL_ATTENTION;

    #[must_use]
    pub const fn allows(self, component: u8) -> bool {
        self.0 & component != 0
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

#[cfg(feature = "donor-route-attribution")]
impl Default for DonorRouteMask {
    fn default() -> Self {
        Self(Self::ALL)
    }
}

#[cfg(feature = "donor-route-attribution")]
impl std::str::FromStr for DonorRouteMask {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "all" {
            return Ok(Self(Self::ALL));
        }
        if value == "none" {
            return Ok(Self(0));
        }
        let mut bits = 0;
        for part in value.split(',') {
            let bit = match part {
                "projection" => Self::PROJECTION,
                "gate" => Self::GATE,
                "local-attention" => Self::LOCAL_ATTENTION,
                "global-attention" => Self::GLOBAL_ATTENTION,
                _ => return Err("unknown donor component"),
            };
            if bits & bit != 0 {
                return Err("duplicate donor component");
            }
            bits |= bit;
        }
        Ok(Self(bits))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalKernelOptions {
    pub qkv_prefill_batch8: bool,
    pub prefill_mma32: bool,
    pub prefill_simd_attention: bool,
    pub quantized_bf16_accumulation: bool,
    /// Explicit experiment, captured once; never inferred from model/device.
    pub research: crate::research::MetalResearchCandidate,
    #[cfg(feature = "donor-route-attribution")]
    pub donor_route_mask: DonorRouteMask,
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
                #[cfg(feature = "donor-route-attribution")]
                donor_route_mask: match std::env::var("RVLLM_METAL_DONOR_COMPONENTS") {
                    Ok(value) => value.parse().unwrap_or_else(|reason| {
                        tracing::warn!(%reason, requested = %value, "Donor components disabled");
                        DonorRouteMask(0)
                    }),
                    Err(std::env::VarError::NotPresent) => DonorRouteMask::default(),
                    Err(std::env::VarError::NotUnicode(_)) => {
                        tracing::warn!("Non-Unicode donor component selector disabled");
                        DonorRouteMask(0)
                    }
                },
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

    #[cfg(feature = "donor-route-attribution")]
    #[test]
    fn donor_component_mask_is_exact_and_fail_closed() {
        let all: DonorRouteMask = "all".parse().unwrap();
        assert_eq!(all.bits(), DonorRouteMask::ALL);
        let none: DonorRouteMask = "none".parse().unwrap();
        assert_eq!(none.bits(), 0);
        let selected: DonorRouteMask = "projection,global-attention".parse().unwrap();
        assert!(selected.allows(DonorRouteMask::PROJECTION));
        assert!(selected.allows(DonorRouteMask::GLOBAL_ATTENTION));
        assert!(!selected.allows(DonorRouteMask::GATE));
        assert!(!selected.allows(DonorRouteMask::LOCAL_ATTENTION));
        for invalid in [
            "",
            "projection,",
            "projection,projection",
            "Projection",
            "attention",
        ] {
            assert!(invalid.parse::<DonorRouteMask>().is_err());
        }
    }

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
