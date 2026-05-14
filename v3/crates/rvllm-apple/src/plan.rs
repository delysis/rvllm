use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
use serde::{Deserialize, Serialize};

use crate::device::AppleAcceleratorTarget;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AppleBackendMode {
    MetalOnly,
    MlxPrototype,
    MetalPrefillMetalDecode,
    MetalPrefillAneFfnRollout,
    MetalPrefillAneRolloutExperimental,
}

impl AppleBackendMode {
    #[must_use]
    pub const fn requires_private_ane(self) -> bool {
        matches!(
            self,
            AppleBackendMode::MetalPrefillAneFfnRollout
                | AppleBackendMode::MetalPrefillAneRolloutExperimental
        )
    }

    #[must_use]
    pub const fn as_flag(self) -> &'static str {
        match self {
            AppleBackendMode::MetalOnly => "metal-only",
            AppleBackendMode::MlxPrototype => "mlx-prototype",
            AppleBackendMode::MetalPrefillMetalDecode => "metal-prefill-metal-decode",
            AppleBackendMode::MetalPrefillAneFfnRollout => "metal-prefill-ane-ffn-rollout",
            AppleBackendMode::MetalPrefillAneRolloutExperimental => {
                "metal-prefill-ane-rollout-experimental"
            }
        }
    }

    #[must_use]
    pub fn from_flag(value: &str) -> Option<Self> {
        match value {
            "metal-only" => Some(AppleBackendMode::MetalOnly),
            "mlx-prototype" => Some(AppleBackendMode::MlxPrototype),
            "metal-prefill-metal-decode" => Some(AppleBackendMode::MetalPrefillMetalDecode),
            "metal-prefill-ane-ffn-rollout" => Some(AppleBackendMode::MetalPrefillAneFfnRollout),
            "metal-prefill-ane-rollout-experimental" => {
                Some(AppleBackendMode::MetalPrefillAneRolloutExperimental)
            }
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct RolloutBucket {
    pub seqs: u32,
    pub tokens: u32,
}

impl RolloutBucket {
    #[must_use]
    pub const fn fits(self, seqs: u32, tokens: u32) -> bool {
        self.seqs >= seqs && self.tokens >= tokens
    }

    #[must_use]
    pub const fn capacity(self) -> u32 {
        self.seqs * self.tokens
    }

    #[must_use]
    pub const fn waste(self, seqs: u32, tokens: u32) -> u32 {
        self.capacity() - seqs * tokens
    }
}

pub const ROLLOUT_BUCKETS: &[RolloutBucket] = &[
    RolloutBucket { seqs: 1, tokens: 1 },
    RolloutBucket { seqs: 2, tokens: 1 },
    RolloutBucket { seqs: 4, tokens: 1 },
    RolloutBucket { seqs: 8, tokens: 1 },
    RolloutBucket {
        seqs: 16,
        tokens: 1,
    },
    RolloutBucket {
        seqs: 32,
        tokens: 1,
    },
    RolloutBucket {
        seqs: 64,
        tokens: 1,
    },
    RolloutBucket {
        seqs: 128,
        tokens: 1,
    },
    RolloutBucket { seqs: 4, tokens: 4 },
    RolloutBucket { seqs: 8, tokens: 4 },
    RolloutBucket {
        seqs: 16,
        tokens: 4,
    },
    RolloutBucket {
        seqs: 32,
        tokens: 4,
    },
    RolloutBucket {
        seqs: 64,
        tokens: 4,
    },
    RolloutBucket { seqs: 8, tokens: 8 },
    RolloutBucket {
        seqs: 16,
        tokens: 8,
    },
    RolloutBucket {
        seqs: 32,
        tokens: 8,
    },
];

#[must_use]
pub fn select_rollout_bucket(seqs: u32, tokens: u32) -> Option<RolloutBucket> {
    select_rollout_bucket_from(ROLLOUT_BUCKETS, seqs, tokens)
}

#[must_use]
fn select_rollout_bucket_from(
    buckets: &[RolloutBucket],
    seqs: u32,
    tokens: u32,
) -> Option<RolloutBucket> {
    if seqs == 0 || tokens == 0 {
        return None;
    }

    buckets
        .iter()
        .copied()
        .filter(|bucket| bucket.fits(seqs, tokens))
        .min_by_key(|bucket| {
            (
                bucket.waste(seqs, tokens),
                bucket.capacity(),
                bucket.seqs,
                bucket.tokens,
            )
        })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AppleMatmulBackend {
    Metal,
    Ane,
}

impl AppleMatmulBackend {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            AppleMatmulBackend::Metal => "metal",
            AppleMatmulBackend::Ane => "ane",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AppleMatmulQuantization {
    Fp16,
    Q8,
    Int8W8A8,
}

impl AppleMatmulQuantization {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            AppleMatmulQuantization::Fp16 => "fp16",
            AppleMatmulQuantization::Q8 => "q8",
            AppleMatmulQuantization::Int8W8A8 => "int8-w8a8",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AppleElementType {
    Fp16,
    Int8,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AppleScaleSpec {
    None,
    F16Block { block_size: u32 },
    F32PerChannel,
    F32PerToken,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AppleMatmulMetadata {
    pub quantization: AppleMatmulQuantization,
    pub weight_dtype: AppleElementType,
    pub activation_dtype: AppleElementType,
    pub output_dtype: AppleElementType,
    pub weight_scale: AppleScaleSpec,
    pub activation_scale: AppleScaleSpec,
}

impl AppleMatmulMetadata {
    #[must_use]
    pub const fn fp16() -> Self {
        Self {
            quantization: AppleMatmulQuantization::Fp16,
            weight_dtype: AppleElementType::Fp16,
            activation_dtype: AppleElementType::Fp16,
            output_dtype: AppleElementType::Fp16,
            weight_scale: AppleScaleSpec::None,
            activation_scale: AppleScaleSpec::None,
        }
    }

    #[must_use]
    pub const fn q8_block(block_size: u32) -> Self {
        Self {
            quantization: AppleMatmulQuantization::Q8,
            weight_dtype: AppleElementType::Int8,
            activation_dtype: AppleElementType::Fp16,
            output_dtype: AppleElementType::Fp16,
            weight_scale: AppleScaleSpec::F16Block { block_size },
            activation_scale: AppleScaleSpec::None,
        }
    }

    #[must_use]
    pub const fn int8_w8a8() -> Self {
        Self {
            quantization: AppleMatmulQuantization::Int8W8A8,
            weight_dtype: AppleElementType::Int8,
            activation_dtype: AppleElementType::Int8,
            output_dtype: AppleElementType::Fp16,
            weight_scale: AppleScaleSpec::F32PerChannel,
            activation_scale: AppleScaleSpec::F32PerToken,
        }
    }

    pub fn validate(self) -> Result<()> {
        match self.quantization {
            AppleMatmulQuantization::Fp16 => self.validate_fp16(),
            AppleMatmulQuantization::Q8 => self.validate_q8(),
            AppleMatmulQuantization::Int8W8A8 => self.validate_int8_w8a8(),
        }
    }

    fn validate_fp16(self) -> Result<()> {
        if self.weight_dtype != AppleElementType::Fp16 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "weight dtype must be fp16",
            ));
        }
        if self.activation_dtype != AppleElementType::Fp16 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "activation dtype must be fp16",
            ));
        }
        if self.output_dtype != AppleElementType::Fp16 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "output dtype must be fp16",
            ));
        }
        if self.weight_scale != AppleScaleSpec::None {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "weight scale must be absent",
            ));
        }
        if self.activation_scale != AppleScaleSpec::None {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "activation scale must be absent",
            ));
        }
        Ok(())
    }

    fn validate_q8(self) -> Result<()> {
        if self.weight_dtype != AppleElementType::Int8 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "weight dtype must be int8",
            ));
        }
        if self.activation_dtype != AppleElementType::Fp16 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "activation dtype must be fp16",
            ));
        }
        if self.output_dtype != AppleElementType::Fp16 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "output dtype must be fp16",
            ));
        }
        match self.weight_scale {
            AppleScaleSpec::F16Block { block_size: 32 } => {}
            _ => {
                return Err(invalid_quantization_metadata(
                    self.quantization,
                    "weight scale must be f16 block32",
                ));
            }
        }
        if self.activation_scale != AppleScaleSpec::None {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "activation scale must be absent",
            ));
        }
        Ok(())
    }

    fn validate_int8_w8a8(self) -> Result<()> {
        if self.weight_dtype != AppleElementType::Int8 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "weight dtype must be int8",
            ));
        }
        if self.activation_dtype != AppleElementType::Int8 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "activation dtype must be int8",
            ));
        }
        if self.output_dtype != AppleElementType::Fp16 {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "output dtype must be fp16",
            ));
        }
        if self.weight_scale != AppleScaleSpec::F32PerChannel {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "weight scale must be f32 per-channel",
            ));
        }
        if self.activation_scale != AppleScaleSpec::F32PerToken {
            return Err(invalid_quantization_metadata(
                self.quantization,
                "activation scale must be f32 per-token",
            ));
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AppleQuantizationMatrixEntry {
    pub backend: AppleMatmulBackend,
    pub quantization: AppleMatmulQuantization,
    pub supported: bool,
}

pub const APPLE_QUANTIZATION_MATRIX: &[AppleQuantizationMatrixEntry] = &[
    AppleQuantizationMatrixEntry {
        backend: AppleMatmulBackend::Metal,
        quantization: AppleMatmulQuantization::Fp16,
        supported: true,
    },
    AppleQuantizationMatrixEntry {
        backend: AppleMatmulBackend::Metal,
        quantization: AppleMatmulQuantization::Q8,
        supported: true,
    },
    AppleQuantizationMatrixEntry {
        backend: AppleMatmulBackend::Metal,
        quantization: AppleMatmulQuantization::Int8W8A8,
        supported: true,
    },
    AppleQuantizationMatrixEntry {
        backend: AppleMatmulBackend::Ane,
        quantization: AppleMatmulQuantization::Fp16,
        supported: true,
    },
    AppleQuantizationMatrixEntry {
        backend: AppleMatmulBackend::Ane,
        quantization: AppleMatmulQuantization::Q8,
        supported: false,
    },
    AppleQuantizationMatrixEntry {
        backend: AppleMatmulBackend::Ane,
        quantization: AppleMatmulQuantization::Int8W8A8,
        supported: true,
    },
];

#[must_use]
pub fn apple_quantization_matrix_entry(
    backend: AppleMatmulBackend,
    quantization: AppleMatmulQuantization,
) -> Option<&'static AppleQuantizationMatrixEntry> {
    APPLE_QUANTIZATION_MATRIX
        .iter()
        .find(|entry| entry.backend == backend && entry.quantization == quantization)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AppleMatmulPlan {
    pub backend: AppleMatmulBackend,
    pub metadata: AppleMatmulMetadata,
}

pub fn plan_apple_matmul(
    backend: AppleMatmulBackend,
    metadata: AppleMatmulMetadata,
) -> Result<AppleMatmulPlan> {
    metadata.validate()?;
    let entry = apple_quantization_matrix_entry(backend, metadata.quantization)
        .ok_or_else(|| unsupported_quantization(backend, metadata.quantization))?;
    if !entry.supported {
        return Err(unsupported_quantization(backend, metadata.quantization));
    }
    Ok(AppleMatmulPlan { backend, metadata })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AppleMatmulConfig {
    pub metal: AppleMatmulMetadata,
    pub ane: AppleMatmulMetadata,
}

impl AppleMatmulConfig {
    #[must_use]
    pub const fn fp16() -> Self {
        Self {
            metal: AppleMatmulMetadata::fp16(),
            ane: AppleMatmulMetadata::fp16(),
        }
    }

    pub fn validate_for_mode(self, mode: AppleBackendMode) -> Result<()> {
        self.metal.validate()?;
        self.ane.validate()?;
        plan_apple_matmul(AppleMatmulBackend::Metal, self.metal)?;
        if mode.requires_private_ane() {
            plan_apple_matmul(AppleMatmulBackend::Ane, self.ane)?;
        }
        Ok(())
    }
}

fn invalid_quantization_metadata(
    quantization: AppleMatmulQuantization,
    reason: &'static str,
) -> RvllmError {
    RvllmError::apple(
        AppleError::InvalidQuantizationMetadata {
            quantization: quantization.as_str(),
            reason,
        },
        quantization_ctx("validate_metadata"),
    )
}

fn unsupported_quantization(
    backend: AppleMatmulBackend,
    quantization: AppleMatmulQuantization,
) -> RvllmError {
    RvllmError::apple(
        AppleError::UnsupportedQuantization {
            backend: backend.as_str(),
            quantization: quantization.as_str(),
        },
        AppleCtx {
            backend: backend.as_str(),
            op: "plan_matmul",
            device: "apple-silicon",
        },
    )
}

const fn quantization_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "rvllm-apple",
        op,
        device: "apple-silicon",
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleRuntimePlan {
    pub target: AppleAcceleratorTarget,
    pub mode: AppleBackendMode,
    pub matmul: AppleMatmulConfig,
    pub rollout_bucket: Option<RolloutBucket>,
    pub rollout_tokens: u32,
    pub private_ane_opt_in: bool,
}

impl AppleRuntimePlan {
    pub fn validate(&self) -> Result<()> {
        if self.rollout_tokens == 0 {
            return Err(RvllmError::apple(
                AppleError::ShapeBucketMissing { seqs: 0, tokens: 0 },
                self.ctx("validate"),
            ));
        }

        if let Some(bucket) = self.rollout_bucket {
            if !ROLLOUT_BUCKETS.contains(&bucket) {
                return Err(RvllmError::apple(
                    AppleError::ShapeBucketMissing {
                        seqs: bucket.seqs,
                        tokens: bucket.tokens,
                    },
                    self.ctx("validate"),
                ));
            }
            if bucket.tokens < self.rollout_tokens {
                return Err(RvllmError::apple(
                    AppleError::ShapeBucketMissing {
                        seqs: bucket.seqs,
                        tokens: self.rollout_tokens,
                    },
                    self.ctx("validate"),
                ));
            }
        }

        self.matmul.validate_for_mode(self.mode)?;
        if self.mode.requires_private_ane() && !self.private_ane_opt_in {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "private-ane",
                    op: "rollout",
                },
                self.ctx("validate"),
            ));
        }
        if self.mode.requires_private_ane() && self.rollout_bucket.is_none() {
            return Err(RvllmError::apple(
                AppleError::ShapeBucketMissing {
                    seqs: 0,
                    tokens: self.rollout_tokens,
                },
                self.ctx("validate"),
            ));
        }

        if self.mode.requires_private_ane()
            && !cfg!(all(target_os = "macos", target_arch = "aarch64"))
        {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "private-ane",
                    op: "rollout",
                },
                self.ctx("validate"),
            ));
        }

        Ok(())
    }

    fn ctx(&self, op: &'static str) -> AppleCtx {
        AppleCtx {
            backend: "rvllm-apple",
            op,
            device: "apple-silicon",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::AppleAcceleratorTarget;

    #[test]
    fn rollout_bucket_minimizes_padding_waste() {
        assert_eq!(
            select_rollout_bucket(1, 1),
            Some(RolloutBucket { seqs: 1, tokens: 1 })
        );
        assert_eq!(
            select_rollout_bucket(3, 1),
            Some(RolloutBucket { seqs: 4, tokens: 1 })
        );
        assert_eq!(
            select_rollout_bucket(3, 4),
            Some(RolloutBucket { seqs: 4, tokens: 4 })
        );
        assert_eq!(
            select_rollout_bucket(9, 4),
            Some(RolloutBucket {
                seqs: 16,
                tokens: 4
            })
        );
        assert_eq!(select_rollout_bucket(33, 8), None);
    }

    #[test]
    fn rollout_bucket_selection_is_not_first_fit() {
        let buckets = [
            RolloutBucket {
                seqs: 32,
                tokens: 4,
            },
            RolloutBucket { seqs: 8, tokens: 4 },
            RolloutBucket {
                seqs: 16,
                tokens: 4,
            },
        ];

        assert_eq!(
            select_rollout_bucket_from(&buckets, 6, 3),
            Some(RolloutBucket { seqs: 8, tokens: 4 })
        );
    }

    #[test]
    fn ane_mode_requires_private_opt_in() {
        let plan = AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillAneFfnRollout,
            matmul: AppleMatmulConfig::fp16(),
            rollout_bucket: Some(RolloutBucket { seqs: 8, tokens: 4 }),
            rollout_tokens: 4,
            private_ane_opt_in: false,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn metal_decode_plan_allows_no_rollout_bucket() {
        let plan = AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillMetalDecode,
            matmul: AppleMatmulConfig::fp16(),
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn ane_mode_requires_rollout_bucket() {
        let plan = AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillAneFfnRollout,
            matmul: AppleMatmulConfig::fp16(),
            rollout_bucket: None,
            rollout_tokens: 4,
            private_ane_opt_in: true,
        };
        assert!(matches!(
            plan.validate(),
            Err(RvllmError::Apple {
                err: AppleError::ShapeBucketMissing { seqs: 0, tokens: 4 },
                ..
            })
        ));
    }

    #[test]
    fn runtime_plan_rejects_unknown_rollout_bucket() {
        let plan = AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillMetalDecode,
            matmul: AppleMatmulConfig::fp16(),
            rollout_bucket: Some(RolloutBucket { seqs: 3, tokens: 7 }),
            rollout_tokens: 7,
            private_ane_opt_in: false,
        };
        assert!(matches!(
            plan.validate(),
            Err(RvllmError::Apple {
                err: AppleError::ShapeBucketMissing { seqs: 3, tokens: 7 },
                ..
            })
        ));
    }

    #[test]
    fn runtime_plan_rejects_bucket_smaller_than_rollout_tokens() {
        let plan = AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillMetalDecode,
            matmul: AppleMatmulConfig::fp16(),
            rollout_bucket: Some(RolloutBucket { seqs: 8, tokens: 4 }),
            rollout_tokens: 8,
            private_ane_opt_in: false,
        };
        assert!(matches!(
            plan.validate(),
            Err(RvllmError::Apple {
                err: AppleError::ShapeBucketMissing { seqs: 8, tokens: 8 },
                ..
            })
        ));
    }

    #[test]
    fn quantization_matrix_has_explicit_metal_and_ane_entries() {
        for backend in [AppleMatmulBackend::Metal, AppleMatmulBackend::Ane] {
            for format in [
                AppleMatmulQuantization::Fp16,
                AppleMatmulQuantization::Q8,
                AppleMatmulQuantization::Int8W8A8,
            ] {
                assert!(
                    apple_quantization_matrix_entry(backend, format).is_some(),
                    "missing matrix entry for {backend:?} {format:?}"
                );
            }
        }
    }

    #[test]
    fn ane_q8_metadata_is_rejected_without_fallback() {
        let err = plan_apple_matmul(AppleMatmulBackend::Ane, AppleMatmulMetadata::q8_block(32))
            .unwrap_err();
        assert!(matches!(
            err,
            RvllmError::Apple {
                err: AppleError::UnsupportedQuantization {
                    backend: "ane",
                    quantization: "q8"
                },
                ..
            }
        ));
    }

    #[test]
    fn int8_w8a8_requires_activation_scale_metadata() {
        let mut metadata = AppleMatmulMetadata::int8_w8a8();
        metadata.activation_scale = AppleScaleSpec::None;
        let err = metadata.validate().unwrap_err();
        assert!(matches!(
            err,
            RvllmError::Apple {
                err: AppleError::InvalidQuantizationMetadata {
                    quantization: "int8-w8a8",
                    reason: "activation scale must be f32 per-token"
                },
                ..
            }
        ));
    }

    #[test]
    fn runtime_plan_rejects_ane_q8_config_without_fallback() {
        let plan = AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillAneFfnRollout,
            matmul: AppleMatmulConfig {
                metal: AppleMatmulMetadata::q8_block(32),
                ane: AppleMatmulMetadata::q8_block(32),
            },
            rollout_bucket: Some(RolloutBucket { seqs: 8, tokens: 4 }),
            rollout_tokens: 4,
            private_ane_opt_in: true,
        };
        let err = plan.validate().unwrap_err();
        assert!(matches!(
            err,
            RvllmError::Apple {
                err: AppleError::UnsupportedQuantization {
                    backend: "ane",
                    quantization: "q8"
                },
                ..
            }
        ));
    }
}
