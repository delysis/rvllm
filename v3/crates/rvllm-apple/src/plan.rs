use rvllm_core::config::{AneComputeProfile, AneFallbackPolicy};
use rvllm_core::DType;
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
use serde::{Deserialize, Serialize};

use crate::device::AppleAcceleratorTarget;

pub const PRIVATE_ANE_ENV_VAR: &str = "RVLLM_ENABLE_PRIVATE_ANE";

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
    ROLLOUT_BUCKETS
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
pub enum AnePartitionOpKind {
    FfnGateUpDown,
    LmHeadVocabBlock,
    QkvProjection,
    OutputProjection,
}

impl AnePartitionOpKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FfnGateUpDown => "ffn_gate_up_down",
            Self::LmHeadVocabBlock => "lm_head_vocab_block",
            Self::QkvProjection => "qkv_projection",
            Self::OutputProjection => "output_projection",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AneLayerRange {
    pub start: u32,
    pub end: u32,
}

impl AneLayerRange {
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AneVocabBlock {
    pub start: u32,
    pub len: u32,
}

impl AneVocabBlock {
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AnePartitionRequest {
    pub op_kind: AnePartitionOpKind,
    pub layer_range: AneLayerRange,
    pub bucket: RolloutBucket,
    pub dtype: DType,
    pub vocab_block: Option<AneVocabBlock>,
}

impl AnePartitionRequest {
    #[must_use]
    pub fn ffn(layer_range: AneLayerRange, bucket: RolloutBucket, dtype: DType) -> Self {
        Self {
            op_kind: AnePartitionOpKind::FfnGateUpDown,
            layer_range,
            bucket,
            dtype,
            vocab_block: None,
        }
    }

    #[must_use]
    pub fn lm_head(
        layer_range: AneLayerRange,
        bucket: RolloutBucket,
        dtype: DType,
        vocab_block: AneVocabBlock,
    ) -> Self {
        Self {
            op_kind: AnePartitionOpKind::LmHeadVocabBlock,
            layer_range,
            bucket,
            dtype,
            vocab_block: Some(vocab_block),
        }
    }

    #[must_use]
    pub fn qkv(layer_range: AneLayerRange, bucket: RolloutBucket, dtype: DType) -> Self {
        Self {
            op_kind: AnePartitionOpKind::QkvProjection,
            layer_range,
            bucket,
            dtype,
            vocab_block: None,
        }
    }

    #[must_use]
    pub fn output_projection(
        layer_range: AneLayerRange,
        bucket: RolloutBucket,
        dtype: DType,
    ) -> Self {
        Self {
            op_kind: AnePartitionOpKind::OutputProjection,
            layer_range,
            bucket,
            dtype,
            vocab_block: None,
        }
    }

    #[must_use]
    pub fn unsupported_reason(&self) -> Option<AneUnsupportedReason> {
        if self.layer_range.is_empty() {
            return Some(AneUnsupportedReason::EmptyLayerRange);
        }
        if !ROLLOUT_BUCKETS.contains(&self.bucket) {
            return Some(AneUnsupportedReason::UnsupportedBucket);
        }
        if !matches!(self.dtype, DType::F16) {
            return Some(AneUnsupportedReason::UnsupportedDType);
        }
        if matches!(self.op_kind, AnePartitionOpKind::LmHeadVocabBlock)
            && self.vocab_block.map_or(true, AneVocabBlock::is_empty)
        {
            return Some(AneUnsupportedReason::EmptyVocabBlock);
        }
        if !matches!(self.op_kind, AnePartitionOpKind::LmHeadVocabBlock)
            && self.vocab_block.is_some()
        {
            return Some(AneUnsupportedReason::UnexpectedVocabBlock);
        }
        None
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AneCapabilityPath {
    PublicCoreMl,
    PrivateAne,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AneUnsupportedReason {
    NonMacOs,
    NonAppleSilicon,
    UnknownNpu,
    NoAneCores,
    PrivateAneFeatureDisabled,
    PrivateAneEnvOptInMissing,
    PublicCoreMlExecutionPathNotEnabled,
    EmptyLayerRange,
    UnsupportedBucket,
    UnsupportedDType,
    EmptyVocabBlock,
    UnexpectedVocabBlock,
}

impl AneUnsupportedReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonMacOs => "non_macos",
            Self::NonAppleSilicon => "non_apple_silicon",
            Self::UnknownNpu => "unknown_npu",
            Self::NoAneCores => "no_ane_cores",
            Self::PrivateAneFeatureDisabled => "private_ane_feature_disabled",
            Self::PrivateAneEnvOptInMissing => "private_ane_env_opt_in_missing",
            Self::PublicCoreMlExecutionPathNotEnabled => "public_coreml_execution_path_not_enabled",
            Self::EmptyLayerRange => "empty_layer_range",
            Self::UnsupportedBucket => "unsupported_bucket",
            Self::UnsupportedDType => "unsupported_dtype",
            Self::EmptyVocabBlock => "empty_vocab_block",
            Self::UnexpectedVocabBlock => "unexpected_vocab_block",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AneCapabilityStatus {
    Available,
    Unsupported { reason: AneUnsupportedReason },
}

impl AneCapabilityStatus {
    #[must_use]
    pub const fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum CoreMlComputeUnitsPlan {
    All,
    CpuAndNeuralEngine,
    NeuralEngineOnly,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoreMlAneComputePlan {
    pub compute_profile: AneComputeProfile,
    pub compute_units: CoreMlComputeUnitsPlan,
    pub uses_public_coreml_execution: bool,
    pub requires_private_ane: bool,
}

impl CoreMlAneComputePlan {
    #[must_use]
    pub const fn from_profile(
        compute_profile: AneComputeProfile,
        requires_private_ane: bool,
    ) -> Self {
        let compute_units = match compute_profile {
            AneComputeProfile::AnyAvailable => CoreMlComputeUnitsPlan::All,
            AneComputeProfile::NeuralEnginePreferred => CoreMlComputeUnitsPlan::CpuAndNeuralEngine,
            AneComputeProfile::NeuralEngineOnly => CoreMlComputeUnitsPlan::NeuralEngineOnly,
        };
        Self {
            compute_profile,
            compute_units,
            uses_public_coreml_execution: !requires_private_ane,
            requires_private_ane,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneCapabilityReport {
    pub path: AneCapabilityPath,
    pub status: AneCapabilityStatus,
    pub target: AppleAcceleratorTarget,
    pub private_ane_feature_enabled: bool,
    pub private_ane_env_opt_in: bool,
    pub compute_plan: CoreMlAneComputePlan,
}

impl AneCapabilityReport {
    #[must_use]
    pub const fn is_available(&self) -> bool {
        self.status.is_available()
    }
}

#[must_use]
pub fn private_ane_feature_enabled() -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        feature = "private-ane"
    ))
}

#[must_use]
pub fn private_ane_env_opted_in() -> bool {
    std::env::var(PRIVATE_ANE_ENV_VAR)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[must_use]
pub fn probe_ane_capability(
    target: &AppleAcceleratorTarget,
    compute_profile: AneComputeProfile,
    private_ane_requested: bool,
) -> AneCapabilityReport {
    let path = if private_ane_requested {
        AneCapabilityPath::PrivateAne
    } else {
        AneCapabilityPath::PublicCoreMl
    };
    let private_ane_feature_enabled = private_ane_feature_enabled();
    let private_ane_env_opt_in = private_ane_env_opted_in();
    let compute_plan = CoreMlAneComputePlan::from_profile(compute_profile, private_ane_requested);
    let status = if !cfg!(target_os = "macos") {
        AneCapabilityStatus::Unsupported {
            reason: AneUnsupportedReason::NonMacOs,
        }
    } else if !cfg!(target_arch = "aarch64") {
        AneCapabilityStatus::Unsupported {
            reason: AneUnsupportedReason::NonAppleSilicon,
        }
    } else if target.ane_cores == 0 {
        AneCapabilityStatus::Unsupported {
            reason: AneUnsupportedReason::NoAneCores,
        }
    } else if matches!(
        target.npu_generation,
        crate::device::AppleNpuGeneration::Unknown
    ) {
        AneCapabilityStatus::Unsupported {
            reason: AneUnsupportedReason::UnknownNpu,
        }
    } else if private_ane_requested && !private_ane_feature_enabled {
        AneCapabilityStatus::Unsupported {
            reason: AneUnsupportedReason::PrivateAneFeatureDisabled,
        }
    } else if private_ane_requested && !private_ane_env_opt_in {
        AneCapabilityStatus::Unsupported {
            reason: AneUnsupportedReason::PrivateAneEnvOptInMissing,
        }
    } else if !private_ane_requested {
        AneCapabilityStatus::Unsupported {
            reason: AneUnsupportedReason::PublicCoreMlExecutionPathNotEnabled,
        }
    } else {
        AneCapabilityStatus::Available
    };

    AneCapabilityReport {
        path,
        status,
        target: target.clone(),
        private_ane_feature_enabled,
        private_ane_env_opt_in,
        compute_plan,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AneOsDeviceKey {
    pub os: String,
    pub arch: String,
    pub device: String,
}

impl AneOsDeviceKey {
    #[must_use]
    pub fn from_target(target: &AppleAcceleratorTarget) -> Self {
        Self {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            device: target.cache_key(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AneCompiledCacheKey {
    pub model_hash: [u8; 32],
    pub layer_range: AneLayerRange,
    pub bucket: RolloutBucket,
    pub dtype: DType,
    pub op_kind: AnePartitionOpKind,
    pub os_device: AneOsDeviceKey,
}

impl AneCompiledCacheKey {
    #[must_use]
    pub fn new(
        model_hash: [u8; 32],
        request: &AnePartitionRequest,
        target: &AppleAcceleratorTarget,
    ) -> Self {
        Self {
            model_hash,
            layer_range: request.layer_range,
            bucket: request.bucket,
            dtype: request.dtype,
            op_kind: request.op_kind,
            os_device: AneOsDeviceKey::from_target(target),
        }
    }

    #[must_use]
    pub fn deterministic_input(&self) -> String {
        format!(
            concat!(
                "rvllm-ane-cache-key-v1\n",
                "model_hash={}\n",
                "layer_start={}\n",
                "layer_end={}\n",
                "bucket_seqs={}\n",
                "bucket_tokens={}\n",
                "dtype={:?}\n",
                "op_kind={}\n",
                "os={}\n",
                "arch={}\n",
                "device={}\n",
            ),
            hex32(&self.model_hash),
            self.layer_range.start,
            self.layer_range.end,
            self.bucket.seqs,
            self.bucket.tokens,
            self.dtype,
            self.op_kind.as_str(),
            self.os_device.os,
            self.os_device.arch,
            self.os_device.device,
        )
    }

    #[must_use]
    pub fn stable_id(&self) -> String {
        format!(
            "ane_static_v1_{}",
            stable_hash_hex(self.deterministic_input().as_bytes())
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AnePlannedBackend {
    Ane,
    MetalFallback,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneStaticPartition {
    pub request: AnePartitionRequest,
    pub backend: AnePlannedBackend,
    pub cache_key: Option<AneCompiledCacheKey>,
    pub fallback_reason: Option<AneUnsupportedReason>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AnePartitionPolicy {
    pub compute_profile: AneComputeProfile,
    pub fallback_policy: AneFallbackPolicy,
}

impl AnePartitionPolicy {
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            compute_profile: AneComputeProfile::NeuralEngineOnly,
            fallback_policy: AneFallbackPolicy::FailFast,
        }
    }

    #[must_use]
    pub const fn allow_metal() -> Self {
        Self {
            compute_profile: AneComputeProfile::NeuralEnginePreferred,
            fallback_policy: AneFallbackPolicy::AllowMetal,
        }
    }

    #[must_use]
    pub const fn requires_strict_ane(self) -> bool {
        self.fallback_policy.is_strict()
            || matches!(self.compute_profile, AneComputeProfile::NeuralEngineOnly)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneStaticPartitionPlan {
    pub backend: AnePlannedBackend,
    pub partitions: Vec<AneStaticPartition>,
}

pub fn plan_ane_static_partitions(
    requests: &[AnePartitionRequest],
    capability: &AneCapabilityReport,
    policy: AnePartitionPolicy,
    model_hash: [u8; 32],
) -> Result<AneStaticPartitionPlan> {
    let mut partitions = Vec::with_capacity(requests.len());
    let mut plan_backend = AnePlannedBackend::Ane;
    for request in requests {
        let reason = request.unsupported_reason().or(match capability.status {
            AneCapabilityStatus::Available => None,
            AneCapabilityStatus::Unsupported { reason } => Some(reason),
        });
        if let Some(reason) = reason {
            if policy.requires_strict_ane() {
                return Err(RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "ane-planner",
                        op: reason.as_str(),
                    },
                    AppleCtx {
                        backend: "ane-planner",
                        op: "plan_ane_static_partitions",
                        device: "apple-silicon",
                    },
                ));
            }
            plan_backend = AnePlannedBackend::MetalFallback;
            partitions.push(AneStaticPartition {
                request: request.clone(),
                backend: AnePlannedBackend::MetalFallback,
                cache_key: None,
                fallback_reason: Some(reason),
            });
        } else {
            partitions.push(AneStaticPartition {
                request: request.clone(),
                backend: AnePlannedBackend::Ane,
                cache_key: Some(AneCompiledCacheKey::new(
                    model_hash,
                    request,
                    &capability.target,
                )),
                fallback_reason: None,
            });
        }
    }

    Ok(AneStaticPartitionPlan {
        backend: plan_backend,
        partitions,
    })
}

#[must_use]
pub(crate) fn stable_hash_hex(input: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in input {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[must_use]
fn hex32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleRuntimePlan {
    pub target: AppleAcceleratorTarget,
    pub mode: AppleBackendMode,
    pub rollout_bucket: Option<RolloutBucket>,
    pub rollout_tokens: u32,
    pub private_ane_opt_in: bool,
    pub strict_ane: bool,
    pub ane_compute_profile: AneComputeProfile,
    pub ane_fallback_policy: AneFallbackPolicy,
    pub ane_hidden_size: usize,
    pub ane_intermediate_size: usize,
    pub ane_num_layers: usize,
    pub model_layout_hash: [u8; 32],
    pub weights_path: Option<std::path::PathBuf>,
}

impl AppleRuntimePlan {
    pub fn validate(&self) -> Result<()> {
        if self.mode.requires_private_ane() && !self.private_ane_opt_in {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "private-ane",
                    op: "rollout",
                },
                self.ctx("validate"),
            ));
        }
        if self.strict_ane {
            if !self.mode.requires_private_ane() || !self.private_ane_opt_in {
                return Err(RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "private-ane",
                        op: "strict_ane_requires_private_ane",
                    },
                    self.ctx("validate"),
                ));
            }
            if !matches!(
                self.ane_compute_profile,
                AneComputeProfile::NeuralEngineOnly
            ) {
                return Err(RvllmError::apple(
                    AppleError::InvalidMil {
                        reason: "strict_ane requires AneComputeProfile::NeuralEngineOnly",
                    },
                    self.ctx("validate"),
                ));
            }
            if !matches!(self.ane_fallback_policy, AneFallbackPolicy::FailFast) {
                return Err(RvllmError::apple(
                    AppleError::InvalidMil {
                        reason: "strict_ane requires AneFallbackPolicy::FailFast",
                    },
                    self.ctx("validate"),
                ));
            }
        }
        if self.private_ane_opt_in && self.ane_num_layers == 0 {
            return Err(RvllmError::apple(
                AppleError::InvalidMil {
                    reason: "ane_num_layers must be >= 1",
                },
                self.ctx("validate"),
            ));
        }
        if self.private_ane_opt_in && self.ane_hidden_size == 0 {
            return Err(RvllmError::apple(
                AppleError::InvalidMil {
                    reason: "ane_hidden_size must be >= 1",
                },
                self.ctx("validate"),
            ));
        }
        if self.private_ane_opt_in && self.ane_intermediate_size == 0 {
            return Err(RvllmError::apple(
                AppleError::InvalidMil {
                    reason: "ane_intermediate_size must be >= 1",
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
    use rvllm_core::DType;

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
    fn ane_mode_requires_private_opt_in() {
        let plan = AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillAneFfnRollout,
            rollout_bucket: Some(RolloutBucket { seqs: 8, tokens: 4 }),
            rollout_tokens: 4,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: AneComputeProfile::AnyAvailable,
            ane_fallback_policy: AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 1,
            ane_intermediate_size: 1,
            ane_num_layers: 1,
            model_layout_hash: [0u8; 32],
            weights_path: None,
        };
        assert!(plan.validate().is_err());
    }

    fn available_ane_report() -> AneCapabilityReport {
        let target = AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1);
        AneCapabilityReport {
            path: AneCapabilityPath::PrivateAne,
            status: AneCapabilityStatus::Available,
            target,
            private_ane_feature_enabled: true,
            private_ane_env_opt_in: true,
            compute_plan: CoreMlAneComputePlan::from_profile(
                AneComputeProfile::NeuralEngineOnly,
                true,
            ),
        }
    }

    fn unavailable_ane_report(reason: AneUnsupportedReason) -> AneCapabilityReport {
        let target = AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1);
        AneCapabilityReport {
            path: AneCapabilityPath::PrivateAne,
            status: AneCapabilityStatus::Unsupported { reason },
            target,
            private_ane_feature_enabled: false,
            private_ane_env_opt_in: false,
            compute_plan: CoreMlAneComputePlan::from_profile(
                AneComputeProfile::NeuralEngineOnly,
                true,
            ),
        }
    }

    #[test]
    fn ane_cache_key_is_deterministic_and_scoped() {
        let target = AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1);
        let request = AnePartitionRequest::ffn(
            AneLayerRange { start: 2, end: 6 },
            RolloutBucket { seqs: 8, tokens: 4 },
            DType::F16,
        );
        let key_a = AneCompiledCacheKey::new([7u8; 32], &request, &target);
        let key_b = AneCompiledCacheKey::new([7u8; 32], &request, &target);
        let mut changed_request = request.clone();
        changed_request.bucket = RolloutBucket {
            seqs: 16,
            tokens: 4,
        };
        let key_c = AneCompiledCacheKey::new([7u8; 32], &changed_request, &target);

        assert_eq!(key_a.deterministic_input(), key_b.deterministic_input());
        assert_eq!(key_a.stable_id(), key_b.stable_id());
        assert_ne!(key_a.deterministic_input(), key_c.deterministic_input());
        assert!(key_a
            .deterministic_input()
            .contains("op_kind=ffn_gate_up_down"));
        assert!(key_a.deterministic_input().contains("bucket_seqs=8"));
    }

    #[test]
    fn ane_partition_selection_models_dense_blocks() {
        let requests = vec![
            AnePartitionRequest::ffn(
                AneLayerRange { start: 0, end: 4 },
                RolloutBucket { seqs: 4, tokens: 4 },
                DType::F16,
            ),
            AnePartitionRequest::lm_head(
                AneLayerRange { start: 4, end: 5 },
                RolloutBucket { seqs: 4, tokens: 4 },
                DType::F16,
                AneVocabBlock {
                    start: 0,
                    len: 4096,
                },
            ),
            AnePartitionRequest::qkv(
                AneLayerRange { start: 0, end: 1 },
                RolloutBucket { seqs: 4, tokens: 4 },
                DType::F16,
            ),
            AnePartitionRequest::output_projection(
                AneLayerRange { start: 0, end: 1 },
                RolloutBucket { seqs: 4, tokens: 4 },
                DType::F16,
            ),
        ];

        let plan = plan_ane_static_partitions(
            &requests,
            &available_ane_report(),
            AnePartitionPolicy::strict(),
            [1u8; 32],
        )
        .unwrap();

        assert_eq!(plan.backend, AnePlannedBackend::Ane);
        assert_eq!(plan.partitions.len(), 4);
        assert!(plan
            .partitions
            .iter()
            .all(|partition| partition.backend == AnePlannedBackend::Ane));
        assert!(plan
            .partitions
            .iter()
            .all(|partition| partition.cache_key.is_some()));
    }

    #[test]
    fn strict_ane_partition_unavailable_fails_fast() {
        let requests = [AnePartitionRequest::ffn(
            AneLayerRange { start: 0, end: 4 },
            RolloutBucket { seqs: 4, tokens: 4 },
            DType::F16,
        )];

        let err = plan_ane_static_partitions(
            &requests,
            &unavailable_ane_report(AneUnsupportedReason::PrivateAneEnvOptInMissing),
            AnePartitionPolicy::strict(),
            [1u8; 32],
        );

        assert!(err.is_err());
    }

    #[test]
    fn allow_metal_policy_selects_fallback_plan_when_ane_unavailable() {
        let requests = [AnePartitionRequest::ffn(
            AneLayerRange { start: 0, end: 4 },
            RolloutBucket { seqs: 4, tokens: 4 },
            DType::F16,
        )];

        let plan = plan_ane_static_partitions(
            &requests,
            &unavailable_ane_report(AneUnsupportedReason::PrivateAneEnvOptInMissing),
            AnePartitionPolicy::allow_metal(),
            [1u8; 32],
        )
        .unwrap();

        assert_eq!(plan.backend, AnePlannedBackend::MetalFallback);
        assert_eq!(plan.partitions[0].backend, AnePlannedBackend::MetalFallback);
        assert_eq!(
            plan.partitions[0].fallback_reason,
            Some(AneUnsupportedReason::PrivateAneEnvOptInMissing)
        );
        assert!(plan.partitions[0].cache_key.is_none());
    }

    #[test]
    fn capability_probe_reports_unsupported_reason_without_private_api() {
        let target = AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1);
        let report = probe_ane_capability(&target, AneComputeProfile::AnyAvailable, false);

        assert!(!report.is_available());
        assert!(matches!(
            report.status,
            AneCapabilityStatus::Unsupported { .. }
        ));
        assert!(!report.compute_plan.requires_private_ane);
    }
}
