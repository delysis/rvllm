use rvllm_core::config::{AneComputeProfile, AneFallbackPolicy};
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
}
