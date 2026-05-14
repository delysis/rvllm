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
    RolloutBucket { seqs: 16, tokens: 1 },
    RolloutBucket { seqs: 32, tokens: 1 },
    RolloutBucket { seqs: 64, tokens: 1 },
    RolloutBucket { seqs: 128, tokens: 1 },
    RolloutBucket { seqs: 4, tokens: 4 },
    RolloutBucket { seqs: 8, tokens: 4 },
    RolloutBucket { seqs: 16, tokens: 4 },
    RolloutBucket { seqs: 32, tokens: 4 },
    RolloutBucket { seqs: 64, tokens: 4 },
    RolloutBucket { seqs: 8, tokens: 8 },
    RolloutBucket { seqs: 16, tokens: 8 },
    RolloutBucket { seqs: 32, tokens: 8 },
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
        .min_by_key(|bucket| (bucket.waste(seqs, tokens), bucket.capacity(), bucket.seqs, bucket.tokens))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleRuntimePlan {
    pub target: AppleAcceleratorTarget,
    pub mode: AppleBackendMode,
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
        assert_eq!(select_rollout_bucket(1, 1), Some(RolloutBucket { seqs: 1, tokens: 1 }));
        assert_eq!(select_rollout_bucket(3, 1), Some(RolloutBucket { seqs: 4, tokens: 1 }));
        assert_eq!(select_rollout_bucket(3, 4), Some(RolloutBucket { seqs: 4, tokens: 4 }));
        assert_eq!(select_rollout_bucket(9, 4), Some(RolloutBucket { seqs: 16, tokens: 4 }));
        assert_eq!(select_rollout_bucket(33, 8), None);
    }

    #[test]
    fn rollout_bucket_selection_is_not_first_fit() {
        let buckets = [
            RolloutBucket { seqs: 32, tokens: 4 },
            RolloutBucket { seqs: 8, tokens: 4 },
            RolloutBucket { seqs: 16, tokens: 4 },
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
}
