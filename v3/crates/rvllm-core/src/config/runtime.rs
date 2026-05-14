//! Frozen runtime configuration. Only constructible via
//! `RuntimeConfigBuilder::build(&model)` in `builder.rs`.

use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AppleBackendMode {
    Disabled,
    MetalOnly,
    MetalPrefillMetalDecode,
    MetalPrefillAneFfnRollout,
    MetalPrefillAneRolloutExperimental,
}

impl AppleBackendMode {
    #[must_use]
    pub const fn requires_private_ane(self) -> bool {
        matches!(
            self,
            Self::MetalPrefillAneFfnRollout | Self::MetalPrefillAneRolloutExperimental
        )
    }

    #[must_use]
    pub const fn requires_rollout(self) -> bool {
        matches!(self, Self::MetalPrefillAneFfnRollout | Self::MetalPrefillAneRolloutExperimental)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AppleRolloutBucket {
    pub seqs: u32,
    pub tokens: u32,
}

impl AppleRolloutBucket {
    #[must_use]
    pub const fn fits(self, seqs: u32, tokens: u32) -> bool {
        self.seqs >= seqs && self.tokens >= tokens
    }

    #[must_use]
    pub const fn capacity_ge(self, tokens: u32) -> bool {
        self.tokens >= tokens
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AppleRolloutBucketPolicy {
    /// Let runtime choose the smallest non-padding bucket for active sequences.
    Auto,
    /// Pin runtime to a fixed bucket. `tokens` is per-rollout token count.
    Fixed { seqs: u32, tokens: u32 },
}

impl Default for AppleRolloutBucket {
    fn default() -> Self {
        Self { seqs: 1, tokens: 1 }
    }
}

impl Default for AppleRolloutBucketPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PreemptionMode {
    Recompute,
    Swap,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphMode {
    Off,
    Buckets(Vec<u32>),
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

/// Validated runtime configuration. Fields private to the config module
/// so callers can't skip the builder via struct-literal construction.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub(super) device_id: u32,
    pub(super) max_batch: u32,
    pub(super) max_context: u32,
    pub(super) kv_block_size: u32,
    pub(super) num_gpu_blocks: u32,
    pub(super) num_cpu_blocks: u32,
    pub(super) gpu_memory_utilization: f32,
    pub(super) fp8_weights: bool,
    pub(super) fp8_kv_cache: bool,
    pub(super) graph_capture: GraphMode,
    pub(super) preemption: PreemptionMode,
    pub(super) log_level: LogLevel,
    pub(super) kernel_dir: Option<PathBuf>,
    pub(super) apple_backend_mode: AppleBackendMode,
    pub(super) apple_private_ane_opt_in: bool,
    pub(super) apple_rollout_tokens: u32,
    pub(super) apple_rollout_bucket_policy: AppleRolloutBucketPolicy,
    pub(super) apple_rollout_bucket: Option<AppleRolloutBucket>,
    pub(super) weights_path: Option<PathBuf>,
}

impl RuntimeConfig {
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    pub fn max_batch(&self) -> u32 {
        self.max_batch
    }
    pub fn max_context(&self) -> u32 {
        self.max_context
    }
    pub fn kv_block_size(&self) -> u32 {
        self.kv_block_size
    }
    pub fn num_gpu_blocks(&self) -> u32 {
        self.num_gpu_blocks
    }
    pub fn num_cpu_blocks(&self) -> u32 {
        self.num_cpu_blocks
    }
    pub fn gpu_memory_utilization(&self) -> f32 {
        self.gpu_memory_utilization
    }
    pub fn fp8_weights(&self) -> bool {
        self.fp8_weights
    }
    pub fn fp8_kv_cache(&self) -> bool {
        self.fp8_kv_cache
    }
    pub fn graph_capture(&self) -> &GraphMode {
        &self.graph_capture
    }
    pub fn preemption(&self) -> PreemptionMode {
        self.preemption
    }
    pub fn log_level(&self) -> LogLevel {
        self.log_level
    }
    pub fn kernel_dir(&self) -> Option<&Path> {
        self.kernel_dir.as_deref()
    }

    pub fn apple_backend_mode(&self) -> AppleBackendMode {
        self.apple_backend_mode
    }

    pub fn apple_private_ane_opt_in(&self) -> bool {
        self.apple_private_ane_opt_in
    }

    pub fn apple_rollout_tokens(&self) -> u32 {
        self.apple_rollout_tokens
    }

    pub fn apple_rollout_bucket_policy(&self) -> AppleRolloutBucketPolicy {
        self.apple_rollout_bucket_policy
    }

    pub fn apple_rollout_bucket(&self) -> Option<AppleRolloutBucket> {
        self.apple_rollout_bucket
    }
    pub fn weights_path(&self) -> Option<&Path> {
        self.weights_path.as_deref()
    }
}
