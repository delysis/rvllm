//! `RuntimeConfigBuilder` — the only path to construct a `RuntimeConfig`.
//! Accumulates every invalid field into `ConfigError::Inconsistent` so the
//! caller sees all problems at once, not the first one.

use std::path::PathBuf;

use crate::error::{ConfigError, Result, RvllmError};

use super::model::ModelConfig;
use super::runtime::{
    AneComputeProfile, AneFallbackPolicy, AppleBackendMode, AppleRolloutBucket,
    AppleRolloutBucketPolicy, GraphMode, LogLevel, PreemptionMode, RuntimeConfig,
};

#[derive(Default)]
pub struct RuntimeConfigBuilder {
    device_id: Option<u32>,
    max_batch: Option<u32>,
    max_context: Option<u32>,
    kv_block_size: Option<u32>,
    num_gpu_blocks: Option<u32>,
    num_cpu_blocks: Option<u32>,
    gpu_memory_utilization: Option<f32>,
    fp8_weights: Option<bool>,
    fp8_kv_cache: Option<bool>,
    graph_capture: Option<GraphMode>,
    preemption: Option<PreemptionMode>,
    log_level: Option<LogLevel>,
    kernel_dir: Option<PathBuf>,
    apple_backend_mode: Option<AppleBackendMode>,
    apple_private_ane_opt_in: Option<bool>,
    strict_ane: Option<bool>,
    ane_compute_profile: Option<AneComputeProfile>,
    ane_fallback_policy: Option<AneFallbackPolicy>,
    ane_hidden_size: Option<usize>,
    ane_intermediate_size: Option<usize>,
    ane_num_layers: Option<usize>,
    model_layout_hash: Option<[u8; 32]>,
    apple_rollout_tokens: Option<u32>,
    apple_rollout_bucket_policy: Option<AppleRolloutBucketPolicy>,
    apple_rollout_bucket: Option<AppleRolloutBucket>,
    weights_path: Option<PathBuf>,
}

macro_rules! setter {
    ($name:ident, $ty:ty) => {
        pub fn $name(mut self, v: $ty) -> Self {
            self.$name = Some(v);
            self
        }
    };
}

impl RuntimeConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    setter!(device_id, u32);
    setter!(max_batch, u32);
    setter!(max_context, u32);
    setter!(kv_block_size, u32);
    setter!(num_gpu_blocks, u32);
    setter!(num_cpu_blocks, u32);
    setter!(gpu_memory_utilization, f32);
    setter!(fp8_weights, bool);
    setter!(fp8_kv_cache, bool);
    setter!(graph_capture, GraphMode);
    setter!(preemption, PreemptionMode);
    setter!(log_level, LogLevel);
    setter!(apple_backend_mode, AppleBackendMode);
    setter!(apple_private_ane_opt_in, bool);
    setter!(strict_ane, bool);
    setter!(ane_compute_profile, AneComputeProfile);
    setter!(ane_fallback_policy, AneFallbackPolicy);
    setter!(ane_hidden_size, usize);
    setter!(ane_intermediate_size, usize);
    setter!(ane_num_layers, usize);
    setter!(model_layout_hash, [u8; 32]);
    setter!(apple_rollout_tokens, u32);
    setter!(apple_rollout_bucket_policy, AppleRolloutBucketPolicy);
    setter!(apple_rollout_bucket, AppleRolloutBucket);
    setter!(weights_path, PathBuf);

    pub fn kernel_dir(mut self, p: PathBuf) -> Self {
        self.kernel_dir = Some(p);
        self
    }

    pub fn build(self, model: &ModelConfig) -> Result<RuntimeConfig> {
        let mut reasons: Vec<String> = Vec::new();
        macro_rules! req {
            ($field:ident) => {
                match self.$field {
                    Some(v) => Some(v),
                    None => {
                        reasons.push(concat!(stringify!($field), " is required").into());
                        None
                    }
                }
            };
        }
        let device_id = req!(device_id);
        let max_batch = req!(max_batch);
        let max_context = req!(max_context);
        let kv_block_size = req!(kv_block_size);
        let num_gpu_blocks = req!(num_gpu_blocks);
        let num_cpu_blocks = req!(num_cpu_blocks);
        let gpu_memory_utilization = req!(gpu_memory_utilization);
        let fp8_weights = req!(fp8_weights);
        let fp8_kv_cache = req!(fp8_kv_cache);
        let graph_capture = req!(graph_capture);
        let preemption = req!(preemption);
        let apple_backend_mode = self
            .apple_backend_mode
            .unwrap_or(AppleBackendMode::Disabled);
        let apple_private_ane_opt_in = self.apple_private_ane_opt_in.unwrap_or(false);
        let apple_rollout_tokens = self.apple_rollout_tokens.unwrap_or(1);
        let apple_rollout_bucket_policy = self.apple_rollout_bucket_policy.unwrap_or_default();
        let mut apple_rollout_bucket = self.apple_rollout_bucket;
        let strict_ane = self.strict_ane.unwrap_or(false);
        let ane_compute_profile = self.ane_compute_profile.unwrap_or(if strict_ane {
            AneComputeProfile::NeuralEngineOnly
        } else {
            AneComputeProfile::AnyAvailable
        });
        let ane_fallback_policy = self.ane_fallback_policy.unwrap_or(if strict_ane {
            AneFallbackPolicy::FailFast
        } else {
            AneFallbackPolicy::AllowMetal
        });
        let ane_hidden_size = self.ane_hidden_size.unwrap_or(model.hidden_size);
        let ane_intermediate_size = self
            .ane_intermediate_size
            .unwrap_or(model.intermediate_size);
        let ane_num_layers = self.ane_num_layers.unwrap_or(model.num_layers);
        let model_layout_hash = self.model_layout_hash.unwrap_or_else(|| {
            let mut h: u64 = 0xcbf29ce484222325u64;
            fn mix(mut h: u64, bytes: &[u8]) -> u64 {
                for b in bytes {
                    h ^= u64::from(*b);
                    h = h.wrapping_mul(0x100000001b3);
                }
                h
            }
            h = mix(
                h,
                format!(
                    "arch={:?};hidden={};intermediate={};layers={};kv={};heads={};head_dim={};max_pos={};dtype={:?}",
                    model.architecture,
                    model.hidden_size,
                    model.intermediate_size,
                    model.num_layers,
                    model.num_kv_heads,
                    model.num_attention_heads,
                    model.head_dim,
                    model.max_position_embeddings,
                    model.torch_dtype,
                )
                .as_bytes(),
            );
            h = mix(h, format!("ane:{ane_hidden_size}:{ane_intermediate_size}:{ane_num_layers}").as_bytes());
            let mut out = [0u8; 32];
            out[..8].copy_from_slice(&h.to_le_bytes());
            out[8..16].copy_from_slice(&h.rotate_left(13).to_le_bytes());
            out[16..24].copy_from_slice(&h.rotate_left(29).to_le_bytes());
            out[24..32].copy_from_slice(&h.rotate_left(47).to_le_bytes());
            out
        });
        let requires_private_ane = apple_backend_mode.requires_private_ane();

        if apple_rollout_tokens == 0 {
            reasons.push("apple_rollout_tokens must be >= 1".into());
        }
        if strict_ane && !requires_private_ane {
            reasons
                .push("strict_ane requires an Apple backend mode that enables ANE rollout".into());
        }
        if strict_ane && !apple_private_ane_opt_in {
            reasons.push("strict_ane requires apple_private_ane_opt_in=true".into());
        }
        if strict_ane && !matches!(ane_fallback_policy, AneFallbackPolicy::FailFast) {
            reasons.push("strict_ane requires ane_fallback_policy=FailFast".into());
        }
        if strict_ane && !matches!(ane_compute_profile, AneComputeProfile::NeuralEngineOnly) {
            reasons.push("strict_ane requires ane_compute_profile=NeuralEngineOnly".into());
        }
        if requires_private_ane && !apple_private_ane_opt_in {
            reasons.push(format!(
                "apple_private_ane_opt_in must be true when apple_backend_mode={apple_backend_mode:?}"
            ));
        }
        if !requires_private_ane {
            if !matches!(apple_rollout_bucket_policy, AppleRolloutBucketPolicy::Auto) {
                reasons.push("apple_rollout_bucket_policy must be Auto unless private ANE rollout is enabled".into());
            }
            if apple_private_ane_opt_in {
                reasons.push(
                    "apple_private_ane_opt_in requires a private ANE mode (MetalPrefillAne* )"
                        .into(),
                );
            }
            if apple_rollout_bucket.is_some() {
                reasons.push(
                    "apple_rollout_bucket is only valid for private ANE rollout modes".into(),
                );
            }
            if apple_rollout_tokens != 1 {
                reasons.push(
                    "apple_rollout_tokens must be 1 unless private ANE mode is enabled".into(),
                );
            }
        }
        if matches!(
            apple_rollout_bucket_policy,
            AppleRolloutBucketPolicy::Fixed { seqs: 0, tokens: _ }
        ) {
            reasons.push("apple_rollout_bucket_policy.Fixed.seqs must be >= 1".into());
        }
        if matches!(
            apple_rollout_bucket_policy,
            AppleRolloutBucketPolicy::Fixed { seqs: _, tokens: 0 }
        ) {
            reasons.push("apple_rollout_bucket_policy.Fixed.tokens must be >= 1".into());
        }
        if let AppleRolloutBucketPolicy::Fixed { seqs, tokens } = apple_rollout_bucket_policy {
            if !requires_private_ane {
                reasons.push(
                    "fixed apple_rollout_bucket_policy is invalid without private ANE rollout"
                        .into(),
                );
            }
            if let Some(requested) = apple_rollout_bucket {
                if requested.seqs != seqs || requested.tokens != tokens {
                    reasons.push(format!(
                        "apple_rollout_bucket {:?} does not match fixed policy (seqs={seqs}, tokens={tokens})",
                        requested
                    ));
                }
            }
            apple_rollout_bucket = Some(AppleRolloutBucket { seqs, tokens });
        }

        if requires_private_ane && apple_rollout_bucket.is_none() {
            reasons.push(
                "apple_rollout_bucket is required when apple_backend_mode uses private ANE".into(),
            );
        }
        if let Some(requested) = apple_rollout_bucket {
            if requested.seqs == 0 || requested.tokens == 0 {
                reasons.push("apple_rollout_bucket requires positive seqs and tokens".into());
            } else if !requested.capacity_ge(apple_rollout_tokens) {
                reasons.push(format!(
                    "apple_rollout_bucket.tokens must be >= apple_rollout_tokens ({apple_rollout_tokens}), got {}",
                    requested.tokens
                ));
            }
            if requested.seqs < 1 {
                reasons.push("apple_rollout_bucket.seqs must be >= 1".into());
            }
        }

        if let Some(v) = kv_block_size {
            if ![16u32, 32, 64].contains(&v) {
                reasons.push(format!("kv_block_size must be 16|32|64, got {v}"));
            }
        }
        if let Some(v) = max_batch {
            if !(1..=256).contains(&v) {
                reasons.push(format!("max_batch must be in 1..=256, got {v}"));
            }
            if v == 0 {
                reasons.push("max_batch must be in 1..=256".into());
            }
        }
        if let Some(ctx) = max_context {
            if ctx as usize > model.max_position_embeddings {
                reasons.push(format!(
                    "max_context {ctx} > model.max_position_embeddings {}",
                    model.max_position_embeddings
                ));
            } else if ctx == 0 {
                reasons.push("max_context must be >= 1".into());
            }
        }
        if let (Some(v), Some(ctx), Some(block), Some(nbg), Some(ncb)) = (
            max_batch,
            max_context,
            kv_block_size,
            num_gpu_blocks,
            num_cpu_blocks,
        ) {
            let min_blocks =
                ((v as u64 * ctx as u64) + (block as u64).saturating_sub(1)) / block as u64;
            if (nbg as u64 + ncb as u64) < min_blocks {
                reasons.push(format!(
                    "num_gpu_blocks + num_cpu_blocks must be >= ceil(max_batch*max_context / kv_block_size) = {min_blocks}, got {}",
                    nbg + ncb
                ));
            }
        }
        if let Some(u) = gpu_memory_utilization {
            if !(u > 0.0 && u <= 0.95) {
                reasons.push(format!(
                    "gpu_memory_utilization must be in (0.0, 0.95], got {u}"
                ));
            }
        }
        if let (Some(true), Some(bs)) = (fp8_kv_cache, kv_block_size) {
            if bs < 32 {
                reasons.push("fp8_kv_cache requires kv_block_size >= 32".into());
            }
        }
        if ane_hidden_size == 0 {
            reasons.push("ane_hidden_size must be >= 1".into());
        }
        if ane_intermediate_size == 0 {
            reasons.push("ane_intermediate_size must be >= 1".into());
        }
        if ane_num_layers == 0 {
            reasons.push("ane_num_layers must be >= 1".into());
        }
        if ane_hidden_size > 0 && ane_hidden_size != model.hidden_size {
            reasons.push("ane_hidden_size should match model.hidden_size".into());
        }
        if ane_intermediate_size > 0 && ane_intermediate_size != model.intermediate_size {
            reasons.push("ane_intermediate_size should match model.intermediate_size".into());
        }
        if ane_num_layers > 0 && ane_num_layers != model.num_layers {
            reasons.push("ane_num_layers should match model.num_layers".into());
        }

        if !reasons.is_empty() {
            return Err(RvllmError::config(
                ConfigError::Inconsistent { reasons },
                "RuntimeConfig::build",
            ));
        }

        // All required present and validated — safe to unwrap.
        #[allow(clippy::unwrap_used)]
        Ok(RuntimeConfig {
            device_id: device_id.unwrap(),
            max_batch: max_batch.unwrap(),
            max_context: max_context.unwrap(),
            kv_block_size: kv_block_size.unwrap(),
            num_gpu_blocks: num_gpu_blocks.unwrap(),
            num_cpu_blocks: num_cpu_blocks.unwrap(),
            gpu_memory_utilization: gpu_memory_utilization.unwrap(),
            fp8_weights: fp8_weights.unwrap(),
            fp8_kv_cache: fp8_kv_cache.unwrap(),
            graph_capture: graph_capture.unwrap(),
            preemption: preemption.unwrap(),
            log_level: self.log_level.unwrap_or(LogLevel::Info),
            kernel_dir: self.kernel_dir,
            apple_backend_mode,
            apple_private_ane_opt_in,
            apple_rollout_tokens,
            apple_rollout_bucket_policy,
            apple_rollout_bucket,
            strict_ane,
            ane_compute_profile,
            ane_fallback_policy,
            ane_hidden_size,
            ane_intermediate_size,
            ane_num_layers,
            model_layout_hash,
            weights_path: self.weights_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelArch, ModelConfig};
    use crate::dtype::DType;

    fn qwen() -> ModelConfig {
        ModelConfig {
            architecture: ModelArch::Qwen2,
            hidden_size: 3584,
            num_layers: 28,
            num_attention_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 18944,
            vocab_size: 152064,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
            torch_dtype: DType::Bf16,
        }
    }

    #[test]
    fn rejects_missing_fields() {
        let err = RuntimeConfigBuilder::new()
            .device_id(0)
            .max_batch(128)
            .build(&qwen())
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("max_context is required"));
        assert!(s.contains("kv_block_size is required"));
    }

    #[test]
    fn rejects_bad_block_size() {
        let err = RuntimeConfigBuilder::new()
            .device_id(0)
            .max_batch(128)
            .max_context(2048)
            .kv_block_size(48)
            .num_gpu_blocks(1024)
            .num_cpu_blocks(0)
            .gpu_memory_utilization(0.9)
            .fp8_weights(true)
            .fp8_kv_cache(false)
            .graph_capture(GraphMode::Off)
            .preemption(PreemptionMode::Recompute)
            .build(&qwen())
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("kv_block_size must be 16|32|64"));
    }

    #[test]
    fn happy_path() {
        let rt = RuntimeConfigBuilder::new()
            .device_id(0)
            .max_batch(128)
            .max_context(2048)
            .kv_block_size(64)
            .num_gpu_blocks(4096)
            .num_cpu_blocks(0)
            .gpu_memory_utilization(0.9)
            .fp8_weights(true)
            .fp8_kv_cache(false)
            .graph_capture(GraphMode::Buckets(vec![1, 2, 4, 8, 16, 32, 64, 128]))
            .preemption(PreemptionMode::Recompute)
            .apple_backend_mode(AppleBackendMode::MetalPrefillMetalDecode)
            .apple_rollout_tokens(1)
            .build(&qwen())
            .unwrap();
        assert_eq!(rt.max_batch(), 128);
        assert_eq!(rt.kv_block_size(), 64);
    }

    #[test]
    fn rejects_private_ane_without_opt_in() {
        let err = RuntimeConfigBuilder::new()
            .device_id(0)
            .max_batch(128)
            .max_context(2048)
            .kv_block_size(64)
            .num_gpu_blocks(1024)
            .num_cpu_blocks(0)
            .gpu_memory_utilization(0.9)
            .fp8_weights(true)
            .fp8_kv_cache(false)
            .graph_capture(GraphMode::Off)
            .preemption(PreemptionMode::Recompute)
            .apple_backend_mode(AppleBackendMode::MetalPrefillAneFfnRollout)
            .apple_private_ane_opt_in(false)
            .build(&qwen())
            .unwrap_err();
        assert!(format!("{err}").contains("apple_private_ane_opt_in"));
    }

    #[test]
    fn validates_fixed_policy_bucket() {
        let err = RuntimeConfigBuilder::new()
            .device_id(0)
            .max_batch(128)
            .max_context(2048)
            .kv_block_size(64)
            .num_gpu_blocks(1024)
            .num_cpu_blocks(0)
            .gpu_memory_utilization(0.9)
            .fp8_weights(true)
            .fp8_kv_cache(false)
            .graph_capture(GraphMode::Off)
            .preemption(PreemptionMode::Recompute)
            .apple_backend_mode(AppleBackendMode::MetalPrefillAneRolloutExperimental)
            .apple_private_ane_opt_in(true)
            .apple_rollout_bucket_policy(AppleRolloutBucketPolicy::Fixed { seqs: 4, tokens: 2 })
            .apple_rollout_bucket(AppleRolloutBucket { seqs: 8, tokens: 2 })
            .build(&qwen())
            .unwrap_err();
        assert!(format!("{err}").contains("does not match fixed policy"));
    }
}
