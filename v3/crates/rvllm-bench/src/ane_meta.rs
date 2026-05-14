//! Shared benchmark/probe CLI metadata for strict-ANE and Apple backend
//! intent capture.

use rvllm_core::{AneComputeProfile, AneFallbackPolicy, AppleBackendMode};
use serde_json::Value;

use std::fmt;
use std::path::PathBuf;
#[derive(Clone, Debug)]
pub struct AppleCliProfile {
    pub backend_profile: String,
    pub apple_mode: Option<AppleBackendMode>,
    pub strict_ane: bool,
    pub private_ane_opt_in: bool,
    pub ane_compute_profile: AneComputeProfile,
    pub ane_fallback_policy: AneFallbackPolicy,
    pub apple_rollout_tokens: u32,
    pub rollout_bucket_seqs: Option<u32>,
    pub rollout_bucket_tokens: Option<u32>,
    pub model_layout_hash: Option<String>,
    pub compile_cache_key: Option<String>,
    pub compile_cache_hit: Option<bool>,
    pub compile_ms: Option<u128>,
    pub compile_reason: Option<String>,
    pub log_dir: Option<PathBuf>,
    pub peer_cuda: Option<Value>,
    pub peer_xla: Option<Value>,
}

#[derive(Clone, Copy, Debug)]
pub enum BackendProfile {
    Cuda,
    Apple,
    Xla,
    Unknown,
}

impl fmt::Display for BackendProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = match self {
            BackendProfile::Cuda => "cuda",
            BackendProfile::Apple => "apple",
            BackendProfile::Xla => "xla",
            BackendProfile::Unknown => "unknown",
        };
        f.write_str(v)
    }
}

impl AppleCliProfile {
    pub fn from_env() -> Self {
        let backend_profile = std::env::var("RVLLM_BACKEND_PROFILE")
            .unwrap_or_else(|_| "cuda".into())
            .to_lowercase();
        let backend = match backend_profile.as_str() {
            "apple" => BackendProfile::Apple,
            "xla" => BackendProfile::Xla,
            "cuda" => BackendProfile::Cuda,
            _ => BackendProfile::Unknown,
        };

        let strict_ane = env_bool("RVLLM_STRICT_ANE");
        let private_ane_opt_in = env_bool("RVLLM_APPLE_PRIVATE_ANE");
        let rollout_tokens = std::env::var("RVLLM_APPLE_ROLLOUT_TOKENS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(1);

        let rollout_bucket_seqs = env_u32_opt("RVLLM_APPLE_BUCKET_SEQS");
        let rollout_bucket_tokens = env_u32_opt("RVLLM_APPLE_BUCKET_TOKENS");
        let compile_cache_hit = match std::env::var("RVLLM_APPLE_COMPILE_CACHE_HIT") {
            Ok(v) => parse_bool(&v),
            Err(_) => None,
        };
        let compile_ms = std::env::var("RVLLM_APPLE_COMPILE_MS")
            .ok()
            .and_then(|s| s.parse::<u128>().ok())
            .filter(|&ms| ms != 0)
            .map(u128::from)
            .or_else(|| {
                std::env::var("RVLLM_APPLE_COMPILE_SECONDS")
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|v| (v * 1000.0) as u128)
            });

        Self {
            backend_profile: backend.to_string(),
            apple_mode: env_apple_mode(),
            strict_ane,
            private_ane_opt_in: private_ane_opt_in.unwrap_or(strict_ane),
            ane_compute_profile: parse_ane_profile(),
            ane_fallback_policy: parse_ane_fallback(),
            apple_rollout_tokens: rollout_tokens,
            rollout_bucket_seqs,
            rollout_bucket_tokens,
            model_layout_hash: std::env::var("RVLLM_APPLE_LAYOUT_HASH").ok(),
            compile_cache_key: std::env::var("RVLLM_APPLE_COMPILE_CACHE_KEY").ok(),
            compile_cache_hit,
            compile_ms,
            compile_reason: std::env::var("RVLLM_APPLE_COMPILE_REASON").ok(),
            log_dir: std::env::var("RVLLM_BENCH_LOG_DIR").ok().map(PathBuf::from),
            peer_cuda: parse_side_by_side("RVLLM_SIDE_BY_SIDE_CUDA"),
            peer_xla: parse_side_by_side("RVLLM_SIDE_BY_SIDE_XLA"),
        }
    }

    pub fn backend(&self) -> BackendProfile {
        match self.backend_profile.as_str() {
            "apple" => BackendProfile::Apple,
            "xla" => BackendProfile::Xla,
            "cuda" => BackendProfile::Cuda,
            _ => BackendProfile::Unknown,
        }
    }

    pub fn is_strict_ane_mode(&self) -> bool {
        self.strict_ane || matches!(self.ane_fallback_policy, AneFallbackPolicy::FailFast)
    }

    pub fn apple_mode_label(&self) -> &'static str {
        match self.apple_mode {
            Some(AppleBackendMode::Disabled) => "disabled",
            Some(AppleBackendMode::MetalOnly) => "metal-only",
            Some(AppleBackendMode::MetalPrefillMetalDecode) => "metal-prefill-metal-decode",
            Some(AppleBackendMode::MetalPrefillAneFfnRollout) => "metal-prefill-ane-ffn-rollout",
            Some(AppleBackendMode::MetalPrefillAneRolloutExperimental) => {
                "metal-prefill-ane-rollout-experimental"
            }
            None => "not-configured",
        }
    }

    pub fn compact_summary(&self) -> String {
        format!(
            "backend={}; apple_mode={}; strict_ane={}; private_ane_opt_in={}; profile={}; fallback={}; rollout_tokens={}; bucket=({:?},{:?}); layout_hash={}; compile_key={}; compile_hit={:?}; compile_ms={:?}",
            self.backend(),
            self.apple_mode_label(),
            self.strict_ane,
            self.private_ane_opt_in,
            self.ane_compute_profile.as_str(),
            self.ane_fallback_policy,
            self.apple_rollout_tokens,
            self.rollout_bucket_seqs,
            self.rollout_bucket_tokens,
            self.model_layout_hash.as_deref().unwrap_or("-"),
            self.compile_cache_key.as_deref().unwrap_or("-"),
            self.compile_cache_hit,
            self.compile_ms
        )
    }

    pub fn compact_apple_object(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled": self.apple_mode.map(|_| true).unwrap_or(false),
            "mode": self.apple_mode_label(),
            "strict_ane": self.strict_ane,
            "private_ane_opt_in": self.private_ane_opt_in,
            "compute_profile": self.ane_compute_profile.as_str(),
            "fallback_policy": ane_fallback_label(self.ane_fallback_policy),
            "rollout_tokens": self.apple_rollout_tokens,
            "rollout_bucket": self.rollout_bucket_seqs.zip(self.rollout_bucket_tokens).map(|(seqs, tokens)| serde_json::json!({"seqs":seqs,"tokens":tokens})),
            "model_layout_hash": self.model_layout_hash,
            "compile_cache_key": self.compile_cache_key,
            "compile_cache_hit": self.compile_cache_hit,
            "compile_ms": self.compile_ms,
            "compile_reason": self.compile_reason,
        })
    }
}

fn parse_side_by_side(name: &str) -> Option<Value> {
    std::env::var(name).ok().and_then(|raw| serde_json::from_str(&raw).ok())
}


fn ane_fallback_label(policy: AneFallbackPolicy) -> &'static str {
    match policy {
        AneFallbackPolicy::FailFast => "fail-fast",
        AneFallbackPolicy::AllowMetal => "allow-metal",
        AneFallbackPolicy::AllowSoft => "allow-soft",
    }
}

fn parse_bool(v: &str) -> Option<bool> {
    match v.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "y" => Some(true),
        "0" | "false" | "no" | "off" | "n" => Some(false),
        _ => None,
    }
}

fn env_bool(name: &str) -> bool {
    std::env::var(name).ok().and_then(|v| parse_bool(&v)).unwrap_or(false)
}

fn env_u32_opt(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|s| s.parse::<u32>().ok())
}

fn parse_ane_profile() -> AneComputeProfile {
    let env = std::env::var("RVLLM_APPLE_ANE_PROFILE")
        .or_else(|_| std::env::var("RVLLM_ANE_PROFILE"))
        .unwrap_or_else(|_| "any".into())
        .to_lowercase();
    match env.as_str() {
        "neural_engine_only" | "neuralengineonly" | "ne" | "neuralengine" | "neural_engine" => {
            AneComputeProfile::NeuralEngineOnly
        }
        "neural_engine_preferred" | "preferred" | "neuralenginepreferred" => {
            AneComputeProfile::NeuralEnginePreferred
        }
        "any" | "any_available" | "anyavailable" => AneComputeProfile::AnyAvailable,
        _ => AneComputeProfile::AnyAvailable,
    }
}

fn parse_ane_fallback() -> AneFallbackPolicy {
    let env = std::env::var("RVLLM_APPLE_ANE_FALLBACK")
        .or_else(|_| std::env::var("RVLLM_ANE_FALLBACK"))
        .unwrap_or_else(|_| "allow_metal".into())
        .to_lowercase();
    match env.as_str() {
        "failfast" | "fail_fast" | "strict" | "true" => AneFallbackPolicy::FailFast,
        "allow_soft" | "allowsoft" | "soft" => AneFallbackPolicy::AllowSoft,
        "allow_metal" | "allowmetal" | "allow" | "false" => AneFallbackPolicy::AllowMetal,
        _ => AneFallbackPolicy::AllowMetal,
    }
}

fn env_apple_mode() -> Option<AppleBackendMode> {
    let mode = std::env::var("RVLLM_APPLE_MODE").ok()?.trim().to_lowercase();
    let parsed = match mode.as_str() {
        "disabled" | "off" => AppleBackendMode::Disabled,
        "metal-only" | "metal_only" | "metalonly" => AppleBackendMode::MetalOnly,
        "metal-prefill-metal-decode" | "metal_prefill_metal_decode" | "metalprefillmetaldecode" => {
            AppleBackendMode::MetalPrefillMetalDecode
        }
        "ane-ffn" | "ane-fn" | "metal-prefill-ane-ffn-rollout" | "metal_prefill_ane_ffn_rollout" => {
            AppleBackendMode::MetalPrefillAneFfnRollout
        }
        "ane-experimental" | "ane-exp" | "metal-prefill-ane-rollout-experimental" | "metal_prefill_ane_rollout_experimental" => {
            AppleBackendMode::MetalPrefillAneRolloutExperimental
        }
        _ => return None,
    };
    Some(parsed)
}
