use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
use serde::{Deserialize, Serialize};

use crate::iosurface::IoSurfaceTensorDesc;
use crate::plan::RolloutBucket;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneRolloutConfig {
    pub bucket: RolloutBucket,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
}

impl AneRolloutConfig {
    #[must_use]
    pub fn activation_desc(&self) -> IoSurfaceTensorDesc {
        IoSurfaceTensorDesc {
            dtype: rvllm_core::DType::F16,
            channels: self.hidden_size,
            spatial: (self.bucket.seqs * self.bucket.tokens) as usize,
        }
    }

    #[must_use]
    pub fn activation_bytes(&self) -> usize {
        self.activation_desc().bytes()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AneProcedure {
    FusedFfn { layer: usize },
    FusedQkv { layer: usize },
    LmHead,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneProgramPlan {
    pub config: AneRolloutConfig,
    pub procedures: Vec<AneProcedure>,
}

impl AneProgramPlan {
    #[must_use]
    pub fn ffn_only(config: AneRolloutConfig) -> Self {
        let procedures = (0..config.num_layers)
            .map(|layer| AneProcedure::FusedFfn { layer })
            .chain(std::iter::once(AneProcedure::LmHead))
            .collect();
        Self { config, procedures }
    }

    #[must_use]
    pub fn qkv_ffn_lm_head(config: AneRolloutConfig) -> Self {
        let mut procedures = Vec::with_capacity(config.num_layers * 2 + 1);
        for layer in 0..config.num_layers {
            procedures.push(AneProcedure::FusedQkv { layer });
            procedures.push(AneProcedure::FusedFfn { layer });
        }
        procedures.push(AneProcedure::LmHead);
        Self { config, procedures }
    }

    #[must_use]
    pub fn num_procedures(&self) -> usize {
        self.procedures.len()
    }
}

pub fn compile_private_ane_program(_plan: &AneProgramPlan) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        rvllm_apple_ane_sys::ffi::load_ane_framework().map_err(|_| {
            RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "private-ane",
                    op: "dlopen_ane_framework",
                },
                AppleCtx {
                    backend: "private-ane",
                    op: "compile",
                    device: "apple-silicon",
                },
            )
        })?;
        
        let _client = rvllm_apple_ane_sys::ffi::get_ane_client().ok_or_else(|| {
            RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "private-ane",
                    op: "get_ane_client",
                },
                AppleCtx {
                    backend: "private-ane",
                    op: "compile",
                    device: "apple-silicon",
                },
            )
        })?;

        // TODO: actually construct .mlpackage from `mil::fused_ffn_mil` bytes, compile it, and load it
        return Ok(());
    }

    #[cfg(not(target_os = "macos"))]
    Err(RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "private-ane",
            op: "compile_private_ane_program",
        },
        AppleCtx {
            backend: "private-ane",
            op: "compile",
            device: "non-apple",
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffn_only_program_has_one_proc_per_layer_plus_lm_head() {
        let config = AneRolloutConfig {
            bucket: RolloutBucket { seqs: 8, tokens: 4 },
            hidden_size: 2048,
            intermediate_size: 6144,
            num_layers: 24,
        };
        let plan = AneProgramPlan::ffn_only(config);
        assert_eq!(plan.num_procedures(), 25);
        assert_eq!(plan.config.activation_bytes(), 8 * 4 * 2048 * 2);
    }

    #[test]
    fn qkv_ffn_program_has_two_procs_per_layer_plus_lm_head() {
        let config = AneRolloutConfig {
            bucket: RolloutBucket { seqs: 4, tokens: 1 },
            hidden_size: 1024,
            intermediate_size: 2816,
            num_layers: 28,
        };
        let plan = AneProgramPlan::qkv_ffn_lm_head(config);
        assert_eq!(plan.num_procedures(), 57);
    }
}
