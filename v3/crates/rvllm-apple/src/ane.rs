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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AneLayerProcedureIndices {
    pub qkv: Option<usize>,
    pub ffn: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneProgramPlan {
    pub config: AneRolloutConfig,
    pub procedures: Vec<AneProcedure>,
    pub layer_procedure_indices: Vec<AneLayerProcedureIndices>,
    pub lm_head_procedure_index: usize,
}

pub type AneBucketCompilePlan = AneProgramPlan;

impl AneProgramPlan {
    #[must_use]
    pub fn ffn_only(config: AneRolloutConfig) -> Self {
        let mut procedures = Vec::with_capacity(config.num_layers + 1);
        let mut layer_procedure_indices = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let ffn = procedures.len();
            procedures.push(AneProcedure::FusedFfn { layer });
            layer_procedure_indices.push(AneLayerProcedureIndices {
                qkv: None,
                ffn: Some(ffn),
            });
        }
        let lm_head_procedure_index = procedures.len();
        procedures.push(AneProcedure::LmHead);
        Self {
            config,
            procedures,
            layer_procedure_indices,
            lm_head_procedure_index,
        }
    }

    #[must_use]
    pub fn qkv_ffn_lm_head(config: AneRolloutConfig) -> Self {
        let mut procedures = Vec::with_capacity(config.num_layers * 2 + 1);
        let mut layer_procedure_indices = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let qkv = procedures.len();
            procedures.push(AneProcedure::FusedQkv { layer });
            let ffn = procedures.len();
            procedures.push(AneProcedure::FusedFfn { layer });
            layer_procedure_indices.push(AneLayerProcedureIndices {
                qkv: Some(qkv),
                ffn: Some(ffn),
            });
        }
        let lm_head_procedure_index = procedures.len();
        procedures.push(AneProcedure::LmHead);
        Self {
            config,
            procedures,
            layer_procedure_indices,
            lm_head_procedure_index,
        }
    }

    #[must_use]
    pub fn num_procedures(&self) -> usize {
        self.procedures.len()
    }

    pub fn procedure_indices_for_layer(&self, layer: usize) -> Result<AneLayerProcedureIndices> {
        self.layer_procedure_indices
            .get(layer)
            .copied()
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::ProcedureIndexMissing { layer },
                    AppleCtx {
                        backend: "private-ane",
                        op: "procedure_indices_for_layer",
                        device: "apple-silicon",
                    },
                )
            })
    }
}

pub fn compile_private_ane_program(_plan: &AneProgramPlan) -> Result<()> {
    Err(RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "private-ane",
            op: "compile_private_ane_program",
        },
        AppleCtx {
            backend: "private-ane",
            op: "compile",
            device: "apple-silicon",
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
        assert_eq!(plan.layer_procedure_indices.len(), 24);
        assert_eq!(
            plan.layer_procedure_indices[0],
            AneLayerProcedureIndices {
                qkv: None,
                ffn: Some(0)
            }
        );
        assert_eq!(
            plan.layer_procedure_indices[23],
            AneLayerProcedureIndices {
                qkv: None,
                ffn: Some(23)
            }
        );
        assert_eq!(plan.lm_head_procedure_index, 24);
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

    #[test]
    fn qkv_ffn_program_maps_layers_to_procedure_indices() {
        let config = AneRolloutConfig {
            bucket: RolloutBucket { seqs: 8, tokens: 4 },
            hidden_size: 1024,
            intermediate_size: 2816,
            num_layers: 3,
        };
        let plan = AneProgramPlan::qkv_ffn_lm_head(config);

        assert_eq!(
            plan.layer_procedure_indices,
            vec![
                AneLayerProcedureIndices {
                    qkv: Some(0),
                    ffn: Some(1)
                },
                AneLayerProcedureIndices {
                    qkv: Some(2),
                    ffn: Some(3)
                },
                AneLayerProcedureIndices {
                    qkv: Some(4),
                    ffn: Some(5)
                },
            ]
        );
        assert_eq!(plan.lm_head_procedure_index, 6);
        let layer_one = match plan.procedure_indices_for_layer(1) {
            Ok(indices) => indices,
            Err(e) => panic!("unexpected procedure index error: {e}"),
        };
        assert_eq!(
            layer_one,
            AneLayerProcedureIndices {
                qkv: Some(2),
                ffn: Some(3)
            }
        );
        assert!(plan.procedure_indices_for_layer(3).is_err());
        assert_eq!(plan.procedures[0], AneProcedure::FusedQkv { layer: 0 });
        assert_eq!(plan.procedures[5], AneProcedure::FusedFfn { layer: 2 });
        assert_eq!(plan.procedures[6], AneProcedure::LmHead);
    }
}
