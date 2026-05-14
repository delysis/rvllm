use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
use serde::{Deserialize, Serialize};

use crate::iosurface::IoSurfaceTensorDesc;
use crate::mil::{fused_ffn_mil, FfnMilOffsets};
use crate::plan::RolloutBucket;
use crate::weight_blob::{build_weight_blob_fp16_named, WeightChunkDesc};

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

#[derive(Copy, Clone, Debug)]
pub struct DenseFfnLayerWeights<'a> {
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FusedFfnProgramArtifact {
    pub procedure: AneProcedure,
    pub mil: String,
    pub weight_blob: Vec<u8>,
    pub weight_descriptors: Vec<WeightChunkDesc>,
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

    pub fn dense_qwen_ffn_artifacts(
        &self,
        layer_weights: &[DenseFfnLayerWeights<'_>],
    ) -> Result<Vec<FusedFfnProgramArtifact>> {
        if layer_weights.len() != self.config.num_layers {
            return Err(invalid_weight_blob("dense FFN layer weight count mismatch"));
        }

        let expected = self
            .config
            .hidden_size
            .checked_mul(self.config.intermediate_size)
            .ok_or_else(|| invalid_weight_blob("dense FFN weight shape overflow"))?;
        let spatial = self
            .config
            .bucket
            .seqs
            .checked_mul(self.config.bucket.tokens)
            .ok_or_else(|| invalid_weight_blob("rollout bucket shape overflow"))?
            as usize;

        let mut seen_layers = vec![false; self.config.num_layers];
        let mut artifacts = Vec::with_capacity(self.config.num_layers);
        for procedure in &self.procedures {
            let layer = match procedure {
                AneProcedure::FusedFfn { layer } => *layer,
                AneProcedure::FusedQkv { .. } | AneProcedure::LmHead => continue,
            };
            if layer >= self.config.num_layers {
                return Err(invalid_mil("dense FFN procedure layer out of range"));
            }
            if seen_layers[layer] {
                return Err(invalid_mil("duplicate dense FFN procedure layer"));
            }
            seen_layers[layer] = true;

            let weights = layer_weights
                .get(layer)
                .ok_or_else(|| invalid_weight_blob("dense FFN layer index out of range"))?;
            validate_dense_ffn_weights(weights, expected)?;

            let (weight_blob, weight_descriptors) = build_weight_blob_fp16_named(&[
                ("gate", weights.gate),
                ("up", weights.up),
                ("down", weights.down),
            ]);
            let offsets = ffn_mil_offsets_from_weight_descriptors(&weight_descriptors)?;
            let mil = fused_ffn_mil(
                self.config.hidden_size,
                self.config.intermediate_size,
                spatial,
                offsets,
            );

            artifacts.push(FusedFfnProgramArtifact {
                procedure: AneProcedure::FusedFfn { layer },
                mil,
                weight_blob,
                weight_descriptors,
            });
        }

        if artifacts.len() != self.config.num_layers || seen_layers.iter().any(|seen| !*seen) {
            return Err(invalid_mil("dense FFN procedure count mismatch"));
        }

        Ok(artifacts)
    }
}

fn validate_dense_ffn_weights(weights: &DenseFfnLayerWeights<'_>, expected: usize) -> Result<()> {
    if weights.gate.len() != expected {
        return Err(invalid_weight_blob("gate FFN weight shape mismatch"));
    }
    if weights.up.len() != expected {
        return Err(invalid_weight_blob("up FFN weight shape mismatch"));
    }
    if weights.down.len() != expected {
        return Err(invalid_weight_blob("down FFN weight shape mismatch"));
    }
    Ok(())
}

fn ffn_mil_offsets_from_weight_descriptors(descs: &[WeightChunkDesc]) -> Result<FfnMilOffsets> {
    let mut gate = None;
    let mut up = None;
    let mut down = None;

    for desc in descs {
        match desc.name.as_str() {
            "gate" => gate = Some(desc.data_offset),
            "up" => up = Some(desc.data_offset),
            "down" => down = Some(desc.data_offset),
            _ => {
                return Err(invalid_weight_blob(
                    "unexpected dense FFN weight descriptor",
                ))
            }
        }
    }

    Ok(FfnMilOffsets {
        gate: gate.ok_or_else(|| invalid_weight_blob("missing gate FFN weight descriptor"))?,
        up: up.ok_or_else(|| invalid_weight_blob("missing up FFN weight descriptor"))?,
        down: down.ok_or_else(|| invalid_weight_blob("missing down FFN weight descriptor"))?,
    })
}

fn invalid_weight_blob(reason: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::InvalidWeightBlob { reason },
        AppleCtx {
            backend: "rvllm-apple",
            op: "dense_qwen_ffn_artifacts",
            device: "apple-silicon",
        },
    )
}

fn invalid_mil(reason: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::InvalidMil { reason },
        AppleCtx {
            backend: "rvllm-apple",
            op: "dense_qwen_ffn_artifacts",
            device: "apple-silicon",
        },
    )
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
    fn dense_qwen_ffn_artifact_uses_gate_up_down_chunk_offsets() {
        let config = AneRolloutConfig {
            bucket: RolloutBucket { seqs: 2, tokens: 2 },
            hidden_size: 4,
            intermediate_size: 6,
            num_layers: 1,
        };
        let plan = AneProgramPlan::ffn_only(config);
        let gate = vec![1.0f32; 24];
        let up = vec![2.0f32; 24];
        let down = vec![3.0f32; 24];

        let artifacts = match plan.dense_qwen_ffn_artifacts(&[DenseFfnLayerWeights {
            gate: &gate,
            up: &up,
            down: &down,
        }]) {
            Ok(artifacts) => artifacts,
            Err(err) => panic!("dense FFN artifacts should build: {err:?}"),
        };

        assert_eq!(artifacts.len(), 1);
        let ffn = &artifacts[0];
        assert_eq!(ffn.procedure, AneProcedure::FusedFfn { layer: 0 });
        assert_eq!(ffn.weight_descriptors.len(), 3);
        assert_eq!(ffn.weight_descriptors[0].name, "gate");
        assert_eq!(ffn.weight_descriptors[1].name, "up");
        assert_eq!(ffn.weight_descriptors[2].name, "down");
        assert_eq!(ffn.weight_descriptors[0].chunk_offset, 64);
        assert_eq!(ffn.weight_descriptors[1].chunk_offset, 176);
        assert_eq!(ffn.weight_descriptors[2].chunk_offset, 288);
        assert!(ffn.mil.contains("offset = tensor<uint64, []>(128)"));
        assert!(ffn.mil.contains("offset = tensor<uint64, []>(240)"));
        assert!(ffn.mil.contains("offset = tensor<uint64, []>(352)"));
    }
}
