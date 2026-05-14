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
    DenseProjection { name: String },
    FusedFfn { layer: usize },
    FusedQkv { layer: usize },
    LmHead,
}

impl AneProcedure {
    #[must_use]
    pub const fn kind_name(&self) -> &'static str {
        match self {
            AneProcedure::DenseProjection { .. } => "dense_projection",
            AneProcedure::FusedFfn { .. } => "fused_ffn",
            AneProcedure::FusedQkv { .. } => "fused_qkv",
            AneProcedure::LmHead => "lm_head",
        }
    }
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

pub fn compile_private_ane_mil(
    procedure: &AneProcedure,
    mil_source: &str,
    weight_blob: &[u8],
) -> Result<()> {
    validate_private_ane_mil_request(procedure, mil_source, weight_blob)?;
    private_ane_compile_unavailable(procedure, "compile_private_ane_mil")
}

fn validate_private_ane_mil_request(
    procedure: &AneProcedure,
    mil_source: &str,
    weight_blob: &[u8],
) -> Result<()> {
    if mil_source.trim().is_empty() {
        return Err(apple_err(
            AppleError::InvalidMil {
                reason: "MIL source is empty",
            },
            "validate_private_ane_mil",
        ));
    }
    if !mil_source.trim_start().starts_with("program(1.0)") {
        return Err(apple_err(
            AppleError::InvalidMil {
                reason: "MIL source must start with program(1.0)",
            },
            "validate_private_ane_mil",
        ));
    }
    if let AneProcedure::DenseProjection { name } = procedure {
        if !mil_source.contains(name) {
            return Err(apple_err(
                AppleError::InvalidMil {
                    reason: "dense projection MIL is missing procedure name",
                },
                "validate_private_ane_mil",
            ));
        }
    }
    if weight_blob.len() < 128 {
        return Err(apple_err(
            AppleError::InvalidWeightBlob {
                reason: "weight blob is too small",
            },
            "validate_private_ane_mil",
        ));
    }
    if weight_blob.first().copied() != Some(0x01) || weight_blob.get(4).copied() != Some(0x02) {
        return Err(apple_err(
            AppleError::InvalidWeightBlob {
                reason: "weight blob header is invalid",
            },
            "validate_private_ane_mil",
        ));
    }
    Ok(())
}

fn private_ane_compile_unavailable(_procedure: &AneProcedure, op: &'static str) -> Result<()> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane"))]
    {
        Err(apple_err(
            AppleError::PrivateApiUnavailable {
                symbol: "ANECompiler",
            },
            op,
        ))
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "private-ane")))]
    {
        Err(apple_err(
            AppleError::FeatureNotAvailable {
                backend: "private-ane",
                op,
            },
            op,
        ))
    }
}

fn apple_err(err: AppleError, op: &'static str) -> RvllmError {
    RvllmError::apple(
        err,
        AppleCtx {
            backend: "private-ane",
            op,
            device: "apple-silicon",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mil::{dense_1x1_conv_mil, fused_ffn_mil, FfnMilOffsets};
    use crate::weight_blob::build_weight_blob_fp16_named;

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
    fn private_ane_mil_compile_reports_typed_unavailable() {
        let weights = vec![1.0f32; 4];
        let (weight_blob, descs) = build_weight_blob_fp16_named(&[("dense", &weights)]);
        let mil = dense_1x1_conv_mil("dense", 2, 2, 1, descs[0].data_offset);
        let err = match compile_private_ane_mil(
            &AneProcedure::DenseProjection {
                name: "dense".to_owned(),
            },
            &mil,
            &weight_blob,
        ) {
            Ok(()) => panic!("private ANE compile should not be available in host tests"),
            Err(err) => err,
        };
        match err {
            RvllmError::Apple {
                err: AppleError::FeatureNotAvailable { backend, op },
                ctx,
                ..
            } => {
                assert_eq!(backend, "private-ane");
                assert_eq!(op, "compile_private_ane_mil");
                assert_eq!(ctx.backend, "private-ane");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn private_ane_mil_compile_validates_weight_blob_before_hardware() {
        let mil = fused_ffn_mil(
            2,
            4,
            1,
            FfnMilOffsets {
                gate: 128,
                up: 256,
                down: 384,
            },
        );
        let err = match compile_private_ane_mil(&AneProcedure::FusedFfn { layer: 0 }, &mil, &[]) {
            Ok(()) => panic!("empty weight blob should fail validation"),
            Err(err) => err,
        };
        match err {
            RvllmError::Apple {
                err: AppleError::InvalidWeightBlob { reason },
                ..
            } => assert_eq!(reason, "weight blob is too small"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
