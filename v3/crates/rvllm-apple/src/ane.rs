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
                AneProcedure::DenseProjection { .. }
                | AneProcedure::FusedQkv { .. }
                | AneProcedure::LmHead => continue,
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct AneSysHandle(u64);

impl AneSysHandle {
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

pub trait AneSys {
    fn compile(&self, plan: &AneProgramPlan) -> Result<AneSysHandle>;
    fn write(&self, handle: AneSysHandle, binding: u32, bytes: &[u8]) -> Result<()>;
    fn eval(&self, handle: AneSysHandle) -> Result<()>;
    fn read(&self, handle: AneSysHandle, binding: u32, out: &mut [u8]) -> Result<()>;
    fn free(&self, handle: AneSysHandle) -> Result<()>;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum AneProgramState {
    Compiled,
    InputsWritten,
    Evaluated,
    Freed,
}

impl AneProgramState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Compiled => "compiled",
            Self::InputsWritten => "inputs-written",
            Self::Evaluated => "evaluated",
            Self::Freed => "freed",
        }
    }
}

pub struct AneProgram<'sys, S: AneSys + ?Sized> {
    sys: &'sys S,
    handle: Option<AneSysHandle>,
    state: AneProgramState,
}

impl<'sys, S: AneSys + ?Sized> AneProgram<'sys, S> {
    pub fn compile(sys: &'sys S, plan: &AneProgramPlan) -> Result<Self> {
        Ok(Self {
            sys,
            handle: Some(sys.compile(plan)?),
            state: AneProgramState::Compiled,
        })
    }

    pub fn write(&mut self, binding: u32, bytes: &[u8]) -> Result<()> {
        let handle = self.live_handle("write")?;
        self.sys.write(handle, binding, bytes)?;
        self.state = AneProgramState::InputsWritten;
        Ok(())
    }

    pub fn eval(&mut self) -> Result<()> {
        if self.state != AneProgramState::InputsWritten {
            return Err(self.lifecycle_err("eval"));
        }
        let handle = self.live_handle("eval")?;
        self.sys.eval(handle)?;
        self.state = AneProgramState::Evaluated;
        Ok(())
    }

    pub fn read(&mut self, binding: u32, out: &mut [u8]) -> Result<()> {
        if self.state != AneProgramState::Evaluated {
            return Err(self.lifecycle_err("read"));
        }
        let handle = self.live_handle("read")?;
        self.sys.read(handle, binding, out)
    }

    pub fn free(mut self) -> Result<()> {
        self.free_handle("free")
    }

    fn live_handle(&self, op: &'static str) -> Result<AneSysHandle> {
        match self.handle {
            Some(handle) => Ok(handle),
            None => Err(self.lifecycle_err(op)),
        }
    }

    fn free_handle(&mut self, op: &'static str) -> Result<()> {
        match self.handle.take() {
            Some(handle) => {
                self.state = AneProgramState::Freed;
                self.sys.free(handle)
            }
            None => Err(self.lifecycle_err(op)),
        }
    }

    fn lifecycle_err(&self, op: &'static str) -> RvllmError {
        RvllmError::apple(
            AppleError::AneLifecycleViolation {
                op,
                state: self.state.as_str(),
            },
            AppleCtx {
                backend: "private-ane",
                op,
                device: "apple-silicon",
            },
        )
    }
}

impl<S: AneSys + ?Sized> Drop for AneProgram<'_, S> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.state = AneProgramState::Freed;
            let _ = self.sys.free(handle);
        }
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
    use std::cell::RefCell;

    #[derive(Default)]
    struct FakeAneSys {
        calls: RefCell<Vec<&'static str>>,
    }

    impl AneSys for FakeAneSys {
        fn compile(&self, _plan: &AneProgramPlan) -> Result<AneSysHandle> {
            self.calls.borrow_mut().push("compile");
            Ok(AneSysHandle::new(7))
        }

        fn write(&self, _handle: AneSysHandle, _binding: u32, _bytes: &[u8]) -> Result<()> {
            self.calls.borrow_mut().push("write");
            Ok(())
        }

        fn eval(&self, _handle: AneSysHandle) -> Result<()> {
            self.calls.borrow_mut().push("eval");
            Ok(())
        }

        fn read(&self, _handle: AneSysHandle, _binding: u32, out: &mut [u8]) -> Result<()> {
            self.calls.borrow_mut().push("read");
            out.copy_from_slice(&[1, 2, 3, 4]);
            Ok(())
        }

        fn free(&self, _handle: AneSysHandle) -> Result<()> {
            self.calls.borrow_mut().push("free");
            Ok(())
        }
    }

    fn test_plan() -> AneProgramPlan {
        AneProgramPlan::ffn_only(AneRolloutConfig {
            bucket: RolloutBucket { seqs: 1, tokens: 1 },
            hidden_size: 4,
            intermediate_size: 16,
            num_layers: 1,
        })
    }

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
    fn safe_program_owns_full_sys_lifecycle() {
        let sys = FakeAneSys::default();
        {
            let mut program = match AneProgram::compile(&sys, &test_plan()) {
                Ok(v) => v,
                Err(e) => panic!("unexpected compile error: {e}"),
            };
            assert!(program.write(0, &[9, 8, 7, 6]).is_ok());
            assert!(program.eval().is_ok());
            let mut out = [0; 4];
            assert!(program.read(1, &mut out).is_ok());
            assert_eq!(out, [1, 2, 3, 4]);
        }

        assert_eq!(
            sys.calls.borrow().as_slice(),
            ["compile", "write", "eval", "read", "free"]
        );
    }

    #[test]
    fn safe_program_rejects_eval_before_write() {
        let sys = FakeAneSys::default();
        {
            let mut program = match AneProgram::compile(&sys, &test_plan()) {
                Ok(v) => v,
                Err(e) => panic!("unexpected compile error: {e}"),
            };
            let err = match program.eval() {
                Ok(()) => panic!("eval before write should fail"),
                Err(e) => e,
            };
            match err {
                RvllmError::Apple {
                    err: AppleError::AneLifecycleViolation { op, state },
                    ..
                } => {
                    assert_eq!(op, "eval");
                    assert_eq!(state, "compiled");
                }
                other => panic!("unexpected error: {other}"),
            }
            assert_eq!(sys.calls.borrow().as_slice(), ["compile"]);
        }
        assert_eq!(sys.calls.borrow().as_slice(), ["compile", "free"]);
    }

    #[test]
    fn explicit_free_is_not_repeated_on_drop() {
        let sys = FakeAneSys::default();
        let program = match AneProgram::compile(&sys, &test_plan()) {
            Ok(v) => v,
            Err(e) => panic!("unexpected compile error: {e}"),
        };
        assert!(program.free().is_ok());
        assert_eq!(sys.calls.borrow().as_slice(), ["compile", "free"]);
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
