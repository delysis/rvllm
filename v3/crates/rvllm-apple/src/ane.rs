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
}
