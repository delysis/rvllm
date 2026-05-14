//! Non-production MLX reference harness scaffolding.
//!
//! This module is a parity-oracle prototype only. It intentionally does not
//! implement `AppleBackend`, and it never falls back to Metal, ANE, CUDA, or
//! CPU execution. Callers must provide an explicit external executor path; the
//! harness only builds a sample invocation that parity tests can wire up later.

use std::path::{Path, PathBuf};

use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
use serde::{Deserialize, Serialize};

use crate::handoff::HandoffCapsule;

const BACKEND: &str = "mlx-reference";
const DEVICE: &str = "host-mlx-reference";

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MlxReferenceMode {
    PrototypeOnly,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MlxReferenceExecution {
    PlannedOnly,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MlxParityOutput {
    PrefillLogits { tolerance_ulps: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MlxParityCase {
    name: String,
    handoff: HandoffCapsule,
    output: MlxParityOutput,
}

impl MlxParityCase {
    pub fn prefill_logits(
        name: impl Into<String>,
        handoff: HandoffCapsule,
        tolerance_ulps: u32,
    ) -> Result<Self> {
        let case = Self {
            name: name.into(),
            handoff,
            output: MlxParityOutput::PrefillLogits { tolerance_ulps },
        };
        case.validate()?;
        Ok(case)
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn output(&self) -> MlxParityOutput {
        self.output
    }

    #[must_use]
    pub const fn handoff(&self) -> &HandoffCapsule {
        &self.handoff
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(err(
                "validate_parity_case",
                AppleError::HandoffMalformed {
                    reason: "mlx parity case name must not be empty",
                },
            ));
        }
        match self.output {
            MlxParityOutput::PrefillLogits { tolerance_ulps } if tolerance_ulps == 0 => {
                return Err(err(
                    "validate_parity_case",
                    AppleError::HandoffMalformed {
                        reason: "mlx parity tolerance must be non-zero",
                    },
                ));
            }
            MlxParityOutput::PrefillLogits { .. } => {}
        }
        self.handoff.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MlxReferenceInvocation {
    pub case_name: String,
    pub program: PathBuf,
    pub args: Vec<String>,
    pub artifact_root: Option<PathBuf>,
    pub output: MlxParityOutput,
    pub execution: MlxReferenceExecution,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MlxReferenceHarness {
    mode: MlxReferenceMode,
    executor: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
}

impl Default for MlxReferenceHarness {
    fn default() -> Self {
        Self::prototype()
    }
}

impl MlxReferenceHarness {
    #[must_use]
    pub const fn prototype() -> Self {
        Self {
            mode: MlxReferenceMode::PrototypeOnly,
            executor: None,
            artifact_root: None,
        }
    }

    #[must_use]
    pub fn with_explicit_executor(mut self, executor: impl Into<PathBuf>) -> Self {
        self.executor = Some(executor.into());
        self
    }

    #[must_use]
    pub fn with_explicit_artifact_root(mut self, artifact_root: impl Into<PathBuf>) -> Self {
        self.artifact_root = Some(artifact_root.into());
        self
    }

    #[must_use]
    pub const fn mode(&self) -> MlxReferenceMode {
        self.mode
    }

    #[must_use]
    pub const fn is_non_production(&self) -> bool {
        matches!(self.mode, MlxReferenceMode::PrototypeOnly)
    }

    #[must_use]
    pub const fn can_fallback(&self) -> bool {
        false
    }

    #[must_use]
    pub fn executor(&self) -> Option<&Path> {
        self.executor.as_deref()
    }

    #[must_use]
    pub fn artifact_root(&self) -> Option<&Path> {
        self.artifact_root.as_deref()
    }

    pub fn plan_parity_case(&self, case: &MlxParityCase) -> Result<MlxReferenceInvocation> {
        self.build_invocation(case, "plan_parity_case")
    }

    /// Prototype-only compatibility hook for future runner integration.
    ///
    /// This does not execute MLX. It returns the same planned-only invocation
    /// as `plan_parity_case` and requires an explicit executor path so callers
    /// cannot silently route through another backend.
    pub fn run_parity_case(&self, case: &MlxParityCase) -> Result<MlxReferenceInvocation> {
        self.build_invocation(case, "run_parity_case")
    }

    fn build_invocation(
        &self,
        case: &MlxParityCase,
        op: &'static str,
    ) -> Result<MlxReferenceInvocation> {
        case.validate()?;
        let Some(executor) = self.executor.clone() else {
            return Err(err(
                op,
                AppleError::FeatureNotAvailable {
                    backend: BACKEND,
                    op,
                },
            ));
        };

        let tolerance_ulps = match case.output {
            MlxParityOutput::PrefillLogits { tolerance_ulps } => tolerance_ulps,
        };
        let mut args = vec![
            "--case".to_owned(),
            case.name().to_owned(),
            "--output".to_owned(),
            match case.output {
                MlxParityOutput::PrefillLogits { .. } => "prefill-logits".to_owned(),
            },
            "--tolerance-ulps".to_owned(),
            tolerance_ulps.to_string(),
        ];
        if let Some(artifact_root) = &self.artifact_root {
            args.push("--artifact-root".to_owned());
            args.push(artifact_root.display().to_string());
        }

        Ok(MlxReferenceInvocation {
            case_name: case.name().to_owned(),
            program: executor,
            args,
            artifact_root: self.artifact_root.clone(),
            output: case.output,
            execution: MlxReferenceExecution::PlannedOnly,
        })
    }
}

fn err(op: &'static str, err: AppleError) -> RvllmError {
    RvllmError::apple(
        err,
        AppleCtx {
            backend: BACKEND,
            op,
            device: DEVICE,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handoff::HandoffKind;
    use rvllm_core::{ReqId, TokenId};

    fn handoff() -> HandoffCapsule {
        HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![ReqId(1)],
            vec![TokenId(2), TokenId(3)],
            vec![0, 2],
            vec![0],
            vec![2],
        )
    }

    #[test]
    fn invocation_is_planned_only_with_explicit_executor() {
        let case = match MlxParityCase::prefill_logits("prefill", handoff(), 8) {
            Ok(case) => case,
            Err(e) => panic!("unexpected parity case error: {e}"),
        };
        let harness = MlxReferenceHarness::prototype()
            .with_explicit_executor("/tmp/run_mlx_reference.py")
            .with_explicit_artifact_root("/tmp/gemma4");

        let invocation = match harness.plan_parity_case(&case) {
            Ok(invocation) => invocation,
            Err(e) => panic!("unexpected invocation error: {e}"),
        };

        assert_eq!(invocation.execution, MlxReferenceExecution::PlannedOnly);
        assert_eq!(
            invocation.program,
            PathBuf::from("/tmp/run_mlx_reference.py")
        );
        assert!(invocation.args.iter().any(|arg| arg == "--artifact-root"));
    }
}
