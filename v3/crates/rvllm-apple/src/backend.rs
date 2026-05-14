use rvllm_core::{AppleCtx, AppleError, ReqId, Result, RvllmError, TokenId};
use serde::{Deserialize, Serialize};

use crate::ane::{compile_private_ane_program, AneProgramPlan, AneRolloutConfig};
use crate::handoff::HandoffCapsule;
use crate::plan::{AppleRuntimePlan, RolloutBucket};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AppleLaunchKind {
    Prefill,
    Rollout,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AppleLaunchTicket {
    pub step_id: u64,
    pub kind: AppleLaunchKind,
    pub bucket: Option<RolloutBucket>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct StepToken {
    pub req_id: ReqId,
    pub token_id: TokenId,
    pub finished: bool,
}

pub trait AppleBackend {
    fn prepare(&mut self, plan: &AppleRuntimePlan) -> Result<()>;
    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket>;
    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: Option<RolloutBucket>,
    ) -> Result<AppleLaunchTicket>;
    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>>;
}

#[derive(Default)]
pub struct ProductionAppleBackend {
    compiled: bool,
    prepared: bool,
    next_step_id: u64,
    last_ticket: Option<u64>,
    pending: Option<Vec<StepToken>>,
    #[cfg(target_os = "macos")]
    handle: Option<rvllm_apple_ane_sys::AneModelHandle>,
}

impl std::fmt::Debug for ProductionAppleBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProductionAppleBackend")
            .field("compiled", &self.compiled)
            .field("prepared", &self.prepared)
            .field("next_step_id", &self.next_step_id)
            .finish()
    }
}

impl ProductionAppleBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn ctx(op: &'static str) -> AppleCtx {
        AppleCtx {
            backend: "production-apple",
            op,
            device: "apple-silicon",
        }
    }

    fn ensure_prepared(&self, op: &'static str) -> Result<()> {
        if self.prepared {
            Ok(())
        } else {
            Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "production-apple",
                },
                Self::ctx(op),
            ))
        }
    }

    fn next_ticket(
        &mut self,
        kind: AppleLaunchKind,
        bucket: Option<RolloutBucket>,
    ) -> AppleLaunchTicket {
        let step_id = self.next_step_id;
        self.next_step_id += 1;
        self.last_ticket = Some(step_id);
        AppleLaunchTicket {
            step_id,
            kind,
            bucket,
        }
    }

    fn compile_if_needed(&mut self, plan: &AppleRuntimePlan) -> Result<()> {
        if !plan.mode.requires_private_ane() {
            return Ok(());
        }
        if self.compiled {
            return Ok(());
        }

        let bucket = plan
            .rollout_bucket
            .unwrap_or(RolloutBucket { seqs: 1, tokens: 1 });
        let ane_plan = AneProgramPlan::ffn_only(AneRolloutConfig {
            bucket,
            hidden_size: 5376, // Gemma4 hidden size
            intermediate_size: 21504,
            num_layers: 1,
        });
        let weights_path = plan.weights_path.as_deref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidMil {
                    reason: "weights_path is required for private ANE",
                },
                Self::ctx("compile_if_needed"),
            )
        })?;
        compile_private_ane_program(&ane_plan, weights_path)?;

        #[cfg(target_os = "macos")]
        {
            let cache_key = ane_plan.cache_key();
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            let cache_root = std::path::PathBuf::from(home).join(".cache/rvllm/ane");
            let cached_compiled = cache_root.join(format!("{}.mlmodelc", cache_key));
            
            if let Some(h) = rvllm_apple_ane_sys::AneModelHandle::load(cached_compiled.to_str().unwrap_or("")) {
                self.handle = Some(h);
            }
        }

        self.compiled = true;
        Ok(())
    }
}

impl AppleBackend for ProductionAppleBackend {
    fn prepare(&mut self, plan: &AppleRuntimePlan) -> Result<()> {
        plan.validate()?;
        self.prepared = true;
        self.compiled = false;
        self.last_ticket = None;
        self.pending = None;
        self.compile_if_needed(plan)?;
        Ok(())
    }

    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        handoff.validate()?;
        self.pending = Some(Vec::new());
        Ok(self.next_ticket(AppleLaunchKind::Prefill, None))
    }

    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: Option<RolloutBucket>,
    ) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_rollout")?;
        handoff.validate()?;

        #[cfg(target_os = "macos")]
        if let Some(ref handle) = self.handle {
            // Resolve IOSurfaces from handoff
            let in_id = handoff
                .input_surface
                .ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::HandoffMalformed {
                            reason: "input_surface missing",
                        },
                        Self::ctx("launch_rollout"),
                    )
                })?
                .0 as u32;
            let out_id = handoff
                .output_surface
                .ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::HandoffMalformed {
                            reason: "output_surface missing",
                        },
                        Self::ctx("launch_rollout"),
                    )
                })?
                .0 as u32;

            let in_surface = rvllm_apple_ane_sys::AneSurface::from_id(in_id).ok_or_else(|| {
                RvllmError::apple(
                    AppleError::HandoffMalformed {
                        reason: "failed to lookup input_surface",
                    },
                    Self::ctx("launch_rollout"),
                )
            })?;
            let out_surface = rvllm_apple_ane_sys::AneSurface::from_id(out_id).ok_or_else(|| {
                RvllmError::apple(
                    AppleError::HandoffMalformed {
                        reason: "failed to lookup output_surface",
                    },
                    Self::ctx("launch_rollout"),
                )
            })?;

            // Create and evaluate ANE request
            let request = rvllm_apple_ane_sys::AneRequest::new(
                &[in_surface],
                &[0],
                &[out_surface.clone()],
                &[0],
                0, // Procedure 0 is the rollout
            )
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "production-apple",
                        op: "create_ane_request",
                    },
                    Self::ctx("launch_rollout"),
                )
            })?;

            handle.evaluate(&request).map_err(|_e| {
                RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "production-apple",
                        op: "ane_evaluate",
                    },
                    Self::ctx("launch_rollout"),
                )
            })?;

            // Readback or synthesize results
            let mut outputs = Vec::with_capacity(handoff.num_sequences());
            for (idx, req_id) in handoff.req_ids.iter().enumerate() {
                // Read the predicted token from the output surface.
                // Assuming tokens are packed as u32s at the start of the surface.
                let token_id = out_surface.read_u32(idx * 4);
                outputs.push(StepToken {
                    req_id: *req_id,
                    token_id: TokenId(token_id),
                    finished: false,
                });
            }
            self.pending = Some(outputs);
        }

        Ok(self.next_ticket(AppleLaunchKind::Rollout, bucket))
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        self.ensure_prepared("collect")?;
        match self.last_ticket {
            Some(expected) if expected == ticket.step_id => Ok(self.pending.take().unwrap_or_default()),
            Some(_) => Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "production-apple",
                    op: "collect_stale_ticket",
                },
                Self::ctx("collect"),
            )),
            None => Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "production-apple",
                },
                Self::ctx("collect"),
            )),
        }
    }
}

#[derive(Debug, Default)]
pub struct StubAppleBackend {
    prepared: bool,
    next_step_id: u64,
    last_ticket: Option<u64>,
    pending: Vec<StepToken>,
}

impl StubAppleBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn ctx(op: &'static str) -> AppleCtx {
        AppleCtx {
            backend: "stub-apple",
            op,
            device: "host-test",
        }
    }

    fn ensure_prepared(&self, op: &'static str) -> Result<()> {
        if self.prepared {
            Ok(())
        } else {
            Err(RvllmError::apple(
                AppleError::NotPrepared { backend: "stub-apple" },
                Self::ctx(op),
            ))
        }
    }

    fn next_ticket(
        &mut self,
        kind: AppleLaunchKind,
        bucket: Option<RolloutBucket>,
    ) -> AppleLaunchTicket {
        let step_id = self.next_step_id;
        self.next_step_id += 1;
        self.last_ticket = Some(step_id);
        AppleLaunchTicket {
            step_id,
            kind,
            bucket,
        }
    }
}

impl AppleBackend for StubAppleBackend {
    fn prepare(&mut self, plan: &AppleRuntimePlan) -> Result<()> {
        plan.validate()?;
        self.prepared = true;
        Ok(())
    }

    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        handoff.validate()?;
        self.pending.clear();
        Ok(self.next_ticket(AppleLaunchKind::Prefill, None))
    }

    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: Option<RolloutBucket>,
    ) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_rollout")?;
        handoff.validate()?;
        self.pending = handoff
            .req_ids
            .iter()
            .enumerate()
            .map(|(idx, req_id)| {
                let base = handoff.tokens_flat.get(idx).copied().unwrap_or(TokenId(0));
                StepToken {
                    req_id: *req_id,
                    token_id: TokenId((base.0 + 1) & 0xffff),
                    finished: false,
                }
            })
            .collect();
        Ok(self.next_ticket(AppleLaunchKind::Rollout, bucket))
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        let expected = self.last_ticket.ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared { backend: "stub-apple" },
                Self::ctx("collect"),
            )
        })?;
        if expected != ticket.step_id {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "stub-apple",
                    op: "collect_step_mismatch",
                },
                Self::ctx("collect"),
            ));
        }

        let mut tokens = Vec::new();
        if ticket.kind == AppleLaunchKind::Rollout {
            tokens.append(&mut self.pending);
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::AppleAcceleratorTarget;
    use crate::handoff::HandoffKind;
    use crate::plan::{AppleBackendMode, RolloutBucket};

    fn plan() -> AppleRuntimePlan {
        AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
            weights_path: None,
        }
    }

    #[test]
    fn backend_requires_prepare_before_launch() {
        let mut backend = StubAppleBackend::new();
        let handoff = HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![ReqId(1)],
            vec![TokenId(10)],
            vec![0, 1],
            vec![0],
            vec![1],
        );
        assert!(backend.launch_prefill(&handoff).is_err());
        assert!(backend.prepare(&plan()).is_ok());
        let ticket = match backend.launch_rollout(
            &handoff,
            Some(RolloutBucket { seqs: 1, tokens: 1 }),
        ) {
            Ok(v) => v,
            Err(e) => panic!("unexpected launch error: {e}"),
        };
        assert_eq!(ticket.kind, AppleLaunchKind::Rollout);
    }
}
