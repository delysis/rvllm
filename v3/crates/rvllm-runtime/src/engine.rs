//! v3 Engine: type-state `step_launch` → `PendingStep::collect`.
//!
//! `step_launch` returns a `PendingStep<'e>` that borrows `&mut Engine`.
//! The only way to drain the step is `PendingStep::collect(self)`,
//! which consumes self and releases the borrow. The borrow checker
//! makes "second launch while ticket is live" a compile error; the
//! `#[must_use]` lint catches silent drops; `Drop` debug_asserts so a
//! mis-use panics in tests rather than silently auto-collecting.
//!
//! There is ONE codepath. Graph capture/replay is an implementation
//! detail inside `step_launch`.

use rvllm_core::{
    ReqId, Result, RuntimeConfig, TokenId,
    AppleError, AppleCtx,
};

use crate::scheduler::{BatchPlan, Scheduler};

#[cfg(feature = "apple")]

/// Output of one step: (request id, new token, finished flag).
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub req_id: ReqId,
    pub new_token: TokenId,
    pub finished: bool,
}

#[cfg(feature = "apple")]
use rvllm_apple::{
    AppleAcceleratorTarget, AppleBackend, AppleBackendMode as AppleBackendModeImpl,
    AppleLaunchTicket, AppleRuntimePlan, HandoffKind,
};

fn apple_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "runtime",
        op,
        device: "apple-silicon",
    }
}

fn apple_unavailable_error(op: &'static str, backend: &'static str) -> rvllm_core::RvllmError {
    rvllm_core::RvllmError::apple(
        AppleError::FeatureNotAvailable { backend, op },
        apple_ctx(op),
    )
}

pub struct Engine {
    pub scheduler: Scheduler,
    #[cfg(feature = "apple")]
    pub apple_backend: Option<Box<dyn AppleBackend>>,
    #[cfg(feature = "apple")]
    pub apple_runtime_plan: Option<AppleRuntimePlan>,
    #[cfg(feature = "apple")]
    pub apple_target: Option<AppleAcceleratorTarget>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            #[cfg(feature = "apple")]
            apple_backend: None,
            #[cfg(feature = "apple")]
            apple_runtime_plan: None,
            #[cfg(feature = "apple")]
            apple_target: None,
        }
    }

    #[cfg(feature = "apple")]
    pub fn with_apple_backend(mut self, backend: Box<dyn AppleBackend>) -> Self {
        self.apple_backend = Some(backend);
        self
    }

    #[cfg(feature = "apple")]
    pub fn with_apple_target(mut self, target: AppleAcceleratorTarget) -> Self {
        self.apple_target = Some(target);
        self
    }

    #[cfg(feature = "apple")]
    pub fn with_apple_runtime_plan(mut self, plan: AppleRuntimePlan) -> Result<Self> {
        plan.validate()?;
        self.apple_runtime_plan = Some(plan);
        Ok(self)
    }

    #[cfg(feature = "apple")]
    pub fn with_apple_runtime_config(
        mut self,
        target: AppleAcceleratorTarget,
        runtime: &RuntimeConfig,
    ) -> Result<Self> {
        let plan = runtime_to_apple_plan(&target, runtime)?;
        if let Some(plan) = &plan {
            plan.validate()?;
        }
        self.apple_runtime_plan = plan;
        self.apple_target = Some(target);
        Ok(self)
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.num_alive() > 0
    }

    pub fn step_launch(&mut self) -> Result<PendingStep<'_>> {
        let plan = self.scheduler.schedule();

        #[cfg(feature = "apple")]
        let mut apple_ticket = None;

        #[cfg(feature = "apple")]
        if let Some(apple_plan) = &self.apple_runtime_plan {
            let Some(backend) = self.apple_backend.as_mut() else {
                return Err(apple_unavailable_error("apple_backend_missing", "apple-runtime"));
            };

            enforce_apple_mode_availability(apple_plan)?;

                    if backend_plan_is_enabled(apple_plan) {
                        match &plan {
                            BatchPlan::Prefill { .. } => {
                                let kind = match_apple_mode_to_handoff_kind(apple_plan.mode);
                                let handoff = crate::apple_bridge::handoff_from_prefill_plan(&plan, kind)?;
                                // Keep prefill fully on-accelerator for now.
                                apple_ticket = Some(backend.launch_prefill(&handoff)?);
                            }
                            BatchPlan::Decode { .. } => {
                                let kind = match_apple_mode_to_handoff_kind(apple_plan.mode);
                                if apple_plan.mode.requires_private_ane() {
                                    let requested_bucket =
                                        apple_plan.rollout_bucket.map(|b| rvllm_core::AppleRolloutBucket {
                                            seqs: b.seqs,
                                            tokens: b.tokens,
                                        });
                                    let bucket = crate::apple_bridge::rollout_bucket_for_decode_with_config(
                                        &plan,
                                        &requested_bucket,
                                        apple_plan.rollout_tokens,
                                    )?;
                                    let handoff = crate::apple_bridge::handoff_from_decode_plan_with_bucket(
                                        &plan,
                                        kind,
                                        Some(bucket),
                                    )?;
                                    apple_ticket = Some(backend.launch_rollout(&handoff, Some(bucket))?);
                                } else {
                                    let handoff = crate::apple_bridge::handoff_from_decode_plan(&plan, kind)?;
                                    apple_ticket = Some(backend.launch_rollout(&handoff, None)?);
                                }
                            }
                            BatchPlan::Idle => {}
                        }
                    }
                }

        Ok(PendingStep {
            engine: self,
            plan: Some(plan),
            #[cfg(feature = "apple")]
            apple_ticket,
        })
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use = "PendingStep must be collect()-ed; silent drop loses the step's scheduler output"]
pub struct PendingStep<'e> {
    engine: &'e mut Engine,
    plan: Option<BatchPlan>,
    #[cfg(feature = "apple")]
    apple_ticket: Option<AppleLaunchTicket>,
}

impl<'e> PendingStep<'e> {
    pub fn plan(&self) -> Option<&BatchPlan> {
        self.plan.as_ref()
    }

    pub fn collect(mut self) -> Result<Vec<StepOutput>> {
        let _plan = self.plan.take().expect("PendingStep::collect called twice");
        
        let mut outputs = Vec::new();
        #[cfg(feature = "apple")]
        let mut decoded = Vec::<(ReqId, TokenId)>::new();
        
        #[cfg(feature = "apple")]
        if let Some(ticket) = self.apple_ticket.take() {
            if let Some(backend) = &mut self.engine.apple_backend {
                let step_tokens = backend.collect(ticket)?;
                for st in step_tokens {
                    decoded.push((st.req_id, st.token_id));
                    outputs.push(StepOutput {
                        req_id: st.req_id,
                        new_token: st.token_id,
                        finished: st.finished,
                    });
                }
            }
        }

        #[cfg(feature = "apple")]
        if !decoded.is_empty() {
            self.engine.scheduler.commit_decode(&decoded);
        }
        
        Ok(outputs)
    }
}

impl<'e> Drop for PendingStep<'e> {
    fn drop(&mut self) {
        debug_assert!(
            self.plan.is_none(),
            "PendingStep dropped without collect(); scheduler output leaked."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched_state::Request;
    use rvllm_core::{ReqId, TokenId};

    #[test]
    fn empty_engine_has_no_pending_work() {
        let e = Engine::new();
        assert!(!e.has_pending_work());
    }

    #[test]
    fn launch_then_collect_releases_borrow_for_next_launch() {
        let mut e = Engine::new();
        e.scheduler
            .enqueue(Request::new(ReqId(1), vec![TokenId(0)], 1));
        assert!(e.has_pending_work());
        let t = e.step_launch().unwrap();
        let _outputs = t.collect().unwrap();
        // Ticket consumed; engine borrow released; can launch again.
        let t2 = e.step_launch().unwrap();
        let _ = t2.collect().unwrap();
    }
    #[test]
    #[cfg(feature = "apple")]
    fn e2e_apple_backend_wiring() {
        use rvllm_apple::{AppleRuntimePlan, StubAppleBackend};

        let mut backend = Box::new(StubAppleBackend::new());
        let plan = AppleRuntimePlan {
            target: rvllm_apple::device::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: rvllm_apple::plan::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: true,
        };
        backend.prepare(&plan).unwrap();

        let e = Engine::new().with_apple_backend(backend);
        let e = match e.with_apple_runtime_plan(plan) {
            Ok(v) => v,
            Err(e) => panic!("unexpected runtime plan error: {e}"),
        };
        let mut e = e;
        e.scheduler.enqueue(Request::new(ReqId(1), vec![TokenId(0)], 1));
        
        // 1. Prefill
        let t1 = e.step_launch().unwrap();
        let outputs1 = t1.collect().unwrap();
        assert!(outputs1.is_empty(), "Prefill returns empty tokens");

        // 2. Decode (Rollout)
        let t2 = e.step_launch().unwrap();
        let outputs2 = t2.collect().unwrap();
        assert_eq!(outputs2.len(), 1, "Decode should return tokens from backend");
        assert_eq!(outputs2[0].new_token, TokenId(1));
    }

    #[test]
    #[cfg(feature = "apple")]
    #[cfg(not(target_os = "macos"))]
    fn private_ane_mode_fails_closed_without_ane_target() {
        let mut backend = Box::new(StubAppleBackend::new());
        let plan = AppleRuntimePlan {
            target: rvllm_apple::device::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: rvllm_apple::plan::AppleBackendMode::MetalPrefillAneFfnRollout,
            rollout_bucket: Some(rvllm_apple::plan::RolloutBucket { seqs: 4, tokens: 4 }),
            rollout_tokens: 1,
            private_ane_opt_in: true,
        };
        let e = Engine::new().with_apple_backend(backend);
        let e = match e.with_apple_runtime_plan(plan) {
            Ok(v) => v,
            Err(e) => panic!("unexpected runtime plan error: {e}"),
        };
        let mut e = e;
        e.scheduler.enqueue(Request::new(ReqId(1), vec![TokenId(10)], 4));

        let t1 = e.step_launch().unwrap();
        let _ = t1.collect().unwrap();

        let t2 = e.step_launch();
        match t2 {
            Err(err) => {
                let s = format!("{err}");
                assert!(s.contains("FeatureNotAvailable"));
            }
            Ok(_) => panic!("private ANE decode should fail closed on non-macOS"),
        }
    }
}

#[cfg(feature = "apple")]
fn backend_plan_is_enabled(plan: &AppleRuntimePlan) -> bool {
    !matches!(
        plan.mode,
        AppleBackendModeImpl::MlxPrototype
    )
}

#[cfg(feature = "apple")]
fn enforce_apple_mode_availability(plan: &AppleRuntimePlan) -> Result<()> {
    if plan.mode.requires_private_ane() && !cfg!(target_os = "macos") {
        return Err(apple_unavailable_error("private_ane_unavailable", "private-ane"));
    }
    if plan.mode.requires_private_ane() && plan.target.ane_cores == 0 {
        return Err(apple_unavailable_error("ane_cores", "private-ane"));
    }
    Ok(())
}

#[cfg(feature = "apple")]
fn runtime_to_apple_plan(
        target: &AppleAcceleratorTarget,
        runtime: &RuntimeConfig,
) -> Result<Option<AppleRuntimePlan>> {
    if matches!(runtime.apple_backend_mode(), rvllm_core::AppleBackendMode::Disabled) {
        return Ok(None);
    }
    let mode = match runtime.apple_backend_mode() {
        rvllm_core::AppleBackendMode::MetalOnly => AppleBackendModeImpl::MetalOnly,
        rvllm_core::AppleBackendMode::MetalPrefillMetalDecode => {
            AppleBackendModeImpl::MetalPrefillMetalDecode
        }
        rvllm_core::AppleBackendMode::MetalPrefillAneFfnRollout => {
            AppleBackendModeImpl::MetalPrefillAneFfnRollout
        }
        rvllm_core::AppleBackendMode::MetalPrefillAneRolloutExperimental => {
            AppleBackendModeImpl::MetalPrefillAneRolloutExperimental
        }
        rvllm_core::AppleBackendMode::Disabled => {
            return Ok(None);
        }
    };

    let rollout_tokens = runtime.apple_rollout_tokens();
    let rollout_bucket = match runtime.apple_rollout_bucket() {
        Some(bucket) => Some(rvllm_apple::plan::RolloutBucket {
            seqs: bucket.seqs,
            tokens: bucket.tokens,
        }),
        None => None,
    };

    let plan = AppleRuntimePlan {
        target: target.clone(),
        mode,
        rollout_bucket,
        rollout_tokens,
        private_ane_opt_in: runtime.apple_private_ane_opt_in(),
        weights_path: runtime.weights_path().map(|p| p.to_path_buf()),
    };
    Ok(Some(plan))
}

#[cfg(feature = "apple")]
fn match_apple_mode_to_handoff_kind(mode: AppleBackendModeImpl) -> HandoffKind {
    match mode {
        AppleBackendModeImpl::MetalOnly | AppleBackendModeImpl::MlxPrototype | AppleBackendModeImpl::MetalPrefillMetalDecode => {
            HandoffKind::MetalPrefillToMetalDecode
        }
        AppleBackendModeImpl::MetalPrefillAneFfnRollout => HandoffKind::MetalPrefillToAneFfnRollout,
        AppleBackendModeImpl::MetalPrefillAneRolloutExperimental => {
            HandoffKind::MetalPrefillToAneRolloutExperimental
        }
        _ => HandoffKind::MetalPrefillToMetalDecode,
    }
}
