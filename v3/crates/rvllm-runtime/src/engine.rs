//! v3 Engine: bounded owned submissions plus a type-state convenience API.
//!
//! `step_submit` returns an owned `SubmittedStep`, allowing distinct request
//! cohorts to occupy the backend's three execution slots. The scheduler marks
//! their requests in flight so per-request state/KV transitions remain
//! ordered. `collect_submitted` owns commit, rollback, and reclamation.
//!
//! `step_launch` / `PendingStep::collect` remains the backward-compatible
//! single-step wrapper over the same path.

#[cfg(feature = "apple")]
use rvllm_core::{AppleCtx, AppleError, RuntimeConfig};
use rvllm_core::{ReqId, Result, TokenId};
use std::collections::HashSet;

use crate::paged_kv::{KvChainHandle, PagedKvConfig, PagedKvError, PagedKvPool, PagedKvStats};
use crate::paged_prompt_cache::{CachePrefixAttachment, PagedPromptCache};
use crate::sched_state::Request;
use crate::scheduler::{BatchPlan, Scheduler};

/// Output of one step: (request id, new token, finished flag).
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub req_id: ReqId,
    pub new_token: TokenId,
    pub finished: bool,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use crate::apple_metal_backend::ModelMetalBackend;
#[cfg(all(feature = "apple", target_os = "macos"))]
use crate::apple_metal_backend::ToyMetalBackend;
#[cfg(all(feature = "apple", target_os = "macos"))]
use rvllm_apple::ProductionAppleBackend;
#[cfg(feature = "apple")]
use rvllm_apple::{
    AppleAcceleratorTarget, AppleBackend, AppleBackendMode as AppleBackendModeImpl,
    AppleLaunchTicket, AppleRuntimePlan, HandoffKind,
};

#[cfg(feature = "apple")]
fn apple_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "runtime",
        op,
        device: "apple-silicon",
    }
}

#[cfg(feature = "apple")]
fn apple_unavailable_error(op: &'static str, backend: &'static str) -> rvllm_core::RvllmError {
    rvllm_core::RvllmError::apple(
        AppleError::FeatureNotAvailable { backend, op },
        apple_ctx(op),
    )
}

fn scheduler_transition_error(
    req_id: Option<ReqId>,
    reason: &'static str,
) -> rvllm_core::RvllmError {
    rvllm_core::RvllmError::Scheduler {
        err: rvllm_core::SchedulerError::InvalidTransition { reason },
        req_id,
    }
}

fn paged_kv_error(
    error: PagedKvError,
    req_id: Option<ReqId>,
    _op: &'static str,
) -> rvllm_core::RvllmError {
    match error {
        PagedKvError::PagePoolExhausted {
            needed_pages,
            free_pages,
        } => rvllm_core::RvllmError::Scheduler {
            err: rvllm_core::SchedulerError::KvExhausted {
                needed_blocks: needed_pages,
                free_blocks: free_pages,
            },
            req_id,
        },
        _ => scheduler_transition_error(req_id, "paged KV allocator rejected the operation"),
    }
}

#[derive(Copy, Clone, Debug)]
struct PagedKvReservationEntry {
    req_id: ReqId,
    handle: KvChainHandle,
    previous_token_len: u32,
    newly_allocated: bool,
}

#[derive(Debug, Default)]
struct PagedKvStepReservation {
    entries: Vec<PagedKvReservationEntry>,
}

pub struct Engine {
    pub scheduler: Scheduler,
    paged_kv: Option<PagedKvPool>,
    paged_kv_layout_fingerprint: [u8; 32],
    /// Requests whose T0 chain is paired with a cache attachment. Normal
    /// terminal/cancellation paths defer reclamation until the worker releases
    /// both the request chain and cache pin atomically.
    cache_managed_kv: HashSet<ReqId>,
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
            paged_kv: None,
            paged_kv_layout_fingerprint: [0; 32],
            cache_managed_kv: HashSet::new(),
            #[cfg(feature = "apple")]
            apple_backend: None,
            #[cfg(feature = "apple")]
            apple_runtime_plan: None,
            #[cfg(feature = "apple")]
            apple_target: None,
        }
    }

    /// Enable request-owned Apple-v1 paged KV. Capacity must be derived from
    /// the backend memory budget; the engine intentionally does not guess a
    /// device-independent default page count.
    pub fn with_paged_kv_pool(
        mut self,
        pool: PagedKvPool,
        model_layout_fingerprint: [u8; 32],
    ) -> Self {
        self.paged_kv = Some(pool);
        self.paged_kv_layout_fingerprint = model_layout_fingerprint;
        self
    }

    pub fn with_paged_kv_config(
        self,
        config: PagedKvConfig,
        model_layout_fingerprint: [u8; 32],
    ) -> Result<Self> {
        let pool = PagedKvPool::new(config)
            .map_err(|error| paged_kv_error(error, None, "paged_kv_config"))?;
        Ok(self.with_paged_kv_pool(pool, model_layout_fingerprint))
    }

    #[must_use]
    pub fn paged_kv_stats(&self) -> Option<PagedKvStats> {
        self.paged_kv.as_ref().map(PagedKvPool::stats)
    }

    #[must_use]
    pub fn paged_kv_pool(&self) -> Option<&PagedKvPool> {
        self.paged_kv.as_ref()
    }

    pub fn paged_kv_pool_mut(&mut self) -> Option<&mut PagedKvPool> {
        self.paged_kv.as_mut()
    }

    /// Transactional request admission for callers using Engine ownership.
    /// Legacy callers may still enqueue directly on `scheduler`; their chain
    /// is allocated lazily at the first Apple launch.
    pub fn enqueue_request(&mut self, mut request: Request) -> Result<()> {
        let allocated = if let Some(pool) = self.paged_kv.as_mut() {
            let handle = pool
                .allocate_chain(request.id, 0)
                .map_err(|error| paged_kv_error(error, Some(request.id), "paged_kv_admit"))?;
            request.bind_kv_chain(handle);
            Some(handle)
        } else {
            None
        };
        if let Err(_error) = self.scheduler.try_enqueue(request) {
            if let (Some(pool), Some(handle)) = (self.paged_kv.as_mut(), allocated) {
                let _ = pool.release_chain(handle);
            }
            return Err(scheduler_transition_error(
                None,
                "scheduler rejected request admission",
            ));
        }
        Ok(())
    }

    /// Adopt a cache-created request chain without allocating a second chain.
    /// The caller retains the attachment until `release_cache_attachment`.
    pub fn enqueue_request_with_cache_attachment(
        &mut self,
        mut request: Request,
        attachment: &CachePrefixAttachment,
    ) -> Result<()> {
        let pool = self.paged_kv.as_ref().ok_or_else(|| {
            scheduler_transition_error(Some(request.id), "paged KV pool is not configured")
        })?;
        let view = pool.view(attachment.chain).map_err(|error| {
            paged_kv_error(error, Some(request.id), "paged_cache_attachment_validate")
        })?;
        if view.owner != request.id || view.token_len != attachment.matched_tokens {
            return Err(scheduler_transition_error(
                Some(request.id),
                "cache attachment does not match request owner and restored length",
            ));
        }
        request.bind_kv_chain(attachment.chain);
        request.begin_restore();
        request
            .finish_restore(attachment.matched_tokens)
            .map_err(|_| {
                scheduler_transition_error(
                    Some(request.id),
                    "cache attachment exceeds prompt length",
                )
            })?;
        let req_id = request.id;
        self.scheduler.try_enqueue(request).map_err(|_| {
            scheduler_transition_error(Some(req_id), "scheduler rejected cached request admission")
        })?;
        self.cache_managed_kv.insert(req_id);
        Ok(())
    }

    /// Keep a cache-eligible miss chain alive past terminal collection so the
    /// worker can capture exact T2 bytes outside the TTFT-critical path.
    pub fn defer_paged_kv_release(&mut self, req_id: ReqId) -> Result<()> {
        let chain = self.scheduler.kv_chain_for(req_id).ok_or_else(|| {
            scheduler_transition_error(Some(req_id), "request has no paged KV chain to defer")
        })?;
        let view = self
            .paged_kv
            .as_ref()
            .ok_or_else(|| {
                scheduler_transition_error(Some(req_id), "paged KV pool is not configured")
            })?
            .view(chain)
            .map_err(|error| paged_kv_error(error, Some(req_id), "paged_kv_defer_release"))?;
        if view.owner != req_id {
            return Err(scheduler_transition_error(
                Some(req_id),
                "deferred paged KV owner does not match request",
            ));
        }
        self.cache_managed_kv.insert(req_id);
        Ok(())
    }

    pub fn release_deferred_paged_kv(&mut self, req_id: ReqId) -> Result<()> {
        if !self.cache_managed_kv.contains(&req_id) {
            return Err(scheduler_transition_error(
                Some(req_id),
                "request does not own deferred paged KV",
            ));
        }
        self.release_paged_kv_for(req_id)?;
        self.cache_managed_kv.remove(&req_id);
        Ok(())
    }

    /// Release the T0 request chain before dropping its T1 metadata pin.
    pub fn release_cache_attachment(
        &mut self,
        cache: &mut PagedPromptCache,
        attachment: &CachePrefixAttachment,
    ) -> Result<()> {
        let owner = self
            .paged_kv
            .as_ref()
            .ok_or_else(|| scheduler_transition_error(None, "paged KV pool is not configured"))?
            .view(attachment.chain)
            .map_err(|error| paged_kv_error(error, None, "paged_cache_attachment_release"))?
            .owner;
        if !self.cache_managed_kv.contains(&owner) {
            return Err(scheduler_transition_error(
                Some(owner),
                "cache attachment is not owned by this engine",
            ));
        }
        cache
            .release_attachment(
                self.paged_kv.as_mut().expect("pool validated above"),
                attachment,
            )
            .map_err(|_| {
                scheduler_transition_error(Some(owner), "failed to release cache attachment")
            })?;
        self.cache_managed_kv.remove(&owner);
        Ok(())
    }

    pub fn cancel_request(&mut self, req_id: ReqId) -> Result<bool> {
        let accelerator_in_flight = self.scheduler.request_is_accelerator_in_flight(req_id);
        let cancelled = self.scheduler.cancel_request(req_id);
        if cancelled && !accelerator_in_flight && !self.cache_managed_kv.contains(&req_id) {
            if let Some(pool) = self.paged_kv.as_mut() {
                pool.cancel_request(req_id)
                    .map_err(|error| paged_kv_error(error, Some(req_id), "paged_kv_cancel"))?;
            }
        }
        Ok(cancelled)
    }

    pub fn finish_request(&mut self, req_id: ReqId) -> Result<bool> {
        let finished = self.scheduler.finish_request(req_id);
        if finished && !self.cache_managed_kv.contains(&req_id) {
            self.release_paged_kv_for(req_id)?;
        }
        Ok(finished)
    }

    /// Pause or resume accelerator eligibility without releasing request KV.
    /// This is used when a bounded per-request response stream is full.
    pub fn set_request_backpressured(&mut self, req_id: ReqId, backpressured: bool) -> Result<()> {
        self.scheduler
            .set_backpressured(req_id, backpressured)
            .map_err(|_| {
                scheduler_transition_error(
                    Some(req_id),
                    "failed to update request backpressure eligibility",
                )
            })
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
        if self.apple_backend.is_none() {
            self.apple_backend = Some(default_apple_backend_for_plan(&plan)?);
        }
        self.apple_runtime_plan = Some(plan);
        if let Some(backend) = self.apple_backend.as_mut() {
            if let Some(runtime_plan) = self.apple_runtime_plan.as_ref() {
                backend.prepare(runtime_plan)?;
            }
        }
        Ok(self)
    }

    #[cfg(feature = "apple")]
    pub fn with_apple_runtime_config(
        mut self,
        target: AppleAcceleratorTarget,
        runtime: &RuntimeConfig,
    ) -> Result<Self> {
        let plan = runtime_to_apple_plan(&target, runtime)?;
        self.apple_target = Some(target);
        let Some(plan) = plan else {
            return Ok(self);
        };
        self.with_apple_runtime_plan(plan)
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.num_alive() > 0
    }

    /// Submit one scheduler batch without borrowing the engine for its entire
    /// GPU lifetime. Requests in the returned step are excluded from later
    /// scheduling, allowing up to the backend's bounded in-flight capacity to
    /// overlap while preserving per-request ordering.
    pub fn step_submit(&mut self) -> Result<SubmittedStep> {
        #[allow(unused_mut)]
        let mut plan = self.scheduler.schedule();

        #[cfg(feature = "apple")]
        let mut apple_ticket = None;
        #[allow(unused_mut)]
        let mut paged_kv_reservation = None;

        #[cfg(feature = "apple")]
        if let Some(apple_plan) = self.apple_runtime_plan.clone() {
            enforce_apple_mode_availability(&apple_plan)?;

            if backend_plan_is_enabled(&apple_plan) {
                if !matches!(plan, BatchPlan::Idle) && self.paged_kv.is_some() {
                    paged_kv_reservation = Some(self.prepare_paged_kv_step(&mut plan)?);
                }

                let launch_result = (|| {
                    let Some(backend) = self.apple_backend.as_mut() else {
                        return Err(apple_unavailable_error(
                            "apple_backend_missing",
                            "apple-runtime",
                        ));
                    };
                    match &plan {
                        BatchPlan::Prefill { .. } => {
                            let kind = match_apple_mode_to_handoff_kind(apple_plan.mode);
                            let handoff = if let Some(pool) = self.paged_kv.as_ref() {
                                crate::apple_bridge::handoff_from_prefill_plan_with_paged_kv(
                                    &plan,
                                    kind,
                                    None,
                                    pool,
                                    self.paged_kv_layout_fingerprint,
                                )?
                            } else {
                                crate::apple_bridge::handoff_from_prefill_plan(&plan, kind)?
                            };
                            // Keep prefill fully on-accelerator for now.
                            backend.launch_prefill(&handoff).map(Some)
                        }
                        BatchPlan::Decode { .. } => {
                            let kind = match_apple_mode_to_handoff_kind(apple_plan.mode);
                            if apple_plan.mode.requires_private_ane() {
                                let requested_bucket = apple_plan.rollout_bucket.map(|b| {
                                    rvllm_core::AppleRolloutBucket {
                                        seqs: b.seqs,
                                        tokens: b.tokens,
                                    }
                                });
                                let bucket =
                                    crate::apple_bridge::rollout_bucket_for_decode_with_config(
                                        &plan,
                                        &requested_bucket,
                                        apple_plan.rollout_tokens,
                                    )?;
                                let handoff = if let Some(pool) = self.paged_kv.as_ref() {
                                    crate::apple_bridge::handoff_from_decode_plan_with_paged_kv(
                                        &plan,
                                        kind,
                                        Some(bucket),
                                        pool,
                                        self.paged_kv_layout_fingerprint,
                                    )?
                                } else {
                                    crate::apple_bridge::handoff_from_decode_plan_with_bucket(
                                        &plan,
                                        kind,
                                        Some(bucket),
                                    )?
                                };
                                backend.launch_rollout(&handoff, Some(bucket)).map(Some)
                            } else {
                                let handoff = if let Some(pool) = self.paged_kv.as_ref() {
                                    crate::apple_bridge::handoff_from_decode_plan_with_paged_kv(
                                        &plan,
                                        kind,
                                        None,
                                        pool,
                                        self.paged_kv_layout_fingerprint,
                                    )?
                                } else {
                                    crate::apple_bridge::handoff_from_decode_plan(&plan, kind)?
                                };
                                backend.launch_rollout(&handoff, None).map(Some)
                            }
                        }
                        BatchPlan::Idle => Ok(None),
                    }
                })();
                match launch_result {
                    Ok(ticket) => apple_ticket = ticket,
                    Err(error) => {
                        if let Some(reservation) = paged_kv_reservation.take() {
                            self.rollback_paged_kv_step(reservation);
                        }
                        return Err(error);
                    }
                }
            }
        }

        self.scheduler
            .mark_accelerator_in_flight(plan.req_ids(), true);
        Ok(SubmittedStep {
            plan: Some(plan),
            paged_kv_reservation,
            #[cfg(feature = "apple")]
            apple_ticket,
        })
    }

    /// Backward-compatible type-state wrapper for callers that prefer a
    /// launch-and-collect lexical scope.
    pub fn step_launch(&mut self) -> Result<PendingStep<'_>> {
        let submitted = self.step_submit()?;
        Ok(PendingStep {
            engine: self,
            submitted: Some(submitted),
        })
    }

    fn prepare_paged_kv_step(&mut self, plan: &mut BatchPlan) -> Result<PagedKvStepReservation> {
        let (req_ids, context_lens, kv_chains) = match plan {
            BatchPlan::Prefill {
                req_ids,
                context_lens,
                kv_chains,
                ..
            }
            | BatchPlan::Decode {
                req_ids,
                context_lens,
                kv_chains,
                ..
            } => (req_ids, context_lens, kv_chains),
            BatchPlan::Idle => return Ok(PagedKvStepReservation::default()),
        };
        if req_ids.len() != context_lens.len() || req_ids.len() != kv_chains.len() {
            return Err(scheduler_transition_error(
                None,
                "paged KV plan vector lengths differ",
            ));
        }

        let mut reservation = PagedKvStepReservation::default();
        for index in 0..req_ids.len() {
            let req_id = req_ids[index];
            let result = (|| {
                let pool = self.paged_kv.as_mut().ok_or_else(|| {
                    scheduler_transition_error(Some(req_id), "paged KV pool disappeared")
                })?;
                let existing = kv_chains[index].or_else(|| pool.chain_for_owner(req_id));
                let (handle, newly_allocated) = match existing {
                    Some(handle) => (handle, false),
                    None => (
                        pool.allocate_chain(req_id, 0).map_err(|error| {
                            paged_kv_error(error, Some(req_id), "paged_kv_allocate")
                        })?,
                        true,
                    ),
                };
                let view = pool
                    .view(handle)
                    .map_err(|error| paged_kv_error(error, Some(req_id), "paged_kv_validate"))?;
                if view.owner != req_id {
                    return Err(scheduler_transition_error(
                        Some(req_id),
                        "paged KV owner does not match scheduled request",
                    ));
                }
                let previous_token_len = view.token_len;
                reservation.entries.push(PagedKvReservationEntry {
                    req_id,
                    handle,
                    previous_token_len,
                    newly_allocated,
                });

                if self.scheduler.kv_chain_for(req_id) != Some(handle) {
                    self.scheduler.bind_kv_chain(req_id, handle).map_err(|_| {
                        scheduler_transition_error(
                            Some(req_id),
                            "failed to bind request-owned KV chain",
                        )
                    })?;
                }
                kv_chains[index] = Some(handle);
                let target_len = context_lens[index];
                if previous_token_len < target_len {
                    pool.append_tokens(handle, target_len - previous_token_len)
                        .map_err(|error| paged_kv_error(error, Some(req_id), "paged_kv_reserve"))?;
                }
                Ok(())
            })();
            if let Err(error) = result {
                self.rollback_paged_kv_step(reservation);
                return Err(error);
            }
        }
        Ok(reservation)
    }

    fn rollback_paged_kv_step(&mut self, reservation: PagedKvStepReservation) {
        let Some(pool) = self.paged_kv.as_mut() else {
            return;
        };
        for entry in reservation.entries.into_iter().rev() {
            if entry.newly_allocated {
                let _ = pool.release_chain(entry.handle);
                let _ = self.scheduler.unbind_kv_chain(entry.req_id, entry.handle);
            } else {
                let _ = pool.truncate_tokens(entry.handle, entry.previous_token_len);
            }
        }
    }

    fn release_paged_kv_for(&mut self, req_id: ReqId) -> Result<()> {
        let Some(pool) = self.paged_kv.as_mut() else {
            return Ok(());
        };
        pool.cancel_request(req_id)
            .map_err(|error| paged_kv_error(error, Some(req_id), "paged_kv_release"))?;
        Ok(())
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use = "SubmittedStep must be passed to Engine::collect_submitted"]
pub struct SubmittedStep {
    plan: Option<BatchPlan>,
    paged_kv_reservation: Option<PagedKvStepReservation>,
    #[cfg(feature = "apple")]
    apple_ticket: Option<AppleLaunchTicket>,
}

impl SubmittedStep {
    pub fn plan(&self) -> Option<&BatchPlan> {
        self.plan.as_ref()
    }
}

impl Drop for SubmittedStep {
    fn drop(&mut self) {
        debug_assert!(
            self.plan.is_none(),
            "SubmittedStep dropped without Engine::collect_submitted; accelerator ownership leaked"
        );
    }
}

impl Engine {
    /// Collect a previously submitted step. Completion is the sole owner of
    /// scheduler commit, slot reclamation, backend errors, and deferred KV
    /// release for requests cancelled while Metal was using their pages.
    pub fn collect_submitted(&mut self, mut submitted: SubmittedStep) -> Result<Vec<StepOutput>> {
        let plan = submitted
            .plan
            .take()
            .expect("SubmittedStep collected twice");
        let plan_req_ids = plan.req_ids().to_vec();

        #[allow(unused_mut)]
        let mut outputs = Vec::<StepOutput>::new();
        #[cfg(feature = "apple")]
        let mut decoded = Vec::<(ReqId, TokenId)>::new();

        #[cfg(feature = "apple")]
        if let Some(ticket) = submitted.apple_ticket.take() {
            if let Some(backend) = &mut self.apple_backend {
                let step_tokens = match backend.collect(ticket) {
                    Ok(tokens) => tokens,
                    Err(error) => {
                        self.scheduler
                            .mark_accelerator_in_flight(&plan_req_ids, false);
                        if let Some(reservation) = submitted.paged_kv_reservation.take() {
                            self.rollback_paged_kv_step(reservation);
                        }
                        self.release_cancelled_step_kv(&plan_req_ids)?;
                        return Err(error);
                    }
                };
                // Cancellation is final: a completion may reclaim resources,
                // but it must never surface a late token to the caller.
                for st in step_tokens {
                    if self.scheduler.request_is_alive(st.req_id) {
                        decoded.push((st.req_id, st.token_id));
                        outputs.push(StepOutput {
                            req_id: st.req_id,
                            new_token: st.token_id,
                            finished: st.finished,
                        });
                    }
                }
            }
        }

        #[cfg(feature = "apple")]
        if !decoded.is_empty() {
            self.scheduler.commit_decode(&decoded);
            for output in &outputs {
                if output.finished {
                    self.scheduler.finish_request(output.req_id);
                }
            }
        }

        if let BatchPlan::Prefill {
            req_ids,
            cu_seqlens_q,
            ..
        } = &plan
        {
            let completed: Vec<_> = req_ids
                .iter()
                .zip(cu_seqlens_q.windows(2))
                .filter(|(req_id, _)| self.scheduler.request_is_alive(**req_id))
                .map(|(&req_id, span)| (req_id, span[1] - span[0]))
                .collect();
            if let Err(error) = self.scheduler.commit_prefill(&completed) {
                self.scheduler
                    .mark_accelerator_in_flight(&plan_req_ids, false);
                if let Some(reservation) = submitted.paged_kv_reservation.take() {
                    self.rollback_paged_kv_step(reservation);
                }
                self.release_cancelled_step_kv(&plan_req_ids)?;
                let req_id = match error {
                    crate::scheduler::SchedulerError::EmptyPrompt { req_id }
                    | crate::scheduler::SchedulerError::DuplicateRequest { req_id }
                    | crate::scheduler::SchedulerError::UnknownRequest { req_id }
                    | crate::scheduler::SchedulerError::InvalidPrefillCommit { req_id, .. } => {
                        Some(req_id)
                    }
                };
                return Err(rvllm_core::RvllmError::Scheduler {
                    err: rvllm_core::SchedulerError::InvalidTransition {
                        reason: "backend completed a prefill plan that scheduler could not commit",
                    },
                    req_id,
                });
            }
        }

        self.scheduler
            .mark_accelerator_in_flight(&plan_req_ids, false);
        // The allocator reservation becomes committed only after both backend
        // completion and scheduler state transition succeed.
        submitted.paged_kv_reservation.take();
        self.release_cancelled_step_kv(&plan_req_ids)?;

        let terminal: Vec<_> = outputs
            .iter()
            .filter(|output| output.finished || !self.scheduler.request_is_alive(output.req_id))
            .map(|output| output.req_id)
            .collect();
        for req_id in terminal {
            if !self.cache_managed_kv.contains(&req_id) {
                self.release_paged_kv_for(req_id)?;
            }
        }
        Ok(outputs)
    }

    fn release_cancelled_step_kv(&mut self, req_ids: &[ReqId]) -> Result<()> {
        for &req_id in req_ids {
            if !self.scheduler.request_is_alive(req_id) && !self.cache_managed_kv.contains(&req_id)
            {
                self.release_paged_kv_for(req_id)?;
            }
        }
        Ok(())
    }
}

#[must_use = "PendingStep must be collect()-ed; silent drop loses the step's scheduler output"]
pub struct PendingStep<'e> {
    engine: &'e mut Engine,
    submitted: Option<SubmittedStep>,
}

impl<'e> PendingStep<'e> {
    pub fn plan(&self) -> Option<&BatchPlan> {
        self.submitted.as_ref().and_then(SubmittedStep::plan)
    }

    pub fn collect(mut self) -> Result<Vec<StepOutput>> {
        let submitted = self
            .submitted
            .take()
            .expect("PendingStep::collect called twice");
        self.engine.collect_submitted(submitted)
    }
}

impl<'e> Drop for PendingStep<'e> {
    fn drop(&mut self) {
        debug_assert!(
            self.submitted.is_none(),
            "PendingStep dropped without collect(); scheduler output leaked."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_cache::{CacheIdentity, CacheNamespace, MemoryPressure, PromptCacheConfig};
    use crate::sched_state::Request;
    #[cfg(feature = "apple")]
    use rvllm_core::config::{AneComputeProfile, AneFallbackPolicy};
    use rvllm_core::{ReqId, TokenId};

    #[cfg(all(feature = "apple", target_os = "macos"))]
    static TOY_METAL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(all(feature = "apple", target_os = "macos"))]
    struct ToyMetalEnvGuard {
        _guard: std::sync::MutexGuard<'static, ()>,
        previous: Option<std::ffi::OsString>,
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    impl ToyMetalEnvGuard {
        fn new() -> Self {
            Self {
                _guard: TOY_METAL_ENV_LOCK.lock().expect("lock toy Metal env"),
                previous: std::env::var_os("RVLLM_APPLE_TOY_METAL"),
            }
        }
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    impl Drop for ToyMetalEnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.as_ref() {
                std::env::set_var("RVLLM_APPLE_TOY_METAL", previous);
            } else {
                std::env::remove_var("RVLLM_APPLE_TOY_METAL");
            }
        }
    }

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
    fn paged_kv_reservation_rolls_back_all_requests_on_exhaustion() {
        let mut e = Engine::new()
            .with_paged_kv_config(PagedKvConfig::apple_v1(2, 4), [1; 32])
            .unwrap();
        e.enqueue_request(Request::new(ReqId(1), vec![TokenId(1); 33], 1))
            .unwrap();
        e.enqueue_request(Request::new(ReqId(2), vec![TokenId(2); 33], 1))
            .unwrap();

        let mut plan = e.scheduler.schedule();
        assert!(e.prepare_paged_kv_step(&mut plan).is_err());
        let pool = e.paged_kv.as_ref().unwrap();
        assert_eq!(pool.stats().used_pages, 0);
        assert_eq!(
            pool.view(pool.chain_for_owner(ReqId(1)).unwrap())
                .unwrap()
                .token_len,
            0
        );
        assert_eq!(
            pool.view(pool.chain_for_owner(ReqId(2)).unwrap())
                .unwrap()
                .token_len,
            0
        );
    }

    #[test]
    fn cancellation_releases_pages_and_request_id_reuse_gets_new_generation() {
        let mut e = Engine::new()
            .with_paged_kv_config(PagedKvConfig::apple_v1(4, 1), [2; 32])
            .unwrap();
        e.enqueue_request(Request::new(ReqId(7), vec![TokenId(3); 33], 1))
            .unwrap();
        let mut plan = e.scheduler.schedule();
        let reservation = e.prepare_paged_kv_step(&mut plan).unwrap();
        let old = e.scheduler.kv_chain_for(ReqId(7)).unwrap();
        // Treat the reservation as a completed accelerator step.
        drop(reservation);
        e.scheduler.commit_prefill(&[(ReqId(7), 33)]).unwrap();
        assert_eq!(e.paged_kv_stats().unwrap().used_pages, 2);

        e.set_request_backpressured(ReqId(7), true).unwrap();
        assert!(matches!(e.scheduler.schedule(), BatchPlan::Idle));
        assert_eq!(e.scheduler.kv_chain_for(ReqId(7)), Some(old));
        assert_eq!(e.paged_kv_stats().unwrap().used_pages, 2);
        assert!(e.cancel_request(ReqId(7)).unwrap());
        assert_eq!(e.paged_kv_stats().unwrap().used_pages, 0);
        e.enqueue_request(Request::new(ReqId(7), vec![TokenId(4)], 1))
            .unwrap();
        let new = e.scheduler.kv_chain_for(ReqId(7)).unwrap();
        assert_eq!(old.id, new.id);
        assert_ne!(old.generation, new.generation);
        assert!(matches!(
            e.paged_kv.as_mut().unwrap().release_chain(old),
            Err(PagedKvError::StaleChain { .. })
        ));
        assert_eq!(
            e.paged_kv.as_ref().unwrap().chain_for_owner(ReqId(7)),
            Some(new)
        );
    }

    #[test]
    fn cached_attachment_restores_exact_cursor_and_defers_cancel_release() {
        const LAYOUT: [u8; 32] = [17; 32];
        let identity = CacheIdentity {
            namespace: CacheNamespace::new("engine-cache-test").unwrap(),
            model: [1; 32],
            tokenizer: [2; 32],
            adapter: None,
            kv_layout: LAYOUT,
            numeric_path: [3; 32],
            format_version: 1,
        };
        let mut cache = PagedPromptCache::new(
            PromptCacheConfig {
                hot_bytes: 4096,
                warm_bytes: 0,
                protected_fraction_percent: 80,
                frequency_aging_interval: 100,
            },
            LAYOUT,
            64,
        )
        .unwrap();
        let prompt: Vec<_> = (0..65).map(TokenId).collect();
        let mut engine = Engine::new()
            .with_paged_kv_config(PagedKvConfig::apple_v1(8, 4), LAYOUT)
            .unwrap();
        let source = engine
            .paged_kv_pool_mut()
            .unwrap()
            .allocate_chain(ReqId(100), 65)
            .unwrap();
        cache
            .promote_hot(
                engine.paged_kv_pool_mut().unwrap(),
                identity.clone(),
                &prompt,
                source,
            )
            .unwrap();
        engine
            .paged_kv_pool_mut()
            .unwrap()
            .release_chain(source)
            .unwrap();
        let mut unavailable = crate::paged_prompt_cache::UnavailableKvPageIo;
        let attachment = cache
            .attach(
                engine.paged_kv_pool_mut().unwrap(),
                &mut unavailable,
                ReqId(7),
                &identity,
                &prompt,
                || false,
            )
            .unwrap()
            .unwrap();
        assert_eq!(attachment.matched_tokens, 64);
        engine
            .enqueue_request_with_cache_attachment(
                Request::new(ReqId(7), prompt.clone(), 1),
                &attachment,
            )
            .unwrap();
        let plan = engine.scheduler.schedule();
        let BatchPlan::Prefill {
            query_start_positions,
            cu_seqlens_q,
            ..
        } = plan
        else {
            panic!("expected one-token private-tail prefill")
        };
        assert_eq!(query_start_positions, vec![64]);
        assert_eq!(cu_seqlens_q, vec![0, 1]);

        assert!(engine.cancel_request(ReqId(7)).unwrap());
        assert!(engine
            .paged_kv_pool()
            .unwrap()
            .view(attachment.chain)
            .is_ok());
        engine
            .release_cache_attachment(&mut cache, &attachment)
            .unwrap();
        assert!(engine
            .paged_kv_pool()
            .unwrap()
            .chain_for_owner(ReqId(7))
            .is_none());
        cache
            .handle_memory_pressure(
                engine.paged_kv_pool_mut().unwrap(),
                MemoryPressure::Critical,
            )
            .unwrap();
        assert_eq!(engine.paged_kv_stats().unwrap().used_pages, 0);
    }

    #[test]
    fn failed_stale_cache_release_keeps_engine_ownership_guard() {
        const LAYOUT: [u8; 32] = [19; 32];
        let identity = CacheIdentity {
            namespace: CacheNamespace::new("engine-stale-test").unwrap(),
            model: [1; 32],
            tokenizer: [2; 32],
            adapter: None,
            kv_layout: LAYOUT,
            numeric_path: [3; 32],
            format_version: 1,
        };
        let mut cache = PagedPromptCache::new(
            PromptCacheConfig {
                hot_bytes: 4096,
                warm_bytes: 0,
                protected_fraction_percent: 80,
                frequency_aging_interval: 100,
            },
            LAYOUT,
            64,
        )
        .unwrap();
        let prompt: Vec<_> = (0..33).map(TokenId).collect();
        let mut engine = Engine::new()
            .with_paged_kv_config(PagedKvConfig::apple_v1(4, 3), LAYOUT)
            .unwrap();
        let source = engine
            .paged_kv_pool_mut()
            .unwrap()
            .allocate_chain(ReqId(100), 33)
            .unwrap();
        cache
            .promote_hot(
                engine.paged_kv_pool_mut().unwrap(),
                identity.clone(),
                &prompt,
                source,
            )
            .unwrap();
        engine
            .paged_kv_pool_mut()
            .unwrap()
            .release_chain(source)
            .unwrap();
        let mut unavailable = crate::paged_prompt_cache::UnavailableKvPageIo;
        let attachment = cache
            .attach(
                engine.paged_kv_pool_mut().unwrap(),
                &mut unavailable,
                ReqId(8),
                &identity,
                &prompt,
                || false,
            )
            .unwrap()
            .unwrap();
        engine
            .enqueue_request_with_cache_attachment(Request::new(ReqId(8), prompt, 1), &attachment)
            .unwrap();
        engine
            .paged_kv_pool_mut()
            .unwrap()
            .release_chain(attachment.chain)
            .unwrap();
        assert!(engine
            .release_cache_attachment(&mut cache, &attachment)
            .is_err());
        assert!(engine.cache_managed_kv.contains(&ReqId(8)));
        let release = cache
            .handle_memory_pressure(
                engine.paged_kv_pool_mut().unwrap(),
                MemoryPressure::Critical,
            )
            .unwrap();
        assert!(release.hot.is_empty());
    }

    #[test]
    fn failed_terminal_warm_capture_does_not_leak_deferred_chain() {
        struct FailingCapture;
        impl crate::paged_prompt_cache::KvPageIo for FailingCapture {
            fn page_bytes(&self) -> Option<usize> {
                Some(64)
            }

            fn capture_page(
                &mut self,
                _page: rvllm_core::BlockId,
            ) -> std::result::Result<std::sync::Arc<[u8]>, crate::KvPageIoError> {
                Err(crate::KvPageIoError::DeviceCopyFailed(
                    "injected".to_owned(),
                ))
            }

            fn restore_page(
                &mut self,
                _page: rvllm_core::BlockId,
                _bytes: &[u8],
            ) -> std::result::Result<(), crate::KvPageIoError> {
                unreachable!()
            }

            fn copy_page(
                &mut self,
                _copy: crate::CowPageCopy,
            ) -> std::result::Result<(), crate::KvPageIoError> {
                unreachable!()
            }
        }

        const LAYOUT: [u8; 32] = [23; 32];
        let identity = CacheIdentity {
            namespace: CacheNamespace::new("terminal-capture-failure").unwrap(),
            model: [1; 32],
            tokenizer: [2; 32],
            adapter: None,
            kv_layout: LAYOUT,
            numeric_path: [3; 32],
            format_version: 1,
        };
        let prompt: Vec<_> = (0..33).map(TokenId).collect();
        let mut engine = Engine::new()
            .with_paged_kv_config(PagedKvConfig::apple_v1(4, 2), LAYOUT)
            .unwrap();
        engine
            .enqueue_request(Request::new(ReqId(9), prompt.clone(), 1))
            .unwrap();
        engine.defer_paged_kv_release(ReqId(9)).unwrap();
        let mut plan = engine.scheduler.schedule();
        let reservation = engine.prepare_paged_kv_step(&mut plan).unwrap();
        drop(reservation);
        engine.scheduler.commit_prefill(&[(ReqId(9), 33)]).unwrap();
        let chain = engine.scheduler.kv_chain_for(ReqId(9)).unwrap();
        let terminal_token = TokenId(99);
        engine
            .scheduler
            .commit_decode(&[(ReqId(9), terminal_token)]);
        assert!(!engine.scheduler.request_is_alive(ReqId(9)));
        assert!(engine.paged_kv_pool().unwrap().view(chain).is_ok());

        let mut warm = PagedPromptCache::new(
            PromptCacheConfig {
                hot_bytes: 0,
                warm_bytes: 4096,
                protected_fraction_percent: 80,
                frequency_aging_interval: 100,
            },
            LAYOUT,
            64,
        )
        .unwrap();
        let mut io = FailingCapture;
        assert!(warm
            .capture_warm(
                engine.paged_kv_pool().unwrap(),
                &mut io,
                identity,
                &prompt,
                chain,
                || false,
            )
            .is_err());
        assert_eq!(terminal_token, TokenId(99));
        engine.release_deferred_paged_kv(ReqId(9)).unwrap();
        assert_eq!(engine.paged_kv_stats().unwrap().used_pages, 0);
    }

    #[test]
    fn chunked_prefill_and_decode_reserve_absolute_request_pages() {
        let mut e = Engine::new()
            .with_paged_kv_config(PagedKvConfig::apple_v1(8, 2), [4; 32])
            .unwrap();
        e.scheduler = Scheduler::with_config(crate::scheduler::SchedulerConfig {
            max_prefill_tokens: 32,
            ..crate::scheduler::SchedulerConfig::default()
        });
        e.enqueue_request(Request::new(ReqId(5), vec![TokenId(6); 65], 2))
            .unwrap();

        for (expected_start, expected_len) in [(0, 32), (32, 32), (64, 1)] {
            let mut plan = e.scheduler.schedule();
            let reservation = e.prepare_paged_kv_step(&mut plan).unwrap();
            let BatchPlan::Prefill {
                query_start_positions,
                context_lens,
                ..
            } = plan
            else {
                panic!("expected prefill")
            };
            assert_eq!(query_start_positions, vec![expected_start]);
            assert_eq!(context_lens, vec![expected_start + expected_len]);
            drop(reservation);
            e.scheduler
                .commit_prefill(&[(ReqId(5), expected_len)])
                .unwrap();
        }
        let handle = e.scheduler.kv_chain_for(ReqId(5)).unwrap();
        assert_eq!(
            e.paged_kv.as_ref().unwrap().view(handle).unwrap().token_len,
            65
        );

        // First decode reads/writes position 64, already covered by prefill.
        let mut first_decode = e.scheduler.schedule();
        let reservation = e.prepare_paged_kv_step(&mut first_decode).unwrap();
        drop(reservation);
        e.scheduler.commit_decode(&[(ReqId(5), TokenId(7))]);

        // The next decode writes the generated token at absolute position 65,
        // extending the same request-owned third page without batch ordinals.
        let mut second_decode = e.scheduler.schedule();
        let reservation = e.prepare_paged_kv_step(&mut second_decode).unwrap();
        drop(reservation);
        assert_eq!(
            e.paged_kv.as_ref().unwrap().view(handle).unwrap().token_len,
            66
        );
    }

    #[test]
    fn engine_fails_closed_on_shared_partial_tail_without_device_cow() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(4, 3)).unwrap();
        let source = pool.allocate_chain(ReqId(100), 17).unwrap();
        let fork = pool.fork_chain(source, ReqId(1)).unwrap();
        let before = pool.view(fork).unwrap().pages.to_vec();
        let mut request = Request::new(ReqId(1), vec![TokenId(9); 33], 1);
        request.bind_kv_chain(fork);
        request.finish_restore(17).unwrap();

        let mut e = Engine::new().with_paged_kv_pool(pool, [5; 32]);
        e.scheduler.enqueue(request);
        let mut plan = e.scheduler.schedule();
        assert!(e.prepare_paged_kv_step(&mut plan).is_err());
        let pool = e.paged_kv.as_ref().unwrap();
        let view = pool.view(fork).unwrap();
        assert_eq!(view.token_len, 17);
        assert_eq!(view.pages, before.as_slice());
        assert_eq!(pool.page_ref_count(before[0]), Some(2));
    }

    #[test]
    fn inserting_a_cache_miss_cannot_reassign_an_active_requests_pages() {
        let mut e = Engine::new()
            .with_paged_kv_config(PagedKvConfig::apple_v1(8, 4), [6; 32])
            .unwrap();
        e.enqueue_request(Request::new(ReqId(1), vec![TokenId(1)], 3))
            .unwrap();
        let mut prefill_a = e.scheduler.schedule();
        drop(e.prepare_paged_kv_step(&mut prefill_a).unwrap());
        e.scheduler.commit_prefill(&[(ReqId(1), 1)]).unwrap();
        let chain_a = e.scheduler.kv_chain_for(ReqId(1)).unwrap();
        let pages_a = e
            .paged_kv
            .as_ref()
            .unwrap()
            .view(chain_a)
            .unwrap()
            .pages
            .to_vec();

        let mut decode_a = e.scheduler.schedule();
        drop(e.prepare_paged_kv_step(&mut decode_a).unwrap());
        e.scheduler.commit_decode(&[(ReqId(1), TokenId(2))]);

        // A now has a future decode deadline, so newly inserted B gets a
        // chunked prefill without changing A's ownership.
        e.enqueue_request(Request::new(ReqId(2), vec![TokenId(3); 33], 2))
            .unwrap();
        let mut prefill_b = e.scheduler.schedule();
        assert!(matches!(
            &prefill_b,
            BatchPlan::Prefill { req_ids, .. } if req_ids == &vec![ReqId(2)]
        ));
        drop(e.prepare_paged_kv_step(&mut prefill_b).unwrap());
        e.scheduler.commit_prefill(&[(ReqId(2), 33)]).unwrap();
        assert_eq!(
            e.paged_kv.as_ref().unwrap().chain_for_owner(ReqId(1)),
            Some(chain_a)
        );
        assert_eq!(
            e.paged_kv.as_ref().unwrap().view(chain_a).unwrap().pages,
            pages_a
        );

        let decode = e.scheduler.schedule();
        let BatchPlan::Decode {
            req_ids, kv_chains, ..
        } = decode
        else {
            panic!("expected decode batch")
        };
        assert_eq!(req_ids, vec![ReqId(2), ReqId(1)]);
        assert_eq!(kv_chains[1], Some(chain_a));
    }

    #[cfg(feature = "apple")]
    #[derive(Default)]
    struct FailingPagedBackend {
        handoffs: std::sync::Arc<std::sync::Mutex<Vec<rvllm_apple::HandoffCapsule>>>,
        fail_launch: bool,
        fail_collect: bool,
    }

    #[cfg(feature = "apple")]
    impl rvllm_apple::AppleBackend for FailingPagedBackend {
        fn prepare(&mut self, _plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
            Ok(())
        }

        fn launch_prefill(
            &mut self,
            handoff: &rvllm_apple::HandoffCapsule,
        ) -> Result<rvllm_apple::AppleLaunchTicket> {
            self.handoffs.lock().unwrap().push(handoff.clone());
            if self.fail_launch {
                return Err(apple_unavailable_error(
                    "test_launch_failure",
                    "test-backend",
                ));
            }
            Ok(rvllm_apple::AppleLaunchTicket {
                step_id: 1,
                kind: rvllm_apple::AppleLaunchKind::Prefill,
                bucket: None,
            })
        }

        fn launch_rollout(
            &mut self,
            handoff: &rvllm_apple::HandoffCapsule,
            _bucket: Option<rvllm_apple::RolloutBucket>,
        ) -> Result<rvllm_apple::AppleLaunchTicket> {
            self.launch_prefill(handoff)
        }

        fn collect(
            &mut self,
            _ticket: rvllm_apple::AppleLaunchTicket,
        ) -> Result<Vec<rvllm_apple::StepToken>> {
            if self.fail_collect {
                return Err(apple_unavailable_error(
                    "test_collect_failure",
                    "test-backend",
                ));
            }
            Ok(Vec::new())
        }
    }

    #[cfg(feature = "apple")]
    #[derive(Default)]
    struct MultiTicketBackend {
        next_step_id: u64,
    }

    #[cfg(feature = "apple")]
    impl rvllm_apple::AppleBackend for MultiTicketBackend {
        fn prepare(&mut self, _plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
            Ok(())
        }

        fn launch_prefill(
            &mut self,
            _handoff: &rvllm_apple::HandoffCapsule,
        ) -> Result<rvllm_apple::AppleLaunchTicket> {
            let ticket = rvllm_apple::AppleLaunchTicket {
                step_id: self.next_step_id,
                kind: rvllm_apple::AppleLaunchKind::Prefill,
                bucket: None,
            };
            self.next_step_id += 1;
            Ok(ticket)
        }

        fn launch_rollout(
            &mut self,
            _handoff: &rvllm_apple::HandoffCapsule,
            bucket: Option<rvllm_apple::RolloutBucket>,
        ) -> Result<rvllm_apple::AppleLaunchTicket> {
            let ticket = rvllm_apple::AppleLaunchTicket {
                step_id: self.next_step_id,
                kind: rvllm_apple::AppleLaunchKind::Rollout,
                bucket,
            };
            self.next_step_id += 1;
            Ok(ticket)
        }

        fn collect(
            &mut self,
            _ticket: rvllm_apple::AppleLaunchTicket,
        ) -> Result<Vec<rvllm_apple::StepToken>> {
            Ok(Vec::new())
        }
    }

    #[cfg(feature = "apple")]
    struct TerminalPagedBackend;

    #[cfg(feature = "apple")]
    impl rvllm_apple::AppleBackend for TerminalPagedBackend {
        fn prepare(&mut self, _plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
            Ok(())
        }

        fn launch_prefill(
            &mut self,
            _handoff: &rvllm_apple::HandoffCapsule,
        ) -> Result<rvllm_apple::AppleLaunchTicket> {
            Ok(rvllm_apple::AppleLaunchTicket {
                step_id: 1,
                kind: rvllm_apple::AppleLaunchKind::Prefill,
                bucket: None,
            })
        }

        fn launch_rollout(
            &mut self,
            _handoff: &rvllm_apple::HandoffCapsule,
            bucket: Option<rvllm_apple::RolloutBucket>,
        ) -> Result<rvllm_apple::AppleLaunchTicket> {
            Ok(rvllm_apple::AppleLaunchTicket {
                step_id: 2,
                kind: rvllm_apple::AppleLaunchKind::Rollout,
                bucket,
            })
        }

        fn collect(
            &mut self,
            ticket: rvllm_apple::AppleLaunchTicket,
        ) -> Result<Vec<rvllm_apple::StepToken>> {
            if ticket.kind == rvllm_apple::AppleLaunchKind::Rollout {
                Ok(vec![rvllm_apple::StepToken {
                    req_id: ReqId(77),
                    token_id: TokenId(12),
                    finished: true,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }

    #[cfg(feature = "apple")]
    fn paged_test_plan() -> rvllm_apple::AppleRuntimePlan {
        rvllm_apple::AppleRuntimePlan {
            target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple test", 0),
            mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: AneComputeProfile::AnyAvailable,
            ane_fallback_policy: AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 1,
            ane_intermediate_size: 1,
            ane_num_layers: 1,
            model_layout_hash: [3; 32],
            weights_path: None,
        }
    }

    #[test]
    #[cfg(feature = "apple")]
    fn launch_and_collect_failures_rollback_paged_reservations() {
        for fail_collect in [false, true] {
            let handoffs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let backend = FailingPagedBackend {
                handoffs: std::sync::Arc::clone(&handoffs),
                fail_launch: !fail_collect,
                fail_collect,
            };
            let mut e = Engine::new()
                .with_apple_backend(Box::new(backend))
                .with_apple_runtime_plan(paged_test_plan())
                .unwrap()
                .with_paged_kv_config(PagedKvConfig::apple_v1(4, 2), [0xCC; 32])
                .unwrap();
            e.enqueue_request(Request::new(ReqId(42), vec![TokenId(8); 33], 1))
                .unwrap();

            let result = match e.step_launch() {
                Ok(pending) => pending.collect().map(|_| ()),
                Err(error) => Err(error),
            };
            assert!(result.is_err());
            assert_eq!(e.paged_kv_stats().unwrap().used_pages, 0);
            let capsule = handoffs.lock().unwrap().first().unwrap().clone();
            assert!(capsule.is_well_formed());
            assert_eq!(capsule.max_blocks_per_seq, 2);
            assert_eq!(capsule.block_tables.len(), 2);
            assert_eq!(capsule.slot_mapping.len(), 33);
            assert_eq!(capsule.model_layout_fingerprint, [0xCC; 32]);
        }
    }

    #[test]
    #[cfg(feature = "apple")]
    fn owned_submissions_exclude_in_flight_requests_and_collect_out_of_order() {
        let mut e = Engine::new()
            .with_apple_backend(Box::new(MultiTicketBackend::default()))
            .with_apple_runtime_plan(paged_test_plan())
            .unwrap();
        e.scheduler = Scheduler::with_config(crate::scheduler::SchedulerConfig {
            max_prefill_tokens: 1,
            ..crate::scheduler::SchedulerConfig::default()
        });
        for req_id in 1..=3 {
            e.scheduler
                .enqueue(Request::new(ReqId(req_id), vec![TokenId(req_id as u32)], 1));
        }

        let first = e.step_submit().expect("submit first request");
        let second = e.step_submit().expect("submit second request");
        let third = e.step_submit().expect("submit third request");
        let plan_id = |step: &SubmittedStep| step.plan().unwrap().req_ids()[0];
        assert_eq!(
            [plan_id(&first), plan_id(&second), plan_id(&third)],
            [ReqId(1), ReqId(2), ReqId(3)]
        );
        let idle = e.step_submit().expect("all live requests are owned");
        assert!(matches!(idle.plan(), Some(BatchPlan::Idle)));
        e.collect_submitted(idle).expect("collect idle step");

        e.collect_submitted(second).expect("collect middle");
        e.collect_submitted(third).expect("collect last");
        e.collect_submitted(first).expect("collect first");
        assert_eq!(e.scheduler.num_alive(), 3);
    }

    #[test]
    #[cfg(feature = "apple")]
    fn cancellation_defers_paged_kv_release_until_submitted_completion() {
        let mut e = Engine::new()
            .with_apple_backend(Box::new(MultiTicketBackend::default()))
            .with_apple_runtime_plan(paged_test_plan())
            .unwrap()
            .with_paged_kv_config(PagedKvConfig::apple_v1(2, 2), [3; 32])
            .unwrap();
        e.enqueue_request(Request::new(ReqId(9), vec![TokenId(4)], 1))
            .unwrap();
        let submitted = e.step_submit().expect("submit prefill");
        assert_eq!(e.paged_kv_stats().unwrap().used_pages, 1);
        assert!(e.cancel_request(ReqId(9)).unwrap());
        assert_eq!(
            e.paged_kv_stats().unwrap().used_pages,
            1,
            "GPU-owned pages must remain live until completion"
        );
        assert!(e.collect_submitted(submitted).unwrap().is_empty());
        assert_eq!(e.paged_kv_stats().unwrap().used_pages, 0);
    }

    #[test]
    #[cfg(feature = "apple")]
    fn terminal_backend_output_releases_request_owned_pages() {
        let mut e = Engine::new()
            .with_apple_backend(Box::new(TerminalPagedBackend))
            .with_apple_runtime_plan(paged_test_plan())
            .unwrap()
            .with_paged_kv_config(PagedKvConfig::apple_v1(2, 2), [0xDD; 32])
            .unwrap();
        e.enqueue_request(Request::new(ReqId(77), vec![TokenId(11)], 8))
            .unwrap();
        e.step_launch().unwrap().collect().unwrap();
        assert_eq!(e.paged_kv_stats().unwrap().used_pages, 1);

        let outputs = e.step_launch().unwrap().collect().unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].finished);
        assert_eq!(e.paged_kv_stats().unwrap().used_pages, 0);
        assert!(!e.has_pending_work());
    }
    #[test]
    #[cfg(feature = "apple")]
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn apple_runtime_plan_fails_closed_on_unsupported_platform() {
        use rvllm_apple::AppleRuntimePlan;

        let plan = AppleRuntimePlan {
            target: rvllm_apple::device::AppleAcceleratorTarget::from_device_name(
                "Apple M4 Max",
                1,
            ),
            mode: rvllm_apple::plan::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: true,
            strict_ane: false,
            ane_compute_profile: AneComputeProfile::AnyAvailable,
            ane_fallback_policy: AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 1,
            ane_intermediate_size: 1,
            ane_num_layers: 1,
            model_layout_hash: [0u8; 32],
            weights_path: None,
        };

        let err = match Engine::new().with_apple_runtime_plan(plan) {
            Ok(_) => panic!("unsupported platform must reject Apple backend preparation"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            rvllm_core::RvllmError::Apple {
                err: AppleError::FeatureNotAvailable {
                    op: "apple_backend_unavailable_on_target",
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    #[ignore = "requires Metal hardware and explicit shader assets; excluded from host checks"]
    #[cfg(feature = "apple")]
    #[cfg(target_os = "macos")]
    fn e2e_apple_backend_wiring_toy_metal() {
        use rvllm_apple::AppleRuntimePlan;

        let seed = TokenId(11);
        let plan = AppleRuntimePlan {
            target: rvllm_apple::device::AppleAcceleratorTarget::from_device_name(
                "Apple M4 Max",
                1,
            ),
            mode: rvllm_apple::plan::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: AneComputeProfile::AnyAvailable,
            ane_fallback_policy: AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 1,
            ane_intermediate_size: 1,
            ane_num_layers: 1,
            model_layout_hash: [0u8; 32],
            weights_path: None,
        };

        let e = match Engine::new()
            .with_apple_backend(Box::new(ToyMetalBackend::new()))
            .with_apple_runtime_plan(plan)
        {
            Ok(v) => v,
            Err(e) => panic!("unexpected runtime plan error: {e}"),
        };
        let mut e = e;
        e.scheduler.enqueue(Request::new(ReqId(1), vec![seed], 2));

        let t1 = e.step_launch().unwrap();
        let outputs1 = t1.collect().unwrap();
        assert!(outputs1.is_empty(), "Prefill returns empty tokens");

        let t2 = e.step_launch().unwrap();
        let outputs2 = t2.collect().unwrap();
        assert_eq!(
            outputs2.len(),
            1,
            "Decode should return tokens from Metal backend"
        );
        assert_eq!(outputs2[0].new_token, seed);
    }

    #[test]
    #[cfg(feature = "apple")]
    #[cfg(target_os = "macos")]
    fn apple_default_metal_route_requires_model_dir_unless_toy_is_explicitly_enabled() {
        use rvllm_apple::AppleRuntimePlan;

        let _guard = ToyMetalEnvGuard::new();
        std::env::remove_var("RVLLM_APPLE_TOY_METAL");

        let plan = AppleRuntimePlan {
            target: rvllm_apple::device::AppleAcceleratorTarget::from_device_name(
                "Apple M4 Max",
                1,
            ),
            mode: rvllm_apple::plan::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: AneComputeProfile::AnyAvailable,
            ane_fallback_policy: AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 1,
            ane_intermediate_size: 1,
            ane_num_layers: 1,
            model_layout_hash: [0u8; 32],
            weights_path: None,
        };

        // Check route selection without preparing a device or loading shaders.
        // Actual Metal execution belongs to the ignored integration fixture.
        let err = match default_apple_backend_for_plan(&plan) {
            Ok(_) => panic!("default Metal route should require a model directory"),
            Err(err) => err,
        };
        assert!(
            format!("{err}").contains("metal_model_dir_required"),
            "unexpected default Metal route error: {err}"
        );

        std::env::set_var("RVLLM_APPLE_TOY_METAL", "1");
        default_apple_backend_for_plan(&plan)
            .expect("toy Metal route should be available only with explicit env opt-in");
    }

    #[test]
    #[cfg(feature = "apple")]
    #[cfg(not(target_os = "macos"))]
    fn private_ane_mode_fails_closed_without_ane_target() {
        let plan = AppleRuntimePlan {
            target: rvllm_apple::device::AppleAcceleratorTarget::from_device_name(
                "Apple M4 Max",
                1,
            ),
            mode: rvllm_apple::plan::AppleBackendMode::MetalPrefillAneFfnRollout,
            rollout_bucket: Some(rvllm_apple::plan::RolloutBucket { seqs: 4, tokens: 4 }),
            rollout_tokens: 1,
            private_ane_opt_in: true,
            strict_ane: false,
            ane_compute_profile: AneComputeProfile::AnyAvailable,
            ane_fallback_policy: AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 1,
            ane_intermediate_size: 1,
            ane_num_layers: 1,
            model_layout_hash: [0u8; 32],
            weights_path: None,
        };
        let err = match Engine::new().with_apple_runtime_plan(plan) {
            Ok(_) => panic!("private ANE preparation must fail closed on non-macOS"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            rvllm_core::RvllmError::Apple {
                err: AppleError::FeatureNotAvailable {
                    op: "private_ane_unavailable",
                    ..
                },
                ..
            }
        ));
    }
}

#[cfg(feature = "apple")]
fn backend_plan_is_enabled(plan: &AppleRuntimePlan) -> bool {
    !matches!(plan.mode, AppleBackendModeImpl::MlxPrototype)
}

#[cfg(feature = "apple")]
fn enforce_apple_mode_availability(plan: &AppleRuntimePlan) -> Result<()> {
    if plan.mode.requires_private_ane() && !cfg!(target_os = "macos") {
        return Err(apple_unavailable_error(
            "private_ane_unavailable",
            "private-ane",
        ));
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
    if matches!(
        runtime.apple_backend_mode(),
        rvllm_core::AppleBackendMode::Disabled
    ) {
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
        strict_ane: runtime.strict_ane(),
        ane_compute_profile: runtime.ane_compute_profile(),
        ane_fallback_policy: runtime.ane_fallback_policy(),
        ane_hidden_size: runtime.ane_hidden_size(),
        ane_intermediate_size: runtime.ane_intermediate_size(),
        ane_num_layers: runtime.ane_num_layers(),
        model_layout_hash: *runtime.model_layout_hash(),
        weights_path: runtime.weights_path().map(|p| p.to_path_buf()),
    };
    Ok(Some(plan))
}

#[cfg(feature = "apple")]
fn default_apple_backend_for_plan(plan: &AppleRuntimePlan) -> Result<Box<dyn AppleBackend>> {
    if plan.mode.requires_private_ane() {
        #[cfg(target_os = "macos")]
        {
            return Ok(Box::new(ProductionAppleBackend::new()));
        }
        #[cfg(not(target_os = "macos"))]
        return Err(apple_unavailable_error(
            "private_ane_unavailable",
            "private-ane",
        ));
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if let Some(model_dir) = plan.weights_path.clone() {
            #[cfg(target_os = "macos")]
            return Ok(Box::new(ModelMetalBackend::new(model_dir)));
            #[cfg(target_os = "ios")]
            return ModelMetalBackend::from_model_package_path(model_dir)
                .map(|backend| Box::new(backend) as Box<dyn AppleBackend>);
        }
        #[cfg(target_os = "macos")]
        if std::env::var("RVLLM_APPLE_TOY_METAL").ok().as_deref() == Some("1") {
            return Ok(Box::new(ToyMetalBackend::new()));
        }
        return Err(apple_unavailable_error(
            "metal_model_dir_required",
            "apple-metal",
        ));
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let _ = plan;
        Err(apple_unavailable_error(
            "apple_backend_unavailable_on_target",
            "apple-runtime",
        ))
    }
}

#[cfg(feature = "apple")]
fn match_apple_mode_to_handoff_kind(mode: AppleBackendModeImpl) -> HandoffKind {
    match mode {
        AppleBackendModeImpl::MetalOnly
        | AppleBackendModeImpl::MlxPrototype
        | AppleBackendModeImpl::MetalPrefillMetalDecode => HandoffKind::MetalPrefillToMetalDecode,
        AppleBackendModeImpl::MetalPrefillAneFfnRollout => HandoffKind::MetalPrefillToAneFfnRollout,
        AppleBackendModeImpl::MetalPrefillAneRolloutExperimental => {
            HandoffKind::MetalPrefillToAneRolloutExperimental
        }
        _ => HandoffKind::MetalPrefillToMetalDecode,
    }
}
