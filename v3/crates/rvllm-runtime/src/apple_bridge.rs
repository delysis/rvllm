//! Feature-gated bridge from rvLLM scheduler output to Apple backend capsules.
//!
//! This module deliberately has no Metal/ANE FFI. It only translates the real
//! `BatchPlan` values into host-testable `rvllm-apple` contracts.

use rvllm_apple::{select_rollout_bucket, HandoffCapsule, HandoffKind, RolloutBucket};
use rvllm_core::{AppleCtx, AppleError, ReqId, Result, RvllmError, TokenId};

use crate::scheduler::BatchPlan;

fn apple_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "apple-bridge",
        op,
        device: "apple-silicon",
    }
}

fn err(reason: &'static str, op: &'static str) -> RvllmError {
    RvllmError::apple(AppleError::HandoffMalformed { reason }, apple_ctx(op))
}

fn shape_err(seqs: u32, tokens: u32, op: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::ShapeBucketMissing { seqs, tokens },
        apple_ctx(op),
    )
}

/// One speculative branch rooted at a scheduler decode request.
///
/// The core scheduler still owns request lifecycle and decode ordering. Apple
/// rollout policy only decides how these branch candidates map onto static ANE
/// rollout rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AneSpeculativeBranch {
    pub req_id: ReqId,
    pub branch_id: u32,
    pub tokens: Vec<TokenId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AneRolloutBranchPlan {
    pub tokens_per_branch: u32,
    pub branches: Vec<AneSpeculativeBranch>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AneRolloutSlot {
    pub req_id: ReqId,
    pub branch_id: u32,
    pub decode_slot: u32,
    pub token_offset: u32,
    pub token_len: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AneRolloutBatch {
    pub bucket: RolloutBucket,
    pub capsule: HandoffCapsule,
    pub slots: Vec<AneRolloutSlot>,
}

/// Convert a scheduler prefill plan into a Metal/ANE handoff capsule.
///
/// Positions and context lengths are derived from `cu_seqlens_q`: for each
/// sequence, position is `len - 1` and context length is `len`.
pub fn handoff_from_prefill_plan(plan: &BatchPlan, kind: HandoffKind) -> Result<HandoffCapsule> {
    let BatchPlan::Prefill {
        req_ids,
        prompt_tokens_flat,
        cu_seqlens_q,
    } = plan else {
        return Err(err("expected BatchPlan::Prefill", "prefill_handoff"));
    };

    if cu_seqlens_q.len() != req_ids.len() + 1 {
        return Err(err("cu_seqlens length must equal req_ids + 1", "prefill_handoff"));
    }
    if cu_seqlens_q.first().copied() != Some(0) {
        return Err(err("cu_seqlens must start at 0", "prefill_handoff"));
    }
    if cu_seqlens_q.last().copied() != Some(prompt_tokens_flat.len() as u32) {
        return Err(err("cu_seqlens must end at token length", "prefill_handoff"));
    }

    let mut positions = Vec::with_capacity(req_ids.len());
    let mut context_lens = Vec::with_capacity(req_ids.len());
    for span in cu_seqlens_q.windows(2) {
        let len = span[1].saturating_sub(span[0]);
        if len == 0 {
            return Err(err("empty prefill sequence", "prefill_handoff"));
        }
        positions.push(len - 1);
        context_lens.push(len);
    }

    let capsule = HandoffCapsule::new(
        kind,
        req_ids.clone(),
        prompt_tokens_flat.clone(),
        cu_seqlens_q.clone(),
        positions,
        context_lens,
    );
    capsule.validate().map(|()| capsule)
}

/// Convert a decode plan into a one-token-per-sequence Apple handoff capsule.
pub fn handoff_from_decode_plan(plan: &BatchPlan, kind: HandoffKind) -> Result<HandoffCapsule> {
    let BatchPlan::Decode {
        req_ids,
        last_tokens,
        positions,
        context_lens,
        ..
    } = plan else {
        return Err(err("expected BatchPlan::Decode", "decode_handoff"));
    };

    if req_ids.len() != last_tokens.len()
        || req_ids.len() != positions.len()
        || req_ids.len() != context_lens.len()
    {
        return Err(err("decode vector lengths differ", "decode_handoff"));
    }

    let mut cu = Vec::with_capacity(req_ids.len() + 1);
    cu.push(0);
    for i in 0..req_ids.len() {
        cu.push((i + 1) as u32);
    }

    let capsule = HandoffCapsule::new(
        kind,
        req_ids.clone(),
        last_tokens.clone(),
        cu,
        positions.clone(),
        context_lens.clone(),
    );
    capsule.validate().map(|()| capsule)
}

pub fn rollout_bucket_for_decode(plan: &BatchPlan, tokens_per_rollout: u32) -> Result<RolloutBucket> {
    let BatchPlan::Decode { req_ids, .. } = plan else {
        return Err(err("expected BatchPlan::Decode", "rollout_bucket"));
    };
    select_rollout_bucket(req_ids.len() as u32, tokens_per_rollout).ok_or_else(|| {
        RvllmError::apple(
            AppleError::ShapeBucketMissing {
                seqs: req_ids.len() as u32,
                tokens: tokens_per_rollout,
            },
            apple_ctx("rollout_bucket"),
        )
    })
}

/// Form a static ANE rollout batch from a scheduler decode plan plus branch candidates.
///
/// Branches are emitted in scheduler decode order, with stable branch-plan order
/// inside each request. Each branch becomes one rollout row; duplicate request
/// ids are intentional and disambiguated by `AneRolloutSlot::branch_id`.
pub fn ane_rollout_batch_from_decode_plan(
    plan: &BatchPlan,
    kind: HandoffKind,
    branch_plan: &AneRolloutBranchPlan,
) -> Result<AneRolloutBatch> {
    let BatchPlan::Decode {
        req_ids,
        last_tokens,
        positions,
        context_lens,
        ..
    } = plan else {
        return Err(err("expected BatchPlan::Decode", "ane_rollout_batch"));
    };

    if req_ids.len() != last_tokens.len()
        || req_ids.len() != positions.len()
        || req_ids.len() != context_lens.len()
    {
        return Err(err("decode vector lengths differ", "ane_rollout_batch"));
    }
    if branch_plan.tokens_per_branch == 0 {
        return Err(err(
            "tokens_per_branch must be nonzero",
            "ane_rollout_batch",
        ));
    }
    if branch_plan.branches.is_empty() {
        return Err(err(
            "rollout branch plan must not be empty",
            "ane_rollout_batch",
        ));
    }
    for (idx, branch) in branch_plan.branches.iter().enumerate() {
        let token_len = u32::try_from(branch.tokens.len())
            .map_err(|_| err("too many rollout tokens", "ane_rollout_batch"))?;
        if token_len != branch_plan.tokens_per_branch {
            return Err(err(
                "branch token length must equal tokens_per_branch",
                "ane_rollout_batch",
            ));
        }
        if !req_ids.contains(&branch.req_id) {
            return Err(err(
                "branch req_id absent from decode plan",
                "ane_rollout_batch",
            ));
        }
        if branch_plan.branches[idx + 1..]
            .iter()
            .any(|other| other.req_id == branch.req_id && other.branch_id == branch.branch_id)
        {
            return Err(err("duplicate branch id for request", "ane_rollout_batch"));
        }
    }

    let seqs = u32::try_from(branch_plan.branches.len())
        .map_err(|_| err("too many rollout branches", "ane_rollout_batch"))?;
    let bucket = select_rollout_bucket(seqs, branch_plan.tokens_per_branch)
        .ok_or_else(|| shape_err(seqs, branch_plan.tokens_per_branch, "ane_rollout_batch"))?;

    let mut capsule_req_ids = Vec::with_capacity(branch_plan.branches.len());
    let token_capacity = branch_plan
        .branches
        .len()
        .checked_mul(branch_plan.tokens_per_branch as usize)
        .ok_or_else(|| err("too many rollout tokens", "ane_rollout_batch"))?;
    let mut tokens_flat = Vec::with_capacity(token_capacity);
    let mut cu_seqlens = Vec::with_capacity(branch_plan.branches.len() + 1);
    let mut rollout_positions = Vec::with_capacity(branch_plan.branches.len());
    let mut rollout_context_lens = Vec::with_capacity(branch_plan.branches.len());
    let mut slots = Vec::with_capacity(branch_plan.branches.len());
    cu_seqlens.push(0);

    for (decode_slot, req_id) in req_ids.iter().copied().enumerate() {
        let decode_slot = u32::try_from(decode_slot)
            .map_err(|_| err("too many decode slots", "ane_rollout_batch"))?;
        for branch in branch_plan
            .branches
            .iter()
            .filter(|branch| branch.req_id == req_id)
        {
            let token_offset = u32::try_from(tokens_flat.len())
                .map_err(|_| err("too many rollout tokens", "ane_rollout_batch"))?;
            tokens_flat.extend(branch.tokens.iter().copied());
            let token_len = branch_plan.tokens_per_branch;
            cu_seqlens.push(
                u32::try_from(tokens_flat.len())
                    .map_err(|_| err("too many rollout tokens", "ane_rollout_batch"))?,
            );
            capsule_req_ids.push(req_id);
            rollout_positions.push(positions[decode_slot as usize]);
            rollout_context_lens.push(context_lens[decode_slot as usize]);
            slots.push(AneRolloutSlot {
                req_id,
                branch_id: branch.branch_id,
                decode_slot,
                token_offset,
                token_len,
            });
        }
    }

    let capsule = HandoffCapsule::new(
        kind,
        capsule_req_ids,
        tokens_flat,
        cu_seqlens,
        rollout_positions,
        rollout_context_lens,
    );
    capsule.validate()?;

    Ok(AneRolloutBatch {
        bucket,
        capsule,
        slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::{AppleError, ReqId, RvllmError, TokenId};

    use crate::{Request, Scheduler};

    #[test]
    fn prefill_plan_maps_to_well_formed_capsule() {
        let plan = BatchPlan::Prefill {
            req_ids: vec![ReqId(1), ReqId(2)],
            prompt_tokens_flat: vec![TokenId(10), TokenId(11), TokenId(20)],
            cu_seqlens_q: vec![0, 2, 3],
        };
        let capsule = match handoff_from_prefill_plan(&plan, HandoffKind::MetalPrefillToAneFfnRollout) {
            Ok(v) => v,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert!(capsule.is_well_formed());
        assert_eq!(capsule.positions, vec![1, 0]);
        assert_eq!(capsule.context_lens, vec![2, 1]);
    }

    #[test]
    fn decode_plan_maps_to_unit_spans_and_bucket() {
        let plan = BatchPlan::Decode {
            req_ids: vec![ReqId(1), ReqId(2), ReqId(3)],
            bucket: 4,
            last_tokens: vec![TokenId(10), TokenId(20), TokenId(30)],
            positions: vec![7, 8, 9],
            context_lens: vec![8, 9, 10],
        };
        let capsule = match handoff_from_decode_plan(&plan, HandoffKind::MetalPrefillToAneFfnRollout) {
            Ok(v) => v,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert!(capsule.is_well_formed());
        assert_eq!(capsule.cu_seqlens, vec![0, 1, 2, 3]);
        let bucket = match rollout_bucket_for_decode(&plan, 4) {
            Ok(v) => v,
            Err(e) => panic!("unexpected bucket error: {e}"),
        };
        assert_eq!(bucket, RolloutBucket { seqs: 4, tokens: 4 });
    }

    #[test]
    fn ane_rollout_policy_batches_scheduler_decode_requests_and_spec_branches() {
        let mut scheduler = Scheduler::new();
        scheduler.enqueue(Request::new(ReqId(1), vec![TokenId(10), TokenId(11)], 8));
        scheduler.enqueue(Request::new(ReqId(2), vec![TokenId(20), TokenId(21)], 8));

        match scheduler.schedule() {
            BatchPlan::Prefill { req_ids, .. } => assert_eq!(req_ids, vec![ReqId(1), ReqId(2)]),
            other => panic!("expected Prefill, got {other:?}"),
        }

        let decode = scheduler.schedule();
        let branch_plan = AneRolloutBranchPlan {
            tokens_per_branch: 4,
            branches: vec![
                AneSpeculativeBranch {
                    req_id: ReqId(1),
                    branch_id: 0,
                    tokens: vec![TokenId(101), TokenId(102), TokenId(103), TokenId(104)],
                },
                AneSpeculativeBranch {
                    req_id: ReqId(1),
                    branch_id: 1,
                    tokens: vec![TokenId(111), TokenId(112), TokenId(113), TokenId(114)],
                },
                AneSpeculativeBranch {
                    req_id: ReqId(2),
                    branch_id: 0,
                    tokens: vec![TokenId(201), TokenId(202), TokenId(203), TokenId(204)],
                },
            ],
        };

        let rollout = match ane_rollout_batch_from_decode_plan(
            &decode,
            HandoffKind::MetalPrefillToAneRolloutExperimental,
            &branch_plan,
        ) {
            Ok(v) => v,
            Err(e) => panic!("unexpected rollout policy error: {e}"),
        };

        assert_eq!(rollout.bucket, RolloutBucket { seqs: 4, tokens: 4 });
        assert_eq!(rollout.capsule.req_ids, vec![ReqId(1), ReqId(1), ReqId(2)]);
        assert_eq!(
            rollout.capsule.tokens_flat,
            vec![
                TokenId(101),
                TokenId(102),
                TokenId(103),
                TokenId(104),
                TokenId(111),
                TokenId(112),
                TokenId(113),
                TokenId(114),
                TokenId(201),
                TokenId(202),
                TokenId(203),
                TokenId(204),
            ]
        );
        assert_eq!(rollout.capsule.cu_seqlens, vec![0, 4, 8, 12]);
        assert_eq!(rollout.capsule.positions, vec![1, 1, 1]);
        assert_eq!(rollout.capsule.context_lens, vec![2, 2, 2]);
        assert_eq!(
            rollout
                .slots
                .iter()
                .map(|slot| (
                    slot.req_id,
                    slot.branch_id,
                    slot.token_offset,
                    slot.token_len
                ))
                .collect::<Vec<_>>(),
            vec![
                (ReqId(1), 0, 0, 4),
                (ReqId(1), 1, 4, 4),
                (ReqId(2), 0, 8, 4)
            ]
        );
    }

    #[test]
    fn ane_rollout_policy_rejects_unsupported_branch_shape() {
        let decode = BatchPlan::Decode {
            req_ids: vec![ReqId(1)],
            bucket: 1,
            last_tokens: vec![TokenId(10)],
            positions: vec![4],
            context_lens: vec![5],
        };
        let branch_plan = AneRolloutBranchPlan {
            tokens_per_branch: 8,
            branches: (0..33)
                .map(|branch_id| AneSpeculativeBranch {
                    req_id: ReqId(1),
                    branch_id,
                    tokens: vec![TokenId(200); 8],
                })
                .collect(),
        };

        let err = match ane_rollout_batch_from_decode_plan(
            &decode,
            HandoffKind::MetalPrefillToAneRolloutExperimental,
            &branch_plan,
        ) {
            Ok(v) => panic!("expected unsupported shape error, got {v:?}"),
            Err(e) => e,
        };

        match err {
            RvllmError::Apple {
                err: AppleError::ShapeBucketMissing { seqs, tokens },
                ..
            } => {
                assert_eq!(seqs, 33);
                assert_eq!(tokens, 8);
            }
            other => panic!("expected ShapeBucketMissing, got {other:?}"),
        }
    }
}
