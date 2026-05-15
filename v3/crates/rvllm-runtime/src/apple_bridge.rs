//! Feature-gated bridge from rvLLM scheduler output to Apple backend capsules.
//!
//! This module deliberately has no Metal/ANE FFI. It only translates the real
//! `BatchPlan` values into host-testable `rvllm-apple` contracts.

use rvllm_apple::{select_rollout_bucket, HandoffCapsule, HandoffKind, RolloutBucket};
use rvllm_core::{
    AppleCtx, AppleError, AppleRolloutBucket, AppleRolloutBucketPolicy, Result, RvllmError,
};

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

/// Convert a scheduler prefill plan into a Metal/ANE handoff capsule.
///
/// Positions and context lengths are derived from `cu_seqlens_q`: for each
/// sequence, position is `len - 1` and context length is `len`.
pub fn handoff_from_prefill_plan(plan: &BatchPlan, kind: HandoffKind) -> Result<HandoffCapsule> {
    handoff_from_prefill_plan_with_bucket(plan, kind, None)
}

pub fn handoff_from_prefill_plan_with_bucket(
    plan: &BatchPlan,
    kind: HandoffKind,
    rollout_bucket: Option<RolloutBucket>,
) -> Result<HandoffCapsule> {
    let BatchPlan::Prefill {
        req_ids,
        prompt_tokens_flat,
        cu_seqlens_q,
    } = plan
    else {
        return Err(err("expected BatchPlan::Prefill", "prefill_handoff"));
    };

    if cu_seqlens_q.len() != req_ids.len() + 1 {
        return Err(err(
            "cu_seqlens length must equal req_ids + 1",
            "prefill_handoff",
        ));
    }
    if cu_seqlens_q.first().copied() != Some(0) {
        return Err(err("cu_seqlens must start at 0", "prefill_handoff"));
    }
    if cu_seqlens_q.last().copied() != Some(prompt_tokens_flat.len() as u32) {
        return Err(err(
            "cu_seqlens must end at token length",
            "prefill_handoff",
        ));
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

    let mut capsule = HandoffCapsule::new(
        kind,
        req_ids.clone(),
        prompt_tokens_flat.clone(),
        cu_seqlens_q.clone(),
        positions,
        context_lens,
    );
    if matches!(
        kind,
        HandoffKind::MetalPrefillToAneFfnRollout
            | HandoffKind::MetalPrefillToAneRolloutExperimental
    ) {
        capsule = capsule.with_rollout_bucket(rollout_bucket);
    } else if rollout_bucket.is_some() {
        return Err(err(
            "non-ANE prefill handoff must not have a rollout bucket",
            "prefill_handoff",
        ));
    }
    capsule.validate().map(|()| capsule)
}

/// Convert a decode plan into a one-token-per-sequence Apple handoff capsule.
pub fn handoff_from_decode_plan(plan: &BatchPlan, kind: HandoffKind) -> Result<HandoffCapsule> {
    handoff_from_decode_plan_with_bucket(plan, kind, None)
}

pub fn handoff_from_decode_plan_with_bucket(
    plan: &BatchPlan,
    kind: HandoffKind,
    rollout_bucket: Option<RolloutBucket>,
) -> Result<HandoffCapsule> {
    let BatchPlan::Decode {
        req_ids,
        last_tokens,
        positions,
        context_lens,
        ..
    } = plan
    else {
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

    let mut capsule = HandoffCapsule::new(
        kind,
        req_ids.clone(),
        last_tokens.clone(),
        cu,
        positions.clone(),
        context_lens.clone(),
    );
    capsule = capsule.with_rollout_bucket(rollout_bucket);
    capsule.validate().map(|()| capsule)
}

pub fn rollout_bucket_for_decode(
    plan: &BatchPlan,
    tokens_per_rollout: u32,
) -> Result<RolloutBucket> {
    rollout_bucket_for_decode_with_config(plan, &None, tokens_per_rollout)
}
pub fn rollout_bucket_for_decode_with_runtime(
    plan: &BatchPlan,
    policy: &Option<AppleRolloutBucket>,
    tokens_per_rollout: u32,
) -> Result<RolloutBucket> {
    rollout_bucket_for_decode_with_config(plan, policy, tokens_per_rollout)
}

pub fn rollout_bucket_for_decode_with_config(
    plan: &BatchPlan,
    requested_bucket: &Option<AppleRolloutBucket>,
    tokens_per_rollout: u32,
) -> Result<RolloutBucket> {
    let BatchPlan::Decode { req_ids, .. } = plan else {
        return Err(err("expected BatchPlan::Decode", "rollout_bucket"));
    };

    if tokens_per_rollout == 0 {
        return Err(err("tokens_per_rollout must be > 0", "rollout_bucket"));
    }

    let seqs = req_ids.len() as u32;
    let bucket = match requested_bucket {
        Some(b) => {
            if b.tokens < tokens_per_rollout || b.seqs < seqs {
                return Err(RvllmError::apple(
                    AppleError::ShapeBucketMissing {
                        seqs,
                        tokens: tokens_per_rollout,
                    },
                    apple_ctx("rollout_bucket"),
                ));
            }
            RolloutBucket {
                seqs: b.seqs,
                tokens: b.tokens,
            }
        }
        None => select_rollout_bucket(seqs, tokens_per_rollout).ok_or_else(|| {
            RvllmError::apple(
                AppleError::ShapeBucketMissing {
                    seqs,
                    tokens: tokens_per_rollout,
                },
                apple_ctx("rollout_bucket"),
            )
        })?,
    };

    if !bucket.fits(seqs, tokens_per_rollout) {
        return Err(RvllmError::apple(
            AppleError::ShapeBucketMissing {
                seqs,
                tokens: tokens_per_rollout,
            },
            apple_ctx("rollout_bucket"),
        ));
    }

    Ok(bucket)
}

#[cfg(feature = "apple")]
#[allow(dead_code)]
pub fn rollout_bucket_for_decode_with_runtime_config(
    plan: &BatchPlan,
    rollout_tokens: u32,
    policy: AppleRolloutBucketPolicy,
    fixed_bucket: Option<AppleRolloutBucket>,
) -> Result<RolloutBucket> {
    let bucket_override = match policy {
        AppleRolloutBucketPolicy::Auto => None,
        AppleRolloutBucketPolicy::Fixed { seqs, tokens } => {
            Some(AppleRolloutBucket { seqs, tokens })
        }
    };
    let requested = fixed_bucket.or(bucket_override);
    rollout_bucket_for_decode_with_config(plan, &requested, rollout_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_core::{ReqId, TokenId};

    #[test]
    fn prefill_plan_maps_to_well_formed_capsule() {
        let plan = BatchPlan::Prefill {
            req_ids: vec![ReqId(1), ReqId(2)],
            prompt_tokens_flat: vec![TokenId(10), TokenId(11), TokenId(20)],
            cu_seqlens_q: vec![0, 2, 3],
        };
        let capsule = match handoff_from_prefill_plan_with_bucket(
            &plan,
            HandoffKind::MetalPrefillToAneFfnRollout,
            Some(RolloutBucket { seqs: 4, tokens: 1 }),
        ) {
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
        let bucket = match rollout_bucket_for_decode(&plan, 4) {
            Ok(v) => v,
            Err(e) => panic!("unexpected bucket error: {e}"),
        };
        let capsule = match handoff_from_decode_plan_with_bucket(
            &plan,
            HandoffKind::MetalPrefillToAneFfnRollout,
            Some(bucket),
        ) {
            Ok(v) => v,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert!(capsule.is_well_formed());
        assert_eq!(capsule.cu_seqlens, vec![0, 1, 2, 3]);
        assert_eq!(bucket, RolloutBucket { seqs: 4, tokens: 4 });
    }
}
