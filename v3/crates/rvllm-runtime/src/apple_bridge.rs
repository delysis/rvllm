//! Feature-gated bridge from rvLLM scheduler output to Apple backend capsules.
//!
//! This module deliberately has no Metal/ANE FFI. It only translates the real
//! `BatchPlan` values into host-testable `rvllm-apple` contracts.

use rvllm_apple::{
    select_rollout_bucket, HandoffCapsule, HandoffKind, HandoffKvChain, RolloutBucket,
};
use rvllm_core::{
    AppleCtx, AppleError, AppleRolloutBucket, AppleRolloutBucketPolicy, Result, RvllmError,
};

use crate::paged_kv::{PagedKvPool, APPLE_KV_PAGE_SIZE};
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
    handoff_from_prefill_plan_impl(plan, kind, rollout_bucket, None)
}

/// Build a prefill handoff with complete request-owned page tables. The pool
/// is the source of truth: scheduler order only selects rows in the flattened
/// table and never determines physical ownership.
pub fn handoff_from_prefill_plan_with_paged_kv(
    plan: &BatchPlan,
    kind: HandoffKind,
    rollout_bucket: Option<RolloutBucket>,
    pool: &PagedKvPool,
    model_layout_fingerprint: [u8; 32],
) -> Result<HandoffCapsule> {
    handoff_from_prefill_plan_impl(
        plan,
        kind,
        rollout_bucket,
        Some((pool, model_layout_fingerprint)),
    )
}

fn handoff_from_prefill_plan_impl(
    plan: &BatchPlan,
    kind: HandoffKind,
    rollout_bucket: Option<RolloutBucket>,
    paged_kv: Option<(&PagedKvPool, [u8; 32])>,
) -> Result<HandoffCapsule> {
    let BatchPlan::Prefill {
        req_ids,
        prompt_tokens_flat,
        cu_seqlens_q,
        query_start_positions,
        context_lens,
        kv_chains,
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
    if query_start_positions.len() != req_ids.len() || context_lens.len() != req_ids.len() {
        return Err(err(
            "prefill query starts/context lengths must match req_ids",
            "prefill_handoff",
        ));
    }

    let mut positions = Vec::with_capacity(req_ids.len());
    for (index, span) in cu_seqlens_q.windows(2).enumerate() {
        let len = span[1].saturating_sub(span[0]);
        if len == 0 {
            return Err(err("empty prefill sequence", "prefill_handoff"));
        }
        let expected_context = query_start_positions[index]
            .checked_add(len)
            .ok_or_else(|| err("prefill context length overflow", "prefill_handoff"))?;
        if context_lens[index] != expected_context {
            return Err(err(
                "prefill context length must equal query start + query length",
                "prefill_handoff",
            ));
        }
        positions.push(expected_context - 1);
    }

    let mut capsule = HandoffCapsule::new(
        kind,
        req_ids.clone(),
        prompt_tokens_flat.clone(),
        cu_seqlens_q.clone(),
        positions,
        context_lens.clone(),
    );
    if let Some((pool, fingerprint)) = paged_kv {
        let token_positions = cu_seqlens_q
            .windows(2)
            .enumerate()
            .map(|(index, span)| {
                (0..span[1] - span[0])
                    .map(|offset| query_start_positions[index] + offset)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        capsule = attach_paged_kv(
            capsule,
            req_ids,
            kv_chains,
            context_lens,
            &token_positions,
            pool,
            fingerprint,
        )?;
    } else if kv_chains.iter().any(Option::is_some) {
        return Err(err(
            "paged KV chains require allocator page-table metadata",
            "prefill_handoff",
        ));
    }
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
    handoff_from_decode_plan_impl(plan, kind, rollout_bucket, None)
}

pub fn handoff_from_decode_plan_with_paged_kv(
    plan: &BatchPlan,
    kind: HandoffKind,
    rollout_bucket: Option<RolloutBucket>,
    pool: &PagedKvPool,
    model_layout_fingerprint: [u8; 32],
) -> Result<HandoffCapsule> {
    handoff_from_decode_plan_impl(
        plan,
        kind,
        rollout_bucket,
        Some((pool, model_layout_fingerprint)),
    )
}

fn handoff_from_decode_plan_impl(
    plan: &BatchPlan,
    kind: HandoffKind,
    rollout_bucket: Option<RolloutBucket>,
    paged_kv: Option<(&PagedKvPool, [u8; 32])>,
) -> Result<HandoffCapsule> {
    let BatchPlan::Decode {
        req_ids,
        last_tokens,
        positions,
        context_lens,
        kv_chains,
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
    if let Some((pool, fingerprint)) = paged_kv {
        let token_positions = positions
            .iter()
            .map(|&position| vec![position])
            .collect::<Vec<_>>();
        capsule = attach_paged_kv(
            capsule,
            req_ids,
            kv_chains,
            context_lens,
            &token_positions,
            pool,
            fingerprint,
        )?;
    } else if kv_chains.iter().any(Option::is_some) {
        return Err(err(
            "paged KV chains require allocator page-table metadata",
            "decode_handoff",
        ));
    }
    capsule = capsule.with_rollout_bucket(rollout_bucket);
    capsule.validate().map(|()| capsule)
}

fn attach_paged_kv(
    capsule: HandoffCapsule,
    req_ids: &[rvllm_core::ReqId],
    chains: &[Option<crate::paged_kv::KvChainHandle>],
    context_lens: &[u32],
    token_positions: &[Vec<u32>],
    pool: &PagedKvPool,
    model_layout_fingerprint: [u8; 32],
) -> Result<HandoffCapsule> {
    if chains.len() != req_ids.len()
        || context_lens.len() != req_ids.len()
        || token_positions.len() != req_ids.len()
    {
        return Err(err(
            "paged KV metadata vector lengths must match req_ids",
            "paged_kv_handoff",
        ));
    }

    let mut views = Vec::with_capacity(req_ids.len());
    let mut handoff_chains = Vec::with_capacity(req_ids.len());
    let mut max_blocks_per_seq = 0usize;
    for (index, &req_id) in req_ids.iter().enumerate() {
        let handle = chains[index].ok_or_else(|| {
            err(
                "every request in a paged KV batch must own a chain",
                "paged_kv_handoff",
            )
        })?;
        let view = pool
            .view(handle)
            .map_err(|_| err("paged KV chain is stale", "paged_kv_handoff"))?;
        if view.owner != req_id {
            return Err(err(
                "paged KV chain owner does not match request",
                "paged_kv_handoff",
            ));
        }
        if view.token_len < context_lens[index] {
            return Err(err(
                "paged KV chain does not cover request context",
                "paged_kv_handoff",
            ));
        }
        max_blocks_per_seq = max_blocks_per_seq.max(view.pages.len());
        handoff_chains.push(HandoffKvChain {
            id: handle.id.0,
            generation: handle.generation,
        });
        views.push(view);
    }
    if max_blocks_per_seq == 0 || max_blocks_per_seq > u32::MAX as usize {
        return Err(err(
            "paged KV block-table width is invalid",
            "paged_kv_handoff",
        ));
    }

    let mut block_tables = vec![u32::MAX; req_ids.len() * max_blocks_per_seq];
    let mut slot_mapping = Vec::with_capacity(capsule.tokens_flat.len());
    for (seq, view) in views.iter().enumerate() {
        for (block_index, page) in view.pages.iter().enumerate() {
            block_tables[seq * max_blocks_per_seq + block_index] = page.0;
        }
        for &position in &token_positions[seq] {
            let page_index = (position / APPLE_KV_PAGE_SIZE) as usize;
            let page = view.pages.get(page_index).ok_or_else(|| {
                err(
                    "slot position is outside the request page table",
                    "paged_kv_handoff",
                )
            })?;
            let slot = u64::from(page.0) * u64::from(APPLE_KV_PAGE_SIZE)
                + u64::from(position % APPLE_KV_PAGE_SIZE);
            if slot > i32::MAX as u64 {
                return Err(err(
                    "paged KV slot exceeds signed metadata range",
                    "paged_kv_handoff",
                ));
            }
            slot_mapping.push(slot as i32);
        }
    }
    if slot_mapping.len() != capsule.tokens_flat.len() {
        return Err(err(
            "paged KV slot count must equal input token count",
            "paged_kv_handoff",
        ));
    }

    Ok(capsule.with_paged_kv(
        handoff_chains,
        max_blocks_per_seq as u32,
        block_tables,
        slot_mapping,
        model_layout_fingerprint,
    ))
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
    use crate::paged_kv::{PagedKvConfig, PagedKvPool};
    use rvllm_core::{ReqId, TokenId};

    #[test]
    fn prefill_plan_maps_to_well_formed_capsule() {
        let plan = BatchPlan::Prefill {
            req_ids: vec![ReqId(1), ReqId(2)],
            prompt_tokens_flat: vec![TokenId(10), TokenId(11), TokenId(20)],
            cu_seqlens_q: vec![0, 2, 3],
            query_start_positions: vec![0, 0],
            context_lens: vec![2, 1],
            kv_chains: vec![None, None],
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
            kv_chains: vec![None, None, None],
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

    #[test]
    fn paged_handoff_uses_request_owned_pages_after_batch_reorder() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(8, 4)).unwrap();
        let chain_a = pool.allocate_chain(ReqId(10), 65).unwrap();
        let chain_b = pool.allocate_chain(ReqId(20), 33).unwrap();
        let pages_a = pool.view(chain_a).unwrap().pages.to_vec();
        let pages_b = pool.view(chain_b).unwrap().pages.to_vec();

        // B precedes A in this batch even though A was admitted first.
        let plan = BatchPlan::Prefill {
            req_ids: vec![ReqId(20), ReqId(10)],
            prompt_tokens_flat: vec![TokenId(2), TokenId(1)],
            cu_seqlens_q: vec![0, 1, 2],
            query_start_positions: vec![32, 64],
            context_lens: vec![33, 65],
            kv_chains: vec![Some(chain_b), Some(chain_a)],
        };
        let capsule = handoff_from_prefill_plan_with_paged_kv(
            &plan,
            HandoffKind::MetalPrefillToMetalDecode,
            None,
            &pool,
            [0x5A; 32],
        )
        .unwrap();

        assert!(capsule.is_well_formed());
        assert_eq!(capsule.max_blocks_per_seq, 3);
        assert_eq!(
            capsule.block_tables,
            vec![
                pages_b[0].0,
                pages_b[1].0,
                u32::MAX,
                pages_a[0].0,
                pages_a[1].0,
                pages_a[2].0,
            ]
        );
        assert_eq!(
            capsule.slot_mapping,
            vec![(pages_b[1].0 * 32) as i32, (pages_a[2].0 * 32) as i32]
        );
        assert_eq!(capsule.model_layout_fingerprint, [0x5A; 32]);
        assert_eq!(capsule.kv_chains[0].id, chain_b.id.0);
        assert_eq!(capsule.kv_chains[1].id, chain_a.id.0);
    }

    #[test]
    fn chain_only_handoff_fails_closed_instead_of_emitting_zero_width() {
        let plan = BatchPlan::Decode {
            req_ids: vec![ReqId(1)],
            bucket: 1,
            last_tokens: vec![TokenId(9)],
            positions: vec![0],
            context_lens: vec![1],
            kv_chains: vec![Some(crate::paged_kv::KvChainHandle {
                id: crate::paged_kv::KvChainId(0),
                generation: 1,
            })],
        };
        assert!(handoff_from_decode_plan(&plan, HandoffKind::MetalPrefillToMetalDecode).is_err());
    }
}
