//! Feature-gated bridge from rvLLM scheduler output to Apple backend capsules.
//!
//! This module deliberately has no Metal/ANE FFI. It only translates the real
//! `BatchPlan` values into host-testable `rvllm-apple` contracts.

use rvllm_apple::{select_rollout_bucket, HandoffCapsule, HandoffKind, RolloutBucket};
use rvllm_core::{AppleCtx, AppleError, ReqId, Result, RvllmError, TokenId};
use rvllm_loader::gemma4_arch::{Gemma4Arch, Gemma4LayerType};
use rvllm_loader::ModelArch;

use crate::gemma4_layer_exec::{Gemma4LayerDims, Gemma4Phase};
use crate::layer_exec::{LayerDims, LayerPhase};
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

/// Host-side shape facts derived from the same packed QKV layout used by layer_exec.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AppleLayerDerivedShape {
    pub q_dim: u32,
    pub kv_dim: u32,
    pub qkv_rows: u32,
    pub qkv_out_bytes: usize,
    pub k_out_byte_offset: u64,
    pub v_out_byte_offset: u64,
    pub kv_cache_elems_per_layer: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct AppleLayerParity {
    pub dims: LayerDims,
    pub phase: LayerPhase,
    pub shape: AppleLayerDerivedShape,
}

#[derive(Copy, Clone, Debug)]
pub struct AppleGemma4LayerParity {
    pub dims: Gemma4LayerDims,
    pub phase: Gemma4Phase,
    pub shape: AppleLayerDerivedShape,
}

fn layer_shape_err(reason: &'static str, op: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::LayerShapeInvalid { reason },
        AppleCtx {
            backend: "apple-layer-parity",
            op,
            device: "apple-silicon",
        },
    )
}

fn require_nonzero(value: u32, reason: &'static str, op: &'static str) -> Result<u32> {
    if value == 0 {
        return Err(layer_shape_err(reason, op));
    }
    Ok(value)
}

fn usize_to_u32(value: usize, reason: &'static str, op: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| layer_shape_err(reason, op))
}

fn checked_add_u32(a: u32, b: u32, reason: &'static str, op: &'static str) -> Result<u32> {
    a.checked_add(b).ok_or_else(|| layer_shape_err(reason, op))
}

fn checked_mul_u32(a: u32, b: u32, reason: &'static str, op: &'static str) -> Result<u32> {
    a.checked_mul(b).ok_or_else(|| layer_shape_err(reason, op))
}

fn checked_mul_u64(a: u64, b: u64, reason: &'static str, op: &'static str) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| layer_shape_err(reason, op))
}

fn checked_usize(bytes: u64, reason: &'static str, op: &'static str) -> Result<usize> {
    usize::try_from(bytes).map_err(|_| layer_shape_err(reason, op))
}

fn div_ceil_u32(n: u32, d: u32, reason: &'static str, op: &'static str) -> Result<u32> {
    require_nonzero(d, "block_size must be nonzero", op)?;
    let numerator = checked_add_u32(n, d - 1, reason, op)?;
    Ok(numerator / d)
}

fn phase_num_tokens(num_seqs: u32, phase: LayerPhase, op: &'static str) -> Result<u32> {
    require_nonzero(num_seqs, "num_seqs must be nonzero", op)?;
    match phase {
        LayerPhase::Decode => Ok(num_seqs),
        LayerPhase::Prefill { max_seqlen_q, .. } => {
            require_nonzero(max_seqlen_q, "max_seqlen_q must be nonzero", op)?;
            checked_mul_u32(num_seqs, max_seqlen_q, "prefill token count overflow", op)
        }
    }
}

fn derive_shape(dims: LayerDims, op: &'static str) -> Result<AppleLayerDerivedShape> {
    require_nonzero(dims.num_tokens, "num_tokens must be nonzero", op)?;
    require_nonzero(dims.num_heads, "num_heads must be nonzero", op)?;
    require_nonzero(dims.num_kv_heads, "num_kv_heads must be nonzero", op)?;
    require_nonzero(dims.head_dim, "head_dim must be nonzero", op)?;
    require_nonzero(dims.block_size, "block_size must be nonzero", op)?;
    require_nonzero(
        dims.num_blocks_total,
        "num_blocks_total must be nonzero",
        op,
    )?;

    let q_dim = checked_mul_u32(dims.num_heads, dims.head_dim, "q_dim overflow", op)?;
    let kv_dim = checked_mul_u32(dims.num_kv_heads, dims.head_dim, "kv_dim overflow", op)?;
    let two_kv = checked_mul_u32(2, kv_dim, "qkv_rows overflow", op)?;
    let qkv_rows = checked_add_u32(q_dim, two_kv, "qkv_rows overflow", op)?;

    let tokens = dims.num_tokens as u64;
    let q_dim64 = q_dim as u64;
    let kv_dim64 = kv_dim as u64;
    let qkv_rows64 = qkv_rows as u64;
    let q_bytes = checked_mul_u64(tokens, q_dim64, "q offset overflow", op)?;
    let q_bytes = checked_mul_u64(q_bytes, 2, "q offset overflow", op)?;
    let kv_bytes = checked_mul_u64(tokens, kv_dim64, "kv offset overflow", op)?;
    let kv_bytes = checked_mul_u64(kv_bytes, 2, "kv offset overflow", op)?;
    let qkv_out_bytes = checked_mul_u64(tokens, qkv_rows64, "qkv output size overflow", op)?;
    let qkv_out_bytes = checked_mul_u64(qkv_out_bytes, 2, "qkv output size overflow", op)?;

    let kv_cache_elems = checked_mul_u64(
        2,
        dims.num_blocks_total as u64,
        "kv cache size overflow",
        op,
    )?;
    let kv_cache_elems = checked_mul_u64(
        kv_cache_elems,
        dims.block_size as u64,
        "kv cache size overflow",
        op,
    )?;
    let kv_cache_elems = checked_mul_u64(
        kv_cache_elems,
        dims.num_kv_heads as u64,
        "kv cache size overflow",
        op,
    )?;
    let kv_cache_elems = checked_mul_u64(
        kv_cache_elems,
        dims.head_dim as u64,
        "kv cache size overflow",
        op,
    )?;

    Ok(AppleLayerDerivedShape {
        q_dim,
        kv_dim,
        qkv_rows,
        qkv_out_bytes: checked_usize(qkv_out_bytes, "qkv output size does not fit usize", op)?,
        k_out_byte_offset: q_bytes,
        v_out_byte_offset: checked_add_u64(q_bytes, kv_bytes, "v offset overflow", op)?,
        kv_cache_elems_per_layer: kv_cache_elems,
    })
}

fn checked_add_u64(a: u64, b: u64, reason: &'static str, op: &'static str) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| layer_shape_err(reason, op))
}

pub fn qwen_layer_parity(
    arch: &ModelArch,
    num_seqs: u32,
    phase: LayerPhase,
    block_size: u32,
    num_blocks_total: u32,
) -> Result<AppleLayerParity> {
    let op = "qwen_layer_parity";
    let hidden = usize_to_u32(arch.hidden_size, "hidden_size does not fit u32", op)?;
    let num_heads = usize_to_u32(
        arch.num_attention_heads,
        "num_attention_heads does not fit u32",
        op,
    )?;
    let num_kv_heads = usize_to_u32(
        arch.num_key_value_heads,
        "num_key_value_heads does not fit u32",
        op,
    )?;
    let head_dim = usize_to_u32(arch.head_dim, "head_dim does not fit u32", op)?;
    let intermediate = usize_to_u32(
        arch.intermediate_size,
        "intermediate_size does not fit u32",
        op,
    )?;
    require_nonzero(hidden, "hidden_size must be nonzero", op)?;
    require_nonzero(head_dim, "head_dim must be nonzero", op)?;
    require_nonzero(block_size, "block_size must be nonzero", op)?;
    require_nonzero(num_blocks_total, "num_blocks_total must be nonzero", op)?;

    let num_tokens = phase_num_tokens(num_seqs, phase, op)?;
    let dims = LayerDims {
        num_tokens,
        hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate,
        block_size,
        max_blocks_per_seq: (num_blocks_total / num_seqs).max(1),
        num_blocks_total,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        rms_eps: arch.rms_norm_eps,
    };
    let shape = derive_shape(dims, op)?;
    Ok(AppleLayerParity { dims, phase, shape })
}

pub fn gemma4_layer_parity(
    arch: &Gemma4Arch,
    layer_idx: usize,
    num_seqs: u32,
    phase: LayerPhase,
    block_size: u32,
    num_blocks_total: u32,
    f16_kv_decode: bool,
) -> Result<AppleGemma4LayerParity> {
    let op = "gemma4_layer_parity";
    let Some(layer_type) = arch.layer_types.get(layer_idx).copied() else {
        return Err(layer_shape_err("layer_idx out of range", op));
    };
    require_nonzero(num_seqs, "num_seqs must be nonzero", op)?;
    require_nonzero(block_size, "block_size must be nonzero", op)?;
    require_nonzero(num_blocks_total, "num_blocks_total must be nonzero", op)?;

    let hidden = usize_to_u32(arch.hidden_size, "hidden_size does not fit u32", op)?;
    let num_heads = usize_to_u32(
        arch.num_attention_heads,
        "num_attention_heads does not fit u32",
        op,
    )?;
    let head_dim = usize_to_u32(
        arch.head_dim_for_layer(layer_idx),
        "head_dim does not fit u32",
        op,
    )?;
    let num_kv_heads = usize_to_u32(
        arch.num_kv_heads_for_layer(layer_idx),
        "num_kv_heads does not fit u32",
        op,
    )?;
    let rotary_dim = usize_to_u32(
        arch.rotary_dim_for_layer(layer_idx),
        "rotary_dim does not fit u32",
        op,
    )?;
    let intermediate = usize_to_u32(
        arch.intermediate_size,
        "intermediate_size does not fit u32",
        op,
    )?;
    let sliding_window = usize_to_u32(
        arch.sliding_window_size,
        "sliding_window_size does not fit u32",
        op,
    )?;
    require_nonzero(hidden, "hidden_size must be nonzero", op)?;
    require_nonzero(head_dim, "head_dim must be nonzero", op)?;
    require_nonzero(num_heads, "num_attention_heads must be nonzero", op)?;
    require_nonzero(num_kv_heads, "num_kv_heads must be nonzero", op)?;

    let sliding_blocks = div_ceil_u32(
        sliding_window,
        block_size,
        "sliding block count overflow",
        op,
    )?
    .min(num_blocks_total);
    let layer_blocks = match layer_type {
        Gemma4LayerType::SlidingAttention => sliding_blocks,
        Gemma4LayerType::GlobalAttention => num_blocks_total,
    };
    require_nonzero(layer_blocks, "layer block count must be nonzero", op)?;

    let num_tokens = phase_num_tokens(num_seqs, phase, op)?;
    let gemma_phase = match phase {
        LayerPhase::Decode => Gemma4Phase::Decode,
        LayerPhase::Prefill {
            cu_seqlens_q,
            max_seqlen_q,
        } => Gemma4Phase::Prefill {
            cu_seqlens_q,
            max_seqlen_q,
            num_seqs,
        },
    };

    let dims = Gemma4LayerDims {
        num_tokens,
        hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        intermediate,
        block_size,
        max_blocks_per_seq: layer_blocks,
        num_blocks_total: layer_blocks,
        attn_scale: 1.0,
        rms_eps: arch.rms_norm_eps,
        layer_type,
        sliding_window,
        // Runtime Gemma 4 prefill uses FP8 KV; F16 KV is decode-only.
        f16_kv: matches!(phase, LayerPhase::Decode) && f16_kv_decode,
    };
    let common_dims = LayerDims {
        num_tokens: dims.num_tokens,
        hidden: dims.hidden,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        intermediate: dims.intermediate,
        block_size: dims.block_size,
        max_blocks_per_seq: dims.max_blocks_per_seq,
        num_blocks_total: dims.num_blocks_total,
        attn_scale: dims.attn_scale,
        rms_eps: dims.rms_eps,
    };
    let shape = derive_shape(common_dims, op)?;
    Ok(AppleGemma4LayerParity {
        dims,
        phase: gemma_phase,
        shape,
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
    use crate::layer_exec::LayerPhase;
    use crate::{Request, Scheduler};
    use rvllm_core::{AppleError, ReqId, RvllmError, TokenId};
    use rvllm_loader::gemma4_arch::{Gemma4Arch, Gemma4LayerType};
    use rvllm_loader::{LayerAttnType, ModelArch};

    fn scheduled_prefill_plan() -> BatchPlan {
        let mut scheduler = Scheduler::new();
        scheduler.enqueue(Request::new(ReqId(1), vec![TokenId(10), TokenId(11)], 4));
        scheduler.enqueue(Request::new(ReqId(2), vec![TokenId(20)], 4));
        scheduler.schedule()
    }

    fn scheduled_decode_plan() -> BatchPlan {
        let mut scheduler = Scheduler::new();
        scheduler.enqueue(Request::new(ReqId(1), vec![TokenId(10), TokenId(11)], 4));
        scheduler.enqueue(Request::new(ReqId(2), vec![TokenId(20)], 4));
        let _prefill = scheduler.schedule();
        scheduler.commit_decode(&[(ReqId(1), TokenId(12))]);
        scheduler.schedule()
    }

    #[test]
    fn prefill_plan_maps_to_well_formed_capsule() {
        let plan = scheduled_prefill_plan();
        let capsule = match handoff_from_prefill_plan(&plan, HandoffKind::MetalPrefillToAneFfnRollout) {
            Ok(v) => v,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert!(capsule.is_well_formed());
        assert_eq!(capsule.req_ids, vec![ReqId(1), ReqId(2)]);
        assert_eq!(capsule.tokens_flat, vec![TokenId(10), TokenId(11), TokenId(20)]);
        assert_eq!(capsule.cu_seqlens, vec![0, 2, 3]);
        assert_eq!(capsule.positions, vec![1, 0]);
        assert_eq!(capsule.context_lens, vec![2, 1]);
    }

    #[test]
    fn decode_plan_maps_to_unit_spans_and_bucket() {
        let plan = scheduled_decode_plan();
        let capsule = match handoff_from_decode_plan(&plan, HandoffKind::MetalPrefillToAneFfnRollout) {
            Ok(v) => v,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert!(capsule.is_well_formed());
        assert_eq!(capsule.req_ids, vec![ReqId(1), ReqId(2)]);
        assert_eq!(capsule.tokens_flat, vec![TokenId(12), TokenId(20)]);
        assert_eq!(capsule.cu_seqlens, vec![0, 1, 2]);
        assert_eq!(capsule.positions, vec![2, 0]);
        assert_eq!(capsule.context_lens, vec![3, 1]);
        let bucket = match rollout_bucket_for_decode(&plan, 4) {
            Ok(v) => v,
            Err(e) => panic!("unexpected bucket error: {e}"),
        };
        assert_eq!(bucket, RolloutBucket { seqs: 4, tokens: 4 });
    }

    fn qwen_arch() -> ModelArch {
        ModelArch {
            num_hidden_layers: 28,
            hidden_size: 3584,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            head_dim: 128,
            intermediate_size: 18944,
            vocab_size: 151_936,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32_768,
            attention_bias: true,
            rms_norm_eps: 1e-6,
            layer_types: vec![LayerAttnType::Full; 28],
            global_head_dim: None,
            num_global_key_value_heads: None,
            global_rope_theta: None,
            partial_rotary_factor: None,
            sliding_window: None,
            final_logit_softcapping: None,
            hidden_activation: None,
            tie_word_embeddings: false,
            attention_k_eq_v: false,
        }
    }

    #[test]
    fn qwen_layer_parity_derives_layer_exec_shapes() {
        let arch = qwen_arch();
        let decode = match qwen_layer_parity(&arch, 8, LayerPhase::Decode, 32, 1024) {
            Ok(v) => v,
            Err(e) => panic!("unexpected qwen decode parity error: {e}"),
        };
        assert_eq!(decode.dims.num_tokens, 8);
        assert_eq!(decode.dims.hidden, 3584);
        assert_eq!(decode.dims.num_heads, 28);
        assert_eq!(decode.dims.num_kv_heads, 4);
        assert_eq!(decode.dims.head_dim, 128);
        assert_eq!(decode.dims.max_blocks_per_seq, 128);
        assert_eq!(decode.shape.q_dim, 3584);
        assert_eq!(decode.shape.kv_dim, 512);
        assert_eq!(decode.shape.qkv_rows, 4608);
        assert_eq!(decode.shape.k_out_byte_offset, 8 * 3584 * 2);
        assert_eq!(decode.shape.v_out_byte_offset, 8 * (3584 + 512) * 2);

        let prefill = match qwen_layer_parity(
            &arch,
            8,
            LayerPhase::Prefill {
                cu_seqlens_q: 0x1000,
                max_seqlen_q: 16,
            },
            32,
            1024,
        ) {
            Ok(v) => v,
            Err(e) => panic!("unexpected qwen prefill parity error: {e}"),
        };
        assert_eq!(prefill.dims.num_tokens, 128);
        assert_eq!(prefill.shape.qkv_out_bytes, 128 * 4608 * 2);
        assert_eq!(prefill.shape.k_out_byte_offset, 128 * 3584 * 2);
        assert_eq!(prefill.shape.v_out_byte_offset, 128 * (3584 + 512) * 2);
    }

    fn gemma4_arch() -> Gemma4Arch {
        let mut layer_types = vec![Gemma4LayerType::SlidingAttention; 6];
        layer_types[5] = Gemma4LayerType::GlobalAttention;
        Gemma4Arch {
            num_hidden_layers: 6,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim_sliding: 256,
            head_dim_global: 512,
            num_kv_heads_sliding: 16,
            num_kv_heads_global: 4,
            intermediate_size: 21_504,
            vocab_size: 262_144,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262_144,
            sliding_window_size: 1024,
            rope_theta_sliding: 10_000.0,
            rope_theta_global: 1_000_000.0,
            partial_rotary_factor_global: 0.25,
            logit_softcap: 30.0,
            layer_types,
            weight_prefix: "model".to_owned(),
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn gemma4_layer_parity_derives_sliding_and_global_shapes() {
        let arch = gemma4_arch();
        let sliding = match gemma4_layer_parity(
            &arch,
            0,
            2,
            LayerPhase::Prefill {
                cu_seqlens_q: 0x2000,
                max_seqlen_q: 4,
            },
            32,
            1024,
            false,
        ) {
            Ok(v) => v,
            Err(e) => panic!("unexpected gemma4 sliding parity error: {e}"),
        };
        assert_eq!(sliding.dims.num_tokens, 8);
        assert_eq!(sliding.dims.hidden, 5376);
        assert_eq!(sliding.dims.num_heads, 32);
        assert_eq!(sliding.dims.num_kv_heads, 16);
        assert_eq!(sliding.dims.head_dim, 256);
        assert_eq!(sliding.dims.rotary_dim, 256);
        assert_eq!(sliding.dims.max_blocks_per_seq, 32);
        assert_eq!(sliding.dims.num_blocks_total, 32);
        assert_eq!(sliding.shape.q_dim, 8192);
        assert_eq!(sliding.shape.kv_dim, 4096);
        assert_eq!(sliding.shape.qkv_rows, 16_384);
        assert_eq!(sliding.shape.k_out_byte_offset, 8 * 8192 * 2);
        assert_eq!(sliding.shape.v_out_byte_offset, 8 * (8192 + 4096) * 2);

        let global = match gemma4_layer_parity(&arch, 5, 2, LayerPhase::Decode, 32, 1024, true) {
            Ok(v) => v,
            Err(e) => panic!("unexpected gemma4 global parity error: {e}"),
        };
        assert_eq!(global.dims.num_tokens, 2);
        assert_eq!(global.dims.num_kv_heads, 4);
        assert_eq!(global.dims.head_dim, 512);
        assert_eq!(global.dims.rotary_dim, 128);
        assert_eq!(global.dims.max_blocks_per_seq, 1024);
        assert_eq!(global.dims.num_blocks_total, 1024);
        assert!(global.dims.f16_kv);
        assert_eq!(global.shape.q_dim, 16_384);
        assert_eq!(global.shape.kv_dim, 2048);
        assert_eq!(global.shape.qkv_rows, 20_480);
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
