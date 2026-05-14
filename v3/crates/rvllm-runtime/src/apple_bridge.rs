//! Feature-gated bridge from rvLLM scheduler output to Apple backend capsules.
//!
//! This module deliberately has no Metal/ANE FFI. It only translates the real
//! `BatchPlan` values into host-testable `rvllm-apple` contracts.

use rvllm_apple::{
    select_rollout_bucket, AneProgramPlan, AneRolloutConfig, AppleAcceleratorTarget,
    AppleBackendMode, AppleMatmulConfig, AppleRuntimePlan, HandoffCapsule, HandoffKind,
    MetalBufferArenaPlan, MetalBufferRequest, MetalBufferRole, MetalPrefillCommandBufferRecipe,
    PrefillLayerGroup, RolloutBucket,
};
use rvllm_core::{AppleCtx, AppleError, ReqId, Result, RvllmError, TokenId};
use rvllm_loader::gemma4_arch::{Gemma4Arch, Gemma4LayerType};
use rvllm_loader::ModelArch;

use crate::bring_up::{BenchResult, PplResult};
use crate::gemma4_layer_exec::{Gemma4LayerDims, Gemma4Phase};
use crate::layer_exec::{LayerDims, LayerPhase};
use crate::sched_state::Request;
use crate::scheduler::{bucket_for, BatchPlan, Scheduler};

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

pub fn rollout_bucket_for_decode(
    plan: &BatchPlan,
    tokens_per_rollout: u32,
) -> Result<RolloutBucket> {
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
    } = plan
    else {
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

const GEMMA4_NATIVE_BLOCK_SIZE: u32 = 32;
const GEMMA4_NATIVE_ROLLOUT_TOKENS: u32 = 1;
const GEMMA4_NATIVE_KV_ELEM_BYTES: u64 = 2;

#[derive(Clone, Debug)]
pub struct AppleGemma4NativePlan {
    pub runtime_plan: AppleRuntimePlan,
    pub prefill_handoff: Option<HandoffCapsule>,
    pub decode_handoff: HandoffCapsule,
    pub prefill_layers: Vec<AppleGemma4LayerParity>,
    pub decode_layers: Vec<AppleGemma4LayerParity>,
    pub metal_arena: MetalBufferArenaPlan,
    pub prefill_recipe: MetalPrefillCommandBufferRecipe,
    pub ane_program: AneProgramPlan,
    pub block_size: u32,
    pub num_blocks_total: u32,
    pub max_context_tokens: u32,
}

pub fn run_gemma4_native_apple_bench(
    arch: &Gemma4Arch,
    num_seqs: u32,
    iters: u32,
    warmup: u32,
) -> Result<BenchResult> {
    let _plan = plan_gemma4_native_apple_bench(arch, num_seqs, iters, warmup)?;
    Err(native_execution_unavailable(
        "run_gemma4_native_apple_bench",
    ))
}

pub fn run_gemma4_native_apple_ppl(arch: &Gemma4Arch, token_ids: &[u32]) -> Result<PplResult> {
    let _plan = plan_gemma4_native_apple_ppl(arch, token_ids)?;
    Err(native_execution_unavailable("run_gemma4_native_apple_ppl"))
}

pub fn run_gemma4_native_apple_generate(
    arch: &Gemma4Arch,
    prompt_ids: &[u32],
    max_new: usize,
    eos_ids: &[u32],
) -> Result<Vec<u32>> {
    let _plan = plan_gemma4_native_apple_generate(arch, prompt_ids, max_new, eos_ids)?;
    Err(native_execution_unavailable(
        "run_gemma4_native_apple_generate",
    ))
}

pub fn plan_gemma4_native_apple_bench(
    arch: &Gemma4Arch,
    num_seqs: u32,
    iters: u32,
    _warmup: u32,
) -> Result<AppleGemma4NativePlan> {
    let op = "run_gemma4_native_apple_bench";
    validate_gemma4_native_arch(arch, op)?;
    native_require_nonzero_u32(num_seqs, "num_seqs must be nonzero", op)?;
    native_require_nonzero_u32(iters, "iters must be nonzero", op)?;

    let context_len = 1;
    let (decode_plan, rollout_bucket) = native_decode_batch_plan(num_seqs, context_len, op)?;
    let decode_handoff =
        handoff_from_decode_plan(&decode_plan, HandoffKind::MetalPrefillToMetalDecode)?;
    build_gemma4_native_plan(
        arch,
        None,
        decode_handoff,
        rollout_bucket,
        0,
        context_len,
        op,
    )
}

pub fn plan_gemma4_native_apple_ppl(
    arch: &Gemma4Arch,
    token_ids: &[u32],
) -> Result<AppleGemma4NativePlan> {
    let op = "run_gemma4_native_apple_ppl";
    validate_gemma4_native_arch(arch, op)?;
    if token_ids.len() < 2 {
        return Err(native_shape_err("ppl requires at least two token ids", op));
    }
    validate_gemma4_token_ids(arch, token_ids, op)?;

    let max_context_tokens =
        native_usize_to_u32(token_ids.len(), "token count does not fit u32", op)?;
    let max_output_tokens = max_context_tokens - 1;
    let (prefill_handoff, decode_handoff, rollout_bucket) =
        native_prefill_decode_handoffs(token_ids, max_output_tokens, op)?;
    build_gemma4_native_plan(
        arch,
        Some(prefill_handoff),
        decode_handoff,
        rollout_bucket,
        max_context_tokens,
        max_context_tokens,
        op,
    )
}

pub fn plan_gemma4_native_apple_generate(
    arch: &Gemma4Arch,
    prompt_ids: &[u32],
    max_new: usize,
    eos_ids: &[u32],
) -> Result<AppleGemma4NativePlan> {
    let op = "run_gemma4_native_apple_generate";
    validate_gemma4_native_arch(arch, op)?;
    if prompt_ids.is_empty() {
        return Err(native_shape_err("prompt must not be empty", op));
    }
    validate_gemma4_token_ids(arch, prompt_ids, op)?;
    validate_gemma4_token_ids(arch, eos_ids, op)?;

    let prompt_tokens =
        native_usize_to_u32(prompt_ids.len(), "prompt length does not fit u32", op)?;
    let max_new = native_usize_to_u32(max_new, "max_new does not fit u32", op)?;
    native_require_nonzero_u32(max_new, "max_new must be nonzero", op)?;
    let max_context_tokens = native_checked_add_u32(
        prompt_tokens,
        max_new,
        "generation context length overflow",
        op,
    )?;
    let (prefill_handoff, decode_handoff, rollout_bucket) =
        native_prefill_decode_handoffs(prompt_ids, max_new, op)?;
    build_gemma4_native_plan(
        arch,
        Some(prefill_handoff),
        decode_handoff,
        rollout_bucket,
        prompt_tokens,
        max_context_tokens,
        op,
    )
}

fn build_gemma4_native_plan(
    arch: &Gemma4Arch,
    prefill_handoff: Option<HandoffCapsule>,
    decode_handoff: HandoffCapsule,
    rollout_bucket: RolloutBucket,
    prefill_tokens: u32,
    max_context_tokens: u32,
    op: &'static str,
) -> Result<AppleGemma4NativePlan> {
    native_require_nonzero_u32(max_context_tokens, "max_context_tokens must be nonzero", op)?;
    let decode_seqs = native_usize_to_u32(
        decode_handoff.req_ids.len(),
        "decode sequence count does not fit u32",
        op,
    )?;
    native_require_nonzero_u32(decode_seqs, "decode sequence count must be nonzero", op)?;

    let runtime_plan = AppleRuntimePlan {
        target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
        mode: AppleBackendMode::MetalPrefillMetalDecode,
        matmul: AppleMatmulConfig::fp16(),
        rollout_bucket: Some(rollout_bucket),
        rollout_tokens: GEMMA4_NATIVE_ROLLOUT_TOKENS,
        private_ane_opt_in: false,
    };
    runtime_plan.validate()?;

    let num_blocks_total = native_num_blocks_total(decode_seqs, max_context_tokens, op)?;
    let decode_layers = native_gemma4_layer_plans(
        arch,
        decode_seqs,
        LayerPhase::Decode,
        num_blocks_total,
        true,
    )?;

    let prefill_layers = match &prefill_handoff {
        Some(handoff) => {
            native_require_nonzero_u32(prefill_tokens, "prefill token count must be nonzero", op)?;
            let prefill_seqs = native_usize_to_u32(
                handoff.req_ids.len(),
                "prefill sequence count does not fit u32",
                op,
            )?;
            native_gemma4_layer_plans(
                arch,
                prefill_seqs,
                LayerPhase::Prefill {
                    cu_seqlens_q: 0,
                    max_seqlen_q: prefill_tokens,
                },
                num_blocks_total,
                false,
            )?
        }
        None => Vec::new(),
    };

    let metal_arena =
        native_metal_arena_plan(arch, rollout_bucket, prefill_tokens, &decode_layers, op)?;
    let prefill_recipe = native_prefill_recipe(arch, op)?;
    let ane_program = native_ane_program(arch, rollout_bucket)?;

    Ok(AppleGemma4NativePlan {
        runtime_plan,
        prefill_handoff,
        decode_handoff,
        prefill_layers,
        decode_layers,
        metal_arena,
        prefill_recipe,
        ane_program,
        block_size: GEMMA4_NATIVE_BLOCK_SIZE,
        num_blocks_total,
        max_context_tokens,
    })
}

fn native_prefill_decode_handoffs(
    prompt_ids: &[u32],
    max_output_tokens: u32,
    op: &'static str,
) -> Result<(HandoffCapsule, HandoffCapsule, RolloutBucket)> {
    native_require_nonzero_u32(max_output_tokens, "max output tokens must be nonzero", op)?;
    let prompt_tokens: Vec<TokenId> = prompt_ids.iter().copied().map(TokenId).collect();
    let mut scheduler = Scheduler::new();
    scheduler.enqueue(Request::new(ReqId(1), prompt_tokens, max_output_tokens));

    let prefill_plan = scheduler.schedule();
    let prefill_handoff =
        handoff_from_prefill_plan(&prefill_plan, HandoffKind::MetalPrefillToMetalDecode)?;
    let decode_plan = scheduler.schedule();
    let rollout_bucket = rollout_bucket_for_decode(&decode_plan, GEMMA4_NATIVE_ROLLOUT_TOKENS)?;
    let decode_handoff =
        handoff_from_decode_plan(&decode_plan, HandoffKind::MetalPrefillToMetalDecode)?;

    Ok((prefill_handoff, decode_handoff, rollout_bucket))
}

fn native_decode_batch_plan(
    num_seqs: u32,
    context_len: u32,
    op: &'static str,
) -> Result<(BatchPlan, RolloutBucket)> {
    native_require_nonzero_u32(num_seqs, "num_seqs must be nonzero", op)?;
    native_require_nonzero_u32(context_len, "context_len must be nonzero", op)?;
    let rollout_bucket = select_rollout_bucket(num_seqs, GEMMA4_NATIVE_ROLLOUT_TOKENS)
        .ok_or_else(|| shape_err(num_seqs, GEMMA4_NATIVE_ROLLOUT_TOKENS, op))?;
    let scheduler_bucket = bucket_for(num_seqs)
        .ok_or_else(|| shape_err(num_seqs, GEMMA4_NATIVE_ROLLOUT_TOKENS, op))?;
    let n = num_seqs as usize;
    let req_ids = (0..n).map(|i| ReqId((i + 1) as u64)).collect();
    let plan = BatchPlan::Decode {
        req_ids,
        bucket: scheduler_bucket,
        last_tokens: vec![TokenId(0); n],
        positions: vec![context_len - 1; n],
        context_lens: vec![context_len; n],
    };
    let checked_bucket = rollout_bucket_for_decode(&plan, GEMMA4_NATIVE_ROLLOUT_TOKENS)?;
    debug_assert_eq!(checked_bucket, rollout_bucket);
    Ok((plan, rollout_bucket))
}

fn native_gemma4_layer_plans(
    arch: &Gemma4Arch,
    num_seqs: u32,
    phase: LayerPhase,
    num_blocks_total: u32,
    f16_kv_decode: bool,
) -> Result<Vec<AppleGemma4LayerParity>> {
    let mut layers = Vec::with_capacity(arch.num_hidden_layers);
    for layer_idx in 0..arch.num_hidden_layers {
        layers.push(gemma4_layer_parity(
            arch,
            layer_idx,
            num_seqs,
            phase,
            GEMMA4_NATIVE_BLOCK_SIZE,
            num_blocks_total,
            f16_kv_decode,
        )?);
    }
    Ok(layers)
}

fn native_metal_arena_plan(
    arch: &Gemma4Arch,
    rollout_bucket: RolloutBucket,
    prefill_tokens: u32,
    decode_layers: &[AppleGemma4LayerParity],
    op: &'static str,
) -> Result<MetalBufferArenaPlan> {
    let hidden = native_usize_to_u32(arch.hidden_size, "hidden_size does not fit u32", op)?;
    let intermediate = native_usize_to_u32(
        arch.intermediate_size,
        "intermediate_size does not fit u32",
        op,
    )?;
    let vocab = native_usize_to_u32(arch.vocab_size, "vocab_size does not fit u32", op)?;
    let max_head_dim =
        native_usize_to_u32(arch.max_head_dim(), "max head_dim does not fit u32", op)?;
    let max_kv_heads =
        native_usize_to_u32(arch.max_kv_heads(), "max num_kv_heads does not fit u32", op)?;
    let num_heads = native_usize_to_u32(
        arch.num_attention_heads,
        "num_attention_heads does not fit u32",
        op,
    )?;
    let max_q_dim = native_checked_mul_u32(num_heads, max_head_dim, "q_dim overflow", op)?;
    let max_kv_dim = native_checked_mul_u32(max_kv_heads, max_head_dim, "kv_dim overflow", op)?;
    let max_qkv_rows = native_checked_add_u32(
        max_q_dim,
        native_checked_mul_u32(2, max_kv_dim, "qkv rows overflow", op)?,
        "qkv rows overflow",
        op,
    )?;

    let max_tokens = prefill_tokens
        .max(rollout_bucket.capacity())
        .max(GEMMA4_NATIVE_ROLLOUT_TOKENS);
    let token_bytes = native_bytes(max_tokens as u64, 4, "token buffer overflow", op)?;
    let metadata_bytes = native_bytes(
        native_checked_add_u64(max_tokens as u64, 1, "metadata buffer overflow", op)?,
        4,
        "metadata buffer overflow",
        op,
    )?;
    let hidden_bytes = native_bytes(
        native_checked_mul_u64(
            max_tokens as u64,
            hidden as u64,
            "hidden buffer overflow",
            op,
        )?,
        2,
        "hidden buffer overflow",
        op,
    )?;
    let qkv_bytes = native_bytes(
        native_checked_mul_u64(
            max_tokens as u64,
            max_qkv_rows as u64,
            "qkv buffer overflow",
            op,
        )?,
        2,
        "qkv buffer overflow",
        op,
    )?;
    let attn_bytes = native_bytes(
        native_checked_mul_u64(
            max_tokens as u64,
            max_q_dim as u64,
            "attention buffer overflow",
            op,
        )?,
        2,
        "attention buffer overflow",
        op,
    )?;
    let ffn_bytes = native_bytes(
        native_checked_mul_u64(
            native_checked_mul_u64(max_tokens as u64, 2, "ffn buffer overflow", op)?,
            intermediate as u64,
            "ffn buffer overflow",
            op,
        )?,
        2,
        "ffn buffer overflow",
        op,
    )?;
    let logits_bytes = native_bytes(vocab as u64, 4, "logits buffer overflow", op)?;
    let parameter_bytes = native_bytes(hidden as u64, 2, "parameter buffer overflow", op)?;

    let mut kv_cache_elems = 0u64;
    for layer in decode_layers {
        kv_cache_elems = native_checked_add_u64(
            kv_cache_elems,
            layer.shape.kv_cache_elems_per_layer,
            "kv cache size overflow",
            op,
        )?;
    }
    let kv_cache_bytes = native_bytes(
        kv_cache_elems,
        GEMMA4_NATIVE_KV_ELEM_BYTES,
        "kv cache size overflow",
        op,
    )?;

    let requests = vec![
        MetalBufferRequest::new("tokens", token_bytes, 16, MetalBufferRole::Activation),
        MetalBufferRequest::new("positions", metadata_bytes, 16, MetalBufferRole::Activation),
        MetalBufferRequest::new("hidden", hidden_bytes, 16, MetalBufferRole::Activation),
        MetalBufferRequest::new("qkv", qkv_bytes, 64, MetalBufferRole::Scratch),
        MetalBufferRequest::new("attention", attn_bytes, 64, MetalBufferRole::Scratch),
        MetalBufferRequest::new("ffn", ffn_bytes, 64, MetalBufferRole::Scratch),
        MetalBufferRequest::new("kv-cache", kv_cache_bytes, 256, MetalBufferRole::KvCache),
        MetalBufferRequest::new("logits", logits_bytes, 64, MetalBufferRole::Activation),
        MetalBufferRequest::new(
            "parameters",
            parameter_bytes,
            256,
            MetalBufferRole::Parameters,
        ),
    ];

    let mut capacity = 0usize;
    for request in &requests {
        capacity = native_checked_add_usize(capacity, request.bytes, "arena size overflow", op)?;
        capacity = native_checked_add_usize(capacity, request.align, "arena size overflow", op)?;
    }
    MetalBufferArenaPlan::new(capacity, &requests)
}

fn native_prefill_recipe(
    arch: &Gemma4Arch,
    op: &'static str,
) -> Result<MetalPrefillCommandBufferRecipe> {
    let layer_count = native_usize_to_u32(
        arch.num_hidden_layers,
        "num_hidden_layers does not fit u32",
        op,
    )?;
    let group = PrefillLayerGroup::new(0, 0, layer_count)?;
    MetalPrefillCommandBufferRecipe::for_layer_group(group)
}

fn native_ane_program(arch: &Gemma4Arch, bucket: RolloutBucket) -> Result<AneProgramPlan> {
    let config = AneRolloutConfig {
        bucket,
        hidden_size: arch.hidden_size,
        intermediate_size: arch.intermediate_size,
        num_layers: arch.num_hidden_layers,
    };
    config.activation_desc().validate()?;
    Ok(AneProgramPlan::qkv_ffn_lm_head(config))
}

fn native_num_blocks_total(
    num_seqs: u32,
    max_context_tokens: u32,
    op: &'static str,
) -> Result<u32> {
    let blocks_per_seq = div_ceil_u32(
        max_context_tokens,
        GEMMA4_NATIVE_BLOCK_SIZE,
        "context block count overflow",
        op,
    )?;
    native_checked_mul_u32(num_seqs, blocks_per_seq, "total block count overflow", op)
}

fn validate_gemma4_native_arch(arch: &Gemma4Arch, op: &'static str) -> Result<()> {
    native_require_nonzero_usize(
        arch.num_hidden_layers,
        "num_hidden_layers must be nonzero",
        op,
    )?;
    native_require_nonzero_usize(arch.hidden_size, "hidden_size must be nonzero", op)?;
    native_require_nonzero_usize(
        arch.num_attention_heads,
        "num_attention_heads must be nonzero",
        op,
    )?;
    native_require_nonzero_usize(
        arch.head_dim_sliding,
        "head_dim_sliding must be nonzero",
        op,
    )?;
    native_require_nonzero_usize(arch.head_dim_global, "head_dim_global must be nonzero", op)?;
    native_require_nonzero_usize(
        arch.num_kv_heads_sliding,
        "num_kv_heads_sliding must be nonzero",
        op,
    )?;
    native_require_nonzero_usize(
        arch.num_kv_heads_global,
        "num_kv_heads_global must be nonzero",
        op,
    )?;
    native_require_nonzero_usize(
        arch.intermediate_size,
        "intermediate_size must be nonzero",
        op,
    )?;
    native_require_nonzero_usize(arch.vocab_size, "vocab_size must be nonzero", op)?;
    native_require_nonzero_usize(
        arch.sliding_window_size,
        "sliding_window_size must be nonzero",
        op,
    )?;
    if arch.layer_types.len() != arch.num_hidden_layers {
        return Err(native_shape_err(
            "layer_types length must equal num_hidden_layers",
            op,
        ));
    }
    if !arch.rms_norm_eps.is_finite() || arch.rms_norm_eps <= 0.0 {
        return Err(native_shape_err("rms_norm_eps must be positive", op));
    }
    if !arch.logit_softcap.is_finite() || arch.logit_softcap <= 0.0 {
        return Err(native_shape_err("logit_softcap must be positive", op));
    }
    for layer_idx in 0..arch.num_hidden_layers {
        native_require_nonzero_usize(
            arch.head_dim_for_layer(layer_idx),
            "layer head_dim must be nonzero",
            op,
        )?;
        native_require_nonzero_usize(
            arch.num_kv_heads_for_layer(layer_idx),
            "layer num_kv_heads must be nonzero",
            op,
        )?;
        native_require_nonzero_usize(
            arch.rotary_dim_for_layer(layer_idx),
            "layer rotary_dim must be nonzero",
            op,
        )?;
    }
    Ok(())
}

fn validate_gemma4_token_ids(arch: &Gemma4Arch, token_ids: &[u32], op: &'static str) -> Result<()> {
    let vocab = native_usize_to_u32(arch.vocab_size, "vocab_size does not fit u32", op)?;
    native_require_nonzero_u32(vocab, "vocab_size must be nonzero", op)?;
    for &token_id in token_ids {
        if token_id >= vocab {
            return Err(native_shape_err("token id out of vocabulary", op));
        }
    }
    Ok(())
}

fn native_execution_unavailable(op: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "native-apple-gemma4",
            op,
        },
        native_apple_ctx(op),
    )
}

fn native_shape_err(reason: &'static str, op: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::LayerShapeInvalid { reason },
        native_apple_ctx(op),
    )
}

fn native_apple_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "native-apple-gemma4",
        op,
        device: "apple-silicon",
    }
}

fn native_require_nonzero_u32(value: u32, reason: &'static str, op: &'static str) -> Result<u32> {
    if value == 0 {
        return Err(native_shape_err(reason, op));
    }
    Ok(value)
}

fn native_require_nonzero_usize(
    value: usize,
    reason: &'static str,
    op: &'static str,
) -> Result<usize> {
    if value == 0 {
        return Err(native_shape_err(reason, op));
    }
    Ok(value)
}

fn native_usize_to_u32(value: usize, reason: &'static str, op: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| native_shape_err(reason, op))
}

fn native_usize_from_u64(value: u64, reason: &'static str, op: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| native_shape_err(reason, op))
}

fn native_checked_add_u32(a: u32, b: u32, reason: &'static str, op: &'static str) -> Result<u32> {
    a.checked_add(b).ok_or_else(|| native_shape_err(reason, op))
}

fn native_checked_mul_u32(a: u32, b: u32, reason: &'static str, op: &'static str) -> Result<u32> {
    a.checked_mul(b).ok_or_else(|| native_shape_err(reason, op))
}

fn native_checked_add_u64(a: u64, b: u64, reason: &'static str, op: &'static str) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| native_shape_err(reason, op))
}

fn native_checked_mul_u64(a: u64, b: u64, reason: &'static str, op: &'static str) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| native_shape_err(reason, op))
}

fn native_checked_add_usize(
    a: usize,
    b: usize,
    reason: &'static str,
    op: &'static str,
) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| native_shape_err(reason, op))
}

fn native_bytes(
    elements: u64,
    elem_bytes: u64,
    reason: &'static str,
    op: &'static str,
) -> Result<usize> {
    native_usize_from_u64(
        native_checked_mul_u64(elements, elem_bytes, reason, op)?,
        reason,
        op,
    )
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
        let capsule =
            match handoff_from_prefill_plan(&plan, HandoffKind::MetalPrefillToAneFfnRollout) {
                Ok(v) => v,
                Err(e) => panic!("unexpected error: {e}"),
            };
        assert!(capsule.is_well_formed());
        assert_eq!(capsule.req_ids, vec![ReqId(1), ReqId(2)]);
        assert_eq!(
            capsule.tokens_flat,
            vec![TokenId(10), TokenId(11), TokenId(20)]
        );
        assert_eq!(capsule.cu_seqlens, vec![0, 2, 3]);
        assert_eq!(capsule.positions, vec![1, 0]);
        assert_eq!(capsule.context_lens, vec![2, 1]);
    }

    #[test]
    fn decode_plan_maps_to_unit_spans_and_bucket() {
        let plan = scheduled_decode_plan();
        let capsule =
            match handoff_from_decode_plan(&plan, HandoffKind::MetalPrefillToAneFfnRollout) {
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
    fn gemma4_native_generate_plans_scheduler_and_apple_contracts() {
        let arch = gemma4_arch();
        let plan = match plan_gemma4_native_apple_generate(&arch, &[1, 2, 3], 4, &[9]) {
            Ok(v) => v,
            Err(e) => panic!("unexpected native generate plan error: {e}"),
        };

        assert_eq!(
            plan.runtime_plan.mode,
            rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode
        );
        let prefill = match &plan.prefill_handoff {
            Some(v) => v,
            None => panic!("expected prefill handoff"),
        };
        assert_eq!(
            prefill.tokens_flat,
            vec![TokenId(1), TokenId(2), TokenId(3)]
        );
        assert_eq!(prefill.cu_seqlens, vec![0, 3]);
        assert_eq!(plan.decode_handoff.tokens_flat, vec![TokenId(3)]);
        assert_eq!(plan.decode_handoff.positions, vec![2]);
        assert_eq!(plan.decode_handoff.context_lens, vec![3]);
        assert_eq!(plan.prefill_layers.len(), arch.num_hidden_layers);
        assert_eq!(plan.decode_layers.len(), arch.num_hidden_layers);
        assert_eq!(plan.prefill_layers[0].dims.num_tokens, 3);
        assert_eq!(plan.decode_layers[0].dims.num_tokens, 1);
        assert_eq!(
            plan.ane_program.num_procedures(),
            arch.num_hidden_layers * 2 + 1
        );
        assert_eq!(
            plan.prefill_recipe.encoded_ops().count(),
            arch.num_hidden_layers * rvllm_apple::PREFILL_LAYER_OPS.len()
        );
        assert!(!plan.metal_arena.has_overlaps());
    }

    #[test]
    fn gemma4_native_generate_reports_first_party_executor_gap_after_planning() {
        let arch = gemma4_arch();
        let err = match run_gemma4_native_apple_generate(&arch, &[1, 2, 3], 4, &[9]) {
            Ok(v) => panic!("expected native Apple execution gap, got {v:?}"),
            Err(e) => e,
        };

        match err {
            RvllmError::Apple {
                err:
                    AppleError::FeatureNotAvailable {
                        backend: "native-apple-gemma4",
                        op: "run_gemma4_native_apple_generate",
                    },
                ctx,
                ..
            } => {
                assert_eq!(ctx.backend, "native-apple-gemma4");
                assert_eq!(ctx.op, "run_gemma4_native_apple_generate");
            }
            other => panic!("expected native FeatureNotAvailable, got {other:?}"),
        }
    }

    #[test]
    fn gemma4_native_ppl_rejects_oov_token_before_executor_gap() {
        let arch = gemma4_arch();
        let err = match run_gemma4_native_apple_ppl(&arch, &[1, arch.vocab_size as u32]) {
            Ok(v) => panic!("expected oov token error, got {v:?}"),
            Err(e) => e,
        };

        match err {
            RvllmError::Apple {
                err:
                    AppleError::LayerShapeInvalid {
                        reason: "token id out of vocabulary",
                    },
                ctx,
                ..
            } => {
                assert_eq!(ctx.backend, "native-apple-gemma4");
                assert_eq!(ctx.op, "run_gemma4_native_apple_ppl");
            }
            other => panic!("expected native token shape error, got {other:?}"),
        }
    }

    #[test]
    fn gemma4_native_bench_rejects_unsupported_rollout_shape() {
        let arch = gemma4_arch();
        let err = match run_gemma4_native_apple_bench(&arch, 129, 1, 0) {
            Ok(v) => panic!("expected unsupported Apple bucket, got {v:?}"),
            Err(e) => e,
        };

        match err {
            RvllmError::Apple {
                err: AppleError::ShapeBucketMissing { seqs, tokens },
                ctx,
                ..
            } => {
                assert_eq!(seqs, 129);
                assert_eq!(tokens, 1);
                assert_eq!(ctx.op, "run_gemma4_native_apple_bench");
            }
            other => panic!("expected ShapeBucketMissing, got {other:?}"),
        }
    }

    #[test]
    fn gemma4_native_bench_rejects_zero_iters() {
        let arch = gemma4_arch();
        let err = match run_gemma4_native_apple_bench(&arch, 1, 0, 0) {
            Ok(v) => panic!("expected zero iters error, got {v:?}"),
            Err(e) => e,
        };

        match err {
            RvllmError::Apple {
                err:
                    AppleError::LayerShapeInvalid {
                        reason: "iters must be nonzero",
                    },
                ctx,
                ..
            } => {
                assert_eq!(ctx.backend, "native-apple-gemma4");
                assert_eq!(ctx.op, "run_gemma4_native_apple_bench");
            }
            other => panic!("expected native iters shape error, got {other:?}"),
        }
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
