#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::{cmp::max, collections::BTreeMap, path::Path, ptr};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use half::f16;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use rvllm_apple::AppleLowBitTensorRole;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use rvllm_loader::{
    gemma4_validate::{
        Gemma4DryRunAttentionKind as HostGemma4DryRunAttentionKind,
        Gemma4DryRunFp8ScaleSummary as HostGemma4DryRunFp8ScaleSummary,
        Gemma4DryRunLayerValidation as HostGemma4DryRunLayerValidation,
        Gemma4DryRunLmHeadStatus as HostGemma4DryRunLmHeadStatus,
        Gemma4DryRunValidation as HostGemma4DryRunValidation,
    },
    load::{LayerAttnType, ModelArch},
};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::{
    arena::{MetalBufferArena, MetalRegion},
    context::MetalContext,
    memory_budget::AppleMemoryBudgetError,
    weight_loader::{
        load_safetensor_entry_f32, load_safetensor_entry_for_float_type,
        map_safetensor_to_arena_with_float_type, scan_safetensor_tensors, SafetensorTensorInfo,
    },
    MetalFloatType,
};

#[cfg(any(target_os = "macos", target_os = "ios"))]
const PROBE_METAL_ARENA_BYTES: usize = 1024 * 1024;
#[cfg(any(target_os = "macos", target_os = "ios"))]
const PROBE_METAL_SOFTCAP: f32 = 0.0;
#[cfg(any(target_os = "macos", target_os = "ios"))]
const METAL_DEFAULT_MAX_TOTAL_TOKENS: usize = 2048;
#[cfg(any(target_os = "macos", target_os = "ios"))]
const METAL_DEFAULT_MAX_BATCH_SEQUENCES: usize = 1;
/// Version-1 Apple KV ABI page size. Kept local to the Metal crate so the
/// embedded backend does not depend on the host scheduler crate.
pub const APPLE_KV_PAGE_TOKENS: usize = 32;
#[cfg(target_os = "macos")]
const RVLLM_METAL_MAX_TOTAL_TOKENS_ENV: &str = "RVLLM_METAL_MAX_TOTAL_TOKENS";
#[cfg(target_os = "macos")]
const RVLLM_METAL_MAX_PROBE_TOKENS_ENV: &str = "RVLLM_METAL_MAX_PROBE_TOKENS";
#[cfg(target_os = "macos")]
const RVLLM_METAL_MAX_BATCH_TOKENS_ENV: &str = "RVLLM_METAL_MAX_BATCH_TOKENS";
#[cfg(target_os = "macos")]
const RVLLM_METAL_MAX_BATCH_SEQUENCES_ENV: &str = "RVLLM_METAL_MAX_BATCH_SEQUENCES";
#[cfg(target_os = "macos")]
const RVLLM_METAL_DEBUG_TRACE_LAYER_ENV: &str = "RVLLM_METAL_DEBUG_TRACE_LAYER";

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn probe_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "model-metal-backend",
        op,
        device: "apple-silicon",
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn model_arena_overflow() -> RvllmError {
    RvllmError::apple(
        AppleError::InvalidWeightBlob {
            reason: "model arena byte overflow",
        },
        probe_ctx("prepare"),
    )
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn invalid_low_bit_replacement(reason: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::InvalidWeightBlob { reason },
        probe_ctx("low_bit_replacement"),
    )
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn align_up_checked(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(model_arena_overflow)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn low_bit_packed_values_bytes(
    format: rvllm_apple::AppleLowBitWeightFormat,
    n: usize,
    k: usize,
) -> Result<usize> {
    if n == 0 || k == 0 {
        return Err(invalid_low_bit_replacement(
            "low-bit replacement shape has a zero dimension",
        ));
    }
    let row_bytes = match format {
        rvllm_apple::AppleLowBitWeightFormat::W4A16 => k
            .checked_add(1)
            .map(|value| value / 2)
            .ok_or_else(model_arena_overflow)?,
        rvllm_apple::AppleLowBitWeightFormat::W8A16 => k,
    };
    n.checked_mul(row_bytes).ok_or_else(model_arena_overflow)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn low_bit_scales_bytes(n: usize, k: usize) -> Result<usize> {
    let groups = k
        .checked_add(rvllm_apple::APPLE_LOW_BIT_GROUP_SIZE - 1)
        .map(|value| value / rvllm_apple::APPLE_LOW_BIT_GROUP_SIZE)
        .ok_or_else(model_arena_overflow)?;
    n.checked_mul(groups)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f16>()))
        .ok_or_else(model_arena_overflow)
}

#[cfg(target_os = "macos")]
fn debug_trace_layers_from_env() -> Vec<usize> {
    std::env::var(RVLLM_METAL_DEBUG_TRACE_LAYER_ENV)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(target_os = "ios")]
fn debug_trace_layers_from_env() -> Vec<usize> {
    Vec::new()
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn debug_trace_layer_enabled(layers: &[usize], layer_idx: usize) -> bool {
    layers.contains(&layer_idx)
}

#[derive(Debug, Clone)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct Gemma4MetalState {
    pub float_type: MetalFloatType,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcap: f32,
    pub embedding_scale: f32,
    pub max_probe_tokens: usize,
    pub max_batch_tokens: usize,
    pub max_batch_sequences: usize,
    pub memory_budget: Gemma4MetalMemoryReport,
    pub embedding: MetalRegion,
    pub final_norm: MetalRegion,
    pub lm_head: MetalRegion,
    pub residual: MetalRegion,
    pub logits: MetalRegion,
    pub normed_hidden: MetalRegion,
    pub sampled: MetalRegion,
    pub final_argmax_partial_max: MetalRegion,
    pub final_argmax_partial_idx: MetalRegion,
    pub token_ids: MetalRegion,
    pub ple: Option<MetalPleState>,

    pub layers: Vec<MetalOneLayerState>,
    /// Three independently allocated execution slots. Weights, immutable RoPE
    /// tables, and physical KV pages remain shared; every region written by a
    /// launch is private to its slot.
    pub execution_slots: Vec<Gemma4MetalExecutionSlot>,
}

/// Mutable buffers owned by one committed Metal step.
#[derive(Debug, Clone)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct Gemma4MetalExecutionSlot {
    pub residual: MetalRegion,
    pub logits: MetalRegion,
    pub normed_hidden: MetalRegion,
    pub sampled: MetalRegion,
    pub final_argmax_partial_max: MetalRegion,
    pub final_argmax_partial_idx: MetalRegion,
    pub token_ids: MetalRegion,
    pub ple_token_inputs: Option<MetalRegion>,
    pub ple_context_inputs: Option<MetalRegion>,
    pub layers: Vec<MetalLayerExecutionSlot>,
}

#[derive(Debug, Clone)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalLayerExecutionSlot {
    pub qkv_out: MetalRegion,
    pub q: MetalRegion,
    pub k: MetalRegion,
    pub v: MetalRegion,
    pub attn_out: MetalRegion,
    pub global_decode_partials: Option<MetalRegion>,
    pub gate_up_out: MetalRegion,
    pub activated: MetalRegion,
    pub mlp_out: MetalRegion,
    pub moe_topk_indices: Option<MetalRegion>,
    pub moe_topk_weights: Option<MetalRegion>,
    pub moe_activated: Option<MetalRegion>,
    pub moe_out: Option<MetalRegion>,
    pub trace: Option<MetalLayerTraceState>,
    pub positions: MetalRegion,
    pub slot_mapping: MetalRegion,
    pub block_tables: MetalRegion,
    pub context_lens: MetalRegion,
    pub cu_seqlens: MetalRegion,
}

/// Honest live-device accounts used to size the prepared Metal arena and its
/// physical paged-KV pool. Scratch is both charged and physically allocated as
/// three independent in-flight execution slots.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct Gemma4MetalMemoryReport {
    pub recommended_working_set_bytes: u64,
    pub reserve_bytes: u64,
    pub usable_bytes: u64,
    pub weights_bytes: u64,
    pub scratch_slot_bytes: u64,
    pub scratch_budget_bytes: u64,
    pub metadata_bytes: u64,
    pub kv_budget_bytes: u64,
    pub kv_page_bytes: u64,
    pub allocated_kv_bytes: u64,
    pub prepared_arena_bytes: u64,
    pub max_useful_kv_pages: u32,
    pub admission_required_kv_pages: u32,
    pub physical_kv_pages: u32,
}

/// An authenticated low-bit tensor that is intended to replace one native
/// projection allocation in the prepared Metal model.
///
/// The descriptor contains the complete storage identity needed by the
/// planner. Callers must load the two payload regions in the same
/// lexicographic tensor-name order used by the planner.
#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalLowBitWeightReplacement {
    pub tensor_name: String,
    pub role: AppleLowBitTensorRole,
    pub format: rvllm_apple::AppleLowBitWeightFormat,
    /// `[N, K]`, matching the native `[hidden, intermediate]` weight.
    pub shape: [usize; 2],
    pub packed_values_bytes: usize,
    pub scales_bytes: usize,
}

#[derive(Debug, Clone)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalPleState {
    pub ple_dim: usize,
    pub ple_vocab_size: usize,
    pub embed_tokens_per_layer: MetalRegion,
    pub per_layer_model_projection: MetalRegion,
    pub per_layer_projection_norm: MetalRegion,
    pub token_inputs: MetalRegion,
    pub context_inputs: MetalRegion,
}

#[derive(Debug, Clone)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalOneLayerState {
    pub layer_idx: usize,
    /// Exact authenticated checkpoint tensor selected for this layer.
    pub down_proj_name: String,
    /// Optional authenticated low-bit sidecars stored in the model arena.
    pub low_bit_q_proj: Option<crate::low_bit_metal::MetalLowBitProjectionOffsets>,
    pub low_bit_k_proj: Option<crate::low_bit_metal::MetalLowBitProjectionOffsets>,
    pub low_bit_v_proj: Option<crate::low_bit_metal::MetalLowBitProjectionOffsets>,
    pub low_bit_o_proj: Option<crate::low_bit_metal::MetalLowBitProjectionOffsets>,
    pub low_bit_gate_proj: Option<crate::low_bit_metal::MetalLowBitProjectionOffsets>,
    pub low_bit_up_proj: Option<crate::low_bit_metal::MetalLowBitProjectionOffsets>,
    pub low_bit_down_proj: Option<crate::low_bit_metal::MetalLowBitProjectionOffsets>,
    pub dims: MetalProbeLayerDims,
    pub shared_kv_source_layer: Option<usize>,

    pub attn_norm: MetalRegion,
    pub qkv: MetalRegion,
    pub q_norm: Option<MetalRegion>,
    pub k_norm: Option<MetalRegion>,
    pub v_norm: Option<MetalRegion>,
    pub o_proj: MetalRegion,
    pub mlp_norm: MetalRegion,
    pub post_attn_norm: Option<MetalRegion>,
    pub pre_ff_norm: Option<MetalRegion>,
    pub post_ff_norm: Option<MetalRegion>,
    pub layer_scalar: Option<MetalRegion>,
    pub layer_scalar_dim: u32,
    pub gate_up: MetalRegion,
    /// Native dense source. Absent only when the planner authenticated an
    /// explicit low-bit replacement for this exact tensor.
    pub down_proj: Option<MetalRegion>,
    pub moe: Option<MetalMoeState>,
    pub per_layer_input_gate: Option<MetalRegion>,
    pub per_layer_projection: Option<MetalRegion>,
    pub post_per_layer_input_norm: Option<MetalRegion>,

    pub qkv_out: MetalRegion,
    pub q: MetalRegion,
    pub k: MetalRegion,
    pub v: MetalRegion,
    pub attn_out: MetalRegion,
    pub global_decode_partials: Option<MetalRegion>,
    pub gate_up_out: MetalRegion,
    pub activated: MetalRegion,
    pub mlp_out: MetalRegion,
    pub moe_topk_indices: Option<MetalRegion>,
    pub moe_topk_weights: Option<MetalRegion>,
    pub moe_activated: Option<MetalRegion>,
    pub moe_out: Option<MetalRegion>,
    pub trace: Option<MetalLayerTraceState>,

    pub positions: MetalRegion,
    pub slot_mapping: MetalRegion,
    pub cos: MetalRegion,
    pub sin: MetalRegion,
    pub block_tables: MetalRegion,
    pub context_lens: MetalRegion,
    pub cu_seqlens: MetalRegion,

    pub kv_cache_k: MetalRegion,
    pub kv_cache_v: MetalRegion,

    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
}

#[derive(Debug, Clone)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalMoeState {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub router_proj: MetalRegion,
    pub router_scale: MetalRegion,
    pub router_per_expert_scale: MetalRegion,
    pub pre_ff2_norm: MetalRegion,
    pub post_ff1_norm: MetalRegion,
    pub post_ff2_norm: MetalRegion,
    pub expert_gate_up: MetalRegion,
    pub expert_down: MetalRegion,
}

#[derive(Debug, Clone)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalLayerTraceState {
    pub input_to_layer: MetalRegion,
    pub after_input_layernorm: MetalRegion,
    pub q_projection: MetalRegion,
    pub k_projection: MetalRegion,
    pub v_projection: MetalRegion,
    pub after_q_norm: MetalRegion,
    pub after_k_norm: MetalRegion,
    pub after_v_norm: MetalRegion,
    pub after_rope_q: MetalRegion,
    pub after_rope_k: MetalRegion,
    pub attention_output: MetalRegion,
    pub after_o_proj: MetalRegion,
    pub after_post_attention_layernorm: MetalRegion,
    pub after_pre_feedforward_layernorm: MetalRegion,
    pub gate_up_out: MetalRegion,
    pub ffn_activation: MetalRegion,
    pub after_ffn_branch: MetalRegion,
    pub after_post_feedforward_layernorm: MetalRegion,
    pub per_layer_input: Option<MetalRegion>,
    pub per_layer_input_gate: Option<MetalRegion>,
    pub per_layer_projection: Option<MetalRegion>,
    pub post_per_layer_input_norm: Option<MetalRegion>,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl Gemma4MetalState {
    /// Materialize a descriptor-only view for one preallocated execution
    /// slot. This is intended to run during backend preparation, never in the
    /// launch path.
    pub fn state_for_execution_slot(&self, slot_index: usize) -> Result<Self> {
        let slot = self.execution_slots.get(slot_index).ok_or_else(|| {
            RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "execution_slot_out_of_range",
                },
                probe_ctx("prepare"),
            )
        })?;
        if slot.layers.len() != self.layers.len() {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "Metal execution slot layer count mismatch",
                },
                probe_ctx("prepare"),
            ));
        }

        let mut state = self.clone();
        state.residual = slot.residual.clone();
        state.logits = slot.logits.clone();
        state.normed_hidden = slot.normed_hidden.clone();
        state.sampled = slot.sampled.clone();
        state.final_argmax_partial_max = slot.final_argmax_partial_max.clone();
        state.final_argmax_partial_idx = slot.final_argmax_partial_idx.clone();
        state.token_ids = slot.token_ids.clone();
        if let Some(ple) = state.ple.as_mut() {
            ple.token_inputs = slot.ple_token_inputs.clone().ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "Metal execution slot is missing PLE token inputs",
                    },
                    probe_ctx("prepare"),
                )
            })?;
            ple.context_inputs = slot.ple_context_inputs.clone().ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "Metal execution slot is missing PLE context inputs",
                    },
                    probe_ctx("prepare"),
                )
            })?;
        }
        for (layer, execution) in state.layers.iter_mut().zip(&slot.layers) {
            layer.qkv_out = execution.qkv_out.clone();
            layer.q = execution.q.clone();
            layer.k = execution.k.clone();
            layer.v = execution.v.clone();
            layer.attn_out = execution.attn_out.clone();
            layer.global_decode_partials = execution.global_decode_partials.clone();
            layer.gate_up_out = execution.gate_up_out.clone();
            layer.activated = execution.activated.clone();
            layer.mlp_out = execution.mlp_out.clone();
            layer.moe_topk_indices = execution.moe_topk_indices.clone();
            layer.moe_topk_weights = execution.moe_topk_weights.clone();
            layer.moe_activated = execution.moe_activated.clone();
            layer.moe_out = execution.moe_out.clone();
            layer.trace = execution.trace.clone();
            layer.positions = execution.positions.clone();
            layer.slot_mapping = execution.slot_mapping.clone();
            layer.block_tables = execution.block_tables.clone();
            layer.context_lens = execution.context_lens.clone();
            layer.cu_seqlens = execution.cu_seqlens.clone();
        }
        // Execution views are terminal descriptors and do not need to retain
        // the preparation-only slot table.
        state.execution_slots.clear();
        Ok(state)
    }
}

#[derive(Debug, Clone)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
struct SharedLayerScratch {
    qkv_out: MetalRegion,
    q: MetalRegion,
    k: MetalRegion,
    v: MetalRegion,
    attn_out: MetalRegion,
    global_decode_partials: Option<MetalRegion>,
    gate_up_out: MetalRegion,
    activated: MetalRegion,
    mlp_out: MetalRegion,
    moe_topk_indices: Option<MetalRegion>,
    moe_topk_weights: Option<MetalRegion>,
    moe_activated: Option<MetalRegion>,
    moe_out: Option<MetalRegion>,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn execution_layer_from_state(layer: &MetalOneLayerState) -> MetalLayerExecutionSlot {
    MetalLayerExecutionSlot {
        qkv_out: layer.qkv_out.clone(),
        q: layer.q.clone(),
        k: layer.k.clone(),
        v: layer.v.clone(),
        attn_out: layer.attn_out.clone(),
        global_decode_partials: layer.global_decode_partials.clone(),
        gate_up_out: layer.gate_up_out.clone(),
        activated: layer.activated.clone(),
        mlp_out: layer.mlp_out.clone(),
        moe_topk_indices: layer.moe_topk_indices.clone(),
        moe_topk_weights: layer.moe_topk_weights.clone(),
        moe_activated: layer.moe_activated.clone(),
        moe_out: layer.moe_out.clone(),
        trace: layer.trace.clone(),
        positions: layer.positions.clone(),
        slot_mapping: layer.slot_mapping.clone(),
        block_tables: layer.block_tables.clone(),
        context_lens: layer.context_lens.clone(),
        cu_seqlens: layer.cu_seqlens.clone(),
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn allocate_region_like(
    arena: &mut MetalBufferArena,
    name: &str,
    source: &MetalRegion,
    alignment: usize,
) -> Result<MetalRegion> {
    arena.region(name, source.size, alignment)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn allocate_optional_region_like(
    arena: &mut MetalBufferArena,
    name: &str,
    source: Option<&MetalRegion>,
    alignment: usize,
) -> Result<Option<MetalRegion>> {
    source
        .map(|source| allocate_region_like(arena, name, source, alignment))
        .transpose()
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn allocate_trace_like(
    arena: &mut MetalBufferArena,
    slot_index: usize,
    layer_idx: usize,
    source: Option<&MetalLayerTraceState>,
) -> Result<Option<MetalLayerTraceState>> {
    let Some(source) = source else {
        return Ok(None);
    };
    let name = |field: &str| format!("metal_slot_{slot_index}_layer_{layer_idx}_trace_{field}");
    Ok(Some(MetalLayerTraceState {
        input_to_layer: allocate_region_like(arena, &name("input"), &source.input_to_layer, 16)?,
        after_input_layernorm: allocate_region_like(
            arena,
            &name("input_norm"),
            &source.after_input_layernorm,
            16,
        )?,
        q_projection: allocate_region_like(arena, &name("q"), &source.q_projection, 16)?,
        k_projection: allocate_region_like(arena, &name("k"), &source.k_projection, 16)?,
        v_projection: allocate_region_like(arena, &name("v"), &source.v_projection, 16)?,
        after_q_norm: allocate_region_like(arena, &name("q_norm"), &source.after_q_norm, 16)?,
        after_k_norm: allocate_region_like(arena, &name("k_norm"), &source.after_k_norm, 16)?,
        after_v_norm: allocate_region_like(arena, &name("v_norm"), &source.after_v_norm, 16)?,
        after_rope_q: allocate_region_like(arena, &name("rope_q"), &source.after_rope_q, 16)?,
        after_rope_k: allocate_region_like(arena, &name("rope_k"), &source.after_rope_k, 16)?,
        attention_output: allocate_region_like(
            arena,
            &name("attention"),
            &source.attention_output,
            16,
        )?,
        after_o_proj: allocate_region_like(arena, &name("o_proj"), &source.after_o_proj, 16)?,
        after_post_attention_layernorm: allocate_region_like(
            arena,
            &name("post_attention_norm"),
            &source.after_post_attention_layernorm,
            16,
        )?,
        after_pre_feedforward_layernorm: allocate_region_like(
            arena,
            &name("pre_ff_norm"),
            &source.after_pre_feedforward_layernorm,
            16,
        )?,
        gate_up_out: allocate_region_like(arena, &name("gate_up"), &source.gate_up_out, 16)?,
        ffn_activation: allocate_region_like(
            arena,
            &name("ffn_activation"),
            &source.ffn_activation,
            16,
        )?,
        after_ffn_branch: allocate_region_like(
            arena,
            &name("ffn_branch"),
            &source.after_ffn_branch,
            16,
        )?,
        after_post_feedforward_layernorm: allocate_region_like(
            arena,
            &name("post_ff_norm"),
            &source.after_post_feedforward_layernorm,
            16,
        )?,
        per_layer_input: allocate_optional_region_like(
            arena,
            &name("ple_input"),
            source.per_layer_input.as_ref(),
            16,
        )?,
        per_layer_input_gate: allocate_optional_region_like(
            arena,
            &name("ple_gate"),
            source.per_layer_input_gate.as_ref(),
            16,
        )?,
        per_layer_projection: allocate_optional_region_like(
            arena,
            &name("ple_projection"),
            source.per_layer_projection.as_ref(),
            16,
        )?,
        post_per_layer_input_norm: allocate_optional_region_like(
            arena,
            &name("ple_post_norm"),
            source.post_per_layer_input_norm.as_ref(),
            16,
        )?,
    }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub enum MetalProbeLayerAttentionKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct MetalProbeLayerDims {
    pub attention_kind: MetalProbeLayerAttentionKind,
    /// Exact causal attention width for sliding layers. Zero denotes full
    /// attention and is never inferred from the head shape.
    pub attention_window: u32,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub qkv_rows: usize,
    pub attn_scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct Gemma4DryRunValidation {
    pub weight_prefix: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub attention_sliding_layers: usize,
    pub attention_full_layers: usize,
    pub v_uses_k_proj_layers: usize,
    pub embed_tokens: String,
    pub final_norm: String,
    pub lm_head: Option<String>,
    pub lm_head_status: HostGemma4DryRunLmHeadStatus,
    pub final_logit_softcap: Option<f32>,
    pub fp8_scale_summary: HostGemma4DryRunFp8ScaleSummary,
    pub layers: Vec<Gemma4DryRunLayerValidation>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct Gemma4DryRunLayerValidation {
    pub layer_idx: usize,
    pub attention_kind: MetalProbeLayerAttentionKind,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: Option<String>,
    pub v_uses_k_proj: bool,
    pub input_layernorm: String,
    pub post_attention_layernorm: String,
    pub pre_feedforward_layernorm: String,
    pub post_feedforward_layernorm: String,
    pub q_norm: String,
    pub k_norm: String,
    pub layer_scalar: Option<String>,
    pub layer_scalar_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl Gemma4DryRunValidation {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        HostGemma4DryRunValidation::from_model_dir(model_dir).map(Self::from_host)
    }

    fn from_host(host: HostGemma4DryRunValidation) -> Self {
        Self {
            weight_prefix: host.weight_prefix,
            num_layers: host.num_layers,
            hidden_size: host.hidden_size,
            vocab_size: host.vocab_size,
            tie_word_embeddings: host.tie_word_embeddings,
            attention_sliding_layers: host.attention_sliding_layers,
            attention_full_layers: host.attention_full_layers,
            v_uses_k_proj_layers: host.v_uses_k_proj_layers,
            embed_tokens: host.embed_tokens,
            final_norm: host.final_norm,
            lm_head: host.lm_head,
            lm_head_status: host.lm_head_status,
            final_logit_softcap: host.final_logit_softcap,
            fp8_scale_summary: host.fp8_scale_summary,
            layers: host
                .layers
                .into_iter()
                .map(Gemma4DryRunLayerValidation::from_host)
                .collect(),
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl Gemma4DryRunLayerValidation {
    fn from_host(host: HostGemma4DryRunLayerValidation) -> Self {
        Self {
            layer_idx: host.layer_idx,
            attention_kind: match host.attention_kind {
                HostGemma4DryRunAttentionKind::Sliding => MetalProbeLayerAttentionKind::Sliding,
                HostGemma4DryRunAttentionKind::Full => MetalProbeLayerAttentionKind::Full,
            },
            q_proj: host.q_proj,
            k_proj: host.k_proj,
            v_proj: host.v_proj,
            v_uses_k_proj: host.v_uses_k_proj,
            input_layernorm: host.input_layernorm,
            post_attention_layernorm: host.post_attention_layernorm,
            pre_feedforward_layernorm: host.pre_feedforward_layernorm,
            post_feedforward_layernorm: host.post_feedforward_layernorm,
            q_norm: host.q_norm,
            k_norm: host.k_norm,
            layer_scalar: host.layer_scalar,
            layer_scalar_dim: host.layer_scalar_dim,
            rope_dim: host.rope_dim,
            rope_theta: host.rope_theta,
            sliding_window: host.sliding_window,
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalProbeLayerDims {
    fn from_arch_layer(arch: &ModelArch, layer_idx: usize) -> Result<Self> {
        let layer_type = arch
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerAttnType::Full);
        let (attention_kind, attention_window, head_dim, num_kv_heads, rope_dim, rope_theta) =
            match layer_type {
                LayerAttnType::SlidingAttention => (
                    MetalProbeLayerAttentionKind::Sliding,
                    u32::try_from(arch.sliding_window.filter(|&window| window > 0).ok_or_else(
                        || {
                            RvllmError::apple(
                                AppleError::InvalidWeightBlob {
                                    reason: "sliding attention layer requires a positive sliding_window",
                                },
                                probe_ctx("prepare"),
                            )
                        },
                    )?)
                    .map_err(|_| {
                        RvllmError::apple(
                            AppleError::InvalidWeightBlob {
                                reason: "sliding_window must fit in the Apple attention ABI",
                            },
                            probe_ctx("prepare"),
                        )
                    })?,
                    arch.head_dim,
                    arch.num_key_value_heads,
                    arch.head_dim,
                    arch.rope_theta,
                ),
            LayerAttnType::Full => {
                let head_dim = arch.global_head_dim.unwrap_or(arch.head_dim);
                let rotary_factor = arch.partial_rotary_factor.unwrap_or(1.0);
                let rope_dim = ((head_dim as f32 * rotary_factor) as usize / 2) * 2;
                (
                    MetalProbeLayerAttentionKind::Full,
                    0,
                    head_dim,
                    arch.num_global_key_value_heads
                        .unwrap_or(arch.num_key_value_heads),
                    rope_dim,
                    arch.global_rope_theta.unwrap_or(arch.rope_theta),
                )
            }
            LayerAttnType::Linear => {
                return Err(RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "model-metal-backend",
                        op: "unsupported_probe_linear_attention_layer",
                    },
                    probe_ctx("prepare"),
                ));
            }
        };
        let num_heads = arch.num_attention_heads;
        if num_heads == 0
            || num_kv_heads == 0
            || num_heads % num_kv_heads != 0
            || head_dim == 0
            || head_dim % 2 != 0
            || rope_dim == 0
            || rope_dim % 2 != 0
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
        {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "synthetic probe requires nonzero grouped attention heads, even nonzero head/rope dims, and positive rope theta",
                },
                probe_ctx("prepare"),
            ));
        }

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        Ok(Self {
            attention_kind,
            attention_window,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            rope_theta,
            q_dim,
            kv_dim,
            qkv_rows: q_dim + 2 * kv_dim,
            // Gemma 4 text attention uses unscaled scores in HF
            // (`Gemma4TextAttention.scaling = 1.0`).
            attn_scale: 1.0,
        })
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
struct ProbeModelPlan {
    debug_trace_layers: Vec<usize>,
    explicit_batch_sequences: bool,
    arch: ModelArch,
    tensors: BTreeMap<String, SafetensorTensorInfo>,
    embed_name: String,
    final_norm_name: String,
    lm_head_name: String,
    tie_embeddings: bool,
    ple_names: Option<ProbePleNames>,
    layer_names: Vec<ProbeLayerNames>,
    low_bit_replacements: BTreeMap<String, MetalLowBitWeightReplacement>,
    names: Vec<String>,
    arena_bytes: usize,
    unfloored_arena_bytes: usize,
    weights_bytes: usize,
    scratch_slot_bytes: usize,
    metadata_bytes: usize,
    kv_page_bytes: usize,
    physical_kv_pages: u32,
    memory_budget: Option<Gemma4MetalMemoryReport>,
    max_probe_tokens: usize,
    max_batch_tokens: usize,
    max_batch_sequences: usize,
}

/// A validated, device-budgeted native model plan. Build once, allocate its
/// reported arena size, then consume the same plan to load the model.
pub struct MetalModelLoadPlan {
    model_dir: std::path::PathBuf,
    plan: ProbeModelPlan,
    memory_report: Gemma4MetalMemoryReport,
}

impl MetalModelLoadPlan {
    pub fn new(
        ctx: &MetalContext,
        model_dir: &Path,
        limits: crate::MetalModelLimits,
    ) -> Result<Self> {
        let plan =
            ProbeModelPlan::with_limits(model_dir, Some(limits))?.apply_working_set_budget(ctx)?;
        let memory_report = plan
            .memory_budget
            .clone()
            .ok_or_else(model_arena_overflow)?;
        Ok(Self {
            model_dir: model_dir.to_owned(),
            plan,
            memory_report,
        })
    }

    #[must_use]
    pub fn arena_bytes(&self) -> usize {
        self.plan.arena_bytes
    }

    #[must_use]
    pub fn memory_report(&self) -> &Gemma4MetalMemoryReport {
        &self.memory_report
    }

    pub fn load(
        self,
        ctx: &MetalContext,
        arena: &mut MetalBufferArena,
        float_type: MetalFloatType,
    ) -> Result<Gemma4MetalState> {
        Gemma4MetalState::load_probe_model_from_plan(
            ctx,
            arena,
            &self.model_dir,
            float_type,
            self.plan,
        )
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
struct ProbePleNames {
    ple_dim: usize,
    ple_vocab_size: usize,
    embed_tokens_per_layer_name: String,
    per_layer_model_projection_name: String,
    per_layer_projection_norm_name: String,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
struct ProbeLayerNames {
    dims: MetalProbeLayerDims,
    attn_norm_name: String,
    o_proj_name: String,
    mlp_norm_name: String,
    down_proj_name: String,
    prefused_qkv_name: String,
    q_name: String,
    k_name: String,
    v_name: String,
    v_uses_k_proj: bool,
    q_norm_name: Option<String>,
    k_norm_name: Option<String>,
    v_norm_name: Option<String>,
    post_attn_norm_name: Option<String>,
    pre_ff_norm_name: Option<String>,
    post_ff_norm_name: Option<String>,
    layer_scalar_name: Option<String>,
    layer_scalar_dim: usize,
    intermediate_size: usize,
    prefused_gate_up_name: String,
    gate_name: String,
    up_name: String,
    moe: Option<ProbeMoeNames>,
    per_layer_input_gate_name: Option<String>,
    per_layer_projection_name: Option<String>,
    post_per_layer_input_norm_name: Option<String>,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
struct ProbeMoeNames {
    num_experts: usize,
    top_k: usize,
    intermediate_size: usize,
    router_proj_name: String,
    router_scale_name: String,
    router_per_expert_scale_name: String,
    pre_ff2_norm_name: String,
    post_ff1_norm_name: String,
    post_ff2_norm_name: String,
    expert_gate_up_name: String,
    expert_down_name: String,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl ProbeModelPlan {
    fn new(model_dir: &Path) -> Result<Self> {
        Self::with_limits(model_dir, None)
    }

    fn with_limits(model_dir: &Path, limits: Option<crate::MetalModelLimits>) -> Result<Self> {
        let arch = ModelArch::from_dir(model_dir)?;
        let (
            max_probe_tokens,
            max_batch_sequences,
            max_batch_tokens,
            debug_trace_layers,
            explicit_batch_sequences,
        ) = if let Some(limits) = limits {
            limits.validate(arch.max_position_embeddings)?;
            (
                limits.max_context_tokens,
                limits.max_batch_sequences,
                limits.max_batch_tokens,
                Vec::new(),
                true,
            )
        } else {
            let context = configured_max_total_tokens(&arch)?;
            let sequences = configured_max_batch_sequences()?;
            (
                context,
                sequences,
                configured_max_batch_tokens(context, sequences)?,
                debug_trace_layers_from_env(),
                batch_sequences_explicitly_configured(),
            )
        };
        if arch.hidden_size == 0 || arch.vocab_size == 0 {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "model has zero hidden_size or vocab_size",
                },
                probe_ctx("prepare"),
            ));
        }

        let tensors = scan_safetensor_tensors(model_dir)?;
        let weight_prefix = resolve_weight_prefix(&tensors);
        let embed_name = format!("{weight_prefix}.embed_tokens.weight");
        let final_norm_name = format!("{weight_prefix}.norm.weight");
        let prefixed_lm_head_name = format!("{weight_prefix}.lm_head.weight");
        let has_lm_head =
            tensors.contains_key("lm_head.weight") || tensors.contains_key(&prefixed_lm_head_name);
        let tie_embeddings = arch.tie_word_embeddings && !has_lm_head;
        let lm_head_name = if tensors.contains_key("lm_head.weight") {
            "lm_head.weight".to_owned()
        } else if tensors.contains_key(&prefixed_lm_head_name) {
            prefixed_lm_head_name.clone()
        } else if tie_embeddings {
            embed_name.clone()
        } else {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing lm_head weights",
                },
                probe_ctx("prepare"),
            ));
        };

        let embed_info = tensors.get(&embed_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing embed_tokens.weight",
                },
                probe_ctx("prepare"),
            )
        })?;
        if embed_info.shape.len() != 2
            || embed_info.shape[0] != arch.vocab_size
            || embed_info.shape[1] != arch.hidden_size
        {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "embed_tokens weight shape mismatch",
                },
                probe_ctx("prepare"),
            ));
        }

        let final_norm_info = tensors.get(&final_norm_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing final layer norm weight",
                },
                probe_ctx("prepare"),
            )
        })?;
        if final_norm_info.shape.len() != 1 || final_norm_info.shape[0] != arch.hidden_size {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "final layer norm shape mismatch",
                },
                probe_ctx("prepare"),
            ));
        }

        let lm_head_info = tensors.get(&lm_head_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing lm_head weight",
                },
                probe_ctx("prepare"),
            )
        })?;
        if !tie_embeddings {
            if lm_head_info.shape.len() != 2
                || lm_head_info.shape[0] != arch.vocab_size
                || lm_head_info.shape[1] != arch.hidden_size
            {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "lm_head weight shape mismatch",
                    },
                    probe_ctx("prepare"),
                ));
            }
        }

        let mut names = vec![embed_name.clone(), final_norm_name.clone()];
        if lm_head_name != embed_name {
            names.push(lm_head_name.clone());
        }

        let mut ple_weight_bytes = 0usize;
        let ple_embed_name = format!("{weight_prefix}.embed_tokens_per_layer.weight");
        let ple_names = if tensors.contains_key(&ple_embed_name) {
            let ple_dim = arch.hidden_size_per_layer_input;
            let ple_vocab_size = if arch.vocab_size_per_layer_input > 0 {
                arch.vocab_size_per_layer_input
            } else {
                arch.vocab_size
            };
            if ple_dim == 0 {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "Gemma4 per-layer embedding dim is zero",
                    },
                    probe_ctx("prepare"),
                ));
            }
            let ple_stride = arch.num_hidden_layers.checked_mul(ple_dim).ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "Gemma4 PLE stride overflow",
                    },
                    probe_ctx("prepare"),
                )
            })?;
            let per_layer_model_projection_name =
                format!("{weight_prefix}.per_layer_model_projection.weight");
            let per_layer_projection_norm_name =
                format!("{weight_prefix}.per_layer_projection_norm.weight");
            validate_tensor_shape(
                &tensors,
                &ple_embed_name,
                &[ple_vocab_size, ple_stride],
                "embed_tokens_per_layer weight shape mismatch",
            )?;
            validate_tensor_shape(
                &tensors,
                &per_layer_model_projection_name,
                &[ple_stride, arch.hidden_size],
                "per_layer_model_projection weight shape mismatch",
            )?;
            validate_tensor_shape(
                &tensors,
                &per_layer_projection_norm_name,
                &[ple_dim],
                "per_layer_projection_norm weight shape mismatch",
            )?;
            for name in [
                &ple_embed_name,
                &per_layer_model_projection_name,
                &per_layer_projection_norm_name,
            ] {
                let info = tensors.get(name).ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "missing Gemma4 PLE tensor",
                        },
                        probe_ctx("prepare"),
                    )
                })?;
                ple_weight_bytes = ple_weight_bytes.checked_add(info.nbytes).ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "Gemma4 PLE byte size overflow",
                        },
                        probe_ctx("prepare"),
                    )
                })?;
                names.push((*name).clone());
            }
            Some(ProbePleNames {
                ple_dim,
                ple_vocab_size,
                embed_tokens_per_layer_name: ple_embed_name,
                per_layer_model_projection_name,
                per_layer_projection_norm_name,
            })
        } else {
            None
        };

        let mut layer_weight_bytes = 0;
        let mut converted_router_weight_bytes = 0;
        let mut fused_qkv_bytes = 0;
        let mut fused_gate_up_bytes = 0;
        let mut layer_names = Vec::new();

        if arch.num_hidden_layers > 0 {
            let hidden = arch.hidden_size;

            if arch.intermediate_size == 0 {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "one-layer probe requires nonzero intermediate",
                    },
                    probe_ctx("prepare"),
                ));
            }

            let mut add_tensor_size = |name: &str| -> Result<()> {
                let info = tensors.get(name).ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "missing layer weight",
                        },
                        probe_ctx("prepare"),
                    )
                })?;
                layer_weight_bytes += info.nbytes;
                Ok(())
            };

            for layer_idx in 0..arch.num_hidden_layers {
                let dims = MetalProbeLayerDims::from_arch_layer(&arch, layer_idx)?;
                let q_dim = dims.q_dim;
                let kv_dim = dims.kv_dim;
                let qkv_rows = dims.qkv_rows;
                let intermediate = arch.intermediate_size_for_layer(layer_idx);
                let lprefix = format!("{weight_prefix}.layers.{layer_idx}");
                let attn_norm_name = resolve_tensor_alias(
                    &tensors,
                    vec![
                        format!("{lprefix}.input_layernorm.weight"),
                        format!("{lprefix}.pre_attention_layernorm.weight"),
                    ],
                    "missing attention norm weight",
                )?;
                let o_proj_name = format!("{lprefix}.self_attn.o_proj.weight");
                let mlp_norm_name = resolve_tensor_alias(
                    &tensors,
                    vec![
                        format!("{lprefix}.mlp_norm.weight"),
                        format!("{lprefix}.pre_feedforward_layernorm.weight"),
                        format!("{lprefix}.post_attention_layernorm.weight"),
                    ],
                    "missing mlp norm weight",
                )?;
                let down_proj_name = format!("{lprefix}.mlp.down_proj.weight");

                let prefused_qkv_name = format!("{lprefix}.self_attn.qkv.weight");
                let q_name = format!("{lprefix}.self_attn.q_proj.weight");
                let k_name = format!("{lprefix}.self_attn.k_proj.weight");
                let v_name = format!("{lprefix}.self_attn.v_proj.weight");
                let use_prefused_qkv = tensors.contains_key(&prefused_qkv_name);
                let v_uses_k_proj = !use_prefused_qkv
                    && !tensors.contains_key(&v_name)
                    && arch.attention_k_eq_v
                    && dims.attention_kind == MetalProbeLayerAttentionKind::Full;
                let q_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.self_attn.q_norm.weight")],
                );
                let k_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.self_attn.k_norm.weight")],
                );
                let v_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.self_attn.v_norm.weight")],
                );
                let post_attn_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.post_attention_layernorm.weight")],
                );
                let pre_ff_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.pre_feedforward_layernorm.weight")],
                );
                let post_ff_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.post_feedforward_layernorm.weight")],
                );
                let layer_scalar_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![
                        format!("{lprefix}.layer_scalar"),
                        format!("{lprefix}.layer_scalar.weight"),
                    ],
                );
                let prefused_gate_up_name = format!("{lprefix}.mlp.gate_up.weight");
                let gate_name = format!("{lprefix}.mlp.gate_proj.weight");
                let up_name = format!("{lprefix}.mlp.up_proj.weight");
                let use_prefused_gate_up = tensors.contains_key(&prefused_gate_up_name);
                let (
                    per_layer_input_gate_name,
                    per_layer_projection_name,
                    post_per_layer_input_norm_name,
                ) = if let Some(ple) = &ple_names {
                    let input_gate = format!("{lprefix}.per_layer_input_gate.weight");
                    let projection = format!("{lprefix}.per_layer_projection.weight");
                    let post_norm = format!("{lprefix}.post_per_layer_input_norm.weight");
                    validate_tensor_shape(
                        &tensors,
                        &input_gate,
                        &[ple.ple_dim, hidden],
                        "per_layer_input_gate weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &projection,
                        &[hidden, ple.ple_dim],
                        "per_layer_projection weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &post_norm,
                        &[hidden],
                        "post_per_layer_input_norm weight shape mismatch",
                    )?;
                    for name in [&input_gate, &projection, &post_norm] {
                        add_tensor_size(name)?;
                        names.push(name.clone());
                    }
                    (Some(input_gate), Some(projection), Some(post_norm))
                } else {
                    (None, None, None)
                };

                validate_tensor_shape(
                    &tensors,
                    &attn_norm_name,
                    &[hidden],
                    "attention norm weight shape mismatch",
                )?;
                validate_tensor_shape(
                    &tensors,
                    &o_proj_name,
                    &[hidden, q_dim],
                    "o_proj weight shape mismatch",
                )?;
                validate_tensor_shape(
                    &tensors,
                    &mlp_norm_name,
                    &[hidden],
                    "mlp norm weight shape mismatch",
                )?;
                validate_tensor_shape(
                    &tensors,
                    &down_proj_name,
                    &[hidden, intermediate],
                    "down_proj weight shape mismatch",
                )?;
                add_tensor_size(&attn_norm_name)?;
                add_tensor_size(&o_proj_name)?;
                add_tensor_size(&mlp_norm_name)?;
                add_tensor_size(&down_proj_name)?;

                names.push(attn_norm_name.clone());
                names.push(o_proj_name.clone());
                names.push(mlp_norm_name.clone());
                names.push(down_proj_name.clone());

                if use_prefused_qkv {
                    validate_tensor_shape(
                        &tensors,
                        &prefused_qkv_name,
                        &[qkv_rows, hidden],
                        "qkv weight shape mismatch",
                    )?;
                    add_tensor_size(&prefused_qkv_name)?;
                    names.push(prefused_qkv_name.clone());
                } else {
                    validate_tensor_shape(
                        &tensors,
                        &q_name,
                        &[q_dim, hidden],
                        "q_proj weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &k_name,
                        &[kv_dim, hidden],
                        "k_proj weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        if v_uses_k_proj { &k_name } else { &v_name },
                        &[kv_dim, hidden],
                        "v_proj weight shape mismatch",
                    )?;
                    fused_qkv_bytes += qkv_rows * hidden * std::mem::size_of::<f16>();
                }

                validate_optional_norm_shape(
                    &tensors,
                    &q_norm_name,
                    dims.head_dim,
                    "q_norm weight shape mismatch",
                )?;
                validate_optional_norm_shape(
                    &tensors,
                    &k_norm_name,
                    dims.head_dim,
                    "k_norm weight shape mismatch",
                )?;
                validate_optional_norm_shape(
                    &tensors,
                    &v_norm_name,
                    dims.head_dim,
                    "v_norm weight shape mismatch",
                )?;
                for norm_name in [&q_norm_name, &k_norm_name, &v_norm_name]
                    .into_iter()
                    .flatten()
                {
                    add_tensor_size(norm_name)?;
                    names.push(norm_name.clone());
                }
                validate_optional_norm_shape(
                    &tensors,
                    &post_attn_norm_name,
                    hidden,
                    "post_attention_layernorm weight shape mismatch",
                )?;
                validate_optional_norm_shape(
                    &tensors,
                    &pre_ff_norm_name,
                    hidden,
                    "pre_feedforward_layernorm weight shape mismatch",
                )?;
                validate_optional_norm_shape(
                    &tensors,
                    &post_ff_norm_name,
                    hidden,
                    "post_feedforward_layernorm weight shape mismatch",
                )?;
                for norm_name in [&post_attn_norm_name, &pre_ff_norm_name, &post_ff_norm_name]
                    .into_iter()
                    .flatten()
                {
                    if norm_name != &mlp_norm_name {
                        add_tensor_size(norm_name)?;
                        names.push(norm_name.clone());
                    }
                }
                let layer_scalar_dim =
                    validate_optional_layer_scalar_shape(&tensors, &layer_scalar_name, hidden)?;
                if let Some(layer_scalar_name) = &layer_scalar_name {
                    add_tensor_size(layer_scalar_name)?;
                    names.push(layer_scalar_name.clone());
                }

                if use_prefused_gate_up {
                    validate_tensor_shape(
                        &tensors,
                        &prefused_gate_up_name,
                        &[2 * intermediate, hidden],
                        "gate_up weight shape mismatch",
                    )?;
                    add_tensor_size(&prefused_gate_up_name)?;
                    names.push(prefused_gate_up_name.clone());
                } else {
                    validate_tensor_shape(
                        &tensors,
                        &gate_name,
                        &[intermediate, hidden],
                        "gate_proj weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &up_name,
                        &[intermediate, hidden],
                        "up_proj weight shape mismatch",
                    )?;
                    fused_gate_up_bytes += 2 * intermediate * hidden * std::mem::size_of::<f16>();
                }

                let moe = if plan_moe_enabled(&arch) {
                    let num_experts = arch.num_experts.unwrap_or(0);
                    let top_k = arch.top_k_experts.unwrap_or(0);
                    let moe_intermediate = arch.moe_intermediate_size.unwrap_or(0);
                    if num_experts == 0
                        || top_k == 0
                        || top_k > num_experts
                        || num_experts > 256
                        || top_k > 16
                        || moe_intermediate == 0
                    {
                        return Err(RvllmError::apple(
                            AppleError::FeatureNotAvailable {
                                backend: "model-metal-backend",
                                op: "unsupported_gemma4_moe_shape",
                            },
                            probe_ctx("prepare"),
                        ));
                    }
                    let router_proj_name = format!("{lprefix}.router.proj.weight");
                    let router_scale_name = format!("{lprefix}.router.scale");
                    let router_per_expert_scale_name = format!("{lprefix}.router.per_expert_scale");
                    let pre_ff2_norm_name = format!("{lprefix}.pre_feedforward_layernorm_2.weight");
                    let post_ff1_norm_name =
                        format!("{lprefix}.post_feedforward_layernorm_1.weight");
                    let post_ff2_norm_name =
                        format!("{lprefix}.post_feedforward_layernorm_2.weight");
                    let expert_gate_up_name = format!("{lprefix}.experts.gate_up_proj");
                    let expert_down_name = format!("{lprefix}.experts.down_proj");
                    validate_tensor_shape(
                        &tensors,
                        &router_proj_name,
                        &[num_experts, hidden],
                        "router.proj weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &router_scale_name,
                        &[hidden],
                        "router.scale shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &router_per_expert_scale_name,
                        &[num_experts],
                        "router.per_expert_scale shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &pre_ff2_norm_name,
                        &[hidden],
                        "pre_feedforward_layernorm_2 shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &post_ff1_norm_name,
                        &[hidden],
                        "post_feedforward_layernorm_1 shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &post_ff2_norm_name,
                        &[hidden],
                        "post_feedforward_layernorm_2 shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &expert_gate_up_name,
                        &[num_experts, 2 * moe_intermediate, hidden],
                        "experts.gate_up_proj shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &expert_down_name,
                        &[num_experts, hidden, moe_intermediate],
                        "experts.down_proj shape mismatch",
                    )?;
                    converted_router_weight_bytes +=
                        (num_experts * hidden + hidden + num_experts) * std::mem::size_of::<f32>();
                    for name in [
                        &pre_ff2_norm_name,
                        &post_ff1_norm_name,
                        &post_ff2_norm_name,
                        &expert_gate_up_name,
                        &expert_down_name,
                    ] {
                        add_tensor_size(name)?;
                        names.push(name.clone());
                    }
                    Some(ProbeMoeNames {
                        num_experts,
                        top_k,
                        intermediate_size: moe_intermediate,
                        router_proj_name,
                        router_scale_name,
                        router_per_expert_scale_name,
                        pre_ff2_norm_name,
                        post_ff1_norm_name,
                        post_ff2_norm_name,
                        expert_gate_up_name,
                        expert_down_name,
                    })
                } else {
                    None
                };

                layer_names.push(ProbeLayerNames {
                    dims,
                    attn_norm_name,
                    o_proj_name,
                    mlp_norm_name,
                    down_proj_name,
                    prefused_qkv_name,
                    q_name,
                    k_name,
                    v_name,
                    v_uses_k_proj,
                    q_norm_name,
                    k_norm_name,
                    v_norm_name,
                    post_attn_norm_name,
                    pre_ff_norm_name,
                    post_ff_norm_name,
                    layer_scalar_name,
                    layer_scalar_dim,
                    intermediate_size: intermediate,
                    prefused_gate_up_name,
                    gate_name,
                    up_name,
                    moe,
                    per_layer_input_gate_name,
                    per_layer_projection_name,
                    post_per_layer_input_norm_name,
                });
            }
        }

        let half_bytes = std::mem::size_of::<f16>();
        let i32_bytes = std::mem::size_of::<i32>();
        let f32_bytes = std::mem::size_of::<f32>();
        let embed_bytes = embed_info.nbytes;
        let final_norm_bytes = final_norm_info.nbytes;
        let lm_head_bytes = if tie_embeddings {
            0
        } else {
            lm_head_info.nbytes
        };
        let residual_bytes = arch
            .hidden_size
            .checked_mul(max_batch_tokens)
            .and_then(|v| v.checked_mul(half_bytes))
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "residual buffer size overflow",
                    },
                    probe_ctx("prepare"),
                )
            })?;
        let logits_bytes = arch
            .vocab_size
            .checked_mul(max_batch_tokens)
            .and_then(|v| v.checked_mul(half_bytes))
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "logits buffer size overflow",
                    },
                    probe_ctx("prepare"),
                )
            })?;
        let normed_hidden_bytes = residual_bytes;
        let sampled_bytes = max_batch_tokens * i32_bytes;
        let final_argmax_tile_count = arch.vocab_size.div_ceil(8);
        let final_argmax_partial_max_bytes = max_batch_tokens * final_argmax_tile_count * f32_bytes;
        let final_argmax_partial_idx_bytes = max_batch_tokens * final_argmax_tile_count * i32_bytes;
        let token_ids_bytes = max_batch_tokens * 4;
        let ple_inputs_bytes = ple_names
            .as_ref()
            .map(|ple| {
                max_batch_tokens
                    * plan_num_layers_stride(arch.num_hidden_layers, ple.ple_dim)
                    * half_bytes
                    * 2
            })
            .unwrap_or(0);

        let mut shared_scratch_bytes = 0;
        if arch.num_hidden_layers > 0 {
            let max_qkv_rows = layer_names
                .iter()
                .map(|layer| layer.dims.qkv_rows)
                .max()
                .unwrap_or(0);
            let max_q_dim = layer_names
                .iter()
                .map(|layer| layer.dims.q_dim)
                .max()
                .unwrap_or(0);
            let max_kv_dim = layer_names
                .iter()
                .map(|layer| layer.dims.kv_dim)
                .max()
                .unwrap_or(0);
            let max_intermediate = layer_names
                .iter()
                .map(|layer| layer.intermediate_size)
                .max()
                .unwrap_or(0);
            let max_moe_top_k = layer_names
                .iter()
                .filter_map(|layer| layer.moe.as_ref().map(|moe| moe.top_k))
                .max()
                .unwrap_or(0);
            let max_moe_intermediate = layer_names
                .iter()
                .filter_map(|layer| layer.moe.as_ref().map(|moe| moe.intermediate_size))
                .max()
                .unwrap_or(0);
            let qkv_bytes = if arch.hidden_size == 3840 && arch.num_hidden_layers == 48 {
                f32_bytes
            } else {
                half_bytes
            };
            shared_scratch_bytes = max_batch_tokens * max_qkv_rows * qkv_bytes
                + max_batch_tokens * max_q_dim * half_bytes
                + max_batch_tokens * max_kv_dim * half_bytes
                + max_batch_tokens * max_kv_dim * half_bytes
                + max_batch_tokens * max_q_dim * half_bytes
                + max_batch_tokens * 2 * max_intermediate * half_bytes
                + max_batch_tokens * max_intermediate * half_bytes
                + max_batch_tokens * arch.hidden_size * half_bytes
                + max_batch_tokens * max_moe_top_k * i32_bytes
                + max_batch_tokens * max_moe_top_k * f32_bytes
                + max_batch_tokens * max_moe_top_k * max_moe_intermediate * half_bytes
                + max_batch_tokens * arch.hidden_size * half_bytes
                + crate::attention_global_decode::SPLIT_SCRATCH_BYTES
                + 64;
        }

        let mut per_layer_trace_bytes = 0usize;
        let mut per_layer_mutable_metadata_bytes = 0usize;
        let mut immutable_metadata_bytes = 0usize;
        let mut kv_page_bytes = 0usize;
        if arch.num_hidden_layers > 0 {
            let hidden = arch.hidden_size;
            for (layer_idx, layer_names) in layer_names.iter().enumerate() {
                let dims = layer_names.dims;
                let intermediate = layer_names.intermediate_size;
                let trace_bytes = if debug_trace_layer_enabled(&debug_trace_layers, layer_idx) {
                    let ple_dim = ple_names.as_ref().map_or(0, |ple| ple.ple_dim);
                    let ple_trace_elems = if ple_dim > 0 {
                        2 * ple_dim + 2 * hidden
                    } else {
                        0
                    };
                    max_batch_tokens
                        * (7 * hidden
                            + 4 * dims.q_dim
                            + 5 * dims.kv_dim
                            + 3 * intermediate
                            + ple_trace_elems)
                        * half_bytes
                } else {
                    0
                };

                let block_size = APPLE_KV_PAGE_TOKENS;
                let max_blocks_per_seq = max_probe_tokens.div_ceil(block_size);
                let layer_kv_page_bytes = block_size * dims.kv_dim * half_bytes * 2;
                let layer_metadata_bytes = (2 * max_batch_tokens
                    + 2 * max_batch_sequences
                    + max_batch_sequences * max_blocks_per_seq
                    + 1)
                    * i32_bytes;

                let half_rope = dims.rope_dim / 2;
                let max_pos = max_probe_tokens;
                let rope_table_bytes = max_pos * half_rope * f32_bytes;

                per_layer_trace_bytes = per_layer_trace_bytes
                    .checked_add(trace_bytes)
                    .ok_or_else(model_arena_overflow)?;
                kv_page_bytes = kv_page_bytes
                    .checked_add(layer_kv_page_bytes)
                    .ok_or_else(model_arena_overflow)?;
                per_layer_mutable_metadata_bytes = per_layer_mutable_metadata_bytes
                    .checked_add(layer_metadata_bytes)
                    .and_then(|bytes| bytes.checked_add(64))
                    .ok_or_else(model_arena_overflow)?;
                immutable_metadata_bytes = immutable_metadata_bytes
                    .checked_add(rope_table_bytes * 2)
                    .and_then(|bytes| bytes.checked_add(64))
                    .ok_or_else(model_arena_overflow)?;
            }
        }

        let weights_bytes = embed_bytes
            .checked_add(final_norm_bytes)
            .and_then(|v| v.checked_add(lm_head_bytes))
            .and_then(|v| v.checked_add(ple_weight_bytes))
            .and_then(|v| v.checked_add(layer_weight_bytes))
            .and_then(|v| v.checked_add(converted_router_weight_bytes))
            .and_then(|v| v.checked_add(fused_qkv_bytes))
            .and_then(|v| v.checked_add(fused_gate_up_bytes))
            .ok_or_else(model_arena_overflow)?;
        let scratch_slot_bytes = residual_bytes
            .checked_add(logits_bytes)
            .and_then(|v| v.checked_add(normed_hidden_bytes))
            .and_then(|v| v.checked_add(sampled_bytes))
            .and_then(|v| v.checked_add(final_argmax_partial_max_bytes))
            .and_then(|v| v.checked_add(final_argmax_partial_idx_bytes))
            .and_then(|v| v.checked_add(token_ids_bytes))
            .and_then(|v| v.checked_add(ple_inputs_bytes))
            .and_then(|v| v.checked_add(shared_scratch_bytes))
            .and_then(|v| v.checked_add(per_layer_trace_bytes))
            .and_then(|v| v.checked_add(per_layer_mutable_metadata_bytes))
            .ok_or_else(model_arena_overflow)?;
        let metadata_bytes = immutable_metadata_bytes
            .checked_add(64 * 1024)
            .ok_or_else(model_arena_overflow)?;
        let max_blocks_per_seq = max_probe_tokens.div_ceil(APPLE_KV_PAGE_TOKENS);
        let physical_kv_pages = max_batch_sequences
            .checked_mul(max_blocks_per_seq)
            .and_then(|pages| u32::try_from(pages).ok())
            .ok_or_else(model_arena_overflow)?;
        let allocated_kv_bytes = kv_page_bytes
            .checked_mul(physical_kv_pages as usize)
            .ok_or_else(model_arena_overflow)?;
        let unfloored_arena_bytes = weights_bytes
            .checked_add(
                scratch_slot_bytes
                    .checked_mul(crate::memory_budget::IN_FLIGHT_SCRATCH_SLOTS)
                    .ok_or_else(model_arena_overflow)?,
            )
            .and_then(|v| v.checked_add(metadata_bytes))
            .and_then(|v| v.checked_add(allocated_kv_bytes))
            .ok_or_else(model_arena_overflow)?;
        let arena_bytes = max(unfloored_arena_bytes, PROBE_METAL_ARENA_BYTES);

        Ok(Self {
            debug_trace_layers,
            explicit_batch_sequences,
            arch,
            tensors,
            embed_name,
            final_norm_name,
            lm_head_name,
            tie_embeddings,
            ple_names,
            layer_names,
            low_bit_replacements: BTreeMap::new(),
            names,
            arena_bytes,
            unfloored_arena_bytes,
            weights_bytes,
            scratch_slot_bytes,
            metadata_bytes,
            kv_page_bytes,
            physical_kv_pages,
            memory_budget: None,
            max_probe_tokens,
            max_batch_tokens,
            max_batch_sequences,
        })
    }

    fn with_low_bit_replacements(
        mut self,
        replacements: &[MetalLowBitWeightReplacement],
    ) -> Result<Self> {
        if replacements.is_empty() {
            return Ok(self);
        }

        // Validate a sorted copy before changing names or accounting. This
        // makes failure deterministic even when the package manifest order is
        // not, and guarantees that no partial plan can escape.
        let mut replacements = replacements.to_vec();
        replacements.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));

        let mut validated = BTreeMap::new();
        let mut displaced_native_bytes = 0usize;
        let mut low_bit_arena_bytes = 0usize;
        for replacement in replacements {
            if replacement.tensor_name.is_empty() {
                return Err(invalid_low_bit_replacement(
                    "low-bit replacement tensor name is empty",
                ));
            }
            if validated.contains_key(&replacement.tensor_name) {
                return Err(invalid_low_bit_replacement(
                    "duplicate low-bit replacement tensor name",
                ));
            }
            if replacement.role != AppleLowBitTensorRole::DenseDownProjection {
                return Err(invalid_low_bit_replacement(
                    "authenticated low-bit role is not wired into the normal Metal route",
                ));
            }

            let layer = self
                .layer_names
                .iter()
                .find(|layer| layer.down_proj_name == replacement.tensor_name)
                .ok_or_else(|| {
                    invalid_low_bit_replacement(
                        "low-bit replacement does not name a prepared dense down projection",
                    )
                })?;
            if layer.moe.is_some() {
                return Err(invalid_low_bit_replacement(
                    "low-bit replacement of a mixture-of-experts layer is unsupported",
                ));
            }
            let expected_shape = [self.arch.hidden_size, layer.intermediate_size];
            if replacement.shape != expected_shape {
                return Err(invalid_low_bit_replacement(
                    "low-bit replacement shape does not match the prepared dense down projection",
                ));
            }

            let expected_values = low_bit_packed_values_bytes(
                replacement.format,
                replacement.shape[0],
                replacement.shape[1],
            )?;
            let expected_scales = low_bit_scales_bytes(replacement.shape[0], replacement.shape[1])?;
            if replacement.packed_values_bytes != expected_values {
                return Err(invalid_low_bit_replacement(
                    "low-bit replacement packed-value byte length is invalid",
                ));
            }
            if replacement.scales_bytes != expected_scales {
                return Err(invalid_low_bit_replacement(
                    "low-bit replacement scale byte length is invalid",
                ));
            }

            let native = self.tensors.get(&replacement.tensor_name).ok_or_else(|| {
                invalid_low_bit_replacement(
                    "low-bit replacement native tensor is missing from the checkpoint",
                )
            })?;
            displaced_native_bytes = displaced_native_bytes
                .checked_add(native.nbytes)
                .ok_or_else(model_arena_overflow)?;

            // Sidecars are loaded into two 16-byte-aligned arena regions. The
            // first starts from an aligned model-arena cursor; account the
            // exact inter-region and inter-tensor padding in sorted load order.
            low_bit_arena_bytes = align_up_checked(low_bit_arena_bytes, 16)?;
            low_bit_arena_bytes = low_bit_arena_bytes
                .checked_add(replacement.packed_values_bytes)
                .ok_or_else(model_arena_overflow)?;
            low_bit_arena_bytes = align_up_checked(low_bit_arena_bytes, 16)?;
            low_bit_arena_bytes = low_bit_arena_bytes
                .checked_add(replacement.scales_bytes)
                .ok_or_else(model_arena_overflow)?;

            validated.insert(replacement.tensor_name.clone(), replacement);
        }

        self.weights_bytes = self
            .weights_bytes
            .checked_sub(displaced_native_bytes)
            .and_then(|bytes| bytes.checked_add(low_bit_arena_bytes))
            .ok_or_else(model_arena_overflow)?;
        self.unfloored_arena_bytes = self
            .unfloored_arena_bytes
            .checked_sub(displaced_native_bytes)
            .and_then(|bytes| bytes.checked_add(low_bit_arena_bytes))
            .ok_or_else(model_arena_overflow)?;
        self.arena_bytes = max(self.unfloored_arena_bytes, PROBE_METAL_ARENA_BYTES);
        self.names
            .retain(|name| !validated.contains_key(name.as_str()));
        self.low_bit_replacements = validated;
        Ok(self)
    }

    fn with_additional_weight_bytes(mut self, additional_weight_bytes: usize) -> Result<Self> {
        if additional_weight_bytes == 0 {
            return Ok(self);
        }
        self.weights_bytes = self
            .weights_bytes
            .checked_add(additional_weight_bytes)
            .ok_or_else(model_arena_overflow)?;
        self.unfloored_arena_bytes = self
            .unfloored_arena_bytes
            .checked_add(additional_weight_bytes)
            .ok_or_else(model_arena_overflow)?;
        self.arena_bytes = self
            .arena_bytes
            .checked_add(additional_weight_bytes)
            .ok_or_else(model_arena_overflow)?;
        Ok(self)
    }

    fn apply_working_set_budget(mut self, ctx: &MetalContext) -> Result<Self> {
        let weights_bytes = u64::try_from(self.weights_bytes)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let scratch_slot_bytes = u64::try_from(self.scratch_slot_bytes)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let metadata_bytes = u64::try_from(self.metadata_bytes)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let kv_page_bytes = u64::try_from(self.kv_page_bytes)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let max_blocks_per_seq = self.max_probe_tokens.div_ceil(APPLE_KV_PAGE_TOKENS);
        let admission_sequences = if self.explicit_batch_sequences {
            self.max_batch_sequences
        } else {
            1
        };
        let admission_required_pages = max_blocks_per_seq
            .checked_mul(admission_sequences)
            .and_then(|pages| u64::try_from(pages).ok())
            .ok_or_else(|| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let max_useful_pages = u64::from(self.physical_kv_pages);
        let budget = ctx
            .memory_budget(weights_bytes, scratch_slot_bytes, metadata_bytes)
            .map_err(memory_budget_error)?;
        let capacity = budget
            .plan_kv_pages(kv_page_bytes, max_useful_pages, admission_required_pages)
            .map_err(memory_budget_error)?;
        let physical_kv_pages = u32::try_from(capacity.physical_pages)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let max_useful_kv_pages = u32::try_from(capacity.max_useful_pages)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let admission_required_kv_pages = u32::try_from(admission_required_pages)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let allocated_kv_bytes = usize::try_from(capacity.allocated_kv_bytes)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        let arena_bytes = self
            .weights_bytes
            .checked_add(
                self.scratch_slot_bytes
                    .checked_mul(crate::memory_budget::IN_FLIGHT_SCRATCH_SLOTS)
                    .ok_or_else(model_arena_overflow)?,
            )
            .and_then(|bytes| bytes.checked_add(self.metadata_bytes))
            .and_then(|bytes| bytes.checked_add(allocated_kv_bytes))
            .ok_or_else(model_arena_overflow)?;

        self.physical_kv_pages = physical_kv_pages;
        self.unfloored_arena_bytes = arena_bytes;
        self.arena_bytes = max(arena_bytes, PROBE_METAL_ARENA_BYTES);
        let arena_bytes_u64 = u64::try_from(self.arena_bytes)
            .map_err(|_| memory_budget_error(AppleMemoryBudgetError::ArithmeticNarrowing))?;
        if arena_bytes_u64 > capacity.budget.usable_bytes {
            return Err(memory_budget_error(
                AppleMemoryBudgetError::FixedResourcesExceedBudget {
                    required: arena_bytes_u64,
                    usable: capacity.budget.usable_bytes,
                },
            ));
        }
        self.memory_budget = Some(Gemma4MetalMemoryReport {
            recommended_working_set_bytes: capacity.budget.working_set_bytes,
            reserve_bytes: capacity.budget.reserve_bytes,
            usable_bytes: capacity.budget.usable_bytes,
            weights_bytes: capacity.budget.weights_bytes,
            scratch_slot_bytes,
            scratch_budget_bytes: capacity.budget.scratch_bytes,
            metadata_bytes: capacity.budget.metadata_bytes,
            kv_budget_bytes: capacity.budget.kv_pool_bytes,
            kv_page_bytes: capacity.bytes_per_page,
            allocated_kv_bytes: capacity.allocated_kv_bytes,
            prepared_arena_bytes: arena_bytes_u64,
            max_useful_kv_pages,
            admission_required_kv_pages,
            physical_kv_pages,
        });
        Ok(self)
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn memory_budget_error(error: AppleMemoryBudgetError) -> RvllmError {
    let reason = match error {
        AppleMemoryBudgetError::MissingWorkingSetLimit => {
            "Metal device did not report a recommended working-set limit"
        }
        AppleMemoryBudgetError::InvalidReservePercent => "invalid Apple memory reserve percentage",
        AppleMemoryBudgetError::ArithmeticOverflow
        | AppleMemoryBudgetError::ArithmeticNarrowing => "Metal memory budget arithmetic overflow",
        AppleMemoryBudgetError::FixedResourcesExceedBudget { .. } => {
            "Metal fixed resources exceed recommended working-set budget"
        }
        AppleMemoryBudgetError::ZeroSizedKvPage => "Metal model has a zero-sized KV page",
        AppleMemoryBudgetError::ZeroUsefulKvPages => "Metal model has no useful KV pages",
        AppleMemoryBudgetError::InsufficientKvPages { .. } => {
            "Metal KV budget cannot satisfy configured max-context admission"
        }
    };
    RvllmError::apple(
        AppleError::InvalidWeightBlob { reason },
        probe_ctx("memory_budget"),
    )
}

#[cfg(target_os = "macos")]
fn parse_positive_env_usize(name: &'static str, op: &'static str) -> Result<Option<usize>> {
    match std::env::var(name) {
        Ok(raw) => {
            let value = raw.trim().parse::<usize>().map_err(|_| {
                RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "model-metal-backend",
                        op,
                    },
                    probe_ctx("prepare"),
                )
            })?;
            if value == 0 {
                return Err(RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "model-metal-backend",
                        op,
                    },
                    probe_ctx("prepare"),
                ));
            }
            Ok(Some(value))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op,
            },
            probe_ctx("prepare"),
        )),
    }
}

#[cfg(target_os = "macos")]
fn batch_sequences_explicitly_configured() -> bool {
    std::env::var_os(RVLLM_METAL_MAX_BATCH_SEQUENCES_ENV).is_some()
}

#[cfg(target_os = "ios")]
fn batch_sequences_explicitly_configured() -> bool {
    false
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn configured_max_total_tokens(arch: &ModelArch) -> Result<usize> {
    let configured = configured_max_total_tokens_override()?;
    let value = configured
        .unwrap_or_else(|| METAL_DEFAULT_MAX_TOTAL_TOKENS.min(arch.max_position_embeddings.max(1)));
    if value > arch.max_position_embeddings {
        return Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op: "unsupported_context_length_exceeds_model_max_position",
            },
            probe_ctx("prepare"),
        ));
    }
    Ok(value)
}

#[cfg(target_os = "macos")]
fn configured_max_total_tokens_override() -> Result<Option<usize>> {
    Ok(parse_positive_env_usize(
        RVLLM_METAL_MAX_TOTAL_TOKENS_ENV,
        "invalid_RVLLM_METAL_MAX_TOTAL_TOKENS",
    )?
    .or(parse_positive_env_usize(
        RVLLM_METAL_MAX_PROBE_TOKENS_ENV,
        "invalid_RVLLM_METAL_MAX_PROBE_TOKENS",
    )?))
}

#[cfg(target_os = "ios")]
fn configured_max_total_tokens_override() -> Result<Option<usize>> {
    Ok(None)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn configured_max_batch_sequences() -> Result<usize> {
    configured_max_batch_sequences_override()
        .map(|value| value.unwrap_or(METAL_DEFAULT_MAX_BATCH_SEQUENCES))
}

#[cfg(target_os = "macos")]
fn configured_max_batch_sequences_override() -> Result<Option<usize>> {
    parse_positive_env_usize(
        RVLLM_METAL_MAX_BATCH_SEQUENCES_ENV,
        "invalid_RVLLM_METAL_MAX_BATCH_SEQUENCES",
    )
}

#[cfg(target_os = "ios")]
fn configured_max_batch_sequences_override() -> Result<Option<usize>> {
    Ok(None)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn configured_max_batch_tokens(
    max_total_tokens: usize,
    max_batch_sequences: usize,
) -> Result<usize> {
    let default = max_total_tokens
        .checked_mul(max_batch_sequences)
        .ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "Metal batch token capacity overflow",
                },
                probe_ctx("prepare"),
            )
        })?;
    configured_max_batch_tokens_override().map(|value| value.unwrap_or(default))
}

#[cfg(target_os = "macos")]
fn configured_max_batch_tokens_override() -> Result<Option<usize>> {
    parse_positive_env_usize(
        RVLLM_METAL_MAX_BATCH_TOKENS_ENV,
        "invalid_RVLLM_METAL_MAX_BATCH_TOKENS",
    )
}

#[cfg(target_os = "ios")]
fn configured_max_batch_tokens_override() -> Result<Option<usize>> {
    Ok(None)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn plan_num_layers_stride(num_layers: usize, ple_dim: usize) -> usize {
    num_layers.saturating_mul(ple_dim)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn plan_moe_enabled(arch: &ModelArch) -> bool {
    arch.enable_moe_block
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl Gemma4MetalState {
    pub fn dry_run_validate_gemma4_model_dir(model_dir: &Path) -> Result<Gemma4DryRunValidation> {
        Gemma4DryRunValidation::from_model_dir(model_dir)
    }

    pub fn required_probe_model_arena_bytes(model_dir: &Path) -> Result<usize> {
        Ok(ProbeModelPlan::new(model_dir)?.arena_bytes)
    }

    /// Arena requirement after replacing selected native dense down
    /// projections with authenticated low-bit sidecars.
    pub fn required_probe_model_arena_bytes_with_low_bit_replacements(
        model_dir: &Path,
        replacements: &[MetalLowBitWeightReplacement],
    ) -> Result<usize> {
        Ok(ProbeModelPlan::new(model_dir)?
            .with_low_bit_replacements(replacements)?
            .arena_bytes)
    }

    /// Device-budgeted arena requirement and its independently accounted
    /// memory report. This is the production preparation path.
    pub fn required_probe_model_arena_bytes_for_device(
        ctx: &MetalContext,
        model_dir: &Path,
    ) -> Result<(usize, Gemma4MetalMemoryReport)> {
        Self::required_probe_model_arena_bytes_for_device_with_additional_weights(ctx, model_dir, 0)
    }

    /// Device-budgeted arena requirement including immutable package sidecars.
    ///
    /// Extra bytes are accounted as weights before the physical KV page pool
    /// is sized, so low-bit residency can never be hidden in unreported Metal
    /// allocations.
    pub fn required_probe_model_arena_bytes_for_device_with_additional_weights(
        ctx: &MetalContext,
        model_dir: &Path,
        additional_weight_bytes: usize,
    ) -> Result<(usize, Gemma4MetalMemoryReport)> {
        let plan = ProbeModelPlan::new(model_dir)?
            .with_additional_weight_bytes(additional_weight_bytes)?
            .apply_working_set_budget(ctx)?;
        let report = plan.memory_budget.ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing Metal model memory budget",
                },
                probe_ctx("memory_budget"),
            )
        })?;
        Ok((plan.arena_bytes, report))
    }

    /// Device-budgeted requirement for resident low-bit replacement weights.
    ///
    /// Validation and accounting complete before a caller creates or mutates
    /// the destination arena.
    pub fn required_probe_model_arena_bytes_for_device_with_low_bit_replacements(
        ctx: &MetalContext,
        model_dir: &Path,
        replacements: &[MetalLowBitWeightReplacement],
    ) -> Result<(usize, Gemma4MetalMemoryReport)> {
        let plan = ProbeModelPlan::new(model_dir)?
            .with_low_bit_replacements(replacements)?
            .apply_working_set_budget(ctx)?;
        let report = plan.memory_budget.ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing Metal model memory budget",
                },
                probe_ctx("memory_budget"),
            )
        })?;
        Ok((plan.arena_bytes, report))
    }

    pub fn preferred_probe_model_float_type(model_dir: &Path) -> Result<MetalFloatType> {
        let tensors = scan_safetensor_tensors(model_dir)?;
        if tensors
            .values()
            .any(|tensor| tensor.dtype == rvllm_core::DType::Bf16)
        {
            Ok(MetalFloatType::Bf16)
        } else {
            Ok(MetalFloatType::F16)
        }
    }

    pub fn load_probe_model(
        ctx: &MetalContext,
        arena: &mut MetalBufferArena,
        model_dir: &Path,
    ) -> Result<Self> {
        let float_type = Self::preferred_probe_model_float_type(model_dir)?;
        Self::load_probe_model_with_float_type(ctx, arena, model_dir, float_type)
    }

    pub fn load_probe_model_with_float_type(
        ctx: &MetalContext,
        arena: &mut MetalBufferArena,
        model_dir: &Path,
        float_type: MetalFloatType,
    ) -> Result<Self> {
        Self::load_probe_model_with_float_type_and_additional_weights(
            ctx, arena, model_dir, float_type, 0,
        )
    }

    /// Load native checkpoint weights while reserving and accounting immutable
    /// package sidecars in the same arena.
    pub fn load_probe_model_with_float_type_and_additional_weights(
        ctx: &MetalContext,
        arena: &mut MetalBufferArena,
        model_dir: &Path,
        float_type: MetalFloatType,
        additional_weight_bytes: usize,
    ) -> Result<Self> {
        let plan = ProbeModelPlan::new(model_dir)?
            .with_additional_weight_bytes(additional_weight_bytes)?
            .apply_working_set_budget(ctx)?;
        Self::load_probe_model_from_plan(ctx, arena, model_dir, float_type, plan)
    }

    /// Load a model whose selected native dense down projections have been
    /// omitted so authenticated low-bit sidecars can occupy their residency.
    pub fn load_probe_model_with_float_type_and_low_bit_replacements(
        ctx: &MetalContext,
        arena: &mut MetalBufferArena,
        model_dir: &Path,
        float_type: MetalFloatType,
        replacements: &[MetalLowBitWeightReplacement],
    ) -> Result<Self> {
        let plan = ProbeModelPlan::new(model_dir)?
            .with_low_bit_replacements(replacements)?
            .apply_working_set_budget(ctx)?;
        Self::load_probe_model_from_plan(ctx, arena, model_dir, float_type, plan)
    }

    fn load_probe_model_from_plan(
        ctx: &MetalContext,
        arena: &mut MetalBufferArena,
        model_dir: &Path,
        float_type: MetalFloatType,
        plan: ProbeModelPlan,
    ) -> Result<Self> {
        let _ = ctx;
        let debug_trace_layers = &plan.debug_trace_layers;
        let mut mapped_refs = map_safetensor_to_arena_with_float_type(
            arena,
            model_dir,
            &plan
                .names
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>(),
            float_type,
        )?;
        let embedding = region_lookup(&mut mapped_refs, &plan.embed_name)?;
        let final_norm = region_lookup(&mut mapped_refs, &plan.final_norm_name)?;
        let lm_head = if plan.tie_embeddings {
            embedding.clone()
        } else {
            region_lookup(&mut mapped_refs, &plan.lm_head_name)?
        };

        let half_bytes = std::mem::size_of::<f16>();
        let f32_bytes = std::mem::size_of::<f32>();
        let max_probe_tokens = plan.max_probe_tokens;
        let max_batch_tokens = plan.max_batch_tokens;
        let max_batch_sequences = plan.max_batch_sequences;
        let residual_bytes = max_batch_tokens * plan.arch.hidden_size * half_bytes;
        let logits_bytes = max_batch_tokens * plan.arch.vocab_size * half_bytes;
        let normed_hidden_bytes = residual_bytes;
        let sampled_bytes = max_batch_tokens * std::mem::size_of::<i32>();
        let final_argmax_tile_count = plan.arch.vocab_size.div_ceil(8);
        let final_argmax_partial_max_bytes = max_batch_tokens * final_argmax_tile_count * f32_bytes;
        let final_argmax_partial_idx_bytes =
            max_batch_tokens * final_argmax_tile_count * std::mem::size_of::<i32>();
        let token_ids_bytes = max_batch_tokens * 4;

        let ple = if let Some(ple_names) = &plan.ple_names {
            let embed_tokens_per_layer =
                region_lookup(&mut mapped_refs, &ple_names.embed_tokens_per_layer_name)?;
            let per_layer_model_projection =
                region_lookup(&mut mapped_refs, &ple_names.per_layer_model_projection_name)?;
            let per_layer_projection_norm =
                region_lookup(&mut mapped_refs, &ple_names.per_layer_projection_norm_name)?;
            let stride = plan_num_layers_stride(plan.arch.num_hidden_layers, ple_names.ple_dim);
            let token_inputs = arena.region(
                "metal_model_ple_token_inputs",
                max_batch_tokens * stride * half_bytes,
                16,
            )?;
            let context_inputs = arena.region(
                "metal_model_ple_context_inputs",
                max_batch_tokens * stride * half_bytes,
                16,
            )?;
            Some(MetalPleState {
                ple_dim: ple_names.ple_dim,
                ple_vocab_size: ple_names.ple_vocab_size,
                embed_tokens_per_layer,
                per_layer_model_projection,
                per_layer_projection_norm,
                token_inputs,
                context_inputs,
            })
        } else {
            None
        };

        let mut layers = Vec::new();
        let shared_kv_source_layers = plan.arch.shared_kv_source_layers();
        let shared_scratch = if plan.arch.num_hidden_layers > 0 {
            let max_qkv_rows = plan
                .layer_names
                .iter()
                .map(|layer| layer.dims.qkv_rows)
                .max()
                .unwrap_or(0);
            let max_q_dim = plan
                .layer_names
                .iter()
                .map(|layer| layer.dims.q_dim)
                .max()
                .unwrap_or(0);
            let max_kv_dim = plan
                .layer_names
                .iter()
                .map(|layer| layer.dims.kv_dim)
                .max()
                .unwrap_or(0);
            let max_intermediate = plan
                .layer_names
                .iter()
                .map(|layer| layer.intermediate_size)
                .max()
                .unwrap_or(0);
            let max_moe_top_k = plan
                .layer_names
                .iter()
                .filter_map(|layer| layer.moe.as_ref().map(|moe| moe.top_k))
                .max()
                .unwrap_or(0);
            let max_moe_intermediate = plan
                .layer_names
                .iter()
                .filter_map(|layer| layer.moe.as_ref().map(|moe| moe.intermediate_size))
                .max()
                .unwrap_or(0);
            let qkv_bytes = if plan.arch.hidden_size == 3840 && plan.arch.num_hidden_layers == 48 {
                std::mem::size_of::<f32>()
            } else {
                half_bytes
            };
            Some(SharedLayerScratch {
                qkv_out: arena.region(
                    "metal_shared_layer_qkv_out",
                    max_batch_tokens * max_qkv_rows * qkv_bytes,
                    16,
                )?,
                q: arena.region(
                    "metal_shared_layer_q",
                    max_batch_tokens * max_q_dim * half_bytes,
                    16,
                )?,
                k: arena.region(
                    "metal_shared_layer_k",
                    max_batch_tokens * max_kv_dim * half_bytes,
                    16,
                )?,
                v: arena.region(
                    "metal_shared_layer_v",
                    max_batch_tokens * max_kv_dim * half_bytes,
                    16,
                )?,
                attn_out: arena.region(
                    "metal_shared_layer_attn_out",
                    max_batch_tokens * max_q_dim * half_bytes,
                    16,
                )?,
                global_decode_partials: Some(arena.region(
                    "metal_shared_global_decode_partials",
                    crate::attention_global_decode::SPLIT_SCRATCH_BYTES,
                    16,
                )?),
                gate_up_out: arena.region(
                    "metal_shared_layer_gate_up_out",
                    max_batch_tokens * 2 * max_intermediate * half_bytes,
                    16,
                )?,
                activated: arena.region(
                    "metal_shared_layer_activated",
                    max_batch_tokens * max_intermediate * half_bytes,
                    16,
                )?,
                mlp_out: arena.region(
                    "metal_shared_layer_mlp_out",
                    max_batch_tokens * plan.arch.hidden_size * half_bytes,
                    16,
                )?,
                moe_topk_indices: (max_moe_top_k > 0)
                    .then(|| {
                        arena.region(
                            "metal_shared_layer_moe_topk_indices",
                            max_batch_tokens * max_moe_top_k * std::mem::size_of::<i32>(),
                            4,
                        )
                    })
                    .transpose()?,
                moe_topk_weights: (max_moe_top_k > 0)
                    .then(|| {
                        arena.region(
                            "metal_shared_layer_moe_topk_weights",
                            max_batch_tokens * max_moe_top_k * f32_bytes,
                            4,
                        )
                    })
                    .transpose()?,
                moe_activated: (max_moe_top_k > 0 && max_moe_intermediate > 0)
                    .then(|| {
                        arena.region(
                            "metal_shared_layer_moe_activated",
                            max_batch_tokens * max_moe_top_k * max_moe_intermediate * half_bytes,
                            16,
                        )
                    })
                    .transpose()?,
                moe_out: (max_moe_top_k > 0)
                    .then(|| {
                        arena.region(
                            "metal_shared_layer_moe_out",
                            max_batch_tokens * plan.arch.hidden_size * half_bytes,
                            16,
                        )
                    })
                    .transpose()?,
            })
        } else {
            None
        };

        for layer_idx in 0..plan.arch.num_hidden_layers {
            let layer_names = plan.layer_names.get(layer_idx).ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "missing layer name plan",
                    },
                    probe_ctx("prepare"),
                )
            })?;
            let hidden = plan.arch.hidden_size;
            let intermediate = layer_names.intermediate_size;
            let dims = layer_names.dims;
            let q_dim = dims.q_dim;
            let kv_dim = dims.kv_dim;

            let attn_norm = region_lookup(&mut mapped_refs, &layer_names.attn_norm_name)?;
            let o_proj = region_lookup(&mut mapped_refs, &layer_names.o_proj_name)?;
            let mlp_norm = region_lookup(&mut mapped_refs, &layer_names.mlp_norm_name)?;
            let down_proj = if plan
                .low_bit_replacements
                .contains_key(&layer_names.down_proj_name)
            {
                None
            } else {
                Some(region_lookup(
                    &mut mapped_refs,
                    &layer_names.down_proj_name,
                )?)
            };
            let q_norm =
                optional_region_lookup(&mut mapped_refs, layer_names.q_norm_name.as_deref())?;
            let k_norm =
                optional_region_lookup(&mut mapped_refs, layer_names.k_norm_name.as_deref())?;
            let v_norm =
                optional_region_lookup(&mut mapped_refs, layer_names.v_norm_name.as_deref())?;
            let post_attn_norm = optional_distinct_region_lookup(
                &mut mapped_refs,
                layer_names.post_attn_norm_name.as_deref(),
                &layer_names.mlp_norm_name,
            )?;
            let pre_ff_norm = optional_region_or_alias_lookup(
                &mut mapped_refs,
                layer_names.pre_ff_norm_name.as_deref(),
                &layer_names.mlp_norm_name,
                &mlp_norm,
            )?;
            let post_ff_norm = optional_distinct_region_lookup(
                &mut mapped_refs,
                layer_names.post_ff_norm_name.as_deref(),
                &layer_names.mlp_norm_name,
            )?;
            let layer_scalar =
                optional_region_lookup(&mut mapped_refs, layer_names.layer_scalar_name.as_deref())?;
            let per_layer_input_gate = optional_region_lookup(
                &mut mapped_refs,
                layer_names.per_layer_input_gate_name.as_deref(),
            )?;
            let per_layer_projection = optional_region_lookup(
                &mut mapped_refs,
                layer_names.per_layer_projection_name.as_deref(),
            )?;
            let post_per_layer_input_norm = optional_region_lookup(
                &mut mapped_refs,
                layer_names.post_per_layer_input_norm_name.as_deref(),
            )?;

            let qkv = if plan.tensors.contains_key(&layer_names.prefused_qkv_name) {
                region_lookup(&mut mapped_refs, &layer_names.prefused_qkv_name)?
            } else {
                let bytes = concat_f16_tensors(
                    &plan.tensors,
                    &[
                        layer_names.q_name.clone(),
                        layer_names.k_name.clone(),
                        if layer_names.v_uses_k_proj {
                            layer_names.k_name.clone()
                        } else {
                            layer_names.v_name.clone()
                        },
                    ],
                    &[
                        vec![q_dim, hidden],
                        vec![kv_dim, hidden],
                        vec![kv_dim, hidden],
                    ],
                    "fuse_qkv",
                    float_type,
                )?;
                map_fused_bytes_to_arena(arena, &format!("metal_fused_qkv_{layer_idx}"), &bytes)?
            };

            let gate_up = if plan
                .tensors
                .contains_key(&layer_names.prefused_gate_up_name)
            {
                region_lookup(&mut mapped_refs, &layer_names.prefused_gate_up_name)?
            } else {
                let bytes = concat_f16_tensors(
                    &plan.tensors,
                    &[layer_names.gate_name.clone(), layer_names.up_name.clone()],
                    &[vec![intermediate, hidden], vec![intermediate, hidden]],
                    "fuse_gate_up",
                    float_type,
                )?;
                map_fused_bytes_to_arena(
                    arena,
                    &format!("metal_fused_gate_up_{layer_idx}"),
                    &bytes,
                )?
            };

            let moe = layer_names
                .moe
                .as_ref()
                .map(|moe_names| -> Result<MetalMoeState> {
                    Ok(MetalMoeState {
                        num_experts: moe_names.num_experts,
                        top_k: moe_names.top_k,
                        intermediate_size: moe_names.intermediate_size,
                        router_proj: map_router_tensor_to_arena(
                            arena,
                            &plan.tensors,
                            &moe_names.router_proj_name,
                            &format!("metal_moe_router_proj_f32_{layer_idx}"),
                            float_type,
                        )?,
                        router_scale: map_router_tensor_to_arena(
                            arena,
                            &plan.tensors,
                            &moe_names.router_scale_name,
                            &format!("metal_moe_router_scale_f32_{layer_idx}"),
                            float_type,
                        )?,
                        router_per_expert_scale: map_router_tensor_to_arena(
                            arena,
                            &plan.tensors,
                            &moe_names.router_per_expert_scale_name,
                            &format!("metal_moe_router_per_expert_scale_f32_{layer_idx}"),
                            float_type,
                        )?,
                        pre_ff2_norm: region_lookup(
                            &mut mapped_refs,
                            &moe_names.pre_ff2_norm_name,
                        )?,
                        post_ff1_norm: region_lookup(
                            &mut mapped_refs,
                            &moe_names.post_ff1_norm_name,
                        )?,
                        post_ff2_norm: region_lookup(
                            &mut mapped_refs,
                            &moe_names.post_ff2_norm_name,
                        )?,
                        expert_gate_up: region_lookup(
                            &mut mapped_refs,
                            &moe_names.expert_gate_up_name,
                        )?,
                        expert_down: region_lookup(&mut mapped_refs, &moe_names.expert_down_name)?,
                    })
                })
                .transpose()?;

            let scratch = shared_scratch.as_ref().ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "missing shared layer scratch",
                    },
                    probe_ctx("prepare"),
                )
            })?;
            let qkv_out = scratch.qkv_out.clone();
            let q = scratch.q.clone();
            let k = scratch.k.clone();
            let v = scratch.v.clone();
            let attn_out = scratch.attn_out.clone();
            let global_decode_partials = scratch.global_decode_partials.clone();
            let gate_up_out = scratch.gate_up_out.clone();
            let activated = scratch.activated.clone();
            let mlp_out = scratch.mlp_out.clone();
            let moe_topk_indices = scratch.moe_topk_indices.clone();
            let moe_topk_weights = scratch.moe_topk_weights.clone();
            let moe_activated = scratch.moe_activated.clone();
            let moe_out = scratch.moe_out.clone();
            let trace = if debug_trace_layer_enabled(&debug_trace_layers, layer_idx) {
                let trace_region = |arena: &mut MetalBufferArena,
                                    name: &str,
                                    elems_per_token: usize|
                 -> Result<MetalRegion> {
                    arena.region(
                        &format!("metal_layer_{layer_idx}_trace_{name}"),
                        max_batch_tokens * elems_per_token * half_bytes,
                        16,
                    )
                };
                let ple_dim = plan.ple_names.as_ref().map_or(0, |ple| ple.ple_dim);
                Some(MetalLayerTraceState {
                    input_to_layer: trace_region(arena, "input_to_layer", hidden)?,
                    after_input_layernorm: trace_region(arena, "after_input_layernorm", hidden)?,
                    q_projection: trace_region(arena, "q_projection", q_dim)?,
                    k_projection: trace_region(arena, "k_projection", kv_dim)?,
                    v_projection: trace_region(arena, "v_projection", kv_dim)?,
                    after_q_norm: trace_region(arena, "after_q_norm", q_dim)?,
                    after_k_norm: trace_region(arena, "after_k_norm", kv_dim)?,
                    after_v_norm: trace_region(arena, "after_v_norm", kv_dim)?,
                    after_rope_q: trace_region(arena, "after_rope_q", q_dim)?,
                    after_rope_k: trace_region(arena, "after_rope_k", kv_dim)?,
                    attention_output: trace_region(arena, "attention_output", q_dim)?,
                    after_o_proj: trace_region(arena, "after_o_proj", hidden)?,
                    after_post_attention_layernorm: trace_region(
                        arena,
                        "after_post_attention_layernorm",
                        hidden,
                    )?,
                    after_pre_feedforward_layernorm: trace_region(
                        arena,
                        "after_pre_feedforward_layernorm",
                        hidden,
                    )?,
                    gate_up_out: trace_region(arena, "gate_up_out", 2 * intermediate)?,
                    ffn_activation: trace_region(arena, "ffn_activation", intermediate)?,
                    after_ffn_branch: trace_region(arena, "after_ffn_branch", hidden)?,
                    after_post_feedforward_layernorm: trace_region(
                        arena,
                        "after_post_feedforward_layernorm",
                        hidden,
                    )?,
                    per_layer_input: (ple_dim > 0)
                        .then(|| trace_region(arena, "per_layer_input", ple_dim))
                        .transpose()?,
                    per_layer_input_gate: (ple_dim > 0)
                        .then(|| trace_region(arena, "per_layer_input_gate", ple_dim))
                        .transpose()?,
                    per_layer_projection: (ple_dim > 0)
                        .then(|| trace_region(arena, "per_layer_projection", hidden))
                        .transpose()?,
                    post_per_layer_input_norm: (ple_dim > 0)
                        .then(|| trace_region(arena, "post_per_layer_input_norm", hidden))
                        .transpose()?,
                })
            } else {
                None
            };

            let block_size = APPLE_KV_PAGE_TOKENS as u32;
            let max_blocks_per_seq = max_probe_tokens.div_ceil(APPLE_KV_PAGE_TOKENS) as u32;
            let num_blocks_total = plan.physical_kv_pages;
            let kv_cache_k = arena.region(
                &format!("metal_layer_{layer_idx}_kv_cache_k"),
                (num_blocks_total as usize) * (block_size as usize) * kv_dim * half_bytes,
                16,
            )?;
            let kv_cache_v = arena.region(
                &format!("metal_layer_{layer_idx}_kv_cache_v"),
                (num_blocks_total as usize) * (block_size as usize) * kv_dim * half_bytes,
                16,
            )?;

            let positions = arena.region(
                &format!("metal_layer_{layer_idx}_positions"),
                max_batch_tokens * 4,
                4,
            )?;
            let slot_mapping = arena.region(
                &format!("metal_layer_{layer_idx}_slot_mapping"),
                max_batch_tokens * 4,
                4,
            )?;
            let context_lens = arena.region(
                &format!("metal_layer_{layer_idx}_context_lens"),
                max_batch_sequences * 4,
                4,
            )?;
            let block_tables = arena.region(
                &format!("metal_layer_{layer_idx}_block_tables"),
                max_batch_sequences * (max_blocks_per_seq as usize) * 4,
                4,
            )?;
            let cu_seqlens = arena.region(
                &format!("metal_layer_{layer_idx}_cu_seqlens"),
                (max_batch_sequences + 1) * 4,
                4,
            )?;

            let half_rope = dims.rope_dim / 2;
            let max_pos = max_probe_tokens;
            let cos = arena.region(
                &format!("metal_layer_{layer_idx}_rope_cos"),
                max_pos * half_rope * f32_bytes,
                16,
            )?;
            let sin = arena.region(
                &format!("metal_layer_{layer_idx}_rope_sin"),
                max_pos * half_rope * f32_bytes,
                16,
            )?;

            write_i32_region(arena, &positions, &vec![0; max_batch_tokens])?;
            write_i32_region(arena, &slot_mapping, &vec![0; max_batch_tokens])?;
            write_i32_region(arena, &context_lens, &vec![0; max_batch_sequences])?;
            write_i32_region(
                arena,
                &block_tables,
                &vec![0; max_batch_sequences * max_blocks_per_seq as usize],
            )?;
            write_i32_region(arena, &cu_seqlens, &vec![0; max_batch_sequences + 1])?;
            let (cos_table, sin_table) =
                build_rope_tables(max_pos, half_rope, dims.head_dim, dims.rope_theta);
            write_f32_region(arena, &cos, &cos_table)?;
            write_f32_region(arena, &sin, &sin_table)?;

            layers.push(MetalOneLayerState {
                layer_idx,
                down_proj_name: layer_names.down_proj_name.clone(),
                low_bit_q_proj: None,
                low_bit_k_proj: None,
                low_bit_v_proj: None,
                low_bit_o_proj: None,
                low_bit_gate_proj: None,
                low_bit_up_proj: None,
                low_bit_down_proj: None,
                dims,
                shared_kv_source_layer: shared_kv_source_layers[layer_idx],
                attn_norm,
                qkv,
                q_norm,
                k_norm,
                v_norm,
                o_proj,
                mlp_norm,
                post_attn_norm,
                pre_ff_norm,
                post_ff_norm,
                layer_scalar,
                layer_scalar_dim: layer_names.layer_scalar_dim as u32,
                gate_up,
                down_proj,
                moe,
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
                qkv_out,
                q,
                k,
                v,
                attn_out,
                global_decode_partials,
                gate_up_out,
                activated,
                mlp_out,
                moe_topk_indices,
                moe_topk_weights,
                moe_activated,
                moe_out,
                trace,
                positions,
                slot_mapping,
                cos,
                sin,
                block_tables,
                context_lens,
                cu_seqlens,
                kv_cache_k,
                kv_cache_v,
                block_size,
                max_blocks_per_seq,
                num_blocks_total,
            });
        }

        if !mapped_refs.is_empty() {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "unexpected mapped tensor entries",
                },
                probe_ctx("prepare"),
            ));
        }

        let residual = arena.region("metal_model_residual", residual_bytes, 16)?;
        let logits = arena.region("metal_model_logits", logits_bytes, 16)?;
        let normed_hidden = arena.region("metal_model_normed_hidden", normed_hidden_bytes, 16)?;
        let sampled = arena.region("metal_model_sampled", sampled_bytes, 4)?;
        let final_argmax_partial_max = arena.region(
            "metal_model_final_argmax_partial_max",
            final_argmax_partial_max_bytes,
            16,
        )?;
        let final_argmax_partial_idx = arena.region(
            "metal_model_final_argmax_partial_idx",
            final_argmax_partial_idx_bytes,
            16,
        )?;
        let token_ids = arena.region("metal_model_token_ids", token_ids_bytes, 4)?;

        let mut execution_slots = Vec::with_capacity(crate::memory_budget::IN_FLIGHT_SCRATCH_SLOTS);
        execution_slots.push(Gemma4MetalExecutionSlot {
            residual: residual.clone(),
            logits: logits.clone(),
            normed_hidden: normed_hidden.clone(),
            sampled: sampled.clone(),
            final_argmax_partial_max: final_argmax_partial_max.clone(),
            final_argmax_partial_idx: final_argmax_partial_idx.clone(),
            token_ids: token_ids.clone(),
            ple_token_inputs: ple.as_ref().map(|ple| ple.token_inputs.clone()),
            ple_context_inputs: ple.as_ref().map(|ple| ple.context_inputs.clone()),
            layers: layers.iter().map(execution_layer_from_state).collect(),
        });

        // Allocate every mutable launch region twice more. Layer weights,
        // immutable RoPE tables, and physical KV pages intentionally remain
        // shared. Shared layer scratch remains shared *within* one command
        // buffer because layer encoders execute in order, but never across
        // execution slots.
        for slot_index in 1..crate::memory_budget::IN_FLIGHT_SCRATCH_SLOTS {
            let base_layer = layers.first();
            let shared = base_layer
                .map(|layer| -> Result<MetalLayerExecutionSlot> {
                    let name = |field: &str| format!("metal_slot_{slot_index}_shared_{field}");
                    Ok(MetalLayerExecutionSlot {
                        qkv_out: allocate_region_like(arena, &name("qkv_out"), &layer.qkv_out, 16)?,
                        q: allocate_region_like(arena, &name("q"), &layer.q, 16)?,
                        k: allocate_region_like(arena, &name("k"), &layer.k, 16)?,
                        v: allocate_region_like(arena, &name("v"), &layer.v, 16)?,
                        attn_out: allocate_region_like(
                            arena,
                            &name("attn_out"),
                            &layer.attn_out,
                            16,
                        )?,
                        global_decode_partials: allocate_optional_region_like(
                            arena,
                            &name("global_decode_partials"),
                            layer.global_decode_partials.as_ref(),
                            16,
                        )?,
                        gate_up_out: allocate_region_like(
                            arena,
                            &name("gate_up_out"),
                            &layer.gate_up_out,
                            16,
                        )?,
                        activated: allocate_region_like(
                            arena,
                            &name("activated"),
                            &layer.activated,
                            16,
                        )?,
                        mlp_out: allocate_region_like(arena, &name("mlp_out"), &layer.mlp_out, 16)?,
                        moe_topk_indices: allocate_optional_region_like(
                            arena,
                            &name("moe_topk_indices"),
                            layer.moe_topk_indices.as_ref(),
                            4,
                        )?,
                        moe_topk_weights: allocate_optional_region_like(
                            arena,
                            &name("moe_topk_weights"),
                            layer.moe_topk_weights.as_ref(),
                            4,
                        )?,
                        moe_activated: allocate_optional_region_like(
                            arena,
                            &name("moe_activated"),
                            layer.moe_activated.as_ref(),
                            16,
                        )?,
                        moe_out: allocate_optional_region_like(
                            arena,
                            &name("moe_out"),
                            layer.moe_out.as_ref(),
                            16,
                        )?,
                        // Per-layer fields are replaced below.
                        trace: None,
                        positions: layer.positions.clone(),
                        slot_mapping: layer.slot_mapping.clone(),
                        block_tables: layer.block_tables.clone(),
                        context_lens: layer.context_lens.clone(),
                        cu_seqlens: layer.cu_seqlens.clone(),
                    })
                })
                .transpose()?;

            let mut slot_layers = Vec::with_capacity(layers.len());
            for layer in &layers {
                let mut execution = shared.clone().ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "missing layer scratch for Metal execution slot",
                        },
                        probe_ctx("prepare"),
                    )
                })?;
                let layer_idx = layer.layer_idx;
                let name =
                    |field: &str| format!("metal_slot_{slot_index}_layer_{layer_idx}_{field}");
                execution.trace =
                    allocate_trace_like(arena, slot_index, layer_idx, layer.trace.as_ref())?;
                execution.positions =
                    allocate_region_like(arena, &name("positions"), &layer.positions, 4)?;
                execution.slot_mapping =
                    allocate_region_like(arena, &name("slot_mapping"), &layer.slot_mapping, 4)?;
                execution.block_tables =
                    allocate_region_like(arena, &name("block_tables"), &layer.block_tables, 4)?;
                execution.context_lens =
                    allocate_region_like(arena, &name("context_lens"), &layer.context_lens, 4)?;
                execution.cu_seqlens =
                    allocate_region_like(arena, &name("cu_seqlens"), &layer.cu_seqlens, 4)?;
                slot_layers.push(execution);
            }

            let slot_name = |field: &str| format!("metal_slot_{slot_index}_{field}");
            execution_slots.push(Gemma4MetalExecutionSlot {
                residual: allocate_region_like(arena, &slot_name("residual"), &residual, 16)?,
                logits: allocate_region_like(arena, &slot_name("logits"), &logits, 16)?,
                normed_hidden: allocate_region_like(
                    arena,
                    &slot_name("normed_hidden"),
                    &normed_hidden,
                    16,
                )?,
                sampled: allocate_region_like(arena, &slot_name("sampled"), &sampled, 4)?,
                final_argmax_partial_max: allocate_region_like(
                    arena,
                    &slot_name("final_argmax_partial_max"),
                    &final_argmax_partial_max,
                    16,
                )?,
                final_argmax_partial_idx: allocate_region_like(
                    arena,
                    &slot_name("final_argmax_partial_idx"),
                    &final_argmax_partial_idx,
                    16,
                )?,
                token_ids: allocate_region_like(arena, &slot_name("token_ids"), &token_ids, 4)?,
                ple_token_inputs: ple
                    .as_ref()
                    .map(|ple| {
                        allocate_region_like(
                            arena,
                            &slot_name("ple_token_inputs"),
                            &ple.token_inputs,
                            16,
                        )
                    })
                    .transpose()?,
                ple_context_inputs: ple
                    .as_ref()
                    .map(|ple| {
                        allocate_region_like(
                            arena,
                            &slot_name("ple_context_inputs"),
                            &ple.context_inputs,
                            16,
                        )
                    })
                    .transpose()?,
                layers: slot_layers,
            });
        }

        Ok(Self {
            float_type,
            hidden_size: plan.arch.hidden_size,
            vocab_size: plan.arch.vocab_size,
            num_layers: plan.arch.num_hidden_layers,
            rms_norm_eps: plan.arch.rms_norm_eps,
            final_logit_softcap: plan
                .arch
                .final_logit_softcapping
                .unwrap_or(PROBE_METAL_SOFTCAP),
            embedding_scale: (plan.arch.hidden_size as f32).sqrt(),
            max_probe_tokens,
            max_batch_tokens,
            max_batch_sequences,
            memory_budget: plan.memory_budget.ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "missing Metal model memory budget",
                    },
                    probe_ctx("memory_budget"),
                )
            })?,
            embedding,
            final_norm,
            lm_head,
            residual,
            logits,
            normed_hidden,
            sampled,
            final_argmax_partial_max,
            final_argmax_partial_idx,
            token_ids,
            ple,
            layers,
            execution_slots,
        })
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn resolve_tensor_alias(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    candidates: Vec<String>,
    missing_reason: &'static str,
) -> Result<String> {
    candidates
        .into_iter()
        .find(|name| tensors.contains_key(name))
        .ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: missing_reason,
                },
                probe_ctx("prepare"),
            )
        })
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn resolve_optional_tensor_alias(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    candidates: Vec<String>,
) -> Option<String> {
    candidates
        .into_iter()
        .find(|name| tensors.contains_key(name))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn validate_tensor_shape(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    name: &str,
    expected: &[usize],
    reason: &'static str,
) -> Result<()> {
    let info = tensors.get(name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing layer weight",
            },
            probe_ctx("prepare"),
        )
    })?;
    if info.shape.as_slice() != expected {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob { reason },
            probe_ctx("prepare"),
        ));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn validate_optional_norm_shape(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    name: &Option<String>,
    expected_dim: usize,
    reason: &'static str,
) -> Result<()> {
    let Some(name) = name else {
        return Ok(());
    };
    let info = tensors.get(name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing optional norm tensor",
            },
            probe_ctx("prepare"),
        )
    })?;
    if info.shape.len() != 1 || info.shape[0] != expected_dim {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob { reason },
            probe_ctx("prepare"),
        ));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn validate_optional_layer_scalar_shape(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    name: &Option<String>,
    hidden: usize,
) -> Result<usize> {
    let Some(name) = name else {
        return Ok(0);
    };
    let info = tensors.get(name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing layer_scalar tensor",
            },
            probe_ctx("prepare"),
        )
    })?;
    if info.shape.len() == 1 && (info.shape[0] == 1 || info.shape[0] == hidden) {
        Ok(info.shape[0])
    } else {
        Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "layer_scalar shape mismatch",
            },
            probe_ctx("prepare"),
        ))
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn resolve_weight_prefix(tensors: &BTreeMap<String, SafetensorTensorInfo>) -> String {
    if tensors.contains_key("model.embed_tokens.weight") {
        "model".to_owned()
    } else if tensors.contains_key("model.language_model.embed_tokens.weight") {
        "model.language_model".to_owned()
    } else if tensors.contains_key("language_model.model.embed_tokens.weight") {
        "language_model.model".to_owned()
    } else {
        "model".to_owned()
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn region_lookup(refs: &mut Vec<(String, MetalRegion)>, name: &str) -> Result<MetalRegion> {
    let idx = refs.iter().position(|(n, _)| n == name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing mapped tensor in map",
            },
            probe_ctx("resolve_regions"),
        )
    })?;
    Ok(refs.swap_remove(idx).1)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn optional_region_lookup(
    refs: &mut Vec<(String, MetalRegion)>,
    name: Option<&str>,
) -> Result<Option<MetalRegion>> {
    match name {
        Some(name) => Ok(Some(region_lookup(refs, name)?)),
        None => Ok(None),
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn optional_distinct_region_lookup(
    refs: &mut Vec<(String, MetalRegion)>,
    name: Option<&str>,
    alias_name: &str,
) -> Result<Option<MetalRegion>> {
    match name {
        Some(name) if name != alias_name => Ok(Some(region_lookup(refs, name)?)),
        _ => Ok(None),
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn optional_region_or_alias_lookup(
    refs: &mut Vec<(String, MetalRegion)>,
    name: Option<&str>,
    alias_name: &str,
    alias_region: &MetalRegion,
) -> Result<Option<MetalRegion>> {
    match name {
        Some(name) if name == alias_name => Ok(Some(alias_region.clone())),
        Some(name) => Ok(Some(region_lookup(refs, name)?)),
        None => Ok(None),
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn map_fused_bytes_to_arena(
    arena: &mut MetalBufferArena,
    name: &str,
    bytes: &[u8],
) -> Result<MetalRegion> {
    let region = arena.region(name, bytes.len(), 16)?;
    unsafe {
        arena.write_region(&region, bytes)?;
    }
    Ok(region)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn map_f32_tensor_to_arena(
    arena: &mut MetalBufferArena,
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    tensor_name: &str,
    region_name: &str,
) -> Result<MetalRegion> {
    let info = tensors.get(tensor_name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing f32 tensor",
            },
            probe_ctx("prepare"),
        )
    })?;
    let bytes = load_safetensor_entry_f32(info)?;
    map_fused_bytes_to_arena(arena, region_name, &bytes)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn map_router_tensor_to_arena(
    arena: &mut MetalBufferArena,
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    tensor_name: &str,
    region_name: &str,
    float_type: MetalFloatType,
) -> Result<MetalRegion> {
    match float_type {
        MetalFloatType::F16 => map_f32_tensor_to_arena(arena, tensors, tensor_name, region_name),
        MetalFloatType::Bf16 => {
            let info = tensors.get(tensor_name).ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "missing router tensor",
                    },
                    probe_ctx("prepare"),
                )
            })?;
            let bytes = load_safetensor_entry_for_float_type(info, float_type)?;
            map_fused_bytes_to_arena(arena, region_name, &bytes)
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn concat_f16_tensors(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    names: &[String],
    expected_shapes: &[Vec<usize>],
    op: &'static str,
    float_type: MetalFloatType,
) -> Result<Vec<u8>> {
    if names.len() != expected_shapes.len() {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "concat shape metadata mismatch",
            },
            probe_ctx(op),
        ));
    }

    let mut out = Vec::new();
    for (name, expected_shape) in names.iter().zip(expected_shapes.iter()) {
        let info = tensors.get(name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing tensor for concat",
                },
                probe_ctx(op),
            )
        })?;
        if info.shape != *expected_shape {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "tensor shape mismatch for concat",
                },
                probe_ctx(op),
            ));
        }
        let bytes = load_safetensor_entry_for_float_type(info, float_type)?;
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_i32_region(arena: &MetalBufferArena, region: &MetalRegion, values: &[i32]) -> Result<()> {
    unsafe {
        let dst = arena.host_ptr(region) as *mut i32;
        ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn write_f32_region(arena: &MetalBufferArena, region: &MetalRegion, values: &[f32]) -> Result<()> {
    unsafe {
        let dst = arena.host_ptr(region) as *mut f32;
        ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn build_rope_tables(
    max_positions: usize,
    half_rope: usize,
    head_dim: usize,
    rope_theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0f32; max_positions * half_rope];
    let mut sin = vec![0.0f32; max_positions * half_rope];
    for pos in 0..max_positions {
        for pair in 0..half_rope {
            let exponent = (2 * pair) as f32 / head_dim as f32;
            let inv_freq = 1.0 / rope_theta.powf(exponent);
            let angle = pos as f32 * inv_freq;
            let idx = pos * half_rope + pair;
            cos[idx] = angle.cos();
            sin[idx] = angle.sin();
        }
    }
    (cos, sin)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use serde_json::{Map, Value};
    use std::{
        fs::{self, File},
        io::Write,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn shared_kv_test_arch(layer_types: Vec<LayerAttnType>, shared_layers: usize) -> ModelArch {
        let num_hidden_layers = layer_types.len();
        ModelArch {
            num_hidden_layers,
            hidden_size: 1536,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 192,
            intermediate_size: 4096,
            use_double_wide_mlp: true,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            num_kv_shared_layers: shared_layers,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            vocab_size: 262_144,
            rope_theta: 10_000.0,
            max_position_embeddings: 4096,
            attention_bias: false,
            rms_norm_eps: 1e-6,
            layer_types,
            global_head_dim: Some(256),
            num_global_key_value_heads: Some(4),
            global_rope_theta: Some(1_000_000.0),
            partial_rotary_factor: Some(0.25),
            sliding_window: Some(1024),
            final_logit_softcapping: Some(30.0),
            hidden_activation: Some("gelu_pytorch_tanh".to_string()),
            tie_word_embeddings: true,
            attention_k_eq_v: false,
        }
    }

    #[test]
    fn shared_kv_source_layers_maps_tail_to_last_unshared_same_attention_kind() {
        let arch = shared_kv_test_arch(
            vec![
                LayerAttnType::SlidingAttention,
                LayerAttnType::Full,
                LayerAttnType::SlidingAttention,
                LayerAttnType::Full,
                LayerAttnType::SlidingAttention,
                LayerAttnType::Full,
            ],
            2,
        );

        let sources = arch.shared_kv_source_layers();

        assert_eq!(sources, vec![None, None, None, None, Some(2), Some(3)]);
    }

    #[test]
    fn layer_dims_require_explicit_window_only_for_sliding_attention() {
        let mut sliding = shared_kv_test_arch(vec![LayerAttnType::SlidingAttention], 0);
        sliding.sliding_window = None;
        assert!(MetalProbeLayerDims::from_arch_layer(&sliding, 0).is_err());
        sliding.sliding_window = Some(u32::MAX as usize + 1);
        assert!(MetalProbeLayerDims::from_arch_layer(&sliding, 0).is_err());

        let mut full = shared_kv_test_arch(vec![LayerAttnType::Full], 0);
        full.sliding_window = None;
        let dims = MetalProbeLayerDims::from_arch_layer(&full, 0)
            .expect("full attention must not require a sliding window");
        assert_eq!(dims.attention_window, 0);
    }

    #[test]
    fn shared_kv_source_layers_leave_tail_without_matching_source_unmapped() {
        let arch = shared_kv_test_arch(
            vec![
                LayerAttnType::SlidingAttention,
                LayerAttnType::SlidingAttention,
                LayerAttnType::Full,
            ],
            1,
        );

        let sources = arch.shared_kv_source_layers();

        assert_eq!(sources, vec![None, None, None]);
    }

    fn test_fixture_dir(name: &str) -> PathBuf {
        let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rvllm-metal-{name}-{}-{}-{id}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create fixture dir");
        dir
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect()
    }

    fn add_tensor(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        data: &[f32],
        shape: &[usize],
    ) {
        let start = payload.len();
        payload.extend_from_slice(&f16_bytes(data));
        let end = payload.len();
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
        meta.insert(
            "shape".to_owned(),
            Value::Array(
                shape
                    .iter()
                    .map(|n| Value::Number((*n as u64).into()))
                    .collect(),
            ),
        );
        meta.insert(
            "data_offsets".to_owned(),
            Value::Array(vec![
                Value::Number((start as u64).into()),
                Value::Number((end as u64).into()),
            ]),
        );
        header.insert(name.to_string(), Value::Object(meta));
    }

    fn write_two_layer_sliding_global_plan_fixture() -> PathBuf {
        let dir = test_fixture_dir("sliding-global-dims");
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let sliding_head_dim = 128usize;
        let global_head_dim = 256usize;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();
        let zeros_embed = vec![0.0f32; vocab * hidden];
        let ones_hidden = vec![1.0f32; hidden];
        let zeros_lm_head = vec![0.0f32; vocab * hidden];
        let zeros_gate_up = vec![0.0f32; 2 * intermediate * hidden];
        let zeros_down = vec![0.0f32; hidden * intermediate];

        add_tensor(
            &mut header,
            &mut payload,
            "model.embed_tokens.weight",
            &zeros_embed,
            &[vocab, hidden],
        );
        add_tensor(
            &mut header,
            &mut payload,
            "model.norm.weight",
            &ones_hidden,
            &[hidden],
        );
        add_tensor(
            &mut header,
            &mut payload,
            "lm_head.weight",
            &zeros_lm_head,
            &[vocab, hidden],
        );

        for (layer_idx, head_dim) in [(0usize, sliding_head_dim), (1usize, global_head_dim)] {
            let qkv_rows = 3 * head_dim;
            let zeros_qkv = vec![0.0f32; qkv_rows * hidden];
            let zeros_o = vec![0.0f32; hidden * head_dim];
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &ones_hidden,
                &[hidden],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.self_attn.qkv.weight"),
                &zeros_qkv,
                &[qkv_rows, hidden],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &zeros_o,
                &[hidden, head_dim],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.mlp_norm.weight"),
                &ones_hidden,
                &[hidden],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.mlp.gate_up.weight"),
                &zeros_gate_up,
                &[2 * intermediate, hidden],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &zeros_down,
                &[hidden, intermediate],
            );
        }

        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 2,
    "hidden_size": {hidden},
    "intermediate_size": {intermediate},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {sliding_head_dim},
    "global_head_dim": {global_head_dim},
    "num_global_key_value_heads": 1,
    "layer_types": ["sliding_attention", "full_attention"],
    "sliding_window": 32,
    "vocab_size": {vocab},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false,
    "rope_parameters": {{
      "sliding_attention": {{"rope_theta": 10000.0}},
      "full_attention": {{"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}}
    }}
  }}
}}"#
            ),
        )
        .expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out = File::create(dir.join("model.safetensors")).expect("create safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[test]
    fn explicit_limits_keep_independent_owners_and_reject_excess_before_tensor_scan() {
        let dir = write_two_layer_sliding_global_plan_fixture();
        let small = crate::MetalModelLimits {
            max_context_tokens: 8,
            max_batch_tokens: 4,
            max_batch_sequences: 1,
        };
        let large = crate::MetalModelLimits {
            max_context_tokens: 16,
            max_batch_tokens: 16,
            max_batch_sequences: 2,
        };
        let a = ProbeModelPlan::with_limits(&dir, Some(small)).unwrap();
        let b = ProbeModelPlan::with_limits(&dir, Some(large)).unwrap();
        let again = ProbeModelPlan::with_limits(&dir, Some(small)).unwrap();
        assert_eq!(
            (
                a.max_probe_tokens,
                a.max_batch_tokens,
                a.max_batch_sequences
            ),
            (8, 4, 1)
        );
        assert_eq!(
            (
                b.max_probe_tokens,
                b.max_batch_tokens,
                b.max_batch_sequences
            ),
            (16, 16, 2)
        );
        assert_eq!(a.arena_bytes, again.arena_bytes);
        assert!(b.scratch_slot_bytes > a.scratch_slot_bytes);
        assert_eq!(a.physical_kv_pages, 1);
        assert_eq!(b.physical_kv_pages, 2);
        assert!(a.explicit_batch_sequences && a.debug_trace_layers.is_empty());
        // A bad limit must be rejected even when tensor scanning would fail.
        fs::remove_file(dir.join("model.safetensors")).unwrap();
        let err = ProbeModelPlan::with_limits(
            &dir,
            Some(crate::MetalModelLimits {
                max_context_tokens: 17,
                ..large
            }),
        )
        .err()
        .unwrap();
        assert!(err
            .to_string()
            .contains("invalid explicit Metal model limits"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn probe_model_plan_selects_sliding_and_global_layer_dims() {
        let dir = write_two_layer_sliding_global_plan_fixture();
        let plan = ProbeModelPlan::new(&dir).expect("build probe model plan");

        assert_eq!(plan.layer_names.len(), 2);
        let sliding = plan.layer_names[0].dims;
        assert_eq!(
            sliding.attention_kind,
            MetalProbeLayerAttentionKind::Sliding
        );
        assert_eq!(sliding.attention_window, 32);
        assert_eq!(sliding.num_heads, 1);
        assert_eq!(sliding.num_kv_heads, 1);
        assert_eq!(sliding.head_dim, 128);
        assert_eq!(sliding.rope_dim, 128);
        assert_eq!(sliding.rope_theta, 10000.0);
        assert_eq!(sliding.q_dim, 128);
        assert_eq!(sliding.kv_dim, 128);
        assert_eq!(sliding.qkv_rows, 384);

        let full = plan.layer_names[1].dims;
        assert_eq!(full.attention_kind, MetalProbeLayerAttentionKind::Full);
        assert_eq!(full.attention_window, 0);
        assert_eq!(full.num_heads, 1);
        assert_eq!(full.num_kv_heads, 1);
        assert_eq!(full.head_dim, 256);
        assert_eq!(full.rope_dim, 64);
        assert_eq!(full.rope_theta, 1000000.0);
        assert_eq!(full.q_dim, 256);
        assert_eq!(full.kv_dim, 256);
        assert_eq!(full.qkv_rows, 768);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn additional_package_weights_are_accounted_before_device_budgeting() {
        let dir = write_two_layer_sliding_global_plan_fixture();
        let base = ProbeModelPlan::new(&dir).expect("build base probe model plan");
        let base_weights = base.weights_bytes;
        let base_arena = base.arena_bytes;
        let additional = 12_345;
        let with_sidecar = base
            .with_additional_weight_bytes(additional)
            .expect("account package sidecar");
        assert_eq!(with_sidecar.weights_bytes, base_weights + additional);
        assert_eq!(with_sidecar.arena_bytes, base_arena + additional);

        let overflow = ProbeModelPlan::new(&dir)
            .expect("rebuild probe model plan")
            .with_additional_weight_bytes(usize::MAX);
        assert!(overflow.is_err());
        let _ = fs::remove_dir_all(dir);
    }

    fn low_bit_replacement(
        tensor_name: impl Into<String>,
        format: rvllm_apple::AppleLowBitWeightFormat,
        shape: [usize; 2],
    ) -> MetalLowBitWeightReplacement {
        MetalLowBitWeightReplacement {
            tensor_name: tensor_name.into(),
            role: AppleLowBitTensorRole::DenseDownProjection,
            format,
            shape,
            packed_values_bytes: low_bit_packed_values_bytes(format, shape[0], shape[1])
                .expect("packed bytes"),
            scales_bytes: low_bit_scales_bytes(shape[0], shape[1]).expect("scale bytes"),
        }
    }

    #[test]
    fn low_bit_replacement_plan_omits_native_weights_and_accounts_exact_storage() {
        let dir = write_two_layer_sliding_global_plan_fixture();
        let base = ProbeModelPlan::new(&dir).expect("build base plan");
        let native_names = base
            .layer_names
            .iter()
            .map(|layer| layer.down_proj_name.clone())
            .collect::<Vec<_>>();
        let native_bytes = native_names
            .iter()
            .map(|name| base.tensors.get(name).expect("native tensor").nbytes)
            .sum::<usize>();
        let base_weights_bytes = base.weights_bytes;
        let base_unfloored_arena_bytes = base.unfloored_arena_bytes;
        let replacements = vec![
            low_bit_replacement(
                native_names[1].clone(),
                rvllm_apple::AppleLowBitWeightFormat::W8A16,
                [128, 256],
            ),
            low_bit_replacement(
                native_names[0].clone(),
                rvllm_apple::AppleLowBitWeightFormat::W4A16,
                [128, 256],
            ),
        ];
        let mut expected_low_bit_arena_bytes = 0usize;
        for replacement in replacements.iter().rev() {
            expected_low_bit_arena_bytes =
                align_up_checked(expected_low_bit_arena_bytes, 16).expect("align values");
            expected_low_bit_arena_bytes += replacement.packed_values_bytes;
            expected_low_bit_arena_bytes =
                align_up_checked(expected_low_bit_arena_bytes, 16).expect("align scales");
            expected_low_bit_arena_bytes += replacement.scales_bytes;
        }

        let planned = base
            .with_low_bit_replacements(&replacements)
            .expect("replace two dense projections");
        assert_eq!(planned.low_bit_replacements.len(), 2);
        assert!(native_names
            .iter()
            .all(|name| !planned.names.contains(name)));
        assert_eq!(
            planned.weights_bytes,
            base_weights_bytes - native_bytes + expected_low_bit_arena_bytes
        );
        assert_eq!(
            planned.unfloored_arena_bytes,
            base_unfloored_arena_bytes - native_bytes + expected_low_bit_arena_bytes
        );
        assert_eq!(
            planned.arena_bytes,
            max(planned.unfloored_arena_bytes, PROBE_METAL_ARENA_BYTES)
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn low_bit_replacement_plan_rejects_duplicate_missing_shape_and_moe_inputs() {
        let dir = write_two_layer_sliding_global_plan_fixture();
        let plan = ProbeModelPlan::new(&dir).expect("build base plan");
        let name = plan.layer_names[0].down_proj_name.clone();
        let valid = low_bit_replacement(
            name.clone(),
            rvllm_apple::AppleLowBitWeightFormat::W4A16,
            [128, 256],
        );

        assert!(ProbeModelPlan::new(&dir)
            .expect("duplicate plan")
            .with_low_bit_replacements(&[valid.clone(), valid.clone()])
            .is_err());
        let mut unwired_role = valid.clone();
        unwired_role.role = AppleLowBitTensorRole::QueryProjection;
        assert!(ProbeModelPlan::new(&dir)
            .expect("unwired role plan")
            .with_low_bit_replacements(&[unwired_role])
            .is_err());
        assert!(ProbeModelPlan::new(&dir)
            .expect("missing plan")
            .with_low_bit_replacements(&[low_bit_replacement(
                "model.layers.99.mlp.down_proj.weight",
                rvllm_apple::AppleLowBitWeightFormat::W4A16,
                [128, 256],
            )])
            .is_err());

        let mut bad_shape = valid.clone();
        bad_shape.shape = [128, 255];
        bad_shape.packed_values_bytes =
            low_bit_packed_values_bytes(bad_shape.format, 128, 255).expect("bad packed bytes");
        bad_shape.scales_bytes = low_bit_scales_bytes(128, 255).expect("bad scale bytes");
        assert!(ProbeModelPlan::new(&dir)
            .expect("shape plan")
            .with_low_bit_replacements(&[bad_shape])
            .is_err());

        let mut bad_payload = valid.clone();
        bad_payload.packed_values_bytes -= 1;
        assert!(ProbeModelPlan::new(&dir)
            .expect("payload plan")
            .with_low_bit_replacements(&[bad_payload])
            .is_err());

        let mut moe_plan = ProbeModelPlan::new(&dir).expect("moe plan");
        moe_plan.layer_names[0].moe = Some(ProbeMoeNames {
            num_experts: 2,
            top_k: 1,
            intermediate_size: 64,
            router_proj_name: String::new(),
            router_scale_name: String::new(),
            router_per_expert_scale_name: String::new(),
            pre_ff2_norm_name: String::new(),
            post_ff1_norm_name: String::new(),
            post_ff2_norm_name: String::new(),
            expert_gate_up_name: String::new(),
            expert_down_name: String::new(),
        });
        assert!(moe_plan.with_low_bit_replacements(&[valid]).is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn build_rope_tables_position_zero_identity_position_one_uses_theta() {
        let (cos, sin) = build_rope_tables(2, 2, 8, 10000.0);

        assert_eq!(cos[0], 1.0);
        assert_eq!(sin[0], 0.0);
        assert_eq!(cos[1], 1.0);
        assert_eq!(sin[1], 0.0);
        assert!((cos[2] - 1.0f32.cos()).abs() < 1e-6);
        assert!((sin[2] - 1.0f32.sin()).abs() < 1e-6);
        assert!((cos[3] - 0.1f32.cos()).abs() < 1e-6);
        assert!((sin[3] - 0.1f32.sin()).abs() < 1e-6);
    }

    fn add_zero_tensor(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
    ) {
        let count = shape.iter().copied().product::<usize>();
        add_tensor(header, payload, name, &vec![0.0f32; count], shape);
    }

    fn write_dry_run_full_gemma_style_fixture(
        tie_embeddings: bool,
        attention_k_eq_v: bool,
        omit_lm_head: bool,
        omit_v_proj_layer: Option<usize>,
        q_proj0_shape: Option<&[usize]>,
        q_proj1_shape: Option<&[usize]>,
        omit_q_norm0: bool,
    ) -> PathBuf {
        let dir = test_fixture_dir("dry-run-full-gemma-style");
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let sliding_head_dim = 128usize;
        let global_head_dim = 256usize;
        let prefix = "model.language_model";

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        add_zero_tensor(
            &mut header,
            &mut payload,
            &format!("{prefix}.embed_tokens.weight"),
            &[vocab, hidden],
        );
        add_zero_tensor(
            &mut header,
            &mut payload,
            &format!("{prefix}.norm.weight"),
            &[hidden],
        );
        if !omit_lm_head {
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{prefix}.lm_head.weight"),
                &[vocab, hidden],
            );
        }

        for (layer_idx, head_dim) in [(0usize, sliding_head_dim), (1usize, global_head_dim)] {
            let q_dim = head_dim;
            let kv_dim = head_dim;
            let lprefix = format!("{prefix}.layers.{layer_idx}");
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ] {
                add_zero_tensor(
                    &mut header,
                    &mut payload,
                    &format!("{lprefix}.{suffix}"),
                    &[hidden],
                );
            }

            let q_shape = match layer_idx {
                0 => q_proj0_shape
                    .map(|shape| shape.to_vec())
                    .unwrap_or_else(|| vec![q_dim, hidden]),
                1 => q_proj1_shape
                    .map(|shape| shape.to_vec())
                    .unwrap_or_else(|| vec![q_dim, hidden]),
                _ => vec![q_dim, hidden],
            };
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.q_proj.weight"),
                &q_shape,
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.k_proj.weight"),
                &[kv_dim, hidden],
            );
            if omit_v_proj_layer != Some(layer_idx) {
                add_zero_tensor(
                    &mut header,
                    &mut payload,
                    &format!("{lprefix}.self_attn.v_proj.weight"),
                    &[kv_dim, hidden],
                );
            }
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.o_proj.weight"),
                &[hidden, q_dim],
            );
            if !(layer_idx == 0 && omit_q_norm0) {
                add_zero_tensor(
                    &mut header,
                    &mut payload,
                    &format!("{lprefix}.self_attn.q_norm.weight"),
                    &[head_dim],
                );
            }
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.k_norm.weight"),
                &[head_dim],
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.layer_scalar"),
                &[hidden],
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.mlp.gate_proj.weight"),
                &[intermediate, hidden],
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.mlp.up_proj.weight"),
                &[intermediate, hidden],
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.mlp.down_proj.weight"),
                &[hidden, intermediate],
            );
        }

        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "text_config": {{
    "num_hidden_layers": 2,
    "hidden_size": {hidden},
    "intermediate_size": {intermediate},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {sliding_head_dim},
    "global_head_dim": {global_head_dim},
    "num_global_key_value_heads": 1,
    "layer_types": ["sliding_attention", "full_attention"],
    "vocab_size": {vocab},
    "max_position_embeddings": 1024,
    "sliding_window": 32,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": {tie_embeddings},
    "attention_k_eq_v": {attention_k_eq_v},
    "rope_parameters": {{
      "sliding_attention": {{"rope_theta": 10000.0}},
      "full_attention": {{"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}}
    }}
  }}
}}"#
            ),
        )
        .expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out = File::create(dir.join("model.safetensors")).expect("create safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[test]
    fn dry_run_validates_text_config_and_model_language_model_prefix() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, false, false, None, None, None, false);
        let validation =
            Gemma4DryRunValidation::from_model_dir(&dir).expect("dry-run validates fixture");

        assert_eq!(validation.weight_prefix, "model.language_model");
        assert_eq!(validation.num_layers, 2);
        assert_eq!(validation.hidden_size, 128);
        assert_eq!(validation.vocab_size, 8);
        assert_eq!(validation.final_logit_softcap, Some(30.0));
        assert_eq!(
            validation.layers[0].attention_kind,
            MetalProbeLayerAttentionKind::Sliding
        );
        assert_eq!(validation.layers[0].sliding_window, Some(32));
        assert_eq!(
            validation.layers[1].attention_kind,
            MetalProbeLayerAttentionKind::Full
        );
        assert_eq!(validation.layers[1].rope_dim, 64);
        assert_eq!(validation.layers[1].rope_theta, 1000000.0);
        assert_eq!(validation.layers[0].layer_scalar_dim, 128);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_allows_tied_embeddings_without_lm_head() {
        let dir =
            write_dry_run_full_gemma_style_fixture(true, false, true, None, None, None, false);
        let validation = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect("tied embeddings do not require lm_head");

        assert!(validation.tie_word_embeddings);
        assert_eq!(validation.lm_head, None);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_rejects_missing_lm_head_when_embeddings_are_not_tied() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, false, true, None, None, None, false);
        let err = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect_err("untied embeddings require lm_head");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.lm_head.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn probe_model_plan_allows_tied_embeddings_without_lm_head() {
        let dir =
            write_dry_run_full_gemma_style_fixture(true, false, true, None, None, None, false);
        let plan =
            ProbeModelPlan::new(&dir).expect("tied embeddings can reuse embed_tokens in prepare");

        assert!(plan.tie_embeddings);
        assert_eq!(plan.lm_head_name, plan.embed_name);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn probe_model_plan_rejects_missing_lm_head_when_embeddings_are_not_tied() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, false, true, None, None, None, false);
        let err = match ProbeModelPlan::new(&dir) {
            Ok(_) => panic!("untied embeddings require lm_head in prepare plan"),
            Err(err) => err,
        };
        let msg = format!("{err}");

        assert!(msg.contains("missing lm_head weights"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_allows_missing_v_proj_when_attention_k_eq_v() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, true, false, Some(1), None, None, false);
        let validation = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect("attention_k_eq_v permits missing v_proj");

        assert!(validation.layers[1].v_uses_k_proj);
        assert_eq!(validation.layers[1].v_proj, None);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_rejects_missing_sliding_v_proj_when_attention_k_eq_v() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, true, false, Some(0), None, None, false);
        let err = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect_err("sliding attention requires v_proj");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.v_proj.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_missing_tensor_error_names_missing_tensor() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, false, false, Some(0), None, None, false);
        let err = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect_err("missing v_proj must fail");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.v_proj.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_shape_mismatch_error_names_tensor_and_expected_actual_shapes() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            Some(&[127, 128]),
            None,
            false,
        );
        let err = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect_err("q_proj shape mismatch must fail");
        let msg = format!("{err}");

        assert!(msg.contains("ShapeMismatch"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(msg.contains("expected: [128, 128]"));
        assert!(msg.contains("got: [127, 128]"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_missing_q_norm_error_names_missing_tensor() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, false, false, None, None, None, true);
        let err =
            Gemma4DryRunValidation::from_model_dir(&dir).expect_err("missing q_norm must fail");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_norm.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_global_layer_shape_mismatch_uses_global_head_dim() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            None,
            Some(&[255, 128]),
            false,
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("global q_proj shape mismatch must fail");
        let msg = format!("{err}");

        assert!(msg.contains("ShapeMismatch"));
        assert!(msg.contains("model.language_model.layers.1.self_attn.q_proj.weight"));
        assert!(msg.contains("expected: [256, 128]"));
        assert!(msg.contains("got: [255, 128]"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore = "set RVLLM_GEMMA4_MODEL_DIR to validate a real Gemma4 model directory"]
    fn real_gemma4_model_dir_dry_run_validates_when_env_is_set() {
        let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
            eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
            return;
        };
        let model_dir = PathBuf::from(model_dir);
        let validation = Gemma4DryRunValidation::from_model_dir(&model_dir).unwrap_or_else(|err| {
            panic!(
                "Gemma4 dry-run validation failed for {}: {err}",
                model_dir.display()
            )
        });

        assert!(validation.num_layers > 0);
        assert!(validation.hidden_size > 0);
        assert!(validation.vocab_size > 0);
        assert_eq!(validation.layers.len(), validation.num_layers);
        eprintln!(
            "validated Gemma4 dry-run shapes: dir={} prefix={} layers={} hidden={} vocab={} tie_word_embeddings={} attention=sliding:{} full:{} v_uses_k_proj:{} lm_head_status={} fp8_scales={} fp8_scaled_weights={}",
            model_dir.display(),
            validation.weight_prefix,
            validation.num_layers,
            validation.hidden_size,
            validation.vocab_size,
            validation.tie_word_embeddings,
            validation.attention_sliding_layers,
            validation.attention_full_layers,
            validation.v_uses_k_proj_layers,
            validation.lm_head_status.as_str(),
            validation.fp8_scale_summary.mode.as_str(),
            validation.fp8_scale_summary.scaled_weights
        );
    }

    #[test]
    #[ignore = "set RVLLM_GEMMA4_MODEL_DIR to plan a real Gemma4 model directory"]
    fn real_gemma4_model_dir_arena_bytes_are_computable() {
        let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
            eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
            return;
        };
        let model_dir = PathBuf::from(model_dir);
        let bytes = Gemma4MetalState::required_probe_model_arena_bytes(&model_dir)
            .expect("Gemma4 Metal arena bytes should be computable without a layer-count opt-in");
        eprintln!(
            "Gemma4 Metal arena requirement: {bytes} bytes ({:.2} GiB)",
            bytes as f64 / 1024.0 / 1024.0 / 1024.0
        );
        assert!(bytes > 1024 * 1024 * 1024);
    }
}

#[derive(Debug, Default, Clone)]
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct Gemma4MetalState;
