//! Metal layer forward pass for transformer decoder blocks.
//!
//! Mirrors the CUDA `layer_exec::forward_phase()` but uses Metal compute
//! encoders instead of CUDA kernel launches.

use crate::arena::MetalBufferArena;
use crate::context::MetalContext;
use crate::low_bit_metal::MetalLowBitProjectionOffsets;
use crate::pipeline::PipelineCache;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};
use rvllm_apple::device::AppleGpuFamily;
use rvllm_apple::AppleLowBitTensorRole;
use rvllm_core::Result;

fn low_bit_qkv_dispatch_columns(skip_kv: bool, q_dim: u32, kv_dim: u32) -> [(bool, u32); 3] {
    [(true, 0), (!skip_kv, q_dim), (!skip_kv, q_dim + kv_dim)]
}

fn qkv_fusion_allowed(has_low_bit_qkv: bool, otherwise_allowed: bool) -> bool {
    !has_low_bit_qkv && otherwise_allowed
}

fn low_bit_descriptor_matches(
    projection: MetalLowBitProjectionOffsets,
    role: AppleLowBitTensorRole,
    shape: [u32; 2],
) -> bool {
    projection.role() == role && projection.shape() == shape
}

// Shared research shape check; all FFI/resource checks stay in this boundary crate.
fn research_layer_eligible(
    pipelines: &PipelineCache,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    candidate: crate::MetalResearchCandidate,
) -> bool {
    pipelines.kernel_options().research == candidate
        && !pipelines.kernel_options().quantized_bf16_accumulation
        && matches!(phase, MetalPhase::Prefill { .. })
        && (crate::research::Gemma12bResearchShape {
            tokens: dims.num_tokens,
            hidden: dims.hidden,
            intermediate: dims.intermediate,
            layers: dims.num_layers,
            heads: dims.num_heads,
            kv_heads: dims.num_kv_heads,
            head_dim: dims.head_dim,
            attention_window: dims.attention_window,
            moe_experts: dims.moe_num_experts,
            moe_top_k: dims.moe_top_k,
            moe_intermediate: dims.moe_intermediate,
            ple: dims.ple_dim,
        })
        .supports(candidate)
}

/// The same decision drives execution and diagnostic encoder accounting.
/// No savings are attributed to a missing PSO, unsupported layer or aliasing
/// scratch plan. This is a dispatch predicate, not a hardware acceptance flag.
#[allow(clippy::too_many_arguments)]
pub fn supports_research_rounded_gate(
    pipelines: &PipelineCache,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    weights: &MetalLayerWeights,
    scratch: &MetalScratch,
    capture_gate_up: bool,
    arena_bytes: usize,
) -> bool {
    research_layer_eligible(
        pipelines,
        dims,
        phase,
        crate::MetalResearchCandidate::RoundedGate32,
    ) && pipelines
        .research_pso("research_rounded_gate32", 128, 14_336)
        .is_some()
        && crate::research::rounded_gate_buffers_fit(
            [
                scratch.normed_hidden,
                weights.gate_up_offset,
                scratch.gate_up_out,
                scratch.activated,
            ],
            dims.num_tokens,
            dims.hidden,
            dims.intermediate,
            arena_bytes,
            capture_gate_up,
        )
}

/// Construct the exact same request for execution and diagnostic accounting.
#[allow(clippy::too_many_arguments)]
pub fn research_bf16_gate_request(
    pipelines: &PipelineCache,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    weights: &MetalLayerWeights,
    scratch: &MetalScratch,
    capture_gate_up: bool,
    arena_bytes: usize,
) -> crate::research_decode::GateUpRequest {
    crate::research_decode::GateUpRequest {
        selected: pipelines.kernel_options().research,
        dtype: pipelines.float_type(),
        decode: matches!(phase, MetalPhase::Decode),
        quantized_accumulation: pipelines.kernel_options().quantized_bf16_accumulation,
        model: crate::research::Gemma12bResearchShape {
            tokens: dims.num_tokens,
            hidden: dims.hidden,
            intermediate: dims.intermediate,
            layers: dims.num_layers,
            heads: dims.num_heads,
            kv_heads: dims.num_kv_heads,
            head_dim: dims.head_dim,
            attention_window: dims.attention_window,
            moe_experts: dims.moe_num_experts,
            moe_top_k: dims.moe_top_k,
            moe_intermediate: dims.moe_intermediate,
            ple: dims.ple_dim,
        },
        capture_gate_up,
        has_low_bit_gate_or_up: weights.low_bit_gate_proj.is_some()
            || weights.low_bit_up_proj.is_some(),
        offsets: [
            scratch.normed_hidden,
            weights.gate_up_offset,
            scratch.activated,
        ],
        arena_bytes,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn supports_research_bf16_gate(
    pipelines: &PipelineCache,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    weights: &MetalLayerWeights,
    scratch: &MetalScratch,
    capture_gate_up: bool,
    arena_bytes: usize,
) -> bool {
    crate::research_decode_metal::gate_up_plan(
        pipelines,
        research_bf16_gate_request(
            pipelines,
            dims,
            phase,
            weights,
            scratch,
            capture_gate_up,
            arena_bytes,
        ),
    )
    .is_some()
}

// Measured on M4 Max with Gemma 4 E2B: cooperative projection remains faster
// through M=18, is effectively tied at M=19, and regresses sharply at M=20.
const COOPERATIVE_GEMV_MAX_M: u32 = 19;

// Gemma 4 12B's gate/up, output, and down projections benefit from sharing
// weights across prompt rows. Keep the larger window limited to these shapes.
fn is_gemma4_12b_prompt_projection(m: u32, n: u32, k: u32) -> bool {
    (20..=32).contains(&m) && matches!((n, k), (30_720, 3_840) | (3_840, 4_096 | 8_192 | 15_360))
}

/// Eligibility shared by encoding and diagnostic encoder accounting.
#[must_use]
pub fn supports_qkv_prefill_projection(pipelines: &PipelineCache, dims: &MetalLayerDims) -> bool {
    (pipelines.kernel_options().qkv_prefill_batch8
        || prefill_mma_enabled(pipelines))
        // The diagnostic scalar-rounding rewrite does not apply to batch8.
        && !pipelines.kernel_options().quantized_bf16_accumulation
        && matches!(pipelines.gpu_family(), AppleGpuFamily::Apple9 | AppleGpuFamily::Apple10)
        && ((prefill_mma_enabled(pipelines) && (6..=1024).contains(&dims.num_tokens))
            || (20..=32).contains(&dims.num_tokens))
        && dims.hidden == 3840
        && dims.intermediate == 15360
        && dims.num_layers == 48
        && dims.moe_num_experts == 0
        && dims.moe_top_k == 0
        && dims.moe_intermediate == 0
        && dims.ple_dim == 0
        && dims.num_heads == 16
        && matches!((dims.num_kv_heads, dims.head_dim), (8, 256) | (1, 512))
}

fn prefill_mma_enabled(pipelines: &PipelineCache) -> bool {
    pipelines.kernel_options().prefill_mma32
        && !pipelines.kernel_options().quantized_bf16_accumulation
        && pipelines.float_type() == Some(crate::MetalFloatType::Bf16)
        && matches!(
            pipelines.gpu_family(),
            AppleGpuFamily::Apple9 | AppleGpuFamily::Apple10
        )
}

// Arithmetic-changing prefill experiment. Keep the route separate from MMA
// and limited to the model, dtype and GPU families qualified by the component.
fn supports_gemma4_prefill_simd_attention(
    pipelines: &PipelineCache,
    dims: &MetalLayerDims,
) -> bool {
    pipelines.kernel_options().prefill_simd_attention
        && pipelines.float_type() == Some(crate::MetalFloatType::Bf16)
        && matches!(
            pipelines.gpu_family(),
            AppleGpuFamily::Apple9 | AppleGpuFamily::Apple10
        )
        && (6..=1024).contains(&dims.num_tokens)
        && dims.hidden == 3840
        && dims.intermediate == 15360
        && dims.num_layers == 48
        && dims.moe_num_experts == 0
        && dims.moe_top_k == 0
        && dims.moe_intermediate == 0
        && dims.ple_dim == 0
        && dims.num_heads == 16
        && matches!(
            (dims.num_kv_heads, dims.head_dim, dims.attention_window),
            (8, 256, 1024) | (1, 512, 0)
        )
}

fn is_prefill_mma_shape(m: u32, n: u32, k: u32, output_f32: bool) -> bool {
    (6..=1024).contains(&m)
        && if output_f32 {
            k == 3840 && matches!(n, 8192 | 9216)
        } else {
            matches!((n, k), (30_720, 3_840) | (3_840, 4_096 | 8_192 | 15_360))
        }
}

/// Resolve the matrix policy once from the actual layer and execution phase.
/// Batched decode and unrelated architectures retain their existing routing.
#[must_use]
pub fn supports_gemma4_prefill_mma(
    pipelines: &PipelineCache,
    dims: &MetalLayerDims,
    phase: MetalPhase,
) -> bool {
    matches!(phase, MetalPhase::Prefill { .. })
        && prefill_mma_enabled(pipelines)
        && supports_qkv_prefill_projection(pipelines, dims)
}

#[cfg(test)]
#[path = "qkv_prefill_tests.rs"]
mod qkv_prefill_tests;

#[cfg(test)]
#[path = "prefill_attention_tests.rs"]
mod prefill_attention_tests;

#[cfg(test)]
#[path = "pointwise_dispatch_tests.rs"]
mod pointwise_dispatch_tests;

#[cfg(test)]
#[path = "prefill_mma_tile_tests.rs"]
mod prefill_mma_tile_tests;

#[cfg(test)]
#[path = "prefill_mma_tests.rs"]
mod prefill_mma_tests;

/// Canonical lower bound for a causal attention extent. `attention_window ==
/// 0` is the stable full-attention encoding; otherwise the returned extent
/// contains at most the requested number of tokens, including the query.
#[must_use]
pub const fn attention_window_start(attention_len: u32, attention_window: u32) -> u32 {
    if attention_window == 0 {
        0
    } else {
        attention_len.saturating_sub(attention_window)
    }
}

/// Dimensions for a single decoder layer, matching rvllm-runtime's LayerDims.
#[derive(Copy, Clone, Debug)]
pub struct MetalLayerDims {
    pub layer_idx: u32,
    /// Exact causal attention width. Zero denotes full attention.
    pub attention_window: u32,
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub moe_num_experts: u32,
    pub moe_top_k: u32,
    pub moe_intermediate: u32,
    pub ple_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub attn_scale: f32,
    pub rms_eps: f32,
    pub rope_dim: u32,
    pub softcap: f32,
}

/// Per-layer weight offsets within the arena buffer.
#[derive(Copy, Clone, Debug)]
pub struct MetalLayerWeights {
    pub attn_norm_offset: usize,
    pub qkv_offset: usize,
    pub qkv_bias_offset: Option<usize>,
    pub q_norm_offset: Option<usize>,
    pub k_norm_offset: Option<usize>,
    pub v_norm_offset: Option<usize>,
    pub o_proj_offset: usize,
    pub mlp_norm_offset: usize,
    pub post_attn_norm_offset: Option<usize>,
    pub pre_ff_norm_offset: Option<usize>,
    pub post_ff_norm_offset: Option<usize>,
    pub layer_scalar_offset: Option<usize>,
    pub layer_scalar_dim: u32,
    pub gate_up_offset: usize,
    pub down_proj_offset: Option<usize>,
    pub low_bit_q_proj: Option<MetalLowBitProjectionOffsets>,
    pub low_bit_k_proj: Option<MetalLowBitProjectionOffsets>,
    pub low_bit_v_proj: Option<MetalLowBitProjectionOffsets>,
    pub low_bit_o_proj: Option<MetalLowBitProjectionOffsets>,
    pub low_bit_gate_proj: Option<MetalLowBitProjectionOffsets>,
    pub low_bit_up_proj: Option<MetalLowBitProjectionOffsets>,
    pub low_bit_down_proj: Option<MetalLowBitProjectionOffsets>,
    pub moe: Option<MetalMoeWeights>,
    pub per_layer_inputs_offset: Option<usize>,
    pub per_layer_input_gate_offset: Option<usize>,
    pub per_layer_projection_offset: Option<usize>,
    pub post_per_layer_input_norm_offset: Option<usize>,
}

#[derive(Copy, Clone, Debug)]
pub struct MetalMoeWeights {
    pub router_proj_offset: usize,
    pub router_scale_offset: usize,
    pub router_per_expert_scale_offset: usize,
    pub pre_ff2_norm_offset: usize,
    pub post_ff1_norm_offset: usize,
    pub post_ff2_norm_offset: usize,
    pub expert_gate_up_offset: usize,
    pub expert_down_offset: usize,
}

/// Pre-allocated scratch buffer offsets.
#[derive(Copy, Clone, Debug)]
pub struct MetalScratch {
    pub normed_hidden: usize,
    pub qkv_out: usize, // packed [Q, K, V]
    pub q_offset: usize,
    pub k_offset: usize,
    pub v_offset: usize,
    pub attn_out: usize,
    /// Dedicated FP32 sufficient-statistic storage for bounded split-KV decode.
    pub global_decode_partials: Option<usize>,
    pub gate_up_out: usize,
    pub activated: usize,
    pub mlp_out: usize,
    pub moe_topk_indices: Option<usize>,
    pub moe_topk_weights: Option<usize>,
    pub moe_activated: Option<usize>,
    pub moe_out: Option<usize>,
}

/// Optional trace-only snapshot offsets within the arena buffer.
///
/// These regions are populated only when a caller passes a trace scratch block
/// into `metal_forward_layer`. Normal production execution passes `None` and
/// keeps the original scratch plan unchanged.
#[derive(Copy, Clone, Debug)]
pub struct MetalLayerTraceScratch {
    pub input_to_layer: usize,
    pub after_input_layernorm: usize,
    pub q_projection: usize,
    pub k_projection: usize,
    pub v_projection: usize,
    pub after_q_norm: usize,
    pub after_k_norm: usize,
    pub after_v_norm: usize,
    pub after_rope_q: usize,
    pub after_rope_k: usize,
    pub attention_output: usize,
    pub after_o_proj: usize,
    pub after_post_attention_layernorm: usize,
    pub after_pre_feedforward_layernorm: usize,
    pub gate_up_out: usize,
    pub ffn_activation: usize,
    pub after_ffn_branch: usize,
    pub after_post_feedforward_layernorm: usize,
    pub per_layer_input: Option<usize>,
    pub per_layer_input_gate: Option<usize>,
    pub per_layer_projection: Option<usize>,
    pub post_per_layer_input_norm: Option<usize>,
}

/// Test/debug-only execution switches for bounded diagnostics.
///
/// Normal callers pass `Default::default()`. These flags deliberately do not
/// encode an optimization contract; they exist so the runtime can compare the
/// current E2B shared-KV path against narrower skip hypotheses.
#[derive(Copy, Clone, Debug, Default)]
pub struct MetalLayerDebugSkip {
    pub skip_kv_projection: bool,
    pub skip_local_kv_cache_write: bool,
}

fn validate_debug_skip(debug_skip: MetalLayerDebugSkip) -> Result<()> {
    if debug_skip.skip_kv_projection && !debug_skip.skip_local_kv_cache_write {
        return Err(rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::InvalidWeightBlob {
                reason: "cannot write the local KV cache after skipping K/V projection",
            },
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "debug_skip_kv_projection",
                device: "apple-silicon",
            },
        ));
    }
    Ok(())
}

#[cfg(test)]
mod debug_skip_tests {
    use super::*;

    #[test]
    fn skipping_kv_projection_requires_skipping_its_cache_write() {
        assert!(validate_debug_skip(MetalLayerDebugSkip::default()).is_ok());
        assert!(validate_debug_skip(MetalLayerDebugSkip {
            skip_kv_projection: true,
            skip_local_kv_cache_write: true,
        })
        .is_ok());
        assert!(validate_debug_skip(MetalLayerDebugSkip {
            skip_kv_projection: true,
            skip_local_kv_cache_write: false,
        })
        .is_err());
    }
}

/// Metadata buffer offsets (positions, slot mapping, etc).
#[derive(Copy, Clone, Debug)]
pub struct MetalMetadata {
    pub positions_offset: usize,
    pub slot_mapping_offset: usize,
    pub cos_offset: usize,
    pub sin_offset: usize,
    pub block_tables_offset: usize,
    pub context_lens_offset: usize,
    pub cu_seqlens_offset: Option<usize>,
}

/// Offsets for Gemma 4 per-layer embedding input preparation.
#[derive(Copy, Clone, Debug)]
pub struct MetalPlePrepare {
    pub embedding_offset: usize,
    pub token_ids_offset: usize,
    pub residual_offset: usize,
    pub per_layer_model_projection_offset: usize,
    pub per_layer_projection_norm_offset: usize,
    pub token_inputs_offset: usize,
    pub context_inputs_offset: usize,
    pub num_tokens: u32,
    pub hidden: u32,
    pub vocab: u32,
    pub num_layers: u32,
    pub ple_dim: u32,
    pub rms_eps: f32,
}

/// Which phase: decode (1 token/seq) or prefill (multi-token/seq).
#[derive(Copy, Clone, Debug)]
pub enum MetalPhase {
    Decode,
    Prefill { max_seqlen_q: u32, batch_size: u32 },
}

fn ranges_overlap(a_start: usize, a_len: usize, b_start: usize, b_len: usize) -> bool {
    let Some(a_end) = a_start.checked_add(a_len) else {
        return true;
    };
    let Some(b_end) = b_start.checked_add(b_len) else {
        return true;
    };
    a_start < b_end && b_start < a_end
}

fn validate_qkv_scratch_planar(
    scratch: &MetalScratch,
    num_tokens: usize,
    q_dim: usize,
    kv_dim: usize,
    projection_f32: bool,
) -> Result<()> {
    let elem = std::mem::size_of::<u16>();
    let qkv_elems = num_tokens.checked_mul(q_dim + 2 * kv_dim).ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "qkv scratch element overflow",
            },
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "layer_forward_scratch_planar",
                device: "apple-silicon",
            },
        )
    })?;
    let projection_bytes = if projection_f32 {
        std::mem::size_of::<f32>()
    } else {
        elem
    };
    let qkv_len = qkv_elems.checked_mul(projection_bytes).ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "qkv scratch byte overflow",
            },
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "layer_forward_scratch_planar",
                device: "apple-silicon",
            },
        )
    })?;
    let q_len = num_tokens * q_dim * elem;
    let kv_len = num_tokens * kv_dim * elem;

    let bad = ranges_overlap(scratch.qkv_out, qkv_len, scratch.q_offset, q_len)
        || ranges_overlap(scratch.qkv_out, qkv_len, scratch.k_offset, kv_len)
        || ranges_overlap(scratch.qkv_out, qkv_len, scratch.v_offset, kv_len)
        || ranges_overlap(scratch.q_offset, q_len, scratch.k_offset, kv_len)
        || ranges_overlap(scratch.q_offset, q_len, scratch.v_offset, kv_len)
        || ranges_overlap(scratch.k_offset, kv_len, scratch.v_offset, kv_len);
    if bad {
        return Err(rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::FeatureNotAvailable {
                backend: "metal",
                op: "qkv planar scratch regions overlap",
            },
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "layer_forward_scratch_planar",
                device: "apple-silicon",
            },
        ));
    }
    Ok(())
}

fn supports_attention_decode_online(dims: &MetalLayerDims) -> bool {
    dims.num_heads > 0
        && dims.num_kv_heads > 0
        && dims.num_heads % dims.num_kv_heads == 0
        && dims.head_dim <= 256
        && dims.block_size > 0
        && dims.max_blocks_per_seq > 0
}

fn supports_tiled_gemm(m: u32, n: u32, k: u32) -> bool {
    m > 0 && n > 0 && k > 0 && m <= 16 && n <= 262_144 && k <= 16_384
}

fn supports_vec_gemm(m: u32, n: u32, k: u32) -> bool {
    m > 0
        && n > 0
        && k > 0
        && (m <= COOPERATIVE_GEMV_MAX_M || is_gemma4_12b_prompt_projection(m, n, k))
        && n <= 262_144
        && k <= 65_536
}

/// Apple9/10 batch-eight projection path. Each simdgroup owns one output
/// column and accumulates eight request rows, so the dominant `[N,K]` weight
/// stream is shared across the batch instead of being fetched once per row.
/// The eight-row path also covers 20–32-token Gemma 4 12B prompt projections;
/// smaller microbatches retain the established cooperative GEMV reduction.
fn supports_batch8_gemm(gpu_family: AppleGpuFamily, m: u32, n: u32, k: u32) -> bool {
    matches!(gpu_family, AppleGpuFamily::Apple9 | AppleGpuFamily::Apple10)
        && (m == 8 || is_gemma4_12b_prompt_projection(m, n, k))
        && n >= 1_024
        && n <= 262_144
        && n.is_multiple_of(8)
        && k >= 1_024
        && k <= 65_536
        && k.is_multiple_of(32)
}

/// Number of encoders used by the projection + RMSNorm policy.
///
/// For decode and prompt microbatches, the cooperative GEMV kernel exposes
/// enough parallelism to keep the GPU occupied. The single-encoder fused
/// kernel launches only one threadgroup per token and serializes large output
/// projections, so the small-batch path deliberately spends a second encoder
/// on RMSNorm. Qualified native BF16 matrix prefills also materialize the
/// projection before RMSNorm; other larger prefills retain the fused fallback.
#[must_use]
pub fn metal_gemm_rmsnorm_encoder_count(m: u32, n: u32, k: u32, allow_prefill_mma: bool) -> u64 {
    if supports_vec_gemm(m, n, k) || (allow_prefill_mma && is_prefill_mma_shape(m, n, k, false)) {
        2
    } else {
        1
    }
}

fn supports_fused_final_logits_small(num_tokens: u32, hidden: u32, vocab: u32) -> bool {
    num_tokens > 0 && num_tokens <= 256 && hidden > 0 && hidden <= 4096 && vocab > 0 && vocab <= 256
}

fn supports_final_sample_tiles(num_tokens: u32, hidden: u32, vocab: u32) -> bool {
    num_tokens > 0
        && num_tokens <= 256
        && hidden > 0
        && hidden <= 65_536
        && vocab > 256
        && vocab <= 262_144
}

fn supports_batch8_final_sample(
    gpu_family: AppleGpuFamily,
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
) -> bool {
    matches!(gpu_family, AppleGpuFamily::Apple9 | AppleGpuFamily::Apple10)
        && num_tokens == 8
        && hidden >= 1_024
        && hidden <= 65_536
        && hidden.is_multiple_of(32)
        && vocab >= 1_024
        && vocab <= 262_144
}

fn supports_batch4_final_sample(
    gpu_family: AppleGpuFamily,
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
) -> bool {
    matches!(gpu_family, AppleGpuFamily::Apple9 | AppleGpuFamily::Apple10)
        && num_tokens == 4
        && hidden >= 1_024
        && hidden <= 65_536
        && hidden.is_multiple_of(32)
        && vocab >= 1_024
        && vocab <= 262_144
}

#[must_use]
pub fn supports_qkv_rope_cache_fusion(dims: &MetalLayerDims) -> bool {
    dims.num_tokens > 0
        && dims.hidden > 0
        && dims.head_dim > 0
        && dims.head_dim <= 512
        && dims.rope_dim > 0
        && dims.rope_dim <= dims.head_dim
        && dims.rope_dim % 2 == 0
        && dims.num_heads > 0
        && dims.num_kv_heads > 0
}

#[must_use]
pub fn metal_finalize_logits_encoder_count(
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
    _softcap: f32,
) -> u64 {
    if supports_fused_final_logits_small(num_tokens, hidden, vocab) {
        1
    } else {
        3
    }
}

#[must_use]
pub fn metal_finalize_sample_encoder_count(num_tokens: u32, hidden: u32, vocab: u32) -> u64 {
    if supports_fused_final_logits_small(num_tokens, hidden, vocab) {
        1
    } else if supports_final_sample_tiles(num_tokens, hidden, vocab) {
        3
    } else {
        metal_finalize_logits_encoder_count(num_tokens, hidden, vocab, 0.0)
    }
}

pub unsafe fn metal_prepare_ple_inputs(
    ctx: &MetalContext,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    params: &MetalPlePrepare,
) -> Result<()> {
    let queue = ctx.queue_retained();
    let cmd_buf = queue.commandBuffer().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "ple_prepare",
                device: "apple-silicon",
            },
        )
    })?;

    metal_encode_prepare_ple_inputs(&cmd_buf, pipelines, arena, params)?;

    cmd_buf.commit();
    Ok(())
}

pub unsafe fn metal_encode_prepare_ple_inputs(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    params: &MetalPlePrepare,
) -> Result<()> {
    let stride = params.num_layers.saturating_mul(params.ple_dim);
    let ple_scale = (params.ple_dim as f32).sqrt();
    let buf = arena.buffer_retained();
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "ple_embedding_gather",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("embedding_gather_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), params.embedding_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), params.token_ids_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), params.token_inputs_offset, 2);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&params.num_tokens as *const _ as *mut _),
            4,
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&stride as *const _ as *mut _),
            4,
            4,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&params.vocab as *const _ as *mut _),
            4,
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&ple_scale as *const _ as *mut _),
            4,
            6,
        );
        let threads_per_group = MTLSize {
            width: 1,
            height: (stride as usize).clamp(1, 256),
            depth: 1,
        };
        let groups = MTLSize {
            width: params.num_tokens as usize,
            height: (stride as usize + threads_per_group.height - 1) / threads_per_group.height,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();
    }

    let alpha = 1.0f32 / (params.hidden as f32).sqrt();
    encode_gemm(
        &cmd_buf,
        pipelines,
        buf,
        params.residual_offset,
        params.per_layer_model_projection_offset,
        params.context_inputs_offset,
        params.num_tokens,
        stride,
        params.hidden,
        alpha,
        0.0,
    )?;
    encode_headwise_rmsnorm(
        &cmd_buf,
        pipelines,
        buf,
        params.context_inputs_offset,
        params.context_inputs_offset,
        params.per_layer_projection_norm_offset,
        params.ple_dim,
        params.num_layers,
        params.rms_eps,
        params.num_tokens,
        "ple_context_norm",
    )?;

    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "ple_combine",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("ple_combine_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), params.token_inputs_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), params.context_inputs_offset, 1);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&params.num_tokens as *const _ as *mut _),
            4,
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&stride as *const _ as *mut _),
            4,
            3,
        );
        let groups = MTLSize {
            width: params.num_tokens as usize,
            height: stride as usize,
            depth: 1,
        };
        let tpg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    }

    Ok(())
}

/// Execute one decoder layer on Metal.
///
/// All buffers are pre-allocated in the arena. This function only
/// encodes compute commands — no allocation, no buffer creation.
///
/// # Safety
/// Caller must ensure all buffer offsets are valid within the arena
/// and that no concurrent GPU work is modifying the same regions.
pub unsafe fn metal_forward_layer(
    ctx: &MetalContext,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    dims: &MetalLayerDims,
    weights: &MetalLayerWeights,
    scratch: &MetalScratch,
    trace: Option<&MetalLayerTraceScratch>,
    meta: &MetalMetadata,
    residual_offset: usize,
    phase: MetalPhase,
    kv_cache_k_offset: usize,
    kv_cache_v_offset: usize,
    attention_kv_cache_k_offset: usize,
    attention_kv_cache_v_offset: usize,
    debug_skip: MetalLayerDebugSkip,
) -> Result<()> {
    let queue = ctx.queue_retained();
    let cmd_buf = queue.commandBuffer().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "layer_forward",
                device: "apple-silicon",
            },
        )
    })?;
    metal_encode_forward_layer(
        &cmd_buf,
        pipelines,
        arena,
        dims,
        weights,
        scratch,
        trace,
        meta,
        residual_offset,
        phase,
        kv_cache_k_offset,
        kv_cache_v_offset,
        attention_kv_cache_k_offset,
        attention_kv_cache_v_offset,
        debug_skip,
        #[cfg(feature = "metal-stage-instrumentation")]
        None,
    )?;
    cmd_buf.commit();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_encode_forward_layer(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    dims: &MetalLayerDims,
    weights: &MetalLayerWeights,
    scratch: &MetalScratch,
    trace: Option<&MetalLayerTraceScratch>,
    meta: &MetalMetadata,
    residual_offset: usize,
    phase: MetalPhase,
    kv_cache_k_offset: usize,
    kv_cache_v_offset: usize,
    attention_kv_cache_k_offset: usize,
    attention_kv_cache_v_offset: usize,
    debug_skip: MetalLayerDebugSkip,
    #[cfg(feature = "metal-stage-instrumentation")] mut stage_profiler: Option<
        &mut crate::stage_instrumentation::MetalStageProfiler,
    >,
) -> Result<()> {
    if crate::donor12b::simdgroups(pipelines.kernel_options().research).is_some() {
        pipelines.reset_donor_layer_encoder_correction();
    }
    let allow_prefill_mma = trace.is_none() && supports_gemma4_prefill_mma(pipelines, dims, phase);
    let buf = arena.buffer_retained();
    let num_tokens = dims.num_tokens;
    let hidden = dims.hidden;
    let q_dim = dims.num_heads * dims.head_dim;
    let kv_dim = dims.num_kv_heads * dims.head_dim;
    let qkv_n = q_dim + 2 * kv_dim;
    validate_debug_skip(debug_skip)?;
    let native_down_proj_offset = validate_down_projection_sources(
        weights.down_proj_offset,
        weights.low_bit_down_proj.is_some(),
    )?
    // Low-bit branches never read this value. The source validation above
    // guarantees every native branch receives the real offset.
    .unwrap_or(0);
    if let Some(projection) = weights.low_bit_down_proj {
        let expected = [hidden, dims.intermediate];
        if !low_bit_descriptor_matches(
            projection,
            AppleLowBitTensorRole::DenseDownProjection,
            expected,
        ) {
            return Err(rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::InvalidWeightBlob {
                    reason: "low-bit down projection shape does not match prepared layer",
                },
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "low_bit_down_projection_shape",
                    device: "apple-silicon",
                },
            ));
        }
    }
    let low_bit_qkv = match (
        weights.low_bit_q_proj,
        weights.low_bit_k_proj,
        weights.low_bit_v_proj,
    ) {
        (None, None, None) => None,
        (Some(q), Some(k), Some(v)) => {
            if !low_bit_descriptor_matches(
                q,
                AppleLowBitTensorRole::QueryProjection,
                [q_dim, hidden],
            ) || !low_bit_descriptor_matches(
                k,
                AppleLowBitTensorRole::KeyProjection,
                [kv_dim, hidden],
            ) || !low_bit_descriptor_matches(
                v,
                AppleLowBitTensorRole::ValueProjection,
                [kv_dim, hidden],
            ) {
                return Err(rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::InvalidWeightBlob {
                        reason: "low-bit Q/K/V projection shape does not match prepared layer",
                    },
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "low_bit_qkv_projection_shape",
                        device: "apple-silicon",
                    },
                ));
            }
            Some((q, k, v))
        }
        _ => {
            return Err(rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::InvalidWeightBlob {
                    reason: "low-bit Q/K/V projections must be installed as one complete set",
                },
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "low_bit_qkv_projection_set",
                    device: "apple-silicon",
                },
            ));
        }
    };
    let low_bit_gate_up = match (weights.low_bit_gate_proj, weights.low_bit_up_proj) {
        (None, None) => None,
        (Some(gate), Some(up)) => {
            if !low_bit_descriptor_matches(
                gate,
                AppleLowBitTensorRole::DenseGateProjection,
                [dims.intermediate, hidden],
            ) || !low_bit_descriptor_matches(
                up,
                AppleLowBitTensorRole::DenseUpProjection,
                [dims.intermediate, hidden],
            ) {
                return Err(rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::InvalidWeightBlob {
                        reason: "low-bit gate/up projection shape does not match prepared layer",
                    },
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "low_bit_gate_up_projection_shape",
                        device: "apple-silicon",
                    },
                ));
            }
            Some((gate, up))
        }
        _ => {
            return Err(rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::InvalidWeightBlob {
                    reason: "low-bit gate/up projections must be installed as one complete set",
                },
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "low_bit_gate_up_projection_set",
                    device: "apple-silicon",
                },
            ));
        }
    };
    let use_fused_qkv_rope_cache = qkv_fusion_allowed(
        low_bit_qkv.is_some(),
        trace.is_none()
            && !debug_skip.skip_kv_projection
            && !debug_skip.skip_local_kv_cache_write
            && weights.q_norm_offset.is_some()
            && weights.k_norm_offset.is_some()
            && supports_qkv_rope_cache_fusion(dims),
    );
    // Despite its historical name, this is also the donor's decode FP32
    // projection lane. Four-byte scratch is already reserved for dense 12B.
    let use_qkv_prefill_projection = use_fused_qkv_rope_cache
        && ((matches!(phase, MetalPhase::Prefill { .. })
            && supports_qkv_prefill_projection(pipelines, dims))
            || crate::donor12b_metal::projected_qkv_allowed(pipelines, dims, phase));
    validate_qkv_scratch_planar(
        scratch,
        num_tokens as usize,
        q_dim as usize,
        kv_dim as usize,
        use_qkv_prefill_projection,
    )?;
    if let Some(trace) = trace {
        encode_trace_copy(
            &cmd_buf,
            pipelines,
            buf,
            residual_offset,
            trace.input_to_layer,
            num_tokens * hidden,
            "trace_input_to_layer",
        )?;
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    if let Some(profiler) = stage_profiler.as_deref_mut() {
        profiler.begin(
            cmd_buf,
            crate::stage_instrumentation::MetalStage::NormResidual,
        );
    }
    // 1. RMSNorm(residual) → normed_hidden
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "rmsnorm_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("rmsnorm_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.normed_hidden, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), weights.attn_norm_offset, 2);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&hidden as *const u32 as *mut _),
            4,
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&dims.rms_eps as *const f32 as *mut _),
            4,
            4,
        );
        let threads_per_group = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: num_tokens as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();
    }
    if let Some(trace) = trace {
        encode_trace_copy(
            &cmd_buf,
            pipelines,
            buf,
            scratch.normed_hidden,
            trace.after_input_layernorm,
            num_tokens * hidden,
            "trace_after_input_layernorm",
        )?;
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    if let Some(profiler) = stage_profiler.as_deref_mut() {
        profiler.end(cmd_buf);
        profiler.begin(
            cmd_buf,
            if dims.attention_window == 0 {
                crate::stage_instrumentation::MetalStage::QkvFull
            } else {
                crate::stage_instrumentation::MetalStage::QkvSliding
            },
        );
    }
    // 2-4. QKV projection and optional Gemma-style Q/K/V norms before RoPE.
    if let Some((q_projection, k_projection, v_projection)) = low_bit_qkv {
        if !debug_skip.skip_kv_projection {
            let donor_qkv = crate::donor12b_metal::try_encode_qkv(
                pipelines,
                cmd_buf,
                buf,
                dims,
                phase,
                [q_projection, k_projection, v_projection],
                scratch.normed_hidden,
                scratch.qkv_out,
                trace.is_some(),
                false,
            )?;
            if !donor_qkv {
                for ((enabled, column), projection) in
                    low_bit_qkv_dispatch_columns(debug_skip.skip_kv_projection, q_dim, kv_dim)
                        .into_iter()
                        .zip([q_projection, k_projection, v_projection])
                {
                    debug_assert!(enabled);
                    encode_low_bit_projection_strided(
                        &cmd_buf,
                        pipelines,
                        buf,
                        projection,
                        scratch.normed_hidden,
                        scratch.qkv_out,
                        num_tokens,
                        qkv_n,
                        column,
                        phase,
                        dims,
                    )?;
                }
            }
            encode_split_qkv(
                &cmd_buf,
                pipelines,
                buf,
                scratch.qkv_out,
                scratch.q_offset,
                scratch.k_offset,
                scratch.v_offset,
                num_tokens,
                q_dim,
                kv_dim,
            )?;
        } else {
            encode_low_bit_projection_strided(
                &cmd_buf,
                pipelines,
                buf,
                q_projection,
                scratch.normed_hidden,
                scratch.q_offset,
                num_tokens,
                q_dim,
                0,
                phase,
                dims,
            )?;
        }
        if let Some(q_norm_offset) = weights.q_norm_offset {
            encode_headwise_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.q_offset,
                scratch.q_offset,
                q_norm_offset,
                dims.head_dim,
                dims.num_heads,
                dims.rms_eps,
                num_tokens,
                "low_bit_q_norm",
            )?;
        }
        if !debug_skip.skip_kv_projection {
            if let Some(k_norm_offset) = weights.k_norm_offset {
                encode_headwise_rmsnorm(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.k_offset,
                    scratch.k_offset,
                    k_norm_offset,
                    dims.head_dim,
                    dims.num_kv_heads,
                    dims.rms_eps,
                    num_tokens,
                    "low_bit_k_norm",
                )?;
            }
            if let Some(v_norm_offset) = weights.v_norm_offset {
                encode_headwise_rmsnorm(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.v_offset,
                    scratch.v_offset,
                    v_norm_offset,
                    dims.head_dim,
                    dims.num_kv_heads,
                    dims.rms_eps,
                    num_tokens,
                    "low_bit_v_norm",
                )?;
            } else {
                encode_headwise_rmsnorm_unit(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.v_offset,
                    scratch.v_offset,
                    dims.head_dim,
                    dims.num_kv_heads,
                    dims.rms_eps,
                    num_tokens,
                    "low_bit_v_norm_unit",
                )?;
            }
        }
    } else if let (Some(q_norm_offset), Some(k_norm_offset)) =
        (weights.q_norm_offset, weights.k_norm_offset)
    {
        if let Some(trace) = trace {
            encode_gemm_with_output(
                &cmd_buf,
                pipelines,
                buf,
                scratch.normed_hidden,
                weights.qkv_offset,
                scratch.qkv_out,
                num_tokens,
                qkv_n,
                hidden,
                1.0,
                0.0,
                false,
                allow_prefill_mma,
            )?;
            encode_split_qkv(
                &cmd_buf,
                pipelines,
                buf,
                scratch.qkv_out,
                scratch.q_offset,
                scratch.k_offset,
                scratch.v_offset,
                num_tokens,
                q_dim,
                kv_dim,
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.q_offset,
                trace.q_projection,
                num_tokens * q_dim,
                "trace_q_projection",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.k_offset,
                trace.k_projection,
                num_tokens * kv_dim,
                "trace_k_projection",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.v_offset,
                trace.v_projection,
                num_tokens * kv_dim,
                "trace_v_projection",
            )?;
        }
        if trace.is_none() && !debug_skip.skip_kv_projection {
            let v_gamma_offset = weights.v_norm_offset.unwrap_or(q_norm_offset);
            if use_fused_qkv_rope_cache {
                if use_qkv_prefill_projection {
                    encode_gemm_with_output(
                        &cmd_buf,
                        pipelines,
                        buf,
                        scratch.normed_hidden,
                        weights.qkv_offset,
                        scratch.qkv_out,
                        num_tokens,
                        qkv_n,
                        hidden,
                        1.0,
                        0.0,
                        true,
                        allow_prefill_mma,
                    )?;
                    if crate::donor12b::simdgroups(pipelines.kernel_options().research).is_some()
                        && !(matches!(phase, MetalPhase::Prefill { .. })
                            && supports_qkv_prefill_projection(pipelines, dims))
                    {
                        pipelines.add_donor_layer_encoder_correction(1);
                    }
                }
                encode_qkv_headwise_rmsnorm_rope_cache(
                    &cmd_buf,
                    pipelines,
                    buf,
                    if use_qkv_prefill_projection {
                        scratch.qkv_out
                    } else {
                        scratch.normed_hidden
                    },
                    weights.qkv_offset,
                    q_norm_offset,
                    k_norm_offset,
                    v_gamma_offset,
                    scratch.q_offset,
                    scratch.k_offset,
                    scratch.v_offset,
                    meta.cos_offset,
                    meta.sin_offset,
                    meta.positions_offset,
                    meta.slot_mapping_offset,
                    kv_cache_k_offset,
                    kv_cache_v_offset,
                    num_tokens,
                    hidden,
                    dims.head_dim,
                    dims.num_heads,
                    dims.num_kv_heads,
                    0,
                    q_dim,
                    q_dim + kv_dim,
                    dims.rms_eps,
                    weights.v_norm_offset.is_some(),
                    dims.rope_dim,
                    "qkv_headwise_norm_rope_cache",
                    use_qkv_prefill_projection,
                )?;
            } else {
                encode_qkv_headwise_rmsnorm(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.normed_hidden,
                    weights.qkv_offset,
                    q_norm_offset,
                    k_norm_offset,
                    v_gamma_offset,
                    scratch.q_offset,
                    scratch.k_offset,
                    scratch.v_offset,
                    num_tokens,
                    hidden,
                    dims.head_dim,
                    dims.num_heads,
                    dims.num_kv_heads,
                    0,
                    q_dim,
                    q_dim + kv_dim,
                    dims.rms_eps,
                    weights.v_norm_offset.is_some(),
                    "qkv_headwise_norm",
                )?;
            }
        } else {
            encode_gemm_headwise_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.normed_hidden,
                weights.qkv_offset,
                q_norm_offset,
                scratch.q_offset,
                num_tokens,
                hidden,
                dims.head_dim,
                dims.num_heads,
                0,
                dims.rms_eps,
                "q_norm",
            )?;
        }
        if trace.is_some() && !debug_skip.skip_kv_projection {
            encode_gemm_headwise_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.normed_hidden,
                weights.qkv_offset,
                k_norm_offset,
                scratch.k_offset,
                num_tokens,
                hidden,
                dims.head_dim,
                dims.num_kv_heads,
                q_dim,
                dims.rms_eps,
                "k_norm",
            )?;
            if let Some(v_norm_offset) = weights.v_norm_offset {
                encode_gemm_headwise_rmsnorm(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.normed_hidden,
                    weights.qkv_offset,
                    v_norm_offset,
                    scratch.v_offset,
                    num_tokens,
                    hidden,
                    dims.head_dim,
                    dims.num_kv_heads,
                    q_dim + kv_dim,
                    dims.rms_eps,
                    "v_norm",
                )?;
            } else {
                // Gemma 4 uses an unscaled RMSNorm for V; there is no v_norm.weight
                // tensor in the checkpoint.
                encode_gemm_headwise_rmsnorm_unit(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.normed_hidden,
                    weights.qkv_offset,
                    scratch.v_offset,
                    num_tokens,
                    hidden,
                    dims.head_dim,
                    dims.num_kv_heads,
                    q_dim + kv_dim,
                    dims.rms_eps,
                    "v_norm_unit",
                )?;
            }
        }
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.q_offset,
                trace.after_q_norm,
                num_tokens * q_dim,
                "trace_after_q_norm",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.k_offset,
                trace.after_k_norm,
                num_tokens * kv_dim,
                "trace_after_k_norm",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.v_offset,
                trace.after_v_norm,
                num_tokens * kv_dim,
                "trace_after_v_norm",
            )?;
        }
    } else {
        encode_gemm_with_output(
            &cmd_buf,
            pipelines,
            buf,
            scratch.normed_hidden,
            weights.qkv_offset,
            scratch.qkv_out,
            num_tokens,
            qkv_n,
            hidden,
            1.0,
            0.0,
            false,
            allow_prefill_mma,
        )?;

        encode_split_qkv(
            &cmd_buf,
            pipelines,
            buf,
            scratch.qkv_out,
            scratch.q_offset,
            scratch.k_offset,
            scratch.v_offset,
            num_tokens,
            q_dim,
            kv_dim,
        )?;
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.q_offset,
                trace.q_projection,
                num_tokens * q_dim,
                "trace_q_projection",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.k_offset,
                trace.k_projection,
                num_tokens * kv_dim,
                "trace_k_projection",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.v_offset,
                trace.v_projection,
                num_tokens * kv_dim,
                "trace_v_projection",
            )?;
        }

        if let Some(q_norm_offset) = weights.q_norm_offset {
            encode_headwise_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.q_offset,
                scratch.q_offset,
                q_norm_offset,
                dims.head_dim,
                dims.num_heads,
                dims.rms_eps,
                num_tokens,
                "q_norm",
            )?;
        }
        if !debug_skip.skip_kv_projection {
            if let Some(k_norm_offset) = weights.k_norm_offset {
                encode_headwise_rmsnorm(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.k_offset,
                    scratch.k_offset,
                    k_norm_offset,
                    dims.head_dim,
                    dims.num_kv_heads,
                    dims.rms_eps,
                    num_tokens,
                    "k_norm",
                )?;
            }
            if let Some(v_norm_offset) = weights.v_norm_offset {
                encode_headwise_rmsnorm(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.v_offset,
                    scratch.v_offset,
                    v_norm_offset,
                    dims.head_dim,
                    dims.num_kv_heads,
                    dims.rms_eps,
                    num_tokens,
                    "v_norm",
                )?;
            }
        }
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.q_offset,
                trace.after_q_norm,
                num_tokens * q_dim,
                "trace_after_q_norm",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.k_offset,
                trace.after_k_norm,
                num_tokens * kv_dim,
                "trace_after_k_norm",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.v_offset,
                trace.after_v_norm,
                num_tokens * kv_dim,
                "trace_after_v_norm",
            )?;
        }
    }

    // 5. RoPE: apply partial RoPE to Q and K. The normal Gemma 4 path fuses
    // this into QKV projection/norm above; keep the standalone encoder for
    // trace/debug/fallback paths.
    if !use_fused_qkv_rope_cache {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "rope_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("rope_partial_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.q_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.k_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), meta.cos_offset, 2);
        encoder.setBuffer_offset_atIndex(Some(buf), meta.sin_offset, 3);
        encoder.setBuffer_offset_atIndex(Some(buf), meta.positions_offset, 4);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
            4,
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&dims.num_heads as *const _ as *mut _),
            4,
            6,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&dims.num_kv_heads as *const _ as *mut _),
            4,
            7,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&dims.head_dim as *const _ as *mut _),
            4,
            8,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&dims.rope_dim as *const _ as *mut _),
            4,
            9,
        );
        let half_rope = dims.rope_dim / 2;
        let groups = MTLSize {
            width: num_tokens as usize,
            height: half_rope as usize,
            depth: 1,
        };
        let tpg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    }
    if let Some(trace) = trace {
        encode_trace_copy(
            &cmd_buf,
            pipelines,
            buf,
            scratch.q_offset,
            trace.after_rope_q,
            num_tokens * q_dim,
            "trace_after_rope_q",
        )?;
        encode_trace_copy(
            &cmd_buf,
            pipelines,
            buf,
            scratch.k_offset,
            trace.after_rope_k,
            num_tokens * kv_dim,
            "trace_after_rope_k",
        )?;
    }

    // 6. KV cache write. The normal Gemma 4 path fuses this into QKV
    // projection/norm above; keep the standalone encoder for trace/debug/
    // fallback paths.
    if !use_fused_qkv_rope_cache && !debug_skip.skip_local_kv_cache_write {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "kv_write_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("kv_cache_write_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.k_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.v_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_k_offset, 2);
        encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_v_offset, 3);
        encoder.setBuffer_offset_atIndex(Some(buf), meta.slot_mapping_offset, 4);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
            4,
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&kv_dim as *const _ as *mut _),
            4,
            6,
        );
        let groups = MTLSize {
            width: num_tokens as usize,
            height: kv_dim as usize,
            depth: 1,
        };
        let tpg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    if let Some(profiler) = stage_profiler.as_deref_mut() {
        profiler.end(cmd_buf);
        profiler.begin(
            cmd_buf,
            if dims.attention_window == 0 {
                crate::stage_instrumentation::MetalStage::AttentionFull
            } else {
                crate::stage_instrumentation::MetalStage::AttentionSliding
            },
        );
    }
    // 7. Attention
    match phase {
        MetalPhase::Decode => {
            let donor_attention = crate::donor12b_metal::try_encode_attention(
                pipelines,
                cmd_buf,
                buf,
                dims,
                phase,
                [
                    scratch.q_offset,
                    attention_kv_cache_k_offset,
                    attention_kv_cache_v_offset,
                    scratch.attn_out,
                    meta.block_tables_offset,
                    meta.context_lens_offset,
                    meta.positions_offset,
                ],
            )?;
            // Explicit opt-in only; unsupported requests retain the unchanged
            // incumbent route. The adapter records only actual candidate encodes.
            let split_encoded = if donor_attention {
                None
            } else if let Some(partials) = scratch.global_decode_partials {
                crate::attention_global_decode_metal::try_encode_split_global_decode(
                    pipelines,
                    cmd_buf,
                    buf,
                    dims,
                    phase,
                    crate::attention_global_decode::SplitDecodeBuffers {
                        common: crate::attention_global_decode::DecodeBuffers {
                            q: scratch.q_offset,
                            k: attention_kv_cache_k_offset,
                            v: attention_kv_cache_v_offset,
                            output: scratch.attn_out,
                            block_tables: meta.block_tables_offset,
                            context_lens: meta.context_lens_offset,
                            positions: meta.positions_offset,
                        },
                        partials,
                    },
                    crate::attention_global_decode::DecodeOutput::Bf16,
                )?
            } else {
                None
            };
            if !donor_attention
                && split_encoded.is_none()
                && crate::attention_global_decode_metal::try_encode_global_decode(
                    pipelines,
                    cmd_buf,
                    buf,
                    dims,
                    phase,
                    crate::attention_global_decode::DecodeBuffers {
                        q: scratch.q_offset,
                        k: attention_kv_cache_k_offset,
                        v: attention_kv_cache_v_offset,
                        output: scratch.attn_out,
                        block_tables: meta.block_tables_offset,
                        context_lens: meta.context_lens_offset,
                        positions: meta.positions_offset,
                    },
                    crate::attention_global_decode::DecodeOutput::Bf16,
                )?
                .is_none()
            {
                let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                    rvllm_core::RvllmError::apple(
                        rvllm_core::AppleError::MetalUnavailable,
                        rvllm_core::AppleCtx {
                            backend: "metal",
                            op: "attn_decode",
                            device: "apple-silicon",
                        },
                    )
                })?;
                let use_online = supports_attention_decode_online(dims);
                let pso = pipelines.get(if use_online {
                    "attention_decode_online_f16"
                } else {
                    "attention_decode_f16"
                })?;
                encoder.setComputePipelineState(pso);
                encoder.setBuffer_offset_atIndex(Some(buf), scratch.q_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(buf), attention_kv_cache_k_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(buf), attention_kv_cache_v_offset, 2);
                encoder.setBuffer_offset_atIndex(Some(buf), scratch.attn_out, 3);
                encoder.setBuffer_offset_atIndex(Some(buf), meta.block_tables_offset, 4);
                encoder.setBuffer_offset_atIndex(Some(buf), meta.context_lens_offset, 5);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
                    4,
                    6,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&dims.num_heads as *const _ as *mut _),
                    4,
                    7,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&dims.num_kv_heads as *const _ as *mut _),
                    4,
                    8,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&dims.head_dim as *const _ as *mut _),
                    4,
                    9,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&dims.block_size as *const _ as *mut _),
                    4,
                    10,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(
                        &dims.max_blocks_per_seq as *const _ as *mut _,
                    ),
                    4,
                    11,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&dims.attn_scale as *const _ as *mut _),
                    4,
                    12,
                );
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new_unchecked(&dims.attention_window as *const _ as *mut _),
                    4,
                    13,
                );
                let total_heads = num_tokens * dims.num_heads;
                let groups = MTLSize {
                    width: total_heads as usize,
                    height: 1,
                    depth: 1,
                };
                let tpg = MTLSize {
                    width: if use_online { 32 } else { 1 },
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
                encoder.endEncoding();
            }
        }
        MetalPhase::Prefill {
            max_seqlen_q: _,
            batch_size,
        } => {
            let total_q = num_tokens;
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "attn_prefill",
                        device: "apple-silicon",
                    },
                )
            })?;
            let research_kind = pipelines.kernel_options().research;
            let temporal = research_kind == crate::MetalResearchCandidate::AttentionQ4;
            let research_name = match (temporal, dims.head_dim) {
                (true, 256) => "research_attn_q4_d256",
                (true, _) => "research_attn_q4_d512",
                (false, 256) => "research_gqa_kv8_d256",
                (false, _) => "research_gqa_kv8_d512",
            };
            let research_threads = if temporal || dims.num_kv_heads != 8 {
                128
            } else {
                64
            };
            let research_pso = if batch_size == 1
                && dims.attn_scale == 1.0
                && matches!(
                    research_kind,
                    crate::MetalResearchCandidate::GqaKv8
                        | crate::MetalResearchCandidate::AttentionQ4
                )
                && research_layer_eligible(pipelines, dims, phase, research_kind)
                && (!temporal
                    || crate::research_next::temporal_context_capacity_fits(
                        dims.block_size,
                        dims.max_blocks_per_seq,
                    )) {
                meta.cu_seqlens_offset.and_then(|cu| {
                    let shape = crate::research::GqaBufferShape {
                        tokens: num_tokens,
                        kv_heads: dims.num_kv_heads,
                        head_dim: dims.head_dim,
                        block_size: dims.block_size,
                        max_blocks: dims.max_blocks_per_seq,
                        num_blocks: dims.num_blocks_total,
                    };
                    if !shape.buffers_fit(
                        [
                            scratch.q_offset,
                            attention_kv_cache_k_offset,
                            attention_kv_cache_v_offset,
                            scratch.attn_out,
                            meta.block_tables_offset,
                            meta.context_lens_offset,
                            cu,
                            meta.positions_offset,
                        ],
                        buf.length(),
                    ) {
                        return None;
                    }
                    pipelines.research_pso(
                        research_name,
                        research_threads,
                        if temporal {
                            crate::research_next::temporal_threadgroup_bytes(dims.head_dim)?
                        } else {
                            2 * 8 * dims.head_dim as usize * 2 + 8 * 4
                        },
                    )
                })
            } else {
                None
            };
            let use_research = research_pso.is_some();
            let use_simd =
                trace.is_none() && supports_gemma4_prefill_simd_attention(pipelines, dims);
            let pso = if let Some(pso) = research_pso {
                encoder.setLabel(Some(&objc2_foundation::NSString::from_str(research_name)));
                tracing::debug!(
                    candidate = research_kind.name(),
                    kernel = research_name,
                    tokens = num_tokens,
                    "Research dispatch"
                );
                pso
            } else {
                pipelines.get(if use_simd {
                    "attention_prefill_simdgroup_f16"
                } else {
                    "attention_prefill_f16"
                })?
            };
            encoder.setComputePipelineState(pso);
            encoder.setBuffer_offset_atIndex(Some(buf), scratch.q_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), attention_kv_cache_k_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), attention_kv_cache_v_offset, 2);
            encoder.setBuffer_offset_atIndex(Some(buf), scratch.attn_out, 3);
            encoder.setBuffer_offset_atIndex(Some(buf), meta.block_tables_offset, 4);
            encoder.setBuffer_offset_atIndex(Some(buf), meta.context_lens_offset, 5);
            if let Some(cu_off) = meta.cu_seqlens_offset {
                encoder.setBuffer_offset_atIndex(Some(buf), cu_off, 6);
            }
            encoder.setBuffer_offset_atIndex(Some(buf), meta.positions_offset, 7);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&total_q as *const _ as *mut _),
                4,
                8,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&batch_size as *const _ as *mut _),
                4,
                9,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.num_heads as *const _ as *mut _),
                4,
                10,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.num_kv_heads as *const _ as *mut _),
                4,
                11,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.head_dim as *const _ as *mut _),
                4,
                12,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.block_size as *const _ as *mut _),
                4,
                13,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.max_blocks_per_seq as *const _ as *mut _),
                4,
                14,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.attn_scale as *const _ as *mut _),
                4,
                15,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.attention_window as *const _ as *mut _),
                4,
                16,
            );
            if use_research {
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&dims.num_blocks_total).cast(),
                    4,
                    17,
                );
            }
            let groups = if use_research && temporal {
                MTLSize {
                    width: (total_q as usize).div_ceil(4),
                    height: dims.num_heads as usize,
                    depth: 1,
                }
            } else if use_research {
                MTLSize {
                    width: total_q as usize,
                    height: dims.num_kv_heads as usize,
                    depth: (dims.num_heads / dims.num_kv_heads).div_ceil(4) as usize,
                }
            } else {
                MTLSize {
                    width: total_q as usize,
                    height: dims.num_heads as usize,
                    depth: 1,
                }
            };
            let tpg = MTLSize {
                width: if use_research {
                    research_threads
                } else if use_simd {
                    32
                } else {
                    1
                },
                height: 1,
                depth: 1,
            };
            if use_research || use_simd {
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
            } else {
                encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
            }
            encoder.endEncoding();
            if use_research {
                use crate::research_evidence::ResearchKernel;
                pipelines.record_research_dispatch(match (temporal, dims.head_dim) {
                    (true, 256) => ResearchKernel::Temporal256,
                    (true, _) => ResearchKernel::Temporal512,
                    (false, 256) => ResearchKernel::Gqa256,
                    (false, _) => ResearchKernel::Gqa512,
                });
            }
        }
    }
    if let Some(trace) = trace {
        encode_trace_copy(
            &cmd_buf,
            pipelines,
            buf,
            scratch.attn_out,
            trace.attention_output,
            num_tokens * q_dim,
            "trace_attention_output",
        )?;
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    if let Some(profiler) = stage_profiler.as_deref_mut() {
        profiler.end(cmd_buf);
        profiler.begin(
            cmd_buf,
            if dims.attention_window == 0 {
                crate::stage_instrumentation::MetalStage::OProjectionFull
            } else {
                crate::stage_instrumentation::MetalStage::OProjectionSliding
            },
        );
    }
    // 8. O projection, post-attention norm, then residual add.
    let attn_addition_offset = if let Some(post_attn_norm_offset) = weights.post_attn_norm_offset {
        if let Some(projection) = weights.low_bit_o_proj {
            if !low_bit_descriptor_matches(
                projection,
                AppleLowBitTensorRole::OutputProjection,
                [hidden, q_dim],
            ) {
                return Err(rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::InvalidWeightBlob {
                        reason: "low-bit output projection shape does not match prepared layer",
                    },
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "low_bit_o_projection_shape",
                        device: "apple-silicon",
                    },
                ));
            }
            encode_low_bit_projection_strided(
                &cmd_buf,
                pipelines,
                buf,
                projection,
                scratch.attn_out,
                scratch.mlp_out,
                num_tokens,
                hidden,
                0,
                phase,
                dims,
            )?;
            encode_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.mlp_out,
                scratch.normed_hidden,
                post_attn_norm_offset,
                hidden,
                dims.rms_eps,
                num_tokens,
                "post_attn_norm_low_bit",
            )?;
            if let Some(trace) = trace {
                encode_trace_copy(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.mlp_out,
                    trace.after_o_proj,
                    num_tokens * hidden,
                    "trace_after_o_proj",
                )?;
            }
        } else {
            if let Some(trace) = trace {
                encode_gemm_with_output(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.attn_out,
                    weights.o_proj_offset,
                    trace.after_o_proj,
                    num_tokens,
                    hidden,
                    q_dim,
                    1.0,
                    0.0,
                    false,
                    allow_prefill_mma,
                )?;
            }
            encode_gemm_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.attn_out,
                weights.o_proj_offset,
                post_attn_norm_offset,
                scratch.normed_hidden,
                num_tokens,
                hidden,
                q_dim,
                dims.rms_eps,
                "post_attn_norm",
                allow_prefill_mma,
            )?;
        }
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.normed_hidden,
                trace.after_post_attention_layernorm,
                num_tokens * hidden,
                "trace_after_post_attention_layernorm",
            )?;
        }
        scratch.normed_hidden
    } else {
        if let Some(projection) = weights.low_bit_o_proj {
            if !low_bit_descriptor_matches(
                projection,
                AppleLowBitTensorRole::OutputProjection,
                [hidden, q_dim],
            ) {
                return Err(rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::InvalidWeightBlob {
                        reason: "low-bit output projection shape does not match prepared layer",
                    },
                    rvllm_core::AppleCtx {
                        backend: "metal",
                        op: "low_bit_o_projection_shape",
                        device: "apple-silicon",
                    },
                ));
            }
            encode_low_bit_projection_strided(
                &cmd_buf,
                pipelines,
                buf,
                projection,
                scratch.attn_out,
                scratch.mlp_out,
                num_tokens,
                hidden,
                0,
                phase,
                dims,
            )?;
        } else {
            encode_gemm_with_output(
                &cmd_buf,
                pipelines,
                buf,
                scratch.attn_out,
                weights.o_proj_offset,
                scratch.mlp_out,
                num_tokens,
                hidden,
                q_dim,
                1.0,
                0.0,
                false,
                allow_prefill_mma,
            )?;
        }
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.mlp_out,
                trace.after_o_proj,
                num_tokens * hidden,
                "trace_after_o_proj",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.mlp_out,
                trace.after_post_attention_layernorm,
                num_tokens * hidden,
                "trace_after_post_attention_layernorm",
            )?;
        }
        scratch.mlp_out
    };
    // Fuse the half-rounded attention residual add with the immediately
    // following pre-FFN RMSNorm. The kernel stores the updated residual before
    // accumulating the RMS sum, preserving the prior two-kernel rounding order.
    encode_residual_add_rmsnorm(
        &cmd_buf,
        pipelines,
        buf,
        residual_offset,
        attn_addition_offset,
        scratch.normed_hidden,
        weights
            .pre_ff_norm_offset
            .unwrap_or(weights.mlp_norm_offset),
        hidden,
        dims.rms_eps,
        num_tokens,
        None,
        0,
    )?;
    if let Some(trace) = trace {
        encode_trace_copy(
            &cmd_buf,
            pipelines,
            buf,
            scratch.normed_hidden,
            trace.after_pre_feedforward_layernorm,
            num_tokens * hidden,
            "trace_after_pre_feedforward_layernorm",
        )?;
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    if let Some(profiler) = stage_profiler.as_deref_mut() {
        profiler.end(cmd_buf);
        profiler.begin(
            cmd_buf,
            crate::stage_instrumentation::MetalStage::FfnGateUpActivation,
        );
    }
    // 11. Dense FFN branch: Gate||Up projection, GELU, Down projection.
    let two_inter = 2 * dims.intermediate;
    let has_per_layer_input_branch = dims.ple_dim > 0
        && weights.per_layer_inputs_offset.is_some()
        && weights.per_layer_input_gate_offset.is_some()
        && weights.per_layer_projection_offset.is_some()
        && weights.post_per_layer_input_norm_offset.is_some();
    let mut layer_scale_fused = false;
    let donor_gate = crate::donor12b_metal::try_encode_gate(
        pipelines,
        cmd_buf,
        buf,
        dims,
        phase,
        weights,
        scratch,
        trace.is_some(),
    )?;
    let bf16_gate = !donor_gate
        && crate::research_decode_metal::try_encode_gate_up(
            pipelines,
            cmd_buf,
            &buf,
            research_bf16_gate_request(
                pipelines,
                dims,
                phase,
                weights,
                scratch,
                trace.is_some(),
                buf.length(),
            ),
        )?;
    let rounded_gate = !donor_gate
        && !bf16_gate
        && low_bit_gate_up.is_none()
        && supports_research_rounded_gate(
            pipelines,
            dims,
            phase,
            weights,
            scratch,
            trace.is_some(),
            buf.length(),
        )
        && try_encode_research_rounded_gate(
            cmd_buf,
            pipelines,
            buf,
            dims,
            weights,
            scratch,
            trace.is_some(),
        )?;
    if !donor_gate {
        if let Some((gate_projection, up_projection)) = low_bit_gate_up {
            encode_low_bit_projection_strided(
                &cmd_buf,
                pipelines,
                buf,
                gate_projection,
                scratch.normed_hidden,
                scratch.gate_up_out,
                num_tokens,
                two_inter,
                0,
                phase,
                dims,
            )?;
            encode_low_bit_projection_strided(
                &cmd_buf,
                pipelines,
                buf,
                up_projection,
                scratch.normed_hidden,
                scratch.gate_up_out,
                num_tokens,
                two_inter,
                dims.intermediate,
                phase,
                dims,
            )?;
        } else if !rounded_gate && !bf16_gate {
            encode_gemm_with_output(
                &cmd_buf,
                pipelines,
                buf,
                scratch.normed_hidden,
                weights.gate_up_offset,
                scratch.gate_up_out,
                num_tokens,
                two_inter,
                hidden,
                1.0,
                0.0,
                false,
                allow_prefill_mma,
            )?;
        }
    }
    if let Some(trace) = trace {
        encode_trace_copy(
            &cmd_buf,
            pipelines,
            buf,
            scratch.gate_up_out,
            trace.gate_up_out,
            num_tokens * two_inter,
            "trace_gate_up_out",
        )?;
    }

    if !donor_gate && !rounded_gate && !bf16_gate {
        encode_gelu_mul(
            &cmd_buf,
            pipelines,
            buf,
            scratch.gate_up_out,
            scratch.activated,
            num_tokens,
            dims.intermediate,
            "gelu_mul",
        )?;
    }
    if let Some(trace) = trace {
        encode_trace_copy(
            &cmd_buf,
            pipelines,
            buf,
            scratch.activated,
            trace.ffn_activation,
            num_tokens * dims.intermediate,
            "trace_ffn_activation",
        )?;
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    if let Some(profiler) = stage_profiler.as_deref_mut() {
        profiler.end(cmd_buf);
        profiler.begin(cmd_buf, crate::stage_instrumentation::MetalStage::FfnDown);
    }
    let ffn_addition_offset = if let Some(moe) = weights.moe {
        let Some(topk_indices_offset) = scratch.moe_topk_indices else {
            return Err(missing_moe_scratch("moe_topk_indices"));
        };
        let Some(topk_weights_offset) = scratch.moe_topk_weights else {
            return Err(missing_moe_scratch("moe_topk_weights"));
        };
        let Some(moe_activated_offset) = scratch.moe_activated else {
            return Err(missing_moe_scratch("moe_activated"));
        };
        let Some(moe_out_offset) = scratch.moe_out else {
            return Err(missing_moe_scratch("moe_out"));
        };
        if dims.moe_num_experts == 0 || dims.moe_top_k == 0 || dims.moe_intermediate == 0 {
            return Err(missing_moe_scratch("moe_dims"));
        }

        // HF Gemma4 MoE keeps the dense MLP and adds a separately routed
        // expert branch. Dense branch uses post_feedforward_layernorm_1.
        if let Some(projection) = weights.low_bit_down_proj {
            encode_low_bit_down_projection(
                &cmd_buf,
                pipelines,
                buf,
                projection,
                scratch.activated,
                scratch.mlp_out,
                num_tokens,
                phase,
                dims,
            )?;
            encode_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.mlp_out,
                scratch.normed_hidden,
                moe.post_ff1_norm_offset,
                hidden,
                dims.rms_eps,
                num_tokens,
                "post_ff1_norm_low_bit",
            )?;
        } else {
            encode_gemm_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.activated,
                native_down_proj_offset,
                moe.post_ff1_norm_offset,
                scratch.normed_hidden,
                num_tokens,
                hidden,
                dims.intermediate,
                dims.rms_eps,
                "post_ff1_norm",
                allow_prefill_mma,
            )?;
        }
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.normed_hidden,
                trace.after_ffn_branch,
                num_tokens * hidden,
                "trace_after_ffn_branch",
            )?;
        }

        encode_moe_router_topk(
            &cmd_buf,
            pipelines,
            buf,
            residual_offset,
            moe.router_proj_offset,
            moe.router_scale_offset,
            moe.router_per_expert_scale_offset,
            topk_indices_offset,
            topk_weights_offset,
            num_tokens,
            hidden,
            dims.moe_num_experts,
            dims.moe_top_k,
            dims.rms_eps,
        )?;
        encode_rmsnorm(
            &cmd_buf,
            pipelines,
            buf,
            residual_offset,
            scratch.mlp_out,
            moe.pre_ff2_norm_offset,
            hidden,
            dims.rms_eps,
            num_tokens,
            "pre_ff2_norm",
        )?;
        encode_moe_expert_gate_up(
            &cmd_buf,
            pipelines,
            buf,
            scratch.mlp_out,
            moe.expert_gate_up_offset,
            topk_indices_offset,
            topk_weights_offset,
            moe_activated_offset,
            num_tokens,
            hidden,
            dims.moe_num_experts,
            dims.moe_top_k,
            dims.moe_intermediate,
        )?;
        encode_moe_expert_down(
            &cmd_buf,
            pipelines,
            buf,
            moe_activated_offset,
            moe.expert_down_offset,
            topk_indices_offset,
            moe_out_offset,
            num_tokens,
            hidden,
            dims.moe_num_experts,
            dims.moe_top_k,
            dims.moe_intermediate,
        )?;
        encode_rmsnorm(
            &cmd_buf,
            pipelines,
            buf,
            moe_out_offset,
            moe_out_offset,
            moe.post_ff2_norm_offset,
            hidden,
            dims.rms_eps,
            num_tokens,
            "post_ff2_norm",
        )?;

        // Sum the normalized dense and expert branches, apply the final
        // post_feedforward_layernorm, then add the result to the residual.
        encode_residual_add_rmsnorm(
            &cmd_buf,
            pipelines,
            buf,
            scratch.normed_hidden,
            moe_out_offset,
            scratch.mlp_out,
            weights
                .post_ff_norm_offset
                .unwrap_or(weights.mlp_norm_offset),
            hidden,
            dims.rms_eps,
            num_tokens,
            None,
            0,
        )?;
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.mlp_out,
                trace.after_post_feedforward_layernorm,
                num_tokens * hidden,
                "trace_after_post_feedforward_layernorm",
            )?;
        }
        scratch.mlp_out
    } else if let Some(post_ff_norm_offset) = weights.post_ff_norm_offset {
        if let Some(projection) = weights.low_bit_down_proj {
            let raw_output = trace
                .map(|trace| trace.after_ffn_branch)
                .unwrap_or(scratch.mlp_out);
            encode_low_bit_down_projection(
                &cmd_buf,
                pipelines,
                buf,
                projection,
                scratch.activated,
                raw_output,
                num_tokens,
                phase,
                dims,
            )?;
            encode_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                raw_output,
                scratch.normed_hidden,
                post_ff_norm_offset,
                hidden,
                dims.rms_eps,
                num_tokens,
                "post_ff_norm_low_bit",
            )?;
        } else {
            if let Some(trace) = trace {
                encode_gemm_with_output(
                    &cmd_buf,
                    pipelines,
                    buf,
                    scratch.activated,
                    native_down_proj_offset,
                    trace.after_ffn_branch,
                    num_tokens,
                    hidden,
                    dims.intermediate,
                    1.0,
                    0.0,
                    false,
                    allow_prefill_mma,
                )?;
            }
            encode_gemm_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.activated,
                native_down_proj_offset,
                post_ff_norm_offset,
                scratch.normed_hidden,
                num_tokens,
                hidden,
                dims.intermediate,
                dims.rms_eps,
                "post_ff_norm",
                allow_prefill_mma,
            )?;
        }
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.normed_hidden,
                trace.after_post_feedforward_layernorm,
                num_tokens * hidden,
                "trace_after_post_feedforward_layernorm",
            )?;
        }
        scratch.normed_hidden
    } else {
        if let Some(projection) = weights.low_bit_down_proj {
            encode_low_bit_down_projection(
                &cmd_buf,
                pipelines,
                buf,
                projection,
                scratch.activated,
                scratch.mlp_out,
                num_tokens,
                phase,
                dims,
            )?;
        } else {
            encode_gemm_with_output(
                &cmd_buf,
                pipelines,
                buf,
                scratch.activated,
                native_down_proj_offset,
                scratch.mlp_out,
                num_tokens,
                hidden,
                dims.intermediate,
                1.0,
                0.0,
                false,
                allow_prefill_mma,
            )?;
        }
        if let Some(trace) = trace {
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.mlp_out,
                trace.after_ffn_branch,
                num_tokens * hidden,
                "trace_after_ffn_branch",
            )?;
            encode_trace_copy(
                &cmd_buf,
                pipelines,
                buf,
                scratch.mlp_out,
                trace.after_post_feedforward_layernorm,
                num_tokens * hidden,
                "trace_after_post_feedforward_layernorm",
            )?;
        }
        scratch.mlp_out
    };
    if !has_per_layer_input_branch {
        if let Some(layer_scalar_offset) = weights.layer_scalar_offset {
            encode_residual_add_then_scale(
                &cmd_buf,
                pipelines,
                buf,
                residual_offset,
                ffn_addition_offset,
                num_tokens * hidden,
                hidden,
                layer_scalar_offset,
                weights.layer_scalar_dim,
            )?;
            layer_scale_fused = true;
        } else {
            encode_residual_add(
                &cmd_buf,
                pipelines,
                buf,
                residual_offset,
                ffn_addition_offset,
                num_tokens * hidden,
                hidden,
                None,
                0,
            )?;
        }
    } else {
        encode_residual_add(
            &cmd_buf,
            pipelines,
            buf,
            residual_offset,
            ffn_addition_offset,
            num_tokens * hidden,
            hidden,
            None,
            0,
        )?;
    }

    if let (
        Some(per_layer_inputs_offset),
        Some(per_layer_input_gate_offset),
        Some(per_layer_projection_offset),
        Some(post_per_layer_input_norm_offset),
    ) = (
        weights.per_layer_inputs_offset,
        weights.per_layer_input_gate_offset,
        weights.per_layer_projection_offset,
        weights.post_per_layer_input_norm_offset,
    ) {
        if has_per_layer_input_branch {
            if let Some(trace) = trace {
                if let Some(per_layer_input) = trace.per_layer_input {
                    encode_trace_copy_ple_layer(
                        &cmd_buf,
                        pipelines,
                        buf,
                        per_layer_inputs_offset,
                        per_layer_input,
                        num_tokens,
                        dims.num_layers,
                        dims.layer_idx,
                        dims.ple_dim,
                    )?;
                }
            }
            encode_gemm_with_output(
                &cmd_buf,
                pipelines,
                buf,
                residual_offset,
                per_layer_input_gate_offset,
                scratch.activated,
                num_tokens,
                dims.ple_dim,
                hidden,
                1.0,
                0.0,
                false,
                allow_prefill_mma,
            )?;
            if let Some(trace) = trace {
                if let Some(per_layer_input_gate) = trace.per_layer_input_gate {
                    encode_trace_copy(
                        &cmd_buf,
                        pipelines,
                        buf,
                        scratch.activated,
                        per_layer_input_gate,
                        num_tokens * dims.ple_dim,
                        "trace_per_layer_input_gate",
                    )?;
                }
            }
            encode_ple_gelu_mul(
                &cmd_buf,
                pipelines,
                buf,
                scratch.activated,
                per_layer_inputs_offset,
                num_tokens,
                dims.num_layers,
                dims.layer_idx,
                dims.ple_dim,
            )?;
            if let Some(trace) = trace {
                if let Some(per_layer_projection) = trace.per_layer_projection {
                    encode_gemm_with_output(
                        &cmd_buf,
                        pipelines,
                        buf,
                        scratch.activated,
                        per_layer_projection_offset,
                        per_layer_projection,
                        num_tokens,
                        hidden,
                        dims.ple_dim,
                        1.0,
                        0.0,
                        false,
                        allow_prefill_mma,
                    )?;
                }
            }
            encode_gemm_rmsnorm(
                &cmd_buf,
                pipelines,
                buf,
                scratch.activated,
                per_layer_projection_offset,
                post_per_layer_input_norm_offset,
                scratch.normed_hidden,
                num_tokens,
                hidden,
                dims.ple_dim,
                dims.rms_eps,
                "post_per_layer_input_norm",
                allow_prefill_mma,
            )?;
            if let Some(trace) = trace {
                if let Some(post_per_layer_input_norm) = trace.post_per_layer_input_norm {
                    encode_trace_copy(
                        &cmd_buf,
                        pipelines,
                        buf,
                        scratch.normed_hidden,
                        post_per_layer_input_norm,
                        num_tokens * hidden,
                        "trace_post_per_layer_input_norm",
                    )?;
                }
            }
            if let Some(layer_scalar_offset) = weights.layer_scalar_offset {
                encode_residual_add_then_scale(
                    &cmd_buf,
                    pipelines,
                    buf,
                    residual_offset,
                    scratch.normed_hidden,
                    num_tokens * hidden,
                    hidden,
                    layer_scalar_offset,
                    weights.layer_scalar_dim,
                )?;
                layer_scale_fused = true;
            } else {
                encode_residual_add(
                    &cmd_buf,
                    pipelines,
                    buf,
                    residual_offset,
                    scratch.normed_hidden,
                    num_tokens * hidden,
                    hidden,
                    None,
                    0,
                )?;
            }
        }
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    if let Some(profiler) = stage_profiler.as_deref_mut() {
        profiler.end(cmd_buf);
        profiler.begin(
            cmd_buf,
            crate::stage_instrumentation::MetalStage::NormResidual,
        );
    }
    if !layer_scale_fused {
        encode_layer_scale(
            &cmd_buf,
            pipelines,
            buf,
            residual_offset,
            num_tokens * hidden,
            hidden,
            weights.layer_scalar_offset,
            weights.layer_scalar_dim,
        )?;
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    if let Some(profiler) = stage_profiler.as_deref_mut() {
        profiler.end(cmd_buf);
    }
    Ok(())
}

unsafe fn encode_trace_copy(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    src_offset: usize,
    dst_offset: usize,
    len: u32,
    op: &'static str,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("copy_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), src_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), dst_offset, 1);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&len as *const _ as *mut _),
        4,
        2,
    );
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: len as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_trace_copy_ple_layer(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    packed_ple_offset: usize,
    dst_offset: usize,
    num_tokens: u32,
    num_layers: u32,
    layer_idx: u32,
    ple_dim: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "trace_copy_ple_layer",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("copy_ple_layer_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), packed_ple_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), dst_offset, 1);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        2,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_layers as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&layer_idx as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&ple_dim as *const _ as *mut _),
        4,
        5,
    );
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: num_tokens as usize,
            height: ple_dim as usize,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

fn missing_moe_scratch(name: &'static str) -> rvllm_core::RvllmError {
    rvllm_core::RvllmError::apple(
        rvllm_core::AppleError::FeatureNotAvailable {
            backend: "metal",
            op: name,
        },
        rvllm_core::AppleCtx {
            backend: "metal",
            op: "moe_scratch",
            device: "apple-silicon",
        },
    )
}

fn validate_down_projection_sources(
    native_offset: Option<usize>,
    has_low_bit: bool,
) -> Result<Option<usize>> {
    match (native_offset, has_low_bit) {
        (Some(offset), false) => Ok(Some(offset)),
        (None, true) => Ok(None),
        (Some(_), true) => Err(rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::InvalidWeightBlob {
                reason: "layer has both native and low-bit down projection execution sources",
            },
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "down_projection_source",
                device: "apple-silicon",
            },
        )),
        (None, false) => Err(rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::InvalidWeightBlob {
                reason: "layer has no down projection execution source",
            },
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "down_projection_source",
                device: "apple-silicon",
            },
        )),
    }
}

pub(crate) unsafe fn encode_gelu_mul(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    gate_up_offset: usize,
    output_offset: usize,
    num_tokens: u32,
    intermediate: u32,
    op: &'static str,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("gelu_mul_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), gate_up_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), output_offset, 1);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        2,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&intermediate as *const _ as *mut _),
        4,
        3,
    );
    let groups = MTLSize {
        width: num_tokens as usize,
        height: intermediate as usize,
        depth: 1,
    };
    // Each thread owns one independent element. Pack contiguous dimensions
    // into full groups without changing arithmetic, indexing or rounding.
    let tpg = MTLSize {
        width: 1,
        height: 256,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_moe_router_topk(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    hidden_offset: usize,
    router_proj_offset: usize,
    router_scale_offset: usize,
    router_per_expert_scale_offset: usize,
    topk_indices_offset: usize,
    topk_weights_offset: usize,
    num_tokens: u32,
    hidden: u32,
    num_experts: u32,
    top_k: u32,
    rms_eps: f32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "moe_router_topk",
                device: "apple-silicon",
            },
        )
    })?;
    let scalar_root_size = 1.0f32 / (hidden as f32).sqrt();
    let pso = pipelines.get("moe_router_topk_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), hidden_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), router_proj_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), router_scale_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), router_per_expert_scale_offset, 3);
    encoder.setBuffer_offset_atIndex(Some(buf), topk_indices_offset, 4);
    encoder.setBuffer_offset_atIndex(Some(buf), topk_weights_offset, 5);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        6,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        7,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_experts as *const _ as *mut _),
        4,
        8,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&top_k as *const _ as *mut _),
        4,
        9,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&rms_eps as *const _ as *mut _),
        4,
        10,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&scalar_root_size as *const _ as *mut _),
        4,
        11,
    );
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: num_tokens as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_moe_expert_gate_up(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    hidden_offset: usize,
    expert_gate_up_offset: usize,
    topk_indices_offset: usize,
    topk_weights_offset: usize,
    activated_offset: usize,
    num_tokens: u32,
    hidden: u32,
    num_experts: u32,
    top_k: u32,
    intermediate: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "moe_expert_gate_up",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("moe_expert_gate_up_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), hidden_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), expert_gate_up_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), topk_indices_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), topk_weights_offset, 3);
    encoder.setBuffer_offset_atIndex(Some(buf), activated_offset, 4);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        5,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        6,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_experts as *const _ as *mut _),
        4,
        7,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&top_k as *const _ as *mut _),
        4,
        8,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&intermediate as *const _ as *mut _),
        4,
        9,
    );
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: num_tokens as usize,
            height: top_k as usize,
            depth: intermediate as usize,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_moe_expert_down(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    activated_offset: usize,
    expert_down_offset: usize,
    topk_indices_offset: usize,
    output_offset: usize,
    num_tokens: u32,
    hidden: u32,
    num_experts: u32,
    top_k: u32,
    intermediate: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "moe_expert_down",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("moe_expert_down_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), activated_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), expert_down_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), topk_indices_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), output_offset, 3);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        5,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_experts as *const _ as *mut _),
        4,
        6,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&top_k as *const _ as *mut _),
        4,
        7,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&intermediate as *const _ as *mut _),
        4,
        8,
    );
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: num_tokens as usize,
            height: hidden as usize,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

// Preserve the incumbent internal helper API for unrelated callers/fixtures.
unsafe fn encode_rmsnorm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    input_offset: usize,
    output_offset: usize,
    gamma_offset: usize,
    hidden: u32,
    eps: f32,
    num_tokens: u32,
    op: &'static str,
) -> Result<()> {
    encode_rmsnorm_with_policy(
        cmd_buf,
        pipelines,
        buf,
        input_offset,
        output_offset,
        gamma_offset,
        hidden,
        eps,
        num_tokens,
        op,
        false,
    )
}

unsafe fn encode_rmsnorm_with_policy(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    input_offset: usize,
    output_offset: usize,
    gamma_offset: usize,
    hidden: u32,
    eps: f32,
    num_tokens: u32,
    op: &'static str,
    full_prefill_projection: bool,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let decision = crate::research_projection::postnorm_plan(
        pipelines.kernel_options().research,
        full_prefill_projection,
        pipelines.float_type() == Some(crate::MetalFloatType::Bf16)
            && !pipelines.kernel_options().quantized_bf16_accumulation,
        [input_offset, output_offset, gamma_offset],
        num_tokens,
        hidden,
        eps,
        buf.length(),
    );
    let research = decision.ok().and_then(|plan| {
        let (threads, shared) = plan.kernel.limits();
        pipelines
            .research_pso(plan.kernel.name(), threads, shared)
            .map(|pso| (plan, pso))
    });
    let pso = if let Some((plan, pso)) = research {
        encoder.setLabel(Some(&objc2_foundation::NSString::from_str(
            plan.kernel.name(),
        )));
        pso
    } else {
        pipelines.get("rmsnorm_f16")?
    };
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), input_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), output_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), gamma_offset, 2);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
        4,
        4,
    );
    if research.is_some_and(|(plan, _)| plan.binds_token_count) {
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&num_tokens).cast(), 4, 5);
    }
    let tpg = MTLSize {
        width: research.map_or(256, |(plan, _)| plan.kernel.limits().0),
        height: 1,
        depth: 1,
    };
    let groups = MTLSize {
        width: num_tokens as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    if let Some((plan, _)) = research {
        pipelines.record_research_dispatch(plan.kernel);
        tracing::debug!(
            candidate = plan.kernel.owner().name(),
            num_tokens,
            hidden,
            op,
            "Research dispatch"
        );
    }
    Ok(())
}

unsafe fn encode_headwise_rmsnorm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    input_offset: usize,
    output_offset: usize,
    gamma_offset: usize,
    head_dim: u32,
    num_heads: u32,
    eps: f32,
    num_tokens: u32,
    op: &'static str,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("rmsnorm_headwise_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), input_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), output_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), gamma_offset, 2);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&head_dim as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_heads as *const _ as *mut _),
        4,
        5,
    );
    let groups = MTLSize {
        width: num_tokens.saturating_mul(num_heads) as usize,
        height: 1,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_headwise_rmsnorm_unit(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    input_offset: usize,
    output_offset: usize,
    head_dim: u32,
    num_heads: u32,
    eps: f32,
    num_tokens: u32,
    op: &'static str,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    encoder.setComputePipelineState(pipelines.get("rmsnorm_headwise_unit_f16")?);
    encoder.setBuffer_offset_atIndex(Some(buf), input_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), output_offset, 1);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&head_dim).cast(), 4, 2);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&eps).cast(), 4, 3);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&num_heads).cast(), 4, 4);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: num_tokens.saturating_mul(num_heads) as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_gemm_rmsnorm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    gamma_offset: usize,
    c_offset: usize,
    m: u32,
    n: u32,
    k: u32,
    eps: f32,
    op: &'static str,
    allow_prefill_mma: bool,
) -> Result<()> {
    if metal_gemm_rmsnorm_encoder_count(m, n, k, allow_prefill_mma) == 2 {
        // The fused kernel has one threadgroup per token. That is useful for
        // large-prefill locality, but severely under-occupies decode-sized
        // projections. Materializing the dtype-rounded projection, from GEMV
        // or native matrix instructions, preserves the two-op model boundary.
        encode_gemm_with_output(
            cmd_buf,
            pipelines,
            buf,
            a_offset,
            b_offset,
            c_offset,
            m,
            n,
            k,
            1.0,
            0.0,
            false,
            allow_prefill_mma,
        )?;
        return encode_rmsnorm_with_policy(
            cmd_buf,
            pipelines,
            buf,
            c_offset,
            c_offset,
            gamma_offset,
            n,
            eps,
            m,
            op,
            allow_prefill_mma,
        );
    }

    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("gemm_rmsnorm_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), gamma_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), c_offset, 3);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&m as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&n as *const _ as *mut _),
        4,
        5,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&k as *const _ as *mut _),
        4,
        6,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
        4,
        7,
    );
    let groups = MTLSize {
        width: m as usize,
        height: 1,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_gemm_headwise_rmsnorm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    gamma_offset: usize,
    c_offset: usize,
    m: u32,
    k: u32,
    head_dim: u32,
    num_heads: u32,
    b_row_offset: u32,
    eps: f32,
    op: &'static str,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("gemm_headwise_rmsnorm_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), gamma_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), c_offset, 3);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&m as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&k as *const _ as *mut _),
        4,
        5,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&head_dim as *const _ as *mut _),
        4,
        6,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_heads as *const _ as *mut _),
        4,
        7,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&b_row_offset as *const _ as *mut _),
        4,
        8,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
        4,
        9,
    );
    let groups = MTLSize {
        width: m.saturating_mul(num_heads) as usize,
        height: 1,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_gemm_headwise_rmsnorm_unit(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    c_offset: usize,
    m: u32,
    k: u32,
    head_dim: u32,
    num_heads: u32,
    b_row_offset: u32,
    eps: f32,
    op: &'static str,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("gemm_headwise_rmsnorm_unit_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), c_offset, 2);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&m as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&k as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&head_dim as *const _ as *mut _),
        4,
        5,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_heads as *const _ as *mut _),
        4,
        6,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&b_row_offset as *const _ as *mut _),
        4,
        7,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
        4,
        8,
    );
    let groups = MTLSize {
        width: m.saturating_mul(num_heads) as usize,
        height: 1,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_qkv_headwise_rmsnorm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    q_gamma_offset: usize,
    k_gamma_offset: usize,
    v_gamma_offset: usize,
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    m: u32,
    hidden_k: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    q_row_offset: u32,
    k_row_offset: u32,
    v_row_offset: u32,
    eps: f32,
    v_has_gamma: bool,
    op: &'static str,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("qkv_headwise_rmsnorm_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), q_gamma_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), k_gamma_offset, 3);
    encoder.setBuffer_offset_atIndex(Some(buf), v_gamma_offset, 4);
    encoder.setBuffer_offset_atIndex(Some(buf), q_offset, 5);
    encoder.setBuffer_offset_atIndex(Some(buf), k_offset, 6);
    encoder.setBuffer_offset_atIndex(Some(buf), v_offset, 7);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&m as *const _ as *mut _),
        4,
        8,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden_k as *const _ as *mut _),
        4,
        9,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&head_dim as *const _ as *mut _),
        4,
        10,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_q_heads as *const _ as *mut _),
        4,
        11,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_kv_heads as *const _ as *mut _),
        4,
        12,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&q_row_offset as *const _ as *mut _),
        4,
        13,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&k_row_offset as *const _ as *mut _),
        4,
        14,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&v_row_offset as *const _ as *mut _),
        4,
        15,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
        4,
        16,
    );
    let v_has_gamma_u32 = u32::from(v_has_gamma);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&v_has_gamma_u32 as *const _ as *mut _),
        4,
        17,
    );
    let groups = MTLSize {
        width: m.saturating_mul(num_q_heads + 2 * num_kv_heads) as usize,
        height: 1,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_qkv_headwise_rmsnorm_rope_cache(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    q_gamma_offset: usize,
    k_gamma_offset: usize,
    v_gamma_offset: usize,
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    cos_offset: usize,
    sin_offset: usize,
    positions_offset: usize,
    slot_mapping_offset: usize,
    kv_cache_k_offset: usize,
    kv_cache_v_offset: usize,
    m: u32,
    hidden_k: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    q_row_offset: u32,
    k_row_offset: u32,
    v_row_offset: u32,
    eps: f32,
    v_has_gamma: bool,
    rope_dim: u32,
    op: &'static str,
    projected_f32: bool,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op,
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get(if projected_f32 {
        "qkv_projected_rmsnorm_rope_cache_f16"
    } else {
        "qkv_headwise_rmsnorm_rope_cache_f16"
    })?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), q_gamma_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), k_gamma_offset, 3);
    encoder.setBuffer_offset_atIndex(Some(buf), v_gamma_offset, 4);
    encoder.setBuffer_offset_atIndex(Some(buf), q_offset, 5);
    encoder.setBuffer_offset_atIndex(Some(buf), k_offset, 6);
    encoder.setBuffer_offset_atIndex(Some(buf), v_offset, 7);
    encoder.setBuffer_offset_atIndex(Some(buf), cos_offset, 8);
    encoder.setBuffer_offset_atIndex(Some(buf), sin_offset, 9);
    encoder.setBuffer_offset_atIndex(Some(buf), positions_offset, 10);
    encoder.setBuffer_offset_atIndex(Some(buf), slot_mapping_offset, 11);
    encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_k_offset, 12);
    encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_v_offset, 13);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&m as *const _ as *mut _),
        4,
        14,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden_k as *const _ as *mut _),
        4,
        15,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&head_dim as *const _ as *mut _),
        4,
        16,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_q_heads as *const _ as *mut _),
        4,
        17,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_kv_heads as *const _ as *mut _),
        4,
        18,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&q_row_offset as *const _ as *mut _),
        4,
        19,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&k_row_offset as *const _ as *mut _),
        4,
        20,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&v_row_offset as *const _ as *mut _),
        4,
        21,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
        4,
        22,
    );
    let v_has_gamma_u32 = u32::from(v_has_gamma);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&v_has_gamma_u32 as *const _ as *mut _),
        4,
        23,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&rope_dim as *const _ as *mut _),
        4,
        24,
    );
    let groups = MTLSize {
        width: m.saturating_mul(num_q_heads + 2 * num_kv_heads) as usize,
        height: 1,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_residual_add(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    residual_offset: usize,
    addition_offset: usize,
    count: u32,
    hidden: u32,
    layer_scalar_offset: Option<usize>,
    layer_scalar_dim: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "residual_add",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("residual_add_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), addition_offset, 1);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&count as *const _ as *mut _),
        4,
        2,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        3,
    );
    let layer_scalar_dim = if layer_scalar_offset.is_some() {
        layer_scalar_dim
    } else {
        0
    };
    encoder.setBuffer_offset_atIndex(Some(buf), layer_scalar_offset.unwrap_or(residual_offset), 4);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&layer_scalar_dim as *const _ as *mut _),
        4,
        5,
    );
    let groups = MTLSize {
        width: count as usize,
        height: 1,
        depth: 1,
    };
    // Each thread owns one independent element. Pack contiguous dimensions
    // into full groups without changing arithmetic, indexing or rounding.
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_residual_add_then_scale(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    residual_offset: usize,
    addition_offset: usize,
    count: u32,
    hidden: u32,
    layer_scalar_offset: usize,
    layer_scalar_dim: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "residual_add_then_scale",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("residual_add_then_scale_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), addition_offset, 1);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&count as *const _ as *mut _),
        4,
        2,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBuffer_offset_atIndex(Some(buf), layer_scalar_offset, 4);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&layer_scalar_dim as *const _ as *mut _),
        4,
        5,
    );
    let groups = MTLSize {
        width: count as usize,
        height: 1,
        depth: 1,
    };
    // Each thread owns one independent element. Pack contiguous dimensions
    // into full groups without changing arithmetic, indexing or rounding.
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_residual_add_rmsnorm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    residual_offset: usize,
    addition_offset: usize,
    output_offset: usize,
    gamma_offset: usize,
    hidden: u32,
    eps: f32,
    num_tokens: u32,
    layer_scalar_offset: Option<usize>,
    layer_scalar_dim: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "residual_add_rmsnorm",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("residual_add_rmsnorm_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), addition_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), output_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), gamma_offset, 3);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&eps as *const _ as *mut _),
        4,
        5,
    );
    let layer_scalar_dim = if layer_scalar_offset.is_some() {
        layer_scalar_dim
    } else {
        0
    };
    encoder.setBuffer_offset_atIndex(Some(buf), layer_scalar_offset.unwrap_or(residual_offset), 6);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&layer_scalar_dim as *const _ as *mut _),
        4,
        7,
    );
    let groups = MTLSize {
        width: num_tokens as usize,
        height: 1,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_layer_scale(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    x_offset: usize,
    count: u32,
    hidden: u32,
    layer_scalar_offset: Option<usize>,
    layer_scalar_dim: u32,
) -> Result<()> {
    let Some(layer_scalar_offset) = layer_scalar_offset else {
        return Ok(());
    };
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "layer_scale",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("layer_scale_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), x_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), layer_scalar_offset, 1);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&count as *const _ as *mut _),
        4,
        2,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&layer_scalar_dim as *const _ as *mut _),
        4,
        4,
    );
    let groups = MTLSize {
        width: count as usize,
        height: 1,
        depth: 1,
    };
    // Each thread owns one independent element. Pack contiguous dimensions
    // into full groups without changing arithmetic, indexing or rounding.
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_ple_gelu_mul(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    gate_offset: usize,
    packed_ple_offset: usize,
    num_tokens: u32,
    num_layers: u32,
    layer_idx: u32,
    ple_dim: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "ple_gelu_mul",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("ple_gelu_mul_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), gate_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), packed_ple_offset, 1);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        2,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_layers as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&layer_idx as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&ple_dim as *const _ as *mut _),
        4,
        5,
    );
    let groups = MTLSize {
        width: num_tokens as usize,
        height: ple_dim as usize,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

unsafe fn encode_logits_head(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
    rms_eps: f32,
    softcap: f32,
    residual_offset: usize,
    final_norm_offset: usize,
    lm_head_offset: usize,
    logits_offset: usize,
    normed_hidden_offset: usize,
    sampled_tokens_offset: usize,
) -> Result<()> {
    let buf = arena.buffer_retained();

    if supports_fused_final_logits_small(num_tokens, hidden, vocab) {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "final_fused_logits_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("final_norm_lm_head_argmax_small_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), final_norm_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), lm_head_offset, 2);
        encoder.setBuffer_offset_atIndex(Some(buf), logits_offset, 3);
        encoder.setBuffer_offset_atIndex(Some(buf), sampled_tokens_offset, 4);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
            4,
            5,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&vocab as *const _ as *mut _),
            4,
            6,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&rms_eps as *const _ as *mut _),
            4,
            7,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&softcap as *const _ as *mut _),
            4,
            8,
        );
        let groups = MTLSize {
            width: num_tokens as usize,
            height: 1,
            depth: 1,
        };
        let tpg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
        return Ok(());
    }

    // 1) final RMSNorm -> normed_hidden
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "final_rmsnorm_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("rmsnorm_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), normed_hidden_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), final_norm_offset, 2);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
            4,
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&rms_eps as *const _ as *mut _),
            4,
            4,
        );
        let groups = MTLSize {
            width: num_tokens as usize,
            height: 1,
            depth: 1,
        };
        let tpg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    }

    // 2) LM-head projection
    encode_gemm(
        cmd_buf,
        pipelines,
        buf,
        normed_hidden_offset,
        lm_head_offset,
        logits_offset,
        num_tokens,
        vocab,
        hidden,
        1.0,
        0.0,
    )?;

    // 3) optional fused softcap + argmax, or plain argmax when softcap is disabled.
    if softcap > 0.0 {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "final_softcap_argmax_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("softcap_argmax_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), logits_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), sampled_tokens_offset, 1);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
            4,
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&vocab as *const _ as *mut _),
            4,
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&softcap as *const _ as *mut _),
            4,
            4,
        );
        let groups = MTLSize {
            width: num_tokens as usize,
            height: 1,
            depth: 1,
        };
        let tpg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    } else {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "final_argmax_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("argmax_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), logits_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), sampled_tokens_offset, 1);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
            4,
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&vocab as *const _ as *mut _),
            4,
            3,
        );
        let groups = MTLSize {
            width: num_tokens as usize,
            height: 1,
            depth: 1,
        };
        let tpg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    }

    Ok(())
}

unsafe fn encode_final_rmsnorm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    num_tokens: u32,
    hidden: u32,
    rms_eps: f32,
    residual_offset: usize,
    final_norm_offset: usize,
    normed_hidden_offset: usize,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "final_sample_rmsnorm_encoder",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("rmsnorm_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), normed_hidden_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), final_norm_offset, 2);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&rms_eps as *const _ as *mut _),
        4,
        4,
    );
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: num_tokens as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_final_lm_head_argmax_tiles(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
    softcap: f32,
    normed_hidden_offset: usize,
    lm_head_offset: usize,
    partial_max_offset: usize,
    partial_idx_offset: usize,
) -> Result<u32> {
    let tile_count = vocab.div_ceil(8);
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "final_lm_head_argmax_tiles_encoder",
                device: "apple-silicon",
            },
        )
    })?;
    let use_batch8 =
        supports_batch8_final_sample(pipelines.gpu_family(), num_tokens, hidden, vocab);
    let use_batch4 =
        supports_batch4_final_sample(pipelines.gpu_family(), num_tokens, hidden, vocab);
    let pso = pipelines.get(if use_batch8 {
        "final_lm_head_argmax_tiles_batch8_f16"
    } else if use_batch4 {
        "final_lm_head_argmax_tiles_batch4_f16"
    } else {
        "final_lm_head_argmax_tiles_f16"
    })?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), normed_hidden_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), lm_head_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), partial_max_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), partial_idx_offset, 3);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&vocab as *const _ as *mut _),
        4,
        5,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
        4,
        6,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&tile_count as *const _ as *mut _),
        4,
        7,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&softcap as *const _ as *mut _),
        4,
        8,
    );
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        if use_batch8 || use_batch4 {
            MTLSize {
                width: tile_count as usize,
                height: 1,
                depth: 1,
            }
        } else {
            MTLSize {
                width: num_tokens as usize,
                height: tile_count as usize,
                depth: 1,
            }
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(tile_count)
}

unsafe fn encode_final_argmax_reduce(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    num_tokens: u32,
    tile_count: u32,
    partial_max_offset: usize,
    partial_idx_offset: usize,
    sampled_tokens_offset: usize,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "final_argmax_reduce_encoder",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("final_argmax_reduce_f32")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), partial_max_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), partial_idx_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), sampled_tokens_offset, 2);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&tile_count as *const _ as *mut _),
        4,
        4,
    );
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: num_tokens as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_encode_finalize_sample(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
    rms_eps: f32,
    softcap: f32,
    residual_offset: usize,
    final_norm_offset: usize,
    lm_head_offset: usize,
    logits_offset: usize,
    normed_hidden_offset: usize,
    sampled_tokens_offset: usize,
    partial_max_offset: usize,
    partial_idx_offset: usize,
) -> Result<()> {
    if !supports_final_sample_tiles(num_tokens, hidden, vocab) {
        return encode_logits_head(
            cmd_buf,
            pipelines,
            arena,
            num_tokens,
            hidden,
            vocab,
            rms_eps,
            softcap,
            residual_offset,
            final_norm_offset,
            lm_head_offset,
            logits_offset,
            normed_hidden_offset,
            sampled_tokens_offset,
        );
    }

    let buf = arena.buffer_retained();
    encode_final_rmsnorm(
        cmd_buf,
        pipelines,
        buf,
        num_tokens,
        hidden,
        rms_eps,
        residual_offset,
        final_norm_offset,
        normed_hidden_offset,
    )?;
    let tile_count = encode_final_lm_head_argmax_tiles(
        cmd_buf,
        pipelines,
        buf,
        num_tokens,
        hidden,
        vocab,
        softcap,
        normed_hidden_offset,
        lm_head_offset,
        partial_max_offset,
        partial_idx_offset,
    )?;
    encode_final_argmax_reduce(
        cmd_buf,
        pipelines,
        buf,
        num_tokens,
        tile_count,
        partial_max_offset,
        partial_idx_offset,
        sampled_tokens_offset,
    )
}

/// Run final normalization + LM head projection + optional softcap + argmax.
///
/// This path is intentionally non-allocating in the hot path; all buffers
/// are pre-allocated in the arena.
#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_finalize_logits(
    ctx: &MetalContext,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
    rms_eps: f32,
    softcap: f32,
    residual_offset: usize,
    final_norm_offset: usize,
    lm_head_offset: usize,
    logits_offset: usize,
    normed_hidden_offset: usize,
    sampled_tokens_offset: usize,
) -> Result<()> {
    let queue = ctx.queue_retained();
    let cmd_buf = queue.commandBuffer().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "final_logits_encode",
                device: "apple-silicon",
            },
        )
    })?;

    metal_encode_finalize_logits(
        &cmd_buf,
        pipelines,
        arena,
        num_tokens,
        hidden,
        vocab,
        rms_eps,
        softcap,
        residual_offset,
        final_norm_offset,
        lm_head_offset,
        logits_offset,
        normed_hidden_offset,
        sampled_tokens_offset,
    )?;

    cmd_buf.commit();
    Ok(())
}

/// Encode final normalization + LM head projection + optional softcap + argmax
/// onto an existing command buffer.
#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_encode_finalize_logits(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
    rms_eps: f32,
    softcap: f32,
    residual_offset: usize,
    final_norm_offset: usize,
    lm_head_offset: usize,
    logits_offset: usize,
    normed_hidden_offset: usize,
    sampled_tokens_offset: usize,
) -> Result<()> {
    encode_logits_head(
        cmd_buf,
        pipelines,
        arena,
        num_tokens,
        hidden,
        vocab,
        rms_eps,
        softcap,
        residual_offset,
        final_norm_offset,
        lm_head_offset,
        logits_offset,
        normed_hidden_offset,
        sampled_tokens_offset,
    )
}

/// Same as `metal_finalize_logits`, but blocks until completion.
#[allow(clippy::too_many_arguments)]
pub unsafe fn metal_finalize_logits_blocking(
    ctx: &MetalContext,
    pipelines: &PipelineCache,
    arena: &MetalBufferArena,
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
    rms_eps: f32,
    softcap: f32,
    residual_offset: usize,
    final_norm_offset: usize,
    lm_head_offset: usize,
    logits_offset: usize,
    normed_hidden_offset: usize,
    sampled_tokens_offset: usize,
) -> Result<()> {
    let queue = ctx.queue_retained();
    let cmd_buf = queue.commandBuffer().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "final_logits_encode_blocking",
                device: "apple-silicon",
            },
        )
    })?;

    metal_encode_finalize_logits(
        &cmd_buf,
        pipelines,
        arena,
        num_tokens,
        hidden,
        vocab,
        rms_eps,
        softcap,
        residual_offset,
        final_norm_offset,
        lm_head_offset,
        logits_offset,
        normed_hidden_offset,
        sampled_tokens_offset,
    )?;

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

unsafe fn encode_split_qkv(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    qkv_offset: usize,
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    num_tokens: u32,
    q_dim: u32,
    kv_dim: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "split_qkv_encode",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("split_qkv_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), qkv_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), q_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), k_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), v_offset, 3);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&q_dim as *const _ as *mut _),
        4,
        5,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&kv_dim as *const _ as *mut _),
        4,
        6,
    );
    let max_dim = q_dim.saturating_add(2u32.saturating_mul(kv_dim));
    let groups = MTLSize {
        width: num_tokens as usize,
        height: max_dim as usize,
        depth: 1,
    };
    let tpg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_low_bit_descriptor(role: AppleLowBitTensorRole) -> MetalLowBitProjectionOffsets {
        MetalLowBitProjectionOffsets::new_for_role(
            role,
            rvllm_apple::AppleLowBitWeightFormat::W4A16,
            4,
            32,
            0,
            64,
            64,
            8,
        )
        .expect("test descriptor")
    }

    #[test]
    fn low_bit_qkv_forces_standalone_route_and_skip_omits_kv_dispatches() {
        assert!(qkv_fusion_allowed(false, true));
        assert!(!qkv_fusion_allowed(true, true));
        assert_eq!(
            low_bit_qkv_dispatch_columns(false, 8, 4),
            [(true, 0), (true, 8), (true, 12)]
        );
        assert_eq!(
            low_bit_qkv_dispatch_columns(true, 8, 4),
            [(true, 0), (false, 8), (false, 12)]
        );
    }

    #[test]
    fn low_bit_descriptor_roles_cannot_be_swapped_at_equal_shape() {
        let q = test_low_bit_descriptor(AppleLowBitTensorRole::QueryProjection);
        let k = test_low_bit_descriptor(AppleLowBitTensorRole::KeyProjection);
        assert!(low_bit_descriptor_matches(
            q,
            AppleLowBitTensorRole::QueryProjection,
            [4, 32]
        ));
        assert!(!low_bit_descriptor_matches(
            k,
            AppleLowBitTensorRole::QueryProjection,
            [4, 32]
        ));
    }

    #[test]
    fn down_projection_execution_source_is_exactly_one() {
        assert_eq!(
            validate_down_projection_sources(Some(64), false).expect("native source"),
            Some(64)
        );
        assert_eq!(
            validate_down_projection_sources(None, true).expect("low-bit source"),
            None
        );
        assert!(validate_down_projection_sources(Some(64), true).is_err());
        assert!(validate_down_projection_sources(None, false).is_err());
    }

    #[test]
    fn attention_window_start_preserves_full_attention_and_restored_prefixes() {
        assert_eq!(attention_window_start(0, 0), 0);
        assert_eq!(attention_window_start(4097, 0), 0);
        assert_eq!(attention_window_start(17, 32), 0);
        assert_eq!(attention_window_start(33, 32), 1);
        // A chunk beginning at absolute position 96 still attends 65..=96;
        // the lower bound must not be derived from its chunk-local offset.
        assert_eq!(attention_window_start(97, 32), 65);
    }

    #[test]
    fn decode_online_shape_gate_fails_closed_to_scalar_fallback() {
        let supported = MetalLayerDims {
            layer_idx: 0,
            attention_window: 0,
            num_tokens: 1,
            hidden: 128,
            num_layers: 1,
            intermediate: 256,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            moe_num_experts: 0,
            moe_top_k: 0,
            moe_intermediate: 0,
            ple_dim: 0,
            block_size: 32,
            max_blocks_per_seq: 2,
            num_blocks_total: 4,
            rope_dim: 32,
            rms_eps: 1e-6,
            attn_scale: 1.0,
            softcap: 0.0,
        };
        assert!(supports_attention_decode_online(&supported));

        let mut oversized_head = supported;
        oversized_head.head_dim = 257;
        assert!(!supports_attention_decode_online(&oversized_head));

        let mut invalid_gqa = supported;
        invalid_gqa.num_heads = 3;
        assert!(!supports_attention_decode_online(&invalid_gqa));

        let mut missing_page_layout = supported;
        missing_page_layout.max_blocks_per_seq = 0;
        assert!(!supports_attention_decode_online(&missing_page_layout));
    }

    fn attention_decode_reference(
        q: &[half::f16],
        k_cache: &[half::f16],
        v_cache: &[half::f16],
        block_tables: &[i32],
        context_lens: &[i32],
        num_seqs: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        max_blocks: u32,
        scale: f32,
        attention_window: u32,
    ) -> Vec<f32> {
        let num_seqs = num_seqs as usize;
        let num_heads = num_heads as usize;
        let num_kv_heads = num_kv_heads as usize;
        let head_dim = head_dim as usize;
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;

        let mut out = vec![0f32; num_seqs * q_dim];
        for seq in 0..num_seqs {
            let ctx_len = context_lens[seq] as usize;
            if ctx_len == 0 {
                continue;
            }

            for head in 0..num_heads {
                let kv_head = head / (num_heads / num_kv_heads);
                let mut q_local = vec![0f32; head_dim];
                for d in 0..head_dim {
                    q_local[d] = q[seq * q_dim + head * head_dim + d].to_f32();
                }

                let mut scores = vec![f32::NEG_INFINITY; ctx_len];
                let attention_start = attention_window_start(ctx_len as u32, attention_window);
                for t in attention_start as usize..ctx_len {
                    let block_idx = t / block_size as usize;
                    let block_offset = t % block_size as usize;
                    let block_id = block_tables[seq * max_blocks as usize + block_idx];
                    if block_id < 0 {
                        continue;
                    }

                    let block_base = block_id as usize * block_size as usize * kv_dim;
                    let mut score = 0f32;
                    for d in 0..head_dim {
                        let k_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
                        score += q_local[d] * k_cache[k_idx].to_f32();
                    }
                    scores[t] = score * scale;
                }

                let max_score = scores.iter().fold(f32::NEG_INFINITY, |acc, &s| acc.max(s));
                let mut sum_exp = 0f32;
                for &score in &scores {
                    if score > f32::NEG_INFINITY {
                        sum_exp += (score - max_score).exp();
                    }
                }

                for t in 0..ctx_len {
                    if scores[t] <= f32::NEG_INFINITY {
                        continue;
                    }
                    let block_idx = t / block_size as usize;
                    let block_offset = t % block_size as usize;
                    let block_id = block_tables[seq * max_blocks as usize + block_idx];
                    if block_id < 0 {
                        continue;
                    }

                    let block_base = block_id as usize * block_size as usize * kv_dim;
                    let weight = (scores[t] - max_score).exp() / sum_exp;

                    for d in 0..head_dim {
                        let v_idx = block_base + block_offset * kv_dim + kv_head * head_dim + d;
                        out[seq * q_dim + head * head_dim + d] += weight * v_cache[v_idx].to_f32();
                    }
                }
            }
        }

        out
    }

    #[test]
    fn qkv_scratch_planar_layout_rejects_canonical_aliasing() {
        let scratch = MetalScratch {
            normed_hidden: 0,
            qkv_out: 64,
            q_offset: 64,
            k_offset: 80,
            v_offset: 88,
            attn_out: 0,
            global_decode_partials: None,
            gate_up_out: 0,
            activated: 0,
            mlp_out: 0,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_activated: None,
            moe_out: None,
        };
        assert!(validate_qkv_scratch_planar(&scratch, 1, 8, 4, false).is_err());
    }

    #[test]
    fn qkv_scratch_planar_layout_rejects_aliasing() {
        let scratch = MetalScratch {
            normed_hidden: 0,
            qkv_out: 64,
            q_offset: 68,
            k_offset: 80,
            v_offset: 96,
            attn_out: 0,
            global_decode_partials: None,
            gate_up_out: 0,
            activated: 0,
            mlp_out: 0,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_activated: None,
            moe_out: None,
        };
        assert!(validate_qkv_scratch_planar(&scratch, 1, 8, 4, false).is_err());
    }

    #[test]
    fn qkv_scratch_planar_layout_accepts_non_overlapping_offsets() {
        let scratch = MetalScratch {
            normed_hidden: 0,
            qkv_out: 0,
            q_offset: 1024,
            k_offset: 2048,
            v_offset: 3072,
            attn_out: 0,
            global_decode_partials: None,
            gate_up_out: 0,
            activated: 0,
            mlp_out: 0,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_activated: None,
            moe_out: None,
        };
        assert!(
            validate_qkv_scratch_planar(&scratch, 1, 8, 4, false).is_ok(),
            "non-overlapping planar scratch regions should be accepted"
        );
    }

    #[test]
    fn qkv_scratch_planar_checks_the_fp32_projection_extent() {
        let mut scratch = MetalScratch {
            normed_hidden: 0,
            qkv_out: 0,
            q_offset: 32,
            k_offset: 128,
            v_offset: 160,
            attn_out: 0,
            global_decode_partials: None,
            gate_up_out: 0,
            activated: 0,
            mlp_out: 0,
            moe_topk_indices: None,
            moe_topk_weights: None,
            moe_activated: None,
            moe_out: None,
        };
        // Sixteen projection elements occupy 32 bytes in FP16, 64 in FP32.
        assert!(validate_qkv_scratch_planar(&scratch, 1, 8, 4, false).is_ok());
        assert!(validate_qkv_scratch_planar(&scratch, 1, 8, 4, true).is_err());
        scratch.q_offset = 64;
        assert!(validate_qkv_scratch_planar(&scratch, 1, 8, 4, true).is_ok());
    }

    #[test]
    fn prefill_mma_limits_projection_shapes_and_output_storage() {
        for m in [6, 8, 19, 20, 21, 28, 32, 63, 84, 230, 650, 1024] {
            assert!(is_prefill_mma_shape(m, 8192, 3840, true));
            assert!(is_prefill_mma_shape(m, 9216, 3840, true));
            assert!(is_prefill_mma_shape(m, 30720, 3840, false));
            for k in [4096, 8192, 15360] {
                assert!(is_prefill_mma_shape(m, 3840, k, false));
            }
            // A QKV FP32 buffer and ordinary BF16 projection buffers cannot
            // select each other's entry points based on dimensions alone.
            assert!(!is_prefill_mma_shape(m, 8192, 3840, false));
            assert!(!is_prefill_mma_shape(m, 30720, 3840, true));
        }
        for m in [0, 1, 5, 1025, u32::MAX] {
            assert!(!is_prefill_mma_shape(m, 8192, 3840, true));
            assert!(!is_prefill_mma_shape(m, 3840, 15360, false));
        }
        assert!(!is_prefill_mma_shape(21, 30719, 3840, false));
        assert!(!is_prefill_mma_shape(21, 8192, 3839, true));
        for m in [84, 230, 652, 1024] {
            for k in [4096, 8192, 15360] {
                assert_eq!(metal_gemm_rmsnorm_encoder_count(m, 3840, k, true), 2);
                assert_eq!(metal_gemm_rmsnorm_encoder_count(m, 3840, k, false), 1);
            }
            assert_eq!(metal_gemm_rmsnorm_encoder_count(m, 2304, 9216, true), 1);
        }
    }

    fn split_qkv_ref(
        qkv: &[half::f16],
        num_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> (Vec<half::f16>, Vec<half::f16>, Vec<half::f16>) {
        let qkv_dim = q_dim + 2 * kv_dim;
        let mut q = vec![half::f16::from_f32(0.0); num_tokens * q_dim];
        let mut k = vec![half::f16::from_f32(0.0); num_tokens * kv_dim];
        let mut v = vec![half::f16::from_f32(0.0); num_tokens * kv_dim];

        for token in 0..num_tokens {
            let base = token * qkv_dim;
            for d in 0..q_dim {
                q[token * q_dim + d] = qkv[base + d];
            }
            for d in 0..kv_dim {
                k[token * kv_dim + d] = qkv[base + q_dim + d];
                v[token * kv_dim + d] = qkv[base + q_dim + kv_dim + d];
            }
        }
        (q, k, v)
    }

    #[test]
    fn split_qkv_reference_matches_interleaved_layout() {
        let num_tokens = 2usize;
        let q_dim = 3usize;
        let kv_dim = 2usize;
        let qkv: Vec<half::f16> = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(5.0),
            half::f16::from_f32(6.0),
            half::f16::from_f32(7.0),
            half::f16::from_f32(8.0),
            half::f16::from_f32(9.0),
            half::f16::from_f32(10.0),
            half::f16::from_f32(11.0),
            half::f16::from_f32(12.0),
            half::f16::from_f32(13.0),
            half::f16::from_f32(14.0),
        ];
        let (q, k, v) = split_qkv_ref(&qkv, num_tokens, q_dim, kv_dim);
        assert_eq!(
            q,
            vec![
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
                half::f16::from_f32(3.0),
                half::f16::from_f32(8.0),
                half::f16::from_f32(9.0),
                half::f16::from_f32(10.0)
            ]
        );
        assert_eq!(
            k,
            vec![
                half::f16::from_f32(4.0),
                half::f16::from_f32(5.0),
                half::f16::from_f32(11.0),
                half::f16::from_f32(12.0)
            ]
        );
        assert_eq!(
            v,
            vec![
                half::f16::from_f32(6.0),
                half::f16::from_f32(7.0),
                half::f16::from_f32(13.0),
                half::f16::from_f32(14.0)
            ]
        );
    }

    fn rmsnorm_ref(
        input: &[half::f16],
        gamma: &[half::f16],
        hidden: usize,
        eps: f32,
    ) -> Vec<half::f16> {
        let mut out = Vec::with_capacity(input.len());
        for token in 0..(input.len() / hidden) {
            let base = token * hidden;
            let mut sum_sq = 0.0f32;
            for d in 0..hidden {
                let v = input[base + d].to_f32();
                sum_sq += v * v;
            }
            let inv_rms = 1.0f32 / (sum_sq / hidden as f32 + eps).sqrt();
            for d in 0..hidden {
                out.push(half::f16::from_f32(
                    input[base + d].to_f32() * inv_rms * gamma[d].to_f32(),
                ));
            }
        }
        out
    }

    fn softcap_ref(logits: &mut [f32], cap: f32) {
        for v in logits {
            *v = cap * (*v / cap).tanh();
        }
    }

    fn lm_head_ref(
        residual: &[half::f16],
        final_norm: &[half::f16],
        lm_head: &[half::f16],
        num_tokens: usize,
        hidden: usize,
        vocab: usize,
        eps: f32,
        softcap: f32,
    ) -> Vec<i32> {
        let normed = rmsnorm_ref(residual, final_norm, hidden, eps);
        let mut logits = vec![0f32; num_tokens * vocab];
        for t in 0..num_tokens {
            for v in 0..vocab {
                let mut acc = 0.0f32;
                for d in 0..hidden {
                    acc += normed[t * hidden + d].to_f32() * lm_head[v * hidden + d].to_f32();
                }
                logits[t * vocab + v] = acc;
            }
        }
        if softcap > 0.0 {
            softcap_ref(&mut logits, softcap);
        }

        let mut out = vec![0i32; num_tokens];
        for t in 0..num_tokens {
            let mut best_idx = 0usize;
            let mut best_val = -f32::INFINITY;
            for v in 0..vocab {
                let val = logits[t * vocab + v];
                if val > best_val {
                    best_val = val;
                    best_idx = v;
                }
            }
            out[t] = best_idx as i32;
        }
        out
    }

    fn gemm_ref(
        a: &[half::f16],
        b_transposed: &[half::f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<half::f16> {
        let mut out = vec![half::f16::from_f32(0.0); m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[row * k + kk].to_f32() * b_transposed[col * k + kk].to_f32();
                }
                out[row * n + col] = half::f16::from_f32(acc);
            }
        }
        out
    }

    #[test]
    fn fused_final_logits_small_cpu_reference_selects_expected_tokens() {
        let residual = vec![
            half::f16::from_f32(0.2),
            half::f16::from_f32(0.4),
            half::f16::from_f32(0.6),
            half::f16::from_f32(0.8),
            half::f16::from_f32(-0.5),
            half::f16::from_f32(0.3),
            half::f16::from_f32(0.7),
            half::f16::from_f32(0.1),
        ];
        let final_norm = vec![half::f16::from_f32(1.0); 4];
        let lm_head = vec![
            half::f16::from_f32(0.1),
            half::f16::from_f32(0.0),
            half::f16::from_f32(-0.1),
            half::f16::from_f32(0.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(0.2),
            half::f16::from_f32(0.0),
            half::f16::from_f32(-0.1),
            half::f16::from_f32(0.0),
            half::f16::from_f32(0.1),
            half::f16::from_f32(0.4),
            half::f16::from_f32(0.0),
        ];

        let got = lm_head_ref(&residual, &final_norm, &lm_head, 2, 4, 3, 1e-5, 30.0);
        assert_eq!(got, vec![2, 2]);
    }

    #[test]
    fn final_logits_encoder_count_gates_fused_small_vocab_path() {
        assert_eq!(metal_finalize_logits_encoder_count(2, 4, 3, 30.0), 1);
        assert_eq!(metal_finalize_logits_encoder_count(2, 4, 257, 30.0), 3);
        assert_eq!(metal_finalize_logits_encoder_count(2, 4, 257, 0.0), 3);
        assert_eq!(metal_finalize_sample_encoder_count(2, 4, 3), 1);
        assert_eq!(metal_finalize_sample_encoder_count(2, 4, 257), 3);
    }

    #[test]
    fn gemm_rmsnorm_policy_prefers_parallel_microbatch_path() {
        assert_eq!(metal_gemm_rmsnorm_encoder_count(1, 2304, 9216, false), 2);
        assert_eq!(metal_gemm_rmsnorm_encoder_count(19, 2304, 9216, false), 2);
        assert_eq!(metal_gemm_rmsnorm_encoder_count(20, 2304, 9216, false), 1);
        assert_eq!(metal_gemm_rmsnorm_encoder_count(1, 2304, 65_537, false), 1);
    }

    #[test]
    fn batch8_projection_policy_is_shape_and_gpu_family_gated() {
        assert!(!supports_batch8_gemm(
            AppleGpuFamily::Apple9,
            4,
            2_304,
            9_216
        ));
        assert!(supports_batch8_gemm(
            AppleGpuFamily::Apple10,
            8,
            9_216,
            2_304
        ));
        assert!(!supports_batch8_gemm(
            AppleGpuFamily::Apple9,
            1,
            2_304,
            9_216
        ));
        assert!(!supports_batch8_gemm(
            AppleGpuFamily::Apple8,
            8,
            2_304,
            9_216
        ));
        assert!(!supports_batch8_gemm(
            AppleGpuFamily::Apple9,
            8,
            2_304,
            9_215
        ));
        assert!(!supports_batch8_gemm(
            AppleGpuFamily::Unknown,
            8,
            2_304,
            9_216
        ));
        assert!(!supports_batch8_gemm(
            AppleGpuFamily::Apple9,
            8,
            2_305,
            9_216
        ));
    }

    #[test]
    fn batch8_lm_head_policy_is_shape_and_gpu_family_gated() {
        assert!(supports_batch8_final_sample(
            AppleGpuFamily::Apple9,
            8,
            2_304,
            262_144
        ));
        assert!(!supports_batch8_final_sample(
            AppleGpuFamily::Apple9,
            1,
            2_304,
            262_144
        ));
        assert!(!supports_batch8_final_sample(
            AppleGpuFamily::Apple9,
            4,
            2_304,
            262_144
        ));
        assert!(!supports_batch8_final_sample(
            AppleGpuFamily::Apple8,
            8,
            2_304,
            262_144
        ));
        assert!(!supports_batch8_final_sample(
            AppleGpuFamily::Apple9,
            8,
            2_303,
            262_144
        ));
    }

    #[test]
    fn batch4_lm_head_policy_is_shape_and_gpu_family_gated() {
        assert!(supports_batch4_final_sample(
            AppleGpuFamily::Apple9,
            4,
            2_304,
            262_144
        ));
        assert!(supports_batch4_final_sample(
            AppleGpuFamily::Apple10,
            4,
            2_304,
            262_144
        ));
        assert!(!supports_batch4_final_sample(
            AppleGpuFamily::Apple9,
            8,
            2_304,
            262_144
        ));
        for num_tokens in 1..4 {
            assert!(!supports_batch4_final_sample(
                AppleGpuFamily::Apple9,
                num_tokens,
                2_304,
                262_144
            ));
        }
        assert!(!supports_batch4_final_sample(
            AppleGpuFamily::Apple8,
            4,
            2_304,
            262_144
        ));
    }

    #[test]
    fn softcap_ref_matches_definition() {
        let cap = 30.0f32;
        let mut logits = vec![-60.0f32, -30.0, -15.0, -1.0, 0.0, 1.0, 15.0, 30.0, 60.0];
        let expected: Vec<f32> = logits.iter().map(|x| cap * (x / cap).tanh()).collect();

        softcap_ref(&mut logits, cap);
        for (got, exp) in logits.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn final_logits_macos_smoke_matches_cpu() -> rvllm_core::Result<()> {
        let mut ctx = MetalContext::new()?;
        ctx.compile_library(crate::kernels::KERNEL_SOURCE)?;
        let mut pipelines = PipelineCache::new();
        pipelines.compile_all(&ctx)?;
        let mut arena = MetalBufferArena::new(ctx.device(), 16 * 1024)?;

        const NUM_TOKENS: u32 = 2;
        const HIDDEN: u32 = 4;
        const VOCAB: u32 = 3;
        const EPS: f32 = 1e-5;
        const SOFTCAP: f32 = 30.0;
        let residual = vec![
            half::f16::from_f32(0.2),
            half::f16::from_f32(0.4),
            half::f16::from_f32(0.6),
            half::f16::from_f32(0.8),
            half::f16::from_f32(0.1),
            half::f16::from_f32(-0.2),
            half::f16::from_f32(0.5),
            half::f16::from_f32(0.3),
        ];
        let final_norm = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(1.0),
        ];
        let lm_head = vec![
            half::f16::from_f32(0.2),
            half::f16::from_f32(0.1),
            half::f16::from_f32(-0.1),
            half::f16::from_f32(0.0),
            half::f16::from_f32(-0.2),
            half::f16::from_f32(0.3),
            half::f16::from_f32(0.25),
            half::f16::from_f32(-0.4),
            half::f16::from_f32(0.15),
            half::f16::from_f32(0.05),
            half::f16::from_f32(0.6),
            half::f16::from_f32(-0.3),
        ];

        let half_bytes = std::mem::size_of::<half::f16>();
        let i32_bytes = std::mem::size_of::<i32>();
        let residual_region = arena.region("residual", residual.len() * half_bytes, 2)?;
        let final_norm_region = arena.region("final_norm", final_norm.len() * half_bytes, 2)?;
        let lm_head_region = arena.region("lm_head", lm_head.len() * half_bytes, 2)?;
        let normed_region = arena.region("normed", residual.len() * half_bytes, 2)?;
        let logits_region = arena.region(
            "logits",
            (NUM_TOKENS as usize * VOCAB as usize) * half_bytes,
            2,
        )?;
        let token_region = arena.region("tokens", (NUM_TOKENS as usize) * i32_bytes, 4)?;

        unsafe {
            let residual_ptr = arena.host_ptr(&residual_region) as *mut half::f16;
            for (idx, value) in residual.iter().enumerate() {
                *residual_ptr.add(idx) = *value;
            }
            let final_norm_ptr = arena.host_ptr(&final_norm_region) as *mut half::f16;
            for (idx, value) in final_norm.iter().enumerate() {
                *final_norm_ptr.add(idx) = *value;
            }
            let lm_head_ptr = arena.host_ptr(&lm_head_region) as *mut half::f16;
            for (idx, value) in lm_head.iter().enumerate() {
                *lm_head_ptr.add(idx) = *value;
            }
        }

        let queue = ctx.queue_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "final_logits_smoke_cmdbuf",
                    device: "apple-silicon",
                },
            )
        })?;
        unsafe {
            encode_logits_head(
                &cmd_buf,
                &pipelines,
                &arena,
                NUM_TOKENS,
                HIDDEN,
                VOCAB,
                EPS,
                SOFTCAP,
                residual_region.offset,
                final_norm_region.offset,
                lm_head_region.offset,
                logits_region.offset,
                normed_region.offset,
                token_region.offset,
            )?;
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let got = unsafe {
            let ptr = arena.host_ptr(&token_region) as *const i32;
            std::slice::from_raw_parts(ptr, NUM_TOKENS as usize).to_vec()
        };
        let expected = lm_head_ref(
            &residual,
            &final_norm,
            &lm_head,
            NUM_TOKENS as usize,
            HIDDEN as usize,
            VOCAB as usize,
            EPS,
            SOFTCAP,
        );
        assert_eq!(got, expected);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiled_gemm_macos_smoke_matches_cpu() -> rvllm_core::Result<()> {
        let mut ctx = MetalContext::new()?;
        ctx.compile_library(crate::kernels::KERNEL_SOURCE)?;
        let mut pipelines = PipelineCache::new();
        pipelines.compile_all(&ctx)?;
        let mut arena = MetalBufferArena::new(ctx.device(), 16 * 1024)?;

        const M: u32 = 2;
        const N: u32 = 16;
        const K: u32 = 8;
        let a: Vec<half::f16> = (0..(M * K))
            .map(|i| half::f16::from_f32((i as f32 + 1.0) * 0.01))
            .collect();
        let b: Vec<half::f16> = (0..(N * K))
            .map(|i| half::f16::from_f32((i as f32 % 7.0 - 3.0) * 0.02))
            .collect();
        let c = vec![half::f16::from_f32(0.0); (M * N) as usize];

        let half_bytes = std::mem::size_of::<half::f16>();
        let a_region = arena.region("gemm_a", a.len() * half_bytes, 2)?;
        let b_region = arena.region("gemm_b", b.len() * half_bytes, 2)?;
        let c_region = arena.region("gemm_c", c.len() * half_bytes, 2)?;

        unsafe {
            let a_ptr = arena.host_ptr(&a_region) as *mut half::f16;
            for (idx, value) in a.iter().enumerate() {
                *a_ptr.add(idx) = *value;
            }
            let b_ptr = arena.host_ptr(&b_region) as *mut half::f16;
            for (idx, value) in b.iter().enumerate() {
                *b_ptr.add(idx) = *value;
            }
            let c_ptr = arena.host_ptr(&c_region) as *mut half::f16;
            for (idx, value) in c.iter().enumerate() {
                *c_ptr.add(idx) = *value;
            }
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "tiled_gemm_smoke_cmdbuf",
                    device: "apple-silicon",
                },
            )
        })?;
        unsafe {
            encode_gemm(
                &cmd_buf,
                &pipelines,
                buf,
                a_region.offset,
                b_region.offset,
                c_region.offset,
                M,
                N,
                K,
                1.0,
                0.0,
            )?;
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let expected = gemm_ref(&a, &b, M as usize, N as usize, K as usize);
        let got = unsafe {
            std::slice::from_raw_parts(arena.host_ptr(&c_region) as *const half::f16, c.len())
        };
        for i in 0..got.len() {
            assert!((got[i].to_f32() - expected[i].to_f32()).abs() < 1e-2);
        }
        Ok(())
    }

    #[test]
    fn attention_decode_macos_smoke_matches_cpu() -> rvllm_core::Result<()> {
        let mut ctx = MetalContext::new()?;
        ctx.compile_library(crate::kernels::KERNEL_SOURCE)?;
        let mut pipelines = PipelineCache::new();
        pipelines.compile_all(&ctx)?;
        let mut arena = MetalBufferArena::new(ctx.device(), 64 * 1024)?;

        const NUM_TOKENS: u32 = 2;
        const NUM_HEADS: u32 = 2;
        const NUM_KV_HEADS: u32 = 1;
        const HEAD_DIM: u32 = 8;
        const BLOCK_SIZE: u32 = 4;
        const MAX_BLOCKS_PER_SEQ: u32 = 2;
        const PHYSICAL_BLOCKS: u32 = 3;
        let attn_scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let attention_window = 5u32;

        let q_dim = NUM_HEADS * HEAD_DIM;
        let kv_dim = NUM_KV_HEADS * HEAD_DIM;

        let q: Vec<half::f16> = (0..NUM_TOKENS * q_dim)
            .map(|i| half::f16::from_f32(((i as f32) + 1.0) * 0.03))
            .collect();
        let k_cache: Vec<half::f16> = (0..(PHYSICAL_BLOCKS * BLOCK_SIZE * kv_dim) as usize)
            .map(|i| half::f16::from_f32(((i as f32) + 1.0) * 0.01))
            .collect();
        let v_cache: Vec<half::f16> = (0..(PHYSICAL_BLOCKS * BLOCK_SIZE * kv_dim) as usize)
            .map(|i| half::f16::from_f32(((i as f32) + 1.0) * 0.02))
            .collect();
        // Both requests share physical block 2 as their immutable prefix.
        // Their logical tails live in different and non-contiguous pages.
        let block_tables = vec![2_i32, 0, 2, 1];
        let context_lens = vec![7_i32, 8_i32];

        let half_bytes = std::mem::size_of::<half::f16>();
        let i32_bytes = std::mem::size_of::<i32>();

        let q_region = arena.region("attn_decode_q", q.len() * half_bytes, 2)?;
        let k_cache_region = arena.region("attn_decode_k", k_cache.len() * half_bytes, 2)?;
        let v_cache_region = arena.region("attn_decode_v", v_cache.len() * half_bytes, 2)?;
        let out_region = arena.region("attn_decode_out", q.len() * half_bytes, 2)?;
        let block_table_region = arena.region(
            "attn_decode_block_tables",
            block_tables.len() * i32_bytes,
            4,
        )?;
        let context_region =
            arena.region("attn_decode_context", context_lens.len() * i32_bytes, 4)?;

        unsafe {
            let q_ptr = arena.host_ptr(&q_region) as *mut half::f16;
            for (idx, value) in q.iter().enumerate() {
                *q_ptr.add(idx) = *value;
            }

            let k_ptr = arena.host_ptr(&k_cache_region) as *mut half::f16;
            for (idx, value) in k_cache.iter().enumerate() {
                *k_ptr.add(idx) = *value;
            }

            let v_ptr = arena.host_ptr(&v_cache_region) as *mut half::f16;
            for (idx, value) in v_cache.iter().enumerate() {
                *v_ptr.add(idx) = *value;
            }

            let slot_ptr = arena.host_ptr(&block_table_region) as *mut i32;
            for (idx, value) in block_tables.iter().enumerate() {
                *slot_ptr.add(idx) = *value;
            }

            let ctx_ptr = arena.host_ptr(&context_region) as *mut i32;
            for (idx, value) in context_lens.iter().enumerate() {
                *ctx_ptr.add(idx) = *value;
            }
        }

        let queue = ctx.queue_retained();
        let buf = arena.buffer_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "attn_decode_smoke",
                    device: "apple-silicon",
                },
            )
        })?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "attn_decode_smoke_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("attention_decode_online_f16")?;
        unsafe {
            encoder.setComputePipelineState(pso);
            encoder.setBuffer_offset_atIndex(Some(buf), q_region.offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), k_cache_region.offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), v_cache_region.offset, 2);
            encoder.setBuffer_offset_atIndex(Some(buf), out_region.offset, 3);
            encoder.setBuffer_offset_atIndex(Some(buf), block_table_region.offset, 4);
            encoder.setBuffer_offset_atIndex(Some(buf), context_region.offset, 5);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&NUM_TOKENS as *const _ as *mut _),
                4,
                6,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&NUM_HEADS as *const _ as *mut _),
                4,
                7,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&NUM_KV_HEADS as *const _ as *mut _),
                4,
                8,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&HEAD_DIM as *const _ as *mut _),
                4,
                9,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&BLOCK_SIZE as *const _ as *mut _),
                4,
                10,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&MAX_BLOCKS_PER_SEQ as *const _ as *mut _),
                4,
                11,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&attn_scale as *const _ as *mut _),
                4,
                12,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&attention_window as *const _ as *mut _),
                4,
                13,
            );

            let groups = MTLSize {
                width: (NUM_TOKENS * NUM_HEADS) as usize,
                height: 1,
                depth: 1,
            };
            let tpg = MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let expected = attention_decode_reference(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            NUM_TOKENS,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            MAX_BLOCKS_PER_SEQ,
            attn_scale,
            attention_window,
        );

        let got = unsafe {
            std::slice::from_raw_parts(arena.host_ptr(&out_region) as *const half::f16, q.len())
        };
        for i in 0..got.len() {
            assert!((got[i].to_f32() - expected[i]).abs() < 1e-2);
        }
        Ok(())
    }
}

fn low_bit_research_encoding_error() -> rvllm_core::RvllmError {
    rvllm_core::RvllmError::apple(
        rvllm_core::AppleError::InvalidWeightBlob {
            reason: "native BF16 low-bit research encode failed",
        },
        rvllm_core::AppleCtx {
            backend: "metal",
            op: "low_bit_research",
            device: "apple-silicon",
        },
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_low_bit_down_projection(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    projection: MetalLowBitProjectionOffsets,
    activation_offset: usize,
    output_offset: usize,
    num_tokens: u32,
    phase: MetalPhase,
    dims: &MetalLayerDims,
) -> Result<()> {
    if crate::donor12b::simdgroups(pipelines.kernel_options().research).is_some()
        && pipelines.float_type() == Some(crate::MetalFloatType::Bf16)
    {
        let encoded = crate::donor12b_metal::try_encode_low_bit_projection(
            pipelines,
            cmd_buf,
            buf,
            dims,
            phase,
            projection,
            activation_offset,
            output_offset,
            projection.shape()[0],
            0,
        )?;
        if !encoded {
            projection
                .encode_strided_bf16_n4(
                    cmd_buf,
                    pipelines,
                    buf,
                    activation_offset,
                    output_offset,
                    num_tokens as usize,
                    projection.shape()[0] as usize,
                    0,
                )
                .map_err(|_| low_bit_research_encoding_error())?;
        }
        return Ok(());
    }
    let selected = pipelines.kernel_options().research;
    let targeted = matches!(
        (selected, projection.role()),
        (
            crate::MetalResearchCandidate::QmvW4G32R8Sg2,
            AppleLowBitTensorRole::DenseDownProjection
        ) | (
            crate::MetalResearchCandidate::QmvW4G32R4Sg8K8,
            AppleLowBitTensorRole::DenseDownProjection
        ) | (
            crate::MetalResearchCandidate::QmvW8G32R8Sg2,
            AppleLowBitTensorRole::OutputProjection
        ) | (
            crate::MetalResearchCandidate::QmvW8G32R4Sg8K8,
            AppleLowBitTensorRole::OutputProjection
        )
    );
    // This selector owns decode only. Prefill and other phases retain the
    // incumbent route even when the research candidate is selected.
    if targeted
        && pipelines.float_type() == Some(crate::MetalFloatType::Bf16)
        && matches!(phase, MetalPhase::Decode)
    {
        // Never reinterpret BF16 activations as the legacy F16 low-bit ABI on
        // refusal. The existing n4 BF16 schedule is the role-specific control.
        let encoded = projection
            .try_encode_strided_bf16_decode_candidate(
                cmd_buf,
                pipelines,
                buf,
                activation_offset,
                output_offset,
                num_tokens as usize,
                projection.shape()[0] as usize,
                0,
            )
            .map_err(|_| low_bit_research_encoding_error())?;
        if !encoded {
            projection
                .encode_strided_bf16_n4(
                    cmd_buf,
                    pipelines,
                    buf,
                    activation_offset,
                    output_offset,
                    num_tokens as usize,
                    projection.shape()[0] as usize,
                    0,
                )
                .map_err(|_| low_bit_research_encoding_error())?;
        }
        return Ok(());
    }

    projection
        .encode(
            cmd_buf,
            pipelines,
            buf,
            activation_offset,
            output_offset,
            num_tokens as usize,
        )
        .map_err(|_| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::InvalidWeightBlob {
                    reason: "low-bit down projection encoding failed",
                },
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "low_bit_down_projection",
                    device: "apple-silicon",
                },
            )
        })
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_low_bit_projection_strided(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    projection: MetalLowBitProjectionOffsets,
    activation_offset: usize,
    output_offset: usize,
    num_tokens: u32,
    output_row_stride: u32,
    output_column: u32,
    phase: MetalPhase,
    dims: &MetalLayerDims,
) -> Result<()> {
    if crate::donor12b::simdgroups(pipelines.kernel_options().research).is_some()
        && pipelines.float_type() == Some(crate::MetalFloatType::Bf16)
    {
        let encoded = crate::donor12b_metal::try_encode_low_bit_projection(
            pipelines,
            cmd_buf,
            buf,
            dims,
            phase,
            projection,
            activation_offset,
            output_offset,
            output_row_stride,
            output_column,
        )?;
        if !encoded {
            projection
                .encode_strided_bf16_n4(
                    cmd_buf,
                    pipelines,
                    buf,
                    activation_offset,
                    output_offset,
                    num_tokens as usize,
                    output_row_stride as usize,
                    output_column as usize,
                )
                .map_err(|_| low_bit_research_encoding_error())?;
        }
        return Ok(());
    }
    let selected = pipelines.kernel_options().research;
    let targeted = matches!(
        (selected, projection.role()),
        (
            crate::MetalResearchCandidate::QmvW4G32R8Sg2,
            AppleLowBitTensorRole::DenseDownProjection
        ) | (
            crate::MetalResearchCandidate::QmvW4G32R4Sg8K8,
            AppleLowBitTensorRole::DenseDownProjection
        ) | (
            crate::MetalResearchCandidate::QmvW8G32R8Sg2,
            AppleLowBitTensorRole::OutputProjection
        ) | (
            crate::MetalResearchCandidate::QmvW8G32R4Sg8K8,
            AppleLowBitTensorRole::OutputProjection
        )
    );
    // This selector owns decode only. Prefill and other phases retain the
    // incumbent route even when the research candidate is selected.
    if targeted
        && pipelines.float_type() == Some(crate::MetalFloatType::Bf16)
        && matches!(phase, MetalPhase::Decode)
    {
        // Never reinterpret BF16 activations as the legacy F16 low-bit ABI on
        // refusal. The existing n4 BF16 schedule is the role-specific control.
        let encoded = projection
            .try_encode_strided_bf16_decode_candidate(
                cmd_buf,
                pipelines,
                buf,
                activation_offset,
                output_offset,
                num_tokens as usize,
                output_row_stride as usize,
                output_column as usize,
            )
            .map_err(|_| low_bit_research_encoding_error())?;
        if !encoded {
            projection
                .encode_strided_bf16_n4(
                    cmd_buf,
                    pipelines,
                    buf,
                    activation_offset,
                    output_offset,
                    num_tokens as usize,
                    output_row_stride as usize,
                    output_column as usize,
                )
                .map_err(|_| low_bit_research_encoding_error())?;
        }
        return Ok(());
    }

    projection
        .encode_strided(
            cmd_buf,
            pipelines,
            buf,
            activation_offset,
            output_offset,
            num_tokens as usize,
            output_row_stride as usize,
            output_column as usize,
        )
        .map_err(|_| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::InvalidWeightBlob {
                    reason: "low-bit strided projection encoding failed",
                },
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "low_bit_projection_strided",
                    device: "apple-silicon",
                },
            )
        })
}

/// Encode a GEMM operation into the command buffer.
pub(crate) unsafe fn encode_gemm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    c_offset: usize,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    encode_gemm_with_output(
        cmd_buf, pipelines, buf, a_offset, b_offset, c_offset, m, n, k, alpha, beta, false, false,
    )
}

/// The FP32 output mode is used only for the bounded QKV prefill projection.
/// Its caller reserves four-byte scratch and checks separation from planar QKV.
unsafe fn encode_gemm_with_output(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    c_offset: usize,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    output_f32: bool,
    allow_prefill_mma: bool,
) -> Result<()> {
    if crate::donor12b_metal::try_encode_native_projection(
        pipelines,
        cmd_buf,
        buf,
        [a_offset, b_offset, c_offset],
        [m, n, k],
        alpha,
        beta,
        output_f32,
    )? {
        return Ok(());
    }
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "gemm_encode",
                device: "apple-silicon",
            },
        )
    })?;
    let candidate = pipelines.kernel_options().research;
    let decision = crate::research_projection::ProjectionRequest {
        candidate,
        full_prefill: allow_prefill_mma,
        native_bf16: pipelines.float_type() == Some(crate::MetalFloatType::Bf16)
            && !pipelines.kernel_options().quantized_bf16_accumulation,
        alpha,
        beta,
        shape: [m, n, k],
        output_f32,
        offsets: [a_offset, b_offset, c_offset],
        arena_bytes: buf.length(),
    }
    .plan();
    let research = decision.ok().and_then(|plan| {
        let (threads, shared) = plan.kernel.limits();
        pipelines
            .research_pso(plan.kernel.name(), threads, shared)
            .map(|pso| (plan, pso))
    });
    let use_research = research.is_some();
    if let Err(reason) = decision {
        if reason != crate::research_projection::FallbackReason::NotThisOperation {
            tracing::debug!(
                candidate = candidate.name(),
                ?reason,
                m,
                n,
                k,
                "Research projection fallback"
            );
        }
    } else if !use_research {
        tracing::debug!(
            candidate = candidate.name(),
            m,
            n,
            k,
            reason = "pso-or-resource-unavailable",
            "Research projection fallback"
        );
    }
    let use_mma = !use_research && allow_prefill_mma && is_prefill_mma_shape(m, n, k, output_f32);
    let use_batch8 = !use_research
        && !use_mma
        && (output_f32 || supports_batch8_gemm(pipelines.gpu_family(), m, n, k));
    let use_vec = !use_research && !use_mma && !use_batch8 && supports_vec_gemm(m, n, k);
    let use_tiled =
        !use_research && !use_mma && !use_batch8 && !use_vec && supports_tiled_gemm(m, n, k);
    let pso = if let Some((plan, pso)) = research {
        encoder.setLabel(Some(&objc2_foundation::NSString::from_str(
            plan.kernel.name(),
        )));
        tracing::debug!(
            candidate = candidate.name(),
            kernel = plan.kernel.name(),
            m,
            n,
            k,
            output_f32,
            "Research dispatch"
        );
        pso
    } else {
        pipelines.get(if use_mma && output_f32 {
            "qkv_project_f32_mma32"
        } else if use_mma {
            "gemm_f16_mma32"
        } else if output_f32 {
            "qkv_project_f32_batch8"
        } else if use_batch8 {
            "gemm_f16_batch8"
        } else if use_vec {
            "gemm_f16_vec8"
        } else if use_tiled {
            "gemm_f16_tiled16"
        } else {
            "gemm_f16"
        })?
    };
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), c_offset, 2);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&m as *const _ as *mut _),
        4,
        3,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&n as *const _ as *mut _),
        4,
        4,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&k as *const _ as *mut _),
        4,
        5,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&alpha as *const _ as *mut _),
        4,
        6,
    );
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&beta as *const _ as *mut _),
        4,
        7,
    );

    let (groups, tpg) = if let Some((plan, _)) = research {
        (
            MTLSize {
                width: (m as usize).div_ceil(plan.tile_m),
                height: (n as usize).div_ceil(plan.tile_n),
                depth: 1,
            },
            MTLSize {
                width: plan.kernel.limits().0,
                height: 1,
                depth: 1,
            },
        )
    } else if use_mma {
        (
            MTLSize {
                width: (m as usize).div_ceil(32),
                height: (n as usize).div_ceil(32),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        )
    } else if use_batch8 {
        (
            MTLSize {
                width: (m as usize).div_ceil(8),
                height: (n as usize).div_ceil(8),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    } else if use_vec {
        (
            MTLSize {
                width: m as usize,
                height: (n as usize).div_ceil(8),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    } else {
        let tile_m: usize = if use_tiled { 16 } else { 8 };
        let tile_n: usize = if use_tiled { 16 } else { 8 };
        (
            MTLSize {
                width: (m as usize).div_ceil(tile_m),
                height: (n as usize).div_ceil(tile_n),
                depth: 1,
            },
            MTLSize {
                width: tile_m,
                height: tile_n,
                depth: 1,
            },
        )
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    if let Some((plan, _)) = research {
        pipelines.record_research_dispatch(plan.kernel);
    }
    Ok(())
}

/// One encoder replaces the gate/up GEMM and activation encoder, without
/// changing the command-buffer dependency or arena ownership model.
unsafe fn try_encode_research_rounded_gate(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    weights: &MetalLayerWeights,
    scratch: &MetalScratch,
    capture_gate_up: bool,
) -> Result<bool> {
    let Some(pso) = pipelines.research_pso("research_rounded_gate32", 128, 14_336) else {
        return Ok(false);
    };
    if !crate::research::rounded_gate_buffers_fit(
        [
            scratch.normed_hidden,
            weights.gate_up_offset,
            scratch.gate_up_out,
            scratch.activated,
        ],
        dims.num_tokens,
        dims.hidden,
        dims.intermediate,
        buf.length(),
        capture_gate_up,
    ) {
        return Ok(false);
    }
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "research_rounded_gate32",
                device: "apple-silicon",
            },
        )
    })?;
    encoder.setLabel(Some(&objc2_foundation::NSString::from_str(
        "metal-rounded-gate32",
    )));
    encoder.setComputePipelineState(pso);
    for (index, offset) in [
        scratch.normed_hidden,
        weights.gate_up_offset,
        scratch.activated,
        scratch.gate_up_out,
    ]
    .into_iter()
    .enumerate()
    {
        encoder.setBuffer_offset_atIndex(Some(buf), offset, index);
    }
    let capture = u32::from(capture_gate_up);
    for (index, value) in [dims.num_tokens, dims.hidden, dims.intermediate, capture]
        .iter()
        .enumerate()
    {
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(value).cast(), 4, index + 4);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: (dims.num_tokens as usize).div_ceil(32),
            height: (dims.intermediate as usize).div_ceil(32),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    pipelines.record_research_dispatch(crate::research_evidence::ResearchKernel::RoundedGate);
    tracing::debug!(
        candidate = "metal-rounded-gate32",
        tokens = dims.num_tokens,
        capture_gate_up,
        "Research dispatch"
    );
    Ok(true)
}
