//! Metal layer forward pass for transformer decoder blocks.
//!
//! Mirrors the CUDA `layer_exec::forward_phase()` but uses Metal compute
//! encoders instead of CUDA kernel launches.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLSize,
    MTLCommandQueue, MTLCommandEncoder,
};
use crate::arena::{MetalBufferArena, MetalRegion};
use crate::pipeline::PipelineCache;
use crate::context::MetalContext;
use rvllm_core::Result;

/// Dimensions for a single decoder layer, matching rvllm-runtime's LayerDims.
#[derive(Copy, Clone, Debug)]
pub struct MetalLayerDims {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
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
    pub o_proj_offset: usize,
    pub mlp_norm_offset: usize,
    pub gate_up_offset: usize,
    pub down_proj_offset: usize,
}

/// Pre-allocated scratch buffer offsets.
#[derive(Copy, Clone, Debug)]
pub struct MetalScratch {
    pub normed_hidden: usize,
    pub qkv_out: usize,  // packed [Q, K, V]
    pub q_offset: usize,
    pub k_offset: usize,
    pub v_offset: usize,
    pub attn_out: usize,
    pub gate_up_out: usize,
    pub activated: usize,
    pub mlp_out: usize,
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

/// Which phase: decode (1 token/seq) or prefill (multi-token/seq).
#[derive(Copy, Clone, Debug)]
pub enum MetalPhase {
    Decode,
    Prefill {
        max_seqlen_q: u32,
        batch_size: u32,
    },
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
    meta: &MetalMetadata,
    residual_offset: usize,
    phase: MetalPhase,
    kv_cache_k_offset: usize,
    kv_cache_v_offset: usize,
) -> Result<()> {
    let queue = ctx.queue_retained();
    let buf = arena.buffer_retained();
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

    let num_tokens = dims.num_tokens;
    let hidden = dims.hidden;
    let q_dim = dims.num_heads * dims.head_dim;
    let kv_dim = dims.num_kv_heads * dims.head_dim;
    let qkv_n = q_dim + 2 * kv_dim;

    // 1. RMSNorm(residual) → normed_hidden
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx { backend: "metal", op: "rmsnorm_encoder", device: "apple-silicon" },
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
        let threads_per_group = MTLSize { width: 256, height: 1, depth: 1 };
        let groups = MTLSize { width: num_tokens as usize, height: 1, depth: 1 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();
    }

    // 2. QKV GEMM: normed_hidden[M,K] × W_qkv[K,N] → qkv_out[M,N]
    encode_gemm(
        &cmd_buf, pipelines, buf,
        scratch.normed_hidden, weights.qkv_offset, scratch.qkv_out,
        num_tokens, qkv_n, hidden,
        1.0, 0.0,
    )?;

    // 3. RoPE: apply partial RoPE to Q and K portions of qkv_out
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx { backend: "metal", op: "rope_encoder", device: "apple-silicon" },
            )
        })?;
        let pso = pipelines.get("rope_partial_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.q_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.k_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), meta.cos_offset, 2);
        encoder.setBuffer_offset_atIndex(Some(buf), meta.sin_offset, 3);
        encoder.setBuffer_offset_atIndex(Some(buf), meta.positions_offset, 4);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _), 4, 5);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.num_heads as *const _ as *mut _), 4, 6);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.num_kv_heads as *const _ as *mut _), 4, 7);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.head_dim as *const _ as *mut _), 4, 8);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.rope_dim as *const _ as *mut _), 4, 9);
        let half_rope = dims.rope_dim / 2;
        let groups = MTLSize { width: num_tokens as usize, height: half_rope as usize, depth: 1 };
        let tpg = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    }

    // 4. KV cache write
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx { backend: "metal", op: "kv_write_encoder", device: "apple-silicon" },
            )
        })?;
        let pso = pipelines.get("kv_cache_write_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.k_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.v_offset, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_k_offset, 2);
        encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_v_offset, 3);
        encoder.setBuffer_offset_atIndex(Some(buf), meta.slot_mapping_offset, 4);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _), 4, 5);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&kv_dim as *const _ as *mut _), 4, 6);
        let groups = MTLSize { width: num_tokens as usize, height: kv_dim as usize, depth: 1 };
        let tpg = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    }

    // 5. Attention
    match phase {
        MetalPhase::Decode => {
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx { backend: "metal", op: "attn_decode", device: "apple-silicon" },
                )
            })?;
            let pso = pipelines.get("attention_decode_f16")?;
            encoder.setComputePipelineState(pso);
            encoder.setBuffer_offset_atIndex(Some(buf), scratch.q_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_k_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_v_offset, 2);
            encoder.setBuffer_offset_atIndex(Some(buf), scratch.attn_out, 3);
            encoder.setBuffer_offset_atIndex(Some(buf), meta.block_tables_offset, 4);
            encoder.setBuffer_offset_atIndex(Some(buf), meta.context_lens_offset, 5);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _), 4, 6);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.num_heads as *const _ as *mut _), 4, 7);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.num_kv_heads as *const _ as *mut _), 4, 8);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.head_dim as *const _ as *mut _), 4, 9);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.block_size as *const _ as *mut _), 4, 10);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.max_blocks_per_seq as *const _ as *mut _), 4, 11);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.attn_scale as *const _ as *mut _), 4, 12);
            let total_heads = num_tokens * dims.num_heads;
            let groups = MTLSize { width: total_heads as usize, height: 1, depth: 1 };
            let tpg = MTLSize { width: 256.min(dims.head_dim as usize), height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
            encoder.endEncoding();
        }
        MetalPhase::Prefill { max_seqlen_q: _, batch_size } => {
            let total_q = num_tokens;
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                rvllm_core::RvllmError::apple(
                    rvllm_core::AppleError::MetalUnavailable,
                    rvllm_core::AppleCtx { backend: "metal", op: "attn_prefill", device: "apple-silicon" },
                )
            })?;
            let pso = pipelines.get("attention_prefill_f16")?;
            encoder.setComputePipelineState(pso);
            encoder.setBuffer_offset_atIndex(Some(buf), scratch.q_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_k_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_v_offset, 2);
            encoder.setBuffer_offset_atIndex(Some(buf), scratch.attn_out, 3);
            encoder.setBuffer_offset_atIndex(Some(buf), meta.block_tables_offset, 4);
            encoder.setBuffer_offset_atIndex(Some(buf), meta.context_lens_offset, 5);
            if let Some(cu_off) = meta.cu_seqlens_offset {
                encoder.setBuffer_offset_atIndex(Some(buf), cu_off, 6);
            }
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&total_q as *const _ as *mut _), 4, 7);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&batch_size as *const _ as *mut _), 4, 8);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.num_heads as *const _ as *mut _), 4, 9);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.num_kv_heads as *const _ as *mut _), 4, 10);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.head_dim as *const _ as *mut _), 4, 11);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.block_size as *const _ as *mut _), 4, 12);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.max_blocks_per_seq as *const _ as *mut _), 4, 13);
            encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.attn_scale as *const _ as *mut _), 4, 14);
            let groups = MTLSize { width: total_q as usize, height: dims.num_heads as usize, depth: 1 };
            let tpg = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
            encoder.endEncoding();
        }
    }

    // 6. O projection with residual: residual += attn_out × W_o
    encode_gemm_residual(
        &cmd_buf, pipelines, buf,
        scratch.attn_out, weights.o_proj_offset, residual_offset, residual_offset,
        num_tokens, hidden, q_dim,
    )?;

    // 7. Pre-FFN RMSNorm
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx { backend: "metal", op: "ffn_norm", device: "apple-silicon" },
            )
        })?;
        let pso = pipelines.get("rmsnorm_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.normed_hidden, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), weights.mlp_norm_offset, 2);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _), 4, 3);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.rms_eps as *const _ as *mut _), 4, 4);
        let threads_per_group = MTLSize { width: 256, height: 1, depth: 1 };
        let groups = MTLSize { width: num_tokens as usize, height: 1, depth: 1 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();
    }

    // 8. Gate||Up projection: normed_hidden × W_gate_up → gate_up_out
    let two_inter = 2 * dims.intermediate;
    encode_gemm(
        &cmd_buf, pipelines, buf,
        scratch.normed_hidden, weights.gate_up_offset, scratch.gate_up_out,
        num_tokens, two_inter, hidden,
        1.0, 0.0,
    )?;

    // 9. GELU(gate) * up → activated
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx { backend: "metal", op: "gelu_mul", device: "apple-silicon" },
            )
        })?;
        let pso = pipelines.get("gelu_mul_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.gate_up_out, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.activated, 1);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _), 4, 2);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&dims.intermediate as *const _ as *mut _), 4, 3);
        let groups = MTLSize { width: num_tokens as usize, height: dims.intermediate as usize, depth: 1 };
        let tpg = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.dispatchThreads_threadsPerThreadgroup(groups, tpg);
        encoder.endEncoding();
    }

    // 10. Down projection with residual: residual += activated × W_down
    encode_gemm_residual(
        &cmd_buf, pipelines, buf,
        scratch.activated, weights.down_proj_offset, residual_offset, residual_offset,
        num_tokens, hidden, dims.intermediate,
    )?;

    cmd_buf.commit();

    Ok(())
}

/// Encode a GEMM operation into the command buffer.
unsafe fn encode_gemm(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    c_offset: usize,
    m: u32, n: u32, k: u32,
    alpha: f32, beta: f32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx { backend: "metal", op: "gemm_encode", device: "apple-silicon" },
        )
    })?;
    let pso = pipelines.get("gemm_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), c_offset, 2);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&m as *const _ as *mut _), 4, 3);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&n as *const _ as *mut _), 4, 4);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&k as *const _ as *mut _), 4, 5);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&alpha as *const _ as *mut _), 4, 6);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&beta as *const _ as *mut _), 4, 7);

    let tile_m: usize = 8;
    let tile_n: usize = 8;
    let groups_x = (m as usize + tile_m - 1) / tile_m;
    let groups_y = (n as usize + tile_n - 1) / tile_n;
    let groups = MTLSize { width: groups_x, height: groups_y, depth: 1 };
    let tpg = MTLSize { width: tile_m, height: tile_n, depth: 1 };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}

/// Encode a GEMM + residual add operation.
unsafe fn encode_gemm_residual(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &PipelineCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    a_offset: usize,
    b_offset: usize,
    c_offset: usize,
    residual_offset: usize,
    m: u32, n: u32, k: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx { backend: "metal", op: "gemm_res_encode", device: "apple-silicon" },
        )
    })?;
    let pso = pipelines.get("gemm_residual_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), c_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 3);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&m as *const _ as *mut _), 4, 4);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&n as *const _ as *mut _), 4, 5);
    encoder.setBytes_length_atIndex(std::ptr::NonNull::new_unchecked(&k as *const _ as *mut _), 4, 6);

    let tile_m: usize = 8;
    let tile_n: usize = 8;
    let groups_x = (m as usize + tile_m - 1) / tile_m;
    let groups_y = (n as usize + tile_n - 1) / tile_n;
    let groups = MTLSize { width: groups_x, height: groups_y, depth: 1 };
    let tpg = MTLSize { width: tile_m, height: tile_n, depth: 1 };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}
