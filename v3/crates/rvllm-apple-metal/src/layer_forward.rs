//! Metal layer forward pass for transformer decoder blocks.
//!
//! Mirrors the CUDA `layer_exec::forward_phase()` but uses Metal compute
//! encoders instead of CUDA kernel launches.

use crate::arena::MetalBufferArena;
use crate::context::MetalContext;
use crate::pipeline::PipelineCache;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};
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
    pub down_proj_offset: usize,
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
    let qkv_len = qkv_elems.checked_mul(elem).ok_or_else(|| {
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

fn supports_attention_decode_reduction(dims: &MetalLayerDims) -> bool {
    dims.num_heads > 0
        && dims.num_kv_heads > 0
        && dims.num_heads % dims.num_kv_heads == 0
        && dims.head_dim <= 256
        && dims.block_size > 0
        && dims.max_blocks_per_seq > 0
}

fn supports_tiled_gemm(m: u32, n: u32, k: u32) -> bool {
    m > 0 && n > 0 && k > 0 && m <= 16 && n <= 1024 && k <= 1024
}

fn supports_fused_final_logits_small(num_tokens: u32, hidden: u32, vocab: u32) -> bool {
    num_tokens > 0 && num_tokens <= 256 && hidden > 0 && hidden <= 4096 && vocab > 0 && vocab <= 256
}

#[must_use]
pub fn metal_finalize_logits_encoder_count(
    num_tokens: u32,
    hidden: u32,
    vocab: u32,
    softcap: f32,
) -> u64 {
    if supports_fused_final_logits_small(num_tokens, hidden, vocab) {
        1
    } else {
        3 + (softcap > 0.0) as u64
    }
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
    validate_qkv_scratch_planar(
        scratch,
        num_tokens as usize,
        q_dim as usize,
        kv_dim as usize,
    )?;

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

    // 2. QKV GEMM: normed_hidden[M,K] × W_qkv[K,N] → qkv_out[M,N]
    encode_gemm(
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
    )?;

    // 3. Split fused QKV into planar Q/K/V scratch regions.
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

    // 4. Optional Gemma-style Q/K/V norms before RoPE and KV cache write.
    if let Some(q_norm_offset) = weights.q_norm_offset {
        encode_rmsnorm(
            &cmd_buf,
            pipelines,
            buf,
            scratch.q_offset,
            scratch.q_offset,
            q_norm_offset,
            q_dim,
            dims.rms_eps,
            num_tokens,
            "q_norm",
        )?;
    }
    if let Some(k_norm_offset) = weights.k_norm_offset {
        encode_rmsnorm(
            &cmd_buf,
            pipelines,
            buf,
            scratch.k_offset,
            scratch.k_offset,
            k_norm_offset,
            kv_dim,
            dims.rms_eps,
            num_tokens,
            "k_norm",
        )?;
    }
    if let Some(v_norm_offset) = weights.v_norm_offset {
        encode_rmsnorm(
            &cmd_buf,
            pipelines,
            buf,
            scratch.v_offset,
            scratch.v_offset,
            v_norm_offset,
            kv_dim,
            dims.rms_eps,
            num_tokens,
            "v_norm",
        )?;
    }

    // 5. RoPE: apply partial RoPE to Q and K
    {
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

    // 6. KV cache write
    {
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

    // 7. Attention
    match phase {
        MetalPhase::Decode => {
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
            let use_reduction = supports_attention_decode_reduction(dims);
            let pso = pipelines.get(if use_reduction {
                "attention_decode_reduction_f16"
            } else {
                "attention_decode_f16"
            })?;
            encoder.setComputePipelineState(pso);
            encoder.setBuffer_offset_atIndex(Some(buf), scratch.q_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_k_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), kv_cache_v_offset, 2);
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
                std::ptr::NonNull::new_unchecked(&dims.max_blocks_per_seq as *const _ as *mut _),
                4,
                11,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.attn_scale as *const _ as *mut _),
                4,
                12,
            );
            let total_heads = num_tokens * dims.num_heads;
            let groups = MTLSize {
                width: total_heads as usize,
                height: 1,
                depth: 1,
            };
            let tpg = MTLSize {
                width: if use_reduction { 256 } else { 1 },
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
            encoder.endEncoding();
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
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&total_q as *const _ as *mut _),
                4,
                7,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&batch_size as *const _ as *mut _),
                4,
                8,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.num_heads as *const _ as *mut _),
                4,
                9,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.num_kv_heads as *const _ as *mut _),
                4,
                10,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.head_dim as *const _ as *mut _),
                4,
                11,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.block_size as *const _ as *mut _),
                4,
                12,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.max_blocks_per_seq as *const _ as *mut _),
                4,
                13,
            );
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&dims.attn_scale as *const _ as *mut _),
                4,
                14,
            );
            let groups = MTLSize {
                width: total_q as usize,
                height: dims.num_heads as usize,
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
    }

    // 8. O projection with residual: residual += attn_out × W_o
    encode_gemm_residual(
        &cmd_buf,
        pipelines,
        buf,
        scratch.attn_out,
        weights.o_proj_offset,
        residual_offset,
        residual_offset,
        num_tokens,
        hidden,
        q_dim,
        weights.layer_scalar_offset,
        weights.layer_scalar_dim,
    )?;

    // 9. Optional Gemma-style post-attention RMSNorm on the residual stream.
    if let Some(post_attn_norm_offset) = weights.post_attn_norm_offset {
        encode_rmsnorm(
            &cmd_buf,
            pipelines,
            buf,
            residual_offset,
            residual_offset,
            post_attn_norm_offset,
            hidden,
            dims.rms_eps,
            num_tokens,
            "post_attn_norm",
        )?;
    }

    // 10. Pre-FFN RMSNorm. Explicit pre-FF Gemma norm overrides the legacy MLP norm.
    encode_rmsnorm(
        &cmd_buf,
        pipelines,
        buf,
        residual_offset,
        scratch.normed_hidden,
        weights
            .pre_ff_norm_offset
            .unwrap_or(weights.mlp_norm_offset),
        hidden,
        dims.rms_eps,
        num_tokens,
        "ffn_norm",
    )?;

    // 11. Gate||Up projection: normed_hidden × W_gate_up → gate_up_out
    let two_inter = 2 * dims.intermediate;
    encode_gemm(
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
    )?;

    // 12. GELU(gate) * up → activated
    {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "gelu_mul",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("gelu_mul_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.gate_up_out, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), scratch.activated, 1);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&num_tokens as *const _ as *mut _),
            4,
            2,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&dims.intermediate as *const _ as *mut _),
            4,
            3,
        );
        let groups = MTLSize {
            width: num_tokens as usize,
            height: dims.intermediate as usize,
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

    // 13. Down projection with residual: residual += activated × W_down
    encode_gemm_residual(
        &cmd_buf,
        pipelines,
        buf,
        scratch.activated,
        weights.down_proj_offset,
        residual_offset,
        residual_offset,
        num_tokens,
        hidden,
        dims.intermediate,
        weights.layer_scalar_offset,
        weights.layer_scalar_dim,
    )?;

    // 14. Optional Gemma-style post-FF RMSNorm on the residual stream.
    if let Some(post_ff_norm_offset) = weights.post_ff_norm_offset {
        encode_rmsnorm(
            &cmd_buf,
            pipelines,
            buf,
            residual_offset,
            residual_offset,
            post_ff_norm_offset,
            hidden,
            dims.rms_eps,
            num_tokens,
            "post_ff_norm",
        )?;
    }

    cmd_buf.commit();

    Ok(())
}

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
    let pso = pipelines.get("rmsnorm_f16")?;
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
    let tpg = MTLSize {
        width: 256,
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

    // 3) optional softcap
    if softcap > 0.0 {
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            rvllm_core::RvllmError::apple(
                rvllm_core::AppleError::MetalUnavailable,
                rvllm_core::AppleCtx {
                    backend: "metal",
                    op: "final_softcap_encoder",
                    device: "apple-silicon",
                },
            )
        })?;
        let pso = pipelines.get("softcap_f16")?;
        encoder.setComputePipelineState(pso);
        encoder.setBuffer_offset_atIndex(Some(buf), logits_offset, 0);
        let count = num_tokens.saturating_mul(vocab);
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&count as *const _ as *mut _),
            4,
            1,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&softcap as *const _ as *mut _),
            4,
            2,
        );
        let groups = MTLSize {
            width: (count as usize + 255) / 256,
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

    // 4) argmax -> sampled token IDs
    {
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

    encode_logits_head(
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

    encode_logits_head(
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
                for t in 0..ctx_len {
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
            gate_up_out: 0,
            activated: 0,
            mlp_out: 0,
        };
        assert!(validate_qkv_scratch_planar(&scratch, 1, 8, 4).is_err());
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
            gate_up_out: 0,
            activated: 0,
            mlp_out: 0,
        };
        assert!(validate_qkv_scratch_planar(&scratch, 1, 8, 4).is_err());
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
            gate_up_out: 0,
            activated: 0,
            mlp_out: 0,
        };
        assert!(
            validate_qkv_scratch_planar(&scratch, 1, 8, 4).is_ok(),
            "non-overlapping planar scratch regions should be accepted"
        );
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
        assert_eq!(metal_finalize_logits_encoder_count(2, 4, 257, 30.0), 4);
        assert_eq!(metal_finalize_logits_encoder_count(2, 4, 257, 0.0), 3);
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

        const NUM_TOKENS: u32 = 1;
        const NUM_HEADS: u32 = 2;
        const NUM_KV_HEADS: u32 = 1;
        const HEAD_DIM: u32 = 8;
        const BLOCK_SIZE: u32 = 4;
        const MAX_BLOCKS_PER_SEQ: u32 = 1;
        let attn_scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let q_dim = NUM_HEADS * HEAD_DIM;
        let kv_dim = NUM_KV_HEADS * HEAD_DIM;

        let q: Vec<half::f16> = (0..q_dim)
            .map(|i| half::f16::from_f32(((i as f32) + 1.0) * 0.03))
            .collect();
        let k_cache: Vec<half::f16> = (0..(BLOCK_SIZE * kv_dim) as usize)
            .map(|i| half::f16::from_f32(((i as f32) + 1.0) * 0.01))
            .collect();
        let v_cache: Vec<half::f16> = (0..(BLOCK_SIZE * kv_dim) as usize)
            .map(|i| half::f16::from_f32(((i as f32) + 1.0) * 0.02))
            .collect();
        let block_tables = vec![0_i32];
        let context_lens = vec![4_i32];

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
        let pso = pipelines.get("attention_decode_reduction_f16")?;
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

            let groups = MTLSize {
                width: (NUM_TOKENS * NUM_HEADS) as usize,
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

/// Encode a GEMM operation into the command buffer.
unsafe fn encode_gemm(
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
    let use_tiled = supports_tiled_gemm(m, n, k);
    let pso = pipelines.get(if use_tiled {
        "gemm_f16_tiled16"
    } else {
        "gemm_f16"
    })?;
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

    let tile_m: usize = if use_tiled { 16 } else { 8 };
    let tile_n: usize = if use_tiled { 16 } else { 8 };
    let groups_x = (m as usize + tile_m - 1) / tile_m;
    let groups_y = (n as usize + tile_n - 1) / tile_n;
    let groups = MTLSize {
        width: groups_x,
        height: groups_y,
        depth: 1,
    };
    let tpg = MTLSize {
        width: tile_m,
        height: tile_n,
        depth: 1,
    };
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
    m: u32,
    n: u32,
    k: u32,
    layer_scalar_offset: Option<usize>,
    layer_scalar_dim: u32,
) -> Result<()> {
    let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
        rvllm_core::RvllmError::apple(
            rvllm_core::AppleError::MetalUnavailable,
            rvllm_core::AppleCtx {
                backend: "metal",
                op: "gemm_res_encode",
                device: "apple-silicon",
            },
        )
    })?;
    let pso = pipelines.get("gemm_residual_f16")?;
    encoder.setComputePipelineState(pso);
    encoder.setBuffer_offset_atIndex(Some(buf), a_offset, 0);
    encoder.setBuffer_offset_atIndex(Some(buf), b_offset, 1);
    encoder.setBuffer_offset_atIndex(Some(buf), c_offset, 2);
    encoder.setBuffer_offset_atIndex(Some(buf), residual_offset, 3);
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
    let layer_scalar_dim = if layer_scalar_offset.is_some() {
        layer_scalar_dim
    } else {
        0
    };
    encoder.setBuffer_offset_atIndex(Some(buf), layer_scalar_offset.unwrap_or(residual_offset), 7);
    encoder.setBytes_length_atIndex(
        std::ptr::NonNull::new_unchecked(&layer_scalar_dim as *const _ as *mut _),
        4,
        8,
    );

    let tile_m: usize = 8;
    let tile_n: usize = 8;
    let groups_x = (m as usize + tile_m - 1) / tile_m;
    let groups_y = (n as usize + tile_n - 1) / tile_n;
    let groups = MTLSize {
        width: groups_x,
        height: groups_y,
        depth: 1,
    };
    let tpg = MTLSize {
        width: tile_m,
        height: tile_n,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tpg);
    encoder.endEncoding();
    Ok(())
}
