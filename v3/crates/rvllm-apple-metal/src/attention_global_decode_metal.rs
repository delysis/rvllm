//! The single normal-route adapter for global cooperative decode. No source
//! compilation, labels/NSString creation, logging, allocation, environment reads
//! or clocks here. Rejected requests do not create an encoder or record work.

use crate::attention_global_decode::{
    DecodeBuffers, DecodeOutput, DecodePlan, DecodeShape, SplitDecodeBuffers, SplitDecodePlan,
};
use crate::layer_forward::{MetalLayerDims, MetalPhase};
use crate::research::Gemma12bResearchShape;
use crate::research_evidence::ResearchKernel;
use crate::{MetalFloatType, PipelineCache};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodedGlobalDecode {
    pub plan: DecodePlan,
    pub kernel: ResearchKernel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodedSplitGlobalDecode {
    pub plan: SplitDecodePlan,
    pub partial_kernel: ResearchKernel,
    pub merge_kernel: ResearchKernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitGlobalDecodeStage {
    Partial,
    Merge,
}

#[allow(clippy::too_many_arguments)]
fn try_encode_split_global_decode_impl(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    buffers: SplitDecodeBuffers,
    output: DecodeOutput,
    stages: (bool, bool),
) -> Result<Option<EncodedSplitGlobalDecode>> {
    let candidate = pipelines.kernel_options().research;
    let Some(tile) = candidate.split_global_decode_tile() else {
        return Ok(None);
    };
    if !matches!(phase, MetalPhase::Decode)
        || pipelines.float_type() != Some(MetalFloatType::Bf16)
        || pipelines.kernel_options().quantized_bf16_accumulation
        || !(Gemma12bResearchShape {
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
    {
        return Ok(None);
    }
    let shape = DecodeShape {
        sequences: dims.num_tokens,
        heads: dims.num_heads,
        kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        block_size: dims.block_size,
        max_blocks: dims.max_blocks_per_seq,
        num_blocks: dims.num_blocks_total,
        window: dims.attention_window,
        scale: dims.attn_scale,
    };
    let Some(plan) = SplitDecodePlan::new(tile, shape, output) else {
        return Ok(None);
    };
    if !plan.buffers_fit(buffers, buffer.length()) {
        return Ok(None);
    }
    let [partial_kernel, merge_kernel] = candidate.kernels() else {
        return Ok(None);
    };
    // Resolve the complete two-PSO route before creating either encoder.
    let Some(partial_pso) = pipelines.research_pso(
        partial_kernel.name(),
        plan.partial_threads[0],
        plan.partial_threadgroup_bytes,
    ) else {
        return Ok(None);
    };
    let Some(merge_pso) = pipelines.research_pso(
        merge_kernel.name(),
        plan.merge_threads[0],
        plan.merge_threadgroup_bytes,
    ) else {
        return Ok(None);
    };
    let size = |xyz: [usize; 3]| MTLSize {
        width: xyz[0],
        height: xyz[1],
        depth: xyz[2],
    };
    let params = plan.params();
    if stages.0 {
        let partial = command.computeCommandEncoder().ok_or_else(|| {
            RvllmError::apple(
                AppleError::MetalUnavailable,
                AppleCtx {
                    backend: "metal",
                    op: "global_d512_split_partial",
                    device: "apple-silicon",
                },
            )
        })?;
        partial.setComputePipelineState(partial_pso);
        unsafe {
            let offsets = [
                buffers.common.q,
                buffers.common.k,
                buffers.common.v,
                buffers.partials,
                buffers.common.block_tables,
                buffers.common.context_lens,
                buffers.common.positions,
            ];
            for (index, offset) in offsets.into_iter().enumerate() {
                partial.setBuffer_offset_atIndex(Some(buffer), offset, index);
            }
            partial.setBytes_length_atIndex(
                std::ptr::NonNull::from(&params).cast(),
                std::mem::size_of_val(&params),
                7,
            );
        }
        partial.dispatchThreadgroups_threadsPerThreadgroup(
            size(plan.partial_grid),
            size(plan.partial_threads),
        );
        partial.endEncoding();
        pipelines.record_research_dispatch(*partial_kernel);
    }

    if stages.1 {
        let merge = command.computeCommandEncoder().ok_or_else(|| {
            RvllmError::apple(
                AppleError::MetalUnavailable,
                AppleCtx {
                    backend: "metal",
                    op: "global_d512_split_merge",
                    device: "apple-silicon",
                },
            )
        })?;
        merge.setComputePipelineState(merge_pso);
        unsafe {
            for (index, offset) in [
                buffers.partials,
                buffers.common.output,
                buffers.common.block_tables,
                buffers.common.context_lens,
                buffers.common.positions,
            ]
            .into_iter()
            .enumerate()
            {
                merge.setBuffer_offset_atIndex(Some(buffer), offset, index);
            }
            merge.setBytes_length_atIndex(
                std::ptr::NonNull::from(&params).cast(),
                std::mem::size_of_val(&params),
                5,
            );
        }
        merge.dispatchThreadgroups_threadsPerThreadgroup(
            size(plan.merge_grid),
            size(plan.merge_threads),
        );
        merge.endEncoding();
        pipelines.record_research_dispatch(*merge_kernel);
    }
    Ok(Some(EncodedSplitGlobalDecode {
        plan,
        partial_kernel: *partial_kernel,
        merge_kernel: *merge_kernel,
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn try_encode_split_global_decode(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    buffers: SplitDecodeBuffers,
    output: DecodeOutput,
) -> Result<Option<EncodedSplitGlobalDecode>> {
    try_encode_split_global_decode_impl(
        pipelines,
        command,
        buffer,
        dims,
        phase,
        buffers,
        output,
        (true, true),
    )
}

/// Encodes one half of the split route for identity-bound native timing. This
/// uses the same predicate, plan, PSOs and bindings as the production adapter;
/// callers remain responsible for ordered partial-before-merge submission.
#[allow(clippy::too_many_arguments)]
pub fn try_encode_split_global_decode_stage(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    buffers: SplitDecodeBuffers,
    output: DecodeOutput,
    stage: SplitGlobalDecodeStage,
) -> Result<Option<EncodedSplitGlobalDecode>> {
    let stages = match stage {
        SplitGlobalDecodeStage::Partial => (true, false),
        SplitGlobalDecodeStage::Merge => (false, true),
    };
    try_encode_split_global_decode_impl(
        pipelines, command, buffer, dims, phase, buffers, output, stages,
    )
}

/// Used by the real layer path and the ignored native oracle, not an alternate
/// test-only dispatch predicate. Producer completion/resource ownership remains
/// the caller's existing Metal command-buffer/arena contract.
#[allow(clippy::too_many_arguments)]
pub fn try_encode_global_decode(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    buffers: DecodeBuffers,
    output: DecodeOutput,
) -> Result<Option<EncodedGlobalDecode>> {
    let candidate = pipelines.kernel_options().research;
    let Some(tile) = candidate.global_decode_tile() else {
        return Ok(None);
    };
    if !matches!(phase, MetalPhase::Decode)
        || pipelines.float_type() != Some(MetalFloatType::Bf16)
        || pipelines.kernel_options().quantized_bf16_accumulation
        || !(Gemma12bResearchShape {
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
    {
        return Ok(None);
    }
    let shape = DecodeShape {
        sequences: dims.num_tokens,
        heads: dims.num_heads,
        kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        block_size: dims.block_size,
        max_blocks: dims.max_blocks_per_seq,
        num_blocks: dims.num_blocks_total,
        window: dims.attention_window,
        scale: dims.attn_scale,
    };
    let Some(plan) = DecodePlan::new(tile, shape, output) else {
        return Ok(None);
    };
    if !plan.buffers_fit(buffers, buffer.length()) {
        return Ok(None);
    }
    let [kernel] = candidate.kernels() else {
        return Ok(None);
    };
    let Some(pso) =
        pipelines.research_pso(kernel.name(), tile.threads as usize, plan.threadgroup_bytes)
    else {
        return Ok(None);
    };
    let encoder = command.computeCommandEncoder().ok_or_else(|| {
        RvllmError::apple(
            AppleError::MetalUnavailable,
            AppleCtx {
                backend: "metal",
                op: "global_d512_decode",
                device: "apple-silicon",
            },
        )
    })?;
    encoder.setComputePipelineState(pso);
    let params = plan.params();
    // SAFETY: the exact shape and checked, aligned, nonaliasing spans above
    // cover every binding. Params is a repr(C) stack value copied by setBytes.
    // Metadata is bounds-checked uniformly in MSL before reads or output writes.
    unsafe {
        for (index, offset) in buffers.offsets().into_iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
        }
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&params).cast(),
            std::mem::size_of_val(&params),
            7,
        );
    }
    let size = |xyz: [usize; 3]| MTLSize {
        width: xyz[0],
        height: xyz[1],
        depth: xyz[2],
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(size(plan.grid), size(plan.threads));
    encoder.endEncoding();
    // Existing append-only receipt hook. A count means encoded, not completed,
    // numerically correct, timing eligible, or promoted.
    pipelines.record_research_dispatch(*kernel);
    Ok(Some(EncodedGlobalDecode {
        plan,
        kernel: *kernel,
    }))
}
