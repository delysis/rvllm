//! Checked, allocation-free native bindings for the donor12b plans.
//! No pack conversion, shader compilation, environment reads, labels, clocks,
//! synchronization waits or host readback occur in these encoding functions.
use crate::donor12b::{
    self, AttentionRequest, GateRequest, Plan, Policy, ProjectionRequest, QkvRequest, Weight,
};
use crate::layer_forward::{MetalLayerDims, MetalLayerWeights, MetalPhase, MetalScratch};
use crate::low_bit_metal::MetalLowBitProjectionOffsets;
use crate::research::Gemma12bResearchShape;
use crate::{MetalFloatType, PipelineCache};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

pub fn policy(pipelines: &PipelineCache, dims: &MetalLayerDims, phase: MetalPhase) -> Policy {
    Policy {
        selected: pipelines.kernel_options().research,
        dtype: pipelines.float_type(),
        quantized_accumulation: pipelines.kernel_options().quantized_bf16_accumulation,
        decode: matches!(phase, MetalPhase::Decode),
        model: Gemma12bResearchShape {
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
    }
}
fn weight(w: MetalLowBitProjectionOffsets) -> Weight {
    let [n, k] = w.shape();
    Weight {
        format: w.format(),
        role: w.role(),
        n,
        k,
        values: w.packed_values_offset(),
        scales: w.scales_offset(),
    }
}
fn available(pipelines: &PipelineCache, plan: &Plan) -> bool {
    pipelines
        .research_pso(plan.kernel.name(), plan.threads()[0], plan.shared_bytes())
        .is_some()
}
// The only unsafe boundary. Every public entry point constructs a fresh checked
// plan using the actual buffer length and owner policy before calling here.
fn emit(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    plan: Plan,
) -> Result<bool> {
    let Some(pso) =
        pipelines.research_pso(plan.kernel.name(), plan.threads()[0], plan.shared_bytes())
    else {
        return Ok(false);
    };
    let encoder = command.computeCommandEncoder().ok_or_else(|| {
        RvllmError::apple(
            AppleError::MetalUnavailable,
            AppleCtx {
                backend: "metal",
                op: "donor12b_encode",
                device: "apple-silicon",
            },
        )
    })?;
    encoder.setComputePipelineState(pso);
    // SAFETY: the plan checks all aligned, full-size buffer ranges and live
    // write/read disjointness, and PSO admission checks thread/TG resources.
    // Metal copies the stack constants during this call. The prepared owner
    // retains the arena and command resources for the complete submission.
    unsafe {
        for (i, &offset) in plan.offsets[..plan.buffer_count].iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buffer), offset, i);
        }
        if plan.has_params {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(&plan.params).cast(),
                std::mem::size_of_val(&plan.params),
                plan.buffer_count,
            );
        }
    }
    let size = |v: [usize; 3]| MTLSize {
        width: v[0],
        height: v[1],
        depth: v[2],
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(size(plan.grid), size(plan.threads()));
    encoder.endEncoding();
    pipelines.record_research_dispatch(plan.kernel);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub fn try_encode_low_bit_projection(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    projection: MetalLowBitProjectionOffsets,
    activation: usize,
    output: usize,
    stride: u32,
    column: u32,
) -> Result<bool> {
    let owner = policy(pipelines, dims, phase);
    if !owner.allowed() {
        return Ok(false);
    }
    let [n, k] = projection.shape();
    let request = ProjectionRequest {
        selected: owner.selected,
        dtype: owner.dtype,
        quantized_accumulation: owner.quantized_accumulation,
        shape: [dims.num_tokens, n, k],
        activation,
        native_weights: 0,
        low_bit: Some(weight(projection)),
        output,
        output_stride: stride,
        output_column: column,
        output_f32: false,
        arena_bytes: buffer.length(),
    };
    let Some(plan) = request.plan() else {
        return Ok(false);
    };
    let encoded = emit(pipelines, command, buffer, plan)?;
    if encoded {
        pipelines.record_low_bit_dispatch(projection.format(), projection.role());
    }
    Ok(encoded)
}

/// Primitive entry point for the existing central GEMM dispatcher. This checks
/// exact matrix shapes, actual typed owner and explicit FP32 output, but does
/// not change which native model tensors the central dispatcher supplies.
#[allow(clippy::too_many_arguments)]
pub fn try_encode_native_projection(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offsets: [usize; 3],
    shape: [u32; 3],
    alpha: f32,
    beta: f32,
    output_f32: bool,
) -> Result<bool> {
    if alpha != 1.0 || beta != 0.0 {
        return Ok(false);
    }
    let request = ProjectionRequest {
        selected: pipelines.kernel_options().research,
        dtype: pipelines.float_type(),
        quantized_accumulation: pipelines.kernel_options().quantized_bf16_accumulation,
        shape,
        activation: offsets[0],
        native_weights: offsets[1],
        low_bit: None,
        output: offsets[2],
        output_stride: shape[1],
        output_column: 0,
        output_f32,
        arena_bytes: buffer.length(),
    };
    let Some(plan) = request.plan() else {
        return Ok(false);
    };
    emit(pipelines, command, buffer, plan)
}

pub fn gate_request(
    pipelines: &PipelineCache,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    weights: &MetalLayerWeights,
    scratch: &MetalScratch,
    capture: bool,
    arena_bytes: usize,
) -> Option<GateRequest> {
    let low_bit = match (weights.low_bit_gate_proj, weights.low_bit_up_proj) {
        (None, None) => None,
        (Some(g), Some(u)) => Some([weight(g), weight(u)]),
        _ => return None,
    };
    Some(GateRequest {
        policy: policy(pipelines, dims, phase),
        capture_gate_up: capture,
        activation: scratch.normed_hidden,
        native_weights: weights.gate_up_offset,
        low_bit,
        output: scratch.activated,
        arena_bytes,
    })
}

/// Used by both execution and optional external encoder accounting. A trace
/// retains the materialized gate/up buffers and refuses this fused route.
pub fn gate_plan(pipelines: &PipelineCache, request: GateRequest) -> Option<Plan> {
    if request.policy.selected != pipelines.kernel_options().research
        || request.policy.dtype != pipelines.float_type()
        || request.policy.quantized_accumulation
            != pipelines.kernel_options().quantized_bf16_accumulation
    {
        return None;
    }
    let plan = request.plan()?;
    available(pipelines, &plan).then_some(plan)
}

#[allow(clippy::too_many_arguments)]
pub fn try_encode_gate(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    weights: &MetalLayerWeights,
    scratch: &MetalScratch,
    capture: bool,
) -> Result<bool> {
    let Some(request) = gate_request(
        pipelines,
        dims,
        phase,
        weights,
        scratch,
        capture,
        buffer.length(),
    ) else {
        return Ok(false);
    };
    let Some(plan) = gate_plan(pipelines, request) else {
        return Ok(false);
    };
    let encoded = emit(pipelines, command, buffer, plan)?;
    if encoded {
        pipelines.add_donor_layer_encoder_correction(if request.low_bit.is_some() {
            -2
        } else {
            -1
        });
        if let Some([g, u]) = request.low_bit {
            // Logical role participation, not physical dispatch count: the
            // separate research ledger records ONE physical fused dispatch.
            pipelines.record_low_bit_dispatch(g.format, g.role);
            pipelines.record_low_bit_dispatch(u.format, u.role);
        }
    }
    Ok(encoded)
}

#[allow(clippy::too_many_arguments)]
pub fn try_encode_qkv(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    projections: [MetalLowBitProjectionOffsets; 3],
    activation: usize,
    output: usize,
    capture: bool,
    skip_kv: bool,
) -> Result<bool> {
    let request = QkvRequest {
        policy: policy(pipelines, dims, phase),
        capture_projection: capture,
        skip_kv,
        activation,
        weights: projections.map(weight),
        output,
        arena_bytes: buffer.length(),
    };
    let Some(plan) = request.plan() else {
        return Ok(false);
    };
    let encoded = emit(pipelines, command, buffer, plan)?;
    if encoded {
        pipelines.add_donor_layer_encoder_correction(-2);
        for p in projections {
            pipelines.record_low_bit_dispatch(p.format(), p.role());
        }
    }
    Ok(encoded)
}

#[allow(clippy::too_many_arguments)]
pub fn try_encode_attention(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    dims: &MetalLayerDims,
    phase: MetalPhase,
    offsets: [usize; 7],
) -> Result<bool> {
    let request = AttentionRequest {
        policy: policy(pipelines, dims, phase),
        offsets,
        block_size: dims.block_size,
        max_blocks: dims.max_blocks_per_seq,
        num_blocks: dims.num_blocks_total,
        scale: dims.attn_scale,
        arena_bytes: buffer.length(),
    };
    let Some(plan) = request.plan() else {
        return Ok(false);
    };
    emit(pipelines, command, buffer, plan)
}

/// Enables native QKV projection -> existing FP32 projected norm/RoPE/cache
/// adapter. The actual projection plan still checks four-byte output scratch.
pub fn projected_qkv_allowed(
    pipelines: &PipelineCache,
    dims: &MetalLayerDims,
    phase: MetalPhase,
) -> bool {
    policy(pipelines, dims, phase).allowed()
        && pipelines.float_type() == Some(MetalFloatType::Bf16)
        && donor12b::kernel(
            pipelines.kernel_options().research,
            donor12b::Operation::NativeProjection,
        )
        .is_some_and(|k| {
            pipelines
                .research_pso(k.name(), k.limits().0, k.limits().1)
                .is_some()
        })
}

// Native operator referee invokes the same checked-plan encoder/bindings. This
// hook is absent from non-test builds; it is never a public unchecked route.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn emit_plan_for_oracle(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    plan: Plan,
) -> Result<bool> {
    emit(pipelines, command, buffer, plan)
}
