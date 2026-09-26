//! Safe checked host adapter; unsafe code is confined to typed Metal bindings.
//! No pipeline creation, allocation, labels, clocks, logging or environment reads.
use crate::research_decode::{GateUpPlan, GateUpRequest};
use crate::{MetalFloatType, PipelineCache};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

pub fn gate_up_plan(pipelines: &PipelineCache, request: GateUpRequest) -> Option<GateUpPlan> {
    // Caller metadata cannot bypass the immutable owner policy or typed PSO.
    if request.selected != pipelines.kernel_options().research
        || request.dtype != pipelines.float_type()
        || request.quantized_accumulation != pipelines.kernel_options().quantized_bf16_accumulation
        || pipelines.float_type() != Some(MetalFloatType::Bf16)
    {
        return None;
    }
    let plan = request.plan()?;
    pipelines.research_pso(plan.kernel.name(), plan.threads[0], 0)?;
    Some(plan)
}

/// False means unchanged output and ledger; the caller MUST execute its normal
/// two-dispatch fallback. A command encoder failure is an error, not fallback.
pub fn try_encode_gate_up(
    pipelines: &PipelineCache,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    request: GateUpRequest,
) -> Result<bool> {
    if request.arena_bytes != buffer.length() {
        return Ok(false);
    }
    let Some(plan) = gate_up_plan(pipelines, request) else {
        return Ok(false);
    };
    let Some(pso) = pipelines.research_pso(plan.kernel.name(), plan.threads[0], 0) else {
        return Ok(false);
    };
    let encoder = command.computeCommandEncoder().ok_or_else(|| {
        RvllmError::apple(
            AppleError::MetalUnavailable,
            AppleCtx {
                backend: "metal",
                op: plan.kernel.name(),
                device: "apple-silicon",
            },
        )
    })?;
    encoder.setComputePipelineState(pso);
    // SAFETY: plan validates complete checked/aligned/disjoint arena spans;
    // Metal copies the three stack u32 constants. The owner retains the arena
    // and cache/command lifetimes just as in other checked research adapters.
    unsafe {
        for (index, offset) in plan.offsets.into_iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
        }
        for (index, value) in plan.params.iter().enumerate() {
            encoder.setBytes_length_atIndex(std::ptr::NonNull::from(value).cast(), 4, index + 3);
        }
    }
    let size = |v: [usize; 3]| MTLSize {
        width: v[0],
        height: v[1],
        depth: v[2],
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(size(plan.grid), size(plan.threads));
    encoder.endEncoding();
    pipelines.record_research_dispatch(plan.kernel);
    Ok(true)
}
