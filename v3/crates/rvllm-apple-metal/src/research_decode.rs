//! Allocation-free admission for the next decode round. This extends the
//! existing research owner, storage ABI, pipeline cache and append-only ledger.
#![forbid(unsafe_code)]

use crate::research::{buffer_span, disjoint_writes, matrix_bytes, Gemma12bResearchShape};
use crate::{MetalFloatType, MetalResearchCandidate};
use rvllm_apple::{AppleLowBitTensorRole, AppleLowBitWeightFormat};

#[derive(Clone, Copy, Debug)]
pub struct GateUpRequest {
    pub selected: MetalResearchCandidate,
    pub dtype: Option<MetalFloatType>,
    pub decode: bool,
    pub quantized_accumulation: bool,
    pub model: Gemma12bResearchShape,
    pub capture_gate_up: bool,
    pub has_low_bit_gate_or_up: bool,
    /// Activation, native Gate||Up weights, activated output, all in one arena.
    pub offsets: [usize; 3],
    pub arena_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GateUpPlan {
    pub kernel: crate::research_evidence::ResearchKernel,
    pub offsets: [usize; 3],
    pub params: [u32; 3],
    pub grid: [usize; 3],
    pub threads: [usize; 3],
}

impl GateUpRequest {
    /// No work and no scratch materialization when any boundary is refused.
    pub fn plan(self) -> Option<GateUpPlan> {
        if !self.selected.ffn_decode()
            || self.dtype != Some(MetalFloatType::Bf16)
            || !self.decode
            || self.quantized_accumulation
            || self.capture_gate_up
            || self.has_low_bit_gate_or_up
            || !self.model.supports(self.selected)
        {
            return None;
        }
        let input = buffer_span(self.offsets[0], matrix_bytes(1, 3840, 2)?, self.arena_bytes)?;
        let weights = buffer_span(
            self.offsets[1],
            matrix_bytes(30720, 3840, 2)?,
            self.arena_bytes,
        )?;
        let output = buffer_span(
            self.offsets[2],
            matrix_bytes(1, 15360, 2)?,
            self.arena_bytes,
        )?;
        if !disjoint_writes(&[input, weights], &[output]) {
            return None;
        }
        let launch = operator_launch(self.selected)?;
        Some(GateUpPlan {
            kernel: launch.kernel,
            offsets: self.offsets,
            params: [1, 3840, 15360],
            grid: [15360 / launch.rows_per_group, 1, 1],
            threads: [launch.threads, 1, 1],
        })
    }
}

/// Role and precision are independent admission criteria, not inferred from N/K.
/// The descriptor's constructor/loader remain responsible for authentication,
/// signed payload validity, exact byte counts, and immutable scale ownership.
pub fn qmv_decode_contract(
    selected: MetalResearchCandidate,
    format: AppleLowBitWeightFormat,
    role: AppleLowBitTensorRole,
    m: usize,
    n: u32,
    k: u32,
) -> bool {
    if m != 1 || n != 3840 {
        return false;
    }
    qmv_role_target(selected, role)
        && selected.operator_keys().contains(&(k as usize))
        && ((selected.qmv_w4() && format == AppleLowBitWeightFormat::W4A16)
            || (selected.qmv_w8() && format == AppleLowBitWeightFormat::W8A16))
}

/// A single fused encode replaces exactly two encoders (Gate||Up, GELU).
/// Refusal saves nothing. This is for the existing external encoder ledger;
/// completion/quality never follows from this arithmetic or from encode counts.
pub const fn gate_up_encoder_count(fused: bool) -> u64 {
    if fused {
        1
    } else {
        2
    }
}

#[cfg(test)]
#[path = "research_decode_reference.rs"]
pub(crate) mod reference;
#[cfg(test)]
#[path = "research_decode_tests.rs"]
mod tests;

/// Role targeting is distinct from exact-shape admission: a refused targeted
/// BF16 decode still uses the existing BF16 n4 fallback, never the F16 ABI.
pub const fn qmv_role_target(
    selected: MetalResearchCandidate,
    role: AppleLowBitTensorRole,
) -> bool {
    (selected.qmv_w4() && matches!(role, AppleLowBitTensorRole::DenseDownProjection))
        || (selected.qmv_w8() && matches!(role, AppleLowBitTensorRole::OutputProjection))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperatorLaunch {
    pub kernel: crate::research_evidence::ResearchKernel,
    pub rows_per_group: usize,
    pub threads: usize,
}

pub const fn operator_launch(selected: MetalResearchCandidate) -> Option<OperatorLaunch> {
    use crate::research_evidence::ResearchKernel as K;
    use MetalResearchCandidate as C;
    let (kernel, rows_per_group, threads) = match selected {
        C::FfnBf16R4Sg2 => (K::FfnBf16R4Sg2, 8, 64),
        C::FfnBf16R2Sg2 => (K::FfnBf16R2Sg2, 4, 64),
        C::FfnBf16R4Sg4 => (K::FfnBf16R4Sg4, 16, 128),
        C::QmvW4G32R8Sg2 => (K::QmvW4G32R8Sg2, 16, 64),
        C::QmvW8G32R8Sg2 => (K::QmvW8G32R8Sg2, 16, 64),
        C::QmvW4G32R4Sg8K8 => (K::QmvW4G32R4Sg8K8, 32, 256),
        C::QmvW8G32R4Sg8K8 => (K::QmvW8G32R4Sg8K8, 32, 256),
        C::QmvW4G32R4Sg4 => (K::QmvW4G32R4Sg4, 16, 128),
        C::QmvW8G32R4Sg4K8192 => (K::QmvW8G32R4Sg4K8192, 16, 128),
        C::QmvW8G32R2Sg4K4096 => (K::QmvW8G32R2Sg4K4096, 8, 128),
        _ => return None,
    };
    Some(OperatorLaunch {
        kernel,
        rows_per_group,
        threads,
    })
}
