//! Default-off, native BF16 prefill tournament. Pure planning, no device access.
//! The historical `allow_prefill_mma` layer flag also carries this explicitly
//! selected family; it is never inferred from matrix dimensions alone.
#![forbid(unsafe_code)]

use crate::research::{Gemma12bResearchShape, GqaBufferShape, MetalResearchCandidate};
use crate::research_evidence::ResearchKernel;
use crate::research_projection::{FallbackReason, ProjectionPlan, ProjectionRequest};
use std::ops::Range;

pub const MIN_M: u32 = 6;
pub const MAX_M: u32 = 2048;
pub const MAX_CONTEXT_CAPACITY: u32 = 4096;
pub const RUNGS: [u32; 7] = [6, 17, 64, 256, 512, 1024, 2048];
pub const FAMILY: [MetalResearchCandidate; 4] = [
    MetalResearchCandidate::PrefillLoad4Control,
    MetalResearchCandidate::PrefillPipeline32x64,
    MetalResearchCandidate::PrefillQ4K16,
    MetalResearchCandidate::PrefillPipeline32x64Q4K16,
];

pub const fn projection_kernels(c: MetalResearchCandidate) -> Option<[ResearchKernel; 2]> {
    use MetalResearchCandidate as C;
    use ResearchKernel as K;
    match c {
        C::PrefillLoad4Control => Some([K::PrefillControlGemm, K::PrefillControlQkv]),
        C::PrefillPipeline32x64 => Some([K::PrefillPipelineGemm, K::PrefillPipelineQkv]),
        C::PrefillPipeline32x64Q4K16 => Some([K::PrefillCombinedGemm, K::PrefillCombinedQkv]),
        _ => None,
    }
}

/// Distinct raw-output and normalization identities for the fused-normal-route
/// precision boundary. No BF16 projection is introduced before this RMSNorm.
pub const fn postnorm_kernels(c: MetalResearchCandidate) -> Option<[ResearchKernel; 2]> {
    use MetalResearchCandidate as C;
    use ResearchKernel as K;
    match c {
        C::PrefillLoad4Control => Some([K::PrefillControlRaw, K::PrefillControlNorm]),
        C::PrefillPipeline32x64 => Some([K::PrefillPipelineRaw, K::PrefillPipelineNorm]),
        C::PrefillPipeline32x64Q4K16 => Some([K::PrefillCombinedRaw, K::PrefillCombinedNorm]),
        _ => None,
    }
}
pub fn raw_norm_projection_shape(m: u32, n: u32, k: u32) -> bool {
    (MIN_M..=MAX_M).contains(&m) && n == 3840 && matches!(k, 4096 | 8192 | 15360)
}
/// The incumbent normal route already materializes BF16 at M<=32 for these
/// exact O/down shapes. Preserve that boundary in short full-route screens;
/// raw FP32 materialization replaces only the large-prefill fused operator.
/// Direct component oracles may still exercise raw kernels at M=6 and M=17.
pub fn raw_norm_route(m: u32, n: u32, k: u32) -> bool {
    m > 32 && raw_norm_projection_shape(m, n, k)
}

/// X=FP32 projection, O=BF16 normalized result, G=BF16 gamma. All spans
/// disjoint: X must not overwrite G during its immediately preceding producer.
pub fn postnorm_plan(
    c: MetalResearchCandidate,
    permission: bool,
    native: bool,
    offsets: [usize; 3],
    m: u32,
    n: u32,
    eps: f32,
    bytes: usize,
) -> Result<crate::research_projection::PostnormPlan, FallbackReason> {
    let [_, kernel] = postnorm_kernels(c).ok_or(FallbackReason::NotThisOperation)?;
    if !permission {
        return Err(FallbackReason::ModelPhaseOrTrace);
    }
    if !native {
        return Err(FallbackReason::StoragePrecision);
    }
    if !(MIN_M..=MAX_M).contains(&m) || n != 3840 || eps != 1.0e-6 || offsets[0] % 4 != 0 {
        return Err(FallbackReason::Shape);
    }
    let check = || {
        use crate::research::{buffer_span, disjoint_writes, matrix_bytes};
        let x = buffer_span(offsets[0], matrix_bytes(m, n, 4)?, bytes)?;
        let o = buffer_span(offsets[1], matrix_bytes(m, n, 2)?, bytes)?;
        let g = buffer_span(offsets[2], matrix_bytes(1, n, 2)?, bytes)?;
        Some(disjoint_writes(&[g], &[x, o]))
    };
    if !check().unwrap_or(false) {
        return Err(FallbackReason::BufferOrAlias);
    }
    Ok(crate::research_projection::PostnormPlan {
        kernel,
        binds_token_count: true,
    })
}

pub const fn attention_kernel(c: MetalResearchCandidate, dim: u32) -> Option<ResearchKernel> {
    use MetalResearchCandidate as C;
    use ResearchKernel as K;
    match (c, dim) {
        (C::PrefillQ4K16, 256) => Some(K::PrefillQ4K16D256),
        (C::PrefillQ4K16, 512) => Some(K::PrefillQ4K16D512),
        (C::PrefillPipeline32x64Q4K16, 256) => Some(K::PrefillCombinedD256),
        (C::PrefillPipeline32x64Q4K16, 512) => Some(K::PrefillCombinedD512),
        _ => None,
    }
}

/// Model/phase identity is checked separately. This is the exact operation ABI.
pub fn projection_shape(m: u32, n: u32, k: u32, fp32: bool) -> bool {
    (MIN_M..=MAX_M).contains(&m)
        && if fp32 {
            k == 3840 && matches!(n, 8192 | 9216)
        } else {
            matches!((n, k), (30720, 3840) | (3840, 4096 | 8192 | 15360))
        }
}

pub fn projection_plan(r: ProjectionRequest) -> Result<ProjectionPlan, FallbackReason> {
    let [gemm, qkv] = projection_kernels(r.candidate).ok_or(FallbackReason::NotThisOperation)?;
    if !r.full_prefill {
        return Err(FallbackReason::ModelPhaseOrTrace);
    }
    if !r.native_bf16 {
        return Err(FallbackReason::StoragePrecision);
    }
    if r.alpha != 1.0 || r.beta != 0.0 {
        return Err(FallbackReason::ProjectionScale);
    }
    let [m, n, k] = r.shape;
    let raw = r.output_f32 && raw_norm_projection_shape(m, n, k);
    if (!raw && !projection_shape(m, n, k, r.output_f32)) || n % 64 != 0 || k % 32 != 0 {
        return Err(FallbackReason::Shape);
    }
    if r.offsets[0] % 8 != 0 || r.offsets[1] % 8 != 0 {
        return Err(FallbackReason::Alignment);
    }
    if !crate::research::projection_buffers_fit(
        r.offsets,
        r.shape,
        if r.output_f32 { 4 } else { 2 },
        r.arena_bytes,
    ) {
        return Err(FallbackReason::BufferOrAlias);
    }
    Ok(ProjectionPlan {
        kernel: if raw {
            postnorm_kernels(r.candidate).ok_or(FallbackReason::NotThisOperation)?[0]
        } else if r.output_f32 {
            qkv
        } else {
            gemm
        },
        tile_m: 32,
        tile_n: 64,
    })
}

/// No environment reads or allocations. Hardware knowledge enters explicitly
/// from the existing owner; this predicate is not hardware qualification.
pub fn projection_layer_supported(
    candidate: MetalResearchCandidate,
    model: Gemma12bResearchShape,
    native_bf16: bool,
    apple9_or_10: bool,
) -> bool {
    projection_kernels(candidate).is_some()
        && native_bf16
        && apple9_or_10
        && model.supports(candidate)
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionRequest {
    pub candidate: MetalResearchCandidate,
    pub model: Gemma12bResearchShape,
    pub native_bf16: bool,
    pub apple9_or_10: bool,
    pub prefill_single_sequence: bool,
    pub trace: bool,
    pub block_size: u32,
    pub max_blocks: u32,
    pub num_blocks: u32,
    pub scale: f32,
    /// Existing attention ABI: Q, K, V, O, table, contexts, cu, positions.
    pub offsets: [usize; 8],
    pub arena_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionPlan {
    pub kernel: ResearchKernel,
    pub grid: [usize; 3],
}

impl AttentionRequest {
    pub fn plan(self) -> Result<AttentionPlan, FallbackReason> {
        let kernel = attention_kernel(self.candidate, self.model.head_dim)
            .ok_or(FallbackReason::NotThisOperation)?;
        if !self.prefill_single_sequence || self.trace || !self.model.supports(self.candidate) {
            return Err(FallbackReason::ModelPhaseOrTrace);
        }
        if !self.native_bf16 || !self.apple9_or_10 {
            return Err(FallbackReason::StoragePrecision);
        }
        if self.scale != 1.0
            || !matches!(
                (
                    self.model.kv_heads,
                    self.model.head_dim,
                    self.model.attention_window
                ),
                (8, 256, 1024) | (1, 512, 0)
            )
            || self.block_size == 0
            || self.max_blocks == 0
            || self.num_blocks == 0
        {
            return Err(FallbackReason::Shape);
        }
        let capacity = self
            .block_size
            .checked_mul(self.max_blocks)
            .ok_or(FallbackReason::Shape)?;
        if capacity < self.model.tokens || capacity > MAX_CONTEXT_CAPACITY {
            return Err(FallbackReason::Shape);
        }
        let shape = GqaBufferShape {
            tokens: self.model.tokens,
            kv_heads: self.model.kv_heads,
            head_dim: self.model.head_dim,
            block_size: self.block_size,
            max_blocks: self.max_blocks,
            num_blocks: self.num_blocks,
        };
        if !shape.buffers_fit(self.offsets, self.arena_bytes) {
            return Err(FallbackReason::BufferOrAlias);
        }
        let cache_bytes = (self.num_blocks as usize)
            .checked_mul(self.block_size as usize)
            .and_then(|n| n.checked_mul(self.model.kv_heads as usize))
            .and_then(|n| n.checked_mul(self.model.head_dim as usize))
            .and_then(|n| n.checked_mul(2))
            .ok_or(FallbackReason::BufferOrAlias)?;
        // Read/read aliases in general are legal, but this kernel's K and V
        // contract is explicitly two separate transformed cache allocations.
        let k = self.offsets[1]
            ..self.offsets[1]
                .checked_add(cache_bytes)
                .ok_or(FallbackReason::BufferOrAlias)?;
        let v = self.offsets[2]
            ..self.offsets[2]
                .checked_add(cache_bytes)
                .ok_or(FallbackReason::BufferOrAlias)?;
        if k.start < v.end && v.start < k.end {
            return Err(FallbackReason::BufferOrAlias);
        }
        Ok(AttentionPlan {
            kernel,
            grid: [(self.model.tokens as usize).div_ceil(4), 16, 1],
        })
    }
}

/// Independent host definition of visibility: positions are absolute, including
/// the newest token. A speculative suffix in `context` is not automatically visible.
pub fn visible_range(
    position: i32,
    context: i32,
    capacity: u32,
    window: u32,
) -> Result<Range<u32>, &'static str> {
    if context <= 0 || context as u32 > capacity || position < 0 || position >= context {
        return Err("invalid absolute position/context");
    }
    let end = position as u32 + 1;
    Ok((if window == 0 {
        0
    } else {
        end.saturating_sub(window)
    })..end)
}

/// Negative pages are holes. A nonnegative out-of-range page is an error, never
/// an address or an empty token. No pointer arithmetic precedes these checks.
pub fn physical_token(
    table: &[i32],
    block: u32,
    physical_blocks: u32,
    token: u32,
) -> Result<Option<usize>, &'static str> {
    if block == 0 || physical_blocks == 0 {
        return Err("zero page geometry");
    }
    let page = *table
        .get((token / block) as usize)
        .ok_or("logical page outside table")?;
    if page < 0 {
        return Ok(None);
    }
    if page as u32 >= physical_blocks {
        return Err("physical page outside cache");
    }
    let at = (page as usize)
        .checked_mul(block as usize)
        .and_then(|n| n.checked_add((token % block) as usize))
        .ok_or("page offset overflow")?;
    Ok(Some(at))
}

/// Arithmetic operation counts, not timing attribution or a bandwidth estimate.
pub fn projection_flops(m: u32, n: u32, k: u32) -> Option<u64> {
    u64::from(m)
        .checked_mul(u64::from(n))?
        .checked_mul(u64::from(k))?
        .checked_mul(2)
}

/// Expected NEW slots at M=256/512/1024/2048 for one complete, untraced,
/// single-chunk 48-layer prefill. Short-prefill O/down keeps its BF16 boundary.
/// Decode never contributes to these counters. Any fallback invalidates this
/// full-route coverage claim, even when a process otherwise returns success.
pub fn expected_prefill_dispatches(c: MetalResearchCandidate) -> Option<Vec<(&'static str, u64)>> {
    if !FAMILY.contains(&c) {
        return None;
    }
    let mut out = Vec::new();
    if let Some([gemm, qkv]) = projection_kernels(c) {
        out.push((gemm.name(), 48)); // gate/up only; O/down preserve FP32 before norm.
        out.push((qkv.name(), 48));
    }
    if let Some([raw, norm]) = postnorm_kernels(c) {
        out.push((raw.name(), 96));
        out.push((norm.name(), 96));
    }
    if let Some(k) = attention_kernel(c, 256) {
        out.push((k.name(), 40));
    }
    if let Some(k) = attention_kernel(c, 512) {
        out.push((k.name(), 8));
    }
    Some(out)
}

#[cfg(test)]
#[path = "prefill_round_tests.rs"]
mod tests;
