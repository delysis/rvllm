//! Allocation-free admission and launch ABI for the 12B donor schedules.
//! Plans describe encoded work only: they do not certify device correctness,
//! checkpoint quality, GPU completion or performance. Existing defaults stay off.
#![forbid(unsafe_code)]

use crate::research::{buffer_span, disjoint_writes, matrix_bytes, Gemma12bResearchShape};
use crate::research_evidence::ResearchKernel;
use crate::{MetalFloatType, MetalResearchCandidate};
use rvllm_apple::{AppleLowBitTensorRole as Role, AppleLowBitWeightFormat as Format};
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    W4,
    W8,
    BatchW4,
    BatchW8,
    GateW4,
    GateW8,
    QkvW4,
    QkvW8,
    NativeGate,
    NativeProjection,
    LocalAttention,
    GlobalAttention,
}

pub const fn simdgroups(candidate: MetalResearchCandidate) -> Option<usize> {
    match candidate {
        MetalResearchCandidate::Donor12bSg8 => Some(8),
        MetalResearchCandidate::Donor12bSg4 => Some(4),
        _ => None,
    }
}

/// The prepared runtime may install group-32/FP16-scale sidecars into a
/// BF16 activation model ONLY for this explicit native-BF16 family.
pub fn bf16_sidecars_allowed(options: crate::MetalKernelOptions) -> bool {
    simdgroups(options.research).is_some() && !options.quantized_bf16_accumulation
}

pub fn kernel(candidate: MetalResearchCandidate, operation: Operation) -> Option<ResearchKernel> {
    // Catalog order is explicit, checked exhaustively by registry tests.
    let index = match operation {
        Operation::W4 => 0,
        Operation::W8 => 1,
        Operation::BatchW4 => 2,
        Operation::BatchW8 => 3,
        Operation::GateW4 => 4,
        Operation::GateW8 => 5,
        Operation::QkvW4 => 6,
        Operation::QkvW8 => 7,
        Operation::NativeGate => 8,
        Operation::NativeProjection => 9,
        Operation::LocalAttention => 10,
        Operation::GlobalAttention => 11,
    };
    simdgroups(candidate)?;
    candidate.kernels().get(index).copied()
}

#[derive(Clone, Copy, Debug)]
pub struct Policy {
    pub selected: MetalResearchCandidate,
    pub dtype: Option<MetalFloatType>,
    pub quantized_accumulation: bool,
    pub model: Gemma12bResearchShape,
    pub decode: bool,
}
impl Policy {
    pub fn allowed(self) -> bool {
        simdgroups(self.selected).is_some()
            && self.dtype == Some(MetalFloatType::Bf16)
            && !self.quantized_accumulation
            && self.model.supports(self.selected)
            && (!self.decode || self.model.tokens == 1)
            && match (self.model.kv_heads, self.model.head_dim) {
                (8, 256) => self.model.attention_window > 0,
                (1, 512) => self.model.attention_window == 0,
                _ => false,
            }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Plan {
    pub kernel: ResearchKernel,
    pub offsets: [usize; 8],
    pub buffer_count: usize,
    /// Exact Metal constant ABI: eight little/native-endian u32 fields.
    pub params: [u32; 8],
    pub has_params: bool,
    pub grid: [usize; 3],
}
impl Plan {
    pub fn threads(self) -> [usize; 3] {
        [self.kernel.limits().0, 1, 1]
    }
    pub fn shared_bytes(self) -> usize {
        self.kernel.limits().1
    }
}

fn span(offset: usize, bytes: usize, alignment: usize, arena: usize) -> Option<Range<usize>> {
    if offset % alignment != 0 {
        return None;
    }
    buffer_span(offset, bytes, arena)
}
fn output_span(
    offset: usize,
    m: u32,
    n: u32,
    stride: u32,
    column: u32,
    element: usize,
    arena: usize,
) -> Option<Range<usize>> {
    if m == 0 || n == 0 || column.checked_add(n)? > stride {
        return None;
    }
    // Conservatively reserve row padding too; an output never aliases a read
    // merely because this particular invocation leaves some columns untouched.
    let elements = (m as usize - 1)
        .checked_mul(stride as usize)?
        .checked_add(column as usize)?
        .checked_add(n as usize)?;
    span(offset, elements.checked_mul(element)?, element, arena)
}

/// Portable mirror used ONLY for planning. The native adapter constructs it
/// from immutable, already-authenticated MetalLowBitProjectionOffsets.
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub format: Format,
    pub role: Role,
    pub n: u32,
    pub k: u32,
    pub values: usize,
    pub scales: usize,
}
impl Weight {
    fn spans(self, arena: usize) -> Option<[Range<usize>; 2]> {
        if self.n == 0 || self.k == 0 || self.k % 32 != 0 {
            return None;
        }
        let elements = (self.n as usize).checked_mul(self.k as usize)?;
        let packed = match self.format {
            Format::W4A16 => elements / 2,
            Format::W8A16 => elements,
        };
        let a = span(self.values, packed, 4, arena)?;
        let b = span(
            self.scales,
            elements.checked_div(32)?.checked_mul(2)?,
            4,
            arena,
        )?;
        if a.start < b.end && b.start < a.end {
            return None;
        }
        Some([a, b])
    }
}

pub fn projection_role_matches(role: Role, n: u32, k: u32) -> bool {
    match role {
        Role::QueryProjection => k == 3840 && matches!(n, 4096 | 8192),
        Role::KeyProjection | Role::ValueProjection => k == 3840 && matches!(n, 512 | 2048),
        Role::OutputProjection => n == 3840 && matches!(k, 4096 | 8192),
        Role::DenseGateProjection | Role::DenseUpProjection => n == 15360 && k == 3840,
        Role::DenseDownProjection => n == 3840 && k == 15360,
        Role::LmHead => n == 262144 && k == 3840,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectionRequest {
    pub selected: MetalResearchCandidate,
    pub dtype: Option<MetalFloatType>,
    pub quantized_accumulation: bool,
    /// The primitive is model-neutral arithmetic on these exact 12B matrices.
    /// Full layer fusion additionally requires Policy::allowed().
    pub shape: [u32; 3],
    pub activation: usize,
    pub native_weights: usize,
    pub low_bit: Option<Weight>,
    pub output: usize,
    pub output_stride: u32,
    pub output_column: u32,
    pub output_f32: bool,
    pub arena_bytes: usize,
}
impl ProjectionRequest {
    pub fn plan(self) -> Option<Plan> {
        let sg = simdgroups(self.selected)?;
        let [m, n, k] = self.shape;
        if self.dtype != Some(MetalFloatType::Bf16)
            || self.quantized_accumulation
            || !(1..=128).contains(&m)
            || k % 256 != 0
            || n % 4 != 0
        {
            return None;
        }
        let input = span(self.activation, matrix_bytes(m, k, 2)?, 2, self.arena_bytes)?;
        let output = output_span(
            self.output,
            m,
            n,
            self.output_stride,
            self.output_column,
            if self.output_f32 { 4 } else { 2 },
            self.arena_bytes,
        )?;
        let (operation, offsets, count) = if let Some(w) = self.low_bit {
            if [w.n, w.k] != [n, k] || !projection_role_matches(w.role, n, k) {
                return None;
            }
            let [values, scales] = w.spans(self.arena_bytes)?;
            if !disjoint_writes(&[input, values, scales], &[output]) {
                return None;
            }
            let operation = match (w.format, m == 1) {
                (Format::W4A16, true) => Operation::W4,
                (Format::W8A16, true) => Operation::W8,
                (Format::W4A16, false) => Operation::BatchW4,
                (Format::W8A16, false) => Operation::BatchW8,
            };
            (
                operation,
                [self.activation, w.values, w.scales, self.output, 0, 0, 0, 0],
                4,
            )
        } else {
            if !matches!(
                (n, k),
                (8192 | 9216 | 30720 | 15360 | 262144, 3840) | (3840, 4096 | 8192 | 15360)
            ) {
                return None;
            }
            let weights = span(
                self.native_weights,
                matrix_bytes(n, k, 2)?,
                2,
                self.arena_bytes,
            )?;
            if !disjoint_writes(&[input, weights], &[output]) {
                return None;
            }
            (
                Operation::NativeProjection,
                [
                    self.activation,
                    self.native_weights,
                    self.output,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                3,
            )
        };
        let rows_per_group = sg * if m == 1 { 4 } else { 2 };
        Some(Plan {
            kernel: kernel(self.selected, operation)?,
            offsets,
            buffer_count: count,
            params: [
                m,
                n,
                k,
                self.output_stride,
                self.output_column,
                u32::from(self.output_f32),
                0,
                0,
            ],
            has_params: true,
            grid: [
                (n as usize).div_ceil(rows_per_group),
                (m as usize).div_ceil(8),
                1,
            ],
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GateRequest {
    pub policy: Policy,
    pub capture_gate_up: bool,
    pub activation: usize,
    pub native_weights: usize,
    pub low_bit: Option<[Weight; 2]>,
    pub output: usize,
    pub arena_bytes: usize,
}
impl GateRequest {
    pub fn plan(self) -> Option<Plan> {
        if !self.policy.allowed() || !self.policy.decode || self.capture_gate_up {
            return None;
        }
        let input = span(self.activation, 3840 * 2, 2, self.arena_bytes)?;
        let output = span(self.output, 15360 * 2, 2, self.arena_bytes)?;
        let (operation, offsets, count) = if let Some([g, u]) = self.low_bit {
            if g.format != u.format
                || g.role != Role::DenseGateProjection
                || u.role != Role::DenseUpProjection
                || [g.n, g.k, u.n, u.k] != [15360, 3840, 15360, 3840]
            {
                return None;
            }
            let [gv, gs] = g.spans(self.arena_bytes)?;
            let [uv, us] = u.spans(self.arena_bytes)?;
            if !disjoint_writes(&[input, gv, gs, uv, us], &[output]) {
                return None;
            }
            (
                match g.format {
                    Format::W4A16 => Operation::GateW4,
                    Format::W8A16 => Operation::GateW8,
                },
                [
                    self.activation,
                    g.values,
                    g.scales,
                    u.values,
                    u.scales,
                    self.output,
                    0,
                    0,
                ],
                6,
            )
        } else {
            let w = span(
                self.native_weights,
                matrix_bytes(30720, 3840, 2)?,
                2,
                self.arena_bytes,
            )?;
            if !disjoint_writes(&[input, w], &[output]) {
                return None;
            }
            (
                Operation::NativeGate,
                [
                    self.activation,
                    self.native_weights,
                    self.output,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                3,
            )
        };
        Some(Plan {
            kernel: kernel(self.policy.selected, operation)?,
            offsets,
            buffer_count: count,
            params: [0; 8],
            has_params: false,
            grid: [15360 / (4 * simdgroups(self.policy.selected)?), 1, 1],
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QkvRequest {
    pub policy: Policy,
    pub capture_projection: bool,
    pub skip_kv: bool,
    pub activation: usize,
    pub weights: [Weight; 3],
    pub output: usize,
    pub arena_bytes: usize,
}
impl QkvRequest {
    pub fn plan(self) -> Option<Plan> {
        if !self.policy.allowed() || !self.policy.decode || self.capture_projection || self.skip_kv
        {
            return None;
        }
        let [q, k, v] = self.weights;
        let qn = self
            .policy
            .model
            .heads
            .checked_mul(self.policy.model.head_dim)?;
        let kn = self
            .policy
            .model
            .kv_heads
            .checked_mul(self.policy.model.head_dim)?;
        if q.format != k.format
            || q.format != v.format
            || [q.role, k.role, v.role]
                != [
                    Role::QueryProjection,
                    Role::KeyProjection,
                    Role::ValueProjection,
                ]
            || [q.n, q.k, k.n, k.k, v.n, v.k] != [qn, 3840, kn, 3840, kn, 3840]
        {
            return None;
        }
        let [qv, qs] = q.spans(self.arena_bytes)?;
        let [kv, ks] = k.spans(self.arena_bytes)?;
        let [vv, vs] = v.spans(self.arena_bytes)?;
        let input = span(self.activation, 3840 * 2, 2, self.arena_bytes)?;
        let output = span(
            self.output,
            matrix_bytes(1, qn + 2 * kn, 2)?,
            2,
            self.arena_bytes,
        )?;
        if !disjoint_writes(&[input, qv, qs, kv, ks, vv, vs], &[output]) {
            return None;
        }
        let reuse = kn == 512 && k.values == v.values && k.scales == v.scales;
        let rows = qn + if reuse { kn } else { 2 * kn };
        Some(Plan {
            kernel: kernel(
                self.policy.selected,
                match q.format {
                    Format::W4A16 => Operation::QkvW4,
                    Format::W8A16 => Operation::QkvW8,
                },
            )?,
            offsets: [
                self.activation,
                q.values,
                q.scales,
                k.values,
                k.scales,
                v.values,
                v.scales,
                self.output,
            ],
            buffer_count: 8,
            params: [1, qn, kn, 0, 0, 0, u32::from(reuse), 0],
            has_params: true,
            grid: [
                (rows as usize).div_ceil(4 * simdgroups(self.policy.selected)?),
                1,
                1,
            ],
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionRequest {
    pub policy: Policy,
    /// Q, paged K, paged V, output, block table, context length, position.
    pub offsets: [usize; 7],
    pub block_size: u32,
    pub max_blocks: u32,
    pub num_blocks: u32,
    pub scale: f32,
    pub arena_bytes: usize,
}
impl AttentionRequest {
    pub fn plan(self) -> Option<Plan> {
        if !self.policy.allowed()
            || !self.policy.decode
            || self.scale != 1.0
            || self.block_size == 0
            || self.max_blocks == 0
            || self.num_blocks == 0
            || u64::from(self.block_size) * u64::from(self.max_blocks) > i32::MAX as u64
        {
            return None;
        }
        let d = self.policy.model.head_dim;
        let kv = self.policy.model.kv_heads;
        let cache = (self.num_blocks as usize)
            .checked_mul(self.block_size as usize)?
            .checked_mul(kv as usize)?
            .checked_mul(d as usize)?
            .checked_mul(2)?;
        let [q, k, v, out, table, context, position] = self.offsets;
        let q = span(q, matrix_bytes(16, d, 2)?, 2, self.arena_bytes)?;
        let k = span(k, cache, 2, self.arena_bytes)?;
        let v = span(v, cache, 2, self.arena_bytes)?;
        let table = span(
            table,
            (self.max_blocks as usize).checked_mul(4)?,
            4,
            self.arena_bytes,
        )?;
        let context = span(context, 4, 4, self.arena_bytes)?;
        let position = span(position, 4, 4, self.arena_bytes)?;
        let out = span(out, matrix_bytes(16, d, 2)?, 2, self.arena_bytes)?;
        if !disjoint_writes(&[q, k, v, table, context, position], &[out]) {
            return None;
        }
        let mut offsets = [0; 8];
        offsets[..7].copy_from_slice(&self.offsets);
        Some(Plan {
            kernel: kernel(
                self.policy.selected,
                if d == 256 {
                    Operation::LocalAttention
                } else {
                    Operation::GlobalAttention
                },
            )?,
            offsets,
            buffer_count: 7,
            params: [
                self.block_size,
                self.max_blocks,
                self.num_blocks,
                self.policy.model.attention_window,
                0,
                0,
                0,
                0,
            ],
            has_params: true,
            grid: [16, 1, 1],
        })
    }
}

#[cfg(test)]
#[path = "donor12b_tests.rs"]
mod tests;
