//! Checked, allocation-free planning for the opt-in global D512/g16 decode family.
//! This is a source contract, not device qualification. Cache ownership and
//! producer synchronization remain with the existing paged-cache owner.
#![forbid(unsafe_code)]

use std::ops::Range;

pub const HEADS: u32 = 16;
pub const DIM: u32 = 512;
pub const KV_TILE: u32 = 8;
pub const LIVE_LENGTHS: [u32; 5] = [256, 512, 1024, 2048, 4096];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeTile {
    pub rows: u32,
    pub panel: u32,
    pub threads: u32,
}

pub const DECODE_TILES: [DecodeTile; 9] = [
    DecodeTile {
        rows: 8,
        panel: 64,
        threads: 64,
    },
    DecodeTile {
        rows: 8,
        panel: 64,
        threads: 128,
    },
    DecodeTile {
        rows: 8,
        panel: 128,
        threads: 64,
    },
    DecodeTile {
        rows: 8,
        panel: 128,
        threads: 128,
    },
    DecodeTile {
        rows: 16,
        panel: 64,
        threads: 64,
    },
    DecodeTile {
        rows: 16,
        panel: 64,
        threads: 128,
    },
    DecodeTile {
        rows: 16,
        panel: 128,
        threads: 64,
    },
    DecodeTile {
        rows: 16,
        panel: 128,
        threads: 128,
    },
    DecodeTile {
        rows: 1,
        panel: 128,
        threads: 32,
    },
];

impl DecodeTile {
    pub const fn supported(self) -> bool {
        (self.rows == 1 && self.panel == 128 && self.threads == 32)
            || (matches!(self.rows, 8 | 16)
                && matches!(self.panel, 64 | 128)
                && matches!(self.threads, 64 | 128))
    }

    /// Q is staged once; one K or V panel reuses the same storage. Scores,
    /// corrections and weights are FP32. Eight signed page IDs are separate.
    pub const fn threadgroup_bytes(self) -> usize {
        (self.rows * DIM * 2 + KV_TILE * self.panel * 2 + 3 * self.rows * KV_TILE * 4 + KV_TILE * 4)
            as usize
    }

    /// Source-level FP32 output state per lane. Not a compiler register count.
    pub const fn output_floats_per_thread(self) -> u32 {
        self.rows * DIM / self.threads
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum DecodeOutput {
    Bf16 = 0,
    /// Oracle only. The normal layer route always selects Bf16.
    F32 = 1,
}

impl DecodeOutput {
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

/// The attention operator geometry; full model/phase/dtype checks precede this
/// planner in the Metal adapter. Batch=1 is deliberately the initial scope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeShape {
    pub sequences: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks: u32,
    pub num_blocks: u32,
    pub window: u32,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeBuffers {
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub output: usize,
    pub block_tables: usize,
    pub context_lens: usize,
    pub positions: usize,
}

impl DecodeBuffers {
    pub const fn offsets(self) -> [usize; 7] {
        [
            self.q,
            self.k,
            self.v,
            self.output,
            self.block_tables,
            self.context_lens,
            self.positions,
        ]
    }
}

/// All fields are four-byte scalars, in exactly the MSL struct order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct DecodeParams {
    pub sequences: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks: u32,
    pub num_blocks: u32,
    pub window: u32,
    pub scale: f32,
    pub output_kind: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodePlan {
    pub tile: DecodeTile,
    pub shape: DecodeShape,
    pub output: DecodeOutput,
    pub grid: [usize; 3],
    pub threads: [usize; 3],
    pub threadgroup_bytes: usize,
    /// No global partial-state storage, reduction launch or split allocation.
    pub scratch_bytes: usize,
    pub cache_bytes: usize,
}

fn span(offset: usize, bytes: usize, align: usize, capacity: usize) -> Option<Range<usize>> {
    let end = offset.checked_add(bytes)?;
    (bytes != 0 && offset % align == 0 && end <= capacity).then_some(offset..end)
}

fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

impl DecodePlan {
    pub fn new(tile: DecodeTile, shape: DecodeShape, output: DecodeOutput) -> Option<Self> {
        if !tile.supported()
            || shape.sequences != 1
            || shape.heads != HEADS
            || shape.kv_heads != 1
            || shape.head_dim != DIM
            || shape.window != 0
            || shape.scale.to_bits() != 1.0_f32.to_bits()
            || shape.block_size == 0
            || shape.max_blocks == 0
            || shape.num_blocks == 0
            || shape.max_blocks.checked_mul(shape.block_size)? > i32::MAX as u32
        {
            return None;
        }
        let cache_bytes = (shape.num_blocks as usize)
            .checked_mul(shape.block_size as usize)?
            .checked_mul(DIM as usize)?
            .checked_mul(2)?;
        // Check the complete table size even on a 32-bit host.
        (shape.max_blocks as usize).checked_mul(4)?;
        Some(Self {
            tile,
            shape,
            output,
            grid: [(HEADS / tile.rows) as usize, 1, 1],
            threads: [tile.threads as usize, 1, 1],
            threadgroup_bytes: tile.threadgroup_bytes(),
            scratch_bytes: 0,
            cache_bytes,
        })
    }

    pub const fn params(self) -> DecodeParams {
        DecodeParams {
            sequences: self.shape.sequences,
            heads: self.shape.heads,
            kv_heads: self.shape.kv_heads,
            head_dim: self.shape.head_dim,
            block_size: self.shape.block_size,
            max_blocks: self.shape.max_blocks,
            num_blocks: self.shape.num_blocks,
            window: self.shape.window,
            scale: self.shape.scale,
            output_kind: self.output as u32,
        }
    }

    pub fn buffers_fit(self, buffers: DecodeBuffers, capacity: usize) -> bool {
        let elements = (HEADS * DIM) as usize;
        let Some(q) = span(buffers.q, elements * 2, 2, capacity) else {
            return false;
        };
        let Some(k) = span(buffers.k, self.cache_bytes, 2, capacity) else {
            return false;
        };
        let Some(v) = span(buffers.v, self.cache_bytes, 2, capacity) else {
            return false;
        };
        let Some(table_bytes) = (self.shape.max_blocks as usize).checked_mul(4) else {
            return false;
        };
        let Some(table) = span(buffers.block_tables, table_bytes, 4, capacity) else {
            return false;
        };
        let Some(context) = span(buffers.context_lens, 4, 4, capacity) else {
            return false;
        };
        let Some(positions) = span(buffers.positions, 4, 4, capacity) else {
            return false;
        };
        let Some(output) = span(
            buffers.output,
            elements * self.output.element_bytes(),
            self.output.element_bytes(),
            capacity,
        ) else {
            return false;
        };
        // Separate transformed K/V is the only admitted cache ABI.
        !overlaps(&k, &v)
            && [q, k, v, table, context, positions]
                .iter()
                .all(|read| !overlaps(read, &output))
    }
}

/// The packed row is a head coordinate, never a causal position. Verification
/// chunks and prefill are not admitted by this first operator.
pub fn packed_head(tile: DecodeTile, group: u32, simdgroup: u32, row: u32) -> Option<u32> {
    if !tile.supported() {
        return None;
    }
    let simdgroups = tile.threads / 32;
    if group >= HEADS / tile.rows || simdgroup >= simdgroups || row >= tile.rows / simdgroups {
        return None;
    }
    Some(group * tile.rows + simdgroup + row * simdgroups)
}

/// Validate metadata before *any* output write. Negative physical page IDs are
/// holes, not addresses. Positions limit visibility even with a speculative
/// suffix in the table. Neither this function nor the shader mutates metadata.
pub fn visible_end(shape: DecodeShape, table: &[i32], context: i32, position: i32) -> Option<u32> {
    let capacity = shape.max_blocks.checked_mul(shape.block_size)?;
    if shape.block_size == 0
        || context <= 0
        || context as u32 > capacity
        || position < 0
        || position >= context
        || table.len() < shape.max_blocks as usize
    {
        return None;
    }
    let end = position as u32 + 1;
    let pages = (end - 1) / shape.block_size + 1;
    if table[..pages as usize]
        .iter()
        .any(|&page| page >= 0 && page as u32 >= shape.num_blocks)
    {
        return None;
    }
    Some(end)
}

/// Returns None for a negative page (an intentionally absent token).
pub fn physical_base(shape: DecodeShape, table: &[i32], token: u32) -> Option<usize> {
    if shape.block_size == 0 || token / shape.block_size >= shape.max_blocks {
        return None;
    }
    let page = *table.get((token / shape.block_size) as usize)?;
    if page < 0 || page as u32 >= shape.num_blocks {
        return None;
    }
    (page as usize)
        .checked_mul(shape.block_size as usize)?
        .checked_add((token % shape.block_size) as usize)?
        .checked_mul(DIM as usize)
}

/// Explicit BF16 nearest-even conversion. No intermediate BF16 probability or
/// accumulator exists. NaNs remain NaNs; finite inputs are the oracle domain.
pub fn round_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x40;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn widen_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

#[cfg(test)]
#[path = "attention_global_decode_reference.rs"]
pub mod reference;

#[cfg(test)]
#[path = "attention_global_decode_tests.rs"]
mod tests;
