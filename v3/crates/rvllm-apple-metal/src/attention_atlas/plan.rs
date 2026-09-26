//! Pure checked planning. No environment reads, FFI, clocks, or auto-selection.
#![forbid(unsafe_code)]
use super::{Error, Result};
use serde::{Deserialize, Serialize};
use std::ops::Range;

pub const ABI: u32 = 0x4154_0001;
pub const MAX_QUERIES: u32 = 4096;
pub const MAX_KEYS: u32 = 262_144;
pub const HEADS: u32 = 16;
pub const GUARD: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    Vector,
    Cooperative,
    Matrix,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Softmax {
    PerKey,
    PerTile,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheFormat {
    SeparateBf16,
    /// A NEW experimental producer ABI, NOT the fused FP32 projection ABI.
    /// z=BF16, factor=FP32, gamma=BF16, compact cos/sin=FP32.
    BaseBf16FactorF32RopeV1,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Output {
    Bf16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub strategy: Strategy,
    pub softmax: Softmax,
    /// Packed rows: original query position times GQA, plus head-in-group.
    pub rows: u32,
    pub keys: u32,
    pub panel: u32,
    pub threads: u32,
    pub splits: u32,
}
impl Candidate {
    pub fn validate(self) -> Result<()> {
        if !matches!(self.splits, 1 | 2 | 4 | 8 | 16 | 32) || !matches!(self.panel, 64 | 128) {
            return Err(Error::new("invalid split count or D panel"));
        }
        let valid = match self.strategy {
            Strategy::Vector => {
                self.rows == 1
                    && self.keys == 1
                    && self.threads == 32
                    && self.softmax == Softmax::PerKey
            }
            Strategy::Cooperative => {
                matches!(self.rows, 2 | 4 | 8 | 16 | 32)
                    && matches!(self.keys, 8 | 16 | 32 | 64)
                    && matches!(self.threads, 64 | 128)
            }
            Strategy::Matrix => {
                matches!(self.rows, 8 | 16 | 32)
                    && matches!(self.keys, 16 | 32 | 64)
                    && matches!(self.threads, 64 | 128)
                    && self.softmax == Softmax::PerTile
                    && self.splits == 1
            }
        };
        if !valid {
            return Err(Error::new("unsupported candidate geometry"));
        }
        Ok(())
    }
    pub fn name(self) -> String {
        let strategy = match self.strategy {
            Strategy::Vector => "vector",
            Strategy::Cooperative => "coop",
            Strategy::Matrix => "mma",
        };
        let softmax = match self.softmax {
            Softmax::PerKey => "key",
            Softmax::PerTile => "tile",
        };
        format!(
            "atlas-{strategy}-{softmax}-r{}-k{}-p{}-t{}-s{}",
            self.rows, self.keys, self.panel, self.threads, self.splits
        )
    }
    pub fn source_threadgroup_bytes(self) -> usize {
        let r = self.rows as usize;
        let k = self.keys as usize;
        let p = self.panel as usize;
        match self.strategy {
            Strategy::Vector | Strategy::Cooperative => {
                2 * (r * p + k * p) + 12 * r * k + 8 * r + 4 * k
            }
            Strategy::Matrix => 4 * (r * p + k * p) + 8 * r * k + 12 * r + 4 * k,
        }
    }
    pub fn output_floats_per_thread(self, dim: u32) -> u32 {
        if self.validate().is_err() {
            return u32::MAX;
        }
        self.rows.div_ceil(self.threads / 32) * (dim / 32)
    }
}

/// A bounded source search space, not a list of qualified configurations.
/// All entries fit a conservative 32-KiB SOURCE budget. Queried PSO/device
/// limits remain mandatory; even admitted entries can spill or lose badly.
pub fn catalog() -> Vec<Candidate> {
    let mut all = Vec::new();
    for splits in [1, 2, 4, 8, 16, 32] {
        all.push(Candidate {
            strategy: Strategy::Vector,
            softmax: Softmax::PerKey,
            rows: 1,
            keys: 1,
            panel: 64,
            threads: 32,
            splits,
        });
    }
    for rows in [2, 4, 8, 16] {
        for panel in [64, 128] {
            for threads in [64, 128] {
                for splits in [1, 2, 4, 8, 16, 32] {
                    all.push(Candidate {
                        strategy: Strategy::Cooperative,
                        softmax: Softmax::PerKey,
                        rows,
                        keys: 8,
                        panel,
                        threads,
                        splits,
                    });
                }
            }
        }
    }
    for strategy in [Strategy::Cooperative, Strategy::Matrix] {
        for rows in [8, 16, 32] {
            for keys in [16, 32, 64] {
                for panel in [64, 128] {
                    for threads in [64, 128] {
                        let c = Candidate {
                            strategy,
                            softmax: Softmax::PerTile,
                            rows,
                            keys,
                            panel,
                            threads,
                            splits: 1,
                        };
                        if c.source_threadgroup_bytes() <= 32_768 {
                            all.push(c);
                        }
                    }
                }
            }
        }
    }
    all
}

pub fn parse_candidate(name: &str) -> Result<Candidate> {
    catalog()
        .into_iter()
        .find(|c| c.name() == name)
        .ok_or_else(|| Error::new("unknown exact candidate name; no automatic fallback"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub queries: u32,
    pub live_keys: u32,
    pub kv_heads: u32,
    pub dim: u32,
    pub window: u32,
    pub page_size: u32,
    pub max_blocks: u32,
    pub physical_blocks: u32,
}
impl Shape {
    pub fn validate(self) -> Result<()> {
        if !matches!(
            (self.kv_heads, self.dim, self.window),
            (8, 256, 1024) | (1, 512, 0)
        ) || !(1..=MAX_QUERIES).contains(&self.queries)
            || !(self.queries..=MAX_KEYS).contains(&self.live_keys)
            || self.page_size == 0
            || self.page_size > MAX_KEYS
            || self.max_blocks == 0
            || self.physical_blocks == 0
        {
            return Err(Error::new(
                "requires batch-one Gemma-12B text attention geometry",
            ));
        }
        let capacity = self
            .max_blocks
            .checked_mul(self.page_size)
            .ok_or_else(|| Error::new("logical page capacity overflow"))?;
        if capacity > i32::MAX as u32 || self.live_keys > capacity {
            return Err(Error::new(
                "live context exceeds signed page-table capacity",
            ));
        }
        self.cache_elements()?;
        Ok(())
    }
    pub fn gqa(self) -> u32 {
        HEADS / self.kv_heads
    }
    pub fn output_elements(self) -> usize {
        self.queries as usize * HEADS as usize * self.dim as usize
    }
    pub fn cache_elements(self) -> Result<usize> {
        product(&[
            self.physical_blocks as usize,
            self.page_size as usize,
            self.kv_heads as usize,
            self.dim as usize,
        ])
    }
    pub fn key_start(self, position: u32) -> u32 {
        if self.window == 0 {
            0
        } else {
            (position + 1).saturating_sub(self.window)
        }
    }
}
fn product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| Error::new("byte-size multiplication overflow"))
}

/// Stable 64-byte Metal ABI. No usize, bool, enum or implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub abi: u32,
    pub queries: u32,
    pub live_keys: u32,
    pub kv_heads: u32,
    pub dim: u32,
    pub window: u32,
    pub page_size: u32,
    pub max_blocks: u32,
    pub physical_blocks: u32,
    pub splits: u32,
    pub output_kind: u32,
    pub cache_kind: u32,
    pub rows: u32,
    pub keys: u32,
    pub panel: u32,
    pub threads: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Plan {
    pub candidate: Candidate,
    pub shape: Shape,
    pub cache: CacheFormat,
    pub output: Output,
    pub grid: [usize; 3],
    pub partial_bytes: usize,
    pub threadgroup_bytes: usize,
    pub encoded_dispatches: u32,
}
impl Plan {
    pub fn new(
        candidate: Candidate,
        shape: Shape,
        cache: CacheFormat,
        output: Output,
    ) -> Result<Self> {
        candidate.validate()?;
        shape.validate()?;
        if candidate.strategy == Strategy::Cooperative
            && candidate.rows < 8
            && (shape.dim != 256 || shape.queries > 8)
        {
            return Err(Error::new(
                "two/four-row cooperative tiles are local decode/verification only",
            ));
        }
        if candidate.splits > 1 && shape.queries > 8 {
            return Err(Error::new(
                "split-KV is bounded to decode/verification (Q <= 8)",
            ));
        }
        if cache != CacheFormat::SeparateBf16 && (shape.dim != 512 || shape.kv_heads != 1) {
            return Err(Error::new("reconstructible ABI is global-only"));
        }
        let packed = shape
            .queries
            .checked_mul(shape.gqa())
            .ok_or_else(|| Error::new("packed row overflow"))?;
        let partial_bytes = if candidate.splits == 1 {
            0
        } else {
            product(&[
                shape.queries as usize,
                HEADS as usize,
                candidate.splits as usize,
                shape.dim as usize + 2,
                4,
            ])?
        };
        Ok(Self {
            candidate,
            shape,
            cache,
            output,
            grid: [
                packed.div_ceil(candidate.rows) as usize,
                shape.kv_heads as usize,
                candidate.splits as usize,
            ],
            partial_bytes,
            threadgroup_bytes: candidate.source_threadgroup_bytes(),
            encoded_dispatches: if candidate.splits == 1 { 2 } else { 3 },
        })
    }
    pub fn params(self) -> Params {
        let s = self.shape;
        let c = self.candidate;
        Params {
            abi: ABI,
            queries: s.queries,
            live_keys: s.live_keys,
            kv_heads: s.kv_heads,
            dim: s.dim,
            window: s.window,
            page_size: s.page_size,
            max_blocks: s.max_blocks,
            physical_blocks: s.physical_blocks,
            splits: c.splits,
            output_kind: if self.output == Output::Bf16 { 0 } else { 1 },
            cache_kind: if self.cache == CacheFormat::SeparateBf16 {
                0
            } else {
                1
            },
            rows: c.rows,
            keys: c.keys,
            panel: c.panel,
            threads: c.threads,
        }
    }
    pub fn pso_fits(
        self,
        width: usize,
        max_threads: usize,
        static_bytes: usize,
        device_bytes: usize,
    ) -> bool {
        width == 32
            && max_threads >= self.candidate.threads as usize
            && static_bytes <= device_bytes
            && self.threadgroup_bytes <= device_bytes
    }
    /// Includes all auxiliary raw-cache bytes, not a misleading payload-only estimate.
    pub fn buffer_lengths(self) -> Result<[usize; 14]> {
        if Self::new(self.candidate, self.shape, self.cache, self.output)? != self {
            return Err(Error::new("forged or stale plan fields"));
        }
        let s = self.shape;
        let cache_bytes = product(&[s.cache_elements()?, 2])?;
        let raw = self.cache != CacheFormat::SeparateBf16;
        Ok([
            product(&[s.output_elements(), 2])?,   // 0 Q
            cache_bytes,                           // 1 K or raw z
            if raw { 2 } else { cache_bytes },     // 2 V (dummy when raw)
            product(&[s.max_blocks as usize, 4])?, // 3 page table
            product(&[s.queries as usize, 4])?,    // 4 absolute positions
            product(&[s.output_elements(), 4])?,   // 5 output, roomy for FP32 or BF16
            self.partial_bytes.max(4),             // 6 unnormalized split states
            4,                                     // 7 status, overwritten by validation
            if raw {
                product(&[s.physical_blocks as usize, s.page_size as usize, 4])?
            } else {
                4
            },
            if raw { 512 * 2 } else { 2 }, // 9 K gamma
            if raw {
                product(&[s.live_keys as usize, 64, 4])?
            } else {
                4
            }, // 10 cos
            if raw {
                product(&[s.live_keys as usize, 64, 4])?
            } else {
                4
            }, // 11 sin
            4,                             // 12 incumbent context
            8,                             // 13 incumbent cu_seqlens
        ])
    }
}

/// Reference validator for the device metadata prepass. Codes are part of ABI.
/// Positions must form a contiguous chunk; suffixes after the last query remain
/// invisible. Negative page IDs are holes. No KV mutation is performed.
pub fn metadata_status(s: Shape, positions: &[i32], pages: &[i32]) -> u32 {
    if s.validate().is_err()
        || positions.len() != s.queries as usize
        || pages.len() < s.max_blocks as usize
    {
        return 1;
    }
    let first = positions[0];
    if first < 0 {
        return 2;
    }
    for (i, p) in positions.iter().enumerate() {
        if *p < 0 || *p as u32 >= s.live_keys || i64::from(*p) != i64::from(first) + i as i64 {
            return 2;
        }
    }
    let lo = s.key_start(first as u32) / s.page_size;
    let hi = *positions.last().unwrap() as u32 / s.page_size;
    for page in &pages[lo as usize..=hi as usize] {
        if *page >= 0 && *page as u32 >= s.physical_blocks {
            return 3;
        }
    }
    0
}

pub fn packed_query_head(s: Shape, packed: u32, kv_head: u32) -> Option<(u32, u32)> {
    if s.validate().is_err() || kv_head >= s.kv_heads || packed >= s.queries * s.gqa() {
        return None;
    }
    Some((packed / s.gqa(), kv_head * s.gqa() + packed % s.gqa()))
}

pub fn partition(start: u32, end: u32, split: u32, splits: u32) -> Option<Range<u32>> {
    if start > end || !matches!(splits, 1 | 2 | 4 | 8 | 16 | 32) || split >= splits {
        return None;
    }
    let n = u64::from(end - start);
    Some(
        start + (n * u64::from(split) / u64::from(splits)) as u32
            ..start + (n * u64::from(split + 1) / u64::from(splits)) as u32,
    )
}

#[derive(Clone, Debug, Serialize)]
pub struct Layout {
    pub spans: [Range<usize>; 14],
    pub capacity: usize,
}
impl Layout {
    pub fn new(plan: Plan, byte_limit: usize) -> Result<Self> {
        let lengths = plan.buffer_lengths()?;
        let mut cursor = 0usize;
        let mut spans = std::array::from_fn(|_| 0..0);
        for (i, len) in lengths.into_iter().enumerate() {
            let start = cursor
                .checked_add(GUARD)
                .and_then(|x| x.checked_add(31))
                .map(|x| x & !31)
                .ok_or_else(|| Error::new("arena alignment overflow"))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| Error::new("arena span overflow"))?;
            cursor = end
                .checked_add(GUARD)
                .ok_or_else(|| Error::new("arena guard overflow"))?;
            if cursor > byte_limit {
                return Err(Error::new("explicit arena budget exceeded"));
            }
            spans[i] = start..end;
        }
        Ok(Self {
            spans,
            capacity: cursor,
        })
    }
    pub fn validate(&self, plan: Plan, capacity: usize) -> Result<()> {
        let lengths = plan.buffer_lengths()?;
        for (i, span) in self.spans.iter().enumerate() {
            if span.start % 4 != 0
                || span.end < span.start
                || span.end - span.start != lengths[i]
                || span.end > capacity
                || self.spans[..i]
                    .iter()
                    .any(|b| span.start < b.end && b.start < span.end)
            {
                return Err(Error::new("invalid/overlapping/undersized arena binding"));
            }
        }
        Ok(())
    }
}
