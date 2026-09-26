//! Independent scalar FP64 operator oracle and deterministic adversarial data.
//! Neither the oracle nor the fixture generator calls a candidate implementation.
#![forbid(unsafe_code)]
use super::{plan, CacheFormat, Error, Plan, Result, Shape};
use serde::{Deserialize, Serialize};

pub fn widen(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}
pub fn round(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return (b >> 16) as u16 | 0x40;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Pattern {
    Mixed,
    ZeroQueries,
    Peaked,
    Alternating,
    AllHoles,
    FirstHole,
    FuturePoison,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub seed: u64,
    pub first_position: u32,
    pub pattern: Pattern,
}
/// Byte-compatible producer snapshot. For the raw format, k contains z and v is
/// empty; the normalization/rotation helper below defines the NEW producer ABI.
#[derive(Clone, Debug)]
pub struct Data {
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub pages: Vec<i32>,
    pub positions: Vec<i32>,
    pub factor: Vec<f32>,
    pub gamma: Vec<u16>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}
fn checked_len(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        Err(Error::new(format!(
            "input length {actual}, expected {expected}"
        )))
    } else {
        Ok(())
    }
}
impl Data {
    pub fn validate(&self, plan: Plan) -> Result<()> {
        let s = plan.shape;
        s.validate()?;
        checked_len(self.q.len(), s.output_elements())?;
        checked_len(self.k.len(), s.cache_elements()?)?;
        checked_len(self.pages.len(), s.max_blocks as usize)?;
        checked_len(self.positions.len(), s.queries as usize)?;
        if plan::metadata_status(s, &self.positions, &self.pages) != 0 {
            return Err(Error::new(
                "invalid metadata; no unsafe incumbent dispatch permitted",
            ));
        }
        let raw = plan.cache != CacheFormat::SeparateBf16;
        checked_len(self.v.len(), if raw { 0 } else { s.cache_elements()? })?;
        checked_len(
            self.factor.len(),
            if raw {
                s.physical_blocks as usize * s.page_size as usize
            } else {
                0
            },
        )?;
        checked_len(self.gamma.len(), if raw { 512 } else { 0 })?;
        checked_len(
            self.cos.len(),
            if raw { s.live_keys as usize * 64 } else { 0 },
        )?;
        checked_len(self.sin.len(), self.cos.len())?;
        // The entire allocated snapshot, including padding, is finite. Page-table
        // poison is separately allowed outside the visible logical range.
        if self
            .q
            .iter()
            .chain(&self.k)
            .chain(&self.v)
            .chain(&self.gamma)
            .any(|x| !widen(*x).is_finite())
            || self
                .factor
                .iter()
                .chain(&self.cos)
                .chain(&self.sin)
                .any(|x| !x.is_finite())
            || self.factor.iter().any(|x| *x <= 0.0)
        {
            return Err(Error::new(
                "non-finite data or non-positive normalization factor",
            ));
        }
        Ok(())
    }
    pub(crate) fn physical_token(&self, shape: Shape, logical: u32) -> Option<usize> {
        let page = self.pages[(logical / shape.page_size) as usize];
        if page < 0 {
            None
        } else {
            Some(page as usize * shape.page_size as usize + (logical % shape.page_size) as usize)
        }
    }
    pub(crate) fn kv(&self, plan: Plan, logical: u32, head: u32, d: u32) -> Option<(u16, u16)> {
        let pt = self.physical_token(plan.shape, logical)?;
        let base = (pt * plan.shape.kv_heads as usize + head as usize) * plan.shape.dim as usize;
        if plan.cache == CacheFormat::SeparateBf16 {
            return Some((self.k[base + d as usize], self.v[base + d as usize]));
        }
        let normalized = |coord: usize| widen(self.k[base + coord]) * self.factor[pt];
        let scaled = |coord: usize| round(normalized(coord) * widen(self.gamma[coord]));
        let v = round(normalized(d as usize));
        let k = if d < 64 || (256..320).contains(&d) {
            let pair = if d < 64 { d } else { d - 256 } as usize;
            let c = self.cos[logical as usize * 64 + pair];
            let s = self.sin[logical as usize * 64 + pair];
            let x0 = widen(scaled(pair));
            let x1 = widen(scaled(pair + 256));
            round(if d < 64 {
                (-x1).mul_add(s, x0 * c)
            } else {
                x0.mul_add(s, x1 * c)
            })
        } else {
            scaled(d as usize)
        };
        Some((k, v))
    }
    /// Materialized control for the same raw ABI. A repeated physical page can
    /// carry different logical RoPE phases: refuse rather than overwrite it.
    pub fn materialized(&self, plan: Plan) -> Result<Self> {
        self.validate(plan)?;
        if plan.cache == CacheFormat::SeparateBf16 {
            return Ok(self.clone());
        }
        let mut seen = vec![false; plan.shape.physical_blocks as usize];
        let blocks = plan.shape.live_keys.div_ceil(plan.shape.page_size) as usize;
        for &page in &self.pages[..blocks] {
            if page < 0 {
                continue;
            }
            if page as usize >= seen.len() {
                return Err(Error::new(
                    "raw materialization requires valid full live-page table",
                ));
            }
            if std::mem::replace(&mut seen[page as usize], true) {
                return Err(Error::new(
                    "raw materialization cannot alias differently rotated logical pages",
                ));
            }
        }
        let mut k = vec![0; plan.shape.cache_elements()?];
        let mut v = k.clone();
        for t in 0..plan.shape.live_keys {
            if let Some(pt) = self.physical_token(plan.shape, t) {
                for d in 0..plan.shape.dim {
                    let (ki, vi) = self.kv(plan, t, 0, d).unwrap();
                    k[pt * plan.shape.dim as usize + d as usize] = ki;
                    v[pt * plan.shape.dim as usize + d as usize] = vi;
                }
            }
        }
        Ok(Self {
            q: self.q.clone(),
            k,
            v,
            pages: self.pages.clone(),
            positions: self.positions.clone(),
            factor: vec![],
            gamma: vec![],
            cos: vec![],
            sin: vec![],
        })
    }
    /// Inputs only. Output/state/status are initialized by the guarded owner.
    pub fn payloads(&self, plan: Plan) -> Result<[Vec<u8>; 14]> {
        self.validate(plan)?;
        let u16s = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let mut a: [Vec<u8>; 14] = std::array::from_fn(|_| vec![]);
        a[0] = u16s(&self.q);
        a[1] = u16s(&self.k);
        a[2] = u16s(&self.v);
        a[3] = self.pages.iter().flat_map(|x| x.to_le_bytes()).collect();
        a[4] = self
            .positions
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        a[8] = f32s(&self.factor);
        a[9] = u16s(&self.gamma);
        a[10] = f32s(&self.cos);
        a[11] = f32s(&self.sin);
        a[12] = (self.positions.last().copied().unwrap() + 1)
            .to_le_bytes()
            .to_vec();
        a[13] = [0i32, plan.shape.queries as i32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let lengths = plan.buffer_lengths()?;
        for i in 0..14 {
            if a[i].is_empty() {
                a[i] = vec![0; lengths[i]];
            }
            checked_len(a[i].len(), lengths[i])?;
        }
        Ok(a)
    }
}

pub fn generate(plan: Plan, spec: &Fixture, byte_limit: usize) -> Result<Data> {
    plan::Layout::new(plan, byte_limit)?;
    let s = plan.shape;
    let end = spec
        .first_position
        .checked_add(s.queries)
        .ok_or_else(|| Error::new("position overflow"))?;
    if end > s.live_keys {
        return Err(Error::new("fixture queries exceed live context"));
    }
    let blocks = s.live_keys.div_ceil(s.page_size);
    if s.physical_blocks < blocks {
        return Err(Error::new(
            "fixture requires one physical page per live logical page",
        ));
    }
    let mut state = spec.seed;
    let mut random = || {
        // SplitMix64, fully specified integer generator. No platform RNG.
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 40) as i32 - 8_388_608) as f32 / 8_388_608.0
    };
    let mut q = (0..s.output_elements())
        .map(|_| round(random() * 0.2))
        .collect::<Vec<_>>();
    let k = (0..s.cache_elements()?)
        .map(|_| round(random() * 0.2))
        .collect::<Vec<_>>();
    let raw = plan.cache != CacheFormat::SeparateBf16;
    let v = if raw {
        vec![]
    } else {
        (0..k.len()).map(|_| round(random())).collect()
    };
    let mut pages = vec![-1; s.max_blocks as usize];
    for b in 0..blocks {
        pages[b as usize] = (blocks - 1 - b) as i32;
    }
    match spec.pattern {
        Pattern::ZeroQueries => q.fill(0),
        Pattern::Alternating => {
            for (i, x) in q.iter_mut().enumerate() {
                *x = round(if i % 2 == 0 { 1.0 } else { -1.0 });
            }
        }
        Pattern::AllHoles => pages.fill(-1),
        Pattern::FirstHole => pages[(s.key_start(spec.first_position) / s.page_size) as usize] = -1,
        Pattern::FuturePoison => {
            let first_invisible = (end - 1) / s.page_size + 1;
            for b in first_invisible..s.max_blocks {
                pages[b as usize] = i32::MAX;
            }
        }
        _ => {}
    }
    let mut data = Data {
        q,
        k,
        v,
        pages,
        positions: (spec.first_position..end).map(|p| p as i32).collect(),
        factor: vec![],
        gamma: vec![],
        cos: vec![],
        sin: vec![],
    };
    if raw {
        data.factor = (0..s.physical_blocks as usize * s.page_size as usize)
            .map(|pt| {
                let z = &data.k[pt * 512..(pt + 1) * 512];
                let sum = z
                    .iter()
                    .map(|z| {
                        let f = widen(*z);
                        f * f
                    })
                    .sum::<f32>();
                (sum / 512.0 + 1e-6).powf(-0.5)
            })
            .collect();
        data.gamma = (0..512)
            .map(|i| round(0.5 + (i % 23) as f32 / 23.0))
            .collect();
        for t in 0..s.live_keys {
            for pair in 0..64 {
                let angle = t as f64 * 1_000_000.0f64.powf(-(2 * pair) as f64 / 512.0);
                data.cos.push(angle.cos() as f32);
                data.sin.push(angle.sin() as f32);
            }
        }
    }
    if spec.pattern == Pattern::Peaked {
        for t in 0..s.queries {
            for head in 0..16 {
                for d in 0..s.dim {
                    if let Some((k, _)) = data.kv(plan, spec.first_position + t, head / s.gqa(), d)
                    {
                        data.q[(t as usize * 16 + head as usize) * s.dim as usize + d as usize] =
                            round(widen(k) * 32.0);
                    }
                }
            }
        }
    }
    data.validate(plan)?;
    Ok(data)
}

/// Scalar FP64 online oracle. Dot association, exp and PV all differ from GPU
/// algorithms. Bounds cap accidental huge host work; never samples away checks.
pub fn oracle(plan: Plan, data: &Data, max_fmas: u64) -> Result<Vec<f64>> {
    data.validate(plan)?;
    let s = plan.shape;
    let pairs = data
        .positions
        .iter()
        .try_fold(0u64, |n, p| {
            n.checked_add(u64::from(*p as u32 + 1 - s.key_start(*p as u32)))
        })
        .ok_or_else(|| Error::new("oracle pair count overflow"))?;
    if pairs
        .checked_mul(32 * u64::from(s.dim))
        .ok_or_else(|| Error::new("oracle work overflow"))?
        > max_fmas
    {
        return Err(Error::new(
            "complete FP64 oracle exceeds declared work budget; increase explicitly, never sample",
        ));
    }
    let mut output = vec![0.0; s.output_elements()];
    for t in 0..s.queries {
        for head in 0..16 {
            let row = (t as usize * 16 + head as usize) * s.dim as usize;
            let pos = data.positions[t as usize] as u32;
            let mut m = f64::NEG_INFINITY;
            let mut l = 0.0;
            for key in s.key_start(pos)..=pos {
                if data.physical_token(s, key).is_none() {
                    continue;
                }
                let mut score = 0.0;
                for d in 0..s.dim {
                    score += f64::from(widen(data.q[row + d as usize]))
                        * f64::from(widen(data.kv(plan, key, head / s.gqa(), d).unwrap().0));
                }
                let next = m.max(score);
                let a = if l == 0.0 { 0.0 } else { (m - next).exp() };
                let w = (score - next).exp();
                for d in 0..s.dim {
                    output[row + d as usize] = output[row + d as usize] * a
                        + w * f64::from(widen(data.kv(plan, key, head / s.gqa(), d).unwrap().1));
                }
                l = l * a + w;
                m = next;
            }
            if l > 0.0 {
                for d in 0..s.dim {
                    output[row + d as usize] /= l;
                }
            }
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    pub max_abs: f64,
    pub rel_l2: f64,
}
impl Tolerance {
    pub fn validate(self) -> Result<()> {
        if !self.max_abs.is_finite()
            || !self.rel_l2.is_finite()
            || self.max_abs < 0.0
            || self.rel_l2 < 0.0
        {
            Err(Error::new("invalid predeclared tolerance"))
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Accuracy {
    pub max_abs: f64,
    pub rel_l2: f64,
    pub elements: usize,
    pub passed: bool,
}
pub fn compare(actual: &[f32], expected: &[f64], tol: Tolerance) -> Result<Accuracy> {
    tol.validate()?;
    checked_len(actual.len(), expected.len())?;
    if actual.is_empty() {
        return Err(Error::new("empty oracle comparison"));
    }
    let (mut max_abs, mut sum, mut norm) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() {
            return Err(Error::new("nonfinite oracle output"));
        }
        let e = f64::from(*a) - *b;
        max_abs = max_abs.max(e.abs());
        sum += e * e;
        norm += b * b;
    }
    let rel_l2 = if norm > 0.0 {
        (sum / norm).sqrt()
    } else {
        sum.sqrt()
    };
    Ok(Accuracy {
        max_abs,
        rel_l2,
        elements: actual.len(),
        passed: max_abs <= tol.max_abs && rel_l2 <= tol.rel_l2,
    })
}
