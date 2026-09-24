//! Test-only deterministic operator reference. The FP32 schedule mirrors the
//! declared panel/tree/FMA order, not SIMD work ownership. FP64 is independent.
use super::{
    physical_base, round_bf16, visible_end, widen_bf16, DecodePlan, DecodeShape, DIM, HEADS,
};

#[derive(Clone, Debug)]
pub struct Fixture {
    pub shape: DecodeShape,
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub table: Vec<i32>,
    pub context: i32,
    pub position: i32,
}

impl Fixture {
    pub fn new(length: u32, block_size: u32) -> Self {
        assert!(length > 0 && block_size > 0);
        let max_blocks = (length - 1) / block_size + 3;
        let num_blocks = max_blocks + 3;
        let shape = DecodeShape {
            sequences: 1,
            heads: HEADS,
            kv_heads: 1,
            head_dim: DIM,
            block_size,
            max_blocks,
            num_blocks,
            window: 0,
            scale: 1.0,
        };
        let noise = |mut x: u32| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x & 65535) as f32 / 32768.0 - 1.0
        };
        let q = (0..HEADS * DIM)
            .map(|i| {
                round_bf16(
                    noise(i.wrapping_mul(179).wrapping_add(1009))
                        * [0.0001, 0.1, 1.0, 4.0][(i / DIM) as usize % 4],
                )
            })
            .collect();
        // Poison physical padding: a tail/future read must not look plausible.
        let mut k = vec![0x7fc1; num_blocks as usize * block_size as usize * DIM as usize];
        let mut v = k.clone();
        let table: Vec<_> = (0..max_blocks)
            .map(|b| (num_blocks - 1 - b) as i32)
            .collect();
        for t in 0..length {
            let base = physical_base(shape, &table, t).unwrap();
            for d in 0..DIM as usize {
                let seed = t.wrapping_mul(12347).wrapping_add(d as u32 * 139 + 913);
                k[base + d] = round_bf16(noise(seed) * 0.5);
                v[base + d] = round_bf16(noise(seed.wrapping_add(7919)));
            }
        }
        Self {
            shape,
            q,
            k,
            v,
            table,
            context: length as i32,
            position: length as i32 - 1,
        }
    }

    pub fn validated_end(&self, plan: DecodePlan) -> Result<u32, &'static str> {
        if self.shape != plan.shape
            || self.q.len() != (HEADS * DIM) as usize
            || self.k.len() != plan.cache_bytes / 2
            || self.v.len() != plan.cache_bytes / 2
        {
            return Err("fixture geometry/storage mismatch");
        }
        visible_end(self.shape, &self.table, self.context, self.position)
            .ok_or("invalid visible page metadata")
    }
}

pub fn dot_f32(q: &[u16], k: &[u16], panel: u32) -> f32 {
    assert!(q.len() == DIM as usize && k.len() == DIM as usize && matches!(panel, 64 | 128));
    let mut score = 0.0_f32;
    // The staging panel is not a numerical-association parameter.
    for first in (0..DIM as usize).step_by(64) {
        let mut lanes = [0.0_f32; 32];
        for (lane, partial) in lanes.iter_mut().enumerate() {
            for d in (lane..64).step_by(32) {
                *partial = widen_bf16(q[first + d]).mul_add(widen_bf16(k[first + d]), *partial);
            }
        }
        for delta in [16, 8, 4, 2, 1] {
            for lane in 0..delta {
                lanes[lane] += lanes[lane + delta];
            }
        }
        score += lanes[0];
    }
    score
}

pub fn dot_f64(q: &[u16], k: &[u16]) -> f64 {
    assert_eq!(q.len(), k.len());
    q.iter()
        .zip(k)
        .map(|(&a, &b)| widen_bf16(a) as f64 * widen_bf16(b) as f64)
        .sum()
}

pub fn output_f32(f: &Fixture, plan: DecodePlan) -> Result<Vec<f32>, &'static str> {
    let end = f.validated_end(plan)?;
    let mut output = vec![0.0_f32; (HEADS * DIM) as usize];
    for head in 0..HEADS as usize {
        let q = &f.q[head * DIM as usize..(head + 1) * DIM as usize];
        if q.iter().any(|&x| !widen_bf16(x).is_finite()) {
            return Err("nonfinite Q");
        }
        let mut u = [0.0_f32; DIM as usize];
        let (mut maximum, mut denominator) = (f32::NEG_INFINITY, 0.0_f32);
        for t in 0..end {
            let Some(base) = physical_base(f.shape, &f.table, t) else {
                continue;
            };
            let k = &f.k[base..base + DIM as usize];
            let v = &f.v[base..base + DIM as usize];
            if k.iter().chain(v).any(|&x| !widen_bf16(x).is_finite()) {
                return Err("nonfinite KV");
            }
            let score = dot_f32(q, k, plan.tile.panel);
            if !score.is_finite() {
                return Err("nonfinite QK");
            }
            let next = maximum.max(score);
            let alpha = if denominator == 0.0 {
                0.0
            } else if maximum == next {
                1.0
            } else {
                (maximum - next).exp()
            };
            let weight = if score == next {
                1.0
            } else {
                (score - next).exp()
            };
            denominator = denominator.mul_add(alpha, weight);
            for (accumulator, &value) in u.iter_mut().zip(v) {
                *accumulator = weight.mul_add(widen_bf16(value), *accumulator * alpha);
            }
            maximum = next;
        }
        let inverse = if denominator > 0.0 {
            1.0 / denominator
        } else {
            0.0
        };
        for (d, &value) in u.iter().enumerate() {
            output[head * DIM as usize + d] = value * inverse;
        }
    }
    Ok(output)
}

/// Independent two-pass FP64 attention: no panels, SIMD tree or online update.
pub fn output_f64(f: &Fixture, plan: DecodePlan) -> Result<Vec<f64>, &'static str> {
    let end = f.validated_end(plan)?;
    let mut output = vec![0.0_f64; (HEADS * DIM) as usize];
    for head in 0..HEADS as usize {
        let q = &f.q[head * DIM as usize..(head + 1) * DIM as usize];
        let mut scored = Vec::new();
        for t in 0..end {
            if let Some(base) = physical_base(f.shape, &f.table, t) {
                let score = dot_f64(q, &f.k[base..base + DIM as usize]);
                if !score.is_finite() {
                    return Err("nonfinite FP64 QK");
                }
                scored.push((base, score));
            }
        }
        let maximum = scored
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut denominator = 0.0;
        for (base, score) in scored {
            let weight = (score - maximum).exp();
            denominator += weight;
            for d in 0..DIM as usize {
                output[head * DIM as usize + d] += weight * widen_bf16(f.v[base + d]) as f64;
            }
        }
        if denominator > 0.0 {
            for d in 0..DIM as usize {
                output[head * DIM as usize + d] /= denominator;
            }
        }
    }
    Ok(output)
}

/// Sampled dots use a predeclared forward-error bound, not an output tolerance
/// chosen after native results. Accumulation operations are all FP32.
pub fn sampled_dots(
    f: &Fixture,
    plan: DecodePlan,
) -> Result<Vec<(u32, u32, f32, f64)>, &'static str> {
    let end = f.validated_end(plan)?;
    let mut dots = Vec::new();
    for head in [0_u32, 7, 15] {
        for token in [0, end / 2, end - 1] {
            let Some(base) = physical_base(f.shape, &f.table, token) else {
                continue;
            };
            let q = &f.q[(head * DIM) as usize..((head + 1) * DIM) as usize];
            let k = &f.k[base..base + DIM as usize];
            let a = dot_f32(q, k, plan.tile.panel);
            let b = dot_f64(q, k);
            let absolute_products: f64 = q
                .iter()
                .zip(k)
                .map(|(&x, &y)| (widen_bf16(x) as f64 * widen_bf16(y) as f64).abs())
                .sum();
            let bound = 32.0 * f32::EPSILON as f64 * absolute_products + 1e-30;
            if !a.is_finite() || !b.is_finite() || (a as f64 - b).abs() > bound {
                return Err("sampled FP64 dot bound exceeded");
            }
            dots.push((head, token, a, b));
        }
    }
    Ok(dots)
}
