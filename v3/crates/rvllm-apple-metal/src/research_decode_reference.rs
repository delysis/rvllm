//! Test-only independent scalar FP64 references. No shader source parsing,
//! model outputs, candidate result, or candidate reduction tree is an oracle.
#![forbid(unsafe_code)]
use half::f16;

pub fn widen(bits: u16) -> f64 {
    f32::from_bits(u32::from(bits) << 16) as f64
}
pub fn bf16(value: f64) -> u16 {
    let bits = (value as f32).to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x40;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
pub fn gelu(value: f64) -> f64 {
    if value >= 5.0 {
        value
    } else if value <= -5.0 {
        0.0
    } else {
        0.5 * value * (1.0 + (0.797_884_560_8 * (value + 0.044_715 * value * value * value)).tanh())
    }
}
/// Retain both incumbent BF16 materialization boundaries before activation.
pub fn activate(gate: f64, up: f64) -> u16 {
    bf16(gelu(widen(bf16(gate))) * widen(bf16(up)))
}
pub fn dense_gate_up(x: &[u16], weights: &[u16], intermediate: usize) -> Vec<u16> {
    assert_eq!(weights.len(), 2 * intermediate * x.len());
    (0..intermediate)
        .map(|r| {
            let dot = |row: usize| {
                x.iter()
                    .enumerate()
                    .map(|(k, &a)| widen(a) * widen(weights[row * x.len() + k]))
                    .sum::<f64>()
            };
            activate(dot(r), dot(intermediate + r))
        })
        .collect()
}
/// Sparse nonzero exact-shape fixture: all rows vary, including both GELU tails.
/// Only this fixture construction knows its eight nonzero coordinates per row.
pub fn sparse_term(row: usize, term: usize) -> (usize, u16) {
    (
        (row * 37 + term * 479) % 3840,
        bf16((((row * 13 + term * 7) % 31) as f64 - 15.0) / 8.0),
    )
}
pub fn sparse_activation() -> Vec<u16> {
    (0..3840)
        .map(|k| bf16(((k * 19 % 41) as f64 - 20.0) / 8.0))
        .collect()
}
pub fn sparse_gate_up() -> Vec<u16> {
    let x = sparse_activation();
    (0..15360)
        .map(|r| {
            let dot = |row| {
                (0..8)
                    .map(|term| {
                        let (k, w) = sparse_term(row, term);
                        widen(x[k]) * widen(w)
                    })
                    .sum::<f64>()
            };
            activate(dot(r), dot(r + 15360))
        })
        .collect()
}

#[derive(Clone)]
pub struct Group32Fixture {
    pub bits: u32,
    pub n: usize,
    pub k: usize,
    pub values: Vec<u8>,
    pub scales: Vec<u16>,
    pub x: Vec<u16>,
}
impl Group32Fixture {
    pub fn new(bits: u32, n: usize, k: usize) -> Self {
        assert!(matches!(bits, 4 | 8) && n > 0 && k > 0);
        let stride = if bits == 4 { k.div_ceil(2) } else { k };
        let groups = k.div_ceil(32);
        let mut values = vec![0; n * stride];
        for row in 0..n {
            for col in 0..k {
                let q = ((row * 19 + col * 7) % (1usize << bits)) as u8;
                if bits == 4 {
                    values[row * stride + col / 2] |= q << (4 * (col % 2));
                } else {
                    values[row * stride + col] = q;
                }
            }
        }
        let scales = (0..n * groups)
            .map(|i| f16::from_f64([0.03125, 0.125, 0.5, 1.5][i % 4]).to_bits())
            .collect();
        let x = (0..k)
            .map(|i| bf16(((i * 11 % 37) as f64 - 18.0) / 32.0))
            .collect();
        Self {
            bits,
            n,
            k,
            values,
            scales,
            x,
        }
    }
    pub fn output(&self) -> Vec<u16> {
        group32_fp64(
            self.bits,
            self.n,
            self.k,
            &self.values,
            &self.scales,
            &self.x,
        )
        .into_iter()
        .map(bf16)
        .collect()
    }
}
/// The checkpoint ABI is signed symmetric W4/W8, little-nibble W4, row-major
/// groups of 32, FP16 scales. There is no affine zero point, bias, or repacker.
pub fn group32_fp64(
    bits: u32,
    n: usize,
    k: usize,
    values: &[u8],
    scales: &[u16],
    x: &[u16],
) -> Vec<f64> {
    assert!(matches!(bits, 4 | 8));
    let stride = if bits == 4 { k.div_ceil(2) } else { k };
    let groups = k.div_ceil(32);
    assert_eq!(values.len(), n * stride);
    assert_eq!(scales.len(), n * groups);
    assert_eq!(x.len(), k);
    (0..n)
        .map(|row| {
            (0..k)
                .map(|col| {
                    let byte = values[row * stride + if bits == 4 { col / 2 } else { col }];
                    let q = if bits == 4 {
                        let nibble = (byte >> (4 * (col % 2))) & 15;
                        if nibble >= 8 {
                            i32::from(nibble) - 16
                        } else {
                            i32::from(nibble)
                        }
                    } else {
                        i32::from(byte as i8)
                    };
                    widen(x[col])
                        * f64::from(q)
                        * f16::from_bits(scales[row * groups + col / 32]).to_f64()
                })
                .sum()
        })
        .collect()
}
