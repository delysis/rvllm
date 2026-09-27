//! Independent deterministic fixtures. Structured cases have closed-form full
//! output references; periodic dense cases exercise cancellation/dynamic range.
//! These are operator screens, not checkpoint-quality evidence.
#![forbid(unsafe_code)]
use super::{digest, Result};
use half::bf16;
use rvllm_apple_metal::{prefill_round as plan, MetalResearchCandidate as Candidate};
use serde_json::{json, Value};
use std::{fs, path::Path};

pub const GUARD: usize = 32;
pub struct Buffer {
    pub id: usize,
    pub bytes: Vec<u8>,
    pub read_only: bool,
}
pub struct Fixture {
    pub buffers: Vec<Buffer>,
    pub passes: Vec<Value>,
    pub output: usize,
    pub width: usize,
    pub rows: usize,
    pub columns: usize,
    pub oracle: Oracle,
    pub label: String,
    pub intermediate: Option<ProjectionCheck>,
}
pub struct ProjectionCheck {
    pub values: Vec<f64>,
    pub absolute_sums: Vec<f64>,
    pub period: usize,
    pub rows: usize,
    pub columns: usize,
    pub width: usize,
    pub exact: bool,
    pub k: usize,
    pub ordered_fp32: Option<Vec<f32>>,
}
pub enum Oracle {
    Projection {
        values: Vec<f64>,
        absolute_sums: Vec<f64>,
        period: usize,
        exact: bool,
        norm: bool,
        gelu: bool,
    },
    Attention {
        values: Vec<f64>,
    },
    Untouched,
    Poison,
}
fn guarded(payload: Vec<u8>) -> Vec<u8> {
    let mut out = vec![0xa5; GUARD];
    out.extend(payload);
    out.extend([0xa5; GUARD]);
    out
}
fn bf(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values
        .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}
fn ints(values: impl Iterator<Item = i32>) -> Vec<u8> {
    values.flat_map(i32::to_le_bytes).collect()
}
fn buffer(id: usize, payload: Vec<u8>, read_only: bool) -> Buffer {
    Buffer {
        id,
        bytes: guarded(payload),
        read_only,
    }
}
fn binding(index: usize, id: usize, offset: usize) -> Value {
    json!({"index":index,"id":id,"offset":offset})
}
fn constant(index: usize, word: u32) -> Value {
    json!({"index":index,"word":word})
}
fn pass(
    kernel: &str,
    grid: [usize; 3],
    threads: [usize; 3],
    bindings: Vec<Value>,
    constants: Vec<Value>,
    shared: usize,
) -> Value {
    json!({"kernel":kernel,"grid":grid,"threads":threads,"bindings":bindings,
        "constants":constants,"sourceSharedBytes":shared})
}
fn structured_a(r: usize, k: usize) -> f32 {
    let a = (r * 5 % 7) as i32 - 3;
    let b = (r * 11 % 5) as i32 - 2;
    (a * if k % 2 == 0 { 1 } else { -1 } + b * if k % 8 < 4 { 1 } else { -1 }) as f32 / 32.0
}
fn structured_b(c: usize, k: usize) -> f32 {
    let a = (c * 7 % 31) as i32 - 15;
    let b = (c * 11 % 17) as i32 - 8;
    (a * if k % 2 == 0 { 1 } else { -1 } + b * if k % 8 < 4 { 1 } else { -1 }) as f32 / 64.0
}
fn exact_dot(r: usize, c: usize, k: usize) -> f64 {
    let a = (r * 5 % 7) as i64 - 3;
    let b = (r * 11 % 5) as i64 - 2;
    let x = (c * 7 % 31) as i64 - 15;
    let y = (c * 11 % 17) as i64 - 8;
    (k as i64 * (a * x + b * y)) as f64 / 2048.0
}
fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
fn dense(r: usize, k: usize, salt: u32) -> f32 {
    let h = mix((r as u32).wrapping_mul(2654435761) ^ (k as u32).wrapping_mul(2246822519) ^ salt);
    // Exact BF16, mixed exponents/signs. No host float RNG/library dependency.
    let bits =
        (((h >> 31) as u16) << 15) | (((119 + (h >> 7) % 9) as u16) << 7) | ((h & 127) as u16);
    bf16::from_bits(bits).to_f32()
}
fn gelu(x: f64) -> f64 {
    if x >= 5.0 {
        x
    } else if x <= -5.0 {
        0.0
    } else {
        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
    }
}
fn round(x: f64) -> f64 {
    bf16::from_f32(x as f32).to_f64()
}

// Read the actual fixture bytes rather than regenerating the operands. BF16
// products fit exactly in FP32; mul_add specifies one FP32 rounding per K step.
fn ordered_fp32_dot(a: &[u8], b: &[u8], row: usize, col: usize, k: usize) -> f32 {
    let mut acc = 0.0f32;
    for kk in 0..k {
        let ai = 2 * (row * k + kk);
        let bi = 2 * (col * k + kk);
        let av = bf16::from_bits(u16::from_le_bytes([a[ai], a[ai + 1]])).to_f32();
        let bv = bf16::from_bits(u16::from_le_bytes([b[bi], b[bi + 1]])).to_f32();
        acc = av.mul_add(bv, acc);
    }
    acc
}

#[cfg(test)]
mod ordered_fp32_tests {
    use super::*;

    #[test]
    fn long_bf16_dot_explains_the_preserved_periodic_failure() {
        let k = 15_360;
        let a = bf((0..k).map(|kk| dense(0, kk % 256, 11)));
        let b = bf((0..k).map(|kk| dense(25, kk % 256, 79)));
        assert_eq!(ordered_fp32_dot(&a, &b, 0, 0, k), 347.127_72_f32);
    }
}

/// `normal` selects scalar GEMM / fused norm or batch8 FP32 QKV controls.
/// These component calls are not a full normal-route QKV capture.
/// `metal-mma32-load4` retains its shader's 1024-row admission; at M=2048
/// it uses two 1024-row chunks. The incumbent MMA shader itself admits M=2048
/// in an operator fixture, although its normal-route host selector does not.
pub fn projection(arm: &str, role: &str, m: usize, kind: &str) -> Result<Fixture> {
    if !(6..=2048).contains(&m) {
        return Err("projection M outside tournament".into());
    }
    let norm = role.ends_with("-norm");
    let gelu = role == "gate-gelu";
    let base = role.strip_suffix("-norm").unwrap_or(role);
    let (n, k, qkv) = match base {
        "qkv-local" => (8192, 3840, true),
        "qkv-global" => (9216, 3840, true),
        "gate-up" | "gate-gelu" => (30720, 3840, false),
        "o-local" => (3840, 4096, false),
        "o-global" => (3840, 8192, false),
        "down" => (3840, 15360, false),
        _ => return Err("unknown projection role".into()),
    };
    if norm && (qkv || n != 3840) {
        return Err("invalid norm role".into());
    }
    let periodic = kind == "periodic";
    if periodic && m > 17 {
        return Err("periodic full-output screen is short-only".into());
    }
    if !["structured", "periodic", "wrong-threads", "wrong-m"].contains(&kind) {
        return Err("unknown fixture".into());
    }
    let period = if periodic { 64 } else { n };
    let a = bf((0..m * k).map(|i| {
        if periodic {
            dense(i / k, i % k % 256, 11)
        } else {
            structured_a(i / k, i % k)
        }
    }));
    let b = bf((0..n * k).map(|i| {
        if periodic {
            dense(i / k % 64, i % k % 256, 79)
        } else {
            structured_b(i / k, i % k)
        }
    }));
    let mut values = Vec::with_capacity(m * period);
    let mut absolute_sums = Vec::with_capacity(m * period);
    for r in 0..m {
        for c in 0..period {
            if periodic {
                let mut dot = 0.0;
                let mut abs = 0.0;
                for kk in 0..256 {
                    let v = dense(r, kk, 11) as f64 * dense(c, kk, 79) as f64;
                    dot += v;
                    abs += v.abs();
                }
                values.push(dot * (k / 256) as f64);
                absolute_sums.push(abs * (k / 256) as f64);
            } else {
                values.push(exact_dot(r, c, k));
                absolute_sums.push(0.0);
            }
        }
    }
    let mut buffers = vec![buffer(0, a, true), buffer(1, b, true)];
    let mut passes = Vec::new();
    let new = arm
        .parse::<Candidate>()
        .ok()
        .filter(|c| plan::projection_kernels(*c).is_some());
    let fp32 = qkv || (norm && new.is_some());
    let width = if fp32 { 4 } else { 2 };
    buffers.push(buffer(2, vec![0xff; m * n * width], false));
    if norm {
        buffers.push(buffer(
            3,
            bf((0..n).map(|d| (d % 7 + 1) as f32 / 8.0)),
            true,
        ));
        buffers.push(buffer(4, vec![0xff; m * n * 2], false));
    }
    if gelu {
        buffers.push(buffer(4, vec![0xff; m * (n / 2) * 2], false));
    }
    if norm && arm == "normal" {
        passes.push(pass(
            "gemm_rmsnorm_f16",
            [m, 1, 1],
            [256, 1, 1],
            vec![
                binding(0, 0, 32),
                binding(1, 1, 32),
                binding(2, 3, 32),
                binding(3, 4, 32),
            ],
            vec![
                constant(4, m as u32),
                constant(5, n as u32),
                constant(6, k as u32),
                constant(7, 1e-6f32.to_bits()),
            ],
            25600,
        ));
    } else {
        let (kernel, tile, threads, shared, chunk) = if let Some(c) = new {
            let ks = plan::projection_kernels(c).unwrap();
            let kernel = if norm {
                plan::postnorm_kernels(c).unwrap()[0]
            } else if qkv {
                ks[1]
            } else {
                ks[0]
            };
            (
                kernel.name().to_string(),
                [32, 64],
                [128, 1, 1],
                kernel.limits().1,
                2048,
            )
        } else if arm == "mma32" {
            (
                (if qkv {
                    "qkv_project_f32_mma32"
                } else {
                    "gemm_f16_mma32"
                })
                .into(),
                [32, 32],
                [128, 1, 1],
                8192,
                2048,
            )
        } else if arm == "metal-mma32-load4" {
            (
                (if qkv {
                    "wave2_qkv_mma32_load4"
                } else {
                    "wave2_gemm_mma32_load4"
                })
                .into(),
                [32, 32],
                [128, 1, 1],
                8192,
                1024,
            )
        } else if arm == "normal" {
            (
                (if qkv {
                    "qkv_project_f32_batch8"
                } else {
                    "gemm_f16"
                })
                .into(),
                [8, 8],
                if qkv { [256, 1, 1] } else { [8, 8, 1] },
                0,
                2048,
            )
        } else {
            return Err("arm does not own this projection".into());
        };
        for start in (0..m).step_by(chunk) {
            let rows = (m - start).min(chunk);
            let mut constants = vec![
                constant(3, rows as u32),
                constant(4, n as u32),
                constant(5, k as u32),
                constant(6, 1f32.to_bits()),
                constant(7, 0),
            ];
            if kind == "wrong-m" {
                constants[0] = constant(3, 2049);
            }
            passes.push(pass(
                &kernel,
                [rows.div_ceil(tile[0]), n.div_ceil(tile[1]), 1],
                if kind == "wrong-threads" {
                    [64, 1, 1]
                } else {
                    threads
                },
                vec![
                    binding(0, 0, 32 + start * k * 2),
                    binding(1, 1, 32),
                    binding(2, 2, 32 + start * n * width),
                ],
                constants,
                shared,
            ));
        }
        if norm {
            let (name, t, shared) = if let Some(c) = new {
                let kernel = plan::postnorm_kernels(c).unwrap()[1];
                (kernel.name(), 256, kernel.limits().1)
            } else {
                ("rmsnorm_f16", 256, 1024)
            };
            let mut constants = vec![constant(3, n as u32), constant(4, 1e-6f32.to_bits())];
            if new.is_some() {
                constants.push(constant(5, m as u32));
            }
            passes.push(pass(
                name,
                [m, 1, 1],
                [t, 1, 1],
                vec![binding(0, 2, 32), binding(1, 4, 32), binding(2, 3, 32)],
                constants,
                shared,
            ));
        }
    }
    if gelu {
        passes.push(pass(
            "gelu_mul_f16",
            [m, (n / 2).div_ceil(256), 1],
            [1, 256, 1],
            vec![binding(0, 2, 32), binding(1, 4, 32)],
            vec![constant(2, m as u32), constant(3, (n / 2) as u32)],
            0,
        ));
    }
    let ordered_fp32 = if periodic && norm && new.is_some() {
        let actual_a = &buffers[0].bytes[GUARD..buffers[0].bytes.len() - GUARD];
        let actual_b = &buffers[1].bytes[GUARD..buffers[1].bytes.len() - GUARD];
        Some(
            (0..m)
                .flat_map(|r| (0..period).map(move |c| (r, c)))
                .map(|(r, c)| ordered_fp32_dot(actual_a, actual_b, r, c, k))
                .collect(),
        )
    } else {
        None
    };
    let intermediate = if (norm && arm != "normal") || gelu {
        Some(ProjectionCheck {
            values: values.clone(),
            absolute_sums: absolute_sums.clone(),
            period,
            rows: m,
            columns: n,
            width,
            exact: !periodic,
            k,
            ordered_fp32,
        })
    } else {
        None
    };
    // Old MMA/load4 postnorm intentionally has a different BF16 boundary. Its
    // oracle describes that arm; it is never used to bless the new raw-norm route.
    if norm && new.is_none() && arm != "normal" {
        for v in &mut values {
            *v = round(*v);
        }
    }
    if norm {
        for r in 0..m {
            let sum = (0..n)
                .map(|c| values[r * period + c % period].powi(2))
                .sum::<f64>();
            let inv = 1.0 / (sum / n as f64 + 1e-6).sqrt();
            for c in 0..period {
                values[r * period + c] *= inv;
            }
        }
    }
    let oracle = if kind.starts_with("wrong-") {
        Oracle::Untouched
    } else {
        Oracle::Projection {
            values,
            absolute_sums,
            period,
            exact: !periodic && !norm && !gelu,
            norm,
            gelu,
        }
    };
    if kind.starts_with("wrong-") && (new.is_none() || norm || gelu) {
        return Err("negative geometry fixture owns new raw projections only".into());
    }
    Ok(Fixture {
        buffers,
        passes,
        output: if norm || gelu { 4 } else { 2 },
        width: if norm || gelu { 2 } else { width },
        rows: m,
        columns: if gelu { n / 2 } else { n },
        oracle,
        intermediate,
        label: format!("{arm}/{role}/{m}/{kind}"),
    })
}

pub fn attention(arm: &str, role: &str, m: usize, kind: &str) -> Result<Fixture> {
    if !(6..=2048).contains(&m) {
        return Err("attention M outside tournament".into());
    }
    let (d, kv, window) = match role {
        "attention-local" => (256, 8, 1024),
        "attention-global" => (512, 1, 0),
        _ => return Err("attention role".into()),
    };
    let candidate = arm.parse::<Candidate>().ok();
    let kernel = candidate.and_then(|c| plan::attention_kernel(c, d as u32));
    if kernel.is_none() && !["normal", "simd-attention"].contains(&arm) {
        return Err("arm does not own attention".into());
    }
    let allowed = [
        "structured",
        "periodic",
        "holes",
        "newest",
        "all-holes",
        "bad-page",
        "bad-metadata",
        "wrong-threads",
        "wrong-m",
    ];
    if !allowed.contains(&kind) {
        return Err("unknown attention fixture".into());
    }
    if kernel.is_none() && ["bad-page", "bad-metadata", "wrong-threads", "wrong-m"].contains(&kind)
    {
        return Err("never send malformed metadata to unguarded incumbent".into());
    }
    if kind == "periodic" && m > 17 {
        return Err("dense attention fixture is short-only".into());
    }
    if arm == "normal" && ["holes", "all-holes"].contains(&kind) {
        return Err("scalar incumbent has no negative-page guard; holes are not sent to it".into());
    }
    let prefix = if m < 64 { 32 } else { 97 };
    let context = m + prefix + 3;
    let pages = context.div_ceil(32);
    let physical = (pages + 2).next_power_of_two();
    let mut table: Vec<i32> = (0..pages)
        .map(|p| ((p * 5 + 3) % physical) as i32)
        .collect();
    if kind == "holes" {
        for (p, b) in table.iter_mut().enumerate() {
            if p % 5 == 1 {
                *b = -1;
            }
        }
    }
    if kind == "all-holes" {
        table.fill(-1);
    }
    let mut kc = vec![bf16::NAN; physical * 32 * kv * d];
    let mut vc = kc.clone();
    let coeff = |t: usize, h: usize| -> (f64, f64) {
        let spike = if kind == "newest" && t == prefix + m - 1 {
            32.0
        } else {
            0.0
        };
        (
            ((t * 7 + h * 3) % 23) as f64 / 32.0 - 0.34375 + spike,
            ((t * 11 + h * 5) % 19) as f64 / 64.0 - 0.140625,
        )
    };
    let kval = |t: usize, h: usize, dd: usize| {
        (((t * 13 + h * 7 + dd * 3) % 31) as i32 - 15) as f32 / 32.0
    };
    for t in 0..context {
        let page = table[t / 32];
        if page < 0 {
            continue;
        }
        let pt = page as usize * 32 + t % 32;
        for h in 0..kv {
            let (a, b) = coeff(t, h);
            for dd in 0..d {
                kc[(pt * kv + h) * d + dd] = bf16::from_f32(kval(t, h, dd));
                let v = a * if dd % 2 == 0 { 1.0 } else { -1.0 }
                    + b * if dd % 4 < 2 { 1.0 } else { -1.0 };
                vc[(pt * kv + h) * d + dd] = bf16::from_f32(v as f32);
            }
        }
    }
    if kind == "bad-page" {
        table[1] = physical as i32;
    }
    let mut q = vec![0f32; m * 16 * d];
    for r in 0..m {
        for h in 0..16 {
            if kind == "periodic" {
                for dd in 0..d {
                    q[(r * 16 + h) * d + dd] = dense(r * 16 + h, dd, 23) / 32.0;
                }
            } else {
                q[(r * 16 + h) * d + (r * 7 + h * 11) % d] = (r % 5 + 1) as f32 / 4.0;
            }
        }
    }
    let mut expected = vec![0f64; m * 16 * d];
    for r in 0..m {
        let end = prefix + r + 1;
        let start = if window == 0 {
            0
        } else {
            end.saturating_sub(window)
        };
        for h in 0..16 {
            let kh = h / (16 / kv);
            let dd = (r * 7 + h * 11) % d;
            let amp = (r % 5 + 1) as f64 / 4.0;
            let mut terms = Vec::new();
            let mut max = f64::NEG_INFINITY;
            for t in start..end {
                let pg = table[t / 32];
                if pg < 0 {
                    continue;
                }
                if pg as usize >= physical {
                    continue;
                }
                let s = if kind == "periodic" {
                    (0..d)
                        .map(|col| q[(r * 16 + h) * d + col] as f64 * kval(t, kh, col) as f64)
                        .sum::<f64>()
                } else {
                    amp * kval(t, kh, dd) as f64
                };
                max = max.max(s);
                terms.push((t, s));
            }
            let sum = terms.iter().map(|(_, s)| (*s - max).exp()).sum::<f64>();
            // Read the independently materialized BF16 V values, not an ideal
            // rank formula: the newest spike can round its small coefficients.
            let mut basis = [0.0f64; 4];
            if sum > 0.0 {
                for &(t, s) in &terms {
                    let pt = table[t / 32] as usize * 32 + t % 32;
                    let weight = (s - max).exp() / sum;
                    for col in 0..4 {
                        basis[col] += weight * vc[(pt * kv + kh) * d + col].to_f64();
                    }
                }
            }
            for col in 0..d {
                expected[(r * 16 + h) * d + col] = basis[col % 4];
            }
        }
    }
    let mut buffers = vec![
        buffer(0, bf(q.into_iter()), true),
        buffer(
            1,
            kc.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect(),
            true,
        ),
        buffer(
            2,
            vc.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect(),
            true,
        ),
        buffer(3, vec![0xff; m * 16 * d * 2], false),
        buffer(4, ints(table.into_iter()), true),
        buffer(
            5,
            ints(
                [if kind == "bad-metadata" {
                    0
                } else {
                    context as i32
                }]
                .into_iter(),
            ),
            true,
        ),
        buffer(6, ints([0, m as i32].into_iter()), true),
        buffer(7, ints((0..m).map(|i| (prefix + i) as i32)), true),
    ];
    let mut constants = vec![
        constant(8, if kind == "wrong-m" { 2049 } else { m as u32 }),
        constant(9, 1),
        constant(10, 16),
        constant(11, kv as u32),
        constant(12, d as u32),
        constant(13, 32),
        constant(14, pages as u32),
        constant(15, 1f32.to_bits()),
        constant(16, window as u32),
    ];
    let (name, grid, threads, shared) = if let Some(kernel) = kernel {
        constants.push(constant(17, physical as u32));
        (
            kernel.name(),
            [m.div_ceil(4), 16, 1],
            [128, 1, 1],
            kernel.limits().1,
        )
    } else if arm == "simd-attention" {
        ("attention_prefill_simdgroup_f16", [m, 16, 1], [32, 1, 1], 0)
    } else {
        ("attention_prefill_f16", [m, 16, 1], [1, 1, 1], 0)
    };
    let oracle = if kind.starts_with("wrong-") {
        Oracle::Untouched
    } else if kind == "bad-page" || kind == "bad-metadata" {
        Oracle::Poison
    } else {
        Oracle::Attention { values: expected }
    };
    // Bad-page fixture is short-only so page one is visible to every row.
    if kind == "bad-page" && m > 17 {
        return Err("bad-page fixture is short-only".into());
    }
    Ok(Fixture {
        buffers: std::mem::take(&mut buffers),
        passes: vec![pass(
            name,
            grid,
            if kind == "wrong-threads" {
                [64, 1, 1]
            } else {
                threads
            },
            (0..8).map(|i| binding(i, i, 32)).collect(),
            constants,
            shared,
        )],
        output: 3,
        width: 2,
        rows: m,
        columns: 16 * d,
        oracle,
        intermediate: None,
        label: format!("{arm}/{role}/{m}/{kind}"),
    })
}

impl Fixture {
    pub fn write(&mut self, dir: &Path, library: &Path, source_digest: &str) -> Result<Value> {
        fs::create_dir_all(dir)?;
        let id = 31; // identity kernel output; outside all operation binding indices
        self.buffers.push(buffer(id, vec![0xff; 32], false));
        let mut specs = Vec::new();
        let mut sealed = Vec::new();
        for b in &self.buffers {
            let path = dir.join(format!("input-{}.bin", b.id));
            fs::write(&path, &b.bytes)?;
            specs.push(json!({"id":b.id,"file":path,"readOnly":b.read_only}));
            sealed.push(
                json!({"id":b.id,"sha256":digest(&b.bytes),"bytes":b.bytes.len(),"path":path}),
            );
        }
        let preflight = pass(
            "pr26_export_identity",
            [1, 1, 1],
            [8, 1, 1],
            vec![binding(0, id, 32)],
            vec![],
            0,
        );
        let job = json!({"schema":"rvllm.prefill-round.driver.v1","library":library,
            "buffers":specs,"preflight":[preflight],"sourceBodySha256":source_digest,
            "passes":self.passes,"warmup":2,"repeats":9});
        fs::write(dir.join("job.json"), serde_json::to_vec_pretty(&job)?)?;
        fs::write(
            dir.join("inputs.json"),
            serde_json::to_vec_pretty(&json!({"inputs":sealed,
            "source_body_sha256":source_digest,"fixture":self.label}))?,
        )?;
        Ok(job)
    }
    pub fn verify(&self, dir: &Path, source_digest: &str) -> Result<Value> {
        let read = |id, suffix| -> Result<Vec<u8>> {
            Ok(fs::read(dir.join(format!("buffer-{id}-{suffix}.bin")))?)
        };
        let identity = read(31, "first")?;
        if identity.len() != 96 || digest_identity(&identity[32..64]) != source_digest {
            return Err("compiled source identity mismatch".into());
        }
        if let Some(check) = &self.intermediate {
            let first = read(2, "first")?;
            let last = read(2, "last")?;
            if first != last || first.len() != check.rows * check.columns * check.width + 64 {
                return Err("intermediate projection extent/determinism failed".into());
            }
            for (i, chunk) in first[32..first.len() - 32]
                .chunks_exact(check.width)
                .enumerate()
            {
                let got = if check.width == 4 {
                    f32::from_le_bytes(chunk.try_into().unwrap()) as f64
                } else {
                    bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f64()
                };
                let at = (i / check.columns) * check.period + (i % check.columns) % check.period;
                let reference = check.values[at];
                let exact = if check.width == 4 {
                    reference as f32 as f64
                } else {
                    round(reference)
                };
                let allowed = 1e-5
                    + 2e-6 * check.absolute_sums[at]
                    + if check.width == 2 {
                        reference.abs() / 256.0
                    } else {
                        0.0
                    };
                if let Some(ordered) = &check.ordered_fp32 {
                    let expected = ordered[at];
                    if check.width != 4 || (got as f32).to_bits() != expected.to_bits() {
                        return Err(format!("declared sequential-FP32 projection mismatch at {i}: {got} vs {expected}").into());
                    }
                    let ku = check.k as f64 * (f32::EPSILON as f64 / 2.0);
                    let analytical_bound = 1e-5 + ku / (1.0 - ku) * check.absolute_sums[at];
                    if !got.is_finite() || (got - reference).abs() > analytical_bound {
                        return Err(format!("FP32 projection exceeds analytical FP64 error bound at {i}: {got} vs {reference}").into());
                    }
                    continue;
                }
                if !got.is_finite()
                    || (check.exact && got.to_bits() != exact.to_bits())
                    || (!check.exact && (got - reference).abs() > allowed)
                {
                    return Err(format!(
                        "raw intermediate projection mismatch at {i}: {got} vs {reference}"
                    )
                    .into());
                }
            }
        }
        let first = read(self.output, "first")?;
        let last = read(self.output, "last")?;
        if first != last {
            return Err("operator output is not bitwise deterministic across repetitions".into());
        }
        let length = self.rows * self.columns * self.width;
        if first.len() != length + 64
            || first[..32]
                .iter()
                .chain(&first[first.len() - 32..])
                .any(|&b| b != 0xa5)
        {
            return Err("output shape/guard mismatch".into());
        }
        let raw = &first[32..32 + length];
        let mut max_abs = 0f64;
        let mut sq = 0f64;
        let mut refs = 0f64;
        let mut compared = 0usize;
        if matches!(self.oracle, Oracle::Untouched) {
            if raw.iter().any(|&b| b != 0xff) {
                return Err("refused dispatch wrote output".into());
            }
            return Ok(json!({"expected_refusal":true,"untouched_bytes":length}));
        }
        for (i, bits) in raw.chunks_exact(self.width).enumerate() {
            let got = if self.width == 4 {
                f32::from_le_bytes(bits.try_into().unwrap()) as f64
            } else {
                bf16::from_bits(u16::from_le_bytes(bits.try_into().unwrap())).to_f64()
            };
            if matches!(self.oracle, Oracle::Poison) {
                if !got.is_nan() {
                    return Err(format!("invalid metadata did not poison output at {i}").into());
                }
                continue;
            }
            if !got.is_finite() {
                return Err(format!("nonfinite output at {i}").into());
            }
            let (reference, allow, exact) = match &self.oracle {
                Oracle::Projection {
                    values,
                    absolute_sums,
                    period,
                    exact,
                    norm,
                    gelu,
                } => {
                    let r = i / self.columns;
                    let c = i % self.columns;
                    let at = r * period + c % period;
                    let mut expected = values[at];
                    if *norm {
                        expected *= (c % 7 + 1) as f64 / 8.0;
                    }
                    if *gelu {
                        let up = values[r * period + (c + self.columns) % period];
                        expected = gelu_reference(expected, up);
                    }
                    (
                        expected,
                        if *norm || *gelu {
                            8e-5
                        } else {
                            1e-5 + 2e-6 * absolute_sums[at]
                        },
                        *exact,
                    )
                }
                Oracle::Attention { values } => (values[i], 5e-5, false),
                _ => return Err("unexpected oracle".into()),
            };
            let exact_value = if self.width == 4 {
                reference as f32 as f64
            } else {
                round(reference)
            };
            let delta = (got - reference).abs();
            let budget = allow
                + if self.width == 2 {
                    reference.abs() / 256.0
                } else {
                    0.0
                };
            if (exact && got.to_bits() != exact_value.to_bits()) || (!exact && delta > budget) {
                return Err(format!("oracle mismatch at [{},{}]: got={got}, ref={reference}, allowed={budget}, exact={exact}",i/self.columns,i%self.columns).into());
            }
            max_abs = max_abs.max(delta);
            sq += delta * delta;
            refs += reference * reference;
            compared += 1;
        }
        Ok(
            json!({"complete_elements":true,"elements":compared,"max_abs":max_abs,
            "relative_l2":if refs>0.0 {(sq/refs).sqrt()}else{sq.sqrt()},
            "poison_expected":matches!(self.oracle,Oracle::Poison),"bitwise_repeat":true}),
        )
    }
}
fn gelu_reference(g: f64, u: f64) -> f64 {
    gelu(round(g)) * round(u)
}
fn digest_identity(bytes: &[u8]) -> String {
    bytes
        .chunks_exact(4)
        .map(|v| format!("{:08x}", u32::from_le_bytes(v.try_into().unwrap())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_closed_form_equals_complete_fp64_dot() {
        for k in [32, 3840, 4096, 8192, 15360] {
            for r in 0..19 {
                for c in 0..71 {
                    let dot = (0..k)
                        .map(|i| structured_a(r, i) as f64 * structured_b(c, i) as f64)
                        .sum::<f64>();
                    assert_eq!(dot, exact_dot(r, c, k));
                }
            }
        }
    }
    #[test]
    fn dense_fixture_is_bf16_exact_and_has_dynamic_range() {
        let mut min = f32::INFINITY;
        let mut max = 0f32;
        for r in 0..9 {
            for k in 0..256 {
                let x = dense(r, k, 11);
                assert_eq!(x, bf16::from_f32(x).to_f32());
                min = min.min(x.abs());
                max = max.max(x.abs());
            }
        }
        assert!(max / min > 128.0);
    }
    #[test]
    fn identity_bytes_have_fixed_endianness() {
        assert_eq!(digest_identity(&[0x78, 0x56, 0x34, 0x12]), "12345678");
    }
    #[test]
    fn gelu_keeps_projection_storage_boundary() {
        let changed = (1..100).any(|i| {
            let g = i as f64 / 31.0;
            round(gelu_reference(g, 0.731)) != round(gelu(g) * 0.731)
        });
        assert!(changed);
    }
}
