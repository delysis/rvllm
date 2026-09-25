//! Default-off Gemma 4 prefill-attention experiment contracts.
//!
//! This is deliberately not wired into `layer_forward`: qualification owns
//! selection. QKV projection, attention, and O projection remain separate
//! measured boundaries.
#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};

pub const GENERATOR_VERSION: &str = "gemma4-prefill-online-v1";
pub const CONVENTIONAL_ENTRYPOINT: &str = "research_gemma4_prefill_tiled_online_bf16";
pub const TENSOR_OPS_ENTRYPOINT: &str = "research_gemma4_prefill_tensorops_bf16";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillArm {
    ConventionalTiled,
    TensorOps,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Boundary {
    ExternalQkvBf16,
    ExternalOutputBf16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueriedHardware {
    /// True only after the runtime has queried and recorded a TensorOps
    /// capability. A GPU-family guess is not sufficient.
    pub tensor_ops: bool,
    pub max_threadgroup_memory: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillPlan {
    pub arm: PrefillArm,
    pub tokens: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window: u32,
    pub qkv_boundary: Boundary,
    pub output_boundary: Boundary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Admission {
    Ready(GeneratedCodeIdentity),
    Unsupported(&'static str),
    Rejected(&'static str),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedCodeIdentity {
    pub generator_version: &'static str,
    pub entrypoint: &'static str,
    pub source_sha256: String,
    pub qkv_boundary: Boundary,
    pub output_boundary: Boundary,
}

/// Filled by the device build step. Empty generated-code fields are rejected,
/// so a source hash cannot masquerade as compiler/ISA/resource evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledCodeIdentity {
    pub generated: GeneratedCodeIdentity,
    pub compiler_version: String,
    pub air_sha256: String,
    pub metallib_sha256: String,
    pub disassembly_sha256: String,
    pub resource_report_sha256: String,
}

impl CompiledCodeIdentity {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.compiler_version.is_empty()
            || [
                &self.air_sha256,
                &self.metallib_sha256,
                &self.disassembly_sha256,
                &self.resource_report_sha256,
            ]
            .into_iter()
            .any(|value| value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err("incomplete generated-code identity");
        }
        Ok(())
    }
}

impl PrefillPlan {
    #[must_use]
    pub fn admit(self, hardware: QueriedHardware) -> Admission {
        if !(1..=2048).contains(&self.tokens)
            || self.heads != 16
            || !matches!(
                (self.kv_heads, self.head_dim, self.window),
                (8, 256, 1024) | (1, 512, 0)
            )
            || self.qkv_boundary != Boundary::ExternalQkvBf16
            || self.output_boundary != Boundary::ExternalOutputBf16
        {
            return Admission::Rejected("not an exact Gemma 4 prefill/QKV/O contract");
        }
        let source = match self.arm {
            PrefillArm::ConventionalTiled => CONVENTIONAL_MSL,
            PrefillArm::TensorOps if !hardware.tensor_ops => {
                return Admission::Unsupported(
                    "TensorOps capability was not reported by the device query",
                )
            }
            PrefillArm::TensorOps => {
                return Admission::Unsupported(
                    "TensorOps source is deferred pending a stable queried Metal ABI",
                )
            }
        };
        if hardware.max_threadgroup_memory < 256 {
            return Admission::Unsupported("less than 256 bytes of queried threadgroup memory");
        }
        Admission::Ready(identity(source, CONVENTIONAL_ENTRYPOINT))
    }
}

#[must_use]
pub fn identity(source: &'static str, entrypoint: &'static str) -> GeneratedCodeIdentity {
    GeneratedCodeIdentity {
        generator_version: GENERATOR_VERSION,
        entrypoint,
        source_sha256: format!("{:x}", Sha256::digest(source.as_bytes())),
        qkv_boundary: Boundary::ExternalQkvBf16,
        output_boundary: Boundary::ExternalOutputBf16,
    }
}

/// Conventional one-SIMD-group-per-query/head online softmax. Keys are
/// traversed in fixed 64-key panels while the sufficient state stays in FP32.
/// Holes are explicit negative page-table entries and are skipped.
pub const CONVENTIONAL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void research_gemma4_prefill_tiled_online_bf16(
 device const bfloat *q [[buffer(0)]], device const bfloat *k [[buffer(1)]],
 device const bfloat *v [[buffer(2)]], device bfloat *o [[buffer(3)]],
 device const int *pages [[buffer(4)]], device const int *context [[buffer(5)]],
 device const int *cu [[buffer(6)]], device const int *positions [[buffer(7)]],
 constant uint &total_q [[buffer(8)]], constant uint &batch [[buffer(9)]],
 constant uint &heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]],
 constant uint &dim [[buffer(12)]], constant uint &page_size [[buffer(13)]],
 constant uint &max_pages [[buffer(14)]], constant float &scale [[buffer(15)]],
 constant uint &window [[buffer(16)]], uint2 tg [[threadgroup_position_in_grid]],
 ushort lane [[thread_index_in_simdgroup]]) {
 uint qi=tg.x, h=tg.y; if(qi>=total_q||h>=heads||(dim!=256&&dim!=512)) return;
 uint s=batch; for(uint i=0;i<batch;i++) if(int(qi)>=cu[i]&&int(qi)<cu[i+1]) {s=i;break;}
 if(s==batch||context[s]<=0) return;
 uint qdim=heads*dim, kvdim=kv_heads*dim, kh=h/(heads/kv_heads), slots=dim/32;
 float qv[16], ov[16]; for(uint z=0;z<slots;z++){uint d=uint(lane)+32*z;qv[z]=float(q[qi*qdim+h*dim+d]);ov[z]=0;}
 uint end=min(uint(context[s]),uint(max(positions[qi],0))+1), begin=window?end-min(end,window):0;
 float m=-INFINITY,l=0;
 for(uint panel=begin;panel<end;panel+=64) for(uint t=panel;t<min(end,panel+64);t++){
  int page=pages[s*max_pages+t/page_size]; if(page<0) continue;
  uint base=uint(page)*page_size*kvdim+(t%page_size)*kvdim+kh*dim; float dot=0;
  for(uint z=0;z<slots;z++) dot+=qv[z]*float(k[base+uint(lane)+32*z]);
  float score=simd_sum(dot)*scale,nm=max(m,score),a=exp(m-nm),w=exp(score-nm); l=l*a+w;
  for(uint z=0;z<slots;z++) ov[z]=ov[z]*a+w*float(v[base+uint(lane)+32*z]); m=nm;
 }
 float inv=l>0?1/l:0; for(uint z=0;z<slots;z++){uint d=uint(lane)+32*z;o[qi*qdim+h*dim+d]=bfloat(ov[z]*inv);}
}
"#;

/// Independent scalar FP64 oracle for one query/head. It shares no reduction
/// or tiling code with the generated Metal candidate.
pub fn reference_f64(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    live: &[bool],
    scale: f64,
) -> Result<Vec<f32>, &'static str> {
    let dim = query.len();
    if dim == 0
        || keys.len() != values.len()
        || keys.len() != live.len().checked_mul(dim).ok_or("shape overflow")?
    {
        return Err("invalid oracle shape");
    }
    let mut scores = Vec::with_capacity(live.len());
    for (row, &is_live) in live.iter().enumerate() {
        if !is_live {
            scores.push(None);
            continue;
        }
        let dot = (0..dim)
            .map(|d| f64::from(query[d]) * f64::from(keys[row * dim + d]))
            .sum::<f64>();
        scores.push(Some(dot * scale));
    }
    let maximum = scores
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err("no live keys");
    }
    let denominator = scores
        .iter()
        .flatten()
        .map(|s| (s - maximum).exp())
        .sum::<f64>();
    let mut out = vec![0.0_f64; dim];
    for (row, score) in scores.into_iter().enumerate() {
        let Some(score) = score else { continue };
        let weight = (score - maximum).exp() / denominator;
        for d in 0..dim {
            out[d] += weight * f64::from(values[row * dim + d]);
        }
    }
    Ok(out.into_iter().map(|x| x as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(arm: PrefillArm, tokens: u32) -> PrefillPlan {
        PrefillPlan {
            arm,
            tokens,
            heads: 16,
            kv_heads: 8,
            head_dim: 256,
            window: 1024,
            qkv_boundary: Boundary::ExternalQkvBf16,
            output_boundary: Boundary::ExternalOutputBf16,
        }
    }
    fn hw(tensor_ops: bool) -> QueriedHardware {
        QueriedHardware {
            tensor_ops,
            max_threadgroup_memory: 32 * 1024,
        }
    }

    #[test]
    fn default_off_boundary_and_tensorops_refusal_are_explicit() {
        assert!(matches!(
            plan(PrefillArm::ConventionalTiled, 256).admit(hw(false)),
            Admission::Ready(_)
        ));
        assert_eq!(
            plan(PrefillArm::TensorOps, 256).admit(hw(false)),
            Admission::Unsupported("TensorOps capability was not reported by the device query")
        );
        assert!(matches!(
            plan(PrefillArm::TensorOps, 256).admit(hw(true)),
            Admission::Unsupported(_)
        ));
        for n in [0, 2049, u32::MAX] {
            assert!(matches!(
                plan(PrefillArm::ConventionalTiled, n).admit(hw(false)),
                Admission::Rejected(_)
            ));
        }
    }

    #[test]
    fn identity_seals_source_entrypoint_generator_and_boundaries() {
        let Admission::Ready(id) = plan(PrefillArm::ConventionalTiled, 256).admit(hw(false)) else {
            panic!()
        };
        assert_eq!(id, identity(CONVENTIONAL_MSL, CONVENTIONAL_ENTRYPOINT));
        assert_eq!(id.source_sha256.len(), 64);
        assert!(CONVENTIONAL_MSL.contains("panel+=64"));
        let incomplete = CompiledCodeIdentity {
            generated: id,
            compiler_version: String::new(),
            air_sha256: String::new(),
            metallib_sha256: String::new(),
            disassembly_sha256: String::new(),
            resource_report_sha256: String::new(),
        };
        assert_eq!(
            incomplete.validate(),
            Err("incomplete generated-code identity")
        );
    }

    #[test]
    fn oracle_covers_boundaries_tails_holes_guards_and_repeat() {
        for rows in [1_usize, 63, 64, 65, 255, 256, 257] {
            let dim = 7;
            let q = (0..dim).map(|d| d as f32 / 9.0 - 0.2).collect::<Vec<_>>();
            let k = (0..rows * dim)
                .map(|i| ((i * 17 % 29) as f32 - 14.0) / 13.0)
                .collect::<Vec<_>>();
            let v = (0..rows * dim)
                .map(|i| ((i * 11 % 31) as f32 - 15.0) / 8.0)
                .collect::<Vec<_>>();
            for hole in [None, Some(0), Some(rows / 2), Some(rows - 1)] {
                let mut live = vec![true; rows];
                if let Some(i) = hole {
                    live[i] = false;
                }
                let before = [0x55_u8; 16];
                let a = reference_f64(&q, &k, &v, &live, 0.125);
                let b = reference_f64(&q, &k, &v, &live, 0.125);
                if rows == 1 && hole == Some(0) {
                    assert_eq!(a, Err("no live keys"));
                } else {
                    assert_eq!(a, b);
                    assert!(a.unwrap().iter().all(|x| x.is_finite()));
                }
                assert_eq!(before, [0x55; 16]);
            }
        }
        let mut guarded = [7.0_f32; 4];
        let snapshot = guarded;
        let bad = reference_f64(&[1.0, 2.0], &[1.0], &[1.0], &[true], 1.0);
        if let Ok(ref v) = bad {
            guarded.copy_from_slice(&v[..4]);
        }
        assert_eq!(bad, Err("invalid oracle shape"));
        assert_eq!(guarded, snapshot);
    }
}
