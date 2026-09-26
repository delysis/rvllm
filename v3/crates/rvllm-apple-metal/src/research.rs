//! Explicit, per-owner research policy. These are not production defaults.
//! No environment reads, FFI, allocation, or hardware execution in routing.
#![forbid(unsafe_code)]

use std::ops::Range;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MetalResearchCandidate {
    #[default]
    Off,
    ShortMma16x64,
    RoundedGate32,
    GqaKv8,
    Mma32Prefetch,
    AttentionQ4,
    RmsSimd32,
    Mma32F32,
    LongMma32x64,
    Mma32Load4,
    RmsnormSimd256,
    DecodeGemvMlx16,
    DecodeGateupMlx16,
    Load4M16N32K64,
    Load4M16N64K64,
    Load4M32N32K64,
    Load4M32N64K32,
    Load4M32N64K64,
    Load4M32N64K128,
    Load4M64N64K64,
    GlobalD512R8P64T64,
    GlobalD512R8P64T128,
    GlobalD512R8P128T64,
    GlobalD512R8P128T128,
    GlobalD512R16P64T64,
    GlobalD512R16P64T128,
    GlobalD512R16P128T64,
    GlobalD512R16P128T128,
    GlobalD512R1P128T32,
    GlobalD512SplitR8S256T128,
    GlobalD512SplitMmaR8K32S256T128,
    GlobalD512SplitCoopKeyR8K8P64T128S32,
    GlobalD512AtlasR16K16P64T128,
    GlobalD512AtlasR16K32P64T128,
    GlobalD512AtlasTileR16K16P64T128,
    GlobalD512AtlasTileR16K32P64T128,
    GlobalD512AtlasMmaR16K16P64T128,
    GlobalD512AtlasMmaR16K32P64T128,
    GlobalD512AtlasMmaR16K16P128T128,
    GlobalD512AtlasMmaR8K32P64T128,
    GlobalD512AtlasMmaR16K16P64T64,
    GlobalD512AtlasMmaR16K64P64T128,
}

impl MetalResearchCandidate {
    /// Explicit decode family, never inferred from a coincidentally matching shape.
    pub const fn global_decode_tile(self) -> Option<crate::attention_global_decode::DecodeTile> {
        use crate::attention_global_decode::DecodeTile;
        let (rows, keys, panel, threads, per_tile_softmax, simd_matrix) = match self {
            Self::GlobalD512R8P64T64 => (8, 8, 64, 64, false, false),
            Self::GlobalD512R8P64T128 => (8, 8, 64, 128, false, false),
            Self::GlobalD512R8P128T64 => (8, 8, 128, 64, false, false),
            Self::GlobalD512R8P128T128 => (8, 8, 128, 128, false, false),
            Self::GlobalD512R16P64T64 => (16, 8, 64, 64, false, false),
            Self::GlobalD512R16P64T128 => (16, 8, 64, 128, false, false),
            Self::GlobalD512R16P128T64 => (16, 8, 128, 64, false, false),
            Self::GlobalD512R16P128T128 => (16, 8, 128, 128, false, false),
            Self::GlobalD512R1P128T32 => (1, 8, 128, 32, false, false),
            Self::GlobalD512AtlasR16K16P64T128 => (16, 16, 64, 128, false, false),
            Self::GlobalD512AtlasR16K32P64T128 => (16, 32, 64, 128, false, false),
            Self::GlobalD512AtlasTileR16K16P64T128 => (16, 16, 64, 128, true, false),
            Self::GlobalD512AtlasTileR16K32P64T128 => (16, 32, 64, 128, true, false),
            Self::GlobalD512AtlasMmaR16K16P64T128 => (16, 16, 64, 128, true, true),
            Self::GlobalD512AtlasMmaR16K32P64T128 => (16, 32, 64, 128, true, true),
            Self::GlobalD512AtlasMmaR16K16P128T128 => (16, 16, 128, 128, true, true),
            Self::GlobalD512AtlasMmaR8K32P64T128 => (8, 32, 64, 128, true, true),
            Self::GlobalD512AtlasMmaR16K16P64T64 => (16, 16, 64, 64, true, true),
            Self::GlobalD512AtlasMmaR16K64P64T128 => (16, 64, 64, 128, true, true),
            _ => return None,
        };
        Some(DecodeTile {
            rows,
            keys,
            panel,
            threads,
            per_tile_softmax,
            simd_matrix,
        })
    }

    pub const fn split_global_decode_tile(
        self,
    ) -> Option<crate::attention_global_decode::SplitDecodeTile> {
        match self {
            Self::GlobalD512SplitR8S256T128 => {
                Some(crate::attention_global_decode::SPLIT_R8S256T128)
            }
            Self::GlobalD512SplitMmaR8K32S256T128 => {
                Some(crate::attention_global_decode::SPLIT_MATRIX_R8K32S256T128)
            }
            Self::GlobalD512SplitCoopKeyR8K8P64T128S32 => {
                Some(crate::attention_global_decode::SPLIT_COOP_KEY_R8K8P64T128S32)
            }
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        self.spec().name
    }

    pub const fn kernels(self) -> &'static [crate::research_evidence::ResearchKernel] {
        self.spec().kernels
    }

    pub(crate) const fn source(self) -> &'static str {
        self.spec().source
    }
}

impl FromStr for MetalResearchCandidate {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        crate::research_catalog::ALL_CANDIDATES
            .into_iter()
            .find(|candidate| candidate.name() == value)
            .ok_or("unknown RVLLM_METAL_RESEARCH candidate; no candidate selected")
    }
}

/// Full layer identity, not just a coincidentally matching matrix dimension.
#[derive(Clone, Copy, Debug)]
pub struct Gemma12bResearchShape {
    pub tokens: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub attention_window: u32,
    pub moe_experts: u32,
    pub moe_top_k: u32,
    pub moe_intermediate: u32,
    pub ple: u32,
}

impl Gemma12bResearchShape {
    pub fn supports(self, candidate: MetalResearchCandidate) -> bool {
        self.hidden == 3840
            && self.intermediate == 15360
            && self.layers == 48
            && self.heads == 16
            && self.moe_experts == 0
            && self.moe_top_k == 0
            && self.moe_intermediate == 0
            && self.ple == 0
            && matches!((self.kv_heads, self.head_dim), (8, 256) | (1, 512))
            // Projection tiling is independent of the attention window and
            // follows the existing MMA contract. Attention/FFN experiments
            // retain the exact archived layer-family tuple.
            && (candidate.spec().window_independent
                || matches!((self.kv_heads, self.head_dim, self.attention_window),
                    (8, 256, 1024) | (1, 512, 0)))
            && (candidate.global_decode_tile().is_none()
                && candidate.split_global_decode_tile().is_none()
                || (self.tokens == 1 && self.kv_heads == 1
                    && self.head_dim == 512 && self.attention_window == 0))
            && candidate != MetalResearchCandidate::Off
            && (candidate.spec().min_tokens..=candidate.spec().max_tokens).contains(&self.tokens)
    }
}

/// Reject a surprising SIMD width or insufficient *queried* PSO/device limits.
/// `planned_bytes` is the shader's conservative source-level scratch bound;
/// the compiler-reported static allocation may differ and is checked too.
pub fn launch_fits(
    execution_width: usize,
    maximum_threads: usize,
    static_bytes: usize,
    device_bytes: usize,
    threads: usize,
    planned_bytes: usize,
) -> bool {
    execution_width == 32
        && matches!(threads, 32 | 64 | 128 | 256)
        && maximum_threads >= threads
        && device_bytes >= planned_bytes
        && device_bytes >= static_bytes
}

pub fn matrix_bytes(rows: u32, columns: u32, element_bytes: usize) -> Option<usize> {
    if rows == 0 || columns == 0 || !matches!(element_bytes, 2 | 4) {
        return None;
    }
    usize::try_from(rows)
        .ok()?
        .checked_mul(usize::try_from(columns).ok()?)?
        .checked_mul(element_bytes)
}

pub fn buffer_span(offset: usize, bytes: usize, arena_bytes: usize) -> Option<Range<usize>> {
    if bytes == 0 || offset % 2 != 0 {
        return None;
    }
    let end = offset.checked_add(bytes)?;
    (end <= arena_bytes).then_some(offset..end)
}

/// Read/read overlap is legal. Every write must be disjoint from all other
/// writes and every live read; a fusion must not shorten scratch lifetimes.
pub fn disjoint_writes(reads: &[Range<usize>], writes: &[Range<usize>]) -> bool {
    let overlap = |a: &Range<usize>, b: &Range<usize>| a.start < b.end && b.start < a.end;
    writes.iter().enumerate().all(|(i, write)| {
        write.start < write.end
            && reads.iter().all(|read| !overlap(read, write))
            && writes[..i].iter().all(|previous| !overlap(previous, write))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(tokens: u32) -> Gemma12bResearchShape {
        Gemma12bResearchShape {
            tokens,
            hidden: 3840,
            intermediate: 15360,
            layers: 48,
            heads: 16,
            kv_heads: 8,
            head_dim: 256,
            attention_window: 1024,
            moe_experts: 0,
            moe_top_k: 0,
            moe_intermediate: 0,
            ple: 0,
        }
    }

    #[test]
    fn selection_is_exact_and_default_off() {
        assert_eq!(
            MetalResearchCandidate::default(),
            MetalResearchCandidate::Off
        );
        for kind in [
            MetalResearchCandidate::Off,
            MetalResearchCandidate::ShortMma16x64,
            MetalResearchCandidate::RoundedGate32,
            MetalResearchCandidate::GqaKv8,
        ] {
            assert_eq!(kind.name().parse(), Ok(kind));
        }
        for bad in [
            "",
            "auto",
            "mma32",
            "metal-short-mma16x64 ",
            "METAL-GQA-KV8",
        ] {
            assert!(bad.parse::<MetalResearchCandidate>().is_err());
        }
    }

    #[test]
    fn near_miss_models_never_enter_any_candidate() {
        for kind in [
            MetalResearchCandidate::ShortMma16x64,
            MetalResearchCandidate::RoundedGate32,
            MetalResearchCandidate::GqaKv8,
        ] {
            let good = model(64);
            assert!(good.supports(kind));
            for bad in [
                Gemma12bResearchShape {
                    hidden: 4096,
                    ..good
                },
                Gemma12bResearchShape { layers: 47, ..good },
                Gemma12bResearchShape { heads: 32, ..good },
                Gemma12bResearchShape {
                    kv_heads: 4,
                    ..good
                },
                Gemma12bResearchShape {
                    head_dim: 128,
                    ..good
                },
                Gemma12bResearchShape {
                    intermediate: 16384,
                    ..good
                },
                Gemma12bResearchShape {
                    moe_experts: 1,
                    ..good
                },
                Gemma12bResearchShape {
                    moe_top_k: 1,
                    ..good
                },
                Gemma12bResearchShape {
                    moe_intermediate: 1,
                    ..good
                },
                Gemma12bResearchShape { ple: 1, ..good },
                Gemma12bResearchShape {
                    tokens: u32::MAX,
                    ..good
                },
            ] {
                assert!(!bad.supports(kind));
            }
            assert_eq!(
                Gemma12bResearchShape {
                    attention_window: 512,
                    ..good
                }
                .supports(kind),
                kind == MetalResearchCandidate::ShortMma16x64
            );
            assert!(Gemma12bResearchShape {
                kv_heads: 1,
                head_dim: 512,
                attention_window: 0,
                ..good
            }
            .supports(kind));
        }
        assert!(!model(64).supports(MetalResearchCandidate::Off));
        assert!(!model(65).supports(MetalResearchCandidate::ShortMma16x64));
        assert!(!model(63).supports(MetalResearchCandidate::GqaKv8));
        assert!(!model(5).supports(MetalResearchCandidate::RoundedGate32));
    }

    #[test]
    fn resource_and_alias_guards_reject_boundary_failures() {
        assert!(launch_fits(32, 128, 16384, 32768, 128, 16384));
        for (width, max_threads, static_bytes, device_bytes) in [
            (64, 128, 16384, 32768),
            (32, 127, 16384, 32768),
            (32, 128, 32769, 32768),
            (32, 128, 8192, 16383),
        ] {
            assert!(!launch_fits(
                width,
                max_threads,
                static_bytes,
                device_bytes,
                128,
                16384
            ));
        }
        assert_eq!(matrix_bytes(84, 30720, 2), Some(5_160_960));
        assert_eq!(matrix_bytes(0, 3840, 2), None);
        assert_eq!(buffer_span(usize::MAX - 1, 8, usize::MAX), None);
        assert_eq!(buffer_span(3, 4, 128), None);
        assert_eq!(buffer_span(120, 10, 128), None);
        assert!(disjoint_writes(&[0..64, 16..32], &[64..128, 128..256]));
        assert!(!disjoint_writes(&[0..64], &[62..128]));
        assert!(!disjoint_writes(&[0..64], &[64..128, 126..256]));
    }
}

/// This candidate varies tiling only within the existing native-BF16 MMA
/// contract, including its already-FP32 QKV result. It does not enable that
/// contract or remove a storage-rounding boundary by itself.
pub fn short_mma_shape(m: u32, n: u32, k: u32, output_f32: bool) -> bool {
    (6..=64).contains(&m)
        && if output_f32 {
            k == 3840 && matches!(n, 8192 | 9216)
        } else {
            matches!((n, k), (30720, 3840) | (3840, 4096 | 8192 | 15360))
        }
}

pub fn projection_buffers_fit(
    offsets: [usize; 3],
    shape: [u32; 3],
    output_bytes: usize,
    arena_bytes: usize,
) -> bool {
    let [m, n, k] = shape;
    let checked = || {
        if offsets[2] % output_bytes != 0 {
            return None;
        }
        let a = buffer_span(offsets[0], matrix_bytes(m, k, 2)?, arena_bytes)?;
        let b = buffer_span(offsets[1], matrix_bytes(n, k, 2)?, arena_bytes)?;
        let c = buffer_span(offsets[2], matrix_bytes(m, n, output_bytes)?, arena_bytes)?;
        Some(disjoint_writes(&[a, b], &[c]))
    };
    matches!(output_bytes, 2 | 4) && checked().unwrap_or(false)
}

#[cfg(test)]
mod short_mma_tests {
    use super::*;
    use crate::{MetalFloatType, MetalKernelOptions};

    #[test]
    fn short_projection_shape_and_buffer_boundaries() {
        for m in [6, 15, 16, 17, 32, 63, 64] {
            assert!(short_mma_shape(m, 8192, 3840, true));
            assert!(short_mma_shape(m, 9216, 3840, true));
            assert!(short_mma_shape(m, 30720, 3840, false));
            assert!(!short_mma_shape(m, 9216, 3840, false));
            assert!(!short_mma_shape(m, 3840, 3840, false));
        }
        for m in [0, 1, 5, 65, 1024, u32::MAX] {
            assert!(!short_mma_shape(m, 30720, 3840, false));
        }
        // A=8 bytes, B=12 bytes, C=24 bytes. The end is exactly 44.
        assert!(projection_buffers_fit([0, 8, 20], [2, 3, 2], 4, 44));
        assert!(!projection_buffers_fit([0, 8, 20], [2, 3, 2], 4, 43));
        assert!(!projection_buffers_fit([0, 8, 18], [2, 3, 2], 4, 44));
        assert!(!projection_buffers_fit([0, 8, 16], [2, 3, 2], 4, 44));
        assert!(!projection_buffers_fit([0, 8, 20], [2, 3, 2], 0, 44));
    }

    #[test]
    fn emitted_storage_and_output_abis_are_explicit() {
        let candidate = MetalResearchCandidate::ShortMma16x64;
        let options = MetalKernelOptions {
            research: candidate,
            ..MetalKernelOptions::default()
        };
        let bf16 = crate::kernels::kernel_source_with_options(MetalFloatType::Bf16, options);
        let extra = bf16.split("// metal-short-mma16x64:").nth(1).unwrap();
        assert!(extra.contains("threadgroup bfloat at[16 * 40]"));
        assert!(extra.contains("simdgroup_matrix<bfloat, 8, 8>"));
        assert!(!extra.contains("half"));
        assert!(extra.contains("device float *C [[buffer(2)]]"));
        assert!(extra.contains("C[output] = bf16_sat(value)"));
        let ordinary = crate::kernels::kernel_source_with_options(
            MetalFloatType::F16,
            MetalKernelOptions::default(),
        );
        assert_eq!(ordinary.as_ref(), crate::kernels::KERNEL_SOURCE);
        assert!(!crate::kernels::KERNEL_NAMES
            .iter()
            .any(|name| name.starts_with("research_")));
    }

    #[test]
    fn cooperative_staging_and_fragment_output_cover_each_element_once() {
        let mut a_writes = [0_u8; 16 * 40];
        let mut b_writes = [0_u8; 64 * 40];
        for tid in 0..128 {
            for i in (tid..512).step_by(128) {
                a_writes[(i / 32) * 40 + i % 32] += 1;
            }
            for i in (tid..2048).step_by(128) {
                b_writes[(i / 32) * 40 + i % 32] += 1;
            }
        }
        for row in a_writes.chunks_exact(40).chain(b_writes.chunks_exact(40)) {
            assert!(row[..32].iter().all(|&count| count == 1));
            assert!(row[32..].iter().all(|&count| count == 0));
        }
        let mut output = [0_u8; 16 * 64];
        for sg in 0..4 {
            for row in 0..16 {
                for col in 0..16 {
                    output[row * 64 + sg * 16 + col] += 1;
                }
            }
        }
        assert!(output.iter().all(|&count| count == 1));
    }
}

/// [input, combined weights, gate/up scratch, activated scratch]. A trace
/// requires both outputs; otherwise the old gate/up allocation is left unused.
pub fn rounded_gate_buffers_fit(
    offsets: [usize; 4],
    tokens: u32,
    hidden: u32,
    intermediate: u32,
    arena_bytes: usize,
    capture_gate_up: bool,
) -> bool {
    let checked = || {
        let two_intermediate = intermediate.checked_mul(2)?;
        let a = buffer_span(offsets[0], matrix_bytes(tokens, hidden, 2)?, arena_bytes)?;
        let w = buffer_span(
            offsets[1],
            matrix_bytes(two_intermediate, hidden, 2)?,
            arena_bytes,
        )?;
        let gu = buffer_span(
            offsets[2],
            matrix_bytes(tokens, two_intermediate, 2)?,
            arena_bytes,
        )?;
        let out = buffer_span(
            offsets[3],
            matrix_bytes(tokens, intermediate, 2)?,
            arena_bytes,
        )?;
        Some(if capture_gate_up {
            disjoint_writes(&[a, w], &[gu, out])
        } else {
            disjoint_writes(&[a, w], &[out])
        })
    };
    checked().unwrap_or(false)
}

#[cfg(test)]
mod rounded_gate_tests {
    use super::*;
    use crate::{MetalFloatType, MetalKernelOptions};

    #[test]
    fn fusion_rejects_shortened_input_or_weight_lifetimes() {
        // M=2,H=4,I=3: input 16, weights 48, gate/up 24, activation 12 bytes.
        assert!(rounded_gate_buffers_fit(
            [0, 16, 64, 88],
            2,
            4,
            3,
            100,
            true
        ));
        assert!(!rounded_gate_buffers_fit(
            [0, 16, 64, 88],
            2,
            4,
            3,
            99,
            true
        ));
        assert!(!rounded_gate_buffers_fit(
            [0, 16, 64, 8],
            2,
            4,
            3,
            100,
            false
        ));
        assert!(!rounded_gate_buffers_fit(
            [0, 16, 64, 32],
            2,
            4,
            3,
            100,
            false
        ));
        assert!(!rounded_gate_buffers_fit(
            [0, 16, 64, 80],
            2,
            4,
            3,
            100,
            true
        ));
        assert!(rounded_gate_buffers_fit(
            [0, 16, 64, 80],
            2,
            4,
            3,
            100,
            false
        ));
        assert!(!rounded_gate_buffers_fit(
            [0, 16, 64, 88],
            2,
            4,
            u32::MAX,
            usize::MAX,
            false
        ));
    }

    #[test]
    fn generated_epilogue_retains_both_storage_rounding_boundaries() {
        for dtype in [MetalFloatType::F16, MetalFloatType::Bf16] {
            let source = crate::kernels::kernel_source_with_options(
                dtype,
                MetalKernelOptions {
                    research: MetalResearchCandidate::RoundedGate32,
                    ..MetalKernelOptions::default()
                },
            );
            let extra = source.split("// metal-rounded-gate32:").nth(1).unwrap();
            let storage = if dtype == MetalFloatType::F16 {
                "half"
            } else {
                "bfloat"
            };
            let round = if dtype == MetalFloatType::F16 {
                "f16_sat"
            } else {
                "bf16_sat"
            };
            assert!(extra.contains(&format!(
                "{storage} rounded_gate = {round}(cg[index] + 0.0f)"
            )));
            assert!(extra.contains(&format!("{storage} rounded_up = {round}(cu[index] + 0.0f)")));
            assert!(extra.contains(&format!(
                "{round}(gelu_tanh(float(rounded_gate)) * float(rounded_up))"
            )));
            assert_eq!(extra.matches("simdgroup_multiply_accumulate(").count(), 8);
            assert!(extra.contains("if (write_gate_up != 0u)"));
        }
    }

    #[test]
    fn removed_bf16_projection_rounding_is_observably_a_different_algorithm() {
        // Independent scalar oracle for the *storage boundary*, not a claim
        // about native matrix instruction reduction or Metal tanh execution.
        use half::bf16;
        let gelu = |g: f32| 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh());
        let mut witnesses = 0;
        for i in -80..80 {
            let g = i as f32 / 31.0 + 0.003;
            let u = 0.731_f32;
            let stored =
                bf16::from_f32(gelu(bf16::from_f32(g).to_f32()) * bf16::from_f32(u).to_f32());
            let unrounded = bf16::from_f32(gelu(g) * u);
            if stored.to_bits() != unrounded.to_bits() {
                witnesses += 1;
            }
        }
        assert!(
            witnesses > 0,
            "the roundings must not be optimized out of the source contract"
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GqaBufferShape {
    pub tokens: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks: u32,
    pub num_blocks: u32,
}

impl GqaBufferShape {
    /// Offsets follow Metal buffers 0..7: Q,K,V,O,tables,lengths,cu,positions.
    pub fn buffers_fit(self, offsets: [usize; 8], arena_bytes: usize) -> bool {
        let checked = || {
            if !matches!((self.kv_heads, self.head_dim), (8, 256) | (1, 512))
                || self.block_size == 0
                || self.max_blocks == 0
                || self.num_blocks == 0
                || offsets[4..].iter().any(|offset| offset % 4 != 0)
            {
                return None;
            }
            self.max_blocks.checked_mul(self.block_size)?;
            let q_bytes = matrix_bytes(self.tokens, 16 * self.head_dim, 2)?;
            let cache_bytes = matrix_bytes(
                self.num_blocks.checked_mul(self.block_size)?,
                self.kv_heads.checked_mul(self.head_dim)?,
                2,
            )?;
            let q = buffer_span(offsets[0], q_bytes, arena_bytes)?;
            let k = buffer_span(offsets[1], cache_bytes, arena_bytes)?;
            let v = buffer_span(offsets[2], cache_bytes, arena_bytes)?;
            let out = buffer_span(offsets[3], q_bytes, arena_bytes)?;
            let table = buffer_span(
                offsets[4],
                matrix_bytes(self.max_blocks, 1, 4)?,
                arena_bytes,
            )?;
            let lengths = buffer_span(offsets[5], 4, arena_bytes)?;
            let cu = buffer_span(offsets[6], 8, arena_bytes)?;
            let positions = buffer_span(offsets[7], matrix_bytes(self.tokens, 1, 4)?, arena_bytes)?;
            Some(disjoint_writes(
                &[q, k, v, table, lengths, cu, positions],
                &[out],
            ))
        };
        checked().unwrap_or(false)
    }
}

#[cfg(test)]
mod gqa_tests {
    use super::*;

    #[test]
    fn grouped_heads_and_native_scratch_cover_both_attention_kinds() {
        for (kv_heads, head_dim, threads, scratch) in [(8, 256, 64, 8224), (1, 512, 128, 16416)] {
            assert_eq!(2 * 8 * head_dim * 2 + 8 * 4, scratch);
            let gqa = 16 / kv_heads;
            let mut heads = [0_u8; 16];
            for kv in 0..kv_heads {
                for chunk in 0..(gqa + 3) / 4 {
                    for sg in 0..threads / 32 {
                        heads[kv * gqa + chunk * 4 + sg] += 1;
                    }
                }
            }
            assert!(heads.iter().all(|&count| count == 1));
        }
    }

    #[test]
    fn metadata_and_kv_bounds_are_checked_including_output_aliases() {
        let shape = GqaBufferShape {
            tokens: 1,
            kv_heads: 1,
            head_dim: 512,
            block_size: 32,
            max_blocks: 1,
            num_blocks: 1,
        };
        // Independent cumulative layout: Q=16384, each cache=32768, O=16384.
        let offsets = [0, 16384, 49152, 81920, 98304, 98308, 98312, 98320];
        assert!(shape.buffers_fit(offsets, 98324));
        assert!(!shape.buffers_fit(offsets, 98323));
        let mut alias = offsets;
        alias[3] = 16384;
        assert!(!shape.buffers_fit(alias, 98324));
        let mut unaligned = offsets;
        unaligned[4] += 2;
        assert!(!shape.buffers_fit(unaligned, 98330));
        assert!(!GqaBufferShape {
            max_blocks: u32::MAX,
            ..shape
        }
        .buffers_fit(offsets, usize::MAX));
        assert!(!GqaBufferShape {
            num_blocks: 0,
            ..shape
        }
        .buffers_fit(offsets, usize::MAX));
    }

    #[test]
    fn source_keeps_online_statistics_float_and_checks_physical_pages() {
        let source = MetalResearchCandidate::GqaKv8.source();
        assert!(source.contains("uint(page) < num_blocks ? 1u : 2u"));
        assert!(source.contains("float max_score = -INFINITY, sum_exp = 0.0f"));
        assert!(source.contains("partial_score += q_lane[slot] * float(kt[j * D + d])"));
        assert!(source.contains(
            "out_lane[slot] = out_lane[slot] * correction + weight * float(vt[j * D + d])"
        ));
        assert!(!source.contains("half weight"));
        assert!(!source.contains("half score"));
        assert!(!source.contains("half sum_exp"));
        assert!(source.contains("poisoned ? half(NAN) : f16_sat(out_lane[slot] * inv_sum)"));
        assert!(!source.contains("f16_sat(poisoned"));
        assert!(source.contains("positions[group.x] < 0 || positions[group.x] >= context_lens[0]"));
    }
}
