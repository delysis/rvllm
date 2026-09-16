//! Bounded public-Metal comparison of an experimental SIMD reduction against
//! the production GEMV. Hardware execution is explicitly opt-in; the candidate
//! is not included in the production library or selected by runtime dispatch.

use crate::{
    arena::MetalBufferArena, context::MetalContext, pipeline::PipelineCache, MetalFloatType,
};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

const BASELINE: &str = "gemm_f16_vec8";
const CANDIDATE: &str = "gemm_f16_vec8_simd_reduce";
const SIMD_REDUCTION: &str = r#"
    if (simd_width == 32) {
        #pragma clang fp reassociate(off)
        // Preserve the original 128/64/32 additions, then finish the
        // 16/8/4/2/1 tree independently in one SIMD group per output column.
        uint lane = tid % 32;
        uint column = tid / 32;
        float p0 = partial[column][lane] + partial[column][lane + 128];
        float p1 = partial[column][lane + 32] + partial[column][lane + 160];
        float p2 = partial[column][lane + 64] + partial[column][lane + 192];
        float p3 = partial[column][lane + 96] + partial[column][lane + 224];
        float sum = (p0 + p2) + (p1 + p3);
        for (ushort stride = 16; stride > 0; stride >>= 1) {
            sum += simd_shuffle_down(sum, stride);
        }
        uint col = col_base + column;
        if (lane == 0 && col < N) {
            uint idx = row * N + col;
            float prior = beta == 0.0f ? 0.0f : float(C[idx]) * beta;
            C[idx] = f16_sat(sum * alpha + prior);
        }
        return;
    }
"#;
const ORIGINAL_REDUCTION: &str = r#"
    for (uint stride = GEMV_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[0][tid] += partial[0][tid + stride];
            partial[1][tid] += partial[1][tid + stride];
            partial[2][tid] += partial[2][tid + stride];
            partial[3][tid] += partial[3][tid + stride];
            partial[4][tid] += partial[4][tid + stride];
            partial[5][tid] += partial[5][tid + stride];
            partial[6][tid] += partial[6][tid + stride];
            partial[7][tid] += partial[7][tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < GEMV_N) {
        uint col = col_base + tid;
        if (col < N) {
            uint idx = row * N + col;
            float prior = beta == 0.0f ? 0.0f : float(C[idx]) * beta;
            C[idx] = f16_sat(partial[tid][0] * alpha + prior);
        }
    }
}
"#;

fn comparison_source(dtype: MetalFloatType) -> String {
    let source = super::kernel_source_for_float_type(dtype);
    let begin = source.find("kernel void gemm_f16_vec8(").unwrap();
    let next = source[begin..]
        .find("// Apple9/10 exact batch-eight projection")
        .unwrap()
        + begin;
    let kernel = &source[begin..next];
    let barrier = "    threadgroup_barrier(mem_flags::mem_threadgroup);";
    let tail = kernel.find(barrier).unwrap() + barrier.len();
    let reduction = format!("{SIMD_REDUCTION}{ORIGINAL_REDUCTION}");
    let reduction = match dtype {
        MetalFloatType::F16 => reduction,
        MetalFloatType::Bf16 => reduction.replace("f16_sat", "bf16_sat"),
    };
    let prefix = kernel[..tail].replacen(BASELINE, CANDIDATE, 1).replacen(
        "[[thread_index_in_threadgroup]]",
        "[[thread_index_in_threadgroup]],\n    uint simd_width [[threads_per_simdgroup]]",
        1,
    );
    format!("{}\n{}{}", source, prefix, reduction)
}

fn inputs(elements: usize, dtype: MetalFloatType, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut bytes = Vec::with_capacity(elements * 2);
    for _ in 0..elements {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let value = (state >> 8) as f32 / 16_777_216.0 - 0.5;
        let bits = match dtype {
            MetalFloatType::F16 => half::f16::from_f32(value).to_bits(),
            MetalFloatType::Bf16 => half::bf16::from_f32(value).to_bits(),
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

#[test]
#[ignore = "manual public-Metal GEMV comparison; no ANE access"]
fn gemv_reduction_preserves_bits_and_measures_gpu_time() -> Result<(), Box<dyn std::error::Error>> {
    let mut reports = Vec::new();
    for dtype in [MetalFloatType::F16, MetalFloatType::Bf16] {
        let mut context = MetalContext::new()?;
        context.compile_library(&comparison_source(dtype))?;
        let mut pipelines = PipelineCache::new();
        for name in [BASELINE, CANDIDATE] {
            pipelines.compile(&context, name)?;
            assert_eq!(pipelines.get(name)?.threadExecutionWidth(), 32);
        }
        for (m, n, k, alpha, beta) in [
            (3u32, 11u32, 259u32, 1.0f32, 0.0f32),
            (3, 11, 259, 1.0 / 3_840.0f32.sqrt(), 0.5),
            (1, 30_720, 3_840, 1.0, 0.0),
            (6, 30_720, 3_840, 1.0, 0.0),
            (6, 3_840, 15_360, 1.0, 0.0),
            (19, 3_840, 4_096, 1.0, 0.0),
        ] {
            let a_len = m as usize * k as usize * 2;
            let b_len = n as usize * k as usize * 2;
            let c_len = m as usize * n as usize * 2;
            let mut arena =
                MetalBufferArena::new(context.device(), a_len + b_len + 2 * c_len + 4096)?;
            let a = arena.region("a", a_len, 32)?;
            let b = arena.region("b", b_len, 32)?;
            let c0 = arena.region("baseline_with_guards", c_len + 64, 32)?;
            let c1 = arena.region("candidate_with_guards", c_len + 64, 32)?;
            let a_bytes = inputs(a_len / 2, dtype, 71);
            let b_bytes = inputs(b_len / 2, dtype, 131);
            let guard = vec![0xA5; c_len + 64];
            // SAFETY: all regions belong to this idle arena, byte lengths match,
            // and the host writes finish before any command is submitted.
            unsafe {
                arena.write_region(&a, &a_bytes)?;
                arena.write_region(&b, &b_bytes)?;
                arena.write_region(&c0, &guard)?;
                arena.write_region(&c1, &guard)?;
            }
            drop(a_bytes);
            drop(b_bytes);
            let buffer = arena.buffer_retained();
            let run = |name: &str, offset: usize| -> Result<f64, Box<dyn std::error::Error>> {
                let command = context
                    .queue()
                    .commandBuffer()
                    .ok_or("command buffer unavailable")?;
                let encoder = command
                    .computeCommandEncoder()
                    .ok_or("compute encoder unavailable")?;
                encoder.setComputePipelineState(pipelines.get(name)?);
                // SAFETY: trusted kernels share this exact ABI and 256-thread
                // geometry. A/B/C are separate, checked arena regions; setBytes
                // copies these scalar values. No host access occurs until wait.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(buffer), a.offset, 0);
                    encoder.setBuffer_offset_atIndex(Some(buffer), b.offset, 1);
                    encoder.setBuffer_offset_atIndex(Some(buffer), offset + 32, 2);
                    for (index, value) in [(3, &m), (4, &n), (5, &k)] {
                        encoder.setBytes_length_atIndex(
                            std::ptr::NonNull::from(value).cast(),
                            4,
                            index,
                        );
                    }
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&alpha).cast(), 4, 6);
                    encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&beta).cast(), 4, 7);
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: m as usize,
                            height: (n as usize).div_ceil(8),
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                encoder.endEncoding();
                command.commit();
                command.waitUntilCompleted();
                if let Some(error) = command.error() {
                    return Err(format!("GPU execution failed: {error}").into());
                }
                let milliseconds = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
                assert!(milliseconds.is_finite() && milliseconds > 0.0);
                Ok(milliseconds)
            };
            run(BASELINE, c0.offset)?;
            run(CANDIDATE, c1.offset)?;
            // SAFETY: both synchronous commands completed. Each slice stays
            // within its allocation and no subsequent command has been issued.
            let (base_bytes, candidate_bytes) = unsafe {
                (
                    std::slice::from_raw_parts(arena.host_ptr(&c0), c_len + 64).to_vec(),
                    std::slice::from_raw_parts(arena.host_ptr(&c1), c_len + 64).to_vec(),
                )
            };
            for bytes in [&base_bytes, &candidate_bytes] {
                assert!(bytes[..32]
                    .iter()
                    .chain(&bytes[c_len + 32..])
                    .all(|&v| v == 0xA5));
                assert!(bytes[32..c_len + 32]
                    .chunks_exact(2)
                    .any(|v| v != [0xA5, 0xA5]));
                for scalar in bytes[32..c_len + 32].chunks_exact(2) {
                    let bits = u16::from_le_bytes([scalar[0], scalar[1]]);
                    let finite = match dtype {
                        MetalFloatType::F16 => half::f16::from_bits(bits).is_finite(),
                        MetalFloatType::Bf16 => half::bf16::from_bits(bits).is_finite(),
                    };
                    assert!(finite, "nonfinite projection output");
                }
            }
            let differences = base_bytes[32..c_len + 32]
                .chunks_exact(2)
                .zip(candidate_bytes[32..c_len + 32].chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                differences, 0,
                "dtype={dtype:?} M={m} N={n} K={k} alpha={alpha} beta={beta}"
            );
            let mut baseline_ms = Vec::new();
            let mut candidate_ms = Vec::new();
            for iteration in 0..6 {
                if iteration % 2 == 0 {
                    baseline_ms.push(run(BASELINE, c0.offset)?);
                    candidate_ms.push(run(CANDIDATE, c1.offset)?);
                } else {
                    candidate_ms.push(run(CANDIDATE, c1.offset)?);
                    baseline_ms.push(run(BASELINE, c0.offset)?);
                }
            }
            let report = serde_json::json!({"dtype":dtype.report_name(), "m":m, "n":n, "k":k, "alpha":alpha, "beta":beta,
                "bit_differences":differences, "baseline_gpu_ms":baseline_ms, "candidate_gpu_ms":candidate_ms});
            eprintln!("{report}");
            reports.push(report);
        }
    }
    if let Some(path) = std::env::var_os("RVLLM_METAL_GEMV_COMPARISON_REPORT") {
        let output = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?;
        serde_json::to_writer_pretty(
            output,
            &serde_json::json!({"cases":reports,
            "claim":"Public Metal synthetic GEMV comparison with guarded outputs; not full-model accuracy or speedup evidence. No ANE access."}),
        )?;
    }
    Ok(())
}
