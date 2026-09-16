//! Isolated public-Metal experiment. No production dispatch selects this kernel.
//! Actual Gemma weights and controlled BF16 inputs; FP32 output and accumulation.

use super::*;
use crate::arena::MetalRegion;
use crate::weight_loader::{load_safetensor_entry_bf16, scan_safetensor_tensors};
use crate::MetalFloatType;
use half::bf16;
use std::path::PathBuf;

const MATRIX_KERNEL: &str = r#"
kernel void prefill_bf16_mma32_probe(
    device const bfloat *A [[buffer(0)]],
    device const bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    const uint sm = uint(sg / 2u) * 16u;
    const uint sn = uint(sg % 2u) * 16u;
    threadgroup bfloat at[32 * 32];
    threadgroup bfloat bt[32 * 32];
    threadgroup float ct[32 * 32];
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint kb = 0; kb < K; kb += 32u) {
        for (uint index = uint(tid); index < 1024u; index += 128u) {
            uint row = index / 32u;
            uint k = kb + index % 32u;
            at[index] = mr + row < M && k < K ? A[(mr + row) * K + k] : bfloat(0.0f);
            bt[index] = nc + row < N && k < K ? B[(nc + row) * K + k] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<bfloat, 8, 8> a0, a1, b0, b1;
            simdgroup_load(a0, at + sm * 32u + kk, 32);
            simdgroup_load(a1, at + (sm + 8u) * 32u + kk, 32);
            simdgroup_load(b0, bt + sn * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(b1, bt + (sn + 8u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        // All four groups must finish reading before any overwrite the tiles.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ct + sm * 32u + sn, 32);
    simdgroup_store(c01, ct + sm * 32u + sn + 8u, 32);
    simdgroup_store(c10, ct + (sm + 8u) * 32u + sn, 32);
    simdgroup_store(c11, ct + (sm + 8u) * 32u + sn + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u;
        uint col = nc + index % 32u;
        if (row < M && col < N) {
            uint output = row * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * C[output];
            C[output] = alpha * ct[index] + prior;
        }
    }
}
"#;

#[test]
#[ignore = "bounded BF16 matrix experiment with real Gemma 4 12B weights; no ANE access"]
fn native_bf16_mma_checks_tails_precision_and_real_projection_time(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let model =
        PathBuf::from(std::env::var_os("RVLLM_METAL_QKV_MODEL_DIR").ok_or("model path required")?);
    let tensors = scan_safetensor_tensors(&model)?;
    let mut ctx = MetalContext::new()?;
    let source = format!(
        "{}\n{MATRIX_KERNEL}",
        crate::kernels::kernel_source_for_float_type(MetalFloatType::Bf16)
    );
    ctx.compile_library(&source)?;
    let mut pipelines = PipelineCache::new();
    for name in [
        "qkv_project_f32_batch8",
        "prefill_bf16_mma32_probe",
        "qkv_project_f32_mma32",
        "gemm_f16_mma32",
        "gemm_f16_batch8",
    ] {
        pipelines.compile(&ctx, name)?;
    }
    let mut reports = Vec::new();
    let mut shapes = vec![
        ("all-tails", 21_u32, 37_u32, 35_u32, vec![]),
        (
            "qkv-sliding",
            21,
            8192,
            3840,
            vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ),
        (
            "gate-up",
            27,
            30720,
            3840,
            vec!["mlp.gate_proj", "mlp.up_proj"],
        ),
        ("output", 28, 3840, 4096, vec!["self_attn.o_proj"]),
        ("down", 32, 3840, 15360, vec!["mlp.down_proj"]),
    ];
    if std::env::var_os("RVLLM_METAL_MMA_EXTENDED").is_some() {
        shapes = vec![
            ("all-tails", 6, 37, 35, vec![]),
            ("all-tails", 63, 37, 35, vec![]),
            (
                "qkv-sliding",
                6,
                8192,
                3840,
                vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ),
            (
                "qkv-sliding",
                84,
                8192,
                3840,
                vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ),
            (
                "gate-up",
                230,
                30720,
                3840,
                vec!["mlp.gate_proj", "mlp.up_proj"],
            ),
            ("output", 650, 3840, 4096, vec!["self_attn.o_proj"]),
            ("down", 1024, 3840, 15360, vec!["mlp.down_proj"]),
        ];
    }
    for (label, m, n, k, names) in shapes {
        let mut weights = Vec::new();
        for name in names {
            let name = format!("model.language_model.layers.0.{name}.weight");
            let entry = tensors.get(&name).ok_or("projection tensor missing")?;
            assert_eq!(entry.shape[1], k as usize);
            weights.extend(load_safetensor_entry_bf16(entry)?);
        }
        if weights.is_empty() {
            for i in 0..n * k {
                weights.extend_from_slice(
                    &bf16::from_f32(((i * 17 + 3) % 71) as f32 / 64.0 - 0.5).to_le_bytes(),
                );
            }
        }
        assert_eq!(weights.len(), n as usize * k as usize * 2);
        let mut input = Vec::new();
        for row in 0..m {
            let scale = [0.1_f32, 1.0, 100.0, 100000.0][row as usize % 4];
            for col in 0..k {
                input.extend_from_slice(
                    &bf16::from_f32((((col * 7 + row * 13) % 47) as f32 / 32.0 - 0.75) * scale)
                        .to_le_bytes(),
                );
            }
        }
        let output_bytes = m as usize * n as usize * 4;
        let mut arena = MetalBufferArena::new(
            ctx.device(),
            weights.len()
                + input.len()
                + 3 * (output_bytes + 64)
                + 2 * (output_bytes / 2 + 64)
                + 1024,
        )?;
        let mut upload = |name: &str, bytes: &[u8]| -> Result<MetalRegion> {
            let region = arena.region(name, bytes.len(), 32)?;
            // SAFETY: an idle arena; this slice exactly fills its new region.
            unsafe {
                arena.write_region(&region, bytes)?;
            }
            Ok(region)
        };
        let a = upload("input", &input)?;
        let b = upload("weights", &weights)?;
        let outputs = [
            upload("baseline", &vec![0xA5; output_bytes + 64])?,
            upload("prototype", &vec![0xA5; output_bytes + 64])?,
            upload("production-f32", &vec![0xA5; output_bytes + 64])?,
            upload("production-bf16", &vec![0xA5; output_bytes / 2 + 64])?,
            upload("baseline-bf16", &vec![0xA5; output_bytes / 2 + 64])?,
        ];
        drop(upload);
        let run = |path: usize| -> std::result::Result<(f64, f64), Box<dyn std::error::Error>> {
            let started = std::time::Instant::now();
            let command = ctx
                .queue()
                .commandBuffer()
                .ok_or("command buffer missing")?;
            let encoder = command.computeCommandEncoder().ok_or("encoder missing")?;
            let name = [
                "qkv_project_f32_batch8",
                "prefill_bf16_mma32_probe",
                "qkv_project_f32_mma32",
                "gemm_f16_mma32",
                "gemm_f16_batch8",
            ][path];
            let candidate = path != 0 && path != 4;
            encoder.setComputePipelineState(pipelines.get(name)?);
            let buffer = arena.buffer_retained();
            // SAFETY: all regions are live and disjoint, sized from the exact
            // dimensions above. The candidate handles every M/N/K tail. Both
            // kernels write only the FP32 interior of the guarded output.
            unsafe {
                for (index, offset) in [a.offset, b.offset, outputs[path].offset + 32]
                    .into_iter()
                    .enumerate()
                {
                    encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
                }
                for (index, value) in [m, n, k].iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::from(value).cast(),
                        4,
                        index + 3,
                    );
                }
                for (index, value) in [1.0_f32, 0.0_f32].iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::from(value).cast(),
                        4,
                        index + 6,
                    );
                }
            }
            let tile = if candidate { 32 } else { 8 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (m as usize).div_ceil(tile),
                    height: (n as usize).div_ceil(tile),
                    depth: 1,
                },
                MTLSize {
                    width: if candidate { 128 } else { 256 },
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return Err(format!("matrix GPU error: {error}").into());
            }
            Ok((
                (command.GPUEndTime() - command.GPUStartTime()) * 1000.0,
                started.elapsed().as_secs_f64() * 1000.0,
            ))
        };
        for path in 0..5 {
            run(path)?;
        }
        let read = |region: &MetalRegion| -> Vec<u8> {
            // SAFETY: synchronous completion above precedes this bounded read.
            unsafe { std::slice::from_raw_parts(arena.host_ptr(region), region.size).to_vec() }
        };
        let read_output = |region: &MetalRegion| -> Vec<f32> {
            let bytes = read(region);
            assert!(bytes[..32]
                .iter()
                .chain(&bytes[bytes.len() - 32..])
                .all(|&v| v == 0xA5));
            bytes[32..bytes.len() - 32]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        };
        let base = read_output(&outputs[0]);
        let candidate = read_output(&outputs[2]);
        assert_eq!(
            candidate,
            read_output(&outputs[1]),
            "production FP32 kernel must match the independently qualified prototype"
        );
        for (path, float_values) in [(3, &candidate), (4, &base)] {
            let bytes = read(&outputs[path]);
            assert!(bytes[..32]
                .iter()
                .chain(&bytes[bytes.len() - 32..])
                .all(|&v| v == 0xA5));
            for (packed, &value) in bytes[32..bytes.len() - 32]
                .chunks_exact(2)
                .zip(float_values)
            {
                assert_eq!(
                    bf16::from_le_bytes(packed.try_into().unwrap()),
                    bf16::from_f32(value),
                    "BF16 output must round once at the projection boundary"
                );
            }
        }
        assert!(base.iter().chain(&candidate).all(|x| x.is_finite()));
        let relative_l2 = (base
            .iter()
            .zip(&candidate)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / base.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>())
        .sqrt();
        assert!(relative_l2 < 0.0001, "{label}: relative L2 {relative_l2}");
        let bf = |bytes: &[u8], index: usize| {
            bf16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]).to_f64()
        };
        let mut max_cpu_error = 0.0_f64;
        for row in [0, m / 2, m - 1] {
            for col in [0, 7, 8, n / 2, n - 1] {
                let expected = (0..k)
                    .map(|i| {
                        bf(&input, (row * k + i) as usize) * bf(&weights, (col * k + i) as usize)
                    })
                    .sum::<f64>();
                let actual = f64::from(candidate[(row * n + col) as usize]);
                max_cpu_error = max_cpu_error.max((actual - expected).abs());
                assert!(
                    (actual - expected).abs() < 0.005 + expected.abs() * 0.0001,
                    "{label} row {row} col {col}: {actual} vs {expected}"
                );
            }
        }
        drop((input, weights));
        let (baseline_path, candidate_path) = if label == "qkv-sliding" || label == "all-tails" {
            (0, 2)
        } else {
            (4, 3)
        };
        for _ in 0..2 {
            run(baseline_path)?;
            run(candidate_path)?;
        }
        let mut baseline_times = Vec::new();
        let mut candidate_times = Vec::new();
        for i in 0..8 {
            if i % 2 == 0 {
                baseline_times.push(run(baseline_path)?);
                candidate_times.push(run(candidate_path)?);
            } else {
                candidate_times.push(run(candidate_path)?);
                baseline_times.push(run(baseline_path)?);
            }
        }
        reports.push(serde_json::json!({"projection":label,"m":m,"n":n,"k":k,"output_dtype":if candidate_path==2 {"float32"} else {"bfloat16"},"production_matches_prototype":true,"bf16_rounds_once":true,"relative_l2":relative_l2,"sampled_cpu_fp64_max_abs":max_cpu_error,"baseline_gpu_wall_ms":baseline_times,"candidate_gpu_wall_ms":candidate_times}));
    }
    let report = serde_json::json!({"cases":reports,"claim":"BF16 matrix kernel with independently checked FP32 and BF16 output ABIs and FP32 accumulation. Actual layer 0 weights, controlled inputs including BF16 values outside FP16 range, masked M/N/K tails, guarded outputs, sampled independent FP64 dots. No production dispatch or full-model quality claim."});
    if let Some(path) = std::env::var_os("RVLLM_METAL_MMA_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("{report}");
    Ok(())
}
