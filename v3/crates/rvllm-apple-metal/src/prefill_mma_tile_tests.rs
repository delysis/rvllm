//! Isolated tile/operand experiment; no production dispatch selects these probes.
use super::*;
use crate::arena::MetalRegion;
use crate::weight_loader::{load_safetensor_entry_bf16, scan_safetensor_tensors};
use crate::MetalFloatType;
use half::bf16;
use objc2_metal::MTLComputePipelineState;
use std::path::PathBuf;

// Isolate reduction depth: identical 32x32 output ownership, four accumulator
// fragments per SIMD group, and ascending 8-wide multiply-accumulate order.
// No production pipeline can select this test-only kernel.
fn reduction64_source(output: &str, name: &str) -> String {
    r#"
kernel void KERNEL_NAME(
    device const bfloat *A [[buffer(0)]], device const bfloat *B [[buffer(1)]],
    device OUTPUT_TYPE *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint mr = group.x * 32u, nc = group.y * 32u;
    const uint sm = uint(sg / 2u) * 16u, sn = uint(sg % 2u) * 16u;
    threadgroup bfloat at[32 * 64];
    threadgroup bfloat bt[32 * 64];
    threadgroup float ct[32 * 32];
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint kb = 0; kb < K; kb += 64u) {
        for (uint index = uint(tid); index < 2048u; index += 128u) {
            uint row = index / 64u, k = kb + index % 64u;
            at[index] = mr + row < M && k < K ? A[(mr + row) * K + k] : bfloat(0.0f);
            bt[index] = nc + row < N && k < K ? B[(nc + row) * K + k] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 64u; kk += 8u) {
            simdgroup_matrix<bfloat, 8, 8> a0, a1, b0, b1;
            simdgroup_load(a0, at + sm * 64u + kk, 64);
            simdgroup_load(a1, at + (sm + 8u) * 64u + kk, 64);
            simdgroup_load(b0, bt + sn * 64u + kk, 64, ulong2(0), true);
            simdgroup_load(b1, bt + (sn + 8u) * 64u + kk, 64, ulong2(0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ct + sm * 32u + sn, 32);
    simdgroup_store(c01, ct + sm * 32u + sn + 8u, 32);
    simdgroup_store(c10, ct + (sm + 8u) * 32u + sn, 32);
    simdgroup_store(c11, ct + (sm + 8u) * 32u + sn + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u, col = nc + index % 32u;
        if (row < M && col < N) {
            uint index_out = row * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[index_out]);
            C[index_out] = OUTPUT_TYPE(alpha * ct[index] + prior);
        }
    }
}
"#
    .replace("KERNEL_NAME", name)
    .replace("OUTPUT_TYPE", output)
}

fn tile_source(stage: &str, output: &str, name: &str) -> String {
    r#"
kernel void KERNEL_NAME(
    device const bfloat *A [[buffer(0)]], device const bfloat *B [[buffer(1)]],
    device OUTPUT_TYPE *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint mr=group.x*32u, nc=group.y*64u;
    const uint sm=uint(sg/2u)*16u, sn=uint(sg%2u)*32u;
    threadgroup STAGE_TYPE at[32*32];
    threadgroup STAGE_TYPE bt[64*32];
    threadgroup float ct[32*64];
    simdgroup_float8x8 c00(0.0f),c01(0.0f),c02(0.0f),c03(0.0f);
    simdgroup_float8x8 c10(0.0f),c11(0.0f),c12(0.0f),c13(0.0f);
    for(uint kb=0;kb<K;kb+=32u) {
        for(uint i=uint(tid);i<1024u;i+=128u) {
            uint row=i/32u, k=kb+i%32u;
            at[i]=mr+row<M && k<K ? STAGE_TYPE(A[(mr+row)*K+k]) : STAGE_TYPE(0.0f);
        }
        for(uint i=uint(tid);i<2048u;i+=128u) {
            uint row=i/32u, k=kb+i%32u;
            bt[i]=nc+row<N && k<K ? STAGE_TYPE(B[(nc+row)*K+k]) : STAGE_TYPE(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kk=0;kk<32u;kk+=8u) {
            simdgroup_matrix<STAGE_TYPE,8,8> a0,a1,b0,b1,b2,b3;
            simdgroup_load(a0,at+sm*32u+kk,32);
            simdgroup_load(a1,at+(sm+8u)*32u+kk,32);
            simdgroup_load(b0,bt+sn*32u+kk,32,ulong2(0),true);
            simdgroup_load(b1,bt+(sn+8u)*32u+kk,32,ulong2(0),true);
            simdgroup_load(b2,bt+(sn+16u)*32u+kk,32,ulong2(0),true);
            simdgroup_load(b3,bt+(sn+24u)*32u+kk,32,ulong2(0),true);
            simdgroup_multiply_accumulate(c00,a0,b0,c00);
            simdgroup_multiply_accumulate(c01,a0,b1,c01);
            simdgroup_multiply_accumulate(c02,a0,b2,c02);
            simdgroup_multiply_accumulate(c03,a0,b3,c03);
            simdgroup_multiply_accumulate(c10,a1,b0,c10);
            simdgroup_multiply_accumulate(c11,a1,b1,c11);
            simdgroup_multiply_accumulate(c12,a1,b2,c12);
            simdgroup_multiply_accumulate(c13,a1,b3,c13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00,ct+sm*64u+sn,64);
    simdgroup_store(c01,ct+sm*64u+sn+8u,64);
    simdgroup_store(c02,ct+sm*64u+sn+16u,64);
    simdgroup_store(c03,ct+sm*64u+sn+24u,64);
    simdgroup_store(c10,ct+(sm+8u)*64u+sn,64);
    simdgroup_store(c11,ct+(sm+8u)*64u+sn+8u,64);
    simdgroup_store(c12,ct+(sm+8u)*64u+sn+16u,64);
    simdgroup_store(c13,ct+(sm+8u)*64u+sn+24u,64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=uint(tid);i<2048u;i+=128u) {
        uint row=mr+i/64u, col=nc+i%64u;
        if(row<M && col<N) {
            uint index=row*N+col;
            float prior=beta==0.0f ? 0.0f : beta*float(C[index]);
            C[index]=OUTPUT_TYPE(alpha*ct[i]+prior);
        }
    }
}
"#
    .replace("KERNEL_NAME", name)
    .replace("STAGE_TYPE", stage)
    .replace("OUTPUT_TYPE", output)
}

#[test]
#[ignore = "bounded real-weight BF16 Metal matrix tile comparison; no ANE access"]
fn native_bf16_tile64_checks_both_output_abis_and_operand_paths(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_tile_comparison(false)
}

#[test]
#[ignore = "bounded M84 BF16 reduction-tile comparison; no ANE access"]
fn native_bf16_bk64_checks_precision_and_m84_ffn(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_tile_comparison(true)
}

fn run_tile_comparison(reduction64: bool) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let qualify_only = std::env::var_os("RVLLM_METAL_MMA_TILE_QUALIFY_ONLY").is_some();
    let model =
        PathBuf::from(std::env::var_os("RVLLM_METAL_QKV_MODEL_DIR").ok_or("model required")?);
    let tensors = scan_safetensor_tensors(&model)?;
    let names = if reduction64 {
        vec![
            "qkv_project_f32_mma32",
            "bk64_bf16_f32",
            "gemm_f16_mma32",
            "bk64_bf16_bf16",
        ]
    } else {
        vec![
            "qkv_project_f32_mma32",
            "tile64_bf16_f32",
            "tile64_f32_f32",
            "gemm_f16_mma32",
            "tile64_bf16_bf16",
            "tile64_f32_bf16",
        ]
    };
    let variant_count = names.len() / 2;
    let mut source =
        crate::kernels::kernel_source_for_float_type(MetalFloatType::Bf16).into_owned();
    if reduction64 {
        source.push_str(&reduction64_source("float", names[1]));
        source.push_str(&reduction64_source("bfloat", names[3]));
    } else {
        for (stage, output, name) in [
            ("bfloat", "float", names[1]),
            ("float", "float", names[2]),
            ("bfloat", "bfloat", names[4]),
            ("float", "bfloat", names[5]),
        ] {
            source.push_str(&tile_source(stage, output, name));
        }
    }
    let mut ctx = MetalContext::new()?;
    ctx.compile_library(&source)?;
    let mut pipelines = PipelineCache::new();
    for name in &names {
        pipelines.compile(&ctx, name)?;
    }
    let resources:Vec<_>=names.iter().map(|name|{let p=pipelines.get(name).unwrap();serde_json::json!({"name":name,"static_shared_bytes":p.staticThreadgroupMemoryLength(),"max_threads":p.maxTotalThreadsPerThreadgroup(),"execution_width":p.threadExecutionWidth()})}).collect();
    let mut reports = Vec::new();
    let shapes = if reduction64 {
        vec![
            ("all-tails", 0, 63_u32, 67_u32, 65_u32, vec![]),
            (
                "gate-up",
                0,
                84,
                30720,
                3840,
                vec!["mlp.gate_proj", "mlp.up_proj"],
            ),
            ("down", 0, 84, 3840, 15360, vec!["mlp.down_proj"]),
        ]
    } else {
        vec![
            ("all-tails", 0, 63_u32, 67_u32, 35_u32, vec![]),
            (
                "sliding-qkv",
                0,
                84,
                8192,
                3840,
                vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ),
            (
                "global-qkv",
                5,
                652,
                9216,
                3840,
                vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.k_proj"],
            ),
            (
                "gate-up",
                0,
                652,
                30720,
                3840,
                vec!["mlp.gate_proj", "mlp.up_proj"],
            ),
            (
                "global-output",
                5,
                652,
                3840,
                8192,
                vec!["self_attn.o_proj"],
            ),
            ("down", 0, 1024, 3840, 15360, vec!["mlp.down_proj"]),
        ]
    };
    for (label, layer, m, n, k, parts) in shapes {
        let mut weights = Vec::new();
        for part in parts {
            let entry = tensors
                .get(&format!(
                    "model.language_model.layers.{layer}.{part}.weight"
                ))
                .ok_or("weight missing")?;
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
        let elements = m as usize * n as usize;
        let mut arena = MetalBufferArena::new(
            ctx.device(),
            weights.len() + input.len() + elements * 18 + 4096,
        )?;
        let mut upload = |name: &str, bytes: &[u8]| -> Result<MetalRegion> {
            let region = arena.region(name, bytes.len(), 32)?;
            // SAFETY: initialization occurs before any submission, with exactly
            // the allocated size and no overlapping regions.
            unsafe {
                arena.write_region(&region, bytes)?;
            }
            Ok(region)
        };
        let a = upload("A", &input)?;
        let b = upload("B", &weights)?;
        let mut outputs = Vec::new();
        for (path, name) in names.iter().enumerate() {
            let bytes = elements * if path < variant_count { 4 } else { 2 };
            let mut poisoned = vec![0xA5; bytes + 64];
            poisoned[32..32 + bytes].fill(0xFF);
            outputs.push(upload(name, &poisoned)?);
        }
        drop(upload);
        let run = |path: usize| -> std::result::Result<(f64, f64), Box<dyn std::error::Error>> {
            let timer = std::time::Instant::now();
            let command = ctx.queue().commandBuffer().ok_or("command missing")?;
            let encoder = command.computeCommandEncoder().ok_or("encoder missing")?;
            encoder.setComputePipelineState(pipelines.get(names[path])?);
            // SAFETY: live exact-size inputs, independently poisoned guarded
            // outputs, and checked shape products. Both tile implementations
            // mask every M/N/K tail and beta=0 must not read output poison.
            unsafe {
                for (i, offset) in [a.offset, b.offset, outputs[path].offset + 32]
                    .into_iter()
                    .enumerate()
                {
                    encoder.setBuffer_offset_atIndex(Some(arena.buffer_retained()), offset, i);
                }
                for (i, value) in [m, n, k].iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::from(value).cast(),
                        4,
                        i + 3,
                    );
                }
                for (i, value) in [1.0_f32, 0.0_f32].iter().enumerate() {
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::from(value).cast(),
                        4,
                        i + 6,
                    );
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (m as usize).div_ceil(32),
                    height: (n as usize).div_ceil(if reduction64 || path % variant_count == 0 {
                        32
                    } else {
                        64
                    }),
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return Err(format!("matrix GPU: {error}").into());
            }
            Ok((
                (command.GPUEndTime() - command.GPUStartTime()) * 1000.0,
                timer.elapsed().as_secs_f64() * 1000.0,
            ))
        };
        for path in 0..names.len() {
            run(path)?;
        }
        let read = |path: usize| -> Vec<f32> {
            let region = &outputs[path];
            // SAFETY: every submission above completed synchronously; bounded
            // read of each guarded allocation with no outstanding GPU use.
            let bytes = unsafe { std::slice::from_raw_parts(arena.host_ptr(region), region.size) };
            assert!(bytes[..32]
                .iter()
                .chain(bytes[bytes.len() - 32..].iter())
                .all(|&b| b == 0xA5));
            let bytes = &bytes[32..bytes.len() - 32];
            if path < variant_count {
                bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect()
            } else {
                bytes
                    .chunks_exact(2)
                    .map(|v| bf16::from_le_bytes(v.try_into().unwrap()).to_f32())
                    .collect()
            }
        };
        let float_outputs: Vec<_> = (0..variant_count).map(read).collect();
        assert!(float_outputs.iter().flatten().all(|x| x.is_finite()));
        let baseline = &float_outputs[0];
        let mut l2 = Vec::new();
        let mut cpu_max = Vec::new();
        for path in 0..variant_count {
            let output = &float_outputs[path];
            let packed = read(path + variant_count);
            if reduction64 {
                assert!(
                    baseline
                        .iter()
                        .zip(output)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label} BK64 must preserve every FP32 accumulator result"
                );
            }
            for (&value, &actual) in output.iter().zip(&packed) {
                assert_eq!(
                    bf16::from_f32(value).to_f32().to_bits(),
                    actual.to_bits(),
                    "{label} variant{path} must round exactly once"
                );
            }
            let relative = (baseline
                .iter()
                .zip(output)
                .map(|(&a, &b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / baseline.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>())
            .sqrt();
            assert!(relative < 0.0001, "{label} path{path} L2 {relative}");
            l2.push(relative);
            let bf = |bytes: &[u8], i: usize| {
                bf16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap()).to_f64()
            };
            let mut max_error = 0.0_f64;
            for row in [0, 15, 16, 31, 32, m - 1].into_iter().filter(|&r| r < m) {
                for col in [0, 31, 32, 63, 64, n - 1].into_iter().filter(|&c| c < n) {
                    let expected = (0..k)
                        .map(|i| {
                            bf(&input, (row * k + i) as usize)
                                * bf(&weights, (col * k + i) as usize)
                        })
                        .sum::<f64>();
                    let actual = f64::from(output[(row * n + col) as usize]);
                    max_error = max_error.max((expected - actual).abs());
                    assert!(
                        (expected - actual).abs() < 0.005 + expected.abs() * 0.0001,
                        "{label} path{path} [{row},{col}]: {actual} vs {expected}"
                    );
                }
            }
            cpu_max.push(max_error);
        }
        let expected_outputs: Vec<_> = (0..names.len()).map(read).collect();
        drop(float_outputs);
        drop((input, weights));
        let mut gpu = vec![Vec::new(); names.len()];
        let mut wall = vec![Vec::new(); names.len()];
        let order: Vec<usize> = if qualify_only {
            Vec::new()
        } else if reduction64 {
            (0..3)
                .flat_map(|block| {
                    (0..2).flat_map(move |abi| {
                        let order = if block % 2 == 0 {
                            [0, 1, 1, 0]
                        } else {
                            [1, 0, 0, 1]
                        };
                        order.map(|variant| abi * 2 + variant)
                    })
                })
                .collect()
        } else {
            (0..6)
                .flat_map(|iteration| (0..names.len()).map(move |delta| (iteration + delta) % 6))
                .collect()
        };
        for &path in &order {
            let (g, w) = run(path)?;
            if !g.is_finite() || g <= 0.0 || !w.is_finite() || w <= 0.0 {
                return Err("invalid matrix timing sample".into());
            }
            gpu[path].push(g);
            wall[path].push(w);
        }
        // Check every guard and result again after repeated submissions, not
        // only before timing. Signed zero remains part of bit identity.
        for (path, expected) in expected_outputs.iter().enumerate() {
            let actual = read(path);
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{label} variant{path} changed after repeated use"
            );
        }
        let median = |v: &[f64]| {
            if v.is_empty() {
                return None;
            }
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            Some((v[2] + v[3]) * 0.5)
        };
        let medians: Vec<_> = gpu.iter().map(|v| median(v)).collect();
        reports.push(serde_json::json!({"projection":label,"m":m,"n":n,"k":k,"relative_l2_vs_mma32":l2,"sampled_fp64_max_abs":cpu_max,"bf16_rounding_exact":true,"fp32_bit_parity_required":reduction64,"guards_intact":true,"commands":names.len()+order.len(),"trial_order":order,"gpu_ms":gpu,"wall_ms":wall,"gpu_median_ms":medians}));
    }
    let report = serde_json::json!({"schema":"rvllm.metal_tile_comparison.v2","qualification_only":qualify_only,"reduction64":reduction64,"variants":names,"pipeline_resources":resources,"cases":reports,"scope":"Real checkpoint weights and synthetic BF16 inputs including values outside FP16 range; FP32 and once-rounded BF16 outputs; isolated tile comparison. No production routing or ANE calls. Timing requires an independently eligible experiment-queue receipt."});
    if let Some(path) = std::env::var_os("RVLLM_METAL_MMA_TILE_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
