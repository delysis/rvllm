//! Isolated tile/operand experiment; no production dispatch selects these probes.
use super::*;
use crate::arena::MetalRegion;
use crate::research_projection::{prefetch_fixture_expectation, validate_fixture_bytes};
use crate::weight_loader::{load_safetensor_entry_bf16, scan_safetensor_tensors};
use crate::MetalFloatType;
use half::bf16;
use objc2_metal::{MTLComputePipelineState, MTLDevice};
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

// The BF16 tile is now the same source used by the explicit runtime route.
// Keep only the independent FP32-operand 32x64 control in this older fixture.
fn float_tile64_control_source(output: &str, name: &str) -> String {
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
    .replace("STAGE_TYPE", "float")
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

#[test]
#[ignore = "explicit real-weight Metal research FP32-operand oracle; no ANE access"]
fn native_fp32_operands_preserve_existing_oracle_gates(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_matrix_comparison(false, Some(crate::MetalResearchCandidate::Mma32F32))
}

#[test]
#[ignore = "explicit real-weight Metal research vector-load oracle; no ANE access"]
fn native_vector_loads_preserve_existing_oracle_gates(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_matrix_comparison(false, Some(crate::MetalResearchCandidate::Mma32Load4))
}

#[test]
#[ignore = "explicit real-weight short-tile component oracle; no ANE access"]
fn native_short_tile_preserves_existing_oracle_gates(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_matrix_comparison(false, Some(crate::MetalResearchCandidate::ShortMma16x64))
}

#[test]
#[ignore = "explicit real-weight prefetch component oracle; no ANE access"]
fn native_prefetch_preserves_existing_oracle_gates(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_matrix_comparison(false, Some(crate::MetalResearchCandidate::Mma32Prefetch))
}

fn run_tile_comparison(reduction64: bool) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_matrix_comparison(reduction64, None)
}

fn run_matrix_comparison(
    reduction64: bool,
    candidate: Option<crate::MetalResearchCandidate>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // New fixtures are numerical-only. Timing belongs to a separately admitted
    // campaign, not to incidental execution of an ignored component test.
    let qualify_only =
        candidate.is_some() || std::env::var_os("RVLLM_METAL_MMA_TILE_QUALIFY_ONLY").is_some();
    let require_fp32_bits = reduction64
        || candidate == Some(crate::MetalResearchCandidate::Mma32Load4)
        || candidate == Some(crate::MetalResearchCandidate::Mma32Prefetch);
    let prefetch = candidate == Some(crate::MetalResearchCandidate::Mma32Prefetch);
    // A fresh companion directory survives an assertion/error without emitting
    // a success receipt. No shader source, admission guard or tolerance changes.
    let evidence_dir = if prefetch {
        let report = PathBuf::from(
            std::env::var_os("RVLLM_METAL_MMA_TILE_REPORT")
                .ok_or("prefetch qualification requires a fresh absolute report path")?,
        );
        if !report.is_absolute() || std::fs::symlink_metadata(&report).is_ok() {
            return Err("prefetch report path must be absolute and absent".into());
        }
        let mut companion = report.as_os_str().to_os_string();
        companion.push(".artifacts");
        let directory = PathBuf::from(companion);
        std::fs::create_dir(&directory)?;
        Some(directory)
    } else {
        None
    };
    let model =
        PathBuf::from(std::env::var_os("RVLLM_METAL_QKV_MODEL_DIR").ok_or("model required")?);
    let tensors = scan_safetensor_tensors(&model)?;
    let names = if let Some(kind) = candidate {
        let kernels = kind.kernels();
        vec![
            "qkv_project_f32_mma32",
            kernels[1].name(),
            "gemm_f16_mma32",
            kernels[0].name(),
        ]
    } else if reduction64 {
        vec![
            "qkv_project_f32_mma32",
            "bk64_bf16_f32",
            "gemm_f16_mma32",
            "bk64_bf16_bf16",
        ]
    } else {
        vec![
            "qkv_project_f32_mma32",
            "wave2_qkv_mma32x64",
            "tile64_f32_f32",
            "gemm_f16_mma32",
            "wave2_gemm_mma32x64",
            "tile64_f32_bf16",
        ]
    };
    let variant_count = names.len() / 2;
    let selected = if reduction64 {
        crate::MetalResearchCandidate::Off
    } else {
        candidate.unwrap_or(crate::MetalResearchCandidate::LongMma32x64)
    };
    let mut source = crate::kernels::kernel_source_with_options(
        MetalFloatType::Bf16,
        crate::MetalKernelOptions {
            research: selected,
            ..crate::MetalKernelOptions::default()
        },
    )
    .into_owned();
    if reduction64 {
        source.push_str(&reduction64_source("float", names[1]));
        source.push_str(&reduction64_source("bfloat", names[3]));
    } else if candidate.is_none() {
        // Preserve the old 32x64 FP32-operand control; native BF16 operands
        // are no longer duplicated under a test-only source/name.
        source.push_str(&float_tile64_control_source("float", names[2]));
        source.push_str(&float_tile64_control_source("bfloat", names[5]));
    }
    if let Some(directory) = &evidence_dir {
        write_matrix_artifact(&directory.join("source.metal"), source.as_bytes())?;
    }
    let mut ctx = MetalContext::new()?;
    ctx.compile_library(&source)?;
    let mut pipelines = PipelineCache::new();
    for name in &names {
        pipelines.compile(&ctx, name)?;
        let pso = pipelines.get(name)?;
        let planned = selected
            .kernels()
            .iter()
            .find(|kernel| kernel.name() == *name)
            .map(|kernel| kernel.limits().1)
            .unwrap_or(0);
        if planned > ctx.device().maxThreadgroupMemoryLength()
            || pso.threadExecutionWidth() != 32
            || pso.maxTotalThreadsPerThreadgroup() < 128
            || pso.staticThreadgroupMemoryLength() > ctx.device().maxThreadgroupMemoryLength()
        {
            return Err(format!("unsupported matrix fixture PSO limits: {name}").into());
        }
    }
    let resources:Vec<_>=names.iter().map(|name|{let p=pipelines.get(name).unwrap();serde_json::json!({"name":name,"static_shared_bytes":p.staticThreadgroupMemoryLength(),"max_threads":p.maxTotalThreadsPerThreadgroup(),"execution_width":p.threadExecutionWidth()})}).collect();
    let mut reports = Vec::new();
    let vector_load = candidate == Some(crate::MetalResearchCandidate::Mma32Load4);
    let short = candidate == Some(crate::MetalResearchCandidate::ShortMma16x64);
    let small_m = if short { 6 } else { 84 };
    let large_m = if short { 63 } else { 652 };
    let shapes = if reduction64 {
        vec![
            ("all-tails", 0, 63_u32, 67_u32, 65_u32, vec![]),
            (
                "gate-up",
                0,
                small_m,
                30720,
                3840,
                vec!["mlp.gate_proj", "mlp.up_proj"],
            ),
            ("down", 0, 84, 3840, 15360, vec!["mlp.down_proj"]),
        ]
    } else {
        vec![
            // Vector loading admits aligned N/K only. Its policy tests reject
            // other N/K; this actual execution still exercises a partial M tile.
            (
                "all-tails",
                0,
                63_u32,
                if vector_load { 64_u32 } else { 67_u32 },
                if vector_load { 64_u32 } else { 35_u32 },
                vec![],
            ),
            (
                "sliding-qkv",
                0,
                small_m,
                8192,
                3840,
                vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ),
            (
                "global-qkv",
                5,
                large_m,
                9216,
                3840,
                vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.k_proj"],
            ),
            (
                "gate-up",
                0,
                large_m,
                30720,
                3840,
                vec!["mlp.gate_proj", "mlp.up_proj"],
            ),
            (
                "global-output",
                5,
                large_m,
                3840,
                8192,
                vec!["self_attn.o_proj"],
            ),
            (
                "down",
                0,
                if short { 64 } else { 1024 },
                3840,
                15360,
                vec!["mlp.down_proj"],
            ),
        ]
    };
    let mut prefetch_positive = [0_usize; 2]; // GEMM, QKV; exclude all refusals.
    let mut prefetch_refusals = 0_usize;
    for (case_index, (label, layer, m, n, k, parts)) in shapes.into_iter().enumerate() {
        let numerical = names
            .iter()
            .map(|name| {
                if prefetch {
                    prefetch_fixture_expectation(name, [m, n, k])
                } else {
                    Ok(true)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let case_dir = evidence_dir
            .as_ref()
            .map(|directory| directory.join(format!("case-{case_index:02}-{label}")));
        if let Some(directory) = &case_dir {
            std::fs::create_dir(directory)?;
        }
        eprintln!("matrix case={label} M={m} N={n} K={k} numerical={numerical:?}");
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
                    width: (m as usize).div_ceil(if path % variant_count == 0 || reduction64 {
                        32
                    } else {
                        crate::research_projection::projection_tile(selected)
                            .ok_or("not a matrix candidate")?
                            .0
                    }),
                    height: (n as usize).div_ceil(if path % variant_count == 0 || reduction64 {
                        32
                    } else {
                        crate::research_projection::projection_tile(selected)
                            .ok_or("not a matrix candidate")?
                            .1
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
            eprintln!(
                "matrix case={label} kernel={} output_f32={} expected={}",
                names[path],
                path < variant_count,
                if numerical[path] {
                    "numerical"
                } else {
                    "guard-refusal"
                }
            );
            run(path).map_err(|error| format!("{label} {}: {error}", names[path]))?;
        }
        let read_raw = |path: usize| -> Vec<u8> {
            let region = &outputs[path];
            // SAFETY: every submission above completed synchronously; bounded
            // read of each guarded allocation with no outstanding GPU use.
            unsafe { std::slice::from_raw_parts(arena.host_ptr(region), region.size) }.to_vec()
        };
        // Capture all raw outputs before any finite/guard/numerical assertion.
        // The all-tails and wrong-output-role dispatches are retained as explicit
        // negative controls; they must leave all output poison untouched.
        let mut validation = Vec::new();
        let mut observations = Vec::new();
        for path in 0..names.len() {
            let bytes = read_raw(path);
            let file = format!("output-{path}.guarded.bin");
            if let Some(directory) = &case_dir {
                write_matrix_artifact(&directory.join(&file), &bytes)?;
            }
            let result =
                validate_fixture_bytes(&bytes, m, n, path < variant_count, numerical[path]);
            observations.push(serde_json::json!({"kernel":names[path],"output_f32":path < variant_count,
                "expected":if numerical[path] {"numerical"} else {"guard-refusal"},
                "raw_file":case_dir.as_ref().map(|_| &file),"guard_bytes_each_end":32,"bytes":bytes.len(),"validation_error":result.as_ref().err()}));
            validation.push(result);
        }
        if let Some(directory) = &case_dir {
            let capture = serde_json::json!({"schema":"rvllm.matrix-component.capture.v1",
                "status":"captured-before-numerical-oracle","candidate":selected.name(),
                "case":label,"shape":[m,n,k],"commands_completed":names.len(),"outputs":observations});
            write_matrix_artifact(
                &directory.join("capture.json"),
                &serde_json::to_vec_pretty(&capture)?,
            )?;
        }
        for (path, result) in validation.into_iter().enumerate() {
            result.map_err(|error| format!("{label} {}: {error}", names[path]))?;
        }
        let read = |path: usize| -> Vec<f32> {
            let bytes = read_raw(path);
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
        let baseline = &float_outputs[0];
        let mut l2 = Vec::new();
        let mut cpu_max = Vec::new();
        for path in 0..variant_count {
            let output = &float_outputs[path];
            let packed = read(path + variant_count);
            if require_fp32_bits && numerical[path] {
                assert!(
                    baseline
                        .iter()
                        .zip(output)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label} layout-only variant must preserve every FP32 accumulator result"
                );
            }
            if numerical[path + variant_count] {
                // Prefetch has no FP32 entry for GEMM shapes. The unchanged
                // baseline FP32 oracle is the exact control for these stored
                // results; do not invent a second, test-only candidate shader.
                let rounding_reference = if numerical[path] { output } else { baseline };
                for (&value, &actual) in rounding_reference.iter().zip(&packed) {
                    assert_eq!(
                        bf16::from_f32(value).to_f32().to_bits(),
                        actual.to_bits(),
                        "{label} variant{path} must round exactly once"
                    );
                }
            }
            if !numerical[path] {
                l2.push(None);
                cpu_max.push(None);
                continue; // Already verified *all* refusal bytes; not a numerical pass.
            }
            let relative = (baseline
                .iter()
                .zip(output)
                .map(|(&a, &b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / baseline.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>())
            .sqrt();
            assert!(relative < 0.0001, "{label} path{path} L2 {relative}");
            l2.push(Some(relative));
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
            cpu_max.push(Some(max_error));
        }
        if prefetch {
            // The exact two candidate positions come from the declared pair
            // above, not from a positive total that could count baseline work.
            prefetch_positive[0] += usize::from(numerical[3]);
            prefetch_positive[1] += usize::from(numerical[1]);
            prefetch_refusals += usize::from(!numerical[1]) + usize::from(!numerical[3]);
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
        reports.push(serde_json::json!({"projection":label,"m":m,"n":n,"k":k,"relative_l2_vs_mma32":l2,"sampled_fp64_max_abs":cpu_max,"bf16_rounding_exact":true,"rounding_reference":if prefetch { "candidate FP32 where admitted; otherwise qualified baseline FP32; refused stored outputs excluded" } else { "corresponding FP32 entry" },"fp32_bit_parity_required":require_fp32_bits,"guards_intact":true,"output_expectations":observations,"commands":names.len()+order.len(),"trial_order":order,"gpu_ms":gpu,"wall_ms":wall,"gpu_median_ms":medians}));
    }
    if prefetch && (prefetch_positive != [3, 2] || prefetch_refusals != 7 || reports.len() != 6) {
        return Err(
            "incomplete prefetch role coverage; guard refusals cannot replace numerical work"
                .into(),
        );
    }
    let report = serde_json::json!({"schema":if prefetch { "rvllm.metal_tile_comparison.v3" } else { "rvllm.metal_tile_comparison.v2" },"prefetch_numerical_gemm":prefetch_positive[0],"prefetch_numerical_qkv":prefetch_positive[1],"prefetch_guard_refusals":prefetch_refusals,"qualification_only":qualify_only,"reduction64":reduction64,"candidate":candidate.map(|kind| kind.name()),"variants":names,"pipeline_resources":resources,"cases":reports,"scope":"Real checkpoint weights and synthetic BF16 inputs including values outside FP16 range; FP32 and once-rounded BF16 outputs; isolated tile comparison. No production routing or ANE calls. Timing requires an independently eligible experiment-queue receipt."});
    if let Some(path) = std::env::var_os("RVLLM_METAL_MMA_TILE_REPORT") {
        let bytes = serde_json::to_vec_pretty(&report)?;
        if candidate.is_some() {
            use std::io::Write;
            let mut file = std::fs::File::create_new(path)?;
            file.write_all(&bytes)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
        } else {
            std::fs::write(path, bytes)?;
        }
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

// Safe host artifact writing after GPU collection; never overwrite a previous
// attempt, including an incomplete capture. Failure remains a fixture failure.
fn write_matrix_artifact(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create_new(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}
