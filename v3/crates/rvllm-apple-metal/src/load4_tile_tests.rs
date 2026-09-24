//! Explicit, ignored device qualification of the actual load4 tile entry points.
//! No ANE calls, timing, automatic candidate sweep, or production promotion.
use crate::arena::MetalRegion;
use crate::research_projection::{load4_tile, validate_fixture_bytes, ProjectionRequest};
use crate::weight_loader::{load_safetensor_entry_bf16, scan_safetensor_tensors};
use crate::{
    MetalBufferArena, MetalContext, MetalFloatType, MetalKernelOptions, MetalResearchCandidate,
    PipelineCache,
};
use half::bf16;
use objc2_metal::*;
use std::path::{Path, PathBuf};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Case {
    label: &'static str,
    layer: usize,
    shape: [u32; 3],
    parts: &'static [&'static str],
    // Independently enumerated positive roles, not inferred from a dispatch count.
    qkv: bool,
    gemm: bool,
}

fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create_new(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[test]
#[ignore = "explicit real-weight BF16 load4 tile oracle; requires Metal and a fresh report directory"]
fn native_load4_tile_preserves_projection_contracts() -> TestResult {
    let candidate: MetalResearchCandidate =
        std::env::var("RVLLM_METAL_LOAD4_TILE_CANDIDATE")?.parse()?;
    let tile = load4_tile(candidate).ok_or("one of the seven load4 tile candidates is required")?;
    let directory = PathBuf::from(
        std::env::var_os("RVLLM_METAL_LOAD4_REPORT_DIR").ok_or("report directory required")?,
    );
    if !directory.is_absolute() || std::fs::symlink_metadata(&directory).is_ok() {
        return Err("report directory must be absolute and absent".into());
    }
    std::fs::create_dir(&directory)?;
    let model =
        PathBuf::from(std::env::var_os("RVLLM_METAL_QKV_MODEL_DIR").ok_or("model required")?);
    let tensors = scan_safetensor_tensors(&model)?;
    let source = crate::kernels::kernel_source_with_options(
        MetalFloatType::Bf16,
        MetalKernelOptions {
            research: candidate,
            ..MetalKernelOptions::default()
        },
    );
    write_new(&directory.join("source.metal"), source.as_bytes())?;
    let mut context = MetalContext::new()?;
    context.compile_library(&source)?;
    let kernels = candidate.kernels();
    let names = [
        "qkv_project_f32_mma32",
        kernels[1].name(),
        "gemm_f16_mma32",
        kernels[0].name(),
    ];
    let mut pipelines = PipelineCache::new();
    let mut resources = Vec::new();
    for (path, name) in names.iter().enumerate() {
        pipelines.compile(&context, name)?;
        let pso = pipelines.get(name)?;
        let (threads, shared) = if path % 2 == 0 {
            (128, 0)
        } else {
            kernels[0].limits()
        };
        if !crate::research::launch_fits(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            pso.staticThreadgroupMemoryLength(),
            context.device().maxThreadgroupMemoryLength(),
            threads,
            shared,
        ) {
            return Err(format!("unsupported PSO resources: {name}").into());
        }
        resources.push(serde_json::json!({"kernel":name,"threads":threads,
            "source_shared_bytes":shared,"static_shared_bytes":pso.staticThreadgroupMemoryLength(),
            "execution_width":pso.threadExecutionWidth(),"maximum_threads":pso.maxTotalThreadsPerThreadgroup()}));
    }
    write_new(
        &directory.join("resources.json"),
        &serde_json::to_vec_pretty(&resources)?,
    )?;
    let short_m = if candidate.spec().min_tokens == 64 {
        65
    } else {
        21
    };
    let qkv = &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"];
    let cases = [
        Case {
            label: "sliding-qkv",
            layer: 0,
            shape: [short_m, 8192, 3840],
            parts: qkv,
            qkv: true,
            gemm: false,
        },
        // Gemma's global K is also the shared V source, as in the incumbent oracle.
        Case {
            label: "global-qkv",
            layer: 5,
            shape: [652, 9216, 3840],
            parts: &["self_attn.q_proj", "self_attn.k_proj", "self_attn.k_proj"],
            qkv: true,
            gemm: false,
        },
        Case {
            label: "gate-up",
            layer: 0,
            shape: [652, 30720, 3840],
            parts: &["mlp.gate_proj", "mlp.up_proj"],
            qkv: false,
            gemm: true,
        },
        Case {
            label: "sliding-output",
            layer: 0,
            shape: [84, 3840, 4096],
            parts: &["self_attn.o_proj"],
            qkv: false,
            gemm: true,
        },
        Case {
            label: "global-output",
            layer: 5,
            shape: [652, 3840, 8192],
            parts: &["self_attn.o_proj"],
            qkv: false,
            gemm: true,
        },
        Case {
            label: "down",
            layer: 0,
            shape: [1024, 3840, 15360],
            parts: &["mlp.down_proj"],
            qkv: false,
            gemm: true,
        },
        Case {
            label: "unsupported-n-k",
            layer: 0,
            shape: [63, 67, 35],
            parts: &[],
            qkv: false,
            gemm: false,
        },
        Case {
            label: "below-minimum-m",
            layer: 0,
            shape: [candidate.spec().min_tokens - 1, 8192, 3840],
            parts: qkv,
            qkv: false,
            gemm: false,
        },
    ];
    let mut reports = Vec::new();
    let mut positive = [0_usize; 2]; // GEMM, QKV; baseline work never counts.
    let mut refusals = 0;
    for (case_index, case) in cases.iter().enumerate() {
        let [m, n, k] = case.shape;
        let case_dir = directory.join(format!("{case_index:02}-{}", case.label));
        std::fs::create_dir(&case_dir)?;
        let numerical = [true, case.qkv, true, case.gemm];
        let mut weights = Vec::new();
        for part in case.parts {
            let name = format!("model.language_model.layers.{}.{part}.weight", case.layer);
            let entry = tensors
                .get(&name)
                .ok_or_else(|| format!("missing weight: {name}"))?;
            if entry.shape.len() != 2 || entry.shape[1] != k as usize {
                return Err(format!("incorrect checkpoint shape: {name}").into());
            }
            weights.extend(load_safetensor_entry_bf16(entry)?);
        }
        if case.parts.is_empty() {
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
                let value = (((col * 7 + row * 13) % 47) as f32 / 32.0 - 0.75) * scale;
                input.extend_from_slice(&bf16::from_f32(value).to_le_bytes());
            }
        }
        let elements = m as usize * n as usize;
        let mut arena = MetalBufferArena::new(
            context.device(),
            input.len() + weights.len() + elements * 12 + 4096,
        )?;
        let mut upload = |name: &str, bytes: &[u8]| -> TestResult<MetalRegion> {
            let region = arena.region(name, bytes.len(), 32)?;
            // SAFETY: exact-size, disjoint allocations initialized before any submission.
            unsafe {
                arena.write_region(&region, bytes)?;
            }
            Ok(region)
        };
        let a = upload("A", &input)?;
        let b = upload("B", &weights)?;
        let mut outputs = Vec::new();
        let mut poisoned_outputs = Vec::new();
        for (path, name) in names.iter().enumerate() {
            let bytes = elements * if path < 2 { 4 } else { 2 };
            let mut raw = vec![0xa5; bytes + 64];
            raw[32..bytes + 32].fill(0xff);
            outputs.push(upload(name, &raw)?);
            poisoned_outputs.push(raw);
        }
        drop(upload);
        for path in [1, 3] {
            let request = ProjectionRequest {
                candidate,
                full_prefill: true,
                native_bf16: true,
                alpha: 1.0,
                beta: 0.0,
                shape: case.shape,
                output_f32: path == 1,
                offsets: [a.offset, b.offset, outputs[path].offset + 32],
                arena_bytes: arena.buffer_retained().length(),
            };
            assert_eq!(
                request.plan().is_ok(),
                numerical[path],
                "{} host admission",
                case.label
            );
        }
        let read = |path: usize| -> Vec<u8> {
            let region = &outputs[path];
            // SAFETY: callers only read after the relevant command has completed.
            unsafe { std::slice::from_raw_parts(arena.host_ptr(region), region.size) }.to_vec()
        };
        let poison = |path: usize| -> TestResult {
            // SAFETY: callers only overwrite a live, exact-size output region after
            // the preceding command has completed and before the next submission.
            unsafe {
                arena.write_region(&outputs[path], &poisoned_outputs[path])?;
            }
            Ok(())
        };
        let run = |path: usize| -> TestResult {
            let command = context
                .queue()
                .commandBuffer()
                .ok_or("command buffer missing")?;
            let encoder = command.computeCommandEncoder().ok_or("encoder missing")?;
            encoder.setComputePipelineState(pipelines.get(names[path])?);
            // SAFETY: live aligned inputs and independently guarded output allocations;
            // dimensions are the bounded fixtures above. Unsupported candidate roles
            // return uniformly before all barriers. Submit and wait before any host read.
            unsafe {
                for (index, offset) in [a.offset, b.offset, outputs[path].offset + 32]
                    .into_iter()
                    .enumerate()
                {
                    encoder.setBuffer_offset_atIndex(Some(arena.buffer_retained()), offset, index);
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
            let (tm, tn, threads) = if path % 2 == 0 {
                (32, 32, 128)
            } else {
                (tile.m, tile.n, kernels[0].limits().0)
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (m as usize).div_ceil(tm),
                    height: (n as usize).div_ceil(tn),
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return Err(format!("{} {}: {error}", case.label, names[path]).into());
            }
            Ok(())
        };
        for path in 0..4 {
            if let Err(error) = run(path) {
                let bytes = read(path);
                write_new(
                    &case_dir.join(format!("command-error-{path}.guarded.bin")),
                    &bytes,
                )?;
                write_new(
                    &case_dir.join(format!("command-error-{path}.json")),
                    &serde_json::to_vec_pretty(&serde_json::json!({
                        "status":"command-error",
                        "shape":case.shape,
                        "kernel":names[path],
                        "error":error.to_string()
                    }))?,
                )?;
                return Err(error);
            }
        }
        let raw: Vec<_> = (0..4).map(read).collect();
        // Preserve diagnostic bytes even when a subsequent numerical assertion fails.
        for (path, bytes) in raw.iter().enumerate() {
            write_new(&case_dir.join(format!("output-{path}.guarded.bin")), bytes)?;
        }
        write_new(
            &case_dir.join("capture.json"),
            &serde_json::to_vec_pretty(&serde_json::json!({
                "status":"captured-before-oracle","shape":case.shape,"kernels":names,"numerical":numerical
            }))?,
        )?;
        for path in 0..4 {
            validate_fixture_bytes(&raw[path], m, n, path < 2, numerical[path])
                .map_err(|error| format!("{} {}: {error}", case.label, names[path]))?;
        }
        let baseline = &raw[0][32..raw[0].len() - 32];
        if case.qkv {
            assert_eq!(
                &raw[1][32..raw[1].len() - 32],
                baseline,
                "{} FP32 bits",
                case.label
            );
        }
        let mut rounded = Vec::with_capacity(elements * 2);
        for value in baseline.chunks_exact(4) {
            rounded.extend_from_slice(
                &bf16::from_f32(f32::from_le_bytes(value.try_into()?)).to_le_bytes(),
            );
        }
        assert_eq!(
            &raw[2][32..raw[2].len() - 32],
            rounded.as_slice(),
            "{} baseline storage",
            case.label
        );
        if case.gemm {
            assert_eq!(
                &raw[3][32..raw[3].len() - 32],
                rounded.as_slice(),
                "{} BF16 bits",
                case.label
            );
        }
        let bf = |bytes: &[u8], index: usize| {
            bf16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]).to_f64()
        };
        let mut maximum_cpu_error = 0.0_f64;
        for row in [0, 15, 16, 31, 32, m - 1].into_iter().filter(|&r| r < m) {
            for col in [0, 31, 32, 63, 64, n - 1].into_iter().filter(|&c| c < n) {
                let expected = (0..k)
                    .map(|i| {
                        bf(&input, (row * k + i) as usize) * bf(&weights, (col * k + i) as usize)
                    })
                    .sum::<f64>();
                let index = (row * n + col) as usize * 4;
                let actual = f64::from(f32::from_le_bytes(baseline[index..index + 4].try_into()?));
                maximum_cpu_error = maximum_cpu_error.max((actual - expected).abs());
                assert!(
                    (actual - expected).abs() < 0.005 + expected.abs() * 0.0001,
                    "{} CPU [{row},{col}]: {actual} vs {expected}",
                    case.label
                );
            }
        }
        for (repeat, path) in [3, 2, 1, 0, 0, 1, 2, 3].into_iter().enumerate() {
            poison(path)?;
            run(path)?;
            let repeated = read(path);
            write_new(
                &case_dir.join(format!("repeat-{repeat:02}-path-{path}.guarded.bin")),
                &repeated,
            )?;
            validate_fixture_bytes(&repeated, m, n, path < 2, numerical[path]).map_err(
                |error| {
                    format!(
                        "{} repeated invocation {repeat} {}: {error}",
                        case.label, names[path]
                    )
                },
            )?;
            assert_eq!(
                &repeated, &raw[path],
                "{} repeated invocation {repeat} {}",
                case.label, names[path]
            );
        }
        positive[0] += usize::from(case.gemm);
        positive[1] += usize::from(case.qkv);
        refusals += usize::from(!case.gemm) + usize::from(!case.qkv);
        reports.push(
            serde_json::json!({"case":case.label,"shape":case.shape,"numerical":numerical,
            "fp32_bits_exact":case.qkv,"bf16_bits_exact":case.gemm,"guards_intact":true,
            "sampled_fp64_max_abs":maximum_cpu_error,"commands_completed":12}),
        );
    }
    assert_eq!(positive, [4, 2]);
    assert_eq!(refusals, 10);
    assert_eq!(reports.len(), 8);
    let report = serde_json::json!({"schema":"rvllm.load4-tile-oracle.v1","status":"component-qualified",
        "candidate":candidate.name(),"numerical_gemm":positive[0],"numerical_qkv":positive[1],
        "guard_refusals":refusals,"cases":reports,"timing_performed":false,
        "scope":"Real checkpoint weights, synthetic BF16 inputs including values above FP16 range, exact baseline bits, independent sampled FP64 dot products, poison/canaries and repeated reuse. Not full-model acceptance or speed evidence."});
    write_new(
        &directory.join("qualification.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
