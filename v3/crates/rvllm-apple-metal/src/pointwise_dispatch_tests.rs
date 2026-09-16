//! Actual GPU comparison of launch geometry; shaders and arithmetic are identical.
use super::*;
use crate::arena::MetalRegion;
use crate::MetalFloatType;
use half::{bf16, f16};

#[derive(Clone, Copy, Debug)]
enum Operation {
    Gelu,
    Residual,
    ResidualThenScale,
    Scale,
}

#[test]
#[ignore = "bounded public-Metal pointwise dispatch comparison; no ANE access"]
fn full_groups_preserve_pointwise_bits_and_measure_gpu_time(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut reports = Vec::new();
    for dtype in [MetalFloatType::F16, MetalFloatType::Bf16] {
        let mut ctx = MetalContext::new()?;
        ctx.compile_library(&crate::kernels::kernel_source_for_float_type(dtype))?;
        let mut pipelines = PipelineCache::new();
        for name in [
            "gelu_mul_f16",
            "residual_add_f16",
            "residual_add_then_scale_f16",
            "layer_scale_f16",
        ] {
            pipelines.compile(&ctx, name)?;
        }
        let pack = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|&x| match dtype {
                    MetalFloatType::F16 => f16::from_f32(x).to_le_bytes(),
                    MetalFloatType::Bf16 => bf16::from_f32(x).to_le_bytes(),
                })
                .collect()
        };
        for (operation, m, width, scalar_dim) in [
            (Operation::Gelu, 3_u32, 37_u32, 0_u32),
            (Operation::Gelu, 21, 15360, 0),
            (Operation::Gelu, 84, 15360, 0),
            (Operation::Gelu, 652, 15360, 0),
            (Operation::Gelu, 1024, 15360, 0),
            (Operation::Residual, 3, 37, 0),
            (Operation::Residual, 652, 3840, 1),
            (Operation::Residual, 652, 3840, 3840),
            (Operation::ResidualThenScale, 3, 37, 37),
            (Operation::ResidualThenScale, 652, 3840, 1),
            (Operation::Scale, 3, 37, 1),
            (Operation::Scale, 652, 3840, 3840),
        ] {
            let count = m * width;
            let count_usize = count as usize;
            let values: Vec<_> = (0..count_usize * 2)
                .map(|i| {
                    let x = ((i * 17 + 3) % 193) as f32 / 16.0 - 6.0;
                    x * [0.01, 1.0, 100.0][i % 3]
                })
                .collect();
            let input = pack(&values);
            let initial = pack(&values[..count_usize]);
            let scalars = pack(
                &(0..width)
                    .map(|i| 0.75 + (i % 7) as f32 / 16.0)
                    .collect::<Vec<_>>(),
            );
            let mut arena = MetalBufferArena::new(
                ctx.device(),
                input.len() + scalars.len() + 2 * (initial.len() + 64) + 1024,
            )?;
            let mut upload = |name: &str, bytes: &[u8]| -> Result<MetalRegion> {
                let region = arena.region(name, bytes.len(), 32)?;
                // SAFETY: idle arena, exact-size write to an exclusively owned region.
                unsafe {
                    arena.write_region(&region, bytes)?;
                }
                Ok(region)
            };
            let source = upload("input", &input)?;
            let scalar = upload("scalar", &scalars)?;
            let mut guarded = vec![0xA5; initial.len() + 64];
            guarded[32..32 + initial.len()].copy_from_slice(&initial);
            let outputs = [
                upload("single-thread", &guarded)?,
                upload("full-group", &guarded)?,
            ];
            drop(upload);
            let run =
                |candidate: bool| -> std::result::Result<(f64, f64), Box<dyn std::error::Error>> {
                    let out = &outputs[usize::from(candidate)];
                    // SAFETY: prior calls completed synchronously. This reset exactly
                    // fills a guarded output region, before the next submission.
                    unsafe {
                        arena.write_region(out, &guarded)?;
                    }
                    let started = std::time::Instant::now();
                    let command = ctx.queue().commandBuffer().ok_or("missing command")?;
                    let buffer = arena.buffer_retained();
                    let output = out.offset + 32;
                    // SAFETY: all live bindings are disjoint, dimensions exactly fit
                    // their regions, scalar width is checked by construction, and
                    // shaders touch only one bounded independent element per thread.
                    unsafe {
                        if candidate {
                            match operation {
                                Operation::Gelu => encode_gelu_mul(
                                    &command,
                                    &pipelines,
                                    buffer,
                                    source.offset,
                                    output,
                                    m,
                                    width,
                                    "pointwise_test",
                                )?,
                                Operation::Residual => encode_residual_add(
                                    &command,
                                    &pipelines,
                                    buffer,
                                    output,
                                    source.offset,
                                    count,
                                    width,
                                    (scalar_dim != 0).then_some(scalar.offset),
                                    scalar_dim,
                                )?,
                                Operation::ResidualThenScale => encode_residual_add_then_scale(
                                    &command,
                                    &pipelines,
                                    buffer,
                                    output,
                                    source.offset,
                                    count,
                                    width,
                                    scalar.offset,
                                    scalar_dim,
                                )?,
                                Operation::Scale => encode_layer_scale(
                                    &command,
                                    &pipelines,
                                    buffer,
                                    output,
                                    count,
                                    width,
                                    Some(scalar.offset),
                                    scalar_dim,
                                )?,
                            }
                        } else {
                            let encoder =
                                command.computeCommandEncoder().ok_or("missing encoder")?;
                            let (name, offsets, uniforms) = match operation {
                                Operation::Gelu => (
                                    "gelu_mul_f16",
                                    vec![(0, source.offset), (1, output)],
                                    vec![(2, m), (3, width)],
                                ),
                                Operation::Residual => (
                                    "residual_add_f16",
                                    vec![(0, output), (1, source.offset), (4, scalar.offset)],
                                    vec![(2, count), (3, width), (5, scalar_dim)],
                                ),
                                Operation::ResidualThenScale => (
                                    "residual_add_then_scale_f16",
                                    vec![(0, output), (1, source.offset), (4, scalar.offset)],
                                    vec![(2, count), (3, width), (5, scalar_dim)],
                                ),
                                Operation::Scale => (
                                    "layer_scale_f16",
                                    vec![(0, output), (1, scalar.offset)],
                                    vec![(2, count), (3, width), (4, scalar_dim)],
                                ),
                            };
                            encoder.setComputePipelineState(pipelines.get(name)?);
                            for (index, offset) in offsets {
                                encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
                            }
                            for (index, value) in uniforms {
                                encoder.setBytes_length_atIndex(
                                    std::ptr::NonNull::from(&value).cast(),
                                    4,
                                    index,
                                );
                            }
                            let grid = if matches!(operation, Operation::Gelu) {
                                MTLSize {
                                    width: m as usize,
                                    height: width as usize,
                                    depth: 1,
                                }
                            } else {
                                MTLSize {
                                    width: count_usize,
                                    height: 1,
                                    depth: 1,
                                }
                            };
                            encoder.dispatchThreads_threadsPerThreadgroup(
                                grid,
                                MTLSize {
                                    width: 1,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            encoder.endEncoding();
                        }
                    }
                    command.commit();
                    command.waitUntilCompleted();
                    if let Some(error) = command.error() {
                        return Err(format!("pointwise GPU error: {error}").into());
                    }
                    Ok((
                        (command.GPUEndTime() - command.GPUStartTime()) * 1000.0,
                        started.elapsed().as_secs_f64() * 1000.0,
                    ))
                };
            run(false)?;
            run(true)?;
            let read = |region: &MetalRegion| -> Vec<u8> {
                // SAFETY: both synchronous submissions completed; bounded read
                // of the complete guarded allocation with no outstanding use.
                unsafe { std::slice::from_raw_parts(arena.host_ptr(region), region.size).to_vec() }
            };
            let original = read(&outputs[0]);
            let grouped = read(&outputs[1]);
            assert_eq!(
                original, grouped,
                "{dtype:?} {operation:?} M{m} width{width}"
            );
            assert!(original[..32]
                .iter()
                .chain(original[original.len() - 32..].iter())
                .all(|&b| b == 0xA5));
            let mut gpu = [Vec::new(), Vec::new()];
            let mut wall = [Vec::new(), Vec::new()];
            for iteration in 0..6 {
                for candidate in if iteration % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let (g, w) = run(candidate)?;
                    gpu[usize::from(candidate)].push(g);
                    wall[usize::from(candidate)].push(w);
                }
            }
            let median = |values: &[f64]| {
                let mut values = values.to_vec();
                values.sort_by(f64::total_cmp);
                (values[2] + values[3]) * 0.5
            };
            reports.push(serde_json::json!({"dtype":format!("{dtype:?}"),"operation":format!("{operation:?}"),"tokens":m,"width":width,"scalar_dim":scalar_dim,"bit_identical":true,"guards_intact":true,"commands":14,"gpu_ms":{"one_thread":gpu[0],"grouped":gpu[1]},"wall_ms":{"one_thread":wall[0],"grouped":wall[1]},"gpu_median_ratio":median(&gpu[0])/median(&gpu[1])}));
        }
    }
    let report = serde_json::json!({"scope":"Same compiled shader, old single-thread launch versus production full-group launch. Alternating GPU timings; exact guarded output comparison.","cases":reports});
    if let Some(path) = std::env::var_os("RVLLM_METAL_POINTWISE_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
