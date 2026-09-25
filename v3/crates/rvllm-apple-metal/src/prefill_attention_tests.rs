//! Compare the opt-in BF16 production attention against its isolated prototype.
use super::*;
use crate::arena::MetalRegion;
use crate::MetalFloatType;
use half::bf16;
use objc2_metal::MTLComputePipelineState;
use sha2::Digest;

const SIMD_PREFILL: &str = r#"
kernel void attention_prefill_simdgroup_probe(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *k_cache [[buffer(1)]],
    device const bfloat *v_cache [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device const int *block_tables [[buffer(4)]],
    device const int *context_lens [[buffer(5)]],
    device const int *cu_seqlens [[buffer(6)]],
    device const int *positions [[buffer(7)]],
    constant uint &total_q [[buffer(8)]],
    constant uint &batch_size [[buffer(9)]],
    constant uint &num_heads [[buffer(10)]],
    constant uint &num_kv_heads [[buffer(11)]],
    constant uint &head_dim [[buffer(12)]],
    constant uint &block_size [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]],
    constant float &scale [[buffer(15)]],
    constant uint &attention_window [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint q_pos = group.x;
    uint head = group.y;
    if (q_pos >= total_q || head >= num_heads || (head_dim != 256 && head_dim != 512)) return;
    uint seq = batch_size;
    for (uint s = 0; s < batch_size; s++) {
        if (int(q_pos) >= cu_seqlens[s] && int(q_pos) < cu_seqlens[s + 1]) { seq = s; break; }
    }
    if (seq == batch_size || context_lens[seq] <= 0) return;
    uint kv_head = head / (num_heads / num_kv_heads);
    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;
    uint attn_len = min(uint(context_lens[seq]), uint(max(positions[q_pos], 0)) + 1u);
    uint attn_start = attention_window == 0 ? 0 : attn_len - min(attn_len, attention_window);
    uint slots = head_dim / 32u;
    float q_lane[16];
    float out_lane[16];
    for (uint slot = 0; slot < slots; slot++) {
        q_lane[slot] = float(q[q_pos * q_dim + head * head_dim + uint(lane) + slot * 32u]);
        out_lane[slot] = 0.0f;
    }
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    for (uint t = attn_start; t < attn_len; t++) {
        int block_id = block_tables[seq * max_blocks + t / block_size];
        if (block_id < 0) continue;
        uint kv_base = uint(block_id) * block_size * kv_dim + (t % block_size) * kv_dim + kv_head * head_dim;
        float partial_score = 0.0f;
        for (uint slot = 0; slot < slots; slot++) {
            uint d = uint(lane) + slot * 32u;
            partial_score += q_lane[slot] * float(k_cache[kv_base + d]);
        }
        float score = simd_sum(partial_score) * scale;
        float next_max = max(max_score, score);
        float correction = exp(max_score - next_max);
        float weight = exp(score - next_max);
        sum_exp = sum_exp * correction + weight;
        for (uint slot = 0; slot < slots; slot++) {
            uint d = uint(lane) + slot * 32u;
            out_lane[slot] = out_lane[slot] * correction + weight * float(v_cache[kv_base + d]);
        }
        max_score = next_max;
    }
    float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
    for (uint slot = 0; slot < slots; slot++) {
        output[q_pos * q_dim + head * head_dim + uint(lane) + slot * 32u] = bf16_sat(out_lane[slot] * inv_sum);
    }
}
"#;

fn noise(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    (x & 0xffff) as f32 / 32768.0 - 1.0
}

#[test]
#[ignore = "bounded BF16 public-Metal attention experiment; no ANE access"]
fn simd_prefill_checks_causality_pages_windows_and_fp64_reference(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut ctx = MetalContext::new()?;
    ctx.compile_library(&format!(
        "{}\n{SIMD_PREFILL}\n{}\n{}",
        crate::kernels::kernel_source_for_float_type(MetalFloatType::Bf16),
        crate::prefill_attention_candidate::CONVENTIONAL_MSL,
        crate::prefill_attention_candidate::TENSOR_OPS_MSL,
    ))?;
    let mut pipelines = PipelineCache::new();
    for name in [
        "attention_prefill_f16",
        "attention_prefill_simdgroup_probe",
        "attention_prefill_simdgroup_f16",
        crate::prefill_attention_candidate::CONVENTIONAL_ENTRYPOINT,
        crate::prefill_attention_candidate::TENSOR_OPS_ENTRYPOINT,
    ] {
        pipelines.compile(&ctx, name)?;
    }
    let candidate_pipeline =
        pipelines.get(crate::prefill_attention_candidate::CONVENTIONAL_ENTRYPOINT)?;
    let tensorops_pipeline =
        pipelines.get(crate::prefill_attention_candidate::TENSOR_OPS_ENTRYPOINT)?;
    let pipeline_resources = serde_json::json!({
        "conventional": {
            "thread_execution_width": candidate_pipeline.threadExecutionWidth(),
            "max_total_threads_per_threadgroup": candidate_pipeline.maxTotalThreadsPerThreadgroup(),
            "static_threadgroup_memory_bytes": candidate_pipeline.staticThreadgroupMemoryLength(),
        },
        "tensorops": {
            "thread_execution_width": tensorops_pipeline.threadExecutionWidth(),
            "max_total_threads_per_threadgroup": tensorops_pipeline.maxTotalThreadsPerThreadgroup(),
            "static_threadgroup_memory_bytes": tensorops_pipeline.staticThreadgroupMemoryLength(),
        },
        "queried_max_threadgroup_memory_bytes": ctx.max_threadgroup_memory(),
        "provenance": "public MTLComputePipelineState/MTLDevice getters after successful candidate compilation"
    });
    let tensor_admission = crate::prefill_attention_candidate::PrefillPlan {
        arm: crate::prefill_attention_candidate::PrefillArm::TensorOps,
        tokens: 256,
        heads: 16,
        kv_heads: 8,
        head_dim: 256,
        window: 1024,
        qkv_boundary: crate::prefill_attention_candidate::Boundary::ExternalQkvBf16,
        output_boundary: crate::prefill_attention_candidate::Boundary::ExternalOutputBf16,
    }
    .admit(crate::prefill_attention_candidate::QueriedHardware {
        tensor_ops: true,
        max_threadgroup_memory: ctx.max_threadgroup_memory(),
    });
    if !matches!(
        tensor_admission,
        crate::prefill_attention_candidate::Admission::Ready(_)
    ) {
        return Err("compiled TensorOps source failed its queried-hardware admission".into());
    }
    let requested = std::env::var("RVLLM_METAL_PREFILL_LENGTH")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(256);
    if !(1..=2048).contains(&requested) {
        return Err("RVLLM_METAL_PREFILL_LENGTH must be 1..=2048".into());
    }
    let mut cases = vec![
        (
            "sliding-21",
            256_u32,
            8_u32,
            1024_u32,
            vec![21_u32],
            vec![0_u32],
            vec![21_u32],
            false,
        ),
        (
            "sliding-84",
            256,
            8,
            1024,
            vec![84],
            vec![0],
            vec![84],
            false,
        ),
        (
            "sliding-652",
            256,
            8,
            1024,
            vec![652],
            vec![0],
            vec![652],
            false,
        ),
        ("global-21", 512, 1, 0, vec![21], vec![0], vec![21], false),
        (
            "global-652",
            512,
            1,
            0,
            vec![652],
            vec![0],
            vec![652],
            false,
        ),
        (
            "sliding-two-chunks",
            256,
            8,
            1024,
            vec![9, 7],
            vec![71, 1023],
            vec![80, 1030],
            false,
        ),
        (
            "window-excludes-prefix",
            256,
            8,
            32,
            vec![5],
            vec![1045],
            vec![1050],
            false,
        ),
        (
            "causal-poison-future",
            512,
            1,
            0,
            vec![1],
            vec![0],
            vec![33],
            true,
        ),
    ];
    for label in ["first-hole", "middle-hole", "last-hole"] {
        cases.push((label, 256, 8, 1024, vec![96], vec![0], vec![96], false));
    }
    for length in [
        requested.saturating_sub(1).max(1),
        requested,
        (requested + 1).min(2048),
    ] {
        cases.push((
            "requested-boundary",
            256,
            8,
            1024,
            vec![length],
            vec![0],
            vec![length],
            false,
        ));
    }
    let mut reports = Vec::new();
    for (label, hd, kv_heads, window, lengths, starts, contexts, poison_future) in cases {
        let heads = 16_u32;
        let block = 32_u32;
        let batch = lengths.len() as u32;
        let total: u32 = lengths.iter().sum();
        let q_dim = heads * hd;
        let kv_dim = kv_heads * hd;
        let max_blocks = contexts.iter().map(|c| c.div_ceil(block)).max().unwrap();
        let pages: u32 = contexts.iter().map(|c| c.div_ceil(block)).sum();
        let mut tables = vec![-1_i32; (batch * max_blocks) as usize];
        let mut cursor = 0;
        for (seq, &context) in contexts.iter().enumerate() {
            for logical in 0..context.div_ceil(block) {
                tables[seq * max_blocks as usize + logical as usize] = (pages - 1 - cursor) as i32;
                cursor += 1;
            }
        }
        if label.ends_with("hole") {
            let logical = match label {
                "first-hole" => 0,
                "middle-hole" => 1,
                _ => 2,
            };
            tables[logical] = -1;
        }
        let mut cu = vec![0_i32];
        let mut positions = Vec::new();
        for (&len, &start) in lengths.iter().zip(&starts) {
            cu.push(cu.last().unwrap() + len as i32);
            positions.extend((start..start + len).map(|p| p as i32));
        }
        let query: Vec<_> = (0..total * q_dim)
            .map(|i| bf16::from_f32(noise(i + 1) * if (i / q_dim) % 7 == 0 { 4.0 } else { 0.25 }))
            .collect();
        let mut keys = vec![bf16::NAN; (pages * block * kv_dim) as usize];
        let mut values = keys.clone();
        for (seq, &context) in contexts.iter().enumerate() {
            for t in 0..context {
                if poison_future && t > 0 {
                    continue;
                }
                let page = tables[seq * max_blocks as usize + (t / block) as usize];
                if page < 0 {
                    continue;
                }
                let page = page as u32;
                let base = (page * block * kv_dim + (t % block) * kv_dim) as usize;
                for d in 0..kv_dim as usize {
                    let seed = (seq as u32).wrapping_mul(10000019) + t * kv_dim + d as u32;
                    keys[base + d] = bf16::from_f32(noise(seed + 77) * 0.5);
                    values[base + d] = bf16::from_f32(noise(seed + 511));
                }
            }
        }
        let mut arena = MetalBufferArena::new(
            ctx.device(),
            (query.len() + keys.len() + values.len() + 2 * query.len()) * 2 + 4096,
        )?;
        let mut upload = |name: &str, bytes: &[u8]| -> Result<MetalRegion> {
            let r = arena.region(name, bytes.len(), 32)?;
            // SAFETY: exact-size initialization of a private idle allocation.
            unsafe {
                arena.write_region(&r, bytes)?;
            }
            Ok(r)
        };
        let pack = |v: &[bf16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let q = upload("q", &pack(&query))?;
        let k = upload("k", &pack(&keys))?;
        let v = upload("v", &pack(&values))?;
        let table = upload(
            "pages",
            &tables
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let context = upload(
            "context",
            &contexts
                .iter()
                .flat_map(|&x| (x as i32).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let cumulative = upload(
            "cu",
            &cu.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        let pos = upload(
            "positions",
            &positions
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let output_bytes = query.len() * 2;
        let outputs = [
            upload("scalar", &vec![0xA5; output_bytes + 64])?,
            upload("candidate", &vec![0xA5; output_bytes + 64])?,
            upload("tensorops", &vec![0xA5; output_bytes + 64])?,
        ];
        drop(upload);
        // mode 0 = scalar control, 1 = conventional candidate,
        // mode 2 = existing SIMD control, 3 = TensorOps candidate.
        let run = |mode: u8|
         -> std::result::Result<(f64, f64), Box<dyn std::error::Error>> {
            let timer = std::time::Instant::now();
            let command = ctx.queue().commandBuffer().ok_or("command missing")?;
            let encoder = command.computeCommandEncoder().ok_or("encoder missing")?;
            let (kernel, output_index, cooperative) = match mode {
                0 => ("attention_prefill_f16", 0usize, false),
                1 => (
                    crate::prefill_attention_candidate::CONVENTIONAL_ENTRYPOINT,
                    1,
                    true,
                ),
                2 => ("attention_prefill_simdgroup_f16", 1, true),
                3 => (
                    crate::prefill_attention_candidate::TENSOR_OPS_ENTRYPOINT,
                    2,
                    true,
                ),
                _ => return Err("invalid prefill referee mode".into()),
            };
            encoder.setComputePipelineState(pipelines.get(kernel)?);
            // SAFETY: exact tensor allocations above, guarded outputs, complete
            // per-sequence page tables, live uniform values copied by Metal.
            // Both kernels use the same ABI; the candidate requires one SIMD group.
            unsafe {
                for (i, offset) in [
                    q.offset,
                    k.offset,
                    v.offset,
                    outputs[output_index].offset + 32,
                    table.offset,
                    context.offset,
                    cumulative.offset,
                    pos.offset,
                ]
                .into_iter()
                .enumerate()
                {
                    encoder.setBuffer_offset_atIndex(Some(arena.buffer_retained()), offset, i);
                }
                for (i, value) in [total, batch, heads, kv_heads, hd, block, max_blocks]
                    .iter()
                    .enumerate()
                {
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::from(value).cast(),
                        4,
                        i + 8,
                    );
                }
                let scale = 1.0_f32;
                encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&scale).cast(), 4, 15);
                encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&window).cast(), 4, 16);
            }
            let grid = MTLSize {
                width: total as usize,
                height: heads as usize,
                depth: 1,
            };
            let threads = MTLSize {
                width: if cooperative { 32 } else { 1 },
                height: 1,
                depth: 1,
            };
            if cooperative {
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
            } else {
                encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
            }
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return Err(format!("attention GPU: {error}").into());
            }
            Ok((
                (command.GPUEndTime() - command.GPUStartTime()) * 1000.0,
                timer.elapsed().as_secs_f64() * 1000.0,
            ))
        };
        if !label.ends_with("hole") {
            run(0)?;
        }
        run(1)?;
        run(3)?;
        let read = |region: &MetalRegion| -> Vec<f32> {
            // SAFETY: synchronous completion precedes this bounded read; no
            // CPU access occurs while either output is in use by the GPU.
            let bytes = unsafe { std::slice::from_raw_parts(arena.host_ptr(region), region.size) };
            assert!(bytes[..32]
                .iter()
                .chain(bytes[bytes.len() - 32..].iter())
                .all(|&b| b == 0xA5));
            bytes[32..bytes.len() - 32]
                .chunks_exact(2)
                .map(|x| bf16::from_le_bytes(x.try_into().unwrap()).to_f32())
                .collect()
        };
        let scalar = read(&outputs[0]);
        let candidate = read(&outputs[1]);
        let tensorops = read(&outputs[2]);
        let mut production_poison = vec![0xA5; output_bytes + 64];
        production_poison[32..32 + output_bytes].fill(0xFF); // BF16 NaNs
                                                             // SAFETY: prototype completion was synchronous. Re-poison its exact
                                                             // allocation so production must independently write every output.
        unsafe {
            arena.write_region(&outputs[1], &production_poison)?;
        }
        run(1)?;
        assert_eq!(
            candidate,
            read(&outputs[1]),
            "candidate output bits changed on repeat"
        );
        unsafe {
            arena.write_region(&outputs[2], &production_poison)?;
        }
        run(3)?;
        assert_eq!(
            tensorops,
            read(&outputs[2]),
            "TensorOps output bits changed on repeat"
        );
        unsafe {
            arena.write_region(&outputs[1], &production_poison)?;
        }
        run(2)?;
        assert_eq!(
            candidate,
            read(&outputs[1]),
            "existing SIMD control must match tiled candidate"
        );
        assert!(candidate.iter().all(|x| x.is_finite()), "{label} conventional finite");
        assert!(tensorops.iter().all(|x| x.is_finite()), "{label} TensorOps finite");
        let relative_l2 = if label.ends_with("hole") {
            0.0
        } else {
            (scalar
                .iter()
                .zip(&candidate)
                .map(|(&a, &b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / scalar.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>())
            .sqrt()
        };
        let max_difference = if label.ends_with("hole") {
            0.0
        } else {
            scalar
                .iter()
                .zip(&candidate)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f32, f32::max)
        };
        assert!(
            relative_l2 < 0.003 && max_difference < 0.032,
            "{label}: conventional L2 {relative_l2} max {max_difference}"
        );
        let tensor_reference = if label.ends_with("hole") {
            &candidate
        } else {
            &scalar
        };
        let tensor_relative_l2 = (tensor_reference
            .iter()
            .zip(&tensorops)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / tensor_reference
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .max(1e-30))
        .sqrt();
        let tensor_max_difference = tensor_reference
            .iter()
            .zip(&tensorops)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            tensor_relative_l2 < 0.004 && tensor_max_difference < 0.032,
            "{label}: TensorOps L2 {tensor_relative_l2} max {tensor_max_difference}"
        );
        let mut squared_error = 0.0;
        let mut squared_reference = 0.0;
        let mut max_cpu_error = 0.0_f64;
        let mut tensor_squared_error = 0.0;
        let mut tensor_max_cpu_error = 0.0_f64;
        let mut sample_rows = vec![0, total / 2, total - 1];
        if total > 1 {
            sample_rows.push(1);
        }
        sample_rows.sort();
        sample_rows.dedup();
        for row in sample_rows {
            let seq = cu
                .windows(2)
                .position(|c| row as i32 >= c[0] && (row as i32) < c[1])
                .unwrap();
            let end = contexts[seq].min(positions[row as usize] as u32 + 1);
            let begin = if window == 0 {
                0
            } else {
                end.saturating_sub(window)
            };
            for head in [0, 7, 15] {
                let kv_head = head / (heads / kv_heads);
                let qb = (row * q_dim + head * hd) as usize;
                let mut scores = Vec::new();
                let mut bases = Vec::new();
                for t in begin..end {
                    let page = tables[seq * max_blocks as usize + (t / block) as usize];
                    if page < 0 {
                        continue;
                    }
                    let page = page as u32;
                    let kb = (page * block * kv_dim + (t % block) * kv_dim + kv_head * hd) as usize;
                    scores.push(
                        (0..hd as usize)
                            .map(|d| query[qb + d].to_f64() * keys[kb + d].to_f64())
                            .sum::<f64>(),
                    );
                    bases.push(kb);
                }
                if scores.is_empty() {
                    continue;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for d in 0..hd as usize {
                    let expected = weights
                        .iter()
                        .zip(&bases)
                        .map(|(&w, &b)| w * values[b + d].to_f64())
                        .sum::<f64>()
                        / sum;
                    let error = f64::from(candidate[qb + d]) - expected;
                    let tensor_error = f64::from(tensorops[qb + d]) - expected;
                    squared_error += error * error;
                    tensor_squared_error += tensor_error * tensor_error;
                    squared_reference += expected * expected;
                    max_cpu_error = max_cpu_error.max(error.abs());
                    tensor_max_cpu_error = tensor_max_cpu_error.max(tensor_error.abs());
                }
            }
        }
        let cpu_relative_l2 = (squared_error / squared_reference).sqrt();
        assert!(
            cpu_relative_l2 < 0.004 && max_cpu_error < 0.01,
            "{label}: conventional FP64 L2 {cpu_relative_l2} max {max_cpu_error}"
        );
        let tensor_cpu_relative_l2 =
            (tensor_squared_error / squared_reference.max(1e-30)).sqrt();
        assert!(
            tensor_cpu_relative_l2 < 0.004 && tensor_max_cpu_error < 0.01,
            "{label}: TensorOps FP64 L2 {tensor_cpu_relative_l2} max {tensor_max_cpu_error}"
        );
        let mut gpu = [Vec::new(), Vec::new(), Vec::new()];
        let mut wall = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0..6 {
            let order = if i % 2 == 0 { [2u8, 1, 3] } else { [3u8, 1, 2] };
            for mode in order {
                let (g, w) = run(mode)?;
                let index = match mode {
                    2 => 0,
                    1 => 1,
                    3 => 2,
                    _ => unreachable!(),
                };
                gpu[index].push(g);
                wall[index].push(w);
            }
        }
        let median = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            (v[2] + v[3]) * 0.5
        };
        reports.push(serde_json::json!({
            "label":label,"tokens":total,"head_dim":hd,"kv_heads":kv_heads,
            "window":window,"contexts":contexts,"starts":starts,"commands":25,
            "guards_unchanged":true,"repeatable_output_bits":true,
            "conventional":{"relative_l2_vs_scalar":relative_l2,
                "max_abs_vs_scalar":max_difference,
                "sampled_fp64_relative_l2":cpu_relative_l2,
                "sampled_fp64_max_abs":max_cpu_error},
            "tensorops":{"relative_l2_vs_reference":tensor_relative_l2,
                "max_abs_vs_reference":tensor_max_difference,
                "sampled_fp64_relative_l2":tensor_cpu_relative_l2,
                "sampled_fp64_max_abs":tensor_max_cpu_error},
            "gpu_ms":{"existing_simd_control":gpu[0],"tiled_candidate":gpu[1],"tensorops_candidate":gpu[2]},
            "wall_ms":{"existing_simd_control":wall[0],"tiled_candidate":wall[1],"tensorops_candidate":wall[2]},
            "gpu_median_ratio":{"control_over_conventional":median(&gpu[0])/median(&gpu[1]),
                "control_over_tensorops":median(&gpu[0])/median(&gpu[2]),
                "conventional_over_tensorops":median(&gpu[1])/median(&gpu[2])}
        }));
    }
    let executable = std::fs::read(std::env::current_exe()?)?;
    let generated = crate::prefill_attention_candidate::identity(
        crate::prefill_attention_candidate::CONVENTIONAL_MSL,
        crate::prefill_attention_candidate::CONVENTIONAL_ENTRYPOINT,
    );
    let tensor_generated = crate::prefill_attention_candidate::identity(
        crate::prefill_attention_candidate::TENSOR_OPS_MSL,
        crate::prefill_attention_candidate::TENSOR_OPS_ENTRYPOINT,
    );
    let report = serde_json::json!({
        "schema":"rvllm.gemma4.metal_prefill_referee.v2","status":"qualified",
        "candidates":[generated.entrypoint,tensor_generated.entrypoint],
        "requested_tokens":requested,"default_off":true,
        "qkv_boundary":"external_bf16","output_projection_boundary":"external_bf16",
        "generated":{
            "generator_version":generated.generator_version,
            "conventional":{"entrypoint":generated.entrypoint,"source_sha256":generated.source_sha256},
            "tensorops":{"entrypoint":tensor_generated.entrypoint,"source_sha256":tensor_generated.source_sha256},
            "executable_sha256":format!("{:x}",sha2::Sha256::digest(&executable)),
            "compiler_artifacts":"jit-library; offline AIR/metallib/disassembly/resource identity still required before promotion",
            "pipeline_resources":pipeline_resources},
        "tensorops":{"status":"compiled-and-refereed",
            "capability_evidence":"candidate source compiled and PSO instantiated on the live MTLDevice; no GPU-family inference"},
        "cases":reports,
        "scope":"Controlled BF16 inputs, permuted physical KV pages, absolute causal positions, windows, tails and holes, guarded outputs, sampled independent FP64 softmax/PV; no ANE execution."
    });
    if let Some(path) = std::env::var_os("RVLLM_METAL_PREFILL_ATTENTION_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
