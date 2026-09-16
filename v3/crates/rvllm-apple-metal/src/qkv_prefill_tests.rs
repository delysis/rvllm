//! Compare the complete new QKV path with the qualified scalar fused shader.
//! Actual checkpoint weights, controlled BF16 activations, guarded outputs.

use super::*;
use crate::arena::MetalRegion;
use crate::weight_loader::{load_safetensor_entry_bf16, scan_safetensor_tensors};
use crate::MetalFloatType;
use half::bf16;
use std::path::PathBuf;

#[test]
#[ignore = "bounded public-Metal QKV comparison; requires the real Gemma 4 12B checkpoint"]
fn qkv_prefill_preserves_norm_boundary_and_measures_gpu_time(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let model =
        PathBuf::from(std::env::var_os("RVLLM_METAL_QKV_MODEL_DIR").ok_or("model path required")?);
    let tensors = scan_safetensor_tensors(&model)?;
    let mut ctx = MetalContext::new()?;
    ctx.compile_library(&crate::kernels::kernel_source_for_float_type(
        MetalFloatType::Bf16,
    ))?;
    let mut pipelines = PipelineCache::new();
    for name in [
        "qkv_project_f32_batch8",
        "qkv_projected_rmsnorm_rope_cache_f16",
        "qkv_headwise_rmsnorm_rope_cache_f16",
    ] {
        pipelines.compile(&ctx, name)?;
    }
    let mut reports = Vec::new();
    for (layer, m, hd, kv_heads, rope_dim, theta) in [
        (0, 21_u32, 256_u32, 8_u32, 256_u32, 10000_f32),
        (5, 28, 512, 1, 128, 1000000_f32),
    ] {
        let hidden = 3840_u32;
        let q_dim = 16 * hd;
        let kv_dim = kv_heads * hd;
        let n = q_dim + 2 * kv_dim;
        let load = |suffix: &str,
                    shape: &[usize]|
         -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
            let name = format!("model.language_model.layers.{layer}.self_attn.{suffix}.weight");
            let entry = tensors.get(&name).ok_or("QKV tensor missing")?;
            assert_eq!(entry.shape, shape);
            Ok(load_safetensor_entry_bf16(entry)?)
        };
        let mut weights = load("q_proj", &[q_dim as usize, hidden as usize])?;
        let keys = load("k_proj", &[kv_dim as usize, hidden as usize])?;
        weights.extend_from_slice(&keys);
        if layer == 5 {
            weights.extend_from_slice(&keys);
        } else {
            weights.extend(load("v_proj", &[kv_dim as usize, hidden as usize])?);
        }
        let q_gamma = load("q_norm", &[hd as usize])?;
        let k_gamma = load("k_norm", &[hd as usize])?;
        let mut input = Vec::new();
        for row in 0..m {
            let scale = [0.1_f32, 1.0, 10.0, 100.0][row as usize % 4];
            for k in 0..hidden {
                let value = (((k * 7 + row * 13) % 47) as f32 / 32.0 - 0.75) * scale;
                input.extend_from_slice(&bf16::from_f32(value).to_le_bytes());
            }
        }
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for pos in 0..m {
            for pair in 0..rope_dim / 2 {
                let angle = pos as f32 * theta.powf(-((2 * pair) as f32) / hd as f32);
                cos.extend_from_slice(&angle.cos().to_le_bytes());
                sin.extend_from_slice(&angle.sin().to_le_bytes());
            }
        }
        let positions: Vec<u8> = (0..m).flat_map(|p| (p as i32).to_le_bytes()).collect();
        let slots: Vec<u8> = (0..m)
            .flat_map(|p| if p == 0 { -1_i32 } else { (m - 1 - p) as i32 }.to_le_bytes())
            .collect();
        let projection_bytes = m as usize * n as usize * 4;
        let q_bytes = m as usize * q_dim as usize * 2;
        let kv_bytes = m as usize * kv_dim as usize * 2;
        let capacity = weights.len()
            + input.len()
            + q_gamma.len()
            + k_gamma.len()
            + cos.len()
            + sin.len()
            + positions.len()
            + slots.len()
            + projection_bytes
            + 2 * (q_bytes + 4 * kv_bytes)
            + 4096;
        let mut arena = MetalBufferArena::new(ctx.device(), capacity)?;
        let mut upload = |name: &str, bytes: &[u8]| -> Result<MetalRegion> {
            let region = arena.region(name, bytes.len(), 32)?;
            // SAFETY: this arena is idle and the slice exactly fills its region.
            unsafe {
                arena.write_region(&region, bytes)?;
            }
            Ok(region)
        };
        let a = upload("input", &input)?;
        let b = upload("weights", &weights)?;
        let qg = upload("q_gamma", &q_gamma)?;
        let kg = upload("k_gamma", &k_gamma)?;
        let ct = upload("cos", &cos)?;
        let st = upload("sin", &sin)?;
        let pos = upload("positions", &positions)?;
        let slots = upload("slots", &slots)?;
        let projected = upload("projection_guarded", &vec![0xA5; projection_bytes + 64])?;
        let mut outputs = Vec::new();
        for path in 0..2 {
            let mut regions = Vec::new();
            for (name, bytes) in [
                ("q", q_bytes),
                ("k", kv_bytes),
                ("v", kv_bytes),
                ("k_cache", kv_bytes),
                ("v_cache", kv_bytes),
            ] {
                regions.push(upload(&format!("{path}-{name}"), &vec![0xA5; bytes + 64])?);
            }
            outputs.push(regions);
        }
        drop(upload);
        drop((weights, input));
        let run = |candidate: bool| -> std::result::Result<(f64, f64), Box<dyn std::error::Error>> {
            let started = std::time::Instant::now();
            let command = ctx
                .queue()
                .commandBuffer()
                .ok_or("command buffer missing")?;
            let buffer = arena.buffer_retained();
            let out = &outputs[usize::from(candidate)];
            // SAFETY: all bindings are separately allocated in this live arena;
            // the FP32 projection and planar/cache regions include guards. The
            // shaders use the checked dimensions above and 256-thread groups.
            unsafe {
                if candidate {
                    encode_gemm_with_output(
                        &command,
                        &pipelines,
                        buffer,
                        a.offset,
                        b.offset,
                        projected.offset + 32,
                        m,
                        n,
                        hidden,
                        1.0,
                        0.0,
                        true,
                        false,
                    )?;
                }
                encode_qkv_headwise_rmsnorm_rope_cache(
                    &command,
                    &pipelines,
                    buffer,
                    if candidate {
                        projected.offset + 32
                    } else {
                        a.offset
                    },
                    b.offset,
                    qg.offset,
                    kg.offset,
                    qg.offset,
                    out[0].offset + 32,
                    out[1].offset + 32,
                    out[2].offset + 32,
                    ct.offset,
                    st.offset,
                    pos.offset,
                    slots.offset,
                    out[3].offset + 32,
                    out[4].offset + 32,
                    m,
                    hidden,
                    hd,
                    16,
                    kv_heads,
                    0,
                    q_dim,
                    q_dim + kv_dim,
                    1e-6,
                    false,
                    rope_dim,
                    "qkv_prefill_comparison",
                    candidate,
                )?;
            }
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return Err(format!("QKV GPU error: {error}").into());
            }
            let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
            assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
            Ok((gpu_ms, started.elapsed().as_secs_f64() * 1000.0))
        };
        run(false)?;
        run(true)?;
        let read = |region: &MetalRegion| -> Vec<u8> {
            // SAFETY: both synchronous commands completed, and this read is
            // bounded by the region; no GPU commands run concurrently here.
            unsafe { std::slice::from_raw_parts(arena.host_ptr(region), region.size).to_vec() }
        };
        let guard = |bytes: &[u8]| {
            assert!(bytes[..32]
                .iter()
                .chain(&bytes[bytes.len() - 32..])
                .all(|&v| v == 0xA5));
        };
        guard(&read(&projected));
        let mut errors = Vec::new();
        for index in 0..5 {
            let base = read(&outputs[0][index]);
            let candidate = read(&outputs[1][index]);
            guard(&base);
            guard(&candidate);
            let mut squared_error = 0.0_f64;
            let mut squared_reference = 0.0_f64;
            let mut maximum = 0.0_f32;
            for (a, b) in base[32..base.len() - 32]
                .chunks_exact(2)
                .zip(candidate[32..candidate.len() - 32].chunks_exact(2))
            {
                let a = bf16::from_le_bytes([a[0], a[1]]).to_f32();
                let b = bf16::from_le_bytes([b[0], b[1]]).to_f32();
                assert!(a.is_finite() && b.is_finite());
                squared_error += f64::from(a - b).powi(2);
                squared_reference += f64::from(a).powi(2);
                maximum = maximum.max((a - b).abs());
            }
            let relative_l2 = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
            assert!(
                relative_l2 <= 0.0025,
                "QKV output {index} relative L2 {relative_l2}"
            );
            if index >= 3 {
                let unwritten = base.len() - 32 - kv_dim as usize * 2;
                assert!(base[unwritten..base.len() - 32].iter().all(|&v| v == 0xA5));
                assert!(candidate[unwritten..candidate.len() - 32]
                    .iter()
                    .all(|&v| v == 0xA5));
            }
            errors.push(
                serde_json::json!({"output":index,"relative_l2":relative_l2,"max_abs":maximum}),
            );
        }
        for _ in 0..2 {
            run(false)?;
            run(true)?;
        }
        let mut baseline = Vec::new();
        let mut candidate = Vec::new();
        for cycle in 0..6 {
            if cycle % 2 == 0 {
                baseline.push(run(false)?);
                candidate.push(run(true)?);
            } else {
                candidate.push(run(true)?);
                baseline.push(run(false)?);
            }
        }
        let report = serde_json::json!({"layer":layer,"m":m,"n":n,"k":hidden,"errors_q_k_v_kcache_vcache":errors,
            "baseline_gpu_wall_ms":baseline,"candidate_gpu_wall_ms":candidate});
        eprintln!("{report}");
        reports.push(report);
    }
    if let Some(path) = std::env::var_os("RVLLM_METAL_QKV_COMPARISON_REPORT") {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        serde_json::to_writer_pretty(
            file,
            &serde_json::json!({"cases":reports,
            "claim":"Actual checkpoint QKV and gamma weights, controlled synthetic BF16 activations; compares complete projection plus normalization/RoPE/cache paths. Guarded outputs and negative cache slots. No ANE access or full-model quality claim."}),
        )?;
    }
    Ok(())
}
