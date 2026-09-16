//! Measured, numerically checked ANE projection on real checkpoint weights or
//! explicit synthetic dimensions. This is not an end-to-end decode benchmark.

#![forbid(unsafe_code)]

use half::f16;
use rvllm_apple::ane_dynamic_ffn::{AneDynamicFfn, AneDynamicFfnProgram};
use rvllm_apple::ane_dynamic_linear::{AneDynamicLinear, AneDynamicLinearProgram};
use rvllm_apple::ane_linear::{AneGatedFfn, AneLinear};
use rvllm_apple_metal::weight_loader::{load_safetensor_f16, scan_safetensor_tensors};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

enum Projection {
    Linear(AneLinear),
    Ffn(AneGatedFfn),
    DynamicFfn(AneDynamicFfn),
    DynamicLinear(AneDynamicLinear),
}

impl Projection {
    fn project(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        match self {
            Self::Linear(projection) => projection.project(input, output),
            Self::Ffn(projection) => projection.project(input, output),
            Self::DynamicFfn(projection) => projection.project(input, output),
            Self::DynamicLinear(projection) => projection.project(input, output),
        }
    }
}

fn cpu_project(weights: &[f16], input: &[f16]) -> Vec<f32> {
    weights
        .chunks_exact(input.len())
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(w, x)| w.to_f32() * x.to_f32())
                .sum()
        })
        .collect()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_dir = None;
    let mut tensor = None;
    let mut ffn_prefix = None;
    let mut dynamic_ffn = false;
    let mut dynamic_linear = false;
    let mut input_channels = 3840_usize;
    let mut output_channels = 15360_usize;
    let mut spatial = 1_usize;
    let mut iterations = 30_usize;
    let mut report = None;
    let mut prepare_ffn_output = None;
    let mut profile_stages = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--tensor" => tensor = Some(value),
            "--ffn-prefix" => ffn_prefix = Some(value),
            "--ffn-mode" => {
                dynamic_ffn = match value.as_str() {
                    "constant" => false,
                    "dynamic" => true,
                    _ => return Err("--ffn-mode must be constant or dynamic".into()),
                };
            }
            "--linear-mode" => {
                dynamic_linear = match value.as_str() {
                    "constant" => false,
                    "dynamic" => true,
                    _ => return Err("--linear-mode must be constant or dynamic".into()),
                };
            }
            "--input-channels" => input_channels = value.parse()?,
            "--output-channels" => output_channels = value.parse()?,
            "--spatial" => spatial = value.parse()?,
            "--iterations" => iterations = value.parse()?,
            "--profile-stages" => profile_stages = value.parse()?,
            "--report" => report = Some(PathBuf::from(value)),
            "--prepare-ffn-output" => prepare_ffn_output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    if !(1..=10000).contains(&iterations) {
        return Err("iterations must be in 1..=10000".into());
    }
    if prepare_ffn_output.is_some() && ffn_prefix.is_none() {
        return Err("--prepare-ffn-output requires --ffn-prefix; it performs no ANE calls".into());
    }
    if dynamic_ffn && ffn_prefix.is_none() {
        return Err("--ffn-mode dynamic requires --ffn-prefix".into());
    }
    if profile_stages && !dynamic_ffn {
        return Err("--profile-stages currently requires --ffn-mode dynamic".into());
    }
    if dynamic_linear && (ffn_prefix.is_some() || dynamic_ffn || spatial != 1) {
        return Err("--linear-mode dynamic requires spatial=1 and no FFN selection".into());
    }
    if let Some(prefix) = &ffn_prefix {
        if tensor.is_some() || model_dir.is_none() || spatial != 1 {
            return Err("ffn-prefix requires model-dir, spatial=1, and no tensor argument".into());
        }
        tensor = Some(format!("{prefix}.gate_proj.weight"));
    }
    let weights: Vec<f16> = match (&model_dir, &tensor) {
        (Some(dir), Some(name)) => {
            let entries = scan_safetensor_tensors(dir)?;
            let entry = entries.get(name).ok_or("checkpoint tensor not found")?;
            let [rows, cols] = entry.shape.as_slice() else {
                return Err("projection tensor must be a matrix [output,input]".into());
            };
            output_channels = *rows;
            input_channels = *cols;
            load_safetensor_f16(dir, name)?
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect()
        }
        (None, None) => {
            let count = input_channels
                .checked_mul(output_channels)
                .filter(|&n| n > 0 && n <= 512 * 1024 * 1024)
                .ok_or("synthetic weight shape must be nonempty and at most 1 GiB")?;
            (0..count)
                .map(|i| f16::from_f32(((i * 17 + 3) % 61) as f32 / 2048.0 - 0.015))
                .collect()
        }
        _ => return Err("model-dir and tensor must be specified together".into()),
    };
    if weights.iter().any(|w| !w.is_finite()) {
        return Err("projection weights contain nonfinite values".into());
    }
    let ffn_weights = if let (Some(dir), Some(prefix)) = (&model_dir, &ffn_prefix) {
        let entries = scan_safetensor_tensors(dir)?;
        let load =
            |suffix: &str, shape: [usize; 2]| -> Result<Vec<f16>, Box<dyn std::error::Error>> {
                let name = format!("{prefix}.{suffix}.weight");
                let entry = entries.get(&name).ok_or("FFN tensor missing")?;
                if entry.shape != shape {
                    return Err(
                        format!("FFN tensor {name} shape {:?} != {shape:?}", entry.shape).into(),
                    );
                }
                let values: Vec<_> = load_safetensor_f16(dir, &name)?
                    .chunks_exact(2)
                    .map(|b| f16::from_le_bytes([b[0], b[1]]))
                    .collect();
                if values.iter().any(|w| !w.is_finite()) {
                    return Err("FFN weights contain nonfinite values".into());
                }
                Ok(values)
            };
        Some((
            load("up_proj", [output_channels, input_channels])?,
            load("down_proj", [input_channels, output_channels])?,
        ))
    } else {
        None
    };
    let weight_bytes = weights.len() * 2 * if ffn_weights.is_some() { 3 } else { 1 };
    let mut hash = Sha256::new();
    hash_f16_weights(&mut hash, &weights);
    if let Some((up, down)) = &ffn_weights {
        hash_f16_weights(&mut hash, up);
        hash_f16_weights(&mut hash, down);
    }
    let weight_sha256 = format!("{:x}", hash.finalize());
    // This branch returns before constructing any private API object. It makes
    // a concrete artifact for later review/validation while ANE is paused.
    if let Some(directory) = prepare_ffn_output {
        let (up, down) = ffn_weights.as_ref().ok_or("FFN weights required")?;
        let layout =
            rvllm_apple::ane_ffn_layout::PackedFfnLayout::new(input_channels, output_channels)?;
        std::fs::create_dir(&directory)?;
        let start = Instant::now();
        let packed = layout.pack_weights(&weights, up, down)?;
        let pack_ms = start.elapsed().as_secs_f64() * 1000.0;
        let packed_sha256 = format!("{:x}", Sha256::digest(&packed));
        std::fs::write(directory.join("ffn.mil"), layout.mil())?;
        std::fs::write(directory.join("ffn-input.fp16"), &packed)?;
        let value = serde_json::json!({
            "schema": "rvllm.ane_dynamic_ffn_prepare.v1", "model_dir": model_dir,
            "ffn_prefix": ffn_prefix, "hidden_size": input_channels, "intermediate_size": output_channels,
            "converted_weight_sha256": weight_sha256, "weight_bytes": weight_bytes,
            "packed_input_bytes": packed.len(), "packed_input_sha256": packed_sha256,
            "output_bytes": layout.output_bytes(), "pack_ms": pack_ms,
            "ane_compilation_attempted": false, "ane_execution_verified": false,
            "claim": "CPU-only checkpoint weight conversion and single-input dynamic FFN packing; compilation, request sharing and ANE execution are unverified."
        });
        let json = serde_json::to_string_pretty(&value)?;
        std::fs::write(directory.join("report.json"), &json)?;
        if let Some(path) = report {
            std::fs::write(path, &json)?;
        }
        println!("{json}");
        return Ok(());
    }
    eprintln!("Compiling ANE projection {output_channels}x{input_channels}, spatial={spatial}");
    let start = Instant::now();
    let intermediate = output_channels;
    let mut linear = if let Some((up, down)) = &ffn_weights {
        output_channels = input_channels;
        if dynamic_ffn {
            let program = AneDynamicFfnProgram::compile(input_channels, intermediate)?;
            Projection::DynamicFfn(program.create_layer(&weights, up, down)?)
        } else {
            Projection::Ffn(AneGatedFfn::compile(
                &weights,
                up,
                down,
                input_channels,
                intermediate,
            )?)
        }
    } else if dynamic_linear {
        let program = AneDynamicLinearProgram::compile(input_channels, output_channels)?;
        Projection::DynamicLinear(program.create_layer(&weights)?)
    } else {
        Projection::Linear(AneLinear::compile(
            &weights,
            input_channels,
            output_channels,
            spatial,
        )?)
    };
    let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut input = vec![f16::ZERO; input_channels];
    let mut output = vec![f16::ZERO; output_channels];
    let mut max_abs_error = 0_f32;
    let mut squared_error = 0_f64;
    let mut squared_reference = 0_f64;
    // Three distinct inputs detect stale buffers as well as layout errors.
    for step in 0..3 {
        for (i, value) in input.iter_mut().enumerate() {
            *value = f16::from_f32(((i * 7 + step * 13) % 47) as f32 / 32.0 - 0.75);
        }
        linear.project(&input, &mut output)?;
        let expected = if let Some((up, down)) = &ffn_weights {
            let g = cpu_project(&weights, &input);
            let u = cpu_project(up, &input);
            let gated: Vec<_> = g
                .iter()
                .zip(u)
                .map(|(g, u)| {
                    let g = f16::from_f32(*g).to_f32();
                    let u = f16::from_f32(u).to_f32();
                    let activated = f16::from_f32(
                        0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
                    );
                    f16::from_f32(activated.to_f32() * u)
                })
                .collect();
            cpu_project(down, &gated)
        } else {
            cpu_project(&weights, &input)
        };
        for (expected, actual) in expected.into_iter().zip(&output) {
            let error = (actual.to_f32() - expected).abs();
            if !actual.is_finite() || error > 0.01 + 0.02 * expected.abs() {
                return Err(format!(
                    "ANE numerical mismatch: actual={actual}, expected={expected}, error={error}"
                )
                .into());
            }
            max_abs_error = max_abs_error.max(error);
            squared_error += f64::from(error).powi(2);
            squared_reference += f64::from(expected).powi(2);
        }
    }
    drop(weights);
    drop(ffn_weights);
    for _ in 0..5 {
        linear.project(&input, &mut output)?;
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut stages = Vec::with_capacity(if profile_stages { iterations } else { 0 });
    for _ in 0..iterations {
        let start = Instant::now();
        if profile_stages {
            let Projection::DynamicFfn(ffn) = &mut linear else {
                unreachable!()
            };
            let measured = ffn.project_profiled(&input, &mut output)?;
            stages.push(serde_json::json!({
                "input_ms": measured.input.as_secs_f64() * 1000.0,
                "evaluate_ms": measured.evaluate.as_secs_f64() * 1000.0,
                "output_ms": measured.output.as_secs_f64() * 1000.0,
            }));
        } else {
            linear.project(&input, &mut output)?;
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let result = serde_json::json!({
        "schema": "rvllm.ane_projection_probe.v1",
        "claim": "Direct ANE projection with CPU numerical reference; not full-model decode or Metal-to-ANE handoff",
        "model_dir": model_dir, "tensor": tensor,
        "operation": if ffn_prefix.is_some() { "gemma_gated_ffn" } else { "linear" },
        "ffn_prefix": ffn_prefix,
        "ffn_mode": ffn_prefix.as_ref().map(|_| if dynamic_ffn { "dynamic_resident_weights_explicit_gelu" } else { "compiled_constant_weights_explicit_gelu" }),
        "linear_mode": ffn_prefix.is_none().then_some(if dynamic_linear { "dynamic_resident_weights" } else { "compiled_constant_weights" }),
        "diagnostic_journal_enabled": std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some(),
        "stage_samples": stages,
        "prepare_includes_weight_packing": true,
        "weight_bytes": weight_bytes,
        "weight_source": if model_dir.is_some() { "checkpoint" } else { "synthetic" },
        "weight_f16_sha256": weight_sha256,
        "input_channels": input_channels, "output_channels": output_channels,
        "logical_spatial": spatial, "iterations": iterations,
        "compile_load_ms": compile_ms,
        "projection_including_io_median_ms": median,
        "projection_including_io_min_ms": samples[0],
        "projection_including_io_p95_ms": samples[(samples.len() * 95).div_ceil(100) - 1],
        "effective_weight_gb_per_s": weight_bytes as f64 / (median * 1e6),
        "numerical_inputs_checked": 3, "max_abs_error": max_abs_error,
        "relative_l2_error": (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt(),
        "ane_execution_verified": true, "cpu_or_gpu_fallback": false,
    });
    let json = serde_json::to_string_pretty(&result)?;
    if let Some(path) = report {
        std::fs::write(path, &json)?;
    }
    println!("{json}");
    Ok(())
}

fn hash_f16_weights(hash: &mut Sha256, weights: &[f16]) {
    let mut encoded = [0_u8; 4096];
    for chunk in weights.chunks(encoded.len() / 2) {
        for (slot, value) in encoded.chunks_exact_mut(2).zip(chunk) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        hash.update(&encoded[..chunk.len() * 2]);
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ANE projection probe failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
