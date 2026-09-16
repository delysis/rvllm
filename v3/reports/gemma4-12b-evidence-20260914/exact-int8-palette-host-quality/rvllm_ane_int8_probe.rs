//! Separate weight quantization error from ANE compressed representation error.
//! All inference remains Rust; no full-model quality claim is made by this probe.
#![forbid(unsafe_code)]

use half::f16;
use rvllm_apple::ane_int8_ffn_weights::AneInt8FfnWeights;
use rvllm_apple::ane_linear::{AneGatedFfn, AneProgramCachePolicy};
use rvllm_apple::ane_lut4_ffn_weights::AneLut4FfnWeights;
use rvllm_apple_metal::weight_loader::{load_safetensor_f16, scan_safetensor_tensors};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

const HIDDEN: usize = 3840;
const INTERMEDIATE: usize = 15360;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Original,
    DenseInt8,
    Int8,
    DenseLut4,
    Lut4,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::DenseInt8 => "dense-int8",
            Self::Int8 => "int8",
            Self::DenseLut4 => "dense-lut4",
            Self::Lut4 => "lut4",
        }
    }
}

enum QuantizedWeights {
    Int8(AneInt8FfnWeights),
    Lut4(AneLut4FfnWeights),
}

impl QuantizedWeights {
    fn dequantized(&self) -> [Vec<f16>; 3] {
        match self {
            Self::Int8(w) => w.dequantized(),
            Self::Lut4(w) => w.dequantized(),
        }
    }

    fn source_blob_bytes(&self) -> usize {
        match self {
            Self::Int8(w) => w.source_blob_bytes(),
            Self::Lut4(w) => w.source_blob_bytes(),
        }
    }

    fn compile(&self, policy: AneProgramCachePolicy) -> Result<AneGatedFfn, String> {
        match self {
            Self::Int8(w) => AneGatedFfn::compile_int8_with_cache_policy(w, policy),
            Self::Lut4(w) => AneGatedFfn::compile_lut4_with_cache_policy(w, policy),
        }
    }
}

#[derive(Default)]
struct ErrorStats {
    count: usize,
    maximum: f32,
    squared_error: f64,
    squared_reference: f64,
}

impl ErrorStats {
    fn observe(&mut self, actual: f32, reference: f32) -> Result<(), String> {
        if !actual.is_finite() || !reference.is_finite() {
            return Err("nonfinite numerical comparison".into());
        }
        let error = actual - reference;
        self.count += 1;
        self.maximum = self.maximum.max(error.abs());
        self.squared_error += f64::from(error).powi(2);
        self.squared_reference += f64::from(reference).powi(2);
        Ok(())
    }

    fn report(&self) -> serde_json::Value {
        json!({"elements":self.count,"max_abs_error":self.maximum,
            "relative_l2_error":(self.squared_error/self.squared_reference.max(f64::MIN_POSITIVE)).sqrt()})
    }
}

struct CpuFfn {
    gate: Vec<f16>,
    up: Vec<f16>,
    gated: Vec<f16>,
    output: Vec<f32>,
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

fn cpu_ffn(weights: &[Vec<f16>; 3], input: &[f16]) -> CpuFfn {
    let gate: Vec<_> = cpu_project(&weights[0], input)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let up: Vec<_> = cpu_project(&weights[1], input)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let gated: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| {
            let g = g.to_f32();
            let activated =
                f16::from_f32(0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()));
            f16::from_f32(activated.to_f32() * u.to_f32())
        })
        .collect();
    let output = cpu_project(&weights[2], &gated);
    CpuFfn {
        gate,
        up,
        gated,
        output,
    }
}

fn weight_hash(weights: &[Vec<f16>; 3]) -> String {
    let mut hash = Sha256::new();
    let mut bytes = [0_u8; 4096];
    for matrix in weights {
        for values in matrix.chunks(bytes.len() / 2) {
            for (value, slot) in values.iter().zip(bytes.chunks_exact_mut(2)) {
                slot.copy_from_slice(&value.to_le_bytes());
            }
            hash.update(&bytes[..values.len() * 2]);
        }
    }
    format!("{:x}", hash.finalize())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_dir = None;
    let mut report = None;
    let mut layer = 0_usize;
    let mut iterations = 12_usize;
    let mut mode = Mode::Int8;
    let mut cache_policy = AneProgramCachePolicy::Compile;
    let mut input_files = Vec::new();
    let mut compare = false;
    let mut cpu_only = false;
    let mut int8_lut_storage = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--report" => report = Some(PathBuf::from(value)),
            "--layer" => layer = value.parse()?,
            "--iterations" => iterations = value.parse()?,
            "--input-fp16" => input_files.push(PathBuf::from(value)),
            "--compare" => compare = value.parse()?,
            "--cpu-only" => cpu_only = value.parse()?,
            "--int8-lut-storage" => int8_lut_storage = value.parse()?,
            "--cache" => {
                cache_policy = match value.as_str() {
                    "compile" => AneProgramCachePolicy::Compile,
                    "reuse" => AneProgramCachePolicy::ReuseOrCompile,
                    "require" => AneProgramCachePolicy::RequireExisting,
                    _ => return Err("cache must be compile, reuse or require".into()),
                }
            }
            "--mode" => {
                mode = match value.as_str() {
                    "original" => Mode::Original,
                    "dense-int8" => Mode::DenseInt8,
                    "int8" => Mode::Int8,
                    "dense-lut4" => Mode::DenseLut4,
                    "lut4" => Mode::Lut4,
                    _ => {
                        return Err(
                            "mode must be original, dense-int8, int8, dense-lut4 or lut4".into(),
                        )
                    }
                }
            }
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    if layer >= 48 || !(1..=10000).contains(&iterations) {
        return Err("layer must be in 0..48 and iterations in 1..=10000".into());
    }
    if compare && !matches!(mode, Mode::Int8 | Mode::Lut4) {
        return Err("--compare requires mode int8 or lut4".into());
    }
    if int8_lut_storage && (!cpu_only || mode != Mode::Int8) {
        return Err(
            "INT8 palette representation audit requires --cpu-only true --mode int8".into(),
        );
    }
    let model_dir = model_dir.ok_or("--model-dir required")?;
    let report = report.ok_or("--report required")?;
    if report.exists() {
        return Err("report already exists".into());
    }
    let mut inputs = Vec::new();
    let mut input_sources = Vec::new();
    for step in 0..3 {
        inputs.push(
            (0..HIDDEN)
                .map(|i| f16::from_f32(((i * 7 + step * 13) % 47) as f32 / 32.0 - 0.75))
                .collect::<Vec<_>>(),
        );
        input_sources.push(json!({"synthetic_pattern":step}));
    }
    for path in input_files {
        let bytes = std::fs::read(&path)?;
        if bytes.len() != HIDDEN * 2 {
            return Err("input file must contain exactly 3840 little-endian FP16 values".into());
        }
        let values: Vec<_> = bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("input file contains nonfinite values".into());
        }
        input_sources.push(json!({"file":path,"sha256":format!("{:x}", Sha256::digest(&bytes))}));
        inputs.push(values);
    }
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let text = config.get("text_config").unwrap_or(&config);
    if text["hidden_activation"] != "gelu_pytorch_tanh"
        || text["hidden_size"] != HIDDEN
        || text["intermediate_size"] != INTERMEDIATE
        || text["num_hidden_layers"] != 48
    {
        return Err("probe requires the dense Gemma 4 12B tanh-GELU architecture".into());
    }
    let started = Instant::now();
    let entries = scan_safetensor_tensors(&model_dir)?;
    let load = |suffix: &str, shape: [usize; 2]| -> Result<Vec<f16>, Box<dyn std::error::Error>> {
        let name = format!("model.language_model.layers.{layer}.mlp.{suffix}.weight");
        if entries.get(&name).ok_or("FFN tensor missing")?.shape != shape {
            return Err(format!("invalid shape for {name}").into());
        }
        Ok(load_safetensor_f16(&model_dir, &name)?
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]))
            .collect())
    };
    let original = [
        load("gate_proj", [INTERMEDIATE, HIDDEN])?,
        load("up_proj", [INTERMEDIATE, HIDDEN])?,
        load("down_proj", [HIDDEN, INTERMEDIATE])?,
    ];
    let weight_load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let is_lut4 = matches!(mode, Mode::Lut4 | Mode::DenseLut4);
    let quantized = if is_lut4 {
        QuantizedWeights::Lut4(AneLut4FfnWeights::quantize(
            &original[0],
            &original[1],
            &original[2],
            HIDDEN,
            INTERMEDIATE,
        )?)
    } else {
        QuantizedWeights::Int8(AneInt8FfnWeights::quantize(
            &original[0],
            &original[1],
            &original[2],
            HIDDEN,
            INTERMEDIATE,
        )?)
    };
    let affine_reference = quantized.dequantized();
    let reconstructed = if int8_lut_storage {
        let QuantizedWeights::Int8(weights) = &quantized else {
            unreachable!()
        };
        let candidate = weights.dequantized_lut8_reference()?;
        for (expected, actual) in affine_reference.iter().zip(&candidate) {
            if !expected
                .iter()
                .zip(actual)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Err("INT8 LUT representation changed a reconstructed weight".into());
            }
        }
        candidate
    } else {
        affine_reference
    };
    let quantize_and_reconstruct_ms = started.elapsed().as_secs_f64() * 1000.0;
    let original_hash = weight_hash(&original);
    let reconstructed_hash = weight_hash(&reconstructed);
    let mut weight_errors = Vec::new();
    for (original, reconstructed) in original.iter().zip(&reconstructed) {
        let mut errors = ErrorStats::default();
        for (original, reconstructed) in original.iter().zip(reconstructed) {
            errors.observe(reconstructed.to_f32(), original.to_f32())?;
        }
        weight_errors.push(errors.report());
    }
    let source_blob_bytes = if int8_lut_storage {
        let QuantizedWeights::Int8(weights) = &quantized else {
            unreachable!()
        };
        weights.lut8_source_blob_bytes()?
    } else {
        quantized.source_blob_bytes()
    };
    if cpu_only {
        let mut per_input = Vec::new();
        for input in &inputs {
            let baseline = cpu_ffn(&original, input);
            let candidate = cpu_ffn(&reconstructed, input);
            let mut errors: [ErrorStats; 4] = std::array::from_fn(|_| ErrorStats::default());
            for (error, (candidate, baseline)) in errors[..3].iter_mut().zip(
                [&candidate.gate, &candidate.up, &candidate.gated]
                    .into_iter()
                    .zip([&baseline.gate, &baseline.up, &baseline.gated]),
            ) {
                for (actual, reference) in candidate.iter().zip(baseline) {
                    error.observe(actual.to_f32(), reference.to_f32())?;
                }
            }
            for (actual, reference) in candidate.output.iter().zip(&baseline.output) {
                errors[3].observe(*actual, *reference)?;
            }
            per_input.push(json!({"gate_up_gated_down":errors.map(|e|e.report()),
                "input_max_abs":input.iter().map(|v|v.to_f32().abs()).fold(0.0_f32,f32::max)}));
        }
        let result = json!({"schema":"rvllm.ane_ffn_host_quality.v1", "model_dir":model_dir,"layer":layer,
            "mode":mode.name(),"int8_lut_storage":int8_lut_storage,"exact_int8_weight_parity":if int8_lut_storage { Some(true) } else { None },"accelerator_calls":0,"compiler_calls":0,
            "original_fp16_sha256":original_hash,"dense_reconstruction_fp16_sha256":reconstructed_hash,
            "weight_load_ms":weight_load_ms,"quantize_and_reconstruct_ms":quantize_and_reconstruct_ms,
            "quantized_source_blob_bytes":source_blob_bytes,"weight_reconstruction_errors_gate_up_down":weight_errors,
            "input_sources":input_sources,"per_input_errors":per_input,
            "claim":"CPU quality rejection gate only. No accelerator execution, speedup, compiled residency or full-model quality claim. INT8 palette storage preserves all reconstructed coefficients exactly."});
        std::fs::write(report, serde_json::to_vec_pretty(&result)?)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    let selected = if mode == Mode::Original {
        &original
    } else {
        &reconstructed
    };
    eprintln!("Preparing {} FFN for layer {layer}", mode.name());
    let started = Instant::now();
    let mut ffn = if matches!(mode, Mode::Int8 | Mode::Lut4) {
        quantized.compile(cache_policy)?
    } else {
        AneGatedFfn::compile_with_cache_policy(
            &selected[0],
            &selected[1],
            &selected[2],
            HIDDEN,
            INTERMEDIATE,
            cache_policy,
        )?
    };
    let compile_load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut controls = if compare {
        let compile = |weights: &[Vec<f16>; 3]| {
            AneGatedFfn::compile_with_cache_policy(
                &weights[0],
                &weights[1],
                &weights[2],
                HIDDEN,
                INTERMEDIATE,
                cache_policy,
            )
        };
        Some([compile(&original)?, compile(&reconstructed)?])
    } else {
        None
    };
    let control_prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
    drop(quantized);
    let mut backend_error = ErrorStats::default();
    let mut total_error = ErrorStats::default();
    let mut quantization_error: [ErrorStats; 4] = std::array::from_fn(|_| ErrorStats::default());
    let mut actual = vec![f16::ZERO; HIDDEN];
    let mut per_input_errors = Vec::new();
    let mut control_errors: [ErrorStats; 2] = std::array::from_fn(|_| ErrorStats::default());
    let mut dense_control_identical = true;
    let mut control_output = vec![f16::ZERO; HIDDEN];
    for input in &inputs {
        ffn.project(input, &mut actual)?;
        let original_cpu = cpu_ffn(&original, input);
        let quantized_cpu = cpu_ffn(&reconstructed, input);
        if let Some(controls) = &mut controls {
            for (index, (control, reference)) in controls
                .iter_mut()
                .zip([&original_cpu.output, &quantized_cpu.output])
                .enumerate()
            {
                control.project(input, &mut control_output)?;
                for (&value, &expected) in control_output.iter().zip(reference) {
                    control_errors[index].observe(value.to_f32(), expected)?;
                    if (value.to_f32() - expected).abs() > 0.01 + 0.02 * expected.abs() {
                        return Err(format!(
                            "ANE control {index} mismatch: {value} versus {expected}"
                        )
                        .into());
                    }
                }
                if index == 1 {
                    dense_control_identical &= control_output
                        .iter()
                        .zip(&actual)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                }
            }
        }
        let expected = if mode == Mode::Original {
            &original_cpu.output
        } else {
            &quantized_cpu.output
        };
        let mut current_backend_error = ErrorStats::default();
        let mut current_total_error = ErrorStats::default();
        for (actual, expected) in actual.iter().zip(expected) {
            let actual = actual.to_f32();
            current_backend_error.observe(actual, *expected)?;
            backend_error.observe(actual, *expected)?;
            if (actual - expected).abs() > 0.01 + 0.02 * expected.abs() {
                return Err(
                    format!("ANE backend/reference mismatch: {actual} versus {expected}").into(),
                );
            }
        }
        for (actual, expected) in actual.iter().zip(&original_cpu.output) {
            current_total_error.observe(actual.to_f32(), *expected)?;
            total_error.observe(actual.to_f32(), *expected)?;
        }
        per_input_errors.push(json!({"backend":current_backend_error.report(),"against_original":current_total_error.report(),
            "input_max_abs":input.iter().map(|v|v.to_f32().abs()).fold(0.0_f32,f32::max)}));
        for (index, (a, b)) in [&quantized_cpu.gate, &quantized_cpu.up, &quantized_cpu.gated]
            .into_iter()
            .zip([&original_cpu.gate, &original_cpu.up, &original_cpu.gated])
            .enumerate()
        {
            for (a, b) in a.iter().zip(b) {
                quantization_error[index].observe(a.to_f32(), b.to_f32())?;
            }
        }
        for (a, b) in quantized_cpu.output.iter().zip(&original_cpu.output) {
            quantization_error[3].observe(*a, *b)?;
        }
    }
    drop(original);
    drop(reconstructed);
    let input = inputs.last().ok_or("missing numerical input")?;
    for _ in 0..5 {
        ffn.project(input, &mut actual)?;
        if let Some(controls) = &mut controls {
            for control in controls {
                control.project(input, &mut actual)?;
            }
        }
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut control_samples: [Vec<f64>; 2] =
        std::array::from_fn(|_| Vec::with_capacity(iterations));
    for iteration in 0..iterations {
        if let Some(controls) = &mut controls {
            // Rotate order every cycle. All programs, requests and buffers
            // remain resident; no weights or CPU references are loaded here.
            for offset in 0..3 {
                let index = (iteration + offset) % 3;
                let started = Instant::now();
                if index == 0 {
                    ffn.project(input, &mut actual)?;
                } else {
                    controls[index - 1].project(input, &mut actual)?;
                }
                let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                if index == 0 {
                    samples.push(elapsed);
                } else {
                    control_samples[index - 1].push(elapsed);
                }
            }
        } else {
            let started = Instant::now();
            ffn.project(input, &mut actual)?;
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let paired_samples = if compare {
        Some(
            json!({"selected":samples,"original":control_samples[0],"dense_reconstruction":control_samples[1]}),
        )
    } else {
        None
    };
    samples.sort_by(f64::total_cmp);
    let result = json!({
        "schema":"rvllm.ane_weight_ffn_probe.v2","mode":mode.name(),"model_dir":model_dir,"layer":layer,
        "quantization":if is_lut4 { "Three scalar per-tensor 16-entry FP16 codebooks; weighted FP16 histogram Lloyd fitting, quantile-midpoint initialization, up to 50 iterations; final assignment to nearest stored FP16 centroid; low nibble first; no format fallback" } else { "One stored FP16 scale per output row; scale=max_abs/127 rounded to FP16; integer=ties_even(weight/stored_scale), clamped to [-127,127]; zero rows use scale one" },
        "original_fp16_sha256":original_hash,"dense_reconstruction_fp16_sha256":reconstructed_hash,
        "weight_load_ms":weight_load_ms,"quantize_and_reconstruct_ms":quantize_and_reconstruct_ms,
        "compile_load_ms":compile_load_ms,"quantized_source_blob_bytes":source_blob_bytes,
        "control_prepare_ms":if compare { Some(control_prepare_ms) } else { None },
        "control_backend_errors_original_dense":if compare { Some(control_errors.map(|e|e.report())) } else { None },
        "dense_control_outputs_bit_identical":if compare { Some(dense_control_identical) } else { None },
        "rotating_order_samples_ms":paired_samples,
        "cache_policy":format!("{cache_policy:?}"),
        "original_weight_payload_bytes":3*HIDDEN*INTERMEDIATE*2,
        "weight_reconstruction_errors_gate_up_down":weight_errors,
        "cpu_quantization_errors_gate_up_gated_down":quantization_error.map(|e|e.report()),
        "ane_vs_selected_cpu_reference":backend_error.report(),"ane_vs_original_cpu_reference":total_error.report(),
        "numerical_inputs_checked":inputs.len(),"input_sources":input_sources,"per_input_errors":per_input_errors,
        "warmups":5,"iterations":iterations,"samples_ms":samples,
        "median_ms":(samples[(samples.len()-1)/2]+samples[samples.len()/2])*0.5,"minimum_ms":samples[0],"p95_ms":samples[(samples.len()*95).div_ceil(100)-1],
        "diagnostic_journal_enabled":std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some(),
        "ane_execution_verified":true,"cpu_or_gpu_fallback":false,"full_model_quality_qualified":false,
        "claim":"One real FFN numerical/latency experiment. Source blob size is not compiled/resident size or proof of compressed runtime bandwidth. Quantization errors are observed, not full-model quality acceptance."
    });
    std::fs::write(report, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ANE weight FFN probe failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
