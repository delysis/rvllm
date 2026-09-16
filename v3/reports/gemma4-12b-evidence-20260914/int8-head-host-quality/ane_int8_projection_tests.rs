//! Bounded actual-checkpoint QKV qualification; never changes the model route.
#![forbid(unsafe_code)]

use super::*;
use crate::apple_measurement::{compare_phase_measurements, PowerMonitor};
use rvllm_apple::ane_int8_ffn_weights::AneInt8LinearWeights;
use rvllm_apple::ane_linear::compile_budget_used;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[path = "ane_int8_head_tests.rs"]
mod head;

fn hash(values: &[f16]) -> String {
    let mut digest = Sha256::new();
    let mut bytes = [0_u8; 4096];
    for values in values.chunks(bytes.len() / 2) {
        for (value, slot) in values.iter().zip(bytes.chunks_exact_mut(2)) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        digest.update(&bytes[..values.len() * 2]);
    }
    format!("{:x}", digest.finalize())
}

fn cpu_projection(weights: &[f16], input: &[f16]) -> Vec<f32> {
    assert!(!input.is_empty() && weights.len() % input.len() == 0);
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

fn errors(actual: &[f32], reference: &[f32]) -> Value {
    assert_eq!(actual.len(), reference.len());
    assert!(!actual.is_empty());
    let mut maximum = 0.0_f32;
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut violations = 0;
    for (&a, &b) in actual.iter().zip(reference) {
        assert!(a.is_finite() && b.is_finite());
        let error = a - b;
        maximum = maximum.max(error.abs());
        squared_error += f64::from(error).powi(2);
        squared_reference += f64::from(b).powi(2);
        violations += usize::from(error.abs() > 0.01 + 0.02 * b.abs());
    }
    json!({"elements":actual.len(),"maximum_absolute_error":maximum,
        "relative_l2_error":(squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt(),
        "backend_tolerance_violations":violations})
}

#[test]
#[ignore = "actual Gemma QKV: explicit paths/mode; cpu zero device calls; restore-original one compile, zero evaluations; qualify at most two compiles; time strictly cached"]
fn checkpoint_qkv_int8_qualification() {
    run().unwrap();
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let model = PathBuf::from(std::env::var("RVLLM_GEMMA4_MODEL_DIR")?);
    let output = PathBuf::from(std::env::var("RVLLM_INT8_QKV_OUTPUT")?);
    let layer: usize = std::env::var("RVLLM_INT8_QKV_LAYER")?.parse()?;
    let mode = std::env::var("RVLLM_INT8_QKV_MODE")?;
    let sources: Vec<PathBuf> = serde_json::from_str(&std::env::var("RVLLM_INT8_QKV_INPUTS")?)?;
    if !matches!(
        mode.as_str(),
        "cpu" | "qualify" | "time" | "restore-original"
    ) || !(1..LAYERS).contains(&layer)
        || sources.is_empty()
        || sources.len() > 8
    {
        return Err(
            "requires layer 1..47, 1..8 captures, mode cpu|qualify|time|restore-original".into(),
        );
    }
    let journal = std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some();
    if (matches!(mode.as_str(), "qualify" | "restore-original") && !journal)
        || (mode == "time" && journal)
    {
        return Err("qualify/restore require the driver journal; time forbids it".into());
    }
    std::fs::create_dir(&output)?;
    assert_eq!(compile_budget_used(), 0);
    let (arch, entries) = validated_weights(&model, 1024)?;
    let shape = layer_shape(&arch, layer);
    let q = shape.query_heads * shape.head_dim;
    let kv = shape.kv_heads * shape.head_dim;
    let shared_value = shape.sliding_window.is_none();
    let prefix = format!("{}.layers.{layer}", arch.weight_prefix);
    let load = |suffix: &str| -> Result<Vec<f16>, String> {
        load_tensor(
            entries
                .get(&format!("{prefix}.{suffix}"))
                .ok_or("tensor missing")?,
        )
    };
    let mut original = load("self_attn.q_proj.weight")?;
    original.extend(load("self_attn.k_proj.weight")?);
    if !shared_value {
        original.extend(load("self_attn.v_proj.weight")?);
    }
    let rows = q + kv * if shared_value { 1 } else { 2 };
    assert_eq!(original.len(), HIDDEN * rows);
    let norm = load("input_layernorm.weight")?;
    let quantized = AneInt8LinearWeights::quantize(&original, HIDDEN, rows)?;
    let reconstructed = quantized.dequantized();
    let mut inputs = Vec::new();
    let mut input_receipts = Vec::new();
    for source in &sources {
        let filename = source
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("input filename")?;
        if !filename.starts_with("ane-position-")
            || !filename.ends_with(&format!("-layer-{:02}.fp16", layer - 1))
        {
            return Err(
                "input must be a post-layer capture immediately before target layer".into(),
            );
        }
        let bytes = std::fs::read(source)?;
        let mut input = vec![f16::ZERO; HIDDEN];
        decode_f16(&bytes, DType::F16, &mut input)?;
        rms_norm_f16_in_place(&mut input, HIDDEN, Some(&norm), arch.rms_norm_eps)?;
        input_receipts.push(
            json!({"file":source,"sha256":format!("{:x}",Sha256::digest(&bytes)),
            "normalized_fp16_sha256":hash(&input),
            "maximum_absolute_input":input.iter().map(|x|x.to_f32().abs()).fold(0.0_f32,f32::max)}),
        );
        inputs.push(input);
    }
    let original_cpu: Vec<_> = inputs
        .iter()
        .map(|x| cpu_projection(&original, x))
        .collect();
    let reconstructed_cpu: Vec<_> = inputs
        .iter()
        .map(|x| cpu_projection(&reconstructed, x))
        .collect();
    let mut ranges = vec![("query", 0, q), ("key", q, q + kv)];
    if !shared_value {
        ranges.push(("value", q + kv, rows));
    }
    let quantization_errors: Vec<_> = original_cpu
        .iter()
        .zip(&reconstructed_cpu)
        .map(|(a, b)| {
            let parts: Vec<_> = ranges.iter().map(|(name,start,end)| {
            json!({"part":name,"error":errors(&b[*start..*end],&a[*start..*end])})
        }).collect();
            json!({"parts":parts,"whole":errors(b,a)})
        })
        .collect();
    let mut report = json!({"schema":"rvllm.gemma4_qkv_int8.v1","mode":mode,"layer":layer,
        "model_dir":model,"input_channels":HIDDEN,"output_channels":rows,
        "shared_key_value":shared_value,"input_sources":input_receipts,
        "original_fp16_sha256":hash(&original),"reconstructed_fp16_sha256":hash(&reconstructed),
        "int8_source_bytes":quantized.source_blob_bytes(),"quantization_errors":quantization_errors,
        "backend_tolerance":"0.01 + 0.02 * abs(cpu_reference)",
        "claim":"One actual QKV projection; quantization error and backend error are separate. No full-model quality or speed claim."});
    // Persist CPU evidence before any private-framework operation.
    std::fs::write(
        output.join("host.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if mode == "cpu" {
        return Ok(());
    }
    if mode == "restore-original" {
        let program = AneLinear::compile_with_cache_policy(
            &original,
            HIDDEN,
            rows,
            1,
            AneProgramCachePolicy::ReuseOrCompileUpTo(1),
        )?;
        drop(program);
        report["compiler_calls"] = json!(compile_budget_used());
        report["evaluations"] = json!(0);
        report["models_dropped"] = json!(1);
        std::fs::write(
            output.join("result.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        return Ok(());
    }
    let policy = if mode == "qualify" {
        AneProgramCachePolicy::ReuseOrCompileUpTo(2)
    } else {
        AneProgramCachePolicy::RequireExisting
    };
    // Only the two new representations may compile; never repair the baseline.
    let baseline = AneLinear::compile_with_cache_policy(
        &original,
        HIDDEN,
        rows,
        1,
        AneProgramCachePolicy::RequireExisting,
    )?;
    let candidate = AneLinear::compile_int8_with_cache_policy(&quantized, 1, policy)?;
    let control = AneLinear::compile_with_cache_policy(&reconstructed, HIDDEN, rows, 1, policy)?;
    let mut programs = [baseline, candidate, control];
    let mut outputs: [Vec<f16>; 3] = std::array::from_fn(|_| vec![f16::ZERO; rows]);
    let mut backend_errors = Vec::new();
    let mut violations = 0_u64;
    for (index, input) in inputs.iter().enumerate() {
        let mut case = Vec::new();
        for (kind, (program, out)) in programs.iter_mut().zip(&mut outputs).enumerate() {
            program.project(input, out)?;
            let reference = if kind == 0 {
                &original_cpu[index]
            } else {
                &reconstructed_cpu[index]
            };
            let actual: Vec<_> = out.iter().map(|x| x.to_f32()).collect();
            let statistics = errors(&actual, reference);
            violations += statistics["backend_tolerance_violations"].as_u64().unwrap();
            let program_name = ["original", "int8", "reconstructed_dense"][kind];
            case.push(json!({"program":program_name,"error":statistics}));
        }
        backend_errors.push(json!(case));
    }
    report["backend_errors"] = json!(backend_errors);
    report["backend_violations"] = json!(violations);
    report["driver_journal_enabled"] = json!(journal);
    if mode == "time" && violations == 0 {
        report["timing"] = measure(&mut programs, &inputs[0], &mut outputs[0])?;
    }
    drop(programs);
    report["compiler_calls"] = json!(compile_budget_used());
    report["models_dropped"] = json!(3);
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if violations != 0 {
        return Err("QKV backend numerical gate failed; see result.json".into());
    }
    Ok(())
}

fn measure(
    programs: &mut [AneLinear; 3],
    input: &[f16],
    output: &mut [f16],
) -> Result<Value, String> {
    let monitor = PowerMonitor::start(None).map_err(|e| e.to_string())?;
    let preflight = monitor.begin().finish(1);
    let mut trials = Vec::new();
    let mut pairs = Vec::new();
    if preflight["sampled_controls_eligible"] != true {
        return Ok(
            json!({"preflight":preflight,"trials":trials,"pairs":pairs,"stopped":"ineligible_preflight"}),
        );
    }
    for program in programs.iter_mut() {
        for _ in 0..8 {
            program.project(input, output)?;
        }
    }
    let repetitions = 128;
    'comparisons: for baseline in [0, 2] {
        for block in 0..3 {
            let start = trials.len();
            let order = if block % 2 == 0 {
                [baseline, 1, 1, baseline]
            } else {
                [1, baseline, baseline, 1]
            };
            for program in order {
                let phase = monitor.begin();
                for _ in 0..repetitions {
                    programs[program].project(input, output)?;
                }
                let measurement = phase.finish(repetitions);
                let eligible = measurement["sampled_controls_eligible"] == true;
                trials.push(json!({"program":program,"measurement":measurement}));
                if !eligible {
                    break 'comparisons;
                }
            }
            for (a, b) in [(start, start + 1), (start + 3, start + 2)] {
                let (a, b) = if trials[a]["program"] == baseline {
                    (a, b)
                } else {
                    (b, a)
                };
                let comparison = compare_phase_measurements(
                    &trials[a]["measurement"],
                    &trials[b]["measurement"],
                );
                pairs.push(match comparison {
                    Ok(value) => json!({"baseline_program":baseline,"baseline_trial":a,"candidate_trial":b,"eligible":true,"comparison":value}),
                    Err(reason) => json!({"baseline_program":baseline,"baseline_trial":a,"candidate_trial":b,"eligible":false,"reason":reason}),
                });
            }
        }
    }
    Ok(
        json!({"preflight":preflight,"repetitions_per_trial":repetitions,"trials":trials,"pairs":pairs}),
    )
}
