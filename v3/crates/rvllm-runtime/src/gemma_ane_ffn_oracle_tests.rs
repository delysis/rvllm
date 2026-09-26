//! Bounded cached-only FFN component qualification on explicitly captured inputs.
//! These ignored fixtures are not run by CI or the delivery gate. No timing,
//! compilation fallback, model download, worker, or queue operation is provided.
#![forbid(unsafe_code)]

use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::PathBuf;

const SCHEMA: &str = "rvllm.ane.ffn-component-input.v1";
const BOUNDED_SCHEMA: &str = "rvllm.ane.ffn-layout-equivalence.v1";
const NEAR_ZERO: f64 = 1.0 / 64.0;
const NEAR_ZERO_ABS: f64 = 1.0 / 1024.0;
const MAX_ABS: f64 = 1.0 / 32.0;
const MAX_MATERIAL_ULP: u32 = 4;
const MAX_NEAR_ZERO_ULP: u32 = 1024;
const MAX_RELATIVE_L2: f64 = 1.0 / 1024.0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    candidate: String,
    layer: usize,
    model_config_sha256: String,
    // Canonical gate/up/down matrix digests, NOT source-file hashes. Each is
    // SHA256(u64 columns LE || u64 rows LE || i8 row bytes || f16 scales LE).
    quantized_matrix_sha256: [String; 3],
    samples: Vec<Sample>,
}

#[derive(Clone, Copy)]
enum Candidate {
    Chunk4,
    Down4,
    Interleaved,
}

impl Candidate {
    fn name(self) -> &'static str {
        match self {
            Self::Chunk4 => "ane-int8-ffn-chunk4",
            Self::Down4 => "ane-int8-ffn-down4",
            Self::Interleaved => "ane-int8-ffn-interleaved",
        }
    }

    fn control(self) -> &'static str {
        match self {
            Self::Interleaved => "stacked-int8",
            _ => "plain-int8",
        }
    }
}

fn valid_hash(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_input(raw: &[u8], candidate: Candidate) -> Result<Input, String> {
    let input: Input = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    if input.schema != SCHEMA
        || input.candidate != candidate.name()
        || input.layer >= LAYERS
        || !(1..=3).contains(&input.samples.len())
        || !valid_hash(&input.model_config_sha256)
        || input.quantized_matrix_sha256.iter().any(|h| !valid_hash(h))
        || input
            .samples
            .iter()
            .any(|s| !s.path.is_absolute() || !valid_hash(&s.sha256))
    {
        return Err("FFN component manifest shape, candidate or pins invalid".into());
    }
    for (i, sample) in input.samples.iter().enumerate() {
        if input.samples[..i]
            .iter()
            .any(|s| s.path == sample.path || s.sha256 == sample.sha256)
        {
            return Err("duplicate FFN sample is not independent coverage".into());
        }
    }
    Ok(input)
}

fn regular_bytes(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let metadata = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !metadata.file_type().is_file() || metadata.len() > limit as u64 {
        return Err("component input must be a bounded regular file".into());
    }
    let file = File::open(path).map_err(|e| e.to_string())?;
    if !file.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err("component input changed type".into());
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > limit {
        return Err("component input exceeds size bound".into());
    }
    Ok(bytes)
}

fn pinned_bytes(path: &Path, expected: &str, limit: usize) -> Result<Vec<u8>, String> {
    if !valid_hash(expected) {
        return Err("invalid input SHA256".into());
    }
    let bytes = regular_bytes(path, limit)?;
    if sha(&bytes) != expected {
        return Err("component input hash changed".into());
    }
    Ok(bytes)
}

fn half_input(bytes: &[u8]) -> Result<Vec<f16>, String> {
    if bytes.len() != HIDDEN * 2 {
        return Err("FFN input must contain exactly 3840 f16 values".into());
    }
    let values: Vec<_> = bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite captured FFN input".into());
    }
    Ok(values)
}

fn matrix_hash(matrix: rvllm_apple::ane_int8_ffn_weights::AneInt8MatrixView<'_>) -> String {
    let mut digest = Sha256::new();
    digest.update((matrix.columns as u64).to_le_bytes());
    digest.update((matrix.scales.len() as u64).to_le_bytes());
    // Bounded stack block, not another model-sized coefficient allocation.
    let mut block = [0_u8; 4096];
    for values in matrix.values.chunks(block.len()) {
        for (dst, value) in block.iter_mut().zip(values) {
            *dst = value.to_le_bytes()[0];
        }
        digest.update(&block[..values.len()]);
    }
    for scale in matrix.scales {
        digest.update(scale.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn record(file: &mut File, event: serde_json::Value) -> Result<(), String> {
    serde_json::to_writer(&mut *file, &event).map_err(|e| e.to_string())?;
    file.write_all(b"\n")
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())
}

fn output(directory: &Path, name: &str, values: &[f16]) -> Result<String, String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(name))
        .map_err(|e| e.to_string())?;
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(sha(&bytes))
}

fn ordered_f16(bits: u16) -> u32 {
    if bits & 0x8000 != 0 {
        u32::from(0x8000_u16 - (bits & 0x7fff))
    } else {
        u32::from(0x8000_u16.wrapping_add(bits))
    }
}

fn ulp_distance(a: f16, b: f16) -> u32 {
    if a == f16::ZERO && b == f16::ZERO {
        return 0;
    }
    ordered_f16(a.to_bits()).abs_diff(ordered_f16(b.to_bits()))
}

fn f16_spacing_toward(control: f16, candidate: f16) -> f64 {
    let bits = control.to_bits();
    if bits == 0 || bits == 0x8000 {
        return f64::from(f16::from_bits(1).to_f32());
    }
    let next_bits = if candidate.to_f32() >= control.to_f32() {
        if bits & 0x8000 == 0 {
            bits + 1
        } else {
            bits - 1
        }
    } else if bits & 0x8000 == 0 {
        bits - 1
    } else {
        bits + 1
    };
    f64::from((f16::from_bits(next_bits).to_f32() - control.to_f32()).abs())
}

#[derive(Debug)]
struct BoundedComparison {
    receipt: serde_json::Value,
    passed: bool,
    squared_error: f64,
    squared_reference: f64,
}

fn bounded_comparison(control: &[f16], candidate: &[f16]) -> BoundedComparison {
    let length_match = control.len() == candidate.len();
    let mut finite_control = true;
    let mut finite_candidate = true;
    let mut control_nan = 0_u64;
    let mut candidate_nan = 0_u64;
    let mut control_pos_inf = 0_u64;
    let mut control_neg_inf = 0_u64;
    let mut candidate_pos_inf = 0_u64;
    let mut candidate_neg_inf = 0_u64;
    let mut mismatch_count = 0_u64;
    let mut hybrid_violations = 0_u64;
    let mut material_ulp_violations = 0_u64;
    let mut near_zero_ulp_violations = 0_u64;
    let mut max_abs_error = 0.0_f64;
    let mut max_abs_index = None;
    let mut max_ulp_all = 0_u32;
    let mut max_ulp_all_index = None;
    let mut max_ulp_material = 0_u32;
    let mut max_ulp_material_index = None;
    let mut max_ulp_near_zero = 0_u32;
    let mut max_ulp_near_zero_index = None;
    let mut first_hybrid_violation = None;
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut histogram = [0_u64; 11];
    let common = control.len().min(candidate.len());
    for index in 0..common {
        let a = control[index];
        let b = candidate[index];
        let af = f64::from(a.to_f32());
        let bf = f64::from(b.to_f32());
        if !a.is_finite() {
            finite_control = false;
            if a.is_nan() {
                control_nan += 1;
            } else if af.is_sign_positive() {
                control_pos_inf += 1;
            } else {
                control_neg_inf += 1;
            }
        }
        if !b.is_finite() {
            finite_candidate = false;
            if b.is_nan() {
                candidate_nan += 1;
            } else if bf.is_sign_positive() {
                candidate_pos_inf += 1;
            } else {
                candidate_neg_inf += 1;
            }
        }
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        squared_reference += af * af;
        let error = (bf - af).abs();
        squared_error += error * error;
        let ulp = ulp_distance(a, b);
        let bucket = match ulp {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5..=8 => 5,
            9..=16 => 6,
            17..=64 => 7,
            65..=256 => 8,
            257..=1024 => 9,
            _ => 10,
        };
        histogram[bucket] += 1;
        if a.to_bits() != b.to_bits() {
            mismatch_count += 1;
        }
        if error > max_abs_error {
            max_abs_error = error;
            max_abs_index = Some(index);
        }
        if ulp > max_ulp_all {
            max_ulp_all = ulp;
            max_ulp_all_index = Some(index);
        }
        let material = af.abs().max(bf.abs()) >= NEAR_ZERO;
        if material {
            if ulp > max_ulp_material {
                max_ulp_material = ulp;
                max_ulp_material_index = Some(index);
            }
            if ulp > MAX_MATERIAL_ULP {
                material_ulp_violations += 1;
            }
        } else {
            if ulp > max_ulp_near_zero {
                max_ulp_near_zero = ulp;
                max_ulp_near_zero_index = Some(index);
            }
            if ulp > MAX_NEAR_ZERO_ULP || error > NEAR_ZERO_ABS {
                near_zero_ulp_violations += 1;
            }
        }
        let hybrid_limit = NEAR_ZERO_ABS.max(4.0 * f16_spacing_toward(a, b));
        if error > hybrid_limit {
            hybrid_violations += 1;
            first_hybrid_violation.get_or_insert(index);
        }
    }
    let denominator = squared_reference.max(common as f64 * (2.0_f64).powi(-28));
    let relative_l2 = (squared_error / denominator).sqrt();
    let absolute_pass = max_abs_error <= MAX_ABS;
    let relative_l2_pass = relative_l2 <= MAX_RELATIVE_L2;
    let passed = length_match
        && common > 0
        && finite_control
        && finite_candidate
        && hybrid_violations == 0
        && material_ulp_violations == 0
        && near_zero_ulp_violations == 0
        && absolute_pass
        && relative_l2_pass;
    let at = |index: Option<usize>| {
        index.map(|i| {
            serde_json::json!({
        "index":i, "control":control[i].to_f32(), "candidate":candidate[i].to_f32(),
        "control_bits":format!("{:04x}",control[i].to_bits()),
        "candidate_bits":format!("{:04x}",candidate[i].to_bits())})
        })
    };
    BoundedComparison {
        receipt: serde_json::json!({
            "schema":BOUNDED_SCHEMA, "passed":passed, "length_match":length_match,
            "control_length":control.len(), "candidate_length":candidate.len(),
            "finite_control":finite_control, "finite_candidate":finite_candidate,
            "nonfinite":{"control_nan":control_nan,"candidate_nan":candidate_nan,
                "control_pos_inf":control_pos_inf,"control_neg_inf":control_neg_inf,
                "candidate_pos_inf":candidate_pos_inf,"candidate_neg_inf":candidate_neg_inf},
            "bit_exact":length_match && mismatch_count == 0, "mismatch_count":mismatch_count,
            "mismatch_fraction":if common == 0 { 0.0 } else { mismatch_count as f64/common as f64 },
            "max_abs_error":max_abs_error,"max_abs_location":at(max_abs_index),
            "relative_l2":relative_l2,"squared_error":squared_error,
            "squared_reference":squared_reference,
            "max_ulp_all":max_ulp_all,"max_ulp_all_location":at(max_ulp_all_index),
            "max_ulp_material":max_ulp_material,"max_ulp_material_location":at(max_ulp_material_index),
            "max_ulp_near_zero":max_ulp_near_zero,"max_ulp_near_zero_location":at(max_ulp_near_zero_index),
            "ulp_histogram":{"0":histogram[0],"1":histogram[1],"2":histogram[2],"3":histogram[3],"4":histogram[4],
                "5_8":histogram[5],"9_16":histogram[6],"17_64":histogram[7],"65_256":histogram[8],
                "257_1024":histogram[9],"gt1024":histogram[10]},
            "hybrid_violation_count":hybrid_violations,"first_hybrid_violation":first_hybrid_violation,
            "material_ulp_violation_count":material_ulp_violations,
            "near_zero_ulp_violation_count":near_zero_ulp_violations,
            "absolute_pass":absolute_pass,"relative_l2_pass":relative_l2_pass}),
        passed,
        squared_error,
        squared_reference,
    }
}

fn load_component_weights(
    model_dir: &Path,
    layer: usize,
    config_sha256: &str,
) -> Result<AneInt8FfnWeights, String> {
    if layer >= LAYERS {
        return Err("component layer out of range".into());
    }
    pinned_bytes(&model_dir.join("config.json"), config_sha256, 1_048_576)?;
    let (arch, entries) = validated_weights(model_dir, 1024)?;
    let prefix = format!("{}.layers.{layer}.mlp", arch.weight_prefix);
    let load = |name: &str| -> Result<Vec<f16>, String> {
        load_tensor(
            entries
                .get(&format!("{prefix}.{name}_proj.weight"))
                .ok_or("missing FFN weight")?,
        )
    };
    let gate = load("gate")?;
    let up = load("up")?;
    let down = load("down")?;
    let weights = AneInt8FfnWeights::quantize(&gate, &up, &down, HIDDEN, INTERMEDIATE)?;
    drop((gate, up, down));
    pinned_bytes(&model_dir.join("config.json"), config_sha256, 1_048_576)?;
    Ok(weights)
}

fn run(
    candidate: Candidate,
    policy: AneProgramCachePolicy,
    cache_policy: &'static str,
) -> Result<(), String> {
    let variable = |name| std::env::var(name).map_err(|_| format!("explicit {name} required"));
    let model_dir = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_MODEL_DIR")?);
    let manifest = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_MANIFEST")?);
    let expected = variable("RVLLM_ANE_FFN_ORACLE_MANIFEST_SHA256")?;
    let directory = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_OUTPUT")?);
    let journal = PathBuf::from(variable("RVLLM_ANE_DIAGNOSTIC_JOURNAL")?);
    if [&model_dir, &manifest, &directory, &journal]
        .iter()
        .any(|p| !p.is_absolute())
    {
        return Err("component paths must be absolute".into());
    }
    let raw = pinned_bytes(&manifest, &expected, 65536)?;
    let input = validate_input(&raw, candidate)?;
    let mut samples = Vec::new();
    for sample in &input.samples {
        samples.push(half_input(&pinned_bytes(
            &sample.path,
            &sample.sha256,
            HIDDEN * 2,
        )?)?);
    }
    // Complete host loading, quantization and all input/weight pin checks before
    // creating the first ANE owner. The same object supplies BOTH layouts.
    let weights = load_component_weights(&model_dir, input.layer, &input.model_config_sha256)?;
    let observed = weights.matrices().map(matrix_hash);
    if observed != input.quantized_matrix_sha256 {
        return Err("FFN coefficient pins disagree".into());
    }
    std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
    let mut events = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("events.jsonl"))
        .map_err(|e| e.to_string())?;
    record(
        &mut events,
        serde_json::json!({"event":"begin", "schema":SCHEMA,
        "candidate":candidate.name(), "control":candidate.control(), "layer":input.layer,
        "manifest_sha256":expected, "quantized_matrix_sha256":observed,
        "samples":samples.len(), "maximum_program_evaluations":2*samples.len(),
        "cache_policy":cache_policy, "timing":false,
        "driver_journal":journal, "driver_lifecycle_validation":"required-separately"}),
    )?;
    let result = (|| -> Result<(), String> {
        record(
            &mut events,
            serde_json::json!({"event":"load-control-begin"}),
        )?;
        let mut control = match candidate {
            Candidate::Interleaved => {
                AneGatedFfn::compile_int8_stacked_with_cache_policy(&weights, policy)?
            }
            _ => AneGatedFfn::compile_int8_with_cache_policy(&weights, policy)?,
        };
        record(
            &mut events,
            serde_json::json!({"event":"load-candidate-begin"}),
        )?;
        let mut proposed = match candidate {
            Candidate::Chunk4 => {
                AneGatedFfn::compile_int8_chunk4_with_cache_policy(&weights, policy)?
            }
            Candidate::Down4 => {
                AneGatedFfn::compile_int8_down4_with_cache_policy(&weights, policy)?
            }
            Candidate::Interleaved => {
                AneGatedFfn::compile_int8_interleaved_with_cache_policy(&weights, policy)?
            }
        };
        let mut a = vec![f16::ZERO; HIDDEN];
        let mut b = vec![f16::ZERO; HIDDEN];
        for (index, sample) in samples.iter().enumerate() {
            record(
                &mut events,
                serde_json::json!({"event":"control-evaluate-begin","sample":index}),
            )?;
            control.project(sample, &mut a)?;
            let a_hash = output(&directory, &format!("sample-{index}-control.f16"), &a)?;
            record(
                &mut events,
                serde_json::json!({"event":"candidate-evaluate-begin","sample":index,
                "control_sha256":a_hash}),
            )?;
            proposed.project(sample, &mut b)?;
            let b_hash = output(&directory, &format!("sample-{index}-candidate.f16"), &b)?;
            // Reuse the incumbent bitwise/finite comparison, without widening
            // tolerances for a new layout. Its historical error prefix says
            // stacked; this receipt identifies the actual candidate/control.
            let matched = check_ffn_output(input.layer, &a, &b);
            record(
                &mut events,
                serde_json::json!({"event":"comparison","sample":index,
                "input_sha256":input.samples[index].sha256, "control_sha256":a_hash,
                "candidate_sha256":b_hash, "bit_exact_finite":matched.is_ok()}),
            )?;
            matched?;
        }
        drop(proposed);
        drop(control);
        Ok(())
    })();
    record(
        &mut events,
        serde_json::json!({"event":"end", "matched_all_inputs":result.is_ok(),
        "error":result.as_ref().err(), "promotion":false,
        "compiler_calls":rvllm_apple::ane_linear::compile_budget_used(),
        "driver_lifecycle_validation":"required-separately", "performance_qualified":false}),
    )?;
    result
}

fn run_chunk4_bounded_equivalence(
    policy: AneProgramCachePolicy,
    cache_policy: &'static str,
) -> Result<(), String> {
    let variable = |name| std::env::var(name).map_err(|_| format!("explicit {name} required"));
    let model_dir = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_MODEL_DIR")?);
    let manifest = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_MANIFEST")?);
    let expected = variable("RVLLM_ANE_FFN_ORACLE_MANIFEST_SHA256")?;
    let directory = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_OUTPUT")?);
    let journal = PathBuf::from(variable("RVLLM_ANE_DIAGNOSTIC_JOURNAL")?);
    if [&model_dir, &manifest, &directory, &journal]
        .iter()
        .any(|p| !p.is_absolute())
    {
        return Err("component paths must be absolute".into());
    }
    let raw = pinned_bytes(&manifest, &expected, 65536)?;
    let input = validate_input(&raw, Candidate::Chunk4)?;
    let mut samples = Vec::new();
    for sample in &input.samples {
        samples.push(half_input(&pinned_bytes(
            &sample.path,
            &sample.sha256,
            HIDDEN * 2,
        )?)?);
    }
    let weights = load_component_weights(&model_dir, input.layer, &input.model_config_sha256)?;
    let observed = weights.matrices().map(matrix_hash);
    if observed != input.quantized_matrix_sha256 {
        return Err("FFN coefficient pins disagree".into());
    }
    std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
    let mut events = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("events.jsonl"))
        .map_err(|e| e.to_string())?;
    let compiler_calls_before = rvllm_apple::ane_linear::compile_budget_used();
    record(
        &mut events,
        serde_json::json!({
        "event":"begin", "schema":BOUNDED_SCHEMA, "input_schema":SCHEMA,
        "candidate":Candidate::Chunk4.name(), "control":Candidate::Chunk4.control(),
        "layer":input.layer, "manifest_sha256":expected,
        "model_config_sha256":input.model_config_sha256,
        "quantized_matrix_sha256":observed, "samples":samples.len(),
        "maximum_program_evaluations":4*samples.len(), "cache_policy":cache_policy,
        "compiler_calls_before":compiler_calls_before,
        "output":{"dtype":"binary16", "length":HIDDEN, "layout":"contiguous-hidden"},
        "contract":{"finite":true,"hybrid":"abs_error <= max(2^-10, 4*ulp_toward(control))",
            "max_absolute":MAX_ABS,"material_threshold":NEAR_ZERO,
            "max_material_ulp":MAX_MATERIAL_ULP,"max_near_zero_ulp":MAX_NEAR_ZERO_ULP,
            "near_zero_max_absolute":NEAR_ZERO_ABS,"max_relative_l2":MAX_RELATIVE_L2,
            "relative_l2_denominator_floor":"sqrt(n)*2^-14",
            "signed_zero_ulp_distance":0,"repeatability":"bit-exact-finite"},
        "timing":false, "driver_journal":journal,
        "driver_lifecycle_validation":"required-separately"}),
    )?;
    let result = (|| -> Result<(), String> {
        record(
            &mut events,
            serde_json::json!({"event":"load-control-begin"}),
        )?;
        let mut control = AneGatedFfn::compile_int8_with_cache_policy(&weights, policy)?;
        record(
            &mut events,
            serde_json::json!({"event":"load-candidate-begin"}),
        )?;
        let mut candidate = AneGatedFfn::compile_int8_chunk4_with_cache_policy(&weights, policy)?;
        let mut control_a = vec![f16::ZERO; HIDDEN];
        let mut control_b = vec![f16::ZERO; HIDDEN];
        let mut candidate_a = vec![f16::ZERO; HIDDEN];
        let mut candidate_b = vec![f16::ZERO; HIDDEN];
        let mut all_passed = true;
        let mut aggregate_squared_error = 0.0_f64;
        let mut aggregate_squared_reference = 0.0_f64;
        let mut completed = 0_usize;
        for (index, sample) in samples.iter().enumerate() {
            record(
                &mut events,
                serde_json::json!({"event":"sample-evaluate-begin","sample":index}),
            )?;
            control.project(sample, &mut control_a)?;
            control.project(sample, &mut control_b)?;
            candidate.project(sample, &mut candidate_a)?;
            candidate.project(sample, &mut candidate_b)?;
            let control_a_hash = output(
                &directory,
                &format!("sample-{index}-control-a.f16"),
                &control_a,
            )?;
            let control_b_hash = output(
                &directory,
                &format!("sample-{index}-control-b.f16"),
                &control_b,
            )?;
            let candidate_a_hash = output(
                &directory,
                &format!("sample-{index}-candidate-a.f16"),
                &candidate_a,
            )?;
            let candidate_b_hash = output(
                &directory,
                &format!("sample-{index}-candidate-b.f16"),
                &candidate_b,
            )?;
            let control_repeat = bounded_comparison(&control_a, &control_b);
            let candidate_repeat = bounded_comparison(&candidate_a, &candidate_b);
            let comparison = bounded_comparison(&control_a, &candidate_a);
            let control_repeat_bit_exact = control_repeat.receipt["bit_exact"] == true
                && control_repeat.receipt["finite_control"] == true
                && control_repeat.receipt["finite_candidate"] == true;
            let candidate_repeat_bit_exact = candidate_repeat.receipt["bit_exact"] == true
                && candidate_repeat.receipt["finite_control"] == true
                && candidate_repeat.receipt["finite_candidate"] == true;
            let sample_passed =
                comparison.passed && control_repeat_bit_exact && candidate_repeat_bit_exact;
            all_passed &= sample_passed;
            aggregate_squared_error += comparison.squared_error;
            aggregate_squared_reference += comparison.squared_reference;
            completed += 1;
            record(
                &mut events,
                serde_json::json!({
                "event":"comparison", "sample":index,
                "input_sha256":input.samples[index].sha256,
                "outputs":{"control_a_sha256":control_a_hash,"control_b_sha256":control_b_hash,
                    "candidate_a_sha256":candidate_a_hash,"candidate_b_sha256":candidate_b_hash},
                "control_repeat_bit_exact_finite":control_repeat_bit_exact,
                "candidate_repeat_bit_exact_finite":candidate_repeat_bit_exact,
                "control_repeat":control_repeat.receipt,
                "candidate_repeat":candidate_repeat.receipt,
                "layout_equivalence":comparison.receipt,
                "sample_passed":sample_passed}),
            )?;
        }
        let denominator =
            aggregate_squared_reference.max(completed as f64 * HIDDEN as f64 * (2.0_f64).powi(-28));
        let aggregate_relative_l2 = (aggregate_squared_error / denominator).sqrt();
        let aggregate_passed =
            all_passed && completed == samples.len() && aggregate_relative_l2 <= MAX_RELATIVE_L2;
        record(
            &mut events,
            serde_json::json!({
            "event":"aggregate-comparison", "samples_expected":samples.len(),
            "samples_completed":completed,"squared_error":aggregate_squared_error,
            "squared_reference":aggregate_squared_reference,
            "relative_l2":aggregate_relative_l2,"relative_l2_limit":MAX_RELATIVE_L2,
            "all_sample_contracts_passed":all_passed,"passed":aggregate_passed}),
        )?;
        drop(candidate);
        drop(control);
        if !aggregate_passed {
            return Err(
                "Chunk4 bounded-equivalence contract failed; see immutable per-sample receipts"
                    .into(),
            );
        }
        Ok(())
    })();
    let compiler_calls_after = rvllm_apple::ane_linear::compile_budget_used();
    let compiler_calls_delta = compiler_calls_after.checked_sub(compiler_calls_before);
    record(
        &mut events,
        serde_json::json!({
        "event":"end", "matched_all_inputs":result.is_ok(), "error":result.as_ref().err(),
        "promotion":false,"compiler_calls_before":compiler_calls_before,
        "compiler_calls_after":compiler_calls_after,"compiler_calls_delta":compiler_calls_delta,
        "driver_lifecycle_validation":"required-separately", "performance_qualified":false}),
    )?;
    result
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PinRequest {
    schema: String,
    candidate: String,
    layer: usize,
    sample_paths: Vec<PathBuf>,
}

fn validate_pin_request(bytes: &[u8]) -> Result<(PinRequest, Candidate), String> {
    let request: PinRequest = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let candidate = match request.candidate.as_str() {
        "ane-int8-ffn-chunk4" => Candidate::Chunk4,
        "ane-int8-ffn-down4" => Candidate::Down4,
        "ane-int8-ffn-interleaved" => Candidate::Interleaved,
        _ => return Err("unknown FFN pin-preparation candidate".into()),
    };
    if request.schema != "rvllm.ane.ffn-pin-request.v1"
        || request.layer >= LAYERS
        || !(1..=3).contains(&request.sample_paths.len())
        || request.sample_paths.iter().any(|path| !path.is_absolute())
        || request
            .sample_paths
            .iter()
            .enumerate()
            .any(|(i, path)| request.sample_paths[..i].contains(path))
    {
        return Err("FFN pin request shape, layer or paths invalid".into());
    }
    Ok((request, candidate))
}

fn prepare_input_pins() -> Result<(), String> {
    let variable = |name| std::env::var(name).map_err(|_| format!("explicit {name} required"));
    let model_dir = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_MODEL_DIR")?);
    let request_path = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_PIN_REQUEST")?);
    let directory = PathBuf::from(variable("RVLLM_ANE_FFN_ORACLE_PIN_OUTPUT")?);
    if [&model_dir, &request_path, &directory]
        .iter()
        .any(|p| !p.is_absolute())
    {
        return Err("pin-preparation paths must be absolute".into());
    }
    let request_bytes = regular_bytes(&request_path, 65536)?;
    let (request, candidate) = validate_pin_request(&request_bytes)?;
    let config_hash = sha(&regular_bytes(&model_dir.join("config.json"), 1_048_576)?);
    let mut samples = Vec::new();
    for path in &request.sample_paths {
        let bytes = regular_bytes(path, HIDDEN * 2)?;
        half_input(&bytes)?;
        samples.push(serde_json::json!({"path":path, "sha256":sha(&bytes)}));
    }
    let weights = load_component_weights(&model_dir, request.layer, &config_hash)?;
    let matrices = weights.matrices().map(matrix_hash);
    let manifest = serde_json::json!({"schema":SCHEMA, "candidate":candidate.name(),
        "layer":request.layer, "model_config_sha256":config_hash,
        "quantized_matrix_sha256":matrices, "samples":samples});
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    validate_input(&manifest_bytes, candidate)?;
    std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
    let mut file = File::create_new(directory.join("manifest.json")).map_err(|e| e.to_string())?;
    file.write_all(&manifest_bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    let report = serde_json::json!({"role":"host-only-FFN-input-pin-preparation",
        "request_sha256":sha(&request_bytes), "manifest_sha256":sha(&manifest_bytes),
        "candidate":candidate.name(), "control":candidate.control(),
        "ane_initialization_requested":false, "hardware_qualification":false,
        "promotion_authorized":false});
    let mut receipt =
        File::create_new(directory.join("host-pin-receipt.json")).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut receipt, &report).map_err(|e| e.to_string())?;
    receipt.sync_all().map_err(|e| e.to_string())?;
    Ok(())
}

#[test]
#[ignore = "explicit host-only model/activation pin preparation; never initializes ANE"]
fn host_prepare_ffn_component_pins() -> Result<(), String> {
    prepare_input_pins()
}

#[test]
#[ignore = "explicit cached-only real-input ANE chunk4 component oracle; no timing"]
fn native_chunk4_matches_plain_cached_ffn() -> Result<(), String> {
    run(
        Candidate::Chunk4,
        AneProgramCachePolicy::RequireExisting,
        "RequireExisting",
    )
}

#[test]
#[ignore = "explicit cached-only real-input ANE down4 component oracle; no timing"]
fn native_down4_matches_plain_cached_ffn() -> Result<(), String> {
    run(
        Candidate::Down4,
        AneProgramCachePolicy::RequireExisting,
        "RequireExisting",
    )
}

#[test]
#[ignore = "explicit cached-only real-input ANE interleaved component oracle; no timing"]
fn native_interleaved_matches_stacked_cached_ffn() -> Result<(), String> {
    run(
        Candidate::Interleaved,
        AneProgramCachePolicy::RequireExisting,
        "RequireExisting",
    )
}

#[test]
#[ignore = "explicit queued ANE Chunk4 provision plus real-input comparison; at most two compiles"]
fn native_chunk4_bounded_provision_matches_plain_ffn() -> Result<(), String> {
    run(
        Candidate::Chunk4,
        AneProgramCachePolicy::ReuseOrCompileUpTo(2),
        "ReuseOrCompileUpTo(2)",
    )
}

#[test]
#[ignore = "explicit queued ANE Chunk4 prospective bounded-equivalence oracle; at most two compiles"]
fn native_chunk4_bounded_equivalence_matches_plain_ffn() -> Result<(), String> {
    run_chunk4_bounded_equivalence(
        AneProgramCachePolicy::ReuseOrCompileUpTo(2),
        "ReuseOrCompileUpTo(2)",
    )
}

#[test]
#[ignore = "explicit queued ANE Down4 provision plus real-input comparison; at most two compiles"]
fn native_down4_bounded_provision_matches_plain_ffn() -> Result<(), String> {
    run(
        Candidate::Down4,
        AneProgramCachePolicy::ReuseOrCompileUpTo(2),
        "ReuseOrCompileUpTo(2)",
    )
}

#[test]
#[ignore = "explicit queued ANE interleaved provision plus real-input comparison; at most two compiles"]
fn native_interleaved_bounded_provision_matches_stacked_ffn() -> Result<(), String> {
    run(
        Candidate::Interleaved,
        AneProgramCachePolicy::ReuseOrCompileUpTo(2),
        "ReuseOrCompileUpTo(2)",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positive(value: f32, ulps: u16) -> f16 {
        let base = f16::from_f32(value);
        f16::from_bits(base.to_bits() + ulps)
    }

    #[test]
    fn bounded_equivalence_material_ulp_boundary_is_exact() {
        let control = vec![f16::from_f32(1.0); 16];
        let mut candidate = control.clone();
        candidate[0] = positive(1.0, 4);
        let at_limit = bounded_comparison(&control, &candidate);
        assert!(at_limit.passed, "{}", at_limit.receipt);
        assert_eq!(at_limit.receipt["max_ulp_material"], 4);
        candidate[0] = positive(1.0, 5);
        let over = bounded_comparison(&control, &candidate);
        assert!(!over.passed);
        assert_eq!(over.receipt["material_ulp_violation_count"], 1);
    }

    #[test]
    fn bounded_equivalence_near_zero_ulp_boundary_is_exact() {
        let control = [f16::ZERO, f16::from_f32(1.0)];
        let at_limit = bounded_comparison(&control, &[f16::from_bits(1024), control[1]]);
        assert!(at_limit.passed, "{}", at_limit.receipt);
        assert_eq!(at_limit.receipt["max_ulp_near_zero"], 1024);
        let over = bounded_comparison(&control, &[f16::from_bits(1025), control[1]]);
        assert!(!over.passed);
        assert_eq!(over.receipt["near_zero_ulp_violation_count"], 1);
        assert!(over.receipt["max_abs_error"].as_f64().unwrap() < NEAR_ZERO_ABS);
    }

    #[test]
    fn bounded_equivalence_absolute_boundary_is_independent() {
        let control = [f16::from_f32(32.0)];
        let at_limit = bounded_comparison(&control, &[f16::from_f32(32.03125)]);
        assert!(at_limit.passed, "{}", at_limit.receipt);
        assert_eq!(at_limit.receipt["max_abs_error"], MAX_ABS);
        let over = bounded_comparison(&control, &[f16::from_f32(32.0625)]);
        assert!(!over.passed);
        assert_eq!(over.receipt["absolute_pass"], false);
        assert_eq!(over.receipt["material_ulp_violation_count"], 0);
    }

    #[test]
    fn bounded_equivalence_relative_l2_boundary_is_independent() {
        let control = [f16::from_f32(1.0), f16::from_f32(1.0)];
        let at_limit = bounded_comparison(&control, &[positive(1.0, 1), positive(1.0, 1)]);
        assert!(at_limit.passed, "{}", at_limit.receipt);
        assert_eq!(at_limit.receipt["relative_l2"], MAX_RELATIVE_L2);
        let over = bounded_comparison(&control, &[positive(1.0, 1), positive(1.0, 2)]);
        assert!(!over.passed);
        assert_eq!(over.receipt["relative_l2_pass"], false);
        assert_eq!(over.receipt["material_ulp_violation_count"], 0);
        assert_eq!(over.receipt["hybrid_violation_count"], 0);
    }

    #[test]
    fn bounded_equivalence_rejects_nonfinite_length_and_large_near_zero_error() {
        for bad in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
            let result = bounded_comparison(&[f16::ZERO], &[bad]);
            assert!(!result.passed);
        }
        assert!(!bounded_comparison(&[f16::ZERO], &[]).passed);
        assert!(!bounded_comparison(&[], &[]).passed);
        let sign_flip = bounded_comparison(&[f16::from_f32(0.000_5)], &[f16::from_f32(-0.000_6)]);
        assert!(!sign_flip.passed);
        assert_eq!(sign_flip.receipt["near_zero_ulp_violation_count"], 1);
    }

    #[test]
    fn bounded_equivalence_signed_zero_is_bit_distinct_but_zero_ulp() {
        let result = bounded_comparison(&[f16::ZERO], &[f16::NEG_ZERO]);
        assert!(result.passed, "{}", result.receipt);
        assert_eq!(result.receipt["bit_exact"], false);
        assert_eq!(result.receipt["max_ulp_all"], 0);
    }

    #[test]
    fn bounded_equivalence_ulp_order_is_contiguous_across_signs() {
        let negative_min = f16::from_bits(0x8001);
        let positive_min = f16::from_bits(0x0001);
        assert_eq!(ulp_distance(negative_min, f16::NEG_ZERO), 1);
        assert_eq!(ulp_distance(f16::NEG_ZERO, f16::ZERO), 0);
        assert_eq!(ulp_distance(f16::ZERO, positive_min), 1);
        assert_eq!(ulp_distance(negative_min, positive_min), 2);

        let negative_one = f16::from_f32(-1.0);
        assert_eq!(
            ulp_distance(negative_one, f16::from_bits(negative_one.to_bits() + 1)),
            1
        );
    }

    #[test]
    fn bounded_equivalence_rejects_layout_permutation() {
        let control = [f16::from_f32(1.0), f16::from_f32(2.0)];
        let result = bounded_comparison(&control, &[control[1], control[0]]);
        assert!(!result.passed);
        assert_eq!(result.receipt["mismatch_count"], 2);
        assert!(result.receipt["hybrid_violation_count"].as_u64().unwrap() > 0);
    }

    fn fixture() -> serde_json::Value {
        serde_json::json!({"schema":SCHEMA,"candidate":"ane-int8-ffn-interleaved", "layer":0,
            "model_config_sha256":"a".repeat(64), "quantized_matrix_sha256":["b".repeat(64),"c".repeat(64),"d".repeat(64)],
            "samples":[{"path":"/nonexistent/captured-input.f16","sha256":"e".repeat(64)}]})
    }
    #[test]
    fn host_pin_request_is_bounded_and_cannot_select_an_unknown_or_duplicate_case() {
        let request = serde_json::json!({"schema": "rvllm.ane.ffn-pin-request.v1",
            "candidate": "ane-int8-ffn-interleaved", "layer": 0,
            "sample_paths": ["/captured/real-input.f16"]});
        let parse =
            |value: &serde_json::Value| validate_pin_request(&serde_json::to_vec(value).unwrap());
        assert!(parse(&request).is_ok());
        for (key, value) in [
            ("candidate", serde_json::json!("auto")),
            ("layer", serde_json::json!(48)),
            ("sample_paths", serde_json::json!([])),
            ("sample_paths", serde_json::json!(["/same", "/same"])),
            ("sample_paths", serde_json::json!(["relative.f16"])),
        ] {
            let mut mutant = request.clone();
            mutant[key] = value;
            assert!(parse(&mutant).is_err());
        }
    }

    #[test]
    fn manifest_refuses_wrong_candidate_unbounded_samples_and_unpinned_coefficients() {
        let parse = |v: &serde_json::Value| {
            validate_input(&serde_json::to_vec(v).unwrap(), Candidate::Interleaved)
        };
        let good = fixture();
        assert!(parse(&good).is_ok());
        for (key, bad) in [
            ("layer", serde_json::json!(48)),
            ("candidate", serde_json::json!("ane-int8-ffn-down4")),
            ("samples", serde_json::json!([])),
            (
                "quantized_matrix_sha256",
                serde_json::json!([null, null, null]),
            ),
        ] {
            let mut mutant = fixture();
            mutant[key] = bad;
            assert!(parse(&mutant).is_err());
        }
        let mut duplicate = fixture();
        let row = duplicate["samples"][0].clone();
        duplicate["samples"] = serde_json::json!([row.clone(), row]);
        assert!(parse(&duplicate).is_err());
        let duplicate_key = String::from_utf8(serde_json::to_vec(&good).unwrap())
            .unwrap()
            .replacen('{', "{\"layer\":0,", 1);
        assert!(validate_input(duplicate_key.as_bytes(), Candidate::Interleaved).is_err());
    }
    #[test]
    fn captured_input_is_exact_width_finite_and_signed_zero_preserving() {
        let mut bytes = vec![0_u8; HIDDEN * 2];
        bytes[..2].copy_from_slice(&0x8000_u16.to_le_bytes());
        assert_eq!(half_input(&bytes).unwrap()[0].to_bits(), 0x8000);
        assert!(half_input(&bytes[..bytes.len() - 1]).is_err());
        bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(half_input(&bytes).is_err());
    }
    #[test]
    fn matrix_identity_hash_binds_shape_coefficients_and_scales_in_order() {
        use rvllm_apple::ane_int8_ffn_weights::AneInt8MatrixView;
        let values = [-2_i8, 7, 1, -1];
        let scales = [f16::ONE, f16::from_f32(0.5)];
        let view = || AneInt8MatrixView {
            columns: 2,
            values: &values,
            scales: &scales,
        };
        let mut bytes = 2_u64.to_le_bytes().to_vec();
        bytes.extend(2_u64.to_le_bytes());
        bytes.extend([254_u8, 7, 1, 255]);
        bytes.extend([0_u8, 60, 0, 56]);
        assert_eq!(matrix_hash(view()), sha(&bytes));
        assert_ne!(
            matrix_hash(AneInt8MatrixView {
                columns: 4,
                ..view()
            }),
            sha(&bytes)
        );
    }
}
