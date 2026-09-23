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

fn run(candidate: Candidate) -> Result<(), String> {
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
        "cache_policy":"RequireExisting", "timing":false,
        "driver_journal":journal, "driver_lifecycle_validation":"required-separately"}),
    )?;
    let result = (|| -> Result<(), String> {
        let policy = AneProgramCachePolicy::RequireExisting;
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
    run(Candidate::Chunk4)
}

#[test]
#[ignore = "explicit cached-only real-input ANE down4 component oracle; no timing"]
fn native_down4_matches_plain_cached_ffn() -> Result<(), String> {
    run(Candidate::Down4)
}

#[test]
#[ignore = "explicit cached-only real-input ANE interleaved component oracle; no timing"]
fn native_interleaved_matches_stacked_cached_ffn() -> Result<(), String> {
    run(Candidate::Interleaved)
}

#[cfg(test)]
mod tests {
    use super::*;
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
