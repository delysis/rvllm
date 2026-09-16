//! CPU-only audit of every vocabulary row on authenticated final-layer states.
#![forbid(unsafe_code)]

use super::*;

#[path = "ane_int8_head_device_tests.rs"]
mod device;

struct HeadInputs {
    inputs: Vec<Vec<f16>>,
    receipts: Vec<Value>,
    expected: Vec<usize>,
    recorded_tops: Vec<Vec<(usize, f32)>>,
}

fn load_head_inputs(
    sources: &[PathBuf],
    norm: &[f16],
    epsilon: f32,
) -> Result<HeadInputs, Box<dyn std::error::Error>> {
    if sources.is_empty() || sources.len() > 8 {
        return Err("head audit requires 1..8 captured step records".into());
    }
    let mut inputs = Vec::new();
    let mut receipts = Vec::new();
    let mut expected = Vec::new();
    let mut recorded_tops = Vec::new();
    for source in sources {
        let record_bytes = std::fs::read(source)?;
        let record: Value = serde_json::from_slice(&record_bytes)?;
        let position = record["position"]
            .as_u64()
            .ok_or("capture position missing")?;
        let layer = record["layer_states"]
            .as_array()
            .ok_or("no layer captures")?
            .iter()
            .find(|item| item["layer"] == 47)
            .ok_or("final layer capture missing")?;
        let filename = format!("ane-position-{position}-layer-47.fp16");
        if layer["file"].as_str() != Some(filename.as_str()) {
            return Err("capture filename mismatch".into());
        }
        let file = source.parent().ok_or("step parent missing")?.join(filename);
        let bytes = std::fs::read(&file)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if layer["sha256"].as_str() != Some(digest.as_str()) {
            return Err("captured final layer hash mismatch".into());
        }
        let token = record["next_token"]
            .as_u64()
            .filter(|&id| id < VOCAB as u64)
            .ok_or("capture next token invalid")?;
        let top: Vec<(usize, f32)> = serde_json::from_value(record["top_five"].clone())?;
        if top.len() != 5
            || top[0].0 != token as usize
            || top.iter().any(|&(id, v)| id >= VOCAB || !v.is_finite())
        {
            return Err("capture top-five record invalid".into());
        }
        let mut input = vec![f16::ZERO; HIDDEN];
        decode_f16(&bytes, DType::F16, &mut input)?;
        rms_norm_f16_in_place(&mut input, HIDDEN, Some(norm), epsilon)?;
        receipts.push(
            json!({"step":source,"step_sha256":format!("{:x}",Sha256::digest(&record_bytes)),
            "capture":file,"capture_sha256":digest,"normalized_fp16_sha256":hash(&input),
            "recorded_next_token":token,"recorded_top_five":top}),
        );
        expected.push(token as usize);
        recorded_tops.push(top);
        inputs.push(input);
    }
    Ok(HeadInputs {
        inputs,
        receipts,
        expected,
        recorded_tops,
    })
}

fn clipped_logits(values: &[f32], softcap: f32) -> Result<Vec<f32>, String> {
    values
        .iter()
        .map(|&value| {
            let projected = f16::from_f32(value);
            if !projected.is_finite() {
                return Err("head projection exceeds FP16".into());
            }
            let divided = f16::from_f32(projected.to_f32() / softcap);
            let squashed = f16::from_f32(divided.to_f32().tanh());
            Ok(f16::from_f32(squashed.to_f32() * softcap).to_f32())
        })
        .collect()
}

fn top_five(values: &[f32]) -> Vec<(usize, f32)> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    // Independent ranking reference; ties retain the smallest vocabulary ID.
    indices.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    indices[..5].iter().map(|&i| (i, values[i])).collect()
}

#[test]
#[ignore = "CPU-only all 262144 Gemma vocabulary rows; explicit model/capture step JSON/output paths; zero accelerator calls"]
fn checkpoint_vocabulary_int8_ranking() {
    run_head().unwrap();
}

fn run_head() -> Result<(), Box<dyn std::error::Error>> {
    let model = PathBuf::from(std::env::var("RVLLM_GEMMA4_MODEL_DIR")?);
    let output = PathBuf::from(std::env::var("RVLLM_INT8_HEAD_OUTPUT")?);
    let sources: Vec<PathBuf> = serde_json::from_str(&std::env::var("RVLLM_INT8_HEAD_STEPS")?)?;
    if sources.is_empty() || sources.len() > 8 {
        return Err("head audit requires 1..8 captured step records".into());
    }
    std::fs::create_dir(&output)?;
    assert_eq!(compile_budget_used(), 0);
    let (arch, entries) = validated_weights(&model, 1024)?;
    let embedding = &entries[&format!("{}.embed_tokens.weight", arch.weight_prefix)];
    let norm = load_tensor(&entries[&format!("{}.norm.weight", arch.weight_prefix)])?;
    let HeadInputs {
        inputs,
        receipts,
        expected,
        recorded_tops,
    } = load_head_inputs(&sources, &norm, arch.rms_norm_eps)?;
    let mut original_raw: Vec<Vec<f32>> =
        inputs.iter().map(|_| Vec::with_capacity(VOCAB)).collect();
    let mut quantized_raw = original_raw.clone();
    let mut tiles = Vec::new();
    for first in (0..VOCAB).step_by(HEAD_ROWS) {
        let original = load_rows(embedding, first, HEAD_ROWS)?;
        let quantized = AneInt8LinearWeights::quantize(&original, HIDDEN, HEAD_ROWS)?;
        let reconstructed = quantized.dequantized();
        for (case, input) in inputs.iter().enumerate() {
            original_raw[case].extend(cpu_projection(&original, input));
            quantized_raw[case].extend(cpu_projection(&reconstructed, input));
        }
        tiles.push(json!({"first_row":first,"row_count":HEAD_ROWS,
            "original_fp16_sha256":hash(&original),"reconstructed_fp16_sha256":hash(&reconstructed),
            "int8_source_bytes":quantized.source_blob_bytes()}));
        eprintln!(
            "CPU vocabulary rows {}..{} checked",
            first,
            first + HEAD_ROWS
        );
    }
    let mut cases = Vec::new();
    let mut original_winners_match = true;
    let mut candidate_winners_match = true;
    let mut reference_violations = 0;
    for case in 0..inputs.len() {
        assert_eq!(original_raw[case].len(), VOCAB);
        assert_eq!(quantized_raw[case].len(), VOCAB);
        let original = clipped_logits(&original_raw[case], arch.logit_softcap)?;
        let candidate = clipped_logits(&quantized_raw[case], arch.logit_softcap)?;
        let original_top = top_five(&original);
        let candidate_top = top_five(&candidate);
        original_winners_match &= original_top[0].0 == expected[case];
        candidate_winners_match &= candidate_top[0].0 == expected[case];
        let mut recorded_errors = Vec::new();
        for &(id, logit) in &recorded_tops[case] {
            let error = (original[id] - logit).abs();
            reference_violations += usize::from(error > 0.01 + 0.02 * logit.abs());
            recorded_errors.push(json!({"token":id,"recorded_logit":logit,"cpu_logit":original[id],"absolute_error":error}));
        }
        cases.push(json!({"input":receipts[case],"original_top_five":original_top,"int8_top_five":candidate_top,
            "original_winner_margin":original_top[0].1-original_top[1].1,
            "int8_winner_margin":candidate_top[0].1-candidate_top[1].1,
            "raw_projection_quantization_error":errors(&quantized_raw[case],&original_raw[case]),
            "clipped_logits_quantization_error":errors(&candidate,&original),
            "original_cpu_vs_recorded_ane_top_logits":recorded_errors}));
    }
    let report = json!({"schema":"rvllm.gemma4_int8_head_host.v1","model_dir":model,
        "vocabulary_rows":VOCAB,"hidden_size":HIDDEN,"tile_rows":HEAD_ROWS,"tiles":tiles,"cases":cases,
        "original_winners_match_recorded_ane":original_winners_match,
        "int8_winners_match_recorded_ane":candidate_winners_match,
        "original_cpu_recorded_ane_tolerance_violations":reference_violations,
        "compiler_calls":compile_budget_used(),"accelerator_evaluations":0,
        "claim":"CPU-only full-vocabulary INT8 quantization audit on saved final-layer states. No ANE INT8 head execution, full-model quality, or speed claim."});
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    assert_eq!(compile_budget_used(), 0);
    if !original_winners_match || reference_violations != 0 {
        return Err(
            "CPU vocabulary reference differs from captured ANE baseline; see result.json".into(),
        );
    }
    if !candidate_winners_match {
        return Err("INT8 vocabulary changes a recorded winner; see result.json".into());
    }
    Ok(())
}
