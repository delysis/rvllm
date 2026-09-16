//! One actual vocabulary tile through the existing single-I/O linear ABI.
#![forbid(unsafe_code)]

use super::*;

#[test]
#[ignore = "one actual head tile: explicit paths/tile/mode; qualify at most two compiles; restore-original at most one compile and zero evaluations; time strict cache"]
fn checkpoint_vocabulary_tile_int8() {
    run_tile().unwrap();
}

fn run_tile() -> Result<(), Box<dyn std::error::Error>> {
    let model = PathBuf::from(std::env::var("RVLLM_GEMMA4_MODEL_DIR")?);
    let output = PathBuf::from(std::env::var("RVLLM_INT8_HEAD_DEVICE_OUTPUT")?);
    let mode = std::env::var("RVLLM_INT8_HEAD_DEVICE_MODE")?;
    let tile: usize = std::env::var("RVLLM_INT8_HEAD_TILE")?.parse()?;
    let sources: Vec<PathBuf> = serde_json::from_str(&std::env::var("RVLLM_INT8_HEAD_STEPS")?)?;
    let journal = std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some();
    if tile >= VOCAB / HEAD_ROWS
        || !matches!(
            mode.as_str(),
            "cpu" | "restore-original" | "qualify" | "time"
        )
        || (matches!(mode.as_str(), "restore-original" | "qualify") && !journal)
        || (mode == "time" && journal)
    {
        return Err("tile 0..15; mode cpu|restore-original|qualify|time; restore/qualify require journal, time forbids it".into());
    }
    std::fs::create_dir(&output)?;
    assert_eq!(compile_budget_used(), 0);
    let (arch, entries) = validated_weights(&model, 1024)?;
    let embedding = &entries[&format!("{}.embed_tokens.weight", arch.weight_prefix)];
    let norm = load_tensor(&entries[&format!("{}.norm.weight", arch.weight_prefix)])?;
    let data = load_head_inputs(&sources, &norm, arch.rms_norm_eps)?;
    let first = tile * HEAD_ROWS;
    let original = load_rows(embedding, first, HEAD_ROWS)?;
    let quantized = AneInt8LinearWeights::quantize(&original, HIDDEN, HEAD_ROWS)?;
    let reconstructed = quantized.dequantized();
    let original_cpu: Vec<_> = data
        .inputs
        .iter()
        .map(|x| cpu_projection(&original, x))
        .collect();
    let candidate_cpu: Vec<_> = data
        .inputs
        .iter()
        .map(|x| cpu_projection(&reconstructed, x))
        .collect();
    let quantization: Vec<_> = original_cpu
        .iter()
        .zip(&candidate_cpu)
        .map(|(a, b)| errors(b, a))
        .collect();
    let mut report = json!({"schema":"rvllm.gemma4_head_tile_int8.v1","model_dir":model,
        "mode":mode,"tile":tile,"first_row":first,"row_count":HEAD_ROWS,"input_sources":data.receipts,
        "original_fp16_sha256":hash(&original),"reconstructed_fp16_sha256":hash(&reconstructed),
        "quantization_errors":quantization,"int8_source_bytes":quantized.source_blob_bytes(),
        "backend_tolerance":"0.01 + 0.02 * abs(cpu_reference)",
        "claim":"One vocabulary tile on authenticated saved states; not full-vocabulary ANE ranking or full-model quality/performance."});
    std::fs::write(
        output.join("host.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if mode == "cpu" {
        return Ok(());
    }
    if mode == "restore-original" {
        let baseline = AneLinear::compile_with_cache_policy(
            &original,
            HIDDEN,
            HEAD_ROWS,
            1,
            AneProgramCachePolicy::ReuseOrCompileUpTo(1),
        )?;
        drop(baseline);
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
    let baseline = AneLinear::compile_with_cache_policy(
        &original,
        HIDDEN,
        HEAD_ROWS,
        1,
        AneProgramCachePolicy::RequireExisting,
    )?;
    let candidate = AneLinear::compile_int8_with_cache_policy(&quantized, 1, policy)?;
    let control =
        AneLinear::compile_with_cache_policy(&reconstructed, HIDDEN, HEAD_ROWS, 1, policy)?;
    let mut programs = [baseline, candidate, control];
    let mut outputs: [Vec<f16>; 3] = std::array::from_fn(|_| vec![f16::ZERO; HEAD_ROWS]);
    let mut backend = Vec::new();
    let mut violations = 0_u64;
    let mut recorded_violations = 0_usize;
    let mut recorded_logits_checked = 0_usize;
    for (index, input) in data.inputs.iter().enumerate() {
        let mut case = Vec::new();
        for (kind, (program, out)) in programs.iter_mut().zip(&mut outputs).enumerate() {
            program.project(input, out)?;
            let actual: Vec<_> = out.iter().map(|v| v.to_f32()).collect();
            let reference = if kind == 0 {
                &original_cpu[index]
            } else {
                &candidate_cpu[index]
            };
            let error = errors(&actual, reference);
            violations += error["backend_tolerance_violations"].as_u64().unwrap();
            let logits = clipped_logits(&actual, arch.logit_softcap)?;
            if kind == 0 {
                for &(id, logit) in &data.recorded_tops[index] {
                    if (first..first + HEAD_ROWS).contains(&id) {
                        recorded_logits_checked += 1;
                        recorded_violations += usize::from(
                            (logits[id - first] - logit).abs() > 0.01 + 0.02 * logit.abs(),
                        );
                    }
                }
            }
            let top: Vec<_> = top_five(&logits)
                .into_iter()
                .map(|(id, v)| (first + id, v))
                .collect();
            let program_name = ["original", "int8", "reconstructed_dense"][kind];
            case.push(json!({"program":program_name,
                "error":error,"within_tile_top_five":top}));
        }
        backend.push(json!(case));
    }
    report["backend_errors"] = json!(backend);
    report["backend_violations"] = json!(violations);
    report["recorded_logits_checked"] = json!(recorded_logits_checked);
    report["recorded_logits_violations"] = json!(recorded_violations);
    report["qualification_evaluations"] = json!(3 * data.inputs.len());
    report["driver_journal_enabled"] = json!(journal);
    if mode == "time" && violations == 0 && recorded_violations == 0 {
        report["timing"] = measure(&mut programs, &data.inputs[0], &mut outputs[0])?;
    }
    drop(programs);
    report["compiler_calls"] = json!(compile_budget_used());
    report["models_dropped"] = json!(3);
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if violations != 0 || recorded_violations != 0 {
        return Err("vocabulary tile backend numerical gate failed; see result.json".into());
    }
    Ok(())
}
