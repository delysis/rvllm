//! Global QKV: two existing single-I/O projections, not a new request ABI.
#![forbid(unsafe_code)]

use super::*;

struct SplitGlobal {
    query: AneLinear,
    shared_kv: AneLinear,
}

impl SplitGlobal {
    fn project(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        if input.len() != HIDDEN || output.len() != 8704 {
            return Err("split global projection requires 3840 inputs and 8704 outputs".into());
        }
        let (query, key) = output.split_at_mut(8192);
        self.query.project(input, query)?;
        self.shared_kv.project(input, key)
    }
}

#[test]
#[ignore = "actual global QKV split: explicit paths/mode; cpu zero accelerator calls, qualify at most two compiles and nine evaluations; time strict cache"]
fn checkpoint_global_split_int8_query() {
    run_split().unwrap();
}

fn run_split() -> Result<(), Box<dyn std::error::Error>> {
    let model = PathBuf::from(std::env::var("RVLLM_GEMMA4_MODEL_DIR")?);
    let output = PathBuf::from(std::env::var("RVLLM_INT8_SPLIT_OUTPUT")?);
    let mode = std::env::var("RVLLM_INT8_SPLIT_MODE")?;
    let sources: Vec<PathBuf> = serde_json::from_str(&std::env::var("RVLLM_INT8_QKV_INPUTS")?)?;
    let layer: usize = std::env::var("RVLLM_INT8_QKV_LAYER")?.parse()?;
    let journal = std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some();
    if !matches!(mode.as_str(), "cpu" | "qualify" | "time")
        || (mode == "qualify" && !journal)
        || (mode == "time" && journal)
    {
        return Err("mode cpu|qualify|time; qualify requires journal, time forbids it".into());
    }
    std::fs::create_dir(&output)?;
    assert_eq!(compile_budget_used(), 0);
    let data = load_qkv_inputs(&model, layer, &sources)?;
    if !data.shared_value || data.query_rows != 8192 || data.kv_rows != 512 {
        return Err("split probe requires a Gemma 4 12B global layer".into());
    }
    let (original_q, original_kv) = data.original.split_at(data.query_rows * HIDDEN);
    let quantized_q = AneInt8LinearWeights::quantize(original_q, HIDDEN, data.query_rows)?;
    let mut reconstructed = quantized_q.dequantized();
    reconstructed.extend_from_slice(original_kv);
    let original_cpu: Vec<_> = data
        .inputs
        .iter()
        .map(|x| cpu_projection(&data.original, x))
        .collect();
    let candidate_cpu: Vec<_> = data
        .inputs
        .iter()
        .map(|x| cpu_projection(&reconstructed, x))
        .collect();
    let mut quantization = Vec::new();
    for (original, candidate) in original_cpu.iter().zip(&candidate_cpu) {
        assert_eq!(&original[data.query_rows..], &candidate[data.query_rows..]);
        quantization.push(
            json!({"query":errors(&candidate[..data.query_rows],&original[..data.query_rows]),
            "shared_kv":errors(&candidate[data.query_rows..],&original[data.query_rows..])}),
        );
    }
    let mut report = json!({"schema":"rvllm.gemma4_global_split_int8.v1","layer":layer,"mode":mode,
        "model_dir":model,"input_sources":data.input_receipts,"query_rows":data.query_rows,"kv_rows":data.kv_rows,
        "original_fp16_sha256":hash(&data.original),"mixed_reconstructed_fp16_sha256":hash(&reconstructed),
        "unchanged_shared_kv_fp16_sha256":hash(original_kv),"quantization_errors":quantization,
        "candidate_evaluations_per_projection":2,"baseline_evaluations_per_projection":1,
        "candidate_query_source_bytes":quantized_q.source_blob_bytes(),
        "candidate_kv_source_bytes":128+original_kv.len()*2,
        "claim":"One global layer with INT8 query and original FP16 shared K/V. Complete two-submission latency; no full-model claim."});
    std::fs::write(
        output.join("host.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if mode == "cpu" {
        return Ok(());
    }
    let policy = if mode == "qualify" {
        AneProgramCachePolicy::ReuseOrCompileUpTo(2)
    } else {
        AneProgramCachePolicy::RequireExisting
    };
    let mut baseline = AneLinear::compile_with_cache_policy(
        &data.original,
        HIDDEN,
        8704,
        1,
        AneProgramCachePolicy::RequireExisting,
    )?;
    let mut candidate = SplitGlobal {
        query: AneLinear::compile_int8_with_cache_policy(&quantized_q, 1, policy)?,
        shared_kv: AneLinear::compile_with_cache_policy(original_kv, HIDDEN, 512, 1, policy)?,
    };
    let mut original_out = vec![f16::ZERO; 8704];
    let mut candidate_out = original_out.clone();
    let mut backend = Vec::new();
    let mut violations = 0_u64;
    for (index, input) in data.inputs.iter().enumerate() {
        baseline.project(input, &mut original_out)?;
        candidate.project(input, &mut candidate_out)?;
        let original_values: Vec<_> = original_out.iter().map(|v| v.to_f32()).collect();
        let candidate_values: Vec<_> = candidate_out.iter().map(|v| v.to_f32()).collect();
        let original_error = errors(&original_values, &original_cpu[index]);
        let query_error = errors(
            &candidate_values[..data.query_rows],
            &candidate_cpu[index][..data.query_rows],
        );
        let kv_error = errors(
            &candidate_values[data.query_rows..],
            &candidate_cpu[index][data.query_rows..],
        );
        for error in [&original_error, &query_error, &kv_error] {
            violations += error["backend_tolerance_violations"].as_u64().unwrap();
        }
        backend.push(json!({"original":original_error,"split_query":query_error,"split_shared_kv":kv_error,
            "shared_kv_output_bits_identical":original_out[data.query_rows..].iter().zip(&candidate_out[data.query_rows..]).all(|(a,b)|a.to_bits()==b.to_bits())}));
    }
    report["backend_errors"] = json!(backend);
    report["backend_violations"] = json!(violations);
    if mode == "time" && violations == 0 {
        report["timing"] = measure_split(
            &mut baseline,
            &mut candidate,
            &data.inputs[0],
            &mut candidate_out,
        )?;
    }
    drop(candidate);
    drop(baseline);
    report["compiler_calls"] = json!(compile_budget_used());
    report["models_dropped"] = json!(3);
    report["driver_journal_enabled"] = json!(journal);
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if violations != 0 {
        return Err("split QKV backend numerical gate failed; see result.json".into());
    }
    Ok(())
}

fn measure_split(
    baseline: &mut AneLinear,
    candidate: &mut SplitGlobal,
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
    for _ in 0..8 {
        baseline.project(input, output)?;
        candidate.project(input, output)?;
    }
    let repetitions = 128;
    'blocks: for block in 0..3 {
        let start = trials.len();
        let order = if block % 2 == 0 {
            [false, true, true, false]
        } else {
            [true, false, false, true]
        };
        for split in order {
            let phase = monitor.begin();
            for _ in 0..repetitions {
                if split {
                    candidate.project(input, output)?;
                } else {
                    baseline.project(input, output)?;
                }
            }
            let measurement = phase.finish(repetitions);
            let eligible = measurement["sampled_controls_eligible"] == true;
            trials.push(json!({"split":split,"measurement":measurement}));
            if !eligible {
                break 'blocks;
            }
        }
        for (a, b) in [(start, start + 1), (start + 3, start + 2)] {
            let (a, b) = if trials[a]["split"] == false {
                (a, b)
            } else {
                (b, a)
            };
            pairs.push(match compare_phase_measurements(&trials[a]["measurement"],&trials[b]["measurement"]) {
                Ok(comparison)=>json!({"baseline_trial":a,"candidate_trial":b,"eligible":true,"comparison":comparison}),
                Err(reason)=>json!({"baseline_trial":a,"candidate_trial":b,"eligible":false,"reason":reason}),
            });
        }
    }
    Ok(
        json!({"preflight":preflight,"complete_projections_per_trial":repetitions,"trials":trials,"pairs":pairs}),
    )
}
