//! Text adapter over the same serial owner used by embedded runtime callers.
use super::*;
use rvllm_runtime::gemma_disaggregated_worker::{
    spawn_gemma_disaggregated_worker, GemmaDisaggregatedConfig,
};
use rvllm_runtime::{AppleEngineConfig, BackendPolicy, CachePolicy, GenerateRequest, TokenEvent};

pub(super) struct Options {
    pub model_dir: PathBuf,
    pub metallib_bf16: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub context_capacity: usize,
    pub compile_budget: usize,
    pub interactive: bool,
    pub max_new_tokens: usize,
    pub text_mode: bool,
}

pub(super) fn run(
    options: Options,
    references: Vec<Reference>,
    tokenizer: tokenizers::Tokenizer,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = AppleEngineConfig {
        backend_policy: BackendPolicy::MetalPrefillAneDecode,
        cache_policy: CachePolicy::Disabled,
        maximum_concurrency: 1,
        ingress_queue_capacity: 4,
        event_queue_capacity: 16,
        ..AppleEngineConfig::default()
    };
    let mut config =
        GemmaDisaggregatedConfig::new(options.model_dir.clone(), options.metallib_bf16);
    config.context_capacity = options.context_capacity;
    config.compile_budget = options.compile_budget;
    config.power_journal = options
        .output_dir
        .as_ref()
        .map(|dir| dir.join("power-observations.jsonl"));
    let (engine, health) = spawn_gemma_disaggregated_worker(engine_config, config)?;
    let preparation = health
        .preparation()
        .ok_or("worker lacks preparation receipt")?;
    eprintln!("Both backends ready in runtime owner; Metal {:.1} ms, ANE {:.1} ms; accepting successive requests",
        preparation.metal.as_secs_f64()*1000.0, preparation.ane.as_secs_f64()*1000.0);
    let config_hash = format!(
        "{:x}",
        Sha256::digest(std::fs::read(options.model_dir.join("config.json"))?)
    );
    let requested = references.len();
    let mut references = references.into_iter();
    let mut cases = Vec::new();
    let mut completed = 0_usize;
    let mut last_report = None;
    loop {
        let reference = if let Some(reference) = references.next() {
            reference
        } else if options.interactive {
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line)? == 0 {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            let prompt = match encode_gemma4_user_prompt(&options.model_dir, &tokenizer, &line) {
                Ok(prompt) => prompt,
                Err(error) => {
                    eprintln!("Prompt rejected: {error}");
                    continue;
                }
            };
            if let Err(error) =
                validate_text_budget(&prompt, options.max_new_tokens, options.context_capacity)
            {
                eprintln!("Prompt rejected: {error}");
                continue;
            }
            Reference {
                path: None,
                prompt,
                generated: None,
                max_new_tokens: options.max_new_tokens,
            }
        } else {
            break;
        };
        let mut request = GenerateRequest::new(
            reference.prompt.iter().copied().map(TokenId).collect(),
            u32::try_from(reference.max_new_tokens)?,
        );
        request.cache_policy = CachePolicy::Disabled;
        let mut stream = engine.submit(request)?;
        let mut generated = Vec::with_capacity(reference.max_new_tokens);
        let mut decoder = IncrementalTextDecoder::default();
        let mut streamed = String::new();
        let (finish_reason, backend) = loop {
            match stream.recv()? {
                TokenEvent::Token { token_id, .. } => {
                    generated.push(token_id.raw());
                    if options.text_mode {
                        stream_token(&mut decoder, &tokenizer, token_id.raw(), &mut streamed)?;
                    }
                }
                TokenEvent::Finished {
                    finish_reason,
                    report,
                    ..
                } => break (finish_reason, report),
            }
        };
        let text = tokenizer
            .decode(&generated, true)
            .map_err(|error| error.to_string())?;
        if options.text_mode {
            let tail = text
                .strip_prefix(&streamed)
                .ok_or("streamed text differs from complete decoding")?;
            std::io::stdout().lock().write_all(tail.as_bytes())?;
            println!();
        }
        let matched = reference
            .generated
            .as_ref()
            .map(|expected| generated == *expected);
        let measurements = backend
            .measurement
            .as_ref()
            .ok_or("missing worker measurements")?;
        let steps = measurements["steps"]
            .as_array()
            .ok_or("missing ANE step measurements")?;
        let case = serde_json::json!({
            "reference":reference.path,"prompt_token_ids":reference.prompt,"generated_tokens":generated,
            "generated_text":text,"matches_reference":matched,"finish_reason":format!("{finish_reason:?}"),
            "prefill_measurement":measurements["prefill"],"ane_import_measurement":measurements["import"],
            "metal_gpu_execution_ms":measurements["metal_gpu_execution_ms"],
            "prefill_command_buffers":1,"metal_decode_steps":0,"ane_decode_steps":steps.len(),"steps":steps,
            "queue_ms":backend.queue_time.as_secs_f64()*1000.0,"prefill_including_import_ms":backend.prefill_time.as_secs_f64()*1000.0,
            "decode_ms":backend.decode_time.as_secs_f64()*1000.0,
        });
        if let Some(directory) = &options.output_dir {
            let directory = directory.join(format!("case-{completed}"));
            std::fs::create_dir(&directory)?;
            std::fs::write(
                directory.join("result.json"),
                serde_json::to_vec_pretty(&case)?,
            )?;
        }
        completed = completed.checked_add(1).ok_or("request count exhausted")?;
        if options.interactive {
            cases.clear();
        }
        cases.push(case);
        let report = serde_json::json!({
            "schema":"rvllm.metal_prefill_ane_decode.v1","runtime_worker":true,
            "model_dir":options.model_dir,"config_sha256":config_hash,
            "measurement_environment":preparation.measurement["environment"],
            "metal_prepare_measurement":preparation.measurement["metal"],"ane_prepare_measurement":preparation.measurement["ane"],
            "metal_prepare_ms":preparation.metal.as_secs_f64()*1000.0,"ane_prepare_ms":preparation.ane.as_secs_f64()*1000.0,
            "ane_weight_plan":"static-int8-ffn-cached","loaded_ane_programs":162,
            "ane_compile_budget":options.compile_budget,"ane_compile_budget_used":preparation.compiler_calls,
            "global_context_capacity":options.context_capacity,"execution_order":"prefill-decode-per-request",
            "metal_residency":"retained-through-ane-decode","cpu_or_gpu_decode_fallback":false,
            "interactive_session_open":options.interactive,"requests_completed":completed,
            "inference_complete":!options.interactive && completed == requested,
            "qualification_complete":!options.text_mode && cases.len() == requested && cases.iter().all(|case|case["matches_reference"]==true),
            "layer_state_capture_enabled":false,"diagnostic_journal_enabled":std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some(),
            "cases":cases,
        });
        if let Some(directory) = &options.output_dir {
            std::fs::write(
                directory.join("report.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
            last_report = Some(report);
        }
        eprintln!(
            "Completed request {}; {} prompt tokens, {} output tokens, {} ANE steps",
            completed - 1,
            reference.prompt.len(),
            generated.len(),
            steps.len()
        );
        if matched == Some(false) {
            return Err("runtime worker continuation differs from reference".into());
        }
    }
    if let (Some(directory), Some(mut report)) = (&options.output_dir, last_report) {
        report["interactive_session_open"] = false.into();
        report["inference_complete"] = true.into();
        std::fs::write(
            directory.join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
    }
    Ok(())
}
