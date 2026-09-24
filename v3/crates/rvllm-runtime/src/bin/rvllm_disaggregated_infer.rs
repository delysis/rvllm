//! Run Metal prefill followed by resident ANE Gemma 4 12B text decode.
#![forbid(unsafe_code)]

use half::f16;
use rvllm_apple::ane_attention_layout::KvImportPacking;
use rvllm_apple::{AppleBackend, AppleRuntimePlan, HandoffKind};
use rvllm_apple_metal::{MetalFloatType, MetalKernelOptions, MetalModelLimits};
use rvllm_core::{ReqId, TokenId};
use rvllm_runtime::ane_prefill::{AneDecodeStart, PrefillScalarType};
use rvllm_runtime::apple_bridge::handoff_from_prefill_plan_with_paged_kv;
use rvllm_runtime::apple_measurement::{metal_counter_capabilities, PowerMonitor};
use rvllm_runtime::apple_metal_backend::{ModelMetalBackend, ModelMetalOptions};
use rvllm_runtime::gemma_ane_decode::{
    inspect_static_cache_with_capacity_until, provision_static_cache_with_capacity_until,
    AneStaticCachePart, AneWeightPlan, GemmaAneDecode,
};
use rvllm_runtime::gemma_head_ranking::{validate_head_ranking_mode, HeadRankingPlan};
use rvllm_runtime::text_generation::{encode_gemma4_user_prompt, IncrementalTextDecoder};
use rvllm_runtime::{BatchPlan, PagedKvConfig, PagedKvPool};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "rvllm_disaggregated_infer/worker.rs"]
mod worker;

#[path = "rvllm_disaggregated_infer/cache_batch.rs"]
mod cache_batch;

#[path = "rvllm_disaggregated_infer/prefill_screen.rs"]
mod prefill_screen;

#[derive(Clone, Copy)]
struct PrefillEvidenceOptions {
    capture_seed: bool,
    candidate: rvllm_apple_metal::MetalResearchCandidate,
}

struct PrefillCase {
    start: AneDecodeStart,
    prefill_sample_capture_ms: f64,
    prefill_measurement: serde_json::Value,
    receipt: serde_json::Value,
}

enum CacheTarget {
    Part(AneStaticCachePart),
    AllInt8,
}

struct Reference {
    path: Option<PathBuf>,
    prompt: Vec<u32>,
    generated: Option<Vec<u32>>,
    max_new_tokens: usize,
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("disaggregated inference failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_dir = None;
    let mut output_dir = None;
    let mut metallib_bf16 = None;
    let mut paths = Vec::new();
    let mut prompts = Vec::new();
    let mut max_new_tokens = 64_usize;
    let mut prefill_only = false;
    let mut capture_layers = false;
    let mut capture_ffn_inputs = false;
    let mut retain_metal = false;
    let mut interleave = false;
    let mut interactive = false;
    let mut runtime_worker = false;
    let mut ane_weights = AneWeightPlan::StaticInt8FfnCached;
    let mut kv_import_packing = KvImportPacking::Baseline;
    let mut head_ranking = HeadRankingPlan::Baseline;
    let mut head_ranking_timing = false;
    let mut prepare_cache = None;
    let mut inspect_cache = false;
    let mut compile_budget = 0_usize;
    let mut context_capacity = 1024_usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("Gemma 4 12B: Metal prefill, ANE decode, greedy text generation.\n\nUsage: rvllm_disaggregated_infer --model-dir PATH --prompt TEXT [options]\n\n  --metallib-bf16 PATH           Precompiled BF16 Metal library (or RVLLM_METAL_METALLIB_BF16)\n  --prompt TEXT                 One user text turn; may be repeated\n  --prompt-file PATH            Read one user text turn from a UTF-8 file\n  --max-new-tokens N             Output limit including EOS (default 64)\n  --context-capacity 64|1024     Prompt plus decoded input capacity (default 1024)\n  --ane-weights PLAN             static-int8-ffn-cached (default), static-all-cached, or research plans\n  --ane-compile-budget 0..16     Bounded recovery of missing cached programs (default 0)\n  --retain-metal BOOL            Keep Metal loaded during ANE decode (default false)\n  --interleave BOOL              Prepare both backends once, then prefill/decode each request\n  --interactive BOOL             Read successive user prompts from stdin, one per line; implies interleave\n  --runtime-worker BOOL          Use the serial runtime owner (INT8/MMA/SIMD; default false)\n  --output-dir PATH              Optional local report directory\n  --hf-reference PATH           Verify against pinned token IDs; requires output directory\n  --capture-layer-states BOOL    Capture first ANE step; requires output directory\n  --prepare-ane-cache PART       qkv, output, ffn, ffn-int8, ffn-lut4, head-attention\n\nText input uses the qualified single-user, non-thinking checkpoint template.\nMultiple prompts share initialization. Model histories, tools and multimodal inputs are unsupported.");
            println!("\n  --prefill-only BOOL            First-token/KV screen; original reference retained; never loads ANE");
            println!("\n  --inspect-ane-cache PART       Strict load inspection of a cache part; zero compiles/evaluations");
            println!(
                "  --kv-import-packing baseline|reuse-scratch|cpu-kv-blocked32 (default baseline)"
            );
            println!("  --head-ranking baseline|cpu-head-softcap-prune (default baseline)");
            println!("  --head-ranking-timing BOOL   CPU softcap/ranking only; default false; same setting on both timing arms");
            println!("  PART=all-int8                  Visit qkv, output, ffn-int8 and head-attention in fresh serial processes");
            println!(
                "  PART=ffn-int8-chunk4           Explicit single-I/O FFN output-channel chunking"
            );
            println!("  --ane-weights static-int8-chunk4-ffn-cached (zero compile budget)");
            println!("  PART=ffn-int8-down4; --ane-weights static-int8-down4-ffn-cached (zero compile budget)");
            println!("  PART=ffn-int8-interleaved; --ane-weights static-int8-interleaved-ffn-cached (zero compile budget; control is stacked INT8)");
            println!("  PART=attention-transpose-flags; --ane-weights static-int8-ffn-transpose-attention-cached (zero compile budget)");
            println!("  PART=ffn-int8-stacked          Prepare/inspect the experimental stacked INT8 FFNs");
            println!(
                "  PART=qkv-sliding-int8-tiles4   Output-row tiling; original FP16 global QKV"
            );
            println!(
                "  --ane-weights static-int8-ffn-sliding-qkv-tiles4-cached (zero compile budget)"
            );
            println!("  PART=qkv-sliding-int8          Experimental INT8 sliding QKV; original FP16 global QKV");
            println!("  --capture-ffn-inputs BOOL      Capture actual FFN inputs for the first two ANE steps; requires output directory; diagnostics only");
            println!("  --ane-weights static-int8-ffn-sliding-qkv-cached\n                                Experimental sliding QKV quantization; full-model quality unqualified");
            println!("  --ane-weights static-int8-stacked-ffn-cached|static-int8-stacked-ffn-checked\n                                Experimental layout; checked requires references, output directory, journal and zero compile budget");
            return Ok(());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            "--metallib-bf16" => metallib_bf16 = Some(PathBuf::from(value)),
            "--hf-reference" => paths.push(PathBuf::from(value)),
            "--prompt" => prompts.push(value),
            "--prompt-file" => prompts.push(std::fs::read_to_string(value)?),
            "--max-new-tokens" => max_new_tokens = value.parse()?,
            "--prefill-only" => prefill_only = value.parse()?,
            "--capture-layer-states" => capture_layers = value.parse()?,
            "--capture-ffn-inputs" => capture_ffn_inputs = value.parse()?,
            "--retain-metal" => retain_metal = value.parse()?,
            "--interleave" => interleave = value.parse()?,
            "--interactive" => interactive = value.parse()?,
            "--runtime-worker" => runtime_worker = value.parse()?,
            "--kv-import-packing" => kv_import_packing = value.parse()?,
            "--head-ranking" => head_ranking = value.parse()?,
            "--head-ranking-timing" => head_ranking_timing = value.parse()?,
            "--ane-compile-budget" => compile_budget = value.parse()?,
            "--context-capacity" => context_capacity = value.parse()?,
            "--prepare-ane-cache" | "--inspect-ane-cache" => {
                if prepare_cache.is_some() {
                    return Err("provide only one cache operation".into());
                }
                inspect_cache = flag == "--inspect-ane-cache";
                prepare_cache = Some(if value == "all-int8" {
                    CacheTarget::AllInt8
                } else {
                    CacheTarget::Part(match value.as_str() {
                    "qkv" => AneStaticCachePart::QueryKeyValue,
                    "qkv-sliding-int8-tiles4" => AneStaticCachePart::QueryKeyValueSlidingInt8Tiles4,
                    "qkv-sliding-int8" => AneStaticCachePart::QueryKeyValueSlidingInt8,
                    "output" => AneStaticCachePart::Output,
                    "ffn" => AneStaticCachePart::FeedForward,
                    "ffn-lut4" => AneStaticCachePart::FeedForwardLut4,
                    "ffn-int8" => AneStaticCachePart::FeedForwardInt8,
                    "ffn-int8-chunk4" => AneStaticCachePart::FeedForwardInt8Chunk4,
                    "ffn-int8-down4" => AneStaticCachePart::FeedForwardInt8Down4,
                    "ffn-int8-interleaved" => AneStaticCachePart::FeedForwardInt8Interleaved,
                    "attention-transpose-flags" => AneStaticCachePart::AttentionTransposeFlags,
                    "ffn-int8-stacked" => AneStaticCachePart::FeedForwardInt8Stacked,
                    "head-attention" => AneStaticCachePart::VocabularyAndAttention,
                    _ => {
                        return Err(
                            "cache part must be all-int8, qkv, qkv-sliding-int8, qkv-sliding-int8-tiles4, output, ffn, ffn-int8, ffn-int8-stacked, ffn-int8-chunk4, ffn-lut4 or head-attention".into(),
                        )
                    }
                })
                });
            }
            "--ane-weights" => {
                ane_weights = match value.as_str() {
                    "dynamic-ffn" => AneWeightPlan::DynamicFfn,
                    "static-ffn" => AneWeightPlan::StaticFfnDynamicOutput,
                    "static-all-cached" => AneWeightPlan::StaticAllCached,
                    "static-lut4-ffn-cached" => AneWeightPlan::StaticLut4FfnCached,
                    "static-int8-ffn-cached" => AneWeightPlan::StaticInt8FfnCached,
                    "static-int8-ffn-sliding-qkv-tiles4-cached" => {
                        AneWeightPlan::StaticInt8FfnSlidingQkvTiles4Cached
                    }
                    "static-int8-ffn-sliding-qkv-cached" => {
                        AneWeightPlan::StaticInt8FfnSlidingQkvCached
                    }
                    "static-int8-chunk4-ffn-cached" => AneWeightPlan::StaticInt8Chunk4FfnCached,
                    "static-int8-down4-ffn-cached" => AneWeightPlan::StaticInt8Down4FfnCached,
                    "static-int8-interleaved-ffn-cached" => {
                        AneWeightPlan::StaticInt8InterleavedFfnCached
                    }
                    "static-int8-ffn-transpose-attention-cached" => {
                        AneWeightPlan::StaticInt8FfnTransposeAttentionCached
                    }
                    "static-int8-stacked-ffn-cached" => AneWeightPlan::StaticInt8StackedFfnCached,
                    "static-int8-stacked-ffn-checked" => AneWeightPlan::StaticInt8StackedFfnChecked,
                    _ => {
                        return Err(
                            "unknown --ane-weights plan; see --help for cached and research plans"
                                .into(),
                        )
                    }
                }
            }
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    validate_head_ranking_mode(
        head_ranking,
        head_ranking_timing,
        prefill_only,
        runtime_worker,
        prepare_cache.is_some(),
        compile_budget,
        ane_weights == AneWeightPlan::StaticInt8FfnCached,
    )?;
    interleave |= interactive || runtime_worker;
    retain_metal |= interleave;
    prefill_screen::PrefillScreenOptions {
        enabled: prefill_only,
        reference_count: paths.len(),
        output_requested: output_dir.is_some(),
        text_input: !prompts.is_empty(),
        interactive,
        interleave,
        retain_metal,
        runtime_worker,
        capture_ane: capture_layers || capture_ffn_inputs,
        cache_operation: prepare_cache.is_some(),
        nonbaseline_ane: ane_weights != AneWeightPlan::StaticInt8FfnCached,
        nonbaseline_packing: kv_import_packing != KvImportPacking::Baseline,
        compile_budget,
    }
    .validate()?;
    let model_dir = model_dir.ok_or("--model-dir required")?;
    if matches!(
        ane_weights,
        AneWeightPlan::StaticInt8Chunk4FfnCached
            | AneWeightPlan::StaticInt8FfnSlidingQkvTiles4Cached
            | AneWeightPlan::StaticInt8Down4FfnCached
            | AneWeightPlan::StaticInt8InterleavedFfnCached
            | AneWeightPlan::StaticInt8FfnTransposeAttentionCached
    ) && compile_budget != 0
    {
        return Err(
            "candidate inference requires zero compile budget; provision separately".into(),
        );
    }
    if ane_weights == AneWeightPlan::StaticInt8StackedFfnChecked
        && (paths.is_empty()
            || output_dir.is_none()
            || compile_budget != 0
            || prepare_cache.is_some()
            || std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_none())
    {
        return Err("stacked FFN checking requires references, an output directory, a driver journal and zero compile budget".into());
    }
    if !matches!(context_capacity, 64 | 1024) {
        return Err("--context-capacity must be a qualified capacity: 64 or 1024".into());
    }
    if compile_budget > 16
        || (compile_budget != 0
            && (prepare_cache.is_some()
                || ane_weights.cache_policy()
                    != rvllm_apple::ane_linear::AneProgramCachePolicy::RequireExisting))
    {
        return Err("--ane-compile-budget must be 0..=16, for cached inference only".into());
    }
    if let Some(target) = prepare_cache {
        if kv_import_packing != KvImportPacking::Baseline
            || !paths.is_empty()
            || !prompts.is_empty()
            || capture_layers
            || capture_ffn_inputs
            || interactive
            || interleave
        {
            return Err(
                "cache operations cannot be combined with prompts, references, interactive/worker mode or layer capture"
                    .into(),
            );
        }
        let output_dir = output_dir.ok_or("--output-dir required for cache provisioning")?;
        std::fs::create_dir(&output_dir)?;
        let output_dir = output_dir.canonicalize()?;
        let cancellation = cache_batch::Cancellation::install()?;
        let part = match target {
            CacheTarget::Part(part) => part,
            CacheTarget::AllInt8 => {
                cache_batch::run_all(
                    &model_dir.canonicalize()?,
                    &output_dir,
                    context_capacity,
                    inspect_cache,
                    &cancellation,
                )?;
                return Ok(());
            }
        };
        let monitor = PowerMonitor::start(Some(&output_dir.join("power-observations.jsonl")))?;
        let measured = monitor.begin();
        let started = Instant::now();
        let inspected = if inspect_cache {
            Some(inspect_static_cache_with_capacity_until(
                &model_dir,
                part,
                context_capacity,
                &|| cancellation.requested(),
            )?)
        } else {
            None
        };
        let programs = if let Some(entries) = &inspected {
            entries.iter().filter(|entry| entry.available).count()
        } else {
            provision_static_cache_with_capacity_until(&model_dir, part, context_capacity, &|| {
                cancellation.requested()
            })?
        };
        let report = serde_json::json!({
            "schema":if inspect_cache {"rvllm.ane_cache_inspection.v1"} else {"rvllm.ane_cache_provision.v1"}, "part":format!("{part:?}"),
            "model_dir":model_dir, "programs_prepared":programs,
            "entries":inspected.as_ref().map(|entries| entries.iter().map(|entry| serde_json::json!({"name":entry.name,"available":entry.available})).collect::<Vec<_>>()),
            "maximum_explicit_compiles":if inspect_cache {0} else {programs},
            "compiler_calls":rvllm_apple::ane_linear::compile_budget_used(),
            "cache_policy":if inspect_cache {"require-existing"} else {"bounded-reuse-or-compile"},
            "prepare_ms":started.elapsed().as_secs_f64()*1000.0, "inference_evaluations":0,"global_context_capacity":context_capacity,
            "measurement":measured.finish(1),
            "claim":"Per-entry availability only; cache entries can later be evicted. This is not full residency or numerical qualification."
        });
        std::fs::write(
            output_dir.join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if (!paths.is_empty() && (!prompts.is_empty() || interactive))
        || (paths.is_empty() && prompts.is_empty() && !interactive)
    {
        return Err("provide --prompt/--prompt-file or --hf-reference, but do not mix them".into());
    }
    if (!paths.is_empty() || capture_layers || capture_ffn_inputs) && output_dir.is_none() {
        return Err("--output-dir required for reference verification or layer capture".into());
    }
    if max_new_tokens == 0 || max_new_tokens > context_capacity + 1 {
        return Err("--max-new-tokens must be positive and fit the selected context".into());
    }
    let text_mode = !prompts.is_empty() || interactive;
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let eos = rvllm_loader::generation::load_eos_token_ids(&model_dir)?;
    let mut references = Vec::with_capacity(paths.len());
    for path in paths {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let prompt: Vec<u32> = serde_json::from_value(value["prompt_token_ids"].clone())?;
        let generated: Vec<u32> = serde_json::from_value(value["generated_tokens"].clone())?;
        let revision = value["model_revision"]
            .as_str()
            .map(str::to_owned)
            .or_else(|| {
                Path::new(value["model_dir"].as_str()?)
                    .file_name()?
                    .to_str()
                    .map(str::to_owned)
            })
            .ok_or("reference lacks checkpoint revision")?;
        if model_dir.file_name().and_then(|n| n.to_str()) != Some(revision.as_str()) {
            return Err("reference/checkpoint revision mismatch".into());
        }
        if prompt.is_empty()
            || generated.is_empty()
            || prompt.len() + generated.len() - 1 > context_capacity
            || prompt.iter().chain(&generated).any(|&id| id >= 262144)
            || generated[..generated.len() - 1]
                .iter()
                .any(|id| eos.contains(id))
        {
            return Err("reference has invalid tokens, EOS placement or context length".into());
        }
        references.push(Reference {
            path: Some(path),
            prompt,
            max_new_tokens: generated.len(),
            generated: Some(generated),
        });
    }
    for prompt in prompts {
        let prompt = encode_gemma4_user_prompt(&model_dir, &tokenizer, &prompt)?;
        validate_text_budget(&prompt, max_new_tokens, context_capacity)?;
        references.push(Reference {
            path: None,
            prompt,
            generated: None,
            max_new_tokens,
        });
    }
    if let Some(directory) = &output_dir {
        std::fs::create_dir(directory)?;
    }
    let config = std::fs::read(model_dir.join("config.json"))?;
    let layout_hash: [u8; 32] = Sha256::digest(&config).into();
    // Resolve diagnostic overrides once, without changing process globals.
    // Encoding independently checks BF16, GPU family and qualified shapes.
    let mut kernels = MetalKernelOptions::from_development_environment();
    if prefill_only {
        let requested = std::env::var("RVLLM_METAL_RESEARCH");
        prefill_screen::validate_requested_candidate(requested.as_deref(), kernels.research)?;
    }
    if std::env::var_os("RVLLM_METAL_PREFILL_GEMM").is_none() {
        kernels.prefill_mma32 = true;
    }
    if std::env::var_os("RVLLM_METAL_PREFILL_ATTENTION").is_none() {
        kernels.prefill_simd_attention = true;
    }
    if kernels.quantized_bf16_accumulation {
        return Err("disaggregated inference requires FP32 accumulation".into());
    }
    let executable_path = std::env::current_exe()?.canonicalize()?;
    let executable_sha256 = sha256_file(&executable_path)?;
    let metallib_bf16 = metallib_bf16
        .or_else(|| std::env::var_os("RVLLM_METAL_METALLIB_BF16").map(PathBuf::from))
        .ok_or("provide --metallib-bf16 or RVLLM_METAL_METALLIB_BF16")?
        .canonicalize()?;
    let metallib_sha256 = sha256_file(&metallib_bf16)?;
    let generated_msl_sha256 = format!(
        "{:x}",
        Sha256::digest(
            rvllm_apple_metal::kernels::kernel_source_with_options(
                MetalFloatType::Bf16,
                kernels,
            )
            .as_bytes(),
        )
    );
    if runtime_worker {
        let qualified = MetalKernelOptions {
            prefill_mma32: true,
            prefill_simd_attention: true,
            ..MetalKernelOptions::default()
        };
        if capture_layers
            || capture_ffn_inputs
            || ane_weights != AneWeightPlan::StaticInt8FfnCached
            || kernels != qualified
            || kv_import_packing != KvImportPacking::Baseline
        {
            return Err(
                "runtime worker requires the qualified INT8/MMA/SIMD route without layer capture"
                    .into(),
            );
        }
        return worker::run(
            worker::Options {
                model_dir,
                metallib_bf16,
                output_dir,
                context_capacity,
                compile_budget,
                interactive,
                max_new_tokens,
                text_mode,
            },
            references,
            tokenizer,
        );
    }
    let metal_options = ModelMetalOptions {
        float_type: MetalFloatType::Bf16,
        kernels,
        limits: MetalModelLimits {
            max_context_tokens: context_capacity,
            max_batch_tokens: if interleave {
                context_capacity
            } else {
                references
                    .iter()
                    .map(|r| r.prompt.len())
                    .max()
                    .ok_or("no prompts")?
            },
            max_batch_sequences: 1,
        },
    };
    let plan = AppleRuntimePlan {
        target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple Silicon", 1),
        // This plan prepares only the Metal backend; explicit ANE ownership
        // below supplies decode rather than the generic backend router.
        mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
        rollout_bucket: None,
        rollout_tokens: 1,
        private_ane_opt_in: false,
        strict_ane: false,
        ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
        ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
        ane_hidden_size: 3840,
        ane_intermediate_size: 15360,
        ane_num_layers: 48,
        model_layout_hash: layout_hash,
        weights_path: Some(model_dir.clone()),
    };
    let mut metal = ModelMetalBackend::with_options(
        model_dir.clone(),
        metallib_bf16.clone(),
        metal_options,
    );
    let power_journal = output_dir
        .as_ref()
        .map(|dir| dir.join("power-observations.jsonl"));
    let measurement_environment = metal_counter_capabilities();
    let monitor = PowerMonitor::start(power_journal.as_deref())?;
    let measured = monitor.begin();
    let timer = Instant::now();
    metal.prepare(&plan)?;
    let metal_prepare_ms = elapsed_ms(timer);
    let metal_prepare_measurement = measured.finish(1);
    if prefill_only {
        let directory = output_dir
            .as_ref()
            .ok_or("prefill-only output directory missing")?;
        let mut cases = Vec::with_capacity(references.len());
        let mut all_selections_exercised = true;
        for (index, reference) in references.iter().enumerate() {
            let case_dir = directory.join(format!("case-{index}"));
            let completed = prefill_case(
                &monitor,
                &mut metal,
                reference,
                &model_dir,
                layout_hash,
                Some(&case_dir),
                PrefillEvidenceOptions {
                    capture_seed: true,
                    candidate: kernels.research,
                },
            )?;
            all_selections_exercised &=
                completed.receipt["research_dispatch"]["complete_family_exercised"] == true;
            cases.push(completed.receipt);
            // Drop each owned KV snapshot here: the screen never queues ANE
            // work and does not retain N prompt caches while screening N cases.
        }
        let report = serde_json::json!({
            "schema": "rvllm.metal_prefill_screen.v1",
            "model_dir": model_dir,
            "config_sha256": format!("{:x}", Sha256::digest(&config)),
            "status": if all_selections_exercised { "first-token-screen-complete" } else { "candidate-not-exercised" },
            "metal_prepare_ms": metal_prepare_ms,
            "metal_prepare_measurement": metal_prepare_measurement,
            "measurement_environment": measurement_environment,
            "requested_candidate": kernels.research.name(),
            "all_selections_exercised": all_selections_exercised,
            "reference_continuation_not_truncated": true,
            "comparison_scope": "first generated token only; full reference continuation retained, not evaluated",
            "requests_completed": cases.len(),
            "ane_load_calls_by_this_mode": 0,
            "ane_compiler_calls_by_this_mode": 0,
            "ane_decode_steps_by_this_mode": 0,
            "ane_driver_journal_verified": false,
            "ane_execution_verified": false,
            "qualification_complete": false,
            "tensor_oracle_passed": null,
            "performance_accepted": false,
            "cases": cases,
        });
        prefill_screen::write_new_json(&directory.join("report.json"), &report)?;
        if !all_selections_exercised {
            return Err("prefill completed through fallback, but the requested candidate was not exercised; GQA requires at least 64 prompt tokens".into());
        }
        eprintln!("First-token-only Metal screen complete; no ANE owner was loaded");
        return Ok(());
    }
    let mut starts = std::collections::VecDeque::with_capacity(references.len());
    if !interleave {
        for (index, reference) in references.iter().enumerate() {
            let case_dir = output_dir
                .as_ref()
                .map(|dir| dir.join(format!("case-{index}")));
            starts.push_back(prefill_case(
                &monitor,
                &mut metal,
                reference,
                &model_dir,
                layout_hash,
                case_dir.as_deref(),
                PrefillEvidenceOptions {
                    capture_seed: !text_mode || capture_layers,
                    candidate: kernels.research,
                },
            )?);
            eprintln!("Metal prefilled and captured case {index}");
        }
    }
    // Each start owns CPU KV vectors after synchronous collection. Keeping the
    // idle Metal arena here would retain another full model during ANE decode.
    // Phased mode batches prefills. Interleaved mode retains both owners on
    // this thread and completes each request before reading another.
    let metal_arena = metal
        .probe_arena_stats()
        .ok_or("Metal arena stats unavailable")?;
    let mut retained_metal = if retain_metal {
        eprintln!(
            "Retaining Metal during ANE execution; arena capacity is {} bytes",
            metal_arena.capacity_bytes
        );
        Some(metal)
    } else {
        drop(metal);
        eprintln!(
            "Released Metal backend before ANE preparation; arena capacity was {} bytes",
            metal_arena.capacity_bytes
        );
        None
    };
    let mut ane = None;
    let mut ane_prepare_ms = None;
    let mut ane_prepare_measurement = None;
    if interleave {
        let measured = monitor.begin();
        let timer = Instant::now();
        ane = Some(GemmaAneDecode::load_with_compile_budget(
            &model_dir,
            context_capacity,
            ane_weights,
            compile_budget,
        )?);
        if head_ranking != HeadRankingPlan::Baseline || head_ranking_timing {
            ane.as_mut()
                .ok_or("ANE owner missing")?
                .configure_head_ranking(head_ranking, head_ranking_timing)?;
        }
        ane_prepare_ms = Some(elapsed_ms(timer));
        ane_prepare_measurement = Some(measured.finish(1));
        eprintln!(
            "Both backends ready; Metal {:.1} ms, ANE {:.1} ms; accepting successive requests",
            metal_prepare_ms,
            ane_prepare_ms.unwrap_or_default()
        );
    }
    let references_requested = references.len();
    let mut references = references.into_iter();
    let mut cases = Vec::new();
    let mut completed_requests = 0_usize;
    let mut any_ane_execution = false;
    let mut last_report = None;
    loop {
        let reference = if let Some(reference) = references.next() {
            reference
        } else if interactive {
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line)? == 0 {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            let prompt = match encode_gemma4_user_prompt(&model_dir, &tokenizer, &line) {
                Ok(prompt) => prompt,
                Err(error) => {
                    eprintln!("Prompt rejected: {error}");
                    continue;
                }
            };
            if let Err(error) = validate_text_budget(&prompt, max_new_tokens, context_capacity) {
                eprintln!("Prompt rejected: {error}");
                continue;
            }
            Reference {
                path: None,
                prompt,
                generated: None,
                max_new_tokens,
            }
        } else {
            break;
        };
        let index = completed_requests;
        let compiler_calls_before = rvllm_apple::ane_linear::compile_budget_used();
        let reference_sha256 = reference
            .path
            .as_deref()
            .map(sha256_file)
            .transpose()?;
        let workload_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&reference.prompt)?)
        );
        let case_dir = output_dir
            .as_ref()
            .map(|dir| dir.join(format!("case-{index}")));
        let PrefillCase {
            start,
            prefill_sample_capture_ms,
            prefill_measurement,
            receipt: prefill_receipt,
        } = if interleave {
            prefill_case(
                &monitor,
                retained_metal
                    .as_mut()
                    .ok_or("resident Metal backend missing")?,
                &reference,
                &model_dir,
                layout_hash,
                case_dir.as_deref(),
                PrefillEvidenceOptions {
                    capture_seed: !text_mode || capture_layers,
                    candidate: kernels.research,
                },
            )?
        } else {
            starts.pop_front().ok_or("prepared prefill missing")?
        };
        let mut generated = vec![start.first_token.raw()];
        let prefill_times = start.times;
        let mut text_decoder = IncrementalTextDecoder::default();
        let mut streamed_text = String::new();
        if text_mode {
            stream_token(
                &mut text_decoder,
                &tokenizer,
                generated[0],
                &mut streamed_text,
            )?;
        }
        let mut steps = Vec::new();
        let mut import_ms = 0.0;
        let mut import_measurement = None;
        if generated.len() < reference.max_new_tokens && !eos.contains(&generated[0]) {
            if ane.is_none() {
                let measured = monitor.begin();
                let timer = Instant::now();
                ane = Some(GemmaAneDecode::load_with_compile_budget(
                    &model_dir,
                    context_capacity,
                    ane_weights,
                    compile_budget,
                )?);
                if head_ranking != HeadRankingPlan::Baseline || head_ranking_timing {
                    ane.as_mut()
                        .ok_or("ANE owner missing")?
                        .configure_head_ranking(head_ranking, head_ranking_timing)?;
                }
                ane_prepare_ms = Some(elapsed_ms(timer));
                ane_prepare_measurement = Some(measured.finish(1));
            }
            let decoder = ane.as_mut().ok_or("ANE decoder missing")?;
            let measured = monitor.begin();
            let timer = Instant::now();
            decoder.import_prefill_with_packing(&start.cache, kv_import_packing)?;
            import_ms = elapsed_ms(timer);
            import_measurement = Some(measured.finish(1));
            drop(start);
            while generated.len() < reference.max_new_tokens
                && !eos.contains(generated.last().ok_or("missing token")?)
            {
                let input = TokenId(*generated.last().ok_or("missing token")?);
                let position = reference.prompt.len() + generated.len() - 1;
                let mut layer_receipts = Vec::new();
                let mut ffn_input_receipts = Vec::new();
                let capture_layer_step = capture_layers && generated.len() == 1;
                let capture_ffn_step = capture_ffn_inputs && generated.len() <= 2;
                let measured = monitor.begin();
                let result = if capture_layer_step || capture_ffn_step {
                    decoder.decode_with_diagnostic_observers(
                        input,
                        &mut |layer, values| {
                            if !capture_layer_step {
                                return Ok(());
                            }
                            let file = format!("ane-position-{position}-layer-{layer:02}.fp16");
                            let hash = write_f16(
                                &case_dir
                                    .as_ref()
                                    .ok_or("capture directory missing")?
                                    .join(&file),
                                values,
                            )
                            .map_err(|e| e.to_string())?;
                            layer_receipts
                                .push(serde_json::json!({"layer":layer,"file":file,"sha256":hash}));
                            Ok(())
                        },
                        &mut |layer, values| {
                            if !capture_ffn_step {
                                return Ok(());
                            }
                            let file =
                                format!("ane-position-{position}-layer-{layer:02}-ffn-input.fp16");
                            let hash = write_f16(
                                &case_dir
                                    .as_ref()
                                    .ok_or("capture directory missing")?
                                    .join(&file),
                                values,
                            )
                            .map_err(|e| e.to_string())?;
                            ffn_input_receipts
                                .push(serde_json::json!({"layer":layer,"file":file,"sha256":hash}));
                            Ok(())
                        },
                    )?
                } else {
                    decoder.decode(input)?
                };
                let measurement = measured.finish(1);
                if result.position != position {
                    return Err("ANE position drift".into());
                }
                generated.push(result.token.raw());
                if text_mode {
                    stream_token(
                        &mut text_decoder,
                        &tokenizer,
                        result.token.raw(),
                        &mut streamed_text,
                    )?;
                }
                let head_observation=decoder.head_ranking_observation().map(|(stats,cpu_ms)|
                    serde_json::json!({"candidate":head_ranking.name(),"considered":stats.considered,
                        "transformed":stats.transformed,"pruned":stats.pruned,"cpu_ms":cpu_ms,
                        "phase":"host-softcap-top5-excluding-ANE-projection"}));
                let t = result.times;
                let step = serde_json::json!({
                    "position": position, "input_token":input.raw(),"next_token":result.token.raw(),
                    "top_five":result.top_five,"layer_states":layer_receipts,
                    "ffn_inputs":ffn_input_receipts,
                    "head_ranking":head_observation,
                    "qkv_ms":t.qkv_ms,"attention_ms":t.attention_ms,"output_ms":t.output_ms,
                    "ffn_ms":t.ffn_ms,"vocabulary_ms":t.vocabulary_ms,"host_ms":t.host_ms,"total_ms":t.total_ms,
                    "measurement":measurement,
                });
                if !text_mode {
                    eprintln!("{}", serde_json::to_string(&step)?);
                }
                if let Some(directory) = &case_dir {
                    std::fs::write(
                        directory.join(format!("step-{position}.json")),
                        serde_json::to_vec_pretty(&step)?,
                    )?;
                }
                steps.push(step);
            }
        }
        let matched = reference
            .generated
            .as_ref()
            .map(|expected| generated == *expected);
        let generated_text = tokenizer
            .decode(&generated, true)
            .map_err(|e| e.to_string())?;
        let compiler_calls_after = rvllm_apple::ane_linear::compile_budget_used();
        let compiler_calls_this_request = compiler_calls_after
            .checked_sub(compiler_calls_before)
            .ok_or("ANE compiler call counter reset during request")?;
        let route_evidence = serde_json::json!({
            "schema":"rvllm.kernel_game.route_evidence.v1",
            "candidate":kernels.research.name(),
            "command_completed":true,
            "reference_match":matched,
            "executable":{"path":executable_path.display().to_string(),"sha256":executable_sha256.clone()},
            "metallib":{"path":metallib_bf16.display().to_string(),"sha256":metallib_sha256.clone()},
            "generated_msl_sha256":generated_msl_sha256.clone(),
            "model_config_sha256":format!("{:x}",Sha256::digest(&config)),
            "reference_sha256":reference_sha256,
            "workload_sha256":workload_sha256,
            "compiler_calls_before":compiler_calls_before,
            "compiler_calls_after":compiler_calls_after,
            "compiler_calls_this_request":compiler_calls_this_request,
            "compile_free_request":compiler_calls_this_request==0,
            "prefill_dispatch":prefill_receipt["research_dispatch"].clone(),
            "output_tokens":generated.len(),
            "ane_decode_steps":steps.len(),
            "expected_output_tokens":reference.generated.as_ref().map(Vec::len),
            "counting_boundary":"compiler counter brackets full request; research dispatch counters bracket synchronous Metal prefill",
            "promotion_claim":false,
        });
        if text_mode {
            let tail = generated_text
                .strip_prefix(&streamed_text)
                .ok_or("streamed text differs from complete token decoding")?;
            std::io::stdout().lock().write_all(tail.as_bytes())?;
        }
        let case = serde_json::json!({
            "reference":reference.path,"prompt_token_ids":reference.prompt,
            "generated_tokens":generated,"generated_text":generated_text,"matches_reference":matched,
            "prefill_sample_capture_ms":prefill_sample_capture_ms,"ane_cache_import_ms":import_ms, "kv_import_packing":kv_import_packing.name(),
            "metal_prefill_complete_ms":prefill_times.metal_execution_ms,"metal_host_non_wait_ms":prefill_times.host_non_wait_ms,
            "metal_command_buffer_wait_ms":prefill_times.command_buffer_wait_ms,"kv_capture_ms":prefill_times.kv_capture_ms,
            "metal_gpu_execution_ms":prefill_times.gpu_execution_ms,
            "prefill_measurement":prefill_measurement,"ane_import_measurement":import_measurement,
            "metal_prefill_receipt":prefill_receipt,
            "kernel_game_route_evidence":route_evidence,
            "prefill_command_buffers":1,"metal_decode_steps":0,"ane_decode_steps":steps.len(),"steps":steps,
        });
        if let Some(directory) = &case_dir {
            std::fs::write(
                directory.join("result.json"),
                serde_json::to_vec_pretty(&case)?,
            )?;
        }
        if text_mode {
            println!();
            let decode_ms: f64 = steps
                .iter()
                .filter_map(|step| step["total_ms"].as_f64())
                .sum();
            eprintln!("{} prompt tokens, {} output tokens; prefill {:.1} ms; {} ANE steps in {:.1} ms ({:.2} tokens/sec)", reference.prompt.len(), generated.len(), prefill_sample_capture_ms, steps.len(), decode_ms, if decode_ms > 0.0 { steps.len() as f64 * 1000.0 / decode_ms } else { 0.0 });
        }
        completed_requests = completed_requests
            .checked_add(1)
            .ok_or("request count exhausted")?;
        any_ane_execution |= !steps.is_empty();
        // Per-case files contain complete receipts. An open-ended session only
        // keeps the latest receipt in memory and in its summary report.
        if interactive {
            cases.clear();
        }
        cases.push(case);
        let report = serde_json::json!({
            "schema":"rvllm.metal_prefill_ane_decode.v1","model_dir":model_dir,
            "config_sha256":format!("{:x}",Sha256::digest(&config)),
            "metal_prepare_ms":metal_prepare_ms,"ane_prepare_ms":ane_prepare_ms,
            "metal_prepare_measurement":metal_prepare_measurement,"ane_prepare_measurement":ane_prepare_measurement,
            "measurement_environment":measurement_environment,
            "execution_order":if interleave {"prefill-decode-per-request"}else{"all-prefills-then-decode"},
            "interactive_session_open":interactive,
            "metal_residency":if retain_metal {"retained-through-ane-decode"} else {"released-after-all-prefills-before-ane-load"},
            "metal_arena_before_release":{"capacity_bytes":metal_arena.capacity_bytes,"allocated_bytes":metal_arena.allocated_bytes,"regions":metal_arena.region_count},
            "ane_weight_plan":ane_weights.name(),
            "head_ranking_plan":head_ranking.name(),"head_ranking_timing":head_ranking_timing,
            "metal_research_candidate":kernels.research.name(),
            "kv_import_packing":kv_import_packing.name(),
            "stacked_ffn_checks_per_layer":ane.as_ref().and_then(GemmaAneDecode::stacked_ffn_checks_per_layer),
            "loaded_ane_programs":if ane.is_some(){ane_weights.program_count()}else{0},
            "ane_cache_policy":if compile_budget != 0 { "reuse-with-compile-budget" } else if ane_weights.cache_policy() == rvllm_apple::ane_linear::AneProgramCachePolicy::RequireExisting {"require-existing"} else {"compile"},
            "ane_compile_budget":compile_budget,"ane_compile_budget_used":rvllm_apple::ane_linear::compile_budget_used(),
            "ane_execution_verified":any_ane_execution,
            "requests_completed":completed_requests,
            "case_history":if interactive {"latest-request-only; full receipts in case-N/result.json"}else{"all-requests"},
            "all_references_match":!text_mode && cases.iter().all(|c| c["matches_reference"]==true),
            "references_requested":if text_mode {0}else{references_requested},"references_completed":if text_mode {0}else{cases.len()},
            "qualification_complete":!text_mode && cases.len()==references_requested && cases.iter().all(|c| c["matches_reference"]==true),
            "inference_complete":!interactive && cases.len()==references_requested,
            "layer_state_capture_enabled":capture_layers,"diagnostic_journal_enabled":std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some(),
            "ffn_input_capture_enabled":capture_ffn_inputs,
            "cpu_or_gpu_decode_fallback":false,"global_context_capacity":context_capacity,"cases":cases,
            "claim": if cases.iter().any(|c| c["ane_decode_steps"].as_u64().unwrap_or(0)>0) {
                "Metal prefill and first-token sampling, actual converted KV imported into all 48 ANE decode layers. Small norms, RoPE, residuals and sampling use CPU Rust. Timings include application scheduling; initialization is reported separately."
            } else { "Metal prefill completed; no ANE decode step was required or executed." },
        });
        if let Some(directory) = &output_dir {
            std::fs::write(
                directory.join("report.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
        }
        if output_dir.is_some() {
            last_report = Some(report);
        }
        eprintln!("Completed request {index}");
        if matched == Some(false) {
            return Err(format!(
                "ANE continuation differs from {}",
                reference
                    .path
                    .as_ref()
                    .ok_or("reference path missing")?
                    .display()
            )
            .into());
        }
    }
    if let (Some(directory), Some(mut report)) = (&output_dir, last_report) {
        report["interactive_session_open"] = false.into();
        report["inference_complete"] = true.into();
        std::fs::write(
            directory.join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
    }
    if let Some(directory) = &output_dir {
        eprintln!("{}", directory.join("report.json").display());
    }
    Ok(())
}

fn prefill_case(
    monitor: &PowerMonitor,
    metal: &mut ModelMetalBackend,
    reference: &Reference,
    model_dir: &Path,
    layout_hash: [u8; 32],
    case_dir: Option<&Path>,
    evidence: PrefillEvidenceOptions,
) -> Result<PrefillCase, Box<dyn std::error::Error>> {
    if let Some(directory) = case_dir {
        std::fs::create_dir(directory)?;
    }
    let owner = ReqId(1);
    let prompt_len = u32::try_from(reference.prompt.len())?;
    let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(
        metal
            .model_capacity()
            .ok_or("Metal capacity unavailable")?
            .physical_kv_pages,
        1,
    ))?;
    let chain = pool.allocate_chain(owner, prompt_len)?;
    let batch = BatchPlan::Prefill {
        req_ids: vec![owner],
        prompt_tokens_flat: reference.prompt.iter().copied().map(TokenId).collect(),
        cu_seqlens_q: vec![0, prompt_len],
        query_start_positions: vec![0],
        context_lens: vec![prompt_len],
        kv_chains: vec![Some(chain)],
    };
    let handoff = handoff_from_prefill_plan_with_paged_kv(
        &batch,
        HandoffKind::MetalPrefillToMetalDecode,
        None,
        &pool,
        layout_hash,
    )?;
    let before = metal.probe_perf_stats();
    let dispatch_before = metal
        .probe_research_dispatches()
        .ok_or("research counters unavailable")?;
    let measured = monitor.begin();
    let timer = Instant::now();
    let start = metal.prefill_for_ane(&handoff)?;
    let prefill_sample_capture_ms = elapsed_ms(timer);
    let measurement = measured.finish(reference.prompt.len());
    let after = metal.probe_perf_stats();
    let dispatch_after = metal
        .probe_research_dispatches()
        .ok_or("research counters unavailable")?;
    let dispatch =
        prefill_screen::dispatch_report(evidence.candidate.name(), dispatch_before, dispatch_after);
    let contract_passed = after.prefill_steps.checked_sub(before.prefill_steps) == Some(1)
        && after.decode_steps == before.decode_steps
        && after.command_buffers.checked_sub(before.command_buffers) == Some(1)
        && start.req_id == owner
        && start.next_position() == reference.prompt.len();
    let expected = reference
        .generated
        .as_ref()
        .and_then(|tokens| tokens.first())
        .copied();
    let receipt = serde_json::json!({
        "schema": "rvllm.metal_prefill_screen.case.v1",
        "reference": reference.path,
        "prompt_token_ids": reference.prompt,
        "first_generated_token": start.first_token.raw(),
        "expected_first_token": expected,
        "first_token_gate_passed": expected.map(|id| id == start.first_token.raw()),
        "full_reference_generated_token_count": reference.generated.as_ref().map(Vec::len),
        "comparison_scope": "first generated token only; not continuation or tensor qualification",
        "command_contract_passed": contract_passed,
        "prefill_command_buffers": after.command_buffers.checked_sub(before.command_buffers),
        "metal_decode_steps": after.decode_steps.checked_sub(before.decode_steps),
        "research_dispatch": dispatch.as_ref().ok(),
        "research_dispatch_error": dispatch.as_ref().err(),
        "prefill_sample_capture_ms": prefill_sample_capture_ms,
        "metal_gpu_execution_ms": start.times.gpu_execution_ms,
        "metal_prefill_complete_ms": start.times.metal_execution_ms,
        "metal_host_non_wait_ms": start.times.host_non_wait_ms,
        "metal_command_buffer_wait_ms": start.times.command_buffer_wait_ms,
        "kv_capture_ms": start.times.kv_capture_ms,
        "prefill_measurement": measurement,
        "ane_execution_verified": false,
        "qualification_complete": false,
        "tensor_oracle_passed": null,
        "performance_accepted": false,
    });
    // Preserve completed prefill evidence before a subsequent cache miss or
    // mismatch can terminate the process. This write is outside its timer.
    if let Some(directory) = case_dir {
        prefill_screen::write_new_json(&directory.join("prefill-result.json"), &receipt)?;
    }
    if !contract_passed {
        return Err("Metal prefill violated command-buffer, request or position contract".into());
    }
    dispatch?;
    if evidence.capture_seed {
        export_seed(
            case_dir.ok_or("capture directory missing")?,
            model_dir,
            &reference.prompt,
            &start,
        )?;
    }
    if let Some(expected) = &reference.generated {
        if start.first_token.raw() != expected[0] {
            return Err(format!(
                "Metal first token {} differs from reference {}",
                start.first_token.raw(),
                expected[0]
            )
            .into());
        }
    }
    Ok(PrefillCase {
        start,
        prefill_sample_capture_ms,
        prefill_measurement: measurement,
        receipt,
    })
}

fn validate_text_budget(
    prompt: &[u32],
    max_new_tokens: usize,
    context_capacity: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if max_new_tokens == 0
        || prompt.is_empty()
        || prompt
            .len()
            .checked_add(max_new_tokens - 1)
            .map_or(true, |total| total > context_capacity)
    {
        return Err(format!("{} prompt tokens and output limit {max_new_tokens} exceed context capacity {context_capacity}", prompt.len()).into());
    }
    Ok(())
}

fn stream_token(
    decoder: &mut IncrementalTextDecoder,
    tokenizer: &tokenizers::Tokenizer,
    token: u32,
    emitted: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(text) = decoder.step(tokenizer, token).map_err(|e| e.to_string())? {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(text.as_bytes())?;
        stdout.flush()?;
        emitted.push_str(&text);
    }
    Ok(())
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn export_seed(
    directory: &Path,
    model_dir: &Path,
    prompt: &[u32],
    start: &AneDecodeStart,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut layers = Vec::with_capacity(start.cache.layers.len());
    for (index, layer) in start.cache.layers.iter().enumerate() {
        let key_file = format!("layer-{index:02}-key.fp16");
        let value_file = format!("layer-{index:02}-value.fp16");
        let key_hash = write_f16(&directory.join(&key_file), &layer.keys)?;
        let value_hash = write_f16(&directory.join(&value_file), &layer.values)?;
        layers.push(serde_json::json!({"layer":index,"kv_heads":layer.shape.kv_heads,"head_dim":layer.shape.head_dim,
            "kv_elements_each":layer.keys.len(),"key_file":key_file,"key_sha256":key_hash,"value_file":value_file,"value_sha256":value_hash}));
    }
    let abi: String = start
        .cache
        .metal_numeric_abi
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let report = serde_json::json!({
        "schema":"rvllm.metal_ane_prefill_handoff.v2","model_dir":model_dir,"prompt_token_ids":prompt,
        "destination_dtype":"float16","source_dtype":match start.cache.source_dtype {PrefillScalarType::F16=>"float16",PrefillScalarType::Bf16=>"bfloat16"},
        "exported_tensor_order":"token, kv_head, head_dim; little-endian IEEE FP16",
        "cache_capture_after_collection":true,"ane_execution_verified":false,
        "first_generated_token":start.first_token.raw(),"ane_next_position":start.next_position(),"metal_numeric_abi":abi,"layers":layers,
    });
    std::fs::write(
        directory.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

fn write_f16(path: &Path, values: &[f16]) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = std::io::BufWriter::new(std::fs::File::create_new(path)?);
    let mut hash = Sha256::new();
    let mut bytes = [0; 4096];
    for chunk in values.chunks(bytes.len() / 2) {
        for (slot, value) in bytes.chunks_exact_mut(2).zip(chunk) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        let bytes = &bytes[..chunk.len() * 2];
        output.write_all(bytes)?;
        hash.update(bytes);
    }
    output.flush()?;
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::validate_text_budget;
    use sha2::Digest as _;

    #[test]
    fn file_identity_hashes_exact_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("artifact");
        std::fs::write(&path, b"kernel-game").unwrap();
        assert_eq!(
            super::sha256_file(&path).unwrap(),
            format!("{:x}", sha2::Sha256::digest(b"kernel-game"))
        );
    }

    #[test]
    fn output_budget_counts_only_tokens_consumed_by_decode() {
        assert!(validate_text_budget(&[2; 1024], 1, 1024).is_ok());
        assert!(validate_text_budget(&[2; 1001], 24, 1024).is_ok());
        assert!(validate_text_budget(&[2; 1002], 24, 1024).is_err());
        assert!(validate_text_budget(&[2; 1024], 2, 1024).is_err());
        assert!(validate_text_budget(&[2], 0, 1024).is_err());
        assert!(validate_text_budget(&[], 1, 1024).is_err());
        assert!(validate_text_budget(&[2; 2], usize::MAX, usize::MAX).is_err());
    }
}
