#![recursion_limit = "256"]

//! Apple Metal text inference workflow for Gemma tokenizers.
//!
//! This is a production-facing shape, not a production-readiness claim. The
//! backend sizes its Metal arena from the requested context and batch limits,
//! then fails clearly when the requested run exceeds those limits.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

const CLAIM: &str =
    "Apple Metal text inference workflow; not production-ready until acceptance gates pass";
const JSON_SCHEMA: &str = "rvllm.apple_metal_text_infer.v1";
const SESSION_JSON_SCHEMA: &str = "rvllm.apple_metal_text_session.v1";
const SESSION_PROFILE_SCHEMA: &str = "rvllm.apple_metal_text_session_profile.v1";
const TEXT_REFERENCE_MANIFEST_SCHEMA: &str = "rvllm.gemma4_e2b_hf_text_reference_suite.v1";
const LARGE_MODEL_ENV: &str = "RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE";
const MAX_TOTAL_TOKENS_ENV: &str = "RVLLM_METAL_MAX_TOTAL_TOKENS";
const LEGACY_MAX_TOKENS_ENV: &str = "RVLLM_METAL_MAX_PROBE_TOKENS";
const MAX_BATCH_TOKENS_ENV: &str = "RVLLM_METAL_MAX_BATCH_TOKENS";
const MAX_BATCH_SEQUENCES_ENV: &str = "RVLLM_METAL_MAX_BATCH_SEQUENCES";
const DEFAULT_MAX_METAL_TOTAL_TOKENS: usize = 2048;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    model_dir: PathBuf,
    prompt: Option<String>,
    prompts_jsonl: Option<PathBuf>,
    reference_manifest: Option<PathBuf>,
    session_backend: SessionBackend,
    max_new_tokens: usize,
    max_total_tokens: Option<usize>,
    eos_token_ids: Vec<u32>,
    no_bos: bool,
    large_model_opt_in: bool,
    hf_reference: Option<PathBuf>,
    report: Option<PathBuf>,
    case_timeout_seconds: Option<u64>,
    profile_samples: usize,
    profile_report: Option<PathBuf>,
    top_logits: usize,
    json_output: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionBackend {
    Direct,
    Engine,
}

impl SessionBackend {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Engine => "engine",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct InferReport {
    model_dir: PathBuf,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    max_new_tokens: usize,
    generated_token_ids: Vec<u32>,
    output_token_ids: Vec<u32>,
    generated_text: String,
    output_text: String,
    finish_reason: FinishReason,
    prepare_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    tok_per_s: f64,
    arena_bytes: usize,
    library_compiles: u64,
    pipeline_state_compiles: u64,
    last_step_gpu_execution_ns: Option<u64>,
    research_dispatch: serde_json::Value,
    command_buffers: u64,
    encoders: u64,
    embedding_encoders: u64,
    ple_encoders: u64,
    layer_encoders: u64,
    layer_scale_encoder_fusions: u64,
    final_sample_encoders: u64,
    final_logits_encoders: u64,
    forced_waits: u64,
    cpu_wall_ns: u64,
    cpu_encode_ns: u64,
    command_buffer_wait_ns: u64,
    last_step_tokens: u64,
    last_step_command_buffers: u64,
    last_step_encoders: u64,
    last_step_forced_waits: u64,
    last_step_cpu_wall_ns: u64,
    last_step_cpu_encode_ns: u64,
    last_step_command_buffer_wait_ns: u64,
    debug_sync: bool,
    large_model_opt_in: bool,
    max_supported_total_tokens: usize,
    metal_compute_dtype: String,
    metal_weight_dtype: String,
    metal_moe_router_weight_dtype: String,
    diagnostic_top_logits: Vec<TopLogit>,
}

#[derive(Debug, Clone, PartialEq)]
struct TopLogit {
    token_id: u32,
    logit: f32,
}

#[derive(Debug)]
struct HfReference {
    path: PathBuf,
    prompt_token_ids: Vec<u32>,
    decode_steps: usize,
    generated_tokens: Vec<u32>,
}

#[derive(Debug)]
struct ReferenceComparison {
    path: PathBuf,
    matched: bool,
    mismatches: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinishReason {
    Eos,
    Length,
}

impl FinishReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::Length => "length",
        }
    }
}

fn parse_token_ids(flag: &str, raw: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let token_id = part
            .parse::<u32>()
            .map_err(|err| format!("invalid {flag} token id {part:?}: {err}"))?;
        out.push(token_id);
    }
    if out.is_empty() {
        return Err(format!("{flag} must contain at least one token id"));
    }
    Ok(out)
}

fn parse_positive_usize(flag: &str, raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag} value {raw:?}: {err}"))?;
    if value == 0 {
        return Err(format!("{flag} must be positive"));
    }
    Ok(value)
}

fn parse_session_backend(raw: &str) -> Result<SessionBackend, String> {
    match raw {
        "direct" => Ok(SessionBackend::Direct),
        "engine" => Ok(SessionBackend::Engine),
        other => Err(format!(
            "--session-backend must be direct or engine; got {other:?}"
        )),
    }
}

fn parse_args_from<I, S>(args: I) -> Result<CliArgs, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut model_dir = None;
    let mut prompt = None;
    let mut prompts_jsonl = None;
    let mut reference_manifest = None;
    let mut session_backend = SessionBackend::Direct;
    let mut max_new_tokens = 1usize;
    let mut max_total_tokens = None;
    let mut eos_token_ids = Vec::new();
    let mut no_bos = false;
    let mut large_model_opt_in = false;
    let mut hf_reference = None;
    let mut report = None;
    let mut case_timeout_seconds = None;
    let mut profile_samples = 0usize;
    let mut profile_report = None;
    let mut top_logits = 0usize;
    let mut json_output = false;

    let mut iter = args.into_iter().map(Into::into).peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--model-dir requires a value".to_owned())?,
                ));
            }
            "--prompt" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--prompt requires a value".to_owned())?;
                if value.is_empty() {
                    return Err("--prompt must not be empty".to_owned());
                }
                prompt = Some(value);
            }
            "--prompts-jsonl" => {
                prompts_jsonl =
                    Some(PathBuf::from(iter.next().ok_or_else(|| {
                        "--prompts-jsonl requires a value".to_owned()
                    })?));
            }
            "--reference-manifest" => {
                reference_manifest =
                    Some(PathBuf::from(iter.next().ok_or_else(|| {
                        "--reference-manifest requires a value".to_owned()
                    })?));
            }
            "--session-backend" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--session-backend requires a value".to_owned())?;
                session_backend = parse_session_backend(&value)?;
            }
            "--max-new-tokens" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--max-new-tokens requires a value".to_owned())?;
                max_new_tokens = parse_positive_usize("--max-new-tokens", &value)?;
                validate_u32_count("--max-new-tokens", max_new_tokens)?;
            }
            "--max-total-tokens" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--max-total-tokens requires a value".to_owned())?;
                max_total_tokens = Some(parse_max_total_tokens("--max-total-tokens", &value)?);
            }
            "--eos-token-ids" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--eos-token-ids requires a value".to_owned())?;
                eos_token_ids = parse_token_ids("--eos-token-ids", &value)?;
            }
            "--no-bos" => {
                no_bos = true;
            }
            "--large-model-opt-in" => {
                large_model_opt_in = true;
            }
            "--hf-reference" => {
                hf_reference =
                    Some(PathBuf::from(iter.next().ok_or_else(|| {
                        "--hf-reference requires a value".to_owned()
                    })?));
            }
            "--report" => {
                report = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--report requires a value".to_owned())?,
                ));
            }
            "--case-timeout-seconds" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--case-timeout-seconds requires a value".to_owned())?;
                case_timeout_seconds =
                    Some(parse_positive_usize("--case-timeout-seconds", &value)? as u64);
            }
            "--profile-samples" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--profile-samples requires a value".to_owned())?;
                profile_samples = parse_positive_usize("--profile-samples", &value)?;
            }
            "--profile-report" => {
                profile_report =
                    Some(PathBuf::from(iter.next().ok_or_else(|| {
                        "--profile-report requires a value".to_owned()
                    })?));
            }
            "--top-logits" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--top-logits requires a value".to_owned())?;
                top_logits = parse_positive_usize("--top-logits", &value)?;
            }
            "--json" => {
                json_output = true;
            }
            "-h" | "--help" => return Err(usage()),
            other if other.starts_with('-') => return Err(format!("unknown argument: {other}")),
            other => return Err(format!("unexpected positional argument: {other}")),
        }
    }

    let session_input_count = usize::from(prompt.is_some())
        + usize::from(prompts_jsonl.is_some())
        + usize::from(reference_manifest.is_some());
    if session_input_count != 1 {
        return Err(
            "exactly one of --prompt, --prompts-jsonl, or --reference-manifest is required"
                .to_owned(),
        );
    }
    if prompt.is_some() && session_backend != SessionBackend::Direct {
        return Err("--session-backend is only valid for session inputs".to_owned());
    }
    if prompt.is_some() && report.is_some() {
        return Err("--report is only valid for session inputs".to_owned());
    }
    if prompt.is_some() && profile_samples > 0 {
        return Err("--profile-samples is only valid for session inputs".to_owned());
    }
    if prompt.is_some() && case_timeout_seconds.is_some() {
        return Err("--case-timeout-seconds is only valid for session inputs".to_owned());
    }
    if prompt.is_some() && profile_report.is_some() {
        return Err("--profile-report is only valid for session inputs".to_owned());
    }
    if prompt.is_none() && top_logits > 0 {
        return Err("--top-logits is only valid for single-prompt inference".to_owned());
    }
    if profile_report.is_some() && profile_samples == 0 {
        return Err("--profile-report requires --profile-samples".to_owned());
    }

    Ok(CliArgs {
        model_dir: model_dir.ok_or_else(|| "--model-dir is required".to_owned())?,
        prompt,
        prompts_jsonl,
        reference_manifest,
        session_backend,
        max_new_tokens,
        max_total_tokens,
        eos_token_ids,
        no_bos,
        large_model_opt_in,
        hf_reference,
        report,
        case_timeout_seconds,
        profile_samples,
        profile_report,
        top_logits,
        json_output,
    })
}

fn usage() -> String {
    "usage: rvllm_metal_infer --model-dir <DIR> (--prompt <TEXT> | --prompts-jsonl <PATH> | --reference-manifest <PATH>) \
     [--session-backend direct|engine] [--max-new-tokens N] [--max-total-tokens N] \
     [--eos-token-ids IDS] [--no-bos] [--large-model-opt-in] [--hf-reference <JSON>] \
     [--report <JSON>] [--case-timeout-seconds N] [--profile-samples N] \
     [--profile-report <JSON>] [--top-logits N] [--json]"
        .to_owned()
}

fn tokenizer_path(model_dir: &std::path::Path) -> PathBuf {
    model_dir.join("tokenizer.json")
}

fn load_tokenizer(model_dir: &std::path::Path) -> Result<tokenizers::Tokenizer, String> {
    let path = tokenizer_path(model_dir);
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|err| format!("load tokenizer {}: {err}", path.display()))
}

fn prompt_token_ids(args: &CliArgs, tokenizer: &tokenizers::Tokenizer) -> Result<Vec<u32>, String> {
    let prompt = args
        .prompt
        .as_deref()
        .ok_or_else(|| "--prompt is required for single-prompt inference".to_owned())?;
    tokenize_prompt(tokenizer, prompt, args.no_bos).map_err(|err| format!("--prompt {err}"))
}

fn tokenize_prompt(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    no_bos: bool,
) -> Result<Vec<u32>, String> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|err| format!("tokenize prompt: {err}"))?;
    let mut token_ids = encoding.get_ids().to_vec();
    if !no_bos {
        token_ids.insert(0, 2);
    }
    if token_ids.is_empty() {
        return Err("prompt produced zero token IDs".to_owned());
    }
    Ok(token_ids)
}

fn parse_max_total_tokens(flag: &str, raw: &str) -> Result<usize, String> {
    let value = parse_positive_usize(flag, raw)?;
    validate_u32_count(flag, value)?;
    Ok(value)
}

fn validate_u32_count(label: &str, value: usize) -> Result<(), String> {
    if u32::try_from(value).is_err() {
        return Err(format!("{label} must be at most {}", u32::MAX));
    }
    Ok(())
}

fn env_configured_max_total_tokens() -> Result<Option<usize>, String> {
    match std::env::var(MAX_TOTAL_TOKENS_ENV) {
        Ok(raw) => parse_max_total_tokens(MAX_TOTAL_TOKENS_ENV, raw.trim()).map(Some),
        Err(std::env::VarError::NotPresent) => match std::env::var(LEGACY_MAX_TOKENS_ENV) {
            Ok(raw) => parse_max_total_tokens(LEGACY_MAX_TOKENS_ENV, raw.trim()).map(Some),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{LEGACY_MAX_TOKENS_ENV} must be valid UTF-8"))
            }
        },
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{MAX_TOTAL_TOKENS_ENV} must be valid UTF-8"))
        }
    }
}

fn default_max_total_tokens(required_total_tokens: usize) -> Result<usize, String> {
    Ok(env_configured_max_total_tokens()?
        .unwrap_or(DEFAULT_MAX_METAL_TOTAL_TOKENS)
        .max(required_total_tokens))
}

fn effective_max_total_tokens(
    args: &CliArgs,
    required_total_tokens: usize,
) -> Result<usize, String> {
    args.max_total_tokens
        .map(Ok)
        .unwrap_or_else(|| default_max_total_tokens(required_total_tokens))
}

fn validate_token_budget(
    prompt_len: usize,
    max_new_tokens: usize,
    max_total_tokens: usize,
) -> Result<(), String> {
    if prompt_len == 0 {
        return Err("Metal prompt must contain at least one token".to_owned());
    }
    if max_new_tokens == 0 {
        return Err("Metal max_new_tokens must be positive".to_owned());
    }
    validate_u32_count("Metal max_total_tokens", max_total_tokens)?;
    let total_tokens = prompt_len
        .checked_add(max_new_tokens)
        .ok_or_else(|| "Metal token budget overflow".to_owned())?;
    validate_u32_count("Metal total token count", total_tokens)?;
    if total_tokens > max_total_tokens {
        return Err(format!(
            "current Metal E2B workflow supports prompt token count + max_new_tokens <= {max_total_tokens}; got {} + {}",
            prompt_len, max_new_tokens
        ));
    }
    Ok(())
}

fn env_large_model_opted_in() -> bool {
    std::env::var(LARGE_MODEL_ENV).ok().as_deref() == Some("1")
}

struct EnvGuard {
    name: &'static str,
    previous: Option<OsString>,
    changed: bool,
}

impl EnvGuard {
    fn set_if(name: &'static str, value: bool) -> Self {
        if value {
            Self::set_value_if(name, Some("1".to_owned()))
        } else {
            Self {
                name,
                previous: None,
                changed: false,
            }
        }
    }

    fn set_value_if(name: &'static str, value: Option<String>) -> Self {
        if let Some(value) = value {
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
            Self {
                name,
                previous,
                changed: true,
            }
        } else {
            Self {
                name,
                previous: None,
                changed: false,
            }
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if !self.changed {
            return;
        }
        if let Some(previous) = self.previous.as_ref() {
            std::env::set_var(self.name, previous);
        } else {
            std::env::remove_var(self.name);
        }
    }
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn json_u32_array(value: &serde_json::Value, field: &str) -> Result<Vec<u32>, String> {
    let items = value
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("HF reference missing array field {field:?}"))?;
    items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            let raw = item
                .as_u64()
                .ok_or_else(|| format!("HF reference {field}[{idx}] is not an unsigned integer"))?;
            u32::try_from(raw)
                .map_err(|_| format!("HF reference {field}[{idx}] exceeds u32: {raw}"))
        })
        .collect()
}

fn parse_hf_reference(path: PathBuf) -> Result<HfReference, String> {
    let raw = std::fs::read_to_string(&path)
        .map_err(|err| format!("read HF reference {}: {err}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("parse HF reference {}: {err}", path.display()))?;
    let prompt_token_ids = json_u32_array(&value, "prompt_token_ids")?;
    let generated_tokens = json_u32_array(&value, "generated_tokens")?;
    let decode_steps = value
        .get("decode_steps")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "HF reference missing unsigned integer field \"decode_steps\"".to_owned())
        .and_then(|raw| {
            usize::try_from(raw).map_err(|_| format!("HF reference decode_steps too large: {raw}"))
        })?;
    Ok(HfReference {
        path,
        prompt_token_ids,
        decode_steps,
        generated_tokens,
    })
}

fn compare_hf_reference(report: &InferReport, reference: &HfReference) -> ReferenceComparison {
    compare_hf_reference_parts(
        &report.prompt_token_ids,
        report.max_new_tokens,
        &report.generated_token_ids,
        reference,
    )
}

fn compare_hf_reference_parts(
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    generated_token_ids: &[u32],
    reference: &HfReference,
) -> ReferenceComparison {
    let mut mismatches = Vec::new();
    if prompt_token_ids != reference.prompt_token_ids.as_slice() {
        mismatches.push(format!(
            "prompt_token_ids differ: metal={:?} reference={:?}",
            prompt_token_ids, reference.prompt_token_ids
        ));
    }
    if max_new_tokens != reference.decode_steps {
        mismatches.push(format!(
            "max_new_tokens/decode_steps differ: metal={} reference={}",
            max_new_tokens, reference.decode_steps
        ));
    }
    if generated_token_ids != reference.generated_tokens.as_slice() {
        mismatches.push(format!(
            "generated_token_ids differ: metal={:?} reference={:?}",
            generated_token_ids, reference.generated_tokens
        ));
    }
    ReferenceComparison {
        path: reference.path.clone(),
        matched: mismatches.is_empty(),
        mismatches,
    }
}

fn hf_reference_value(comparison: Option<&ReferenceComparison>) -> serde_json::Value {
    comparison.map_or(serde_json::Value::Null, |comparison| {
        serde_json::json!({
            "path": comparison.path,
            "matched": comparison.matched,
            "mismatches": comparison.mismatches,
        })
    })
}

fn top_logits_value(top_logits: &[TopLogit]) -> serde_json::Value {
    serde_json::Value::Array(
        top_logits
            .iter()
            .map(|item| {
                serde_json::json!({
                    "token_id": item.token_id,
                    "logit": item.logit,
                })
            })
            .collect(),
    )
}

fn report_value(
    report: &InferReport,
    comparison: Option<&ReferenceComparison>,
) -> serde_json::Value {
    serde_json::json!({
        "schema": JSON_SCHEMA,
        "claim": CLAIM,
        "metal_compute_dtype": report.metal_compute_dtype,
        "metal_weight_dtype": report.metal_weight_dtype,
        "metal_moe_router_weight_dtype": report.metal_moe_router_weight_dtype,
        "diagnostic_top_logits": top_logits_value(&report.diagnostic_top_logits),
        "model_dir": report.model_dir,
        "prompt": report.prompt,
        "prompt_token_ids": report.prompt_token_ids,
        "max_new_tokens": report.max_new_tokens,
        "generated_token_ids": report.generated_token_ids,
        "output_token_ids": report.output_token_ids,
        "generated_text": report.generated_text,
        "output_text": report.output_text,
        "finish_reason": report.finish_reason.as_str(),
        "prepare_ms": report.prepare_ms,
        "prefill_ms": report.prefill_ms,
        "decode_ms": report.decode_ms,
        "tok_per_s": report.tok_per_s,
        "arena_bytes": report.arena_bytes,
        "library_compiles": report.library_compiles,
        "pipeline_state_compiles": report.pipeline_state_compiles,
        "last_step_gpu_execution_ns": report.last_step_gpu_execution_ns,
        "research_dispatch": report.research_dispatch,
        "command_buffers": report.command_buffers,
        "encoders": report.encoders,
        "embedding_encoders": report.embedding_encoders,
        "ple_encoders": report.ple_encoders,
        "layer_encoders": report.layer_encoders,
        "layer_scale_encoder_fusions": report.layer_scale_encoder_fusions,
        "final_sample_encoders": report.final_sample_encoders,
        "final_logits_encoders": report.final_logits_encoders,
        "encoder_counts_by_kernel_family": {
            "embedding": report.embedding_encoders,
            "ple_input": report.ple_encoders,
            "layer_body": report.layer_encoders,
            "layer_scale_fused": report.layer_scale_encoder_fusions,
            "final_sample": report.final_sample_encoders,
            "final_logits_diagnostic": report.final_logits_encoders,
        },
        "forced_waits": report.forced_waits,
        "cpu_wall_ns": report.cpu_wall_ns,
        "cpu_encode_ns": report.cpu_encode_ns,
        "command_buffer_wait_ns": report.command_buffer_wait_ns,
        "last_step_tokens": report.last_step_tokens,
        "last_step_command_buffers": report.last_step_command_buffers,
        "last_step_encoders": report.last_step_encoders,
        "last_step_forced_waits": report.last_step_forced_waits,
        "last_step_cpu_wall_ns": report.last_step_cpu_wall_ns,
        "last_step_cpu_encode_ns": report.last_step_cpu_encode_ns,
        "last_step_command_buffer_wait_ns": report.last_step_command_buffer_wait_ns,
        "debug_sync": report.debug_sync,
        "large_model_opt_in": report.large_model_opt_in,
        "max_supported_total_tokens": report.max_supported_total_tokens,
        "hf_reference": hf_reference_value(comparison),
    })
}

fn print_text_report(report: &InferReport, comparison: Option<&ReferenceComparison>) {
    println!("claim: {CLAIM}");
    println!("model_dir: {}", report.model_dir.display());
    println!("prompt: {}", report.prompt);
    println!(
        "prompt_token_ids: {}",
        serde_json::to_string(&report.prompt_token_ids).expect("serialize prompt token IDs")
    );
    println!("max_new_tokens: {}", report.max_new_tokens);
    println!(
        "generated_token_ids: {}",
        serde_json::to_string(&report.generated_token_ids).expect("serialize generated token IDs")
    );
    println!(
        "output_token_ids: {}",
        serde_json::to_string(&report.output_token_ids).expect("serialize output token IDs")
    );
    println!("finish_reason: {}", report.finish_reason.as_str());
    println!("generated_text: {}", report.generated_text);
    println!("output_text: {}", report.output_text);
    println!("prepare_ms: {:.3}", report.prepare_ms);
    println!("prefill_ms: {:.3}", report.prefill_ms);
    println!("decode_ms: {:.3}", report.decode_ms);
    println!("tok_per_s: {:.6}", report.tok_per_s);
    println!("arena_bytes: {}", report.arena_bytes);
    println!("command_buffers: {}", report.command_buffers);
    println!("encoders: {}", report.encoders);
    println!("embedding_encoders: {}", report.embedding_encoders);
    println!("ple_encoders: {}", report.ple_encoders);
    println!("layer_encoders: {}", report.layer_encoders);
    println!(
        "layer_scale_encoder_fusions: {}",
        report.layer_scale_encoder_fusions
    );
    println!("final_sample_encoders: {}", report.final_sample_encoders);
    println!("final_logits_encoders: {}", report.final_logits_encoders);
    println!("forced_waits: {}", report.forced_waits);
    println!("cpu_wall_ns: {}", report.cpu_wall_ns);
    println!("cpu_encode_ns: {}", report.cpu_encode_ns);
    println!("command_buffer_wait_ns: {}", report.command_buffer_wait_ns);
    println!("debug_sync: {}", report.debug_sync);
    println!("large_model_opt_in: {}", report.large_model_opt_in);
    println!(
        "max_supported_total_tokens: {}",
        report.max_supported_total_tokens
    );
    if !report.diagnostic_top_logits.is_empty() {
        println!(
            "diagnostic_top_logits: {}",
            serde_json::to_string(&top_logits_value(&report.diagnostic_top_logits))
                .expect("serialize diagnostic top logits")
        );
    }
    if let Some(comparison) = comparison {
        println!("hf_reference: {}", comparison.path.display());
        println!("hf_reference_match: {}", comparison.matched);
        println!(
            "hf_reference_mismatches: {}",
            serde_json::to_string(&comparison.mismatches).expect("serialize reference mismatches")
        );
    }
}

fn print_json_report(report: &InferReport, comparison: Option<&ReferenceComparison>) {
    println!(
        "{}",
        serde_json::to_string_pretty(&report_value(report, comparison))
            .expect("serialize JSON infer report")
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SessionCaseSpec {
    name: String,
    prompt: String,
    max_new_tokens: usize,
    max_total_tokens: Option<usize>,
    no_bos: bool,
    hf_reference: Option<PathBuf>,
}

#[derive(Debug)]
struct PreparedSessionCase {
    spec: SessionCaseSpec,
    prompt_token_ids: Vec<u32>,
    max_supported_total_tokens: usize,
    reference: Option<HfReference>,
}

#[derive(Debug)]
struct SessionCaseReport {
    name: String,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    max_new_tokens: usize,
    generated_token_ids: Vec<u32>,
    output_token_ids: Vec<u32>,
    generated_text: String,
    output_text: String,
    finish_reason: FinishReason,
    prefill_ms: f64,
    decode_ms: f64,
    tok_per_s: f64,
    library_compiles: u64,
    pipeline_state_compiles: u64,
    last_step_gpu_execution_ns: Option<u64>,
    research_dispatch: serde_json::Value,
    command_buffers: u64,
    encoders: u64,
    embedding_encoders: u64,
    ple_encoders: u64,
    layer_encoders: u64,
    layer_scale_encoder_fusions: u64,
    final_sample_encoders: u64,
    final_logits_encoders: u64,
    forced_waits: u64,
    cpu_wall_ns: u64,
    cpu_encode_ns: u64,
    command_buffer_wait_ns: u64,
    last_step_tokens: u64,
    last_step_command_buffers: u64,
    last_step_encoders: u64,
    last_step_forced_waits: u64,
    last_step_cpu_wall_ns: u64,
    last_step_cpu_encode_ns: u64,
    last_step_command_buffer_wait_ns: u64,
    max_supported_total_tokens: usize,
    comparison: Option<ReferenceComparison>,
}

fn json_string_field<'a>(
    value: &'a serde_json::Value,
    field: &str,
    case_name: &str,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|item| !item.is_empty())
        .ok_or_else(|| format!("session case {case_name} missing non-empty string field {field:?}"))
}

fn json_optional_string_field<'a>(value: &'a serde_json::Value, field: &str) -> Option<&'a str> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|item| !item.is_empty())
}

fn json_positive_usize_field(
    value: &serde_json::Value,
    field: &str,
    case_name: &str,
) -> Result<Option<usize>, String> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    let Some(raw) = raw.as_u64() else {
        return Err(format!(
            "session case {case_name} field {field:?} must be a positive integer"
        ));
    };
    let parsed = usize::try_from(raw)
        .map_err(|_| format!("session case {case_name} field {field:?} is too large: {raw}"))?;
    if parsed == 0 {
        return Err(format!(
            "session case {case_name} field {field:?} must be positive"
        ));
    }
    Ok(Some(parsed))
}

fn json_optional_bool_field(
    value: &serde_json::Value,
    field: &str,
    case_name: &str,
) -> Result<Option<bool>, String> {
    match value.get(field) {
        Some(raw) => raw
            .as_bool()
            .map(Some)
            .ok_or_else(|| format!("session case {case_name} field {field:?} must be boolean")),
        None => Ok(None),
    }
}

fn resolve_relative_path(base: &std::path::Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn flag_value(command: &[String], flag: &str) -> Option<String> {
    command
        .iter()
        .position(|item| item == flag)
        .and_then(|idx| command.get(idx + 1))
        .cloned()
}

fn session_case_from_value(
    value: &serde_json::Value,
    default_name: String,
    base_dir: &std::path::Path,
    args: &CliArgs,
) -> Result<SessionCaseSpec, String> {
    if !value.is_object() {
        return Err(format!("session case {default_name} must be a JSON object"));
    }
    let name = json_optional_string_field(value, "name")
        .unwrap_or(default_name.as_str())
        .to_owned();
    let prompt = json_string_field(value, "prompt", &name)?.to_owned();
    let max_new_tokens =
        json_positive_usize_field(value, "max_new_tokens", &name)?.unwrap_or(args.max_new_tokens);
    let max_total_tokens =
        json_positive_usize_field(value, "max_total_tokens", &name)?.or(args.max_total_tokens);
    if let Some(value) = max_total_tokens {
        parse_max_total_tokens("max_total_tokens", &value.to_string())?;
    }
    let no_bos = json_optional_bool_field(value, "no_bos", &name)?.unwrap_or(args.no_bos);
    let reference = json_optional_string_field(value, "hf_reference")
        .or_else(|| json_optional_string_field(value, "reference_path"))
        .map(|path| resolve_relative_path(base_dir, path))
        .or_else(|| args.hf_reference.clone());
    Ok(SessionCaseSpec {
        name,
        prompt,
        max_new_tokens,
        max_total_tokens,
        no_bos,
        hf_reference: reference,
    })
}

fn load_jsonl_session_cases(
    args: &CliArgs,
    path: &std::path::Path,
) -> Result<Vec<SessionCaseSpec>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("read prompts JSONL {}: {err}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut cases = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        let line_no = idx + 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).map_err(|err| {
            format!(
                "parse prompts JSONL {} line {line_no}: {err}",
                path.display()
            )
        })?;
        cases.push(session_case_from_value(
            &value,
            format!("case_{line_no}"),
            base_dir,
            args,
        )?);
    }
    if cases.is_empty() {
        return Err(format!(
            "prompts JSONL {} did not contain any cases",
            path.display()
        ));
    }
    validate_unique_case_names(&cases)?;
    Ok(cases)
}

fn manifest_reference_path(
    manifest_path: &std::path::Path,
    manifest: &serde_json::Value,
    raw: &str,
) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        return path;
    }
    manifest
        .get("output_dir")
        .and_then(serde_json::Value::as_str)
        .filter(|item| !item.is_empty())
        .map(|output_dir| PathBuf::from(output_dir).join(&path))
        .unwrap_or_else(|| {
            manifest_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(path)
        })
}

fn load_manifest_session_cases(
    args: &CliArgs,
    manifest_path: &std::path::Path,
) -> Result<Vec<SessionCaseSpec>, String> {
    let raw = std::fs::read_to_string(manifest_path)
        .map_err(|err| format!("read reference manifest {}: {err}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "parse reference manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    let schema = manifest
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "reference manifest missing schema".to_owned())?;
    if schema != TEXT_REFERENCE_MANIFEST_SCHEMA {
        return Err(format!(
            "unexpected reference manifest schema {schema:?}; expected {TEXT_REFERENCE_MANIFEST_SCHEMA:?}"
        ));
    }
    let raw_cases = manifest
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "reference manifest must contain a non-empty cases array".to_owned())?;
    if raw_cases.is_empty() {
        return Err("reference manifest must contain a non-empty cases array".to_owned());
    }
    let mut cases = Vec::with_capacity(raw_cases.len());
    for (idx, raw_case) in raw_cases.iter().enumerate() {
        let case_name = raw_case
            .get("name")
            .and_then(serde_json::Value::as_str)
            .filter(|item| !item.is_empty())
            .map(ToOwned::to_owned)
            .ok_or_else(|| {
                format!(
                    "reference manifest case {} is missing required non-empty name",
                    idx + 1
                )
            })?;
        let command = raw_case
            .get("command")
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| item.as_str().unwrap_or_default().to_owned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let prompt = raw_case
            .get("prompt_text")
            .and_then(serde_json::Value::as_str)
            .or_else(|| raw_case.get("prompt").and_then(serde_json::Value::as_str))
            .map(ToOwned::to_owned)
            .or_else(|| flag_value(&command, "--prompt-text"))
            .filter(|item| !item.is_empty())
            .ok_or_else(|| format!("reference manifest case {case_name} is missing prompt_text"))?;
        let max_new_tokens = json_positive_usize_field(raw_case, "decode_steps", &case_name)?
            .or(json_positive_usize_field(
                raw_case,
                "max_new_tokens",
                &case_name,
            )?)
            .or_else(|| flag_value(&command, "--decode-steps").and_then(|raw| raw.parse().ok()))
            .or_else(|| flag_value(&command, "--max-new-tokens").and_then(|raw| raw.parse().ok()))
            .ok_or_else(|| {
                format!(
                    "reference manifest case {case_name} is missing decode_steps/max_new_tokens"
                )
            })?;
        if max_new_tokens == 0 {
            return Err(format!(
                "reference manifest case {case_name} decode_steps/max_new_tokens must be positive"
            ));
        }
        let max_total_tokens = json_positive_usize_field(raw_case, "max_total_tokens", &case_name)?
            .or(args.max_total_tokens);
        if let Some(value) = max_total_tokens {
            parse_max_total_tokens("max_total_tokens", &value.to_string())?;
        }
        let reference_raw = raw_case
            .get("output")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                raw_case
                    .get("reference")
                    .and_then(serde_json::Value::as_str)
            })
            .or_else(|| {
                raw_case
                    .get("reference_path")
                    .and_then(serde_json::Value::as_str)
            })
            .or_else(|| raw_case.get("path").and_then(serde_json::Value::as_str))
            .filter(|item| !item.is_empty())
            .ok_or_else(|| {
                format!("reference manifest case {case_name} is missing output/reference path")
            })?;
        let no_bos =
            json_optional_bool_field(raw_case, "no_bos", &case_name)?.unwrap_or(args.no_bos);
        cases.push(SessionCaseSpec {
            name: case_name,
            prompt,
            max_new_tokens,
            max_total_tokens,
            no_bos,
            hf_reference: Some(manifest_reference_path(
                manifest_path,
                &manifest,
                reference_raw,
            )),
        });
    }
    validate_unique_case_names(&cases)?;
    Ok(cases)
}

fn validate_unique_case_names(cases: &[SessionCaseSpec]) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for case in cases {
        if !seen.insert(case.name.as_str()) {
            return Err(format!("duplicate session case name: {}", case.name));
        }
    }
    Ok(())
}

fn load_session_case_specs(args: &CliArgs) -> Result<Vec<SessionCaseSpec>, String> {
    match (&args.prompts_jsonl, &args.reference_manifest) {
        (Some(path), None) => load_jsonl_session_cases(args, path),
        (None, Some(path)) => load_manifest_session_cases(args, path),
        _ => Err("session mode requires --prompts-jsonl or --reference-manifest".to_owned()),
    }
}

fn prepare_session_cases(
    tokenizer: &tokenizers::Tokenizer,
    specs: Vec<SessionCaseSpec>,
) -> Result<Vec<PreparedSessionCase>, String> {
    specs
        .into_iter()
        .map(|spec| {
            let prompt_token_ids = tokenize_prompt(tokenizer, &spec.prompt, spec.no_bos)
                .map_err(|err| format!("session case {}: {err}", spec.name))?;
            let required_total_tokens = prompt_token_ids
                .len()
                .checked_add(spec.max_new_tokens)
                .ok_or_else(|| format!("session case {} token budget overflow", spec.name))?;
            let max_supported_total_tokens = spec
                .max_total_tokens
                .map(Ok)
                .unwrap_or_else(|| default_max_total_tokens(required_total_tokens))?;
            validate_token_budget(
                prompt_token_ids.len(),
                spec.max_new_tokens,
                max_supported_total_tokens,
            )
            .map_err(|err| format!("session case {}: {err}", spec.name))?;
            let reference = spec
                .hf_reference
                .clone()
                .map(|path| {
                    if !path.is_file() {
                        return Err(format!(
                            "session case {} reference artifact does not exist: {}",
                            spec.name,
                            path.display()
                        ));
                    }
                    parse_hf_reference(path)
                })
                .transpose()?;
            Ok(PreparedSessionCase {
                spec,
                prompt_token_ids,
                max_supported_total_tokens,
                reference,
            })
        })
        .collect()
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn stats_delta(
    before: rvllm_runtime::apple_metal_backend::MetalProbePerfStats,
    after: rvllm_runtime::apple_metal_backend::MetalProbePerfStats,
) -> rvllm_runtime::apple_metal_backend::MetalProbePerfStats {
    rvllm_runtime::apple_metal_backend::MetalProbePerfStats {
        last_step_gpu_execution_ns: after.last_step_gpu_execution_ns,
        prefill_steps: after.prefill_steps.saturating_sub(before.prefill_steps),
        decode_steps: after.decode_steps.saturating_sub(before.decode_steps),
        tokens: after.tokens.saturating_sub(before.tokens),
        library_compiles: after
            .library_compiles
            .saturating_sub(before.library_compiles),
        pipeline_state_compiles: after
            .pipeline_state_compiles
            .saturating_sub(before.pipeline_state_compiles),
        command_buffers: after.command_buffers.saturating_sub(before.command_buffers),
        encoders: after.encoders.saturating_sub(before.encoders),
        embedding_encoders: after
            .embedding_encoders
            .saturating_sub(before.embedding_encoders),
        ple_encoders: after.ple_encoders.saturating_sub(before.ple_encoders),
        layer_encoders: after.layer_encoders.saturating_sub(before.layer_encoders),
        layer_scale_encoder_fusions: after
            .layer_scale_encoder_fusions
            .saturating_sub(before.layer_scale_encoder_fusions),
        final_sample_encoders: after
            .final_sample_encoders
            .saturating_sub(before.final_sample_encoders),
        final_logits_encoders: after
            .final_logits_encoders
            .saturating_sub(before.final_logits_encoders),
        forced_waits: after.forced_waits.saturating_sub(before.forced_waits),
        cpu_wall_ns: after.cpu_wall_ns.saturating_sub(before.cpu_wall_ns),
        cpu_encode_ns: after.cpu_encode_ns.saturating_sub(before.cpu_encode_ns),
        command_buffer_wait_ns: after
            .command_buffer_wait_ns
            .saturating_sub(before.command_buffer_wait_ns),
        last_step_tokens: after.last_step_tokens,
        last_step_command_buffers: after.last_step_command_buffers,
        last_step_encoders: after.last_step_encoders,
        last_step_forced_waits: after.last_step_forced_waits,
        last_step_cpu_wall_ns: after.last_step_cpu_wall_ns,
        last_step_cpu_encode_ns: after.last_step_cpu_encode_ns,
        last_step_command_buffer_wait_ns: after.last_step_command_buffer_wait_ns,
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn research_dispatch_value(
    snapshot: rvllm_apple_metal::research_evidence::ResearchDispatchSnapshot,
) -> serde_json::Value {
    let counts = rvllm_apple_metal::research_evidence::RESEARCH_KERNEL_NAMES
        .iter()
        .copied()
        .zip(snapshot.counts)
        .filter(|(_, count)| *count != 0)
        .collect::<std::collections::BTreeMap<_, _>>();
    serde_json::json!({
        "schema": rvllm_apple_metal::research_evidence::RESEARCH_DISPATCH_SCHEMA,
        "counts": counts,
        "overflowed": snapshot.overflowed,
    })
}

fn case_report_value(report: &SessionCaseReport) -> serde_json::Value {
    let status = match &report.comparison {
        Some(comparison) if !comparison.matched => "fail",
        _ => "pass",
    };
    serde_json::json!({
        "name": report.name,
        "status": status,
        "prompt": report.prompt,
        "prompt_token_ids": report.prompt_token_ids,
        "max_new_tokens": report.max_new_tokens,
        "generated_token_ids": report.generated_token_ids,
        "output_token_ids": report.output_token_ids,
        "generated_text": report.generated_text,
        "output_text": report.output_text,
        "finish_reason": report.finish_reason.as_str(),
        "prefill_ms": report.prefill_ms,
        "decode_ms": report.decode_ms,
        "tok_per_s": report.tok_per_s,
        "library_compiles": report.library_compiles,
        "pipeline_state_compiles": report.pipeline_state_compiles,
        "last_step_gpu_execution_ns": report.last_step_gpu_execution_ns,
        "research_dispatch": report.research_dispatch,
        "command_buffers": report.command_buffers,
        "encoders": report.encoders,
        "embedding_encoders": report.embedding_encoders,
        "ple_encoders": report.ple_encoders,
        "layer_encoders": report.layer_encoders,
        "layer_scale_encoder_fusions": report.layer_scale_encoder_fusions,
        "final_sample_encoders": report.final_sample_encoders,
        "final_logits_encoders": report.final_logits_encoders,
        "encoder_counts_by_kernel_family": {
            "embedding": report.embedding_encoders,
            "ple_input": report.ple_encoders,
            "layer_body": report.layer_encoders,
            "layer_scale_fused": report.layer_scale_encoder_fusions,
            "final_sample": report.final_sample_encoders,
            "final_logits_diagnostic": report.final_logits_encoders,
        },
        "forced_waits": report.forced_waits,
        "cpu_wall_ns": report.cpu_wall_ns,
        "cpu_encode_ns": report.cpu_encode_ns,
        "command_buffer_wait_ns": report.command_buffer_wait_ns,
        "last_step_tokens": report.last_step_tokens,
        "last_step_command_buffers": report.last_step_command_buffers,
        "last_step_encoders": report.last_step_encoders,
        "last_step_forced_waits": report.last_step_forced_waits,
        "last_step_cpu_wall_ns": report.last_step_cpu_wall_ns,
        "last_step_cpu_encode_ns": report.last_step_cpu_encode_ns,
        "last_step_command_buffer_wait_ns": report.last_step_command_buffer_wait_ns,
        "max_supported_total_tokens": report.max_supported_total_tokens,
        "hf_reference": hf_reference_value(report.comparison.as_ref()),
    })
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn build_metal_runtime_plan(
    model_dir: PathBuf,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    rollout_tokens: usize,
) -> Result<rvllm_apple::AppleRuntimePlan, String> {
    let rollout_tokens = u32::try_from(rollout_tokens)
        .map_err(|_| format!("rollout token count must be at most {}", u32::MAX))?;
    Ok(rvllm_apple::AppleRuntimePlan {
        target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
        mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
        rollout_bucket: None,
        rollout_tokens,
        private_ane_opt_in: false,
        strict_ane: false,
        ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
        ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
        ane_hidden_size: arch.hidden_size,
        ane_intermediate_size: arch.intermediate_size,
        ane_num_layers: arch.num_hidden_layers,
        model_layout_hash: [0u8; 32],
        weights_path: Some(model_dir),
    })
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn validate_large_model_opt_in(
    _arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    args: &CliArgs,
) -> Result<bool, String> {
    let env_opt_in = env_large_model_opted_in();
    Ok(args.large_model_opt_in || env_opt_in)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn max_session_supported_total_tokens(cases: &[PreparedSessionCase]) -> usize {
    cases
        .iter()
        .map(|case| case.max_supported_total_tokens)
        .max()
        .unwrap_or(DEFAULT_MAX_METAL_TOTAL_TOKENS)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn max_session_new_tokens(cases: &[PreparedSessionCase]) -> usize {
    cases
        .iter()
        .map(|case| case.spec.max_new_tokens)
        .max()
        .unwrap_or(1)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn session_batch_token_capacity(cases: &[PreparedSessionCase]) -> Result<usize, String> {
    let prefill_tokens = cases.iter().try_fold(0usize, |total, case| {
        total
            .checked_add(case.prompt_token_ids.len())
            .ok_or_else(|| "session batch token capacity overflow".to_owned())
    })?;
    let capacity = prefill_tokens.max(cases.len()).max(1);
    validate_u32_count("session batch token capacity", capacity)?;
    Ok(capacity)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn session_batch_sequence_capacity(cases: &[PreparedSessionCase]) -> usize {
    cases.len().max(1)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn decode_text(
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    label: &str,
) -> Result<String, String> {
    tokenizer
        .decode(token_ids, true)
        .map_err(|err| format!("decode {label} token IDs: {err}"))
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn session_report_value(
    args: &CliArgs,
    backend: SessionBackend,
    status: &str,
    prepare_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    case_reports: &[SessionCaseReport],
    final_stats: rvllm_runtime::apple_metal_backend::MetalProbePerfStats,
    arena_bytes: usize,
    debug_sync: bool,
    large_model_opt_in: bool,
    max_supported_total_tokens: usize,
    metal_compute_dtype: &str,
    metal_weight_dtype: &str,
    metal_moe_router_weight_dtype: &str,
    research_dispatch: serde_json::Value,
) -> serde_json::Value {
    let generated_tokens: usize = case_reports
        .iter()
        .map(|case| case.generated_token_ids.len())
        .sum();
    let tok_per_s = if decode_ms > 0.0 {
        generated_tokens as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    serde_json::json!({
        "schema": SESSION_JSON_SCHEMA,
        "claim": CLAIM,
        "backend": backend.as_str(),
        "metal_compute_dtype": metal_compute_dtype,
        "metal_weight_dtype": metal_weight_dtype,
        "metal_moe_router_weight_dtype": metal_moe_router_weight_dtype,
        "status": status,
        "model_dir": args.model_dir,
        "case_count": case_reports.len(),
        "prepare_ms": prepare_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "generated_tokens": generated_tokens,
        "tok_per_s": tok_per_s,
        "arena_bytes": arena_bytes,
        "library_compiles": final_stats.library_compiles,
        "pipeline_state_compiles": final_stats.pipeline_state_compiles,
        "last_step_gpu_execution_ns": final_stats.last_step_gpu_execution_ns,
        "research_dispatch": research_dispatch,
        "command_buffers": final_stats.command_buffers,
        "encoders": final_stats.encoders,
        "embedding_encoders": final_stats.embedding_encoders,
        "ple_encoders": final_stats.ple_encoders,
        "layer_encoders": final_stats.layer_encoders,
        "layer_scale_encoder_fusions": final_stats.layer_scale_encoder_fusions,
        "final_sample_encoders": final_stats.final_sample_encoders,
        "final_logits_encoders": final_stats.final_logits_encoders,
        "encoder_counts_by_kernel_family": {
            "embedding": final_stats.embedding_encoders,
            "ple_input": final_stats.ple_encoders,
            "layer_body": final_stats.layer_encoders,
            "layer_scale_fused": final_stats.layer_scale_encoder_fusions,
            "final_sample": final_stats.final_sample_encoders,
            "final_logits_diagnostic": final_stats.final_logits_encoders,
        },
        "forced_waits": final_stats.forced_waits,
        "cpu_wall_ns": final_stats.cpu_wall_ns,
        "cpu_encode_ns": final_stats.cpu_encode_ns,
        "command_buffer_wait_ns": final_stats.command_buffer_wait_ns,
        "last_step_tokens": final_stats.last_step_tokens,
        "last_step_command_buffers": final_stats.last_step_command_buffers,
        "last_step_encoders": final_stats.last_step_encoders,
        "last_step_forced_waits": final_stats.last_step_forced_waits,
        "last_step_cpu_wall_ns": final_stats.last_step_cpu_wall_ns,
        "last_step_cpu_encode_ns": final_stats.last_step_cpu_encode_ns,
        "last_step_command_buffer_wait_ns": final_stats.last_step_command_buffer_wait_ns,
        "debug_sync": debug_sync,
        "large_model_opt_in": large_model_opt_in,
        "max_supported_total_tokens": max_supported_total_tokens,
        "cases": case_reports.iter().map(case_report_value).collect::<Vec<_>>(),
    })
}

fn write_report_if_requested(
    path: Option<&PathBuf>,
    value: &serde_json::Value,
) -> Result<(), String> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("create report directory {}: {err}", parent.display()))?;
        }
        std::fs::write(
            path,
            serde_json::to_string_pretty(value).expect("serialize session report") + "\n",
        )
        .map_err(|err| format!("write report {}: {err}", path.display()))?;
    }
    Ok(())
}

fn check_case_timeout(
    args: &CliArgs,
    case_name: &str,
    start: std::time::Instant,
) -> Result<(), String> {
    if let Some(timeout_seconds) = args.case_timeout_seconds {
        if start.elapsed() > std::time::Duration::from_secs(timeout_seconds) {
            return Err(format!(
                "session case {case_name} timed out after {timeout_seconds}s"
            ));
        }
    }
    Ok(())
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_direct_session(
    args: &CliArgs,
    tokenizer: &tokenizers::Tokenizer,
    cases: Vec<PreparedSessionCase>,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    effective_large_opt_in: bool,
) -> Result<serde_json::Value, String> {
    use rvllm_apple::{AppleBackend, HandoffCapsule, HandoffKind};
    use rvllm_core::{ReqId, TokenId};
    use rvllm_runtime::apple_metal_backend::ModelMetalBackend;

    let max_supported_total_tokens = max_session_supported_total_tokens(&cases);
    let max_new_tokens = max_session_new_tokens(&cases);
    let _large_model_env = EnvGuard::set_if(
        LARGE_MODEL_ENV,
        args.large_model_opt_in && !env_large_model_opted_in(),
    );
    let _max_tokens_env = EnvGuard::set_value_if(
        MAX_TOTAL_TOKENS_ENV,
        Some(max_supported_total_tokens.to_string()),
    );
    let _max_batch_tokens_env = EnvGuard::set_value_if(
        MAX_BATCH_TOKENS_ENV,
        Some(
            cases
                .iter()
                .map(|case| case.prompt_token_ids.len())
                .max()
                .unwrap_or(1)
                .to_string(),
        ),
    );
    let _max_batch_sequences_env =
        EnvGuard::set_value_if(MAX_BATCH_SEQUENCES_ENV, Some("1".to_owned()));
    let mut backend = ModelMetalBackend::new(args.model_dir.clone());
    let plan = build_metal_runtime_plan(args.model_dir.clone(), arch, max_new_tokens)?;

    let total_start = std::time::Instant::now();
    let prepare_start = std::time::Instant::now();
    backend
        .prepare(&plan)
        .map_err(|err| format!("prepare Metal backend: {err}"))?;
    let prepare_ms = ms(prepare_start.elapsed());
    let metal_compute_dtype = backend.metal_compute_dtype_report().to_owned();
    let metal_weight_dtype = backend.metal_weight_dtype_report().to_owned();
    let metal_moe_router_weight_dtype = backend.metal_moe_router_weight_dtype_report().to_owned();

    let mut case_reports = Vec::with_capacity(cases.len());
    for (idx, case) in cases.iter().enumerate() {
        let case_start = std::time::Instant::now();
        let before = backend.probe_perf_stats();
        let dispatch_before = backend.probe_research_dispatches().ok_or_else(|| {
            format!(
                "session case {} research counters unavailable",
                case.spec.name
            )
        })?;
        let req_id = ReqId((idx + 1) as u64);
        let prompt_tokens = case
            .prompt_token_ids
            .iter()
            .copied()
            .map(TokenId)
            .collect::<Vec<_>>();
        let prompt_len = prompt_tokens.len();
        let prefill = HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![req_id],
            prompt_tokens.clone(),
            vec![0, prompt_len as u32],
            vec![(prompt_len - 1) as u32],
            vec![prompt_len as u32],
        );
        let prefill_start = std::time::Instant::now();
        let prefill_ticket = backend
            .launch_prefill(&prefill)
            .map_err(|err| format!("session case {} launch prefill: {err}", case.spec.name))?;
        let prefill_out = backend
            .collect(prefill_ticket)
            .map_err(|err| format!("session case {} collect prefill: {err}", case.spec.name))?;
        if !prefill_out.is_empty() {
            return Err(format!(
                "session case {} prefill unexpectedly returned {} sampled tokens",
                case.spec.name,
                prefill_out.len()
            ));
        }
        check_case_timeout(args, &case.spec.name, case_start)?;
        let prefill_ms = ms(prefill_start.elapsed());

        let mut current = *prompt_tokens.last().expect("prompt token");
        let mut generated_token_ids = Vec::with_capacity(case.spec.max_new_tokens);
        let mut finish_reason = FinishReason::Length;
        let decode_start = std::time::Instant::now();
        for step_idx in 0..case.spec.max_new_tokens {
            let decode = HandoffCapsule::new(
                HandoffKind::MetalPrefillToMetalDecode,
                vec![req_id],
                vec![current],
                vec![0, 1],
                vec![(prompt_len - 1 + step_idx) as u32],
                vec![(prompt_len + step_idx) as u32],
            );
            let ticket = backend.launch_rollout(&decode, None).map_err(|err| {
                format!(
                    "session case {} launch decode step {step_idx}: {err}",
                    case.spec.name
                )
            })?;
            let out = backend.collect(ticket).map_err(|err| {
                format!(
                    "session case {} collect decode step {step_idx}: {err}",
                    case.spec.name
                )
            })?;
            if out.len() != 1 {
                return Err(format!(
                    "session case {} decode step {step_idx} returned {} sampled tokens, expected 1",
                    case.spec.name,
                    out.len()
                ));
            }
            let sampled = out[0].token_id.raw();
            generated_token_ids.push(sampled);
            current = TokenId(sampled);
            if args.eos_token_ids.contains(&sampled) {
                finish_reason = FinishReason::Eos;
                break;
            }
            check_case_timeout(args, &case.spec.name, case_start)?;
        }
        check_case_timeout(args, &case.spec.name, case_start)?;
        let decode_ms = ms(decode_start.elapsed());
        let after = backend.probe_perf_stats();
        let delta = stats_delta(before, after);
        let dispatch_after = backend.probe_research_dispatches().ok_or_else(|| {
            format!(
                "session case {} research counters unavailable",
                case.spec.name
            )
        })?;
        let research_dispatch = dispatch_after
            .checked_since(dispatch_before)
            .map(research_dispatch_value)
            .map_err(|error| format!("session case {}: {error}", case.spec.name))?;
        let mut output_token_ids = case.prompt_token_ids.clone();
        output_token_ids.extend(generated_token_ids.iter().copied());
        let generated_text = decode_text(tokenizer, &generated_token_ids, "generated")?;
        let output_text = decode_text(tokenizer, &output_token_ids, "output")?;
        let comparison = case.reference.as_ref().map(|reference| {
            compare_hf_reference_parts(
                &case.prompt_token_ids,
                case.spec.max_new_tokens,
                &generated_token_ids,
                reference,
            )
        });
        let tok_per_s = if decode_ms > 0.0 {
            generated_token_ids.len() as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        case_reports.push(SessionCaseReport {
            name: case.spec.name.clone(),
            prompt: case.spec.prompt.clone(),
            prompt_token_ids: case.prompt_token_ids.clone(),
            max_new_tokens: case.spec.max_new_tokens,
            generated_token_ids,
            output_token_ids,
            generated_text,
            output_text,
            finish_reason,
            prefill_ms,
            decode_ms,
            tok_per_s,
            library_compiles: delta.library_compiles,
            pipeline_state_compiles: delta.pipeline_state_compiles,
            last_step_gpu_execution_ns: delta.last_step_gpu_execution_ns,
            research_dispatch,
            command_buffers: delta.command_buffers,
            encoders: delta.encoders,
            embedding_encoders: delta.embedding_encoders,
            ple_encoders: delta.ple_encoders,
            layer_encoders: delta.layer_encoders,
            layer_scale_encoder_fusions: delta.layer_scale_encoder_fusions,
            final_sample_encoders: delta.final_sample_encoders,
            final_logits_encoders: delta.final_logits_encoders,
            forced_waits: delta.forced_waits,
            cpu_wall_ns: delta.cpu_wall_ns,
            cpu_encode_ns: delta.cpu_encode_ns,
            command_buffer_wait_ns: delta.command_buffer_wait_ns,
            last_step_tokens: delta.last_step_tokens,
            last_step_command_buffers: delta.last_step_command_buffers,
            last_step_encoders: delta.last_step_encoders,
            last_step_forced_waits: delta.last_step_forced_waits,
            last_step_cpu_wall_ns: delta.last_step_cpu_wall_ns,
            last_step_cpu_encode_ns: delta.last_step_cpu_encode_ns,
            last_step_command_buffer_wait_ns: delta.last_step_command_buffer_wait_ns,
            max_supported_total_tokens: case.max_supported_total_tokens,
            comparison,
        });
    }
    let status = if case_reports
        .iter()
        .any(|case| matches!(&case.comparison, Some(comparison) if !comparison.matched))
    {
        "fail"
    } else {
        "pass"
    };
    let stats = backend.probe_perf_stats();
    let research_dispatch = backend
        .probe_research_dispatches()
        .map(research_dispatch_value)
        .unwrap_or(serde_json::Value::Null);
    let arena_bytes = backend
        .probe_arena_stats()
        .map(|arena| arena.capacity_bytes)
        .unwrap_or(0);
    Ok(session_report_value(
        args,
        SessionBackend::Direct,
        status,
        prepare_ms,
        case_reports.iter().map(|case| case.prefill_ms).sum(),
        case_reports.iter().map(|case| case.decode_ms).sum(),
        ms(total_start.elapsed()),
        &case_reports,
        stats,
        arena_bytes,
        backend.metal_debug_sync_enabled(),
        effective_large_opt_in,
        max_supported_total_tokens,
        &metal_compute_dtype,
        &metal_weight_dtype,
        &metal_moe_router_weight_dtype,
        research_dispatch,
    ))
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Clone)]
struct SharedModelMetalBackend {
    inner: std::rc::Rc<std::cell::RefCell<rvllm_runtime::apple_metal_backend::ModelMetalBackend>>,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl SharedModelMetalBackend {
    fn new(model_dir: PathBuf) -> Self {
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(
                rvllm_runtime::apple_metal_backend::ModelMetalBackend::new(model_dir),
            )),
        }
    }

    fn probe_perf_stats(&self) -> rvllm_runtime::apple_metal_backend::MetalProbePerfStats {
        self.inner.borrow().probe_perf_stats()
    }

    fn probe_arena_bytes(&self) -> usize {
        self.inner
            .borrow()
            .probe_arena_stats()
            .map(|arena| arena.capacity_bytes)
            .unwrap_or(0)
    }

    fn metal_debug_sync_enabled(&self) -> bool {
        self.inner.borrow().metal_debug_sync_enabled()
    }

    fn metal_compute_dtype_report(&self) -> &'static str {
        self.inner.borrow().metal_compute_dtype_report()
    }

    fn metal_weight_dtype_report(&self) -> &'static str {
        self.inner.borrow().metal_weight_dtype_report()
    }

    fn metal_moe_router_weight_dtype_report(&self) -> &'static str {
        self.inner.borrow().metal_moe_router_weight_dtype_report()
    }

    fn probe_research_dispatches(
        &self,
    ) -> Option<rvllm_apple_metal::research_evidence::ResearchDispatchSnapshot> {
        self.inner.borrow().probe_research_dispatches()
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl rvllm_apple::AppleBackend for SharedModelMetalBackend {
    fn prepare(&mut self, plan: &rvllm_apple::AppleRuntimePlan) -> rvllm_core::Result<()> {
        rvllm_apple::AppleBackend::prepare(&mut *self.inner.borrow_mut(), plan)
    }

    fn launch_prefill(
        &mut self,
        handoff: &rvllm_apple::HandoffCapsule,
    ) -> rvllm_core::Result<rvllm_apple::AppleLaunchTicket> {
        rvllm_apple::AppleBackend::launch_prefill(&mut *self.inner.borrow_mut(), handoff)
    }

    fn launch_rollout(
        &mut self,
        handoff: &rvllm_apple::HandoffCapsule,
        bucket: Option<rvllm_apple::RolloutBucket>,
    ) -> rvllm_core::Result<rvllm_apple::AppleLaunchTicket> {
        rvllm_apple::AppleBackend::launch_rollout(&mut *self.inner.borrow_mut(), handoff, bucket)
    }

    fn collect(
        &mut self,
        ticket: rvllm_apple::AppleLaunchTicket,
    ) -> rvllm_core::Result<Vec<rvllm_apple::StepToken>> {
        rvllm_apple::AppleBackend::collect(&mut *self.inner.borrow_mut(), ticket)
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_engine_session(
    args: &CliArgs,
    tokenizer: &tokenizers::Tokenizer,
    cases: Vec<PreparedSessionCase>,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    effective_large_opt_in: bool,
) -> Result<serde_json::Value, String> {
    use rvllm_core::{ReqId, TokenId};
    use rvllm_runtime::{BatchPlan, Engine, Request};

    let max_supported_total_tokens = max_session_supported_total_tokens(&cases);
    let max_new_tokens = max_session_new_tokens(&cases);
    let _large_model_env = EnvGuard::set_if(
        LARGE_MODEL_ENV,
        args.large_model_opt_in && !env_large_model_opted_in(),
    );
    let _max_tokens_env = EnvGuard::set_value_if(
        MAX_TOTAL_TOKENS_ENV,
        Some(max_supported_total_tokens.to_string()),
    );
    let _max_batch_tokens_env = EnvGuard::set_value_if(
        MAX_BATCH_TOKENS_ENV,
        Some(session_batch_token_capacity(&cases)?.to_string()),
    );
    let _max_batch_sequences_env = EnvGuard::set_value_if(
        MAX_BATCH_SEQUENCES_ENV,
        Some(session_batch_sequence_capacity(&cases).to_string()),
    );
    let shared_backend = SharedModelMetalBackend::new(args.model_dir.clone());
    let stats_backend = shared_backend.clone();
    let plan = build_metal_runtime_plan(args.model_dir.clone(), arch, max_new_tokens)?;

    let total_start = std::time::Instant::now();
    let prepare_start = std::time::Instant::now();
    let mut engine = Engine::new()
        .with_apple_backend(Box::new(shared_backend))
        .with_apple_runtime_plan(plan)
        .map_err(|err| format!("prepare Engine Metal backend: {err}"))?;
    let prepare_ms = ms(prepare_start.elapsed());

    let mut req_to_case = HashMap::<ReqId, usize>::new();
    for (idx, case) in cases.iter().enumerate() {
        let req_id = ReqId((idx + 1) as u64);
        req_to_case.insert(req_id, idx);
        let request_max_new_tokens = u32::try_from(case.spec.max_new_tokens).map_err(|_| {
            format!(
                "session case {} max_new_tokens must be at most {}",
                case.spec.name,
                u32::MAX
            )
        })?;
        engine.scheduler.enqueue(Request::new(
            req_id,
            case.prompt_token_ids.iter().copied().map(TokenId).collect(),
            request_max_new_tokens,
        ));
    }
    let case_starts = vec![std::time::Instant::now(); cases.len()];

    let mut generated_by_case = vec![Vec::<u32>::new(); cases.len()];
    let mut prefill_ms_total = 0.0;
    let mut decode_ms_total = 0.0;
    while engine.has_pending_work() {
        let step_start = std::time::Instant::now();
        let step = engine
            .step_launch()
            .map_err(|err| format!("launch Engine session step: {err}"))?;
        let is_decode = matches!(step.plan(), Some(BatchPlan::Decode { .. }));
        let is_prefill = matches!(step.plan(), Some(BatchPlan::Prefill { .. }));
        let outputs = step
            .collect()
            .map_err(|err| format!("collect Engine session step: {err}"))?;
        let step_ms = ms(step_start.elapsed());
        if is_prefill {
            prefill_ms_total += step_ms;
        }
        if is_decode {
            decode_ms_total += step_ms;
        }
        for output in outputs {
            let Some(&case_idx) = req_to_case.get(&output.req_id) else {
                return Err(format!(
                    "Engine session returned unknown request id {}",
                    output.req_id
                ));
            };
            let sampled = output.new_token.raw();
            generated_by_case[case_idx].push(sampled);
            if args.eos_token_ids.contains(&sampled) {
                engine.scheduler.finish_request(output.req_id);
            }
        }
        for (idx, case) in cases.iter().enumerate() {
            check_case_timeout(args, &case.spec.name, case_starts[idx])?;
        }
    }

    let stats = stats_backend.probe_perf_stats();
    let mut case_reports = Vec::with_capacity(cases.len());
    let per_case_prefill_ms = if cases.is_empty() {
        0.0
    } else {
        prefill_ms_total / cases.len() as f64
    };
    let per_case_decode_ms = if cases.is_empty() {
        0.0
    } else {
        decode_ms_total / cases.len() as f64
    };
    for (idx, case) in cases.iter().enumerate() {
        let generated_token_ids = generated_by_case[idx].clone();
        let mut output_token_ids = case.prompt_token_ids.clone();
        output_token_ids.extend(generated_token_ids.iter().copied());
        let finish_reason = if generated_token_ids
            .last()
            .is_some_and(|token| args.eos_token_ids.contains(token))
        {
            FinishReason::Eos
        } else {
            FinishReason::Length
        };
        let generated_text = decode_text(tokenizer, &generated_token_ids, "generated")?;
        let output_text = decode_text(tokenizer, &output_token_ids, "output")?;
        let comparison = case.reference.as_ref().map(|reference| {
            compare_hf_reference_parts(
                &case.prompt_token_ids,
                case.spec.max_new_tokens,
                &generated_token_ids,
                reference,
            )
        });
        let tok_per_s = if per_case_decode_ms > 0.0 {
            generated_token_ids.len() as f64 / (per_case_decode_ms / 1000.0)
        } else {
            0.0
        };
        case_reports.push(SessionCaseReport {
            name: case.spec.name.clone(),
            prompt: case.spec.prompt.clone(),
            prompt_token_ids: case.prompt_token_ids.clone(),
            max_new_tokens: case.spec.max_new_tokens,
            generated_token_ids,
            output_token_ids,
            generated_text,
            output_text,
            finish_reason,
            prefill_ms: per_case_prefill_ms,
            decode_ms: per_case_decode_ms,
            tok_per_s,
            library_compiles: 0,
            pipeline_state_compiles: 0,
            last_step_gpu_execution_ns: None,
            research_dispatch: serde_json::Value::Null,
            command_buffers: 0,
            encoders: 0,
            embedding_encoders: 0,
            ple_encoders: 0,
            layer_encoders: 0,
            layer_scale_encoder_fusions: 0,
            final_sample_encoders: 0,
            final_logits_encoders: 0,
            forced_waits: 0,
            cpu_wall_ns: 0,
            cpu_encode_ns: 0,
            command_buffer_wait_ns: 0,
            last_step_tokens: 0,
            last_step_command_buffers: 0,
            last_step_encoders: 0,
            last_step_forced_waits: 0,
            last_step_cpu_wall_ns: 0,
            last_step_cpu_encode_ns: 0,
            last_step_command_buffer_wait_ns: 0,
            max_supported_total_tokens: case.max_supported_total_tokens,
            comparison,
        });
    }
    let status = if case_reports
        .iter()
        .any(|case| matches!(&case.comparison, Some(comparison) if !comparison.matched))
    {
        "fail"
    } else {
        "pass"
    };
    let research_dispatch = stats_backend
        .probe_research_dispatches()
        .map(research_dispatch_value)
        .unwrap_or(serde_json::Value::Null);
    Ok(session_report_value(
        args,
        SessionBackend::Engine,
        status,
        prepare_ms,
        prefill_ms_total,
        decode_ms_total,
        ms(total_start.elapsed()),
        &case_reports,
        stats,
        stats_backend.probe_arena_bytes(),
        stats_backend.metal_debug_sync_enabled(),
        effective_large_opt_in,
        max_supported_total_tokens,
        stats_backend.metal_compute_dtype_report(),
        stats_backend.metal_weight_dtype_report(),
        stats_backend.metal_moe_router_weight_dtype_report(),
        research_dispatch,
    ))
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_session_once(args: &CliArgs) -> Result<serde_json::Value, String> {
    if !args.model_dir.is_dir() {
        return Err(format!(
            "model path does not exist or is not a directory: {}",
            args.model_dir.display()
        ));
    }
    let tokenizer = load_tokenizer(&args.model_dir)?;
    let specs = load_session_case_specs(args)?;
    let cases = prepare_session_cases(&tokenizer, specs)?;
    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&args.model_dir)
        .map_err(|err| format!("parse Gemma4 architecture: {err}"))?;
    let effective_large_opt_in = validate_large_model_opt_in(&arch, args)?;
    match args.session_backend {
        SessionBackend::Direct => {
            run_direct_session(args, &tokenizer, cases, &arch, effective_large_opt_in)
        }
        SessionBackend::Engine => {
            run_engine_session(args, &tokenizer, cases, &arch, effective_large_opt_in)
        }
    }
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
fn run_session_once(_args: &CliArgs) -> Result<serde_json::Value, String> {
    Err("rvllm_metal_infer session mode requires --features apple on macOS".to_owned())
}

fn json_f64(value: &serde_json::Value, field: &str) -> Result<f64, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("session report missing numeric field {field:?}"))
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn min_f64(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_f64(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn profile_summary(
    args: &CliArgs,
    samples: Vec<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let prepare = samples
        .iter()
        .map(|sample| json_f64(sample, "prepare_ms"))
        .collect::<Result<Vec<_>, _>>()?;
    let prefill = samples
        .iter()
        .map(|sample| json_f64(sample, "prefill_ms"))
        .collect::<Result<Vec<_>, _>>()?;
    let decode = samples
        .iter()
        .map(|sample| json_f64(sample, "decode_ms"))
        .collect::<Result<Vec<_>, _>>()?;
    let tok_per_s = samples
        .iter()
        .map(|sample| json_f64(sample, "tok_per_s"))
        .collect::<Result<Vec<_>, _>>()?;
    let command_buffers_per_token = samples
        .iter()
        .map(|sample| {
            let tokens = json_f64(sample, "generated_tokens")?;
            let command_buffers = json_f64(sample, "command_buffers")?;
            Ok(if tokens > 0.0 {
                command_buffers / tokens
            } else {
                0.0
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let encoders_per_token = samples
        .iter()
        .map(|sample| {
            let tokens = json_f64(sample, "generated_tokens")?;
            let encoders = json_f64(sample, "encoders")?;
            Ok(if tokens > 0.0 { encoders / tokens } else { 0.0 })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let layer_scale_fusions_per_token = samples
        .iter()
        .map(|sample| {
            let tokens = json_f64(sample, "generated_tokens")?;
            let fusions = json_f64(sample, "layer_scale_encoder_fusions")?;
            Ok(if tokens > 0.0 { fusions / tokens } else { 0.0 })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let forced_waits_per_token = samples
        .iter()
        .map(|sample| {
            let tokens = json_f64(sample, "generated_tokens")?;
            let forced_waits = json_f64(sample, "forced_waits")?;
            Ok(if tokens > 0.0 {
                forced_waits / tokens
            } else {
                0.0
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let latency_ms_per_token = samples
        .iter()
        .map(|sample| {
            let tokens = json_f64(sample, "generated_tokens")?;
            let decode_ms = json_f64(sample, "decode_ms")?;
            Ok(if tokens > 0.0 {
                decode_ms / tokens
            } else {
                0.0
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let cpu_encode_ms = samples
        .iter()
        .map(|sample| json_f64(sample, "cpu_encode_ns").map(|ns| ns / 1_000_000.0))
        .collect::<Result<Vec<_>, _>>()?;
    let command_buffer_wait_ms = samples
        .iter()
        .map(|sample| json_f64(sample, "command_buffer_wait_ns").map(|ns| ns / 1_000_000.0))
        .collect::<Result<Vec<_>, _>>()?;
    let cpu_encode_ms_per_token = samples
        .iter()
        .map(|sample| {
            let tokens = json_f64(sample, "generated_tokens")?;
            let ns = json_f64(sample, "cpu_encode_ns")?;
            Ok(if tokens > 0.0 {
                (ns / 1_000_000.0) / tokens
            } else {
                0.0
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let command_buffer_wait_ms_per_token = samples
        .iter()
        .map(|sample| {
            let tokens = json_f64(sample, "generated_tokens")?;
            let ns = json_f64(sample, "command_buffer_wait_ns")?;
            Ok(if tokens > 0.0 {
                (ns / 1_000_000.0) / tokens
            } else {
                0.0
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let first_sample = samples.first();
    let metal_compute_dtype = first_sample
        .and_then(|sample| sample.get("metal_compute_dtype"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let metal_weight_dtype = first_sample
        .and_then(|sample| sample.get("metal_weight_dtype"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let metal_moe_router_weight_dtype = first_sample
        .and_then(|sample| sample.get("metal_moe_router_weight_dtype"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    Ok(serde_json::json!({
        "schema": SESSION_PROFILE_SCHEMA,
        "claim": "Apple Metal text session profile only; not production performance, ANE, external profiler, or production readiness evidence",
        "backend": args.session_backend.as_str(),
        "metal_compute_dtype": metal_compute_dtype,
        "metal_weight_dtype": metal_weight_dtype,
        "metal_moe_router_weight_dtype": metal_moe_router_weight_dtype,
        "model_dir": args.model_dir,
        "sample_count": samples.len(),
        "medians": {
            "prepare_ms": median(prepare.clone()),
            "prefill_ms": median(prefill.clone()),
            "decode_ms": median(decode.clone()),
            "tok_per_s": median(tok_per_s.clone()),
            "command_buffers_per_token": median(command_buffers_per_token.clone()),
            "encoders_per_token": median(encoders_per_token.clone()),
            "layer_scale_fusions_per_token": median(layer_scale_fusions_per_token.clone()),
            "forced_waits_per_token": median(forced_waits_per_token.clone()),
            "latency_ms_per_token": median(latency_ms_per_token.clone()),
            "cpu_encode_ms": median(cpu_encode_ms.clone()),
            "command_buffer_wait_ms": median(command_buffer_wait_ms.clone()),
            "cpu_encode_ms_per_token": median(cpu_encode_ms_per_token.clone()),
            "command_buffer_wait_ms_per_token": median(command_buffer_wait_ms_per_token.clone()),
        },
        "minimums": {
            "prepare_ms": min_f64(&prepare),
            "prefill_ms": min_f64(&prefill),
            "decode_ms": min_f64(&decode),
            "tok_per_s": min_f64(&tok_per_s),
            "command_buffers_per_token": min_f64(&command_buffers_per_token),
            "encoders_per_token": min_f64(&encoders_per_token),
            "layer_scale_fusions_per_token": min_f64(&layer_scale_fusions_per_token),
            "forced_waits_per_token": min_f64(&forced_waits_per_token),
            "latency_ms_per_token": min_f64(&latency_ms_per_token),
            "cpu_encode_ms": min_f64(&cpu_encode_ms),
            "command_buffer_wait_ms": min_f64(&command_buffer_wait_ms),
            "cpu_encode_ms_per_token": min_f64(&cpu_encode_ms_per_token),
            "command_buffer_wait_ms_per_token": min_f64(&command_buffer_wait_ms_per_token),
        },
        "maximums": {
            "prepare_ms": max_f64(&prepare),
            "prefill_ms": max_f64(&prefill),
            "decode_ms": max_f64(&decode),
            "tok_per_s": max_f64(&tok_per_s),
            "command_buffers_per_token": max_f64(&command_buffers_per_token),
            "encoders_per_token": max_f64(&encoders_per_token),
            "layer_scale_fusions_per_token": max_f64(&layer_scale_fusions_per_token),
            "forced_waits_per_token": max_f64(&forced_waits_per_token),
            "latency_ms_per_token": max_f64(&latency_ms_per_token),
            "cpu_encode_ms": max_f64(&cpu_encode_ms),
            "command_buffer_wait_ms": max_f64(&command_buffer_wait_ms),
            "cpu_encode_ms_per_token": max_f64(&cpu_encode_ms_per_token),
            "command_buffer_wait_ms_per_token": max_f64(&command_buffer_wait_ms_per_token),
        },
        "samples": samples,
    }))
}

fn run_session_main(args: &CliArgs) -> Result<(), String> {
    if args.profile_samples > 0 {
        let mut samples = Vec::with_capacity(args.profile_samples);
        for _ in 0..args.profile_samples {
            let sample = run_session_once(args)?;
            if sample.get("status").and_then(serde_json::Value::as_str) != Some("pass") {
                return Err(format!(
                    "session profile sample failed acceptance: {}",
                    serde_json::to_string_pretty(&sample).expect("serialize failed sample")
                ));
            }
            samples.push(sample);
        }
        let summary = profile_summary(args, samples)?;
        write_report_if_requested(
            args.profile_report.as_ref().or(args.report.as_ref()),
            &summary,
        )?;
        println!(
            "{}",
            serde_json::to_string_pretty(&summary).expect("serialize profile summary")
        );
        return Ok(());
    }

    let report = run_session_once(args)?;
    write_report_if_requested(args.report.as_ref(), &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize session report")
    );
    Ok(())
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn select_top_logits(logits: &[f32], limit: usize) -> Vec<TopLogit> {
    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .map(|(idx, logit)| TopLogit {
            token_id: idx as u32,
            logit,
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.logit
            .partial_cmp(&a.logit)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.token_id.cmp(&b.token_id))
    });
    ranked.truncate(limit);
    ranked
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_infer(args: &CliArgs) -> Result<InferReport, String> {
    use rvllm_apple::{AppleBackend, HandoffCapsule, HandoffKind};
    use rvllm_core::{ReqId, TokenId};
    use rvllm_runtime::apple_metal_backend::ModelMetalBackend;

    if !args.model_dir.is_dir() {
        return Err(format!(
            "model path does not exist or is not a directory: {}",
            args.model_dir.display()
        ));
    }
    let tokenizer = load_tokenizer(&args.model_dir)?;
    let prompt_token_ids = prompt_token_ids(args, &tokenizer)?;
    let required_total_tokens = prompt_token_ids
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| "token budget overflow".to_owned())?;
    let max_supported_total_tokens = effective_max_total_tokens(args, required_total_tokens)?;
    validate_token_budget(
        prompt_token_ids.len(),
        args.max_new_tokens,
        max_supported_total_tokens,
    )?;

    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&args.model_dir)
        .map_err(|err| format!("parse Gemma4 architecture: {err}"))?;
    let env_opt_in = env_large_model_opted_in();
    let effective_large_opt_in = args.large_model_opt_in || env_opt_in;
    let _large_model_env =
        EnvGuard::set_if(LARGE_MODEL_ENV, args.large_model_opt_in && !env_opt_in);
    let _max_tokens_env = EnvGuard::set_value_if(
        MAX_TOTAL_TOKENS_ENV,
        Some(max_supported_total_tokens.to_string()),
    );
    let _max_batch_tokens_env = EnvGuard::set_value_if(
        MAX_BATCH_TOKENS_ENV,
        Some(prompt_token_ids.len().max(1).to_string()),
    );
    let _max_batch_sequences_env =
        EnvGuard::set_value_if(MAX_BATCH_SEQUENCES_ENV, Some("1".to_owned()));

    let mut backend = ModelMetalBackend::new(args.model_dir.clone());
    let rollout_tokens = u32::try_from(args.max_new_tokens)
        .map_err(|_| format!("--max-new-tokens must be at most {}", u32::MAX))?;
    let plan = rvllm_apple::AppleRuntimePlan {
        target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
        mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
        rollout_bucket: None,
        rollout_tokens,
        private_ane_opt_in: false,
        strict_ane: false,
        ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
        ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
        ane_hidden_size: arch.hidden_size,
        ane_intermediate_size: arch.intermediate_size,
        ane_num_layers: arch.num_hidden_layers,
        model_layout_hash: [0u8; 32],
        weights_path: Some(args.model_dir.clone()),
    };

    let prepare_start = std::time::Instant::now();
    backend
        .prepare(&plan)
        .map_err(|err| format!("prepare Metal backend: {err}"))?;
    let prepare_ms = ms(prepare_start.elapsed());
    let metal_compute_dtype = backend.metal_compute_dtype_report().to_owned();
    let metal_weight_dtype = backend.metal_weight_dtype_report().to_owned();
    let metal_moe_router_weight_dtype = backend.metal_moe_router_weight_dtype_report().to_owned();

    let prompt_tokens = prompt_token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
    let prompt_len = prompt_tokens.len();
    let prefill = HandoffCapsule::new(
        HandoffKind::MetalPrefillToMetalDecode,
        vec![ReqId(1)],
        prompt_tokens.clone(),
        vec![0, prompt_len as u32],
        vec![(prompt_len - 1) as u32],
        vec![prompt_len as u32],
    );
    let prefill_start = std::time::Instant::now();
    let prefill_ticket = backend
        .launch_prefill(&prefill)
        .map_err(|err| format!("launch prefill: {err}"))?;
    let prefill_out = backend
        .collect(prefill_ticket)
        .map_err(|err| format!("collect prefill: {err}"))?;
    if !prefill_out.is_empty() {
        return Err(format!(
            "prefill unexpectedly returned {} sampled tokens",
            prefill_out.len()
        ));
    }
    let prefill_ms = ms(prefill_start.elapsed());

    let mut current = *prompt_tokens.last().expect("prompt token");
    let mut generated_token_ids = Vec::with_capacity(args.max_new_tokens);
    let mut finish_reason = FinishReason::Length;
    let decode_start = std::time::Instant::now();
    for step_idx in 0..args.max_new_tokens {
        let decode = HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![ReqId(1)],
            vec![current],
            vec![0, 1],
            vec![(prompt_len - 1 + step_idx) as u32],
            vec![(prompt_len + step_idx) as u32],
        );
        let ticket = backend
            .launch_rollout(&decode, None)
            .map_err(|err| format!("launch decode step {step_idx}: {err}"))?;
        let out = backend
            .collect(ticket)
            .map_err(|err| format!("collect decode step {step_idx}: {err}"))?;
        if out.len() != 1 {
            return Err(format!(
                "decode step {step_idx} returned {} sampled tokens, expected 1",
                out.len()
            ));
        }
        let sampled = out[0].token_id.raw();
        generated_token_ids.push(sampled);
        current = TokenId(sampled);
        if args.eos_token_ids.contains(&sampled) {
            finish_reason = FinishReason::Eos;
            break;
        }
    }
    let decode_ms = ms(decode_start.elapsed());
    let generated_count = generated_token_ids.len();
    let tok_per_s = if decode_ms > 0.0 {
        generated_count as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    let mut output_token_ids = prompt_token_ids.clone();
    output_token_ids.extend(generated_token_ids.iter().copied());
    let generated_text = tokenizer
        .decode(&generated_token_ids, true)
        .map_err(|err| format!("decode generated token IDs: {err}"))?;
    let output_text = tokenizer
        .decode(&output_token_ids, true)
        .map_err(|err| format!("decode output token IDs: {err}"))?;
    let diagnostic_top_logits = if args.top_logits > 0 {
        let logits = backend
            .probe_read_decode_logits_f32(1)
            .map_err(|err| format!("read diagnostic top logits: {err}"))?;
        select_top_logits(&logits, args.top_logits)
    } else {
        Vec::new()
    };
    let stats = backend.probe_perf_stats();
    let research_dispatch = backend
        .probe_research_dispatches()
        .map(research_dispatch_value)
        .unwrap_or(serde_json::Value::Null);
    let arena_bytes = backend
        .probe_arena_stats()
        .map(|arena| arena.capacity_bytes)
        .unwrap_or(0);

    Ok(InferReport {
        model_dir: args.model_dir.clone(),
        prompt: args
            .prompt
            .clone()
            .ok_or_else(|| "--prompt is required for single-prompt inference".to_owned())?,
        prompt_token_ids,
        max_new_tokens: args.max_new_tokens,
        generated_token_ids,
        output_token_ids,
        generated_text,
        output_text,
        finish_reason,
        prepare_ms,
        prefill_ms,
        decode_ms,
        tok_per_s,
        arena_bytes,
        library_compiles: stats.library_compiles,
        pipeline_state_compiles: stats.pipeline_state_compiles,
        last_step_gpu_execution_ns: stats.last_step_gpu_execution_ns,
        research_dispatch,
        command_buffers: stats.command_buffers,
        encoders: stats.encoders,
        embedding_encoders: stats.embedding_encoders,
        ple_encoders: stats.ple_encoders,
        layer_encoders: stats.layer_encoders,
        layer_scale_encoder_fusions: stats.layer_scale_encoder_fusions,
        final_sample_encoders: stats.final_sample_encoders,
        final_logits_encoders: stats.final_logits_encoders,
        forced_waits: stats.forced_waits,
        cpu_wall_ns: stats.cpu_wall_ns,
        cpu_encode_ns: stats.cpu_encode_ns,
        command_buffer_wait_ns: stats.command_buffer_wait_ns,
        last_step_tokens: stats.last_step_tokens,
        last_step_command_buffers: stats.last_step_command_buffers,
        last_step_encoders: stats.last_step_encoders,
        last_step_forced_waits: stats.last_step_forced_waits,
        last_step_cpu_wall_ns: stats.last_step_cpu_wall_ns,
        last_step_cpu_encode_ns: stats.last_step_cpu_encode_ns,
        last_step_command_buffer_wait_ns: stats.last_step_command_buffer_wait_ns,
        debug_sync: backend.metal_debug_sync_enabled(),
        large_model_opt_in: effective_large_opt_in,
        max_supported_total_tokens,
        metal_compute_dtype,
        metal_weight_dtype,
        metal_moe_router_weight_dtype,
        diagnostic_top_logits,
    })
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
fn run_infer(_args: &CliArgs) -> Result<InferReport, String> {
    Err("rvllm_metal_infer requires --features apple on macOS".to_owned())
}

fn run_main() -> Result<(), String> {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage());
        return Ok(());
    }
    let mut args = parse_args_from(raw_args)?;
    if args.eos_token_ids.is_empty() {
        args.eos_token_ids = rvllm_loader::generation::load_eos_token_ids(&args.model_dir)?;
    }
    if args.prompts_jsonl.is_some() || args.reference_manifest.is_some() {
        return run_session_main(&args);
    }
    let reference = args
        .hf_reference
        .clone()
        .map(parse_hf_reference)
        .transpose()?;
    let report = run_infer(&args)?;
    let comparison = reference
        .as_ref()
        .map(|reference| compare_hf_reference(&report, reference));
    if args.json_output {
        print_json_report(&report, comparison.as_ref());
    } else {
        print_text_report(&report, comparison.as_ref());
    }
    if comparison.is_some_and(|comparison| !comparison.matched) {
        return Err("generated tokens do not match the Hugging Face reference".into());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            eprintln!("{}", usage());
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_tmp_path(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rvllm_metal_infer_{name}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn session_args_with_jsonl(path: PathBuf) -> CliArgs {
        CliArgs {
            model_dir: PathBuf::from("/tmp/gemma4-e2b"),
            prompt: None,
            prompts_jsonl: Some(path),
            reference_manifest: None,
            session_backend: SessionBackend::Direct,
            max_new_tokens: 3,
            max_total_tokens: Some(32),
            eos_token_ids: vec![1, 2, 107],
            no_bos: true,
            large_model_opt_in: true,
            hf_reference: None,
            report: None,
            case_timeout_seconds: None,
            profile_samples: 0,
            profile_report: None,
            top_logits: 0,
            json_output: true,
        }
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[derive(Clone)]
    struct SharedModelMetalBackend {
        inner:
            std::rc::Rc<std::cell::RefCell<rvllm_runtime::apple_metal_backend::ModelMetalBackend>>,
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    impl SharedModelMetalBackend {
        fn new(model_dir: PathBuf) -> Self {
            Self {
                inner: std::rc::Rc::new(std::cell::RefCell::new(
                    rvllm_runtime::apple_metal_backend::ModelMetalBackend::new(model_dir),
                )),
            }
        }
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    impl rvllm_apple::AppleBackend for SharedModelMetalBackend {
        fn prepare(&mut self, plan: &rvllm_apple::AppleRuntimePlan) -> rvllm_core::Result<()> {
            self.inner.borrow_mut().prepare(plan)
        }

        fn launch_prefill(
            &mut self,
            handoff: &rvllm_apple::HandoffCapsule,
        ) -> rvllm_core::Result<rvllm_apple::AppleLaunchTicket> {
            self.inner.borrow_mut().launch_prefill(handoff)
        }

        fn launch_rollout(
            &mut self,
            handoff: &rvllm_apple::HandoffCapsule,
            bucket: Option<rvllm_apple::RolloutBucket>,
        ) -> rvllm_core::Result<rvllm_apple::AppleLaunchTicket> {
            self.inner.borrow_mut().launch_rollout(handoff, bucket)
        }

        fn collect(
            &mut self,
            ticket: rvllm_apple::AppleLaunchTicket,
        ) -> rvllm_core::Result<Vec<rvllm_apple::StepToken>> {
            self.inner.borrow_mut().collect(ticket)
        }
    }

    #[test]
    fn rvllm_metal_infer_cli_args() {
        let args = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "4",
            "--max-total-tokens",
            "32",
            "--eos-token-ids",
            "1,2,107",
            "--no-bos",
            "--large-model-opt-in",
            "--hf-reference",
            "/tmp/ref.json",
            "--json",
        ])
        .expect("parse CLI args");
        assert_eq!(args.model_dir, PathBuf::from("/tmp/gemma4-e2b"));
        assert_eq!(args.prompt.as_deref(), Some("Hello"));
        assert_eq!(args.prompts_jsonl, None);
        assert_eq!(args.reference_manifest, None);
        assert_eq!(args.session_backend, SessionBackend::Direct);
        assert_eq!(args.max_new_tokens, 4);
        assert_eq!(args.max_total_tokens, Some(32));
        assert_eq!(args.eos_token_ids, vec![1, 2, 107]);
        assert!(args.no_bos);
        assert!(args.large_model_opt_in);
        assert_eq!(args.hf_reference, Some(PathBuf::from("/tmp/ref.json")));
        assert_eq!(args.report, None);
        assert_eq!(args.case_timeout_seconds, None);
        assert_eq!(args.profile_samples, 0);
        assert_eq!(args.profile_report, None);
        assert!(args.json_output);

        let session = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompts-jsonl",
            "/tmp/prompts.jsonl",
            "--session-backend",
            "engine",
            "--report",
            "/tmp/session.json",
            "--case-timeout-seconds",
            "600",
            "--profile-samples",
            "2",
            "--profile-report",
            "/tmp/profile.json",
            "--json",
        ])
        .expect("parse session CLI args");
        assert_eq!(session.prompt, None);
        assert_eq!(
            session.prompts_jsonl,
            Some(PathBuf::from("/tmp/prompts.jsonl"))
        );
        assert_eq!(session.session_backend, SessionBackend::Engine);
        assert_eq!(session.report, Some(PathBuf::from("/tmp/session.json")));
        assert_eq!(session.case_timeout_seconds, Some(600));
        assert_eq!(session.profile_samples, 2);
        assert_eq!(
            session.profile_report,
            Some(PathBuf::from("/tmp/profile.json"))
        );

        let err = parse_args_from(["--model-dir", "/tmp/gemma4-e2b", "--prompt", ""])
            .expect_err("empty prompt should fail");
        assert!(err.contains("--prompt must not be empty"));

        let err = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "0",
        ])
        .expect_err("zero max tokens should fail");
        assert!(err.contains("--max-new-tokens must be positive"));

        let args = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt",
            "Hello",
            "--max-total-tokens",
            "4096",
        ])
        .expect("large total token cap should parse");
        assert_eq!(args.max_total_tokens, Some(4096));

        let err = parse_args_from(["--model-dir", "/tmp/gemma4-e2b"])
            .expect_err("input mode should be required");
        assert!(err.contains("exactly one"));

        let err = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt",
            "Hello",
            "--prompts-jsonl",
            "/tmp/prompts.jsonl",
        ])
        .expect_err("input modes should be mutually exclusive");
        assert!(err.contains("exactly one"));

        let err = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt",
            "Hello",
            "--report",
            "/tmp/session.json",
        ])
        .expect_err("single prompt should not take session report");
        assert!(err.contains("--report is only valid"));

        let err = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt",
            "Hello",
            "--case-timeout-seconds",
            "600",
        ])
        .expect_err("single prompt should not take a case timeout");
        assert!(err.contains("--case-timeout-seconds is only valid"));

        let err = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompts-jsonl",
            "/tmp/prompts.jsonl",
            "--case-timeout-seconds",
            "0",
        ])
        .expect_err("zero case timeout should fail");
        assert!(err.contains("--case-timeout-seconds must be positive"));
    }

    #[test]
    fn rvllm_metal_infer_jsonl_session_cases_validate_and_default() {
        let path = unique_tmp_path("prompts.jsonl");
        std::fs::write(
            &path,
            "{\"name\":\"a\",\"prompt\":\"Hello\",\"max_new_tokens\":1,\"no_bos\":false}\n\
             {\"prompt\":\"Once upon a time\",\"reference_path\":\"ref.json\"}\n",
        )
        .expect("write JSONL fixture");
        let args = session_args_with_jsonl(path.clone());
        let cases = load_jsonl_session_cases(&args, &path).expect("load JSONL cases");
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].name, "a");
        assert_eq!(cases[0].prompt, "Hello");
        assert_eq!(cases[0].max_new_tokens, 1);
        assert!(!cases[0].no_bos);
        assert_eq!(cases[0].max_total_tokens, Some(32));
        assert_eq!(cases[1].name, "case_2");
        assert_eq!(cases[1].max_new_tokens, 3);
        assert!(cases[1].no_bos);
        assert_eq!(
            cases[1].hf_reference,
            Some(path.parent().unwrap().join("ref.json"))
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rvllm_metal_infer_jsonl_duplicate_names_fail() {
        let path = unique_tmp_path("duplicate-prompts.jsonl");
        std::fs::write(
            &path,
            "{\"name\":\"same\",\"prompt\":\"Hello\"}\n{\"name\":\"same\",\"prompt\":\"Bye\"}\n",
        )
        .expect("write JSONL fixture");
        let args = session_args_with_jsonl(path.clone());
        let err = load_jsonl_session_cases(&args, &path).expect_err("duplicates should fail");
        assert!(err.contains("duplicate session case name"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rvllm_metal_infer_reference_manifest_cases_validate() {
        let path = unique_tmp_path("manifest.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "schema": TEXT_REFERENCE_MANIFEST_SCHEMA,
                "output_dir": "/tmp/rvllm-reference-fixtures",
                "cases": [
                    {
                        "name": "text_reference_hello_step1",
                        "prompt_text": "Hello",
                        "decode_steps": 1,
                        "output": "hello.json"
                    }
                ]
            })
            .to_string(),
        )
        .expect("write manifest fixture");
        let mut args = session_args_with_jsonl(PathBuf::from("/tmp/unused.jsonl"));
        args.prompts_jsonl = None;
        args.reference_manifest = Some(path.clone());
        let cases = load_manifest_session_cases(&args, &path).expect("load manifest cases");
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].prompt, "Hello");
        assert_eq!(cases[0].max_new_tokens, 1);
        assert_eq!(
            cases[0].hf_reference,
            Some(PathBuf::from("/tmp/rvllm-reference-fixtures/hello.json"))
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rvllm_metal_infer_reference_manifest_missing_case_name_fails() {
        let path = unique_tmp_path("manifest-missing-name.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "schema": TEXT_REFERENCE_MANIFEST_SCHEMA,
                "cases": [
                    {
                        "prompt_text": "Hello",
                        "decode_steps": 1,
                        "output": "hello.json"
                    }
                ]
            })
            .to_string(),
        )
        .expect("write manifest fixture");
        let mut args = session_args_with_jsonl(PathBuf::from("/tmp/unused.jsonl"));
        args.prompts_jsonl = None;
        args.reference_manifest = Some(path.clone());
        let err = load_manifest_session_cases(&args, &path)
            .expect_err("manifest case names should be required");
        assert!(err.contains("missing required non-empty name"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rvllm_metal_infer_profile_summary_has_non_production_claim() {
        let args = session_args_with_jsonl(PathBuf::from("/tmp/prompts.jsonl"));
        let sample = serde_json::json!({
            "schema": SESSION_JSON_SCHEMA,
            "status": "pass",
            "prepare_ms": 10.0,
            "prefill_ms": 2.0,
            "decode_ms": 4.0,
            "tok_per_s": 1.5,
            "generated_tokens": 6,
            "command_buffers": 12,
            "encoders": 18,
            "layer_scale_encoder_fusions": 6,
            "forced_waits": 6,
            "cpu_encode_ns": 12_000_000,
            "command_buffer_wait_ns": 6_000_000
        });
        let summary = profile_summary(&args, vec![sample]).expect("profile summary");
        assert_eq!(summary["schema"], SESSION_PROFILE_SCHEMA);
        assert!(summary["claim"]
            .as_str()
            .expect("claim")
            .contains("not production performance"));
        assert_eq!(
            summary["medians"]["command_buffers_per_token"].as_f64(),
            Some(2.0)
        );
        assert_eq!(
            summary["minimums"]["command_buffers_per_token"].as_f64(),
            Some(2.0)
        );
        assert_eq!(
            summary["minimums"]["encoders_per_token"].as_f64(),
            Some(3.0)
        );
        assert_eq!(
            summary["minimums"]["forced_waits_per_token"].as_f64(),
            Some(1.0)
        );
        assert_eq!(
            summary["minimums"]["latency_ms_per_token"].as_f64(),
            Some(4.0 / 6.0)
        );
        assert_eq!(
            summary["minimums"]["cpu_encode_ms_per_token"].as_f64(),
            Some(2.0)
        );
        assert_eq!(
            summary["minimums"]["command_buffer_wait_ms_per_token"].as_f64(),
            Some(1.0)
        );
    }

    #[test]
    fn rvllm_metal_infer_token_budget_reports_current_limit() {
        validate_token_budget(31, 1, 32).expect("configured budget edge should pass");
        let err = validate_token_budget(31, 2, 32).expect_err("budget overflow should fail");
        assert!(err.contains("prompt token count + max_new_tokens <= 32"));

        let overflow = validate_token_budget(usize::MAX, 1, u32::MAX as usize)
            .expect_err("usize overflow must fail");
        assert!(overflow.contains("overflow"));
    }

    #[test]
    fn rvllm_metal_infer_env_token_limit_parser_accepts_configured_context() {
        assert_eq!(
            parse_max_total_tokens(MAX_TOTAL_TOKENS_ENV, "16").expect("default cap parses"),
            16
        );
        assert_eq!(
            parse_max_total_tokens(MAX_TOTAL_TOKENS_ENV, "4096").expect("large cap parses"),
            4096
        );
        let err =
            parse_max_total_tokens(MAX_TOTAL_TOKENS_ENV, "0").expect_err("zero cap should fail");
        assert!(err.contains("must be positive"));
        let err = validate_token_budget(31, 2, 32).expect_err("configured budget should fail");
        assert!(err.contains("prompt token count + max_new_tokens <= 32"));
        if usize::BITS > 32 {
            let too_large = (u32::MAX as u64 + 1).to_string();
            let err = parse_max_total_tokens(MAX_TOTAL_TOKENS_ENV, &too_large)
                .expect_err("u32 metadata narrowing must fail");
            assert!(err.contains("at most"));
        }
    }

    #[test]
    fn rvllm_metal_infer_json_report_has_schema_and_claim() {
        let report = InferReport {
            model_dir: PathBuf::from("/tmp/gemma4-e2b"),
            prompt: "Hello".to_owned(),
            prompt_token_ids: vec![2, 4],
            max_new_tokens: 1,
            generated_token_ids: vec![954],
            output_token_ids: vec![2, 4, 954],
            generated_text: " world".to_owned(),
            output_text: "Hello world".to_owned(),
            finish_reason: FinishReason::Length,
            prepare_ms: 1.0,
            prefill_ms: 2.0,
            decode_ms: 3.0,
            tok_per_s: 4.0,
            arena_bytes: 5,
            library_compiles: 1,
            pipeline_state_compiles: 31,
            last_step_gpu_execution_ns: Some(1_000_000),
            research_dispatch: serde_json::json!({"schema":"test"}),
            command_buffers: 6,
            encoders: 7,
            embedding_encoders: 1,
            ple_encoders: 2,
            layer_encoders: 3,
            layer_scale_encoder_fusions: 0,
            final_sample_encoders: 1,
            final_logits_encoders: 0,
            forced_waits: 8,
            cpu_wall_ns: 9,
            cpu_encode_ns: 10,
            command_buffer_wait_ns: 11,
            last_step_tokens: 1,
            last_step_command_buffers: 12,
            last_step_encoders: 13,
            last_step_forced_waits: 14,
            last_step_cpu_wall_ns: 15,
            last_step_cpu_encode_ns: 16,
            last_step_command_buffer_wait_ns: 17,
            debug_sync: false,
            large_model_opt_in: true,
            max_supported_total_tokens: DEFAULT_MAX_METAL_TOTAL_TOKENS,
            metal_compute_dtype: "bfloat16".to_owned(),
            metal_weight_dtype: "bfloat16".to_owned(),
            metal_moe_router_weight_dtype: "bfloat16".to_owned(),
            diagnostic_top_logits: Vec::new(),
        };
        let comparison = ReferenceComparison {
            path: PathBuf::from("/tmp/ref.json"),
            matched: true,
            mismatches: Vec::new(),
        };
        let value = report_value(&report, Some(&comparison));
        assert_eq!(value["schema"], JSON_SCHEMA);
        assert_eq!(value["claim"], CLAIM);
        assert_eq!(value["finish_reason"], "length");
        assert_eq!(value["generated_text"], " world");
        assert_eq!(value["library_compiles"], 1);
        assert_eq!(value["pipeline_state_compiles"], 31);
        assert_eq!(value["last_step_gpu_execution_ns"], 1_000_000);
        assert_eq!(value["research_dispatch"]["schema"], "test");
        assert_eq!(value["hf_reference"]["matched"].as_bool(), Some(true));
        assert_eq!(
            value["max_supported_total_tokens"].as_u64(),
            Some(DEFAULT_MAX_METAL_TOTAL_TOKENS as u64)
        );
    }

    #[test]
    fn rvllm_metal_infer_hf_reference_compare_reports_mismatch() {
        let report = InferReport {
            model_dir: PathBuf::from("/tmp/gemma4-e2b"),
            prompt: "Hello".to_owned(),
            prompt_token_ids: vec![2, 4],
            max_new_tokens: 1,
            generated_token_ids: vec![954],
            output_token_ids: vec![2, 4, 954],
            generated_text: " world".to_owned(),
            output_text: "Hello world".to_owned(),
            finish_reason: FinishReason::Length,
            prepare_ms: 1.0,
            prefill_ms: 2.0,
            decode_ms: 3.0,
            tok_per_s: 4.0,
            arena_bytes: 5,
            library_compiles: 1,
            pipeline_state_compiles: 31,
            last_step_gpu_execution_ns: Some(1_000_000),
            research_dispatch: serde_json::json!({"schema":"test"}),
            command_buffers: 6,
            encoders: 7,
            embedding_encoders: 1,
            ple_encoders: 2,
            layer_encoders: 3,
            layer_scale_encoder_fusions: 0,
            final_sample_encoders: 1,
            final_logits_encoders: 0,
            forced_waits: 8,
            cpu_wall_ns: 9,
            cpu_encode_ns: 10,
            command_buffer_wait_ns: 11,
            last_step_tokens: 1,
            last_step_command_buffers: 12,
            last_step_encoders: 13,
            last_step_forced_waits: 14,
            last_step_cpu_wall_ns: 15,
            last_step_cpu_encode_ns: 16,
            last_step_command_buffer_wait_ns: 17,
            debug_sync: false,
            large_model_opt_in: true,
            max_supported_total_tokens: DEFAULT_MAX_METAL_TOTAL_TOKENS,
            metal_compute_dtype: "bfloat16".to_owned(),
            metal_weight_dtype: "bfloat16".to_owned(),
            metal_moe_router_weight_dtype: "bfloat16".to_owned(),
            diagnostic_top_logits: Vec::new(),
        };
        let reference = HfReference {
            path: PathBuf::from("/tmp/ref.json"),
            prompt_token_ids: vec![2, 4],
            decode_steps: 1,
            generated_tokens: vec![145832],
        };
        let comparison = compare_hf_reference(&report, &reference);
        assert!(!comparison.matched);
        assert!(comparison
            .mismatches
            .iter()
            .any(|item| item.contains("generated_token_ids differ")));
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires cached Gemma4 E2B model directory and Apple Silicon Metal device"]
    fn rvllm_metal_infer_e2b_text_smoke() {
        let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
            eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
            return;
        };
        let model_dir = PathBuf::from(model_dir);
        if !tokenizer_path(&model_dir).is_file() {
            eprintln!(
                "skipping: tokenizer is missing: {}",
                tokenizer_path(&model_dir).display()
            );
            return;
        }
        let args = CliArgs {
            model_dir,
            prompt: Some("Hello".to_owned()),
            prompts_jsonl: None,
            reference_manifest: None,
            session_backend: SessionBackend::Direct,
            max_new_tokens: 1,
            max_total_tokens: None,
            eos_token_ids: vec![1, 2, 107],
            no_bos: false,
            large_model_opt_in: true,
            hf_reference: None,
            report: None,
            case_timeout_seconds: None,
            profile_samples: 0,
            profile_report: None,
            top_logits: 0,
            json_output: true,
        };
        let report = run_infer(&args).expect("run E2B Metal text inference");
        assert_eq!(report.generated_token_ids.len(), 1);
        assert!(!report.output_token_ids.is_empty());
        assert!(report.command_buffers >= 2);
        assert_eq!(
            CLAIM,
            "Apple Metal text inference workflow; not production-ready until acceptance gates pass"
        );
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires cached Gemma4 E2B model directory and Apple Silicon Metal device"]
    fn rvllm_metal_infer_e2b_configured_context_smoke() {
        let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
            eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
            return;
        };
        let model_dir = PathBuf::from(model_dir);
        if !tokenizer_path(&model_dir).is_file() {
            eprintln!(
                "skipping: tokenizer is missing: {}",
                tokenizer_path(&model_dir).display()
            );
            return;
        }
        let args = CliArgs {
            model_dir,
            prompt: Some("Hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello".to_owned()),
            prompts_jsonl: None,
            reference_manifest: None,
            session_backend: SessionBackend::Direct,
            max_new_tokens: 1,
            max_total_tokens: Some(32),
            eos_token_ids: vec![1, 2, 107],
            no_bos: false,
            large_model_opt_in: true,
            hf_reference: None,
            report: None,
            case_timeout_seconds: None,
            profile_samples: 0,
            profile_report: None,
            top_logits: 0,
            json_output: true,
        };
        let report = run_infer(&args).expect("run configured-context E2B Metal text inference");
        assert_eq!(report.max_supported_total_tokens, 32);
        assert_eq!(report.generated_token_ids.len(), 1);
        assert!(report.prompt_token_ids.len() + report.generated_token_ids.len() > 16);
    }

    fn checked_in_e2b_reference_path(filename: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/reference")
            .join(filename)
    }

    #[test]
    fn checked_in_e2b_reference_artifacts_are_consistent() {
        let step1 = parse_hf_reference(checked_in_e2b_reference_path(
            "gemma4-e2b-hf-text-hello-step1.json",
        ))
        .expect("parse checked-in one-step E2B reference");
        let steps2 = parse_hf_reference(checked_in_e2b_reference_path(
            "gemma4-e2b-hf-text-hello-steps2.json",
        ))
        .expect("parse checked-in two-step E2B reference");
        let steps4 = parse_hf_reference(checked_in_e2b_reference_path(
            "gemma4-e2b-hf-text-hello-steps4.json",
        ))
        .expect("parse checked-in four-step E2B reference");

        assert_eq!(step1.prompt_token_ids, vec![2, 9259]);
        assert_eq!(steps2.prompt_token_ids, step1.prompt_token_ids);
        assert_eq!(steps4.prompt_token_ids, step1.prompt_token_ids);
        assert_eq!(step1.generated_tokens, steps2.generated_tokens[..1]);
        assert_eq!(steps2.generated_tokens, steps4.generated_tokens[..2]);
        assert_eq!(steps4.generated_tokens, vec![236764, 108, 236777, 735]);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn run_reference_backed_hello_smoke(
        reference_filename: &str,
        max_new_tokens: usize,
        skip_label: &str,
    ) {
        let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
            eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
            return;
        };
        let model_dir = PathBuf::from(model_dir);
        if !tokenizer_path(&model_dir).is_file() {
            eprintln!(
                "skipping: tokenizer is missing: {}",
                tokenizer_path(&model_dir).display()
            );
            return;
        }
        let reference_path = checked_in_e2b_reference_path(reference_filename);
        if !reference_path.is_file() {
            eprintln!(
                "skipping: {skip_label} HF text reference artifact is missing: {}",
                reference_path.display()
            );
            return;
        }
        let args = CliArgs {
            model_dir,
            prompt: Some("Hello".to_owned()),
            prompts_jsonl: None,
            reference_manifest: None,
            session_backend: SessionBackend::Direct,
            max_new_tokens,
            max_total_tokens: None,
            eos_token_ids: vec![1, 2, 107],
            no_bos: false,
            large_model_opt_in: true,
            hf_reference: Some(reference_path.clone()),
            report: None,
            case_timeout_seconds: None,
            profile_samples: 0,
            profile_report: None,
            top_logits: 0,
            json_output: true,
        };
        let reference = parse_hf_reference(reference_path).expect("parse HF text reference");
        let report = run_infer(&args).expect("run reference-backed E2B Metal text inference");
        let comparison = compare_hf_reference(&report, &reference);
        assert!(comparison.matched, "{:?}", comparison.mismatches);
        assert_eq!(report.generated_token_ids, reference.generated_tokens);
        assert_eq!(report.generated_token_ids.len(), max_new_tokens);
        assert_eq!(report.max_new_tokens, max_new_tokens);
        assert_eq!(
            CLAIM,
            "Apple Metal text inference workflow; not production-ready until acceptance gates pass"
        );
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires cached Gemma4 E2B model directory and Apple Silicon Metal device; HF reference is checked in"]
    fn rvllm_metal_infer_e2b_reference_backed_text_smoke() {
        run_reference_backed_hello_smoke("gemma4-e2b-hf-text-hello-step1.json", 1, "one-step");
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires cached Gemma4 E2B model directory and Apple Silicon Metal device; two-step HF reference is checked in"]
    fn rvllm_metal_infer_e2b_reference_backed_text_two_step_smoke() {
        run_reference_backed_hello_smoke("gemma4-e2b-hf-text-hello-steps2.json", 2, "two-step");
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires cached Gemma4 E2B model directory and Apple Silicon Metal device; four-step HF reference is checked in"]
    fn rvllm_metal_infer_e2b_reference_backed_text_four_step_smoke() {
        run_reference_backed_hello_smoke("gemma4-e2b-hf-text-hello-steps4.json", 4, "four-step");
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires cached Gemma4 E2B model directory and Apple Silicon Metal device; one-step HF reference is checked in"]
    fn rvllm_metal_infer_e2b_engine_reference_backed_text_smoke() {
        let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
            eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
            return;
        };
        let model_dir = PathBuf::from(model_dir);
        if !tokenizer_path(&model_dir).is_file() {
            eprintln!(
                "skipping: tokenizer is missing: {}",
                tokenizer_path(&model_dir).display()
            );
            return;
        }
        let reference_path = checked_in_e2b_reference_path("gemma4-e2b-hf-text-hello-step1.json");
        if !reference_path.is_file() {
            eprintln!(
                "skipping: one-step HF text reference artifact is missing: {}",
                reference_path.display()
            );
            return;
        }

        let tokenizer = load_tokenizer(&model_dir).expect("load E2B tokenizer");
        let args = CliArgs {
            model_dir: model_dir.clone(),
            prompt: Some("Hello".to_owned()),
            prompts_jsonl: None,
            reference_manifest: None,
            session_backend: SessionBackend::Direct,
            max_new_tokens: 1,
            max_total_tokens: None,
            eos_token_ids: vec![1, 2, 107],
            no_bos: false,
            large_model_opt_in: true,
            hf_reference: Some(reference_path.clone()),
            report: None,
            case_timeout_seconds: None,
            profile_samples: 0,
            profile_report: None,
            top_logits: 0,
            json_output: true,
        };
        let prompt_token_ids =
            prompt_token_ids(&args, &tokenizer).expect("tokenize Engine smoke prompt");
        let reference =
            parse_hf_reference(reference_path).expect("parse one-step HF text reference");
        assert_eq!(prompt_token_ids, reference.prompt_token_ids);

        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
            .expect("parse Gemma4 architecture");
        let _large_model_env = EnvGuard::set_if(LARGE_MODEL_ENV, true);
        let shared_backend = SharedModelMetalBackend::new(model_dir.clone());
        let plan = rvllm_apple::AppleRuntimePlan {
            target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: args.max_new_tokens as u32,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
            ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
            ane_hidden_size: arch.hidden_size,
            ane_intermediate_size: arch.intermediate_size,
            ane_num_layers: arch.num_hidden_layers,
            model_layout_hash: [0u8; 32],
            weights_path: Some(model_dir),
        };

        let mut engine = rvllm_runtime::Engine::new()
            .with_apple_backend(Box::new(shared_backend))
            .with_apple_runtime_plan(plan)
            .expect("Engine should prepare real E2B ModelMetalBackend");
        engine
            .scheduler
            .enqueue(rvllm_runtime::sched_state::Request::new(
                rvllm_core::ReqId(1),
                prompt_token_ids
                    .iter()
                    .copied()
                    .map(rvllm_core::TokenId)
                    .collect(),
                args.max_new_tokens as u32,
            ));

        let prefill = engine.step_launch().expect("launch Engine text prefill");
        match prefill.plan().expect("Engine text prefill plan") {
            rvllm_runtime::BatchPlan::Prefill { req_ids, .. } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1)]);
            }
            other => panic!("expected Engine Prefill, got {other:?}"),
        }
        assert!(prefill
            .collect()
            .expect("collect Engine text prefill")
            .is_empty());

        let decode = engine.step_launch().expect("launch Engine text decode");
        match decode.plan().expect("Engine text decode plan") {
            rvllm_runtime::BatchPlan::Decode {
                req_ids,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1)]);
                assert_eq!(
                    positions,
                    &vec![(reference.prompt_token_ids.len() - 1) as u32]
                );
                assert_eq!(context_lens, &vec![reference.prompt_token_ids.len() as u32]);
            }
            other => panic!("expected Engine Decode, got {other:?}"),
        }
        let generated = decode
            .collect()
            .expect("collect Engine text decode")
            .into_iter()
            .map(|output| output.new_token.raw())
            .collect::<Vec<_>>();

        assert_eq!(generated, reference.generated_tokens);
        assert!(!engine.has_pending_work());
        assert_eq!(
            CLAIM,
            "Apple Metal text inference workflow; not production-ready until acceptance gates pass"
        );
    }
}
