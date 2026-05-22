//! Bounded Apple Metal text inference workflow for Gemma tokenizers.
//!
//! This is a production-facing shape, not a production-readiness claim. The
//! current E2B Metal backend still has a bounded probe arena, so this command
//! fails clearly when prompt plus generated tokens exceed that limit.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

const CLAIM: &str =
    "bounded Apple Metal text inference workflow; not production-ready until acceptance gates pass";
const JSON_SCHEMA: &str = "rvllm.apple_metal_text_infer.v1";
const LARGE_MODEL_ENV: &str = "RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE";
const MAX_METAL_E2B_TOKENS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    eos_token_ids: Vec<u32>,
    no_bos: bool,
    large_model_opt_in: bool,
    hf_reference: Option<PathBuf>,
    json_output: bool,
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
    command_buffers: u64,
    encoders: u64,
    forced_waits: u64,
    debug_sync: bool,
    large_model_opt_in: bool,
    max_supported_total_tokens: usize,
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

fn parse_args_from<I, S>(args: I) -> Result<CliArgs, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut model_dir = None;
    let mut prompt = None;
    let mut max_new_tokens = 1usize;
    let mut eos_token_ids = vec![1, 2, 107];
    let mut no_bos = false;
    let mut large_model_opt_in = false;
    let mut hf_reference = None;
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
            "--max-new-tokens" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--max-new-tokens requires a value".to_owned())?;
                max_new_tokens = parse_positive_usize("--max-new-tokens", &value)?;
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
            "--json" => {
                json_output = true;
            }
            "-h" | "--help" => return Err(usage()),
            other if other.starts_with('-') => return Err(format!("unknown argument: {other}")),
            other => return Err(format!("unexpected positional argument: {other}")),
        }
    }

    Ok(CliArgs {
        model_dir: model_dir.ok_or_else(|| "--model-dir is required".to_owned())?,
        prompt: prompt.ok_or_else(|| "--prompt is required".to_owned())?,
        max_new_tokens,
        eos_token_ids,
        no_bos,
        large_model_opt_in,
        hf_reference,
        json_output,
    })
}

fn usage() -> String {
    "usage: rvllm_metal_infer --model-dir <DIR> --prompt <TEXT> \
     [--max-new-tokens N] [--eos-token-ids IDS] [--no-bos] \
     [--large-model-opt-in] [--hf-reference <JSON>] [--json]"
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
    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|err| format!("tokenize prompt: {err}"))?;
    let mut token_ids = encoding.get_ids().to_vec();
    if !args.no_bos {
        token_ids.insert(0, 2);
    }
    if token_ids.is_empty() {
        return Err("--prompt produced zero token IDs".to_owned());
    }
    Ok(token_ids)
}

fn validate_token_budget(prompt_len: usize, max_new_tokens: usize) -> Result<(), String> {
    if prompt_len + max_new_tokens > MAX_METAL_E2B_TOKENS {
        return Err(format!(
            "current Metal E2B workflow supports prompt token count + max_new_tokens <= {MAX_METAL_E2B_TOKENS}; got {} + {}",
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
            let previous = std::env::var_os(name);
            std::env::set_var(name, "1");
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
    let mut mismatches = Vec::new();
    if report.prompt_token_ids != reference.prompt_token_ids {
        mismatches.push(format!(
            "prompt_token_ids differ: metal={:?} reference={:?}",
            report.prompt_token_ids, reference.prompt_token_ids
        ));
    }
    if report.max_new_tokens != reference.decode_steps {
        mismatches.push(format!(
            "max_new_tokens/decode_steps differ: metal={} reference={}",
            report.max_new_tokens, reference.decode_steps
        ));
    }
    if report.generated_token_ids != reference.generated_tokens {
        mismatches.push(format!(
            "generated_token_ids differ: metal={:?} reference={:?}",
            report.generated_token_ids, reference.generated_tokens
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

fn report_value(
    report: &InferReport,
    comparison: Option<&ReferenceComparison>,
) -> serde_json::Value {
    serde_json::json!({
        "schema": JSON_SCHEMA,
        "claim": CLAIM,
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
        "command_buffers": report.command_buffers,
        "encoders": report.encoders,
        "forced_waits": report.forced_waits,
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
    println!("forced_waits: {}", report.forced_waits);
    println!("debug_sync: {}", report.debug_sync);
    println!("large_model_opt_in: {}", report.large_model_opt_in);
    println!(
        "max_supported_total_tokens: {}",
        report.max_supported_total_tokens
    );
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
    validate_token_budget(prompt_token_ids.len(), args.max_new_tokens)?;

    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&args.model_dir)
        .map_err(|err| format!("parse Gemma4 architecture: {err}"))?;
    let env_opt_in = env_large_model_opted_in();
    let effective_large_opt_in = args.large_model_opt_in || env_opt_in;
    if arch.num_hidden_layers > 8 && !effective_large_opt_in {
        return Err(format!(
            "model has {} layers; pass --large-model-opt-in or set {LARGE_MODEL_ENV}=1 for this bounded Metal inference workflow",
            arch.num_hidden_layers
        ));
    }
    let _large_model_env =
        EnvGuard::set_if(LARGE_MODEL_ENV, args.large_model_opt_in && !env_opt_in);

    let mut backend = ModelMetalBackend::new(args.model_dir.clone());
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
        weights_path: Some(args.model_dir.clone()),
    };

    let prepare_start = std::time::Instant::now();
    backend
        .prepare(&plan)
        .map_err(|err| format!("prepare Metal backend: {err}"))?;
    let prepare_ms = ms(prepare_start.elapsed());

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
    let stats = backend.probe_perf_stats();
    let arena_bytes = backend
        .probe_arena_stats()
        .map(|arena| arena.capacity_bytes)
        .unwrap_or(0);

    Ok(InferReport {
        model_dir: args.model_dir.clone(),
        prompt: args.prompt.clone(),
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
        command_buffers: stats.command_buffers,
        encoders: stats.encoders,
        forced_waits: stats.forced_waits,
        debug_sync: backend.metal_debug_sync_enabled(),
        large_model_opt_in: effective_large_opt_in,
        max_supported_total_tokens: MAX_METAL_E2B_TOKENS,
    })
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
fn run_infer(_args: &CliArgs) -> Result<InferReport, String> {
    Err("rvllm_metal_infer requires --features apple on macOS".to_owned())
}

fn run_main() -> Result<(), String> {
    let args = parse_args_from(std::env::args().skip(1))?;
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

    #[test]
    fn rvllm_metal_infer_cli_args() {
        let args = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "4",
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
        assert_eq!(args.prompt, "Hello");
        assert_eq!(args.max_new_tokens, 4);
        assert_eq!(args.eos_token_ids, vec![1, 2, 107]);
        assert!(args.no_bos);
        assert!(args.large_model_opt_in);
        assert_eq!(args.hf_reference, Some(PathBuf::from("/tmp/ref.json")));
        assert!(args.json_output);

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
    }

    #[test]
    fn rvllm_metal_infer_token_budget_reports_current_limit() {
        validate_token_budget(15, 1).expect("budget edge should pass");
        let err = validate_token_budget(15, 2).expect_err("budget overflow should fail");
        assert!(err.contains("prompt token count + max_new_tokens <= 16"));
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
            command_buffers: 6,
            encoders: 7,
            forced_waits: 8,
            debug_sync: false,
            large_model_opt_in: true,
            max_supported_total_tokens: MAX_METAL_E2B_TOKENS,
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
        assert_eq!(value["hf_reference"]["matched"].as_bool(), Some(true));
        assert_eq!(
            value["max_supported_total_tokens"].as_u64(),
            Some(MAX_METAL_E2B_TOKENS as u64)
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
            command_buffers: 6,
            encoders: 7,
            forced_waits: 8,
            debug_sync: false,
            large_model_opt_in: true,
            max_supported_total_tokens: MAX_METAL_E2B_TOKENS,
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
            prompt: "Hello".to_owned(),
            max_new_tokens: 1,
            eos_token_ids: vec![1, 2, 107],
            no_bos: false,
            large_model_opt_in: true,
            hf_reference: None,
            json_output: true,
        };
        let report = run_infer(&args).expect("run bounded E2B Metal text inference");
        assert_eq!(report.generated_token_ids.len(), 1);
        assert!(!report.output_token_ids.is_empty());
        assert!(report.command_buffers >= 2);
        assert_eq!(
            CLAIM,
            "bounded Apple Metal text inference workflow; not production-ready until acceptance gates pass"
        );
    }
}
