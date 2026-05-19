//! Diagnostic Apple Metal decode probe for raw Gemma token IDs.
//!
//! This binary intentionally does not tokenize text and does not claim
//! production inference. It runs the existing `ModelMetalBackend` path for a
//! single raw-token prompt and prints sampled token IDs, top-k logits, and
//! probe counters.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

const CLAIM: &str = "diagnostic Apple Metal probe only; not production inference";
const LARGE_MODEL_ENV: &str = "RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE";
const MAX_PROBE_TOKENS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    model_dir: PathBuf,
    prompt_token_ids: Vec<u32>,
    decode_steps: usize,
    top_k: usize,
    large_model_opt_in: bool,
    hf_reference: Option<PathBuf>,
}

#[derive(Debug)]
struct TopLogit {
    token_id: u32,
    logit: f32,
}

#[derive(Debug)]
struct StepTopK {
    step: usize,
    top_k: Vec<TopLogit>,
}

#[derive(Debug)]
struct StepSelectedLogits {
    step: usize,
    selected_logits: Vec<TopLogit>,
}

#[derive(Debug)]
struct ProbeReport {
    model_dir: PathBuf,
    prompt_token_ids: Vec<u32>,
    decode_steps: usize,
    sampled_token_ids: Vec<u32>,
    per_step_top_k: Vec<StepTopK>,
    per_step_selected_logits: Vec<StepSelectedLogits>,
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
}

#[derive(Debug)]
struct HfReferenceStep {
    step: usize,
    next_token: u32,
    selected_logits: Vec<TopLogit>,
    top_logits: Vec<TopLogit>,
}

#[derive(Debug)]
struct HfReference {
    path: PathBuf,
    prompt_token_ids: Vec<u32>,
    decode_steps: usize,
    generated_tokens: Vec<u32>,
    steps: Vec<HfReferenceStep>,
}

#[derive(Debug)]
struct HfComparison {
    reference_path: PathBuf,
    matched: bool,
    mismatches: Vec<String>,
}

fn parse_token_ids(raw: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let token_id = part
            .parse::<u32>()
            .map_err(|err| format!("invalid token id {part:?}: {err}"))?;
        out.push(token_id);
    }
    if out.is_empty() {
        return Err("--prompt-token-ids must contain at least one token id".to_owned());
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
    let mut prompt_token_ids = None;
    let mut decode_steps = 1usize;
    let mut top_k = 16usize;
    let mut large_model_opt_in = false;
    let mut hf_reference = None;

    let mut iter = args.into_iter().map(Into::into).peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model-dir" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--model-dir requires a value".to_owned())?;
                model_dir = Some(PathBuf::from(value));
            }
            "--prompt-token-ids" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--prompt-token-ids requires a value".to_owned())?;
                prompt_token_ids = Some(parse_token_ids(&value)?);
            }
            "--decode-steps" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--decode-steps requires a value".to_owned())?;
                decode_steps = parse_positive_usize("--decode-steps", &value)?;
            }
            "--top-k" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--top-k requires a value".to_owned())?;
                top_k = parse_positive_usize("--top-k", &value)?;
            }
            "--large-model-opt-in" => {
                large_model_opt_in = true;
            }
            "--hf-reference" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--hf-reference requires a value".to_owned())?;
                hf_reference = Some(PathBuf::from(value));
            }
            "-h" | "--help" => return Err(usage()),
            other if other.starts_with('-') => return Err(format!("unknown argument: {other}")),
            other => return Err(format!("unexpected positional argument: {other}")),
        }
    }

    Ok(CliArgs {
        model_dir: model_dir.ok_or_else(|| "--model-dir is required".to_owned())?,
        prompt_token_ids: prompt_token_ids
            .ok_or_else(|| "--prompt-token-ids is required".to_owned())?,
        decode_steps,
        top_k,
        large_model_opt_in,
        hf_reference,
    })
}

fn usage() -> String {
    "usage: probe_apple_metal_decode --model-dir <DIR> --prompt-token-ids <IDS> \
     [--decode-steps N] [--top-k K] [--large-model-opt-in] [--hf-reference <JSON>]"
        .to_owned()
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn env_large_model_opted_in() -> bool {
    std::env::var(LARGE_MODEL_ENV).ok().as_deref() == Some("1")
}

struct EnvGuard {
    name: &'static str,
    previous: Option<OsString>,
    active: bool,
}

impl EnvGuard {
    fn set_if(name: &'static str, set: bool) -> Self {
        let previous = std::env::var_os(name);
        if set {
            std::env::set_var(name, "1");
        }
        Self {
            name,
            previous,
            active: set,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(previous) = &self.previous {
            std::env::set_var(self.name, previous);
        } else {
            std::env::remove_var(self.name);
        }
    }
}

fn top_k_logits(logits: &[f32], k: usize) -> Vec<TopLogit> {
    let mut indexed = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, logit)| TopLogit {
            token_id: idx as u32,
            logit,
        })
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.logit.total_cmp(&a.logit));
    indexed.truncate(k.min(indexed.len()));
    indexed
}

fn selected_logits(logits: &[f32], token_ids: &[u32]) -> Result<Vec<TopLogit>, String> {
    token_ids
        .iter()
        .copied()
        .map(|token_id| {
            let idx = token_id as usize;
            let logit = *logits
                .get(idx)
                .ok_or_else(|| format!("selected token id {token_id} exceeds vocab size"))?;
            Ok(TopLogit { token_id, logit })
        })
        .collect()
}

fn parse_logit_entries(value: &serde_json::Value, field: &str) -> Result<Vec<TopLogit>, String> {
    let entries = value[field]
        .as_array()
        .ok_or_else(|| format!("HF reference step missing {field} array"))?;
    entries
        .iter()
        .map(|entry| {
            let token_id = entry["token_id"]
                .as_u64()
                .ok_or_else(|| format!("HF reference {field} entry missing token_id"))?;
            let token_id = u32::try_from(token_id)
                .map_err(|_| format!("HF reference {field} token_id exceeds u32"))?;
            let logit = entry["logit"]
                .as_f64()
                .ok_or_else(|| format!("HF reference {field} entry missing logit"))?
                as f32;
            if !logit.is_finite() {
                return Err(format!(
                    "HF reference {field} token {token_id} is non-finite"
                ));
            }
            Ok(TopLogit { token_id, logit })
        })
        .collect()
}

fn parse_hf_reference(
    path: PathBuf,
    expected_prompt_token_ids: &[u32],
    expected_decode_steps: usize,
) -> Result<HfReference, String> {
    let raw = std::fs::read_to_string(&path)
        .map_err(|err| format!("read HF reference {}: {err}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|err| format!("parse HF reference JSON: {err}"))?;
    if value["schema"].as_str() != Some("rvllm.gemma4_hf_reference_logits.v1") {
        return Err("HF reference schema must be rvllm.gemma4_hf_reference_logits.v1".to_owned());
    }
    let prompt_token_ids = value["prompt_token_ids"]
        .as_array()
        .ok_or_else(|| "HF reference missing prompt_token_ids array".to_owned())?
        .iter()
        .map(|item| {
            let token_id = item
                .as_u64()
                .ok_or_else(|| "HF reference prompt token id must be an integer".to_owned())?;
            u32::try_from(token_id)
                .map_err(|_| "HF reference prompt token id exceeds u32".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if prompt_token_ids.as_slice() != expected_prompt_token_ids {
        return Err(format!(
            "HF reference prompt_token_ids {:?} do not match CLI prompt {:?}",
            prompt_token_ids, expected_prompt_token_ids
        ));
    }
    let decode_steps = value["decode_steps"]
        .as_u64()
        .ok_or_else(|| "HF reference missing decode_steps".to_owned())?
        as usize;
    if decode_steps != expected_decode_steps {
        return Err(format!(
            "HF reference decode_steps {decode_steps} does not match CLI decode_steps {expected_decode_steps}"
        ));
    }
    let generated_tokens = value["generated_tokens"]
        .as_array()
        .ok_or_else(|| "HF reference missing generated_tokens array".to_owned())?
        .iter()
        .map(|item| {
            let token_id = item
                .as_u64()
                .ok_or_else(|| "HF reference generated token id must be an integer".to_owned())?;
            u32::try_from(token_id)
                .map_err(|_| "HF reference generated token id exceeds u32".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let steps = value["steps"]
        .as_array()
        .ok_or_else(|| "HF reference missing steps array".to_owned())?
        .iter()
        .enumerate()
        .map(|(idx, step)| {
            let step_idx = step["step"].as_u64().unwrap_or(idx as u64) as usize;
            if step_idx != idx {
                return Err(format!(
                    "HF reference step index {step_idx} does not match position {idx}"
                ));
            }
            let next_token = step["next_token"]
                .as_u64()
                .ok_or_else(|| format!("HF reference step {idx} missing next_token"))?;
            let next_token = u32::try_from(next_token)
                .map_err(|_| format!("HF reference step {idx} next_token exceeds u32"))?;
            Ok(HfReferenceStep {
                step: idx,
                next_token,
                selected_logits: parse_logit_entries(step, "selected_logits")?,
                top_logits: parse_logit_entries(step, "top_logits")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if steps.len() != expected_decode_steps {
        return Err(format!(
            "HF reference has {} steps, expected {expected_decode_steps}",
            steps.len()
        ));
    }
    if generated_tokens.len() != expected_decode_steps {
        return Err(format!(
            "HF reference has {} generated tokens, expected {expected_decode_steps}",
            generated_tokens.len()
        ));
    }
    Ok(HfReference {
        path,
        prompt_token_ids,
        decode_steps,
        generated_tokens,
        steps,
    })
}

fn compare_hf_reference(report: &ProbeReport, reference: &HfReference) -> HfComparison {
    const LOGIT_TOLERANCE: f32 = 1.0;
    let mut mismatches = Vec::new();
    if reference.prompt_token_ids != report.prompt_token_ids {
        mismatches.push(format!(
            "prompt_token_ids differ: metal={:?} hf={:?}",
            report.prompt_token_ids, reference.prompt_token_ids
        ));
    }
    if reference.decode_steps != report.decode_steps {
        mismatches.push(format!(
            "decode_steps differ: metal={} hf={}",
            report.decode_steps, reference.decode_steps
        ));
    }
    if reference.generated_tokens != report.sampled_token_ids {
        mismatches.push(format!(
            "sampled_token_ids differ: metal={:?} hf={:?}",
            report.sampled_token_ids, reference.generated_tokens
        ));
    }

    for (step_idx, reference_step) in reference.steps.iter().enumerate() {
        let Some(metal_top) = report.per_step_top_k.get(step_idx) else {
            mismatches.push(format!("missing Metal top-k for step {step_idx}"));
            continue;
        };
        let Some(metal_selected) = report.per_step_selected_logits.get(step_idx) else {
            mismatches.push(format!("missing Metal selected logits for step {step_idx}"));
            continue;
        };
        if reference_step.step != metal_top.step || reference_step.step != metal_selected.step {
            mismatches.push(format!("step index mismatch at step {step_idx}"));
        }
        if report.sampled_token_ids.get(step_idx).copied() != Some(reference_step.next_token) {
            mismatches.push(format!(
                "step {step_idx} sampled token differs: metal={:?} hf={}",
                report.sampled_token_ids.get(step_idx),
                reference_step.next_token
            ));
        }

        for expected in &reference_step.selected_logits {
            match metal_selected
                .selected_logits
                .iter()
                .find(|item| item.token_id == expected.token_id)
            {
                Some(actual) => {
                    let delta = (actual.logit - expected.logit).abs();
                    if delta > LOGIT_TOLERANCE {
                        mismatches.push(format!(
                            "step {step_idx} selected logit[{}] delta {delta:.6} exceeds {LOGIT_TOLERANCE}: metal={:.6} hf={:.6}",
                            expected.token_id, actual.logit, expected.logit
                        ));
                    }
                }
                None => mismatches.push(format!(
                    "step {step_idx} missing selected logit[{}]",
                    expected.token_id
                )),
            }
        }

        if metal_top.top_k.len() < reference_step.top_logits.len() {
            mismatches.push(format!(
                "step {step_idx} Metal top-k has {} entries, HF has {}",
                metal_top.top_k.len(),
                reference_step.top_logits.len()
            ));
        }
        for (rank, expected) in reference_step.top_logits.iter().enumerate() {
            let Some(actual) = metal_top.top_k.get(rank) else {
                continue;
            };
            let delta = (actual.logit - expected.logit).abs();
            if actual.token_id != expected.token_id || delta > LOGIT_TOLERANCE {
                mismatches.push(format!(
                    "step {step_idx} top-k rank {rank} differs: metal=({}, {:.6}) hf=({}, {:.6}) delta={delta:.6}",
                    actual.token_id, actual.logit, expected.token_id, expected.logit
                ));
            }
        }
    }

    HfComparison {
        reference_path: reference.path.clone(),
        matched: mismatches.is_empty(),
        mismatches,
    }
}

fn json_u32_array(values: &[u32]) -> String {
    serde_json::to_string(values).expect("serialize u32 array")
}

fn per_step_top_k_json(steps: &[StepTopK]) -> String {
    let value = serde_json::Value::Array(
        steps
            .iter()
            .map(|step| {
                serde_json::json!({
                    "step": step.step,
                    "top_k": step.top_k.iter().map(|item| {
                        serde_json::json!({
                            "token_id": item.token_id,
                            "logit": item.logit,
                        })
                    }).collect::<Vec<_>>(),
                })
            })
            .collect(),
    );
    serde_json::to_string(&value).expect("serialize top-k")
}

fn hf_comparison_json(comparison: &HfComparison) -> String {
    serde_json::to_string(&serde_json::json!({
        "reference_path": comparison.reference_path,
        "matched": comparison.matched,
        "mismatches": comparison.mismatches,
    }))
    .expect("serialize HF comparison")
}

fn print_report(report: &ProbeReport, comparison: Option<&HfComparison>) {
    println!("claim: {CLAIM}");
    println!("model_dir: {}", report.model_dir.display());
    println!(
        "prompt_token_ids: {}",
        json_u32_array(&report.prompt_token_ids)
    );
    println!("decode_steps: {}", report.decode_steps);
    println!(
        "sampled_token_ids: {}",
        json_u32_array(&report.sampled_token_ids)
    );
    println!(
        "per_step_top_k: {}",
        per_step_top_k_json(&report.per_step_top_k)
    );
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
    if let Some(comparison) = comparison {
        println!("hf_reference: {}", comparison.reference_path.display());
        println!("hf_reference_match: {}", comparison.matched);
        println!(
            "hf_reference_comparison: {}",
            hf_comparison_json(comparison)
        );
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn run_probe(args: &CliArgs, reference: Option<&HfReference>) -> Result<ProbeReport, String> {
    use rvllm_apple::{AppleBackend, HandoffCapsule, HandoffKind};
    use rvllm_core::{ReqId, TokenId};
    use rvllm_runtime::apple_metal_backend::ModelMetalBackend;

    if !args.model_dir.is_dir() {
        return Err(format!(
            "model path does not exist or is not a directory: {}",
            args.model_dir.display()
        ));
    }
    if args
        .prompt_token_ids
        .len()
        .saturating_add(args.decode_steps)
        > MAX_PROBE_TOKENS
    {
        return Err(format!(
            "current diagnostic Metal probe supports prompt length + decode steps <= {MAX_PROBE_TOKENS}"
        ));
    }

    let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&args.model_dir)
        .map_err(|err| format!("parse Gemma4 architecture: {err}"))?;
    let env_opt_in = env_large_model_opted_in();
    let effective_large_opt_in = args.large_model_opt_in || env_opt_in;
    if arch.num_hidden_layers > 8 && !effective_large_opt_in {
        return Err(format!(
            "model has {} layers; pass --large-model-opt-in or set {LARGE_MODEL_ENV}=1 for this diagnostic probe",
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
        rollout_tokens: args.decode_steps as u32,
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

    let prompt_tokens = args
        .prompt_token_ids
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
    let mut sampled_token_ids = Vec::with_capacity(args.decode_steps);
    let mut per_step_top_k = Vec::with_capacity(args.decode_steps);
    let mut per_step_selected_logits = Vec::with_capacity(args.decode_steps);
    let decode_start = std::time::Instant::now();
    for step_idx in 0..args.decode_steps {
        let decode = HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![ReqId(1)],
            vec![current],
            vec![0, 1],
            vec![(prompt_len - 1 + step_idx) as u32],
            vec![(prompt_len + step_idx) as u32],
        );
        let decode_ticket = backend
            .launch_rollout(&decode, None)
            .map_err(|err| format!("launch decode step {step_idx}: {err}"))?;
        let logits = backend
            .probe_read_decode_logits_f32(1)
            .map_err(|err| format!("read decode logits step {step_idx}: {err}"))?;
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "decode logits contain non-finite values at step {step_idx}"
            ));
        }
        if let Some(reference_step) = reference.and_then(|reference| reference.steps.get(step_idx))
        {
            let token_ids = reference_step
                .selected_logits
                .iter()
                .map(|entry| entry.token_id)
                .collect::<Vec<_>>();
            per_step_selected_logits.push(StepSelectedLogits {
                step: step_idx,
                selected_logits: selected_logits(&logits, &token_ids)?,
            });
        }
        per_step_top_k.push(StepTopK {
            step: step_idx,
            top_k: top_k_logits(&logits, args.top_k),
        });
        let out = backend
            .collect(decode_ticket)
            .map_err(|err| format!("collect decode step {step_idx}: {err}"))?;
        if out.len() != 1 {
            return Err(format!(
                "decode step {step_idx} returned {} sampled tokens, expected 1",
                out.len()
            ));
        }
        let sampled = out[0].token_id.raw();
        sampled_token_ids.push(sampled);
        current = TokenId(sampled);
    }
    let decode_ms = ms(decode_start.elapsed());
    let tok_per_s = if decode_ms > 0.0 {
        (args.decode_steps as f64) / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let stats = backend.probe_perf_stats();
    let arena_bytes = backend
        .probe_arena_stats()
        .map(|arena| arena.capacity_bytes)
        .unwrap_or(0);

    Ok(ProbeReport {
        model_dir: args.model_dir.clone(),
        prompt_token_ids: args.prompt_token_ids.clone(),
        decode_steps: args.decode_steps,
        sampled_token_ids,
        per_step_top_k,
        per_step_selected_logits,
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
    })
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
fn run_probe(_args: &CliArgs, _reference: Option<&HfReference>) -> Result<ProbeReport, String> {
    Err("probe_apple_metal_decode requires --features apple on macOS".to_owned())
}

fn run_main() -> Result<(), String> {
    let args = parse_args_from(std::env::args().skip(1))?;
    let reference = args
        .hf_reference
        .clone()
        .map(|path| parse_hf_reference(path, &args.prompt_token_ids, args.decode_steps))
        .transpose()?;
    let report = run_probe(&args, reference.as_ref())?;
    let comparison = reference
        .as_ref()
        .map(|reference| compare_hf_reference(&report, reference));
    print_report(&report, comparison.as_ref());
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
    fn probe_apple_metal_decode_cli_args() {
        let args = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt-token-ids",
            "2,4",
            "--decode-steps",
            "1",
            "--top-k",
            "16",
            "--large-model-opt-in",
        ])
        .expect("parse CLI args");
        assert_eq!(args.model_dir, PathBuf::from("/tmp/gemma4-e2b"));
        assert_eq!(args.prompt_token_ids, vec![2, 4]);
        assert_eq!(args.decode_steps, 1);
        assert_eq!(args.top_k, 16);
        assert!(args.large_model_opt_in);
        assert_eq!(args.hf_reference, None);

        let err = parse_args_from(["--model-dir", "/tmp/gemma4-e2b", "--prompt-token-ids", ""])
            .expect_err("empty token list should fail");
        assert!(err.contains("at least one token id"));

        let args = parse_args_from([
            "--model-dir",
            "/tmp/gemma4-e2b",
            "--prompt-token-ids",
            "2,4",
            "--hf-reference",
            "/tmp/ref.json",
        ])
        .expect("parse HF reference arg");
        assert_eq!(args.hf_reference, Some(PathBuf::from("/tmp/ref.json")));
    }

    #[test]
    fn probe_apple_metal_decode_hf_reference_compare_reports_mismatch() {
        let report = ProbeReport {
            model_dir: PathBuf::from("/tmp/gemma4-e2b"),
            prompt_token_ids: vec![2, 4],
            decode_steps: 1,
            sampled_token_ids: vec![145832],
            per_step_top_k: vec![StepTopK {
                step: 0,
                top_k: vec![TopLogit {
                    token_id: 145832,
                    logit: 29.734375,
                }],
            }],
            per_step_selected_logits: vec![StepSelectedLogits {
                step: 0,
                selected_logits: vec![TopLogit {
                    token_id: 4,
                    logit: 20.875,
                }],
            }],
            prepare_ms: 1.0,
            prefill_ms: 1.0,
            decode_ms: 1.0,
            tok_per_s: 1.0,
            arena_bytes: 1,
            command_buffers: 1,
            encoders: 1,
            forced_waits: 1,
            debug_sync: false,
            large_model_opt_in: true,
        };
        let reference = HfReference {
            path: PathBuf::from("/tmp/ref.json"),
            prompt_token_ids: vec![2, 4],
            decode_steps: 1,
            generated_tokens: vec![954],
            steps: vec![HfReferenceStep {
                step: 0,
                next_token: 954,
                selected_logits: vec![TopLogit {
                    token_id: 4,
                    logit: 20.875,
                }],
                top_logits: vec![TopLogit {
                    token_id: 954,
                    logit: 22.875,
                }],
            }],
        };
        let comparison = compare_hf_reference(&report, &reference);
        assert!(!comparison.matched);
        assert!(comparison
            .mismatches
            .iter()
            .any(|item| item.contains("sampled_token_ids differ")));
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires cached Gemma4 E2B model directory and Apple Silicon Metal device"]
    fn probe_apple_metal_decode_e2b_raw_tokens_smoke() {
        let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
            eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
            return;
        };
        let args = CliArgs {
            model_dir: PathBuf::from(model_dir),
            prompt_token_ids: vec![2, 4],
            decode_steps: 1,
            top_k: 16,
            large_model_opt_in: true,
            hf_reference: None,
        };
        let report = run_probe(&args, None).expect("run E2B raw-token Metal probe");
        eprintln!("{report:#?}");
        assert_eq!(report.prompt_token_ids, vec![2, 4]);
        assert_eq!(report.decode_steps, 1);
        assert_eq!(report.sampled_token_ids.len(), 1);
        assert_eq!(report.per_step_top_k.len(), 1);
        assert_eq!(report.per_step_top_k[0].top_k.len(), 16);
        assert!(report.arena_bytes > 0);
    }
}
