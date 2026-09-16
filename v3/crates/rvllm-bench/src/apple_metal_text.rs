use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const LARGE_MODEL_ENV: &str = "RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE";
const MAX_TOTAL_TOKENS_ENV: &str = "RVLLM_METAL_MAX_TOTAL_TOKENS";
const MAX_BATCH_TOKENS_ENV: &str = "RVLLM_METAL_MAX_BATCH_TOKENS";
const MAX_BATCH_SEQUENCES_ENV: &str = "RVLLM_METAL_MAX_BATCH_SEQUENCES";
pub const METAL_CLAIM: &str =
    "Apple Metal bench/eval/ppl path; not production-ready or parity-complete";

#[derive(Clone, Debug)]
pub struct MetalTextOptions {
    pub model_dir: PathBuf,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub max_total_tokens: usize,
    pub max_batch_tokens: usize,
    pub no_bos: bool,
    pub eos_token_ids: Vec<u32>,
    pub large_model_opt_in: bool,
}

#[derive(Clone, Debug)]
pub struct MetalBenchOptions {
    pub text: MetalTextOptions,
    pub batch: u32,
    pub iters: u32,
    pub warmup: u32,
}

#[derive(Clone, Debug)]
pub struct MetalCacheGateOptions {
    pub text: MetalTextOptions,
}

#[derive(Clone, Debug)]
pub struct MetalPplOptions {
    pub model_dir: PathBuf,
    pub text: String,
    pub chunk_len: usize,
    pub max_chunks: usize,
    pub no_bos: bool,
    pub max_total_tokens: usize,
    pub large_model_opt_in: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetalEncoderCounts {
    pub embedding: u64,
    pub ple_input: u64,
    pub layer_body: u64,
    pub layer_scale_fused: u64,
    pub final_sample: u64,
    pub final_logits_diagnostic: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetalTextRun {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub output_token_ids: Vec<u32>,
    pub generated_text: String,
    pub output_text: String,
    pub finish_reason: String,
    pub prepare_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub tok_per_sec: f64,
    pub arena_bytes: usize,
    pub command_buffers: u64,
    pub encoders: u64,
    pub embedding_encoders: u64,
    pub ple_encoders: u64,
    pub layer_encoders: u64,
    pub layer_scale_encoder_fusions: u64,
    pub final_sample_encoders: u64,
    pub final_logits_encoders: u64,
    pub encoder_counts_by_kernel_family: MetalEncoderCounts,
    pub forced_waits: u64,
    pub cpu_wall_ns: u64,
    pub cpu_encode_ns: u64,
    pub command_buffer_wait_ns: u64,
    pub last_step_tokens: u64,
    pub last_step_command_buffers: u64,
    pub last_step_encoders: u64,
    pub last_step_forced_waits: u64,
    pub last_step_cpu_wall_ns: u64,
    pub last_step_cpu_encode_ns: u64,
    pub last_step_command_buffer_wait_ns: u64,
    pub claim: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetalPplRun {
    pub perplexity: f64,
    pub tokens: usize,
    pub chunk_len: usize,
    pub elapsed_s: f64,
    pub prepare_ms: f64,
    pub command_buffers: u64,
    pub encoders: u64,
    pub embedding_encoders: u64,
    pub ple_encoders: u64,
    pub layer_encoders: u64,
    pub layer_scale_encoder_fusions: u64,
    pub final_sample_encoders: u64,
    pub final_logits_encoders: u64,
    pub encoder_counts_by_kernel_family: MetalEncoderCounts,
    pub forced_waits: u64,
    pub cpu_wall_ns: u64,
    pub cpu_encode_ns: u64,
    pub command_buffer_wait_ns: u64,
    pub last_step_tokens: u64,
    pub last_step_command_buffers: u64,
    pub last_step_encoders: u64,
    pub last_step_forced_waits: u64,
    pub last_step_cpu_wall_ns: u64,
    pub last_step_cpu_encode_ns: u64,
    pub last_step_command_buffer_wait_ns: u64,
    pub claim: &'static str,
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
pub fn run_eval(_options: MetalTextOptions) -> Result<MetalTextRun, String> {
    Err("Apple Metal eval requires --features apple on macOS".to_owned())
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
pub fn run_bench(_options: MetalBenchOptions) -> Result<Value, String> {
    Err("Apple Metal bench requires --features apple on macOS".to_owned())
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
pub fn run_cache_gate(_options: MetalCacheGateOptions) -> Result<Value, String> {
    Err("Apple Metal cache gate requires --features apple on macOS".to_owned())
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
pub fn run_ppl(_options: MetalPplOptions) -> Result<MetalPplRun, String> {
    Err("Apple Metal PPL requires --features apple on macOS".to_owned())
}

#[cfg(all(feature = "apple", target_os = "macos"))]
pub fn run_eval(options: MetalTextOptions) -> Result<MetalTextRun, String> {
    let tokenizer = load_tokenizer(&options.model_dir)?;
    let prompt_token_ids = tokenize_prompt(&tokenizer, &options.prompt, options.no_bos)?;
    let max_new_tokens = options.max_new_tokens;
    let mut options = options;
    options.max_batch_tokens = options.max_batch_tokens.max(prompt_token_ids.len().max(1));
    let mut prepared = PreparedMetalText::new(options, tokenizer)?;
    let prepare_ms = prepared.prepare_ms;
    let mut run = prepared.generate(&prompt_token_ids, max_new_tokens)?;
    run.prepare_ms = prepare_ms;
    Ok(run)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
pub fn run_bench(options: MetalBenchOptions) -> Result<Value, String> {
    if options.batch == 0 {
        return Err("RVLLM_BATCH must be positive".to_owned());
    }
    if options.iters == 0 {
        return Err("RVLLM_ITERS must be positive".to_owned());
    }
    let tokenizer = load_tokenizer(&options.text.model_dir)?;
    let prompt_token_ids = tokenize_prompt(&tokenizer, &options.text.prompt, options.text.no_bos)?;
    let max_new_tokens = options.text.max_new_tokens;
    validate_max_total_tokens(options.text.max_total_tokens)?;
    validate_max_new_tokens(max_new_tokens)?;
    validate_token_budget(
        prompt_token_ids.len(),
        max_new_tokens,
        options.text.max_total_tokens,
    )?;
    let batch = options.batch;
    let iters = options.iters;
    let warmup = options.warmup;
    let cache_enabled = env_bool("RVLLM_BENCH_PROMPT_CACHE");
    let config = rvllm_serve::ServerConfig {
        addr: "127.0.0.1:0".to_owned(),
        model_dir: options.text.model_dir,
        infer_bin: PathBuf::from("unused-continuous-benchmark"),
        backend: rvllm_serve::ServerBackendMode::MetalDirect,
        max_new_tokens,
        max_total_tokens: Some(options.text.max_total_tokens),
        large_model_opt_in: options.text.large_model_opt_in,
        metallib_bf16: None,
        ane_compile_budget: 0,
    };
    let prompt = options.text.prompt;
    let mut record = rvllm_serve::run_apple_continuous_benchmark(
        config,
        prompt_token_ids.clone(),
        max_new_tokens,
        batch,
        iters,
        warmup,
        cache_enabled,
    )?;
    if let Some(object) = record.as_object_mut() {
        object.insert("prompt".to_owned(), json!(prompt));
        object.insert("prompt_token_ids".to_owned(), json!(prompt_token_ids));
        object.insert("cache_enabled".to_owned(), json!(cache_enabled));
        object.insert("claim".to_owned(), json!(METAL_CLAIM));
    }
    Ok(record)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
pub fn run_cache_gate(options: MetalCacheGateOptions) -> Result<Value, String> {
    const GATE_PROMPT_TOKENS: usize = 513;
    let tokenizer = load_tokenizer(&options.text.model_dir)?;
    let mut prompt_token_ids =
        tokenize_prompt(&tokenizer, &options.text.prompt, options.text.no_bos)?;
    if prompt_token_ids.len() < GATE_PROMPT_TOKENS {
        return Err(format!(
            "cache gate prompt encoded to {} tokens; provide at least {GATE_PROMPT_TOKENS}",
            prompt_token_ids.len()
        ));
    }
    prompt_token_ids.truncate(GATE_PROMPT_TOKENS);
    validate_max_total_tokens(options.text.max_total_tokens)?;
    validate_max_new_tokens(options.text.max_new_tokens)?;
    validate_token_budget(
        prompt_token_ids.len(),
        options.text.max_new_tokens,
        options.text.max_total_tokens,
    )?;
    let config = rvllm_serve::ServerConfig {
        addr: "127.0.0.1:0".to_owned(),
        model_dir: options.text.model_dir,
        infer_bin: PathBuf::from("unused-prompt-cache-gate"),
        backend: rvllm_serve::ServerBackendMode::MetalDirect,
        max_new_tokens: options.text.max_new_tokens,
        max_total_tokens: Some(options.text.max_total_tokens),
        large_model_opt_in: options.text.large_model_opt_in,
        metallib_bf16: None,
        ane_compile_budget: 0,
    };
    let mut record = rvllm_serve::run_apple_prompt_cache_gate(
        config,
        prompt_token_ids.clone(),
        options.text.max_new_tokens,
    )?;
    if let Some(object) = record.as_object_mut() {
        object.insert("prompt_token_ids".to_owned(), json!(prompt_token_ids));
        object.insert("claim".to_owned(), json!(METAL_CLAIM));
    }
    Ok(record)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
pub fn run_ppl(options: MetalPplOptions) -> Result<MetalPplRun, String> {
    if options.chunk_len < 2 {
        return Err("RVLLM_PPL_CHUNK must be at least 2 for Metal PPL".to_owned());
    }
    validate_max_total_tokens(options.max_total_tokens)?;
    if options.chunk_len > options.max_total_tokens {
        return Err(format!(
            "RVLLM_PPL_CHUNK={} exceeds Metal max_total_tokens={}; raise RVLLM_METAL_MAX_TOTAL_TOKENS",
            options.chunk_len, options.max_total_tokens
        ));
    }
    let tokenizer = load_tokenizer(&options.model_dir)?;
    let all_ids = tokenize_prompt(&tokenizer, &options.text, options.no_bos)?;
    if all_ids.len() < options.chunk_len {
        return Err(format!(
            "not enough tokens ({}) for Metal chunk_len={}",
            all_ids.len(),
            options.chunk_len
        ));
    }
    let text_options = MetalTextOptions {
        model_dir: options.model_dir,
        prompt: String::new(),
        max_new_tokens: 1,
        max_total_tokens: options.max_total_tokens,
        max_batch_tokens: options.chunk_len,
        no_bos: options.no_bos,
        eos_token_ids: Vec::new(),
        large_model_opt_in: options.large_model_opt_in,
    };
    let mut prepared = PreparedMetalText::new(text_options, tokenizer)?;
    let mut chunks: Vec<&[u32]> = all_ids.chunks(options.chunk_len).collect();
    if let Some(last) = chunks.last() {
        if last.len() < options.chunk_len {
            chunks.pop();
        }
    }
    if options.max_chunks > 0 && chunks.len() > options.max_chunks {
        chunks.truncate(options.max_chunks);
    }
    let eval_start = Instant::now();
    let before = prepared.backend.probe_perf_stats();
    let mut total_nll = 0.0f64;
    let mut total_tokens = 0usize;
    for chunk in chunks {
        for target_idx in 1..chunk.len() {
            let logits = prepared.next_token_logits(&chunk[..target_idx])?;
            let target = chunk[target_idx] as usize;
            if target >= logits.len() {
                return Err(format!(
                    "target token {target} exceeds logits vocab {}",
                    logits.len()
                ));
            }
            total_nll += negative_log_likelihood(&logits, target);
            total_tokens += 1;
        }
    }
    if total_tokens == 0 {
        return Err("Metal PPL evaluated zero target tokens".to_owned());
    }
    let after = prepared.backend.probe_perf_stats();
    let delta = stats_delta(before, after);
    let elapsed_s = eval_start.elapsed().as_secs_f64();
    Ok(MetalPplRun {
        perplexity: (total_nll / total_tokens as f64).exp(),
        tokens: total_tokens,
        chunk_len: options.chunk_len,
        elapsed_s,
        prepare_ms: prepared.prepare_ms,
        command_buffers: delta.command_buffers,
        encoders: delta.encoders,
        embedding_encoders: delta.embedding_encoders,
        ple_encoders: delta.ple_encoders,
        layer_encoders: delta.layer_encoders,
        layer_scale_encoder_fusions: delta.layer_scale_encoder_fusions,
        final_sample_encoders: delta.final_sample_encoders,
        final_logits_encoders: delta.final_logits_encoders,
        encoder_counts_by_kernel_family: encoder_counts_from_delta(&delta),
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
        claim: METAL_CLAIM,
    })
}

#[cfg(all(feature = "apple", target_os = "macos"))]
struct PreparedMetalText {
    backend: rvllm_runtime::apple_metal_backend::ModelMetalBackend,
    tokenizer: tokenizers::Tokenizer,
    eos_token_ids: Vec<u32>,
    max_total_tokens: usize,
    prepare_ms: f64,
    _large_model_env: EnvGuard,
    _max_tokens_env: EnvGuard,
    _max_batch_tokens_env: EnvGuard,
    _max_batch_sequences_env: EnvGuard,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl PreparedMetalText {
    fn new(options: MetalTextOptions, tokenizer: tokenizers::Tokenizer) -> Result<Self, String> {
        use rvllm_apple::AppleBackend;

        validate_max_total_tokens(options.max_total_tokens)?;
        validate_max_new_tokens(options.max_new_tokens)?;
        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&options.model_dir)
            .map_err(|err| format!("parse Gemma4 architecture: {err}"))?;
        let large_model_env = EnvGuard::set_if(
            LARGE_MODEL_ENV,
            options.large_model_opt_in && !env_truthy(LARGE_MODEL_ENV),
        );
        let max_tokens_env = EnvGuard::set_value_if(
            MAX_TOTAL_TOKENS_ENV,
            Some(options.max_total_tokens.to_string()),
        );
        let max_batch_tokens = options.max_batch_tokens.max(1);
        let max_batch_tokens_env =
            EnvGuard::set_value_if(MAX_BATCH_TOKENS_ENV, Some(max_batch_tokens.to_string()));
        let max_batch_sequences_env =
            EnvGuard::set_value_if(MAX_BATCH_SEQUENCES_ENV, Some("1".to_owned()));
        let rollout_tokens = u32::try_from(options.max_new_tokens)
            .map_err(|_| format!("Metal max_new_tokens must be at most {}", u32::MAX))?;
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
            weights_path: Some(options.model_dir.clone()),
        };
        let mut backend =
            rvllm_runtime::apple_metal_backend::ModelMetalBackend::new(options.model_dir);
        let prepare_start = Instant::now();
        backend
            .prepare(&plan)
            .map_err(|err| format!("prepare Metal backend: {err}"))?;
        Ok(Self {
            backend,
            tokenizer,
            eos_token_ids: options.eos_token_ids,
            max_total_tokens: options.max_total_tokens,
            prepare_ms: ms(prepare_start.elapsed()),
            _large_model_env: large_model_env,
            _max_tokens_env: max_tokens_env,
            _max_batch_tokens_env: max_batch_tokens_env,
            _max_batch_sequences_env: max_batch_sequences_env,
        })
    }

    fn generate(
        &mut self,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<MetalTextRun, String> {
        validate_token_budget(
            prompt_token_ids.len(),
            max_new_tokens,
            self.max_total_tokens,
        )?;
        let before = self.backend.probe_perf_stats();
        let prompt_tokens = prompt_token_ids
            .iter()
            .copied()
            .map(rvllm_core::TokenId)
            .collect::<Vec<_>>();
        let prefill_ms = self.prefill(&prompt_tokens)?;
        let mut current = *prompt_tokens.last().expect("prompt token");
        let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
        let mut finish_reason = "length";
        let decode_start = Instant::now();
        for step_idx in 0..max_new_tokens {
            let token = self.decode_one(current, prompt_tokens.len(), step_idx)?;
            generated_token_ids.push(token);
            current = rvllm_core::TokenId(token);
            if self.eos_token_ids.contains(&token) {
                finish_reason = "eos";
                break;
            }
        }
        let decode_ms = ms(decode_start.elapsed());
        let after = self.backend.probe_perf_stats();
        let delta = stats_delta(before, after);
        let mut output_token_ids = prompt_token_ids.to_vec();
        output_token_ids.extend(generated_token_ids.iter().copied());
        let generated_text = decode_text(&self.tokenizer, &generated_token_ids, "generated")?;
        let output_text = decode_text(&self.tokenizer, &output_token_ids, "output")?;
        let tok_per_sec = if decode_ms > 0.0 {
            generated_token_ids.len() as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        Ok(MetalTextRun {
            prompt_token_ids: prompt_token_ids.to_vec(),
            generated_token_ids,
            output_token_ids,
            generated_text,
            output_text,
            finish_reason: finish_reason.to_owned(),
            prepare_ms: 0.0,
            prefill_ms,
            decode_ms,
            tok_per_sec,
            arena_bytes: self.arena_bytes(),
            command_buffers: delta.command_buffers,
            encoders: delta.encoders,
            embedding_encoders: delta.embedding_encoders,
            ple_encoders: delta.ple_encoders,
            layer_encoders: delta.layer_encoders,
            layer_scale_encoder_fusions: delta.layer_scale_encoder_fusions,
            final_sample_encoders: delta.final_sample_encoders,
            final_logits_encoders: delta.final_logits_encoders,
            encoder_counts_by_kernel_family: encoder_counts_from_delta(&delta),
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
            claim: METAL_CLAIM,
        })
    }

    fn next_token_logits(&mut self, prefix: &[u32]) -> Result<Vec<f32>, String> {
        validate_token_budget(prefix.len(), 1, self.max_total_tokens)?;
        let prompt_tokens = prefix
            .iter()
            .copied()
            .map(rvllm_core::TokenId)
            .collect::<Vec<_>>();
        let _ = self.prefill(&prompt_tokens)?;
        let current = *prompt_tokens.last().expect("prefix token");
        let _ = self.decode_one(current, prompt_tokens.len(), 0)?;
        self.backend
            .probe_read_decode_logits_f32(1)
            .map_err(|err| format!("read Metal decode logits: {err}"))
    }

    fn prefill(&mut self, prompt_tokens: &[rvllm_core::TokenId]) -> Result<f64, String> {
        use rvllm_apple::{AppleBackend, HandoffCapsule, HandoffKind};
        if prompt_tokens.is_empty() {
            return Err("Metal prompt must contain at least one token".to_owned());
        }
        let prompt_len = prompt_tokens.len();
        let prefill = HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            prompt_tokens.to_vec(),
            vec![0, prompt_len as u32],
            vec![(prompt_len - 1) as u32],
            vec![prompt_len as u32],
        );
        let start = Instant::now();
        let ticket = self
            .backend
            .launch_prefill(&prefill)
            .map_err(|err| format!("launch Metal prefill: {err}"))?;
        let out = self
            .backend
            .collect(ticket)
            .map_err(|err| format!("collect Metal prefill: {err}"))?;
        if !out.is_empty() {
            return Err(format!(
                "Metal prefill unexpectedly returned {} sampled tokens",
                out.len()
            ));
        }
        Ok(ms(start.elapsed()))
    }

    fn decode_one(
        &mut self,
        current: rvllm_core::TokenId,
        prompt_len: usize,
        step_idx: usize,
    ) -> Result<u32, String> {
        use rvllm_apple::{AppleBackend, HandoffCapsule, HandoffKind};
        let decode = HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![current],
            vec![0, 1],
            vec![(prompt_len - 1 + step_idx) as u32],
            vec![(prompt_len + step_idx) as u32],
        );
        let ticket = self
            .backend
            .launch_rollout(&decode, None)
            .map_err(|err| format!("launch Metal decode step {step_idx}: {err}"))?;
        let out = self
            .backend
            .collect(ticket)
            .map_err(|err| format!("collect Metal decode step {step_idx}: {err}"))?;
        if out.len() != 1 {
            return Err(format!(
                "Metal decode step {step_idx} returned {} sampled tokens, expected 1",
                out.len()
            ));
        }
        Ok(out[0].token_id.raw())
    }

    fn arena_bytes(&self) -> usize {
        self.backend
            .probe_arena_stats()
            .map(|arena| arena.capacity_bytes)
            .unwrap_or(0)
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Debug)]
struct EnvGuard {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
    active: bool,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl EnvGuard {
    fn set_if(name: &'static str, enabled: bool) -> Self {
        let previous = std::env::var_os(name);
        if enabled {
            std::env::set_var(name, "1");
        }
        Self {
            name,
            previous,
            active: enabled,
        }
    }

    fn set_value_if(name: &'static str, value: Option<String>) -> Self {
        let previous = std::env::var_os(name);
        if let Some(value) = value {
            std::env::set_var(name, value);
            Self {
                name,
                previous,
                active: true,
            }
        } else {
            Self {
                name,
                previous,
                active: false,
            }
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl Drop for EnvGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(previous) = self.previous.as_ref() {
            std::env::set_var(self.name, previous);
        } else {
            std::env::remove_var(self.name);
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Copy, Clone, Debug, Default)]
struct MetalStatsDelta {
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
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn stats_delta(
    before: rvllm_runtime::apple_metal_backend::MetalProbePerfStats,
    after: rvllm_runtime::apple_metal_backend::MetalProbePerfStats,
) -> MetalStatsDelta {
    MetalStatsDelta {
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
fn encoder_counts_from_delta(delta: &MetalStatsDelta) -> MetalEncoderCounts {
    MetalEncoderCounts {
        embedding: delta.embedding_encoders,
        ple_input: delta.ple_encoders,
        layer_body: delta.layer_encoders,
        layer_scale_fused: delta.layer_scale_encoder_fusions,
        final_sample: delta.final_sample_encoders,
        final_logits_diagnostic: delta.final_logits_encoders,
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer, String> {
    let path = model_dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|err| format!("load tokenizer {}: {err}", path.display()))
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn tokenize_prompt(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    no_bos: bool,
) -> Result<Vec<u32>, String> {
    if text.is_empty() {
        return Err("Metal prompt/text must not be empty".to_owned());
    }
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|err| format!("tokenize Metal prompt: {err}"))?;
    let mut ids = Vec::new();
    if !no_bos {
        ids.push(2);
    }
    ids.extend(encoding.get_ids().iter().copied());
    if ids.is_empty() {
        return Err("Metal prompt produced no token IDs".to_owned());
    }
    Ok(ids)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn decode_text(
    tokenizer: &tokenizers::Tokenizer,
    ids: &[u32],
    label: &str,
) -> Result<String, String> {
    tokenizer
        .decode(ids, true)
        .map_err(|err| format!("decode {label} token IDs: {err}"))
}

fn validate_max_total_tokens(max_total_tokens: usize) -> Result<(), String> {
    if max_total_tokens == 0 {
        return Err("Metal max_total_tokens must be positive".to_owned());
    }
    if u32::try_from(max_total_tokens).is_err() {
        return Err(format!(
            "Metal max_total_tokens must be at most {}",
            u32::MAX
        ));
    }
    Ok(())
}

fn validate_max_new_tokens(max_new_tokens: usize) -> Result<(), String> {
    if max_new_tokens == 0 {
        return Err("Metal max_new_tokens must be positive".to_owned());
    }
    if u32::try_from(max_new_tokens).is_err() {
        return Err(format!("Metal max_new_tokens must be at most {}", u32::MAX));
    }
    Ok(())
}

fn validate_token_budget(
    prompt_tokens: usize,
    max_new_tokens: usize,
    max_total_tokens: usize,
) -> Result<(), String> {
    if prompt_tokens == 0 {
        return Err("Metal prompt must contain at least one token".to_owned());
    }
    if max_new_tokens == 0 {
        return Err("Metal max_new_tokens must be positive".to_owned());
    }
    let total = prompt_tokens
        .checked_add(max_new_tokens)
        .ok_or_else(|| "Metal token budget overflow".to_owned())?;
    if u32::try_from(total).is_err() {
        return Err(format!(
            "Metal total token count must be at most {}",
            u32::MAX
        ));
    }
    if total > max_total_tokens {
        return Err(format!(
            "Metal token budget exceeded: prompt_tokens={prompt_tokens} max_new_tokens={max_new_tokens} total={total} max_total_tokens={max_total_tokens}"
        ));
    }
    Ok(())
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn negative_log_likelihood(logits: &[f32], target: usize) -> f64 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exp_sum = logits
        .iter()
        .map(|&logit| ((logit as f64) - max_logit).exp())
        .sum::<f64>();
    max_logit + exp_sum.ln() - logits[target] as f64
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

pub fn env_bool(name: &str) -> bool {
    env_truthy(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_token_budget_enforces_configured_limit() {
        validate_token_budget(2, 1, 3).expect("budget fits");

        let err = validate_token_budget(2, 2, 3).expect_err("budget should exceed cap");
        assert!(err.contains("Metal token budget exceeded"));

        validate_max_total_tokens(65).expect("large configured limit is allowed");
        let err = validate_max_total_tokens(0).expect_err("zero max total should fail");
        assert!(err.contains("Metal max_total_tokens must be positive"));
    }

    #[test]
    fn metal_token_budget_rejects_overflow_and_position_narrowing() {
        let overflow =
            validate_token_budget(usize::MAX, 1, usize::MAX).expect_err("usize overflow must fail");
        assert!(overflow.contains("overflow"));

        if usize::BITS > 32 {
            let too_large = u32::MAX as usize + 1;
            let narrowing = validate_token_budget(too_large - 1, 1, too_large)
                .expect_err("u32 position narrowing must fail");
            assert!(narrowing.contains("at most"));
        }
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    fn negative_log_likelihood_uses_log_softmax() {
        let logits = [1.0, 2.0, 3.0];
        let nll = negative_log_likelihood(&logits, 2);
        let expected = ((1.0f64 - 3.0).exp() + (2.0f64 - 3.0).exp() + 1.0).ln();
        assert!((nll - expected).abs() < 1e-9);
    }
}
