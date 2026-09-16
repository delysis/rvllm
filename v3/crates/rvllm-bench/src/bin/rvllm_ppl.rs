//! rvllm-ppl: perplexity measurement via full-logits readback.
//!
//! Processes tokens one at a time in decode mode, DtoH-ing one logits
//! row per step. Delegates to Bringup::run_ppl which uses the exact
//! same arena layout as the bench to avoid FA3 workspace issues.
//!
//! Env vars (same as rvllm-bench plus):
//!   RVLLM_MODEL_DIR   = HF snapshot dir (must contain tokenizer.json)
//!   RVLLM_KERNELS_DIR, RVLLM_CUTLASS_SO, RVLLM_FA3_SO, RVLLM_POLICY
//!   RVLLM_ARENA_GB    = arena size in GB (default 32)
//!   RVLLM_PPL_CHUNK   = chunk length in tokens (default 128)
//!   RVLLM_PPL_CHUNKS  = max number of chunks (default 0 = all)
//!   RVLLM_PPL_TEXT    = path to plain-text file to evaluate
//!   RVLLM_PROMPT      = inline text (alternative to file)

#[cfg(any(feature = "apple", feature = "cuda"))]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

use rvllm_bench::ane_meta::{AppleCliProfile, BackendProfile};
#[cfg(feature = "cuda")]
use rvllm_core::{ModelArch as HfModelArch, ModelConfig};
#[cfg(feature = "cuda")]
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
#[cfg(feature = "cuda")]
use rvllm_runtime::{Bringup, EnginePaths};

#[cfg(any(feature = "apple", feature = "cuda"))]
fn env_path(k: &str) -> Result<PathBuf, String> {
    std::env::var(k)
        .map_err(|_| format!("missing env var: {k}"))
        .map(PathBuf::from)
}

/// Optional env var: returns `/dev/null` when missing. For paths that
/// the sm_121 backend never opens (SM90-only .so files + policy).
#[cfg(feature = "cuda")]
fn env_path_or_placeholder(k: &str) -> PathBuf {
    std::env::var(k)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/dev/null"))
}

#[cfg(feature = "cuda")]
fn is_gemma4_model_dir(model_dir: &std::path::Path) -> Result<bool, String> {
    Ok(matches!(
        ModelConfig::load_hf(model_dir)
            .map_err(|e| format!("config parse {}: {e}", model_dir.display()))?
            .architecture,
        HfModelArch::Gemma4
    ))
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    if let Err(e) = run() {
        eprintln!("rvllm-ppl: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let profile = AppleCliProfile::from_env();
    if matches!(profile.backend(), BackendProfile::Apple) {
        return run_apple_ppl();
    }
    run_cuda_ppl()
}

#[cfg(feature = "apple")]
fn run_apple_ppl() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let text = read_eval_text()?;
    let chunk_len = std::env::var("RVLLM_PPL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let max_chunks = std::env::var("RVLLM_PPL_CHUNKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let max_total_tokens = rvllm_bench::apple_metal_text::env_usize(
        "RVLLM_METAL_MAX_TOTAL_TOKENS",
        rvllm_bench::apple_metal_text::env_usize("RVLLM_METAL_MAX_PROBE_TOKENS", 2048),
    );
    let report =
        rvllm_bench::apple_metal_text::run_ppl(rvllm_bench::apple_metal_text::MetalPplOptions {
            model_dir,
            text,
            chunk_len,
            max_chunks,
            no_bos: rvllm_bench::apple_metal_text::env_bool("RVLLM_NO_BOS"),
            max_total_tokens,
            large_model_opt_in: rvllm_bench::apple_metal_text::env_bool(
                "RVLLM_APPLE_LARGE_MODEL_OPT_IN",
            ),
        })?;
    eprintln!(
        "metal-ppl: perplexity={:.4} tokens={} elapsed_s={:.1} prepare_ms={:.1}",
        report.perplexity, report.tokens, report.elapsed_s, report.prepare_ms
    );
    eprintln!("{}", report.claim);
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|e| format!("serialize json: {e}"))?
    );
    Ok(())
}

#[cfg(not(feature = "apple"))]
fn run_apple_ppl() -> Result<(), String> {
    Err(
        "RVLLM_BACKEND_PROFILE=apple requires building rvllm-ppl with --features apple on macOS"
            .to_owned(),
    )
}

#[cfg(feature = "cuda")]
fn run_cuda_ppl() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| format!("tokenizer load {}: {e}", tok_path.display()))?;

    let text = read_eval_text()?;
    if text.is_empty() {
        return Err("empty text (set RVLLM_PPL_TEXT or RVLLM_PROMPT)".into());
    }

    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| format!("tokenize: {e}"))?;
    let all_ids: Vec<u32> = std::iter::once(2u32) // BOS
        .chain(encoding.get_ids().iter().copied())
        .collect();
    eprintln!("total tokens: {}", all_ids.len());

    let chunk_len: usize = std::env::var("RVLLM_PPL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let max_chunks: usize = std::env::var("RVLLM_PPL_CHUNKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let paths = EnginePaths {
        model_dir: model_dir.clone(),
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path_or_placeholder("RVLLM_CUTLASS_SO"),
        fa3_so: env_path_or_placeholder("RVLLM_FA3_SO"),
        policy_json: env_path_or_placeholder("RVLLM_POLICY"),
    };
    let arena_gb: usize = std::env::var("RVLLM_ARENA_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let arena_bytes: usize = arena_gb * 1024 * 1024 * 1024;
    let t0 = Instant::now();

    if is_gemma4_model_dir(&model_dir)? {
        let g4_paths = Gemma4EnginePaths {
            model_dir: paths.model_dir,
            kernels_dir: paths.kernels_dir.clone(),
            cutlass_so: paths.cutlass_so,
            fa3_so: paths.fa3_so,
            policy_json: paths.policy_json,
        };
        let g4 = Gemma4Bringup::load(g4_paths, arena_bytes)
            .map_err(|e| format!("gemma4 bringup: {e}"))?;
        eprintln!("bringup: {:.2}s", t0.elapsed().as_secs_f64());

        let embed_mod = g4
            .kernels
            .load_ptx("embedding_gather_f16")
            .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
        let fn_embed = embed_mod
            .get_function("embedding_gather_f16_kernel")
            .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;

        let mut chunks: Vec<&[u32]> = all_ids.chunks(chunk_len).collect();
        if let Some(last) = chunks.last() {
            if last.len() < chunk_len {
                chunks.pop();
            }
        }
        if max_chunks > 0 && chunks.len() > max_chunks {
            chunks.truncate(max_chunks);
        }
        if chunks.is_empty() {
            return Err(format!(
                "not enough tokens ({}) for chunk_len={chunk_len}",
                all_ids.len()
            ));
        }
        eprintln!("evaluating {} chunks of {} tokens", chunks.len(), chunk_len);

        let mut total_nll: f64 = 0.0;
        let mut total_tokens: usize = 0;
        let t_eval = Instant::now();

        for (ci, chunk) in chunks.iter().enumerate() {
            let chunk_t0 = Instant::now();
            let result = unsafe { g4.run_ppl(fn_embed, chunk) }
                .map_err(|e| format!("run_ppl chunk {ci}: {e}"))?;
            total_nll += result.total_nll;
            total_tokens += result.n_evaluated;
            let running_ppl = (total_nll / total_tokens as f64).exp();
            let chunk_ppl = if result.n_evaluated > 0 {
                (result.total_nll / result.n_evaluated as f64).exp()
            } else {
                0.0
            };
            eprintln!(
                "chunk {}/{}: chunk_ppl={:.4} running_ppl={:.4} ({:.1} tok/s, {:.1}s)",
                ci + 1,
                chunks.len(),
                chunk_ppl,
                running_ppl,
                chunk.len() as f64 / chunk_t0.elapsed().as_secs_f64(),
                chunk_t0.elapsed().as_secs_f64()
            );
        }

        let ppl = (total_nll / total_tokens as f64).exp();
        let elapsed = t_eval.elapsed().as_secs_f64();
        eprintln!("perplexity = {ppl:.4} ({total_tokens} tokens, {elapsed:.1}s)");
        println!("{{\"perplexity\":{ppl:.4},\"tokens\":{total_tokens},\"chunk_len\":{chunk_len},\"elapsed_s\":{elapsed:.1}}}");
        return Ok(());
    }

    let br = Bringup::load(paths, arena_bytes).map_err(|e| format!("bringup: {e}"))?;
    eprintln!("bringup: {:.2}s", t0.elapsed().as_secs_f64());

    let embed_mod = br
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
    let fn_embed = embed_mod
        .get_function("embedding_gather_f16_kernel")
        .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;

    // Chunk the input.
    let mut chunks: Vec<&[u32]> = all_ids.chunks(chunk_len).collect();
    if let Some(last) = chunks.last() {
        if last.len() < chunk_len {
            chunks.pop();
        }
    }
    if max_chunks > 0 && chunks.len() > max_chunks {
        chunks.truncate(max_chunks);
    }
    if chunks.is_empty() {
        return Err(format!(
            "not enough tokens ({}) for chunk_len={chunk_len}",
            all_ids.len()
        ));
    }
    eprintln!(
        "evaluating {} chunks of {} tokens ({} total)",
        chunks.len(),
        chunk_len,
        chunks.len() * chunk_len
    );

    let mut total_nll: f64 = 0.0;
    let mut total_tokens: usize = 0;
    let t_eval = Instant::now();

    for (ci, chunk) in chunks.iter().enumerate() {
        let chunk_t0 = Instant::now();
        let result = unsafe { br.run_ppl(fn_embed, chunk) }
            .map_err(|e| format!("run_ppl chunk {ci}: {e}"))?;
        total_nll += result.total_nll;
        total_tokens += result.n_evaluated;
        let running_ppl = (total_nll / total_tokens as f64).exp();
        let chunk_ppl = if result.n_evaluated > 0 {
            (result.total_nll / result.n_evaluated as f64).exp()
        } else {
            0.0
        };
        let chunk_elapsed = chunk_t0.elapsed().as_secs_f64();
        eprintln!(
            "chunk {}/{}: chunk_ppl={:.4} running_ppl={:.4} ({:.1} tok/s, {:.1}s)",
            ci + 1,
            chunks.len(),
            chunk_ppl,
            running_ppl,
            chunk.len() as f64 / chunk_elapsed,
            chunk_elapsed
        );
    }

    let ppl = (total_nll / total_tokens as f64).exp();
    let elapsed = t_eval.elapsed().as_secs_f64();
    eprintln!("perplexity = {ppl:.4} ({total_tokens} tokens, {elapsed:.1}s)");
    println!(
        "{{\"perplexity\":{ppl:.4},\"tokens\":{total_tokens},\"chunk_len\":{chunk_len},\"elapsed_s\":{elapsed:.1}}}"
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_ppl() -> Result<(), String> {
    Err(
        "rvllm-ppl CUDA path requires --features cuda; set RVLLM_BACKEND_PROFILE=apple and build with --features apple for Metal"
            .to_owned(),
    )
}

#[cfg(any(feature = "apple", feature = "cuda"))]
fn read_eval_text() -> Result<String, String> {
    let text = if let Ok(path) = std::env::var("RVLLM_PPL_TEXT") {
        std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?
    } else if let Ok(p) = std::env::var("RVLLM_PROMPT") {
        p
    } else {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
            .map_err(|e| format!("stdin: {e}"))?;
        buf
    };
    if text.is_empty() {
        return Err("empty text (set RVLLM_PPL_TEXT or RVLLM_PROMPT)".into());
    }
    Ok(text)
}
