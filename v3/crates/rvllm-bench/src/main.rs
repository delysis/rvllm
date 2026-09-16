//! rvllm-bench: loads a model + kernels + cutlass .so + fa3 .so, runs
//! `iters` decode-step forwards on a fixed-batch bucket, and reports
//! tokens/sec.
//!
//! Env vars:
//!   RVLLM_MODEL_DIR             = HF snapshot dir with config.json + safetensors (required)
//!   RVLLM_KERNELS_DIR           = dir with manifest.json + compiled PTX         (required)
//!   RVLLM_CUTLASS_SO            = path to libcutlass_kernels.so                 (SM90 only)
//!   RVLLM_FA3_SO                = path to libfa3_kernels.so                     (SM90 only)
//!   RVLLM_POLICY                = path to policy.json                           (SM90 only)
//!   RVLLM_BATCH                 = batch size (default 128)
//!   RVLLM_ITERS                 = decode-step iterations (default 100)
//!   RVLLM_WARMUP                = warmup iterations (default 10)
//!   RVLLM_BENCH_PROMPT_CACHE    = 1 enables production T1/T2 prompt reuse during Apple bench
//!   RVLLM_APPLE_CACHE_GATE      = 1 runs the exact cold/T1/T2 512-token promotion gate
//!   RVLLM_SWEEP                 = if 1, sample a policy parameter sweep
//!   RVLLM_BACKEND_PROFILE       = cuda|apple|xla|unknown (default: cuda)
//!   RVLLM_APPLE_MODE            = disabled|metal-only|metal-prefill-metal-decode|ane-fn|ane-exp
//!   RVLLM_STRICT_ANE            = 1 enables strict fail-fast ANE mode
//!   RVLLM_APPLE_PRIVATE_ANE     = 1 opt-in for private ANE plan path
//!   RVLLM_APPLE_ANE_PROFILE     = any|neural_engine_preferred|neural_engine_only
//!   RVLLM_APPLE_ANE_FALLBACK    = allow-metal|allow-soft|failfast
//!   RVLLM_APPLE_ROLLOUT_TOKENS   = rollout token target (default 1)
//!   RVLLM_APPLE_BUCKET_SEQS      = rollout bucket sequence size (metadata only)
//!   RVLLM_APPLE_BUCKET_TOKENS    = rollout bucket token size (metadata only)
//!   RVLLM_APPLE_LAYOUT_HASH      = optional model layout hash (opaque)
//!   RVLLM_BENCH_LOG_DIR          = directory to append JSON benchmark records
//!   RVLLM_APPLE_COMPILE_CACHE_KEY = optional cache key used by your Apple run
//!   RVLLM_APPLE_COMPILE_CACHE_HIT = 1/0 cache hit/miss marker for Apple
//!   RVLLM_APPLE_COMPILE_MS        = compile wall time in ms for Apple
//!   RVLLM_APPLE_COMPILE_REASON     = optional compile fail reason
//!   RVLLM_SIDE_BY_SIDE_CUDA       = one-line JSON with baseline CUDA metrics
//!   RVLLM_SIDE_BY_SIDE_XLA        = one-line JSON with baseline XLA metrics
//!
//! Prints JSON records with enriched metadata:
//!   {batch,iters,tok_per_sec,ms_per_step,[ttft],backend,backend_profile,side_by_side}

#[cfg(any(feature = "apple", feature = "cuda"))]
use std::fs::OpenOptions;
#[cfg(any(feature = "apple", feature = "cuda"))]
use std::io::Write;
#[cfg(any(feature = "apple", feature = "cuda"))]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use rvllm_core::{AneFallbackPolicy, ModelArch as HfModelArch, ModelConfig};
#[cfg(feature = "cuda")]
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
#[cfg(feature = "cuda")]
use rvllm_runtime::{Bringup, EnginePaths};
#[cfg(any(feature = "apple", feature = "cuda"))]
use serde_json::{json, Value};

use rvllm_bench::ane_meta::{AppleCliProfile, BackendProfile};

#[cfg(any(feature = "apple", feature = "cuda"))]
fn env_path(k: &str) -> Result<PathBuf, String> {
    std::env::var(k)
        .map_err(|_| format!("missing env var: {k}"))
        .map(PathBuf::from)
}

/// Optional env var: returns `/dev/null` when missing. Used for paths
/// that the sm_121 backend never opens. On SM90 an unset value will
/// surface as a clean dlopen error for `/dev/null`.
#[cfg(feature = "cuda")]
fn env_path_or_placeholder(k: &str) -> PathBuf {
    std::env::var(k)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/dev/null"))
}

#[cfg(any(feature = "apple", feature = "cuda"))]
fn env_u32(k: &str, default: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
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

#[cfg(feature = "cuda")]
fn ane_policy_label(policy: AneFallbackPolicy) -> &'static str {
    match policy {
        AneFallbackPolicy::FailFast => "failfast",
        AneFallbackPolicy::AllowMetal => "allow-metal",
        AneFallbackPolicy::AllowSoft => "allow-soft",
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let result = run();
    if let Err(e) = result {
        eprintln!("rvllm-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let profile = AppleCliProfile::from_env();
    if matches!(profile.backend(), BackendProfile::Apple) {
        return run_apple_bench(&profile);
    }

    run_cuda_bench(profile)
}

#[cfg(feature = "apple")]
fn run_apple_bench(profile: &AppleCliProfile) -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let prompt = std::env::var("RVLLM_PROMPT").unwrap_or_else(|_| "Hello".to_owned());
    let cache_gate = rvllm_bench::apple_metal_text::env_bool("RVLLM_APPLE_CACHE_GATE");
    let max_new_tokens = env_u32("RVLLM_MAX_TOKENS", if cache_gate { 32 } else { 1 }) as usize;
    let max_total_tokens = rvllm_bench::apple_metal_text::env_usize(
        "RVLLM_METAL_MAX_TOTAL_TOKENS",
        rvllm_bench::apple_metal_text::env_usize("RVLLM_METAL_MAX_PROBE_TOKENS", 2048),
    );
    let batch = env_u32("RVLLM_BATCH", 1);
    let iters = env_u32("RVLLM_ITERS", 1);
    let warmup = env_u32("RVLLM_WARMUP", 0);
    let text_options = rvllm_bench::apple_metal_text::MetalTextOptions {
        model_dir,
        prompt,
        max_new_tokens,
        max_total_tokens,
        max_batch_tokens: 0,
        no_bos: rvllm_bench::apple_metal_text::env_bool("RVLLM_NO_BOS"),
        eos_token_ids: vec![1, 2, 107],
        large_model_opt_in: rvllm_bench::apple_metal_text::env_bool(
            "RVLLM_APPLE_LARGE_MODEL_OPT_IN",
        ),
    };
    let mut record = if cache_gate {
        rvllm_bench::apple_metal_text::run_cache_gate(
            rvllm_bench::apple_metal_text::MetalCacheGateOptions { text: text_options },
        )?
    } else {
        let options = rvllm_bench::apple_metal_text::MetalBenchOptions {
            text: text_options,
            batch,
            iters,
            warmup,
        };
        rvllm_bench::apple_metal_text::run_bench(options)?
    };
    if !cache_gate {
        let apple_current = record.clone();
        if let Some(object) = record.as_object_mut() {
            object.insert(
                "side_by_side".to_owned(),
                json!({
                    "cuda": profile.peer_cuda.clone().unwrap_or(Value::Null),
                    "apple": apple_current,
                    "xla": profile.peer_xla.clone().unwrap_or(Value::Null),
                }),
            );
        }
    }
    if let Some(log_dir) = profile.log_dir.clone() {
        let mut log = log_dir;
        log.push("rvllm_bench_records.jsonl");
        append_record_to_file(log, &record)?;
    }
    println!(
        "{}",
        serde_json::to_string(&record).map_err(|e| format!("serialize json: {e}"))?
    );
    Ok(())
}

#[cfg(not(feature = "apple"))]
fn run_apple_bench(_profile: &AppleCliProfile) -> Result<(), String> {
    Err(
        "RVLLM_BACKEND_PROFILE=apple requires building rvllm-bench with --features apple on macOS"
            .to_owned(),
    )
}

#[cfg(feature = "cuda")]
fn run_cuda_bench(profile: AppleCliProfile) -> Result<(), String> {
    let paths = EnginePaths {
        model_dir: env_path("RVLLM_MODEL_DIR")?,
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path_or_placeholder("RVLLM_CUTLASS_SO"),
        fa3_so: env_path_or_placeholder("RVLLM_FA3_SO"),
        policy_json: env_path_or_placeholder("RVLLM_POLICY"),
    };
    let batch = env_u32("RVLLM_BATCH", 128);
    let iters = env_u32("RVLLM_ITERS", 100);
    let warmup = env_u32("RVLLM_WARMUP", 10);

    // Arena budget: model (~16 GB fp8) + kv (~8 GB) + scratch/workspace (~4 GB).
    // Override with RVLLM_ARENA_GB if GPU memory is constrained.
    let arena_gb: usize = std::env::var("RVLLM_ARENA_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let arena_bytes: usize = arena_gb * 1024 * 1024 * 1024;

    eprintln!("== rvllm-bench v3 ==");
    eprintln!("model_dir    = {}", paths.model_dir.display());
    eprintln!("kernels_dir  = {}", paths.kernels_dir.display());
    eprintln!("batch       = {batch}");
    eprintln!("iters       = {iters} (warmup {warmup})");
    eprintln!("apple_intent= {}", profile.compact_summary());

    let is_gemma4 = is_gemma4_model_dir(&paths.model_dir)?;

    if is_gemma4 {
        eprintln!("== Gemma 4 detected, using Gemma4Bringup ==");
        let g4_paths = Gemma4EnginePaths {
            model_dir: paths.model_dir,
            kernels_dir: paths.kernels_dir,
            cutlass_so: paths.cutlass_so,
            fa3_so: paths.fa3_so,
            policy_json: paths.policy_json,
        };
        let t0 = Instant::now();
        let g4 = Gemma4Bringup::load(g4_paths, arena_bytes)
            .map_err(|e| format!("gemma4 bringup: {e}"))?;
        let load_ms = t0.elapsed().as_millis();
        eprintln!(
            "bringup: {:.2}s | arch layers={} hidden={} heads={} sliding_kv={} global_kv={}",
            (load_ms as f64) / 1000.0,
            g4.arch.num_hidden_layers,
            g4.arch.hidden_size,
            g4.arch.num_attention_heads,
            g4.arch.num_kv_heads_sliding,
            g4.arch.num_kv_heads_global,
        );
        eprintln!("arena used = {} MiB", g4.arena.used() / (1024 * 1024));
        let result = unsafe { g4.run_bench(batch, iters, warmup) };
        return print_result(&profile, result, load_ms, true);
    }

    let t0 = Instant::now();
    let br = Bringup::load(paths, arena_bytes).map_err(|e| format!("bringup: {e}"))?;
    let load_ms = t0.elapsed().as_millis();
    eprintln!(
        "bringup: {:.2}s | arch layers={} hidden={} heads={} kv_heads={}",
        (load_ms as f64) / 1000.0,
        br.arch.num_hidden_layers,
        br.arch.hidden_size,
        br.arch.num_attention_heads,
        br.arch.num_key_value_heads,
    );
    eprintln!("arena used = {} MiB", br.arena.used() / (1024 * 1024));

    if std::env::var("RVLLM_SWEEP").ok().as_deref() == Some("1") {
        return run_sweep(&br, batch, iters, warmup);
    }

    let result =
        unsafe { br.run_bench(batch, iters, warmup) }.map_err(|e| format!("run_bench: {e}"))?;
    print_result(&profile, result, load_ms, false)
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_bench(_profile: AppleCliProfile) -> Result<(), String> {
    Err(
        "rvllm-bench CUDA path requires --features cuda; set RVLLM_BACKEND_PROFILE=apple and build with --features apple for Metal"
            .to_owned(),
    )
}

#[cfg(feature = "cuda")]
fn print_result(
    profile: &AppleCliProfile,
    r: rvllm_runtime::bring_up::BenchResult,
    load_ms: u128,
    is_gemma4: bool,
) -> Result<(), String> {
    let tok_per_sec = if r.total_ns > 0 {
        (r.iters as f64 * r.num_seqs as f64) * 1.0e9 / r.total_ns as f64
    } else {
        0.0
    };
    let ms_per_step = r.ns_per_step as f64 / 1.0e6;
    let ttft_str = match (r.ttft_ns, r.ttft_hot_ns) {
        (Some(cold), Some(hot)) => format!(
            " ttft_cold={:.2}ms ttft_hot={:.2}ms",
            cold as f64 / 1.0e6,
            hot as f64 / 1.0e6
        ),
        (Some(cold), None) => format!(" ttft={:.2}ms", cold as f64 / 1.0e6),
        _ => String::new(),
    };
    eprintln!(
        "bench: batch={} iters={} -> {:.0} tok/s ({:.3} ms/step){}",
        r.num_seqs, r.iters, tok_per_sec, ms_per_step, ttft_str
    );

    let current = backend_entry(profile.backend(), tok_per_sec, ms_per_step, load_ms, &r);

    let side_by_side = json!({
        "cuda": if matches!(profile.backend(), BackendProfile::Cuda) {
            current.clone()
        } else {
            profile.peer_cuda.clone().unwrap_or(Value::Null)
        },
        "apple": if matches!(profile.backend(), BackendProfile::Apple) {
            profile.compact_apple_object()
        } else {
            Value::Null
        },
        "xla": if matches!(profile.backend(), BackendProfile::Xla) {
            current.clone()
        } else {
            profile.peer_xla.clone().unwrap_or(Value::Null)
        },
    });

    let mut record = json!({
        "batch": r.num_seqs,
        "iters": r.iters,
        "tok_per_sec": tok_per_sec,
        "ms_per_step": ms_per_step,
        "ttft_cold_ms": r.ttft_ns.map(|ns| ns as f64 / 1.0e6),
        "ttft_hot_ms": r.ttft_hot_ns.map(|ns| ns as f64 / 1.0e6),
        "backend": profile.backend().to_string(),
        "backend_profile": profile.backend().to_string(),
        "compile_ns": load_ms * 1_000_000,
        "compile_ms": load_ms as f64,
        "load_plan": {
            "mode": profile.apple_mode_label(),
            "strict_ane": profile.strict_ane,
            "is_strict_ane_mode": profile.is_strict_ane_mode(),
            "private_ane_opt_in": profile.private_ane_opt_in,
            "ane_compute_profile": profile.ane_compute_profile.as_str(),
            "ane_fallback_policy": ane_policy_label(profile.ane_fallback_policy),
            "apple_rollout_tokens": profile.apple_rollout_tokens,
            "apple_rollout_bucket": profile
                .rollout_bucket_seqs
                .zip(profile.rollout_bucket_tokens)
                .map(|(seqs, tokens)| json!({ "seqs": seqs, "tokens": tokens })),
            "model_layout_hash": profile.model_layout_hash,
            "compile_cache_key": profile.compile_cache_key,
            "compile_cache_hit": profile.compile_cache_hit,
            "compile_ms_override": profile.compile_ms,
            "compile_reason": profile.compile_reason,
        },
        "is_gemma4": is_gemma4,
        "side_by_side": side_by_side,
    });
    if let Some(v) = profile.log_dir.clone() {
        let mut log = v.clone();
        log.push("rvllm_bench_records.jsonl");
        append_record_to_file(log, &record)?;
    }
    println!(
        "{}",
        serde_json::to_string(&record).map_err(|e| format!("serialize json: {e}"))?
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn backend_entry(
    backend: BackendProfile,
    tok_per_sec: f64,
    ms_per_step: f64,
    load_ms: u128,
    r: &rvllm_runtime::bring_up::BenchResult,
) -> Value {
    json!({
        "backend": backend.to_string(),
        "enabled": true,
        "compile_ns": load_ms * 1_000_000,
        "compile_ms": load_ms as f64,
        "iter_ns": r.total_ns,
        "iters": r.iters,
        "batch": r.num_seqs,
        "tok_per_sec": tok_per_sec,
        "ms_per_step": ms_per_step,
        "ttft_ns": r.ttft_ns,
        "ttft_hot_ns": r.ttft_hot_ns,
    })
}

#[cfg(any(feature = "apple", feature = "cuda"))]
fn append_record_to_file(path: PathBuf, record: &Value) -> Result<(), String> {
    let dir = path
        .parent()
        .ok_or_else(|| "invalid RVLLM_BENCH_LOG_DIR".to_string())?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create log dir {}: {e}", dir.display()))?;
    let mut fp = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("open log file {}: {e}", path.display()))?;
    let line = serde_json::to_string(record).map_err(|e| format!("serialize json: {e}"))?;
    fp.write_all(line.as_bytes())
        .and_then(|_| fp.write_all(b"\n"))
        .map_err(|e| format!("write log file {}: {e}", path.display()))
}

#[cfg(feature = "cuda")]
fn run_sweep(br: &Bringup, batch: u32, iters: u32, warmup: u32) -> Result<(), String> {
    // Variant grid. Policy knows 40 non-residual + 10 residual (per the
    // autotune .so). Sample a promising subset.
    let nonres: &[u32] = &[0, 2, 5, 8, 10, 12, 14];
    let residuals: &[u32] = &[100, 102, 105, 108];

    let mut best = (u128::MAX, 0u32, 0u32);
    eprintln!("== sweep @ N={batch} ==");
    for &nr in nonres {
        for &r in residuals {
            let ck = br.arena.checkpoint();
            let res =
                unsafe { br.run_bench_with_variants(batch, iters, warmup, Some(nr), Some(r)) };
            unsafe { br.arena.restore(ck) };
            match res {
                Ok(r_) => {
                    let tok_per_sec = if r_.total_ns > 0 {
                        (r_.iters as f64 * r_.num_seqs as f64) * 1.0e9 / r_.total_ns as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "nonres={nr} res={r} -> {:.0} tok/s ({:.3} ms/step)",
                        tok_per_sec,
                        r_.ns_per_step as f64 / 1.0e6
                    );
                    println!(
                        "{{\"nonres\":{nr},\"res\":{r},\"tok_per_sec\":{:.1}}}",
                        tok_per_sec
                    );
                    if r_.ns_per_step < best.0 {
                        best = (r_.ns_per_step, nr, r);
                    }
                }
                Err(e) => {
                    eprintln!("nonres={nr} res={r} -> ERROR: {e}");
                }
            }
        }
    }
    eprintln!(
        "BEST: nonres={} res={} ({:.3} ms/step)",
        best.1,
        best.2,
        best.0 as f64 / 1.0e6
    );
    Ok(())
}
