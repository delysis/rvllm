//! Development-only promotion gate for the production prompt-cache path.

use rvllm_core::TokenId;
use rvllm_runtime::{
    BackendReport, CacheTier, EngineHandle, GenerateRequest, TokenEvent, PROMPT_CACHE_PAGE_TOKENS,
};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const SCHEMA: &str = "rvllm.apple_prompt_cache_gate.v1";
pub const REQUIRED_REUSABLE_TOKENS: usize = 512;
pub const REQUIRED_SKIP_FRACTION: f64 = 0.90;
pub const REQUIRED_TTFT_SPEEDUP: f64 = 2.0;
const CONTINUATION_PROBE_TOKENS: usize = PROMPT_CACHE_PAGE_TOKENS;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExpectedCacheTier {
    Hot,
    WarmOrRecompute,
}

#[derive(Clone, Debug)]
pub struct PromptCacheGateConfig {
    pub prompt_tokens: Vec<TokenId>,
    pub max_output_tokens: u32,
    pub expected_tier: ExpectedCacheTier,
}

#[derive(Debug)]
struct Observation {
    ttft: Duration,
    tokens: Vec<TokenId>,
    report: BackendReport,
}

/// Runs one independent cold request followed by an exact-prefix request on
/// the same production worker. The returned record is deliberately
/// self-gating: a caller must not promote the cache path unless `passed` is
/// true. T2 is allowed to recompute when its measured restore estimate is not
/// favorable; that outcome is reported but is never described as a hit.
pub fn run_prompt_cache_gate(
    engine: &EngineHandle,
    config: PromptCacheGateConfig,
) -> Result<Value, String> {
    validate(&config)?;
    let reusable = rvllm_runtime::reusable_prompt_tokens(config.prompt_tokens.len());
    let request = GenerateRequest::new(config.prompt_tokens.clone(), config.max_output_tokens);

    let cold = run_one(engine, request.clone())?;
    if cold.report.cache_tier != CacheTier::None
        || cold.report.matched_cache_tokens != 0
        || cold.report.saved_prefill_tokens != 0
    {
        return Err(format!(
            "cold cache-gate request was not independent: tier={:?}, matched={}, saved={}",
            cold.report.cache_tier,
            cold.report.matched_cache_tokens,
            cold.report.saved_prefill_tokens
        ));
    }

    let hit = run_one(engine, request)?;
    if cold.tokens != hit.tokens {
        return Err(format!(
            "cache result diverged from independent cold execution: cold={:?}, cached={:?}",
            raw_tokens(&cold.tokens),
            raw_tokens(&hit.tokens)
        ));
    }

    // A full generated page is deliberately appended to the original prompt.
    // If continuation KV had been indexed, this exact probe could match more
    // than the original 512-token prompt prefix. The sentinel makes that
    // generated page reusable while preserving a private writable tail.
    if cold.tokens.len() < CONTINUATION_PROBE_TOKENS {
        return Err(format!(
            "cache-gate generation ended after {} tokens; need {CONTINUATION_PROBE_TOKENS} to verify continuation isolation",
            cold.tokens.len()
        ));
    }
    let mut continuation_prompt = config.prompt_tokens.clone();
    continuation_prompt.extend_from_slice(&cold.tokens[..CONTINUATION_PROBE_TOKENS]);
    continuation_prompt.push(
        config
            .prompt_tokens
            .last()
            .copied()
            .expect("validated non-empty prompt"),
    );
    let continuation_reusable = rvllm_runtime::reusable_prompt_tokens(continuation_prompt.len());
    let continuation_probe = run_one(engine, GenerateRequest::new(continuation_prompt, 1))?;
    let continuation_matched = continuation_probe.report.matched_cache_tokens as usize;
    if continuation_matched > reusable {
        return Err(format!(
            "generated continuation KV entered the prompt cache: base reusable={reusable}, probe reusable={continuation_reusable}, matched={continuation_matched}"
        ));
    }

    let restored = match config.expected_tier {
        ExpectedCacheTier::Hot => {
            if hit.report.cache_tier != CacheTier::Hot {
                return Err(format!(
                    "T1 promotion did not produce a hot hit: got {:?}",
                    hit.report.cache_tier
                ));
            }
            true
        }
        ExpectedCacheTier::WarmOrRecompute => match hit.report.cache_tier {
            CacheTier::Warm => true,
            CacheTier::None => {
                if hit.report.matched_cache_tokens != 0 || hit.report.saved_prefill_tokens != 0 {
                    return Err(
                        "T2 recompute reported saved work without restoring exact pages".to_owned(),
                    );
                }
                false
            }
            other => {
                return Err(format!(
                    "warm-only gate selected an impossible cache tier: {other:?}"
                ));
            }
        },
    };

    let matched = usize::try_from(hit.report.matched_cache_tokens)
        .map_err(|_| "matched token count does not fit usize".to_owned())?;
    let saved = usize::try_from(hit.report.saved_prefill_tokens)
        .map_err(|_| "saved token count does not fit usize".to_owned())?;
    if restored && (matched != reusable || saved != reusable) {
        return Err(format!(
            "cache hit did not save the exact reusable prefix: reusable={reusable}, matched={matched}, saved={saved}"
        ));
    }
    let skip_fraction = if reusable == 0 {
        0.0
    } else {
        saved as f64 / reusable as f64
    };
    let speedup = if hit.ttft.is_zero() {
        f64::INFINITY
    } else {
        cold.ttft.as_secs_f64() / hit.ttft.as_secs_f64()
    };
    let performance_passed =
        restored && skip_fraction >= REQUIRED_SKIP_FRACTION && speedup >= REQUIRED_TTFT_SPEEDUP;

    Ok(json!({
        "schema": SCHEMA,
        "expected_tier": match config.expected_tier {
            ExpectedCacheTier::Hot => "t1_hot",
            ExpectedCacheTier::WarmOrRecompute => "t2_warm_or_recompute",
        },
        "prompt_tokens": config.prompt_tokens.len(),
        "page_tokens": PROMPT_CACHE_PAGE_TOKENS,
        "reusable_tokens": reusable,
        "private_tail_tokens": config.prompt_tokens.len().saturating_sub(reusable),
        "final_prompt_token_private": reusable < config.prompt_tokens.len(),
        "max_output_tokens": config.max_output_tokens,
        "cold": observation_json(&cold),
        "candidate": observation_json(&hit),
        "exact_output_match": true,
        "continuation_cache_probe": {
            "generated_tokens_appended": CONTINUATION_PROBE_TOKENS,
            "probe_reusable_tokens": continuation_reusable,
            "matched_tokens": continuation_matched,
            "no_generated_continuation_kv_cached": continuation_matched <= reusable,
            "observation": observation_json(&continuation_probe),
        },
        "restored": restored,
        "recomputed": !restored,
        "matched_tokens": matched,
        "saved_prefill_tokens": saved,
        "reusable_work_skipped_fraction": skip_fraction,
        "ttft_speedup": finite_or_string(speedup),
        "requirements": {
            "reusable_tokens": REQUIRED_REUSABLE_TOKENS,
            "minimum_skip_fraction": REQUIRED_SKIP_FRACTION,
            "minimum_ttft_speedup": REQUIRED_TTFT_SPEEDUP,
        },
        "performance_passed": performance_passed,
        "passed": match config.expected_tier {
            ExpectedCacheTier::Hot => performance_passed,
            // Recompute is the correct fail-closed T2 outcome when restore is
            // not predicted to beat prefill. It is safe, but not promoted.
            ExpectedCacheTier::WarmOrRecompute => !restored || performance_passed,
        },
        "promotion_eligible": performance_passed,
    }))
}

fn validate(config: &PromptCacheGateConfig) -> Result<(), String> {
    let request = GenerateRequest::new(config.prompt_tokens.clone(), config.max_output_tokens);
    request
        .validate()
        .map_err(|error| format!("invalid cache-gate request: {error}"))?;
    let reusable = rvllm_runtime::reusable_prompt_tokens(config.prompt_tokens.len());
    if reusable != REQUIRED_REUSABLE_TOKENS {
        return Err(format!(
            "cache promotion gate requires exactly {REQUIRED_REUSABLE_TOKENS} reusable tokens; prompt length {} yields {reusable}",
            config.prompt_tokens.len()
        ));
    }
    if config.prompt_tokens.len() != REQUIRED_REUSABLE_TOKENS + 1 {
        return Err(format!(
            "cache promotion gate requires a {}-token prompt so the final token stays private",
            REQUIRED_REUSABLE_TOKENS + 1
        ));
    }
    if config.max_output_tokens < CONTINUATION_PROBE_TOKENS as u32 {
        return Err(format!(
            "cache promotion gate requires at least {CONTINUATION_PROBE_TOKENS} output tokens to verify that generated continuation KV is never cached"
        ));
    }
    Ok(())
}

fn run_one(engine: &EngineHandle, request: GenerateRequest) -> Result<Observation, String> {
    let submitted = Instant::now();
    let mut handle = engine
        .submit(request)
        .map_err(|error| format!("submit cache-gate request: {error}"))?;
    let mut first_token = None;
    let mut tokens = Vec::new();
    loop {
        match handle
            .recv()
            .map_err(|error| format!("receive cache-gate event: {error}"))?
        {
            TokenEvent::Token { token_id, .. } => {
                first_token.get_or_insert_with(Instant::now);
                tokens.push(token_id);
            }
            TokenEvent::Finished { report, .. } => {
                let first_token = first_token
                    .ok_or_else(|| "cache-gate request finished without a token".to_owned())?;
                return Ok(Observation {
                    ttft: first_token.saturating_duration_since(submitted),
                    tokens,
                    report,
                });
            }
        }
    }
}

fn observation_json(observation: &Observation) -> Value {
    json!({
        "ttft_ms": observation.ttft.as_secs_f64() * 1_000.0,
        "generated_token_ids": raw_tokens(&observation.tokens),
        "cache_tier": cache_tier_label(observation.report.cache_tier),
        "matched_tokens": observation.report.matched_cache_tokens,
        "saved_prefill_tokens": observation.report.saved_prefill_tokens,
        "prefill_ms": observation.report.prefill_time.as_secs_f64() * 1_000.0,
        "decode_ms": observation.report.decode_time.as_secs_f64() * 1_000.0,
        "queue_ms": observation.report.queue_time.as_secs_f64() * 1_000.0,
    })
}

fn raw_tokens(tokens: &[TokenId]) -> Vec<u32> {
    tokens.iter().map(|token| token.raw()).collect()
}

fn cache_tier_label(tier: CacheTier) -> &'static str {
    match tier {
        CacheTier::None => "none",
        CacheTier::Active => "active",
        CacheTier::Hot => "hot",
        CacheTier::Warm => "warm",
        CacheTier::Persistent => "persistent",
    }
}

fn finite_or_string(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::String("infinity".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_requires_513_tokens_and_keeps_one_private() {
        let config = PromptCacheGateConfig {
            prompt_tokens: vec![TokenId(7); 513],
            max_output_tokens: 32,
            expected_tier: ExpectedCacheTier::Hot,
        };
        validate(&config).expect("valid gate prompt");
        assert_eq!(rvllm_runtime::reusable_prompt_tokens(513), 512);
        assert!(validate(&PromptCacheGateConfig {
            prompt_tokens: vec![TokenId(7); 512],
            ..config.clone()
        })
        .is_err());
        assert!(validate(&PromptCacheGateConfig {
            prompt_tokens: vec![TokenId(7); 514],
            ..config
        })
        .is_err());
    }
}
