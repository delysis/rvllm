#![forbid(unsafe_code)]

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const PROMPT_TOKENS: usize = 84;
const OUTPUT_TOKENS: usize = 10;
const CONTEXT: u32 = 1024;
const WARMUPS: usize = 2;
const REPETITIONS: usize = 7;
const MODEL_BYTES: u64 = 6_975_879_296;
const MODEL_SHA256: &str = "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b";
const TEMPLATE_SHA256: &str = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4";
const REFERENCE_SHA256: &str = "0b096c565cc841ee625081d23b1752af0590c25c8f48554451e97e1fb7140ee8";
const PROMPT_SHA256: &str = "9269f2ca629b55246f417545e5a4f3acde6d28bf790d7552b63fefd4c875ea8a";
const STANDARD_OUTPUT: [i32; 10] = [818, 3730, 37423, 38167, 1024, 506, 12010, 8858, 236761, 106];

struct Args {
    model: PathBuf,
    prompt_reference: PathBuf,
    prompt_file: PathBuf,
    output: PathBuf,
    threads: i32,
}

fn parse_args() -> Result<Args> {
    let mut values = BTreeMap::<String, OsString>::new();
    let mut seen = HashSet::new();
    let mut args = std::env::args_os().skip(1);
    while let Some(flag) = args.next() {
        let flag = flag.into_string().map_err(|_| "non-UTF8 option name")?;
        if !matches!(
            flag.as_str(),
            "--model" | "--prompt-reference" | "--prompt-file" | "--output-json" | "--threads"
        ) {
            return Err(format!("unknown argument {flag}; required: --model PATH --prompt-reference PATH --prompt-file PATH --output-json NEW_PATH [--threads 8]").into());
        }
        if !seen.insert(flag.clone()) {
            return Err(format!("duplicate argument {flag}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        values.insert(flag, value);
    }
    let mut take = |key: &str| -> Result<OsString> {
        values
            .remove(key)
            .ok_or_else(|| format!("missing {key}").into())
    };
    let model = PathBuf::from(take("--model")?);
    let prompt_reference = PathBuf::from(take("--prompt-reference")?);
    let prompt_file = PathBuf::from(take("--prompt-file")?);
    let output = PathBuf::from(take("--output-json")?);
    let threads = values
        .remove("--threads")
        .unwrap_or_else(|| OsString::from("8"));
    let threads: i32 = threads.to_str().ok_or("non-UTF8 thread count")?.parse()?;
    if !(1..=64).contains(&threads) {
        return Err("threads must be in 1..=64".into());
    }
    Ok(Args {
        model,
        prompt_reference,
        prompt_file,
        output,
        threads,
    })
}

fn small_file(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(1_048_577).read_to_end(&mut bytes)?;
    if bytes.len() > 1_048_576 {
        return Err(format!("input is over 1 MiB: {}", path.display()).into());
    }
    Ok(bytes)
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked_prompt_ids(reference: &[u8], n_vocab: i32) -> Result<Vec<LlamaToken>> {
    let value: Value = serde_json::from_slice(reference)?;
    let ids = value
        .get("prompt_token_ids")
        .and_then(Value::as_array)
        .ok_or("reference must contain a prompt_token_ids array")?;
    if ids.len() != PROMPT_TOKENS {
        return Err(format!("expected {PROMPT_TOKENS} prompt IDs, got {}", ids.len()).into());
    }
    ids.iter()
        .map(|id| {
            let id = id.as_i64().ok_or("prompt token must be an integer")?;
            let id = i32::try_from(id)?;
            if !(0..n_vocab).contains(&id) {
                return Err(format!("prompt token {id} outside vocabulary {n_vocab}").into());
            }
            Ok(LlamaToken::new(id))
        })
        .collect()
}

// First maximum wins, matching llama.cpp's ordinary greedy sampler. No penalties,
// temperature, grammar, RNG, or acceptance state is involved in this bounded trial.
fn greedy(logits: &[f32]) -> Result<LlamaToken> {
    if logits.is_empty() {
        return Err("empty logits".into());
    }
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("non-finite logit at {index}").into());
        }
        if *value > logits[best] {
            best = index;
        }
    }
    Ok(LlamaToken::new(i32::try_from(best)?))
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn repeat(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    prompt: &[LlamaToken],
    index: usize,
) -> Result<(Value, Vec<i32>, bool)> {
    let request_start = Instant::now();
    ctx.clear_kv_cache();
    let reset_ms = milliseconds(request_start);
    ctx.reset_timings();
    let mut batch = LlamaBatch::new(PROMPT_TOKENS, 1);
    batch.add_sequence(prompt, 0, false)?;
    let generation_start = Instant::now();
    ctx.decode(&mut batch)?;
    // The pinned C API calls synchronize() in llama_get_logits_ith(). Reading
    // this safe wrapper is the GPU completion boundary, not decode() returning.
    let logits = ctx.get_logits_ith(i32::try_from(PROMPT_TOKENS - 1)?);
    let prefill_ms = milliseconds(generation_start);
    let mut token = greedy(logits)?;
    let mut ids = vec![token.0];
    let first_token_ms = milliseconds(request_start);
    let decode_start = Instant::now();
    let mut step_ms = Vec::with_capacity(OUTPUT_TOKENS - 1);
    let mut eog = model.is_eog_token(token);
    while ids.len() < OUTPUT_TOKENS && !eog {
        batch.clear();
        let position = PROMPT_TOKENS + ids.len() - 1;
        batch.add(token, i32::try_from(position)?, &[0], true)?;
        let start = Instant::now();
        ctx.decode(&mut batch)?;
        let logits = ctx.get_logits_ith(0);
        step_ms.push(milliseconds(start));
        token = greedy(logits)?;
        ids.push(token.0);
        eog = model.is_eog_token(token);
    }
    let decode_wall_ms = milliseconds(decode_start);
    let generation_wall_ms = milliseconds(generation_start);
    let request_wall_ms = milliseconds(request_start);
    let timings = ctx.timings();
    let completed = step_ms.len();
    let work_match = ids.len() == OUTPUT_TOKENS && completed == OUTPUT_TOKENS - 1;
    let counters_match = timings.n_p_eval() == i32::try_from(PROMPT_TOKENS)?
        && (completed == 0 || timings.n_eval() == i32::try_from(completed)?);
    let decode_eval_sum_ms: f64 = step_ms.iter().sum();
    let row = json!({
        "event": "repetition", "index": index, "warmup": index < WARMUPS,
        "prompt_token_ids": prompt.iter().map(|token| token.0).collect::<Vec<_>>(),
        "prefill_wall_ms": prefill_ms, "decode_wall_ms": decode_wall_ms,
        "actual_decode_steps": completed, "n_generated": ids.len(),
        "prompt_tokens": prompt.len(), "requested_output_tokens": OUTPUT_TOKENS,
        "requested_decode_steps": OUTPUT_TOKENS - 1, "completed_decode_steps": completed,
        "output_token_ids": ids, "output_token_count": ids.len(), "last_token_is_eog": eog,
        "finish_reason": if eog { "eog" } else { "max_tokens" },
        "matches_standard_checkpoint_output": ids == STANDARD_OUTPUT,
        "work_matches_84_prefill_9_decode": work_match, "internal_counters_match": counters_match,
        "kv_reset_wall_ms": reset_ms, "prefill_synchronized_wall_ms": prefill_ms,
        "first_token_wall_ms_including_reset": first_token_ms,
        "decode_eval_wall_ms_each": step_ms, "decode_eval_wall_sum_ms": decode_eval_sum_ms,
        "decode_wall_ms_including_sampling": decode_wall_ms,
        "generation_wall_ms": generation_wall_ms, "request_wall_ms_including_reset": request_wall_ms,
        "internal_prompt_ms": timings.t_p_eval_ms(), "internal_prompt_tokens": timings.n_p_eval(),
        "internal_decode_ms": timings.t_eval_ms(), "internal_decode_runs_raw": timings.n_eval(),
        "internal_zero_count_caveat": "llama.cpp clamps n_eval to at least 1; use completed_decode_steps for actual work",
        "decode_steps_per_second_wall_eval_only": if decode_eval_sum_ms > 0.0 { Some(completed as f64 * 1000.0 / decode_eval_sum_ms) } else { None },
        "decode_steps_per_second_including_sampling": if decode_wall_ms > 0.0 { Some(completed as f64 * 1000.0 / decode_wall_ms) } else { None }
    });
    Ok((row, ids, work_match && counters_match))
}

fn emit(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn run(args: &Args, writer: &mut impl Write) -> Result<()> {
    let reference = small_file(&args.prompt_reference)?;
    let prompt_bytes = small_file(&args.prompt_file)?;
    if digest(&reference) != REFERENCE_SHA256 || digest(&prompt_bytes) != PROMPT_SHA256 {
        return Err("frozen reference or prompt SHA256 mismatch".into());
    }
    let model_path = args.model.canonicalize()?;
    if std::fs::metadata(&model_path)?.len() != MODEL_BYTES {
        return Err("model size differs from the inventoried official Google QAT artifact".into());
    }
    emit(
        writer,
        &json!({"event": "preparation_started", "schema": 1,
        "model_path": model_path, "expected_model_sha256": MODEL_SHA256,
        "model_sha256_verified_by_this_runner": false,
        "model_identity_note": "Hash once externally before a campaign; the runner checks byte length and metadata, not all tensor bytes.",
        "comparison": "Google native QAT Q4_0 GGUF; checkpoint/precision differs from rvllm standard BF16 plus INT8 FFNs",
        "bindings_commit": "a3cf95eb1d4fa748480eb780e6fcbfc1a5c1c391",
        "llama_cpp_commit": "5f55650a78f92aff4d48d671423e888fac0469ff"}),
    )?;
    let preparation_start = Instant::now();
    let backend = LlamaBackend::init()?;
    let params = LlamaModelParams::default()
        .with_n_gpu_layers(99)
        .with_use_mmap(true)
        .with_use_mlock(false);
    let model = LlamaModel::load_from_file(&backend, &model_path, &params)?;
    if model.n_vocab() != 262_144 || model.n_layer() != 48 || model.n_embd() != 3840 {
        return Err("model vocabulary/layer/hidden geometry is not Gemma 4 12B".into());
    }
    let mut metadata = BTreeMap::new();
    for key in [
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "gemma4.block_count",
        "gemma4.embedding_length",
        "gemma4.feed_forward_length",
        "gemma4.context_length",
        "gemma4.attention.sliding_window",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
    ] {
        metadata.insert(key, model.meta_val_str(key).ok());
    }
    if metadata
        .get("general.architecture")
        .and_then(Option::as_deref)
        != Some("gemma4")
    {
        return Err("model architecture is not gemma4".into());
    }
    let template = model.meta_val_str("tokenizer.chat_template")?;
    if digest(template.as_bytes()) != TEMPLATE_SHA256 {
        return Err("unqualified chat template".into());
    }
    let tokens = checked_prompt_ids(&reference, model.n_vocab())?;
    let content = std::str::from_utf8(&prompt_bytes)?
        .trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c));
    let rendered =
        format!("<bos><|turn>user\n{content}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    if model.str_to_token(&rendered, AddBos::Never)? != tokens {
        return Err("GGUF tokenization does not match the exact 84 reference IDs".into());
    }
    if tokens.first().map(|token| token.0) != Some(2) {
        return Err("expected one leading BOS ID 2".into());
    }
    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(CONTEXT))
        .with_n_batch(512)
        .with_n_ubatch(512)
        .with_n_seq_max(1)
        .with_n_threads(args.threads)
        .with_n_threads_batch(args.threads)
        .with_type_k(KvCacheType::F16)
        .with_type_v(KvCacheType::F16)
        .with_no_perf(false)
        .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED);
    let mut context = model.new_context(&backend, context_params)?;
    if context.n_ctx() != CONTEXT
        || tokens.len() + OUTPUT_TOKENS > usize::try_from(context.n_ctx())?
    {
        return Err("context capacity differs from the bounded trial".into());
    }
    emit(
        writer,
        &json!({"event": "prepared", "preparation_wall_ms": milliseconds(preparation_start),
        "model_metadata": metadata, "model_tensor_bytes": model.size(), "model_parameters": model.n_params(),
        "model_vocab": model.n_vocab(), "context_tokens": context.n_ctx(), "threads": args.threads,
        "gpu_layers_requested": 99, "flash_attention_requested": "enabled", "kv_type": "f16",
        "batch": 512, "ubatch": 512, "sequences": 1, "mmap": true, "mlock": false,
        "backend_verification": "Preserve stderr placement logs; requested offload is not proof every operation executed on Metal.",
        "prompt_reference_sha256": REFERENCE_SHA256, "prompt_file_sha256": PROMPT_SHA256,
        "chat_template_sha256": TEMPLATE_SHA256, "rendered_prompt": rendered,
        "prompt_token_ids": tokens.iter().map(|token| token.0).collect::<Vec<_>>(),
        "warmups": WARMUPS, "measured_repetitions": REPETITIONS,
        "timing_scope": "Synchronized host wall and llama internal elapsed time, not GPU device timestamps or cycles. Report serialization is outside request windows and disk writes occur after all repetitions."}),
    )?;
    let mut first_ids: Option<Vec<i32>> = None;
    for index in 0..WARMUPS + REPETITIONS {
        let (mut row, ids, valid_work) = repeat(&model, &mut context, &tokens, index)?;
        let stable = first_ids.as_ref().is_none_or(|first| *first == ids);
        row["matches_first_repetition"] = json!(stable);
        emit(writer, &row)?;
        if !valid_work || !stable {
            return Err(
                "trial work/counters or repeated output IDs diverged; no automatic retry".into(),
            );
        }
        if first_ids.is_none() {
            first_ids = Some(ids);
        }
    }
    emit(
        writer,
        &json!({"event": "complete", "warmups": WARMUPS, "measured_repetitions": REPETITIONS}),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    // Keep receipts in memory during the campaign. A process crash leaves an
    // empty output file; queue status/stderr must reject it, never accept it.
    let mut records = Vec::new();
    let result = run(&args, &mut records);
    if let Err(error) = &result {
        let _ = emit(
            &mut records,
            &json!({"event": "failed", "error": error.to_string()}),
        );
    }
    let events = serde_json::Deserializer::from_slice(&records)
        .into_iter::<Value>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let find = |name: &str| events.iter().find(|event| event["event"] == name);
    let repetitions: Vec<_> = events
        .iter()
        .filter(|event| event["event"] == "repetition")
        .collect();
    let report = json!({
        "schema": "rvllm_llama_native_kit_baseline_v1",
        "backend": "llama_cpp_native_kit_pin",
        "status": if result.is_ok() { "complete" } else { "failed" },
        "preparation": find("preparation_started"), "configuration": find("prepared"),
        "repetitions": repetitions, "failure": find("failed"),
        "warmups_requested": WARMUPS, "measured_repetitions_requested": REPETITIONS
    });
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_ties_select_first_and_nonfinite_values_fail() {
        assert_eq!(greedy(&[1.0, 4.0, 4.0]).unwrap().0, 1);
        assert!(greedy(&[1.0, f32::NAN]).is_err());
        assert!(greedy(&[]).is_err());
    }

    #[test]
    fn reference_rejects_invalid_token_and_count() {
        let mut ids = vec![2; PROMPT_TOKENS];
        ids[10] = 262_144;
        let bytes = serde_json::to_vec(&json!({"prompt_token_ids": ids})).unwrap();
        assert!(checked_prompt_ids(&bytes, 262_144).is_err());
        assert!(checked_prompt_ids(br#"{"prompt_token_ids": [2]}"#, 262_144).is_err());
    }
}
