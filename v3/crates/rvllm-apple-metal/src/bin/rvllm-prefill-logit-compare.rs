//! Fail-closed, offline comparison of two pinned single-prompt Metal receipts.
//! This reports a diagnostic difference; it is not an independent quality oracle.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const MODEL_SHA256: &str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
const PROMPT_SHA256: &str = "72e42ab11ef82d9a1779c2d93c9f30c3bbc39bc9f5ea478cc293ec18a4b68406";
const INFER_SHA256: &str = "7b6487471816cfe71999bf9d54631589105dcf88cb4f569e924c28c048cc1e20";
const NORMAL_LIB_SHA256: &str = "bb88c9667b4b7ac3758760b602abe40cbe72cc907bf3c40e2f2a2d33b0bd8df2";
const COMBINED_LIB_SHA256: &str =
    "8a1234a3ac14cb1c55c6c4c601c4fb2647defbc955b14be3a2c23414552fbfcb";

struct Receipt {
    job: Value,
    queue: Value,
    inference: Value,
    hashes: Value,
}

fn read_json(path: &Path) -> Result<(Value, String)> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let value =
        serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok((value, hash))
}

fn pinned_input(job: &Value, suffix: &str) -> Result<String> {
    let inputs = job["inputs"].as_array().ok_or("missing pinned inputs")?;
    let mut matches = inputs.iter().filter(|input| {
        input["path"]
            .as_str()
            .is_some_and(|path| path.ends_with(suffix))
    });
    let pin = matches
        .next()
        .ok_or_else(|| format!("missing pin {suffix}"))?;
    if matches.next().is_some() {
        return Err(format!("ambiguous pin {suffix}"));
    }
    pin["sha256"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("missing SHA-256 for {suffix}"))
}

fn expected_dispatch(inference: &Value, candidate: bool) -> Result<()> {
    let dispatch = &inference["research_dispatch"];
    if dispatch["overflowed"] != false {
        return Err("dispatch counter overflow or missing flag".into());
    }
    let counts = dispatch["counts"]
        .as_object()
        .ok_or("missing research dispatch counts")?;
    let expected: &[(&str, u64)] = if candidate {
        &[
            ("research_prefill_combined_gemm", 48),
            ("research_prefill_combined_qkv", 48),
            ("research_prefill_combined_raw_norm_projection", 96),
            ("research_prefill_combined_raw_norm", 96),
            ("research_prefill_combined_d256", 40),
            ("research_prefill_combined_d512", 8),
        ]
    } else {
        &[]
    };
    if counts.len() != expected.len()
        || expected
            .iter()
            .any(|(name, count)| counts.get(*name).and_then(Value::as_u64) != Some(*count))
    {
        return Err("fallback or unexpected research dispatch".into());
    }
    Ok(())
}

fn read_receipt(dir: &Path, candidate: bool) -> Result<Receipt> {
    let (job, job_sha256) = read_json(&dir.join("job.json"))?;
    let (queue, queue_sha256) = read_json(&dir.join("report.json"))?;
    let (inference, inference_sha256) = read_json(&dir.join("trial.stdout"))?;
    if queue["id"] != job["id"]
        || queue["status"] != "succeeded"
        || queue["exit_code"] != 0
        || queue["files_unchanged"] != true
    {
        return Err(format!(
            "{}: queue result not successful and sealed",
            dir.display()
        ));
    }
    if job["purpose"] != "correctness"
        || job["command"]["executable"]["sha256"] != INFER_SHA256
        || pinned_input(&job, "model.safetensors")? != MODEL_SHA256
        || pinned_input(&job, "prompt-m304-only.jsonl")? != PROMPT_SHA256
        || pinned_input(&job, "config.json")?
            != "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9"
        || pinned_input(&job, "tokenizer.json")?
            != "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    {
        return Err("model, prompt, tokenizer or executable identity changed".into());
    }
    let selector = if candidate {
        "metal-prefill-pipeline32x64-q4k16"
    } else {
        "off"
    };
    let (lib_name, lib_hash) = if candidate {
        (
            "metal-prefill-pipeline32x64-q4k16.metallib",
            COMBINED_LIB_SHA256,
        )
    } else {
        ("normal.metallib", NORMAL_LIB_SHA256)
    };
    if job["command"]["env"]["RVLLM_METAL_RESEARCH"] != selector
        || pinned_input(&job, lib_name)? != lib_hash
    {
        return Err("wrong Metal route or metallib".into());
    }
    if inference["schema"] != "rvllm.apple_metal_text_infer.v1"
        || inference["metal_compute_dtype"] != "bfloat16"
        || inference["max_new_tokens"] != 1
        || inference["prompt_token_ids"]
            .as_array()
            .is_none_or(|ids| ids.len() != 304)
        || inference["generated_token_ids"]
            .as_array()
            .is_none_or(|ids| ids.len() != 1)
    {
        return Err("wrong model route, prompt length or generated work".into());
    }
    expected_dispatch(&inference, candidate)?;
    let hashes = json!({
        "job_sha256": job_sha256,
        "queue_report_sha256": queue_sha256,
        "inference_stdout_sha256": inference_sha256,
    });
    Ok(Receipt {
        job,
        queue,
        inference,
        hashes,
    })
}

fn top_logits(inference: &Value) -> Result<Vec<(u32, f64)>> {
    let rows = inference["diagnostic_top_logits"]
        .as_array()
        .ok_or("missing top logits")?;
    if rows.len() != 256 {
        return Err(format!("expected 256 top logits, got {}", rows.len()));
    }
    let mut seen = BTreeSet::new();
    let mut result = Vec::with_capacity(rows.len());
    for row in rows {
        let raw_id = row["token_id"].as_u64().ok_or("missing token ID")?;
        let id = u32::try_from(raw_id).map_err(|_| "token ID out of range")?;
        let logit = row["logit"].as_f64().ok_or("missing logit")?;
        if !logit.is_finite() || !seen.insert(id) {
            return Err("nonfinite logit or duplicate token ID".into());
        }
        if result.last().is_some_and(|(_, previous)| *previous < logit) {
            return Err("top logits are not sorted descending".into());
        }
        result.push((id, logit));
    }
    Ok(result)
}

fn compare(off: &Receipt, candidate: &Receipt) -> Result<Value> {
    if off.job["command"]["args"] != candidate.job["command"]["args"]
        || off.inference["prompt_token_ids"] != candidate.inference["prompt_token_ids"]
        || off.inference["generated_token_ids"] != candidate.inference["generated_token_ids"]
    {
        return Err("changed prompt, invocation or generated ID".into());
    }
    let off_top = top_logits(&off.inference)?;
    let candidate_top = top_logits(&candidate.inference)?;
    let candidate_map: BTreeMap<u32, (usize, f64)> = candidate_top
        .iter()
        .enumerate()
        .map(|(rank, (id, value))| (*id, (rank, *value)))
        .collect();
    let mut common = 0_usize;
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut max_rank_shift = 0_usize;
    for (rank, (id, value)) in off_top.iter().enumerate() {
        if let Some((candidate_rank, candidate_value)) = candidate_map.get(id) {
            common += 1;
            let difference = (value - candidate_value).abs();
            max_abs = max_abs.max(difference);
            sum_abs += difference;
            max_rank_shift = max_rank_shift.max(rank.abs_diff(*candidate_rank));
        }
    }
    if common == 0 {
        return Err("top-logit sets have no overlap".into());
    }
    let same_ranked_ids = off_top
        .iter()
        .map(|(id, _)| id)
        .eq(candidate_top.iter().map(|(id, _)| id));
    Ok(json!({
        "schema": "rvllm.prefill_top_logit_comparison.v1",
        "claim": "one-step final decode-logit diagnostic, not a tensor/reference or checkpoint-quality verdict; queue conditions and preparation compilation are recorded separately",
        "model_safetensors_sha256": MODEL_SHA256,
        "prompt_jsonl_sha256": PROMPT_SHA256,
        "inference_executable_sha256": INFER_SHA256,
        "off_job_id": off.job["id"],
        "candidate_job_id": candidate.job["id"],
        "off_receipt_hashes": off.hashes,
        "candidate_receipt_hashes": candidate.hashes,
        "off_sampled_conditions_eligible": off.queue["sampled_conditions_eligible"],
        "candidate_sampled_conditions_eligible": candidate.queue["sampled_conditions_eligible"],
        "off_violations": off.queue["violations"],
        "candidate_violations": candidate.queue["violations"],
        "generated_token_ids": off.inference["generated_token_ids"],
        "same_top_1_id": off_top[0].0 == candidate_top[0].0,
        "same_ranked_top_256_ids": same_ranked_ids,
        "common_top_256_ids": common,
        "missing_from_candidate_top_256": 256 - common,
        "max_rank_shift_on_common_ids": max_rank_shift,
        "max_abs_logit_difference_on_common_ids": max_abs,
        "mean_abs_logit_difference_on_common_ids": sum_abs / common as f64,
        "off_top_16": off_top[..16].iter().map(|(id, value)| json!({"token_id":id,"logit":value})).collect::<Vec<_>>(),
        "candidate_top_16": candidate_top[..16].iter().map(|(id, value)| json!({"token_id":id,"logit":value})).collect::<Vec<_>>(),
    }))
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: rvllm-prefill-logit-compare OFF_QUEUE_DIR CANDIDATE_QUEUE_DIR OUTPUT.json"
                .into(),
        );
    }
    let off = read_receipt(&PathBuf::from(&args[0]), false)?;
    let candidate = read_receipt(&PathBuf::from(&args[1]), true)?;
    let result = compare(&off, &candidate)?;
    let output = PathBuf::from(&args[2]);
    let bytes = serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|error| format!("{}: {error}", output.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("{}: {error}", output.display()))?;
    println!("{}", output.display());
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_logits_rejects_duplicates_and_wrong_count() {
        let rows: Vec<_> = (0..256)
            .map(|index| json!({"token_id":index,"logit":-(index as f64)}))
            .collect();
        let mut report = json!({"diagnostic_top_logits": rows});
        assert_eq!(top_logits(&report).unwrap().len(), 256);
        report["diagnostic_top_logits"][1]["token_id"] = json!(0);
        assert!(top_logits(&report).unwrap_err().contains("duplicate"));
        report["diagnostic_top_logits"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(top_logits(&report).unwrap_err().contains("256"));
    }

    #[test]
    fn dispatch_rejects_fallback() {
        let report = json!({"research_dispatch":{"overflowed":false,"counts":{}}});
        assert!(expected_dispatch(&report, false).is_ok());
        assert!(expected_dispatch(&report, true).is_err());
    }
}
