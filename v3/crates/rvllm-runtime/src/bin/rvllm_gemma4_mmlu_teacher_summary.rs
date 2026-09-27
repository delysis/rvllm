//! Offline, fail-closed summary of two fixed MMLU natural-question trials.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const IDS: [(&str, &str, &str); 4] = [
    (
        "mmlu-formal-logic-2443",
        "off",
        "prefill26-mmlu-natural-logic-off-v1-20260927",
    ),
    (
        "mmlu-formal-logic-2443",
        "combined",
        "prefill26-mmlu-natural-logic-combined-v1-20260927",
    ),
    (
        "mmlu-high-school-computer-science-3185",
        "off",
        "prefill26-mmlu-natural-cs-off-v1-20260927",
    ),
    (
        "mmlu-high-school-computer-science-3185",
        "combined",
        "prefill26-mmlu-natural-cs-combined-v1-20260927",
    ),
];
const COMBINED_DISPATCH: [(&str, u64); 6] = [
    ("research_prefill_combined_gemm", 48),
    ("research_prefill_combined_qkv", 48),
    ("research_prefill_combined_raw_norm_projection", 96),
    ("research_prefill_combined_raw_norm", 96),
    ("research_prefill_combined_d256", 40),
    ("research_prefill_combined_d512", 8),
];
const EXECUTABLE_SHA: &str = "23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463";
const CLI_SOURCE_SHA: &str = "c091c7069ad563271ae2d46361190c0f8f451f521abfe41d7d30b749c4fbdf04";
const TOKEN_BUILDER_SHA: &str = "bcfcc431e39aa9939d07741fe67335cf9c7a8af3fb0f8f398e4594c36200cdbf";
const JOB_GENERATOR_SHA: &str = "e2e68b4854880743ef1f7d2e626e9347ae30715c7bfeae4777aef5a0c094198b";
const SOURCE_SHA: &str = "a1d115405e86ab51bf275473cc04f452f185db2c18f286b5173d2234d4b993b4";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const DATASET_SHA: &str = "58769d67ced092719390e496f1d4bbf8b99c43b77bba39d054b9a09aecb63cab";
const MODEL_SHA: &str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
const CONFIG_SHA: &str = "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9";

type Result<T> = std::result::Result<T, String>;

struct Arm {
    report: Value,
    job: Value,
    trial: Value,
    receipt_sha256: BTreeMap<&'static str, String>,
}

fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    if args.len() != 8 {
        eprintln!("usage: rvllm_gemma4_mmlu_teacher_summary OUTPUT.json TOKENIZER.json SOURCE.json LOGIC_OFF_DIR LOGIC_COMBINED_DIR CS_OFF_DIR CS_COMBINED_DIR");
        std::process::exit(2);
    }
    let paths = args[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    match run(&paths) {
        Ok(summary) => {
            let serialized = serde_json::to_vec_pretty(&summary).expect("serializable summary");
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&paths[0])
            {
                Ok(mut file) => {
                    if let Err(error) = file
                        .write_all(&serialized)
                        .and_then(|_| file.write_all(b"\n"))
                    {
                        eprintln!("write {}: {error}", paths[0].display());
                        std::process::exit(1);
                    }
                }
                Err(error) => {
                    eprintln!("create {}: {error}", paths[0].display());
                    std::process::exit(1);
                }
            }
        }
        Err(error) => {
            eprintln!("rvllm_gemma4_mmlu_teacher_summary: {error}");
            std::process::exit(1);
        }
    }
}

fn read(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse_strict_json(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|value| value.as_str().to_owned())
        .map_err(|error| error.to_string())
}

fn at<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value.get(key).ok_or_else(|| format!("missing {key}"))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    at(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not text"))
}

fn ids(value: &Value, key: &str) -> Result<Vec<u32>> {
    at(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not an array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("{key} contains a non-token ID"))
        })
        .collect()
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| format!("tokenize: {error}"))?;
    let mut result = vec![2];
    result.extend_from_slice(encoded.get_ids());
    Ok(result)
}

fn expected_case(
    tokenizer: &tokenizers::Tokenizer,
    source: &Value,
    id: &str,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let cases = at(source, "cases")?
        .as_array()
        .ok_or("cases is not an array")?;
    let matches = cases
        .iter()
        .filter(|case| case.get("id").and_then(Value::as_str) == Some(id))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected exactly one source case {id}"));
    }
    let prompt = string(matches[0], "prompt")?;
    let continuation = string(matches[0], "continuation")?;
    let prompt_ids = encode(tokenizer, prompt)?;
    let full_ids = encode(tokenizer, &format!("{prompt}{continuation}"))?;
    let targets = full_ids
        .strip_prefix(prompt_ids.as_slice())
        .filter(|tail| !tail.is_empty())
        .ok_or_else(|| format!("case {id} retokenized at boundary"))?;
    Ok((prompt_ids, targets.to_vec()))
}

fn pinned(job: &Value, suffix: &str, expected: &str) -> Result<()> {
    let inputs = at(job, "inputs")?
        .as_array()
        .ok_or("inputs is not an array")?;
    let matches = inputs
        .iter()
        .filter(|item| {
            item.get("path")
                .and_then(Value::as_str)
                .is_some_and(|path| path.ends_with(suffix))
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 || string(matches[0], "sha256")? != expected {
        return Err(format!("wrong or missing pin for {suffix}"));
    }
    Ok(())
}

fn command_arg<'a>(job: &'a Value, flag: &str) -> Result<&'a str> {
    let args = at(at(job, "command")?, "args")?
        .as_array()
        .ok_or("command args are not an array")?;
    let matches = args
        .windows(2)
        .filter(|pair| pair[0].as_str() == Some(flag))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected one {flag} argument"));
    }
    matches[0][1]
        .as_str()
        .ok_or_else(|| format!("{flag} argument is not text"))
}

fn read_arm(
    dir: &Path,
    id: &str,
    mode: &str,
    tokenizer_sha: &str,
    source_sha: &str,
) -> Result<Arm> {
    let report_path = dir.join("report.json");
    let job_path = dir.join("job.json");
    let trial_path = dir.join("trial.stdout");
    let stderr_path = dir.join("trial.stderr");
    let conditions_path = dir.join("conditions.jsonl");
    let report = read(&report_path)?;
    if string(&report, "id")? != id
        || string(&report, "status")? != "succeeded"
        || at(&report, "exit_code")?.as_i64() != Some(0)
        || at(&report, "files_unchanged")?.as_bool() != Some(true)
        || !at(&report, "violations")?
            .as_array()
            .is_some_and(Vec::is_empty)
    {
        return Err(format!("{id}: queue receipt not clean and terminal"));
    }
    let job = read(&job_path)?;
    if string(&job, "id")? != id {
        return Err(format!("{id}: job identity mismatch"));
    }
    pinned(&job, "mmlu-natural-source-v1.json", source_sha)?;
    pinned(&job, "mmlu-test.arrow", DATASET_SHA)?;
    pinned(&job, "rvllm_gemma4_teacher_job_gen.rs", JOB_GENERATOR_SHA)?;
    pinned(&job, "tokenizer.json", tokenizer_sha)?;
    pinned(&job, "rvllm_metal_infer.rs", CLI_SOURCE_SHA)?;
    pinned(&job, "rvllm_gemma4_heldout_tokens.rs", TOKEN_BUILDER_SHA)?;
    pinned(&job, "config.json", CONFIG_SHA)?;
    pinned(&job, "model.safetensors", MODEL_SHA)?;
    let (prompt_file, prompt_sha, library_file, library_sha) = match (id.contains("-logic-"), mode)
    {
        (true, "off") => (
            "mmlu-logic-prompt.jsonl",
            "a3739872c640c5efaf71ed10c6f41469dc6b6c7869253e951cb2553fff8ae0b5",
            "normal.metallib",
            "bb88c9667b4b7ac3758760b602abe40cbe72cc907bf3c40e2f2a2d33b0bd8df2",
        ),
        (true, "combined") => (
            "mmlu-logic-prompt.jsonl",
            "a3739872c640c5efaf71ed10c6f41469dc6b6c7869253e951cb2553fff8ae0b5",
            "metal-prefill-pipeline32x64-q4k16.metallib",
            "8a1234a3ac14cb1c55c6c4c601c4fb2647defbc955b14be3a2c23414552fbfcb",
        ),
        (false, "off") => (
            "mmlu-cs-prompt.jsonl",
            "9ba52ea797d5342a2f9ee33dd4f07c61e608fecd7060eabe34c836df36110427",
            "normal.metallib",
            "bb88c9667b4b7ac3758760b602abe40cbe72cc907bf3c40e2f2a2d33b0bd8df2",
        ),
        (false, "combined") => (
            "mmlu-cs-prompt.jsonl",
            "9ba52ea797d5342a2f9ee33dd4f07c61e608fecd7060eabe34c836df36110427",
            "metal-prefill-pipeline32x64-q4k16.metallib",
            "8a1234a3ac14cb1c55c6c4c601c4fb2647defbc955b14be3a2c23414552fbfcb",
        ),
        _ => return Err(format!("{id}: unsupported mode")),
    };
    pinned(&job, prompt_file, prompt_sha)?;
    pinned(&job, library_file, library_sha)?;
    let command = at(&job, "command")?;
    let environment = at(command, "env")?;
    let expected_selector = if mode == "combined" {
        "metal-prefill-pipeline32x64-q4k16"
    } else {
        "off"
    };
    if !command_arg(&job, "--teacher-prompt-jsonl")?.ends_with(prompt_file)
        || string(at(command, "executable")?, "sha256")? != EXECUTABLE_SHA
        || string(environment, "RVLLM_METAL_RESEARCH")? != expected_selector
        || !string(environment, "RVLLM_METAL_METALLIB_BF16")?.ends_with(library_file)
    {
        return Err(format!("{id}: command identity or route mismatch"));
    }
    let trial = read(&trial_path)?;
    if string(&trial, "schema")? != "rvllm.apple_metal_text_infer.v1"
        || string(&trial, "metal_compute_dtype")? != "bfloat16"
        || string(&trial, "metal_weight_dtype")? != "bfloat16"
        || string(&trial, "model_dir")? != command_arg(&job, "--model-dir")?
    {
        return Err(format!("{id}: wrong trial schema or dtype"));
    }
    let counts = at(at(&trial, "research_dispatch")?, "counts")?
        .as_object()
        .ok_or("dispatch counts are not an object")?;
    let expected = if mode == "combined" {
        &COMBINED_DISPATCH[..]
    } else {
        &[][..]
    };
    if counts.len() != expected.len()
        || expected
            .iter()
            .any(|(name, value)| counts.get(*name).and_then(Value::as_u64) != Some(*value))
    {
        return Err(format!("{id}: wrong actual research dispatch"));
    }
    if at(at(&trial, "research_dispatch")?, "overflowed")?.as_bool() != Some(false) {
        return Err(format!("{id}: dispatch ledger overflowed"));
    }
    let receipt_sha256 = [
        ("job", job_path.as_path()),
        ("report", report_path.as_path()),
        ("trial_stdout", trial_path.as_path()),
        ("trial_stderr", stderr_path.as_path()),
        ("conditions", conditions_path.as_path()),
    ]
    .into_iter()
    .map(|(name, path)| digest(path).map(|hash| (name, hash)))
    .collect::<Result<_>>()?;
    Ok(Arm {
        report,
        job,
        trial,
        receipt_sha256,
    })
}

fn scored_steps(arm: &Arm, prompt: &[u32], targets: &[u32]) -> Result<(Vec<Value>, f64)> {
    if command_arg(&arm.job, "--teacher-token-ids")?
        != targets
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
        || command_arg(&arm.job, "--max-new-tokens")? != targets.len().to_string()
    {
        return Err("predeclared command target IDs/count mismatch".into());
    }
    if ids(&arm.trial, "prompt_token_ids")? != prompt
        || ids(&arm.trial, "generated_token_ids")? != targets
    {
        return Err("prompt/forced-target trajectory mismatch".into());
    }
    let teacher = at(&arm.trial, "teacher_forced")?;
    if string(teacher, "schema")? != "rvllm.metal_teacher_forced_quality.v1" {
        return Err("wrong teacher schema".into());
    }
    let steps = at(teacher, "steps")?
        .as_array()
        .ok_or("steps is not an array")?;
    if steps.len() != targets.len() {
        return Err("teacher step count mismatch".into());
    }
    let mut sum = 0.0;
    for (step, target) in steps.iter().zip(targets) {
        if at(step, "target_token_id")?.as_u64() != Some(u64::from(*target))
            || at(step, "target_rank")?
                .as_u64()
                .is_none_or(|rank| rank == 0)
            || at(step, "sampled_token_id")?.as_u64().is_none()
        {
            return Err("invalid teacher target/rank/greedy ID".into());
        }
        let nll = at(step, "negative_log_likelihood")?
            .as_f64()
            .ok_or("NLL not numeric")?;
        if !nll.is_finite() || nll < 0.0 {
            return Err("NLL not finite and nonnegative".into());
        }
        sum += nll;
    }
    let reported = at(teacher, "total_negative_log_likelihood")?
        .as_f64()
        .ok_or("total NLL not numeric")?;
    if (sum - reported).abs() > 1e-9 {
        return Err("teacher aggregate NLL does not match steps".into());
    }
    Ok((steps.clone(), sum))
}

fn summarize_case(
    name: &str,
    prompt: &[u32],
    targets: &[u32],
    control: &Arm,
    candidate: &Arm,
) -> Result<Value> {
    let (control_steps, control_sum) = scored_steps(control, prompt, targets)?;
    let (candidate_steps, candidate_sum) = scored_steps(candidate, prompt, targets)?;
    let positions = control_steps
        .iter()
        .zip(&candidate_steps)
        .enumerate()
        .map(|(index, (a, b))| {
            let control_nll = at(a, "negative_log_likelihood")?
                .as_f64()
                .ok_or("control NLL not numeric")?;
            let candidate_nll = at(b, "negative_log_likelihood")?
                .as_f64()
                .ok_or("candidate NLL not numeric")?;
            Ok(json!({
                "index": index,
                "target_token_id": targets[index],
                "control_nll": control_nll,
                "candidate_nll": candidate_nll,
                "candidate_minus_control_nll": candidate_nll - control_nll,
                "control_rank": a["target_rank"],
                "candidate_rank": b["target_rank"],
                "control_greedy_id": a["sampled_token_id"],
                "candidate_greedy_id": b["sampled_token_id"],
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let rank_differences = positions
        .iter()
        .filter(|position| position["control_rank"] != position["candidate_rank"])
        .count();
    let greedy_differences = positions
        .iter()
        .filter(|position| position["control_greedy_id"] != position["candidate_greedy_id"])
        .count();
    let first_greedy_difference = positions
        .iter()
        .find(|position| position["control_greedy_id"] != position["candidate_greedy_id"])
        .map(|position| position["index"].clone());
    Ok(json!({
        "id": name,
        "prompt_tokens": prompt.len(),
        "target_tokens": targets.len(),
        "control_total_nll": control_sum,
        "candidate_total_nll": candidate_sum,
        "control_mean_nll": control_sum / targets.len() as f64,
        "candidate_mean_nll": candidate_sum / targets.len() as f64,
        "candidate_minus_control_total_nll": candidate_sum - control_sum,
        "rank_differences": rank_differences,
        "greedy_differences": greedy_differences,
        "first_greedy_difference": first_greedy_difference,
        "control_sampled_conditions_eligible": control.report["sampled_conditions_eligible"],
        "candidate_sampled_conditions_eligible": candidate.report["sampled_conditions_eligible"],
        "control_stratum": control.report["measurement"]["comparison_stratum"],
        "candidate_stratum": candidate.report["measurement"]["comparison_stratum"],
        "control_executable_sha256": control.job["command"]["executable"]["sha256"],
        "candidate_executable_sha256": candidate.job["command"]["executable"]["sha256"],
        "control_receipts_sha256": control.receipt_sha256,
        "candidate_receipts_sha256": candidate.receipt_sha256,
        "positions": positions,
    }))
}

fn run(paths: &[PathBuf]) -> Result<Value> {
    let tokenizer_sha = digest(&paths[1])?;
    let source_sha = digest(&paths[2])?;
    if source_sha != SOURCE_SHA || tokenizer_sha != TOKENIZER_SHA {
        return Err("source or tokenizer is not the predeclared MMLU trial".into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&paths[1])
        .map_err(|error| format!("load tokenizer: {error}"))?;
    let source = read(&paths[2])?;
    if string(&source, "schema")? != "rvllm.gemma4_heldout_text.v1" {
        return Err("wrong source schema".into());
    }
    let arms = IDS
        .iter()
        .zip(&paths[3..])
        .map(|((_, mode, id), dir)| read_arm(dir, id, mode, &tokenizer_sha, &source_sha))
        .collect::<Result<Vec<_>>>()?;
    let mut cases = Vec::new();
    for (index, case_id) in [
        "mmlu-formal-logic-2443",
        "mmlu-high-school-computer-science-3185",
    ]
    .iter()
    .enumerate()
    {
        let (prompt, targets) = expected_case(&tokenizer, &source, case_id)?;
        cases.push(summarize_case(
            case_id,
            &prompt,
            &targets,
            &arms[2 * index],
            &arms[2 * index + 1],
        )?);
    }
    Ok(json!({
        "schema": "rvllm.gemma4_teacher_summary.v1",
        "claim": "Two preselected MMLU test questions in a Codex-authored wrapper; not an independent numerical reference, timing evidence, benchmark score or quality acceptance",
        "source_sha256": source_sha,
        "tokenizer_sha256": tokenizer_sha,
        "cases": cases,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_field() {
        assert!(at(&json!({}), "missing").is_err());
    }

    #[test]
    fn token_id_parser_rejects_non_integral() {
        assert!(ids(&json!({"x":[1.5]}), "x").is_err());
    }

    #[test]
    fn expected_dispatch_has_all_six_roles() {
        assert_eq!(COMBINED_DISPATCH.len(), 6);
        assert_eq!(
            COMBINED_DISPATCH
                .iter()
                .map(|(_, count)| count)
                .sum::<u64>(),
            336
        );
    }

    #[test]
    fn teacher_steps_must_match_forced_targets_and_aggregate() {
        let mut arm = Arm {
            report: json!({}),
            job: json!({"command":{"args":["--teacher-token-ids","7,8","--max-new-tokens","2"]}}),
            trial: json!({
                "prompt_token_ids":[2,3],
                "generated_token_ids":[7,8],
                "teacher_forced":{
                    "schema":"rvllm.metal_teacher_forced_quality.v1",
                    "steps":[
                        {"target_token_id":7,"target_rank":1,"sampled_token_id":7,"negative_log_likelihood":0.25},
                        {"target_token_id":8,"target_rank":2,"sampled_token_id":9,"negative_log_likelihood":0.5}
                    ],
                    "total_negative_log_likelihood":0.75
                }
            }),
            receipt_sha256: BTreeMap::new(),
        };
        assert_eq!(scored_steps(&arm, &[2, 3], &[7, 8]).unwrap().1, 0.75);
        arm.trial["teacher_forced"]["steps"][1]["target_token_id"] = json!(10);
        assert!(scored_steps(&arm, &[2, 3], &[7, 8]).is_err());
        arm.trial["teacher_forced"]["steps"][1]["target_token_id"] = json!(8);
        arm.trial["teacher_forced"]["total_negative_log_likelihood"] = json!(0.7);
        assert!(scored_steps(&arm, &[2, 3], &[7, 8]).is_err());
    }
}
