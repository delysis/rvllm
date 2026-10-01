//! Fail-closed comparison of sealed, one-layer W4/W8 teacher trials.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SOURCE_SHA: &str = "a1d115405e86ab51bf275473cc04f452f185db2c18f286b5173d2234d4b993b4";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const ARMS: [(&str, &str, &str, &str); 6] = [
    (
        "mmlu-formal-logic-2443",
        "native",
        "logic-native-job.json",
        "8bb718fcfbb59f4701567df8b5c2be581ab75d967494ee210d21ab514a061527",
    ),
    (
        "mmlu-formal-logic-2443",
        "w4",
        "logic-w4-job.json",
        "9ae959ab1d442f2eeacab1c1c6c093dd0237bf03004569a84ddfea290efd07f4",
    ),
    (
        "mmlu-formal-logic-2443",
        "w8",
        "logic-w8-job.json",
        "83c011b2b393e16da767f4627555625a80e3952172b7c2c29735e0dcb10780ac",
    ),
    (
        "mmlu-high-school-computer-science-3185",
        "native",
        "cs-native-job.json",
        "85391d90f0baa74ac45c1258ccb4de6369b8bbad771305629c3375d19cb058ef",
    ),
    (
        "mmlu-high-school-computer-science-3185",
        "w4",
        "cs-w4-job.json",
        "e75cde4e1ad13505d2f7960026a99dee7a024c77dd27bcc0b310c8204d2f5c57",
    ),
    (
        "mmlu-high-school-computer-science-3185",
        "w8",
        "cs-w8-job.json",
        "60362b42eb9214565fa925a1570d8b97d9f4f4de61d1b2c924f4a85f8dbdf66b",
    ),
];

type Result<T> = std::result::Result<T, String>;

struct Arm {
    report: Value,
    trial: Value,
    receipts: BTreeMap<&'static str, String>,
}

fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    if args.len() != 11 {
        eprintln!("usage: rvllm_gemma4_lowbit_teacher_summary OUTPUT.json TOKENIZER.json SOURCE.json MANIFEST_DIR LOGIC_NATIVE_DIR LOGIC_W4_DIR LOGIC_W8_DIR CS_NATIVE_DIR CS_W4_DIR CS_W8_DIR");
        std::process::exit(2);
    }
    let paths = args[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    match run(&paths) {
        Ok(summary) => {
            let mut bytes = serde_json::to_vec_pretty(&summary).expect("serializable summary");
            bytes.push(b'\n');
            let result = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&paths[0])
                .and_then(|mut file| file.write_all(&bytes));
            if let Err(error) = result {
                eprintln!("write {}: {error}", paths[0].display());
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("rvllm_gemma4_lowbit_teacher_summary: {error}");
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
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value.get(name).ok_or_else(|| format!("missing {name}"))
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| format!("{name} is not text"))
}

fn token_ids(value: &Value, name: &str) -> Result<Vec<u32>> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| format!("{name} is not an array"))?
        .iter()
        .map(|token| {
            token
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("{name} contains a non-token ID"))
        })
        .collect()
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| format!("tokenize: {error}"))?;
    let mut ids = vec![2];
    ids.extend_from_slice(encoded.get_ids());
    Ok(ids)
}

fn expected_case(
    tokenizer: &tokenizers::Tokenizer,
    source: &Value,
    id: &str,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let cases = field(source, "cases")?
        .as_array()
        .ok_or("cases is not an array")?;
    let found = cases
        .iter()
        .filter(|case| case.get("id").and_then(Value::as_str) == Some(id))
        .collect::<Vec<_>>();
    if found.len() != 1 {
        return Err(format!("expected exactly one source case {id}"));
    }
    let prompt = text(found[0], "prompt")?;
    let continuation = text(found[0], "continuation")?;
    let prompt_ids = encode(tokenizer, prompt)?;
    let full_ids = encode(tokenizer, &format!("{prompt}{continuation}"))?;
    let targets = full_ids
        .strip_prefix(prompt_ids.as_slice())
        .filter(|tail| !tail.is_empty())
        .ok_or_else(|| format!("{id}: retokenized at boundary"))?;
    Ok((prompt_ids, targets.to_vec()))
}

fn command_arg<'a>(job: &'a Value, flag: &str) -> Result<&'a str> {
    let args = field(field(job, "command")?, "args")?
        .as_array()
        .ok_or("command args are not an array")?;
    let matches = args
        .windows(2)
        .filter(|pair| pair[0].as_str() == Some(flag))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected exactly one {flag} argument"));
    }
    matches[0][1]
        .as_str()
        .ok_or_else(|| format!("{flag} argument is not text"))
}

fn dispatch_is_valid(counts: &serde_json::Map<String, Value>, route: &str) -> bool {
    let suffix = match route {
        "native" => None,
        "w4" => Some("w4"),
        "w8" => Some("w8"),
        _ => return false,
    };
    let native_present = counts
        .get("research_donor12b_sg8_native_projection")
        .and_then(Value::as_u64)
        .is_some_and(|count| count > 0);
    let attention_present = counts
        .get("research_donor12b_sg8_global_attention")
        .and_then(Value::as_u64)
        .is_some_and(|count| count > 0);
    let bit_dispatch = |bit: &str| {
        counts
            .get(&format!("research_donor12b_sg8_{bit}"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
    };
    native_present
        && attention_present
        && match suffix {
            None => !bit_dispatch("w4") && !bit_dispatch("w8"),
            Some("w4") => bit_dispatch("w4") && !bit_dispatch("w8"),
            Some("w8") => bit_dispatch("w8") && !bit_dispatch("w4"),
            _ => false,
        }
}

fn read_arm(dir: &Path, manifest: &Path, manifest_sha: &str, route: &str, id: &str) -> Result<Arm> {
    if digest(manifest)? != manifest_sha {
        return Err(format!("{id}: frozen manifest SHA changed"));
    }
    let mut expected_job = read(manifest)?;
    // The serial queue inserts only these two absent optional fields.
    expected_job["kernel_game_submission"] = Value::Null;
    expected_job["validator"] = Value::Null;
    let job_path = dir.join("job.json");
    let job = read(&job_path)?;
    if job != expected_job || text(&job, "id")? != id {
        return Err(format!("{id}: queue job differs from frozen manifest"));
    }
    let report_path = dir.join("report.json");
    let report = read(&report_path)?;
    if text(&report, "id")? != id
        || text(&report, "status")? != "succeeded"
        || field(&report, "exit_code")?.as_i64() != Some(0)
        || field(&report, "files_unchanged")?.as_bool() != Some(true)
        || field(&report, "sampled_conditions_eligible")?
            .as_bool()
            .is_none()
        || !field(&report, "violations")?
            .as_array()
            .is_some_and(Vec::is_empty)
    {
        return Err(format!("{id}: queue receipt is not clean and terminal"));
    }
    let trial_path = dir.join("trial.stdout");
    let trial = read(&trial_path)?;
    if text(&trial, "schema")? != "rvllm.apple_metal_text_infer.v1"
        || text(&trial, "metal_compute_dtype")? != "bfloat16"
        || text(&trial, "metal_weight_dtype")? != "bfloat16"
        || text(&trial, "model_dir")? != command_arg(&job, "--model-dir")?
    {
        return Err(format!("{id}: wrong trial identity or dtype"));
    }
    let dispatch = field(&trial, "research_dispatch")?;
    let counts = field(dispatch, "counts")?
        .as_object()
        .ok_or("dispatch counts are not an object")?;
    if field(dispatch, "overflowed")?.as_bool() != Some(false) || !dispatch_is_valid(counts, route)
    {
        return Err(format!("{id}: expected donor/low-bit dispatch is missing"));
    }
    let stderr_path = dir.join("trial.stderr");
    let conditions_path = dir.join("conditions.jsonl");
    let receipts = [
        ("job", job_path.as_path()),
        ("report", report_path.as_path()),
        ("trial_stdout", trial_path.as_path()),
        ("trial_stderr", stderr_path.as_path()),
        ("conditions", conditions_path.as_path()),
    ];
    let receipts = receipts
        .into_iter()
        .map(|(name, path)| digest(path).map(|hash| (name, hash)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok(Arm {
        report,
        trial,
        receipts,
    })
}

fn scored_steps(arm: &Arm, prompt: &[u32], targets: &[u32]) -> Result<(Vec<Value>, f64)> {
    if token_ids(&arm.trial, "prompt_token_ids")? != prompt
        || token_ids(&arm.trial, "generated_token_ids")? != targets
    {
        return Err("prompt or forced target IDs changed".into());
    }
    let teacher = field(&arm.trial, "teacher_forced")?;
    if text(teacher, "schema")? != "rvllm.metal_teacher_forced_quality.v1" {
        return Err("wrong teacher schema".into());
    }
    let steps = field(teacher, "steps")?
        .as_array()
        .ok_or("steps is not an array")?;
    if steps.len() != targets.len() {
        return Err("teacher step count mismatch".into());
    }
    let mut sum = 0.0;
    for (step, target) in steps.iter().zip(targets) {
        if field(step, "target_token_id")?.as_u64() != Some(u64::from(*target))
            || field(step, "target_rank")?
                .as_u64()
                .is_none_or(|rank| rank == 0)
            || field(step, "sampled_token_id")?.as_u64().is_none()
        {
            return Err("invalid target, rank or sampled ID".into());
        }
        let nll = field(step, "negative_log_likelihood")?
            .as_f64()
            .ok_or("NLL not numeric")?;
        if !nll.is_finite() || nll < 0.0 {
            return Err("NLL not finite and nonnegative".into());
        }
        sum += nll;
    }
    let reported = field(teacher, "total_negative_log_likelihood")?
        .as_f64()
        .ok_or("total NLL not numeric")?;
    if (sum - reported).abs() > 1e-9 {
        return Err("teacher aggregate NLL differs from steps".into());
    }
    Ok((steps.clone(), sum))
}

fn summarize_case(id: &str, prompt: &[u32], targets: &[u32], arms: &[Arm]) -> Result<Value> {
    let scored = arms
        .iter()
        .map(|arm| scored_steps(arm, prompt, targets))
        .collect::<Result<Vec<_>>>()?;
    let positions = (0..targets.len())
        .map(|index| {
            let rows = scored
                .iter()
                .map(|(steps, _)| &steps[index])
                .collect::<Vec<_>>();
            let nll = rows
                .iter()
                .map(|step| {
                    step["negative_log_likelihood"]
                        .as_f64()
                        .expect("validated NLL")
                })
                .collect::<Vec<_>>();
            json!({
                "index": index,
                "target_token_id": targets[index],
                "native_nll": nll[0],
                "w4_nll": nll[1],
                "w8_nll": nll[2],
                "w4_minus_native_nll": nll[1] - nll[0],
                "w8_minus_native_nll": nll[2] - nll[0],
                "native_rank": rows[0]["target_rank"],
                "w4_rank": rows[1]["target_rank"],
                "w8_rank": rows[2]["target_rank"],
                "native_greedy_id": rows[0]["sampled_token_id"],
                "w4_greedy_id": rows[1]["sampled_token_id"],
                "w8_greedy_id": rows[2]["sampled_token_id"],
            })
        })
        .collect::<Vec<_>>();
    let summaries = ["native", "w4", "w8"]
        .iter()
        .zip(arms)
        .zip(&scored)
        .map(|((route, arm), (_, sum))| {
            json!({
                "route": route,
                "total_nll": sum,
                "mean_nll": sum / targets.len() as f64,
                "sampled_conditions_eligible": arm.report["sampled_conditions_eligible"],
                "comparison_stratum": arm.report["measurement"]["comparison_stratum"],
                "research_dispatch": arm.trial["research_dispatch"],
                "receipt_sha256": arm.receipts,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "id": id,
        "prompt_tokens": prompt.len(),
        "target_tokens": targets.len(),
        "arms": summaries,
        "w4_minus_native_total_nll": scored[1].1 - scored[0].1,
        "w8_minus_native_total_nll": scored[2].1 - scored[0].1,
        "w4_rank_differences": positions.iter().filter(|p| p["w4_rank"] != p["native_rank"]).count(),
        "w8_rank_differences": positions.iter().filter(|p| p["w8_rank"] != p["native_rank"]).count(),
        "w4_greedy_differences": positions.iter().filter(|p| p["w4_greedy_id"] != p["native_greedy_id"]).count(),
        "w8_greedy_differences": positions.iter().filter(|p| p["w8_greedy_id"] != p["native_greedy_id"]).count(),
        "positions": positions,
    }))
}

fn run(paths: &[PathBuf]) -> Result<Value> {
    if digest(&paths[1])? != TOKENIZER_SHA || digest(&paths[2])? != SOURCE_SHA {
        return Err("tokenizer or source changed from the sealed trial".into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&paths[1])
        .map_err(|error| format!("load tokenizer: {error}"))?;
    let source = read(&paths[2])?;
    if text(&source, "schema")? != "rvllm.gemma4_heldout_text.v1" {
        return Err("wrong source schema".into());
    }
    let arms = ARMS
        .iter()
        .zip(&paths[4..])
        .map(|((_, route, file, sha), dir)| {
            let manifest = paths[3].join(file);
            let id = format!(
                "prefill26-lowbit-mmlu-{}-{route}-v1-20260927",
                if file.starts_with("logic-") {
                    "logic"
                } else {
                    "cs"
                }
            );
            read_arm(dir, &manifest, sha, route, &id)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut cases = Vec::new();
    for (case_index, case_id) in [
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
            &arms[3 * case_index..3 * case_index + 3],
        )?);
    }
    Ok(json!({
        "schema": "rvllm.gemma4_lowbit_teacher_summary.v1",
        "claim": "One original-BF16-checkpoint layer0 down projection sidecar; donor decode dispatch only; not whole-checkpoint quality, independent numerical reference, benchmark score or timing evidence",
        "source_sha256": SOURCE_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "cases": cases,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_named_low_bit_decode_dispatch() {
        let mut counts = serde_json::Map::new();
        counts.insert("research_donor12b_sg8_native_projection".into(), json!(1));
        counts.insert("research_donor12b_sg8_global_attention".into(), json!(1));
        assert!(dispatch_is_valid(&counts, "native"));
        assert!(!dispatch_is_valid(&counts, "w4"));
        counts.insert("research_donor12b_sg8_w4".into(), json!(1));
        assert!(dispatch_is_valid(&counts, "w4"));
        assert!(!dispatch_is_valid(&counts, "native"));
        assert!(!dispatch_is_valid(&counts, "w8"));
    }

    #[test]
    fn malformed_step_rejected() {
        let arm = Arm {
            report: json!({}),
            trial: json!({"prompt_token_ids":[2],"generated_token_ids":[3],"teacher_forced":{
                "schema":"rvllm.metal_teacher_forced_quality.v1",
                "steps":[{"target_token_id":3,"target_rank":0,"sampled_token_id":3,"negative_log_likelihood":0.1}],
                "total_negative_log_likelihood":0.1}}),
            receipts: BTreeMap::new(),
        };
        assert!(scored_steps(&arm, &[2], &[3]).is_err());
    }
}
