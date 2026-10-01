//! Fail-closed, same-boundary comparison of six frozen HF/Metal prefill jobs.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "a1d115405e86ab51bf275473cc04f452f185db2c18f286b5173d2234d4b993b4";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const ARMS: [(&str, &str, &str, &str); 6] = [
    (
        "mmlu-formal-logic-2443",
        "hf",
        "logic-hf-job.json",
        "144965b1c3dc4967b3d828c66b404921e94631f269760e21aa6512c6d3fab4db",
    ),
    (
        "mmlu-formal-logic-2443",
        "off",
        "logic-off-job.json",
        "c839bdc8831c80f30160bbea9ad96fa9790f9232b393c1cd98ce39223a35f1a2",
    ),
    (
        "mmlu-formal-logic-2443",
        "combined",
        "logic-combined-job.json",
        "e1a50e88d764167331ffe08f4033d9c05cdeec797b752b7301b06cc562021965",
    ),
    (
        "mmlu-high-school-computer-science-3185",
        "hf",
        "cs-hf-job.json",
        "8c69077b89279f8f3fffd858856cd90bbeed5564817543b9d3c82d001fcc1a3b",
    ),
    (
        "mmlu-high-school-computer-science-3185",
        "off",
        "cs-off-job.json",
        "b55e9141f8869fd29fbb610c5d5c53d0e69790f1e7a752749a514f5deb5f1a60",
    ),
    (
        "mmlu-high-school-computer-science-3185",
        "combined",
        "cs-combined-job.json",
        "527d8fc05baf31732485ccffb62188d9c7384c941a9d372c5cb4e21f6c6038f4",
    ),
];

struct Arm {
    report: Value,
    score: Value,
    receipts: BTreeMap<&'static str, String>,
}

fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    if args.len() != 11 {
        eprintln!("usage: rvllm_gemma4_prefill_reference_summary OUTPUT.json TOKENIZER.json SOURCE.json MANIFEST_DIR LOGIC_HF_DIR LOGIC_OFF_DIR LOGIC_COMBINED_DIR CS_HF_DIR CS_OFF_DIR CS_COMBINED_DIR");
        std::process::exit(2);
    }
    let paths = args[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    let result = run(&paths).and_then(|summary| {
        let mut bytes = serde_json::to_vec_pretty(&summary).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&paths[0])
            .and_then(|mut file| file.write_all(&bytes))
            .map_err(|error| format!("{}: {error}", paths[0].display()))
    });
    if let Err(error) = result {
        eprintln!("rvllm_gemma4_prefill_reference_summary: {error}");
        std::process::exit(1);
    }
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|value| value.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn read(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse_strict_json(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value.get(key).ok_or_else(|| format!("missing {key}"))
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not text"))
}

fn number(value: &Value, key: &str) -> Result<f64> {
    let result = field(value, key)?
        .as_f64()
        .ok_or_else(|| format!("{key} is not numeric"))?;
    if !result.is_finite() {
        return Err(format!("{key} is not finite"));
    }
    Ok(result)
}

fn token_ids(value: &Value, key: &str) -> Result<Vec<u32>> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not an array"))?
        .iter()
        .map(|token| {
            token
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("{key} contains a non-token ID"))
        })
        .collect()
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
        return Err(format!("expected exactly one {flag}"));
    }
    matches[0][1]
        .as_str()
        .ok_or_else(|| format!("{flag} is not text"))
}

fn expected_case(
    tokenizer: &tokenizers::Tokenizer,
    source: &Value,
    id: &str,
) -> Result<(Vec<u32>, u32)> {
    let cases = field(source, "cases")?
        .as_array()
        .ok_or("source cases are not an array")?;
    let found = cases
        .iter()
        .filter(|case| case.get("id").and_then(Value::as_str) == Some(id))
        .collect::<Vec<_>>();
    if found.len() != 1 {
        return Err(format!("expected one source case {id}"));
    }
    let prompt = text(found[0], "prompt")?;
    let continuation = text(found[0], "continuation")?;
    let encode = |value: &str| -> Result<Vec<u32>> {
        let encoded = tokenizer
            .encode(value, false)
            .map_err(|error| error.to_string())?;
        let mut ids = vec![2];
        ids.extend_from_slice(encoded.get_ids());
        Ok(ids)
    };
    let prompt_ids = encode(prompt)?;
    let full_ids = encode(&format!("{prompt}{continuation}"))?;
    let targets = full_ids
        .strip_prefix(prompt_ids.as_slice())
        .filter(|tail| !tail.is_empty())
        .ok_or("continuation retokenized at boundary")?;
    Ok((prompt_ids, targets[0]))
}

fn validated_receipt(
    dir: &Path,
    manifest: &Path,
    sha: &str,
    id: &str,
) -> Result<(Value, Value, BTreeMap<&'static str, String>)> {
    if digest(manifest)? != sha {
        return Err(format!("{id}: frozen manifest changed"));
    }
    let mut expected_job = read(manifest)?;
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
    let receipts = [
        ("job", job_path),
        ("report", report_path),
        ("trial_stdout", dir.join("trial.stdout")),
        ("trial_stderr", dir.join("trial.stderr")),
        ("conditions", dir.join("conditions.jsonl")),
    ]
    .into_iter()
    .map(|(name, path)| digest(&path).map(|sha| (name, sha)))
    .collect::<Result<BTreeMap<_, _>>>()?;
    Ok((job, report, receipts))
}

fn hf_score(reference: &Value, prompt: &[u32], target: u32) -> Result<Value> {
    if text(reference, "schema")? != "rvllm.gemma4_hf_reference_logits.v1"
        || token_ids(reference, "prompt_token_ids")? != prompt
        || field(reference, "decode_steps")?.as_u64() != Some(1)
        || field(reference, "full_logits")?.as_bool() != Some(true)
        || token_ids(reference, "selected_token_ids")? != [target]
    {
        return Err("HF reference identity or shape differs".into());
    }
    let steps = field(reference, "steps")?
        .as_array()
        .ok_or("HF steps are not an array")?;
    if steps.len() != 1 {
        return Err("HF did not emit exactly one step".into());
    }
    let step = &steps[0];
    let logits = field(step, "logits")?
        .as_array()
        .ok_or("HF logits are not an array")?;
    if logits.len() != 262_144 || target as usize >= logits.len() {
        return Err("HF vocabulary or target is invalid".into());
    }
    let values = logits
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or("HF logit is nonfinite or not numeric".into())
        })
        .collect::<Result<Vec<f64>>>()?;
    let target_logit = values[target as usize];
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum = values.iter().map(|value| (value - max).exp()).sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("HF log-sum-exp is invalid".into());
    }
    let nll = max + sum.ln() - target_logit;
    let rank = 1 + values
        .iter()
        .enumerate()
        .filter(|(index, value)| {
            **value > target_logit || (**value == target_logit && *index < target as usize)
        })
        .count();
    let greedy = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(index, _)| index as u32)
        .ok_or("HF logits empty")?;
    if field(step, "next_token")?.as_u64() != Some(u64::from(greedy))
        || token_ids(reference, "generated_tokens")? != [greedy]
    {
        return Err("HF reported greedy ID differs from full logits".into());
    }
    let selected = field(step, "selected_logits")?
        .as_array()
        .ok_or("HF selected logits missing")?;
    if selected.len() != 1
        || field(&selected[0], "token_id")?.as_u64() != Some(u64::from(target))
        || number(&selected[0], "logit")? != target_logit
    {
        return Err("HF selected logit differs from full vector".into());
    }
    Ok(
        json!({"target_token_id": target, "target_logit": target_logit,
        "target_rank": rank, "negative_log_likelihood": nll, "sampled_token_id": greedy}),
    )
}

fn metal_score(trial: &Value, prompt: &[u32], target: u32, route: &str) -> Result<Value> {
    if text(trial, "schema")? != "rvllm.apple_metal_text_infer.v1"
        || token_ids(trial, "prompt_token_ids")? != prompt
        || token_ids(trial, "generated_token_ids")? != [target]
        || text(trial, "metal_compute_dtype")? != "bfloat16"
        || text(trial, "metal_weight_dtype")? != "bfloat16"
    {
        return Err("Metal trial identity, prompt or forced target differs".into());
    }
    let dispatch = field(trial, "research_dispatch")?;
    if text(dispatch, "schema")? != "rvllm.metal.research-dispatch.v6"
        || field(dispatch, "overflowed")?.as_bool() != Some(false)
    {
        return Err("Metal research dispatch overflowed".into());
    }
    let counts = field(dispatch, "counts")?
        .as_object()
        .ok_or("dispatch counts missing")?;
    match route {
        "off" if !counts.is_empty() => return Err("control used research kernels".into()),
        "combined" => {
            for (name, expected) in [
                ("research_prefill_combined_gemm", 48),
                ("research_prefill_combined_qkv", 48),
                ("research_prefill_combined_raw_norm_projection", 96),
                ("research_prefill_combined_raw_norm", 96),
                ("research_prefill_combined_d256", 40),
                ("research_prefill_combined_d512", 8),
            ] {
                if counts.get(name).and_then(Value::as_u64) != Some(expected) {
                    return Err(format!("combined route missing exact {name} dispatch"));
                }
            }
            if counts.len() != 6 {
                return Err("unexpected combined research dispatch".into());
            }
        }
        "off" => {}
        _ => return Err("unknown Metal route".into()),
    }
    let teacher = field(trial, "teacher_forced")?;
    if text(teacher, "schema")? != "rvllm.metal_teacher_forced_quality.v1" {
        return Err("wrong Metal teacher schema".into());
    }
    let steps = field(teacher, "steps")?
        .as_array()
        .ok_or("Metal teacher steps are not an array")?;
    if steps.len() != 1 || field(&steps[0], "target_token_id")?.as_u64() != Some(u64::from(target))
    {
        return Err("Metal teacher decode step count or target differs".into());
    }
    let step = field(teacher, "prefill_last_step")?;
    if field(step, "target_token_id")?.as_u64() != Some(u64::from(target))
        || field(step, "target_rank")?
            .as_u64()
            .map_or(true, |rank| rank == 0)
        || field(step, "sampled_token_id")?.as_u64().is_none()
        || number(step, "negative_log_likelihood")? < 0.0
    {
        return Err("Metal prefill-last score invalid".into());
    }
    Ok(step.clone())
}

fn read_arm(
    dir: &Path,
    manifest: &Path,
    sha: &str,
    id: &str,
    route: &str,
    prompt: &[u32],
    target: u32,
) -> Result<Arm> {
    let (job, report, mut receipts) = validated_receipt(dir, manifest, sha, id)?;
    let score = if route == "hf" {
        let output = PathBuf::from(command_arg(&job, "--output")?);
        let reference = read(&output)?;
        let model = field(field(&job, "command")?, "args")?
            .as_array()
            .ok_or("HF arguments missing")?
            .get(1)
            .and_then(Value::as_str)
            .ok_or("HF model argument missing")?;
        if text(&reference, "model_dir")? != model {
            return Err("HF output model differs from pinned command".into());
        }
        let score = hf_score(&reference, prompt, target)?;
        receipts.insert("hf_full_reference", digest(&output)?);
        score
    } else {
        let trial = read(&dir.join("trial.stdout"))?;
        if text(&trial, "model_dir")? != command_arg(&job, "--model-dir")? {
            return Err("Metal output model differs from pinned command".into());
        }
        metal_score(&trial, prompt, target, route)?
    };
    Ok(Arm {
        report,
        score,
        receipts,
    })
}

fn run(paths: &[PathBuf]) -> Result<Value> {
    if digest(&paths[1])? != TOKENIZER_SHA || digest(&paths[2])? != SOURCE_SHA {
        return Err("tokenizer or source changed from frozen trial".into());
    }
    let tokenizer =
        tokenizers::Tokenizer::from_file(&paths[1]).map_err(|error| error.to_string())?;
    let source = read(&paths[2])?;
    if text(&source, "schema")? != "rvllm.gemma4_heldout_text.v1" {
        return Err("wrong source schema".into());
    }
    let mut cases = Vec::new();
    for (index, (slug, case_id)) in [
        ("logic", "mmlu-formal-logic-2443"),
        ("cs", "mmlu-high-school-computer-science-3185"),
    ]
    .into_iter()
    .enumerate()
    {
        let (prompt, target) = expected_case(&tokenizer, &source, case_id)?;
        let arms = ARMS[index * 3..index * 3 + 3]
            .iter()
            .zip(&paths[4 + index * 3..7 + index * 3])
            .map(|((frozen_case, route, file, sha), dir)| {
                if *frozen_case != case_id {
                    return Err("frozen case order differs".into());
                }
                let id = format!("prefill26-mmlu-prefill-ref-{slug}-{route}-v1-20260927");
                read_arm(dir, &paths[3].join(file), sha, &id, route, &prompt, target)
            })
            .collect::<Result<Vec<_>>>()?;
        let comparison = |route_index: usize| -> Result<Value> {
            let hf = &arms[0].score;
            let metal = &arms[route_index].score;
            Ok(json!({
                "target_logit_minus_hf": number(metal, "target_logit")? - number(hf, "target_logit")?,
                "nll_minus_hf": number(metal, "negative_log_likelihood")? - number(hf, "negative_log_likelihood")?,
                "rank_matches_hf": field(metal, "target_rank")? == field(hf, "target_rank")?,
                "greedy_matches_hf": field(metal, "sampled_token_id")? == field(hf, "sampled_token_id")?,
            }))
        };
        cases.push(json!({
            "id": case_id, "prompt_tokens": prompt.len(), "target_token_id": target,
            "arms": arms.iter().zip(["hf", "off", "combined"]).map(|(arm, route)| json!({
                "route": route, "score": arm.score,
                "sampled_conditions_eligible": arm.report["sampled_conditions_eligible"],
                "comparison_stratum": arm.report["measurement"]["comparison_stratum"],
                "receipt_sha256": arm.receipts,
            })).collect::<Vec<_>>(),
            "off_vs_hf": comparison(1)?, "combined_vs_hf": comparison(2)?,
        }));
    }
    Ok(json!({
        "schema": "rvllm.gemma4_prefill_reference_summary.v1",
        "claim": "Two same-token-boundary next-token positions; Metal exposes only target logit/rank/NLL/greedy ID; no full-vector Metal parity, continuation quality, speed, benchmark score or promotion",
        "source_sha256": SOURCE_SHA, "tokenizer_sha256": TOKENIZER_SHA, "cases": cases,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_manifest_hashes_match_checkout() {
        let manifests = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../reports/prefill-round-fullroute-short-20260927/mmlu-prefill-reference-v1-queue",
        );
        for (_, _, file, expected_sha) in ARMS {
            assert_eq!(
                digest(&manifests.join(file)).unwrap(),
                expected_sha,
                "{file}"
            );
        }
    }

    #[test]
    fn hf_ties_use_lowest_token_id() {
        let mut logits = vec![0.0; 262_144];
        logits[3] = 2.0;
        logits[8] = 2.0;
        let reference = json!({"schema":"rvllm.gemma4_hf_reference_logits.v1",
        "prompt_token_ids":[2,10], "decode_steps":1, "full_logits":true,
        "selected_token_ids":[8], "generated_tokens":[3], "steps":[{
            "next_token":3, "selected_logits":[{"token_id":8,"logit":2.0}], "logits":logits
        }]});
        let score = hf_score(&reference, &[2, 10], 8).unwrap();
        assert_eq!(score["target_rank"], 2);
        assert_eq!(score["sampled_token_id"], 3);
    }

    #[test]
    fn rejects_missing_combined_dispatch() {
        let trial = json!({"schema":"rvllm.apple_metal_text_infer.v1",
            "prompt_token_ids":[2], "generated_token_ids":[3],
            "metal_compute_dtype":"bfloat16", "metal_weight_dtype":"bfloat16",
            "research_dispatch":{"schema":"rvllm.metal.research-dispatch.v6","overflowed":false,"counts":{}},
            "teacher_forced":{"schema":"rvllm.metal_teacher_forced_quality.v1",
                "steps":[{"target_token_id":3}],
                "prefill_last_step":{"target_token_id":3,"sampled_token_id":3,
                    "target_logit":1.0,"target_rank":1,"negative_log_likelihood":0.1}}});
        assert!(metal_score(&trial, &[2], 3, "combined").is_err());
    }
}
