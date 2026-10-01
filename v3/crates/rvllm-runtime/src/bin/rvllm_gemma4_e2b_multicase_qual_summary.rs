//! Fail-closed, exact-position referee for the sealed three-arm E2B session trial.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "9a8e792014de7096b0ea89c9a2440448acc00797e26676ce67dd35a4b13cb17c";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const EXECUTABLE_SHA: &str = "149d447d70227edf265c52c721534531bb69a9c31cda44b5512566054db30d06";
const MANIFEST_SHA: [&str; 3] = [
    "a50b5f9533749ca68e35d195eb49a68b0a1e213ea241ceac9b5eea88384c6b4d",
    "a0a046b120d9f066b6484651f9dd770c6b3b3b2fdac8c7cc744668f2570f92db",
    "e3e6f44952b45e663c8ace4796e8f056cf8fa84538bd17e1cec040ab9aa85c26",
];
const MANIFEST_FILES: [&str; 3] = [
    "observatory-single-job.json",
    "kitchen-single-job.json",
    "two-case-batch-job.json",
];
const CASES: [(&str, &[u32]); 2] = [
    ("observatory", &[108355, 236761]),
    ("kitchen", &[6819, 236761]),
];

struct Arm {
    job: Value,
    trial: Value,
    receipts: BTreeMap<&'static str, String>,
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|value| value.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn read(path: &Path) -> Result<Value> {
    parse_strict_json(&fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?)
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key}: missing array"))
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key}: missing text"))
}

fn ids(value: &Value, key: &str) -> Result<Vec<u32>> {
    array(value, key)?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("{key}: non-token"))
        })
        .collect()
}

fn finite(value: &Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("{key}: missing or nonfinite"))
}

fn arg<'a>(job: &'a Value, flag: &str) -> Result<&'a str> {
    let matches = array(&job["command"], "args")?
        .windows(2)
        .filter(|pair| pair[0] == flag)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected exactly one {flag}"));
    }
    matches[0][1]
        .as_str()
        .ok_or_else(|| format!("{flag}: not text"))
}

fn pinned(job: &Value, path: &Path, sha: &str) -> Result<()> {
    let matches = array(job, "inputs")?
        .iter()
        .filter(|entry| entry["path"].as_str() == path.to_str())
        .collect::<Vec<_>>();
    if matches.len() != 1 || matches[0]["sha256"] != sha {
        return Err(format!("{}: input pin differs", path.display()));
    }
    Ok(())
}

fn verify_conditions(report: &Value, journal: &str) -> Result<()> {
    if report["sampled_conditions_eligible"] != true
        || !report["violations"].as_array().is_some_and(Vec::is_empty)
    {
        return Err("queue conditions ineligible".into());
    }
    if journal.lines().next().is_none() {
        return Err("empty condition journal".into());
    }
    for (index, line) in journal.lines().enumerate() {
        let sample: Value = parse_strict_json(line.as_bytes())
            .map_err(|error| format!("condition {}: {error}", index + 1))?;
        if sample["ready"] != true || sample["activity_sampled"] != true {
            return Err(format!("condition {}: incomplete observation", index + 1));
        }
    }
    Ok(())
}

fn selector_off_dispatch(value: &Value) -> bool {
    value["research_dispatch"]["overflowed"] == false
        && value["research_dispatch"]["counts"]
            .as_object()
            .is_some_and(serde_json::Map::is_empty)
}

fn read_arm(
    dir: &Path,
    manifest: &Path,
    expected_sha: &str,
    source: &Path,
    tokenizer: &Path,
) -> Result<Arm> {
    if digest(manifest)? != expected_sha {
        return Err("manifest changed".into());
    }
    let mut expected = read(manifest)?;
    expected["validator"] = Value::Null;
    expected["kernel_game_submission"] = Value::Null;
    let job_path = dir.join("job.json");
    let job = read(&job_path)?;
    if job != expected || job["purpose"] != "correctness" {
        return Err("queued job differs from frozen manifest".into());
    }
    let inputs = array(&job, "inputs")?;
    if inputs.len() != 10 {
        return Err("input pin count differs".into());
    }
    let mut pinned_paths = std::collections::BTreeSet::new();
    for input in inputs {
        let path = text(input, "path")?;
        let sha = text(input, "sha256")?;
        if !Path::new(path).is_absolute() || sha.len() != 64 || !pinned_paths.insert(path) {
            return Err("invalid or duplicated input pin".into());
        }
    }
    pinned(&job, source, SOURCE_SHA)?;
    pinned(&job, tokenizer, TOKENIZER_SHA)?;
    if job["command"]["executable"]["sha256"] != EXECUTABLE_SHA
        || job["command"]["env"]["RVLLM_METAL_RESEARCH"] != "off"
        || job["stable_seconds"] != 0
        || job["conditions"]["minimum_free_bytes"] != (32u64 << 30)
        || !job["conditions"]["thermal_state"].is_null()
    {
        return Err("selector, executable or conditions differ".into());
    }
    let report_path = dir.join("report.json");
    let report = read(&report_path)?;
    if report["id"] != job["id"]
        || report["status"] != "succeeded"
        || report["exit_code"] != 0
        || report["files_unchanged"] != true
        || report["overdue"] != false
        || report["signal_or_missing_exit_code"] != false
    {
        return Err("queue receipt not clean".into());
    }
    let journal_path = dir.join("conditions.jsonl");
    let journal = fs::read_to_string(&journal_path).map_err(|error| error.to_string())?;
    verify_conditions(&report, &journal)?;
    let stdout_path = dir.join("trial.stdout");
    let trial = read(&stdout_path)?;
    if trial["model_dir"] != arg(&job, "--model-dir")?
        || trial["metal_compute_dtype"] != "bfloat16"
        || trial["metal_weight_dtype"] != "bfloat16"
        || !selector_off_dispatch(&trial)
    {
        return Err("trial identity or dispatch differs".into());
    }
    let receipts = [
        ("job", job_path),
        ("report", report_path),
        ("trial_stdout", stdout_path),
        ("trial_stderr", dir.join("trial.stderr")),
        ("conditions", journal_path),
    ]
    .into_iter()
    .map(|(name, path)| digest(&path).map(|sha| (name, sha)))
    .collect::<Result<_>>()?;
    Ok(Arm {
        job,
        trial,
        receipts,
    })
}

fn verify_prompt_inputs(arms: &[Arm], cases: &[Value]) -> Result<()> {
    for (index, ((name, targets), source_case)) in CASES.iter().zip(cases).enumerate() {
        let job = &arms[index].job;
        let path = Path::new(arg(job, "--teacher-prompt-jsonl")?);
        pinned(job, path, &digest(path)?)?;
        let lines = fs::read_to_string(path).map_err(|error| error.to_string())?;
        let records = lines.lines().collect::<Vec<_>>();
        let [line] = records.as_slice() else {
            return Err(format!("{name}: standalone prompt line count differs"));
        };
        let prompt: Value =
            parse_strict_json(line.as_bytes()).map_err(|error| error.to_string())?;
        if prompt["name"] != *name
            || prompt["prompt"] != source_case["prompt"]
            || arg(job, "--teacher-token-ids")?
                != targets
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            || arg(job, "--max-new-tokens")? != targets.len().to_string()
            || arg(job, "--max-total-tokens")? != "512"
        {
            return Err(format!("{name}: standalone prompt or targets differ"));
        }
    }
    let job = &arms[2].job;
    let path = Path::new(arg(job, "--prompts-jsonl")?);
    pinned(job, path, &digest(path)?)?;
    let lines = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let lines = lines.lines().collect::<Vec<_>>();
    if lines.len() != 2 || !array(&job["command"], "args")?.contains(&json!("--teacher-session")) {
        return Err("batch prompt count or teacher-session mode differs".into());
    }
    for (line, ((name, targets), source_case)) in lines.iter().zip(CASES.iter().zip(cases)) {
        let prompt: Value =
            parse_strict_json(line.as_bytes()).map_err(|error| error.to_string())?;
        if prompt["name"] != *name
            || prompt["prompt"] != source_case["prompt"]
            || ids(&prompt, "teacher_token_ids")? != *targets
            || prompt["max_new_tokens"] != targets.len()
            || prompt["max_total_tokens"] != 512
        {
            return Err(format!("{name}: batch prompt or targets differ"));
        }
    }
    Ok(())
}

fn score_case(case: &Value, prompt: &str, prompt_ids: &[u32], targets: &[u32]) -> Result<Value> {
    if text(case, "prompt")? != prompt
        || ids(case, "prompt_token_ids")? != prompt_ids
        || ids(case, "generated_token_ids")? != targets
        || case["finish_reason"] != "length"
    {
        return Err("prompt, forced output or finish reason differs".into());
    }
    let mut output = prompt_ids.to_vec();
    output.extend_from_slice(targets);
    if ids(case, "output_token_ids")? != output {
        return Err("full output IDs differ".into());
    }
    text(case, "generated_text")?;
    text(case, "output_text")?;
    let teacher = &case["teacher_forced"];
    if teacher["schema"] != "rvllm.metal_teacher_forced_quality.v1" {
        return Err("teacher schema differs".into());
    }
    let steps = array(teacher, "steps")?;
    if steps.len() != targets.len() {
        return Err("missing teacher positions".into());
    }
    let mut total = 0.0;
    for (step, target) in steps.iter().zip(targets) {
        if step["target_token_id"].as_u64() != Some(u64::from(*target))
            || step["sampled_token_id"]
                .as_u64()
                .is_none_or(|id| id >= 262_144)
            || step["target_rank"]
                .as_u64()
                .is_none_or(|rank| !(1..=262_144).contains(&rank))
        {
            return Err("target, sampled token or rank differs".into());
        }
        finite(step, "target_logit")?;
        let nll = finite(step, "negative_log_likelihood")?;
        if nll < 0.0 {
            return Err("negative NLL".into());
        }
        total += nll;
    }
    if finite(teacher, "total_negative_log_likelihood")? != total
        || finite(teacher, "mean_negative_log_likelihood")? != total / targets.len() as f64
        || !finite(teacher, "perplexity")?.is_sign_positive()
    {
        return Err("teacher aggregates differ".into());
    }
    Ok(json!({
        "prompt":case["prompt"],
        "prompt_token_ids":case["prompt_token_ids"],
        "generated_token_ids":case["generated_token_ids"],
        "output_token_ids":case["output_token_ids"],
        "finish_reason":case["finish_reason"],
        "generated_text":case["generated_text"],
        "output_text":case["output_text"],
        "teacher_forced":teacher,
    }))
}

fn no_timing(value: &Value) -> Result<()> {
    if value["timing_valid"] != false {
        return Err("batch timing validity differs".into());
    }
    for key in [
        "prepare_ms",
        "prefill_ms",
        "decode_ms",
        "total_ms",
        "tok_per_s",
        "last_step_gpu_execution_ns",
        "cpu_wall_ns",
        "cpu_encode_ns",
        "command_buffer_wait_ns",
        "last_step_cpu_wall_ns",
        "last_step_cpu_encode_ns",
        "last_step_command_buffer_wait_ns",
    ] {
        if value.get(key).is_some() {
            return Err(format!("batch retained invalid timing field {key}"));
        }
    }
    Ok(())
}

fn run(paths: &[PathBuf]) -> Result<Value> {
    let [output, source_path, tokenizer_path, manifest_dir, observatory_dir, kitchen_dir, batch_dir] =
        paths
    else {
        return Err("expected NEW_OUTPUT SOURCE TOKENIZER MANIFEST_DIR THREE_RESULT_DIRS".into());
    };
    if !output.is_absolute()
        || output.exists()
        || digest(source_path)? != SOURCE_SHA
        || digest(tokenizer_path)? != TOKENIZER_SHA
    {
        return Err("output or source/tokenizer identity differs".into());
    }
    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|error| error.to_string())?;
    let source = read(source_path)?;
    let cases = array(&source, "cases")?;
    if source["schema"] != "rvllm.gemma4_heldout_text.v1" || cases.len() != 2 {
        return Err("sealed source shape differs".into());
    }
    let dirs = [observatory_dir, kitchen_dir, batch_dir];
    let mut arms = Vec::new();
    for ((name, sha), dir) in MANIFEST_FILES.iter().zip(MANIFEST_SHA).zip(dirs) {
        arms.push(read_arm(
            dir,
            &manifest_dir.join(name),
            sha,
            source_path,
            tokenizer_path,
        )?);
    }
    verify_prompt_inputs(&arms, cases)?;
    let singles = [&arms[0].trial, &arms[1].trial];
    let batch = &arms[2].trial;
    if batch["schema"] != "rvllm.metal_teacher_forced_session.v1"
        || batch["status"] != "pass"
        || batch["checkpoint_complete"] != true
        || batch["checkpoint_completed_cases"] != 2
        || batch["checkpoint_total_cases"] != 2
        || batch["case_count"] != 2
        || array(batch, "cases")?.len() != 2
    {
        return Err("batch schema or completion differs".into());
    }
    no_timing(batch)?;
    let mut comparisons = Vec::new();
    for (index, ((name, targets), source_case)) in CASES.iter().zip(cases).enumerate() {
        let prompt = text(source_case, "prompt")?;
        if text(source_case, "id")? != *name {
            return Err("source case order differs".into());
        }
        let mut prompt_ids = vec![2];
        prompt_ids.extend_from_slice(
            tokenizer
                .encode(prompt, false)
                .map_err(|error| error.to_string())?
                .get_ids(),
        );
        if prompt_ids.len() != 184 {
            return Err("prompt tokenizer identity differs".into());
        }
        let single = singles[index];
        if single["schema"] != "rvllm.apple_metal_text_infer.v1"
            || single["model_dir"] != batch["model_dir"]
        {
            return Err("standalone schema or model differs".into());
        }
        let batch_case = &batch["cases"][index];
        if batch_case["name"] != *name || batch_case["status"] != "pass" {
            return Err("batch case identity differs".into());
        }
        if !selector_off_dispatch(batch_case) {
            return Err("batch case used a research selector or overflowed".into());
        }
        no_timing(batch_case)?;
        let single_score = score_case(single, prompt, &prompt_ids, targets)?;
        let batch_score = score_case(batch_case, prompt, &prompt_ids, targets)?;
        if single_score != batch_score {
            return Err(format!(
                "{name}: standalone and batch teacher results differ"
            ));
        }
        comparisons
            .push(json!({"case":name,"equal":true,"teacher_forced":single["teacher_forced"]}));
    }
    let receipt_rows = arms
        .iter()
        .map(|arm| json!(arm.receipts))
        .collect::<Vec<_>>();
    let result = json!({
        "schema":"rvllm.e2b_multicase_teacher_qualification.v1",
        "claim":"two synthetic E2B cases had exact standalone/batch teacher equality; no quality or timing claim",
        "source_sha256":SOURCE_SHA,
        "tokenizer_sha256":TOKENIZER_SHA,
        "manifest_sha256":MANIFEST_SHA,
        "receipt_sha256":receipt_rows,
        "comparisons":comparisons,
        "admitted":true,
        "timing_valid":false,
    });
    let mut bytes = serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .and_then(|mut file| file.write_all(&bytes))
        .map_err(|error| format!("{}: {error}", output.display()))?;
    Ok(result)
}

fn main() {
    let paths = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if let Err(error) = run(&paths) {
        eprintln!("rvllm_gemma4_e2b_multicase_qual_summary: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_three_manifest_hashes_are_sealed() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap();
        let root = workspace
            .join("reports/prefill-round-fullroute-short-20260927/e2b-multicase-qual-v1-queue");
        for (name, sha) in MANIFEST_FILES.into_iter().zip(MANIFEST_SHA) {
            assert_eq!(digest(&root.join(name)).unwrap(), sha);
        }
    }

    #[test]
    fn sealed_manifests_match_source_prompt_targets() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap();
        let report = workspace.join("reports/prefill-round-fullroute-short-20260927");
        let root = report.join("e2b-multicase-qual-v1-queue");
        let source = read(&report.join("e2b-multicase-teacher-qual-source-v1.json")).unwrap();
        let arms = MANIFEST_FILES.map(|name| Arm {
            job: read(&root.join(name)).unwrap(),
            trial: Value::Null,
            receipts: BTreeMap::new(),
        });
        assert!(verify_prompt_inputs(&arms, array(&source, "cases").unwrap()).is_ok());
        let mut bad = arms;
        bad[1].job["command"]["args"][5] = json!("8,8");
        assert!(verify_prompt_inputs(&bad, array(&source, "cases").unwrap()).is_err());
    }

    #[test]
    fn timing_fields_fail_closed() {
        assert!(no_timing(&json!({"timing_valid":false})).is_ok());
        assert!(no_timing(&json!({"timing_valid":false,"prefill_ms":1})).is_err());
        assert!(no_timing(&json!({"timing_valid":true})).is_err());
    }

    #[test]
    fn missing_or_unsampled_conditions_fail_closed() {
        let report = json!({"sampled_conditions_eligible":true,"violations":[]});
        assert!(verify_conditions(&report, "").is_err());
        assert!(
            verify_conditions(&report, "{\"ready\":true,\"activity_sampled\":false}\n").is_err()
        );
        assert!(verify_conditions(&report, "{\"ready\":true,\"activity_sampled\":true}\n").is_ok());
    }

    #[test]
    fn score_requires_every_position_and_consistent_aggregate() {
        let valid = json!({
            "prompt":"p", "prompt_token_ids":[2,4],
            "generated_token_ids":[8], "output_token_ids":[2,4,8],
            "finish_reason":"length", "generated_text":"x", "output_text":"px",
            "teacher_forced":{
                "schema":"rvllm.metal_teacher_forced_quality.v1",
                "steps":[{"target_token_id":8,"sampled_token_id":9,"target_rank":2,
                    "target_logit":0.5,"negative_log_likelihood":1.0}],
                "total_negative_log_likelihood":1.0,
                "mean_negative_log_likelihood":1.0,"perplexity":2.718281828459045
            }
        });
        assert!(score_case(&valid, "p", &[2, 4], &[8]).is_ok());
        let mut missing = valid.clone();
        missing["teacher_forced"]["steps"] = json!([]);
        assert!(score_case(&missing, "p", &[2, 4], &[8]).is_err());
        let mut changed = valid;
        changed["teacher_forced"]["total_negative_log_likelihood"] = json!(0.9);
        assert!(score_case(&changed, "p", &[2, 4], &[8]).is_err());
    }
}
