//! Fail-closed referee for the prospective full-text W4 continuation trial.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

// These are the exact, pre-submission package and job identities. Changing a
// job requires a new referee and fresh job IDs, never reinterpretation.
const SOURCE_SHA: &str = "ad075af3147a6c242aed94f0c2f5ac40845c09b38c2929983be5dbd234d5b80a";
const FIXTURE_SHA: &str = "f4899669393b9726953a698eaa3049819de2d83822e67416d891b511e1cff65c";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const MODEL_SHA: &str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
const CONFIG_SHA: &str = "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9";
const SG8_METALLIB_SHA: &str = "21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06";
const PLAN_SHA: &str = "087e5e87f8ef2fd68e6d910af17867ec2cf12ef9344e9b70cf481a6992591d7e";
const PACKAGE_MANIFEST_SHA: &str =
    "abcd5b043322efe1756b6ac81e805f0aee9da5b04c9e10a77ae391532f026ce3";
const EXECUTABLE_SHA: &str = "1f9cfedd5c1a91f4a1046e2930e13071847fc0b90c7bafa2f391f3e99b3082c5";
const CLI_SOURCE_SHA: &str = "5158928cc2f97381b8943e75657b0485f4cbb76abd55a56867e2cfb9c05277ec";
const ARMS: [(&str, &str, &str, &str); 4] = [
    (
        "estuary",
        "native",
        "estuary-native-job.json",
        "ecee5e07697c9797c55510f8aec3084538a2c55ddd707a2c2ffd4d18e21e3f66",
    ),
    (
        "estuary",
        "w4",
        "estuary-w4-job.json",
        "048907938601caa94b0655fe6d2833be0ad45b074bd0656ecdc6b4415e5a0e61",
    ),
    (
        "textile",
        "native",
        "textile-native-job.json",
        "3aaa30f95d374ca1cce6e0c8af340bea687097a1859b6a1f247ae139db354643",
    ),
    (
        "textile",
        "w4",
        "textile-w4-job.json",
        "4a7d23367f99addcd342cf8dd81cc6a48a27efefd48320d28fbb0625cd4ce0f4",
    ),
];

type Result<T> = std::result::Result<T, String>;

struct Arm {
    report: Value,
    trial: Value,
    receipt_sha256: BTreeMap<&'static str, String>,
}

fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    if args.len() != 11 {
        eprintln!("usage: rvllm_gemma4_fulltext_w4_teacher_summary NEW_OUTPUT.json TOKENIZER.json SOURCE.json TOKENS.json SEALED_PLAN.json MANIFEST_DIR ESTUARY_NATIVE_DIR ESTUARY_W4_DIR TEXTILE_NATIVE_DIR TEXTILE_W4_DIR");
        std::process::exit(2);
    }
    let paths = args[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    match run(&paths) {
        Ok(value) => {
            let mut bytes = serde_json::to_vec_pretty(&value).expect("summary is serializable");
            bytes.push(b'\n');
            if let Err(error) = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&paths[0])
                .and_then(|mut file| file.write_all(&bytes))
            {
                eprintln!("write {}: {error}", paths[0].display());
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("rvllm_gemma4_fulltext_w4_teacher_summary: {error}");
            std::process::exit(1);
        }
    }
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|sha| sha.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn read(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse_strict_json(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value.get(key).ok_or_else(|| format!("missing {key}"))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not a string"))
}

fn number(value: &Value, key: &str) -> Result<f64> {
    field(value, key)?
        .as_f64()
        .filter(|n| n.is_finite())
        .ok_or_else(|| format!("{key} is not finite"))
}

fn ids(value: &Value, key: &str) -> Result<Vec<u32>> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not an array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("{key} contains a non-token"))
        })
        .collect()
}

fn encode(tokenizer: &tokenizers::Tokenizer, value: &str) -> Result<Vec<u32>> {
    let mut ids = vec![2];
    ids.extend_from_slice(
        tokenizer
            .encode(value, false)
            .map_err(|error| format!("tokenize: {error}"))?
            .get_ids(),
    );
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
    let matches = cases
        .iter()
        .filter(|case| case["id"] == id)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected exactly one {id} source case"));
    }
    let prompt = string(matches[0], "prompt")?;
    let full = format!("{prompt}{}", string(matches[0], "continuation")?);
    let prompt_ids = encode(tokenizer, prompt)?;
    let full_ids = encode(tokenizer, &full)?;
    let targets = full_ids
        .strip_prefix(prompt_ids.as_slice())
        .filter(|tail| !tail.is_empty())
        .ok_or_else(|| format!("{id}: tokenizer changed across prompt boundary"))?;
    Ok((prompt_ids, targets.to_vec()))
}

fn pinned(job: &Value, name: &str, sha: &str) -> Result<()> {
    let inputs = field(job, "inputs")?
        .as_array()
        .ok_or("inputs is not an array")?;
    let matches = inputs
        .iter()
        .filter(|input| {
            input["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(name))
        })
        .collect::<Vec<_>>();
    if matches.is_empty() || matches.iter().any(|entry| entry["sha256"] != sha) {
        return Err(format!("{name}: missing or changed input pin"));
    }
    Ok(())
}

fn pinned_exact(job: &Value, path: &Path, sha: &str) -> Result<()> {
    let inputs = field(job, "inputs")?
        .as_array()
        .ok_or("inputs is not an array")?;
    let matches = inputs
        .iter()
        .filter(|entry| entry["path"].as_str() == path.to_str())
        .collect::<Vec<_>>();
    if matches.len() != 1 || matches[0]["sha256"] != sha {
        return Err(format!("{}: exact pin missing or changed", path.display()));
    }
    Ok(())
}

fn arg<'a>(job: &'a Value, flag: &str) -> Result<&'a str> {
    let args = job["command"]["args"]
        .as_array()
        .ok_or("command args missing")?;
    let matches = args
        .windows(2)
        .filter(|pair| pair[0] == flag)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected one {flag} argument"));
    }
    matches[0][1]
        .as_str()
        .ok_or_else(|| format!("{flag} argument is not text"))
}

fn check_teacher_command(job: &Value, prompt: &str, targets: &[u32]) -> Result<()> {
    let prompt_path = Path::new(arg(job, "--teacher-prompt-jsonl")?);
    let bytes = fs::read(prompt_path).map_err(|error| error.to_string())?;
    let line = std::str::from_utf8(&bytes)
        .map_err(|error| error.to_string())?
        .trim_end_matches('\n');
    if line.contains('\n') {
        return Err("teacher prompt file has more than one line".into());
    }
    let record: Value = parse_strict_json(line.as_bytes()).map_err(|error| error.to_string())?;
    if record["prompt"] != prompt {
        return Err("teacher prompt differs from sealed source".into());
    }
    let file_name = prompt_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("teacher prompt has no file name")?;
    pinned(job, file_name, &digest(prompt_path)?)?;
    let target_list = targets
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    if arg(job, "--teacher-token-ids")? != target_list
        || arg(job, "--max-new-tokens")? != targets.len().to_string()
    {
        return Err("teacher command target IDs or step count differ".into());
    }
    Ok(())
}

fn verify_fulltext_package(job: &Value, plan_path: &Path, plan: &Value) -> Result<()> {
    if PACKAGE_MANIFEST_SHA.len() != 64 {
        return Err("full-text package identity is not sealed".into());
    }
    pinned_exact(job, plan_path, PLAN_SHA)?;
    let root = Path::new(arg(job, "--model-dir")?);
    let manifest_path = root.join("rvllm-apple-model.json");
    if digest(&manifest_path)? != PACKAGE_MANIFEST_SHA {
        return Err("full-text package manifest changed".into());
    }
    pinned_exact(job, &manifest_path, PACKAGE_MANIFEST_SHA)?;
    let manifest = read(&manifest_path)?;
    if manifest["schema_version"] != 3
        || manifest["model_config"]["sha256"] != CONFIG_SHA
        || manifest["weight_shards"]
            .as_array()
            .is_none_or(|shards| shards.len() != 1)
        || manifest["weight_shards"][0]["file"]["sha256"] != MODEL_SHA
    {
        return Err("full-text package checkpoint identity differs".into());
    }
    let selectors = plan["selectors"]
        .as_array()
        .ok_or("plan selectors missing")?;
    let tensors = manifest["low_bit_tensors"]
        .as_array()
        .ok_or("package tensors missing")?;
    if selectors.len() != 328 || tensors.len() != 328 {
        return Err("full-text plan or package lacks 328 projections".into());
    }
    let mut expected = selectors
        .iter()
        .map(|v| string(v, "tensor_name").map(str::to_owned))
        .collect::<Result<Vec<_>>>()?;
    let mut actual = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        actual.push(string(tensor, "tensor_name")?.to_owned());
        if tensor["format"] != "w4_a16"
            || tensor["group_size"] != 32
            || tensor["activation_float_type"] != "bf16"
        {
            return Err("full-text package sidecar format differs".into());
        }
        for key in ["packed_values", "scales"] {
            let descriptor = field(tensor, key)?;
            let path = root.join(string(descriptor, "path")?);
            pinned_exact(job, &path, string(descriptor, "sha256")?)?;
        }
    }
    expected.sort();
    actual.sort();
    if actual != expected || actual.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("full-text package names differ from sealed plan".into());
    }
    let libraries = manifest["metal_libraries"]
        .as_array()
        .ok_or("package libraries missing")?;
    let matching = libraries
        .iter()
        .filter(|entry| entry["platform"] == "mac_os" && entry["float_type"] == "bf16")
        .collect::<Vec<_>>();
    if matching.len() != 1 || matching[0]["library"]["sha256"] != SG8_METALLIB_SHA {
        return Err("package BF16 Metal library differs from donor".into());
    }
    let library = root.join(string(&matching[0]["library"], "path")?);
    pinned_exact(job, &library, SG8_METALLIB_SHA)?;
    Ok(())
}

fn dispatch_valid(counts: &serde_json::Map<String, Value>, route: &str) -> bool {
    let count = |name: &str| counts.get(name).and_then(Value::as_u64).unwrap_or(0);
    let attention = count("research_donor12b_sg8_local_attention") > 0
        && count("research_donor12b_sg8_global_attention") > 0;
    let w4 = [
        "research_donor12b_sg8_w4",
        "research_donor12b_sg8_gate_w4",
        "research_donor12b_sg8_qkv_w4",
    ];
    let w8 = [
        "research_donor12b_sg8_w8",
        "research_donor12b_sg8_gate_w8",
        "research_donor12b_sg8_qkv_w8",
    ];
    attention
        && w8.iter().all(|name| count(name) == 0)
        && match route {
            "native" => {
                w4.iter().all(|name| count(name) == 0)
                    && count("research_donor12b_sg8_native_gate") > 0
                    && count("research_donor12b_sg8_native_projection") > 0
            }
            "w4" => w4.iter().all(|name| count(name) > 0),
            _ => false,
        }
}

#[allow(clippy::too_many_arguments)]
fn read_arm(
    dir: &Path,
    manifest: &Path,
    manifest_sha: &str,
    plan_path: &Path,
    plan: &Value,
    id: &str,
    route: &str,
    prompt: &str,
    targets: &[u32],
) -> Result<Arm> {
    if manifest_sha.len() != 64 || digest(manifest)? != manifest_sha {
        return Err(format!("{id}: frozen manifest is unset or changed"));
    }
    let mut expected_job = read(manifest)?;
    expected_job["kernel_game_submission"] = Value::Null;
    expected_job["validator"] = Value::Null;
    let job_path = dir.join("job.json");
    let job = read(&job_path)?;
    if job != expected_job || string(&job, "id")? != id || job["purpose"] != "correctness" {
        return Err(format!("{id}: queue job differs from frozen manifest"));
    }
    pinned(
        &job,
        "lowbit-fulltext-w4-quality-source-v1.json",
        SOURCE_SHA,
    )?;
    pinned(
        &job,
        "lowbit-fulltext-w4-quality-tokens-v1.json",
        FIXTURE_SHA,
    )?;
    pinned(&job, "tokenizer.json", TOKENIZER_SHA)?;
    pinned(&job, "config.json", CONFIG_SHA)?;
    pinned(&job, "model.safetensors", MODEL_SHA)?;
    pinned(&job, "rvllm_metal_infer.rs", CLI_SOURCE_SHA)?;
    check_teacher_command(&job, prompt, targets)?;
    if job["command"]["executable"]["sha256"] != EXECUTABLE_SHA
        || job["command"]["env"]["RVLLM_METAL_RESEARCH"] != "metal-donor12b-sg8"
    {
        return Err(format!("{id}: wrong executable or research selector"));
    }
    if route == "w4" {
        verify_fulltext_package(&job, plan_path, plan)?;
    } else {
        pinned(&job, "core.metallib", SG8_METALLIB_SHA)?;
        if job["command"]["env"]["RVLLM_METAL_METALLIB_BF16"]
            .as_str()
            .is_none_or(|path| !path.ends_with("core.metallib"))
        {
            return Err(format!("{id}: native donor library differs"));
        }
    }
    let report_path = dir.join("report.json");
    let report = read(&report_path)?;
    if string(&report, "id")? != id
        || report["status"] != "succeeded"
        || report["exit_code"] != 0
        || report["files_unchanged"] != true
        || report["overdue"] != false
        || report["signal_or_missing_exit_code"] != false
        || report["sampled_conditions_eligible"] != true
        || !report["violations"].as_array().is_some_and(Vec::is_empty)
    {
        return Err(format!("{id}: queue receipt is not clean and eligible"));
    }
    let journal = fs::read_to_string(dir.join("conditions.jsonl")).map_err(|e| e.to_string())?;
    if journal.lines().next().is_none() {
        return Err(format!("{id}: empty conditions journal"));
    }
    for (index, line) in journal.lines().enumerate() {
        let condition: Value = parse_strict_json(line.as_bytes())
            .map_err(|e| format!("{id}: condition line {}: {e}", index + 1))?;
        if condition["ready"] != true {
            return Err(format!("{id}: condition line {} not ready", index + 1));
        }
    }
    let trial_path = dir.join("trial.stdout");
    let trial = read(&trial_path)?;
    if trial["schema"] != "rvllm.apple_metal_text_infer.v1"
        || trial["metal_compute_dtype"] != "bfloat16"
        || trial["metal_weight_dtype"] != "bfloat16"
        || string(&trial, "model_dir")? != arg(&job, "--model-dir")?
        || trial["research_dispatch"]["overflowed"] != false
    {
        return Err(format!(
            "{id}: trial identity, dtype or dispatch ledger differs"
        ));
    }
    let counts = trial["research_dispatch"]["counts"]
        .as_object()
        .ok_or("research counts missing")?;
    if !dispatch_valid(counts, route) {
        return Err(format!("{id}: named decode dispatch is absent"));
    }
    let receipt_sha256 = [
        ("job", job_path),
        ("report", report_path),
        ("trial_stdout", trial_path),
        ("trial_stderr", dir.join("trial.stderr")),
        ("conditions", dir.join("conditions.jsonl")),
    ]
    .into_iter()
    .map(|(name, path)| digest(&path).map(|sha| (name, sha)))
    .collect::<Result<_>>()?;
    Ok(Arm {
        report,
        trial,
        receipt_sha256,
    })
}

fn scored_steps(arm: &Arm, prompt: &[u32], targets: &[u32]) -> Result<(Vec<Value>, f64)> {
    if ids(&arm.trial, "prompt_token_ids")? != prompt
        || ids(&arm.trial, "generated_token_ids")? != targets
    {
        return Err("prompt or forced target IDs changed".into());
    }
    let teacher = field(&arm.trial, "teacher_forced")?;
    if teacher["schema"] != "rvllm.metal_teacher_forced_quality.v1" {
        return Err("wrong teacher schema".into());
    }
    let steps = field(teacher, "steps")?.as_array().ok_or("steps missing")?;
    if steps.len() != targets.len() {
        return Err("teacher step count differs".into());
    }
    let mut total = 0.0;
    for (step, target) in steps.iter().zip(targets) {
        if step["target_token_id"].as_u64() != Some(u64::from(*target))
            || step["target_rank"]
                .as_u64()
                .is_none_or(|rank| !(1..=262_144).contains(&rank))
            || step["sampled_token_id"]
                .as_u64()
                .is_none_or(|id| id >= 262_144)
        {
            return Err("teacher target, rank or sampled ID differs".into());
        }
        let nll = number(step, "negative_log_likelihood")?;
        let _target_logit = number(step, "target_logit")?;
        if nll < 0.0 {
            return Err("negative teacher NLL".into());
        }
        total += nll;
    }
    if (number(teacher, "total_negative_log_likelihood")? - total).abs() > 1e-8
        || (number(teacher, "mean_negative_log_likelihood")? - total / targets.len() as f64).abs()
            > 1e-8
    {
        return Err("teacher aggregate differs from steps".into());
    }
    Ok((steps.clone(), total))
}

fn summarize_case(
    id: &str,
    prompt: &[u32],
    targets: &[u32],
    native: &Arm,
    w4: &Arm,
) -> Result<Value> {
    let (native_steps, native_total) = scored_steps(native, prompt, targets)?;
    let (w4_steps, w4_total) = scored_steps(w4, prompt, targets)?;
    let positions = native_steps
        .iter()
        .zip(&w4_steps)
        .enumerate()
        .map(|(index, (a, b))| {
            let a_nll = a["negative_log_likelihood"]
                .as_f64()
                .expect("validated NLL");
            let b_nll = b["negative_log_likelihood"]
                .as_f64()
                .expect("validated NLL");
            json!({"index": index, "target_token_id": targets[index],
            "native_nll": a_nll, "w4_nll": b_nll, "w4_minus_native_nll": b_nll-a_nll,
            "native_rank": a["target_rank"], "w4_rank": b["target_rank"],
            "native_greedy_id": a["sampled_token_id"], "w4_greedy_id": b["sampled_token_id"]})
        })
        .collect::<Vec<_>>();
    Ok(
        json!({"id": id, "prompt_tokens": prompt.len(), "target_tokens": targets.len(),
        "native_total_nll": native_total, "w4_total_nll": w4_total,
        "native_mean_nll": native_total / targets.len() as f64,
        "w4_mean_nll": w4_total / targets.len() as f64,
        "w4_minus_native_total_nll": w4_total-native_total,
        "rank_differences": positions.iter().filter(|p| p["native_rank"] != p["w4_rank"]).count(),
        "greedy_differences": positions.iter().filter(|p| p["native_greedy_id"] != p["w4_greedy_id"]).count(),
        "first_greedy_difference": positions.iter().position(|p| p["native_greedy_id"] != p["w4_greedy_id"]),
        "native": {"dispatch": native.trial["research_dispatch"], "conditions_eligible": native.report["sampled_conditions_eligible"], "receipt_sha256": native.receipt_sha256},
        "w4": {"dispatch": w4.trial["research_dispatch"], "conditions_eligible": w4.report["sampled_conditions_eligible"], "receipt_sha256": w4.receipt_sha256},
        "positions": positions}),
    )
}

fn run(paths: &[PathBuf]) -> Result<Value> {
    if !seal_ready(PACKAGE_MANIFEST_SHA, &ARMS) {
        return Err("package or job manifests are not sealed; no trial may be adjudicated".into());
    }
    for (path, sha) in [
        (&paths[1], TOKENIZER_SHA),
        (&paths[2], SOURCE_SHA),
        (&paths[3], FIXTURE_SHA),
    ] {
        if digest(path)? != sha {
            return Err(format!("{}: sealed input changed", path.display()));
        }
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&paths[1]).map_err(|e| e.to_string())?;
    let source = read(&paths[2])?;
    let fixture = read(&paths[3])?;
    if digest(&paths[4])? != PLAN_SHA {
        return Err("sealed full-text selector plan changed".into());
    }
    let plan = read(&paths[4])?;
    if source["schema"] != "rvllm.gemma4_heldout_text.v1"
        || fixture["schema"] != "rvllm.gemma4_heldout_tokens.v1"
        || source["cases"]
            .as_array()
            .is_none_or(|cases| cases.len() != 2)
        || fixture["cases"]
            .as_array()
            .is_none_or(|cases| cases.len() != 2)
    {
        return Err("source or fixture schema differs".into());
    }
    let mut cases = Vec::new();
    for (index, (slug, id, prompt_len, target_len)) in [
        ("estuary", "estuary-sensor-v1", 218, 21),
        ("textile", "textile-catalog-v1", 220, 22),
    ]
    .into_iter()
    .enumerate()
    {
        let (prompt, targets) = expected_case(&tokenizer, &source, id)?;
        if prompt.len() != prompt_len || targets.len() != target_len {
            return Err(format!("{id}: source token lengths differ"));
        }
        let fixture_cases = fixture["cases"]
            .as_array()
            .expect("validated fixture cases");
        let matches = fixture_cases
            .iter()
            .filter(|case| case["id"] == id)
            .collect::<Vec<_>>();
        if matches.len() != 1
            || ids(matches[0], "prompt_token_ids")? != prompt
            || ids(matches[0], "target_token_ids")? != targets
        {
            return Err(format!("{id}: sealed token fixture differs"));
        }
        let prompt_text = source["cases"]
            .as_array()
            .expect("validated cases")
            .iter()
            .find(|case| case["id"] == id)
            .and_then(|case| case["prompt"].as_str())
            .ok_or("source prompt missing")?;
        let mut pair = Vec::new();
        for (arm_spec, dir) in ARMS[index * 2..index * 2 + 2]
            .iter()
            .zip(&paths[6 + index * 2..8 + index * 2])
        {
            if arm_spec.0 != slug {
                return Err("frozen arm order differs".into());
            }
            let arm_id = format!(
                "prefill26-fulltext-w4-quality-{slug}-{}-v1-20260927",
                arm_spec.1
            );
            pair.push(read_arm(
                dir,
                &paths[5].join(arm_spec.2),
                arm_spec.3,
                &paths[4],
                &plan,
                &arm_id,
                arm_spec.1,
                prompt_text,
                &targets,
            )?);
        }
        cases.push(summarize_case(id, &prompt, &targets, &pair[0], &pair[1])?);
    }
    Ok(
        json!({"schema": "rvllm.gemma4_fulltext_w4_teacher_summary.v1",
        "claim": "Two Codex-authored synthetic continuations; all text projections W4, embeddings and multimodal BF16; no independent oracle, prefill attribution, timing or checkpoint-wide quality",
        "source_sha256": SOURCE_SHA, "token_fixture_sha256": FIXTURE_SHA,
        "tokenizer_sha256": TOKENIZER_SHA, "package_manifest_sha256": PACKAGE_MANIFEST_SHA,
        "cases": cases}),
    )
}

fn seal_ready(package_sha: &str, arms: &[(&str, &str, &str, &str)]) -> bool {
    package_sha.len() == 64 && arms.iter().all(|arm| arm.3.len() == 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_requires_all_four_hashes() {
        assert!(!seal_ready("", &ARMS));
        assert!(seal_ready(PACKAGE_MANIFEST_SHA, &ARMS));
        let mut missing = ARMS;
        missing[2].3 = "";
        assert!(!seal_ready(PACKAGE_MANIFEST_SHA, &missing));
    }

    #[test]
    fn tracked_job_manifests_match_frozen_hashes() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../reports/prefill-round-fullroute-short-20260927/lowbit-fulltext-w4-quality-v1-queue");
        for (_, _, name, sha) in ARMS {
            assert_eq!(digest(&dir.join(name)).unwrap(), sha);
        }
    }

    #[test]
    fn every_named_w4_decode_role_is_required() {
        let mut counts = serde_json::Map::new();
        counts.insert("research_donor12b_sg8_local_attention".into(), json!(1));
        counts.insert("research_donor12b_sg8_global_attention".into(), json!(1));
        for role in ["w4", "gate_w4", "qkv_w4"] {
            counts.insert(format!("research_donor12b_sg8_{role}"), json!(1));
        }
        assert!(dispatch_valid(&counts, "w4"));
        counts.remove("research_donor12b_sg8_qkv_w4");
        assert!(!dispatch_valid(&counts, "w4"));
        counts.insert("research_donor12b_sg8_qkv_w4".into(), json!(1));
        counts.insert("research_donor12b_sg8_w8".into(), json!(1));
        assert!(!dispatch_valid(&counts, "w4"));
    }

    #[test]
    fn malformed_teacher_step_is_rejected() {
        let arm = Arm {
            report: json!({}),
            receipt_sha256: BTreeMap::new(),
            trial: json!({
            "prompt_token_ids":[2], "generated_token_ids":[107], "teacher_forced": {
                "schema":"rvllm.metal_teacher_forced_quality.v1", "steps":[{
                    "target_token_id":107,"target_rank":0,"sampled_token_id":107,
                    "target_logit":1.0,"negative_log_likelihood":0.1}],
                "total_negative_log_likelihood":0.1,"mean_negative_log_likelihood":0.1}}),
        };
        assert!(scored_steps(&arm, &[2], &[107]).is_err());
    }

    #[test]
    fn teacher_mean_must_match_every_step() {
        let arm = Arm {
            report: json!({}),
            receipt_sha256: BTreeMap::new(),
            trial: json!({
                "prompt_token_ids":[2], "generated_token_ids":[107], "teacher_forced": {
                    "schema":"rvllm.metal_teacher_forced_quality.v1", "steps":[{
                        "target_token_id":107,"target_rank":1,"sampled_token_id":107,
                        "target_logit":1.0,"negative_log_likelihood":0.1}],
                    "total_negative_log_likelihood":0.1,"mean_negative_log_likelihood":0.2}}),
        };
        assert!(scored_steps(&arm, &[2], &[107]).is_err());
    }
}
