//! Fail-closed, all-position referee for the sealed balanced-16 W4 trial.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{Sha256Digest, parse_strict_json};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "c99f34ae929484d019630ada0589ff9fbd7149e86b6109efd513ebf977cd1c22";
const FIXTURE_SHA: &str = "ff019f8554630f2c8c6ca5165728109299e17a66d6cc2a8fc5bc8325fbab2b15";
const DATASET_SHA: &str = "58769d67ced092719390e496f1d4bbf8b99c43b77bba39d054b9a09aecb63cab";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const MODEL_SHA: &str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
const CONFIG_SHA: &str = "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9";
const PLAN_SHA: &str = "087e5e87f8ef2fd68e6d910af17867ec2cf12ef9344e9b70cf481a6992591d7e";
const PACKAGE_SHA: &str = "abcd5b043322efe1756b6ac81e805f0aee9da5b04c9e10a77ae391532f026ce3";
const SG8_SHA: &str = "21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06";
const EXEC_SHA: &str = "1f9cfedd5c1a91f4a1046e2930e13071847fc0b90c7bafa2f391f3e99b3082c5";
const CASES: [(&str, &str); 16] = [
    ("q00", "mmlu-balanced16-10978"),
    ("q01", "mmlu-balanced16-12398"),
    ("q02", "mmlu-balanced16-3441"),
    ("q03", "mmlu-balanced16-5622"),
    ("q04", "mmlu-balanced16-6075"),
    ("q05", "mmlu-balanced16-5793"),
    ("q06", "mmlu-balanced16-13310"),
    ("q07", "mmlu-balanced16-2512"),
    ("q08", "mmlu-balanced16-9802"),
    ("q09", "mmlu-balanced16-12568"),
    ("q10", "mmlu-balanced16-9551"),
    ("q11", "mmlu-balanced16-841"),
    ("q12", "mmlu-balanced16-10376"),
    ("q13", "mmlu-balanced16-1227"),
    ("q14", "mmlu-balanced16-10089"),
    ("q15", "mmlu-balanced16-3236"),
];

// Exact authoritative manifests in q00-native, q00-w4, ... q15-w4 order.
// A changed job requires fresh IDs and a new referee, never reinterpretation.
const MANIFEST_SHA: [&str; 32] = [
    "042c30550f15fde48af070a843538f1d78b41a941f3f868fdda1fb67ae499d3f",
    "73df8445f7307c78cb05c082089be530ebb999f83ab44217b818a51b4532874f",
    "301307b2b8183ce910d9ef48e6dc8438357e97546c12bbf43eaadb5560432ab3",
    "04a0ee3047896613df16336f5363951e9dcbb980861624ae5920c5787111b795",
    "0f025f0d660dabfeccd33820d8dca8b2fedcd8f5274d7f1cfba3cec0cbda2ca1",
    "7b7a62cf5fd398dd1048583b8129b4eddf2b06fd5714d219008111427c5425dc",
    "ce1837aa7b6acf5872036f5baeab4f7b2f52a9a53f65f2f92a852be579467cf4",
    "762825b8a84438477482227a1af8f929c9570ca53387d9c6796bb026634863bd",
    "85a5bd2fbdbc933b109e55fa032455dfb1166d44eb3bdf067e0cc024b5338852",
    "8cd042de29cad338f0f6630ba97c9b8b32e515b1876b96f0914f8635a632adbb",
    "49968095fcd6a0ecdbfb08541e61ee1d9384dce5b307d086e51ba56fc5633317",
    "15e7b7566c390d9afe972c6485a6b85211a623e594087be5fba3c165f72fb8d3",
    "1a460dcffe4bc7550c2d6bc264e76a83703ff432d46a65718bfa857996628816",
    "9c4c1aa244e1ed37f4d0042bcc3ae70779b6ca9576edb94a64ac4bd0a92fdcb9",
    "0850b2cb7406e01ce8f2f5e21049f0a2874acf76e3853ca5bafba25a07c0c121",
    "223d6d764f35f5b50ba12761c0dfe92c2b4dac17f2b89fd8b3d6477abfa97bc0",
    "c76147e47fcc2872ac1732b50d54ef592998b9cd3eec8461e2b1ed0b8a4767fa",
    "86eecefdd40588383bd22d2a3ae6ba626cad14985f0a0e2907de0303966c1c27",
    "475ba928f3ab7e3846a51c50b6b64bb0f357fa551328b8fdaeef9670ba34e355",
    "bd4e41f41ba53fc2fb29f22ce0aafe015aab9432619f0bcb666efb72253517bd",
    "342e27129fb1a919472d0aa8fd14898b644347408e33f6f74f74ff07594d15d0",
    "52d8a23aa439eaa31e16b6722dd8def3f8fa63374b77333dd9b26f751382fbc0",
    "1c03828c4efd088b768e710927f7393d1d74b4f5ab7566076f101515d1e3434f",
    "7495e6c7bbd84dc364a4108ea6e8442eb1f9b91090a9d22a58b8effb918243d3",
    "ec172815c2124c23f27ff6d3784d93c20fa209458000a3ebfd24946eac6be157",
    "351c7cf8d5c708aa09a0ca95ed68c88e8d39155bcd44fe6b15a7c020293cf79f",
    "32449dda6a3cdc1d6b479c2da1ea403cdf7ae112bd494e18e1861042a1ce6fe3",
    "4fc853adcfe7ba241913283c2085d683dfb05f7cd06359114fca5552a3d17181",
    "5f8c14a7bf11439a50277e7d109634c72f19db582e3e4278ca8dfd50fb96270d",
    "58cec15246890575fa523583b9f66bef36000ba3bed948c213967951ac5ff59d",
    "ca7a509c14495fd658f7928ea81767154fa3185eeae0633b21415e2bc31c265e",
    "29e8fcb336869c655c4bdb615fbfcdde0eb77297bfdbcdffd64e0fbf0fa7345e",
];

struct Arm {
    trial: Value,
    receipts: BTreeMap<&'static str, String>,
}

fn main() {
    let args = env::args_os().map(PathBuf::from).collect::<Vec<_>>();
    if args.len() != 40 {
        eprintln!(
            "usage: rvllm_gemma4_mmlu_balanced16_w4_summary NEW_OUTPUT.json TOKENIZER.json SOURCE.json FIXTURE.json PLAN.json DATASET.arrow MANIFEST_DIR 32_RESULT_DIRS_IN_Q00_NATIVE_W4_ORDER"
        );
        std::process::exit(2);
    }
    match run(&args[1..]) {
        Ok(summary) => {
            let mut bytes = serde_json::to_vec_pretty(&summary).expect("summary serializes");
            bytes.push(b'\n');
            if let Err(error) = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&args[1])
                .and_then(|mut file| file.write_all(&bytes))
            {
                eprintln!("{}: {error}", args[1].display());
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("balanced16 referee: {error}");
            std::process::exit(1);
        }
    }
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|sum| sum.as_str().to_owned())
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
        .ok_or_else(|| format!("{key} is not an array"))
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} is not text"))
}

fn finite(value: &Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} is not finite"))
}

fn ids(value: &Value, key: &str) -> Result<Vec<u32>> {
    array(value, key)?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("{key} contains a non-token"))
        })
        .collect()
}

fn arg<'a>(job: &'a Value, flag: &str) -> Result<&'a str> {
    let args = array(&job["command"], "args")?;
    let matches = args
        .windows(2)
        .filter(|pair| pair[0] == flag)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected exactly one {flag}"));
    }
    matches[0][1]
        .as_str()
        .ok_or_else(|| format!("{flag} is not text"))
}

fn pinned(job: &Value, path: &Path, sha: &str) -> Result<()> {
    let matches = array(job, "inputs")?
        .iter()
        .filter(|entry| entry["path"].as_str() == path.to_str())
        .collect::<Vec<_>>();
    if matches.len() != 1 || matches[0]["sha256"] != sha {
        return Err(format!("{}: exact input pin differs", path.display()));
    }
    Ok(())
}

fn pinned_named(job: &Value, name: &str, sha: &str) -> Result<()> {
    let matches = array(job, "inputs")?
        .iter()
        .filter(|entry| {
            entry["path"]
                .as_str()
                .and_then(|path| Path::new(path).file_name())
                == Some(std::ffi::OsStr::new(name))
        })
        .collect::<Vec<_>>();
    if matches.is_empty() || matches.iter().any(|entry| entry["sha256"] != sha) {
        return Err(format!("{name}: named input pin differs"));
    }
    Ok(())
}

fn encode(tokenizer: &tokenizers::Tokenizer, value: &str) -> Result<Vec<u32>> {
    let mut result = vec![2];
    result.extend_from_slice(
        tokenizer
            .encode(value, false)
            .map_err(|e| e.to_string())?
            .get_ids(),
    );
    Ok(result)
}

fn dispatch_valid(counts: &serde_json::Map<String, Value>, route: &str) -> bool {
    let count = |key: &str| counts.get(key).and_then(Value::as_u64).unwrap_or(0);
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
        && w8.iter().all(|key| count(key) == 0)
        && match route {
            "native" => {
                w4.iter().all(|key| count(key) == 0)
                    && count("research_donor12b_sg8_native_gate") > 0
                    && count("research_donor12b_sg8_native_projection") > 0
            }
            "w4" => w4.iter().all(|key| count(key) > 0),
            _ => false,
        }
}

fn verify_package(job: &Value, plan_path: &Path, plan: &Value) -> Result<()> {
    pinned(job, plan_path, PLAN_SHA)?;
    let root = Path::new(arg(job, "--model-dir")?);
    let manifest_path = root.join("rvllm-apple-model.json");
    if digest(&manifest_path)? != PACKAGE_SHA {
        return Err("package manifest changed".into());
    }
    pinned(job, &manifest_path, PACKAGE_SHA)?;
    let manifest = read(&manifest_path)?;
    if manifest["schema_version"] != 3
        || manifest["model_config"]["sha256"] != CONFIG_SHA
        || manifest["weight_shards"][0]["file"]["sha256"] != MODEL_SHA
    {
        return Err("package checkpoint differs".into());
    }
    let selectors = array(plan, "selectors")?;
    let tensors = array(&manifest, "low_bit_tensors")?;
    if selectors.len() != 328 || tensors.len() != 328 {
        return Err("package lacks 328 projections".into());
    }
    let mut expected = selectors
        .iter()
        .map(|v| text(v, "tensor_name").map(str::to_owned))
        .collect::<Result<Vec<_>>>()?;
    let mut actual = Vec::with_capacity(328);
    for tensor in tensors {
        if tensor["format"] != "w4_a16"
            || tensor["group_size"] != 32
            || tensor["activation_float_type"] != "bf16"
        {
            return Err("package sidecar format differs".into());
        }
        actual.push(text(tensor, "tensor_name")?.to_owned());
        for key in ["packed_values", "scales"] {
            let descriptor = &tensor[key];
            pinned(
                job,
                &root.join(text(descriptor, "path")?),
                text(descriptor, "sha256")?,
            )?;
        }
    }
    expected.sort();
    actual.sort();
    if actual != expected || actual.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("package selector set differs".into());
    }
    let matching = array(&manifest, "metal_libraries")?
        .iter()
        .filter(|entry| entry["platform"] == "mac_os" && entry["float_type"] == "bf16")
        .collect::<Vec<_>>();
    if matching.len() != 1 || matching[0]["library"]["sha256"] != SG8_SHA {
        return Err("package donor Metal library differs".into());
    }
    pinned(
        job,
        &root.join(text(&matching[0]["library"], "path")?),
        SG8_SHA,
    )
}

#[allow(clippy::too_many_arguments)]
fn read_arm(
    dir: &Path,
    manifest: &Path,
    sha: &str,
    source: &Path,
    fixture: &Path,
    dataset: &Path,
    tokenizer: &Path,
    plan_path: &Path,
    plan: &Value,
    id: &str,
    case_id: &str,
    route: &str,
    prompt: &str,
    targets: &[u32],
) -> Result<Arm> {
    if sha.len() != 64 || digest(manifest)? != sha {
        return Err(format!("{id}: manifest is not sealed"));
    }
    let mut expected = read(manifest)?;
    expected["kernel_game_submission"] = Value::Null;
    expected["validator"] = Value::Null;
    let job_path = dir.join("job.json");
    let job = read(&job_path)?;
    if job != expected || job["id"] != id || job["purpose"] != "correctness" {
        return Err(format!("{id}: queue job differs from frozen manifest"));
    }
    for (path, sha) in [
        (source, SOURCE_SHA),
        (fixture, FIXTURE_SHA),
        (dataset, DATASET_SHA),
        (tokenizer, TOKENIZER_SHA),
        (plan_path, PLAN_SHA),
    ] {
        pinned(&job, path, sha)?;
    }
    pinned_named(&job, "config.json", CONFIG_SHA)?;
    pinned_named(&job, "model.safetensors", MODEL_SHA)?;
    if job["command"]["executable"]["sha256"] != EXEC_SHA
        || job["command"]["env"]["RVLLM_METAL_RESEARCH"] != "metal-donor12b-sg8"
    {
        return Err(format!("{id}: executable or selector differs"));
    }
    let prompt_path = Path::new(arg(&job, "--teacher-prompt-jsonl")?);
    let bytes = fs::read(prompt_path).map_err(|e| e.to_string())?;
    let line = std::str::from_utf8(&bytes)
        .map_err(|e| e.to_string())?
        .trim_end_matches('\n');
    if line.contains('\n') {
        return Err(format!("{id}: multiple prompt lines"));
    }
    let prompt_record: Value = parse_strict_json(line.as_bytes()).map_err(|e| e.to_string())?;
    if prompt_record["prompt"] != prompt || prompt_record["name"] != case_id {
        return Err(format!("{id}: prompt differs"));
    }
    pinned(&job, prompt_path, &digest(prompt_path)?)?;
    if arg(&job, "--teacher-token-ids")?
        != targets
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
        || arg(&job, "--max-new-tokens")? != targets.len().to_string()
    {
        return Err(format!("{id}: target command differs"));
    }
    if route == "w4" {
        verify_package(&job, plan_path, plan)?;
    } else {
        let root = Path::new(arg(&job, "--model-dir")?);
        pinned(&job, &root.join("config.json"), CONFIG_SHA)?;
        pinned(&job, &root.join("model.safetensors"), MODEL_SHA)?;
        pinned(&job, &root.join("tokenizer.json"), TOKENIZER_SHA)?;
        pinned_named(&job, "core.metallib", SG8_SHA)?;
        if !job["command"]["env"]["RVLLM_METAL_METALLIB_BF16"]
            .as_str()
            .is_some_and(|path| path.ends_with("core.metallib"))
        {
            return Err(format!("{id}: native donor path differs"));
        }
    }
    let report_path = dir.join("report.json");
    let report = read(&report_path)?;
    if report["id"] != id
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
    let journal_path = dir.join("conditions.jsonl");
    let journal = fs::read_to_string(&journal_path).map_err(|e| e.to_string())?;
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
        || text(&trial, "model_dir")? != arg(&job, "--model-dir")?
        || trial["research_dispatch"]["overflowed"] != false
    {
        return Err(format!("{id}: trial identity or ledger differs"));
    }
    let counts = trial["research_dispatch"]["counts"]
        .as_object()
        .ok_or("dispatch counts missing")?;
    if !dispatch_valid(counts, route) {
        return Err(format!("{id}: named decode dispatch absent"));
    }
    let receipts = [
        ("job", job_path),
        ("report", report_path),
        ("trial_stdout", trial_path),
        ("trial_stderr", dir.join("trial.stderr")),
        ("conditions", journal_path),
    ]
    .into_iter()
    .map(|(name, path)| digest(&path).map(|sha| (name, sha)))
    .collect::<Result<_>>()?;
    Ok(Arm { trial, receipts })
}

fn scored(arm: &Arm, prompt: &[u32], targets: &[u32]) -> Result<(Vec<Value>, f64)> {
    if ids(&arm.trial, "prompt_token_ids")? != prompt
        || ids(&arm.trial, "generated_token_ids")? != targets
    {
        return Err("trial prompt or forced targets differ".into());
    }
    let teacher = &arm.trial["teacher_forced"];
    if teacher["schema"] != "rvllm.metal_teacher_forced_quality.v1" {
        return Err("teacher schema differs".into());
    }
    let steps = array(teacher, "steps")?;
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
        let nll = finite(step, "negative_log_likelihood")?;
        let _ = finite(step, "target_logit")?;
        if nll < 0.0 {
            return Err("negative NLL".into());
        }
        total += nll;
    }
    if (finite(teacher, "total_negative_log_likelihood")? - total).abs() > 1e-8
        || (finite(teacher, "mean_negative_log_likelihood")? - total / targets.len() as f64).abs()
            > 1e-8
    {
        return Err("teacher aggregate differs from positions".into());
    }
    Ok((steps.clone(), total))
}

fn run(paths: &[PathBuf]) -> Result<Value> {
    if MANIFEST_SHA.iter().any(|sha| sha.len() != 64) {
        return Err("all 32 authoritative manifest hashes must be sealed".into());
    }
    for (path, sha) in [
        (&paths[1], TOKENIZER_SHA),
        (&paths[2], SOURCE_SHA),
        (&paths[3], FIXTURE_SHA),
        (&paths[4], PLAN_SHA),
        (&paths[5], DATASET_SHA),
    ] {
        if digest(path)? != sha {
            return Err(format!("{}: sealed input changed", path.display()));
        }
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&paths[1]).map_err(|e| e.to_string())?;
    let source = read(&paths[2])?;
    let fixture = read(&paths[3])?;
    let plan = read(&paths[4])?;
    if source["schema"] != "rvllm.gemma4_heldout_text.v1"
        || fixture["schema"] != "rvllm.gemma4_heldout_tokens.v1"
        || array(&source, "cases")?.len() != 16
        || array(&fixture, "cases")?.len() != 16
    {
        return Err("source or fixture schema/count differs".into());
    }
    let mut cases = Vec::with_capacity(16);
    let mut positions_total = 0;
    let mut native_nll_total = 0.0;
    let mut w4_nll_total = 0.0;
    let mut positive_case_deltas = 0;
    let mut negative_case_deltas = 0;
    let mut rank_differences = 0;
    let mut greedy_differences = 0;
    for (index, (slug, case_id)) in CASES.iter().enumerate() {
        let case = &array(&source, "cases")?[index];
        let token_case = &array(&fixture, "cases")?[index];
        if case["id"] != *case_id || token_case["id"] != *case_id {
            return Err(format!("{case_id}: case order differs"));
        }
        let prompt_text = text(case, "prompt")?;
        let full = format!("{prompt_text}{}", text(case, "continuation")?);
        let prompt = encode(&tokenizer, prompt_text)?;
        let full_ids = encode(&tokenizer, &full)?;
        let targets = full_ids
            .strip_prefix(prompt.as_slice())
            .filter(|tail| !tail.is_empty())
            .ok_or("tokenizer changed across prompt boundary")?;
        if ids(token_case, "prompt_token_ids")? != prompt
            || ids(token_case, "target_token_ids")? != targets
        {
            return Err(format!("{case_id}: token fixture differs"));
        }
        positions_total += targets.len();
        let mut pair = Vec::with_capacity(2);
        for (offset, route) in ["native", "w4"].into_iter().enumerate() {
            let arm_index = index * 2 + offset;
            let id = format!("prefill26-mmlu-balanced16-w4-{slug}-{route}-v1-20260928");
            pair.push(read_arm(
                &paths[7 + arm_index],
                &paths[6].join(format!("{slug}-{route}-job.json")),
                MANIFEST_SHA[arm_index],
                &paths[2],
                &paths[3],
                &paths[5],
                &paths[1],
                &paths[4],
                &plan,
                &id,
                case_id,
                route,
                prompt_text,
                targets,
            )?);
        }
        let (native_steps, native_total) = scored(&pair[0], &prompt, targets)?;
        let (w4_steps, w4_total) = scored(&pair[1], &prompt, targets)?;
        native_nll_total += native_total;
        w4_nll_total += w4_total;
        positive_case_deltas += usize::from(w4_total > native_total);
        negative_case_deltas += usize::from(w4_total < native_total);
        let positions = native_steps
            .iter()
            .zip(&w4_steps)
            .enumerate()
            .map(|(at, (a, b))| {
                let a_nll = a["negative_log_likelihood"].as_f64().expect("validated");
                let b_nll = b["negative_log_likelihood"].as_f64().expect("validated");
                json!({"index":at,"target_token_id":targets[at],"native_nll":a_nll,"w4_nll":b_nll,
                "w4_minus_native_nll":b_nll-a_nll,"native_rank":a["target_rank"],
                "w4_rank":b["target_rank"],"native_greedy_id":a["sampled_token_id"],
                "w4_greedy_id":b["sampled_token_id"]})
            })
            .collect::<Vec<_>>();
        rank_differences += positions
            .iter()
            .filter(|p| p["native_rank"] != p["w4_rank"])
            .count();
        greedy_differences += positions
            .iter()
            .filter(|p| p["native_greedy_id"] != p["w4_greedy_id"])
            .count();
        cases.push(json!({"id":case_id,"prompt_tokens":prompt.len(),"target_tokens":targets.len(),
            "native_total_nll":native_total,"w4_total_nll":w4_total,
            "w4_minus_native_total_nll":w4_total-native_total,
            "rank_differences":positions.iter().filter(|p| p["native_rank"] != p["w4_rank"]).count(),
            "greedy_differences":positions.iter().filter(|p| p["native_greedy_id"] != p["w4_greedy_id"]).count(),
            "native":{"dispatch":pair[0].trial["research_dispatch"],"receipt_sha256":pair[0].receipts},
            "w4":{"dispatch":pair[1].trial["research_dispatch"],"receipt_sha256":pair[1].receipts},
            "positions":positions}));
    }
    if positions_total != 341 {
        return Err("sealed 341 target positions differ".into());
    }
    Ok(
        json!({"schema":"rvllm.gemma4_mmlu_balanced16_fulltext_w4_summary.v1",
        "claim":"Selected public MMLU questions, Codex wrapper and answer repetition; diagnostic only, no benchmark score, independent oracle, prefill attribution, timing or promotion",
        "source_sha256":SOURCE_SHA,"fixture_sha256":FIXTURE_SHA,"dataset_sha256":DATASET_SHA,
        "package_manifest_sha256":PACKAGE_SHA,"positions":positions_total,
        "native_total_nll":native_nll_total,"w4_total_nll":w4_nll_total,
        "w4_minus_native_total_nll":w4_nll_total-native_nll_total,
        "positive_case_deltas":positive_case_deltas,"negative_case_deltas":negative_case_deltas,
        "rank_differences":rank_differences,"greedy_differences":greedy_differences,
        "cases":cases}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoritative_manifests_match_all_frozen_hashes() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../reports/prefill-round-fullroute-short-20260927/mmlu-balanced16-w4-queue");
        for (index, (slug, _)) in CASES.iter().enumerate() {
            for (offset, route) in ["native", "w4"].into_iter().enumerate() {
                let path = dir.join(format!("{slug}-{route}-job.json"));
                assert_eq!(digest(&path).unwrap(), MANIFEST_SHA[index * 2 + offset]);
            }
        }
    }

    #[test]
    fn every_named_w4_role_required() {
        let mut counts = serde_json::Map::new();
        for name in [
            "local_attention",
            "global_attention",
            "w4",
            "gate_w4",
            "qkv_w4",
        ] {
            counts.insert(format!("research_donor12b_sg8_{name}"), json!(1));
        }
        assert!(dispatch_valid(&counts, "w4"));
        counts.remove("research_donor12b_sg8_qkv_w4");
        assert!(!dispatch_valid(&counts, "w4"));
        counts.insert("research_donor12b_sg8_qkv_w4".into(), json!(1));
        counts.insert("research_donor12b_sg8_w8".into(), json!(1));
        assert!(!dispatch_valid(&counts, "w4"));
    }

    #[test]
    fn exact_pin_rejects_duplicate_and_changed() {
        let job = json!({"inputs":[{"path":"/a/config.json","sha256":"x"},
            {"path":"/b/config.json","sha256":"y"}]});
        assert!(pinned_named(&job, "config.json", "x").is_err());
        assert!(pinned(&job, Path::new("/a/config.json"), "y").is_err());
    }

    #[test]
    fn teacher_requires_all_forced_positions_and_aggregate() {
        let mut arm = Arm {
            trial: json!({"prompt_token_ids":[2],"generated_token_ids":[107,108],
                "teacher_forced":{"schema":"rvllm.metal_teacher_forced_quality.v1",
                "steps":[{"target_token_id":107,"target_rank":1,"sampled_token_id":107,
                    "target_logit":1.0,"negative_log_likelihood":0.1}],
                "total_negative_log_likelihood":0.1,"mean_negative_log_likelihood":0.1}}),
            receipts: BTreeMap::new(),
        };
        assert!(scored(&arm, &[2], &[107, 108]).is_err());
        arm.trial["generated_token_ids"] = json!([107]);
        arm.trial["teacher_forced"]["steps"][0]["target_rank"] = json!(0);
        assert!(scored(&arm, &[2], &[107]).is_err());
        arm.trial["teacher_forced"]["steps"][0]["target_rank"] = json!(1);
        arm.trial["teacher_forced"]["mean_negative_log_likelihood"] = json!(0.2);
        assert!(scored(&arm, &[2], &[107]).is_err());
    }
}
