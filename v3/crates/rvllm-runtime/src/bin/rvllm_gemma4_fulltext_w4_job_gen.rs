//! Freeze four serial, correctness-only full-text W4 continuation jobs.
#![forbid(unsafe_code)]

use rvllm_apple::AppleModelPackage;
use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "ad075af3147a6c242aed94f0c2f5ac40845c09b38c2929983be5dbd234d5b80a";
const FIXTURE_SHA: &str = "f4899669393b9726953a698eaa3049819de2d83822e67416d891b511e1cff65c";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const MODEL_SHA: &str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
const CONFIG_SHA: &str = "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9";
const PLAN_SHA: &str = "087e5e87f8ef2fd68e6d910af17867ec2cf12ef9344e9b70cf481a6992591d7e";
const TEMPLATE_SHA: &str = "8bb718fcfbb59f4701567df8b5c2be581ab75d967494ee210d21ab514a061527";
const PACKAGE_MANIFEST_SHA: &str =
    "abcd5b043322efe1756b6ac81e805f0aee9da5b04c9e10a77ae391532f026ce3";
const SG8_METALLIB_SHA: &str = "21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06";
const EXECUTABLE_SHA: &str = "1f9cfedd5c1a91f4a1046e2930e13071847fc0b90c7bafa2f391f3e99b3082c5";
const CLI_SOURCE_SHA: &str = "5158928cc2f97381b8943e75657b0485f4cbb76abd55a56867e2cfb9c05277ec";
const CASES: [(&str, &str, usize, usize); 2] = [
    ("estuary", "estuary-sensor-v1", 218, 21),
    ("textile", "textile-catalog-v1", 220, 22),
];

fn main() {
    let args = env::args_os().map(PathBuf::from).collect::<Vec<_>>();
    if args.len() != 10 {
        eprintln!("usage: rvllm_gemma4_fulltext_w4_job_gen SOURCE.json FIXTURE.json TOKENIZER.json PLAN.json NATIVE_TEMPLATE.json W4_PACKAGE_DIR SG8.metallib EXECUTABLE NEW_OUTPUT_DIR");
        std::process::exit(2);
    }
    if let Err(error) = run(&args[1..]) {
        eprintln!("rvllm_gemma4_fulltext_w4_job_gen: {error}");
        std::process::exit(1);
    }
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|sha| sha.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn sealed(path: &Path, expected: &str) -> Result<PathBuf> {
    let path = path.canonicalize().map_err(|error| error.to_string())?;
    if digest(&path)? != expected {
        return Err(format!("{}: sealed SHA-256 differs", path.display()));
    }
    Ok(path)
}

fn verified_path(path: &Path, expected: &str) -> Result<PathBuf> {
    if !path.is_absolute() || digest(path)? != expected {
        return Err(format!("{}: absolute sealed input changed", path.display()));
    }
    // Keep snapshot symlink names: the referee binds tokenizer.json and
    // model.safetensors by their declared package/checkpoint paths.
    Ok(path.to_path_buf())
}

fn read(path: &Path) -> Result<Value> {
    parse_strict_json(&fs::read(path).map_err(|error| error.to_string())?)
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn replace_arg(job: &mut Value, flag: &str, replacement: &str) -> Result<()> {
    let args = job["command"]["args"]
        .as_array_mut()
        .ok_or("command arguments missing")?;
    let matches = args
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| (pair[0] == flag).then_some(index + 1))
        .collect::<Vec<_>>();
    if matches.len() != 1 || !args[matches[0]].is_string() {
        return Err(format!("expected exactly one text-valued {flag}"));
    }
    args[matches[0]] = json!(replacement);
    Ok(())
}

fn arg<'a>(job: &'a Value, flag: &str) -> Result<&'a str> {
    let args = job["command"]["args"]
        .as_array()
        .ok_or("command arguments missing")?;
    let matches = args
        .windows(2)
        .filter(|pair| pair[0] == flag)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected exactly one {flag}"));
    }
    matches[0][1]
        .as_str()
        .ok_or("command argument is not text".into())
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let mut ids = vec![2];
    ids.extend_from_slice(
        tokenizer
            .encode(text, false)
            .map_err(|error| error.to_string())?
            .get_ids(),
    );
    Ok(ids)
}

fn ids(value: &Value, key: &str) -> Result<Vec<u32>> {
    value[key]
        .as_array()
        .ok_or_else(|| format!("{key} missing"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("{key} contains a non-token"))
        })
        .collect()
}

fn pin(pins: &mut Vec<Value>, path: &Path, expected: &str) -> Result<()> {
    let path = verified_path(path, expected)?;
    pins.push(json!({"path":path,"sha256":expected}));
    Ok(())
}

fn package_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let kind = entry.file_type().map_err(|error| error.to_string())?;
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() {
                files.push(entry.path());
            } else {
                return Err(format!(
                    "{}: package contains a link or special file",
                    entry.path().display()
                ));
            }
        }
    }
    files.sort();
    if files.len() != 675 {
        return Err(format!("expected 675 package files, got {}", files.len()));
    }
    Ok(files)
}

fn validate_package(root: &Path, plan: &Value) -> Result<()> {
    sealed(&root.join("rvllm-apple-model.json"), PACKAGE_MANIFEST_SHA)?;
    AppleModelPackage::open(root).map_err(|error| error.to_string())?;
    let manifest = read(&root.join("rvllm-apple-model.json"))?;
    if manifest["schema_version"] != 3
        || manifest["model_config"]["sha256"] != CONFIG_SHA
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
        .map(|v| v["tensor_name"].as_str().unwrap_or(""))
        .collect::<Vec<_>>();
    let mut actual = Vec::new();
    for tensor in tensors {
        if tensor["format"] != "w4_a16"
            || tensor["group_size"] != 32
            || tensor["activation_float_type"] != "bf16"
        {
            return Err("full-text package sidecar format differs".into());
        }
        actual.push(
            tensor["tensor_name"]
                .as_str()
                .ok_or("package tensor name missing")?,
        );
    }
    expected.sort();
    actual.sort();
    if actual != expected || actual.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("full-text package names differ from sealed plan".into());
    }
    let matching = manifest["metal_libraries"]
        .as_array()
        .ok_or("package Metal libraries missing")?
        .iter()
        .filter(|lib| lib["platform"] == "mac_os" && lib["float_type"] == "bf16")
        .collect::<Vec<_>>();
    if matching.len() != 1 || matching[0]["library"]["sha256"] != SG8_METALLIB_SHA {
        return Err("packaged BF16 Metal library differs from native donor".into());
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .and_then(|mut file| file.write_all(bytes))
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn run(paths: &[PathBuf]) -> Result<()> {
    let source_path = sealed(&paths[0], SOURCE_SHA)?;
    let fixture_path = sealed(&paths[1], FIXTURE_SHA)?;
    let tokenizer_path = verified_path(&paths[2], TOKENIZER_SHA)?;
    let plan_path = sealed(&paths[3], PLAN_SHA)?;
    let template_path = sealed(&paths[4], TEMPLATE_SHA)?;
    let package_root = paths[5].canonicalize().map_err(|error| error.to_string())?;
    let donor = sealed(&paths[6], SG8_METALLIB_SHA)?;
    let executable = sealed(&paths[7], EXECUTABLE_SHA)?;
    let output = &paths[8];
    if output.exists() || !output.is_absolute() {
        return Err(format!(
            "{}: output must be a new absolute path",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .ok_or("output has no parent")?
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let output = parent.join(output.file_name().ok_or("output has no name")?);
    let source = read(&source_path)?;
    let fixture = read(&fixture_path)?;
    let plan = read(&plan_path)?;
    validate_package(&package_root, &plan)?;
    let package_paths = package_files(&package_root)?;
    let package_pins = package_paths
        .iter()
        .map(|path| digest(path).map(|sha| json!({"path":path,"sha256":sha})))
        .collect::<Result<Vec<_>>>()?;
    let template = read(&template_path)?;
    if template["id"] != "prefill26-lowbit-mmlu-logic-native-v1-20260927"
        || template["schema"] != "rvllm.experiment_job.v1"
        || template["purpose"] != "correctness"
        || template["stable_seconds"] != 0
        || template["command"]["env"]["RVLLM_METAL_RESEARCH"] != "metal-donor12b-sg8"
        || template["command"]["executable"]["sha256"]
            != "23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463"
    {
        return Err("native teacher template identity differs".into());
    }
    let native_root = PathBuf::from(arg(&template, "--model-dir")?);
    let native_root = native_root
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let mut native_pins = Vec::new();
    for (name, sha) in [
        ("config.json", CONFIG_SHA),
        ("model.safetensors", MODEL_SHA),
        ("tokenizer.json", TOKENIZER_SHA),
    ] {
        let path = verified_path(&native_root.join(name), sha)?;
        native_pins.push(json!({"path":path,"sha256":sha}));
    }
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|error| error.to_string())?;
    if source["schema"] != "rvllm.gemma4_heldout_text.v1"
        || !source["provenance"]
            .as_str()
            .is_some_and(|text| text.contains("Codex-authored synthetic"))
        || source["cases"]
            .as_array()
            .is_none_or(|cases| cases.len() != 2)
        || fixture["schema"] != "rvllm.gemma4_heldout_tokens.v1"
        || fixture["cases"]
            .as_array()
            .is_none_or(|cases| cases.len() != 2)
    {
        return Err("source or token fixture identity differs".into());
    }
    let cli_source = sealed(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_metal_infer.rs"),
        CLI_SOURCE_SHA,
    )?;
    let generator_source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_gemma4_fulltext_w4_job_gen.rs");
    let generator_sha = digest(&generator_source)?;
    let mut files = Vec::new();
    let mut prior = None::<String>;
    for (index, (slug, case_id, prompt_len, target_len)) in CASES.into_iter().enumerate() {
        let case = &source["cases"][index];
        let token_case = &fixture["cases"][index];
        if case["id"] != case_id || token_case["id"] != case_id {
            return Err(format!("{case_id}: case order or identity differs"));
        }
        let prompt = case["prompt"].as_str().ok_or("source prompt missing")?;
        let continuation = case["continuation"]
            .as_str()
            .ok_or("source continuation missing")?;
        let prompt_ids = encode(&tokenizer, prompt)?;
        let full_ids = encode(&tokenizer, &format!("{prompt}{continuation}"))?;
        let targets = full_ids
            .strip_prefix(prompt_ids.as_slice())
            .filter(|tail| !tail.is_empty())
            .ok_or("continuation retokenized across prompt boundary")?;
        if prompt_ids.len() != prompt_len
            || targets.len() != target_len
            || ids(token_case, "prompt_token_ids")? != prompt_ids
            || ids(token_case, "target_token_ids")? != targets
        {
            return Err(format!("{case_id}: sealed tokenizer or target IDs differ"));
        }
        let prompt_path = output.join(format!("{slug}-prompt.jsonl"));
        let mut prompt_bytes = serde_json::to_vec(&json!({"name":case_id,"prompt":prompt}))
            .map_err(|error| error.to_string())?;
        prompt_bytes.push(b'\n');
        let prompt_sha = Sha256Digest::bytes(&prompt_bytes).as_str().to_owned();
        files.push((prompt_path.clone(), prompt_bytes));
        let target_list = targets
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        for route in ["native", "w4"] {
            let id = format!("prefill26-fulltext-w4-quality-{slug}-{route}-v1-20260927");
            let mut job = template.clone();
            job["id"] = json!(id);
            job["after"] = prior.as_ref().map_or_else(|| json!([]), |id| json!([id]));
            job["conditions"]["minimum_free_bytes"] = json!(64u64 << 30);
            job["command"]["executable"] = json!({"path":executable,"sha256":EXECUTABLE_SHA});
            replace_arg(
                &mut job,
                "--teacher-prompt-jsonl",
                &prompt_path.display().to_string(),
            )?;
            replace_arg(&mut job, "--teacher-token-ids", &target_list)?;
            replace_arg(&mut job, "--max-new-tokens", &targets.len().to_string())?;
            let mut pins = Vec::new();
            for (path, sha) in [
                (&source_path, SOURCE_SHA),
                (&fixture_path, FIXTURE_SHA),
                (&tokenizer_path, TOKENIZER_SHA),
                (&plan_path, PLAN_SHA),
                (&template_path, TEMPLATE_SHA),
                (&cli_source, CLI_SOURCE_SHA),
                (&generator_source, generator_sha.as_str()),
                (&prompt_path, prompt_sha.as_str()),
            ] {
                if path == &prompt_path {
                    pins.push(json!({"path":path,"sha256":sha}));
                } else {
                    pin(&mut pins, path, sha)?;
                }
            }
            if route == "w4" {
                replace_arg(&mut job, "--model-dir", &package_root.display().to_string())?;
                job["command"]["env"]
                    .as_object_mut()
                    .ok_or("environment missing")?
                    .remove("RVLLM_METAL_METALLIB_BF16");
                pins.extend(package_pins.iter().cloned());
            } else {
                pins.extend(native_pins.iter().cloned());
                pin(&mut pins, &donor, SG8_METALLIB_SHA)?;
                job["command"]["env"]["RVLLM_METAL_METALLIB_BF16"] = json!(donor);
            }
            pins.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
            for pair in pins.windows(2) {
                if pair[0]["path"] == pair[1]["path"] && pair[0]["sha256"] != pair[1]["sha256"] {
                    return Err("conflicting input pins".into());
                }
            }
            pins.dedup_by(|a, b| a["path"] == b["path"]);
            job["inputs"] = json!(pins);
            let mut bytes = serde_json::to_vec_pretty(&job).map_err(|error| error.to_string())?;
            bytes.push(b'\n');
            files.push((output.join(format!("{slug}-{route}-job.json")), bytes));
            prior = Some(id);
        }
    }
    fs::create_dir(&output).map_err(|error| format!("{}: {error}", output.display()))?;
    for (path, bytes) in files {
        write_new(&path, &bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_flag_is_rejected() {
        let mut job = json!({"command":{"args":["--model-dir","a","--model-dir","b"]}});
        assert!(replace_arg(&mut job, "--model-dir", "c").is_err());
    }

    #[test]
    fn package_count_is_exact() {
        let missing = Path::new("/definitely-missing-fulltext-w4-package");
        assert!(package_files(missing).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn sealed_snapshot_symlink_keeps_declared_file_name() {
        let dir = tempfile::tempdir().unwrap();
        let blob = dir.path().join("blob");
        fs::write(&blob, b"sealed tokenizer").unwrap();
        let link = dir.path().join("tokenizer.json");
        std::os::unix::fs::symlink(&blob, &link).unwrap();
        let sha = digest(&blob).unwrap();
        assert_eq!(verified_path(&link, &sha).unwrap(), link);
        assert_eq!(
            verified_path(&link, &sha).unwrap().file_name().unwrap(),
            "tokenizer.json"
        );
    }
}
