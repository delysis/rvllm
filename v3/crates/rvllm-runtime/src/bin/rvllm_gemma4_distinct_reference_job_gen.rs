//! Seal fresh correctness-only HF/Metal prefill-boundary jobs without touching v1.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "2f212a4b1c316e49b52b78fc80e33a62c35772f0028e4da7d52ee7268c3d008a";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const HF_TEMPLATE_SHA: &str = "144965b1c3dc4967b3d828c66b404921e94631f269760e21aa6512c6d3fab4db";
const OFF_TEMPLATE_SHA: &str = "c839bdc8831c80f30160bbea9ad96fa9790f9232b393c1cd98ce39223a35f1a2";
const COMBINED_TEMPLATE_SHA: &str =
    "e1a50e88d764167331ffe08f4033d9c05cdeec797b752b7301b06cc562021965";
const METAL_EXECUTABLE_SHA: &str =
    "23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463";
const METAL_SOURCE_SHA: &str = "c091c7069ad563271ae2d46361190c0f8f451f521abfe41d7d30b749c4fbdf04";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    schema: String,
    provenance: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    prompt: String,
    continuation: String,
}

fn main() {
    let args = env::args_os().map(PathBuf::from).collect::<Vec<_>>();
    if args.len() != 8 {
        eprintln!("usage: rvllm_gemma4_distinct_reference_job_gen SOURCE.json TOKENIZER.json HF_TEMPLATE.json OFF_TEMPLATE.json COMBINED_TEMPLATE.json METAL_EXECUTABLE NEW_DIR");
        std::process::exit(2);
    }
    if let Err(error) = run(&args[1..]) {
        eprintln!("rvllm_gemma4_distinct_reference_job_gen: {error}");
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
        return Err(format!(
            "{}: SHA-256 differs from sealed input",
            path.display()
        ));
    }
    Ok(path)
}

fn sealed_json(path: &Path, expected: &str) -> Result<(PathBuf, Value)> {
    let path = sealed(path, expected)?;
    let bytes = fs::read(&path).map_err(|error| error.to_string())?;
    let value = parse_strict_json(&bytes).map_err(|error| error.to_string())?;
    Ok((path, value))
}

fn replace_arg(job: &mut Value, flag: &str, new_value: String) -> Result<()> {
    let args = job["command"]["args"]
        .as_array_mut()
        .ok_or("command args missing")?;
    let matches = args
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| (pair[0].as_str() == Some(flag)).then_some(index + 1))
        .collect::<Vec<_>>();
    if matches.len() != 1 || !args[matches[0]].is_string() {
        return Err(format!("expected one text-valued {flag}"));
    }
    args[matches[0]] = json!(new_value);
    Ok(())
}

fn remove_pin(job: &mut Value, suffix: &str) -> Result<()> {
    let pins = job["inputs"].as_array_mut().ok_or("inputs missing")?;
    let matches = pins
        .iter()
        .enumerate()
        .filter_map(|(index, pin)| {
            pin["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(suffix))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected one pin ending {suffix}"));
    }
    pins.remove(matches[0]);
    Ok(())
}

fn pin(job: &mut Value, path: &Path, sha: &str) -> Result<()> {
    job["inputs"]
        .as_array_mut()
        .ok_or("inputs missing")?
        .push(json!({"path":path,"sha256":sha}));
    Ok(())
}

fn encoded(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let mut ids = vec![2];
    ids.extend_from_slice(
        tokenizer
            .encode(text, false)
            .map_err(|error| format!("tokenize: {error}"))?
            .get_ids(),
    );
    Ok(ids)
}

fn case_tokens(tokenizer: &tokenizers::Tokenizer, case: &Case) -> Result<(Vec<u32>, u32)> {
    let prompt = encoded(tokenizer, &case.prompt)?;
    let full = encoded(tokenizer, &format!("{}{}", case.prompt, case.continuation))?;
    let suffix = full
        .strip_prefix(prompt.as_slice())
        .filter(|suffix| !suffix.is_empty())
        .ok_or("continuation retokenized at prompt boundary")?;
    Ok((prompt, suffix[0]))
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
    let source_bytes = fs::read(&source_path).map_err(|error| error.to_string())?;
    let source: Source = parse_strict_json(&source_bytes).map_err(|error| error.to_string())?;
    if source.schema != "rvllm.gemma4_heldout_text.v1"
        || source.provenance.trim().is_empty()
        || source.cases.len() != 2
    {
        return Err("source schema, provenance, or case count changed".into());
    }
    let tokenizer_path = sealed(&paths[1], TOKENIZER_SHA)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("tokenizer: {error}"))?;
    let templates = [
        sealed_json(&paths[2], HF_TEMPLATE_SHA)?,
        sealed_json(&paths[3], OFF_TEMPLATE_SHA)?,
        sealed_json(&paths[4], COMBINED_TEMPLATE_SHA)?,
    ];
    let executable = sealed(&paths[5], METAL_EXECUTABLE_SHA)?;
    let metal_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_metal_infer.rs");
    sealed(&metal_source, METAL_SOURCE_SHA)?;
    let generator = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/rvllm_gemma4_distinct_reference_job_gen.rs");
    let generator_sha = digest(&generator)?;
    if paths[6].exists() {
        return Err(format!("{} already exists", paths[6].display()));
    }
    let parent = paths[6]
        .parent()
        .ok_or("output has no parent")?
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let output = parent.join(paths[6].file_name().ok_or("output has no name")?);
    let mut files = Vec::<(PathBuf, Vec<u8>)>::new();
    let mut after = None::<String>;
    for (case, slug, expected_len, expected_target) in source
        .cases
        .iter()
        .zip([("observatory", 213, 80444), ("library", 203, 506)])
        .map(|(case, (slug, len, target))| (case, slug, len, target))
    {
        if case.id
            != format!(
                "{slug}-{}-v2",
                if slug == "observatory" {
                    "clock"
                } else {
                    "ledger"
                }
            )
        {
            return Err(format!("unexpected case ID: {}", case.id));
        }
        let (prompt_ids, target) = case_tokens(&tokenizer, case)?;
        if prompt_ids.len() != expected_len || target != expected_target {
            return Err(format!("{}: sealed token identity changed", case.id));
        }
        let prompt_path = output.join(format!("{slug}-prompt.jsonl"));
        let mut prompt_bytes = serde_json::to_vec(&json!({"name":case.id,"prompt":case.prompt}))
            .map_err(|error| error.to_string())?;
        prompt_bytes.push(b'\n');
        let prompt_sha = Sha256Digest::bytes(&prompt_bytes).as_str().to_owned();
        files.push((prompt_path.clone(), prompt_bytes));
        let ids = prompt_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        for (route, (template_path, template), template_sha) in ["hf", "off", "combined"]
            .into_iter()
            .zip(templates.iter())
            .zip([HF_TEMPLATE_SHA, OFF_TEMPLATE_SHA, COMBINED_TEMPLATE_SHA])
            .map(|((route, template), sha)| (route, template, sha))
        {
            let id = format!("prefill26-prefill-ref-distinct-v2-{slug}-{route}-20260927");
            let mut job = template.clone();
            job["id"] = json!(id);
            job["after"] = after
                .as_ref()
                .map_or_else(|| json!([]), |prior| json!([prior]));
            remove_pin(&mut job, "mmlu-natural-source-v1.json")?;
            remove_pin(&mut job, "mmlu-test.arrow")?;
            remove_pin(&mut job, "rvllm_gemma4_prefill_reference_job_gen.rs")?;
            if route == "hf" {
                replace_arg(&mut job, "--prompt-token-ids", ids.clone())?;
                replace_arg(&mut job, "--selected-token-ids", target.to_string())?;
                replace_arg(
                    &mut job,
                    "--output",
                    output
                        .join(format!("{slug}-hf-reference.json"))
                        .display()
                        .to_string(),
                )?;
            } else {
                remove_pin(&mut job, "mmlu-logic-prompt.jsonl")?;
                replace_arg(
                    &mut job,
                    "--teacher-prompt-jsonl",
                    prompt_path.display().to_string(),
                )?;
                replace_arg(&mut job, "--teacher-token-ids", target.to_string())?;
                job["command"]["executable"] =
                    json!({"path":executable,"sha256":METAL_EXECUTABLE_SHA});
            }
            pin(&mut job, &source_path, SOURCE_SHA)?;
            pin(&mut job, &prompt_path, &prompt_sha)?;
            pin(&mut job, &generator, &generator_sha)?;
            pin(&mut job, template_path, template_sha)?;
            let mut bytes = serde_json::to_vec_pretty(&job).map_err(|error| error.to_string())?;
            bytes.push(b'\n');
            files.push((output.join(format!("{slug}-{route}-job.json")), bytes));
            after = Some(id);
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
    fn duplicate_argument_is_rejected() {
        let mut job = json!({"command":{"args":["--output","a","--output","b"]}});
        assert!(replace_arg(&mut job, "--output", "c".into()).is_err());
    }

    #[test]
    fn missing_pin_is_rejected() {
        let mut job = json!({"inputs":[]});
        assert!(remove_pin(&mut job, "mmlu-test.arrow").is_err());
    }
}
