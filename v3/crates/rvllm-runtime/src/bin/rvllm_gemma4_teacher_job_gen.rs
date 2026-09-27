//! Build immutable queue inputs for the predeclared two-row MMLU probe.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{Sha256Digest, parse_strict_json};
use serde::Deserialize;
use serde_json::{Value, json};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SOURCE_SHA: &str = "a1d115405e86ab51bf275473cc04f452f185db2c18f286b5173d2234d4b993b4";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const ARROW_SHA: &str = "58769d67ced092719390e496f1d4bbf8b99c43b77bba39d054b9a09aecb63cab";
const EXECUTABLE_SHA: &str = "23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463";
const CASES: [(&str, &str, usize, usize); 2] = [
    ("mmlu-formal-logic-2443", "logic", 230, 24),
    ("mmlu-high-school-computer-science-3185", "cs", 254, 55),
];

type Result<T> = std::result::Result<T, String>;

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
    let args = env::args_os().collect::<Vec<_>>();
    if args.len() != 7 {
        eprintln!(
            "usage: rvllm_gemma4_teacher_job_gen SOURCE.json TOKENIZER.json TEST.arrow OFF_TEMPLATE.json COMBINED_TEMPLATE.json NEW_OUTPUT_DIR"
        );
        std::process::exit(2);
    }
    let paths = args[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    if let Err(error) = run(&paths) {
        eprintln!("rvllm_gemma4_teacher_job_gen: {error}");
        std::process::exit(1);
    }
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|value| value.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn checked_file(path: &Path, expected: &str) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if digest(&path)? != expected {
        return Err(format!("{}: SHA-256 changed", path.display()));
    }
    Ok(path)
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse_strict_json(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let mut ids = vec![2];
    ids.extend_from_slice(
        tokenizer
            .encode(text, false)
            .map_err(|error| format!("tokenize: {error}"))?
            .get_ids(),
    );
    Ok(ids)
}

fn replace_arg(job: &mut Value, flag: &str, replacement: String) -> Result<()> {
    let args = job["command"]["args"]
        .as_array_mut()
        .ok_or("job command args are not an array")?;
    let indices = args
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| (pair[0].as_str() == Some(flag)).then_some(index + 1))
        .collect::<Vec<_>>();
    if indices.len() != 1 || !args[indices[0]].is_string() {
        return Err(format!("expected one text-valued {flag} argument"));
    }
    args[indices[0]] = Value::String(replacement);
    Ok(())
}

fn replace_pin(job: &mut Value, old_suffix: &str, path: &Path, sha: &str) -> Result<()> {
    let inputs = job["inputs"]
        .as_array_mut()
        .ok_or("job inputs are not an array")?;
    let matches = inputs
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            item["path"]
                .as_str()
                .is_some_and(|value| value.ends_with(old_suffix))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected one pin ending {old_suffix}"));
    }
    inputs[matches[0]]["path"] = json!(path);
    inputs[matches[0]]["sha256"] = json!(sha);
    Ok(())
}

fn validate_template(job: &Value, route: &str) -> Result<()> {
    if job["schema"] != "rvllm.experiment_job.v1"
        || job["purpose"] != "correctness"
        || job["stable_seconds"] != 0
        || job["command"]["executable"]["sha256"] != EXECUTABLE_SHA
        || job["command"]["env"]["RVLLM_METAL_RESEARCH"] != route
    {
        return Err(format!("{route} template identity changed"));
    }
    Ok(())
}

fn prepared(job: &Value, id: &str, after: Option<&str>) -> Result<()> {
    if job["id"] != id || job["after"] != after.map_or_else(|| json!([]), |value| json!([value])) {
        return Err("generated job ID or dependency mismatch".into());
    }
    let inputs = job["inputs"].as_array().ok_or("job inputs missing")?;
    if inputs.iter().any(|item| {
        !item["path"]
            .as_str()
            .is_some_and(|path| Path::new(path).is_absolute())
            || item["sha256"].as_str().is_none_or(|sha| sha.len() != 64)
    }) {
        return Err("generated job contains an unsealed input".into());
    }
    Ok(())
}

fn run(paths: &[PathBuf]) -> Result<()> {
    let source_path = checked_file(&paths[0], SOURCE_SHA)?;
    let tokenizer_path = checked_file(&paths[1], TOKENIZER_SHA)?;
    let arrow_path = checked_file(&paths[2], ARROW_SHA)?;
    let off = read_json(&paths[3])?;
    let combined = read_json(&paths[4])?;
    validate_template(&off, "off")?;
    validate_template(&combined, "metal-prefill-pipeline32x64-q4k16")?;
    let source: Source = serde_json::from_value(read_json(&source_path)?)
        .map_err(|error| format!("source schema: {error}"))?;
    if source.schema != "rvllm.gemma4_heldout_text.v1"
        || source.provenance.is_empty()
        || source.cases.len() != CASES.len()
    {
        return Err("source case identity changed".into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("tokenizer: {error}"))?;
    let output = &paths[5];
    if output.exists() {
        return Err(format!("{} already exists", output.display()));
    }
    let parent = output.parent().ok_or("output has no parent")?;
    let parent = parent
        .canonicalize()
        .map_err(|error| format!("{}: {error}", parent.display()))?;
    let output = parent.join(output.file_name().ok_or("output has no name")?);
    let generator_source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_gemma4_heldout_tokens.rs");
    let generator_sha = digest(&generator_source)?;
    let this_source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_gemma4_teacher_job_gen.rs");
    let this_sha = digest(&this_source)?;
    let mut files = Vec::<(PathBuf, Vec<u8>)>::new();
    let mut prior_id = None::<String>;
    for ((expected_id, slug, prompt_len, target_len), case) in CASES.iter().zip(source.cases) {
        if case.id != *expected_id {
            return Err("source case order or ID changed".into());
        }
        let prompt_ids = encode(&tokenizer, &case.prompt)?;
        let full_ids = encode(&tokenizer, &format!("{}{}", case.prompt, case.continuation))?;
        let targets = full_ids
            .strip_prefix(prompt_ids.as_slice())
            .filter(|tail| !tail.is_empty())
            .ok_or("target retokenized the prompt boundary")?;
        if prompt_ids.len() != *prompt_len || targets.len() != *target_len {
            return Err(format!("{} token lengths changed", case.id));
        }
        let prompt_path = output.join(format!("mmlu-{slug}-prompt.jsonl"));
        let mut prompt_bytes = serde_json::to_vec(&json!({"name": case.id, "prompt": case.prompt}))
            .map_err(|error| error.to_string())?;
        prompt_bytes.push(b'\n');
        let prompt_sha = Sha256Digest::bytes(&prompt_bytes).as_str().to_owned();
        files.push((prompt_path.clone(), prompt_bytes));
        for (route, template) in [("off", &off), ("combined", &combined)] {
            let id = format!("prefill26-mmlu-natural-{slug}-{route}-v1-20260927");
            let mut job = template.clone();
            job["id"] = json!(id);
            job["after"] = prior_id
                .as_ref()
                .map_or_else(|| json!([]), |old| json!([old]));
            replace_arg(
                &mut job,
                "--teacher-prompt-jsonl",
                prompt_path.display().to_string(),
            )?;
            replace_arg(
                &mut job,
                "--teacher-token-ids",
                targets
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            )?;
            replace_arg(&mut job, "--max-new-tokens", targets.len().to_string())?;
            replace_pin(&mut job, "heldout-text-v1.json", &source_path, SOURCE_SHA)?;
            replace_pin(
                &mut job,
                "heldout-map-prompt.jsonl",
                &prompt_path,
                &prompt_sha,
            )?;
            replace_pin(
                &mut job,
                "rvllm_gemma4_heldout_tokens.rs",
                &generator_source,
                &generator_sha,
            )?;
            job["inputs"].as_array_mut().unwrap().extend([
                json!({"path": arrow_path, "sha256": ARROW_SHA}),
                json!({"path": this_source, "sha256": this_sha}),
            ]);
            prepared(&job, &id, prior_id.as_deref())?;
            let mut bytes = serde_json::to_vec_pretty(&job).map_err(|error| error.to_string())?;
            bytes.push(b'\n');
            files.push((output.join(format!("mmlu-{slug}-{route}-job.json")), bytes));
            prior_id = Some(id);
        }
    }
    fs::create_dir(&output).map_err(|error| format!("{}: {error}", output.display()))?;
    for (path, bytes) in files {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        file.write_all(&bytes)
            .map_err(|error| format!("{}: {error}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_refuses_ambiguous_argument() {
        let mut job = json!({"command":{"args":["--x","a","--x","b"]}});
        assert!(replace_arg(&mut job, "--x", "c".into()).is_err());
    }

    #[test]
    fn replacement_changes_only_selected_argument() {
        let mut job = json!({"command":{"args":["--x","a","--y","b"]}});
        replace_arg(&mut job, "--x", "c".into()).unwrap();
        assert_eq!(job["command"]["args"], json!(["--x", "c", "--y", "b"]));
    }
}
