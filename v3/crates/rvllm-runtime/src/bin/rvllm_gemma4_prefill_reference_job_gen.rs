//! Freeze same-token-boundary HF and Metal prefill-final reference jobs.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SOURCE_SHA: &str = "a1d115405e86ab51bf275473cc04f452f185db2c18f286b5173d2234d4b993b4";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const LOGIC_SHA: &str = "4a32900cfe775763996088b23e6580a4b12f26dbbf78b819391d9d8899c7ed49";
const CS_SHA: &str = "9e14b98bc1f9b27e23668411e4cf721f43b64644bf8131748d31d4a859ffe433";
const HF_SHA: &str = "7af03b53d77cd57ee47824c1e60b1990f1fe289d98dc79731b60bc9e2b57cd55";
const METAL_OFF_SHA: &str = "f31206fce73fdd169a788d8c40e500cc82dd4fec128836eb406d364d786457b3";
const METAL_COMBINED_SHA: &str = "832fe84814b8bcbe5ab18dd253f41051dafa17d20ca138110b61360d5a3cf679";
const METAL_EXECUTABLE_SHA: &str =
    "23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463";
const METAL_SOURCE_SHA: &str = "c091c7069ad563271ae2d46361190c0f8f451f521abfe41d7d30b749c4fbdf04";
const DATASET_SHA: &str = "58769d67ced092719390e496f1d4bbf8b99c43b77bba39d054b9a09aecb63cab";

type Result<T> = std::result::Result<T, String>;

fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    if args.len() != 10 {
        eprintln!("usage: rvllm_gemma4_prefill_reference_job_gen SOURCE.json TOKENIZER.json LOGIC_OFF.json CS_OFF.json HF_TEMPLATE.json METAL_OFF_TEMPLATE.json METAL_COMBINED_TEMPLATE.json METAL_EXECUTABLE NEW_DIR");
        std::process::exit(2);
    }
    let paths = args[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    if let Err(error) = run(&paths) {
        eprintln!("rvllm_gemma4_prefill_reference_job_gen: {error}");
        std::process::exit(1);
    }
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|value| value.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn sealed(path: &Path, sha: &str) -> Result<(PathBuf, Value)> {
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if digest(&canonical)? != sha {
        return Err(format!("{}: frozen SHA-256 changed", canonical.display()));
    }
    let bytes =
        fs::read(&canonical).map_err(|error| format!("{}: {error}", canonical.display()))?;
    let value = parse_strict_json(&bytes).map_err(|error| error.to_string())?;
    Ok((canonical, value))
}

fn args_mut(job: &mut Value) -> Result<&mut Vec<Value>> {
    job["command"]["args"]
        .as_array_mut()
        .ok_or("command arguments are not an array".into())
}

fn replace_arg(job: &mut Value, flag: &str, value: Value) -> Result<()> {
    let args = args_mut(job)?;
    let matches = args
        .windows(2)
        .enumerate()
        .filter_map(|(i, pair)| (pair[0].as_str() == Some(flag)).then_some(i + 1))
        .collect::<Vec<_>>();
    if matches.len() != 1 || !args[matches[0]].is_string() {
        return Err(format!("expected one text-valued {flag}"));
    }
    args[matches[0]] = value;
    Ok(())
}

fn remove_pin(job: &mut Value, suffix: &str) -> Result<()> {
    let pins = job["inputs"]
        .as_array_mut()
        .ok_or("inputs are not an array")?;
    let matches = pins
        .iter()
        .enumerate()
        .filter_map(|(i, pin)| {
            pin["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(suffix))
                .then_some(i)
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected exactly one pin ending {suffix}"));
    }
    pins.remove(matches[0]);
    Ok(())
}

fn pin(job: &mut Value, path: &Path, sha: &str) -> Result<()> {
    job["inputs"]
        .as_array_mut()
        .ok_or("inputs are not an array")?
        .push(json!({"path": path, "sha256": sha}));
    Ok(())
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| format!("tokenize: {error}"))?;
    let mut ids = vec![2];
    ids.extend_from_slice(encoded.get_ids());
    Ok(ids)
}

fn source_case(
    tokenizer: &tokenizers::Tokenizer,
    source: &Value,
    id: &str,
) -> Result<(Vec<u32>, u32)> {
    let cases = source["cases"].as_array().ok_or("source cases missing")?;
    let matches = cases
        .iter()
        .filter(|case| case["id"].as_str() == Some(id))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected exactly one source case {id}"));
    }
    let prompt = matches[0]["prompt"].as_str().ok_or("prompt missing")?;
    let continuation = matches[0]["continuation"]
        .as_str()
        .ok_or("continuation missing")?;
    let prompt_ids = encode(tokenizer, prompt)?;
    let full_ids = encode(tokenizer, &format!("{prompt}{continuation}"))?;
    let targets = full_ids
        .strip_prefix(prompt_ids.as_slice())
        .filter(|tail| !tail.is_empty())
        .ok_or("continuation retokenized at boundary")?;
    Ok((prompt_ids, targets[0]))
}

fn source_pin(job: &Value, suffix: &str, sha: &str) -> Result<PathBuf> {
    let pins = job["inputs"].as_array().ok_or("inputs missing")?;
    let matches = pins
        .iter()
        .filter(|pin| {
            pin["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(suffix))
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 || matches[0]["sha256"].as_str() != Some(sha) {
        return Err(format!("missing or wrong frozen {suffix} pin"));
    }
    Ok(PathBuf::from(matches[0]["path"].as_str().unwrap()))
}

fn run(paths: &[PathBuf]) -> Result<()> {
    let (source_path, source) = sealed(&paths[0], SOURCE_SHA)?;
    let tokenizer_path = paths[1]
        .canonicalize()
        .map_err(|error| format!("tokenizer: {error}"))?;
    if digest(&tokenizer_path)? != TOKENIZER_SHA {
        return Err("tokenizer changed".into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("tokenizer: {error}"))?;
    let (_, logic_template) = sealed(&paths[2], LOGIC_SHA)?;
    let (_, cs_template) = sealed(&paths[3], CS_SHA)?;
    let (_, hf_template) = sealed(&paths[4], HF_SHA)?;
    let (_, metal_off_template) = sealed(&paths[5], METAL_OFF_SHA)?;
    let (_, metal_combined_template) = sealed(&paths[6], METAL_COMBINED_SHA)?;
    let metal_executable = paths[7]
        .canonicalize()
        .map_err(|error| format!("Metal executable: {error}"))?;
    if digest(&metal_executable)? != METAL_EXECUTABLE_SHA {
        return Err("frozen Metal executable changed".into());
    }
    let metal_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_metal_infer.rs");
    if digest(&metal_source)? != METAL_SOURCE_SHA {
        return Err("Metal CLI source changed".into());
    }
    if paths[8].exists() {
        return Err(format!("{} already exists", paths[8].display()));
    }
    let parent = paths[8]
        .parent()
        .ok_or("output has no parent")?
        .canonicalize()
        .map_err(|error| format!("output parent: {error}"))?;
    let output = parent.join(paths[8].file_name().ok_or("output has no name")?);
    let generator = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/rvllm_gemma4_prefill_reference_job_gen.rs");
    let generator_sha = digest(&generator)?;
    let mut jobs = Vec::new();
    let mut after = None::<String>;
    for (slug, id, original) in [
        ("logic", "mmlu-formal-logic-2443", &logic_template),
        ("cs", "mmlu-high-school-computer-science-3185", &cs_template),
    ] {
        let (prompt_ids, target) = source_case(&tokenizer, &source, id)?;
        let original_targets = original["command"]["args"]
            .as_array()
            .and_then(|args| {
                args.windows(2)
                    .find(|pair| pair[0].as_str() == Some("--teacher-token-ids"))
                    .and_then(|pair| pair[1].as_str())
            })
            .ok_or("original teacher target IDs missing")?;
        if original_targets.split(',').next() != Some(target.to_string().as_str()) {
            return Err(format!("{id}: first target differs from submitted trial"));
        }
        let prompt_file = source_pin(
            original,
            &format!("mmlu-{slug}-prompt.jsonl"),
            if slug == "logic" {
                "a3739872c640c5efaf71ed10c6f41469dc6b6c7869253e951cb2553fff8ae0b5"
            } else {
                "9ba52ea797d5342a2f9ee33dd4f07c61e608fecd7060eabe34c836df36110427"
            },
        )?;
        let dataset = source_pin(original, "mmlu-test.arrow", DATASET_SHA)?;
        let ids_text = prompt_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        for route in ["hf", "off", "combined"] {
            let job_id = format!("prefill26-mmlu-prefill-ref-{slug}-{route}-v1-20260927");
            let mut job = match route {
                "hf" => hf_template.clone(),
                "off" => metal_off_template.clone(),
                "combined" => metal_combined_template.clone(),
                _ => unreachable!(),
            };
            job["id"] = json!(job_id);
            job["after"] = after.as_ref().map_or_else(|| json!([]), |id| json!([id]));
            if route == "hf" {
                replace_arg(&mut job, "--prompt-token-ids", json!(ids_text))?;
                replace_arg(&mut job, "--selected-token-ids", json!(target.to_string()))?;
                replace_arg(
                    &mut job,
                    "--output",
                    json!(output.join(format!("{slug}-hf-reference.json"))),
                )?;
                remove_pin(&mut job, "mlx-source-m304.json")?;
            } else {
                job["command"]["executable"] = json!({
                    "path": metal_executable,
                    "sha256": METAL_EXECUTABLE_SHA,
                });
                replace_arg(&mut job, "--teacher-prompt-jsonl", json!(prompt_file))?;
                replace_arg(&mut job, "--teacher-token-ids", json!(target.to_string()))?;
                remove_pin(&mut job, "prompt-m304-only.jsonl")?;
                remove_pin(&mut job, "hf-m304-one-step-compatible-reference.json")?;
                remove_pin(&mut job, "rvllm_metal_infer.rs")?;
                pin(&mut job, &metal_source, METAL_SOURCE_SHA)?;
                pin(
                    &mut job,
                    &prompt_file,
                    if slug == "logic" {
                        "a3739872c640c5efaf71ed10c6f41469dc6b6c7869253e951cb2553fff8ae0b5"
                    } else {
                        "9ba52ea797d5342a2f9ee33dd4f07c61e608fecd7060eabe34c836df36110427"
                    },
                )?;
            }
            pin(&mut job, &source_path, SOURCE_SHA)?;
            pin(&mut job, &dataset, DATASET_SHA)?;
            pin(&mut job, &generator, &generator_sha)?;
            let mut bytes = serde_json::to_vec_pretty(&job).map_err(|error| error.to_string())?;
            bytes.push(b'\n');
            jobs.push((output.join(format!("{slug}-{route}-job.json")), bytes));
            after = Some(job_id);
        }
    }
    fs::create_dir(&output).map_err(|error| format!("{}: {error}", output.display()))?;
    for (path, bytes) in jobs {
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
    fn duplicate_argument_is_rejected() {
        let mut job = json!({"command":{"args":["--output","a","--output","b"]}});
        assert!(replace_arg(&mut job, "--output", json!("c")).is_err());
    }

    #[test]
    fn missing_pin_is_rejected() {
        let mut job = json!({"inputs":[]});
        assert!(remove_pin(&mut job, "source.json").is_err());
    }
}
