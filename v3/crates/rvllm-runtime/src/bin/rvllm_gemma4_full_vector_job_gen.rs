//! Seal distinct, correctness-only HF/Metal full-prefill-vector jobs.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "f6d06d159a16472c4dba9496f2cbdc0b6f3b3b2498b0e05dfae64cc347f603bd";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const TEMPLATE_SHAS: [&str; 3] = [
    "e39d87c30533067c35065ae745469ce48aa49a53a2b550fab8d89193fbec8382",
    "c30da8b58dcc01c377707475985a74f3f732576dc8b55b887efe2f255b628079",
    "d85cc3b0f6fdf8a06a194dd683060bc897bba94b774bde61a2fc6da2d8589b4e",
];
const METAL_EXECUTABLE_SHA: &str =
    "1f9cfedd5c1a91f4a1046e2930e13071847fc0b90c7bafa2f391f3e99b3082c5";
const METAL_SOURCE_SHA: &str = "5158928cc2f97381b8943e75657b0485f4cbb76abd55a56867e2cfb9c05277ec";
const OLD_METAL_SOURCE_SHA: &str =
    "c091c7069ad563271ae2d46361190c0f8f451f521abfe41d7d30b749c4fbdf04";

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
        eprintln!("usage: rvllm_gemma4_full_vector_job_gen SOURCE.json TOKENIZER.json HF_TEMPLATE.json OFF_TEMPLATE.json COMBINED_TEMPLATE.json METAL_EXECUTABLE NEW_DIR");
        std::process::exit(2);
    }
    if let Err(error) = run(&args[1..]) {
        eprintln!("rvllm_gemma4_full_vector_job_gen: {error}");
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

fn replace_arg(job: &mut Value, flag: &str, replacement: String) -> Result<()> {
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
    args[matches[0]] = json!(replacement);
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
        return Err(format!("expected one input pin ending {suffix}"));
    }
    pins.remove(matches[0]);
    Ok(())
}

fn replace_metal_source_pin(job: &mut Value) -> Result<()> {
    let pins = job["inputs"].as_array_mut().ok_or("inputs missing")?;
    let matches = pins
        .iter_mut()
        .filter(|pin| {
            pin["path"]
                .as_str()
                .is_some_and(|path| path.ends_with("/rvllm_metal_infer.rs"))
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 || matches[0]["sha256"].as_str() != Some(OLD_METAL_SOURCE_SHA) {
        return Err("old Metal source pin absent or changed".into());
    }
    matches.into_iter().next().unwrap()["sha256"] = json!(METAL_SOURCE_SHA);
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
    let source: Source = parse_strict_json(&fs::read(&source_path).map_err(|e| e.to_string())?)
        .map_err(|error| error.to_string())?;
    if source.schema != "rvllm.gemma4_heldout_text.v1"
        || !source.provenance.contains("Codex-authored synthetic")
        || source.cases.len() != 2
    {
        return Err("source schema, provenance, or case count changed".into());
    }
    let tokenizer_path = sealed(&paths[1], TOKENIZER_SHA)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("tokenizer: {error}"))?;
    let templates = [
        sealed_json(&paths[2], TEMPLATE_SHAS[0])?,
        sealed_json(&paths[3], TEMPLATE_SHAS[1])?,
        sealed_json(&paths[4], TEMPLATE_SHAS[2])?,
    ];
    let executable = sealed(&paths[5], METAL_EXECUTABLE_SHA)?;
    let metal_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_metal_infer.rs");
    sealed(&metal_source, METAL_SOURCE_SHA)?;
    let generator =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_gemma4_full_vector_job_gen.rs");
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
    for (case, slug, expected_id, expected_len, expected_target) in source
        .cases
        .iter()
        .zip([
            ("bridge", "bridge-cable-v3", 229, 56896),
            ("seed", "seed-bank-v3", 246, 52102),
        ])
        .map(|(case, (slug, id, len, target))| (case, slug, id, len, target))
    {
        if case.id != expected_id {
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
            .zip(TEMPLATE_SHAS)
            .map(|((route, template), sha)| (route, template, sha))
        {
            let id = format!("prefill26-fullvec-v3-{slug}-{route}-20260927");
            let mut job = template.clone();
            job["id"] = json!(id);
            job["after"] = after
                .as_ref()
                .map_or_else(|| json!([]), |prior| json!([prior]));
            remove_pin(&mut job, "numerical-reference-distinct-v2-source.json")?;
            remove_pin(&mut job, "observatory-prompt.jsonl")?;
            remove_pin(&mut job, "rvllm_gemma4_distinct_reference_job_gen.rs")?;
            remove_pin(&mut job, &format!("logic-{route}-job.json"))?;
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
                replace_metal_source_pin(&mut job)?;
                replace_arg(
                    &mut job,
                    "--teacher-prompt-jsonl",
                    prompt_path.display().to_string(),
                )?;
                replace_arg(&mut job, "--teacher-token-ids", target.to_string())?;
                job["command"]["args"]
                    .as_array_mut()
                    .ok_or("command args missing")?
                    .extend([
                        json!("--teacher-prefill-full-logits-output"),
                        json!(output
                            .join(format!("{slug}-{route}-full-vector.json"))
                            .display()
                            .to_string()),
                    ]);
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
    fn stale_metal_source_pin_is_required() {
        let mut job = json!({"inputs":[{"path":"/x/rvllm_metal_infer.rs","sha256":"wrong"}]});
        assert!(replace_metal_source_pin(&mut job).is_err());
        job["inputs"][0]["sha256"] = json!(OLD_METAL_SOURCE_SHA);
        replace_metal_source_pin(&mut job).unwrap();
        assert_eq!(job["inputs"][0]["sha256"], METAL_SOURCE_SHA);
    }
}
