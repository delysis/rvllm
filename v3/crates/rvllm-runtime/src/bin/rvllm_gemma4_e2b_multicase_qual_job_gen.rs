//! Freeze three fresh serial E2B jobs for multi-case teacher-route qualification.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

const SOURCE_SHA: &str = "9a8e792014de7096b0ea89c9a2440448acc00797e26676ce67dd35a4b13cb17c";
const PROTOCOL_SHA: &str = "8dced743830e7a11efa4bb6790116ac674d16b60bf5381cc7f4c6cb1eb3503c0";
const CLI_SOURCE_SHA: &str = "2682c8abf2dbe6d97bffd2959a37964e541cab2c17853ce910ad98c7377c79a2";
const EXECUTABLE_SHA: &str = "149d447d70227edf265c52c721534531bb69a9c31cda44b5512566054db30d06";
const SNAPSHOT: &str = "6befbaca7398925921802abd1f277b495b78b738";
const MODEL_FILES: [(&str, &str); 4] = [
    (
        "config.json",
        "bbeff1e2fd3fe282536e7ace02309d43e0dbd9b6ac4b6a149b97e3ab6942a878",
    ),
    (
        "generation_config.json",
        "b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9",
    ),
    (
        "model.safetensors",
        "33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137",
    ),
    (
        "tokenizer.json",
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    ),
];
const CASES: [(&str, &[u32]); 2] = [
    ("observatory", &[108355, 236761]),
    ("kitchen", &[6819, 236761]),
];

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

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|value| value.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn pin(path: &Path, expected: &str) -> Result<Value> {
    if !path.is_absolute() || !path.is_file() || digest(path)? != expected {
        return Err(format!("{}: sealed absolute input differs", path.display()));
    }
    // Preserve snapshot symlink basenames used by the model loader.
    Ok(json!({"path":path,"sha256":expected}))
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

fn validate_cases(source: Source, tokenizer: &tokenizers::Tokenizer) -> Result<Vec<Case>> {
    if source.schema != "rvllm.gemma4_heldout_text.v1"
        || !source.provenance.contains("synthetic route-qualification")
        || source.cases.len() != 2
    {
        return Err("source schema, provenance or case count differs".into());
    }
    for (case, (expected_id, targets)) in source.cases.iter().zip(CASES) {
        if case.id != expected_id || case.prompt.is_empty() || case.continuation.is_empty() {
            return Err(format!("{expected_id}: source identity differs"));
        }
        let prompt = encode(tokenizer, &case.prompt)?;
        let full = encode(tokenizer, &format!("{}{}", case.prompt, case.continuation))?;
        if prompt.len() != 184 || full.strip_prefix(prompt.as_slice()) != Some(targets) {
            return Err(format!("{expected_id}: tokenizer prefix or targets differ"));
        }
    }
    Ok(source.cases)
}

fn json_line(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn job(
    id: &str,
    after: &[String],
    args: Vec<String>,
    executable: &Path,
    cwd: &Path,
    inputs: &[Value],
) -> Value {
    json!({
        "schema":"rvllm.experiment_job.v1",
        "id":id,
        "purpose":"correctness",
        "after":after,
        "stable_seconds":0,
        "max_wait_seconds":7200,
        "max_run_seconds":1800,
        "conditions":{
            "disk_path":cwd,
            "minimum_free_bytes":32u64 << 30,
            "power_source":"ac",
            "low_power_mode":null,
            "pmset_power_mode":null,
            "thermal_state":null,
            "idle_llama_servers":[],
            "quiet_process_names":[],
            "observe_process_names":["cargo","rustc","metal","metallib","llama-server","ollama","rvllm_metal_infer"]
        },
        "command":{
            "cwd":cwd,
            "executable":{"path":executable,"sha256":EXECUTABLE_SHA},
            "env":{"RVLLM_METAL_RESEARCH":"off"},
            "args":args
        },
        "inputs":inputs
    })
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
    if paths.len() != 5 {
        return Err("expected SOURCE MODEL_DIR EXECUTABLE PROTOCOL NEW_OUTPUT_DIR".into());
    }
    let [source_path, model, executable, protocol, output] = paths else {
        return Err("five paths required".into());
    };
    if !output.is_absolute() || output.exists() {
        return Err("output must be a new absolute directory".into());
    }
    let cwd = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root unavailable")?;
    let cli_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rvllm_metal_infer.rs");
    let generator_source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/rvllm_gemma4_e2b_multicase_qual_job_gen.rs");
    let mut pins = vec![
        pin(source_path, SOURCE_SHA)?,
        pin(protocol, PROTOCOL_SHA)?,
        pin(executable, EXECUTABLE_SHA)?,
        pin(&cli_source, CLI_SOURCE_SHA)?,
    ];
    pins.push(pin(&generator_source, &digest(&generator_source)?)?);
    if !model.is_absolute() || model.file_name().and_then(|name| name.to_str()) != Some(SNAPSHOT) {
        return Err("model snapshot path differs".into());
    }
    for (name, sha) in MODEL_FILES {
        pins.push(pin(&model.join(name), sha)?);
    }
    let tokenizer = tokenizers::Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|error| error.to_string())?;
    let source: Source =
        parse_strict_json(&fs::read(source_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let cases = validate_cases(source, &tokenizer)?;
    let mut files = Vec::<(PathBuf, Vec<u8>)>::new();
    let mut batch = Vec::<Vec<u8>>::new();
    let mut previous = Vec::<String>::new();
    for (case, (_, targets)) in cases.iter().zip(CASES) {
        let prompt_path = output.join(format!("{}-prompt.jsonl", case.id));
        let prompt_bytes = json_line(&json!({"name":case.id,"prompt":case.prompt}))?;
        let mut job_pins = pins.clone();
        job_pins
            .push(json!({"path":prompt_path,"sha256":Sha256Digest::bytes(&prompt_bytes).as_str()}));
        let id = format!("e2b-multicase-teacher-qual-v1-20261001-{}-single", case.id);
        let args = vec![
            "--model-dir".into(),
            model.display().to_string(),
            "--teacher-prompt-jsonl".into(),
            prompt_path.display().to_string(),
            "--teacher-token-ids".into(),
            targets
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
            "--max-new-tokens".into(),
            targets.len().to_string(),
            "--max-total-tokens".into(),
            "512".into(),
            "--large-model-opt-in".into(),
            "--json".into(),
        ];
        let job = job(&id, &previous, args, executable, cwd, &job_pins);
        files.push((prompt_path, prompt_bytes));
        files.push((
            output.join(format!("{}-single-job.json", case.id)),
            serde_json::to_vec_pretty(&job).map_err(|error| error.to_string())?,
        ));
        previous = vec![id];
        batch.push(json_line(&json!({"name":case.id,"prompt":case.prompt,"max_new_tokens":targets.len(),"max_total_tokens":512,"teacher_token_ids":targets}))?);
    }
    let batch_path = output.join("two-case-teacher.jsonl");
    let batch_bytes = batch.concat();
    let mut batch_pins = pins;
    batch_pins.push(json!({"path":batch_path,"sha256":Sha256Digest::bytes(&batch_bytes).as_str()}));
    let id = "e2b-multicase-teacher-qual-v1-20261001-two-case-batch";
    let args = vec![
        "--model-dir".into(),
        model.display().to_string(),
        "--prompts-jsonl".into(),
        batch_path.display().to_string(),
        "--teacher-session".into(),
        "--large-model-opt-in".into(),
        "--json".into(),
    ];
    let job = job(id, &previous, args, executable, cwd, &batch_pins);
    files.push((batch_path, batch_bytes));
    files.push((
        output.join("two-case-batch-job.json"),
        serde_json::to_vec_pretty(&job).map_err(|error| error.to_string())?,
    ));
    fs::create_dir(output).map_err(|error| format!("{}: {error}", output.display()))?;
    for (path, mut bytes) in files {
        if path.extension().is_some_and(|ext| ext == "json") {
            bytes.push(b'\n');
        }
        write_new(&path, &bytes)?;
    }
    Ok(())
}

fn main() {
    let paths = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if let Err(error) = run(&paths) {
        eprintln!("rvllm_gemma4_e2b_multicase_qual_job_gen: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_requires_exact_order_and_targets() {
        let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let source = Source {
            schema: "wrong".into(),
            provenance: "synthetic route-qualification".into(),
            cases: vec![],
        };
        assert!(validate_cases(source, &tokenizer).is_err());
    }

    #[test]
    fn job_conditions_do_not_gate_on_thermal_stability() {
        let value = job(
            "fresh",
            &[],
            vec![],
            Path::new("/exe"),
            Path::new("/cwd"),
            &[],
        );
        assert_eq!(value["conditions"]["minimum_free_bytes"], 32u64 << 30);
        assert!(value["conditions"]["thermal_state"].is_null());
        assert_eq!(value["stable_seconds"], 0);
    }
}
