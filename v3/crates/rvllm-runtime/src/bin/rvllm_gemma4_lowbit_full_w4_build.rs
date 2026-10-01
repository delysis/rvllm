//! Run one sealed, create-new all-text-projection W4 package export.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

type Result<T> = std::result::Result<T, String>;

const PLAN_SHA: &str = "087e5e87f8ef2fd68e6d910af17867ec2cf12ef9344e9b70cf481a6992591d7e";
const BUILDER_SHA: &str = "c9cc445b139d747548f94c8d6ec141f7059b7111aafd9c5562d4678c3509ae16";
const MODEL_SHA: &str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
const CONFIG_SHA: &str = "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const DONOR_SHA: &str = "21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06";
const PACKAGE_ID: &str = "gemma4-12b-it-fulltext-w4-v1";
const OUTPUT_NAME: &str = "gemma4-12b-it-fulltext-w4-v1.rvllm";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    model_sha256: String,
    config_sha256: String,
    tokenizer_sha256: String,
    donor_metallib_sha256: String,
    group_size: u64,
    text_projection_elements: u64,
    selectors: Vec<Selector>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selector {
    tensor_name: String,
    shape: [u64; 2],
    w4_argument: String,
    w8_argument: String,
}

fn main() {
    let args = env::args_os().map(PathBuf::from).collect::<Vec<_>>();
    if !matches!(args.len(), 6 | 7) {
        eprintln!("usage: rvllm_gemma4_lowbit_full_w4_build SEALED_PLAN.json FROZEN_BUILDER MODEL_DIR METALLIB_ROOT NEW_OUTPUT.rvllm [--verify-only]");
        std::process::exit(2);
    }
    if let Err(error) = run(&args[1..]) {
        eprintln!("rvllm_gemma4_lowbit_full_w4_build: {error}");
        std::process::exit(1);
    }
}

fn verify_file(path: &Path, expected: &str) -> Result<()> {
    let sha = Sha256Digest::file(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if sha.as_str() != expected {
        return Err(format!(
            "{}: SHA-256 differs from sealed input",
            path.display()
        ));
    }
    Ok(())
}

fn verify_plan(plan: &Plan) -> Result<()> {
    if plan.schema != "rvllm.gemma4.lowbit-full-text-selectors.v1"
        || plan.model_sha256 != MODEL_SHA
        || plan.config_sha256 != CONFIG_SHA
        || plan.tokenizer_sha256 != TOKENIZER_SHA
        || plan.donor_metallib_sha256 != DONOR_SHA
        || plan.group_size != 32
        || plan.text_projection_elements != 10_899_947_520
        || plan.selectors.len() != 328
    {
        return Err("sealed selector plan metadata differs".into());
    }
    let mut names = BTreeSet::new();
    for selector in &plan.selectors {
        if !names.insert(selector.tensor_name.as_str())
            || selector.shape[0] == 0
            || selector.shape[1] == 0
            || selector.shape[1] % 32 != 0
            || selector.w4_argument != format!("{}=w4a16-group32", selector.tensor_name)
            || selector.w8_argument != format!("{}=w8a16-group32", selector.tensor_name)
        {
            return Err(format!("invalid selector {}", selector.tensor_name));
        }
    }
    Ok(())
}

fn run(args: &[PathBuf]) -> Result<()> {
    let (args, verify_only) = match args {
        [first, second, third, fourth, fifth] => ([first, second, third, fourth, fifth], false),
        [first, second, third, fourth, fifth, flag] if flag == "--verify-only" => {
            ([first, second, third, fourth, fifth], true)
        }
        _ => return Err("expected five arguments and optional --verify-only".into()),
    };
    let [plan_path, builder, model_dir, metallib_root, output] = args;
    if !output.is_absolute()
        || output.file_name().and_then(|name| name.to_str()) != Some(OUTPUT_NAME)
        || output.exists()
    {
        return Err("output must be the new absolute sealed package path".into());
    }
    let parent = output.parent().ok_or("output has no parent")?;
    let canonical_parent = parent.canonicalize().map_err(|error| error.to_string())?;
    if canonical_parent.file_name().and_then(|name| name.to_str()) != Some("target")
        || canonical_parent
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            != Some("v3")
    {
        return Err("output must be under the selected v3/target".into());
    }
    verify_file(plan_path, PLAN_SHA)?;
    verify_file(builder, BUILDER_SHA)?;
    verify_file(&model_dir.join("model.safetensors"), MODEL_SHA)?;
    verify_file(&model_dir.join("config.json"), CONFIG_SHA)?;
    verify_file(&model_dir.join("tokenizer.json"), TOKENIZER_SHA)?;
    verify_file(&metallib_root.join("macos/bf16/rvllm.metallib"), DONOR_SHA)?;
    let bytes = fs::read(plan_path).map_err(|error| error.to_string())?;
    let plan: Plan =
        serde_json::from_value(parse_strict_json(&bytes).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    verify_plan(&plan)?;

    if verify_only {
        println!("verified_w4_text_projections: {}", plan.selectors.len());
        println!("sealed_plan_sha256: {PLAN_SHA}");
        println!("package_builder_sha256: {BUILDER_SHA}");
        return Ok(());
    }

    let mut command = Command::new(builder);
    command
        .arg("build")
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--metallib-root")
        .arg(metallib_root)
        .arg("--output")
        .arg(output)
        .arg("--package-id")
        .arg(PACKAGE_ID)
        .arg("--weight-format")
        .arg("bf16");
    for selector in &plan.selectors {
        command.arg("--low-bit-proj").arg(&selector.w4_argument);
    }
    println!("sealed_plan_sha256: {PLAN_SHA}");
    println!("package_builder_sha256: {BUILDER_SHA}");
    println!("requested_w4_text_projections: {}", plan.selectors.len());
    let status = command
        .status()
        .map_err(|error| format!("launch package builder: {error}"))?;
    if !status.success() {
        return Err(format!("package builder exited with {status}"));
    }
    let manifest_path = output.join("rvllm-apple-model.json");
    let manifest_bytes = fs::read(&manifest_path).map_err(|error| error.to_string())?;
    let manifest: serde_json::Value =
        parse_strict_json(&manifest_bytes).map_err(|error| error.to_string())?;
    let sidecars = manifest["low_bit_tensors"]
        .as_array()
        .ok_or("built package lacks low-bit tensor list")?;
    if manifest["package_id"] != PACKAGE_ID || sidecars.len() != plan.selectors.len() {
        return Err("built package identity or sidecar count differs".into());
    }
    if sidecars.iter().any(|tensor| {
        tensor["format"] != "w4_a16"
            || tensor["group_size"] != 32
            || tensor["activation_float_type"] != "bf16"
    }) {
        return Err("built package sidecar format differs".into());
    }
    let built = sidecars
        .iter()
        .filter_map(|tensor| tensor["tensor_name"].as_str())
        .collect::<BTreeSet<_>>();
    if built != names_from_plan(&plan) {
        return Err("built package sidecar names differ".into());
    }
    let manifest_sha = Sha256Digest::file(&manifest_path).map_err(|error| error.to_string())?;
    println!("built_package_manifest_sha256: {}", manifest_sha.as_str());
    Ok(())
}

fn names_from_plan(plan: &Plan) -> BTreeSet<&str> {
    plan.selectors
        .iter()
        .map(|selector| selector.tensor_name.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_plan_fails_closed() {
        let plan = Plan {
            schema: "rvllm.gemma4.lowbit-full-text-selectors.v1".into(),
            model_sha256: MODEL_SHA.into(),
            config_sha256: CONFIG_SHA.into(),
            tokenizer_sha256: TOKENIZER_SHA.into(),
            donor_metallib_sha256: DONOR_SHA.into(),
            group_size: 32,
            text_projection_elements: 10_899_947_520,
            selectors: Vec::new(),
        };
        assert!(verify_plan(&plan).is_err());
    }
}
