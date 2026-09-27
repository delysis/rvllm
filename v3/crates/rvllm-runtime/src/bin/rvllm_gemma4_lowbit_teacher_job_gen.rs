//! Freeze a one-layer low-bit quality probe from already-sealed MMLU inputs.
#![forbid(unsafe_code)]

use rvllm_apple::{
    AppleLowBitTensorRole, AppleLowBitWeightFormat, AppleModelPackage, ApplePackageFloatType,
    ApplePackagePlatform,
};
use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde_json::{json, Value};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const LOGIC_TEMPLATE_SHA: &str = "4a32900cfe775763996088b23e6580a4b12f26dbbf78b819391d9d8899c7ed49";
const CS_TEMPLATE_SHA: &str = "9e14b98bc1f9b27e23668411e4cf721f43b64644bf8131748d31d4a859ffe433";
const W4_MANIFEST_SHA: &str = "1b9c43598644619432589b66836ed22e52d1e48c968e4b93134f21bd5e324949";
const W8_MANIFEST_SHA: &str = "1cb72e93d101f23f969b422586bb63058bca0bb23aa80c27e87accff1d162a69";
const EXECUTABLE_SHA: &str = "23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463";
const SG8_METALLIB_SHA: &str = "21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06";
const MODEL_SHA: &str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
const CONFIG_SHA: &str = "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9";
const TOKENIZER_SHA: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const TENSOR: &str = "model.language_model.layers.0.mlp.down_proj.weight";

type Result<T> = std::result::Result<T, String>;

fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    if args.len() != 8 {
        eprintln!(
            "usage: rvllm_gemma4_lowbit_teacher_job_gen LOGIC_OFF_JOB CS_OFF_JOB SG8.metallib W4_PACKAGE W8_PACKAGE EXECUTABLE NEW_OUTPUT_DIR"
        );
        std::process::exit(2);
    }
    let paths = args[1..].iter().map(PathBuf::from).collect::<Vec<_>>();
    if let Err(error) = run(&paths) {
        eprintln!("rvllm_gemma4_lowbit_teacher_job_gen: {error}");
        std::process::exit(1);
    }
}

fn digest(path: &Path) -> Result<String> {
    Sha256Digest::file(path)
        .map(|hash| hash.as_str().to_owned())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn checked(path: &Path, expected: &str) -> Result<PathBuf> {
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if digest(&canonical)? != expected {
        return Err(format!("{}: SHA-256 changed", canonical.display()));
    }
    Ok(canonical)
}

fn read_job(path: &Path, sha: &str, id: &str) -> Result<Value> {
    let path = checked(path, sha)?;
    let bytes = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let value: Value = parse_strict_json(&bytes).map_err(|error| error.to_string())?;
    if value["schema"] != "rvllm.experiment_job.v1"
        || value["id"] != id
        || value["purpose"] != "correctness"
        || value["stable_seconds"] != 0
        || value["command"]["executable"]["sha256"] != EXECUTABLE_SHA
        || value["command"]["env"]["RVLLM_METAL_RESEARCH"] != "off"
    {
        return Err(format!("{id}: template identity changed"));
    }
    Ok(value)
}

fn replace_arg(job: &mut Value, flag: &str, value: &Path) -> Result<()> {
    let args = job["command"]["args"]
        .as_array_mut()
        .ok_or("command arguments are not an array")?;
    let indices = args
        .windows(2)
        .enumerate()
        .filter_map(|(i, pair)| (pair[0].as_str() == Some(flag)).then_some(i + 1))
        .collect::<Vec<_>>();
    if indices.len() != 1 || !args[indices[0]].is_string() {
        return Err(format!("expected one text-valued {flag} argument"));
    }
    args[indices[0]] = json!(value);
    Ok(())
}

fn inputs(job: &mut Value) -> Result<&mut Vec<Value>> {
    job["inputs"]
        .as_array_mut()
        .ok_or("job inputs are not an array".into())
}

fn replace_normal_library(job: &mut Value, path: &Path) -> Result<()> {
    let entries = inputs(job)?;
    let matches = entries
        .iter()
        .enumerate()
        .filter_map(|(i, input)| {
            input["path"]
                .as_str()
                .is_some_and(|path| path.ends_with("/normal.metallib"))
                .then_some(i)
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err("expected exactly one original BF16 library pin".into());
    }
    entries[matches[0]] = json!({"path": path, "sha256": SG8_METALLIB_SHA});
    Ok(())
}

fn pin(job: &mut Value, path: &Path, sha: &str) -> Result<()> {
    inputs(job)?.push(json!({"path": path, "sha256": sha}));
    Ok(())
}

struct Package {
    root: PathBuf,
    manifest_sha: &'static str,
    library: PathBuf,
    packed: PathBuf,
    packed_sha: String,
    scales: PathBuf,
    scales_sha: String,
}

impl Package {
    fn open(
        root: &Path,
        manifest_sha: &'static str,
        format: AppleLowBitWeightFormat,
    ) -> Result<Self> {
        let root = root
            .canonicalize()
            .map_err(|error| format!("{}: {error}", root.display()))?;
        checked(&root.join("rvllm-apple-model.json"), manifest_sha)?;
        let package = AppleModelPackage::open(&root)
            .map_err(|error| format!("{}: {error}", root.display()))?;
        let manifest = package.manifest();
        if manifest.schema_version != 3
            || manifest.model_config.sha256 != CONFIG_SHA
            || manifest.weight_shards.len() != 1
            || manifest.weight_shards[0].file.sha256 != MODEL_SHA
            || !manifest.tokenizer_files.iter().any(|file| {
                file.path == Path::new("tokenizer.json") && file.sha256 == TOKENIZER_SHA
            })
            || manifest.low_bit_tensors.len() != 1
        {
            return Err(format!(
                "{}: checkpoint or sidecar identity changed",
                root.display()
            ));
        }
        let tensor = &manifest.low_bit_tensors[0];
        if tensor.tensor_name != TENSOR
            || tensor.role != AppleLowBitTensorRole::DenseDownProjection
            || tensor.format != format
            || tensor.group_size != 32
            || tensor.activation_float_type != ApplePackageFloatType::Bf16
            || tensor.shape != [3840, 15360]
        {
            return Err(format!(
                "{}: low-bit tensor identity changed",
                root.display()
            ));
        }
        let library = manifest
            .metal_libraries
            .iter()
            .find(|entry| {
                entry.platform == ApplePackagePlatform::MacOs
                    && entry.float_type == ApplePackageFloatType::Bf16
            })
            .ok_or("package lacks a macOS BF16 library")?;
        if library.library.sha256 != SG8_METALLIB_SHA {
            return Err("packaged BF16 library differs from native SG8 control".into());
        }
        Ok(Self {
            library: root.join(&library.library.path),
            packed: root.join(&tensor.packed_values.path),
            packed_sha: tensor.packed_values.sha256.clone(),
            scales: root.join(&tensor.scales.path),
            scales_sha: tensor.scales.sha256.clone(),
            root,
            manifest_sha,
        })
    }

    fn pin(&self, job: &mut Value) -> Result<()> {
        pin(
            job,
            &self.root.join("rvllm-apple-model.json"),
            self.manifest_sha,
        )?;
        pin(job, &self.library, SG8_METALLIB_SHA)?;
        pin(job, &self.packed, &self.packed_sha)?;
        pin(job, &self.scales, &self.scales_sha)?;
        pin(job, &self.root.join("config.json"), CONFIG_SHA)?;
        pin(job, &self.root.join("model.safetensors"), MODEL_SHA)?;
        pin(job, &self.root.join("tokenizer.json"), TOKENIZER_SHA)
    }
}

fn run(paths: &[PathBuf]) -> Result<()> {
    let logic = read_job(
        &paths[0],
        LOGIC_TEMPLATE_SHA,
        "prefill26-mmlu-natural-logic-off-v1-20260927",
    )?;
    let cs = read_job(
        &paths[1],
        CS_TEMPLATE_SHA,
        "prefill26-mmlu-natural-cs-off-v1-20260927",
    )?;
    let sg8 = checked(&paths[2], SG8_METALLIB_SHA)?;
    let w4 = Package::open(&paths[3], W4_MANIFEST_SHA, AppleLowBitWeightFormat::W4A16)?;
    let w8 = Package::open(&paths[4], W8_MANIFEST_SHA, AppleLowBitWeightFormat::W8A16)?;
    let executable = checked(&paths[5], EXECUTABLE_SHA)?;
    let output = &paths[6];
    if output.exists() {
        return Err(format!("{} already exists", output.display()));
    }
    let parent = output.parent().ok_or("output has no parent")?;
    let output = parent
        .canonicalize()
        .map_err(|error| format!("{}: {error}", parent.display()))?
        .join(output.file_name().ok_or("output has no name")?);
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/rvllm_gemma4_lowbit_teacher_job_gen.rs");
    let source_sha = digest(&source)?;
    let mut files = Vec::new();
    let mut after = None::<String>;
    for (slug, template) in [("logic", &logic), ("cs", &cs)] {
        for (route, package) in [("native", None), ("w4", Some(&w4)), ("w8", Some(&w8))] {
            let id = format!("prefill26-lowbit-mmlu-{slug}-{route}-v1-20260927");
            let mut job = template.clone();
            job["id"] = json!(id);
            job["after"] = after.as_ref().map_or_else(|| json!([]), |id| json!([id]));
            job["command"]["executable"]["path"] = json!(executable);
            job["command"]["env"]["RVLLM_METAL_RESEARCH"] = json!("metal-donor12b-sg8");
            replace_normal_library(&mut job, &sg8)?;
            if let Some(package) = package {
                replace_arg(&mut job, "--model-dir", &package.root)?;
                job["command"]["env"]
                    .as_object_mut()
                    .ok_or("command environment is not an object")?
                    .remove("RVLLM_METAL_METALLIB_BF16");
                package.pin(&mut job)?;
            } else {
                job["command"]["env"]["RVLLM_METAL_METALLIB_BF16"] = json!(sg8);
            }
            pin(&mut job, &source, &source_sha)?;
            let mut bytes = serde_json::to_vec_pretty(&job).map_err(|error| error.to_string())?;
            bytes.push(b'\n');
            files.push((output.join(format!("{slug}-{route}-job.json")), bytes));
            after = Some(id);
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
    fn rejects_ambiguous_argument() {
        let mut job = json!({"command":{"args":["--model-dir","a","--model-dir","b"]}});
        assert!(replace_arg(&mut job, "--model-dir", Path::new("c")).is_err());
    }

    #[test]
    fn requires_one_original_library_pin() {
        let mut job = json!({"inputs":[]});
        assert!(replace_normal_library(&mut job, Path::new("/x")).is_err());
    }
}
