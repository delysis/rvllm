//! Deterministic, fail-closed assembly of self-contained Apple model packages.

use crate::model_package::{
    chat_template_fingerprint, fingerprint_assets, hash_file, model_assets_fingerprint,
};
use crate::{
    quantize_apple_low_bit_reference, AppleLowBitTensor, AppleLowBitTensorRole,
    AppleLowBitWeightFormat, AppleMetalLibrary, AppleModelPackage, AppleModelPackageManifest,
    ApplePackageFile, ApplePackageFloatType, ApplePackagePlatform, AppleWeightFormat,
    AppleWeightShard, APPLE_LOW_BIT_GROUP_SIZE, APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
    APPLE_MODEL_PACKAGE_MANIFEST,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

const MAX_SAFETENSORS_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MODEL_METADATA_NAMES: &[&str] = &[
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
];
const TOKENIZER_NAMES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "added_tokens.json",
];
const REQUIRED_METAL_VARIANTS: &[(ApplePackagePlatform, &str, ApplePackageFloatType, &str)] = &[
    (
        ApplePackagePlatform::MacOs,
        "macos",
        ApplePackageFloatType::F16,
        "f16",
    ),
    (
        ApplePackagePlatform::MacOs,
        "macos",
        ApplePackageFloatType::Bf16,
        "bf16",
    ),
    (
        ApplePackagePlatform::Ios,
        "ios",
        ApplePackageFloatType::F16,
        "f16",
    ),
    (
        ApplePackagePlatform::Ios,
        "ios",
        ApplePackageFloatType::Bf16,
        "bf16",
    ),
    (
        ApplePackagePlatform::IosSimulator,
        "ios-simulator",
        ApplePackageFloatType::F16,
        "f16",
    ),
    (
        ApplePackagePlatform::IosSimulator,
        "ios-simulator",
        ApplePackageFloatType::Bf16,
        "bf16",
    ),
];

#[derive(Clone, Debug)]
pub struct AppleModelPackageBuildConfig {
    pub model_dir: PathBuf,
    pub metallib_root: PathBuf,
    pub output_dir: PathBuf,
    pub package_id: String,
    /// `None` detects the one exact native 16-bit dtype present in all shards.
    /// Low-bit packages require a dedicated exporter and are intentionally not
    /// inferred from an ordinary Hugging Face safetensors directory.
    pub weight_format: Option<AppleWeightFormat>,
    /// Explicit tensor-level hybrid exports. The initial production contract
    /// accepts dense MLP down projections only and preserves native weights.
    pub low_bit_down_projections: Vec<AppleLowBitExportRequest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleLowBitExportRequest {
    pub tensor_name: String,
    pub format: AppleLowBitWeightFormat,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleModelPackageBuildReport {
    pub output_dir: PathBuf,
    pub identity_fingerprint: String,
    pub architecture: String,
    pub weight_format: AppleWeightFormat,
    pub weight_shards: usize,
    pub weight_bytes: u64,
    pub low_bit_tensors: usize,
    pub low_bit_bytes: u64,
    pub low_bit_formats: Vec<AppleLowBitWeightFormat>,
    pub metal_variants: usize,
}

pub fn build_apple_model_package(
    config: &AppleModelPackageBuildConfig,
) -> Result<AppleModelPackageBuildReport, String> {
    validate_build_paths(config)?;
    let parsed_config = parse_model_config(&config.model_dir.join("config.json"))?;
    let source_weights = resolve_weight_shards(&config.model_dir)?;
    let inspected = inspect_weight_set(&source_weights)?;
    let weight_format = select_weight_format(config.weight_format, inspected.dtype)?;

    let output_parent = config
        .output_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent)
        .map_err(|error| format!("create output parent {}: {error}", output_parent.display()))?;
    reject_output_inside_sources(config, output_parent)?;
    let output_name = config
        .output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "output directory must have a UTF-8 final component".to_owned())?;
    let staging_path = output_parent.join(format!(
        ".{output_name}.rvllm-staging-{}",
        std::process::id()
    ));
    if staging_path.exists() {
        return Err(format!(
            "staging path already exists; remove it after checking no packager is active: {}",
            staging_path.display()
        ));
    }
    fs::create_dir(&staging_path).map_err(|error| {
        format!(
            "create staging directory {}: {error}",
            staging_path.display()
        )
    })?;
    let mut staging = StagingDir::new(staging_path);

    let model_config = copy_asset(
        &config.model_dir.join("config.json"),
        staging.path(),
        Path::new("config.json"),
    )?;
    let mut model_metadata_files = Vec::new();
    for name in MODEL_METADATA_NAMES {
        let source = config.model_dir.join(name);
        if source.is_file() {
            model_metadata_files.push(copy_asset(&source, staging.path(), Path::new(name))?);
        }
    }
    if let Some(index) = &source_weights.index {
        model_metadata_files.push(copy_asset(
            index,
            staging.path(),
            Path::new("model.safetensors.index.json"),
        )?);
    }
    model_metadata_files.sort_by(|left, right| left.path.cmp(&right.path));

    let mut tokenizer_files = Vec::new();
    for name in TOKENIZER_NAMES {
        let source = config.model_dir.join(name);
        if source.is_file() {
            tokenizer_files.push(copy_asset(&source, staging.path(), Path::new(name))?);
        }
    }
    if !tokenizer_files
        .iter()
        .any(|file| file.path == Path::new("tokenizer.json"))
    {
        return Err("model directory requires tokenizer.json".to_owned());
    }
    tokenizer_files.sort_by(|left, right| left.path.cmp(&right.path));

    let shard_count = u32::try_from(source_weights.shards.len())
        .map_err(|_| "weight shard count exceeds u32".to_owned())?;
    let mut weight_shards = Vec::with_capacity(source_weights.shards.len());
    for (index, source) in source_weights.shards.iter().enumerate() {
        let relative = source
            .file_name()
            .map(PathBuf::from)
            .ok_or_else(|| format!("weight shard has no filename: {}", source.display()))?;
        weight_shards.push(AppleWeightShard {
            shard_index: u32::try_from(index)
                .map_err(|_| "weight shard index overflow".to_owned())?,
            shard_count,
            format: weight_format,
            file: copy_asset(source, staging.path(), &relative)?,
        });
    }

    let low_bit_tensors = export_low_bit_down_projections(
        &config.low_bit_down_projections,
        &inspected,
        staging.path(),
    )?;
    let metal_libraries = copy_and_validate_metal_libraries(&config.metallib_root, staging.path())?;
    let tokenizer_fingerprint =
        fingerprint_assets(b"rvllm.apple.tokenizer-assets.v2\0", tokenizer_files.iter());
    let chat_template_fingerprint = chat_template_fingerprint(&config.model_dir)
        .map_err(|error| format!("fingerprint chat template: {error}"))?;
    let numerical_abi_fingerprint = fingerprint_assets(
        b"rvllm.apple.metal-pipeline-abi.v2\0",
        metal_libraries
            .iter()
            .map(|library| &library.pipeline_manifest),
    );
    let mut manifest = AppleModelPackageManifest {
        schema_version: if low_bit_tensors.is_empty() {
            AppleModelPackageManifest::SCHEMA_V2
        } else {
            AppleModelPackageManifest::SCHEMA_V3
        },
        package_id: config.package_id.clone(),
        architecture: parsed_config.architecture,
        model_fingerprint: String::new(),
        tokenizer_fingerprint,
        chat_template_fingerprint,
        numerical_abi_fingerprint,
        model_config,
        model_metadata_files,
        tokenizer_files,
        weight_shards,
        low_bit_tensors,
        metal_libraries,
    };
    manifest.model_fingerprint = model_assets_fingerprint(&manifest);
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("encode Apple model package manifest: {error}"))?;
    manifest_bytes.push(b'\n');
    let manifest_path = staging.path().join(APPLE_MODEL_PACKAGE_MANIFEST);
    let mut manifest_file = File::create(&manifest_path)
        .map_err(|error| format!("create {}: {error}", manifest_path.display()))?;
    manifest_file
        .write_all(&manifest_bytes)
        .map_err(|error| format!("write {}: {error}", manifest_path.display()))?;
    manifest_file
        .sync_all()
        .map_err(|error| format!("sync {}: {error}", manifest_path.display()))?;

    let package = AppleModelPackage::open(staging.path())
        .map_err(|error| format!("validate assembled Apple model package: {error}"))?;
    let identity_fingerprint = lower_hex(&package.identity_fingerprint());
    let weight_bytes = package
        .manifest()
        .weight_shards
        .iter()
        .map(|shard| shard.file.bytes)
        .sum();
    let low_bit_bytes = package
        .manifest()
        .low_bit_tensors
        .iter()
        .map(|tensor| tensor.packed_values.bytes + tensor.scales.bytes)
        .sum();
    let low_bit_formats = package
        .manifest()
        .low_bit_tensors
        .iter()
        .map(|tensor| tensor.format)
        .collect();
    let report = AppleModelPackageBuildReport {
        output_dir: config.output_dir.clone(),
        identity_fingerprint,
        architecture: package.manifest().architecture.clone(),
        weight_format,
        weight_shards: package.manifest().weight_shards.len(),
        weight_bytes,
        low_bit_tensors: package.manifest().low_bit_tensors.len(),
        low_bit_bytes,
        low_bit_formats,
        metal_variants: package.manifest().metal_libraries.len(),
    };
    drop(package);
    fs::rename(staging.path(), &config.output_dir).map_err(|error| {
        format!(
            "atomically install package {} -> {}: {error}",
            staging.path().display(),
            config.output_dir.display()
        )
    })?;
    staging.keep = true;
    Ok(report)
}

fn validate_build_paths(config: &AppleModelPackageBuildConfig) -> Result<(), String> {
    if config.package_id.trim().is_empty() {
        return Err("package ID must not be empty".to_owned());
    }
    for (label, path) in [
        ("model directory", &config.model_dir),
        ("metallib root", &config.metallib_root),
    ] {
        if !path.is_dir() {
            return Err(format!("{label} is not a directory: {}", path.display()));
        }
    }
    if config.output_dir.exists() {
        return Err(format!(
            "output already exists (packages are immutable): {}",
            config.output_dir.display()
        ));
    }
    Ok(())
}

fn reject_output_inside_sources(
    config: &AppleModelPackageBuildConfig,
    output_parent: &Path,
) -> Result<(), String> {
    let output_parent = fs::canonicalize(output_parent)
        .map_err(|error| format!("resolve output parent {}: {error}", output_parent.display()))?;
    let output_name = config
        .output_dir
        .file_name()
        .ok_or_else(|| "output directory has no final component".to_owned())?;
    let output = output_parent.join(output_name);
    for (label, source) in [
        ("model directory", &config.model_dir),
        ("metallib root", &config.metallib_root),
    ] {
        let source = fs::canonicalize(source)
            .map_err(|error| format!("resolve {label} {}: {error}", source.display()))?;
        if output.starts_with(&source) {
            return Err(format!(
                "output must not be created inside the source {label}: {}",
                output.display()
            ));
        }
    }
    Ok(())
}

struct ParsedModelConfig {
    architecture: String,
}

fn parse_model_config(path: &Path) -> Result<ParsedModelConfig, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("{} root must be an object", path.display()))?;
    let architecture = object
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{} requires architectures[0]", path.display()))?
        .to_owned();
    let model_type = object
        .get("model_type")
        .and_then(Value::as_str)
        .or_else(|| {
            object
                .get("text_config")
                .and_then(Value::as_object)
                .and_then(|text| text.get("model_type"))
                .and_then(Value::as_str)
        });
    if !model_type.is_some_and(|value| !value.trim().is_empty()) {
        return Err(format!(
            "{} requires a non-empty model_type",
            path.display()
        ));
    }
    Ok(ParsedModelConfig { architecture })
}

struct SourceWeightSet {
    index: Option<PathBuf>,
    shards: Vec<PathBuf>,
    weight_map: BTreeMap<String, String>,
}

fn resolve_weight_shards(model_dir: &Path) -> Result<SourceWeightSet, String> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let monolith = model_dir.join("model.safetensors");
    let discovered = fs::read_dir(model_dir)
        .map_err(|error| format!("list {}: {error}", model_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("list {}: {error}", model_dir.display()))?
        .into_iter()
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("safetensors"))
        .collect::<BTreeSet<_>>();

    if index_path.is_file() {
        if monolith.exists() {
            return Err(
                "ambiguous weights: both model.safetensors and model.safetensors.index.json exist"
                    .to_owned(),
            );
        }
        let bytes = fs::read(&index_path)
            .map_err(|error| format!("read {}: {error}", index_path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse {}: {error}", index_path.display()))?;
        let map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .filter(|map| !map.is_empty())
            .ok_or_else(|| format!("{} requires a non-empty weight_map", index_path.display()))?;
        let mut weight_map = BTreeMap::new();
        let mut shard_names = BTreeSet::new();
        for (tensor, shard) in map {
            let shard = shard
                .as_str()
                .ok_or_else(|| format!("weight_map entry {tensor:?} is not a string"))?;
            validate_root_filename(shard)?;
            if !shard.ends_with(".safetensors") {
                return Err(format!("weight_map shard is not safetensors: {shard}"));
            }
            shard_names.insert(shard.to_owned());
            weight_map.insert(tensor.clone(), shard.to_owned());
        }
        let shards = shard_names
            .iter()
            .map(|name| model_dir.join(name))
            .collect::<Vec<_>>();
        let expected = shards.iter().cloned().collect::<BTreeSet<_>>();
        if discovered != expected {
            return Err(format!(
                "ambiguous or incomplete sharded weights: index names {expected:?}, directory contains {discovered:?}"
            ));
        }
        for shard in &shards {
            if !shard.is_file() {
                return Err(format!("missing indexed weight shard: {}", shard.display()));
            }
        }
        Ok(SourceWeightSet {
            index: Some(index_path),
            shards,
            weight_map,
        })
    } else {
        if !monolith.is_file() || discovered != BTreeSet::from([monolith.clone()]) {
            return Err(
                "unsharded models require exactly model.safetensors and no shard-like extras"
                    .to_owned(),
            );
        }
        Ok(SourceWeightSet {
            index: None,
            shards: vec![monolith],
            weight_map: BTreeMap::new(),
        })
    }
}

fn validate_root_filename(value: &str) -> Result<(), String> {
    let path = Path::new(value);
    let mut components = path.components();
    if path.is_absolute()
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(format!("unsafe or nested shard path: {value:?}"));
    }
    Ok(())
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum NativeWeightDtype {
    F16,
    Bf16,
}

struct InspectedWeights {
    dtype: NativeWeightDtype,
    tensors: BTreeMap<String, InspectedTensor>,
}

struct InspectedTensor {
    dtype: NativeWeightDtype,
    shape: Vec<usize>,
    file: PathBuf,
    file_offset: u64,
    nbytes: u64,
}

fn inspect_weight_set(weights: &SourceWeightSet) -> Result<InspectedWeights, String> {
    let mut dtype = None;
    let mut tensors = BTreeMap::new();
    for shard in &weights.shards {
        let shard_name = shard
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("weight shard filename is not UTF-8: {}", shard.display()))?;
        for (tensor, inspected) in inspect_safetensors(shard)? {
            let tensor_dtype = inspected.dtype;
            if let Some(previous) =
                tensors.insert(tensor.clone(), (shard_name.to_owned(), inspected))
            {
                return Err(format!(
                    "tensor {tensor:?} appears in both {:?} and {shard_name:?}",
                    previous.0
                ));
            }
            match dtype {
                Some(previous) if previous != tensor_dtype => {
                    return Err(
                        "mixed F16/BF16 tensors are not a supported package format".to_owned()
                    )
                }
                None => dtype = Some(tensor_dtype),
                _ => {}
            }
        }
    }
    if !weights.weight_map.is_empty() {
        if weights.weight_map.len() != tensors.len() {
            return Err("safetensors index does not cover exactly every tensor".to_owned());
        }
        for (tensor, (actual_shard, _)) in &tensors {
            if weights.weight_map.get(tensor) != Some(actual_shard) {
                return Err(format!(
                    "safetensors index maps tensor {tensor:?} to a different shard"
                ));
            }
        }
    }
    Ok(InspectedWeights {
        dtype: dtype.ok_or_else(|| "safetensors set contains no tensors".to_owned())?,
        tensors: tensors
            .into_iter()
            .map(|(name, (_, tensor))| (name, tensor))
            .collect(),
    })
}

fn inspect_safetensors(path: &Path) -> Result<Vec<(String, InspectedTensor)>, String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let file_bytes = file
        .metadata()
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| format!("read safetensors prefix {}: {error}", path.display()))?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(format!(
            "{} has invalid safetensors header length {header_bytes}",
            path.display()
        ));
    }
    let payload_start = 8_u64
        .checked_add(header_bytes)
        .ok_or_else(|| format!("{} safetensors header overflows", path.display()))?;
    if payload_start > file_bytes {
        return Err(format!(
            "{} safetensors header exceeds file",
            path.display()
        ));
    }
    let header_len = usize::try_from(header_bytes)
        .map_err(|_| format!("{} safetensors header is too large", path.display()))?;
    let mut bytes = vec![0_u8; header_len];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("read safetensors header {}: {error}", path.display()))?;
    let object: serde_json::Map<String, Value> = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse safetensors header {}: {error}", path.display()))?;
    let mut tensors = Vec::new();
    for (name, metadata) in object {
        if name == "__metadata__" {
            continue;
        }
        let metadata = metadata.as_object().ok_or_else(|| {
            format!(
                "{} tensor {name:?} metadata is not an object",
                path.display()
            )
        })?;
        let dtype = match metadata.get("dtype").and_then(Value::as_str) {
            Some("F16") => NativeWeightDtype::F16,
            Some("BF16") => NativeWeightDtype::Bf16,
            Some(other) => {
                return Err(format!(
                    "{} tensor {name:?} uses unsupported dtype {other:?}",
                    path.display()
                ))
            }
            None => return Err(format!("{} tensor {name:?} has no dtype", path.display())),
        };
        let shape = metadata
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{} tensor {name:?} has no shape", path.display()))?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| format!("{} tensor {name:?} has invalid shape", path.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape.is_empty() || shape.contains(&0) {
            return Err(format!(
                "{} tensor {name:?} has an empty or zero shape",
                path.display()
            ));
        }
        let offsets = metadata
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| format!("{} tensor {name:?} has invalid offsets", path.display()))?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("{} tensor {name:?} has invalid start", path.display()))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("{} tensor {name:?} has invalid end", path.display()))?;
        let absolute_end = payload_start.checked_add(end);
        if end < start || absolute_end.map_or(true, |end| end > file_bytes) {
            return Err(format!(
                "{} tensor {name:?} data range is outside the shard",
                path.display()
            ));
        }
        let expected_bytes = shape
            .iter()
            .try_fold(1_u64, |count, dimension| {
                count.checked_mul(*dimension as u64)
            })
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| format!("{} tensor {name:?} byte size overflows", path.display()))?;
        if end - start != expected_bytes {
            return Err(format!(
                "{} tensor {name:?} byte length does not match shape",
                path.display()
            ));
        }
        tensors.push((
            name,
            InspectedTensor {
                dtype,
                shape,
                file: path.to_path_buf(),
                file_offset: payload_start + start,
                nbytes: end - start,
            },
        ));
    }
    Ok(tensors)
}

fn select_weight_format(
    requested: Option<AppleWeightFormat>,
    actual: NativeWeightDtype,
) -> Result<AppleWeightFormat, String> {
    let detected = match actual {
        NativeWeightDtype::F16 => AppleWeightFormat::F16,
        NativeWeightDtype::Bf16 => AppleWeightFormat::Bf16,
    };
    match requested {
        None => Ok(detected),
        Some(AppleWeightFormat::F16 | AppleWeightFormat::Bf16) if requested == Some(detected) => {
            Ok(detected)
        }
        Some(AppleWeightFormat::W4A16Group32 | AppleWeightFormat::W8A16Group32) => Err(
            "low-bit package assembly requires the dedicated W4/W8 exporter; ordinary HF safetensors may not be relabeled"
                .to_owned(),
        ),
        Some(other) => Err(format!(
            "requested weight format {other:?} does not match detected {detected:?}"
        )),
    }
}

fn export_low_bit_down_projections(
    requests: &[AppleLowBitExportRequest],
    inspected: &InspectedWeights,
    output_root: &Path,
) -> Result<Vec<AppleLowBitTensor>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    if inspected.dtype != NativeWeightDtype::F16 {
        return Err(
            "hybrid W4A16/W8A16 sidecars currently require a uniform F16 source checkpoint"
                .to_owned(),
        );
    }

    let mut requests = requests.to_vec();
    requests.sort_by(|left, right| {
        left.tensor_name
            .cmp(&right.tensor_name)
            .then_with(|| left.format.bits().cmp(&right.format.bits()))
    });
    let mut seen = BTreeSet::new();
    let mut exported = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        if !seen.insert(request.tensor_name.as_str()) {
            return Err(format!(
                "duplicate low-bit down-projection request for {:?}",
                request.tensor_name
            ));
        }
        if !request.tensor_name.ends_with(".mlp.down_proj.weight")
            || request.tensor_name.contains(".experts.")
        {
            return Err(format!(
                "low-bit tensor {:?} is not a supported dense MLP down projection",
                request.tensor_name
            ));
        }
        let tensor = inspected.tensors.get(&request.tensor_name).ok_or_else(|| {
            format!(
                "low-bit down-projection tensor {:?} is absent from the source checkpoint",
                request.tensor_name
            )
        })?;
        if tensor.dtype != NativeWeightDtype::F16 || tensor.shape.len() != 2 {
            return Err(format!(
                "low-bit down-projection tensor {:?} must be a two-dimensional F16 tensor",
                request.tensor_name
            ));
        }
        let rows = tensor.shape[0];
        let k = tensor.shape[1];
        let row_bytes = k
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| {
                format!(
                    "low-bit tensor {:?} row size overflows",
                    request.tensor_name
                )
            })?;
        let expected_bytes = rows
            .checked_mul(row_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                format!(
                    "low-bit tensor {:?} source byte size overflows",
                    request.tensor_name
                )
            })?;
        if tensor.nbytes != expected_bytes {
            return Err(format!(
                "low-bit tensor {:?} source byte length disagrees with its shape",
                request.tensor_name
            ));
        }
        let rows_u32 = u32::try_from(rows).map_err(|_| {
            format!(
                "low-bit tensor {:?} row count exceeds u32",
                request.tensor_name
            )
        })?;
        let k_u32 = u32::try_from(k).map_err(|_| {
            format!(
                "low-bit tensor {:?} K dimension exceeds u32",
                request.tensor_name
            )
        })?;

        let relative_dir = PathBuf::from("weights").join("low-bit");
        fs::create_dir_all(output_root.join(&relative_dir)).map_err(|error| {
            format!(
                "create low-bit output directory {}: {error}",
                output_root.join(&relative_dir).display()
            )
        })?;
        let stem = format!("{index:05}-{}", request.format.name());
        let values_relative = relative_dir.join(format!("{stem}.values.bin"));
        let scales_relative = relative_dir.join(format!("{stem}.scales.f16le"));
        let values_path = output_root.join(&values_relative);
        let scales_path = output_root.join(&scales_relative);
        let mut values_file = File::create(&values_path)
            .map_err(|error| format!("create {}: {error}", values_path.display()))?;
        let mut scales_file = File::create(&scales_path)
            .map_err(|error| format!("create {}: {error}", scales_path.display()))?;
        let mut source = File::open(&tensor.file)
            .map_err(|error| format!("open {}: {error}", tensor.file.display()))?;
        source
            .seek(SeekFrom::Start(tensor.file_offset))
            .map_err(|error| format!("seek {}: {error}", tensor.file.display()))?;
        let mut source_row = vec![0_u8; row_bytes];
        let mut source_f32 = vec![0.0_f32; k];
        for row in 0..rows {
            source.read_exact(&mut source_row).map_err(|error| {
                format!(
                    "read row {row} of low-bit source tensor {:?}: {error}",
                    request.tensor_name
                )
            })?;
            for (column, bytes) in source_row.chunks_exact(2).enumerate() {
                source_f32[column] =
                    half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
            }
            let packed = quantize_apple_low_bit_reference(request.format, 1, k, &source_f32)
                .map_err(|error| {
                    format!(
                        "quantize row {row} of low-bit tensor {:?}: {error}",
                        request.tensor_name
                    )
                })?;
            values_file
                .write_all(packed.packed_values())
                .map_err(|error| format!("write {}: {error}", values_path.display()))?;
            for scale in packed.scales() {
                scales_file
                    .write_all(&scale.to_bits().to_le_bytes())
                    .map_err(|error| format!("write {}: {error}", scales_path.display()))?;
            }
        }
        values_file
            .sync_all()
            .map_err(|error| format!("sync {}: {error}", values_path.display()))?;
        scales_file
            .sync_all()
            .map_err(|error| format!("sync {}: {error}", scales_path.display()))?;
        drop(values_file);
        drop(scales_file);

        exported.push(AppleLowBitTensor {
            tensor_name: request.tensor_name.clone(),
            role: AppleLowBitTensorRole::DenseDownProjection,
            format: request.format,
            abi_version: APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
            group_size: u16::try_from(APPLE_LOW_BIT_GROUP_SIZE)
                .map_err(|_| "Apple low-bit group size exceeds u16".to_owned())?,
            activation_float_type: ApplePackageFloatType::F16,
            shape: [rows_u32, k_u32],
            packed_values: describe_asset(output_root, &values_relative)?,
            scales: describe_asset(output_root, &scales_relative)?,
        });
    }
    Ok(exported)
}

fn copy_and_validate_metal_libraries(
    source_root: &Path,
    output_root: &Path,
) -> Result<Vec<AppleMetalLibrary>, String> {
    let mut libraries = Vec::with_capacity(REQUIRED_METAL_VARIANTS.len());
    let mut manifests_by_dtype: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    for &(platform, platform_name, float_type, dtype_name) in REQUIRED_METAL_VARIANTS {
        let source_dir = source_root.join(platform_name).join(dtype_name);
        let source_library = source_dir.join("rvllm.metallib");
        let source_manifest = source_dir.join("pipelines.json");
        if !source_library.is_file() || !source_manifest.is_file() {
            return Err(format!(
                "missing required Metal artifacts for {platform_name}/{dtype_name} under {}",
                source_root.display()
            ));
        }
        let manifest_bytes = fs::read(&source_manifest)
            .map_err(|error| format!("read {}: {error}", source_manifest.display()))?;
        validate_pipeline_manifest(
            &source_manifest,
            &manifest_bytes,
            match float_type {
                ApplePackageFloatType::F16 => "float16",
                ApplePackageFloatType::Bf16 => "bfloat16",
            },
        )?;
        if let Some(previous) = manifests_by_dtype.get(dtype_name) {
            if previous != &manifest_bytes {
                return Err(format!(
                    "pipeline manifests disagree across platforms for dtype {dtype_name}"
                ));
            }
        } else {
            manifests_by_dtype.insert(dtype_name, manifest_bytes);
        }
        let relative_dir = PathBuf::from("metal").join(platform_name).join(dtype_name);
        libraries.push(AppleMetalLibrary {
            platform,
            float_type,
            library: copy_asset(
                &source_library,
                output_root,
                &relative_dir.join("rvllm.metallib"),
            )?,
            pipeline_manifest: copy_asset(
                &source_manifest,
                output_root,
                &relative_dir.join("pipelines.json"),
            )?,
        });
    }
    Ok(libraries)
}

fn describe_asset(output_root: &Path, relative: &Path) -> Result<ApplePackageFile, String> {
    let path = output_root.join(relative);
    let metadata =
        fs::metadata(&path).map_err(|error| format!("stat {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!(
            "generated asset is empty or not a regular file: {}",
            path.display()
        ));
    }
    let sha256 = hash_file(&path).map_err(|error| format!("hash {}: {error}", path.display()))?;
    Ok(ApplePackageFile {
        path: relative.to_path_buf(),
        bytes: metadata.len(),
        sha256,
    })
}

fn validate_pipeline_manifest(path: &Path, bytes: &[u8], dtype: &str) -> Result<(), String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    if value.get("schema").and_then(Value::as_str) != Some("rvllm.apple-metal.pipeline-manifest.v1")
        || value.get("dtype").and_then(Value::as_str) != Some(dtype)
        || value.get("kv_page_tokens").and_then(Value::as_u64) != Some(32)
    {
        return Err(format!(
            "{} has unsupported pipeline schema, dtype, or page size",
            path.display()
        ));
    }
    let kernels = value
        .get("kernels")
        .and_then(Value::as_array)
        .filter(|kernels| !kernels.is_empty())
        .ok_or_else(|| format!("{} requires a non-empty kernels array", path.display()))?;
    if value.get("kernel_count").and_then(Value::as_u64) != Some(kernels.len() as u64) {
        return Err(format!(
            "{} kernel_count does not match kernels",
            path.display()
        ));
    }
    let mut names = BTreeSet::new();
    for kernel in kernels {
        let name = kernel
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| format!("{} contains an invalid kernel name", path.display()))?;
        if !names.insert(name) {
            return Err(format!(
                "{} contains duplicate kernel {name:?}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn copy_asset(
    source: &Path,
    output_root: &Path,
    relative: &Path,
) -> Result<ApplePackageFile, String> {
    if relative.is_absolute()
        || relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "unsafe package output path: {}",
            relative.display()
        ));
    }
    if !source.is_file() {
        return Err(format!("source asset is missing: {}", source.display()));
    }
    let destination = output_root.join(relative);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    fs::copy(source, &destination).map_err(|error| {
        format!(
            "copy {} to {}: {error}",
            source.display(),
            destination.display()
        )
    })?;
    let metadata = fs::metadata(&destination)
        .map_err(|error| format!("stat {}: {error}", destination.display()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!(
            "copied asset is empty or not a regular file: {}",
            destination.display()
        ));
    }
    let sha256 = hash_file(&destination)
        .map_err(|error| format!("hash {}: {error}", destination.display()))?;
    Ok(ApplePackageFile {
        path: relative.to_path_buf(),
        bytes: metadata.len(),
        sha256,
    })
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

struct StagingDir {
    path: PathBuf,
    keep: bool,
}

impl StagingDir {
    fn new(path: PathBuf) -> Self {
        Self { path, keep: false }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!("rvllm-package-builder-{name}-{nonce}"))
    }

    fn write_safetensors(path: &Path, dtype: &str, tensor: &str) {
        let header = serde_json::json!({
            tensor: {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}
        });
        let header = serde_json::to_vec(&header).expect("encode header");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0, 0]);
        fs::write(path, bytes).expect("write safetensors");
    }

    fn write_f16_matrix_safetensors(
        path: &Path,
        tensor: &str,
        rows: usize,
        k: usize,
        values: &[f32],
    ) {
        assert_eq!(values.len(), rows * k);
        let payload = values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let header = serde_json::json!({
            tensor: {
                "dtype": "F16",
                "shape": [rows, k],
                "data_offsets": [0, payload.len()]
            }
        });
        let header = serde_json::to_vec(&header).expect("encode header");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        fs::write(path, bytes).expect("write matrix safetensors");
    }

    fn write_metallibs(root: &Path) {
        for &(_platform, platform_name, float_type, dtype_name) in REQUIRED_METAL_VARIANTS {
            let dir = root.join(platform_name).join(dtype_name);
            fs::create_dir_all(&dir).expect("create metal dir");
            fs::write(dir.join("rvllm.metallib"), b"metallib").expect("write metallib");
            let manifest = serde_json::json!({
                "schema": "rvllm.apple-metal.pipeline-manifest.v1",
                "dtype": match float_type {
                    ApplePackageFloatType::F16 => "float16",
                    ApplePackageFloatType::Bf16 => "bfloat16",
                },
                "kv_page_tokens": 32,
                "kernel_count": 1,
                "kernels": ["kernel"]
            });
            fs::write(
                dir.join("pipelines.json"),
                serde_json::to_vec(&manifest).expect("encode pipeline manifest"),
            )
            .expect("write pipeline manifest");
        }
    }

    fn fixture(name: &str) -> (PathBuf, AppleModelPackageBuildConfig) {
        let root = root(name);
        let model = root.join("model");
        let metal = root.join("metal");
        fs::create_dir_all(&model).expect("create model");
        fs::write(
            model.join("config.json"),
            br#"{"architectures":["TinyForCausalLM"],"model_type":"tiny"}"#,
        )
        .expect("write config");
        fs::write(model.join("tokenizer.json"), b"tokenizer").expect("write tokenizer");
        write_safetensors(&model.join("model.safetensors"), "F16", "weight");
        write_metallibs(&metal);
        let config = AppleModelPackageBuildConfig {
            model_dir: model,
            metallib_root: metal,
            output_dir: root.join("package"),
            package_id: "tiny-test".to_owned(),
            weight_format: None,
            low_bit_down_projections: Vec::new(),
        };
        (root, config)
    }

    #[test]
    fn builds_deterministic_self_contained_package() {
        let (root, config) = fixture("valid");
        let report = build_apple_model_package(&config).expect("build package");
        assert_eq!(report.weight_format, AppleWeightFormat::F16);
        assert_eq!(report.metal_variants, 6);
        assert!(config.output_dir.join("config.json").is_file());
        assert!(config.output_dir.join("model.safetensors").is_file());
        AppleModelPackage::open(&config.output_dir).expect("open built package");
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn rejects_ambiguous_weight_layout() {
        let (root, config) = fixture("ambiguous");
        fs::write(
            config.model_dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .expect("write index");
        let error = build_apple_model_package(&config).expect_err("ambiguous weights fail");
        assert!(error.contains("ambiguous weights"));
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn rejects_missing_platform_dtype_artifact() {
        let (root, config) = fixture("missing-metal");
        fs::remove_file(
            config
                .metallib_root
                .join("ios")
                .join("bf16")
                .join("rvllm.metallib"),
        )
        .expect("remove metallib");
        let error = build_apple_model_package(&config).expect_err("missing artifact fails");
        assert!(error.contains("ios/bf16"));
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn refuses_to_relabel_native_weights_as_low_bit() {
        let (root, mut config) = fixture("low-bit");
        config.weight_format = Some(AppleWeightFormat::W4A16Group32);
        let error = build_apple_model_package(&config).expect_err("low-bit relabel fails");
        assert!(error.contains("dedicated W4/W8 exporter"));
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn exports_authenticated_hybrid_down_projection_without_relabelling_native_shard() {
        let (root, mut config) = fixture("hybrid-low-bit");
        let tensor_name = "model.layers.0.mlp.down_proj.weight";
        let source_values = (0..66)
            .map(|index| index as f32 / 9.0 - 3.5)
            .collect::<Vec<_>>();
        let source_path = config.model_dir.join("model.safetensors");
        write_f16_matrix_safetensors(&source_path, tensor_name, 2, 33, &source_values);
        let source_before = fs::read(&source_path).expect("read source before export");
        config
            .low_bit_down_projections
            .push(AppleLowBitExportRequest {
                tensor_name: tensor_name.to_owned(),
                format: AppleLowBitWeightFormat::W4A16,
            });

        let report = build_apple_model_package(&config).expect("build hybrid package");
        assert_eq!(report.weight_format, AppleWeightFormat::F16);
        assert_eq!(report.low_bit_tensors, 1);
        assert_eq!(report.low_bit_formats, vec![AppleLowBitWeightFormat::W4A16]);
        assert_eq!(
            fs::read(&source_path).expect("read source after export"),
            source_before
        );

        let package = AppleModelPackage::open(&config.output_dir).expect("open hybrid package");
        assert_eq!(
            package.manifest().schema_version,
            AppleModelPackageManifest::SCHEMA_V3
        );
        assert_eq!(
            package.manifest().weight_shards[0].format,
            AppleWeightFormat::F16
        );
        let descriptor = package
            .low_bit_tensor(tensor_name)
            .expect("find down-projection descriptor");
        let expected = quantize_apple_low_bit_reference(
            AppleLowBitWeightFormat::W4A16,
            2,
            33,
            &source_values
                .iter()
                .map(|value| half::f16::from_f32(*value).to_f32())
                .collect::<Vec<_>>(),
        )
        .expect("quantize expected sidecar");
        assert_eq!(
            fs::read(config.output_dir.join(&descriptor.packed_values.path))
                .expect("read packed sidecar"),
            expected.packed_values()
        );
        let expected_scales = expected
            .scales()
            .iter()
            .flat_map(|scale| scale.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            fs::read(config.output_dir.join(&descriptor.scales.path)).expect("read scale sidecar"),
            expected_scales
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn hybrid_export_rejects_non_down_projection_and_bf16_sources() {
        let (root, mut config) = fixture("hybrid-rejections");
        config
            .low_bit_down_projections
            .push(AppleLowBitExportRequest {
                tensor_name: "weight".to_owned(),
                format: AppleLowBitWeightFormat::W8A16,
            });
        let error = build_apple_model_package(&config).expect_err("non-down projection fails");
        assert!(error.contains("not a supported dense MLP down projection"));

        config.low_bit_down_projections[0].tensor_name =
            "model.layers.0.mlp.down_proj.weight".to_owned();
        write_safetensors(
            &config.model_dir.join("model.safetensors"),
            "BF16",
            "model.layers.0.mlp.down_proj.weight",
        );
        let error = build_apple_model_package(&config).expect_err("BF16 hybrid export fails");
        assert!(error.contains("uniform F16 source checkpoint"));
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn refuses_to_create_output_inside_source_model() {
        let (root, mut config) = fixture("source-output");
        config.output_dir = config.model_dir.join("package");
        let error = build_apple_model_package(&config).expect_err("source mutation fails");
        assert!(error.contains("inside the source model directory"));
        assert!(!config.output_dir.exists());
        fs::remove_dir_all(root).expect("remove fixture");
    }
}
