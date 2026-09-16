//! Versioned, fail-closed Apple model-package validation.
//!
//! A package is an ordinary directory that an application may ship as a
//! resource bundle or install into an app-owned container.  The manifest uses
//! relative paths exclusively and authenticates every model configuration,
//! metadata, weight, tokenizer, and Metal artifact before the runtime is
//! allowed to prepare a backend.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use crate::{
    AppleLowBitWeightFormat, PackedAppleLowBitWeights, APPLE_LOW_BIT_GROUP_SIZE,
    APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
};

pub const APPLE_MODEL_PACKAGE_MANIFEST: &str = "rvllm-apple-model.json";
const MODEL_PACKAGE_IDENTITY_V2_DOMAIN: &[u8] = b"rvllm.apple-model-package.v2\0";
const MODEL_PACKAGE_IDENTITY_V3_DOMAIN: &[u8] = b"rvllm.apple-model-package.v3\0";

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplePackagePlatform {
    MacOs,
    Ios,
    IosSimulator,
}

impl ApplePackagePlatform {
    #[must_use]
    pub const fn current() -> Option<Self> {
        if cfg!(target_os = "macos") {
            Some(Self::MacOs)
        } else if cfg!(all(target_os = "ios", target_abi = "sim")) {
            Some(Self::IosSimulator)
        } else if cfg!(target_os = "ios") {
            Some(Self::Ios)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplePackageFloatType {
    F16,
    Bf16,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppleWeightFormat {
    F16,
    Bf16,
    W8A16Group32,
    W4A16Group32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ApplePackageFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleWeightShard {
    pub shard_index: u32,
    pub shard_count: u32,
    pub format: AppleWeightFormat,
    pub file: ApplePackageFile,
}

/// The only transformer role admitted by the initial hybrid low-bit package
/// contract. Other projections remain native until they have independent
/// execution and quality gates.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppleLowBitTensorRole {
    DenseDownProjection,
}

/// One authenticated, tensor-level low-bit sidecar.
///
/// Native F16 safetensors remain present and authoritative in schema v3. This
/// descriptor does not claim that the whole model or weight shard is low-bit.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleLowBitTensor {
    pub tensor_name: String,
    pub role: AppleLowBitTensorRole,
    pub format: AppleLowBitWeightFormat,
    pub abi_version: u16,
    pub group_size: u16,
    pub activation_float_type: ApplePackageFloatType,
    /// Row-major `[N, K]` projection shape.
    pub shape: [u32; 2],
    pub packed_values: ApplePackageFile,
    pub scales: ApplePackageFile,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleMetalLibrary {
    pub platform: ApplePackagePlatform,
    pub float_type: ApplePackageFloatType,
    pub library: ApplePackageFile,
    pub pipeline_manifest: ApplePackageFile,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleModelPackageManifest {
    pub schema_version: u32,
    pub package_id: String,
    pub architecture: String,
    pub model_fingerprint: String,
    pub tokenizer_fingerprint: String,
    pub chat_template_fingerprint: String,
    pub numerical_abi_fingerprint: String,
    /// Authenticated `config.json`. The path is fixed to `config.json` so the
    /// validated package is directly consumable by the HF-compatible loader.
    pub model_config: ApplePackageFile,
    /// Additional model semantics such as the safetensors index and generation
    /// or processor configuration. Every entry is authenticated.
    pub model_metadata_files: Vec<ApplePackageFile>,
    pub tokenizer_files: Vec<ApplePackageFile>,
    pub weight_shards: Vec<AppleWeightShard>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub low_bit_tensors: Vec<AppleLowBitTensor>,
    pub metal_libraries: Vec<AppleMetalLibrary>,
}

impl AppleModelPackageManifest {
    pub const SCHEMA_V2: u32 = 2;
    pub const SCHEMA_V3: u32 = 3;

    pub fn identity_fingerprint(&self) -> Result<[u8; 32], AppleModelPackageError> {
        self.validate_metadata()?;
        let encoded = serde_json::to_vec(self).map_err(AppleModelPackageError::ManifestEncode)?;
        let mut hasher = Sha256::new();
        hasher.update(match self.schema_version {
            Self::SCHEMA_V2 => MODEL_PACKAGE_IDENTITY_V2_DOMAIN,
            Self::SCHEMA_V3 => MODEL_PACKAGE_IDENTITY_V3_DOMAIN,
            _ => unreachable!("schema was validated"),
        });
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(encoded);
        Ok(hasher.finalize().into())
    }

    fn validate_metadata(&self) -> Result<(), AppleModelPackageError> {
        if !matches!(self.schema_version, Self::SCHEMA_V2 | Self::SCHEMA_V3) {
            return Err(AppleModelPackageError::UnsupportedSchema {
                found: self.schema_version,
            });
        }
        if self.schema_version == Self::SCHEMA_V2 && !self.low_bit_tensors.is_empty() {
            return Err(AppleModelPackageError::InvalidLowBitTensor {
                tensor: "<manifest>".to_owned(),
                reason: "schema v2 cannot declare low-bit tensor sidecars".to_owned(),
            });
        }
        if self.schema_version == Self::SCHEMA_V3 && self.low_bit_tensors.is_empty() {
            return Err(AppleModelPackageError::InvalidLowBitTensor {
                tensor: "<manifest>".to_owned(),
                reason: "schema v3 requires at least one low-bit tensor sidecar".to_owned(),
            });
        }
        for (field, value) in [
            ("package_id", self.package_id.as_str()),
            ("architecture", self.architecture.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(AppleModelPackageError::EmptyIdentity { field });
            }
        }
        for (field, value) in [
            ("model_fingerprint", self.model_fingerprint.as_str()),
            ("tokenizer_fingerprint", self.tokenizer_fingerprint.as_str()),
            (
                "chat_template_fingerprint",
                self.chat_template_fingerprint.as_str(),
            ),
            (
                "numerical_abi_fingerprint",
                self.numerical_abi_fingerprint.as_str(),
            ),
        ] {
            if !is_sha256_hex(value) {
                return Err(AppleModelPackageError::InvalidFingerprint { field });
            }
        }
        if self.tokenizer_files.is_empty() {
            return Err(AppleModelPackageError::MissingAsset("tokenizer"));
        }
        if self.weight_shards.is_empty() {
            return Err(AppleModelPackageError::MissingAsset("weights"));
        }
        if self.metal_libraries.is_empty() {
            return Err(AppleModelPackageError::MissingAsset("metallib"));
        }
        if self.model_config.path != Path::new("config.json") {
            return Err(AppleModelPackageError::InvalidModelConfig {
                reason: "model_config path must be config.json".to_owned(),
            });
        }

        let shard_count = self.weight_shards[0].shard_count;
        if shard_count == 0 || shard_count as usize != self.weight_shards.len() {
            return Err(AppleModelPackageError::InvalidShards);
        }
        let format = self.weight_shards[0].format;
        let mut indices = HashSet::with_capacity(self.weight_shards.len());
        for shard in &self.weight_shards {
            if shard.shard_count != shard_count
                || shard.format != format
                || shard.shard_index >= shard_count
                || !indices.insert(shard.shard_index)
            {
                return Err(AppleModelPackageError::InvalidShards);
            }
        }
        if matches!(
            format,
            AppleWeightFormat::W4A16Group32 | AppleWeightFormat::W8A16Group32
        ) {
            return Err(AppleModelPackageError::UnsupportedWeightFormat {
                format,
                reason: "the authenticated serialized low-bit tensor contract and runtime admission are not implemented",
            });
        }
        if self.schema_version == Self::SCHEMA_V3 && format != AppleWeightFormat::F16 {
            return Err(AppleModelPackageError::InvalidLowBitTensor {
                tensor: "<manifest>".to_owned(),
                reason: "schema-v3 hybrid low-bit sidecars require native F16 weight shards"
                    .to_owned(),
            });
        }

        let mut low_bit_names = HashSet::with_capacity(self.low_bit_tensors.len());
        for tensor in &self.low_bit_tensors {
            validate_low_bit_metadata(tensor)?;
            if !low_bit_names.insert(tensor.tensor_name.as_str()) {
                return Err(AppleModelPackageError::InvalidLowBitTensor {
                    tensor: tensor.tensor_name.clone(),
                    reason: "duplicate low-bit tensor name".to_owned(),
                });
            }
        }

        let mut libraries = HashSet::with_capacity(self.metal_libraries.len());
        for library in &self.metal_libraries {
            if !libraries.insert((library.platform, library.float_type)) {
                return Err(AppleModelPackageError::DuplicateMetalLibrary {
                    platform: library.platform,
                    float_type: library.float_type,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AppleModelPackage {
    root: PathBuf,
    manifest: AppleModelPackageManifest,
    identity_fingerprint: [u8; 32],
}

impl AppleModelPackage {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, AppleModelPackageError> {
        let root = root.as_ref().to_path_buf();
        let manifest_path = root.join(APPLE_MODEL_PACKAGE_MANIFEST);
        let bytes = fs::read(&manifest_path).map_err(|source| AppleModelPackageError::Io {
            operation: "read manifest",
            path: manifest_path,
            source,
        })?;
        let manifest: AppleModelPackageManifest =
            serde_json::from_slice(&bytes).map_err(AppleModelPackageError::ManifestDecode)?;
        manifest.validate_metadata()?;

        let mut seen_paths = HashSet::new();
        for file in std::iter::once(&manifest.model_config)
            .chain(manifest.model_metadata_files.iter())
            .chain(manifest.tokenizer_files.iter())
            .chain(manifest.weight_shards.iter().map(|shard| &shard.file))
            .chain(
                manifest
                    .low_bit_tensors
                    .iter()
                    .flat_map(|tensor| [&tensor.packed_values, &tensor.scales]),
            )
            .chain(
                manifest
                    .metal_libraries
                    .iter()
                    .flat_map(|library| [&library.library, &library.pipeline_manifest]),
            )
        {
            validate_file(&root, file, &mut seen_paths)?;
        }
        for tensor in &manifest.low_bit_tensors {
            validate_low_bit_contents(&root, tensor)?;
        }
        validate_model_config(&root, &manifest)?;
        validate_derived_fingerprints(&root, &manifest)?;
        let identity_fingerprint = manifest.identity_fingerprint()?;
        Ok(Self {
            root,
            manifest,
            identity_fingerprint,
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn manifest(&self) -> &AppleModelPackageManifest {
        &self.manifest
    }

    #[must_use]
    pub const fn identity_fingerprint(&self) -> [u8; 32] {
        self.identity_fingerprint
    }

    #[must_use]
    pub fn low_bit_tensor(&self, name: &str) -> Option<&AppleLowBitTensor> {
        self.manifest
            .low_bit_tensors
            .iter()
            .find(|tensor| tensor.tensor_name == name)
    }

    /// Load one low-bit sidecar into its canonical in-memory representation.
    ///
    /// [`Self::open`] validates the complete package, but package directories
    /// are ordinary mutable filesystem state. This method therefore verifies
    /// the length and digest of the exact bytes it consumes as well. The final
    /// constructor revalidates packed semantics before callers can use the
    /// projection.
    pub fn load_low_bit_tensor(
        &self,
        name: &str,
    ) -> Result<PackedAppleLowBitWeights, AppleModelPackageError> {
        let tensor = self.low_bit_tensor(name).ok_or_else(|| {
            AppleModelPackageError::InvalidLowBitTensor {
                tensor: name.to_owned(),
                reason: "tensor is not declared by the authenticated manifest".to_owned(),
            }
        })?;
        let values_len = usize::try_from(tensor.packed_values.bytes).map_err(|_| {
            AppleModelPackageError::InvalidLowBitTensor {
                tensor: name.to_owned(),
                reason: "packed-value byte count exceeds host address space".to_owned(),
            }
        })?;
        let scales_len = usize::try_from(tensor.scales.bytes).map_err(|_| {
            AppleModelPackageError::InvalidLowBitTensor {
                tensor: name.to_owned(),
                reason: "scale byte count exceeds host address space".to_owned(),
            }
        })?;
        let mut values = vec![0_u8; values_len];
        let mut scale_bytes = vec![0_u8; scales_len];
        self.load_low_bit_tensor_into(name, &mut values, &mut scale_bytes)?;
        let scales = scale_bytes
            .chunks_exact(std::mem::size_of::<half::f16>())
            .map(|bytes| half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect::<Vec<_>>();
        PackedAppleLowBitWeights::from_parts(
            tensor.abi_version,
            tensor.format,
            tensor.shape[0] as usize,
            tensor.shape[1] as usize,
            values,
            scales,
        )
        .map_err(|error| AppleModelPackageError::InvalidLowBitTensor {
            tensor: tensor.tensor_name.clone(),
            reason: error.to_string(),
        })
    }

    /// Re-authenticate one low-bit tensor while reading it into caller-owned
    /// fixed-size storage.
    ///
    /// This is the production upload path: it checks the current file length
    /// before reading, reads exactly the authenticated number of bytes plus an
    /// EOF probe, and hashes the bytes copied into the destination. Callers can
    /// therefore stream directly into a pre-budgeted accelerator arena without
    /// an unaccounted full-size staging allocation.
    pub fn load_low_bit_tensor_into(
        &self,
        name: &str,
        packed_values: &mut [u8],
        scales_le: &mut [u8],
    ) -> Result<(), AppleModelPackageError> {
        let tensor = self.low_bit_tensor(name).ok_or_else(|| {
            AppleModelPackageError::InvalidLowBitTensor {
                tensor: name.to_owned(),
                reason: "tensor is not declared by the authenticated manifest".to_owned(),
            }
        })?;
        read_authenticated_asset_into(
            &self.root,
            &tensor.packed_values,
            packed_values,
            "read low-bit values",
        )?;
        read_authenticated_asset_into(
            &self.root,
            &tensor.scales,
            scales_le,
            "read low-bit scales",
        )?;
        Ok(())
    }

    pub fn metal_library(
        &self,
        platform: ApplePackagePlatform,
        float_type: ApplePackageFloatType,
    ) -> Result<(PathBuf, PathBuf), AppleModelPackageError> {
        let library = self
            .manifest
            .metal_libraries
            .iter()
            .find(|library| library.platform == platform && library.float_type == float_type)
            .ok_or(AppleModelPackageError::MetalLibraryUnavailable {
                platform,
                float_type,
            })?;
        Ok((
            self.root.join(&library.library.path),
            self.root.join(&library.pipeline_manifest.path),
        ))
    }
}

fn validate_loaded_asset(
    descriptor: &ApplePackageFile,
    bytes: &[u8],
) -> Result<(), AppleModelPackageError> {
    let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual != descriptor.bytes {
        return Err(AppleModelPackageError::LengthMismatch {
            path: descriptor.path.clone(),
            expected: descriptor.bytes,
            actual,
        });
    }
    if hex_lower(&Sha256::digest(bytes)) != descriptor.sha256.to_ascii_lowercase() {
        return Err(AppleModelPackageError::ChecksumMismatch {
            path: descriptor.path.clone(),
        });
    }
    Ok(())
}

fn read_authenticated_asset_into(
    root: &Path,
    descriptor: &ApplePackageFile,
    output: &mut [u8],
    operation: &'static str,
) -> Result<(), AppleModelPackageError> {
    let output_len = u64::try_from(output.len()).unwrap_or(u64::MAX);
    if output_len != descriptor.bytes {
        return Err(AppleModelPackageError::LengthMismatch {
            path: descriptor.path.clone(),
            expected: descriptor.bytes,
            actual: output_len,
        });
    }
    reject_symlink_components(root, &descriptor.path)?;
    let path = root.join(&descriptor.path);
    let mut file = File::open(&path).map_err(|source| AppleModelPackageError::Io {
        operation,
        path: path.clone(),
        source,
    })?;
    let metadata = file
        .metadata()
        .map_err(|source| AppleModelPackageError::Io {
            operation: "stat low-bit asset",
            path: path.clone(),
            source,
        })?;
    if !metadata.is_file() {
        return Err(AppleModelPackageError::AssetNotRegular {
            path: descriptor.path.clone(),
        });
    }
    let current_len = metadata.len();
    if current_len != descriptor.bytes {
        return Err(AppleModelPackageError::LengthMismatch {
            path: descriptor.path.clone(),
            expected: descriptor.bytes,
            actual: current_len,
        });
    }
    file.read_exact(output)
        .map_err(|source| AppleModelPackageError::Io {
            operation,
            path: path.clone(),
            source,
        })?;
    let mut trailing = [0_u8; 1];
    let trailing_bytes = file
        .read(&mut trailing)
        .map_err(|source| AppleModelPackageError::Io {
            operation,
            path,
            source,
        })?;
    if trailing_bytes != 0 {
        return Err(AppleModelPackageError::LengthMismatch {
            path: descriptor.path.clone(),
            expected: descriptor.bytes,
            actual: descriptor.bytes.saturating_add(trailing_bytes as u64),
        });
    }
    validate_loaded_asset(descriptor, output)
}

fn validate_low_bit_metadata(tensor: &AppleLowBitTensor) -> Result<(), AppleModelPackageError> {
    let invalid = |reason: &str| AppleModelPackageError::InvalidLowBitTensor {
        tensor: tensor.tensor_name.clone(),
        reason: reason.to_owned(),
    };
    if tensor.tensor_name.trim().is_empty() {
        return Err(invalid("tensor name must not be empty"));
    }
    match tensor.role {
        AppleLowBitTensorRole::DenseDownProjection
            if !tensor.tensor_name.ends_with(".mlp.down_proj.weight")
                || tensor.tensor_name.contains(".experts.") =>
        {
            return Err(invalid(
                "dense down-projection tensor name must end in .mlp.down_proj.weight",
            ));
        }
        AppleLowBitTensorRole::DenseDownProjection => {}
    }
    if tensor.abi_version != APPLE_LOW_BIT_WEIGHT_ABI_VERSION {
        return Err(invalid("unsupported low-bit weight ABI version"));
    }
    if usize::from(tensor.group_size) != APPLE_LOW_BIT_GROUP_SIZE {
        return Err(invalid("low-bit group size must be 32"));
    }
    if tensor.activation_float_type != ApplePackageFloatType::F16 {
        return Err(invalid("W4A16/W8A16 sidecars require F16 activations"));
    }
    let rows = usize::try_from(tensor.shape[0]).map_err(|_| invalid("row count overflow"))?;
    let k = usize::try_from(tensor.shape[1]).map_err(|_| invalid("K dimension overflow"))?;
    if rows == 0 || k == 0 {
        return Err(invalid("low-bit tensor dimensions must be nonzero"));
    }
    let packed_row_bytes = match tensor.format {
        AppleLowBitWeightFormat::W4A16 => k
            .checked_add(1)
            .map(|value| value / 2)
            .ok_or_else(|| invalid("packed row byte count overflow"))?,
        AppleLowBitWeightFormat::W8A16 => k,
    };
    let groups = k
        .checked_add(APPLE_LOW_BIT_GROUP_SIZE - 1)
        .map(|value| value / APPLE_LOW_BIT_GROUP_SIZE)
        .ok_or_else(|| invalid("scale group count overflow"))?;
    let expected_values = rows
        .checked_mul(packed_row_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("packed value byte count overflow"))?;
    let expected_scales = rows
        .checked_mul(groups)
        .and_then(|count| count.checked_mul(std::mem::size_of::<half::f16>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("scale byte count overflow"))?;
    if tensor.packed_values.bytes != expected_values {
        return Err(invalid("packed value file length does not match shape"));
    }
    if tensor.scales.bytes != expected_scales {
        return Err(invalid("scale file length does not match shape"));
    }
    Ok(())
}

fn validate_low_bit_contents(
    root: &Path,
    tensor: &AppleLowBitTensor,
) -> Result<(), AppleModelPackageError> {
    let rows = tensor.shape[0] as usize;
    let k = tensor.shape[1] as usize;
    let packed_row_bytes = match tensor.format {
        AppleLowBitWeightFormat::W4A16 => k.div_ceil(2),
        AppleLowBitWeightFormat::W8A16 => k,
    };
    let groups = k.div_ceil(APPLE_LOW_BIT_GROUP_SIZE);
    let mut values = File::open(root.join(&tensor.packed_values.path)).map_err(|source| {
        AppleModelPackageError::Io {
            operation: "open low-bit values",
            path: root.join(&tensor.packed_values.path),
            source,
        }
    })?;
    let mut scales = File::open(root.join(&tensor.scales.path)).map_err(|source| {
        AppleModelPackageError::Io {
            operation: "open low-bit scales",
            path: root.join(&tensor.scales.path),
            source,
        }
    })?;
    let mut row_values = vec![0_u8; packed_row_bytes];
    let mut row_scale_bytes = vec![0_u8; groups * std::mem::size_of::<half::f16>()];
    for row in 0..rows {
        values
            .read_exact(&mut row_values)
            .map_err(|source| AppleModelPackageError::Io {
                operation: "read low-bit values",
                path: root.join(&tensor.packed_values.path),
                source,
            })?;
        scales
            .read_exact(&mut row_scale_bytes)
            .map_err(|source| AppleModelPackageError::Io {
                operation: "read low-bit scales",
                path: root.join(&tensor.scales.path),
                source,
            })?;
        let row_scales = row_scale_bytes
            .chunks_exact(2)
            .map(|bytes| half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect::<Vec<_>>();
        PackedAppleLowBitWeights::from_parts(
            tensor.abi_version,
            tensor.format,
            1,
            k,
            row_values.clone(),
            row_scales,
        )
        .map_err(|error| AppleModelPackageError::InvalidLowBitTensor {
            tensor: tensor.tensor_name.clone(),
            reason: format!("row {row}: {error}"),
        })?;
    }
    Ok(())
}

fn validate_file(
    root: &Path,
    file: &ApplePackageFile,
    seen_paths: &mut HashSet<PathBuf>,
) -> Result<(), AppleModelPackageError> {
    if file.bytes == 0 || !is_sha256_hex(&file.sha256) {
        return Err(AppleModelPackageError::InvalidAssetMetadata {
            path: file.path.clone(),
        });
    }
    if file.path.as_os_str().is_empty()
        || file.path.is_absolute()
        || file
            .path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || !seen_paths.insert(file.path.clone())
    {
        return Err(AppleModelPackageError::UnsafeOrDuplicatePath {
            path: file.path.clone(),
        });
    }
    let path = root.join(&file.path);
    reject_symlink_components(root, &file.path)?;
    let metadata = fs::metadata(&path).map_err(|source| AppleModelPackageError::Io {
        operation: "stat asset",
        path: path.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(AppleModelPackageError::AssetNotRegular {
            path: file.path.clone(),
        });
    }
    if metadata.len() != file.bytes {
        return Err(AppleModelPackageError::LengthMismatch {
            path: file.path.clone(),
            expected: file.bytes,
            actual: metadata.len(),
        });
    }
    let actual = hash_file(&path)?;
    if actual != file.sha256.to_ascii_lowercase() {
        return Err(AppleModelPackageError::ChecksumMismatch {
            path: file.path.clone(),
        });
    }
    Ok(())
}

fn reject_symlink_components(root: &Path, relative: &Path) -> Result<(), AppleModelPackageError> {
    let mut path = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(AppleModelPackageError::UnsafeOrDuplicatePath {
                path: relative.to_path_buf(),
            });
        };
        path.push(component);
        let metadata =
            fs::symlink_metadata(&path).map_err(|source| AppleModelPackageError::Io {
                operation: "inspect asset path",
                path: path.clone(),
                source,
            })?;
        if metadata.file_type().is_symlink() {
            return Err(AppleModelPackageError::SymlinkAsset {
                path: relative.to_path_buf(),
            });
        }
    }
    Ok(())
}

pub(crate) fn hash_file(path: &Path) -> Result<String, AppleModelPackageError> {
    let mut file = File::open(path).map_err(|source| AppleModelPackageError::Io {
        operation: "open asset",
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| AppleModelPackageError::Io {
                operation: "hash asset",
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

pub(crate) fn fingerprint_assets<'a>(
    domain: &[u8],
    assets: impl IntoIterator<Item = &'a ApplePackageFile>,
) -> String {
    let mut sorted = assets.into_iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.path.cmp(&right.path));
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for asset in sorted {
        let path = asset.path.to_string_lossy();
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update(asset.bytes.to_le_bytes());
        hasher.update(asset.sha256.as_bytes());
    }
    hex_lower(&hasher.finalize())
}

pub(crate) fn model_assets_fingerprint(manifest: &AppleModelPackageManifest) -> String {
    let asset_fingerprint = fingerprint_assets(
        match manifest.schema_version {
            AppleModelPackageManifest::SCHEMA_V3 => b"rvllm.apple.model-assets.v3\0".as_slice(),
            _ => b"rvllm.apple.model-assets.v2\0".as_slice(),
        },
        std::iter::once(&manifest.model_config)
            .chain(manifest.model_metadata_files.iter())
            .chain(manifest.weight_shards.iter().map(|shard| &shard.file))
            .chain(
                manifest
                    .low_bit_tensors
                    .iter()
                    .flat_map(|tensor| [&tensor.packed_values, &tensor.scales]),
            ),
    );
    if manifest.schema_version != AppleModelPackageManifest::SCHEMA_V3 {
        return asset_fingerprint;
    }

    // A schema-v3 model identity must change if the same authenticated bytes
    // are assigned a different tensor name, role, shape, format, or ABI.
    let mut hasher = Sha256::new();
    hasher.update(b"rvllm.apple.model-semantics.v3\0");
    hasher.update(asset_fingerprint.as_bytes());
    let mut descriptors = manifest.low_bit_tensors.iter().collect::<Vec<_>>();
    descriptors.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    for tensor in descriptors {
        hasher.update((tensor.tensor_name.len() as u64).to_le_bytes());
        hasher.update(tensor.tensor_name.as_bytes());
        hasher.update([match tensor.role {
            AppleLowBitTensorRole::DenseDownProjection => 1,
        }]);
        hasher.update([tensor.format.bits() as u8]);
        hasher.update(tensor.abi_version.to_le_bytes());
        hasher.update(tensor.group_size.to_le_bytes());
        hasher.update([match tensor.activation_float_type {
            ApplePackageFloatType::F16 => 1,
            ApplePackageFloatType::Bf16 => 2,
        }]);
        hasher.update(tensor.shape[0].to_le_bytes());
        hasher.update(tensor.shape[1].to_le_bytes());
    }
    hex_lower(&hasher.finalize())
}

pub(crate) fn chat_template_fingerprint(root: &Path) -> Result<String, AppleModelPackageError> {
    let path = root.join("tokenizer_config.json");
    let mut hasher = Sha256::new();
    hasher.update(b"rvllm.apple.chat-template.v2\0");
    if path.is_file() {
        let bytes = fs::read(&path).map_err(|source| AppleModelPackageError::Io {
            operation: "read tokenizer config",
            path: path.clone(),
            source,
        })?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            AppleModelPackageError::InvalidTokenizerConfig {
                reason: error.to_string(),
            }
        })?;
        if let Some(template) = value.get("chat_template") {
            let encoded = serde_json::to_vec(template).map_err(|error| {
                AppleModelPackageError::InvalidTokenizerConfig {
                    reason: error.to_string(),
                }
            })?;
            hasher.update(b"present\0");
            hasher.update((encoded.len() as u64).to_le_bytes());
            hasher.update(encoded);
        } else {
            hasher.update(b"absent\0");
        }
    } else {
        hasher.update(b"absent\0");
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn validate_derived_fingerprints(
    root: &Path,
    manifest: &AppleModelPackageManifest,
) -> Result<(), AppleModelPackageError> {
    let expected = [
        (
            "model_fingerprint",
            model_assets_fingerprint(manifest),
            manifest.model_fingerprint.as_str(),
        ),
        (
            "tokenizer_fingerprint",
            fingerprint_assets(
                b"rvllm.apple.tokenizer-assets.v2\0",
                manifest.tokenizer_files.iter(),
            ),
            manifest.tokenizer_fingerprint.as_str(),
        ),
        (
            "numerical_abi_fingerprint",
            fingerprint_assets(
                b"rvllm.apple.metal-pipeline-abi.v2\0",
                manifest
                    .metal_libraries
                    .iter()
                    .map(|library| &library.pipeline_manifest),
            ),
            manifest.numerical_abi_fingerprint.as_str(),
        ),
        (
            "chat_template_fingerprint",
            chat_template_fingerprint(root)?,
            manifest.chat_template_fingerprint.as_str(),
        ),
    ];
    for (field, expected, found) in expected {
        if expected != found.to_ascii_lowercase() {
            return Err(AppleModelPackageError::DerivedFingerprintMismatch { field });
        }
    }
    Ok(())
}

fn validate_model_config(
    root: &Path,
    manifest: &AppleModelPackageManifest,
) -> Result<(), AppleModelPackageError> {
    let bytes = fs::read(root.join(&manifest.model_config.path)).map_err(|source| {
        AppleModelPackageError::Io {
            operation: "read model config",
            path: root.join(&manifest.model_config.path),
            source,
        }
    })?;
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        AppleModelPackageError::InvalidModelConfig {
            reason: format!("config.json is not valid JSON: {error}"),
        }
    })?;
    let object = config
        .as_object()
        .ok_or_else(|| AppleModelPackageError::InvalidModelConfig {
            reason: "config.json root must be an object".to_owned(),
        })?;
    let architectures = object
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| AppleModelPackageError::InvalidModelConfig {
            reason: "config.json requires an architectures array".to_owned(),
        })?;
    let declared = architectures
        .first()
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| AppleModelPackageError::InvalidModelConfig {
            reason: "config.json architectures[0] must be a non-empty string".to_owned(),
        })?;
    if declared != manifest.architecture {
        return Err(AppleModelPackageError::InvalidModelConfig {
            reason: format!(
                "manifest architecture {:?} does not match config.json {:?}",
                manifest.architecture, declared
            ),
        });
    }
    let model_type = object
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            object
                .get("text_config")
                .and_then(serde_json::Value::as_object)
                .and_then(|text| text.get("model_type"))
                .and_then(serde_json::Value::as_str)
        });
    if !model_type.is_some_and(|value| !value.trim().is_empty()) {
        return Err(AppleModelPackageError::InvalidModelConfig {
            reason: "config.json requires a non-empty model_type".to_owned(),
        });
    }
    Ok(())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Debug)]
pub enum AppleModelPackageError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    ManifestDecode(serde_json::Error),
    ManifestEncode(serde_json::Error),
    UnsupportedSchema {
        found: u32,
    },
    EmptyIdentity {
        field: &'static str,
    },
    InvalidFingerprint {
        field: &'static str,
    },
    MissingAsset(&'static str),
    InvalidModelConfig {
        reason: String,
    },
    InvalidTokenizerConfig {
        reason: String,
    },
    DerivedFingerprintMismatch {
        field: &'static str,
    },
    InvalidShards,
    InvalidLowBitTensor {
        tensor: String,
        reason: String,
    },
    UnsupportedWeightFormat {
        format: AppleWeightFormat,
        reason: &'static str,
    },
    DuplicateMetalLibrary {
        platform: ApplePackagePlatform,
        float_type: ApplePackageFloatType,
    },
    InvalidAssetMetadata {
        path: PathBuf,
    },
    UnsafeOrDuplicatePath {
        path: PathBuf,
    },
    SymlinkAsset {
        path: PathBuf,
    },
    AssetNotRegular {
        path: PathBuf,
    },
    LengthMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    ChecksumMismatch {
        path: PathBuf,
    },
    MetalLibraryUnavailable {
        platform: ApplePackagePlatform,
        float_type: ApplePackageFloatType,
    },
}

impl fmt::Display for AppleModelPackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "Apple model package {operation} failed for {}: {source}",
                path.display()
            ),
            Self::ManifestDecode(source) => {
                write!(formatter, "invalid Apple model manifest: {source}")
            }
            Self::ManifestEncode(source) => {
                write!(formatter, "encode Apple model manifest: {source}")
            }
            Self::UnsupportedSchema { found } => {
                write!(formatter, "unsupported Apple model package schema {found}")
            }
            Self::EmptyIdentity { field } => write!(
                formatter,
                "empty Apple model package identity field {field}"
            ),
            Self::InvalidFingerprint { field } => write!(
                formatter,
                "Apple model package identity field {field} is not a SHA-256 digest"
            ),
            Self::MissingAsset(kind) => {
                write!(formatter, "Apple model package has no {kind} assets")
            }
            Self::InvalidModelConfig { reason } => {
                write!(formatter, "invalid Apple model config: {reason}")
            }
            Self::InvalidTokenizerConfig { reason } => {
                write!(formatter, "invalid Apple tokenizer config: {reason}")
            }
            Self::DerivedFingerprintMismatch { field } => write!(
                formatter,
                "Apple model package derived fingerprint does not match {field}"
            ),
            Self::InvalidShards => write!(
                formatter,
                "Apple model package weight shards are inconsistent"
            ),
            Self::InvalidLowBitTensor { tensor, reason } => write!(
                formatter,
                "invalid Apple low-bit tensor {tensor:?}: {reason}"
            ),
            Self::UnsupportedWeightFormat { format, reason } => write!(
                formatter,
                "Apple model package weight format {format:?} is unsupported: {reason}"
            ),
            Self::DuplicateMetalLibrary {
                platform,
                float_type,
            } => write!(
                formatter,
                "duplicate Apple Metal library for {platform:?}/{float_type:?}"
            ),
            Self::InvalidAssetMetadata { path } => write!(
                formatter,
                "invalid Apple model asset metadata for {}",
                path.display()
            ),
            Self::UnsafeOrDuplicatePath { path } => write!(
                formatter,
                "unsafe or duplicate Apple model asset path {}",
                path.display()
            ),
            Self::SymlinkAsset { path } => write!(
                formatter,
                "Apple model package asset may not be a symlink: {}",
                path.display()
            ),
            Self::AssetNotRegular { path } => write!(
                formatter,
                "Apple model package asset is not a regular file: {}",
                path.display()
            ),
            Self::LengthMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "Apple model asset {} has {actual} bytes, expected {expected}",
                path.display()
            ),
            Self::ChecksumMismatch { path } => write!(
                formatter,
                "Apple model asset checksum mismatch for {}",
                path.display()
            ),
            Self::MetalLibraryUnavailable {
                platform,
                float_type,
            } => write!(
                formatter,
                "Apple model package has no Metal library for {platform:?}/{float_type:?}"
            ),
        }
    }
}

impl std::error::Error for AppleModelPackageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::ManifestDecode(source) | Self::ManifestEncode(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!("rvllm-model-package-{name}-{nonce}"))
    }

    fn file(root: &Path, relative: &str, bytes: &[u8]) -> ApplePackageFile {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create asset directory");
        }
        fs::write(&path, bytes).expect("write asset");
        ApplePackageFile {
            path: PathBuf::from(relative),
            bytes: bytes.len() as u64,
            sha256: hex_lower(&Sha256::digest(bytes)),
        }
    }

    fn manifest(root: &Path) -> AppleModelPackageManifest {
        let mut manifest = AppleModelPackageManifest {
            schema_version: AppleModelPackageManifest::SCHEMA_V2,
            package_id: "qwen2.5-0.5b-mobile".into(),
            architecture: "Qwen2ForCausalLM".into(),
            model_fingerprint: String::new(),
            tokenizer_fingerprint: String::new(),
            chat_template_fingerprint: String::new(),
            numerical_abi_fingerprint: String::new(),
            model_config: file(
                root,
                "config.json",
                br#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2"}"#,
            ),
            model_metadata_files: Vec::new(),
            tokenizer_files: vec![file(root, "tokenizer/tokenizer.json", b"tokenizer")],
            weight_shards: vec![AppleWeightShard {
                shard_index: 0,
                shard_count: 1,
                format: AppleWeightFormat::F16,
                file: file(root, "weights/model.bin", b"weights"),
            }],
            low_bit_tensors: Vec::new(),
            metal_libraries: vec![AppleMetalLibrary {
                platform: ApplePackagePlatform::MacOs,
                float_type: ApplePackageFloatType::F16,
                library: file(root, "metal/macos/f16/rvllm.metallib", b"metallib"),
                pipeline_manifest: file(root, "metal/macos/f16/pipelines.json", b"pipelines"),
            }],
        };
        manifest.model_fingerprint = model_assets_fingerprint(&manifest);
        manifest.tokenizer_fingerprint = fingerprint_assets(
            b"rvllm.apple.tokenizer-assets.v2\0",
            manifest.tokenizer_files.iter(),
        );
        manifest.numerical_abi_fingerprint = fingerprint_assets(
            b"rvllm.apple.metal-pipeline-abi.v2\0",
            manifest
                .metal_libraries
                .iter()
                .map(|library| &library.pipeline_manifest),
        );
        manifest.chat_template_fingerprint =
            chat_template_fingerprint(root).expect("fingerprint absent chat template");
        manifest
    }

    fn add_low_bit_tensor(
        root: &Path,
        manifest: &mut AppleModelPackageManifest,
        format: AppleLowBitWeightFormat,
    ) {
        let source = (0..66)
            .map(|index| index as f32 / 11.0 - 3.0)
            .collect::<Vec<_>>();
        let packed =
            crate::quantize_apple_low_bit_reference(format, 2, 33, &source).expect("quantize");
        let scale_bytes = packed
            .scales()
            .iter()
            .flat_map(|scale| scale.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        manifest.schema_version = AppleModelPackageManifest::SCHEMA_V3;
        manifest.low_bit_tensors.push(AppleLowBitTensor {
            tensor_name: "model.layers.0.mlp.down_proj.weight".to_owned(),
            role: AppleLowBitTensorRole::DenseDownProjection,
            format,
            abi_version: APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
            group_size: APPLE_LOW_BIT_GROUP_SIZE as u16,
            activation_float_type: ApplePackageFloatType::F16,
            shape: [2, 33],
            packed_values: file(
                root,
                "weights/low-bit/00000.values.bin",
                packed.packed_values(),
            ),
            scales: file(root, "weights/low-bit/00000.scales.f16le", &scale_bytes),
        });
        manifest.model_fingerprint = model_assets_fingerprint(manifest);
    }

    #[test]
    fn validates_all_assets_and_selects_exact_metallib() {
        let root = test_root("valid");
        fs::create_dir_all(&root).expect("create package root");
        let manifest = manifest(&root);
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        let package = AppleModelPackage::open(&root).expect("validate package");
        assert_ne!(package.identity_fingerprint(), [0; 32]);
        let (library, pipelines) = package
            .metal_library(ApplePackagePlatform::MacOs, ApplePackageFloatType::F16)
            .expect("select library");
        assert!(library.ends_with("rvllm.metallib"));
        assert!(pipelines.ends_with("pipelines.json"));
        fs::remove_dir_all(root).expect("remove package");
    }

    #[test]
    fn rejects_parent_traversal_before_reading_outside_package() {
        let root = test_root("traversal");
        fs::create_dir_all(&root).expect("create package root");
        let mut manifest = manifest(&root);
        manifest.tokenizer_files[0].path = PathBuf::from("../tokenizer.json");
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        assert!(matches!(
            AppleModelPackage::open(&root),
            Err(AppleModelPackageError::UnsafeOrDuplicatePath { .. })
        ));
        fs::remove_dir_all(root).expect("remove package");
    }

    #[test]
    fn rejects_corrupted_asset() {
        let root = test_root("corrupt");
        fs::create_dir_all(&root).expect("create package root");
        let manifest = manifest(&root);
        fs::write(root.join("weights/model.bin"), b"corrupt").expect("corrupt weights");
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        assert!(matches!(
            AppleModelPackage::open(&root),
            Err(AppleModelPackageError::ChecksumMismatch { .. })
        ));
        fs::remove_dir_all(root).expect("remove package");
    }

    #[test]
    fn rejects_config_semantics_that_disagree_with_manifest() {
        let root = test_root("config-mismatch");
        fs::create_dir_all(&root).expect("create package root");
        let mut manifest = manifest(&root);
        manifest.architecture = "OtherForCausalLM".into();
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        assert!(matches!(
            AppleModelPackage::open(&root),
            Err(AppleModelPackageError::InvalidModelConfig { .. })
        ));
        fs::remove_dir_all(root).expect("remove package");
    }

    #[test]
    fn rejects_fingerprint_not_derived_from_authenticated_assets() {
        let root = test_root("fingerprint");
        fs::create_dir_all(&root).expect("create package root");
        let mut manifest = manifest(&root);
        manifest.model_fingerprint = "ab".repeat(32);
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        assert!(matches!(
            AppleModelPackage::open(&root),
            Err(AppleModelPackageError::DerivedFingerprintMismatch {
                field: "model_fingerprint"
            })
        ));
        fs::remove_dir_all(root).expect("remove package");
    }

    #[test]
    fn rejects_externally_relabelled_low_bit_weight_manifests() {
        for format in [
            AppleWeightFormat::W4A16Group32,
            AppleWeightFormat::W8A16Group32,
        ] {
            let root = test_root(match format {
                AppleWeightFormat::W4A16Group32 => "relabel-w4",
                AppleWeightFormat::W8A16Group32 => "relabel-w8",
                _ => unreachable!(),
            });
            fs::create_dir_all(&root).expect("create package root");
            let mut manifest = manifest(&root);
            manifest.weight_shards[0].format = format;
            fs::write(
                root.join(APPLE_MODEL_PACKAGE_MANIFEST),
                serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
            )
            .expect("write manifest");

            let error = AppleModelPackage::open(&root)
                .expect_err("low-bit-labelled package must fail closed");
            assert!(matches!(
                error,
                AppleModelPackageError::UnsupportedWeightFormat {
                    format: rejected,
                    reason: "the authenticated serialized low-bit tensor contract and runtime admission are not implemented",
                } if rejected == format
            ));
            fs::remove_dir_all(root).expect("remove package");
        }
    }

    #[test]
    fn schema_v3_authenticates_and_validates_exact_low_bit_sidecars() {
        for format in [
            AppleLowBitWeightFormat::W4A16,
            AppleLowBitWeightFormat::W8A16,
        ] {
            let root = test_root(format.name());
            fs::create_dir_all(&root).expect("create package root");
            let mut manifest = manifest(&root);
            add_low_bit_tensor(&root, &mut manifest, format);
            fs::write(
                root.join(APPLE_MODEL_PACKAGE_MANIFEST),
                serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
            )
            .expect("write manifest");

            let package = AppleModelPackage::open(&root).expect("open hybrid package");
            let tensor = package
                .low_bit_tensor("model.layers.0.mlp.down_proj.weight")
                .expect("find exact sidecar");
            assert_eq!(tensor.format, format);
            assert_eq!(tensor.shape, [2, 33]);
            assert_eq!(
                package.manifest().weight_shards[0].format,
                AppleWeightFormat::F16
            );
            let loaded = package
                .load_low_bit_tensor("model.layers.0.mlp.down_proj.weight")
                .expect("load validated sidecar");
            assert_eq!(loaded.format(), format);
            assert_eq!(loaded.shape(), [2, 33]);
            let mut packed_values =
                vec![0_u8; tensor.packed_values.bytes.try_into().expect("packed bytes")];
            let mut scales = vec![0_u8; tensor.scales.bytes.try_into().expect("scale bytes")];
            package
                .load_low_bit_tensor_into(
                    "model.layers.0.mlp.down_proj.weight",
                    &mut packed_values,
                    &mut scales,
                )
                .expect("bounded authenticated load");
            assert_eq!(packed_values, loaded.packed_values());
            assert_eq!(
                scales,
                loaded
                    .scales()
                    .iter()
                    .flat_map(|scale| scale.to_bits().to_le_bytes())
                    .collect::<Vec<_>>()
            );
            let shortened_packed_len = packed_values.len() - 1;
            assert!(matches!(
                package.load_low_bit_tensor_into(
                    "model.layers.0.mlp.down_proj.weight",
                    &mut packed_values[..shortened_packed_len],
                    &mut scales,
                ),
                Err(AppleModelPackageError::LengthMismatch { .. })
            ));
            assert!(package
                .load_low_bit_tensor("missing.down_proj.weight")
                .is_err());
            fs::remove_dir_all(root).expect("remove package");
        }
    }

    #[test]
    fn low_bit_reload_rejects_package_mutation_after_open() {
        let root = test_root("reload-mutation");
        fs::create_dir_all(&root).expect("create package root");
        let mut manifest = manifest(&root);
        add_low_bit_tensor(&root, &mut manifest, AppleLowBitWeightFormat::W8A16);
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        let package = AppleModelPackage::open(&root).expect("open hybrid package");
        let descriptor = package.manifest().low_bit_tensors[0].packed_values.clone();
        let path = root.join(&descriptor.path);
        let original = fs::read(&path).expect("read packed values");

        let mut changed = original.clone();
        changed[0] ^= 1;
        fs::write(&path, &changed).expect("mutate packed values");
        assert!(matches!(
            package.load_low_bit_tensor("model.layers.0.mlp.down_proj.weight"),
            Err(AppleModelPackageError::ChecksumMismatch { .. })
        ));

        let mut longer = original;
        longer.push(0);
        fs::write(&path, longer).expect("change packed value length");
        assert!(matches!(
            package.load_low_bit_tensor("model.layers.0.mlp.down_proj.weight"),
            Err(AppleModelPackageError::LengthMismatch { .. })
        ));
        fs::remove_dir_all(root).expect("remove package");
    }

    #[test]
    fn schema_v3_model_fingerprint_binds_low_bit_tensor_semantics() {
        let root = test_root("semantic-fingerprint");
        fs::create_dir_all(&root).expect("create package root");
        let mut manifest = manifest(&root);
        add_low_bit_tensor(&root, &mut manifest, AppleLowBitWeightFormat::W8A16);
        let original = model_assets_fingerprint(&manifest);
        manifest.low_bit_tensors[0].tensor_name = "model.layers.1.mlp.down_proj.weight".to_owned();
        let reassigned = model_assets_fingerprint(&manifest);
        assert_ne!(original, reassigned);
        fs::remove_dir_all(root).expect("remove package");
    }

    #[test]
    fn schema_v2_cannot_smuggle_a_low_bit_sidecar() {
        let root = test_root("v2-sidecar");
        fs::create_dir_all(&root).expect("create package root");
        let mut manifest = manifest(&root);
        add_low_bit_tensor(&root, &mut manifest, AppleLowBitWeightFormat::W4A16);
        manifest.schema_version = AppleModelPackageManifest::SCHEMA_V2;
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        let error = AppleModelPackage::open(&root).expect_err("v2 sidecar must fail");
        assert!(matches!(
            error,
            AppleModelPackageError::InvalidLowBitTensor { ref reason, .. }
                if reason.contains("schema v2")
        ));
        fs::remove_dir_all(root).expect("remove package");
    }

    #[test]
    fn rejects_semantically_invalid_low_bit_values_even_with_fresh_checksum() {
        let root = test_root("reserved-nibble");
        fs::create_dir_all(&root).expect("create package root");
        let mut manifest = manifest(&root);
        add_low_bit_tensor(&root, &mut manifest, AppleLowBitWeightFormat::W4A16);
        let tensor = &mut manifest.low_bit_tensors[0];
        let mut values =
            fs::read(root.join(&tensor.packed_values.path)).expect("read packed values");
        values[0] = (values[0] & 0xf0) | 0x08;
        tensor.packed_values = file(&root, "weights/low-bit/00000.values.bin", &values);
        manifest.model_fingerprint = model_assets_fingerprint(&manifest);
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        let error = AppleModelPackage::open(&root).expect_err("reserved W4 value must fail");
        assert!(matches!(
            error,
            AppleModelPackageError::InvalidLowBitTensor { ref reason, .. }
                if reason.contains("reserved asymmetric minimum")
        ));
        fs::remove_dir_all(root).expect("remove package");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_assets_even_when_target_is_inside_package() {
        use std::os::unix::fs::symlink;

        let root = test_root("symlink");
        fs::create_dir_all(&root).expect("create package root");
        let mut manifest = manifest(&root);
        let target = root.join("tokenizer/real.json");
        fs::rename(root.join("tokenizer/tokenizer.json"), &target).expect("move tokenizer");
        symlink("real.json", root.join("tokenizer/tokenizer.json")).expect("link tokenizer");
        manifest.tokenizer_files[0].bytes = fs::metadata(&target).expect("stat target").len();
        fs::write(
            root.join(APPLE_MODEL_PACKAGE_MANIFEST),
            serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
        )
        .expect("write manifest");

        assert!(matches!(
            AppleModelPackage::open(&root),
            Err(AppleModelPackageError::SymlinkAsset { .. })
        ));
        fs::remove_dir_all(root).expect("remove package");
    }
}
