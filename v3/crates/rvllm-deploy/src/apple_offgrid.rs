use std::path::{Component, Path, PathBuf};

use rvllm_apple::{
    AppleAcceleratorTarget, AppleBackendMode, AppleGpuFamily, AppleNpuGeneration, RolloutBucket,
};
use rvllm_core::{ConfigError, Result, RvllmError};
use serde::{Deserialize, Serialize};

pub const APPLE_OFFGRID_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleArtifact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AppleOffgridModelFormat {
    Safetensors,
    Gguf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppleOffgridModelRef {
    pub format: AppleOffgridModelFormat,
    pub config: BundleArtifact,
    pub tokenizer: BundleArtifact,
    pub weights: Vec<BundleArtifact>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppleOffgridMilProgram {
    pub name: String,
    pub bucket: RolloutBucket,
    pub mil: BundleArtifact,
    pub compiled: BundleArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppleOffgridMilCache {
    pub cache_key: String,
    pub root: PathBuf,
    pub programs: Vec<AppleOffgridMilProgram>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AppleOffgridPolicyMode {
    Sustained,
    Balanced,
    PerformanceCap,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppleOffgridRuntimePolicy {
    pub mode: AppleOffgridPolicyMode,
    pub max_package_watts: u32,
    pub max_gpu_watts: u32,
    pub max_ane_watts: u32,
    pub low_power_mode: bool,
    pub thermal_pressure_limit: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppleOffgridBundleManifest {
    pub schema_version: u32,
    pub bundle_id: String,
    pub revision: String,
    pub model: AppleOffgridModelRef,
    pub metallib: BundleArtifact,
    pub mil_cache: AppleOffgridMilCache,
    pub hardware_profile: AppleAcceleratorTarget,
    pub backend_mode: AppleBackendMode,
    pub energy_policy: AppleOffgridRuntimePolicy,
}

impl BundleArtifact {
    pub fn validate_schema(&self, field: &'static str) -> Result<()> {
        validate_relative_path(field, &self.path)?;
        if !is_sha256_hex(&self.sha256) {
            return Err(invalid(field, "sha256 must be 64 lowercase hex characters"));
        }
        if self.bytes == 0 {
            return Err(invalid(field, "bytes must be nonzero"));
        }
        Ok(())
    }
}

impl AppleOffgridBundleManifest {
    pub fn validate_schema(&self) -> Result<()> {
        if self.schema_version != APPLE_OFFGRID_SCHEMA_VERSION {
            return Err(invalid(
                "schema_version",
                "unsupported Apple off-grid bundle schema",
            ));
        }
        if self.bundle_id.trim().is_empty() {
            return Err(invalid("bundle_id", "bundle id is required"));
        }
        if !is_revision(&self.revision) {
            return Err(invalid(
                "revision",
                "revision must be a 40-character lowercase hex SHA",
            ));
        }

        self.model.validate_schema()?;
        self.metallib.validate_schema("metallib.path")?;
        if self.metallib.path.extension().and_then(|e| e.to_str()) != Some("metallib") {
            return Err(invalid(
                "metallib.path",
                "metallib artifact must end in .metallib",
            ));
        }
        self.mil_cache.validate_schema()?;
        validate_hardware_profile(&self.hardware_profile)?;
        self.energy_policy.validate_schema()?;
        Ok(())
    }
}

impl AppleOffgridModelRef {
    fn validate_schema(&self) -> Result<()> {
        self.config.validate_schema("model.config")?;
        self.tokenizer.validate_schema("model.tokenizer")?;
        if self.weights.is_empty() {
            return Err(invalid(
                "model.weights",
                "at least one model weight artifact is required",
            ));
        }
        for weight in &self.weights {
            weight.validate_schema("model.weights")?;
        }
        Ok(())
    }
}

impl AppleOffgridMilCache {
    fn validate_schema(&self) -> Result<()> {
        if self.cache_key.trim().is_empty() {
            return Err(invalid("mil_cache.cache_key", "cache key is required"));
        }
        validate_relative_path("mil_cache.root", &self.root)?;
        if self.programs.is_empty() {
            return Err(invalid(
                "mil_cache.programs",
                "at least one cached MIL program is required",
            ));
        }
        for program in &self.programs {
            program.validate_schema(&self.root)?;
        }
        Ok(())
    }
}

impl AppleOffgridMilProgram {
    fn validate_schema(&self, root: &Path) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(invalid(
                "mil_cache.programs.name",
                "program name is required",
            ));
        }
        if self.bucket.seqs == 0 || self.bucket.tokens == 0 {
            return Err(invalid(
                "mil_cache.programs.bucket",
                "bucket dimensions must be nonzero",
            ));
        }
        self.mil.validate_schema("mil_cache.programs.mil")?;
        self.compiled
            .validate_schema("mil_cache.programs.compiled")?;
        if !self.mil.path.starts_with(root) {
            return Err(invalid(
                "mil_cache.programs.mil",
                "MIL source must live under mil_cache.root",
            ));
        }
        if !self.compiled.path.starts_with(root) {
            return Err(invalid(
                "mil_cache.programs.compiled",
                "compiled MIL artifact must live under mil_cache.root",
            ));
        }
        Ok(())
    }
}

impl AppleOffgridRuntimePolicy {
    fn validate_schema(&self) -> Result<()> {
        if self.max_package_watts == 0 {
            return Err(invalid(
                "energy_policy.max_package_watts",
                "package watt budget is required",
            ));
        }
        if self.max_gpu_watts == 0 {
            return Err(invalid(
                "energy_policy.max_gpu_watts",
                "GPU watt budget is required",
            ));
        }
        if self.max_ane_watts == 0 {
            return Err(invalid(
                "energy_policy.max_ane_watts",
                "ANE watt budget is required",
            ));
        }
        if self.max_gpu_watts > self.max_package_watts {
            return Err(invalid(
                "energy_policy.max_gpu_watts",
                "GPU watt budget must not exceed package budget",
            ));
        }
        if self.max_ane_watts > self.max_package_watts {
            return Err(invalid(
                "energy_policy.max_ane_watts",
                "ANE watt budget must not exceed package budget",
            ));
        }
        if self.thermal_pressure_limit.trim().is_empty() {
            return Err(invalid(
                "energy_policy.thermal_pressure_limit",
                "thermal pressure limit is required",
            ));
        }
        Ok(())
    }
}

fn validate_hardware_profile(profile: &AppleAcceleratorTarget) -> Result<()> {
    if profile.device_name.trim().is_empty() {
        return Err(invalid(
            "hardware_profile.device_name",
            "device name is required",
        ));
    }
    if profile.gpu_family == AppleGpuFamily::Unknown {
        return Err(invalid(
            "hardware_profile.gpu_family",
            "known Apple GPU family is required",
        ));
    }
    if profile.npu_generation == AppleNpuGeneration::Unknown {
        return Err(invalid(
            "hardware_profile.npu_generation",
            "known Apple NPU generation is required",
        ));
    }
    if profile.architecture_gen != profile.gpu_family.architecture_gen() {
        return Err(invalid(
            "hardware_profile.architecture_gen",
            "architecture generation must match GPU family",
        ));
    }
    if profile.has_nax != profile.gpu_family.has_nax() {
        return Err(invalid(
            "hardware_profile.has_nax",
            "NAX flag must match GPU family",
        ));
    }
    if profile.ane_cores == 0 {
        return Err(invalid(
            "hardware_profile.ane_cores",
            "ANE core count must be nonzero",
        ));
    }
    if profile.die_count == 0 {
        return Err(invalid(
            "hardware_profile.die_count",
            "die count must be nonzero",
        ));
    }
    Ok(())
}

fn validate_relative_path(field: &'static str, path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(invalid(field, "path is required"));
    }
    if path.is_absolute() {
        return Err(invalid(field, "path must be relative to the bundle root"));
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::CurDir | Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(invalid(field, "path must not escape the bundle root"));
    }
    Ok(())
}

fn is_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.to_owned(),
        },
        field,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_uppercase_sha256() {
        let artifact = BundleArtifact {
            path: PathBuf::from("model/config.json"),
            sha256: "A".repeat(64),
            bytes: 1,
        };

        let err = artifact.validate_schema("model.config").unwrap_err();
        assert!(err.to_string().contains("sha256"));
    }

    #[test]
    fn rejects_path_escape() {
        let artifact = BundleArtifact {
            path: PathBuf::from("../model/config.json"),
            sha256: "0".repeat(64),
            bytes: 1,
        };

        let err = artifact.validate_schema("model.config").unwrap_err();
        assert!(err.to_string().contains("bundle root"));
    }
}
