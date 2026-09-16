//! Public Core ML compiled-artifact cache contracts.
//!
//! This module deliberately contains no Core ML or private-framework FFI. A
//! platform adapter compiles into the supplied temporary directory and runs a
//! known-answer check. Only a validated bundle is promoted into the cache.

use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MANIFEST_FILE: &str = "rvllm-coreml-artifact.json";
const IDENTITY_DOMAIN: &[u8] = b"rvllm.public-coreml-artifact.v1\0";
const BUNDLE_DIGEST_DOMAIN: &[u8] = b"rvllm.public-coreml-compiled-bundle.v1\0";
static TEMP_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlPrecision {
    Float16,
    Float32,
    Custom(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct CoreMlArtifactIdentity {
    pub model_fingerprint: String,
    pub exporter_fingerprint: String,
    pub precision: CoreMlPrecision,
    pub static_bucket: String,
    pub os_build: String,
    pub compiler_identity: String,
    pub device_family: String,
}

impl CoreMlArtifactIdentity {
    fn validate(&self) -> Result<(), CoreMlArtifactError> {
        for (field, value) in [
            ("model_fingerprint", self.model_fingerprint.as_str()),
            ("exporter_fingerprint", self.exporter_fingerprint.as_str()),
            ("static_bucket", self.static_bucket.as_str()),
            ("os_build", self.os_build.as_str()),
            ("compiler_identity", self.compiler_identity.as_str()),
            ("device_family", self.device_family.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CoreMlArtifactError::InvalidIdentity { field });
            }
        }
        if matches!(&self.precision, CoreMlPrecision::Custom(value) if value.trim().is_empty()) {
            return Err(CoreMlArtifactError::InvalidIdentity { field: "precision" });
        }
        Ok(())
    }

    /// Stable, domain-separated digest used as the on-disk cache key.
    pub fn cache_key(&self) -> Result<String, CoreMlArtifactError> {
        self.validate()?;
        let encoded = serde_json::to_vec(self).map_err(CoreMlArtifactError::ManifestEncode)?;
        let mut hasher = Sha256::new();
        hasher.update(IDENTITY_DOMAIN);
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlRequestedComputeUnits {
    All,
    CpuAndNeuralEngine,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlDeviceClass {
    Cpu,
    Gpu,
    NeuralEngine,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoreMlDeviceEvidence {
    /// Devices anticipated by public `MLComputePlan` inspection.
    pub compute_plan_anticipated: Option<Vec<CoreMlDeviceClass>>,
    /// Devices observed for the validated model by an Instruments run.
    pub instruments_observed: Option<Vec<CoreMlDeviceClass>>,
    /// Stable reference to the evidence artifact, never a claim by itself.
    pub evidence_reference: Option<String>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlEvidenceLevel {
    RequestedOnly,
    ComputePlanAnticipated,
    InstrumentsObserved,
    ComputePlanAndInstrumentsVerified,
}

impl CoreMlDeviceEvidence {
    #[must_use]
    pub fn evidence_level(&self) -> CoreMlEvidenceLevel {
        let plan = self.compute_plan_anticipated.is_some();
        let instruments = self.instruments_observed.is_some();
        match (plan, instruments) {
            (false, false) => CoreMlEvidenceLevel::RequestedOnly,
            (true, false) => CoreMlEvidenceLevel::ComputePlanAnticipated,
            (false, true) => CoreMlEvidenceLevel::InstrumentsObserved,
            (true, true) => CoreMlEvidenceLevel::ComputePlanAndInstrumentsVerified,
        }
    }

    /// A production ANE-acceleration claim requires both public compute-plan
    /// anticipation and Instruments observation for the same artifact.
    #[must_use]
    pub fn verifies_neural_engine_acceleration(&self) -> bool {
        contains_device(
            self.compute_plan_anticipated.as_deref(),
            CoreMlDeviceClass::NeuralEngine,
        ) && contains_device(
            self.instruments_observed.as_deref(),
            CoreMlDeviceClass::NeuralEngine,
        )
    }

    #[must_use]
    pub fn honest_summary(&self, requested: CoreMlRequestedComputeUnits) -> &'static str {
        if self.verifies_neural_engine_acceleration() {
            "ANE acceleration verified by MLComputePlan anticipation and Instruments observation"
        } else {
            match requested {
                CoreMlRequestedComputeUnits::All => {
                    "Core ML .all compute units requested; ANE execution is not verified"
                }
                CoreMlRequestedComputeUnits::CpuAndNeuralEngine => {
                    "Core ML CPU+Neural Engine compute units requested; ANE execution is not verified"
                }
            }
        }
    }
}

fn contains_device(devices: Option<&[CoreMlDeviceClass]>, expected: CoreMlDeviceClass) -> bool {
    devices
        .map(|values| values.contains(&expected))
        .unwrap_or(false)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnownAnswerValidation {
    pub case_id: String,
    pub input_digest_sha256: String,
    pub expected_output_digest_sha256: String,
    pub actual_output_digest_sha256: String,
    pub max_abs_error: f64,
    pub absolute_tolerance: f64,
    pub validated_at_unix_seconds: u64,
}

impl KnownAnswerValidation {
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.case_id.trim().is_empty()
            && is_sha256_hex(&self.input_digest_sha256)
            && is_sha256_hex(&self.expected_output_digest_sha256)
            && is_sha256_hex(&self.actual_output_digest_sha256)
            && self.max_abs_error.is_finite()
            && self.absolute_tolerance.is_finite()
            && self.max_abs_error >= 0.0
            && self.absolute_tolerance >= 0.0
            && (self.expected_output_digest_sha256 == self.actual_output_digest_sha256
                || self.max_abs_error <= self.absolute_tolerance)
    }
}

/// The exact known-answer contract required by a cache lookup.
///
/// The versioned `case_id` identifies the validator semantics. Digests bind
/// its input and expected output, while the tolerance makes quality-policy
/// changes fail closed instead of inheriting an older, looser validation.
#[derive(Clone, Debug, PartialEq)]
pub struct KnownAnswerPolicy {
    pub case_id: String,
    pub input_digest_sha256: String,
    pub expected_output_digest_sha256: String,
    pub absolute_tolerance: f64,
}

impl KnownAnswerPolicy {
    fn validate(&self) -> Result<(), CoreMlArtifactError> {
        if self.case_id.trim().is_empty() {
            return Err(CoreMlArtifactError::InvalidKnownAnswerPolicy { field: "case_id" });
        }
        if !is_sha256_hex(&self.input_digest_sha256) {
            return Err(CoreMlArtifactError::InvalidKnownAnswerPolicy {
                field: "input_digest_sha256",
            });
        }
        if !is_sha256_hex(&self.expected_output_digest_sha256) {
            return Err(CoreMlArtifactError::InvalidKnownAnswerPolicy {
                field: "expected_output_digest_sha256",
            });
        }
        if !self.absolute_tolerance.is_finite() || self.absolute_tolerance < 0.0 {
            return Err(CoreMlArtifactError::InvalidKnownAnswerPolicy {
                field: "absolute_tolerance",
            });
        }
        Ok(())
    }

    fn mismatch_field(&self, validation: &KnownAnswerValidation) -> Option<&'static str> {
        if self.case_id != validation.case_id {
            Some("case_id")
        } else if self.input_digest_sha256 != validation.input_digest_sha256 {
            Some("input_digest_sha256")
        } else if self.expected_output_digest_sha256 != validation.expected_output_digest_sha256 {
            Some("expected_output_digest_sha256")
        } else if self.absolute_tolerance != validation.absolute_tolerance {
            Some("absolute_tolerance")
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreMlArtifactManifest {
    pub schema_version: u32,
    pub identity: CoreMlArtifactIdentity,
    pub requested_compute_units: CoreMlRequestedComputeUnits,
    pub device_evidence: CoreMlDeviceEvidence,
    pub known_answer: KnownAnswerValidation,
    /// Canonical digest of every directory and regular file in the compiled
    /// bundle except this manifest.
    pub bundle_sha256: String,
}

impl CoreMlArtifactManifest {
    pub const SCHEMA_V2: u32 = 2;

    fn is_reusable_for(
        &self,
        identity: &CoreMlArtifactIdentity,
        requested_compute_units: CoreMlRequestedComputeUnits,
        device_evidence: &CoreMlDeviceEvidence,
    ) -> bool {
        self.schema_version == Self::SCHEMA_V2
            && &self.identity == identity
            && self.requested_compute_units == requested_compute_units
            && &self.device_evidence == device_evidence
            && self.known_answer.passed()
            && is_sha256_hex(&self.bundle_sha256)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArtifactDisposition {
    Reused,
    Built,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledArtifact {
    pub path: PathBuf,
    pub disposition: ArtifactDisposition,
    pub manifest: CoreMlArtifactManifest,
}

#[derive(Debug)]
pub enum CoreMlArtifactError {
    InvalidIdentity {
        field: &'static str,
    },
    Io {
        operation: &'static str,
        source: io::Error,
    },
    ManifestEncode(serde_json::Error),
    ManifestDecode(serde_json::Error),
    LockTimeout {
        path: PathBuf,
    },
    Build(String),
    KnownAnswerRejected {
        case_id: String,
    },
    InvalidKnownAnswerPolicy {
        field: &'static str,
    },
    KnownAnswerPolicyMismatch {
        field: &'static str,
    },
    InvalidBundle {
        reason: String,
    },
}

impl fmt::Display for CoreMlArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentity { field } => {
                write!(formatter, "empty Core ML identity field: {field}")
            }
            Self::Io { operation, source } => {
                write!(formatter, "Core ML cache {operation} failed: {source}")
            }
            Self::ManifestEncode(source) => {
                write!(formatter, "Core ML manifest encoding failed: {source}")
            }
            Self::ManifestDecode(source) => {
                write!(formatter, "Core ML manifest decoding failed: {source}")
            }
            Self::LockTimeout { path } => write!(
                formatter,
                "timed out waiting for Core ML cache lock {}",
                path.display()
            ),
            Self::Build(message) => write!(formatter, "Core ML artifact build failed: {message}"),
            Self::KnownAnswerRejected { case_id } => write!(
                formatter,
                "Core ML known-answer validation failed for {case_id}"
            ),
            Self::InvalidKnownAnswerPolicy { field } => {
                write!(
                    formatter,
                    "invalid Core ML known-answer policy field: {field}"
                )
            }
            Self::KnownAnswerPolicyMismatch { field } => write!(
                formatter,
                "Core ML known-answer result does not match requested policy field: {field}"
            ),
            Self::InvalidBundle { reason } => {
                write!(formatter, "Core ML compiled bundle is invalid: {reason}")
            }
        }
    }
}

impl std::error::Error for CoreMlArtifactError {}

#[derive(Clone, Debug)]
pub struct CoreMlArtifactCache {
    root: PathBuf,
    lock_timeout: Duration,
    lock_poll_interval: Duration,
}

impl CoreMlArtifactCache {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            lock_timeout: Duration::from_secs(120),
            lock_poll_interval: Duration::from_millis(10),
        }
    }

    #[must_use]
    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// Return a validated compiled bundle, building it exactly once per cache
    /// identity. `build` must place the complete bundle in the supplied empty
    /// directory. `validate` must execute a known-answer check against it.
    pub fn get_or_build<B, V>(
        &self,
        identity: CoreMlArtifactIdentity,
        requested_compute_units: CoreMlRequestedComputeUnits,
        device_evidence: CoreMlDeviceEvidence,
        build: B,
        validate: V,
    ) -> Result<CompiledArtifact, CoreMlArtifactError>
    where
        B: FnOnce(&Path) -> Result<(), String>,
        V: FnOnce(&Path) -> Result<KnownAnswerValidation, String>,
    {
        self.get_or_build_inner(
            identity,
            requested_compute_units,
            device_evidence,
            None,
            build,
            validate,
        )
    }

    /// Policy-bound cache lookup for shipping callers.
    ///
    /// Exact policy hits avoid both compilation and validation. If only the
    /// policy changed, the existing authenticated compiled payload is
    /// revalidated in place and its manifest is atomically refreshed.
    pub fn get_or_build_with_known_answer_policy<B, V>(
        &self,
        identity: CoreMlArtifactIdentity,
        requested_compute_units: CoreMlRequestedComputeUnits,
        device_evidence: CoreMlDeviceEvidence,
        policy: KnownAnswerPolicy,
        build: B,
        validate: V,
    ) -> Result<CompiledArtifact, CoreMlArtifactError>
    where
        B: FnOnce(&Path) -> Result<(), String>,
        V: FnOnce(&Path) -> Result<KnownAnswerValidation, String>,
    {
        policy.validate()?;
        self.get_or_build_inner(
            identity,
            requested_compute_units,
            device_evidence,
            Some(&policy),
            build,
            validate,
        )
    }

    fn get_or_build_inner<B, V>(
        &self,
        identity: CoreMlArtifactIdentity,
        requested_compute_units: CoreMlRequestedComputeUnits,
        device_evidence: CoreMlDeviceEvidence,
        policy: Option<&KnownAnswerPolicy>,
        build: B,
        validate: V,
    ) -> Result<CompiledArtifact, CoreMlArtifactError>
    where
        B: FnOnce(&Path) -> Result<(), String>,
        V: FnOnce(&Path) -> Result<KnownAnswerValidation, String>,
    {
        let key = identity.cache_key()?;
        fs::create_dir_all(&self.root).map_err(|source| CoreMlArtifactError::Io {
            operation: "root creation",
            source,
        })?;
        let target = self.root.join(format!("{key}.mlmodelc"));
        let lock_path = self.root.join(format!("{key}.lock"));
        let _lock = CacheLock::acquire(&lock_path, self.lock_timeout, self.lock_poll_interval)?;
        let mut validate = Some(validate);

        if let Some(mut manifest) = read_reusable_manifest(
            &target,
            &identity,
            requested_compute_units,
            &device_evidence,
        )? {
            if let Some(policy) = policy {
                if policy.mismatch_field(&manifest.known_answer).is_some() {
                    let known_answer =
                        validate.take().expect("validator is consumed at most once")(&target)
                            .map_err(CoreMlArtifactError::Build)?;
                    require_policy_match(policy, &known_answer)?;
                    if !known_answer.passed() {
                        return Err(CoreMlArtifactError::KnownAnswerRejected {
                            case_id: known_answer.case_id,
                        });
                    }
                    manifest.known_answer = known_answer;
                    replace_manifest(&target, &manifest)?;
                }
            }
            return Ok(CompiledArtifact {
                path: target,
                disposition: ArtifactDisposition::Reused,
                manifest,
            });
        }

        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let temp = self
            .root
            .join(format!(".{key}.{}.{}.tmp", std::process::id(), nonce));
        fs::create_dir(&temp).map_err(|source| CoreMlArtifactError::Io {
            operation: "temporary bundle creation",
            source,
        })?;
        let mut temp_guard = TemporaryBundle::new(temp.clone());
        build(&temp).map_err(CoreMlArtifactError::Build)?;
        let known_answer = validate.take().expect("validator is consumed at most once")(&temp)
            .map_err(CoreMlArtifactError::Build)?;
        if let Some(policy) = policy {
            require_policy_match(policy, &known_answer)?;
        }
        if !known_answer.passed() {
            return Err(CoreMlArtifactError::KnownAnswerRejected {
                case_id: known_answer.case_id,
            });
        }
        let bundle_sha256 = bundle_digest(&temp)?;

        let manifest = CoreMlArtifactManifest {
            schema_version: CoreMlArtifactManifest::SCHEMA_V2,
            identity,
            requested_compute_units,
            device_evidence,
            known_answer,
            bundle_sha256,
        };
        write_manifest(&temp, &manifest)?;

        let quarantine = self.root.join(format!(".{key}.{}.invalid", nonce));
        let had_target = target.exists();
        if had_target {
            fs::rename(&target, &quarantine).map_err(|source| CoreMlArtifactError::Io {
                operation: "invalid artifact quarantine",
                source,
            })?;
        }
        if let Err(source) = fs::rename(&temp, &target) {
            if had_target {
                let _ = fs::rename(&quarantine, &target);
            }
            return Err(CoreMlArtifactError::Io {
                operation: "atomic artifact promotion",
                source,
            });
        }
        temp_guard.promoted = true;
        if had_target {
            let _ = fs::remove_dir_all(quarantine);
        }

        Ok(CompiledArtifact {
            path: target,
            disposition: ArtifactDisposition::Built,
            manifest,
        })
    }
}

struct CacheLock {
    _file: File,
}

impl CacheLock {
    fn acquire(
        path: &Path,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<Self, CoreMlArtifactError> {
        let started = Instant::now();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|source| CoreMlArtifactError::Io {
                operation: "lock file open",
                source,
            })?;
        loop {
            match FileExt::try_lock_exclusive(&file) {
                Ok(true) => {
                    file.set_len(0)
                        .and_then(|()| {
                            writeln!(
                                file,
                                "pid={} acquired_unix={}",
                                std::process::id(),
                                unix_seconds()
                            )
                        })
                        .and_then(|()| file.sync_all())
                        .map_err(|source| CoreMlArtifactError::Io {
                            operation: "lock metadata write",
                            source,
                        })?;
                    return Ok(Self { _file: file });
                }
                Ok(false) => {
                    if started.elapsed() >= timeout {
                        return Err(CoreMlArtifactError::LockTimeout {
                            path: path.to_path_buf(),
                        });
                    }
                    thread::sleep(poll_interval.min(timeout.saturating_sub(started.elapsed())));
                }
                Err(source) => {
                    return Err(CoreMlArtifactError::Io {
                        operation: "lock acquisition",
                        source,
                    });
                }
            }
        }
    }
}

struct TemporaryBundle {
    path: PathBuf,
    promoted: bool,
}

impl TemporaryBundle {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            promoted: false,
        }
    }
}

impl Drop for TemporaryBundle {
    fn drop(&mut self) {
        if !self.promoted {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn read_reusable_manifest(
    bundle: &Path,
    identity: &CoreMlArtifactIdentity,
    requested_compute_units: CoreMlRequestedComputeUnits,
    device_evidence: &CoreMlDeviceEvidence,
) -> Result<Option<CoreMlArtifactManifest>, CoreMlArtifactError> {
    if !bundle.is_dir() {
        return Ok(None);
    }
    let bytes = match fs::read(bundle.join(MANIFEST_FILE)) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(CoreMlArtifactError::Io {
                operation: "manifest read",
                source,
            });
        }
    };
    let manifest: CoreMlArtifactManifest = match serde_json::from_slice(&bytes) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(None),
    };
    if !manifest.is_reusable_for(identity, requested_compute_units, device_evidence) {
        return Ok(None);
    }
    let actual_digest = match bundle_digest(bundle) {
        Ok(digest) => digest,
        Err(CoreMlArtifactError::InvalidBundle { .. }) => return Ok(None),
        Err(error) => return Err(error),
    };
    Ok((actual_digest == manifest.bundle_sha256).then_some(manifest))
}

fn bundle_digest(bundle: &Path) -> Result<String, CoreMlArtifactError> {
    let mut entries = Vec::new();
    collect_bundle_entries(bundle, Path::new(""), &mut entries)?;
    entries.sort_unstable();
    if entries.is_empty() {
        return Err(CoreMlArtifactError::InvalidBundle {
            reason: "compiled bundle contains no payload entries".to_owned(),
        });
    }

    let mut hasher = Sha256::new();
    hasher.update(BUNDLE_DIGEST_DOMAIN);
    let mut buffer = vec![0_u8; 64 * 1024];
    for relative in entries {
        let path = bundle.join(&relative);
        let metadata = fs::symlink_metadata(&path).map_err(|source| CoreMlArtifactError::Io {
            operation: "bundle entry metadata read",
            source,
        })?;
        let encoded = relative.as_os_str().as_encoded_bytes();
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(encoded);
        if metadata.is_dir() {
            hasher.update([0]);
            continue;
        }
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(CoreMlArtifactError::InvalidBundle {
                reason: format!(
                    "compiled bundle entry {} is not a regular file or directory",
                    relative.display()
                ),
            });
        }
        hasher.update([1]);
        hasher.update(metadata.len().to_le_bytes());
        let mut file = File::open(&path).map_err(|source| CoreMlArtifactError::Io {
            operation: "bundle payload read",
            source,
        })?;
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|source| CoreMlArtifactError::Io {
                    operation: "bundle payload read",
                    source,
                })?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn collect_bundle_entries(
    bundle: &Path,
    relative: &Path,
    entries: &mut Vec<PathBuf>,
) -> Result<(), CoreMlArtifactError> {
    let directory = bundle.join(relative);
    for entry in fs::read_dir(&directory).map_err(|source| CoreMlArtifactError::Io {
        operation: "bundle directory read",
        source,
    })? {
        let entry = entry.map_err(|source| CoreMlArtifactError::Io {
            operation: "bundle directory entry read",
            source,
        })?;
        let child = relative.join(entry.file_name());
        if child == Path::new(MANIFEST_FILE) {
            continue;
        }
        let metadata =
            fs::symlink_metadata(entry.path()).map_err(|source| CoreMlArtifactError::Io {
                operation: "bundle entry metadata read",
                source,
            })?;
        if metadata.file_type().is_symlink() {
            return Err(CoreMlArtifactError::InvalidBundle {
                reason: format!("compiled bundle contains symlink {}", child.display()),
            });
        }
        entries.push(child.clone());
        if metadata.is_dir() {
            collect_bundle_entries(bundle, &child, entries)?;
        } else if !metadata.is_file() {
            return Err(CoreMlArtifactError::InvalidBundle {
                reason: format!(
                    "compiled bundle entry {} is not a regular file or directory",
                    child.display()
                ),
            });
        }
    }
    Ok(())
}

fn write_manifest(
    bundle: &Path,
    manifest: &CoreMlArtifactManifest,
) -> Result<(), CoreMlArtifactError> {
    let bytes = serde_json::to_vec_pretty(manifest).map_err(CoreMlArtifactError::ManifestEncode)?;
    let path = bundle.join(MANIFEST_FILE);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| CoreMlArtifactError::Io {
            operation: "manifest creation",
            source,
        })?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| CoreMlArtifactError::Io {
            operation: "manifest durable write",
            source,
        })
}

fn replace_manifest(
    bundle: &Path,
    manifest: &CoreMlArtifactManifest,
) -> Result<(), CoreMlArtifactError> {
    let bytes = serde_json::to_vec_pretty(manifest).map_err(CoreMlArtifactError::ManifestEncode)?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = bundle.join(format!(
        ".{MANIFEST_FILE}.{}.{}.tmp",
        std::process::id(),
        nonce
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|source| CoreMlArtifactError::Io {
            operation: "replacement manifest creation",
            source,
        })?;
    if let Err(source) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(CoreMlArtifactError::Io {
            operation: "replacement manifest durable write",
            source,
        });
    }
    drop(file);
    if let Err(source) = fs::rename(&temporary, bundle.join(MANIFEST_FILE)) {
        let _ = fs::remove_file(&temporary);
        return Err(CoreMlArtifactError::Io {
            operation: "atomic manifest replacement",
            source,
        });
    }
    File::open(bundle)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| CoreMlArtifactError::Io {
            operation: "manifest directory sync",
            source,
        })
}

fn require_policy_match(
    policy: &KnownAnswerPolicy,
    validation: &KnownAnswerValidation,
) -> Result<(), CoreMlArtifactError> {
    if let Some(field) = policy.mismatch_field(validation) {
        Err(CoreMlArtifactError::KnownAnswerPolicyMismatch { field })
    } else {
        Ok(())
    }
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

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn identity() -> CoreMlArtifactIdentity {
        CoreMlArtifactIdentity {
            model_fingerprint: "model-sha256".to_string(),
            exporter_fingerprint: "exporter-v7".to_string(),
            precision: CoreMlPrecision::Float16,
            static_bucket: "b1-t1".to_string(),
            os_build: "24A335".to_string(),
            compiler_identity: "coreml-compiler-1".to_string(),
            device_family: "apple-gpu-family-9".to_string(),
        }
    }

    fn validation(passes: bool) -> KnownAnswerValidation {
        KnownAnswerValidation {
            case_id: "projection-kat-v1".to_string(),
            input_digest_sha256: "11".repeat(32),
            expected_output_digest_sha256: "22".repeat(32),
            actual_output_digest_sha256: if passes {
                "22".repeat(32)
            } else {
                "33".repeat(32)
            },
            max_abs_error: if passes { 0.0 } else { 1.0 },
            absolute_tolerance: 0.001,
            validated_at_unix_seconds: 1,
        }
    }

    fn no_evidence() -> CoreMlDeviceEvidence {
        CoreMlDeviceEvidence {
            compute_plan_anticipated: None,
            instruments_observed: None,
            evidence_reference: None,
        }
    }

    fn policy(tolerance: f64) -> KnownAnswerPolicy {
        KnownAnswerPolicy {
            case_id: "projection-kat-v1".to_string(),
            input_digest_sha256: "11".repeat(32),
            expected_output_digest_sha256: "22".repeat(32),
            absolute_tolerance: tolerance,
        }
    }

    fn approximate_validation(max_abs_error: f64, tolerance: f64) -> KnownAnswerValidation {
        KnownAnswerValidation {
            case_id: "projection-kat-v1".to_string(),
            input_digest_sha256: "11".repeat(32),
            expected_output_digest_sha256: "22".repeat(32),
            actual_output_digest_sha256: "33".repeat(32),
            max_abs_error,
            absolute_tolerance: tolerance,
            validated_at_unix_seconds: 1,
        }
    }

    #[test]
    fn cache_identity_covers_every_compilation_dimension() -> Result<(), Box<dyn std::error::Error>>
    {
        let base = identity();
        let base_key = base.cache_key()?;
        let variants = [
            CoreMlArtifactIdentity {
                model_fingerprint: "other".into(),
                ..base.clone()
            },
            CoreMlArtifactIdentity {
                exporter_fingerprint: "other".into(),
                ..base.clone()
            },
            CoreMlArtifactIdentity {
                precision: CoreMlPrecision::Float32,
                ..base.clone()
            },
            CoreMlArtifactIdentity {
                static_bucket: "b4-t1".into(),
                ..base.clone()
            },
            CoreMlArtifactIdentity {
                os_build: "other".into(),
                ..base.clone()
            },
            CoreMlArtifactIdentity {
                compiler_identity: "other".into(),
                ..base.clone()
            },
            CoreMlArtifactIdentity {
                device_family: "other".into(),
                ..base
            },
        ];
        for variant in variants {
            assert_ne!(base_key, variant.cache_key()?);
        }
        Ok(())
    }

    #[test]
    fn requested_compute_units_are_not_execution_evidence() {
        let evidence = no_evidence();
        assert_eq!(
            evidence.evidence_level(),
            CoreMlEvidenceLevel::RequestedOnly
        );
        assert!(!evidence.verifies_neural_engine_acceleration());
        assert!(evidence
            .honest_summary(CoreMlRequestedComputeUnits::CpuAndNeuralEngine)
            .contains("requested"));

        let plan_only = CoreMlDeviceEvidence {
            compute_plan_anticipated: Some(vec![CoreMlDeviceClass::NeuralEngine]),
            instruments_observed: None,
            evidence_reference: Some("plan.json".into()),
        };
        assert!(!plan_only.verifies_neural_engine_acceleration());

        let verified = CoreMlDeviceEvidence {
            compute_plan_anticipated: Some(vec![CoreMlDeviceClass::NeuralEngine]),
            instruments_observed: Some(vec![CoreMlDeviceClass::NeuralEngine]),
            evidence_reference: Some("instruments.trace".into()),
        };
        assert!(verified.verifies_neural_engine_acceleration());
    }

    #[test]
    fn validation_failure_never_promotes_artifact() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache = CoreMlArtifactCache::new(root.path());
        let result = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::CpuAndNeuralEngine,
            no_evidence(),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(false)),
        );
        assert!(matches!(
            result,
            Err(CoreMlArtifactError::KnownAnswerRejected { .. })
        ));
        let entries = fs::read_dir(root.path())?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].extension().and_then(|value| value.to_str()),
            Some("lock")
        );
        Ok(())
    }

    #[test]
    fn validated_artifact_is_reused_without_rebuilding() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache = CoreMlArtifactCache::new(root.path());
        let builds = Arc::new(AtomicUsize::new(0));
        let first_builds = Arc::clone(&builds);
        let first = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            move |directory| {
                first_builds.fetch_add(1, Ordering::SeqCst);
                fs::write(directory.join("model.bin"), b"compiled")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(true)),
        )?;
        assert_eq!(first.disposition, ArtifactDisposition::Built);

        let second_builds = Arc::clone(&builds);
        let second = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            move |_| {
                second_builds.fetch_add(1, Ordering::SeqCst);
                Err("must not rebuild".to_string())
            },
            |_| Err("must not revalidate".to_string()),
        )?;
        assert_eq!(second.disposition, ArtifactDisposition::Reused);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn request_and_evidence_changes_never_reuse_stale_reporting(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache = CoreMlArtifactCache::new(root.path());
        let verified = CoreMlDeviceEvidence {
            compute_plan_anticipated: Some(vec![CoreMlDeviceClass::NeuralEngine]),
            instruments_observed: Some(vec![CoreMlDeviceClass::NeuralEngine]),
            evidence_reference: Some("first-run.trace".into()),
        };
        let first = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            verified,
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled-1")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(true)),
        )?;
        assert_eq!(first.disposition, ArtifactDisposition::Built);

        let second = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::CpuAndNeuralEngine,
            no_evidence(),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled-2")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(true)),
        )?;
        assert_eq!(second.disposition, ArtifactDisposition::Built);
        assert_eq!(
            second.manifest.requested_compute_units,
            CoreMlRequestedComputeUnits::CpuAndNeuralEngine
        );
        assert_eq!(second.manifest.device_evidence, no_evidence());
        assert!(!second
            .manifest
            .device_evidence
            .verifies_neural_engine_acceleration());
        Ok(())
    }

    #[test]
    fn policy_change_revalidates_payload_without_recompiling(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache = CoreMlArtifactCache::new(root.path());
        let first = cache.get_or_build_with_known_answer_policy(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            policy(2.0),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(approximate_validation(1.0, 2.0)),
        )?;
        assert_eq!(first.disposition, ArtifactDisposition::Built);

        let builds = AtomicUsize::new(0);
        let validations = AtomicUsize::new(0);
        let second = cache.get_or_build_with_known_answer_policy(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            policy(0.001),
            |_| {
                builds.fetch_add(1, Ordering::SeqCst);
                Err("must not rebuild an authenticated payload".to_string())
            },
            |_| {
                validations.fetch_add(1, Ordering::SeqCst);
                Ok(approximate_validation(0.0005, 0.001))
            },
        )?;
        assert_eq!(second.disposition, ArtifactDisposition::Reused);
        assert_eq!(builds.load(Ordering::SeqCst), 0);
        assert_eq!(validations.load(Ordering::SeqCst), 1);
        assert_eq!(second.manifest.known_answer.absolute_tolerance, 0.001);

        let third = cache.get_or_build_with_known_answer_policy(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            policy(0.001),
            |_| Err("must not rebuild".to_string()),
            |_| Err("must not revalidate exact policy hit".to_string()),
        )?;
        assert_eq!(third.disposition, ArtifactDisposition::Reused);
        Ok(())
    }

    #[test]
    fn rejected_stricter_policy_preserves_existing_manifest_and_payload(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache = CoreMlArtifactCache::new(root.path());
        let first = cache.get_or_build_with_known_answer_policy(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            policy(2.0),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(approximate_validation(1.0, 2.0)),
        )?;
        let manifest_before = fs::read(first.path.join(MANIFEST_FILE))?;
        let payload_before = fs::read(first.path.join("model.bin"))?;

        let result = cache.get_or_build_with_known_answer_policy(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            policy(0.001),
            |_| Err("must not rebuild".to_string()),
            |_| Ok(approximate_validation(0.01, 0.001)),
        );
        assert!(matches!(
            result,
            Err(CoreMlArtifactError::KnownAnswerRejected { .. })
        ));
        assert_eq!(fs::read(first.path.join(MANIFEST_FILE))?, manifest_before);
        assert_eq!(fs::read(first.path.join("model.bin"))?, payload_before);
        Ok(())
    }

    #[test]
    fn invalid_or_mismatched_policy_never_promotes() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache = CoreMlArtifactCache::new(root.path());
        let builds = AtomicUsize::new(0);
        let mut invalid = policy(0.001);
        invalid.absolute_tolerance = f64::NAN;
        let result = cache.get_or_build_with_known_answer_policy(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            invalid,
            |_| {
                builds.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            |_| Ok(approximate_validation(0.0, 0.001)),
        );
        assert!(matches!(
            result,
            Err(CoreMlArtifactError::InvalidKnownAnswerPolicy {
                field: "absolute_tolerance"
            })
        ));
        assert_eq!(builds.load(Ordering::SeqCst), 0);

        let mut mismatched = approximate_validation(0.0, 0.001);
        mismatched.case_id = "different-validator-v1".into();
        let result = cache.get_or_build_with_known_answer_policy(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            policy(0.001),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(mismatched),
        );
        assert!(matches!(
            result,
            Err(CoreMlArtifactError::KnownAnswerPolicyMismatch { field: "case_id" })
        ));
        assert!(!root
            .path()
            .join(format!("{}.mlmodelc", identity().cache_key()?))
            .exists());
        Ok(())
    }

    #[test]
    fn tampered_deleted_or_corrupt_compiled_bundle_is_never_reused(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache = CoreMlArtifactCache::new(root.path());
        let first = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled-1")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(true)),
        )?;
        assert_eq!(first.disposition, ArtifactDisposition::Built);

        fs::write(first.path.join("model.bin"), b"tampered")?;
        let second = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled-2")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(true)),
        )?;
        assert_eq!(second.disposition, ArtifactDisposition::Built);
        assert_eq!(fs::read(second.path.join("model.bin"))?, b"compiled-2");

        fs::remove_file(second.path.join("model.bin"))?;
        let third = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled-3")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(true)),
        )?;
        assert_eq!(third.disposition, ArtifactDisposition::Built);
        assert_eq!(fs::read(third.path.join("model.bin"))?, b"compiled-3");

        fs::write(third.path.join(MANIFEST_FILE), b"{broken")?;
        let fourth = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled-4")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(true)),
        )?;
        assert_eq!(fourth.disposition, ArtifactDisposition::Built);
        assert_eq!(fs::read(fourth.path.join("model.bin"))?, b"compiled-4");
        Ok(())
    }

    #[test]
    fn held_lock_times_out_without_running_builder() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache =
            CoreMlArtifactCache::new(root.path()).with_lock_timeout(Duration::from_millis(5));
        let key = identity().cache_key()?;
        let held = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.path().join(format!("{key}.lock")))?;
        assert!(FileExt::try_lock_exclusive(&held)?);
        let built = AtomicUsize::new(0);
        let result = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            |_| {
                built.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            |_| Ok(validation(true)),
        );
        assert!(matches!(
            result,
            Err(CoreMlArtifactError::LockTimeout { .. })
        ));
        assert_eq!(built.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[test]
    fn stale_lock_path_without_live_owner_does_not_block_build(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let cache =
            CoreMlArtifactCache::new(root.path()).with_lock_timeout(Duration::from_millis(20));
        let key = identity().cache_key()?;
        fs::write(root.path().join(format!("{key}.lock")), b"stale metadata")?;

        let artifact = cache.get_or_build(
            identity(),
            CoreMlRequestedComputeUnits::All,
            no_evidence(),
            |directory| {
                fs::write(directory.join("model.bin"), b"compiled")
                    .map_err(|error| error.to_string())
            },
            |_| Ok(validation(true)),
        )?;
        assert_eq!(artifact.disposition, ArtifactDisposition::Built);
        Ok(())
    }
}
