//! Opt-in encrypted T3 prompt-prefix cache storage.
//!
//! This module intentionally does not obtain or persist encryption keys. Apple
//! hosts must supply a per-install 256-bit key from Keychain and apply iOS Data
//! Protection to the configured directory. The store only writes immutable,
//! complete prompt-prefix pages; generated continuation KV is outside its API.

use crate::prompt_cache::{
    reusable_prompt_tokens, CacheIdentity, CacheNamespace, WarmPage, PROMPT_CACHE_PAGE_TOKENS,
};
use aes_gcm::aead::{Aead, AeadCore, OsRng, Payload};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use rvllm_core::TokenId;
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::ffi::{CStr, CString, OsString};
#[cfg(any(test, not(unix)))]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(all(test, unix))]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
#[cfg(unix)]
use std::path::Component;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use zeroize::{Zeroize, Zeroizing};

const MAGIC: &[u8; 8] = b"RVLLMT3\0";
pub const PERSISTENT_PROMPT_CACHE_FORMAT_VERSION: u32 = 2;
const NONCE_LEN: usize = 12;
const HASH_LEN: usize = 32;
const HEADER_LEN: usize = 8 + 4 + NONCE_LEN + HASH_LEN + HASH_LEN + 8 + 8 + HASH_LEN;
const TAG_LEN: u64 = 16;
const RECORD_EXTENSION: &str = "rvpc";
const TENANT_LOCK_FILE: &str = ".tenant.lock";
const MAX_RECORD_BYTES_IOS: u64 = 32 * 1024 * 1024;
const MAX_RECORD_BYTES_MACOS: u64 = 128 * 1024 * 1024;

/// Serializes the final cancellation check with publication. The continuous
/// worker takes the exclusive side while advancing its I/O epoch, so memory
/// pressure cannot return while an older promotion can still rename a record.
#[derive(Default)]
pub(crate) struct PersistentPublishGate {
    lock: RwLock<()>,
}

impl PersistentPublishGate {
    pub(crate) fn invalidate(&self, invalidate_epoch: impl FnOnce()) {
        let _guard = self
            .lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        invalidate_epoch();
    }

    fn publish_if_current<T>(
        &self,
        cancelled: impl FnOnce() -> bool,
        publish: impl FnOnce() -> Result<T, PersistentCacheError>,
    ) -> Result<T, PersistentCacheError> {
        let _guard = self
            .lock
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        publish()
    }
}

/// Host-owned Keychain material. Debug output is always redacted and the
/// backing bytes are overwritten when the wrapper is dropped.
pub struct PersistentCacheKey([u8; 32]);

impl PersistentCacheKey {
    #[must_use]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    fn take_bytes(mut self) -> [u8; 32] {
        std::mem::replace(&mut self.0, [0; 32])
    }
}

impl std::fmt::Debug for PersistentCacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PersistentCacheKey([REDACTED; 32])")
    }
}

impl Drop for PersistentCacheKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

/// Explicit host authorization and resources for encrypted T3. Shipping hosts
/// populate `key` from a per-install Keychain item and apply platform Data
/// Protection to `root` on iOS.
pub struct PersistentCacheHostConfig {
    pub consent: bool,
    pub root: PathBuf,
    pub quota_bytes: u64,
    pub max_record_bytes: u64,
    /// Engine-level host-derived tenant scope. This is never request input.
    pub cache_namespace: String,
    pub key: Option<PersistentCacheKey>,
}

impl std::fmt::Debug for PersistentCacheHostConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentCacheHostConfig")
            .field("consent", &self.consent)
            .field("root", &self.root)
            .field("quota_bytes", &self.quota_bytes)
            .field("max_record_bytes", &self.max_record_bytes)
            .field("cache_namespace", &"[REDACTED]")
            .field("key", &"[REDACTED]")
            .finish()
    }
}

impl PersistentCacheHostConfig {
    pub fn validate(&self) -> Result<(), PersistentCacheError> {
        let platform_limit = if cfg!(target_os = "ios") {
            MAX_RECORD_BYTES_IOS
        } else {
            MAX_RECORD_BYTES_MACOS
        };
        if !self.consent
            || self.root.as_os_str().is_empty()
            || self.quota_bytes == 0
            || self.max_record_bytes == 0
            || self.max_record_bytes > self.quota_bytes
            || self.max_record_bytes > platform_limit
        {
            return Err(PersistentCacheError::InvalidConfig);
        }
        CacheNamespace::new(&self.cache_namespace)
            .map_err(|_| PersistentCacheError::InvalidConfig)?;
        validate_cache_root(&self.root)?;
        if self.key.is_none() {
            return Err(PersistentCacheError::MissingKey);
        }
        Ok(())
    }

    #[must_use]
    pub fn tenant_namespace(&self) -> String {
        self.cache_namespace.clone()
    }

    pub fn build_store(mut self) -> Result<PersistentPromptCache, PersistentCacheError> {
        self.validate()?;
        let mut key = self
            .key
            .take()
            .ok_or(PersistentCacheError::MissingKey)?
            .take_bytes();
        let result = PersistentPromptCache::new(
            PersistentPromptCacheConfig {
                enabled: true,
                directory: self.root.clone(),
                max_bytes_per_namespace: self.quota_bytes,
                max_record_bytes: self.max_record_bytes,
            },
            key,
        );
        key.zeroize();
        result
    }
}

/// T3 is disabled unless the host explicitly enables it and supplies a path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheConfig {
    pub enabled: bool,
    pub directory: PathBuf,
    /// Independent on-disk quota for each tenant namespace.
    pub max_bytes_per_namespace: u64,
    /// Defensive upper bound for any one encrypted record.
    pub max_record_bytes: u64,
}

impl Default for PersistentPromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            directory: PathBuf::new(),
            max_bytes_per_namespace: 0,
            max_record_bytes: 0,
        }
    }
}

/// A measured, candidate-specific restore/recompute comparison.
///
/// The scheduler should populate these values from device calibration and
/// observed T3 reads. A record is not opened when restore is not predicted to
/// be strictly cheaper than recomputation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RestoreCostEstimate {
    pub measured_restore_ns: u64,
    pub measured_recompute_ns: u64,
}

impl RestoreCostEstimate {
    #[must_use]
    pub const fn should_restore(self) -> bool {
        self.measured_restore_ns > 0 && self.measured_restore_ns < self.measured_recompute_ns
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentCacheRecord {
    pub matched_tokens: usize,
    pub pages: Vec<WarmPage>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PersistentStoreOutcome {
    Disabled,
    Stored { bytes: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PersistentLookup {
    Disabled,
    Recompute,
    Miss,
    Hit(PersistentCacheRecord),
}

#[derive(Debug)]
pub enum PersistentCacheError {
    InvalidConfig,
    MissingKey,
    Cancelled,
    EmptyPrefix,
    PageCountMismatch,
    ZeroSizedPage,
    RecordTooLarge,
    QuotaExceeded,
    CorruptRecord,
    AuthenticationFailed,
    IdentityMismatch,
    Io(std::io::Error),
}

impl std::fmt::Display for PersistentCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid persistent prompt-cache configuration"),
            Self::MissingKey => write!(f, "persistent prompt-cache key is unavailable"),
            Self::Cancelled => write!(f, "persistent prompt-cache operation was cancelled"),
            Self::EmptyPrefix => write!(f, "prompt has no complete private-tail-safe cache page"),
            Self::PageCountMismatch => {
                write!(f, "KV page count does not match reusable prompt prefix")
            }
            Self::ZeroSizedPage => write!(f, "persistent KV pages must not be empty"),
            Self::RecordTooLarge => write!(f, "persistent prompt-cache record exceeds its limit"),
            Self::QuotaExceeded => write!(f, "persistent prompt-cache tenant quota exceeded"),
            Self::CorruptRecord => {
                write!(f, "persistent prompt-cache record is corrupt or partial")
            }
            Self::AuthenticationFailed => {
                write!(f, "persistent prompt-cache authentication failed")
            }
            Self::IdentityMismatch => {
                write!(f, "persistent prompt-cache identity or token mismatch")
            }
            Self::Io(error) => write!(f, "persistent prompt-cache I/O error: {error}"),
        }
    }
}

impl std::error::Error for PersistentCacheError {}

impl From<std::io::Error> for PersistentCacheError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// Encrypted, immutable, per-tenant T3 storage.
pub struct PersistentPromptCache {
    config: PersistentPromptCacheConfig,
    cipher: Aes256Gcm,
    path_key: PersistentCacheKey,
    storage_root: Option<StorageRoot>,
}

impl PersistentPromptCache {
    pub fn new(
        config: PersistentPromptCacheConfig,
        mut per_install_key: [u8; 32],
    ) -> Result<Self, PersistentCacheError> {
        if config.enabled
            && (config.directory.as_os_str().is_empty()
                || config.max_bytes_per_namespace == 0
                || config.max_record_bytes == 0
                || config.max_record_bytes > config.max_bytes_per_namespace
                || config.max_record_bytes
                    > if cfg!(target_os = "ios") {
                        MAX_RECORD_BYTES_IOS
                    } else {
                        MAX_RECORD_BYTES_MACOS
                    })
        {
            return Err(PersistentCacheError::InvalidConfig);
        }
        let storage_root = if config.enabled {
            Some(StorageRoot::open(&config.directory)?)
        } else {
            None
        };
        let hkdf = Hkdf::<Sha256>::new(
            Some(b"rvllm.prompt-cache.key-schedule.v2"),
            &per_install_key,
        );
        let mut encryption_key = [0_u8; 32];
        let mut path_key = [0_u8; 32];
        hkdf.expand(b"aes-256-gcm-record-encryption", &mut encryption_key)
            .map_err(|_| PersistentCacheError::InvalidConfig)?;
        hkdf.expand(b"hmac-sha256-record-paths", &mut path_key)
            .map_err(|_| PersistentCacheError::InvalidConfig)?;
        let cipher = Aes256Gcm::new((&encryption_key).into());
        encryption_key.zeroize();
        per_install_key.zeroize();
        Ok(Self {
            config,
            cipher,
            path_key: PersistentCacheKey::new(path_key),
            storage_root,
        })
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Store only the complete 32-token pages before the final prompt token.
    /// Extra pages are rejected, preventing continuation KV from being folded
    /// into a prompt record accidentally.
    pub fn store_prompt(
        &self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        pages: &[WarmPage],
    ) -> Result<PersistentStoreOutcome, PersistentCacheError> {
        self.store_prompt_cancellable(identity, prompt, pages, || false)
    }

    pub fn store_prompt_cancellable<C>(
        &self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        pages: &[WarmPage],
        cancelled: C,
    ) -> Result<PersistentStoreOutcome, PersistentCacheError>
    where
        C: Fn() -> bool,
    {
        self.store_prompt_cancellable_with_publish_gate(identity, prompt, pages, cancelled, None)
    }

    pub(crate) fn store_prompt_cancellable_with_publish_gate<C>(
        &self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        pages: &[WarmPage],
        cancelled: C,
        publish_gate: Option<&PersistentPublishGate>,
    ) -> Result<PersistentStoreOutcome, PersistentCacheError>
    where
        C: Fn() -> bool,
    {
        if !self.config.enabled {
            return Ok(PersistentStoreOutcome::Disabled);
        }
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        let reusable = reusable_prompt_tokens(prompt.len());
        if reusable == 0 {
            return Err(PersistentCacheError::EmptyPrefix);
        }
        if pages.len() != reusable / PROMPT_CACHE_PAGE_TOKENS {
            return Err(PersistentCacheError::PageCountMismatch);
        }
        if pages.iter().any(|page| page.bytes.is_empty()) {
            return Err(PersistentCacheError::ZeroSizedPage);
        }

        let prefix = &prompt[..reusable];
        let raw_identity = identity.fingerprint().0;
        let raw_tokens = prefix_fingerprint(raw_identity, prefix);
        let identity_fingerprint = self.locator(b"identity", &[&raw_identity]);
        let token_fingerprint = self.locator(b"tokens", &[&raw_identity, &raw_tokens]);
        let plaintext = Zeroizing::new(encode_payload(identity, prefix, pages)?);
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        let plaintext_len =
            u64::try_from(plaintext.len()).map_err(|_| PersistentCacheError::RecordTooLarge)?;
        let ciphertext_len = plaintext_len
            .checked_add(TAG_LEN)
            .ok_or(PersistentCacheError::RecordTooLarge)?;
        let record_len = (HEADER_LEN as u64)
            .checked_add(ciphertext_len)
            .ok_or(PersistentCacheError::RecordTooLarge)?;
        if record_len > self.config.max_record_bytes {
            return Err(PersistentCacheError::RecordTooLarge);
        }

        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let aad = aad(
            identity_fingerprint,
            token_fingerprint,
            plaintext_len,
            ciphertext_len,
        );
        let ciphertext = self
            .cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: &plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| PersistentCacheError::AuthenticationFailed)?;
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        if ciphertext.len() as u64 != ciphertext_len {
            return Err(PersistentCacheError::CorruptRecord);
        }
        let ciphertext_checksum: [u8; 32] = Sha256::digest(&ciphertext).into();

        let namespace_name = self.namespace_name(&identity.namespace);
        let namespace_dir = self
            .storage_root()?
            .open_or_create_private_directory(&namespace_name)?;
        let _tenant_lock = TenantAdvisoryLock::acquire_cancellable(&namespace_dir, &cancelled)?;
        cleanup_stale_temporary_files(&namespace_dir)?;
        let destination = self.record_name(identity_fingerprint, token_fingerprint);
        self.enforce_quota(&namespace_dir, &destination, record_len)?;

        let temporary = format!(".{}.tmp", hex(&nonce));
        let write_result = (|| -> Result<(), PersistentCacheError> {
            if cancelled() {
                return Err(PersistentCacheError::Cancelled);
            }
            let mut file = namespace_dir.create_private_file(&temporary)?;
            file.write_all(MAGIC)?;
            file.write_all(&PERSISTENT_PROMPT_CACHE_FORMAT_VERSION.to_le_bytes())?;
            file.write_all(&nonce)?;
            file.write_all(&identity_fingerprint)?;
            file.write_all(&token_fingerprint)?;
            file.write_all(&plaintext_len.to_le_bytes())?;
            file.write_all(&ciphertext_len.to_le_bytes())?;
            file.write_all(&ciphertext_checksum)?;
            file.write_all(&ciphertext)?;
            file.sync_all()?;
            let metadata = file.metadata()?;
            #[cfg(unix)]
            if !metadata.is_file()
                || metadata.nlink() != 1
                || metadata.permissions().mode() & 0o777 != 0o600
            {
                return Err(PersistentCacheError::CorruptRecord);
            }
            #[cfg(not(unix))]
            if !metadata.is_file() {
                return Err(PersistentCacheError::CorruptRecord);
            }
            if let Some(gate) = publish_gate {
                gate.publish_if_current(&cancelled, || {
                    namespace_dir.rename(&temporary, &destination)
                })?;
                namespace_dir.sync()?;
            } else {
                if cancelled() {
                    return Err(PersistentCacheError::Cancelled);
                }
                namespace_dir.rename(&temporary, &destination)?;
                namespace_dir.sync()?;
            }
            Ok(())
        })();
        if write_result.is_err() {
            let _ = namespace_dir.unlink(&temporary);
        }
        write_result?;
        Ok(PersistentStoreOutcome::Stored { bytes: record_len })
    }

    pub fn restore_prompt(
        &self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        cost: RestoreCostEstimate,
    ) -> Result<PersistentLookup, PersistentCacheError> {
        self.restore_prompt_cancellable(identity, prompt, cost, || false)
    }

    pub fn restore_prompt_cancellable<C>(
        &self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        cost: RestoreCostEstimate,
        cancelled: C,
    ) -> Result<PersistentLookup, PersistentCacheError>
    where
        C: Fn() -> bool,
    {
        if !self.config.enabled {
            return Ok(PersistentLookup::Disabled);
        }
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        if !cost.should_restore() {
            return Ok(PersistentLookup::Recompute);
        }
        let reusable = reusable_prompt_tokens(prompt.len());
        if reusable == 0 {
            return Ok(PersistentLookup::Miss);
        }
        let prefix = &prompt[..reusable];
        let raw_identity = identity.fingerprint().0;
        let raw_tokens = prefix_fingerprint(raw_identity, prefix);
        let identity_fingerprint = self.locator(b"identity", &[&raw_identity]);
        let token_fingerprint = self.locator(b"tokens", &[&raw_identity, &raw_tokens]);
        let namespace = match self
            .storage_root()?
            .open_private_directory(&self.namespace_name(&identity.namespace))
        {
            Ok(directory) => directory,
            Err(PersistentCacheError::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                return Ok(PersistentLookup::Miss);
            }
            Err(error) => return Err(error),
        };
        let record_name = self.record_name(identity_fingerprint, token_fingerprint);
        let mut file = match namespace.open_regular_file(&record_name) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(PersistentLookup::Miss)
            }
            Err(error) => return Err(error.into()),
        };
        let metadata_len = file.metadata()?.len();
        if metadata_len < HEADER_LEN as u64
            || metadata_len > self.config.max_record_bytes
            || metadata_len > self.config.max_bytes_per_namespace
        {
            return Err(PersistentCacheError::CorruptRecord);
        }
        let mut header = [0_u8; HEADER_LEN];
        file.read_exact(&mut header)
            .map_err(|_| PersistentCacheError::CorruptRecord)?;
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        let parsed = ParsedHeader::parse(&header)?;
        if parsed.identity_fingerprint != identity_fingerprint
            || parsed.token_fingerprint != token_fingerprint
        {
            return Err(PersistentCacheError::IdentityMismatch);
        }
        if parsed.ciphertext_len != parsed.plaintext_len.checked_add(TAG_LEN).unwrap_or(0)
            || metadata_len != HEADER_LEN as u64 + parsed.ciphertext_len
        {
            return Err(PersistentCacheError::CorruptRecord);
        }
        let ciphertext_len = usize::try_from(parsed.ciphertext_len)
            .map_err(|_| PersistentCacheError::RecordTooLarge)?;
        let mut ciphertext = vec![0_u8; ciphertext_len];
        file.read_exact(&mut ciphertext)
            .map_err(|_| PersistentCacheError::CorruptRecord)?;
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        if <[u8; 32]>::from(Sha256::digest(&ciphertext)) != parsed.ciphertext_checksum {
            return Err(PersistentCacheError::CorruptRecord);
        }
        let aad = aad(
            parsed.identity_fingerprint,
            parsed.token_fingerprint,
            parsed.plaintext_len,
            parsed.ciphertext_len,
        );
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        let plaintext = Zeroizing::new(
            self.cipher
                .decrypt(
                    Nonce::from_slice(&parsed.nonce),
                    Payload {
                        msg: &ciphertext,
                        aad: &aad,
                    },
                )
                .map_err(|_| PersistentCacheError::AuthenticationFailed)?,
        );
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        if plaintext.len() as u64 != parsed.plaintext_len {
            return Err(PersistentCacheError::CorruptRecord);
        }
        let decoded = decode_payload(&plaintext)?;
        if &decoded.identity != identity || decoded.tokens != prefix {
            return Err(PersistentCacheError::IdentityMismatch);
        }
        if decoded.pages.len() != reusable / PROMPT_CACHE_PAGE_TOKENS {
            return Err(PersistentCacheError::CorruptRecord);
        }
        Ok(PersistentLookup::Hit(PersistentCacheRecord {
            matched_tokens: reusable,
            pages: decoded.pages,
        }))
    }

    /// Remove a repeatedly failing exact record from the active lookup path.
    /// The renamed file remains tenant-local for diagnostics/quota cleanup and
    /// can never be interpreted as a valid cache record.
    pub fn quarantine_prompt(
        &self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
    ) -> Result<(), PersistentCacheError> {
        self.quarantine_prompt_cancellable(identity, prompt, || false)
    }

    pub fn quarantine_prompt_cancellable<C>(
        &self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        cancelled: C,
    ) -> Result<(), PersistentCacheError>
    where
        C: Fn() -> bool,
    {
        if !self.config.enabled {
            return Ok(());
        }
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        let reusable = reusable_prompt_tokens(prompt.len());
        if reusable == 0 {
            return Ok(());
        }
        let raw_identity = identity.fingerprint().0;
        let raw_tokens = prefix_fingerprint(raw_identity, &prompt[..reusable]);
        let identity_fingerprint = self.locator(b"identity", &[&raw_identity]);
        let token_fingerprint = self.locator(b"tokens", &[&raw_identity, &raw_tokens]);
        let namespace = match self
            .storage_root()?
            .open_private_directory(&self.namespace_name(&identity.namespace))
        {
            Ok(directory) => directory,
            Err(PersistentCacheError::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let _tenant_lock = match TenantAdvisoryLock::try_acquire(&namespace) {
            Ok(Some(lock)) => lock,
            Ok(None) => return Ok(()),
            Err(PersistentCacheError::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        cleanup_stale_temporary_files(&namespace)?;
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        let source = self.record_name(identity_fingerprint, token_fingerprint);
        match namespace.validate_regular_file(&source) {
            Ok(()) => {}
            Err(PersistentCacheError::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        namespace.unlink(&source)?;
        namespace.sync()
    }

    fn storage_root(&self) -> Result<&StorageRoot, PersistentCacheError> {
        self.storage_root
            .as_ref()
            .ok_or(PersistentCacheError::InvalidConfig)
    }

    fn namespace_name(&self, namespace: &CacheNamespace) -> String {
        let mut h = <Hmac<Sha256> as Mac>::new_from_slice(&self.path_key.0)
            .expect("HMAC accepts a 256-bit key");
        h.update(b"rvllm.prompt-cache.tenant-directory.v2");
        h.update(namespace.as_str().as_bytes());
        hex(&h.finalize().into_bytes())
    }

    fn record_name(&self, identity: [u8; 32], tokens: [u8; 32]) -> String {
        let mut h = <Hmac<Sha256> as Mac>::new_from_slice(&self.path_key.0)
            .expect("HMAC accepts a 256-bit key");
        h.update(b"rvllm.prompt-cache.record-path.v2");
        h.update(&identity);
        h.update(&tokens);
        format!("{}.{}", hex(&h.finalize().into_bytes()), RECORD_EXTENSION)
    }

    fn locator(&self, domain: &[u8], values: &[&[u8]]) -> [u8; 32] {
        let mut h = <Hmac<Sha256> as Mac>::new_from_slice(&self.path_key.0)
            .expect("HMAC accepts a 256-bit key");
        h.update(b"rvllm.prompt-cache.header-locator.v2");
        h.update(domain);
        for value in values {
            h.update(value);
        }
        h.finalize().into_bytes().into()
    }

    fn enforce_quota(
        &self,
        namespace_dir: &StorageDirectory,
        destination: &str,
        incoming_bytes: u64,
    ) -> Result<(), PersistentCacheError> {
        let mut used = 0_u64;
        for entry_name in namespace_dir.entry_names()? {
            if entry_name == TENANT_LOCK_FILE {
                continue;
            }
            let metadata = namespace_dir.regular_file_metadata(&entry_name)?;
            used = used
                .checked_add(metadata.len())
                .ok_or(PersistentCacheError::QuotaExceeded)?;
        }
        let replaced = match namespace_dir.regular_file_metadata(destination) {
            Ok(metadata) => metadata.len(),
            Err(PersistentCacheError::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                0
            }
            Err(error) => return Err(error),
        };
        let projected = used
            .saturating_sub(replaced)
            .checked_add(incoming_bytes)
            .ok_or(PersistentCacheError::QuotaExceeded)?;
        if projected > self.config.max_bytes_per_namespace {
            return Err(PersistentCacheError::QuotaExceeded);
        }
        Ok(())
    }
}

struct ParsedHeader {
    nonce: [u8; NONCE_LEN],
    identity_fingerprint: [u8; HASH_LEN],
    token_fingerprint: [u8; HASH_LEN],
    plaintext_len: u64,
    ciphertext_len: u64,
    ciphertext_checksum: [u8; HASH_LEN],
}

impl ParsedHeader {
    fn parse(bytes: &[u8; HEADER_LEN]) -> Result<Self, PersistentCacheError> {
        if &bytes[..8] != MAGIC {
            return Err(PersistentCacheError::CorruptRecord);
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != PERSISTENT_PROMPT_CACHE_FORMAT_VERSION {
            return Err(PersistentCacheError::CorruptRecord);
        }
        let mut nonce = [0; NONCE_LEN];
        nonce.copy_from_slice(&bytes[12..24]);
        let mut identity_fingerprint = [0; HASH_LEN];
        identity_fingerprint.copy_from_slice(&bytes[24..56]);
        let mut token_fingerprint = [0; HASH_LEN];
        token_fingerprint.copy_from_slice(&bytes[56..88]);
        let plaintext_len = u64::from_le_bytes(bytes[88..96].try_into().unwrap());
        let ciphertext_len = u64::from_le_bytes(bytes[96..104].try_into().unwrap());
        let mut ciphertext_checksum = [0; HASH_LEN];
        ciphertext_checksum.copy_from_slice(&bytes[104..136]);
        Ok(Self {
            nonce,
            identity_fingerprint,
            token_fingerprint,
            plaintext_len,
            ciphertext_len,
            ciphertext_checksum,
        })
    }
}

struct DecodedPayload {
    identity: CacheIdentity,
    tokens: Vec<TokenId>,
    pages: Vec<WarmPage>,
}

fn encode_payload(
    identity: &CacheIdentity,
    tokens: &[TokenId],
    pages: &[WarmPage],
) -> Result<Vec<u8>, PersistentCacheError> {
    let mut out = Vec::new();
    out.extend_from_slice(b"rvllm.prompt-cache.payload.v1\0");
    put_bytes(&mut out, identity.namespace.as_str().as_bytes())?;
    out.extend_from_slice(&identity.model);
    out.extend_from_slice(&identity.tokenizer);
    match identity.adapter {
        Some(adapter) => {
            out.push(1);
            out.extend_from_slice(&adapter);
        }
        None => out.push(0),
    }
    out.extend_from_slice(&identity.kv_layout);
    out.extend_from_slice(&identity.numeric_path);
    out.extend_from_slice(&identity.format_version.to_le_bytes());
    put_u32(&mut out, tokens.len())?;
    for token in tokens {
        out.extend_from_slice(&token.raw().to_le_bytes());
    }
    put_u32(&mut out, pages.len())?;
    for page in pages {
        put_bytes(&mut out, &page.bytes)?;
    }
    let checksum: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&checksum);
    Ok(out)
}

fn decode_payload(bytes: &[u8]) -> Result<DecodedPayload, PersistentCacheError> {
    if bytes.len() < 32 {
        return Err(PersistentCacheError::CorruptRecord);
    }
    let split = bytes.len() - 32;
    if <[u8; 32]>::from(Sha256::digest(&bytes[..split])) != bytes[split..] {
        return Err(PersistentCacheError::CorruptRecord);
    }
    let mut reader = Reader::new(&bytes[..split]);
    reader.expect(b"rvllm.prompt-cache.payload.v1\0")?;
    let namespace = String::from_utf8(reader.bytes()?.to_vec())
        .map_err(|_| PersistentCacheError::CorruptRecord)?;
    let namespace =
        CacheNamespace::new(namespace).map_err(|_| PersistentCacheError::CorruptRecord)?;
    let model = reader.array()?;
    let tokenizer = reader.array()?;
    let adapter = match reader.u8()? {
        0 => None,
        1 => Some(reader.array()?),
        _ => return Err(PersistentCacheError::CorruptRecord),
    };
    let kv_layout = reader.array()?;
    let numeric_path = reader.array()?;
    let format_version = reader.u32()?;
    let token_count = reader.u32()? as usize;
    let mut tokens = Vec::with_capacity(token_count);
    for _ in 0..token_count {
        tokens.push(TokenId(reader.u32()?));
    }
    let page_count = reader.u32()? as usize;
    let mut pages = Vec::with_capacity(page_count);
    for _ in 0..page_count {
        let bytes = reader.bytes()?;
        if bytes.is_empty() {
            return Err(PersistentCacheError::CorruptRecord);
        }
        pages.push(WarmPage {
            bytes: Arc::from(bytes),
        });
    }
    if !reader.is_empty() {
        return Err(PersistentCacheError::CorruptRecord);
    }
    Ok(DecodedPayload {
        identity: CacheIdentity {
            namespace,
            model,
            tokenizer,
            adapter,
            kv_layout,
            numeric_path,
            format_version,
        },
        tokens,
        pages,
    })
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], PersistentCacheError> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or(PersistentCacheError::CorruptRecord)?;
        let value = self
            .bytes
            .get(self.cursor..end)
            .ok_or(PersistentCacheError::CorruptRecord)?;
        self.cursor = end;
        Ok(value)
    }

    fn expect(&mut self, expected: &[u8]) -> Result<(), PersistentCacheError> {
        if self.take(expected.len())? != expected {
            return Err(PersistentCacheError::CorruptRecord);
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, PersistentCacheError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, PersistentCacheError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], PersistentCacheError> {
        Ok(self.take(N)?.try_into().unwrap())
    }

    fn bytes(&mut self) -> Result<&'a [u8], PersistentCacheError> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    fn is_empty(&self) -> bool {
        self.cursor == self.bytes.len()
    }
}

fn put_u32(out: &mut Vec<u8>, value: usize) -> Result<(), PersistentCacheError> {
    let value = u32::try_from(value).map_err(|_| PersistentCacheError::RecordTooLarge)?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), PersistentCacheError> {
    put_u32(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

fn prefix_fingerprint(identity: [u8; 32], tokens: &[TokenId]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"rvllm.prompt-cache.prompt-prefix.v1");
    h.update(identity);
    h.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        h.update(token.raw().to_le_bytes());
    }
    h.finalize().into()
}

fn aad(identity: [u8; 32], tokens: [u8; 32], plaintext_len: u64, ciphertext_len: u64) -> Vec<u8> {
    let mut result = Vec::with_capacity(8 + 4 + 32 + 32 + 8 + 8);
    result.extend_from_slice(MAGIC);
    result.extend_from_slice(&PERSISTENT_PROMPT_CACHE_FORMAT_VERSION.to_le_bytes());
    result.extend_from_slice(&identity);
    result.extend_from_slice(&tokens);
    result.extend_from_slice(&plaintext_len.to_le_bytes());
    result.extend_from_slice(&ciphertext_len.to_le_bytes());
    result
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    result
}

fn validate_cache_root(path: &Path) -> Result<(), PersistentCacheError> {
    StorageRoot::open(path).map(|_| ())
}

#[cfg(unix)]
struct StorageRoot {
    directory: StorageDirectory,
}

#[cfg(unix)]
impl StorageRoot {
    fn open(path: &Path) -> Result<Self, PersistentCacheError> {
        if !path.is_absolute() {
            return Err(PersistentCacheError::InvalidConfig);
        }
        let slash = CStr::from_bytes_with_nul(b"/\0").expect("static root path is valid");
        let root_fd = loop {
            let fd = unsafe {
                libc::open(
                    slash.as_ptr(),
                    libc::O_RDONLY
                        | libc::O_DIRECTORY
                        | libc::O_NOFOLLOW
                        | libc::O_CLOEXEC
                        | libc::O_NONBLOCK,
                )
            };
            if fd >= 0 {
                break fd;
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        };
        let mut file = unsafe { File::from_raw_fd(root_fd) };
        for component in path.components() {
            match component {
                Component::RootDir => {}
                Component::Normal(component) => {
                    let component = CString::new(component.as_bytes())
                        .map_err(|_| PersistentCacheError::InvalidConfig)?;
                    let fd = openat_retry(
                        file.as_raw_fd(),
                        &component,
                        libc::O_RDONLY
                            | libc::O_DIRECTORY
                            | libc::O_NOFOLLOW
                            | libc::O_CLOEXEC
                            | libc::O_NONBLOCK,
                        0,
                    )?;
                    let next = unsafe { File::from_raw_fd(fd) };
                    if !next.metadata()?.is_dir() {
                        return Err(PersistentCacheError::InvalidConfig);
                    }
                    file = next;
                }
                Component::CurDir | Component::ParentDir | Component::Prefix(_) => {
                    return Err(PersistentCacheError::InvalidConfig);
                }
            }
        }
        validate_private_directory_file(&file, PersistentCacheError::InvalidConfig)?;
        Ok(Self {
            directory: StorageDirectory { file },
        })
    }

    fn open_or_create_private_directory(
        &self,
        name: &str,
    ) -> Result<StorageDirectory, PersistentCacheError> {
        let name = c_name(name)?;
        let created = loop {
            let result = unsafe {
                libc::mkdirat(
                    self.directory.file.as_raw_fd(),
                    name.as_ptr(),
                    libc::S_IRWXU,
                )
            };
            if result == 0 {
                break true;
            }
            let error = std::io::Error::last_os_error();
            match error.kind() {
                std::io::ErrorKind::Interrupted => continue,
                std::io::ErrorKind::AlreadyExists => break false,
                _ => return Err(error.into()),
            }
        };
        let directory = self.open_private_directory_cstr(&name)?;
        if created {
            let result = unsafe { libc::fchmod(directory.file.as_raw_fd(), libc::S_IRWXU) };
            if result != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            directory.sync()?;
            self.directory.sync()?;
        }
        validate_private_directory_file(&directory.file, PersistentCacheError::CorruptRecord)?;
        Ok(directory)
    }

    fn open_private_directory(&self, name: &str) -> Result<StorageDirectory, PersistentCacheError> {
        self.open_private_directory_cstr(&c_name(name)?)
    }

    fn open_private_directory_cstr(
        &self,
        name: &CStr,
    ) -> Result<StorageDirectory, PersistentCacheError> {
        let fd = openat_retry(
            self.directory.file.as_raw_fd(),
            name,
            libc::O_RDONLY
                | libc::O_DIRECTORY
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC
                | libc::O_NONBLOCK,
            0,
        )?;
        let file = unsafe { File::from_raw_fd(fd) };
        validate_private_directory_file(&file, PersistentCacheError::CorruptRecord)?;
        Ok(StorageDirectory { file })
    }
}

#[cfg(unix)]
struct StorageDirectory {
    file: File,
}

#[cfg(unix)]
impl StorageDirectory {
    #[cfg(test)]
    fn open_path(path: &Path) -> Result<Self, PersistentCacheError> {
        Ok(StorageRoot::open(&fs::canonicalize(path)?)?.directory)
    }

    fn create_private_file(&self, name: &str) -> Result<File, PersistentCacheError> {
        let fd = openat_retry(
            self.file.as_raw_fd(),
            &c_name(name)?,
            libc::O_WRONLY
                | libc::O_CREAT
                | libc::O_EXCL
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC
                | libc::O_NONBLOCK,
            libc::S_IRUSR | libc::S_IWUSR,
        )?;
        let file = unsafe { File::from_raw_fd(fd) };
        let result = unsafe { libc::fchmod(file.as_raw_fd(), libc::S_IRUSR | libc::S_IWUSR) };
        if result != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        validate_private_regular_file(&file)?;
        Ok(file)
    }

    fn open_regular_file(&self, name: &str) -> Result<File, std::io::Error> {
        let name = CString::new(name.as_bytes()).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid cache entry name")
        })?;
        let fd = openat_retry_io(
            self.file.as_raw_fd(),
            &name,
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
            0,
        )?;
        let file = unsafe { File::from_raw_fd(fd) };
        validate_private_regular_file_io(&file)?;
        Ok(file)
    }

    fn validate_regular_file(&self, name: &str) -> Result<(), PersistentCacheError> {
        self.open_regular_file(name).map(drop).map_err(Into::into)
    }

    fn regular_file_metadata(&self, name: &str) -> Result<fs::Metadata, PersistentCacheError> {
        Ok(self.open_regular_file(name)?.metadata()?)
    }

    fn rename(&self, source: &str, destination: &str) -> Result<(), PersistentCacheError> {
        let source = c_name(source)?;
        let destination = c_name(destination)?;
        loop {
            let result = unsafe {
                libc::renameat(
                    self.file.as_raw_fd(),
                    source.as_ptr(),
                    self.file.as_raw_fd(),
                    destination.as_ptr(),
                )
            };
            if result == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
    }

    fn unlink(&self, name: &str) -> Result<(), PersistentCacheError> {
        let name = c_name(name)?;
        loop {
            let result = unsafe { libc::unlinkat(self.file.as_raw_fd(), name.as_ptr(), 0) };
            if result == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
    }

    fn entry_names(&self) -> Result<Vec<String>, PersistentCacheError> {
        // `dup` would share the directory offset with the pinned descriptor.
        // Opening "." relative to it yields a fresh open file description while
        // remaining anchored to the exact same directory inode.
        let dot = CStr::from_bytes_with_nul(b".\0").expect("static directory name is valid");
        let iterator_fd = openat_retry(
            self.file.as_raw_fd(),
            dot,
            libc::O_RDONLY
                | libc::O_DIRECTORY
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC
                | libc::O_NONBLOCK,
            0,
        )?;
        let raw_stream = unsafe { libc::fdopendir(iterator_fd) };
        if raw_stream.is_null() {
            let error = std::io::Error::last_os_error();
            unsafe {
                libc::close(iterator_fd);
            }
            return Err(error.into());
        }
        let stream = DirectoryStream(raw_stream);
        let mut names = Vec::new();
        loop {
            set_errno(0);
            let entry = unsafe { libc::readdir(stream.0) };
            if entry.is_null() {
                let error = get_errno();
                return if error == 0 {
                    Ok(names)
                } else {
                    Err(std::io::Error::from_raw_os_error(error).into())
                };
            }
            let bytes = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
            if bytes == b"." || bytes == b".." {
                continue;
            }
            let name = OsString::from_vec(bytes.to_vec())
                .into_string()
                .map_err(|_| PersistentCacheError::CorruptRecord)?;
            names.push(name);
        }
    }

    fn sync(&self) -> Result<(), PersistentCacheError> {
        self.file.sync_all()?;
        Ok(())
    }
}

#[cfg(unix)]
struct DirectoryStream(*mut libc::DIR);

#[cfg(unix)]
impl Drop for DirectoryStream {
    fn drop(&mut self) {
        unsafe {
            libc::closedir(self.0);
        }
    }
}

#[cfg(unix)]
fn c_name(name: &str) -> Result<CString, PersistentCacheError> {
    if name.is_empty() || name == "." || name == ".." || name.as_bytes().contains(&b'/') {
        return Err(PersistentCacheError::CorruptRecord);
    }
    CString::new(name.as_bytes()).map_err(|_| PersistentCacheError::CorruptRecord)
}

#[cfg(unix)]
fn openat_retry(
    directory: RawFd,
    name: &CStr,
    flags: libc::c_int,
    mode: libc::mode_t,
) -> Result<RawFd, PersistentCacheError> {
    openat_retry_io(directory, name, flags, mode).map_err(Into::into)
}

#[cfg(unix)]
fn openat_retry_io(
    directory: RawFd,
    name: &CStr,
    flags: libc::c_int,
    mode: libc::mode_t,
) -> Result<RawFd, std::io::Error> {
    loop {
        let fd = unsafe { libc::openat(directory, name.as_ptr(), flags, libc::c_uint::from(mode)) };
        if fd >= 0 {
            return Ok(fd);
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[cfg(unix)]
fn validate_private_directory_file(
    file: &File,
    invalid: PersistentCacheError,
) -> Result<(), PersistentCacheError> {
    let metadata = file.metadata()?;
    if !metadata.is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.permissions().mode() & 0o777 != 0o700
    {
        return Err(invalid);
    }
    Ok(())
}

#[cfg(unix)]
fn validate_private_regular_file(file: &File) -> Result<(), PersistentCacheError> {
    validate_private_regular_file_io(file).map_err(Into::into)
}

#[cfg(unix)]
fn validate_private_regular_file_io(file: &File) -> Result<(), std::io::Error> {
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.permissions().mode() & 0o777 != 0o600
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "persistent cache entry is not a private single-link regular file",
        ));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn set_errno(value: libc::c_int) {
    unsafe {
        *libc::__error() = value;
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn get_errno() -> libc::c_int {
    unsafe { *libc::__error() }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn set_errno(value: libc::c_int) {
    unsafe {
        *libc::__errno_location() = value;
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn get_errno() -> libc::c_int {
    unsafe { *libc::__errno_location() }
}

#[cfg(all(
    unix,
    not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android"
    ))
))]
compile_error!("persistent T3 dirfd storage needs an errno accessor for this Unix target");

#[cfg(not(unix))]
struct StorageRoot {
    path: PathBuf,
}

#[cfg(not(unix))]
impl StorageRoot {
    fn open(path: &Path) -> Result<Self, PersistentCacheError> {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(PersistentCacheError::InvalidConfig);
        }
        Ok(Self {
            path: path.to_owned(),
        })
    }

    fn open_or_create_private_directory(
        &self,
        name: &str,
    ) -> Result<StorageDirectory, PersistentCacheError> {
        let path = self.path.join(name);
        match fs::create_dir(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        self.open_private_directory(name)
    }

    fn open_private_directory(&self, name: &str) -> Result<StorageDirectory, PersistentCacheError> {
        let path = self.path.join(name);
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(PersistentCacheError::CorruptRecord);
        }
        Ok(StorageDirectory { path })
    }
}

#[cfg(not(unix))]
struct StorageDirectory {
    path: PathBuf,
}

#[cfg(not(unix))]
impl StorageDirectory {
    fn create_private_file(&self, name: &str) -> Result<File, PersistentCacheError> {
        Ok(OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.path.join(name))?)
    }

    fn open_regular_file(&self, name: &str) -> Result<File, std::io::Error> {
        let file = File::open(self.path.join(name))?;
        if !file.metadata()?.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "persistent cache entry is not a regular file",
            ));
        }
        Ok(file)
    }

    fn validate_regular_file(&self, name: &str) -> Result<(), PersistentCacheError> {
        self.open_regular_file(name).map(drop).map_err(Into::into)
    }

    fn regular_file_metadata(&self, name: &str) -> Result<fs::Metadata, PersistentCacheError> {
        Ok(self.open_regular_file(name)?.metadata()?)
    }

    fn rename(&self, source: &str, destination: &str) -> Result<(), PersistentCacheError> {
        fs::rename(self.path.join(source), self.path.join(destination))?;
        Ok(())
    }

    fn unlink(&self, name: &str) -> Result<(), PersistentCacheError> {
        fs::remove_file(self.path.join(name))?;
        Ok(())
    }

    fn entry_names(&self) -> Result<Vec<String>, PersistentCacheError> {
        fs::read_dir(&self.path)?
            .map(|entry| {
                entry?
                    .file_name()
                    .into_string()
                    .map_err(|_| PersistentCacheError::CorruptRecord)
            })
            .collect()
    }

    fn sync(&self) -> Result<(), PersistentCacheError> {
        File::open(&self.path)?.sync_all()?;
        Ok(())
    }
}

struct TenantAdvisoryLock {
    file: File,
}

impl TenantAdvisoryLock {
    #[cfg(test)]
    fn acquire(namespace_dir: &StorageDirectory) -> Result<Self, PersistentCacheError> {
        Self::acquire_cancellable(namespace_dir, || false)
    }

    #[cfg(unix)]
    fn open_validated(namespace_dir: &StorageDirectory) -> Result<File, PersistentCacheError> {
        let name = c_name(TENANT_LOCK_FILE)?;
        let base_flags = libc::O_RDWR | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;
        let (fd, created) = match openat_retry_io(
            namespace_dir.file.as_raw_fd(),
            &name,
            base_flags | libc::O_CREAT | libc::O_EXCL,
            libc::S_IRUSR | libc::S_IWUSR,
        ) {
            Ok(fd) => (fd, true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (
                openat_retry(namespace_dir.file.as_raw_fd(), &name, base_flags, 0)?,
                false,
            ),
            Err(error) => return Err(error.into()),
        };
        let file = unsafe { File::from_raw_fd(fd) };
        if created {
            let result = unsafe { libc::fchmod(file.as_raw_fd(), libc::S_IRUSR | libc::S_IWUSR) };
            if result != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            file.sync_all()?;
            namespace_dir.sync()?;
        }
        validate_private_regular_file(&file)?;
        Ok(file)
    }

    #[cfg(not(unix))]
    fn open_validated(namespace_dir: &StorageDirectory) -> Result<File, PersistentCacheError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(namespace_dir.path.join(TENANT_LOCK_FILE))?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(PersistentCacheError::CorruptRecord);
        }
        Ok(file)
    }

    fn try_acquire(namespace_dir: &StorageDirectory) -> Result<Option<Self>, PersistentCacheError> {
        let file = Self::open_validated(namespace_dir)?;
        #[cfg(unix)]
        loop {
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if result == 0 {
                return Ok(Some(Self { file }));
            }
            let error = std::io::Error::last_os_error();
            match error.kind() {
                std::io::ErrorKind::Interrupted => continue,
                std::io::ErrorKind::WouldBlock => return Ok(None),
                _ => return Err(error.into()),
            }
        }
        #[cfg(not(unix))]
        Ok(Some(Self { file }))
    }

    fn acquire_cancellable<C>(
        namespace_dir: &StorageDirectory,
        cancelled: C,
    ) -> Result<Self, PersistentCacheError>
    where
        C: Fn() -> bool,
    {
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        let file = Self::open_validated(namespace_dir)?;
        #[cfg(unix)]
        loop {
            if cancelled() {
                return Err(PersistentCacheError::Cancelled);
            }
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if result == 0 {
                break;
            }
            let error = std::io::Error::last_os_error();
            match error.kind() {
                std::io::ErrorKind::Interrupted => continue,
                std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                _ => return Err(error.into()),
            }
        }
        let acquired = Self { file };
        if cancelled() {
            return Err(PersistentCacheError::Cancelled);
        }
        Ok(acquired)
    }
}

impl Drop for TenantAdvisoryLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn cleanup_stale_temporary_files(
    namespace_dir: &StorageDirectory,
) -> Result<(), PersistentCacheError> {
    let mut removed = false;
    for file_name in namespace_dir.entry_names()? {
        if !is_temporary_record_name(&file_name) {
            continue;
        }
        namespace_dir.validate_regular_file(&file_name)?;
        namespace_dir.unlink(&file_name)?;
        removed = true;
    }
    if removed {
        namespace_dir.sync()?;
    }
    Ok(())
}

fn is_temporary_record_name(file_name: &str) -> bool {
    let Some(hex_nonce) = file_name
        .strip_prefix('.')
        .and_then(|name| name.strip_suffix(".tmp"))
    else {
        return false;
    };
    hex_nonce.len() == NONCE_LEN * 2
        && hex_nonce
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::ffi::OsStr;
    use std::io::{Seek, SeekFrom};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    const FAST_RESTORE: RestoreCostEstimate = RestoreCostEstimate {
        measured_restore_ns: 10,
        measured_recompute_ns: 100,
    };

    fn config(path: &Path) -> PersistentPromptCacheConfig {
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        PersistentPromptCacheConfig {
            enabled: true,
            directory: fs::canonicalize(path).unwrap(),
            max_bytes_per_namespace: 1024 * 1024,
            max_record_bytes: 512 * 1024,
        }
    }

    fn identity(namespace: &str) -> CacheIdentity {
        CacheIdentity {
            namespace: CacheNamespace::new(namespace).unwrap(),
            model: [1; 32],
            tokenizer: [2; 32],
            adapter: Some([3; 32]),
            kv_layout: [4; 32],
            numeric_path: [5; 32],
            format_version: 7,
        }
    }

    fn prompt() -> Vec<TokenId> {
        (0..65).map(TokenId).collect()
    }

    fn pages() -> Vec<WarmPage> {
        vec![
            WarmPage {
                bytes: Arc::from([11_u8; 64]),
            },
            WarmPage {
                bytes: Arc::from([22_u8; 64]),
            },
        ]
    }

    fn host(path: &Path, key: Option<[u8; 32]>) -> PersistentCacheHostConfig {
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        PersistentCacheHostConfig {
            consent: true,
            root: fs::canonicalize(path).unwrap(),
            quota_bytes: 1024 * 1024,
            max_record_bytes: 512 * 1024,
            cache_namespace: "tenant-a".to_owned(),
            key: key.map(PersistentCacheKey::new),
        }
    }

    fn sole_record(root: &Path) -> PathBuf {
        let tenant = fs::read_dir(root).unwrap().next().unwrap().unwrap().path();
        fs::read_dir(tenant)
            .unwrap()
            .map(Result::unwrap)
            .find(|entry| {
                entry.path().extension().and_then(|v| v.to_str()) == Some(RECORD_EXTENSION)
            })
            .unwrap()
            .path()
    }

    #[test]
    fn disabled_by_default_and_does_not_touch_disk() {
        let temp = tempfile::tempdir().unwrap();
        let store =
            PersistentPromptCache::new(PersistentPromptCacheConfig::default(), [9; 32]).unwrap();
        assert_eq!(
            store
                .store_prompt(&identity("a"), &prompt(), &pages())
                .unwrap(),
            PersistentStoreOutcome::Disabled
        );
        assert_eq!(
            store
                .restore_prompt(&identity("a"), &prompt(), FAST_RESTORE)
                .unwrap(),
            PersistentLookup::Disabled
        );
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
    }

    #[test]
    fn host_config_requires_key_root_quota_consent_and_redacts_secret() {
        let temp = tempfile::tempdir().unwrap();
        let missing = host(temp.path(), None);
        assert!(matches!(
            missing.validate(),
            Err(PersistentCacheError::MissingKey)
        ));

        let valid = host(temp.path(), Some([0x5a; 32]));
        let debug = format!("{valid:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("5a"));
        valid.validate().unwrap();
    }

    #[test]
    fn encrypted_roundtrip_and_restore_cost_gate() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        store
            .store_prompt(&identity("a"), &prompt(), &pages())
            .unwrap();
        assert_eq!(
            store
                .restore_prompt(
                    &identity("a"),
                    &prompt(),
                    RestoreCostEstimate {
                        measured_restore_ns: 100,
                        measured_recompute_ns: 100
                    }
                )
                .unwrap(),
            PersistentLookup::Recompute
        );
        let PersistentLookup::Hit(hit) = store
            .restore_prompt(&identity("a"), &prompt(), FAST_RESTORE)
            .unwrap()
        else {
            panic!("expected hit")
        };
        assert_eq!(hit.matched_tokens, 64);
        assert_eq!(hit.pages, pages());
    }

    #[cfg(unix)]
    #[test]
    fn replacing_configured_root_cannot_redirect_store_or_restore() {
        let temp = tempfile::tempdir().unwrap();
        let configured_root = temp.path().join("cache");
        fs::create_dir(&configured_root).unwrap();
        fs::set_permissions(&configured_root, fs::Permissions::from_mode(0o700)).unwrap();
        let store = PersistentPromptCache::new(config(&configured_root), [9; 32]).unwrap();

        let pinned_root = temp.path().join("cache-pinned");
        fs::rename(&configured_root, &pinned_root).unwrap();
        fs::create_dir(&configured_root).unwrap();
        fs::set_permissions(&configured_root, fs::Permissions::from_mode(0o700)).unwrap();

        store
            .store_prompt(&identity("a"), &prompt(), &pages())
            .unwrap();
        assert_eq!(
            fs::read_dir(&configured_root).unwrap().count(),
            0,
            "a replacement at the configured path must receive no cache data"
        );
        assert_eq!(fs::read_dir(&pinned_root).unwrap().count(), 1);
        assert!(matches!(
            store
                .restore_prompt(&identity("a"), &prompt(), FAST_RESTORE)
                .unwrap(),
            PersistentLookup::Hit(_)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn cache_root_rejects_symlinked_ancestor_components() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let canonical_temp = fs::canonicalize(temp.path()).unwrap();
        let real_parent = canonical_temp.join("real");
        let root = real_parent.join("cache");
        fs::create_dir(&real_parent).unwrap();
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        let linked_parent = canonical_temp.join("linked");
        symlink(&real_parent, &linked_parent).unwrap();

        assert!(PersistentPromptCache::new(
            PersistentPromptCacheConfig {
                enabled: true,
                directory: linked_parent.join("cache"),
                max_bytes_per_namespace: 1024 * 1024,
                max_record_bytes: 512 * 1024,
            },
            [9; 32]
        )
        .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn replacing_tenant_while_lock_waits_cannot_redirect_publication() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        let cache_identity = identity("a");
        store
            .store_prompt(&cache_identity, &prompt(), &pages())
            .unwrap();
        let tenant = fs::read_dir(temp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let pinned_tenant = StorageDirectory::open_path(&tenant).unwrap();
        let tenant_lock = TenantAdvisoryLock::acquire(&pinned_tenant).unwrap();

        let mut second_prompt = prompt();
        second_prompt[0] = TokenId(867_530);
        let cancellation_polls = Arc::new(AtomicU64::new(0));
        let (result_tx, result_rx) = mpsc::channel();
        thread::scope(|scope| {
            let polls = Arc::clone(&cancellation_polls);
            let store = &store;
            let cache_identity = &cache_identity;
            let second_prompt = &second_prompt;
            scope.spawn(move || {
                result_tx
                    .send(store.store_prompt_cancellable(
                        cache_identity,
                        second_prompt,
                        &pages(),
                        || {
                            polls.fetch_add(1, Ordering::AcqRel);
                            false
                        },
                    ))
                    .unwrap();
            });

            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while cancellation_polls.load(Ordering::Acquire) < 5 {
                assert!(
                    std::time::Instant::now() < deadline,
                    "store did not reach the contended tenant lock"
                );
                thread::yield_now();
            }

            let detached_tenant = temp.path().join("detached-tenant");
            fs::rename(&tenant, &detached_tenant).unwrap();
            fs::create_dir(&tenant).unwrap();
            fs::set_permissions(&tenant, fs::Permissions::from_mode(0o700)).unwrap();
            drop(tenant_lock);

            assert!(matches!(
                result_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
                Ok(PersistentStoreOutcome::Stored { .. })
            ));
            assert_eq!(
                fs::read_dir(&tenant).unwrap().count(),
                0,
                "an in-flight store must not publish into a replacement tenant"
            );
            assert!(
                fs::read_dir(&detached_tenant)
                    .unwrap()
                    .map(Result::unwrap)
                    .filter(|entry| {
                        entry.path().extension() == Some(OsStr::new(RECORD_EXTENSION))
                    })
                    .count()
                    >= 2
            );
        });
    }

    #[test]
    fn record_header_and_name_do_not_expose_raw_identity_or_token_digests() {
        let temp = tempfile::tempdir().unwrap();
        let identity = identity("a");
        let prompt = prompt();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        store.store_prompt(&identity, &prompt, &pages()).unwrap();
        let record = sole_record(temp.path());
        let bytes = fs::read(&record).unwrap();
        let raw_identity = identity.fingerprint().0;
        let raw_tokens = prefix_fingerprint(
            raw_identity,
            &prompt[..reusable_prompt_tokens(prompt.len())],
        );
        assert!(!bytes[..HEADER_LEN]
            .windows(raw_identity.len())
            .any(|window| window == raw_identity));
        assert!(!bytes[..HEADER_LEN]
            .windows(raw_tokens.len())
            .any(|window| window == raw_tokens));
        let name = record.file_name().unwrap().to_string_lossy();
        assert!(!name.contains(&hex(&raw_identity)));
        assert!(!name.contains(&hex(&raw_tokens)));
    }

    #[cfg(unix)]
    #[test]
    fn record_reads_reject_hardlinks_and_symlinks_without_blocking() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        store
            .store_prompt(&identity("a"), &prompt(), &pages())
            .unwrap();
        let record = sole_record(temp.path());
        let metadata = fs::symlink_metadata(&record).unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);

        let hardlink = record.with_extension("hardlink");
        fs::hard_link(&record, &hardlink).unwrap();
        assert!(matches!(
            store.restore_prompt(&identity("a"), &prompt(), FAST_RESTORE),
            Err(PersistentCacheError::Io(_))
        ));
        fs::remove_file(&hardlink).unwrap();

        let target = record.with_extension("target");
        fs::rename(&record, &target).unwrap();
        symlink(&target, &record).unwrap();
        assert!(matches!(
            store.restore_prompt(&identity("a"), &prompt(), FAST_RESTORE),
            Err(PersistentCacheError::Io(_))
        ));

        fs::remove_file(&record).unwrap();
        assert!(std::process::Command::new("mkfifo")
            .arg(&record)
            .status()
            .unwrap()
            .success());
        assert!(matches!(
            store.restore_prompt(&identity("a"), &prompt(), FAST_RESTORE),
            Err(PersistentCacheError::Io(_))
        ));
    }

    #[test]
    fn wrong_key_cannot_locate_or_authenticate_records() {
        let temp = tempfile::tempdir().unwrap();
        PersistentPromptCache::new(config(temp.path()), [9; 32])
            .unwrap()
            .store_prompt(&identity("a"), &prompt(), &pages())
            .unwrap();
        let wrong = PersistentPromptCache::new(config(temp.path()), [8; 32]).unwrap();
        assert_eq!(
            wrong
                .restore_prompt(&identity("a"), &prompt(), FAST_RESTORE)
                .unwrap(),
            PersistentLookup::Miss
        );
    }

    #[test]
    fn corruption_and_partial_records_are_rejected() {
        for truncate in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
            store
                .store_prompt(&identity("a"), &prompt(), &pages())
                .unwrap();
            let path = sole_record(temp.path());
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .unwrap();
            if truncate {
                file.set_len(HEADER_LEN as u64 + 3).unwrap();
            } else {
                file.seek(SeekFrom::End(-1)).unwrap();
                let mut byte = [0_u8; 1];
                file.read_exact(&mut byte).unwrap();
                file.seek(SeekFrom::End(-1)).unwrap();
                file.write_all(&[byte[0] ^ 0x80]).unwrap();
            }
            assert!(matches!(
                store.restore_prompt(&identity("a"), &prompt(), FAST_RESTORE),
                Err(PersistentCacheError::CorruptRecord)
            ));
        }
    }

    #[test]
    fn quarantined_corruption_becomes_an_exact_miss() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        store
            .store_prompt(&identity("a"), &prompt(), &pages())
            .unwrap();
        let path = sole_record(temp.path());
        let mut file = OpenOptions::new().write(true).open(path).unwrap();
        file.write_all(b"broken").unwrap();
        file.sync_all().unwrap();
        assert!(matches!(
            store.restore_prompt(&identity("a"), &prompt(), FAST_RESTORE),
            Err(PersistentCacheError::CorruptRecord)
        ));
        store.quarantine_prompt(&identity("a"), &prompt()).unwrap();
        assert_eq!(
            store
                .restore_prompt(&identity("a"), &prompt(), FAST_RESTORE)
                .unwrap(),
            PersistentLookup::Miss
        );
    }

    #[test]
    fn cancelled_store_never_publishes_a_record() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        assert!(matches!(
            store.store_prompt_cancellable(&identity("a"), &prompt(), &pages(), || true),
            Err(PersistentCacheError::Cancelled)
        ));
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
    }

    #[test]
    fn cancellation_after_sync_removes_temporary_record_before_publish() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        let polls = Cell::new(0);
        assert!(matches!(
            store.store_prompt_cancellable(&identity("a"), &prompt(), &pages(), || {
                let next = polls.get() + 1;
                polls.set(next);
                next >= 5
            }),
            Err(PersistentCacheError::Cancelled)
        ));
        let tenant = fs::read_dir(temp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let entries = fs::read_dir(tenant)
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_name(), TENANT_LOCK_FILE);
    }

    #[test]
    fn epoch_invalidation_cannot_return_during_final_publication() {
        let gate = Arc::new(PersistentPublishGate::default());
        let epoch = Arc::new(AtomicU64::new(7));
        let publication_entered = Arc::new(AtomicBool::new(false));
        let (release_tx, release_rx) = mpsc::channel();
        let (published_tx, published_rx) = mpsc::channel();
        let (invalidated_tx, invalidated_rx) = mpsc::channel();

        thread::scope(|scope| {
            let publisher_gate = Arc::clone(&gate);
            let publisher_entered = Arc::clone(&publication_entered);
            scope.spawn(move || {
                publisher_gate
                    .publish_if_current(
                        || false,
                        || {
                            publisher_entered.store(true, Ordering::Release);
                            release_rx.recv().unwrap();
                            published_tx.send(()).unwrap();
                            Ok(())
                        },
                    )
                    .unwrap();
            });
            while !publication_entered.load(Ordering::Acquire) {
                thread::yield_now();
            }

            let invalidator_gate = Arc::clone(&gate);
            let invalidator_epoch = Arc::clone(&epoch);
            scope.spawn(move || {
                invalidator_gate.invalidate(|| {
                    invalidator_epoch.fetch_add(1, Ordering::AcqRel);
                });
                invalidated_tx.send(()).unwrap();
            });

            assert!(invalidated_rx
                .recv_timeout(Duration::from_millis(30))
                .is_err());
            release_tx.send(()).unwrap();
            published_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            invalidated_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        });
        assert_eq!(epoch.load(Ordering::Acquire), 8);
    }

    #[cfg(unix)]
    #[test]
    fn tenant_advisory_lock_serializes_independent_open_file_descriptions() {
        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let directory = StorageDirectory::open_path(temp.path()).unwrap();
        let first = TenantAdvisoryLock::acquire(&directory).unwrap();
        let (attempting_tx, attempting_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| {
                attempting_tx.send(()).unwrap();
                let directory = StorageDirectory::open_path(temp.path()).unwrap();
                let _second = TenantAdvisoryLock::acquire(&directory).unwrap();
                acquired_tx.send(()).unwrap();
            });
            attempting_rx.recv().unwrap();
            assert!(acquired_rx.recv_timeout(Duration::from_millis(30)).is_err());
            drop(first);
            acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        });
    }

    #[cfg(unix)]
    #[test]
    fn tenant_lock_wait_is_cancellation_safe() {
        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let directory = StorageDirectory::open_path(temp.path()).unwrap();
        let first = TenantAdvisoryLock::acquire(&directory).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (attempting_tx, attempting_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let tenant_path = temp.path();
        thread::scope(|scope| {
            let waiter_cancelled = Arc::clone(&cancelled);
            scope.spawn(move || {
                attempting_tx.send(()).unwrap();
                let directory = StorageDirectory::open_path(tenant_path).unwrap();
                let result = TenantAdvisoryLock::acquire_cancellable(&directory, || {
                    waiter_cancelled.load(Ordering::Acquire)
                });
                result_tx
                    .send(matches!(result, Err(PersistentCacheError::Cancelled)))
                    .unwrap();
            });
            attempting_rx.recv().unwrap();
            assert!(result_rx.recv_timeout(Duration::from_millis(30)).is_err());
            cancelled.store(true, Ordering::Release);
            assert!(
                result_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
                "lock waiter must abort without acquiring after epoch cancellation"
            );
        });
        drop(first);
    }

    #[cfg(unix)]
    #[test]
    fn best_effort_quarantine_never_waits_for_tenant_lock() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        let cache_identity = identity("a");
        let cache_prompt = prompt();
        store
            .store_prompt(&cache_identity, &cache_prompt, &pages())
            .unwrap();
        let tenant = fs::read_dir(temp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let tenant_directory = StorageDirectory::open_path(&tenant).unwrap();
        let lock = TenantAdvisoryLock::acquire(&tenant_directory).unwrap();
        let (result_tx, result_rx) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| {
                result_tx
                    .send(store.quarantine_prompt_cancellable(
                        &cache_identity,
                        &cache_prompt,
                        || false,
                    ))
                    .unwrap();
            });
            result_rx
                .recv_timeout(Duration::from_millis(100))
                .expect("best-effort quarantine must not wait for a live lock")
                .unwrap();
        });
        drop(lock);
        assert!(fs::read_dir(&tenant)
            .unwrap()
            .map(Result::unwrap)
            .any(|entry| entry.path().extension() == Some(OsStr::new(RECORD_EXTENSION))));
        store
            .quarantine_prompt(&cache_identity, &cache_prompt)
            .unwrap();
        assert!(!fs::read_dir(&tenant)
            .unwrap()
            .map(Result::unwrap)
            .any(|entry| entry.path().extension() == Some(OsStr::new(RECORD_EXTENSION))));
    }

    #[test]
    fn stale_well_formed_temporary_record_is_removed_under_tenant_lock() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        store
            .store_prompt(&identity("a"), &prompt(), &pages())
            .unwrap();
        let tenant = fs::read_dir(temp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let stale = tenant.join(".000102030405060708090a0b.tmp");
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        options.open(&stale).unwrap().write_all(b"partial").unwrap();

        let mut replacement_prompt = prompt();
        replacement_prompt[0] = TokenId(404);
        store
            .store_prompt(&identity("a"), &replacement_prompt, &pages())
            .unwrap();
        assert!(!stale.exists());
    }

    #[test]
    fn quota_is_enforced_per_namespace() {
        let temp = tempfile::tempdir().unwrap();
        let sizing = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        let PersistentStoreOutcome::Stored { bytes } = sizing
            .store_prompt(&identity("sizing"), &prompt(), &pages())
            .unwrap()
        else {
            panic!("expected sizing record")
        };
        let quota_root = temp.path().join("quota");
        fs::create_dir(&quota_root).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&quota_root, fs::Permissions::from_mode(0o700)).unwrap();
        let store = PersistentPromptCache::new(
            PersistentPromptCacheConfig {
                enabled: true,
                directory: fs::canonicalize(quota_root).unwrap(),
                max_bytes_per_namespace: bytes + 16,
                max_record_bytes: bytes,
            },
            [9; 32],
        )
        .unwrap();
        store
            .store_prompt(&identity("a"), &prompt(), &pages())
            .unwrap();
        let mut second_prompt = prompt();
        second_prompt[0] = TokenId(999);
        assert!(matches!(
            store.store_prompt(&identity("a"), &second_prompt, &pages()),
            Err(PersistentCacheError::QuotaExceeded)
        ));
    }

    #[test]
    fn tenant_namespaces_are_isolated() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        store
            .store_prompt(&identity("tenant-a"), &prompt(), &pages())
            .unwrap();
        assert_eq!(
            store
                .restore_prompt(&identity("tenant-b"), &prompt(), FAST_RESTORE)
                .unwrap(),
            PersistentLookup::Miss
        );
        store
            .store_prompt(&identity("tenant-b"), &prompt(), &pages())
            .unwrap();
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
    }

    #[test]
    fn continuation_pages_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let store = PersistentPromptCache::new(config(temp.path()), [9; 32]).unwrap();
        let mut extra = pages();
        extra.push(WarmPage {
            bytes: Arc::from([33_u8; 64]),
        });
        assert!(matches!(
            store.store_prompt(&identity("a"), &prompt(), &extra),
            Err(PersistentCacheError::PageCountMismatch)
        ));
    }
}
