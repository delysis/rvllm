use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{AppleAcceleratorTarget, RolloutBucket};

pub const ANE_COMPILE_CACHE_SCHEMA: &str = "rvllm-ane-v1";

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct CompileCacheHash([u8; 32]);

impl CompileCacheHash {
    #[must_use]
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        Self(out)
    }

    #[must_use]
    pub const fn as_digest(&self) -> [u8; 32] {
        self.0
    }

    #[must_use]
    pub fn to_hex(self) -> String {
        let mut hex = String::with_capacity(64);
        for byte in self.0 {
            hex.push(hex_digit(byte >> 4));
            hex.push(hex_digit(byte & 0x0f));
        }
        hex
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AneCompileCacheKey {
    pub mil_hash: CompileCacheHash,
    pub weight_hash: CompileCacheHash,
    pub target: AppleAcceleratorTarget,
    pub bucket: RolloutBucket,
}

impl AneCompileCacheKey {
    #[must_use]
    pub fn new(
        mil_bytes: &[u8],
        weight_bytes: &[u8],
        target: &AppleAcceleratorTarget,
        bucket: RolloutBucket,
    ) -> Self {
        Self::from_hashes(
            CompileCacheHash::from_bytes(mil_bytes),
            CompileCacheHash::from_bytes(weight_bytes),
            target,
            bucket,
        )
    }

    #[must_use]
    pub fn from_hashes(
        mil_hash: CompileCacheHash,
        weight_hash: CompileCacheHash,
        target: &AppleAcceleratorTarget,
        bucket: RolloutBucket,
    ) -> Self {
        Self {
            mil_hash,
            weight_hash,
            target: target.clone(),
            bucket,
        }
    }

    #[must_use]
    pub fn path_component(&self) -> String {
        format!(
            "{ANE_COMPILE_CACHE_SCHEMA}_mil-{}_weight-{}_target-{}_bucket-s{}-t{}",
            self.mil_hash.to_hex(),
            self.weight_hash.to_hex(),
            self.target.cache_key(),
            self.bucket.seqs,
            self.bucket.tokens
        )
    }

    #[must_use]
    pub fn cache_path(&self, root: impl AsRef<Path>) -> PathBuf {
        root.as_ref().join(self.path_component())
    }
}

fn hex_digit(nibble: u8) -> char {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    HEX[nibble as usize] as char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_cache_key_schema_is_stable() {
        let target = AppleAcceleratorTarget::from_device_name("Apple M5 Pro", 1);
        let key = AneCompileCacheKey::new(
            b"program(1.0)\nconst W\n",
            b"weights\0bytes\xff",
            &target,
            RolloutBucket { seqs: 8, tokens: 4 },
        );

        assert_eq!(
            key.path_component(),
            "rvllm-ane-v1_mil-992b991192f51944218b405de266420b15d52487c98ebf5724ac0be74f1f235a_weight-8390e6cc07b05a5764a4cb460fbc2880ad895bb1d6d57bec46bbb70b45554655_target-apple-gpu-apple10-arch17-npu-m5-tier-pro-nax1-ane16-dies1_bucket-s8-t4"
        );
    }

    #[test]
    fn compile_cache_key_changes_for_each_schema_input() {
        let target = AppleAcceleratorTarget::from_device_name("Apple M5 Pro", 1);
        let other_target = AppleAcceleratorTarget::from_device_name("Apple M4 Pro", 1);
        let bucket = RolloutBucket { seqs: 8, tokens: 4 };
        let key = AneCompileCacheKey::from_hashes(
            CompileCacheHash::from_digest([0x11; 32]),
            CompileCacheHash::from_digest([0x22; 32]),
            &target,
            bucket,
        );

        assert_ne!(
            key.path_component(),
            AneCompileCacheKey::from_hashes(
                CompileCacheHash::from_digest([0x33; 32]),
                CompileCacheHash::from_digest([0x22; 32]),
                &target,
                bucket,
            )
            .path_component()
        );
        assert_ne!(
            key.path_component(),
            AneCompileCacheKey::from_hashes(
                CompileCacheHash::from_digest([0x11; 32]),
                CompileCacheHash::from_digest([0x33; 32]),
                &target,
                bucket,
            )
            .path_component()
        );
        assert_ne!(
            key.path_component(),
            AneCompileCacheKey::from_hashes(
                CompileCacheHash::from_digest([0x11; 32]),
                CompileCacheHash::from_digest([0x22; 32]),
                &other_target,
                bucket,
            )
            .path_component()
        );
        assert_ne!(
            key.path_component(),
            AneCompileCacheKey::from_hashes(
                CompileCacheHash::from_digest([0x11; 32]),
                CompileCacheHash::from_digest([0x22; 32]),
                &target,
                RolloutBucket { seqs: 4, tokens: 4 },
            )
            .path_component()
        );
    }

    #[test]
    fn compile_cache_path_component_is_safe() {
        let target = AppleAcceleratorTarget::from_device_name("../Apple M5 Pro/../../bad", 1);
        let key = AneCompileCacheKey::from_hashes(
            CompileCacheHash::from_digest([0xaa; 32]),
            CompileCacheHash::from_digest([0xbb; 32]),
            &target,
            RolloutBucket {
                seqs: 32,
                tokens: 8,
            },
        );
        let component = key.path_component();

        assert!(!component.contains(".."));
        assert!(!component.contains('/'));
        assert!(!component.contains('\\'));
        assert!(component
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_'));

        let root = std::path::Path::new("/tmp/rvllm-ane-cache");
        let path = key.cache_path(root);
        assert!(path.starts_with(root));
        assert_eq!(
            path.file_name().and_then(std::ffi::OsStr::to_str),
            Some(component.as_str())
        );
    }
}
