//! Remove only a verified, redundant copy of the caller's source weights.
//! This never locates or removes daemon cache entries or lowered programs.

use std::fs::File;
use std::io::Read;
use std::path::Path;

pub(super) fn remove_verified_copy(path: &Path, source: &[u8]) -> Result<bool, String> {
    let metadata = match path.symlink_metadata() {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && source.is_empty() => {
            return Ok(false);
        }
        Err(error) => return Err(format!("ANE source staging metadata: {error}")),
    };
    if !metadata.is_file() || metadata.len() != source.len() as u64 {
        return Err("ANE data staging is not a regular source-weight copy".into());
    }
    let mut file = File::open(path).map_err(|e| format!("ANE source staging open: {e}"))?;
    let mut buffer = [0_u8; 65536];
    for expected in source.chunks(buffer.len()) {
        let actual = &mut buffer[..expected.len()];
        file.read_exact(actual)
            .map_err(|e| format!("ANE source staging read: {e}"))?;
        if actual != expected {
            return Err("ANE data staging differs from source weights; retained".into());
        }
    }
    if file.read(&mut buffer[..1]).map_err(|e| e.to_string())? != 0 {
        return Err("ANE data staging grew during verification; retained".into());
    }
    drop(file);
    std::fs::remove_file(path).map_err(|e| format!("ANE source staging removal: {e}"))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_only_identical_regular_source_copies() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("rvllm-ane-source-{}-{nonce}", std::process::id()));
        let source = vec![42_u8; 65539]; // exercise the partial final chunk
        std::fs::write(&path, &source).unwrap();
        let different = vec![41_u8; source.len()];
        assert!(remove_verified_copy(&path, &different).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), source);
        assert!(remove_verified_copy(&path, &source).unwrap());
        assert!(!path.exists());
        assert!(!remove_verified_copy(&path, &[]).unwrap());
    }
}
