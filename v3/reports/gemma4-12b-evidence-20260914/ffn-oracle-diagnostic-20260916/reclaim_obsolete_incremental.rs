#![forbid(unsafe_code)]

use std::fs::{self, OpenOptions};
use std::path::Path;
use std::time::{Duration, SystemTime};

fn old_tree(path: &Path, now: SystemTime) -> std::io::Result<Option<u64>> {
    let meta = fs::symlink_metadata(path)?;
    if meta.file_type().is_symlink()
        || now.duration_since(meta.modified()?).unwrap_or_default() < Duration::from_secs(2 * 3600)
    {
        return Ok(None);
    }
    if meta.is_file() { return Ok(Some(meta.len())); }
    if !meta.is_dir() { return Ok(None); }
    let mut bytes = 0;
    for entry in fs::read_dir(path)? {
        let Some(size) = old_tree(&entry?.path(), now)? else { return Ok(None); };
        bytes += size;
    }
    Ok(Some(bytes))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apply = match std::env::args().nth(1).as_deref() {
        Some("--inspect") => false,
        Some("--apply") => true,
        _ => return Err("expected --inspect or --apply".into()),
    };
    // This helper cannot select another tree or delete linked artifacts.
    let root = Path::new("/Users/george/Downloads/rvllm/v3/target");
    if root.canonicalize()? != root { return Err("noncanonical target".into()); }
    let tag = fs::read_to_string(root.join("CACHEDIR.TAG"))?;
    if !tag.starts_with("Signature: 8a477f597d28d172789f06886806bc55") || !tag.contains("created by cargo") {
        return Err("unrecognized Cargo target".into());
    }
    let debug = root.join("debug");
    let incremental = debug.join("incremental");
    if debug.canonicalize()? != debug || incremental.canonicalize()? != incremental {
        return Err("linked profile or incremental tree".into());
    }
    let lock = OpenOptions::new().read(true).write(true).open(debug.join(".cargo-lock"))?;
    lock.try_lock()?;
    let now = SystemTime::now();
    let mut total = 0;
    for entry in fs::read_dir(incremental)? {
        let path = entry?.path();
        if !fs::symlink_metadata(&path)?.is_dir() { continue; }
        if let Some(bytes) = old_tree(&path, now)? {
            if apply { fs::remove_dir_all(&path)?; }
            println!("{}\t{}\t{}", if apply { "REMOVED" } else { "CANDIDATE" }, bytes, path.display());
            total += bytes;
        }
    }
    println!("TOTAL\t{total}");
    drop(lock);
    Ok(())
}
