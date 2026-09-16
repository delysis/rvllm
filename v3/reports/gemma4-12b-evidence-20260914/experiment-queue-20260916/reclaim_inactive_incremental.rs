#![forbid(unsafe_code)]

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

// Reject an entire candidate on any recent entry or non-regular object. Never
// follow links, and keep Cargo's profile locks held until deletion completes.
fn old_tree(path: &Path, now: SystemTime) -> Result<Option<u64>, Box<dyn std::error::Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || now.duration_since(metadata.modified()?).unwrap_or_default() < Duration::from_secs(7 * 86400)
    {
        return Ok(None);
    }
    if metadata.is_file() {
        return Ok(Some(metadata.len()));
    }
    if !metadata.is_dir() {
        return Ok(None);
    }
    let mut bytes = 0_u64;
    for entry in fs::read_dir(path)? {
        let Some(size) = old_tree(&entry?.path(), now)? else { return Ok(None); };
        bytes = bytes.checked_add(size).ok_or("size overflow")?;
    }
    Ok(Some(bytes))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let apply = match args.next().as_deref() {
        Some("--inspect") => false,
        Some("--apply") => true,
        _ => return Err("expected --inspect or --apply, then explicit target paths".into()),
    };
    let now = SystemTime::now();
    for raw in args {
        let root = PathBuf::from(raw);
        if !root.is_absolute() || root.canonicalize()? != root
            || fs::symlink_metadata(&root)?.file_type().is_symlink()
            || root.file_name().and_then(|x| x.to_str()) != Some("target")
        {
            return Err("noncanonical Cargo target refused".into());
        }
        let tag = fs::read_to_string(root.join("CACHEDIR.TAG"))?;
        if !tag.starts_with("Signature: 8a477f597d28d172789f06886806bc55") || !tag.contains("created by cargo") {
            return Err("unrecognized Cargo cache identity".into());
        }
        let mut locks: Vec<File> = Vec::new();
        let mut active = false;
        for profile in ["debug", "release"] {
            let path = root.join(profile);
            if !path.exists() { continue; }
            if fs::symlink_metadata(&path)?.file_type().is_symlink() { return Err("symlink profile".into()); }
            let file = OpenOptions::new().read(true).write(true).open(path.join(".cargo-lock"))?;
            if file.try_lock().is_err() { active = true; break; }
            locks.push(file);
        }
        if active || locks.is_empty() {
            println!("SKIP\tactive-or-unidentified\t{}", root.display());
            continue;
        }
        for profile in ["debug", "release"] {
            let path = root.join(profile).join("incremental");
            if !path.exists() { continue; }
            match old_tree(&path, now)? {
                Some(bytes) => {
                    if apply { fs::remove_dir_all(&path)?; }
                    println!("{}\t{}\t{}", if apply { "REMOVED" } else { "CANDIDATE" }, bytes, path.display());
                }
                None => println!("SKIP\trecent-or-nonregular\t{}", path.display()),
            }
        }
        drop(locks);
    }
    Ok(())
}
