#![forbid(unsafe_code)]

use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let apply = match args.next().as_deref() {
        Some("--inspect") => false,
        Some("--apply") => true,
        _ => return Err("expected --inspect or --apply, then explicit Cargo target directories".into()),
    };
    let minimum_age = Duration::from_secs(2 * 86400);
    let now = SystemTime::now();
    for raw in args {
        let root = PathBuf::from(raw);
        if !root.is_absolute() || fs::symlink_metadata(&root)?.file_type().is_symlink()
            || root.canonicalize()? != root || root.file_name().and_then(|x| x.to_str()) != Some("target")
        {
            return Err(format!("not an explicit, nonsymlink target: {}", root.display()).into());
        }
        let tag = fs::read_to_string(root.join("CACHEDIR.TAG"))?;
        if !tag.starts_with("Signature: 8a477f597d28d172789f06886806bc55") || !tag.contains("created by cargo") {
            return Err("unrecognized Cargo cache identity".into());
        }
        let mut locks: Vec<File> = Vec::new();
        let mut inactive = true;
        for profile in ["debug", "release"] {
            let directory = root.join(profile);
            if !directory.exists() { continue; }
            if fs::symlink_metadata(&directory)?.file_type().is_symlink() {
                return Err("symlink profile refused".into());
            }
            let file = OpenOptions::new().read(true).write(true).open(directory.join(".cargo-lock"))?;
            if file.try_lock().is_err() {
                inactive = false;
                break;
            }
            locks.push(file);
        }
        if !inactive || locks.is_empty() {
            println!("SKIP active-or-unidentified {}", root.display());
            continue;
        }
        let mut total = 0_u64;
        let mut count = 0_u64;
        for profile in ["debug", "release"] {
            let deps = root.join(profile).join("deps");
            if !deps.exists() { continue; }
            if fs::symlink_metadata(&deps)?.file_type().is_symlink() { return Err("symlink deps refused".into()); }
            for entry in fs::read_dir(deps)? {
                let path = entry?.path();
                if !matches!(path.extension().and_then(|x| x.to_str()), Some("rlib" | "rmeta" | "o")) { continue; }
                let metadata = fs::symlink_metadata(&path)?;
                if !metadata.is_file() || metadata.file_type().is_symlink()
                    || now.duration_since(metadata.modified()?).unwrap_or_default() < minimum_age
                { continue; }
                // Locks remain held across discovery and unlink. This never
                // descends into a source tree, bundle, binary, or model cache.
                if apply { fs::remove_file(&path)?; }
                println!("{}\t{}\t{}", if apply { "REMOVED" } else { "CANDIDATE" }, metadata.len(), path.display());
                total += metadata.len();
                count += 1;
            }
        }
        println!("TOTAL\t{count}\t{total}\t{}", root.display());
        drop(locks);
    }
    Ok(())
}
