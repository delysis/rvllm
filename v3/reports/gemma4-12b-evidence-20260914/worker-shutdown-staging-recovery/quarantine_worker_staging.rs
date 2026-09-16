#![forbid(unsafe_code)]
use std::{fs, path::PathBuf, process::Command, time::UNIX_EPOCH};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = PathBuf::from("reports/gemma4-12b-evidence-20260914/worker-shutdown-staging-recovery");
    let destination = base.join("quarantined-source-staging");
    fs::create_dir(&destination)?;
    let temporary = PathBuf::from("/var/folders/t0/4s921_v11fv9vlymtx6g5qgm0000gn/T");
    let mut count = 0;
    for id in fs::read_to_string(base.join("known-model-ids.txt"))?.lines() {
        if id.len() != 194 || !id.bytes().all(|c| c.is_ascii_hexdigit() || c == b'_') { return Err("invalid known identity".into()); }
        let path = temporary.join(id);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let created = metadata.created()?.duration_since(UNIX_EPOCH)?.as_secs();
        if !metadata.is_dir() || metadata.file_type().is_symlink() || !(1789517543..=1789517711).contains(&created) {
            return Err(format!("not staging from the completed CLI process: {}", path.display()).into());
        }
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            match entry.file_name().to_str() {
                Some("model.mil" | "net.plist") if kind.is_file() => (),
                Some("weights") if kind.is_dir() && fs::read_dir(entry.path())?.next().is_none() => (),
                _ => return Err(format!("unexpected staging content: {}", entry.path().display()).into()),
            }
        }
        if fs::read(path.join("model.mil"))? != fs::read(path.join("net.plist"))? { return Err("MIL staging differs".into()); }
        let hash = Command::new("/usr/bin/shasum").args(["-a", "256"]).arg(path.join("model.mil")).output()?;
        let digest = String::from_utf8(hash.stdout)?;
        if !hash.status.success() || digest.split_whitespace().next().map(str::to_uppercase).as_deref() != Some(&id[..64]) {
            return Err("MIL digest does not match known model identity".into());
        }
        // Preserve every byte through a same-volume rename. No daemon path,
        // weight asset, or unknown staging directory is removed or replaced.
        fs::rename(&path, destination.join(id))?;
        println!("QUARANTINED\t{created}\t{id}");
        count += 1;
    }
    println!("TOTAL\t{count}");
    Ok(())
}
