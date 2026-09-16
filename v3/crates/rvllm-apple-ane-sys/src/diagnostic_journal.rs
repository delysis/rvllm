//! Opt-in durable phase records for private-driver validation. This is off by
//! default because syncing every evaluation is incompatible with benchmarking.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};

pub(super) fn record(stage: &'static str, model_id: Option<&str>) -> Result<(), String> {
    static JOURNAL: OnceLock<Result<Option<Mutex<File>>, String>> = OnceLock::new();
    let journal = JOURNAL.get_or_init(|| {
        let Some(path) = std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL") else {
            return Ok(None);
        };
        open(std::path::Path::new(&path)).map(Some)
    });
    let Some(file) = journal.as_ref().map_err(Clone::clone)? else {
        return Ok(());
    };
    append(file, stage, model_id)
}

fn open(path: &std::path::Path) -> Result<Mutex<File>, String> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map(Mutex::new)
        .map_err(|e| format!("create exclusive ANE diagnostic journal {path:?}: {e}"))
}

fn append(file: &Mutex<File>, stage: &'static str, model_id: Option<&str>) -> Result<(), String> {
    // All stage names are source literals and model IDs have been validated as
    // alphanumeric/underscore/hyphen. No tensor, prompt, or weight data is logged.
    let id = model_id.unwrap_or("");
    if !id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err("ANE diagnostic model ID is not a safe string".into());
    }
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    let mut file = file
        .lock()
        .map_err(|_| "ANE diagnostic journal lock poisoned")?;
    writeln!(
        file,
        "{{\"pid\":{},\"unix_ms\":{timestamp},\"stage\":\"{stage}\",\"model_id\":\"{id}\"}}",
        std::process::id()
    )
    .and_then(|()| file.sync_all())
    .map_err(|e| format!("persist ANE diagnostic phase: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_records_are_readable_and_existing_evidence_is_not_truncated() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rvllm-ane-journal-{}-{nonce}.jsonl",
            std::process::id()
        ));
        let journal = open(&path).unwrap();
        append(&journal, "compile_begin", Some("0x1234abcd")).unwrap();
        assert!(open(&path).is_err());
        assert!(append(&journal, "invalid", Some("bad\"id")).is_err());
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 1);
        assert!(text.contains("\"stage\":\"compile_begin\""));
        assert!(text.contains("\"model_id\":\"0x1234abcd\""));
        drop(journal);
        std::fs::remove_file(path).unwrap();
    }
}
