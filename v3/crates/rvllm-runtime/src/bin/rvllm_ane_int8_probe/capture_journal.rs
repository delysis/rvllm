//! Append completed diagnostic records before starting the next device call.
#![forbid(unsafe_code)]

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub(super) struct CaptureJournal {
    writer: BufWriter<File>,
}

impl CaptureJournal {
    pub(super) fn create(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// The caller supplies one compact JSON record. A complete line survives
    /// ordinary later errors; this does not promise durability across power loss.
    pub(super) fn append(&mut self, record: &[u8]) -> io::Result<()> {
        self.writer.write_all(record)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_record_survives_later_error_and_cannot_be_overwritten() {
        let directory = std::env::temp_dir().join(format!(
            "rvllm-capture-journal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("capture.jsonl");
        let record = br#"{"collection_complete":false,"phase":"serial","source_index":0}"#;
        let result: io::Result<()> = (|| {
            let mut journal = CaptureJournal::create(&path)?;
            journal.append(record)?;
            Err(io::Error::other("simulated later device failure"))
        })();
        assert!(result.is_err());
        let mut expected = record.to_vec();
        expected.push(b'\n');
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        assert_eq!(
            CaptureJournal::create(&path).err().unwrap().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&directory).unwrap();
    }
}
