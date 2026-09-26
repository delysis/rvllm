//! Run a pinned ABBA test and retain a complete drift-invalid collection.
#![forbid(unsafe_code)]

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const REPORT_DIR: &str = "RVLLM_METAL_GLOBAL_DECODE_REPORT_DIR";
const CANDIDATE: &str = "RVLLM_METAL_GLOBAL_DECODE_CANDIDATE";
const LENGTH: &str = "RVLLM_METAL_GLOBAL_DECODE_LENGTH";

fn digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_retained_receipt(path: &Path) -> Result<()> {
    let receipt: Value = serde_json::from_reader(File::open(path)?)?;
    let candidate = std::env::var(CANDIDATE)?;
    let length: u64 = std::env::var(LENGTH)?.parse()?;
    let balanced = receipt["balanced_abba_baab"] == true;
    let blocks = if balanced { 10 } else { 5 };
    if receipt["blocks"].as_u64() != Some(blocks)
        || receipt["samples"].as_array().is_some_and(|samples| {
            samples.iter().enumerate().any(|(i, s)| {
                let order = if balanced && i / 4 % 2 == 1 {
                    ["B", "A", "A", "B"]
                } else {
                    ["A", "B", "B", "A"]
                };
                s["block"].as_u64() != Some((i / 4) as u64)
                    || s["position"].as_u64() != Some((i % 4) as u64)
                    || s["arm"] != order[i % 4]
                    || s["dispatches"] != 100
                    || !s["gpu_seconds"]
                        .as_f64()
                        .is_some_and(|x| x.is_finite() && x > 0.0)
            })
        })
    {
        return Err("incomplete or misordered counterbalanced collection".into());
    }
    if receipt["schema"] != "rvllm.global-decode.abba.v1"
        || receipt["status"] != "collected"
        || receipt["candidate"] != candidate
        || receipt["length"].as_u64() != Some(length)
        || receipt["samples"].as_array().map(Vec::len) != Some(blocks as usize * 4)
        || receipt["control_drift_passed"] != false
        || receipt["timing_eligible"] != false
        || receipt["source_compiles_during_samples"] != 0
    {
        return Err("nonzero test did not leave a complete drift-invalid ABBA receipt".into());
    }
    Ok(())
}

fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let executable = PathBuf::from(args.next().ok_or("missing test executable")?);
    let expected = args
        .next()
        .ok_or("missing test executable SHA-256")?
        .into_string()
        .map_err(|_| "test executable SHA-256 is not UTF-8")?;
    if digest(&executable)? != expected.to_ascii_lowercase() {
        return Err("test executable identity drift".into());
    }
    let status = Command::new(&executable).args(args).status()?;
    if status.success() {
        return Ok(());
    }
    let directory = PathBuf::from(std::env::var_os(REPORT_DIR).ok_or("missing report directory")?);
    validate_retained_receipt(&directory.join("abba.json"))?;
    eprintln!(
        "retained complete drift-invalid ABBA collection after child exit {:?}",
        status.code()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rvllm-retain-abba: {error}");
            ExitCode::FAILURE
        }
    }
}
