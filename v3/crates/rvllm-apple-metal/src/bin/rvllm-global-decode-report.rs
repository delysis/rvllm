//! Summarize every retained global-decode ABBA cell without sample selection.
#![forbid(unsafe_code)]

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}

fn mean(values: &[f64]) -> Result<f64> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("timing samples must be nonempty, finite and positive".into());
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn median(mut values: Vec<f64>) -> Result<f64> {
    mean(&values)?;
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn cell(directory: &Path) -> Result<Value> {
    let receipt = read(&directory.join("native/abba.json"))?;
    if receipt["schema"] != "rvllm.global-decode.abba.v1"
        || receipt["status"] != "collected"
        || receipt["samples"].as_array().map(Vec::len) != Some(20)
    {
        return Err(format!("invalid ABBA receipt: {}", directory.display()).into());
    }
    let mut arms = BTreeMap::<&str, Vec<f64>>::from([("A", Vec::new()), ("B", Vec::new())]);
    let mut blocks = BTreeMap::<u64, BTreeMap<&str, Vec<f64>>>::new();
    for sample in receipt["samples"]
        .as_array()
        .ok_or("ABBA samples missing")?
    {
        let arm = sample["arm"].as_str().ok_or("sample arm missing")?;
        let block = sample["block"].as_u64().ok_or("sample block missing")?;
        let seconds = sample["gpu_seconds"]
            .as_f64()
            .ok_or("sample GPU seconds missing")?;
        arms.get_mut(arm).ok_or("unknown ABBA arm")?.push(seconds);
        blocks
            .entry(block)
            .or_default()
            .entry(arm)
            .or_default()
            .push(seconds);
    }
    if arms["A"].len() != 10 || arms["B"].len() != 10 || blocks.len() != 5 {
        return Err("ABBA receipt must contain ten samples per arm in five blocks".into());
    }
    let baseline = mean(&arms["A"])?;
    let candidate = mean(&arms["B"])?;
    let mut paired = Vec::new();
    for block in blocks.values() {
        if block.get("A").map(Vec::len) != Some(2) || block.get("B").map(Vec::len) != Some(2) {
            return Err("each ABBA block must contain two samples per arm".into());
        }
        paired.push(mean(&block["A"])? / mean(&block["B"])?);
    }
    let queue = read(&directory.join("report.json"))?;
    let drift_passed = receipt["control_drift_passed"] == true;
    let queue_succeeded = queue["status"] == "succeeded";
    let classification = if !drift_passed {
        "inconclusive_control_drift"
    } else if !queue_succeeded {
        "inconclusive_queue"
    } else if queue["sampled_conditions_eligible"] != true {
        "exploratory_condition_ineligible"
    } else {
        "exploratory_stable"
    };
    Ok(json!({
        "candidate":receipt["candidate"], "length":receipt["length"],
        "classification":classification,
        "queue_status":queue["status"],
        "sampled_conditions_eligible":queue["sampled_conditions_eligible"],
        "control_drift_fraction":receipt["control_drift_fraction"],
        "control_drift_passed":drift_passed,
        "baseline_mean_ms_per_dispatch":baseline * 10.0,
        "candidate_mean_ms_per_dispatch":candidate * 10.0,
        "ratio_of_means":baseline / candidate,
        "median_paired_block_ratio":median(paired.clone())?,
        "paired_block_ratios":paired,
        "samples_retained":20,
        "promotion":false
    }))
}

fn run() -> Result<()> {
    let queue = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: rvllm-global-decode-report ABSOLUTE_QUEUE")?,
    );
    if !queue.is_absolute() {
        return Err("queue path must be absolute".into());
    }
    let mut cells = Vec::new();
    for entry in std::fs::read_dir(queue.join("results"))? {
        let directory = entry?.path();
        if directory
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("-abba-L"))
            && directory.join("native/abba.json").is_file()
        {
            cells.push(cell(&directory)?);
        }
    }
    cells.sort_by(|left, right| {
        left["candidate"]
            .as_str()
            .cmp(&right["candidate"].as_str())
            .then_with(|| left["length"].as_u64().cmp(&right["length"].as_u64()))
    });
    let value = json!({
        "schema":"rvllm.global-decode.summary.v1",
        "queue":queue,
        "cells":cells,
        "cell_count":cells.len(),
        "claim":"Exploratory raw-operator evidence only. No cell is promotion evidence without full-route real-weight qualification and independent confirmation."
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rvllm-global-decode-report: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{mean, median};

    #[test]
    fn summaries_retain_all_positive_samples() {
        assert_eq!(mean(&[1.0, 2.0, 9.0]).unwrap(), 4.0);
        assert_eq!(median(vec![9.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
    }

    #[test]
    fn summaries_reject_missing_or_invalid_timing() {
        assert!(mean(&[]).is_err());
        assert!(mean(&[1.0, 0.0]).is_err());
        assert!(mean(&[1.0, f64::NAN]).is_err());
    }
}
