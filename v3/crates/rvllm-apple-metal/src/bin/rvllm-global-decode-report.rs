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

/// Deterministic whole-block bootstrap, no within-block resampling or trimming.
/// Diagnostic interval only: ten blocks do not establish full-route performance.
fn bootstrap_median_95(values: &[f64]) -> Result<[f64; 2]> {
    mean(values)?;
    let mut state = 0x8fb6_04e1_5a52_bdf4_u64;
    let mut medians = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let mut draw = Vec::with_capacity(values.len());
        for _ in values {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            draw.push(values[(state % values.len() as u64) as usize]);
        }
        medians.push(median(draw)?);
    }
    medians.sort_by(f64::total_cmp);
    Ok([medians[249], medians[9749]])
}

fn cell(directory: &Path) -> Result<Value> {
    let receipt = read(&directory.join("native/abba.json"))?;
    let balanced = receipt["balanced_abba_baab"] == true;
    let count = if balanced { 10 } else { 5 };
    if receipt["blocks"].as_u64() != Some(count as u64) {
        return Err("block count does not match collection contract".into());
    }
    if receipt["schema"] != "rvllm.global-decode.abba.v1"
        || receipt["status"] != "collected"
        || receipt["samples"].as_array().map(Vec::len) != Some(count * 4)
    {
        return Err(format!("invalid ABBA receipt: {}", directory.display()).into());
    }
    let mut arms = BTreeMap::<&str, Vec<f64>>::from([("A", Vec::new()), ("B", Vec::new())]);
    let mut blocks = BTreeMap::<u64, BTreeMap<&str, Vec<f64>>>::new();
    for (index, sample) in receipt["samples"]
        .as_array()
        .ok_or("ABBA samples missing")?
        .iter()
        .enumerate()
    {
        let order = if balanced && index / 4 % 2 == 1 {
            ["B", "A", "A", "B"]
        } else {
            ["A", "B", "B", "A"]
        };
        if sample["block"].as_u64() != Some((index / 4) as u64)
            || sample["position"].as_u64() != Some((index % 4) as u64)
            || sample["arm"] != order[index % 4]
            || sample["dispatches"] != 100
        {
            return Err("missing, duplicated, reordered or unbalanced samples".into());
        }
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
    if arms["A"].len() != count * 2 || arms["B"].len() != count * 2 || blocks.len() != count {
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
    let interval = bootstrap_median_95(&paired)?;
    let abba = paired
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| (!balanced || i % 2 == 0).then_some(x))
        .collect::<Vec<_>>();
    let baab = paired
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| (balanced && i % 2 == 1).then_some(x))
        .collect::<Vec<_>>();
    let abba_median = median(abba)?;
    let baab_median = if balanced { Some(median(baab)?) } else { None };
    let order_agrees = balanced && abba_median > 1.0 && baab_median.is_some_and(|x| x > 1.0);
    let computed_drift = arms["A"].iter().copied().fold(f64::NEG_INFINITY, f64::max)
        / arms["A"].iter().copied().fold(f64::INFINITY, f64::min)
        - 1.0;
    if (receipt["control_drift_fraction"]
        .as_f64()
        .ok_or("missing drift")?
        - computed_drift)
        .abs()
        > 1e-12
    {
        return Err("drift value disagrees with retained samples".into());
    }
    let queue = read(&directory.join("report.json"))?;
    let drift_passed = receipt["control_drift_passed"] == true && computed_drift <= 0.05;
    let queue_succeeded = queue["status"] == "succeeded" && queue["files_unchanged"] == true;
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
        "operator_k":receipt["operator_k"],
        "baseline_mean_ms_per_iteration":baseline * 10.0,
        "candidate_mean_ms_per_iteration":candidate * 10.0,
        "timing_unit":"one complete operator invocation; fused baseline may use two encoders",
        "ratio_of_means":baseline / candidate,
        "median_paired_block_ratio":median(paired.clone())?,
        "paired_block_ratios":paired,
        "samples_retained":count*4,"balanced_abba_baab":balanced,
        "bootstrap_resamples":10000,"paired_median_bootstrap_95":interval,
        "abba_median_ratio":abba_median,"baab_median_ratio":baab_median,
        "order_strata_agree":order_agrees,
        "screening_evidence_only":balanced && order_agrees && interval[0]>1.0
            && drift_passed && queue_succeeded && queue["sampled_conditions_eligible"]==true,
        "promotion":false
    }))
}

fn matches_campaign(name: &str, campaign: Option<&str>) -> bool {
    campaign.is_none_or(|campaign| {
        name.strip_prefix(campaign)
            .is_some_and(|suffix| suffix.starts_with('-'))
    })
}

fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let queue = PathBuf::from(
        args.next()
            .ok_or("usage: rvllm-global-decode-report ABSOLUTE_QUEUE [CAMPAIGN]")?,
    );
    let campaign = args.next();
    if args.next().is_some() {
        return Err("usage: rvllm-global-decode-report ABSOLUTE_QUEUE [CAMPAIGN]".into());
    }
    let campaign = campaign
        .as_deref()
        .map(|value| value.to_str().ok_or("campaign must be UTF-8"))
        .transpose()?;
    if !queue.is_absolute() {
        return Err("queue path must be absolute".into());
    }
    let mut cells = Vec::new();
    for entry in std::fs::read_dir(queue.join("results"))? {
        let directory = entry?.path();
        if directory
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("-abba-L") && matches_campaign(name, campaign))
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
    use super::{matches_campaign, mean, median};

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

    #[test]
    fn campaign_filter_has_an_id_boundary() {
        assert!(matches_campaign("round-01-arm-abba-L0", Some("round-01")));
        assert!(!matches_campaign("round-010-arm-abba-L0", Some("round-01")));
        assert!(matches_campaign("anything-abba-L0", None));
    }
}

#[cfg(test)]
mod round_statistics_tests {
    use super::*;
    #[test]
    fn bootstrap_is_deterministic_and_does_not_hide_regression() {
        let values = [1.0; 10];
        assert_eq!(bootstrap_median_95(&values).unwrap(), [1.0, 1.0]);
        let varied = [0.8, 1.1, 0.9, 1.2, 0.85, 1.0, 1.05, 0.95, 1.01, 0.99];
        let a = bootstrap_median_95(&varied).unwrap();
        assert_eq!(a, bootstrap_median_95(&varied).unwrap());
        assert!(a[0] < 1.0);
        assert!(bootstrap_median_95(&[f64::NAN]).is_err());
    }
}
