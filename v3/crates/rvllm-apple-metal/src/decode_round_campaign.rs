//! Pure, host-testable validation for the second decode round's offline tools.
//! No queue execution, clocks, filesystem access, sample selection or promotion.
#![forbid(unsafe_code)]

use crate::MetalResearchCandidate;
use serde_json::{json, Value};

pub const PROTOCOL: &str = "rvllm.decode-round.screen.v3";
pub const DRIFT_LIMIT: f64 = 0.05;

#[derive(Clone, Debug, PartialEq)]
pub struct CollectionStats {
    pub paired_ratios: Vec<f64>,
    pub abba_median: f64,
    pub baab_median: f64,
    pub bootstrap_median_95: [f64; 2],
    pub control_drift: f64,
}

impl CollectionStats {
    /// A screening rule, NOT statistical independence or model qualification.
    pub fn promising(&self) -> bool {
        self.control_drift <= DRIFT_LIMIT
            && self.abba_median > 1.0
            && self.baab_median > 1.0
            && self.bootstrap_median_95[0] > 1.0
    }

    /// Spend another exploratory context, never qualify timing or promotion.
    /// Pairing can survive changing host conditions; the drift failure remains
    /// visible and must be resolved by independent confirmation later.
    pub fn exploratory_advance(&self) -> bool {
        self.promising()
            || (self.abba_median > 1.10
                && self.baab_median > 1.10
                && self.bootstrap_median_95[0] > 1.10
                && self.paired_ratios.len() == 10
                && self.paired_ratios.iter().all(|ratio| *ratio > 1.0)
                && self.abba_median.max(self.baab_median) / self.abba_median.min(self.baab_median)
                    <= 1.10)
    }

    pub fn receipt(&self) -> Value {
        json!({"paired_block_ratios":self.paired_ratios,
            "abba_median_speedup":self.abba_median,
            "baab_median_speedup":self.baab_median,
            "whole_block_bootstrap_median_95":self.bootstrap_median_95,
            "control_drift_fraction":self.control_drift,"control_drift_limit":DRIFT_LIMIT,
            "control_drift_passed":self.control_drift<=DRIFT_LIMIT,
            "promising_screen_only":self.promising(),
            "exploratory_advance_only":self.exploratory_advance(),"promotion":false})
    }
}

pub fn supported(candidate: MetalResearchCandidate) -> bool {
    candidate.decode_round_operator()
        || candidate
            .global_decode_tile()
            .is_some_and(|t| t.streaming())
        || candidate
            .split_global_decode_tile()
            .is_some_and(|t| t.streaming())
}

/// Return the sole permitted predecessor. A fresh campaign always starts at
/// L256 (operators at L0); there is no exception for a previously winning name.
pub fn predecessor(candidate: MetalResearchCandidate, length: u32) -> Result<Option<u32>, String> {
    if !supported(candidate) {
        return Err("candidate is not in the second-round protocol".into());
    }
    if candidate.decode_round_operator() {
        return if length == 0 {
            Ok(None)
        } else {
            Err("operator length must be zero".into())
        };
    }
    if length > candidate.global_capacity_tokens() {
        return Err("length exceeds declared logical capacity".into());
    }
    match length {
        256 => Ok(None),
        512 => Ok(Some(256)),
        1024 => Ok(Some(512)),
        2048 => Ok(Some(1024)),
        _ => Err("attention stage must be L256/L512/L1024/L2048".into()),
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

// Same deterministic whole-block bootstrap as the existing summary tool.
// This diagnostic interval never rescues drift, order disagreement or a fail.
fn bootstrap(values: &[f64]) -> [f64; 2] {
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
        medians.push(median(&draw));
    }
    medians.sort_by(f64::total_cmp);
    [medians[249], medians[9749]]
}

/// Validate complete raw collection, work counts and recomputed drift. Drift
/// failures return valid stats with `promising=false` so all observations can
/// be retained. Missing/misordered work, contradictory flags or bad identities
/// are errors, never a reason to retry. File hashes are checked by the caller.
pub fn validate_collection(receipt: &Value) -> Result<CollectionStats, String> {
    let candidate: MetalResearchCandidate = receipt["candidate"]
        .as_str()
        .ok_or("missing candidate")?
        .parse()
        .map_err(|_| "invalid candidate")?;
    let length = receipt["length"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("missing length")?;
    predecessor(candidate, length)?;
    let split = candidate.split_global_decode_tile().is_some();
    let expected_schema = if split {
        "rvllm.global-decode.abba.v3"
    } else {
        "rvllm.global-decode.abba.v1"
    };
    if receipt["schema"] != expected_schema
        || receipt["status"] != "collected"
        || receipt["blocks"] != 10
        || receipt["balanced_abba_baab"] != true
        || receipt["warmups_per_arm"] != 5
        || receipt["dispatches_per_sample"] != 100
        || receipt["source_compiles_during_samples"] != 0
        || receipt["timing_eligible"] != false
        || receipt["promotion"] != false
        || receipt["control_drift_limit"] != DRIFT_LIMIT
        || receipt["identity"]["candidate"] != candidate.name()
    {
        return Err("collection protocol/identity mismatch".into());
    }
    if split
        && (receipt["timing_metric"] != "complete-interleaved-partial-merge-command-buffer"
            || receipt["candidate_dispatches_per_operation"] != 2
            || receipt["conditions_are_observations_only"] != true)
    {
        return Err("split receipt does not time complete interleaved operations".into());
    }
    if candidate.decode_round_operator() {
        let k = receipt["operator_k"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("missing exact operator K")?;
        if !candidate.operator_keys().contains(&k) {
            return Err("operator K refused".into());
        }
    }
    let samples = receipt["samples"].as_array().ok_or("samples missing")?;
    if samples.len() != 40 {
        return Err("exactly 40 retained samples required".into());
    }
    let mut ratios = Vec::with_capacity(10);
    let mut control_min = f64::INFINITY;
    let mut control_max = f64::NEG_INFINITY;
    for (block, group) in samples.chunks_exact(4).enumerate() {
        let order = if block % 2 == 0 {
            ["A", "B", "B", "A"]
        } else {
            ["B", "A", "A", "B"]
        };
        let mut a = 0.0;
        let mut b = 0.0;
        for (position, sample) in group.iter().enumerate() {
            let arm = order[position];
            let expected_encoders =
                if (split && arm == "B") || (candidate.ffn_decode() && arm == "A") {
                    200
                } else {
                    100
                };
            if sample["block"].as_u64() != Some(block as u64)
                || sample["position"].as_u64() != Some(position as u64)
                || sample["arm"] != arm
                || sample["dispatches"] != 100
                || sample["operator_iterations"] != 100
                || sample["actual_compute_encoders"] != expected_encoders
                || sample["controls_before"].is_null()
                || sample["controls_after"].is_null()
            {
                return Err("missing/reordered samples, work or observations".into());
            }
            if split
                && (sample["partial_dispatches"] != (if arm == "B" { 100 } else { 0 })
                    || sample["merge_dispatches"] != (if arm == "B" { 100 } else { 0 })
                    || sample["baseline_dispatches"] != (if arm == "A" { 100 } else { 0 }))
            {
                return Err("incomplete split/merge dispatch accounting".into());
            }
            let seconds = sample["gpu_seconds"].as_f64().ok_or("GPU time missing")?;
            let wall = sample["synchronized_wall_seconds"]
                .as_f64()
                .ok_or("wall time missing")?;
            if !seconds.is_finite() || seconds <= 0.0 || !wall.is_finite() || wall <= 0.0 {
                return Err("nonpositive or nonfinite timing".into());
            }
            if arm == "A" {
                a += seconds / 2.0;
                control_min = control_min.min(seconds);
                control_max = control_max.max(seconds);
            } else {
                b += seconds / 2.0;
            }
        }
        let ratio = a / b;
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err("unrepresentable speed ratio".into());
        }
        ratios.push(ratio);
    }
    let drift = control_max / control_min - 1.0;
    let reported = receipt["control_drift_fraction"]
        .as_f64()
        .ok_or("drift missing")?;
    if !drift.is_finite()
        || !reported.is_finite()
        || (drift - reported).abs() > 1e-12
        || receipt["control_drift_passed"].as_bool() != Some(drift <= DRIFT_LIMIT)
    {
        return Err("drift contradicts retained controls".into());
    }
    let abba: Vec<_> = ratios.iter().step_by(2).copied().collect();
    let baab: Vec<_> = ratios.iter().skip(1).step_by(2).copied().collect();
    Ok(CollectionStats {
        abba_median: median(&abba),
        baab_median: median(&baab),
        bootstrap_median_95: bootstrap(&ratios),
        paired_ratios: ratios,
        control_drift: drift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn collection(split: bool) -> Value {
        let candidate = if split {
            MetalResearchCandidate::GlobalD512SplitStreamR4S256T128C2048
        } else {
            MetalResearchCandidate::GlobalD512StreamR4T128C2048
        };
        let mut samples = Vec::new();
        for block in 0..10 {
            let order = if block % 2 == 0 {
                ["A", "B", "B", "A"]
            } else {
                ["B", "A", "A", "B"]
            };
            for (position, arm) in order.into_iter().enumerate() {
                samples.push(json!({"block":block,"position":position,"arm":arm,
                    "dispatches":100,"operator_iterations":100,
                    "actual_compute_encoders":if split && arm=="B" {200} else {100},
                    "partial_dispatches":if arm=="B" {100} else {0},
                    "merge_dispatches":if arm=="B" {100} else {0},
                    "baseline_dispatches":if arm=="A" {100} else {0},
                    "gpu_seconds":if arm=="A" {2.0} else {1.0},
                    "synchronized_wall_seconds":3.0,"controls_before":[],"controls_after":[]}));
            }
        }
        json!({"schema":if split {"rvllm.global-decode.abba.v3"} else {"rvllm.global-decode.abba.v1"},
            "status":"collected","candidate":candidate.name(),"length":256,
            "identity":{"candidate":candidate.name()},"blocks":10,"balanced_abba_baab":true,
            "warmups_per_arm":5,"dispatches_per_sample":100,"source_compiles_during_samples":0,
            "timing_eligible":false,"promotion":false,"control_drift_limit":0.05,
            "control_drift_fraction":0.0,"control_drift_passed":true,"samples":samples,
            "timing_metric":"complete-interleaved-partial-merge-command-buffer",
            "candidate_dispatches_per_operation":2,"conditions_are_observations_only":true})
    }
    #[test]
    fn accepts_complete_balanced_work_without_promoting() {
        for split in [false, true] {
            let result = validate_collection(&collection(split)).unwrap();
            assert!(result.promising());
            assert_eq!(result.paired_ratios, vec![2.0; 10]);
            assert_eq!(result.receipt()["promotion"], false);
        }
    }
    #[test]
    fn rejects_partial_merge_misorder_pruning_and_fabricated_drift() {
        for case in 0..8 {
            let mut r = collection(true);
            match case {
                0 => {
                    r["samples"].as_array_mut().unwrap().pop();
                }
                1 => r["samples"][1]["merge_dispatches"] = json!(0),
                2 => r["samples"][1]["arm"] = json!("A"),
                3 => r["source_compiles_during_samples"] = json!(1),
                4 => r["control_drift_fraction"] = json!(0.01),
                5 => r["control_drift_limit"] = json!(0.06),
                6 => r["samples"][0]["gpu_seconds"] = json!(0),
                _ => r["schema"] = json!("rvllm.global-decode.abba.v2"),
            }
            assert!(validate_collection(&r).is_err(), "case {case}");
        }
    }
    #[test]
    fn drift_and_order_disagreement_cannot_advance() {
        let mut r = collection(false);
        r["samples"][0]["gpu_seconds"] = json!(2.12);
        r["control_drift_fraction"] = json!(0.06);
        r["control_drift_passed"] = json!(false);
        let result = validate_collection(&r).unwrap();
        assert!(!result.promising());
        assert!(result.exploratory_advance());
        let mut r = collection(false);
        for sample in r["samples"].as_array_mut().unwrap() {
            if sample["block"].as_u64().unwrap() % 2 == 1 && sample["arm"] == "B" {
                sample["gpu_seconds"] = json!(3.0);
            }
        }
        let result = validate_collection(&r).unwrap();
        assert!(!result.promising());
        assert!(!result.exploratory_advance());
    }
    #[test]
    fn variance_tolerant_continuation_requires_all_blocks_and_order_agreement() {
        let mut ratios = CollectionStats {
            paired_ratios: vec![2.0; 10],
            abba_median: 2.0,
            baab_median: 1.9,
            bootstrap_median_95: [1.8, 2.1],
            control_drift: 1.5,
        };
        assert!(!ratios.promising());
        assert!(ratios.exploratory_advance());
        ratios.paired_ratios[0] = 0.99;
        assert!(!ratios.exploratory_advance());
        ratios.paired_ratios[0] = 2.0;
        ratios.baab_median = 1.7;
        assert!(!ratios.exploratory_advance());
        ratios.baab_median = 1.9;
        ratios.bootstrap_median_95[0] = 1.09;
        assert!(!ratios.exploratory_advance());
    }
    #[test]
    fn stage_bounds_and_operator_separation() {
        let c = MetalResearchCandidate::GlobalD512StreamR4T128C2048;
        assert_eq!(predecessor(c, 256), Ok(None));
        assert_eq!(predecessor(c, 2048), Ok(Some(1024)));
        assert!(predecessor(c, 4096).is_err());
        assert!(predecessor(MetalResearchCandidate::GlobalD512ShortR4T128, 1024).is_err());
        assert!(predecessor(MetalResearchCandidate::FfnBf16R2Sg2, 256).is_err());
        assert_eq!(
            predecessor(MetalResearchCandidate::FfnBf16R2Sg2, 0),
            Ok(None)
        );
    }
}
