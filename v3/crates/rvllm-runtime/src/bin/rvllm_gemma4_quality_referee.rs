//! Fail-closed checkpoint-quality referee for experimental Gemma 4 low-bit routes.
#![forbid(unsafe_code)]

use rvllm_runtime::kernel_game::{parse_strict_json, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "rvllm.gemma4_quality_referee.input.v1";
const FIXTURE_SCHEMA: &str = "rvllm.gemma4_token_fixture.v1";
const OBSERVATION_SCHEMA: &str = "rvllm.gemma4_quality_observations.v1";
const CALIBRATION_SCHEMA: &str = "rvllm.gemma4_quality_calibration.v1";
const OUTPUT_SCHEMA: &str = "rvllm.gemma4_quality_referee.receipt.v1";
const REQUIRED_ROLES: [&str; 7] = [
    "q_projection",
    "k_projection",
    "v_projection",
    "o_projection",
    "gate_projection",
    "up_projection",
    "down_projection",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    checkpoint: CheckpointIdentity,
    config: FileIdentity,
    quantizer: QuantizerIdentity,
    reference_route: RouteIdentity,
    candidate_route: RouteIdentity,
    fixture: FileIdentity,
    reference_observations: FileIdentity,
    candidate_observations: FileIdentity,
    calibration: Option<FileIdentity>,
    coverage: Coverage,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    path: PathBuf,
    sha256: Sha256Digest,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointIdentity {
    repository: String,
    revision: String,
    manifest_sha256: Sha256Digest,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct QuantizerIdentity {
    name: String,
    implementation_revision: String,
    weight_bits: u8,
    group_size: u32,
    scale_dtype: String,
    zero_point: bool,
    rounding: String,
    calibration_dataset_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RouteIdentity {
    name: String,
    source_tree_sha256: Sha256Digest,
    executable_sha256: Sha256Digest,
    model_package_sha256: Sha256Digest,
    full_model_route: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Coverage {
    full_checkpoint_quantized: bool,
    quantized_roles: Vec<String>,
    operator_correctness_receipt_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema: String,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureCase {
    id: String,
    token_ids: Vec<u32>,
    target_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observations {
    schema: String,
    checkpoint_manifest_sha256: Sha256Digest,
    config_sha256: Sha256Digest,
    fixture_sha256: Sha256Digest,
    route: RouteIdentity,
    cases: Vec<ObservedCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedCase {
    id: String,
    positions: Vec<ObservedPosition>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedPosition {
    target_token_id: u32,
    target_negative_log_likelihood: f64,
    logits: BTreeMap<u32, f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Calibration {
    schema: String,
    protocol: String,
    calibration_corpus_sha256: Sha256Digest,
    held_out_fixture_sha256: Sha256Digest,
    minimum_target_positions: usize,
    max_absolute_logit_delta: f64,
    max_mean_absolute_logit_delta: f64,
    max_mean_nll_increase: f64,
    max_perplexity_ratio: f64,
    minimum_representative_top1_agreement: f64,
    rationale: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Decision {
    CalibrationRequired,
    Rejected,
    BoundedSliceEvidence,
    FullModelAccepted,
}

#[derive(Debug, Serialize)]
struct Metrics {
    cases: usize,
    target_positions: usize,
    compared_logits: usize,
    max_absolute_logit_delta: f64,
    mean_absolute_logit_delta: f64,
    mean_nll_increase: f64,
    perplexity_ratio: f64,
    representative_top1_agreement: f64,
}

#[derive(Debug, Serialize)]
struct Receipt<'a> {
    schema: &'static str,
    decision: Decision,
    full_model_quality_accepted: bool,
    operator_correctness_is_separate: bool,
    checkpoint: &'a CheckpointIdentity,
    config_sha256: &'a Sha256Digest,
    quantizer: &'a QuantizerIdentity,
    reference_route: &'a RouteIdentity,
    candidate_route: &'a RouteIdentity,
    fixture_sha256: &'a Sha256Digest,
    reference_observations_sha256: &'a Sha256Digest,
    candidate_observations_sha256: &'a Sha256Digest,
    calibration_sha256: Option<&'a Sha256Digest>,
    calibration_protocol: Option<String>,
    calibration_corpus_sha256: Option<Sha256Digest>,
    calibration_rationale: Option<String>,
    operator_correctness_receipt_sha256: Option<&'a Sha256Digest>,
    metrics: Metrics,
    failed_gates: Vec<String>,
    limitations: Vec<String>,
}

fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    if args.len() != 2 && args.len() != 4 {
        eprintln!("usage: rvllm_gemma4_quality_referee INPUT.json [--require bounded_slice_evidence|full_model_accepted]");
        std::process::exit(2);
    }
    let required = if args.len() == 4 {
        if args[2] != "--require" {
            eprintln!("expected --require");
            std::process::exit(2);
        }
        match args[3].to_str() {
            Some("bounded_slice_evidence") => Some(Decision::BoundedSliceEvidence),
            Some("full_model_accepted") => Some(Decision::FullModelAccepted),
            _ => {
                eprintln!("invalid required decision");
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    match run(Path::new(&args[1])) {
        Ok(receipt) => {
            let matched = required.map_or(true, |value| value == receipt.decision);
            println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            if !matched {
                std::process::exit(3);
            }
        }
        Err(error) => {
            eprintln!("rvllm_gemma4_quality_referee: {error}");
            std::process::exit(1);
        }
    }
}

fn read_strict<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_strict_json(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn verify_file(identity: &FileIdentity) -> Result<(), String> {
    let actual = Sha256Digest::file(&identity.path).map_err(|e| e.to_string())?;
    if actual != identity.sha256 {
        return Err(format!(
            "sha256 mismatch for {}: expected {}, got {}",
            identity.path.display(),
            identity.sha256.as_str(),
            actual.as_str()
        ));
    }
    Ok(())
}

fn validate_identity(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(format!("invalid {label}"));
    }
    Ok(())
}

fn run(path: &Path) -> Result<Receipt<'static>, String> {
    // Leak one small manifest so the receipt can borrow sealed identities until process exit.
    let input: &'static Input = Box::leak(Box::new(read_strict(path)?));
    if input.schema != INPUT_SCHEMA {
        return Err(format!("wrong input schema: {}", input.schema));
    }
    for (label, value) in [
        ("repository", input.checkpoint.repository.as_str()),
        ("revision", input.checkpoint.revision.as_str()),
        ("quantizer", input.quantizer.name.as_str()),
        (
            "quantizer revision",
            input.quantizer.implementation_revision.as_str(),
        ),
    ] {
        validate_identity(label, value)?;
    }
    if !matches!(input.quantizer.weight_bits, 4 | 8) || input.quantizer.group_size == 0 {
        return Err("quality referee accepts only W4/W8 with nonzero group size".into());
    }
    for artifact in [
        &input.config,
        &input.fixture,
        &input.reference_observations,
        &input.candidate_observations,
    ] {
        verify_file(artifact)?;
    }
    if let Some(calibration) = &input.calibration {
        verify_file(calibration)?;
    }

    let fixture: Fixture = read_strict(&input.fixture.path)?;
    let reference: Observations = read_strict(&input.reference_observations.path)?;
    let candidate: Observations = read_strict(&input.candidate_observations.path)?;
    if fixture.schema != FIXTURE_SCHEMA
        || reference.schema != OBSERVATION_SCHEMA
        || candidate.schema != OBSERVATION_SCHEMA
    {
        return Err("wrong fixture or observation schema".into());
    }
    validate_observation_identity(input, &reference, &input.reference_route)?;
    validate_observation_identity(input, &candidate, &input.candidate_route)?;
    let metrics = compare(&fixture, &reference, &candidate)?;

    let mut failures = Vec::new();
    let mut limitations = vec!["operator correctness is an independent prerequisite and is not inferred from logits or perplexity".into()];
    let calibration = input
        .calibration
        .as_ref()
        .map(|f| read_strict::<Calibration>(&f.path))
        .transpose()?;
    if let Some(c) = &calibration {
        validate_calibration(c, &input.fixture.sha256)?;
        if metrics.target_positions < c.minimum_target_positions {
            failures.push("insufficient_target_positions".into());
        }
        if metrics.max_absolute_logit_delta > c.max_absolute_logit_delta {
            failures.push("max_absolute_logit_delta".into());
        }
        if metrics.mean_absolute_logit_delta > c.max_mean_absolute_logit_delta {
            failures.push("mean_absolute_logit_delta".into());
        }
        if metrics.mean_nll_increase > c.max_mean_nll_increase {
            failures.push("mean_nll_increase".into());
        }
        if metrics.perplexity_ratio > c.max_perplexity_ratio {
            failures.push("perplexity_ratio".into());
        }
        if metrics.representative_top1_agreement < c.minimum_representative_top1_agreement {
            failures.push("representative_top1_agreement".into());
        }
    } else {
        failures.push("missing_calibration".into());
        limitations.push("no acceptance thresholds were supplied; this receipt records metrics but cannot accept quality".into());
    }

    let roles = input
        .coverage
        .quantized_roles
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let required_roles = REQUIRED_ROLES.into_iter().collect::<BTreeSet<_>>();
    let all_roles = roles == required_roles;
    let full_route = input.coverage.full_checkpoint_quantized
        && all_roles
        && input.reference_route.full_model_route
        && input.candidate_route.full_model_route;
    if input.coverage.operator_correctness_receipt_sha256.is_none() {
        limitations.push("no independently sealed operator-correctness receipt was bound".into());
    }
    if !full_route {
        limitations.push("bounded or partial route: this receipt must not be used as full-model W4/W8 acceptance".into());
    }
    let decision = if calibration.is_none() {
        Decision::CalibrationRequired
    } else if !failures.is_empty() {
        Decision::Rejected
    } else if !full_route || input.coverage.operator_correctness_receipt_sha256.is_none() {
        Decision::BoundedSliceEvidence
    } else {
        Decision::FullModelAccepted
    };
    let accepted = matches!(decision, Decision::FullModelAccepted);
    Ok(Receipt {
        schema: OUTPUT_SCHEMA,
        decision,
        full_model_quality_accepted: accepted,
        operator_correctness_is_separate: true,
        checkpoint: &input.checkpoint,
        config_sha256: &input.config.sha256,
        quantizer: &input.quantizer,
        reference_route: &input.reference_route,
        candidate_route: &input.candidate_route,
        fixture_sha256: &input.fixture.sha256,
        reference_observations_sha256: &input.reference_observations.sha256,
        candidate_observations_sha256: &input.candidate_observations.sha256,
        calibration_sha256: input.calibration.as_ref().map(|v| &v.sha256),
        calibration_protocol: calibration.as_ref().map(|v| v.protocol.clone()),
        calibration_corpus_sha256: calibration
            .as_ref()
            .map(|v| v.calibration_corpus_sha256.clone()),
        calibration_rationale: calibration.as_ref().map(|v| v.rationale.clone()),
        operator_correctness_receipt_sha256: input
            .coverage
            .operator_correctness_receipt_sha256
            .as_ref(),
        metrics,
        failed_gates: failures,
        limitations,
    })
}

fn validate_observation_identity(
    input: &Input,
    value: &Observations,
    route: &RouteIdentity,
) -> Result<(), String> {
    if value.checkpoint_manifest_sha256 != input.checkpoint.manifest_sha256
        || value.config_sha256 != input.config.sha256
        || value.fixture_sha256 != input.fixture.sha256
    {
        return Err("observation checkpoint/config/fixture identity mismatch".into());
    }
    if value.route.name != route.name
        || value.route.source_tree_sha256 != route.source_tree_sha256
        || value.route.executable_sha256 != route.executable_sha256
        || value.route.model_package_sha256 != route.model_package_sha256
        || value.route.full_model_route != route.full_model_route
    {
        return Err("observation route identity mismatch".into());
    }
    Ok(())
}

fn validate_calibration(value: &Calibration, fixture_sha: &Sha256Digest) -> Result<(), String> {
    if value.schema != CALIBRATION_SCHEMA
        || value.protocol.trim().is_empty()
        || value.rationale.trim().is_empty()
        || value.minimum_target_positions == 0
        || &value.held_out_fixture_sha256 != fixture_sha
    {
        return Err("invalid or non-held-out calibration contract".into());
    }
    let thresholds = [
        value.max_absolute_logit_delta,
        value.max_mean_absolute_logit_delta,
        value.max_mean_nll_increase,
        value.max_perplexity_ratio,
        value.minimum_representative_top1_agreement,
    ];
    if thresholds.iter().any(|x| !x.is_finite())
        || value.max_absolute_logit_delta < 0.0
        || value.max_mean_absolute_logit_delta < 0.0
        || value.max_mean_nll_increase < 0.0
        || value.max_perplexity_ratio < 1.0
        || !(0.0..=1.0).contains(&value.minimum_representative_top1_agreement)
    {
        return Err("invalid calibration thresholds".into());
    }
    Ok(())
}

fn compare(
    fixture: &Fixture,
    reference: &Observations,
    candidate: &Observations,
) -> Result<Metrics, String> {
    let fixture_ids = fixture
        .cases
        .iter()
        .map(|c| c.id.as_str())
        .collect::<BTreeSet<_>>();
    if fixture_ids.len() != fixture.cases.len() || fixture.cases.is_empty() {
        return Err("fixture case ids must be unique and nonempty".into());
    }
    let refs = observations_by_id(reference, &fixture_ids)?;
    let cands = observations_by_id(candidate, &fixture_ids)?;
    let mut count = 0usize;
    let mut sum_delta = 0.0;
    let mut max_delta = 0.0_f64;
    let mut target_count = 0usize;
    let mut sum_ref_nll = 0.0;
    let mut sum_cand_nll = 0.0;
    let mut top1_matches = 0usize;
    for f in &fixture.cases {
        if f.token_ids.is_empty() || f.target_token_ids.len() != f.token_ids.len() {
            return Err(format!("fixture case {} is empty", f.id));
        }
        let r = refs[&f.id.as_str()];
        let c = cands[&f.id.as_str()];
        if r.positions.len() != f.target_token_ids.len()
            || c.positions.len() != f.target_token_ids.len()
        {
            return Err(format!("case {} target coverage mismatch", f.id));
        }
        for ((rpos, cpos), target) in r
            .positions
            .iter()
            .zip(&c.positions)
            .zip(&f.target_token_ids)
        {
            if rpos.target_token_id != *target || cpos.target_token_id != *target {
                return Err(format!("case {} target identity mismatch", f.id));
            }
            if !rpos.target_negative_log_likelihood.is_finite()
                || !cpos.target_negative_log_likelihood.is_finite()
                || rpos.target_negative_log_likelihood < 0.0
                || cpos.target_negative_log_likelihood < 0.0
            {
                return Err("invalid negative log likelihood".into());
            }
            if rpos.logits.keys().collect::<Vec<_>>() != cpos.logits.keys().collect::<Vec<_>>()
                || rpos.logits.is_empty()
            {
                return Err("representative logit token sets differ or are empty".into());
            }
            let mut rbest: Option<(u32, f64)> = None;
            let mut cbest: Option<(u32, f64)> = None;
            for (&token, &rv) in &rpos.logits {
                let cv = cpos.logits[&token];
                if !rv.is_finite() || !cv.is_finite() {
                    return Err("non-finite logit".into());
                }
                let delta = (rv - cv).abs();
                max_delta = max_delta.max(delta);
                sum_delta += delta;
                count += 1;
                if rbest.map_or(true, |(t, v)| rv > v || (rv == v && token < t)) {
                    rbest = Some((token, rv));
                }
                if cbest.map_or(true, |(t, v)| cv > v || (cv == v && token < t)) {
                    cbest = Some((token, cv));
                }
            }
            top1_matches += usize::from(rbest.unwrap().0 == cbest.unwrap().0);
            sum_ref_nll += rpos.target_negative_log_likelihood;
            sum_cand_nll += cpos.target_negative_log_likelihood;
            target_count += 1;
        }
    }
    let mean_ref_nll = sum_ref_nll / target_count as f64;
    let mean_cand_nll = sum_cand_nll / target_count as f64;
    Ok(Metrics {
        cases: fixture.cases.len(),
        target_positions: target_count,
        compared_logits: count,
        max_absolute_logit_delta: max_delta,
        mean_absolute_logit_delta: sum_delta / count as f64,
        mean_nll_increase: mean_cand_nll - mean_ref_nll,
        perplexity_ratio: (mean_cand_nll - mean_ref_nll).exp(),
        representative_top1_agreement: top1_matches as f64 / target_count as f64,
    })
}

fn observations_by_id<'a>(
    observations: &'a Observations,
    fixture_ids: &BTreeSet<&str>,
) -> Result<BTreeMap<&'a str, &'a ObservedCase>, String> {
    let map = observations
        .cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    if map.len() != observations.cases.len()
        || map.keys().copied().collect::<BTreeSet<_>>() != *fixture_ids
    {
        return Err("observation cases must exactly match fixture ids".into());
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_json_rejects_duplicate_keys() {
        let bytes = br#"{"schema":"x","schema":"y"}"#;
        assert!(parse_strict_json::<Observations>(bytes).is_err());
    }

    #[test]
    fn required_roles_are_stable_and_complete() {
        assert_eq!(REQUIRED_ROLES.len(), 7);
        assert!(REQUIRED_ROLES.contains(&"down_projection"));
    }
    #[test]
    fn calibration_rejects_invented_permissive_numbers() {
        let sha = Sha256Digest::bytes(b"fixture");
        let bad = Calibration {
            schema: CALIBRATION_SCHEMA.into(),
            protocol: "held-out".into(),
            calibration_corpus_sha256: Sha256Digest::bytes(b"cal"),
            held_out_fixture_sha256: sha.clone(),
            minimum_target_positions: 1,
            max_absolute_logit_delta: f64::INFINITY,
            max_mean_absolute_logit_delta: 0.0,
            max_mean_nll_increase: 0.0,
            max_perplexity_ratio: 1.0,
            minimum_representative_top1_agreement: 1.0,
            rationale: "measured".into(),
        };
        assert!(validate_calibration(&bad, &sha).is_err());
    }
}
