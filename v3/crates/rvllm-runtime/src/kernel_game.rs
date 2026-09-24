//! Host-only contracts for the kernel evaluation game.
//!
//! This module is deliberately model/device independent. It turns immutable
//! candidate/task identities and trusted receipts into monotonic evidence and
//! timing decisions. It never launches device work or promotes a selector.
#![forbid(unsafe_code)]

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

pub const SUBMISSION_SCHEMA: &str = "rvllm.kernel_game.submission.v1";
pub const EVIDENCE_SCHEMA: &str = "rvllm.kernel_game.evidence.v1";
pub const RESULT_SCHEMA: &str = "rvllm.kernel_game.result.v1";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn parse(value: impl Into<String>) -> Result<Self, KernelGameError> {
        let value = value.into();
        if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(KernelGameError::InvalidDigest(value));
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    pub fn bytes(bytes: &[u8]) -> Self {
        Self(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn file(path: &Path) -> Result<Self, KernelGameError> {
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        Ok(Self(format!("{:x}", hasher.finalize())))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateClass {
    MetalProjection,
    MetalAttention,
    MetalNorm,
    AneSingleIo,
    CpuTransform,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactIdentity {
    pub sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedSubmission {
    pub schema: String,
    pub id: String,
    pub revision: u32,
    pub class: CandidateClass,
    pub task_id: String,
    pub candidate: String,
    pub control: String,
    pub source_tree: Sha256Digest,
    pub executable: ArtifactIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallib: Option<ArtifactIdentity>,
    pub generated_source: Sha256Digest,
    pub model: Sha256Digest,
    pub reference: Sha256Digest,
    pub workload: Sha256Digest,
    pub oracle: Sha256Digest,
    #[serde(default)]
    pub expected_dispatch: BTreeMap<String, u64>,
}

impl SealedSubmission {
    pub fn validate(&self) -> Result<(), KernelGameError> {
        if self.schema != SUBMISSION_SCHEMA {
            return Err(KernelGameError::WrongSchema(self.schema.clone()));
        }
        for (label, value) in [
            ("id", self.id.as_str()),
            ("task_id", self.task_id.as_str()),
            ("candidate", self.candidate.as_str()),
            ("control", self.control.as_str()),
        ] {
            if value.is_empty()
                || value.len() > 128
                || !value
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/'))
            {
                return Err(KernelGameError::InvalidIdentity(label));
            }
        }
        if self.candidate == self.control {
            return Err(KernelGameError::CandidateEqualsControl);
        }
        if self
            .expected_dispatch
            .iter()
            .any(|(name, count)| name.is_empty() || *count == 0)
        {
            return Err(KernelGameError::InvalidDispatchContract);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStage {
    Submitted,
    SourceReady,
    Compiled,
    ComponentQualified,
    DispatchQualified,
    CacheReady,
    FullRouteQualified,
    TimingScreened,
    IndependentlyConfirmed,
}

impl EvidenceStage {
    pub fn permits(self, next: Self) -> bool {
        next >= self
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteEvidence {
    pub schema: String,
    pub submission_id: String,
    pub task_id: String,
    pub candidate: String,
    pub executable_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallib_sha256: Option<Sha256Digest>,
    pub generated_source_sha256: Sha256Digest,
    pub model_sha256: Sha256Digest,
    pub reference_sha256: Sha256Digest,
    pub workload_sha256: Sha256Digest,
    pub command_completed: bool,
    pub matches_reference: bool,
    pub compiler_calls: u64,
    pub dispatch_counts: BTreeMap<String, u64>,
    pub output_tokens: u64,
    pub decode_steps: u64,
}

impl RouteEvidence {
    pub fn validate_against(&self, submission: &SealedSubmission) -> Result<(), KernelGameError> {
        if self.schema != EVIDENCE_SCHEMA {
            return Err(KernelGameError::WrongSchema(self.schema.clone()));
        }
        let identities_match = self.submission_id == submission.id
            && self.task_id == submission.task_id
            && self.candidate == submission.candidate
            && self.executable_sha256 == submission.executable.sha256
            && self.metallib_sha256 == submission.metallib.as_ref().map(|a| a.sha256.clone())
            && self.generated_source_sha256 == submission.generated_source
            && self.model_sha256 == submission.model
            && self.reference_sha256 == submission.reference
            && self.workload_sha256 == submission.workload;
        if !identities_match {
            return Err(KernelGameError::IdentityMismatch);
        }
        if !self.command_completed || !self.matches_reference || self.compiler_calls != 0 {
            return Err(KernelGameError::RouteNotQualified);
        }
        if self.output_tokens == 0 || self.decode_steps + 1 < self.output_tokens {
            return Err(KernelGameError::InvalidWorkCount);
        }
        for (name, expected) in &submission.expected_dispatch {
            if self.dispatch_counts.get(name) != Some(expected) {
                return Err(KernelGameError::DispatchMismatch(name.clone()));
            }
        }
        let allowed: BTreeSet<_> = submission.expected_dispatch.keys().collect();
        if self
            .dispatch_counts
            .iter()
            .any(|(name, count)| *count != 0 && !allowed.contains(name))
        {
            return Err(KernelGameError::ForeignDispatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingArm {
    Control,
    Candidate,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimingSample {
    pub arm: TimingArm,
    pub elapsed_ms: f64,
    pub power_source: String,
    pub low_power_mode: bool,
    pub pmset_power_mode: u64,
    pub thermal_state: u64,
    pub compiler_calls: u64,
}

impl TimingSample {
    fn validate(&self) -> Result<(), KernelGameError> {
        if !self.elapsed_ms.is_finite() || self.elapsed_ms <= 0.0 {
            return Err(KernelGameError::InvalidTiming);
        }
        if self.compiler_calls != 0 {
            return Err(KernelGameError::CompileDuringTiming);
        }
        if !matches!(self.power_source.as_str(), "ac" | "battery") || self.thermal_state > 1 {
            return Err(KernelGameError::MixedTimingStratum);
        }
        Ok(())
    }

    fn same_stratum(&self, other: &Self) -> bool {
        self.power_source == other.power_source
            && self.low_power_mode == other.low_power_mode
            && self.pmset_power_mode == other.pmset_power_mode
            && self.thermal_state == other.thermal_state
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimingPlan {
    pub warmups_per_arm: u32,
    pub measured_blocks: u32,
    pub order: Vec<TimingArm>,
    pub max_control_drift_fraction: f64,
    pub practical_improvement_fraction: f64,
}

impl TimingPlan {
    pub fn abba(measured_blocks: u32, practical_improvement_fraction: f64) -> Self {
        let mut order = Vec::with_capacity(measured_blocks as usize * 4);
        for _ in 0..measured_blocks {
            order.extend([
                TimingArm::Control,
                TimingArm::Candidate,
                TimingArm::Candidate,
                TimingArm::Control,
            ]);
        }
        Self {
            warmups_per_arm: 2,
            measured_blocks,
            order,
            max_control_drift_fraction: 0.05,
            practical_improvement_fraction,
        }
    }

    pub fn validate(&self) -> Result<(), KernelGameError> {
        if self.warmups_per_arm == 0
            || self.measured_blocks == 0
            || self.order.len() != self.measured_blocks as usize * 4
            || !self.max_control_drift_fraction.is_finite()
            || !(0.0..=0.25).contains(&self.max_control_drift_fraction)
            || !self.practical_improvement_fraction.is_finite()
            || !(0.0..=0.50).contains(&self.practical_improvement_fraction)
        {
            return Err(KernelGameError::InvalidTimingPlan);
        }
        for chunk in self.order.chunks_exact(4) {
            if chunk
                != [
                    TimingArm::Control,
                    TimingArm::Candidate,
                    TimingArm::Candidate,
                    TimingArm::Control,
                ]
            {
                return Err(KernelGameError::InvalidTimingPlan);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TimingScore {
    pub control_median_ms: f64,
    pub candidate_median_ms: f64,
    pub improvement_fraction: f64,
    pub maximum_control_drift_fraction: f64,
    pub practical_improvement_met: bool,
}

pub fn score_timing(
    plan: &TimingPlan,
    samples: &[TimingSample],
) -> Result<TimingScore, KernelGameError> {
    plan.validate()?;
    if samples.len() != plan.order.len() {
        return Err(KernelGameError::WrongTimingSampleCount);
    }
    for (sample, expected) in samples.iter().zip(&plan.order) {
        sample.validate()?;
        if sample.arm != *expected {
            return Err(KernelGameError::TimingOrderMismatch);
        }
    }
    if samples
        .windows(2)
        .any(|pair| !pair[0].same_stratum(&pair[1]))
    {
        return Err(KernelGameError::MixedTimingStratum);
    }

    let control: Vec<_> = samples
        .iter()
        .filter(|s| s.arm == TimingArm::Control)
        .map(|s| s.elapsed_ms)
        .collect();
    let candidate: Vec<_> = samples
        .iter()
        .filter(|s| s.arm == TimingArm::Candidate)
        .map(|s| s.elapsed_ms)
        .collect();
    let control_median_ms = median(control.clone());
    let candidate_median_ms = median(candidate);
    let improvement_fraction = (control_median_ms - candidate_median_ms) / control_median_ms;

    let anchor = control[0];
    let maximum_control_drift_fraction = control
        .iter()
        .map(|value| (value - anchor).abs() / anchor)
        .fold(0.0_f64, f64::max);
    if maximum_control_drift_fraction > plan.max_control_drift_fraction {
        return Err(KernelGameError::ControlDrift {
            observed: maximum_control_drift_fraction,
            allowed: plan.max_control_drift_fraction,
        });
    }
    Ok(TimingScore {
        control_median_ms,
        candidate_median_ms,
        improvement_fraction,
        maximum_control_drift_fraction,
        practical_improvement_met: improvement_fraction >= plan.practical_improvement_fraction,
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcome {
    Promotable,
    Deferred,
    Failed,
    Blocked,
    Inconclusive,
}

#[derive(Clone, Debug, Serialize)]
pub struct PublishedResult {
    pub schema: &'static str,
    pub submission_id: String,
    pub highest_stage: EvidenceStage,
    pub outcome: TerminalOutcome,
    pub blockers: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingScore>,
}

pub fn reduce_result(
    submission: &SealedSubmission,
    route: Option<&RouteEvidence>,
    timing: Option<(&TimingPlan, &[TimingSample])>,
    independently_confirmed: bool,
) -> Result<PublishedResult, KernelGameError> {
    submission.validate()?;
    let Some(route) = route else {
        return Ok(PublishedResult {
            schema: RESULT_SCHEMA,
            submission_id: submission.id.clone(),
            highest_stage: EvidenceStage::Compiled,
            outcome: TerminalOutcome::Blocked,
            blockers: vec!["missing_full_route_evidence".into()],
            timing: None,
        });
    };
    route.validate_against(submission)?;

    let Some((plan, samples)) = timing else {
        return Ok(PublishedResult {
            schema: RESULT_SCHEMA,
            submission_id: submission.id.clone(),
            highest_stage: EvidenceStage::FullRouteQualified,
            outcome: TerminalOutcome::Deferred,
            blockers: vec!["timing_not_run".into()],
            timing: None,
        });
    };
    let score = score_timing(plan, samples)?;
    if !score.practical_improvement_met {
        return Ok(PublishedResult {
            schema: RESULT_SCHEMA,
            submission_id: submission.id.clone(),
            highest_stage: EvidenceStage::TimingScreened,
            outcome: TerminalOutcome::Inconclusive,
            blockers: vec!["practical_improvement_not_met".into()],
            timing: Some(score),
        });
    }
    if !independently_confirmed {
        return Ok(PublishedResult {
            schema: RESULT_SCHEMA,
            submission_id: submission.id.clone(),
            highest_stage: EvidenceStage::TimingScreened,
            outcome: TerminalOutcome::Deferred,
            blockers: vec!["independent_confirmation_required".into()],
            timing: Some(score),
        });
    }
    Ok(PublishedResult {
        schema: RESULT_SCHEMA,
        submission_id: submission.id.clone(),
        highest_stage: EvidenceStage::IndependentlyConfirmed,
        outcome: TerminalOutcome::Promotable,
        blockers: Vec::new(),
        timing: Some(score),
    })
}

/// Parse JSON recursively while rejecting duplicate object keys before serde
/// gets an opportunity to overwrite the earlier value.
pub fn parse_strict_json<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, KernelGameError> {
    struct StrictValue(Value);

    impl<'de> Deserialize<'de> for StrictValue {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            use serde::de::{MapAccess, SeqAccess, Visitor};
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = Value;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("valid JSON without duplicate object keys")
                }
                fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
                    Ok(Value::Bool(v))
                }
                fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
                    Ok(Value::Number(v.into()))
                }
                fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
                    Ok(Value::Number(v.into()))
                }
                fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Value, E> {
                    serde_json::Number::from_f64(v)
                        .map(Value::Number)
                        .ok_or_else(|| E::custom("non-finite JSON number"))
                }
                fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Value, E> {
                    Ok(Value::String(v.to_owned()))
                }
                fn visit_string<E>(self, v: String) -> Result<Value, E> {
                    Ok(Value::String(v))
                }
                fn visit_none<E>(self) -> Result<Value, E> {
                    Ok(Value::Null)
                }
                fn visit_unit<E>(self) -> Result<Value, E> {
                    Ok(Value::Null)
                }
                fn visit_some<D2: serde::Deserializer<'de>>(
                    self,
                    d: D2,
                ) -> Result<Value, D2::Error> {
                    StrictValue::deserialize(d).map(|v| v.0)
                }
                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
                    let mut values = Vec::new();
                    while let Some(value) = seq.next_element::<StrictValue>()? {
                        values.push(value.0);
                    }
                    Ok(Value::Array(values))
                }
                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
                    let mut values = serde_json::Map::new();
                    while let Some(key) = map.next_key::<String>()? {
                        if values.contains_key(&key) {
                            return Err(serde::de::Error::custom(format!(
                                "duplicate JSON key: {key}"
                            )));
                        }
                        let value = map.next_value::<StrictValue>()?;
                        values.insert(key, value.0);
                    }
                    Ok(Value::Object(values))
                }
            }
            deserializer.deserialize_any(V).map(StrictValue)
        }
    }

    let mut de = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValue::deserialize(&mut de)
        .map_err(|e| KernelGameError::Json(e.to_string()))?;
    de.end().map_err(|e| KernelGameError::Json(e.to_string()))?;
    serde_json::from_value(value.0).map_err(|e| KernelGameError::Json(e.to_string()))
}

#[derive(Debug)]
pub enum KernelGameError {
    Io(std::io::Error),
    Json(String),
    WrongSchema(String),
    InvalidDigest(String),
    InvalidIdentity(&'static str),
    CandidateEqualsControl,
    InvalidDispatchContract,
    IdentityMismatch,
    RouteNotQualified,
    InvalidWorkCount,
    DispatchMismatch(String),
    ForeignDispatch,
    InvalidTiming,
    CompileDuringTiming,
    MixedTimingStratum,
    InvalidTimingPlan,
    WrongTimingSampleCount,
    TimingOrderMismatch,
    ControlDrift { observed: f64, allowed: f64 },
}

impl fmt::Display for KernelGameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Json(e) => write!(f, "{e}"),
            Self::WrongSchema(v) => write!(f, "wrong kernel-game schema: {v}"),
            Self::InvalidDigest(v) => write!(f, "invalid SHA-256 digest: {v}"),
            Self::InvalidIdentity(v) => write!(f, "invalid identity field: {v}"),
            Self::CandidateEqualsControl => f.write_str("candidate must differ from control"),
            Self::InvalidDispatchContract => f.write_str("invalid positive dispatch contract"),
            Self::IdentityMismatch => f.write_str("route evidence does not match sealed submission"),
            Self::RouteNotQualified => f.write_str("route did not complete, match, or remain compile-free"),
            Self::InvalidWorkCount => f.write_str("route work counts are impossible"),
            Self::DispatchMismatch(v) => write!(f, "dispatch count mismatch for {v}"),
            Self::ForeignDispatch => f.write_str("route contains foreign candidate dispatch"),
            Self::InvalidTiming => f.write_str("timing sample must be finite and positive"),
            Self::CompileDuringTiming => f.write_str("compiler call observed during timing"),
            Self::MixedTimingStratum => f.write_str("timing samples cross environment strata"),
            Self::InvalidTimingPlan => f.write_str("invalid timing plan"),
            Self::WrongTimingSampleCount => f.write_str("timing sample count differs from plan"),
            Self::TimingOrderMismatch => f.write_str("timing sample order differs from plan"),
            Self::ControlDrift { observed, allowed } => {
                write!(f, "control drift {observed:.6} exceeds {allowed:.6}")
            }
        }
    }
}

impl std::error::Error for KernelGameError {}

impl From<std::io::Error> for KernelGameError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn submission() -> SealedSubmission {
        SealedSubmission {
            schema: SUBMISSION_SCHEMA.into(),
            id: "prefetch-r1".into(),
            revision: 1,
            class: CandidateClass::MetalProjection,
            task_id: "gemma4-prefill-projection".into(),
            candidate: "metal-mma32-prefetch".into(),
            control: "off".into(),
            source_tree: digest(1),
            executable: ArtifactIdentity { sha256: digest(2), path: None },
            metallib: Some(ArtifactIdentity { sha256: digest(3), path: None }),
            generated_source: digest(4),
            model: digest(5),
            reference: digest(6),
            workload: digest(7),
            oracle: digest(8),
            expected_dispatch: BTreeMap::from([
                ("research_gemm_mma32_prefetch".into(), 144),
                ("research_qkv_mma32_prefetch".into(), 48),
            ]),
        }
    }

    fn route() -> RouteEvidence {
        let s = submission();
        RouteEvidence {
            schema: EVIDENCE_SCHEMA.into(),
            submission_id: s.id,
            task_id: s.task_id,
            candidate: s.candidate,
            executable_sha256: digest(2),
            metallib_sha256: Some(digest(3)),
            generated_source_sha256: digest(4),
            model_sha256: digest(5),
            reference_sha256: digest(6),
            workload_sha256: digest(7),
            command_completed: true,
            matches_reference: true,
            compiler_calls: 0,
            dispatch_counts: BTreeMap::from([
                ("research_gemm_mma32_prefetch".into(), 144),
                ("research_qkv_mma32_prefetch".into(), 48),
            ]),
            output_tokens: 10,
            decode_steps: 9,
        }
    }

    #[test]
    fn strict_json_rejects_duplicate_keys_before_typed_deserialization() {
        let bad = br#"{"schema":"rvllm.kernel_game.submission.v1","schema":"x"}"#;
        assert!(parse_strict_json::<SealedSubmission>(bad).is_err());
    }

    #[test]
    fn route_identity_and_exact_dispatch_are_hard_prerequisites() {
        let s = submission();
        let mut r = route();
        assert!(r.validate_against(&s).is_ok());
        r.executable_sha256 = digest(99);
        assert!(matches!(
            r.validate_against(&s),
            Err(KernelGameError::IdentityMismatch)
        ));
        let mut r = route();
        r.dispatch_counts
            .insert("research_gemm_mma32_prefetch".into(), 143);
        assert!(matches!(
            r.validate_against(&s),
            Err(KernelGameError::DispatchMismatch(_))
        ));
    }

    fn sample(arm: TimingArm, ms: f64) -> TimingSample {
        TimingSample {
            arm,
            elapsed_ms: ms,
            power_source: "ac".into(),
            low_power_mode: true,
            pmset_power_mode: 1,
            thermal_state: 0,
            compiler_calls: 0,
        }
    }

    #[test]
    fn abba_scoring_requires_order_stratum_zero_compiles_and_drift_gate() {
        let plan = TimingPlan::abba(2, 0.02);
        let samples = [
            sample(TimingArm::Control, 100.0),
            sample(TimingArm::Candidate, 90.0),
            sample(TimingArm::Candidate, 91.0),
            sample(TimingArm::Control, 101.0),
            sample(TimingArm::Control, 100.5),
            sample(TimingArm::Candidate, 89.5),
            sample(TimingArm::Candidate, 90.5),
            sample(TimingArm::Control, 99.5),
        ];
        let score = score_timing(&plan, &samples).unwrap();
        assert!(score.practical_improvement_met);
        let mut bad = samples.to_vec();
        bad[3].elapsed_ms = 110.0;
        assert!(matches!(
            score_timing(&plan, &bad),
            Err(KernelGameError::ControlDrift { .. })
        ));
        let mut bad = samples.to_vec();
        bad[2].compiler_calls = 1;
        assert!(matches!(
            score_timing(&plan, &bad),
            Err(KernelGameError::CompileDuringTiming)
        ));
    }

    #[test]
    fn promotion_eligibility_requires_route_timing_effect_and_confirmation() {
        let s = submission();
        let r = route();
        let plan = TimingPlan::abba(1, 0.02);
        let samples = [
            sample(TimingArm::Control, 100.0),
            sample(TimingArm::Candidate, 90.0),
            sample(TimingArm::Candidate, 91.0),
            sample(TimingArm::Control, 101.0),
        ];
        let unconfirmed = reduce_result(&s, Some(&r), Some((&plan, &samples)), false).unwrap();
        assert_eq!(unconfirmed.outcome, TerminalOutcome::Deferred);
        let confirmed = reduce_result(&s, Some(&r), Some((&plan, &samples)), true).unwrap();
        assert_eq!(confirmed.outcome, TerminalOutcome::Promotable);
        assert_eq!(confirmed.highest_stage, EvidenceStage::IndependentlyConfirmed);
    }

    #[test]
    fn missing_route_is_blocked_not_a_candidate_failure() {
        let result = reduce_result(&submission(), None, None, false).unwrap();
        assert_eq!(result.outcome, TerminalOutcome::Blocked);
        assert_eq!(result.blockers, ["missing_full_route_evidence"]);
    }
}
