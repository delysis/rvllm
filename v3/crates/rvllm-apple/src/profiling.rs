use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BenchmarkCategory {
    MetalOnly,
    AnePartitioned,
    FallbackPath,
    ToyDisabled,
}

impl BenchmarkCategory {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MetalOnly => "metal_only",
            Self::AnePartitioned => "ane_partitioned",
            Self::FallbackPath => "fallback_path",
            Self::ToyDisabled => "toy_disabled",
        }
    }
}

pub const ROADMAP_BENCHMARK_CATEGORIES: &[BenchmarkCategory] = &[
    BenchmarkCategory::MetalOnly,
    BenchmarkCategory::AnePartitioned,
    BenchmarkCategory::FallbackPath,
    BenchmarkCategory::ToyDisabled,
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OptionalMetric<T> {
    Measured { value: T },
    Unmeasured { reason: String },
    Unsupported { reason: String },
}

impl<T> OptionalMetric<T> {
    #[must_use]
    pub fn measured(value: T) -> Self {
        Self::Measured { value }
    }

    #[must_use]
    pub fn unmeasured(reason: impl Into<String>) -> Self {
        Self::Unmeasured {
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported {
            reason: reason.into(),
        }
    }

    #[must_use]
    pub const fn is_measured(&self) -> bool {
        matches!(self, Self::Measured { .. })
    }

    #[must_use]
    pub fn is_recorded_honestly(&self) -> bool {
        match self {
            Self::Measured { .. } => true,
            Self::Unmeasured { reason } | Self::Unsupported { reason } => !reason.trim().is_empty(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackendProfileMetrics {
    pub first_token_latency_ms: OptionalMetric<f64>,
    pub steady_decode_tokens_per_second: OptionalMetric<f64>,
    pub prefill_tokens_per_second: OptionalMetric<f64>,
    pub memory_peak_bytes: OptionalMetric<u64>,
    pub command_buffers_per_token: OptionalMetric<f64>,
    pub cpu_utilization_percent: OptionalMetric<f64>,
    pub gpu_utilization_percent: OptionalMetric<f64>,
    pub ane_utilization_percent: OptionalMetric<f64>,
    pub energy_joules: OptionalMetric<f64>,
}

impl BackendProfileMetrics {
    #[must_use]
    pub fn core_metrics_measured(&self) -> bool {
        self.first_token_latency_ms.is_measured()
            && self.steady_decode_tokens_per_second.is_measured()
            && self.prefill_tokens_per_second.is_measured()
            && self.memory_peak_bytes.is_measured()
            && self.command_buffers_per_token.is_measured()
            && self.cpu_utilization_percent.is_measured()
            && self.gpu_utilization_percent.is_measured()
    }

    #[must_use]
    pub fn all_metric_slots_recorded_honestly(&self) -> bool {
        self.first_token_latency_ms.is_recorded_honestly()
            && self.steady_decode_tokens_per_second.is_recorded_honestly()
            && self.prefill_tokens_per_second.is_recorded_honestly()
            && self.memory_peak_bytes.is_recorded_honestly()
            && self.command_buffers_per_token.is_recorded_honestly()
            && self.cpu_utilization_percent.is_recorded_honestly()
            && self.gpu_utilization_percent.is_recorded_honestly()
            && self.ane_utilization_percent.is_recorded_honestly()
            && self.energy_joules.is_recorded_honestly()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackendProfileSample {
    pub sample_id: String,
    pub category: BenchmarkCategory,
    pub backend_name: String,
    pub model_id: String,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub toy_backend_enabled: bool,
    pub metrics: BackendProfileMetrics,
}

impl BackendProfileSample {
    #[must_use]
    pub fn new(
        sample_id: impl Into<String>,
        category: BenchmarkCategory,
        backend_name: impl Into<String>,
        model_id: impl Into<String>,
        metrics: BackendProfileMetrics,
    ) -> Self {
        Self {
            sample_id: sample_id.into(),
            category,
            backend_name: backend_name.into(),
            model_id: model_id.into(),
            prompt_tokens: 0,
            generated_tokens: 0,
            toy_backend_enabled: false,
            metrics,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EvidenceState {
    Present { evidence_id: String },
    Missing { reason: String },
    Failed { reason: String },
}

impl EvidenceState {
    #[must_use]
    pub fn present(evidence_id: impl Into<String>) -> Self {
        Self::Present {
            evidence_id: evidence_id.into(),
        }
    }

    #[must_use]
    pub fn missing(reason: impl Into<String>) -> Self {
        Self::Missing {
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn failed(reason: impl Into<String>) -> Self {
        Self::Failed {
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn is_present(&self) -> bool {
        matches!(self, Self::Present { evidence_id } if !evidence_id.trim().is_empty())
    }

    #[must_use]
    pub fn failure_reason(&self, missing_label: &'static str) -> Option<String> {
        match self {
            Self::Present { evidence_id } if evidence_id.trim().is_empty() => {
                Some(format!("{missing_label}: evidence_id is empty"))
            }
            Self::Present { .. } => None,
            Self::Missing { reason } => Some(format!("{missing_label}: {reason}")),
            Self::Failed { reason } => Some(reason.clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PerformanceRegressionEvidence {
    SampleComparison {
        baseline_sample_id: String,
        current_sample_id: String,
        max_allowed_regression_percent: f64,
        observed_regression_percent: f64,
    },
    ComparableEvidence {
        evidence_id: String,
        description: String,
    },
    NotTracked {
        reason: String,
    },
}

impl PerformanceRegressionEvidence {
    #[must_use]
    pub fn is_tracked(&self) -> bool {
        match self {
            Self::SampleComparison {
                baseline_sample_id,
                current_sample_id,
                ..
            } => !baseline_sample_id.trim().is_empty() && !current_sample_id.trim().is_empty(),
            Self::ComparableEvidence {
                evidence_id,
                description,
            } => !evidence_id.trim().is_empty() && !description.trim().is_empty(),
            Self::NotTracked { .. } => false,
        }
    }

    #[must_use]
    pub fn failure_reason(&self) -> Option<String> {
        match self {
            Self::SampleComparison {
                baseline_sample_id,
                current_sample_id,
                observed_regression_percent,
                max_allowed_regression_percent,
            } => {
                if baseline_sample_id.trim().is_empty() || current_sample_id.trim().is_empty() {
                    Some(
                        "performance regression tracking requires non-empty baseline and current sample IDs"
                            .to_string(),
                    )
                } else if observed_regression_percent > max_allowed_regression_percent {
                    Some(format!(
                        "observed regression {observed_regression_percent:.2}% exceeds allowed {max_allowed_regression_percent:.2}%"
                    ))
                } else {
                    None
                }
            }
            Self::ComparableEvidence {
                evidence_id,
                description,
            } => {
                if evidence_id.trim().is_empty() || description.trim().is_empty() {
                    Some(
                        "performance regression tracking requires a non-empty comparable evidence ID and description"
                            .to_string(),
                    )
                } else {
                    None
                }
            }
            Self::NotTracked { reason } => {
                Some(format!("performance regressions are not tracked: {reason}"))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppleProductionAcceptanceEvidence {
    pub evidence_id: String,
    pub samples: Vec<BackendProfileSample>,
    pub correctness_against_reference: EvidenceState,
    pub default_toy_path_disabled: EvidenceState,
    pub unsupported_models_fail_clearly: EvidenceState,
    pub no_hot_path_allocation: EvidenceState,
    pub no_hot_path_pipeline_compilation: EvidenceState,
    pub direct_backend_smoke: EvidenceState,
    pub engine_smoke: EvidenceState,
    pub performance_regression: PerformanceRegressionEvidence,
}

impl AppleProductionAcceptanceEvidence {
    #[must_use]
    pub fn current_incomplete() -> Self {
        Self {
            evidence_id: "current-apple-backend-incomplete".to_string(),
            samples: Vec::new(),
            correctness_against_reference: EvidenceState::missing(
                "no reference-model correctness evidence supplied for production profiling",
            ),
            default_toy_path_disabled: EvidenceState::missing(
                "default production path has not supplied toy-disabled evidence",
            ),
            unsupported_models_fail_clearly: EvidenceState::missing(
                "unsupported-model failure evidence was not supplied",
            ),
            no_hot_path_allocation: EvidenceState::missing(
                "hot-path allocation audit evidence was not supplied",
            ),
            no_hot_path_pipeline_compilation: EvidenceState::missing(
                "hot-path pipeline-compilation audit evidence was not supplied",
            ),
            direct_backend_smoke: EvidenceState::missing("direct backend smoke was not supplied"),
            engine_smoke: EvidenceState::missing("engine smoke was not supplied"),
            performance_regression: PerformanceRegressionEvidence::NotTracked {
                reason: "no baseline/current sample IDs or comparable evidence supplied"
                    .to_string(),
            },
        }
    }

    #[must_use]
    pub fn current_real_e2b_probe() -> Self {
        let mut metal_probe_sample = BackendProfileSample::new(
            "real-e2b-metal-probe-command-buffer-combined-2026-05-18",
            BenchmarkCategory::MetalOnly,
            "rvllm-runtime-model-metal-backend-probe",
            "google/gemma-4-E2B",
            BackendProfileMetrics {
                first_token_latency_ms: OptionalMetric::unmeasured(
                    "probe harness reports aggregate prefill and decode wall time, not isolated first-token latency",
                ),
                steady_decode_tokens_per_second: OptionalMetric::measured(2.2637),
                prefill_tokens_per_second: OptionalMetric::measured(3.4130),
                memory_peak_bytes: OptionalMetric::unmeasured(
                    "probe reports planned arena bytes separately; no external peak RSS/GPU allocation profile was captured",
                ),
                command_buffers_per_token: OptionalMetric::measured(0.8333),
                cpu_utilization_percent: OptionalMetric::unmeasured(
                    "no Instruments or powermetrics CPU utilization capture was attached",
                ),
                gpu_utilization_percent: OptionalMetric::unmeasured(
                    "no Metal System Trace GPU utilization capture was attached",
                ),
                ane_utilization_percent: OptionalMetric::unsupported(
                    "real E2B probe currently uses Metal only; private ANE execution is not established",
                ),
                energy_joules: OptionalMetric::unmeasured(
                    "energy measurement requires an external powermetrics capture",
                ),
            },
        );
        metal_probe_sample.prompt_tokens = 2;
        metal_probe_sample.generated_tokens = 4;

        Self {
            evidence_id: "current-real-e2b-probe-partial".to_string(),
            samples: vec![metal_probe_sample],
            correctness_against_reference: EvidenceState::present(
                "real-e2b-full-vocab-hf-parity-prompts-and-forced-decode-2026-05-18",
            ),
            default_toy_path_disabled: EvidenceState::missing(
                "direct probe tests bypass production default routing; no production-default toy-disabled rollout evidence supplied",
            ),
            unsupported_models_fail_clearly: EvidenceState::present(
                "real-e2b-large-model-default-gate-and-dry-run-validation-errors",
            ),
            no_hot_path_allocation: EvidenceState::present(
                "metal-probe-arena-region-count-stable-after-rollout",
            ),
            no_hot_path_pipeline_compilation: EvidenceState::present(
                "metal-probe-pipeline-compile-counters-stable-after-rollout",
            ),
            direct_backend_smoke: EvidenceState::present(
                "real-e2b-direct-model-metal-backend-full-vocab-parity",
            ),
            engine_smoke: EvidenceState::present(
                "real-e2b-engine-batch-two-full-vocab-hf-parity",
            ),
            performance_regression: PerformanceRegressionEvidence::NotTracked {
                reason: "single-host probe measurement has no baseline/current comparison or dashboard evidence"
                    .to_string(),
            },
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AcceptanceCriterion {
    BenchmarkCoverage,
    ProfileMetrics,
    CorrectnessAgainstReference,
    DefaultToyPathDisabled,
    UnsupportedModelsFailClearly,
    NoHotPathAllocation,
    NoHotPathPipelineCompilation,
    PerformanceRegressionsTracked,
}

impl AcceptanceCriterion {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BenchmarkCoverage => "benchmark_coverage",
            Self::ProfileMetrics => "profile_metrics",
            Self::CorrectnessAgainstReference => "correctness_against_reference",
            Self::DefaultToyPathDisabled => "default_toy_path_disabled",
            Self::UnsupportedModelsFailClearly => "unsupported_models_fail_clearly",
            Self::NoHotPathAllocation => "no_hot_path_allocation",
            Self::NoHotPathPipelineCompilation => "no_hot_path_pipeline_compilation",
            Self::PerformanceRegressionsTracked => "performance_regressions_tracked",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceFailure {
    pub criterion: AcceptanceCriterion,
    pub reason: String,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProductionCandidateStatus {
    ProductionCandidate,
    NotProductionCandidate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleProductionAcceptanceReport {
    pub evidence_id: String,
    pub status: ProductionCandidateStatus,
    pub failures: Vec<AcceptanceFailure>,
}

impl AppleProductionAcceptanceReport {
    #[must_use]
    pub fn is_production_candidate(&self) -> bool {
        self.status == ProductionCandidateStatus::ProductionCandidate
    }
}

#[must_use]
pub fn evaluate_apple_production_acceptance(
    evidence: &AppleProductionAcceptanceEvidence,
) -> AppleProductionAcceptanceReport {
    let mut failures = Vec::new();

    for category in ROADMAP_BENCHMARK_CATEGORIES {
        if !evidence
            .samples
            .iter()
            .any(|sample| sample.category == *category)
        {
            failures.push(AcceptanceFailure {
                criterion: AcceptanceCriterion::BenchmarkCoverage,
                reason: format!("missing {} benchmark sample", category.as_str()),
            });
        }
    }

    for sample in &evidence.samples {
        if sample.sample_id.trim().is_empty() {
            failures.push(AcceptanceFailure {
                criterion: AcceptanceCriterion::BenchmarkCoverage,
                reason: format!("{} sample has an empty sample_id", sample.category.as_str()),
            });
        }
        if sample.toy_backend_enabled {
            failures.push(AcceptanceFailure {
                criterion: AcceptanceCriterion::DefaultToyPathDisabled,
                reason: format!("sample {} used the toy backend", sample.sample_id),
            });
        }
        if !sample.metrics.core_metrics_measured() {
            failures.push(AcceptanceFailure {
                criterion: AcceptanceCriterion::ProfileMetrics,
                reason: format!(
                    "sample {} is missing measured core latency/throughput/memory/CPU/GPU/command-buffer metrics",
                    sample.sample_id
                ),
            });
        }
        if !sample.metrics.all_metric_slots_recorded_honestly() {
            failures.push(AcceptanceFailure {
                criterion: AcceptanceCriterion::ProfileMetrics,
                reason: format!(
                    "sample {} has unmeasured or unsupported metrics without reasons",
                    sample.sample_id
                ),
            });
        }
    }

    push_evidence_failure(
        &mut failures,
        AcceptanceCriterion::CorrectnessAgainstReference,
        evidence
            .correctness_against_reference
            .failure_reason("correctness against reference is missing"),
    );
    push_evidence_failure(
        &mut failures,
        AcceptanceCriterion::DefaultToyPathDisabled,
        evidence
            .default_toy_path_disabled
            .failure_reason("default toy-disabled evidence is missing"),
    );
    push_evidence_failure(
        &mut failures,
        AcceptanceCriterion::UnsupportedModelsFailClearly,
        evidence
            .unsupported_models_fail_clearly
            .failure_reason("unsupported-model failure evidence is missing"),
    );
    push_evidence_failure(
        &mut failures,
        AcceptanceCriterion::NoHotPathAllocation,
        evidence
            .no_hot_path_allocation
            .failure_reason("hot-path allocation evidence is missing"),
    );
    push_evidence_failure(
        &mut failures,
        AcceptanceCriterion::NoHotPathPipelineCompilation,
        evidence
            .no_hot_path_pipeline_compilation
            .failure_reason("hot-path pipeline-compilation evidence is missing"),
    );
    push_evidence_failure(
        &mut failures,
        AcceptanceCriterion::CorrectnessAgainstReference,
        evidence
            .direct_backend_smoke
            .failure_reason("direct backend smoke evidence is missing"),
    );
    push_evidence_failure(
        &mut failures,
        AcceptanceCriterion::CorrectnessAgainstReference,
        evidence
            .engine_smoke
            .failure_reason("engine smoke evidence is missing"),
    );

    if let Some(reason) = evidence.performance_regression.failure_reason() {
        failures.push(AcceptanceFailure {
            criterion: AcceptanceCriterion::PerformanceRegressionsTracked,
            reason,
        });
    }

    let status = if failures.is_empty() {
        ProductionCandidateStatus::ProductionCandidate
    } else {
        ProductionCandidateStatus::NotProductionCandidate
    };

    AppleProductionAcceptanceReport {
        evidence_id: evidence.evidence_id.clone(),
        status,
        failures,
    }
}

#[must_use]
pub fn current_apple_production_acceptance_report() -> AppleProductionAcceptanceReport {
    evaluate_apple_production_acceptance(&AppleProductionAcceptanceEvidence::current_incomplete())
}

#[must_use]
pub fn current_real_e2b_probe_acceptance_report() -> AppleProductionAcceptanceReport {
    evaluate_apple_production_acceptance(
        &AppleProductionAcceptanceEvidence::current_real_e2b_probe(),
    )
}

fn push_evidence_failure(
    failures: &mut Vec<AcceptanceFailure>,
    criterion: AcceptanceCriterion,
    reason: Option<String>,
) {
    if let Some(reason) = reason {
        failures.push(AcceptanceFailure { criterion, reason });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_metrics() -> BackendProfileMetrics {
        BackendProfileMetrics {
            first_token_latency_ms: OptionalMetric::measured(10.0),
            steady_decode_tokens_per_second: OptionalMetric::measured(20.0),
            prefill_tokens_per_second: OptionalMetric::measured(30.0),
            memory_peak_bytes: OptionalMetric::measured(1024),
            command_buffers_per_token: OptionalMetric::measured(2.0),
            cpu_utilization_percent: OptionalMetric::measured(12.5),
            gpu_utilization_percent: OptionalMetric::measured(45.0),
            ane_utilization_percent: OptionalMetric::unmeasured(
                "ANE utilization is not observable through public profiler counters",
            ),
            energy_joules: OptionalMetric::unmeasured(
                "energy measurement requires an external powermetrics capture",
            ),
        }
    }

    fn sample(category: BenchmarkCategory, id: &str) -> BackendProfileSample {
        let mut sample = BackendProfileSample::new(
            id,
            category,
            category.as_str(),
            "reference-model",
            complete_metrics(),
        );
        sample.prompt_tokens = 16;
        sample.generated_tokens = 8;
        sample
    }

    fn complete_evidence() -> AppleProductionAcceptanceEvidence {
        AppleProductionAcceptanceEvidence {
            evidence_id: "complete-evidence".to_string(),
            samples: ROADMAP_BENCHMARK_CATEGORIES
                .iter()
                .map(|category| sample(*category, category.as_str()))
                .collect(),
            correctness_against_reference: EvidenceState::present("correctness-report"),
            default_toy_path_disabled: EvidenceState::present("toy-disabled-report"),
            unsupported_models_fail_clearly: EvidenceState::present("unsupported-model-report"),
            no_hot_path_allocation: EvidenceState::present("allocation-audit"),
            no_hot_path_pipeline_compilation: EvidenceState::present("pipeline-audit"),
            direct_backend_smoke: EvidenceState::present("direct-smoke"),
            engine_smoke: EvidenceState::present("engine-smoke"),
            performance_regression: PerformanceRegressionEvidence::SampleComparison {
                baseline_sample_id: "baseline-metal-only".to_string(),
                current_sample_id: "current-metal-only".to_string(),
                max_allowed_regression_percent: 5.0,
                observed_regression_percent: 0.0,
            },
        }
    }

    #[test]
    fn complete_evidence_can_pass_evaluator_in_isolation() {
        let report = evaluate_apple_production_acceptance(&complete_evidence());

        assert_eq!(
            report.status,
            ProductionCandidateStatus::ProductionCandidate
        );
        assert!(report.failures.is_empty());
    }

    #[test]
    fn current_incomplete_evidence_fails_with_clear_reasons() {
        let report = current_apple_production_acceptance_report();

        assert_eq!(
            report.status,
            ProductionCandidateStatus::NotProductionCandidate
        );
        assert!(report.failures.iter().any(|failure| failure
            .reason
            .contains("missing metal_only benchmark sample")));
        assert!(report
            .failures
            .iter()
            .any(|failure| failure.reason.contains("correctness against reference")));
        assert!(report.failures.iter().any(|failure| failure
            .reason
            .contains("performance regressions are not tracked")));
    }

    #[test]
    fn current_real_e2b_probe_evidence_records_progress_but_not_production_readiness() {
        let report = current_real_e2b_probe_acceptance_report();

        assert_eq!(
            report.status,
            ProductionCandidateStatus::NotProductionCandidate
        );
        assert!(!report.failures.iter().any(|failure| failure
            .reason
            .contains("correctness against reference is missing")));
        assert!(report.failures.iter().any(|failure| failure
            .reason
            .contains("missing ane_partitioned benchmark sample")));
        assert!(report.failures.iter().any(|failure| failure
            .reason
            .contains("missing fallback_path benchmark sample")));
        assert!(report.failures.iter().any(|failure| {
            failure.criterion == AcceptanceCriterion::ProfileMetrics
                && failure
                    .reason
                    .contains("missing measured core latency/throughput/memory/CPU/GPU/command-buffer metrics")
        }));
        assert!(report.failures.iter().any(|failure| {
            failure.criterion == AcceptanceCriterion::DefaultToyPathDisabled
                && failure.reason.contains("production-default toy-disabled")
        }));
        assert!(!report.failures.iter().any(|failure| {
            failure.criterion == AcceptanceCriterion::NoHotPathAllocation
                && failure.reason.contains("hot-path allocation")
        }));
        assert!(!report.failures.iter().any(|failure| {
            failure.criterion == AcceptanceCriterion::NoHotPathPipelineCompilation
                && failure.reason.contains("pipeline-compilation")
        }));
        assert!(!report.failures.iter().any(|failure| {
            failure.criterion == AcceptanceCriterion::CorrectnessAgainstReference
                && failure.reason.contains("scheduler/Engine")
        }));
        assert!(report.failures.iter().any(|failure| {
            failure.criterion == AcceptanceCriterion::PerformanceRegressionsTracked
        }));
    }

    #[test]
    fn toy_backend_evidence_fails_and_default_toy_disabled_is_required() {
        let mut evidence = complete_evidence();
        evidence.samples[0].toy_backend_enabled = true;
        evidence.default_toy_path_disabled =
            EvidenceState::missing("toy-disabled production-default evidence missing");

        let report = evaluate_apple_production_acceptance(&evidence);

        assert_eq!(
            report.status,
            ProductionCandidateStatus::NotProductionCandidate
        );
        assert!(report.failures.iter().any(|failure| failure.criterion
            == AcceptanceCriterion::DefaultToyPathDisabled
            && failure.reason.contains("toy backend")));
        assert!(report.failures.iter().any(|failure| failure.criterion
            == AcceptanceCriterion::DefaultToyPathDisabled
            && failure.reason.contains("toy-disabled")));
    }

    #[test]
    fn ane_utilization_may_be_unobservable_when_reason_is_recorded() {
        let mut evidence = complete_evidence();
        for sample in &mut evidence.samples {
            if sample.category == BenchmarkCategory::AnePartitioned {
                sample.metrics.ane_utilization_percent = OptionalMetric::unmeasured(
                    "ANE utilization is unavailable from public counters on this OS build",
                );
            }
        }

        let report = evaluate_apple_production_acceptance(&evidence);

        assert_eq!(
            report.status,
            ProductionCandidateStatus::ProductionCandidate
        );
    }

    #[test]
    fn ane_utilization_unobservable_requires_an_honest_reason() {
        let mut evidence = complete_evidence();
        for sample in &mut evidence.samples {
            if sample.category == BenchmarkCategory::AnePartitioned {
                sample.metrics.ane_utilization_percent = OptionalMetric::unmeasured("");
            }
        }

        let report = evaluate_apple_production_acceptance(&evidence);

        assert_eq!(
            report.status,
            ProductionCandidateStatus::NotProductionCandidate
        );
        assert!(report.failures.iter().any(|failure| {
            failure.criterion == AcceptanceCriterion::ProfileMetrics
                && failure.reason.contains("without reasons")
        }));
    }

    #[test]
    fn performance_regression_tracking_requires_sample_ids_or_comparable_evidence() {
        let mut evidence = complete_evidence();
        evidence.performance_regression = PerformanceRegressionEvidence::NotTracked {
            reason: "no baseline/current profile samples".to_string(),
        };
        let report = evaluate_apple_production_acceptance(&evidence);
        assert_eq!(
            report.status,
            ProductionCandidateStatus::NotProductionCandidate
        );
        assert!(report.failures.iter().any(|failure| {
            failure.criterion == AcceptanceCriterion::PerformanceRegressionsTracked
        }));

        evidence.performance_regression = PerformanceRegressionEvidence::SampleComparison {
            baseline_sample_id: String::new(),
            current_sample_id: "current".to_string(),
            max_allowed_regression_percent: 5.0,
            observed_regression_percent: 0.0,
        };
        let report = evaluate_apple_production_acceptance(&evidence);
        assert!(report
            .failures
            .iter()
            .any(|failure| failure.reason.contains("baseline and current sample IDs")));

        evidence.performance_regression = PerformanceRegressionEvidence::ComparableEvidence {
            evidence_id: "regression-dashboard-run".to_string(),
            description: "dashboard compares the same Apple profile suite".to_string(),
        };
        let report = evaluate_apple_production_acceptance(&evidence);
        assert_eq!(
            report.status,
            ProductionCandidateStatus::ProductionCandidate
        );
    }
}
