use crate::handoff::HandoffCapsule;
use crate::plan::{AnePlannedBackend, AneStaticPartitionPlan, AneUnsupportedReason};
use rvllm_core::config::AneFallbackPolicy;
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum DensePartitionExecutionPath {
    Ane,
    MetalFallback,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AneExecutionAvailability {
    Unavailable,
    AvailableButNotExecutedByScaffold,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PartitionExecutionCounters {
    pub metal_time_ns: u128,
    pub ane_time_ns: u128,
    pub cpu_sync_time_ns: u128,
    pub metal_partitions: u32,
    pub ane_partitions: u32,
    pub ane_unavailable_partitions: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PartitionExecutionReport {
    pub path: DensePartitionExecutionPath,
    pub ane_availability: Option<AneExecutionAvailability>,
    pub fallback_reason: Option<AneUnsupportedReason>,
    pub counters: PartitionExecutionCounters,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticFfnPartition {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticDenseOutput {
    pub values: Vec<f32>,
    pub report: PartitionExecutionReport,
}

impl SyntheticFfnPartition {
    #[must_use]
    pub fn identity(hidden_size: usize) -> Self {
        let mut gate_up = vec![0.0; hidden_size * hidden_size];
        let mut down = vec![0.0; hidden_size * hidden_size];
        for idx in 0..hidden_size {
            gate_up[idx * hidden_size + idx] = 1.0;
            down[idx * hidden_size + idx] = 1.0;
        }
        Self {
            hidden_size,
            intermediate_size: hidden_size,
            gate_up,
            down,
        }
    }

    #[must_use]
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        gate_up: Vec<f32>,
        down: Vec<f32>,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            gate_up,
            down,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(err("synthetic_ffn_validate", "empty synthetic dense shape"));
        }
        if self.gate_up.len() != self.hidden_size * self.intermediate_size {
            return Err(err(
                "synthetic_ffn_validate",
                "gate_up length must equal hidden_size * intermediate_size",
            ));
        }
        if self.down.len() != self.intermediate_size * self.hidden_size {
            return Err(err(
                "synthetic_ffn_validate",
                "down length must equal intermediate_size * hidden_size",
            ));
        }
        Ok(())
    }
}

pub struct DisaggregatedDenseExecutor {
    fallback_policy: AneFallbackPolicy,
    static_plan: AneStaticPartitionPlan,
}

impl DisaggregatedDenseExecutor {
    #[must_use]
    pub const fn new(
        fallback_policy: AneFallbackPolicy,
        static_plan: AneStaticPartitionPlan,
    ) -> Self {
        Self {
            fallback_policy,
            static_plan,
        }
    }

    pub fn execute_synthetic_ffn(
        &self,
        handoff: &HandoffCapsule,
        partition: &SyntheticFfnPartition,
        input: &[f32],
    ) -> Result<SyntheticDenseOutput> {
        handoff.validate()?;
        partition.validate()?;
        validate_input(partition, input)?;

        let mut counters = PartitionExecutionCounters::default();
        let fallback_reason = self
            .static_plan
            .partitions
            .iter()
            .find_map(|partition| partition.fallback_reason);
        let has_ane_partition = self
            .static_plan
            .partitions
            .iter()
            .any(|partition| partition.backend == AnePlannedBackend::Ane);
        let has_metal_fallback = self
            .static_plan
            .partitions
            .iter()
            .any(|partition| partition.backend == AnePlannedBackend::MetalFallback);

        if has_ane_partition {
            counters.ane_unavailable_partitions += 1;
            if self.fallback_policy.is_strict() {
                return Err(err(
                    "execute_synthetic_ffn",
                    "strict ANE partition is unavailable in synthetic scaffold",
                ));
            }
            let output = run_synthetic_metal_fallback(partition, input, &mut counters);
            return Ok(SyntheticDenseOutput {
                values: output,
                report: PartitionExecutionReport {
                    path: DensePartitionExecutionPath::MetalFallback,
                    ane_availability: Some(
                        AneExecutionAvailability::AvailableButNotExecutedByScaffold,
                    ),
                    fallback_reason,
                    counters,
                },
            });
        }

        if has_metal_fallback {
            counters.ane_unavailable_partitions += self.static_plan.partitions.len() as u32;
            let output = run_synthetic_metal_fallback(partition, input, &mut counters);
            return Ok(SyntheticDenseOutput {
                values: output,
                report: PartitionExecutionReport {
                    path: DensePartitionExecutionPath::MetalFallback,
                    ane_availability: Some(AneExecutionAvailability::Unavailable),
                    fallback_reason,
                    counters,
                },
            });
        }

        Err(err(
            "execute_synthetic_ffn",
            "static partition plan contains no executable dense partitions",
        ))
    }
}

#[must_use]
pub fn synthetic_ffn_metal_only_reference(
    partition: &SyntheticFfnPartition,
    input: &[f32],
) -> Vec<f32> {
    synthetic_ffn(partition, input)
}

fn run_synthetic_metal_fallback(
    partition: &SyntheticFfnPartition,
    input: &[f32],
    counters: &mut PartitionExecutionCounters,
) -> Vec<f32> {
    let metal_start = Instant::now();
    let output = synthetic_ffn(partition, input);
    counters.metal_time_ns = counters
        .metal_time_ns
        .saturating_add(metal_start.elapsed().as_nanos());
    counters.metal_partitions += 1;

    let cpu_sync_start = Instant::now();
    let output = output.to_vec();
    counters.cpu_sync_time_ns = counters
        .cpu_sync_time_ns
        .saturating_add(cpu_sync_start.elapsed().as_nanos());
    output
}

fn synthetic_ffn(partition: &SyntheticFfnPartition, input: &[f32]) -> Vec<f32> {
    let tokens = input.len() / partition.hidden_size;
    let mut intermediate = vec![0.0; tokens * partition.intermediate_size];
    for token in 0..tokens {
        for col in 0..partition.intermediate_size {
            let mut acc = 0.0;
            for row in 0..partition.hidden_size {
                acc += input[token * partition.hidden_size + row]
                    * partition.gate_up[row * partition.intermediate_size + col];
            }
            intermediate[token * partition.intermediate_size + col] = acc.max(0.0);
        }
    }

    let mut output = vec![0.0; tokens * partition.hidden_size];
    for token in 0..tokens {
        for col in 0..partition.hidden_size {
            let mut acc = 0.0;
            for row in 0..partition.intermediate_size {
                acc += intermediate[token * partition.intermediate_size + row]
                    * partition.down[row * partition.hidden_size + col];
            }
            output[token * partition.hidden_size + col] = acc;
        }
    }
    output
}

fn validate_input(partition: &SyntheticFfnPartition, input: &[f32]) -> Result<()> {
    if input.is_empty() || input.len() % partition.hidden_size != 0 {
        return Err(err(
            "synthetic_ffn_validate",
            "input length must be a non-zero multiple of hidden_size",
        ));
    }
    Ok(())
}

fn err(op: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::FeatureNotAvailable {
            backend: "disaggregated-apple",
            op: reason,
        },
        AppleCtx {
            backend: "disaggregated-apple",
            op,
            device: "host-test",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::AppleAcceleratorTarget;
    use crate::handoff::{
        HandoffKind, HandoffSurfaceBinding, HandoffSurfaceRole, StateHandle, StateHandleKind,
        SurfaceId,
    };
    use crate::iosurface::IoSurfaceTensorDesc;
    use crate::plan::{
        plan_ane_static_partitions, AneCapabilityPath, AneCapabilityReport, AneCapabilityStatus,
        AneLayerRange, AnePartitionPolicy, AnePartitionRequest, AneUnsupportedReason,
        CoreMlAneComputePlan, RolloutBucket,
    };
    use rvllm_core::config::AneComputeProfile;
    use rvllm_core::{DType, ReqId, TokenId};

    fn unavailable_report(reason: AneUnsupportedReason) -> AneCapabilityReport {
        let target = AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1);
        AneCapabilityReport {
            path: AneCapabilityPath::PrivateAne,
            status: AneCapabilityStatus::Unsupported { reason },
            target,
            private_ane_feature_enabled: false,
            private_ane_env_opt_in: false,
            compute_plan: CoreMlAneComputePlan::from_profile(
                AneComputeProfile::NeuralEngineOnly,
                true,
            ),
        }
    }

    fn available_report() -> AneCapabilityReport {
        let target = AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1);
        AneCapabilityReport {
            path: AneCapabilityPath::PrivateAne,
            status: AneCapabilityStatus::Available,
            target,
            private_ane_feature_enabled: true,
            private_ane_env_opt_in: true,
            compute_plan: CoreMlAneComputePlan::from_profile(
                AneComputeProfile::NeuralEngineOnly,
                true,
            ),
        }
    }

    fn request() -> AnePartitionRequest {
        AnePartitionRequest::ffn(
            AneLayerRange { start: 0, end: 1 },
            RolloutBucket { seqs: 1, tokens: 1 },
            DType::F16,
        )
    }

    fn handoff(hidden_size: usize) -> HandoffCapsule {
        let desc = IoSurfaceTensorDesc {
            dtype: DType::F16,
            channels: hidden_size,
            spatial: 1,
        };
        HandoffCapsule::new(
            HandoffKind::MetalPrefillToAneFfnRollout,
            vec![ReqId(7)],
            vec![TokenId(11)],
            vec![0, 1],
            vec![0],
            vec![1],
        )
        .with_rollout_bucket(Some(RolloutBucket { seqs: 1, tokens: 1 }))
        .with_surfaces(Some(SurfaceId(101)), Some(SurfaceId(102)))
        .with_activation_surface(HandoffSurfaceBinding {
            role: HandoffSurfaceRole::ActivationInput,
            surface_id: SurfaceId(101),
            desc,
        })
        .with_activation_surface(HandoffSurfaceBinding {
            role: HandoffSurfaceRole::ActivationOutput,
            surface_id: SurfaceId(102),
            desc,
        })
        .with_state_handle(StateHandle {
            kind: StateHandleKind::KvCache,
            id: 77,
            bytes: 4096,
        })
    }

    #[test]
    fn synthetic_one_layer_ane_ffn_noop_matches_metal_only_fallback() {
        let static_plan = plan_ane_static_partitions(
            &[request()],
            &unavailable_report(AneUnsupportedReason::PrivateAneEnvOptInMissing),
            AnePartitionPolicy::allow_metal(),
            [3u8; 32],
        )
        .unwrap();
        let partition = SyntheticFfnPartition::identity(4);
        let input = [0.0, 1.0, 2.0, 3.0];
        let executor = DisaggregatedDenseExecutor::new(AneFallbackPolicy::AllowMetal, static_plan);
        let output = executor
            .execute_synthetic_ffn(&handoff(4), &partition, &input)
            .unwrap();

        assert_eq!(
            output.values,
            synthetic_ffn_metal_only_reference(&partition, &input)
        );
        assert_eq!(output.values, input);
        assert_eq!(
            output.report.path,
            DensePartitionExecutionPath::MetalFallback
        );
        assert_eq!(output.report.counters.ane_time_ns, 0);
        assert_eq!(output.report.counters.metal_partitions, 1);
    }

    #[test]
    fn synthetic_one_layer_ane_ffn_nonzero_matches_metal_only_fallback() {
        let static_plan = plan_ane_static_partitions(
            &[request()],
            &unavailable_report(AneUnsupportedReason::PublicCoreMlExecutionPathNotEnabled),
            AnePartitionPolicy::allow_metal(),
            [4u8; 32],
        )
        .unwrap();
        let partition = SyntheticFfnPartition::new(
            2,
            3,
            vec![1.0, -1.0, 2.0, 0.5, 3.0, -2.0],
            vec![1.0, 2.0, -1.0, 0.5, 0.25, 4.0],
        );
        let input = [2.0, 4.0];
        let executor = DisaggregatedDenseExecutor::new(AneFallbackPolicy::AllowMetal, static_plan);
        let output = executor
            .execute_synthetic_ffn(&handoff(2), &partition, &input)
            .unwrap();
        let reference = synthetic_ffn_metal_only_reference(&partition, &input);

        assert_eq!(output.values, reference);
        assert_ne!(output.values, input);
        assert_eq!(output.values, vec![-6.0, 13.0]);
        assert_eq!(output.report.counters.ane_time_ns, 0);
        assert_eq!(
            output.report.fallback_reason,
            Some(AneUnsupportedReason::PublicCoreMlExecutionPathNotEnabled)
        );
    }

    #[test]
    fn strict_ane_static_partition_unavailable_does_not_fake_execution() {
        let static_plan = plan_ane_static_partitions(
            &[request()],
            &available_report(),
            AnePartitionPolicy::strict(),
            [5u8; 32],
        )
        .unwrap();
        let executor = DisaggregatedDenseExecutor::new(AneFallbackPolicy::FailFast, static_plan);
        let err = executor.execute_synthetic_ffn(
            &handoff(2),
            &SyntheticFfnPartition::identity(2),
            &[1.0, 2.0],
        );

        assert!(err.is_err());
    }
}
