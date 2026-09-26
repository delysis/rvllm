//! rvllm-apple: Apple Silicon backend contracts for Metal prefill and ANE rollout.
//!
//! The default build is safe and host-testable. It contains planning, handoff,
//! layout, MIL, and weight-blob invariants, but no Metal or private ANE FFI.

#![deny(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(
    feature = "macos-private-ane-research",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!(
    "macos-private-ane-research uses unsupported private APIs and is restricted to macOS/aarch64 research builds"
);

pub mod ane;
#[cfg(feature = "macos-private-ane-research")]
pub mod ane_attention;
pub mod ane_attention_layout;
#[cfg(feature = "macos-private-ane-research")]
pub mod ane_dynamic_ffn;
#[cfg(feature = "macos-private-ane-research")]
pub mod ane_dynamic_linear;
pub mod ane_ffn_layout;
pub mod ane_int8_candidates;
pub mod ane_int8_ffn_weights;
#[cfg(feature = "macos-private-ane-research")]
pub mod ane_linear;
pub mod ane_lut4_ffn_weights;
pub mod ane_output_ffn;
pub mod ane_packed32_layout;
pub mod backend;
pub mod coreml_artifact;
pub mod coreml_projection;
pub mod device;
pub mod disaggregated;
pub mod gemma_decode_math;
pub mod handoff;
pub mod iosurface;
pub mod low_bit_weights;
pub mod metal;
pub mod mil;
pub mod model_package;
#[cfg(any(feature = "package-builder", test))]
pub mod model_package_builder;
pub mod plan;
pub mod profiling;
pub mod weight_blob;

pub use ane::{AneProcedure, AneProgramPlan, AneRolloutConfig};
#[cfg(test)]
pub use backend::StubAppleBackend;
pub use backend::{
    AppleBackend, AppleLaunchKind, AppleLaunchTicket, ProductionAppleBackend, StepToken,
};
pub use coreml_artifact::{
    ArtifactDisposition, CompiledArtifact, CoreMlArtifactCache, CoreMlArtifactError,
    CoreMlArtifactIdentity, CoreMlArtifactManifest, CoreMlDeviceClass, CoreMlDeviceEvidence,
    CoreMlEvidenceLevel, CoreMlPrecision, CoreMlRequestedComputeUnits, KnownAnswerPolicy,
    KnownAnswerValidation,
};
pub use coreml_projection::{
    export_checkpoint_gated_ffn, export_checkpoint_projection, validate_public_coreml_gated_ffn,
    validate_public_coreml_projection, CoreMlGatedFfnArtifact, CoreMlGatedFfnExportRequest,
    CoreMlGatedFfnRuntimeValidation, CoreMlGatedFfnTensorIdentity, CoreMlProjectionArtifact,
    CoreMlProjectionError, CoreMlProjectionExportRequest, CoreMlProjectionKnownAnswer,
    CoreMlProjectionRuntimeValidation, CoreMlProjectionValidation, CoreMlProjectionWeightDtype,
};
pub use device::{AppleAcceleratorTarget, AppleGpuFamily, AppleNpuGeneration, DeviceTier};
pub use disaggregated::{
    synthetic_ffn_metal_only_reference, AneExecutionAvailability, DensePartitionExecutionPath,
    DisaggregatedDenseExecutor, PartitionExecutionCounters, PartitionExecutionReport,
    SyntheticDenseOutput, SyntheticFfnPartition,
};
pub use handoff::{
    HandoffCapsule, HandoffKind, HandoffKvChain, HandoffSequenceMeta, HandoffSurfaceBinding,
    HandoffSurfaceRole, StateHandle, StateHandleKind, SurfaceId, HANDOFF_SCHEMA_V2,
};
pub use iosurface::{IoSurfaceTensorDesc, PackedField, PackedInputLayout};
pub use low_bit_weights::{
    dequantize_apple_low_bit_reference, project_apple_low_bit_reference,
    quantize_apple_low_bit_reference, AppleLowBitWeightFormat, LowBitWeightError,
    PackedAppleLowBitWeights, APPLE_LOW_BIT_GROUP_SIZE, APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
};
pub use metal::{MetalPrefillBackend, MetalPrefillConfig, PrefillContract};
pub use mil::{dense_1x1_conv_mil, fused_ffn_mil, fused_qkv_mil, FfnMilOffsets, QkvMilOffsets};
pub use model_package::{
    AppleLowBitTensor, AppleLowBitTensorRole, AppleMetalLibrary, AppleModelPackage,
    AppleModelPackageError, AppleModelPackageManifest, ApplePackageFile, ApplePackageFloatType,
    ApplePackagePlatform, AppleWeightFormat, AppleWeightShard, APPLE_MODEL_PACKAGE_MANIFEST,
};
#[cfg(any(feature = "package-builder", test))]
pub use model_package_builder::{
    build_apple_model_package, AppleLowBitExportRequest, AppleModelPackageBuildConfig,
    AppleModelPackageBuildReport,
};
pub use plan::{
    plan_ane_static_partitions, private_ane_env_opted_in, private_ane_feature_enabled,
    probe_ane_capability, select_rollout_bucket, AneCapabilityPath, AneCapabilityReport,
    AneCapabilityStatus, AneCompiledCacheKey, AneLayerRange, AneOsDeviceKey, AnePartitionOpKind,
    AnePartitionPolicy, AnePartitionRequest, AnePlannedBackend, AneStaticPartition,
    AneStaticPartitionPlan, AneUnsupportedReason, AneVocabBlock, AppleBackendMode,
    AppleRuntimePlan, CoreMlAneComputePlan, CoreMlComputeUnitsPlan, RolloutBucket,
    PRIVATE_ANE_ENV_VAR, ROLLOUT_BUCKETS,
};
pub use profiling::{
    current_apple_production_acceptance_report, current_real_e2b_probe_acceptance_report,
    evaluate_apple_production_acceptance, AcceptanceCriterion, AcceptanceFailure,
    AppleProductionAcceptanceEvidence, AppleProductionAcceptanceReport, BackendProfileMetrics,
    BackendProfileSample, BenchmarkCategory, EvidenceState, OptionalMetric,
    PerformanceRegressionEvidence, ProductionCandidateStatus, ROADMAP_BENCHMARK_CATEGORIES,
};
pub use rvllm_apple_coreml_runtime::{CoreMlComputeUnits, CoreMlExecutionReport};
pub use weight_blob::{build_weight_blob_fp16, build_weight_blob_fp16_named, WeightChunkDesc};
