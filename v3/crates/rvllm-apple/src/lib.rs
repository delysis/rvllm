//! rvllm-apple: Apple Silicon backend contracts for Metal prefill and ANE rollout.
//!
//! The default build is safe and host-testable. It contains planning, handoff,
//! layout, MIL, and weight-blob invariants, but no Metal or private ANE FFI.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod ane;
pub mod backend;
pub mod device;
pub mod handoff;
pub mod iosurface;
pub mod metal;
pub mod mil;
pub mod plan;
pub mod weight_blob;

pub use ane::{AneProcedure, AneProgramPlan, AneRolloutConfig};
pub use backend::{AppleBackend, AppleLaunchKind, AppleLaunchTicket, StubAppleBackend, StepToken};
pub use device::{
    AppleAcceleratorTarget, AppleGpuFamily, AppleNpuGeneration, DeviceTier,
};
pub use handoff::{HandoffCapsule, HandoffKind, StateHandle, StateHandleKind, SurfaceId};
pub use iosurface::{IoSurfaceTensorDesc, PackedField, PackedInputLayout};
pub use metal::{MetalPrefillBackend, MetalPrefillConfig, PrefillContract};
pub use mil::{
    dense_1x1_conv_mil, fused_ffn_mil, fused_qkv_mil, FfnMilOffsets, QkvMilOffsets,
};
pub use plan::{
    apple_quantization_matrix_entry, plan_apple_matmul, select_rollout_bucket, AppleBackendMode,
    AppleElementType, AppleMatmulBackend, AppleMatmulConfig, AppleMatmulMetadata, AppleMatmulPlan,
    AppleMatmulQuantization, AppleQuantizationMatrixEntry, AppleRuntimePlan, AppleScaleSpec,
    RolloutBucket, APPLE_QUANTIZATION_MATRIX, ROLLOUT_BUCKETS,
};
pub use weight_blob::{
    build_weight_blob_fp16, build_weight_blob_fp16_named, WeightChunkDesc,
};
