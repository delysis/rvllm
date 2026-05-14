//! rvllm-apple: Apple Silicon backend contracts for Metal prefill and ANE rollout.
//!
//! The default build is safe and host-testable. It contains planning, handoff,
//! layout, MIL, and weight-blob invariants, but no Metal or private ANE FFI.

#![cfg_attr(not(all(feature = "metal", target_os = "macos")), forbid(unsafe_code))]
#![cfg_attr(all(feature = "metal", target_os = "macos"), deny(unsafe_code))]
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod ane;
pub mod backend;
pub mod compile_cache;
pub mod device;
pub mod handoff;
pub mod iosurface;
pub mod metal;
pub mod mil;
#[cfg(feature = "mlx")]
pub mod mlx;
pub mod plan;
pub mod weight_blob;

pub use ane::{
    compile_private_ane_mil, compile_private_ane_program, AneProcedure, AneProgram,
    AneProgramPlan, AneRolloutConfig, AneSys, AneSysHandle,
};
pub use backend::{AppleBackend, AppleLaunchKind, AppleLaunchTicket, StubAppleBackend, StepToken};
pub use compile_cache::{AneCompileCacheKey, CompileCacheHash};
pub use device::{
    AppleAcceleratorTarget, AppleGpuFamily, AppleNpuGeneration, DeviceTier,
};
pub use handoff::{HandoffCapsule, HandoffKind, StateHandle, StateHandleKind, SurfaceId};
pub use iosurface::{
    ByteSurfaceShape, IoSurfaceTensorDesc, PackedField, PackedFieldLayout, PackedFieldStrides,
    PackedInputLayout,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::DirectMetalContext;
pub use metal::{
    DirectMetalContextConfig, DirectMetalPipelineName, MetalBufferAllocation, MetalBufferArenaPlan,
    MetalBufferBinding, MetalBufferRequest, MetalBufferRole, MetalPrefillBackend,
    MetalPrefillCommand, MetalPrefillCommandBufferRecipe, MetalPrefillCommandEvent,
    MetalPrefillConfig, MetalPrefillOp, PrefillContract, PrefillLayerGroup, PREFILL_LAYER_OPS,
};
pub use mil::{
    dense_1x1_conv_mil, fused_ffn_mil, fused_ffn_mil_from_descs, fused_qkv_mil,
    fused_qkv_mil_from_descs, FfnMilOffsets, FfnMilWeightDescs, QkvMilOffsets,
    QkvMilWeightDescs,
};
#[cfg(feature = "mlx")]
pub use mlx::{
    MlxParityCase, MlxParityOutput, MlxReferenceExecution, MlxReferenceHarness,
    MlxReferenceInvocation, MlxReferenceMode,
};
pub use plan::{
    select_rollout_bucket, AppleBackendMode, AppleRuntimePlan, RolloutBucket, ROLLOUT_BUCKETS,
};
pub use weight_blob::{
    build_weight_blob_fp16, build_weight_blob_fp16_described, build_weight_blob_fp16_named,
    AneFp16WeightSpec, WeightChunkDesc,
};
