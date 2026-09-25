use half::{bf16, f16};
use rvllm_apple::{
    AppleBackend, AppleLaunchKind, AppleLaunchTicket, AppleModelPackage, ApplePackageFloatType,
    ApplePackagePlatform, HandoffCapsule, StepToken,
};
use rvllm_core::{AppleCtx, AppleError, BlockId, Result, RvllmError, TokenId};
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use std::cell::Cell;
#[cfg(feature = "metal-stage-instrumentation")]
use std::cell::RefCell;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use std::cmp::max;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use std::path::PathBuf;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use std::ptr;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use std::sync::Arc;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use std::time::{Duration, Instant};

#[cfg(not(all(feature = "apple", any(target_os = "macos", target_os = "ios"))))]
#[derive(Debug, Default)]
pub struct RuntimeMetalBackend;

#[cfg(not(all(feature = "apple", any(target_os = "macos", target_os = "ios"))))]
impl RuntimeMetalBackend {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(not(all(feature = "apple", any(target_os = "macos", target_os = "ios"))))]
impl AppleBackend for RuntimeMetalBackend {
    fn prepare(&mut self, _plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "runtime-metal-backend",
                op: "prepare",
            },
            AppleCtx {
                backend: "runtime-metal-backend",
                op: "prepare",
                device: "apple-silicon",
            },
        ))
    }

    fn launch_prefill(&mut self, _handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "runtime-metal-backend",
                op: "launch_prefill",
            },
            AppleCtx {
                backend: "runtime-metal-backend",
                op: "launch_prefill",
                device: "apple-silicon",
            },
        ))
    }

    fn launch_rollout(
        &mut self,
        _handoff: &HandoffCapsule,
        _bucket: Option<rvllm_apple::plan::RolloutBucket>,
    ) -> Result<AppleLaunchTicket> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "runtime-metal-backend",
                op: "launch_rollout",
            },
            AppleCtx {
                backend: "runtime-metal-backend",
                op: "launch_rollout",
                device: "apple-silicon",
            },
        ))
    }

    fn collect(&mut self, _ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "runtime-metal-backend",
                op: "collect",
            },
            AppleCtx {
                backend: "runtime-metal-backend",
                op: "collect",
                device: "apple-silicon",
            },
        ))
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use crate::paged_kv::{CowPageCopy, APPLE_KV_PAGE_SIZE};
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use crate::paged_prompt_cache::{KvPageIo, KvPageIoError};
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use objc2::rc::Retained;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use rvllm_apple::RolloutBucket;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use rvllm_apple_metal::arena::{MetalBufferArena, MetalRegion};
#[cfg(all(
    feature = "metal-stage-instrumentation",
    any(target_os = "macos", target_os = "ios")
))]
use rvllm_apple_metal::stage_instrumentation::{MetalStage, MetalStageProfiler};
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use rvllm_apple_metal::{
    context::MetalContext,
    gemma4_model::{
        Gemma4MetalState, MetalLayerTraceState, MetalLowBitWeightReplacement, APPLE_KV_PAGE_TOKENS,
    },
    kernels,
    layer_forward::{
        metal_encode_finalize_sample, metal_encode_forward_layer, metal_encode_prepare_ple_inputs,
        metal_finalize_logits_blocking, metal_finalize_logits_encoder_count,
        metal_finalize_sample_encoder_count, metal_forward_layer, metal_gemm_rmsnorm_encoder_count,
        metal_prepare_ple_inputs, supports_qkv_prefill_projection, supports_qkv_rope_cache_fusion,
        MetalLayerDebugSkip, MetalLayerDims, MetalLayerTraceScratch, MetalLayerWeights,
        MetalMetadata, MetalMoeWeights, MetalPhase, MetalPlePrepare, MetalScratch,
    },
    pipeline::PipelineCache,
    AppleLowBitWeightFormat, MetalFloatType, MetalLowBitProjectionOffsets,
    APPLE_LOW_BIT_GROUP_SIZE, APPLE_LOW_BIT_WEIGHT_ABI_VERSION,
};
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
use sha2::{Digest, Sha256};

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
const METAL_HIDDEN: usize = 256;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
const METAL_VOCAB: usize = 256;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
const METAL_MAX_TOKENS: usize = 256;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
const METAL_EPS: f32 = 1e-5;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
const METAL_SOFTCAP: f32 = 0.0;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
const METAL_ARENA_BYTES: usize = 1 * 1024 * 1024;
#[cfg(all(feature = "apple", target_os = "macos"))]
pub const RVLLM_METAL_DEBUG_SYNC_ENV: &str = "RVLLM_METAL_DEBUG_SYNC";
#[cfg(all(feature = "apple", target_os = "macos"))]
pub const RVLLM_METAL_DTYPE_ENV: &str = "RVLLM_METAL_DTYPE";
#[cfg(all(feature = "apple", target_os = "macos"))]
pub const RVLLM_METAL_METALLIB_F16_ENV: &str = "RVLLM_METAL_METALLIB_F16";
#[cfg(all(feature = "apple", target_os = "macos"))]
pub const RVLLM_METAL_METALLIB_BF16_ENV: &str = "RVLLM_METAL_METALLIB_BF16";
#[cfg(all(feature = "apple", target_os = "macos"))]
pub const RVLLM_EXPERIMENTAL_METAL_KV_INT8_ENV: &str = "RVLLM_EXPERIMENTAL_METAL_KV_INT8";
#[cfg(all(test, feature = "apple", target_os = "macos"))]
const RVLLM_METAL_DEBUG_FINITE_LAYERS_ENV: &str = "RVLLM_METAL_DEBUG_FINITE_LAYERS";
#[cfg(all(test, feature = "apple", target_os = "macos"))]
const RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS_ENV: &str = "RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS";
#[cfg(all(test, feature = "apple", target_os = "macos"))]
const RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV: &str = "RVLLM_METAL_DEBUG_STOP_AFTER_LAYER";
#[cfg(all(test, feature = "apple", target_os = "macos"))]
const RVLLM_METAL_DEBUG_TRACE_LAYER_ENV: &str = "RVLLM_METAL_DEBUG_TRACE_LAYER";
#[cfg(all(test, feature = "apple", target_os = "macos"))]
const RVLLM_METAL_DEBUG_TRACE_JSON_ENV: &str = "RVLLM_METAL_DEBUG_TRACE_JSON";
#[cfg(all(test, feature = "apple", target_os = "macos"))]
const RVLLM_METAL_DEBUG_SKIP_FINAL_LOGITS_ENV: &str = "RVLLM_METAL_DEBUG_SKIP_FINAL_LOGITS";
#[cfg(all(test, feature = "apple", target_os = "macos"))]
const RVLLM_METAL_DEBUG_SHARED_KV_SKIP_MODE_ENV: &str = "RVLLM_METAL_DEBUG_SHARED_KV_SKIP_MODE";
#[cfg(all(test, feature = "apple", target_os = "macos"))]
const RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV: &str = "RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE";

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
enum MetalDebugSharedKvSkipMode {
    #[default]
    None,
    SkipLocalKvCacheWriteOnly,
    SkipTailKvProjectionAndCache,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MetalProbePerfStats {
    pub prefill_steps: u64,
    pub decode_steps: u64,
    pub tokens: u64,
    pub library_compiles: u64,
    pub pipeline_state_compiles: u64,
    pub command_buffers: u64,
    pub encoders: u64,
    pub embedding_encoders: u64,
    pub ple_encoders: u64,
    pub layer_encoders: u64,
    pub layer_scale_encoder_fusions: u64,
    pub final_sample_encoders: u64,
    pub final_logits_encoders: u64,
    pub forced_waits: u64,
    pub cpu_wall_ns: u64,
    pub cpu_encode_ns: u64,
    pub command_buffer_wait_ns: u64,
    pub last_step_tokens: u64,
    pub last_step_command_buffers: u64,
    pub last_step_encoders: u64,
    pub last_step_forced_waits: u64,
    pub last_step_cpu_wall_ns: u64,
    pub last_step_cpu_encode_ns: u64,
    pub last_step_command_buffer_wait_ns: u64,
    /// Completed asynchronous command buffer GPU end minus start. No host wait.
    pub last_step_gpu_execution_ns: Option<u64>,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MetalProbeArenaStats {
    pub region_count: usize,
    pub allocated_bytes: usize,
    pub capacity_bytes: usize,
}

/// Prepared physical limits used to construct backend-neutral scheduling
/// metadata. Callers must mirror these exact limits rather than independently
/// guessing a page-pool shape from request configuration.
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MetalModelCapacity {
    pub max_context_tokens: usize,
    pub max_batch_tokens: usize,
    pub max_batch_sequences: usize,
    pub kv_page_size: u32,
    pub physical_kv_pages: u32,
    pub recommended_working_set_bytes: u64,
    pub reserve_bytes: u64,
    pub usable_bytes: u64,
    pub weights_bytes: u64,
    pub scratch_slot_bytes: u64,
    pub scratch_budget_bytes: u64,
    pub metadata_bytes: u64,
    pub kv_budget_bytes: u64,
    pub kv_page_bytes: u64,
    pub allocated_kv_bytes: u64,
    pub prepared_arena_bytes: u64,
    pub max_useful_kv_pages: u32,
    pub admission_required_kv_pages: u32,
    /// Stable public name of the actual Metal compute/cache element type.
    pub metal_float_type: &'static str,
    /// Stable name of the prepared KV storage representation.
    pub kv_storage_format: &'static str,
    /// The utility-path opt-in is recorded even while production KV remains
    /// native 16-bit storage, preventing ambiguous cache identities.
    pub experimental_kv_int8_opt_in: bool,
    /// True only when the prepared production KV arena actually stores int8.
    pub experimental_kv_int8_active: bool,
    /// Number of authenticated tensor sidecars selected for real execution.
    pub low_bit_projection_count: u32,
    /// Arena-resident packed values plus FP16 scales, excluding alignment.
    pub low_bit_weight_bytes: u64,
    pub numeric_abi_version: u32,
    pub numeric_abi_fingerprint: [u8; 32],
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
pub const METAL_NUMERIC_ABI_VERSION: u32 = 10;

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum MetalLowBitResidencyPolicy {
    /// Keep the native projection resident while selecting the authenticated
    /// low-bit sidecar for execution.
    #[default]
    HybridFallback,
    /// Omit every selected native projection from the Metal arena.
    ReplaceNative,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl MetalLowBitResidencyPolicy {
    const fn identity_tag(self) -> u8 {
        match self {
            Self::HybridFallback => 1,
            Self::ReplaceNative => 2,
        }
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[cfg(test)]
fn metal_numeric_abi_fingerprint(
    float_type: MetalFloatType,
    experimental_kv_int8_opt_in: bool,
    experimental_kv_int8_active: bool,
) -> [u8; 32] {
    metal_numeric_abi_fingerprint_impl(
        float_type,
        experimental_kv_int8_opt_in,
        experimental_kv_int8_active,
        None,
        None,
        MetalLowBitResidencyPolicy::HybridFallback,
        rvllm_apple_metal::MetalKernelOptions::from_development_environment(),
    )
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn metal_numeric_abi_fingerprint_for_state(
    float_type: MetalFloatType,
    experimental_kv_int8_opt_in: bool,
    experimental_kv_int8_active: bool,
    state: &Gemma4MetalState,
    package_identity: Option<[u8; 32]>,
    low_bit_residency_policy: MetalLowBitResidencyPolicy,
    kernel_options: rvllm_apple_metal::MetalKernelOptions,
) -> [u8; 32] {
    metal_numeric_abi_fingerprint_impl(
        float_type,
        experimental_kv_int8_opt_in,
        experimental_kv_int8_active,
        Some(state),
        package_identity,
        low_bit_residency_policy,
        kernel_options,
    )
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn metal_numeric_abi_fingerprint_impl(
    float_type: MetalFloatType,
    experimental_kv_int8_opt_in: bool,
    experimental_kv_int8_active: bool,
    state: Option<&Gemma4MetalState>,
    package_identity: Option<[u8; 32]>,
    low_bit_residency_policy: MetalLowBitResidencyPolicy,
    kernel_options: rvllm_apple_metal::MetalKernelOptions,
) -> [u8; 32] {
    let shader_source = kernels::kernel_source_with_options(float_type, kernel_options);
    let bf16_accumulation =
        if matches!(float_type, MetalFloatType::Bf16) && shader_source.contains("acc = bf16_acc") {
            b"quantized-bf16".as_slice()
        } else {
            b"float32".as_slice()
        };
    let mut hasher = Sha256::new();
    hasher.update(b"rvllm.apple.metal.numeric-abi\0");
    hasher.update(METAL_NUMERIC_ABI_VERSION.to_le_bytes());
    hasher.update(b"attention.decode=causal-paged-online-softmax-v2\0");
    hasher.update(b"attention.decode-online=causal-paged-simdgroup-one-pass-v3\0");
    hasher.update(b"attention.prefill=absolute-causal-paged-v2\0");
    hasher.update(b"attention.prefill-simdgroup-d256-d512=");
    hasher.update([u8::from(kernel_options.prefill_simd_attention)]);
    hasher.update(b"attention.window=zero-full-positive-exact-width\0");
    hasher.update(b"projection.microbatch=apple9-batch8-weight-reuse-v1\0");
    hasher.update(b"projection.gemma4-12b-prefill=shape-limited-batch8-m20-32-v1\0");
    hasher.update(b"projection.qkv-prefill-f32-batch8=");
    hasher.update([u8::from(kernel_options.qkv_prefill_batch8)]);
    hasher.update(b"projection.lm-head=apple9-batch8-argmax-weight-reuse-v1\0");
    hasher.update(b"projection.gemma4-12b-bf16-mma32=");
    hasher.update([u8::from(kernel_options.prefill_mma32)]);
    hasher.update(b"kv.page-tokens=");
    hasher.update((APPLE_KV_PAGE_TOKENS as u32).to_le_bytes());
    hasher.update(b"\0bf16.accumulation=");
    hasher.update(bf16_accumulation);
    hasher.update(b"\0weights.quantization=native16-with-exact-tensor-sidecars-v1\0low-bit.abi=");
    hasher.update(APPLE_LOW_BIT_WEIGHT_ABI_VERSION.to_le_bytes());
    hasher.update(b"\0low-bit.group-size=");
    hasher.update((APPLE_LOW_BIT_GROUP_SIZE as u32).to_le_bytes());
    hasher.update(b"\0low-bit.formats=");
    hasher.update([AppleLowBitWeightFormat::W4A16.bits() as u8]);
    hasher.update([AppleLowBitWeightFormat::W8A16.bits() as u8]);
    hasher.update(b"\0low-bit.residency-policy=");
    hasher.update([low_bit_residency_policy.identity_tag()]);
    hasher.update([match float_type {
        MetalFloatType::F16 => 1,
        MetalFloatType::Bf16 => 2,
    }]);
    hasher.update([u8::from(experimental_kv_int8_opt_in)]);
    hasher.update([u8::from(experimental_kv_int8_active)]);
    hasher.update(b"\0low-bit.selected=");
    let selected_count = state.map_or(0, |state| {
        state
            .layers
            .iter()
            .filter(|layer| layer.low_bit_down_proj.is_some())
            .count()
    });
    hasher.update((selected_count as u32).to_le_bytes());
    hasher.update(b"\0low-bit.package-identity=");
    match package_identity {
        Some(identity) => {
            hasher.update([1]);
            hasher.update(identity);
        }
        _ => hasher.update([0]),
    }
    if let Some(state) = state {
        let mut selected = state
            .layers
            .iter()
            .filter_map(|layer| {
                layer
                    .low_bit_down_proj
                    .map(|projection| (layer, projection))
            })
            .collect::<Vec<_>>();
        selected.sort_by(|(left, _), (right, _)| left.down_proj_name.cmp(&right.down_proj_name));
        for (layer, projection) in selected {
            hasher.update((layer.layer_idx as u32).to_le_bytes());
            hasher.update((layer.down_proj_name.len() as u32).to_le_bytes());
            hasher.update(layer.down_proj_name.as_bytes());
            hasher.update([projection.format().bits() as u8]);
            for dimension in projection.shape() {
                hasher.update(dimension.to_le_bytes());
            }
            hasher.update((APPLE_LOW_BIT_GROUP_SIZE as u32).to_le_bytes());
        }
    }
    hasher.update(b"\0shader-manifest=");
    for name in kernels::KERNEL_NAMES {
        hasher.update((name.len() as u32).to_le_bytes());
        hasher.update(name.as_bytes());
    }
    hasher.update(b"\0shader-source=");
    hasher.update((shader_source.len() as u64).to_le_bytes());
    hasher.update(shader_source.as_bytes());
    hasher.finalize().into()
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug, Default)]
struct MetalProbePerfCounters {
    prefill_steps: Cell<u64>,
    decode_steps: Cell<u64>,
    tokens: Cell<u64>,
    library_compiles: Cell<u64>,
    pipeline_state_compiles: Cell<u64>,
    command_buffers: Cell<u64>,
    encoders: Cell<u64>,
    embedding_encoders: Cell<u64>,
    ple_encoders: Cell<u64>,
    layer_encoders: Cell<u64>,
    layer_scale_encoder_fusions: Cell<u64>,
    final_sample_encoders: Cell<u64>,
    final_logits_encoders: Cell<u64>,
    forced_waits: Cell<u64>,
    cpu_wall_ns: Cell<u64>,
    command_buffer_wait_ns: Cell<u64>,
    last_step_tokens: Cell<u64>,
    last_step_command_buffers: Cell<u64>,
    last_step_encoders: Cell<u64>,
    last_step_forced_waits: Cell<u64>,
    last_step_cpu_wall_ns: Cell<u64>,
    last_step_command_buffer_wait_ns: Cell<u64>,
    last_step_gpu_execution_ns: Cell<Option<u64>>,
    #[cfg(feature = "metal-stage-instrumentation")]
    last_stage_timing_receipt: RefCell<Option<serde_json::Value>>,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl MetalProbePerfCounters {
    fn clear(&self) {
        self.prefill_steps.set(0);
        self.decode_steps.set(0);
        self.tokens.set(0);
        self.library_compiles.set(0);
        self.pipeline_state_compiles.set(0);
        self.command_buffers.set(0);
        self.encoders.set(0);
        self.embedding_encoders.set(0);
        self.ple_encoders.set(0);
        self.layer_encoders.set(0);
        self.layer_scale_encoder_fusions.set(0);
        self.final_sample_encoders.set(0);
        self.final_logits_encoders.set(0);
        self.forced_waits.set(0);
        self.cpu_wall_ns.set(0);
        self.command_buffer_wait_ns.set(0);
        self.last_step_tokens.set(0);
        self.last_step_command_buffers.set(0);
        self.last_step_encoders.set(0);
        self.last_step_forced_waits.set(0);
        self.last_step_cpu_wall_ns.set(0);
        self.last_step_command_buffer_wait_ns.set(0);
        self.last_step_gpu_execution_ns.set(None);
        #[cfg(feature = "metal-stage-instrumentation")]
        self.last_stage_timing_receipt.borrow_mut().take();
    }

    fn snapshot(&self) -> MetalProbePerfStats {
        let cpu_wall_ns = self.cpu_wall_ns.get();
        let wait_ns = self.command_buffer_wait_ns.get();
        let last_cpu_wall_ns = self.last_step_cpu_wall_ns.get();
        let last_wait_ns = self.last_step_command_buffer_wait_ns.get();
        MetalProbePerfStats {
            prefill_steps: self.prefill_steps.get(),
            decode_steps: self.decode_steps.get(),
            tokens: self.tokens.get(),
            library_compiles: self.library_compiles.get(),
            pipeline_state_compiles: self.pipeline_state_compiles.get(),
            command_buffers: self.command_buffers.get(),
            encoders: self.encoders.get(),
            embedding_encoders: self.embedding_encoders.get(),
            ple_encoders: self.ple_encoders.get(),
            layer_encoders: self.layer_encoders.get(),
            layer_scale_encoder_fusions: self.layer_scale_encoder_fusions.get(),
            final_sample_encoders: self.final_sample_encoders.get(),
            final_logits_encoders: self.final_logits_encoders.get(),
            forced_waits: self.forced_waits.get(),
            cpu_wall_ns,
            cpu_encode_ns: cpu_wall_ns.saturating_sub(wait_ns),
            command_buffer_wait_ns: wait_ns,
            last_step_tokens: self.last_step_tokens.get(),
            last_step_command_buffers: self.last_step_command_buffers.get(),
            last_step_encoders: self.last_step_encoders.get(),
            last_step_forced_waits: self.last_step_forced_waits.get(),
            last_step_cpu_wall_ns: last_cpu_wall_ns,
            last_step_cpu_encode_ns: last_cpu_wall_ns.saturating_sub(last_wait_ns),
            last_step_command_buffer_wait_ns: last_wait_ns,
            last_step_gpu_execution_ns: self.last_step_gpu_execution_ns.get(),
        }
    }

    fn add_command_buffers(&self, count: u64) {
        self.command_buffers
            .set(self.command_buffers.get().saturating_add(count));
    }

    fn add_library_compile(&self) {
        self.library_compiles
            .set(self.library_compiles.get().saturating_add(1));
    }

    fn add_pipeline_state_compiles(&self, count: u64) {
        self.pipeline_state_compiles
            .set(self.pipeline_state_compiles.get().saturating_add(count));
    }

    fn add_encoders(&self, count: u64) {
        self.encoders.set(self.encoders.get().saturating_add(count));
    }

    fn add_embedding_encoders(&self, count: u64) {
        self.add_encoders(count);
        self.embedding_encoders
            .set(self.embedding_encoders.get().saturating_add(count));
    }

    fn add_ple_encoders(&self, count: u64) {
        self.add_encoders(count);
        self.ple_encoders
            .set(self.ple_encoders.get().saturating_add(count));
    }

    fn add_layer_encoders(&self, count: u64) {
        self.add_encoders(count);
        self.layer_encoders
            .set(self.layer_encoders.get().saturating_add(count));
    }

    fn add_layer_scale_encoder_fusions(&self, count: u64) {
        self.layer_scale_encoder_fusions
            .set(self.layer_scale_encoder_fusions.get().saturating_add(count));
    }

    fn add_final_sample_encoders(&self, count: u64) {
        self.add_encoders(count);
        self.final_sample_encoders
            .set(self.final_sample_encoders.get().saturating_add(count));
    }

    fn add_final_logits_encoders(&self, count: u64) {
        self.add_encoders(count);
        self.final_logits_encoders
            .set(self.final_logits_encoders.get().saturating_add(count));
    }

    fn add_forced_wait(&self) {
        self.forced_waits
            .set(self.forced_waits.get().saturating_add(1));
    }

    fn add_forced_wait_duration(&self, elapsed: Duration) {
        self.add_forced_wait();
        self.command_buffer_wait_ns.set(
            self.command_buffer_wait_ns
                .get()
                .saturating_add(duration_ns_u64(elapsed)),
        );
    }

    fn finish_step(
        &self,
        decode: bool,
        tokens: u64,
        before: MetalProbePerfStats,
        elapsed: Duration,
    ) {
        let elapsed_ns = duration_ns_u64(elapsed);
        if decode {
            self.decode_steps
                .set(self.decode_steps.get().saturating_add(1));
        } else {
            self.prefill_steps
                .set(self.prefill_steps.get().saturating_add(1));
        }
        self.tokens.set(self.tokens.get().saturating_add(tokens));
        self.cpu_wall_ns
            .set(self.cpu_wall_ns.get().saturating_add(elapsed_ns));
        self.last_step_tokens.set(tokens);
        self.last_step_command_buffers.set(
            self.command_buffers
                .get()
                .saturating_sub(before.command_buffers),
        );
        self.last_step_encoders
            .set(self.encoders.get().saturating_sub(before.encoders));
        self.last_step_forced_waits
            .set(self.forced_waits.get().saturating_sub(before.forced_waits));
        self.last_step_cpu_wall_ns.set(elapsed_ns);
        self.last_step_command_buffer_wait_ns.set(
            self.command_buffer_wait_ns
                .get()
                .saturating_sub(before.command_buffer_wait_ns),
        );
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn duration_ns_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn metal_debug_sync_enabled() -> bool {
    std::env::var(RVLLM_METAL_DEBUG_SYNC_ENV).ok().as_deref() == Some("1")
}

#[cfg(all(feature = "apple", target_os = "ios"))]
const fn metal_debug_sync_enabled() -> bool {
    false
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn experimental_metal_kv_int8_enabled() -> bool {
    std::env::var(RVLLM_EXPERIMENTAL_METAL_KV_INT8_ENV)
        .ok()
        .as_deref()
        == Some("1")
}

#[cfg(all(feature = "apple", target_os = "ios"))]
const fn experimental_metal_kv_int8_enabled() -> bool {
    false
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn configured_metal_float_type(model_dir: &std::path::Path) -> Result<MetalFloatType> {
    match std::env::var(RVLLM_METAL_DTYPE_ENV) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Gemma4MetalState::preferred_probe_model_float_type(model_dir),
            "f16" | "float16" | "fp16" => Ok(MetalFloatType::F16),
            "bf16" | "bfloat16" => Ok(MetalFloatType::Bf16),
            _ => Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "invalid_RVLLM_METAL_DTYPE",
                },
                model_ctx("prepare"),
            )),
        },
        Err(std::env::VarError::NotPresent) => {
            Gemma4MetalState::preferred_probe_model_float_type(model_dir)
        }
        Err(std::env::VarError::NotUnicode(_)) => Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op: "invalid_RVLLM_METAL_DTYPE",
            },
            model_ctx("prepare"),
        )),
    }
}

#[cfg(all(feature = "apple", target_os = "ios"))]
fn configured_metal_float_type(model_dir: &std::path::Path) -> Result<MetalFloatType> {
    Gemma4MetalState::preferred_probe_model_float_type(model_dir)
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn load_model_kernel_library(
    ctx: &mut MetalContext,
    float_type: MetalFloatType,
    explicit_path: Option<&std::path::Path>,
    kernel_options: rvllm_apple_metal::MetalKernelOptions,
) -> Result<()> {
    if let Some(path) = explicit_path {
        return ctx.load_metallib(path);
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(path) = ModelMetalBackend::development_metallib_path(float_type) {
            return ctx.load_metallib(&path);
        }
        if cfg!(debug_assertions) {
            let kernel_source = kernels::kernel_source_with_options(float_type, kernel_options);
            return ctx.compile_library(&kernel_source);
        }
    }
    #[cfg(target_os = "ios")]
    let _ = (float_type, kernel_options);

    Err(RvllmError::apple(
        AppleError::MetallibMissing {
            path: PathBuf::from("explicit_precompiled_metallib_required"),
        },
        model_ctx("load_shipping_metallib"),
    ))
}

/// Compile-time contract for the real model-owning Metal backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelMetalPlatformPolicy {
    pub real_model_backend: bool,
    pub environment_configuration: bool,
    pub development_source_compilation: bool,
    pub requires_explicit_precompiled_metallib: bool,
}

impl ModelMetalPlatformPolicy {
    #[must_use]
    pub const fn current() -> Self {
        let apple_runtime = cfg!(all(
            feature = "apple",
            any(target_os = "macos", target_os = "ios")
        ));
        Self {
            real_model_backend: apple_runtime,
            environment_configuration: cfg!(all(feature = "apple", target_os = "macos")),
            development_source_compilation: cfg!(all(feature = "apple", target_os = "macos")),
            requires_explicit_precompiled_metallib: cfg!(all(feature = "apple", target_os = "ios")),
        }
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn metal_u16_to_f32(bits: u16, float_type: MetalFloatType) -> f32 {
    match float_type {
        MetalFloatType::F16 => f16::from_bits(bits).to_f32(),
        MetalFloatType::Bf16 => bf16::from_bits(bits).to_f32(),
    }
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_finite_layers_enabled() -> bool {
    fn env_truthy(name: &str) -> bool {
        std::env::var(name)
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    }

    env_truthy(RVLLM_METAL_DEBUG_CHECK_FINITE_LAYERS_ENV)
        || env_truthy(RVLLM_METAL_DEBUG_FINITE_LAYERS_ENV)
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_stop_after_layer() -> Option<usize> {
    let raw = std::env::var(RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV).ok()?;
    match raw.parse::<usize>() {
        Ok(layer_idx) => Some(layer_idx),
        Err(err) => {
            eprintln!(
                "metal debug finite: ignoring invalid {RVLLM_METAL_DEBUG_STOP_AFTER_LAYER_ENV}={raw:?}: {err}"
            );
            None
        }
    }
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_trace_layers() -> Vec<usize> {
    std::env::var(RVLLM_METAL_DEBUG_TRACE_LAYER_ENV)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|part| {
                    let part = part.trim();
                    if part.is_empty() {
                        return None;
                    }
                    match part.parse::<usize>() {
                        Ok(layer_idx) => Some(layer_idx),
                        Err(err) => {
                            eprintln!(
                                "metal debug trace: ignoring invalid trace layer {part:?}: {err}"
                            );
                            None
                        }
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_trace_json_path() -> Option<PathBuf> {
    std::env::var_os(RVLLM_METAL_DEBUG_TRACE_JSON_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_trace_json_path_for_layer(path: &std::path::Path, layer_idx: usize) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw.contains("{layer}") {
        PathBuf::from(raw.replace("{layer}", &layer_idx.to_string()))
    } else {
        path.to_path_buf()
    }
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_skip_final_logits_enabled() -> bool {
    std::env::var(RVLLM_METAL_DEBUG_SKIP_FINAL_LOGITS_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_shared_kv_skip_mode() -> MetalDebugSharedKvSkipMode {
    match std::env::var(RVLLM_METAL_DEBUG_SHARED_KV_SKIP_MODE_ENV)
        .unwrap_or_default()
        .trim()
    {
        "" | "none" => MetalDebugSharedKvSkipMode::None,
        "skip_local_kv_cache_write_only" => MetalDebugSharedKvSkipMode::SkipLocalKvCacheWriteOnly,
        "skip_tail_kv_projection_and_cache" => {
            MetalDebugSharedKvSkipMode::SkipTailKvProjectionAndCache
        }
        raw => {
            eprintln!(
                "metal debug shared-KV: ignoring invalid {RVLLM_METAL_DEBUG_SHARED_KV_SKIP_MODE_ENV}={raw:?}"
            );
            MetalDebugSharedKvSkipMode::None
        }
    }
}

#[cfg(not(all(test, feature = "apple", target_os = "macos")))]
fn metal_debug_skip_final_logits_enabled() -> bool {
    false
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_layer_controls_enabled() -> bool {
    metal_debug_finite_layers_enabled()
        || metal_debug_stop_after_layer().is_some()
        || !metal_debug_trace_layers().is_empty()
}

#[cfg(all(not(test), feature = "apple", target_os = "macos"))]
fn metal_debug_layer_controls_enabled() -> bool {
    false
}

#[cfg(all(not(test), feature = "apple", target_os = "ios"))]
fn metal_debug_layer_controls_enabled() -> bool {
    false
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn debug_print_f16_region_token_stats(
    arena: &MetalBufferArena,
    label: &str,
    offset: usize,
    num_tokens: usize,
    elems_per_token: usize,
) -> usize {
    let elem_count = num_tokens.saturating_mul(elems_per_token);
    let region = MetalRegion {
        name: label.to_owned(),
        offset,
        size: elem_count.saturating_mul(std::mem::size_of::<f16>()),
    };
    let ptr = unsafe { arena.host_ptr(&region) as *const u16 };
    let bits = unsafe { std::slice::from_raw_parts(ptr, elem_count) };
    let mut total_nonfinite = 0usize;
    for token in 0..num_tokens {
        let start = token.saturating_mul(elems_per_token);
        let end = start.saturating_add(elems_per_token);
        let mut nonfinite = 0usize;
        let mut max_abs = 0.0f32;
        for raw in &bits[start..end] {
            let value = f16::from_bits(*raw).to_f32();
            if value.is_finite() {
                max_abs = max_abs.max(value.abs());
            } else {
                nonfinite += 1;
            }
        }
        total_nonfinite += nonfinite;
        eprintln!(
            "metal debug finite: region={label} token={token} nonfinite={nonfinite}/{elems_per_token} max_abs={max_abs:e}"
        );
    }
    total_nonfinite
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn debug_f16_summary_json(
    arena: &MetalBufferArena,
    label: &str,
    offset: usize,
    num_tokens: usize,
    elems_per_token: usize,
) -> String {
    let elem_count = num_tokens.saturating_mul(elems_per_token);
    let region = MetalRegion {
        name: label.to_owned(),
        offset,
        size: elem_count.saturating_mul(std::mem::size_of::<f16>()),
    };
    let ptr = unsafe { arena.host_ptr(&region) as *const u16 };
    let bits = unsafe { std::slice::from_raw_parts(ptr, elem_count) };
    let mut finite_count = 0usize;
    let mut first_nonfinite_index = None;
    let mut max_abs = 0.0f32;
    let mut abs_sum = 0.0f64;
    let mut first_values = String::new();
    let mut selected_values = Vec::new();
    const SELECTED_TRACE_INDICES: &[usize] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 32, 64, 128, 256, 512, 1024, 1535,
    ];
    for (idx, raw) in bits.iter().enumerate() {
        let value = f16::from_bits(*raw).to_f32();
        if idx < 16 {
            if idx > 0 {
                first_values.push(',');
            }
            if value.is_finite() {
                first_values.push_str(&format!("{value:.9e}"));
            } else {
                first_values.push_str("null");
            }
        }
        if SELECTED_TRACE_INDICES.contains(&idx) {
            let value_json = if value.is_finite() {
                format!("{value:.9e}")
            } else {
                "null".to_owned()
            };
            selected_values.push(format!("{{\"index\":{idx},\"value\":{value_json}}}"));
        }
        if value.is_finite() {
            finite_count += 1;
            max_abs = max_abs.max(value.abs());
            abs_sum += value.abs() as f64;
        } else if first_nonfinite_index.is_none() {
            first_nonfinite_index = Some(idx);
        }
    }
    let mean_abs = if finite_count == 0 {
        0.0
    } else {
        (abs_sum / finite_count as f64) as f32
    };
    let first_nonfinite =
        first_nonfinite_index.map_or_else(|| "null".to_owned(), |idx| idx.to_string());
    format!(
        "\"{label}\":{{\"shape\":[{num_tokens},{elems_per_token}],\"total_count\":{elem_count},\"finite_count\":{finite_count},\"max_abs\":{max_abs:.9e},\"mean_abs\":{mean_abs:.9e},\"first_nonfinite_index\":{first_nonfinite},\"first_values\":[{first_values}],\"selected\":[{}]}}",
        selected_values.join(",")
    )
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn debug_write_layer_trace_json(
    arena: &MetalBufferArena,
    path: &std::path::Path,
    op: &'static str,
    phase: MetalPhase,
    layer_idx: usize,
    num_tokens: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    intermediate: usize,
    residual_offset: usize,
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    attn_out_offset: usize,
    gate_up_out_offset: usize,
    activated_offset: usize,
    ple_dim: usize,
    kv_cache_rows: usize,
    local_kv_cache_k_offset: usize,
    local_kv_cache_v_offset: usize,
    attention_kv_cache_k_offset: usize,
    attention_kv_cache_v_offset: usize,
    shared_kv_source_layer: Option<usize>,
    trace: Option<&MetalLayerTraceState>,
) -> Result<()> {
    let phase_name = match phase {
        MetalPhase::Decode => "decode",
        MetalPhase::Prefill { .. } => "prefill",
    };
    let mut summaries = Vec::new();
    if let Some(trace) = trace {
        summaries.extend([
            debug_f16_summary_json(
                arena,
                "input_to_layer",
                trace.input_to_layer.offset,
                num_tokens,
                hidden,
            ),
            debug_f16_summary_json(
                arena,
                "after_input_layernorm",
                trace.after_input_layernorm.offset,
                num_tokens,
                hidden,
            ),
            debug_f16_summary_json(
                arena,
                "q_projection",
                trace.q_projection.offset,
                num_tokens,
                q_dim,
            ),
            debug_f16_summary_json(
                arena,
                "k_projection",
                trace.k_projection.offset,
                num_tokens,
                kv_dim,
            ),
            debug_f16_summary_json(
                arena,
                "v_projection",
                trace.v_projection.offset,
                num_tokens,
                kv_dim,
            ),
            debug_f16_summary_json(
                arena,
                "after_q_norm",
                trace.after_q_norm.offset,
                num_tokens,
                q_dim,
            ),
            debug_f16_summary_json(
                arena,
                "after_k_norm",
                trace.after_k_norm.offset,
                num_tokens,
                kv_dim,
            ),
            debug_f16_summary_json(
                arena,
                "after_v_norm",
                trace.after_v_norm.offset,
                num_tokens,
                kv_dim,
            ),
            debug_f16_summary_json(
                arena,
                "after_rope_q",
                trace.after_rope_q.offset,
                num_tokens,
                q_dim,
            ),
            debug_f16_summary_json(
                arena,
                "after_rope_k",
                trace.after_rope_k.offset,
                num_tokens,
                kv_dim,
            ),
            debug_f16_summary_json(
                arena,
                "attention_output",
                trace.attention_output.offset,
                num_tokens,
                q_dim,
            ),
            debug_f16_summary_json(
                arena,
                "after_o_proj",
                trace.after_o_proj.offset,
                num_tokens,
                hidden,
            ),
            debug_f16_summary_json(
                arena,
                "after_post_attention_layernorm",
                trace.after_post_attention_layernorm.offset,
                num_tokens,
                hidden,
            ),
            debug_f16_summary_json(
                arena,
                "after_pre_feedforward_layernorm",
                trace.after_pre_feedforward_layernorm.offset,
                num_tokens,
                hidden,
            ),
            debug_f16_summary_json(
                arena,
                "gate_up_out",
                trace.gate_up_out.offset,
                num_tokens,
                intermediate.saturating_mul(2),
            ),
            debug_f16_summary_json(
                arena,
                "ffn_activation",
                trace.ffn_activation.offset,
                num_tokens,
                intermediate,
            ),
            debug_f16_summary_json(
                arena,
                "after_ffn_branch",
                trace.after_ffn_branch.offset,
                num_tokens,
                hidden,
            ),
            debug_f16_summary_json(
                arena,
                "after_post_feedforward_layernorm",
                trace.after_post_feedforward_layernorm.offset,
                num_tokens,
                hidden,
            ),
        ]);
        if let Some(region) = &trace.per_layer_input {
            summaries.push(debug_f16_summary_json(
                arena,
                "per_layer_input",
                region.offset,
                num_tokens,
                ple_dim,
            ));
        }
        if let Some(region) = &trace.per_layer_input_gate {
            summaries.push(debug_f16_summary_json(
                arena,
                "per_layer_input_gate",
                region.offset,
                num_tokens,
                ple_dim,
            ));
        }
        if let Some(region) = &trace.per_layer_projection {
            summaries.push(debug_f16_summary_json(
                arena,
                "per_layer_projection",
                region.offset,
                num_tokens,
                hidden,
            ));
        }
        if let Some(region) = &trace.post_per_layer_input_norm {
            summaries.push(debug_f16_summary_json(
                arena,
                "post_per_layer_input_norm",
                region.offset,
                num_tokens,
                hidden,
            ));
        }
    } else {
        summaries.extend([
            debug_f16_summary_json(arena, "after_rope_q", q_offset, num_tokens, q_dim),
            debug_f16_summary_json(arena, "after_rope_k", k_offset, num_tokens, kv_dim),
            debug_f16_summary_json(arena, "after_v_norm", v_offset, num_tokens, kv_dim),
            debug_f16_summary_json(
                arena,
                "attention_output",
                attn_out_offset,
                num_tokens,
                q_dim,
            ),
            debug_f16_summary_json(
                arena,
                "gate_up_out",
                gate_up_out_offset,
                num_tokens,
                intermediate.saturating_mul(2),
            ),
            debug_f16_summary_json(
                arena,
                "ffn_activation",
                activated_offset,
                num_tokens,
                intermediate,
            ),
        ]);
    }
    summaries.push(debug_f16_summary_json(
        arena,
        "final_residual_after_layer",
        residual_offset,
        num_tokens,
        hidden,
    ));
    summaries.extend([
        debug_f16_summary_json(
            arena,
            "local_kv_cache_k",
            local_kv_cache_k_offset,
            kv_cache_rows,
            kv_dim,
        ),
        debug_f16_summary_json(
            arena,
            "local_kv_cache_v",
            local_kv_cache_v_offset,
            kv_cache_rows,
            kv_dim,
        ),
        debug_f16_summary_json(
            arena,
            "attention_kv_cache_k",
            attention_kv_cache_k_offset,
            kv_cache_rows,
            kv_dim,
        ),
        debug_f16_summary_json(
            arena,
            "attention_kv_cache_v",
            attention_kv_cache_v_offset,
            kv_cache_rows,
            kv_dim,
        ),
    ]);
    let shared_kv_source_layer_json = shared_kv_source_layer
        .map(|layer| layer.to_string())
        .unwrap_or_else(|| "null".to_owned());
    let json = format!(
        "{{\"schema\":\"rvllm.gemma4_metal_layer_trace.v1\",\"op\":\"{op}\",\"phase\":\"{phase_name}\",\"layer\":{layer_idx},\"shared_kv_source_layer\":{shared_kv_source_layer_json},\"num_tokens\":{num_tokens},\"kv_cache_rows\":{kv_cache_rows},\"summaries\":{{{}}},\"claim\":\"rvLLM Metal layer debug summary only; no final logits, ANE, or production claim.\"}}\n",
        summaries.join(",")
    );
    std::fs::write(path, json).map_err(|_| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "failed to write Metal layer trace JSON",
            },
            model_ctx("debug_layer_trace"),
        )
    })
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "runtime-metal-backend",
        op,
        device: "apple-silicon",
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn model_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "model-metal-backend",
        op,
        device: "apple-silicon",
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
struct MetalState {
    residual: MetalRegion,
    final_norm: MetalRegion,
    lm_head: MetalRegion,
    logits: MetalRegion,
    normed_hidden: MetalRegion,
    sampled: MetalRegion,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
const MODEL_METAL_IN_FLIGHT_SLOTS: usize = 3;

/// A bounded completion record for one launched model step.
///
/// Production launches retain their committed command buffer here and return
/// immediately. Collection owns the wait, error propagation, readback, and
/// reclamation. CPU-ready debug completions may occupy the other ring slots.
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
enum InFlightCompletion {
    Reserved,
    Submitted(ModelGpuSubmission),
    Ready(Result<Vec<StepToken>>),
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
enum ModelGpuOutput {
    Prefill,
    Tokens {
        req_ids: Vec<rvllm_core::ReqId>,
        sampled: MetalRegion,
    },
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn ane_full_prompt_pages(
    handoff: &HandoffCapsule,
) -> std::result::Result<Vec<BlockId>, crate::ane_prefill::AnePrefillError> {
    use crate::ane_prefill::AnePrefillError;
    if handoff.positions.contains(&u32::MAX) {
        return Err(AnePrefillError::Invalid(
            "prompt position overflows its context length",
        ));
    }
    handoff.validate()?;
    let tokens = u32::try_from(handoff.tokens_flat.len())
        .map_err(|_| AnePrefillError::Invalid("prompt length exceeds u32"))?;
    if tokens == 0
        || handoff.req_ids.len() != 1
        || handoff.cu_seqlens.as_slice() != [0, tokens]
        || handoff.query_start_positions.as_slice() != [0]
        || handoff.context_lens.as_slice() != [tokens]
        || handoff.kv_chains.len() != 1
        || handoff.max_blocks_per_seq == 0
    {
        return Err(AnePrefillError::Invalid(
            "requires one complete prompt with allocator-owned paged KV",
        ));
    }
    let count = handoff
        .tokens_flat
        .len()
        .div_ceil(APPLE_KV_PAGE_SIZE as usize);
    let raw_pages = handoff
        .block_tables
        .get(..count)
        .ok_or(AnePrefillError::Invalid("prompt page table is incomplete"))?;
    let mut seen = std::collections::HashSet::with_capacity(count);
    if raw_pages
        .iter()
        .any(|&page| page == u32::MAX || !seen.insert(page))
    {
        return Err(AnePrefillError::Invalid(
            "prompt pages are missing or repeated",
        ));
    }
    if handoff.slot_mapping.len() != handoff.tokens_flat.len() {
        return Err(AnePrefillError::Invalid(
            "prompt slot mapping is incomplete",
        ));
    }
    for (position, &slot) in handoff.slot_mapping.iter().enumerate() {
        let page_size = APPLE_KV_PAGE_SIZE as usize;
        let expected = u64::from(raw_pages[position / page_size]) * page_size as u64
            + (position % page_size) as u64;
        if u64::try_from(slot).ok() != Some(expected) {
            return Err(AnePrefillError::Invalid(
                "prompt slots do not match the logical page order",
            ));
        }
    }
    Ok(raw_pages.iter().copied().map(BlockId).collect())
}

/// Everything collection needs after a command buffer has been committed.
///
/// The command buffer is retained by its ticket, so launch can return without
/// waiting. Its output region belongs to the execution slot retained by the
/// corresponding ring entry and cannot be reused before collection.
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
struct ModelGpuSubmission {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    output: ModelGpuOutput,
    perf_before: MetalProbePerfStats,
    wall_start: Instant,
    num_tokens: usize,
    is_decode: bool,
    #[cfg(feature = "metal-stage-instrumentation")]
    stage_profiler: MetalStageProfiler,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn metal_command_buffer_completion_error(status: MTLCommandBufferStatus) -> Option<RvllmError> {
    (status == MTLCommandBufferStatus::Error).then(|| {
        RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op: "metal_command_buffer_failed",
            },
            model_ctx("collect"),
        )
    })
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl ModelGpuSubmission {
    fn is_complete(&self) -> bool {
        matches!(
            self.command_buffer.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
    }

    fn wait(&self, perf: &MetalProbePerfCounters) {
        if self.is_complete() {
            return;
        }
        let wait_start = Instant::now();
        self.command_buffer.waitUntilCompleted();
        perf.add_forced_wait_duration(wait_start.elapsed());
    }

    fn finish(
        self,
        arena: &MetalBufferArena,
        perf: &MetalProbePerfCounters,
    ) -> Result<Vec<StepToken>> {
        if let Some(error) = metal_command_buffer_completion_error(self.command_buffer.status()) {
            perf.finish_step(
                self.is_decode,
                self.num_tokens as u64,
                self.perf_before,
                self.wall_start.elapsed(),
            );
            return Err(error);
        }
        if self.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "collect_step_not_complete",
                },
                model_ctx("collect"),
            ));
        }

        let gpu_start = self.command_buffer.GPUStartTime();
        let gpu_end = self.command_buffer.GPUEndTime();
        perf.last_step_gpu_execution_ns.set(
            (gpu_start.is_finite()
                && gpu_end.is_finite()
                && gpu_start > 0.0
                && gpu_end > gpu_start)
                .then(|| ((gpu_end - gpu_start) * 1e9) as u64),
        );
        #[cfg(feature = "metal-stage-instrumentation")]
        {
            let receipt = unsafe { self.stage_profiler.receipt() }.map_err(|_| {
                RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "model-metal-backend",
                        op: "resolve_stage_timing",
                    },
                    model_ctx("resolve_stage_timing"),
                )
            })?;
            *perf.last_stage_timing_receipt.borrow_mut() = Some(receipt);
        }

        let outputs = match self.output {
            ModelGpuOutput::Prefill => Vec::new(),
            ModelGpuOutput::Tokens { req_ids, sampled } => {
                let sampled_ptr = unsafe { arena.host_ptr(&sampled) as *const i32 };
                let sampled = unsafe { std::slice::from_raw_parts(sampled_ptr, req_ids.len()) };
                req_ids
                    .into_iter()
                    .zip(sampled.iter().copied())
                    .map(|(req_id, token_id)| StepToken {
                        req_id,
                        token_id: TokenId(token_id as u32),
                        finished: false,
                    })
                    .collect()
            }
        };
        perf.finish_step(
            self.is_decode,
            self.num_tokens as u64,
            self.perf_before,
            self.wall_start.elapsed(),
        );
        Ok(outputs)
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
enum ModelLaunchResult {
    Ready(Vec<StepToken>),
    Submitted(ModelGpuSubmission),
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
struct InFlightStep {
    ticket: AppleLaunchTicket,
    execution_slot: usize,
    completion: InFlightCompletion,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
struct ModelInFlightRing {
    slots: [Option<InFlightStep>; MODEL_METAL_IN_FLIGHT_SLOTS],
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl Default for ModelInFlightRing {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl ModelInFlightRing {
    fn reserve(
        &mut self,
        step_id: u64,
        kind: AppleLaunchKind,
        bucket: Option<RolloutBucket>,
    ) -> Result<AppleLaunchTicket> {
        if self.slots.iter().any(|slot| {
            slot.as_ref()
                .is_some_and(|step| step.ticket.step_id == step_id)
        }) {
            return Err(Self::stale_ticket_error("duplicate_in_flight_step_id"));
        }
        let slot_idx = self.slots.iter().position(Option::is_none).ok_or_else(|| {
            RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "in_flight_ring_full",
                },
                model_ctx("launch"),
            )
        })?;
        let ticket = AppleLaunchTicket {
            step_id,
            kind,
            bucket,
        };
        self.slots[slot_idx] = Some(InFlightStep {
            ticket,
            execution_slot: slot_idx,
            completion: InFlightCompletion::Reserved,
        });
        Ok(ticket)
    }

    fn execution_slot(&self, ticket: AppleLaunchTicket) -> Result<usize> {
        let step = self
            .slots
            .iter()
            .filter_map(Option::as_ref)
            .find(|step| step.ticket.step_id == ticket.step_id)
            .ok_or_else(|| Self::stale_ticket_error("execution_slot_stale_ticket"))?;
        if step.ticket != ticket {
            return Err(Self::stale_ticket_error("execution_slot_ticket_mismatch"));
        }
        Ok(step.execution_slot)
    }

    fn complete(
        &mut self,
        ticket: AppleLaunchTicket,
        result: Result<Vec<StepToken>>,
    ) -> Result<()> {
        let step = self
            .slots
            .iter_mut()
            .filter_map(Option::as_mut)
            .find(|step| step.ticket.step_id == ticket.step_id)
            .ok_or_else(|| Self::stale_ticket_error("complete_stale_ticket"))?;
        if step.ticket != ticket {
            return Err(Self::stale_ticket_error("complete_ticket_mismatch"));
        }
        if !matches!(step.completion, InFlightCompletion::Reserved) {
            return Err(Self::stale_ticket_error("complete_step_already_ready"));
        }
        step.completion = InFlightCompletion::Ready(result);
        Ok(())
    }

    fn submit(&mut self, ticket: AppleLaunchTicket, submission: ModelGpuSubmission) -> Result<()> {
        let step = self
            .slots
            .iter_mut()
            .filter_map(Option::as_mut)
            .find(|step| step.ticket.step_id == ticket.step_id)
            .ok_or_else(|| Self::stale_ticket_error("submit_stale_ticket"))?;
        if step.ticket != ticket {
            return Err(Self::stale_ticket_error("submit_ticket_mismatch"));
        }
        if !matches!(step.completion, InFlightCompletion::Reserved) {
            return Err(Self::stale_ticket_error("submit_step_already_owned"));
        }
        step.completion = InFlightCompletion::Submitted(submission);
        Ok(())
    }

    fn has_submitted(&self) -> bool {
        self.slots.iter().any(|slot| {
            matches!(
                slot.as_ref().map(|step| &step.completion),
                Some(InFlightCompletion::Submitted(_))
            )
        })
    }

    fn take_submission(&mut self, ticket: AppleLaunchTicket) -> Result<Option<ModelGpuSubmission>> {
        let step = self
            .slots
            .iter_mut()
            .filter_map(Option::as_mut)
            .find(|step| step.ticket.step_id == ticket.step_id)
            .ok_or_else(|| Self::stale_ticket_error("collect_stale_ticket"))?;
        if step.ticket != ticket {
            return Err(Self::stale_ticket_error("collect_ticket_mismatch"));
        }
        if !matches!(step.completion, InFlightCompletion::Submitted(_)) {
            return Ok(None);
        }
        match std::mem::replace(&mut step.completion, InFlightCompletion::Reserved) {
            InFlightCompletion::Submitted(submission) => Ok(Some(submission)),
            _ => unreachable!("submitted completion changed while exclusively borrowed"),
        }
    }

    fn restore_submission(
        &mut self,
        ticket: AppleLaunchTicket,
        submission: ModelGpuSubmission,
    ) -> Result<()> {
        self.submit(ticket, submission)
    }

    fn abort(&mut self, step_id: u64) {
        if let Some(slot) = self.slots.iter_mut().find(|slot| {
            slot.as_ref()
                .is_some_and(|step| step.ticket.step_id == step_id)
        }) {
            *slot = None;
        }
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        let slot_idx = self
            .slots
            .iter()
            .position(|slot| {
                slot.as_ref()
                    .is_some_and(|step| step.ticket.step_id == ticket.step_id)
            })
            .ok_or_else(|| Self::stale_ticket_error("collect_stale_ticket"))?;

        let matches = self.slots[slot_idx]
            .as_ref()
            .is_some_and(|step| step.ticket == ticket);
        if !matches {
            return Err(Self::stale_ticket_error("collect_ticket_mismatch"));
        }
        if matches!(
            self.slots[slot_idx].as_ref().map(|step| &step.completion),
            Some(InFlightCompletion::Reserved)
        ) {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "collect_step_not_complete",
                },
                model_ctx("collect"),
            ));
        }

        // Taking the slot before returning the completion makes collection the
        // sole reclamation point for both successful and failed GPU steps.
        let step = self.slots[slot_idx]
            .take()
            .expect("located in-flight slot must remain occupied");
        match step.completion {
            InFlightCompletion::Ready(result) => result,
            InFlightCompletion::Reserved | InFlightCompletion::Submitted(_) => {
                unreachable!("unfinished completion returned above")
            }
        }
    }

    fn stale_ticket_error(op: &'static str) -> RvllmError {
        RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op,
            },
            model_ctx("collect"),
        )
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    #[cfg(test)]
    fn is_submitted(&self, ticket: AppleLaunchTicket) -> bool {
        self.slots.iter().any(|slot| {
            slot.as_ref().is_some_and(|step| {
                step.ticket == ticket && matches!(step.completion, InFlightCompletion::Submitted(_))
            })
        })
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Debug, Default)]
pub struct RuntimeMetalBackend {
    prepared: bool,
    next_step_id: u64,
    last_ticket: Option<u64>,
    pending: Option<Vec<StepToken>>,
    ctx: Option<MetalContext>,
    pipelines: Option<PipelineCache>,
    arena: Option<MetalBufferArena>,
    state: Option<MetalState>,
}

/// Explicit configuration for a native Metal model owner. No environment
/// controls participate in this route; its precompiled library is mandatory.
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Clone, Copy, Debug)]
pub struct ModelMetalOptions {
    pub float_type: MetalFloatType,
    pub limits: rvllm_apple_metal::MetalModelLimits,
    pub kernels: rvllm_apple_metal::MetalKernelOptions,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
pub struct ModelMetalBackend {
    explicit_options: Option<ModelMetalOptions>,
    kernel_options: rvllm_apple_metal::MetalKernelOptions,
    pub model_dir: PathBuf,
    model_package: Option<AppleModelPackage>,
    metallib_f16: Option<PathBuf>,
    metallib_bf16: Option<PathBuf>,
    low_bit_residency_policy: MetalLowBitResidencyPolicy,
    pub prepared: bool,
    pub next_step_id: u64,
    prepared_model_layout_fingerprint: Option<[u8; 32]>,
    in_flight: ModelInFlightRing,
    ctx: Option<MetalContext>,
    pipelines: Option<PipelineCache>,
    arena: Option<MetalBufferArena>,
    pub state: Option<Gemma4MetalState>,
    execution_states: Vec<Gemma4MetalState>,
    float_type: Option<MetalFloatType>,
    debug_sync: bool,
    experimental_kv_int8: bool,
    perf: MetalProbePerfCounters,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct ModelKvLayerPageLayout {
    k_offset: usize,
    v_offset: usize,
    page_bytes_per_tensor: usize,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelKvPageLayout {
    total_pages: u32,
    block_size: u32,
    serialized_page_bytes: usize,
    layers: Vec<ModelKvLayerPageLayout>,
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
fn preflight_low_bit_replacement_descriptors(
    model_dir: &std::path::Path,
    float_type: MetalFloatType,
    replacements: &[MetalLowBitWeightReplacement],
) -> Result<Vec<MetalLowBitWeightReplacement>> {
    let mut sorted = replacements.to_vec();
    sorted.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    if sorted
        .windows(2)
        .any(|pair| pair[0].tensor_name == pair[1].tensor_name)
    {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "duplicate low-bit replacement tensor name",
            },
            model_ctx("prepare_low_bit_weights"),
        ));
    }
    if !sorted.is_empty() && float_type != MetalFloatType::F16 {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "W4A16/W8A16 sidecars require an F16 Metal model",
            },
            model_ctx("prepare_low_bit_weights"),
        ));
    }

    // The portable planner validates the complete sorted set before the
    // runtime creates a Metal context or arena. It rejects missing, MoE,
    // duplicate, shape-incompatible, and malformed storage descriptors.
    Gemma4MetalState::required_probe_model_arena_bytes_with_low_bit_replacements(
        model_dir, &sorted,
    )?;
    Ok(sorted)
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl ModelMetalBackend {
    #[must_use]
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            explicit_options: None,
            kernel_options: rvllm_apple_metal::MetalKernelOptions::default(),
            model_dir,
            model_package: None,
            metallib_f16: None,
            metallib_bf16: None,
            low_bit_residency_policy: MetalLowBitResidencyPolicy::HybridFallback,
            prepared: false,
            next_step_id: 0,
            prepared_model_layout_fingerprint: None,
            in_flight: ModelInFlightRing::default(),
            ctx: None,
            pipelines: None,
            arena: None,
            state: None,
            execution_states: Vec::new(),
            float_type: None,
            debug_sync: metal_debug_sync_enabled(),
            experimental_kv_int8: experimental_metal_kv_int8_enabled(),
            perf: MetalProbePerfCounters::default(),
        }
    }

    /// Construct an embedded backend with caller-owned precompiled Metal
    /// libraries. iOS preparation never consults environment variables and
    /// requires the selected dtype's path to be present here.
    #[must_use]
    pub fn with_precompiled_metallibs(
        model_dir: PathBuf,
        metallib_f16: Option<PathBuf>,
        metallib_bf16: Option<PathBuf>,
    ) -> Self {
        let mut backend = Self::new(model_dir);
        backend.metallib_f16 = metallib_f16;
        backend.metallib_bf16 = metallib_bf16;
        backend
    }

    /// Native checkpoint backend for embedded workers. Choices remain fixed
    /// through preparation, sizing, encoding and numeric identity reporting.
    #[must_use]
    pub fn with_options(model_dir: PathBuf, metallib: PathBuf, options: ModelMetalOptions) -> Self {
        let mut backend = Self::new(model_dir);
        match options.float_type {
            MetalFloatType::F16 => backend.metallib_f16 = Some(metallib),
            MetalFloatType::Bf16 => backend.metallib_bf16 = Some(metallib),
        }
        backend.explicit_options = Some(options);
        backend.kernel_options = options.kernels;
        backend.debug_sync = false;
        backend.experimental_kv_int8 = false;
        backend
    }

    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_low_bit_residency_policy(
        mut self,
        policy: MetalLowBitResidencyPolicy,
    ) -> Self {
        self.low_bit_residency_policy = policy;
        self
    }

    /// Open a validated Apple model package and select only the Metal assets
    /// for the current device/simulator platform. Package validation checks
    /// checksums before any model or shader resource is prepared.
    pub fn from_model_package_path(root: PathBuf) -> Result<Self> {
        let package = AppleModelPackage::open(&root).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "invalid Apple model package",
                },
                model_ctx("open_model_package"),
            )
        })?;
        let platform = ApplePackagePlatform::current().ok_or_else(|| {
            RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "apple_model_package_platform",
                },
                model_ctx("open_model_package"),
            )
        })?;
        let library = |float_type| {
            package
                .metal_library(platform, float_type)
                .ok()
                .map(|(library, _manifest)| library)
        };
        let f16 = library(ApplePackageFloatType::F16);
        let bf16 = library(ApplePackageFloatType::Bf16);
        if f16.is_none() && bf16.is_none() {
            return Err(RvllmError::apple(
                AppleError::MetallibMissing {
                    path: root.join("rvllm-apple-model.json"),
                },
                model_ctx("open_model_package"),
            ));
        }
        let mut backend = Self::with_precompiled_metallibs(root, f16, bf16);
        backend.model_package = Some(package);
        Ok(backend)
    }

    #[must_use]
    pub fn probe_perf_stats(&self) -> MetalProbePerfStats {
        self.perf.snapshot()
    }

    #[cfg(feature = "metal-stage-instrumentation")]
    pub fn last_stage_timing_receipt(&self) -> Option<serde_json::Value> {
        self.perf.last_stage_timing_receipt.borrow().clone()
    }

    /// Encoded research dispatches, not proof of GPU completion or accuracy.
    #[must_use]
    pub fn probe_research_dispatches(
        &self,
    ) -> Option<rvllm_apple_metal::research_evidence::ResearchDispatchSnapshot> {
        self.pipelines
            .as_ref()
            .map(PipelineCache::research_dispatch_snapshot)
    }

    #[must_use]
    pub fn probe_arena_stats(&self) -> Option<MetalProbeArenaStats> {
        self.arena.as_ref().map(|arena| MetalProbeArenaStats {
            region_count: arena.regions().len(),
            allocated_bytes: arena.allocated(),
            capacity_bytes: arena.capacity(),
        })
    }

    #[must_use]
    pub fn model_capacity(&self) -> Option<MetalModelCapacity> {
        let state = self.state.as_ref()?;
        let first_layer = state.layers.first()?;
        let float_type = state.float_type;
        // The opt-in currently exposes conversion/readback utilities only;
        // production Gemma KV pages remain exact native F16/BF16 storage.
        let experimental_kv_int8_active = false;
        let low_bit_projection_count = state
            .layers
            .iter()
            .filter(|layer| layer.low_bit_down_proj.is_some())
            .count() as u32;
        let low_bit_weight_bytes = state
            .layers
            .iter()
            .filter_map(|layer| layer.low_bit_down_proj)
            .try_fold(0_u64, |total, projection| {
                total.checked_add(projection.resident_bytes() as u64)
            })
            .unwrap_or(u64::MAX);
        Some(MetalModelCapacity {
            max_context_tokens: state.max_probe_tokens,
            max_batch_tokens: state.max_batch_tokens,
            max_batch_sequences: state.max_batch_sequences,
            kv_page_size: first_layer.block_size,
            physical_kv_pages: first_layer.num_blocks_total,
            recommended_working_set_bytes: state.memory_budget.recommended_working_set_bytes,
            reserve_bytes: state.memory_budget.reserve_bytes,
            usable_bytes: state.memory_budget.usable_bytes,
            weights_bytes: state.memory_budget.weights_bytes,
            scratch_slot_bytes: state.memory_budget.scratch_slot_bytes,
            scratch_budget_bytes: state.memory_budget.scratch_budget_bytes,
            metadata_bytes: state.memory_budget.metadata_bytes,
            kv_budget_bytes: state.memory_budget.kv_budget_bytes,
            kv_page_bytes: state.memory_budget.kv_page_bytes,
            allocated_kv_bytes: state.memory_budget.allocated_kv_bytes,
            prepared_arena_bytes: state.memory_budget.prepared_arena_bytes,
            max_useful_kv_pages: state.memory_budget.max_useful_kv_pages,
            admission_required_kv_pages: state.memory_budget.admission_required_kv_pages,
            metal_float_type: float_type.report_name(),
            kv_storage_format: if experimental_kv_int8_active {
                "int8-experimental-v1"
            } else {
                float_type.report_name()
            },
            experimental_kv_int8_opt_in: self.experimental_kv_int8,
            experimental_kv_int8_active,
            low_bit_projection_count,
            low_bit_weight_bytes,
            numeric_abi_version: METAL_NUMERIC_ABI_VERSION,
            numeric_abi_fingerprint: metal_numeric_abi_fingerprint_for_state(
                float_type,
                self.experimental_kv_int8,
                experimental_kv_int8_active,
                state,
                self.model_package
                    .as_ref()
                    .map(AppleModelPackage::identity_fingerprint),
                self.low_bit_residency_policy,
                self.kernel_options,
            ),
        })
    }

    /// Capture a request's logical KV prefix after its Metal ticket has been
    /// collected. The caller supplies the live allocator-owned page order.
    /// This converts BF16 to ANE FP16 once; it never starts ANE execution.
    pub fn export_ane_prefill(
        &mut self,
        pages: &[BlockId],
        tokens: usize,
    ) -> std::result::Result<crate::ane_prefill::AnePrefillSnapshot, KvPageIoError> {
        use crate::ane_prefill::{capture_prefill, PrefillLayerShape, PrefillScalarType};
        self.ensure_kv_page_io_idle()?;
        let capacity = self
            .model_capacity()
            .ok_or(KvPageIoError::BackendUnavailable)?;
        if tokens > capacity.max_context_tokens {
            return Err(KvPageIoError::LayoutMismatch(
                "ANE prefill exceeds prepared Metal context",
            ));
        }
        let state = self
            .state
            .as_ref()
            .ok_or(KvPageIoError::BackendUnavailable)?;
        if state
            .layers
            .iter()
            .any(|layer| layer.shared_kv_source_layer.is_some())
        {
            return Err(KvPageIoError::LayoutMismatch(
                "ANE prefill export does not yet support cross-layer shared KV",
            ));
        }
        let dtype = match state.float_type {
            MetalFloatType::F16 => PrefillScalarType::F16,
            MetalFloatType::Bf16 => PrefillScalarType::Bf16,
        };
        let shapes: Vec<_> = state
            .layers
            .iter()
            .map(|layer| PrefillLayerShape {
                query_heads: layer.dims.num_heads,
                kv_heads: layer.dims.num_kv_heads,
                head_dim: layer.dims.head_dim,
                sliding_window: (layer.dims.attention_window != 0)
                    .then_some(layer.dims.attention_window as usize),
            })
            .collect();
        capture_prefill(
            self,
            pages,
            tokens,
            &shapes,
            dtype,
            capacity.numeric_abi_fingerprint,
        )
    }

    /// Run a complete single-request prompt and sample its first output on
    /// Metal, then capture the prompt KV for ANE. This operation is synchronous
    /// and invokes no ANE API. The caller retains ownership of the supplied
    /// allocator chain throughout the call. Stop on model EOS before asking an
    /// ANE decoder to consume the returned token at `start.next_position()`.
    pub fn prefill_for_ane(
        &mut self,
        handoff: &HandoffCapsule,
    ) -> std::result::Result<crate::ane_prefill::AneDecodeStart, crate::ane_prefill::AnePrefillError>
    {
        use crate::ane_prefill::{AneDecodeStart, AnePrefillError, AnePrefillTimes};
        self.ensure_prepared("prefill_for_ane")?;
        self.ensure_kv_page_io_idle()?;
        let pages = ane_full_prompt_pages(handoff)?;
        self.validate_paged_kv_contract(handoff, "prefill_for_ane")?;
        let layout = self.kv_page_layout()?;
        for &page in &pages {
            Self::validate_kv_page(&layout, page)?;
        }
        let state = self
            .state
            .as_ref()
            .ok_or(KvPageIoError::BackendUnavailable)?;
        if state
            .layers
            .iter()
            .any(|layer| layer.shared_kv_source_layer.is_some())
        {
            return Err(AnePrefillError::Invalid(
                "cross-layer shared KV is not supported",
            ));
        }
        if self.debug_sync
            || (self.explicit_options.is_none() && metal_debug_layer_controls_enabled())
            || (self.explicit_options.is_none() && metal_debug_skip_final_logits_enabled())
        {
            return Err(AnePrefillError::Invalid(
                "requires complete asynchronous Metal execution without debug skips",
            ));
        }
        let vocab = state.vocab_size;
        let before = self.probe_perf_stats();
        let metal_started = Instant::now();
        let (ticket, slot) = self.reserve_step(AppleLaunchKind::Prefill, None)?;
        let result = self.run_prefill_step(handoff, slot, true);
        let ticket = self.finish_launch(ticket, result)?;
        let outputs = self.collect(ticket)?;
        let [first] = outputs.as_slice() else {
            return Err(AnePrefillError::Invalid(
                "Metal did not return one first token",
            ));
        };
        if first.req_id != handoff.req_ids[0] || first.token_id.raw() as usize >= vocab {
            return Err(AnePrefillError::Invalid(
                "Metal returned an invalid first token or request",
            ));
        }
        let metal_execution_ms = metal_started.elapsed().as_secs_f64() * 1000.0;
        let after = self.probe_perf_stats();
        let capture_started = Instant::now();
        let cache = self.export_ane_prefill(&pages, handoff.tokens_flat.len())?;
        let kv_capture_ms = capture_started.elapsed().as_secs_f64() * 1000.0;
        Ok(AneDecodeStart {
            req_id: first.req_id,
            first_token: first.token_id,
            cache,
            times: AnePrefillTimes {
                metal_execution_ms,
                gpu_execution_ms: after.last_step_gpu_execution_ns.map(|ns| ns as f64 / 1e6),
                host_non_wait_ms: after.cpu_encode_ns.saturating_sub(before.cpu_encode_ns) as f64
                    / 1e6,
                command_buffer_wait_ms: after
                    .command_buffer_wait_ns
                    .saturating_sub(before.command_buffer_wait_ns)
                    as f64
                    / 1e6,
                kv_capture_ms,
            },
        })
    }

    fn kv_page_layout(&self) -> std::result::Result<ModelKvPageLayout, KvPageIoError> {
        if !self.prepared {
            return Err(KvPageIoError::BackendUnavailable);
        }
        let state = self
            .state
            .as_ref()
            .ok_or(KvPageIoError::BackendUnavailable)?;
        let first = state
            .layers
            .first()
            .ok_or(KvPageIoError::LayoutMismatch("model has no KV layers"))?;
        if first.block_size != APPLE_KV_PAGE_SIZE {
            return Err(KvPageIoError::LayoutMismatch(
                "Metal layer does not use Apple-v1 32-token pages",
            ));
        }
        if first.num_blocks_total == 0 {
            return Err(KvPageIoError::LayoutMismatch(
                "Metal KV page pool has zero physical pages",
            ));
        }

        let mut serialized_page_bytes = 0usize;
        let mut layers = Vec::with_capacity(state.layers.len());
        for layer in &state.layers {
            if layer.block_size != first.block_size
                || layer.num_blocks_total != first.num_blocks_total
            {
                return Err(KvPageIoError::LayoutMismatch(
                    "Metal layers disagree on physical page geometry",
                ));
            }
            let tensor_page_bytes = (layer.block_size as usize)
                .checked_mul(layer.dims.kv_dim)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                .ok_or(KvPageIoError::LayoutMismatch("KV page byte size overflow"))?;
            let tensor_bytes = tensor_page_bytes
                .checked_mul(layer.num_blocks_total as usize)
                .ok_or(KvPageIoError::LayoutMismatch(
                    "KV tensor byte size overflow",
                ))?;
            if layer.kv_cache_k.size != tensor_bytes || layer.kv_cache_v.size != tensor_bytes {
                return Err(KvPageIoError::LayoutMismatch(
                    "Metal KV regions do not match page geometry",
                ));
            }
            serialized_page_bytes = serialized_page_bytes
                .checked_add(tensor_page_bytes.checked_mul(2).ok_or(
                    KvPageIoError::LayoutMismatch("serialized KV page byte size overflow"),
                )?)
                .ok_or(KvPageIoError::LayoutMismatch(
                    "serialized KV page byte size overflow",
                ))?;
            layers.push(ModelKvLayerPageLayout {
                k_offset: layer.kv_cache_k.offset,
                v_offset: layer.kv_cache_v.offset,
                page_bytes_per_tensor: tensor_page_bytes,
            });
        }
        if serialized_page_bytes == 0 {
            return Err(KvPageIoError::LayoutMismatch(
                "serialized KV page has zero bytes",
            ));
        }
        Ok(ModelKvPageLayout {
            total_pages: first.num_blocks_total,
            block_size: first.block_size,
            serialized_page_bytes,
            layers,
        })
    }

    fn ensure_kv_page_io_idle(&self) -> std::result::Result<(), KvPageIoError> {
        if self.in_flight.has_submitted() {
            Err(KvPageIoError::BackendBusy)
        } else {
            Ok(())
        }
    }

    fn validate_kv_page(
        layout: &ModelKvPageLayout,
        page: BlockId,
    ) -> std::result::Result<usize, KvPageIoError> {
        if page.0 >= layout.total_pages {
            return Err(KvPageIoError::InvalidPageId {
                page: page.0,
                total_pages: layout.total_pages,
            });
        }
        Ok(page.0 as usize)
    }

    #[must_use]
    pub const fn metal_debug_sync_enabled(&self) -> bool {
        self.debug_sync
    }

    #[must_use]
    pub fn metal_float_type(&self) -> MetalFloatType {
        self.float_type
            .or_else(|| self.state.as_ref().map(|state| state.float_type))
            .unwrap_or(MetalFloatType::F16)
    }

    #[must_use]
    pub fn metal_compute_dtype_report(&self) -> &'static str {
        self.metal_float_type().report_name()
    }

    #[must_use]
    pub fn metal_weight_dtype_report(&self) -> &'static str {
        self.metal_float_type().report_name()
    }

    fn reserve_step(
        &mut self,
        kind: AppleLaunchKind,
        bucket: Option<RolloutBucket>,
    ) -> Result<(AppleLaunchTicket, usize)> {
        if self.next_step_id == u64::MAX {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "step_id_exhausted",
                },
                model_ctx("launch"),
            ));
        }
        let ticket = self.in_flight.reserve(self.next_step_id, kind, bucket)?;
        let execution_slot = self.in_flight.execution_slot(ticket)?;
        Ok((ticket, execution_slot))
    }

    /// Paged KV metadata is meaningful only for the exact model/layout that
    /// sized and prepared the physical page pool. Reject missing or stale
    /// fingerprints before reserving a ticket or touching encoder state.
    fn validate_paged_kv_contract(&self, handoff: &HandoffCapsule, op: &'static str) -> Result<()> {
        // Treat any part of the paged ABI as opting into the full contract.
        // In particular, do not let a producer omit `kv_chains` while still
        // supplying physical block tables or slots to bypass the fingerprint
        // check. A legacy ordinal handoff leaves every one of these fields at
        // its constructor default.
        let has_paged_metadata = !handoff.kv_chains.is_empty()
            || handoff.max_blocks_per_seq != 0
            || !handoff.block_tables.is_empty()
            || !handoff.slot_mapping.is_empty()
            || handoff.model_layout_fingerprint != [0; 32];
        if !has_paged_metadata {
            return Ok(());
        }
        let expected = self.prepared_model_layout_fingerprint.ok_or_else(|| {
            RvllmError::apple(
                AppleError::HandoffMalformed {
                    reason: "paged KV backend has no prepared model layout fingerprint",
                },
                model_ctx(op),
            )
        })?;
        if expected == [0; 32] {
            return Err(RvllmError::apple(
                AppleError::HandoffMalformed {
                    reason: "paged KV prepared model layout fingerprint must be nonzero",
                },
                model_ctx(op),
            ));
        }
        if handoff.model_layout_fingerprint == [0; 32] {
            return Err(RvllmError::apple(
                AppleError::HandoffMalformed {
                    reason: "paged KV handoff model layout fingerprint must be nonzero",
                },
                model_ctx(op),
            ));
        }
        if handoff.model_layout_fingerprint != expected {
            return Err(RvllmError::apple(
                AppleError::HandoffMalformed {
                    reason: "paged KV handoff model layout fingerprint mismatch",
                },
                model_ctx(op),
            ));
        }
        Ok(())
    }

    fn finish_launch(
        &mut self,
        ticket: AppleLaunchTicket,
        result: Result<ModelLaunchResult>,
    ) -> Result<AppleLaunchTicket> {
        match result {
            Ok(ModelLaunchResult::Ready(outputs)) => {
                if let Err(error) = self.in_flight.complete(ticket, Ok(outputs)) {
                    self.in_flight.abort(ticket.step_id);
                    return Err(error);
                }
                self.next_step_id += 1;
                Ok(ticket)
            }
            Ok(ModelLaunchResult::Submitted(submission)) => {
                if let Err(error) = self.in_flight.submit(ticket, submission) {
                    self.in_flight.abort(ticket.step_id);
                    return Err(error);
                }
                self.next_step_id += 1;
                Ok(ticket)
            }
            Err(error) => {
                self.in_flight.abort(ticket.step_id);
                Err(error)
            }
        }
    }

    #[must_use]
    pub fn metal_moe_router_weight_dtype_report(&self) -> &'static str {
        match self.metal_float_type() {
            MetalFloatType::F16 => "float32",
            MetalFloatType::Bf16 => "bfloat16",
        }
    }

    /// Returns whether the experimental Metal KV int8 utility path was
    /// explicitly opted in with `RVLLM_EXPERIMENTAL_METAL_KV_INT8=1`.
    ///
    /// The production probe path still uses F16 KV cache storage regardless of
    /// this flag; the compressed path is currently test-only/readback-only.
    #[must_use]
    pub const fn experimental_kv_int8_enabled(&self) -> bool {
        self.experimental_kv_int8
    }

    /// Poll a submitted step without blocking the accelerator worker.
    ///
    /// `Ok(None)` means Metal still owns the shared scratch slot. The ticket
    /// remains valid and must be polled again or passed to `collect`.
    pub fn try_collect(&mut self, ticket: AppleLaunchTicket) -> Result<Option<Vec<StepToken>>> {
        self.ensure_prepared("try_collect")?;
        self.collect_model_step(ticket, false)
    }

    fn collect_model_step(
        &mut self,
        ticket: AppleLaunchTicket,
        wait: bool,
    ) -> Result<Option<Vec<StepToken>>> {
        if let Some(submission) = self.in_flight.take_submission(ticket)? {
            if !wait && !submission.is_complete() {
                self.in_flight.restore_submission(ticket, submission)?;
                return Ok(None);
            }
            if wait {
                submission.wait(&self.perf);
            }
            let arena = self.arena.as_ref().ok_or_else(|| {
                RvllmError::apple(
                    AppleError::NotPrepared {
                        backend: "model-metal-backend",
                    },
                    model_ctx("collect"),
                )
            })?;
            let result = submission.finish(arena, &self.perf);
            self.in_flight.complete(ticket, result)?;
        }
        self.in_flight.collect(ticket).map(Some)
    }

    pub fn probe_read_decode_logits_f32(&self, num_tokens: usize) -> Result<Vec<f32>> {
        let state = self.state.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("debug_read_decode_logits_f32"),
            )
        })?;
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("debug_read_decode_logits_f32"),
            )
        })?;
        let elem_count = num_tokens.checked_mul(state.vocab_size).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "debug logits element count overflow",
                },
                model_ctx("debug_read_decode_logits_f32"),
            )
        })?;
        let byte_count = elem_count
            .checked_mul(std::mem::size_of::<f16>())
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "debug logits byte count overflow",
                    },
                    model_ctx("debug_read_decode_logits_f32"),
                )
            })?;
        if byte_count > state.logits.size {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "debug logits read exceeds logits buffer",
                },
                model_ctx("debug_read_decode_logits_f32"),
            ));
        }

        let ctx = self.ctx.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("debug_read_decode_logits_f32"),
            )
        })?;
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("debug_read_decode_logits_f32"),
            )
        })?;
        let num_tokens_u32 = u32::try_from(num_tokens).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "debug logits token count exceeds u32",
                },
                model_ctx("debug_read_decode_logits_f32"),
            )
        })?;
        let hidden_u32 = u32::try_from(state.hidden_size).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "hidden_size exceeds u32",
                },
                model_ctx("debug_read_decode_logits_f32"),
            )
        })?;
        let vocab_u32 = u32::try_from(state.vocab_size).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "vocab_size exceeds u32",
                },
                model_ctx("debug_read_decode_logits_f32"),
            )
        })?;
        unsafe {
            metal_finalize_logits_blocking(
                ctx,
                pipelines,
                arena,
                num_tokens_u32,
                hidden_u32,
                vocab_u32,
                state.rms_norm_eps,
                state.final_logit_softcap,
                state.residual.offset,
                state.final_norm.offset,
                state.lm_head.offset,
                state.logits.offset,
                state.normed_hidden.offset,
                state.sampled.offset,
            )?;
        }
        self.perf.add_command_buffers(1);
        self.perf
            .add_final_logits_encoders(metal_finalize_logits_encoder_count(
                num_tokens_u32,
                hidden_u32,
                vocab_u32,
                state.final_logit_softcap,
            ));
        self.perf.add_forced_wait();

        let logits_ptr = unsafe { arena.host_ptr(&state.logits) as *const u16 };
        let logits_bits = unsafe { std::slice::from_raw_parts(logits_ptr, elem_count) };
        let float_type = self.metal_float_type();
        Ok(logits_bits
            .iter()
            .map(|bits| metal_u16_to_f32(*bits, float_type))
            .collect())
    }

    #[cfg(test)]
    fn debug_read_decode_logits_f32(&self, num_tokens: usize) -> Result<Vec<f32>> {
        self.probe_read_decode_logits_f32(num_tokens)
    }

    #[cfg(test)]
    fn debug_read_residual_f32(&self, num_tokens: usize) -> Result<Vec<f32>> {
        let state = self.state.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("debug_read_residual_f32"),
            )
        })?;
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("debug_read_residual_f32"),
            )
        })?;
        let elem_count = num_tokens.checked_mul(state.hidden_size).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "debug residual element count overflow",
                },
                model_ctx("debug_read_residual_f32"),
            )
        })?;
        let byte_count = elem_count
            .checked_mul(std::mem::size_of::<f16>())
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "debug residual byte count overflow",
                    },
                    model_ctx("debug_read_residual_f32"),
                )
            })?;
        if byte_count > state.residual.size {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "debug residual read exceeds residual buffer",
                },
                model_ctx("debug_read_residual_f32"),
            ));
        }

        let ptr = unsafe { arena.host_ptr(&state.residual) as *const u16 };
        let bits = unsafe { std::slice::from_raw_parts(ptr, elem_count) };
        let float_type = self.metal_float_type();
        Ok(bits
            .iter()
            .map(|bits| metal_u16_to_f32(*bits, float_type))
            .collect())
    }

    fn ensure_prepared(&self, op: &'static str) -> Result<()> {
        if self.prepared {
            Ok(())
        } else {
            Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx(op),
            ))
        }
    }

    fn initialize_model_resources(&mut self) -> Result<Gemma4MetalState> {
        if self
            .explicit_options
            .is_some_and(|options| options.kernels.quantized_bf16_accumulation)
        {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "explicit native Metal libraries require FP32 accumulation",
                },
                model_ctx("prepare"),
            ));
        }
        let float_type = if let Some(options) = self.explicit_options {
            options.float_type
        } else {
            configured_metal_float_type(&self.model_dir)?
        };
        self.kernel_options = self
            .explicit_options
            .map(|options| options.kernels)
            .unwrap_or_else(rvllm_apple_metal::MetalKernelOptions::from_development_environment);
        let low_bit_replacements = self.preflight_package_low_bit_replacements(float_type)?;
        let additional_weight_bytes =
            Self::hybrid_low_bit_arena_budget_bytes(&low_bit_replacements)?;
        let mut ctx = MetalContext::new()?;
        load_model_kernel_library(
            &mut ctx,
            float_type,
            self.explicit_metallib_path(float_type),
            self.kernel_options,
        )?;
        let mut pipelines = PipelineCache::with_kernel_options(self.kernel_options);
        pipelines.compile_all_for_type(&ctx, float_type)?;
        self.perf.add_library_compile();
        self.perf
            .add_pipeline_state_compiles(pipelines.len() as u64);

        let (mut arena, mut state, memory_report) = if let Some(options) = self.explicit_options {
            if !low_bit_replacements.is_empty() || self.model_package.is_some() {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "explicit native Metal options cannot load package sidecars",
                    },
                    model_ctx("prepare"),
                ));
            }
            let load_plan =
                rvllm_apple_metal::gemma4_model::MetalModelLoadPlan::new_with_research_candidate(
                    &ctx,
                    &self.model_dir,
                    options.limits,
                    options.kernels.research,
                )?;
            let memory_report = *load_plan.memory_report();
            let mut arena = MetalBufferArena::new(ctx.device(), load_plan.arena_bytes())?;
            let state = load_plan.load(&ctx, &mut arena, float_type)?;
            (arena, state, memory_report)
        } else {
            let (arena_bytes, memory_report) = match self.low_bit_residency_policy {
            MetalLowBitResidencyPolicy::HybridFallback => {
                Gemma4MetalState::required_probe_model_arena_bytes_for_device_with_additional_weights_and_research(
                    &ctx,
                    &self.model_dir,
                    additional_weight_bytes,
                    self.kernel_options.research,
                )?
            }
            MetalLowBitResidencyPolicy::ReplaceNative => {
                Gemma4MetalState::required_probe_model_arena_bytes_for_device_with_low_bit_replacements_and_research(
                    &ctx,
                    &self.model_dir,
                    &low_bit_replacements,
                    self.kernel_options.research,
                )?
            }
        };
            let mut arena = MetalBufferArena::new(ctx.device(), arena_bytes)?;
            let state = match self.low_bit_residency_policy {
                MetalLowBitResidencyPolicy::HybridFallback => {
                    Gemma4MetalState::load_probe_model_with_float_type_additional_weights_and_research(
                        &ctx,
                        &mut arena,
                        &self.model_dir,
                        float_type,
                        additional_weight_bytes,
                        self.kernel_options.research,
                    )?
                }
                MetalLowBitResidencyPolicy::ReplaceNative => {
                    Gemma4MetalState::load_probe_model_with_float_type_low_bit_replacements_and_research(
                        &ctx,
                        &mut arena,
                        &self.model_dir,
                        float_type,
                        &low_bit_replacements,
                        self.kernel_options.research,
                    )?
                }
            };
            (arena, state, memory_report)
        };
        self.install_package_low_bit_sidecars(
            &mut arena,
            &mut state,
            float_type,
            &low_bit_replacements,
        )?;
        if state.memory_budget != memory_report {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "Metal memory budget changed during model preparation",
                },
                model_ctx("prepare"),
            ));
        }

        self.execution_states = (0..MODEL_METAL_IN_FLIGHT_SLOTS)
            .map(|slot| state.state_for_execution_slot(slot))
            .collect::<Result<Vec<_>>>()?;
        self.ctx = Some(ctx);
        self.pipelines = Some(pipelines);
        self.arena = Some(arena);
        self.float_type = Some(float_type);
        Ok(state)
    }

    fn preflight_package_low_bit_replacements(
        &self,
        float_type: MetalFloatType,
    ) -> Result<Vec<MetalLowBitWeightReplacement>> {
        let Some(package) = self.model_package.as_ref() else {
            return Ok(Vec::new());
        };
        let replacements = package
            .manifest()
            .low_bit_tensors
            .iter()
            .map(|tensor| {
                let values = usize::try_from(tensor.packed_values.bytes).map_err(|_| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "low-bit packed value byte count exceeds address space",
                        },
                        model_ctx("prepare_low_bit_weights"),
                    )
                })?;
                let scales = usize::try_from(tensor.scales.bytes).map_err(|_| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "low-bit scale byte count exceeds address space",
                        },
                        model_ctx("prepare_low_bit_weights"),
                    )
                })?;
                Ok(MetalLowBitWeightReplacement {
                    tensor_name: tensor.tensor_name.clone(),
                    role: tensor.role,
                    format: tensor.format,
                    shape: [tensor.shape[0] as usize, tensor.shape[1] as usize],
                    packed_values_bytes: values,
                    scales_bytes: scales,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        preflight_low_bit_replacement_descriptors(&self.model_dir, float_type, &replacements)
    }

    fn hybrid_low_bit_arena_budget_bytes(
        replacements: &[MetalLowBitWeightReplacement],
    ) -> Result<usize> {
        let mut total = 0usize;
        for replacement in replacements {
            total = total
                .checked_add(15)
                .map(|bytes| bytes & !15)
                .and_then(|bytes| bytes.checked_add(replacement.packed_values_bytes))
                .and_then(|bytes| bytes.checked_add(15))
                .map(|bytes| bytes & !15)
                .and_then(|bytes| bytes.checked_add(replacement.scales_bytes))
                .ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "low-bit arena byte count overflow",
                        },
                        model_ctx("prepare_low_bit_weights"),
                    )
                })?;
        }
        Ok(total)
    }

    fn install_package_low_bit_sidecars(
        &self,
        arena: &mut MetalBufferArena,
        state: &mut Gemma4MetalState,
        float_type: MetalFloatType,
        replacements: &[MetalLowBitWeightReplacement],
    ) -> Result<()> {
        let Some(package) = self.model_package.as_ref() else {
            if replacements.is_empty() {
                return Ok(());
            }
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "low-bit replacement plan has no authenticated package",
                },
                model_ctx("prepare_low_bit_weights"),
            ));
        };
        if replacements.is_empty() {
            return Ok(());
        }
        if float_type != MetalFloatType::F16 {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "W4A16/W8A16 sidecars require an F16 Metal model",
                },
                model_ctx("prepare_low_bit_weights"),
            ));
        }

        for replacement in replacements {
            let tensor = package
                .low_bit_tensor(&replacement.tensor_name)
                .ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "planned low-bit tensor is missing from authenticated package",
                        },
                        model_ctx("prepare_low_bit_weights"),
                    )
                })?;
            let layer_index = state
                .layers
                .iter()
                .position(|layer| match replacement.role {
                    rvllm_apple::AppleLowBitTensorRole::QueryProjection => {
                        layer.q_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::KeyProjection => {
                        layer.k_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::ValueProjection => {
                        layer.v_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::OutputProjection => {
                        layer.o_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::DenseGateProjection => {
                        layer.gate_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::DenseUpProjection => {
                        layer.up_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::DenseDownProjection => {
                        layer.down_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::LmHead => false,
                })
                .ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason:
                                "low-bit tensor does not match its prepared dense projection role",
                        },
                        model_ctx("prepare_low_bit_weights"),
                    )
                })?;
            let layer = &state.layers[layer_index];
            if layer.moe.is_some() {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "initial Metal low-bit route does not support MoE layers",
                    },
                    model_ctx("prepare_low_bit_weights"),
                ));
            }
            let intermediate = replacements
                .iter()
                .find(|candidate| {
                    candidate.role == rvllm_apple::AppleLowBitTensorRole::DenseDownProjection
                        && candidate.tensor_name == layer.down_proj_name
                })
                .map(|candidate| candidate.shape[1])
                .or_else(|| {
                    layer.down_proj.as_ref().and_then(|region| {
                        region
                            .size
                            .checked_div(std::mem::size_of::<f16>())
                            .and_then(|elements| elements.checked_div(state.hidden_size))
                    })
                })
                .ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "prepared dense FFN intermediate size is unavailable",
                        },
                        model_ctx("prepare_low_bit_weights"),
                    )
                })?;
            let expected_shape = match replacement.role {
                rvllm_apple::AppleLowBitTensorRole::QueryProjection => {
                    [layer.dims.q_dim, state.hidden_size]
                }
                rvllm_apple::AppleLowBitTensorRole::KeyProjection
                | rvllm_apple::AppleLowBitTensorRole::ValueProjection => {
                    [layer.dims.kv_dim, state.hidden_size]
                }
                rvllm_apple::AppleLowBitTensorRole::OutputProjection => {
                    [state.hidden_size, layer.dims.q_dim]
                }
                rvllm_apple::AppleLowBitTensorRole::DenseGateProjection
                | rvllm_apple::AppleLowBitTensorRole::DenseUpProjection => {
                    [intermediate, state.hidden_size]
                }
                rvllm_apple::AppleLowBitTensorRole::DenseDownProjection => {
                    [state.hidden_size, intermediate]
                }
                rvllm_apple::AppleLowBitTensorRole::LmHead => {
                    unreachable!("LM head rejected above")
                }
            };
            if replacement.shape != expected_shape
                || tensor.shape
                    != [
                        u32::try_from(expected_shape[0]).map_err(|_| {
                            RvllmError::apple(
                                AppleError::InvalidWeightBlob {
                                    reason: "Metal hidden size exceeds low-bit ABI",
                                },
                                model_ctx("prepare_low_bit_weights"),
                            )
                        })?,
                        u32::try_from(expected_shape[1]).map_err(|_| {
                            RvllmError::apple(
                                AppleError::InvalidWeightBlob {
                                    reason: "Metal intermediate size exceeds low-bit ABI",
                                },
                                model_ctx("prepare_low_bit_weights"),
                            )
                        })?,
                    ]
            {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason:
                            "low-bit tensor shape does not match prepared dense projection role",
                    },
                    model_ctx("prepare_low_bit_weights"),
                ));
            }

            let values_len = replacement.packed_values_bytes;
            let values = arena.region(
                &format!(
                    "metal_low_bit_layer_{layer_index}_{}_packed_values",
                    replacement.role.report_name()
                ),
                values_len,
                16,
            )?;
            let scale_bytes = replacement.scales_bytes;
            let scales = arena.region(
                &format!(
                    "metal_low_bit_layer_{layer_index}_{}_scales",
                    replacement.role.report_name()
                ),
                scale_bytes,
                16,
            )?;
            // Re-authenticate the exact mutable package bytes directly into
            // the pre-budgeted Metal arena. Fixed-size destination slices and
            // the package reader's EOF probe prevent an oversized transient
            // staging allocation.
            unsafe {
                let values_destination =
                    std::slice::from_raw_parts_mut(arena.host_ptr(&values), values.size);
                let scales_destination =
                    std::slice::from_raw_parts_mut(arena.host_ptr(&scales), scales.size);
                package
                    .load_low_bit_tensor_into(
                        &replacement.tensor_name,
                        values_destination,
                        scales_destination,
                    )
                    .map_err(|_| {
                        RvllmError::apple(
                            AppleError::InvalidWeightBlob {
                                reason: "low-bit sidecar failed bounded authenticated reload",
                            },
                            model_ctx("prepare_low_bit_weights"),
                        )
                    })?;
            }
            let projection = MetalLowBitProjectionOffsets::new_for_role(
                replacement.role,
                replacement.format,
                replacement.shape[0],
                replacement.shape[1],
                values.offset,
                values.size,
                scales.offset,
                scales.size,
            )
            .map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "low-bit arena descriptor validation failed",
                    },
                    model_ctx("prepare_low_bit_weights"),
                )
            })?;
            let layer = &mut state.layers[layer_index];
            match replacement.role {
                rvllm_apple::AppleLowBitTensorRole::QueryProjection => {
                    layer.low_bit_q_proj = Some(projection)
                }
                rvllm_apple::AppleLowBitTensorRole::KeyProjection => {
                    layer.low_bit_k_proj = Some(projection)
                }
                rvllm_apple::AppleLowBitTensorRole::ValueProjection => {
                    layer.low_bit_v_proj = Some(projection)
                }
                rvllm_apple::AppleLowBitTensorRole::OutputProjection => {
                    layer.low_bit_o_proj = Some(projection)
                }
                rvllm_apple::AppleLowBitTensorRole::DenseGateProjection => {
                    layer.low_bit_gate_proj = Some(projection)
                }
                rvllm_apple::AppleLowBitTensorRole::DenseUpProjection => {
                    layer.low_bit_up_proj = Some(projection)
                }
                rvllm_apple::AppleLowBitTensorRole::DenseDownProjection => {
                    layer.low_bit_down_proj = Some(projection)
                }
                rvllm_apple::AppleLowBitTensorRole::LmHead => {
                    unreachable!("LM head rejected above")
                }
            }
        }
        for replacement in replacements {
            let layer = state
                .layers
                .iter()
                .find(|layer| match replacement.role {
                    rvllm_apple::AppleLowBitTensorRole::QueryProjection => {
                        layer.q_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::KeyProjection => {
                        layer.k_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::ValueProjection => {
                        layer.v_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::OutputProjection => {
                        layer.o_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::DenseGateProjection => {
                        layer.gate_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::DenseUpProjection => {
                        layer.up_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::DenseDownProjection => {
                        layer.down_proj_name == replacement.tensor_name
                    }
                    rvllm_apple::AppleLowBitTensorRole::LmHead => false,
                })
                .ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "low-bit replacement disappeared during preparation",
                        },
                        model_ctx("prepare_low_bit_weights"),
                    )
                })?;
            let installed = match replacement.role {
                rvllm_apple::AppleLowBitTensorRole::QueryProjection => {
                    layer.low_bit_q_proj.is_some()
                }
                rvllm_apple::AppleLowBitTensorRole::KeyProjection => layer.low_bit_k_proj.is_some(),
                rvllm_apple::AppleLowBitTensorRole::ValueProjection => {
                    layer.low_bit_v_proj.is_some()
                }
                rvllm_apple::AppleLowBitTensorRole::OutputProjection => {
                    layer.low_bit_o_proj.is_some()
                }
                rvllm_apple::AppleLowBitTensorRole::DenseGateProjection => {
                    layer.low_bit_gate_proj.is_some()
                }
                rvllm_apple::AppleLowBitTensorRole::DenseUpProjection => {
                    layer.low_bit_up_proj.is_some()
                }
                rvllm_apple::AppleLowBitTensorRole::DenseDownProjection => {
                    layer.low_bit_down_proj.is_some()
                }
                rvllm_apple::AppleLowBitTensorRole::LmHead => false,
            };
            if !installed
                || (replacement.role == rvllm_apple::AppleLowBitTensorRole::DenseDownProjection
                    && self.low_bit_residency_policy == MetalLowBitResidencyPolicy::ReplaceNative
                    && layer.down_proj.is_some())
            {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "low-bit replacement installation is incomplete",
                    },
                    model_ctx("prepare_low_bit_weights"),
                ));
            }
        }
        Ok(())
    }

    fn explicit_metallib_path(&self, float_type: MetalFloatType) -> Option<&std::path::Path> {
        match float_type {
            MetalFloatType::F16 => self.metallib_f16.as_deref(),
            MetalFloatType::Bf16 => self.metallib_bf16.as_deref(),
        }
    }

    #[cfg(target_os = "macos")]
    fn development_metallib_path(float_type: MetalFloatType) -> Option<PathBuf> {
        let env_name = match float_type {
            MetalFloatType::F16 => RVLLM_METAL_METALLIB_F16_ENV,
            MetalFloatType::Bf16 => RVLLM_METAL_METALLIB_BF16_ENV,
        };
        std::env::var_os(env_name).map(PathBuf::from)
    }

    fn enqueue_embedding_gather(&self, state: &Gemma4MetalState, num_tokens: usize) -> Result<()> {
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("enqueue_embedding_gather"),
            )
        })?;
        let queue = ctx.queue_retained();
        let cmd_buf = queue.commandBuffer().ok_or_else(|| {
            RvllmError::apple(
                AppleError::MetalUnavailable,
                model_ctx("embedding_gather_command_buffer"),
            )
        })?;
        self.encode_embedding_gather(&cmd_buf, state, num_tokens)?;
        cmd_buf.commit();
        self.perf.add_command_buffers(1);
        self.perf.add_embedding_encoders(1);
        if self.debug_sync {
            let wait_start = Instant::now();
            cmd_buf.waitUntilCompleted();
            self.perf.add_forced_wait_duration(wait_start.elapsed());
        }
        Ok(())
    }

    fn encode_embedding_gather(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        state: &Gemma4MetalState,
        num_tokens: usize,
    ) -> Result<()> {
        self.encode_embedding_gather_from(
            cmd_buf,
            state,
            num_tokens,
            state.token_ids.offset,
        )
    }

    fn encode_embedding_gather_from(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        state: &Gemma4MetalState,
        num_tokens: usize,
        token_ids_offset: usize,
    ) -> Result<()> {
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("embedding_gather"),
            )
        })?;
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("embedding_gather"),
            )
        })?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            RvllmError::apple(
                AppleError::MetalUnavailable,
                model_ctx("embedding_gather_encoder"),
            )
        })?;

        let pso = pipelines.get("embedding_gather_f16")?;
        let buf = arena.buffer_retained();
        let num_tokens_u32 = u32::try_from(num_tokens).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "token count exceeds u32",
                },
                model_ctx("embedding_gather"),
            )
        })?;
        let hidden = u32::try_from(state.hidden_size).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "hidden_size does not fit in u32",
                },
                model_ctx("embedding_gather"),
            )
        })?;
        let vocab = u32::try_from(state.vocab_size).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "vocab_size does not fit in u32",
                },
                model_ctx("embedding_gather"),
            )
        })?;
        let scale = state.embedding_scale;
        unsafe {
            encoder.setComputePipelineState(pso);
            encoder.setBuffer_offset_atIndex(Some(buf), state.embedding.offset, 0);
            encoder.setBuffer_offset_atIndex(Some(buf), token_ids_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(buf), state.residual.offset, 2);
            encoder.setBytes_length_atIndex(
                ptr::NonNull::new_unchecked(&num_tokens_u32 as *const _ as *mut _),
                4,
                3,
            );
            encoder.setBytes_length_atIndex(
                ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
                4,
                4,
            );
            encoder.setBytes_length_atIndex(
                ptr::NonNull::new_unchecked(&vocab as *const _ as *mut _),
                4,
                5,
            );
            encoder.setBytes_length_atIndex(
                ptr::NonNull::new_unchecked(&scale as *const _ as *mut _),
                4,
                6,
            );
        }
        let hidden_usize = hidden as usize;
        let threads_per_group = MTLSize {
            width: 1,
            height: max(1, hidden_usize.min(256)),
            depth: 1,
        };
        let groups = MTLSize {
            width: num_tokens,
            height: (hidden_usize + threads_per_group.height - 1) / threads_per_group.height,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();
        Ok(())
    }

    fn enqueue_ple_inputs(&self, state: &Gemma4MetalState, num_tokens: usize) -> Result<()> {
        let Some(ple) = &state.ple else {
            return Ok(());
        };
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("enqueue_ple_inputs"),
            )
        })?;
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("enqueue_ple_inputs"),
            )
        })?;
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("enqueue_ple_inputs"),
            )
        })?;
        let params = MetalPlePrepare {
            embedding_offset: ple.embed_tokens_per_layer.offset,
            token_ids_offset: state.token_ids.offset,
            residual_offset: state.residual.offset,
            per_layer_model_projection_offset: ple.per_layer_model_projection.offset,
            per_layer_projection_norm_offset: ple.per_layer_projection_norm.offset,
            token_inputs_offset: ple.token_inputs.offset,
            context_inputs_offset: ple.context_inputs.offset,
            num_tokens: u32::try_from(num_tokens).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "token count exceeds u32",
                    },
                    model_ctx("enqueue_ple_inputs"),
                )
            })?,
            hidden: u32::try_from(state.hidden_size).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "hidden_size exceeds u32",
                    },
                    model_ctx("enqueue_ple_inputs"),
                )
            })?,
            vocab: u32::try_from(ple.ple_vocab_size).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "PLE vocab size exceeds u32",
                    },
                    model_ctx("enqueue_ple_inputs"),
                )
            })?,
            num_layers: u32::try_from(state.num_layers).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "layer count exceeds u32",
                    },
                    model_ctx("enqueue_ple_inputs"),
                )
            })?,
            ple_dim: u32::try_from(ple.ple_dim).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "PLE dim exceeds u32",
                    },
                    model_ctx("enqueue_ple_inputs"),
                )
            })?,
            rms_eps: state.rms_norm_eps,
        };
        unsafe {
            metal_prepare_ple_inputs(ctx, pipelines, arena, &params)?;
        }
        self.perf.add_command_buffers(1);
        self.perf.add_ple_encoders(4);
        if self.debug_sync {
            self.wait_for_metal_queue("enqueue_ple_inputs")?;
        }
        Ok(())
    }

    fn encode_ple_inputs(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        state: &Gemma4MetalState,
        num_tokens: usize,
    ) -> Result<u64> {
        self.encode_ple_inputs_from(
            cmd_buf,
            state,
            num_tokens,
            state.token_ids.offset,
        )
    }

    fn encode_ple_inputs_from(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        state: &Gemma4MetalState,
        num_tokens: usize,
        token_ids_offset: usize,
    ) -> Result<u64> {
        let Some(ple) = &state.ple else {
            return Ok(0);
        };
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("encode_ple_inputs"),
            )
        })?;
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("encode_ple_inputs"),
            )
        })?;
        let params = MetalPlePrepare {
            embedding_offset: ple.embed_tokens_per_layer.offset,
            token_ids_offset,
            residual_offset: state.residual.offset,
            per_layer_model_projection_offset: ple.per_layer_model_projection.offset,
            per_layer_projection_norm_offset: ple.per_layer_projection_norm.offset,
            token_inputs_offset: ple.token_inputs.offset,
            context_inputs_offset: ple.context_inputs.offset,
            num_tokens: u32::try_from(num_tokens).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "token count exceeds u32",
                    },
                    model_ctx("encode_ple_inputs"),
                )
            })?,
            hidden: u32::try_from(state.hidden_size).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "hidden_size exceeds u32",
                    },
                    model_ctx("encode_ple_inputs"),
                )
            })?,
            vocab: u32::try_from(ple.ple_vocab_size).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "PLE vocab size exceeds u32",
                    },
                    model_ctx("encode_ple_inputs"),
                )
            })?,
            num_layers: u32::try_from(state.num_layers).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "layer count exceeds u32",
                    },
                    model_ctx("encode_ple_inputs"),
                )
            })?,
            ple_dim: u32::try_from(ple.ple_dim).map_err(|_| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "PLE dim exceeds u32",
                    },
                    model_ctx("encode_ple_inputs"),
                )
            })?,
            rms_eps: state.rms_norm_eps,
        };
        unsafe {
            metal_encode_prepare_ple_inputs(cmd_buf, pipelines, arena, &params)?;
        }
        Ok(4)
    }

    #[cfg(test)]
    fn encode_device_resident_decode_advance_single(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        state: &Gemma4MetalState,
        current_position: u32,
        current_slot: i32,
    ) -> Result<()> {
        const MAX_RESEARCH_LAYERS: usize = 64;
        if state.layers.is_empty()
            || state.layers.len() > MAX_RESEARCH_LAYERS
            || current_slot < 0
        {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "device_resident_decode_slice_shape",
                },
                model_ctx("device_resident_decode_slice"),
            ));
        }
        let block_size = state.layers[0].block_size;
        let next_position = current_position.checked_add(1).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "device-resident decode position overflow",
                },
                model_ctx("device_resident_decode_slice"),
            )
        })?;
        if block_size == 0
            || next_position as usize >= state.max_probe_tokens
            || current_position % block_size == block_size - 1
            || (current_slot as u32) % block_size != current_position % block_size
        {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "device_resident_decode_slice_page_crossing",
                },
                model_ctx("device_resident_decode_slice"),
            ));
        }

        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("device_resident_decode_slice"),
            )
        })?;
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("device_resident_decode_slice"),
            )
        })?;
        let pso = pipelines.get("research_decode_advance_single")?;
        let mut offsets = [0u32; MAX_RESEARCH_LAYERS * 3];
        for (layer_idx, layer) in state.layers.iter().enumerate() {
            if layer.block_size != block_size {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "device-resident decode layer block-size mismatch",
                    },
                    model_ctx("device_resident_decode_slice"),
                ));
            }
            for (slot, offset) in [
                layer.positions.offset,
                layer.slot_mapping.offset,
                layer.context_lens.offset,
            ]
            .into_iter()
            .enumerate()
            {
                offsets[layer_idx * 3 + slot] = u32::try_from(offset).map_err(|_| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "device-resident metadata offset exceeds u32",
                        },
                        model_ctx("device_resident_decode_slice"),
                    )
                })?;
            }
        }
        let num_layers = u32::try_from(state.layers.len()).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "device-resident layer count exceeds u32",
                },
                model_ctx("device_resident_decode_slice"),
            )
        })?;
        let byte_len = state.layers.len() * 3 * std::mem::size_of::<u32>();
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            RvllmError::apple(
                AppleError::MetalUnavailable,
                model_ctx("device_resident_decode_slice"),
            )
        })?;
        unsafe {
            encoder.setComputePipelineState(pso);
            encoder.setBuffer_offset_atIndex(Some(arena.buffer()), 0, 0);
            encoder.setBytes_length_atIndex(
                ptr::NonNull::new_unchecked(offsets.as_ptr() as *mut _),
                byte_len,
                1,
            );
            encoder.setBytes_length_atIndex(
                ptr::NonNull::new_unchecked(&num_layers as *const _ as *mut _),
                std::mem::size_of_val(&num_layers),
                2,
            );
        }
        let threads = max(
            1,
            state
                .layers
                .len()
                .min(pso.maxTotalThreadsPerThreadgroup()),
        );
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: state.layers.len(),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }

    fn write_i32_metadata_region(
        arena: &MetalBufferArena,
        region: &MetalRegion,
        values: &[i32],
        op: &'static str,
    ) -> Result<()> {
        let byte_len = values
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "metadata byte length overflow",
                    },
                    model_ctx(op),
                )
            })?;
        if byte_len > region.size {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "metadata region too small",
                },
                model_ctx(op),
            ));
        }

        unsafe {
            let dst = arena.host_ptr(region) as *mut i32;
            ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
        }
        Ok(())
    }

    fn write_prefill_layer_metadata(
        &self,
        state: &Gemma4MetalState,
        handoff: &HandoffCapsule,
    ) -> Result<()> {
        let num_tokens = handoff.tokens_flat.len();
        if num_tokens == 0 || num_tokens > state.max_batch_tokens {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_prefill_length",
                },
                model_ctx("launch_prefill"),
            ));
        }
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_prefill"),
            )
        })?;

        let num_seqs = handoff.num_sequences();
        if num_seqs == 0 || num_seqs > state.max_batch_sequences {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_prefill_batch_size",
                },
                model_ctx("launch_prefill"),
            ));
        }
        if handoff.cu_seqlens.len() != num_seqs + 1 {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "prefill cu_seqlens length must equal req_ids + 1",
                },
                model_ctx("launch_prefill"),
            ));
        }

        let mut positions = Vec::with_capacity(num_tokens);
        let mut slot_mapping = Vec::with_capacity(num_tokens);
        let mut max_seqlen_q = 0usize;
        for seq in 0..num_seqs {
            let seq_start = handoff.cu_seqlens[seq] as usize;
            let seq_end = handoff.cu_seqlens[seq + 1] as usize;
            if seq_end < seq_start || seq_end > num_tokens {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "prefill cu_seqlens must be monotonic and within token count",
                    },
                    model_ctx("launch_prefill"),
                ));
            }
            let seq_len = seq_end - seq_start;
            if seq_len == 0 {
                return Err(RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "model-metal-backend",
                        op: "unsupported_empty_prefill_sequence",
                    },
                    model_ctx("launch_prefill"),
                ));
            }
            max_seqlen_q = max_seqlen_q.max(seq_len);
            let last_position = handoff.positions[seq] as usize;
            let Some(first_position) = (last_position + 1).checked_sub(seq_len) else {
                return Err(RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "model-metal-backend",
                        op: "unsupported_prefill_position_span",
                    },
                    model_ctx("launch_prefill"),
                ));
            };
            if last_position >= state.max_probe_tokens {
                return Err(RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "model-metal-backend",
                        op: "unsupported_context_length",
                    },
                    model_ctx("launch_prefill"),
                ));
            }
            for token_offset in 0..seq_len {
                let position = first_position + token_offset;
                if position >= state.max_probe_tokens {
                    return Err(RvllmError::apple(
                        AppleError::FeatureNotAvailable {
                            backend: "model-metal-backend",
                            op: "unsupported_context_length",
                        },
                        model_ctx("launch_prefill"),
                    ));
                }
                positions.push(position as i32);
                let block_size = state.layers[0].block_size as usize;
                let max_blocks = state.layers[0].max_blocks_per_seq as usize;
                let block = seq * max_blocks + position / block_size;
                slot_mapping.push((block * block_size + position % block_size) as i32);
            }
        }
        if !handoff.slot_mapping.is_empty() {
            slot_mapping = Self::validated_slot_mapping(state, &handoff.slot_mapping)?;
        }
        let context_lens = handoff
            .context_lens
            .iter()
            .map(|&context_len| context_len as i32)
            .collect::<Vec<_>>();
        let block_tables = Self::materialize_block_tables(state, handoff)?;
        let cu_seqlens = handoff
            .cu_seqlens
            .iter()
            .map(|&cu| cu as i32)
            .collect::<Vec<_>>();

        for layer in &state.layers {
            Self::write_i32_metadata_region(arena, &layer.positions, &positions, "launch_prefill")?;
            Self::write_i32_metadata_region(
                arena,
                &layer.slot_mapping,
                &slot_mapping,
                "launch_prefill",
            )?;
            Self::write_i32_metadata_region(
                arena,
                &layer.context_lens,
                &context_lens,
                "launch_prefill",
            )?;
            Self::write_i32_metadata_region(
                arena,
                &layer.block_tables,
                &block_tables,
                "launch_prefill",
            )?;
            Self::write_i32_metadata_region(
                arena,
                &layer.cu_seqlens,
                &cu_seqlens,
                "launch_prefill",
            )?;
        }
        if max_seqlen_q == 0 {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "prefill max sequence length must be positive",
                },
                model_ctx("launch_prefill"),
            ));
        }
        Ok(())
    }

    fn write_decode_layer_metadata(
        &self,
        state: &Gemma4MetalState,
        handoff: &HandoffCapsule,
    ) -> Result<()> {
        let num_seqs = handoff.num_sequences();
        if num_seqs == 0 || num_seqs > state.max_batch_sequences {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_decode_batch_size",
                },
                model_ctx("launch_rollout"),
            ));
        }
        if handoff
            .positions
            .iter()
            .zip(&handoff.context_lens)
            .any(|(&position, &context_len)| {
                position as usize >= state.max_probe_tokens
                    || context_len == 0
                    || context_len as usize > state.max_probe_tokens
            })
        {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_context_length",
                },
                model_ctx("launch_rollout"),
            ));
        }
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_rollout"),
            )
        })?;

        let positions = handoff
            .positions
            .iter()
            .map(|&position| position as i32)
            .collect::<Vec<_>>();
        let context_lens = handoff
            .context_lens
            .iter()
            .map(|&context_len| context_len as i32)
            .collect::<Vec<_>>();
        let slot_mapping = if handoff.slot_mapping.is_empty() {
            let block_size = state.layers[0].block_size as usize;
            let max_blocks = state.layers[0].max_blocks_per_seq as usize;
            handoff
                .positions
                .iter()
                .enumerate()
                .map(|(seq, &position)| {
                    let position = position as usize;
                    let block = seq * max_blocks + position / block_size;
                    (block * block_size + position % block_size) as i32
                })
                .collect::<Vec<_>>()
        } else {
            Self::validated_slot_mapping(state, &handoff.slot_mapping)?
        };
        let block_tables = Self::materialize_block_tables(state, handoff)?;
        let cu_seqlens = (0..=num_seqs).map(|idx| idx as i32).collect::<Vec<_>>();

        for layer in &state.layers {
            Self::write_i32_metadata_region(arena, &layer.positions, &positions, "launch_rollout")?;
            Self::write_i32_metadata_region(
                arena,
                &layer.slot_mapping,
                &slot_mapping,
                "launch_rollout",
            )?;
            Self::write_i32_metadata_region(
                arena,
                &layer.context_lens,
                &context_lens,
                "launch_rollout",
            )?;
            Self::write_i32_metadata_region(
                arena,
                &layer.block_tables,
                &block_tables,
                "launch_rollout",
            )?;
            Self::write_i32_metadata_region(
                arena,
                &layer.cu_seqlens,
                &cu_seqlens,
                "launch_rollout",
            )?;
        }
        Ok(())
    }

    fn validated_slot_mapping(state: &Gemma4MetalState, slots: &[i32]) -> Result<Vec<i32>> {
        let Some(layer) = state.layers.first() else {
            return Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("paged_kv_metadata"),
            ));
        };
        let capacity = u64::from(layer.num_blocks_total) * u64::from(layer.block_size);
        if slots
            .iter()
            .any(|&slot| slot < 0 || slot as u64 >= capacity)
        {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "paged KV slot mapping exceeds the physical page pool",
                },
                model_ctx("paged_kv_metadata"),
            ));
        }
        Ok(slots.to_vec())
    }

    fn materialize_block_tables(
        state: &Gemma4MetalState,
        handoff: &HandoffCapsule,
    ) -> Result<Vec<i32>> {
        let Some(layer) = state.layers.first() else {
            return Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("paged_kv_metadata"),
            ));
        };
        if state.layers.iter().any(|candidate| {
            candidate.block_size != layer.block_size
                || candidate.max_blocks_per_seq != layer.max_blocks_per_seq
                || candidate.num_blocks_total != layer.num_blocks_total
        }) {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "paged KV layout differs between model layers",
                },
                model_ctx("paged_kv_metadata"),
            ));
        }
        let num_seqs = handoff.num_sequences();
        let target_width = layer.max_blocks_per_seq as usize;
        let block_size = layer.block_size as usize;
        let mut output = vec![-1_i32; num_seqs * target_width];

        if handoff.block_tables.is_empty() {
            for (seq, &context_len) in handoff.context_lens.iter().enumerate() {
                let required = (context_len as usize).div_ceil(block_size);
                if required > target_width {
                    return Err(RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "paged KV context exceeds block-table width",
                        },
                        model_ctx("paged_kv_metadata"),
                    ));
                }
                for block_index in 0..required {
                    output[seq * target_width + block_index] =
                        (seq * target_width + block_index) as i32;
                }
            }
            return Ok(output);
        }

        let source_width = handoff.max_blocks_per_seq as usize;
        if source_width == 0 || source_width > target_width {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "paged KV handoff block-table width is incompatible",
                },
                model_ctx("paged_kv_metadata"),
            ));
        }
        for (seq, &context_len) in handoff.context_lens.iter().enumerate() {
            let required = (context_len as usize).div_ceil(block_size);
            if required > source_width {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "paged KV handoff omits required context pages",
                    },
                    model_ctx("paged_kv_metadata"),
                ));
            }
            for block_index in 0..source_width {
                let raw = handoff.block_tables[seq * source_width + block_index];
                if raw == u32::MAX {
                    if block_index < required {
                        return Err(RvllmError::apple(
                            AppleError::InvalidWeightBlob {
                                reason: "paged KV handoff has a hole in the live prefix",
                            },
                            model_ctx("paged_kv_metadata"),
                        ));
                    }
                    continue;
                }
                if raw >= layer.num_blocks_total {
                    return Err(RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "paged KV block id exceeds the physical page pool",
                        },
                        model_ctx("paged_kv_metadata"),
                    ));
                }
                output[seq * target_width + block_index] = raw as i32;
            }
        }
        Ok(output)
    }

    fn enqueue_probe_layers(
        &self,
        external_cmd_buf: Option<&ProtocolObject<dyn MTLCommandBuffer>>,
        #[cfg(feature = "metal-stage-instrumentation")] mut stage_profiler: Option<
            &mut MetalStageProfiler,
        >,
        state: &Gemma4MetalState,
        num_tokens: usize,
        phase: MetalPhase,
        op: &'static str,
    ) -> Result<()> {
        if num_tokens == 0 || num_tokens > state.max_batch_tokens {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_probe_token_count",
                },
                model_ctx(op),
            ));
        }
        if state.num_layers != state.layers.len() {
            return Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx(op),
            ));
        }

        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx(op),
            )
        })?;
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx(op),
            )
        })?;
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx(op),
            )
        })?;

        let num_tokens_u32 = u32::try_from(num_tokens).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "token count exceeds u32",
                },
                model_ctx(op),
            )
        })?;
        let half_bytes = std::mem::size_of::<f16>();
        #[cfg(test)]
        let stop_after_layer = self
            .explicit_options
            .is_none()
            .then(metal_debug_stop_after_layer)
            .flatten();
        #[cfg(test)]
        let trace_layers = if self.explicit_options.is_none() {
            metal_debug_trace_layers()
        } else {
            Vec::new()
        };
        #[cfg(test)]
        let trace_json_path = self
            .explicit_options
            .is_none()
            .then(metal_debug_trace_json_path)
            .flatten();
        #[cfg(test)]
        let shared_kv_skip_mode = if self.explicit_options.is_none() {
            metal_debug_shared_kv_skip_mode()
        } else {
            MetalDebugSharedKvSkipMode::None
        };
        #[cfg(not(test))]
        let shared_kv_skip_mode = MetalDebugSharedKvSkipMode::None;
        #[cfg(test)]
        let debug_layer_checks = (self.explicit_options.is_none()
            && metal_debug_finite_layers_enabled())
            || stop_after_layer.is_some()
            || !trace_layers.is_empty();
        #[cfg(not(test))]
        let debug_layer_checks = false;

        let owned_cmd_buf =
            if debug_layer_checks || external_cmd_buf.is_some() {
                None
            } else {
                let queue = ctx.queue_retained();
                Some(queue.commandBuffer().ok_or_else(|| {
                    RvllmError::apple(AppleError::MetalUnavailable, model_ctx(op))
                })?)
            };
        let batched_cmd_buf = external_cmd_buf.or_else(|| owned_cmd_buf.as_deref());

        for one in &state.layers {
            if one.layer_idx >= state.num_layers {
                return Err(RvllmError::apple(
                    AppleError::NotPrepared {
                        backend: "model-metal-backend",
                    },
                    model_ctx(op),
                ));
            }

            let hidden = state.hidden_size;
            // Replacement residency deliberately omits the native fused
            // gate/up allocation. Preserve the authenticated logical shape
            // from the low-bit down descriptor instead of inferring zero from
            // that absent storage.
            let intermediate = one.low_bit_down_proj.map_or_else(
                || one.gate_up.size / 2 / half_bytes / hidden,
                |projection| projection.shape()[1] as usize,
            );
            let down_proj_offset = match (one.down_proj.as_ref(), one.low_bit_down_proj.is_some()) {
                (Some(native), false) => Some(native.offset),
                (_, true) => None,
                (None, false) => {
                    return Err(RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "prepared layer has no down projection execution source",
                        },
                        model_ctx(op),
                    ));
                }
            };

            let dims = MetalLayerDims {
                layer_idx: one.layer_idx as u32,
                attention_window: one.dims.attention_window,
                num_tokens: num_tokens_u32,
                hidden: state.hidden_size as u32,
                num_layers: state.num_layers as u32,
                num_heads: one.dims.num_heads as u32,
                num_kv_heads: one.dims.num_kv_heads as u32,
                head_dim: one.dims.head_dim as u32,
                intermediate: intermediate as u32,
                moe_num_experts: one.moe.as_ref().map_or(0, |moe| moe.num_experts as u32),
                moe_top_k: one.moe.as_ref().map_or(0, |moe| moe.top_k as u32),
                moe_intermediate: one
                    .moe
                    .as_ref()
                    .map_or(0, |moe| moe.intermediate_size as u32),
                ple_dim: state.ple.as_ref().map_or(0, |ple| ple.ple_dim as u32),
                block_size: one.block_size,
                max_blocks_per_seq: one.max_blocks_per_seq,
                num_blocks_total: one.num_blocks_total,
                attn_scale: one.dims.attn_scale,
                rms_eps: state.rms_norm_eps,
                rope_dim: one.dims.rope_dim as u32,
                softcap: state.final_logit_softcap,
            };

            let weights = MetalLayerWeights {
                attn_norm_offset: one.attn_norm.offset,
                qkv_offset: one.qkv.offset,
                qkv_bias_offset: None,
                q_norm_offset: one.q_norm.as_ref().map(|region| region.offset),
                k_norm_offset: one.k_norm.as_ref().map(|region| region.offset),
                v_norm_offset: one.v_norm.as_ref().map(|region| region.offset),
                o_proj_offset: one.o_proj.offset,
                mlp_norm_offset: one.mlp_norm.offset,
                post_attn_norm_offset: one.post_attn_norm.as_ref().map(|region| region.offset),
                pre_ff_norm_offset: one.pre_ff_norm.as_ref().map(|region| region.offset),
                post_ff_norm_offset: one.post_ff_norm.as_ref().map(|region| region.offset),
                layer_scalar_offset: one.layer_scalar.as_ref().map(|region| region.offset),
                layer_scalar_dim: one.layer_scalar_dim,
                gate_up_offset: one.gate_up.offset,
                down_proj_offset,
                low_bit_q_proj: one.low_bit_q_proj,
                low_bit_k_proj: one.low_bit_k_proj,
                low_bit_v_proj: one.low_bit_v_proj,
                low_bit_o_proj: one.low_bit_o_proj,
                low_bit_gate_proj: one.low_bit_gate_proj,
                low_bit_up_proj: one.low_bit_up_proj,
                low_bit_down_proj: one.low_bit_down_proj,
                moe: one.moe.as_ref().map(|moe| MetalMoeWeights {
                    router_proj_offset: moe.router_proj.offset,
                    router_scale_offset: moe.router_scale.offset,
                    router_per_expert_scale_offset: moe.router_per_expert_scale.offset,
                    pre_ff2_norm_offset: moe.pre_ff2_norm.offset,
                    post_ff1_norm_offset: moe.post_ff1_norm.offset,
                    post_ff2_norm_offset: moe.post_ff2_norm.offset,
                    expert_gate_up_offset: moe.expert_gate_up.offset,
                    expert_down_offset: moe.expert_down.offset,
                }),
                per_layer_inputs_offset: state.ple.as_ref().map(|ple| ple.token_inputs.offset),
                per_layer_input_gate_offset: one
                    .per_layer_input_gate
                    .as_ref()
                    .map(|region| region.offset),
                per_layer_projection_offset: one
                    .per_layer_projection
                    .as_ref()
                    .map(|region| region.offset),
                post_per_layer_input_norm_offset: one
                    .post_per_layer_input_norm
                    .as_ref()
                    .map(|region| region.offset),
            };

            let scratch = MetalScratch {
                normed_hidden: state.normed_hidden.offset,
                qkv_out: one.qkv_out.offset,
                q_offset: one.q.offset,
                k_offset: one.k.offset,
                v_offset: one.v.offset,
                attn_out: one.attn_out.offset,
                global_decode_partials: one
                    .global_decode_partials
                    .as_ref()
                    .map(|region| region.offset),
                gate_up_out: one.gate_up_out.offset,
                activated: one.activated.offset,
                mlp_out: one.mlp_out.offset,
                moe_topk_indices: one.moe_topk_indices.as_ref().map(|region| region.offset),
                moe_topk_weights: one.moe_topk_weights.as_ref().map(|region| region.offset),
                moe_activated: one.moe_activated.as_ref().map(|region| region.offset),
                moe_out: one.moe_out.as_ref().map(|region| region.offset),
            };

            let meta = MetalMetadata {
                positions_offset: one.positions.offset,
                slot_mapping_offset: one.slot_mapping.offset,
                cos_offset: one.cos.offset,
                sin_offset: one.sin.offset,
                block_tables_offset: one.block_tables.offset,
                context_lens_offset: one.context_lens.offset,
                cu_seqlens_offset: Some(one.cu_seqlens.offset),
            };

            #[cfg(test)]
            let layer_trace_state = if trace_layers.contains(&one.layer_idx) {
                one.trace.as_ref()
            } else {
                None
            };
            #[cfg(not(test))]
            let layer_trace_state: Option<&MetalLayerTraceState> = None;
            let layer_trace_scratch = layer_trace_state.map(|trace| MetalLayerTraceScratch {
                input_to_layer: trace.input_to_layer.offset,
                after_input_layernorm: trace.after_input_layernorm.offset,
                q_projection: trace.q_projection.offset,
                k_projection: trace.k_projection.offset,
                v_projection: trace.v_projection.offset,
                after_q_norm: trace.after_q_norm.offset,
                after_k_norm: trace.after_k_norm.offset,
                after_v_norm: trace.after_v_norm.offset,
                after_rope_q: trace.after_rope_q.offset,
                after_rope_k: trace.after_rope_k.offset,
                attention_output: trace.attention_output.offset,
                after_o_proj: trace.after_o_proj.offset,
                after_post_attention_layernorm: trace.after_post_attention_layernorm.offset,
                after_pre_feedforward_layernorm: trace.after_pre_feedforward_layernorm.offset,
                gate_up_out: trace.gate_up_out.offset,
                ffn_activation: trace.ffn_activation.offset,
                after_ffn_branch: trace.after_ffn_branch.offset,
                after_post_feedforward_layernorm: trace.after_post_feedforward_layernorm.offset,
                per_layer_input: trace.per_layer_input.as_ref().map(|region| region.offset),
                per_layer_input_gate: trace
                    .per_layer_input_gate
                    .as_ref()
                    .map(|region| region.offset),
                per_layer_projection: trace
                    .per_layer_projection
                    .as_ref()
                    .map(|region| region.offset),
                post_per_layer_input_norm: trace
                    .post_per_layer_input_norm
                    .as_ref()
                    .map(|region| region.offset),
            });
            let attention_kv_layer = one
                .shared_kv_source_layer
                .and_then(|source_idx| state.layers.get(source_idx));
            let attention_kv_cache_k_offset = attention_kv_layer
                .map(|layer| layer.kv_cache_k.offset)
                .unwrap_or(one.kv_cache_k.offset);
            let attention_kv_cache_v_offset = attention_kv_layer
                .map(|layer| layer.kv_cache_v.offset)
                .unwrap_or(one.kv_cache_v.offset);
            let shared_kv_debug_skip =
                match (one.shared_kv_source_layer.is_some(), shared_kv_skip_mode) {
                    (true, MetalDebugSharedKvSkipMode::SkipLocalKvCacheWriteOnly) => {
                        MetalLayerDebugSkip {
                            skip_kv_projection: false,
                            skip_local_kv_cache_write: true,
                        }
                    }
                    (true, MetalDebugSharedKvSkipMode::SkipTailKvProjectionAndCache) => {
                        MetalLayerDebugSkip {
                            skip_kv_projection: true,
                            skip_local_kv_cache_write: true,
                        }
                    }
                    _ => MetalLayerDebugSkip::default(),
                };

            unsafe {
                if let Some(cmd_buf) = batched_cmd_buf.as_ref() {
                    metal_encode_forward_layer(
                        cmd_buf,
                        pipelines,
                        arena,
                        &dims,
                        &weights,
                        &scratch,
                        layer_trace_scratch.as_ref(),
                        &meta,
                        state.residual.offset,
                        phase,
                        one.kv_cache_k.offset,
                        one.kv_cache_v.offset,
                        attention_kv_cache_k_offset,
                        attention_kv_cache_v_offset,
                        shared_kv_debug_skip,
                        #[cfg(feature = "metal-stage-instrumentation")]
                        stage_profiler.as_deref_mut(),
                    )?;
                } else {
                    metal_forward_layer(
                        ctx,
                        pipelines,
                        arena,
                        &dims,
                        &weights,
                        &scratch,
                        layer_trace_scratch.as_ref(),
                        &meta,
                        state.residual.offset,
                        phase,
                        one.kv_cache_k.offset,
                        one.kv_cache_v.offset,
                        attention_kv_cache_k_offset,
                        attention_kv_cache_v_offset,
                        shared_kv_debug_skip,
                    )?;
                    self.perf.add_command_buffers(1);
                }
            }
            self.perf
                .add_layer_encoders(Self::estimate_layer_encoder_count(
                    &weights,
                    &dims,
                    layer_trace_scratch.is_some(),
                    shared_kv_debug_skip,
                    layer_trace_scratch.is_none()
                        && rvllm_apple_metal::layer_forward::supports_gemma4_prefill_mma(
                            pipelines, &dims, phase,
                        ),
                    matches!(phase, MetalPhase::Prefill { .. })
                        && supports_qkv_prefill_projection(pipelines, &dims),
                    rvllm_apple_metal::layer_forward::supports_research_rounded_gate(
                        pipelines,
                        &dims,
                        phase,
                        &weights,
                        &scratch,
                        layer_trace_scratch.is_some(),
                        arena.capacity(),
                    ),
                ));
            if weights.layer_scalar_offset.is_some() {
                self.perf.add_layer_scale_encoder_fusions(1);
            }

            #[cfg(test)]
            if (self.explicit_options.is_none() && metal_debug_finite_layers_enabled())
                || stop_after_layer.is_some()
                || trace_layers.contains(&one.layer_idx)
            {
                self.wait_for_metal_queue("debug_layer_finite")?;
                let residual_nonfinite = debug_print_f16_region_token_stats(
                    arena,
                    "residual",
                    state.residual.offset,
                    num_tokens,
                    state.hidden_size,
                );
                if residual_nonfinite > 0 {
                    let q_dim = one.dims.q_dim;
                    let kv_dim = one.dims.kv_dim;
                    let two_intermediate = 2usize.saturating_mul(intermediate);
                    eprintln!(
                        "metal debug finite: op={op} layer={} residual_nonfinite={residual_nonfinite}/{}",
                        one.layer_idx,
                        num_tokens.saturating_mul(state.hidden_size)
                    );
                    debug_print_f16_region_token_stats(arena, "q", one.q.offset, num_tokens, q_dim);
                    debug_print_f16_region_token_stats(
                        arena,
                        "k",
                        one.k.offset,
                        num_tokens,
                        kv_dim,
                    );
                    debug_print_f16_region_token_stats(
                        arena,
                        "v",
                        one.v.offset,
                        num_tokens,
                        kv_dim,
                    );
                    debug_print_f16_region_token_stats(
                        arena,
                        "attn_out",
                        one.attn_out.offset,
                        num_tokens,
                        q_dim,
                    );
                    debug_print_f16_region_token_stats(
                        arena,
                        "gate_up_out",
                        one.gate_up_out.offset,
                        num_tokens,
                        two_intermediate,
                    );
                    debug_print_f16_region_token_stats(
                        arena,
                        "activated",
                        one.activated.offset,
                        num_tokens,
                        intermediate,
                    );
                    debug_print_f16_region_token_stats(
                        arena,
                        "mlp_out",
                        one.mlp_out.offset,
                        num_tokens,
                        state.hidden_size,
                    );
                    return Err(RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "nonfinite residual after Metal layer",
                        },
                        model_ctx("debug_layer_finite"),
                    ));
                }
                if trace_layers.contains(&one.layer_idx) {
                    if let Some(path) = trace_json_path.as_deref() {
                        let layer_path = metal_debug_trace_json_path_for_layer(path, one.layer_idx);
                        let q_dim = one.dims.q_dim;
                        let kv_dim = one.dims.kv_dim;
                        debug_write_layer_trace_json(
                            arena,
                            &layer_path,
                            op,
                            phase,
                            one.layer_idx,
                            num_tokens,
                            state.hidden_size,
                            q_dim,
                            kv_dim,
                            intermediate,
                            state.residual.offset,
                            one.q.offset,
                            one.k.offset,
                            one.v.offset,
                            one.attn_out.offset,
                            one.gate_up_out.offset,
                            one.activated.offset,
                            state.ple.as_ref().map_or(0, |ple| ple.ple_dim),
                            state.max_probe_tokens,
                            one.kv_cache_k.offset,
                            one.kv_cache_v.offset,
                            attention_kv_cache_k_offset,
                            attention_kv_cache_v_offset,
                            one.shared_kv_source_layer,
                            layer_trace_state,
                        )?;
                        eprintln!(
                            "metal debug trace: wrote layer {} summary to {}",
                            one.layer_idx,
                            layer_path.display()
                        );
                    }
                }
            }

            #[cfg(test)]
            if stop_after_layer == Some(one.layer_idx) {
                eprintln!(
                    "metal debug finite: op={op} stopped after layer={}",
                    one.layer_idx
                );
                break;
            }
        }

        if let Some(cmd_buf) = owned_cmd_buf {
            cmd_buf.commit();
            self.perf.add_command_buffers(1);
        }

        Ok(())
    }

    fn wait_for_metal_queue(&self, op: &'static str) -> Result<()> {
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx(op),
            )
        })?;
        let queue = ctx.queue_retained();
        let cmd_buf = queue
            .commandBuffer()
            .ok_or_else(|| RvllmError::apple(AppleError::MetalUnavailable, model_ctx(op)))?;
        cmd_buf.commit();
        self.perf.add_command_buffers(1);
        let wait_start = Instant::now();
        cmd_buf.waitUntilCompleted();
        self.perf.add_forced_wait_duration(wait_start.elapsed());
        Ok(())
    }

    fn estimate_layer_encoder_count(
        weights: &MetalLayerWeights,
        dims: &MetalLayerDims,
        trace_enabled: bool,
        debug_skip: MetalLayerDebugSkip,
        allow_prefill_mma: bool,
        qkv_prefill_projection_eligible: bool,
        rounded_gate_encoder_fused: bool,
    ) -> u64 {
        let mut count = 10;
        let low_bit_qkv = weights.low_bit_q_proj.is_some()
            && weights.low_bit_k_proj.is_some()
            && weights.low_bit_v_proj.is_some();
        let low_bit_gate_up =
            weights.low_bit_gate_proj.is_some() && weights.low_bit_up_proj.is_some();
        if low_bit_qkv {
            if debug_skip.skip_kv_projection {
                // Native Q projection+norm is one fused encoder; low-bit Q is
                // a projection followed by standalone normalization.
                count += weights.q_norm_offset.is_some() as u64;
            } else {
                // Three projections replace one native QKV projection, then a
                // split and standalone per-head norms replace fused handling.
                count += 3;
                count += weights.q_norm_offset.is_some() as u64;
                count += weights.k_norm_offset.is_some() as u64;
                // V always receives either learned or unit headwise RMSNorm.
                count += 1;
            }
        } else if !(weights.q_norm_offset.is_some() && weights.k_norm_offset.is_some()) {
            count += weights.q_norm_offset.is_some() as u64;
            count += weights.k_norm_offset.is_some() as u64;
            count += weights.v_norm_offset.is_some() as u64;
        }
        count += weights.post_attn_norm_offset.is_some() as u64;
        count += weights.post_ff_norm_offset.is_some() as u64;
        if weights.per_layer_inputs_offset.is_some()
            && weights.per_layer_input_gate_offset.is_some()
            && weights.per_layer_projection_offset.is_some()
            && weights.post_per_layer_input_norm_offset.is_some()
        {
            count += 4;
        }
        if weights.moe.is_some() {
            // Dense branch post_norm_1 replaces the dense post_ff_norm path;
            // router, expert input norm, expert gate/up, expert down,
            // expert post-norm, and branch-sum final norm are additional.
            count += 6;
        }

        // Each projection+RMSNorm site is one encoder in the fused fallback,
        // already represented above. Decode/microbatch shapes use cooperative
        // GEMV plus a separate RMSNorm encoder, so account for the extra
        // encoder at every active site.
        let projection_norm_bonus = |n: u32, k: u32| {
            metal_gemm_rmsnorm_encoder_count(dims.num_tokens, n, k, allow_prefill_mma)
                .saturating_sub(1)
        };
        if weights.post_attn_norm_offset.is_some() {
            if weights.low_bit_o_proj.is_some() {
                // Low-bit projection and RMSNorm are always separate.
                count += 1;
            } else {
                count += projection_norm_bonus(dims.hidden, dims.num_heads * dims.head_dim);
            }
        }
        if weights.moe.is_some() || weights.post_ff_norm_offset.is_some() {
            count += projection_norm_bonus(dims.hidden, dims.intermediate);
        }
        if weights.per_layer_inputs_offset.is_some()
            && weights.per_layer_input_gate_offset.is_some()
            && weights.per_layer_projection_offset.is_some()
            && weights.post_per_layer_input_norm_offset.is_some()
        {
            count += projection_norm_bonus(dims.hidden, dims.ple_dim);
        }
        if !trace_enabled
            && !low_bit_qkv
            && !debug_skip.skip_kv_projection
            && !debug_skip.skip_local_kv_cache_write
            && weights.q_norm_offset.is_some()
            && weights.k_norm_offset.is_some()
            && supports_qkv_rope_cache_fusion(dims)
        {
            count = count.saturating_sub(2);
            count += u64::from(qkv_prefill_projection_eligible);
        }
        // Separate low-bit gate and up projections replace one native fused
        // gate-up projection encoder.
        count += u64::from(low_bit_gate_up);
        // The candidate replaces two encoders with one. Its predicate includes
        // live PSO limits and buffer bounds, not merely the requested selector.
        count.saturating_sub(u64::from(rounded_gate_encoder_fused && !low_bit_gate_up))
    }

    fn encode_prefill_first_token(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        state: &Gemma4MetalState,
        num_tokens: usize,
    ) -> Result<()> {
        let invalid = || {
            RvllmError::apple(
                AppleError::HandoffMalformed {
                    reason: "prefill final row exceeds its scratch region",
                },
                model_ctx("prefill_first_token"),
            )
        };
        let row_bytes = state.hidden_size.checked_mul(2).ok_or_else(invalid)?;
        let row_offset = num_tokens
            .checked_sub(1)
            .and_then(|row| row.checked_mul(row_bytes))
            .ok_or_else(invalid)?;
        if row_bytes == 0
            || row_offset
                .checked_add(row_bytes)
                .filter(|&end| end <= state.residual.size)
                .is_none()
            || state.normed_hidden.size < row_bytes
            || state.sampled.size < std::mem::size_of::<i32>()
        {
            return Err(invalid());
        }
        let residual_offset = state
            .residual
            .offset
            .checked_add(row_offset)
            .ok_or_else(invalid)?;
        let hidden = u32::try_from(state.hidden_size).map_err(|_| invalid())?;
        let vocab = u32::try_from(state.vocab_size).map_err(|_| invalid())?;
        let arena = self.arena.as_ref().ok_or_else(invalid)?;
        let pipelines = self.pipelines.as_ref().ok_or_else(invalid)?;
        // SAFETY: one complete residual row is checked above. All output and
        // workspace regions belong to this execution slot; the loaded state
        // sized them for at least one token. The preceding layer encoders and
        // this head share one ordered command buffer, with no intervening host
        // access. Collection retains the slot through output readback.
        unsafe {
            metal_encode_finalize_sample(
                command,
                pipelines,
                arena,
                1,
                hidden,
                vocab,
                state.rms_norm_eps,
                state.final_logit_softcap,
                residual_offset,
                state.final_norm.offset,
                state.lm_head.offset,
                state.logits.offset,
                state.normed_hidden.offset,
                state.sampled.offset,
                state.final_argmax_partial_max.offset,
                state.final_argmax_partial_idx.offset,
            )?;
        }
        self.perf
            .add_final_sample_encoders(metal_finalize_sample_encoder_count(1, hidden, vocab));
        Ok(())
    }

    fn run_prefill_step(
        &self,
        handoff: &HandoffCapsule,
        execution_slot: usize,
        sample_first_token: bool,
    ) -> Result<ModelLaunchResult> {
        let perf_before = self.perf.snapshot();
        let wall_start = Instant::now();
        let state = self.execution_states.get(execution_slot).ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_prefill"),
            )
        })?;

        let num_tokens = handoff.tokens_flat.len();
        if num_tokens == 0 || num_tokens > state.max_batch_tokens {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_prefill_length",
                },
                model_ctx("launch_prefill"),
            ));
        }
        for &tok in &handoff.tokens_flat {
            if (tok.raw() as usize) >= state.vocab_size {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "token id exceeds vocabulary size",
                    },
                    model_ctx("launch_prefill"),
                ));
            }
        }

        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_prefill"),
            )
        })?;
        unsafe {
            let dst = arena.host_ptr(&state.token_ids) as *mut u32;
            for (idx, token) in handoff.tokens_flat.iter().enumerate() {
                dst.add(idx).write(token.raw());
            }
        }

        let max_seqlen_q = handoff
            .cu_seqlens
            .windows(2)
            .map(|window| window[1].saturating_sub(window[0]))
            .max()
            .unwrap_or(0);
        self.write_prefill_layer_metadata(state, handoff)?;
        if !self.debug_sync
            && (self.explicit_options.is_some() || !metal_debug_layer_controls_enabled())
        {
            let ctx = self.ctx.as_ref().ok_or_else(|| {
                RvllmError::apple(
                    AppleError::NotPrepared {
                        backend: "model-metal-backend",
                    },
                    model_ctx("launch_prefill"),
                )
            })?;
            let queue = ctx.queue_retained();
            let cmd_buf = queue.commandBuffer().ok_or_else(|| {
                RvllmError::apple(AppleError::MetalUnavailable, model_ctx("launch_prefill"))
            })?;
            #[cfg(feature = "metal-stage-instrumentation")]
            let mut stage_profiler =
                MetalStageProfiler::new(ctx.device(), state.num_layers * 7 + 2).map_err(|_| {
                    RvllmError::apple(
                        AppleError::FeatureNotAvailable {
                            backend: "model-metal-backend",
                            op: "create_stage_timing",
                        },
                        model_ctx("launch_prefill"),
                    )
                })?;
            #[cfg(feature = "metal-stage-instrumentation")]
            unsafe {
                stage_profiler.begin(&cmd_buf, MetalStage::Embedding)
            };
            self.encode_embedding_gather(&cmd_buf, state, num_tokens)?;
            #[cfg(feature = "metal-stage-instrumentation")]
            unsafe {
                stage_profiler.end(&cmd_buf)
            };
            self.perf.add_embedding_encoders(1);
            let ple_encoders = self.encode_ple_inputs(&cmd_buf, state, num_tokens)?;
            self.perf.add_ple_encoders(ple_encoders);
            self.enqueue_probe_layers(
                Some(&cmd_buf),
                #[cfg(feature = "metal-stage-instrumentation")]
                Some(&mut stage_profiler),
                state,
                num_tokens,
                MetalPhase::Prefill {
                    max_seqlen_q,
                    batch_size: handoff.num_sequences() as u32,
                },
                "launch_prefill",
            )?;
            let output = if sample_first_token {
                #[cfg(feature = "metal-stage-instrumentation")]
                unsafe {
                    stage_profiler.begin(&cmd_buf, MetalStage::LmHead)
                };
                self.encode_prefill_first_token(&cmd_buf, state, num_tokens)?;
                #[cfg(feature = "metal-stage-instrumentation")]
                unsafe {
                    stage_profiler.end(&cmd_buf)
                };
                ModelGpuOutput::Tokens {
                    req_ids: handoff.req_ids.clone(),
                    sampled: state.sampled.clone(),
                }
            } else {
                ModelGpuOutput::Prefill
            };
            cmd_buf.commit();
            self.perf.add_command_buffers(1);
            return Ok(ModelLaunchResult::Submitted(ModelGpuSubmission {
                command_buffer: cmd_buf,
                output,
                perf_before,
                wall_start,
                num_tokens,
                is_decode: false,
                #[cfg(feature = "metal-stage-instrumentation")]
                stage_profiler,
            }));
        }
        self.enqueue_embedding_gather(state, num_tokens)?;
        self.enqueue_ple_inputs(state, num_tokens)?;
        self.enqueue_probe_layers(
            None,
            #[cfg(feature = "metal-stage-instrumentation")]
            None,
            state,
            num_tokens,
            MetalPhase::Prefill {
                max_seqlen_q,
                batch_size: handoff.num_sequences() as u32,
            },
            "launch_prefill",
        )?;
        self.wait_for_metal_queue("launch_prefill")?;
        self.perf
            .finish_step(false, num_tokens as u64, perf_before, wall_start.elapsed());
        Ok(ModelLaunchResult::Ready(Vec::new()))
    }

    fn run_decode_step(
        &self,
        handoff: &HandoffCapsule,
        execution_slot: usize,
    ) -> Result<ModelLaunchResult> {
        let perf_before = self.perf.snapshot();
        let wall_start = Instant::now();
        if handoff.tokens_flat.is_empty() {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "decode requires at least one token",
                },
                model_ctx("launch_rollout"),
            ));
        }
        if handoff.tokens_flat.len() != handoff.num_sequences() {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "launch_rollout",
                },
                model_ctx("launch_rollout"),
            ));
        }

        let state = self.execution_states.get(execution_slot).ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_rollout"),
            )
        })?;

        for &tok in &handoff.tokens_flat {
            if (tok.raw() as usize) >= state.vocab_size {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "token id exceeds vocabulary size",
                    },
                    model_ctx("launch_rollout"),
                ));
            }
        }
        let num_tokens = handoff.tokens_flat.len();
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_rollout"),
            )
        })?;
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_rollout"),
            )
        })?;
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_rollout"),
            )
        })?;

        unsafe {
            let dst = arena.host_ptr(&state.token_ids) as *mut u32;
            for (idx, token) in handoff.tokens_flat.iter().enumerate() {
                dst.add(idx).write(token.raw());
            }
        }

        let num_tokens_u32 = u32::try_from(num_tokens).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "token count exceeds u32",
                },
                model_ctx("launch_rollout"),
            )
        })?;
        let vocab_u32 = u32::try_from(state.vocab_size).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "vocab_size exceeds u32",
                },
                model_ctx("launch_rollout"),
            )
        })?;
        let hidden_u32 = u32::try_from(state.hidden_size).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "hidden_size exceeds u32",
                },
                model_ctx("launch_rollout"),
            )
        })?;

        self.write_decode_layer_metadata(state, handoff)?;
        if !self.debug_sync
            && (self.explicit_options.is_some() || !metal_debug_layer_controls_enabled())
        {
            let queue = ctx.queue_retained();
            let cmd_buf = queue.commandBuffer().ok_or_else(|| {
                RvllmError::apple(AppleError::MetalUnavailable, model_ctx("launch_rollout"))
            })?;
            #[cfg(feature = "metal-stage-instrumentation")]
            let mut stage_profiler =
                MetalStageProfiler::new(ctx.device(), state.num_layers * 7 + 2).map_err(|_| {
                    RvllmError::apple(
                        AppleError::FeatureNotAvailable {
                            backend: "model-metal-backend",
                            op: "create_stage_timing",
                        },
                        model_ctx("launch_rollout"),
                    )
                })?;
            #[cfg(feature = "metal-stage-instrumentation")]
            unsafe {
                stage_profiler.begin(&cmd_buf, MetalStage::Embedding)
            };
            self.encode_embedding_gather(&cmd_buf, state, num_tokens)?;
            #[cfg(feature = "metal-stage-instrumentation")]
            unsafe {
                stage_profiler.end(&cmd_buf)
            };
            self.perf.add_embedding_encoders(1);
            let ple_encoders = self.encode_ple_inputs(&cmd_buf, state, num_tokens)?;
            self.perf.add_ple_encoders(ple_encoders);
            self.enqueue_probe_layers(
                Some(&cmd_buf),
                #[cfg(feature = "metal-stage-instrumentation")]
                Some(&mut stage_profiler),
                state,
                num_tokens,
                MetalPhase::Decode,
                "launch_rollout",
            )?;
            if self.explicit_options.is_none() && metal_debug_skip_final_logits_enabled() {
                cmd_buf.commit();
                self.perf.add_command_buffers(1);
                let wait_start = Instant::now();
                cmd_buf.waitUntilCompleted();
                self.perf.add_forced_wait_duration(wait_start.elapsed());
                let outputs = handoff
                    .req_ids
                    .iter()
                    .copied()
                    .map(|req_id| StepToken {
                        req_id,
                        token_id: TokenId(0),
                        finished: false,
                    })
                    .collect();
                self.perf
                    .finish_step(true, num_tokens as u64, perf_before, wall_start.elapsed());
                return Ok(ModelLaunchResult::Ready(outputs));
            }
            unsafe {
                #[cfg(feature = "metal-stage-instrumentation")]
                stage_profiler.begin(&cmd_buf, MetalStage::LmHead);
                metal_encode_finalize_sample(
                    &cmd_buf,
                    pipelines,
                    arena,
                    num_tokens_u32,
                    hidden_u32,
                    vocab_u32,
                    state.rms_norm_eps,
                    state.final_logit_softcap,
                    state.residual.offset,
                    state.final_norm.offset,
                    state.lm_head.offset,
                    state.logits.offset,
                    state.normed_hidden.offset,
                    state.sampled.offset,
                    state.final_argmax_partial_max.offset,
                    state.final_argmax_partial_idx.offset,
                )?;
                #[cfg(feature = "metal-stage-instrumentation")]
                stage_profiler.end(&cmd_buf);
            }
            self.perf
                .add_final_sample_encoders(metal_finalize_sample_encoder_count(
                    num_tokens_u32,
                    hidden_u32,
                    vocab_u32,
                ));
            cmd_buf.commit();
            self.perf.add_command_buffers(1);
            return Ok(ModelLaunchResult::Submitted(ModelGpuSubmission {
                command_buffer: cmd_buf,
                output: ModelGpuOutput::Tokens {
                    req_ids: handoff.req_ids.clone(),
                    sampled: state.sampled.clone(),
                },
                perf_before,
                wall_start,
                num_tokens,
                is_decode: true,
                #[cfg(feature = "metal-stage-instrumentation")]
                stage_profiler,
            }));
        } else {
            self.enqueue_embedding_gather(state, num_tokens)?;
            self.enqueue_ple_inputs(state, num_tokens)?;
            self.enqueue_probe_layers(
                None,
                #[cfg(feature = "metal-stage-instrumentation")]
                None,
                state,
                num_tokens,
                MetalPhase::Decode,
                "launch_rollout",
            )?;
            if self.explicit_options.is_none() && metal_debug_skip_final_logits_enabled() {
                let outputs = handoff
                    .req_ids
                    .iter()
                    .copied()
                    .map(|req_id| StepToken {
                        req_id,
                        token_id: TokenId(0),
                        finished: false,
                    })
                    .collect();
                self.perf
                    .finish_step(true, num_tokens as u64, perf_before, wall_start.elapsed());
                return Ok(ModelLaunchResult::Ready(outputs));
            }

            unsafe {
                let queue = ctx.queue_retained();
                let cmd_buf = queue.commandBuffer().ok_or_else(|| {
                    RvllmError::apple(AppleError::MetalUnavailable, model_ctx("launch_rollout"))
                })?;
                metal_encode_finalize_sample(
                    &cmd_buf,
                    pipelines,
                    arena,
                    num_tokens_u32,
                    hidden_u32,
                    vocab_u32,
                    state.rms_norm_eps,
                    state.final_logit_softcap,
                    state.residual.offset,
                    state.final_norm.offset,
                    state.lm_head.offset,
                    state.logits.offset,
                    state.normed_hidden.offset,
                    state.sampled.offset,
                    state.final_argmax_partial_max.offset,
                    state.final_argmax_partial_idx.offset,
                )?;
                cmd_buf.commit();
                let wait_start = Instant::now();
                cmd_buf.waitUntilCompleted();
                self.perf.add_forced_wait_duration(wait_start.elapsed());
            }
            self.perf.add_command_buffers(1);
            self.perf
                .add_final_sample_encoders(metal_finalize_sample_encoder_count(
                    num_tokens_u32,
                    hidden_u32,
                    vocab_u32,
                ));
        }

        let sampled_ptr = unsafe { arena.host_ptr(&state.sampled) as *const i32 };
        let sampled = unsafe { std::slice::from_raw_parts(sampled_ptr, num_tokens) };
        let mut outputs = Vec::with_capacity(num_tokens);
        for (idx, &req_id) in handoff.req_ids.iter().enumerate() {
            let token = TokenId(sampled[idx] as u32);
            outputs.push(StepToken {
                req_id,
                token_id: token,
                finished: false,
            });
        }
        self.perf
            .finish_step(true, num_tokens as u64, perf_before, wall_start.elapsed());
        Ok(ModelLaunchResult::Ready(outputs))
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl AppleBackend for ModelMetalBackend {
    fn prepare(&mut self, plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
        plan.validate()?;
        self.prepared = false;
        self.prepared_model_layout_fingerprint = None;
        if !self.model_dir.exists() || !self.model_dir.is_dir() {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing model path",
                },
                model_ctx("prepare"),
            ));
        }
        self.next_step_id = 0;
        self.in_flight = ModelInFlightRing::default();
        self.ctx = None;
        self.pipelines = None;
        self.arena = None;
        self.state = None;
        self.execution_states.clear();
        self.float_type = None;
        self.debug_sync = self.explicit_options.is_none() && metal_debug_sync_enabled();
        self.experimental_kv_int8 =
            self.explicit_options.is_none() && experimental_metal_kv_int8_enabled();
        self.perf.clear();
        let state = self.initialize_model_resources()?;
        self.state = Some(state);
        self.prepared_model_layout_fingerprint = Some(plan.model_layout_hash);
        self.prepared = true;
        Ok(())
    }

    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        handoff.validate()?;
        self.validate_paged_kv_contract(handoff, "launch_prefill")?;
        let (ticket, execution_slot) = self.reserve_step(AppleLaunchKind::Prefill, None)?;
        let result = self.run_prefill_step(handoff, execution_slot, false);
        self.finish_launch(ticket, result)
    }

    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: Option<RolloutBucket>,
    ) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_rollout")?;
        handoff.validate()?;
        self.validate_paged_kv_contract(handoff, "launch_rollout")?;
        let (ticket, execution_slot) = self.reserve_step(AppleLaunchKind::Rollout, bucket)?;
        let result = self.run_decode_step(handoff, execution_slot);
        self.finish_launch(ticket, result)
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        self.ensure_prepared("collect")?;
        self.collect_model_step(ticket, true)?
            .ok_or_else(|| ModelInFlightRing::stale_ticket_error("collect_step_not_complete"))
    }
}

/// Exact host-visible page I/O over the shared-mode unified-memory arena.
///
/// The canonical serialized order is `(layer 0 K, layer 0 V, layer 1 K,
/// layer 1 V, ...)`, with each tensor containing all 32 token rows and every
/// element preserved as its native two-byte F16/BF16 bit pattern.
///
/// These operations intentionally use CPU memcpy rather than command-buffer
/// blits: every KV region is allocated from one `MTLStorageModeShared` arena on
/// Apple Silicon. The backend rejects access while any submitted command
/// buffer still owns that arena, including a command buffer that has completed
/// but has not yet passed through collection/reclamation.
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl KvPageIo for ModelMetalBackend {
    fn page_bytes(&self) -> Option<usize> {
        self.kv_page_layout()
            .ok()
            .map(|layout| layout.serialized_page_bytes)
    }

    fn capture_page(&mut self, page: BlockId) -> std::result::Result<Arc<[u8]>, KvPageIoError> {
        self.ensure_kv_page_io_idle()?;
        let layout = self.kv_page_layout()?;
        let page = Self::validate_kv_page(&layout, page)?;
        let arena = self
            .arena
            .as_ref()
            .ok_or(KvPageIoError::BackendUnavailable)?;
        let base = arena.buffer().contents().as_ptr() as *const u8;
        let mut serialized = Vec::with_capacity(layout.serialized_page_bytes);
        for layer in &layout.layers {
            let page_offset = page
                .checked_mul(layer.page_bytes_per_tensor)
                .ok_or(KvPageIoError::LayoutMismatch("KV page offset overflow"))?;
            for tensor_offset in [layer.k_offset, layer.v_offset] {
                let offset = tensor_offset
                    .checked_add(page_offset)
                    .ok_or(KvPageIoError::LayoutMismatch("KV page offset overflow"))?;
                // SAFETY: layout validation proves the range is inside its KV
                // region; busy-state rejection proves no GPU submission owns
                // the shared arena for the duration of this exclusive borrow.
                let bytes = unsafe {
                    std::slice::from_raw_parts(base.add(offset), layer.page_bytes_per_tensor)
                };
                serialized.extend_from_slice(bytes);
            }
        }
        if serialized.len() != layout.serialized_page_bytes {
            return Err(KvPageIoError::LayoutMismatch(
                "captured KV page length disagrees with validated layout",
            ));
        }
        Ok(serialized.into())
    }

    fn restore_page(
        &mut self,
        page: BlockId,
        bytes: &[u8],
    ) -> std::result::Result<(), KvPageIoError> {
        self.ensure_kv_page_io_idle()?;
        let layout = self.kv_page_layout()?;
        let page = Self::validate_kv_page(&layout, page)?;
        if bytes.len() != layout.serialized_page_bytes {
            return Err(KvPageIoError::InvalidPageBytes {
                expected: layout.serialized_page_bytes,
                got: bytes.len(),
            });
        }
        let arena = self
            .arena
            .as_ref()
            .ok_or(KvPageIoError::BackendUnavailable)?;
        let base = arena.buffer().contents().as_ptr() as *mut u8;
        let mut source_offset = 0usize;
        for layer in &layout.layers {
            let page_offset = page
                .checked_mul(layer.page_bytes_per_tensor)
                .ok_or(KvPageIoError::LayoutMismatch("KV page offset overflow"))?;
            for tensor_offset in [layer.k_offset, layer.v_offset] {
                let destination_offset = tensor_offset
                    .checked_add(page_offset)
                    .ok_or(KvPageIoError::LayoutMismatch("KV page offset overflow"))?;
                // SAFETY: validated region geometry and exact input length
                // prove both ranges; no submitted GPU command owns the arena.
                unsafe {
                    ptr::copy_nonoverlapping(
                        bytes.as_ptr().add(source_offset),
                        base.add(destination_offset),
                        layer.page_bytes_per_tensor,
                    );
                }
                source_offset += layer.page_bytes_per_tensor;
            }
        }
        debug_assert_eq!(source_offset, bytes.len());
        Ok(())
    }

    fn copy_page(&mut self, copy: CowPageCopy) -> std::result::Result<(), KvPageIoError> {
        self.ensure_kv_page_io_idle()?;
        let layout = self.kv_page_layout()?;
        if copy.valid_tokens == 0 || copy.valid_tokens > layout.block_size {
            return Err(KvPageIoError::LayoutMismatch(
                "COW valid-token count is outside the physical page",
            ));
        }
        let source = Self::validate_kv_page(&layout, copy.source)?;
        let destination = Self::validate_kv_page(&layout, copy.destination)?;
        if source == destination {
            return Err(KvPageIoError::LayoutMismatch(
                "COW source and destination pages are identical",
            ));
        }
        let arena = self
            .arena
            .as_ref()
            .ok_or(KvPageIoError::BackendUnavailable)?;
        let base = arena.buffer().contents().as_ptr() as *mut u8;
        for layer in &layout.layers {
            let source_page_offset = source
                .checked_mul(layer.page_bytes_per_tensor)
                .ok_or(KvPageIoError::LayoutMismatch("KV page offset overflow"))?;
            let destination_page_offset = destination
                .checked_mul(layer.page_bytes_per_tensor)
                .ok_or(KvPageIoError::LayoutMismatch("KV page offset overflow"))?;
            for tensor_offset in [layer.k_offset, layer.v_offset] {
                let source_offset = tensor_offset
                    .checked_add(source_page_offset)
                    .ok_or(KvPageIoError::LayoutMismatch("KV page offset overflow"))?;
                let destination_offset = tensor_offset
                    .checked_add(destination_page_offset)
                    .ok_or(KvPageIoError::LayoutMismatch("KV page offset overflow"))?;
                // SAFETY: source and destination are distinct validated
                // physical pages in shared-mode arena regions. Full-page copy
                // preserves the exact initialized F16/BF16 bits and harmlessly
                // carries the unused tail bytes along with the valid prefix.
                unsafe {
                    ptr::copy_nonoverlapping(
                        base.add(source_offset),
                        base.add(destination_offset),
                        layer.page_bytes_per_tensor,
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
pub type ToyMetalBackend = RuntimeMetalBackend;

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl RuntimeMetalBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_prepared(&self, op: &'static str) -> Result<()> {
        if self.prepared {
            Ok(())
        } else {
            Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "runtime-metal-backend",
                },
                ctx(op),
            ))
        }
    }

    fn next_ticket(
        &mut self,
        kind: AppleLaunchKind,
        bucket: Option<RolloutBucket>,
    ) -> AppleLaunchTicket {
        let step_id = self.next_step_id;
        self.next_step_id += 1;
        self.last_ticket = Some(step_id);
        AppleLaunchTicket {
            step_id,
            kind,
            bucket,
        }
    }

    fn initialize_metal_resources(&mut self) -> Result<()> {
        if self.ctx.is_some()
            && self.pipelines.is_some()
            && self.arena.is_some()
            && self.state.is_some()
        {
            return Ok(());
        }

        let mut ctx = MetalContext::new()?;
        load_model_kernel_library(
            &mut ctx,
            MetalFloatType::F16,
            None,
            rvllm_apple_metal::MetalKernelOptions::from_development_environment(),
        )?;

        let mut pipelines = PipelineCache::new();
        pipelines.compile_all(&ctx)?;

        let mut arena = MetalBufferArena::new(ctx.device(), METAL_ARENA_BYTES)?;
        let half_bytes = std::mem::size_of::<f16>();
        let i32_bytes = std::mem::size_of::<i32>();
        let hidden = METAL_HIDDEN;
        let vocab = METAL_VOCAB;
        let max_tokens = METAL_MAX_TOKENS;

        let residual = arena.region(
            "metal_decode_residual",
            max_tokens * hidden * half_bytes,
            16,
        )?;
        let final_norm = arena.region("metal_decode_final_norm", hidden * half_bytes, 16)?;
        let lm_head = arena.region("metal_decode_lm_head", vocab * hidden * half_bytes, 16)?;
        let logits = arena.region("metal_decode_logits", max_tokens * vocab * half_bytes, 16)?;
        let normed_hidden = arena.region(
            "metal_decode_normed_hidden",
            max_tokens * hidden * half_bytes,
            16,
        )?;
        let sampled = arena.region("metal_decode_sampled", max_tokens * i32_bytes, 4)?;

        let state = MetalState {
            residual,
            final_norm,
            lm_head,
            logits,
            normed_hidden,
            sampled,
        };
        self.fill_model_weights(&arena, &state)?;
        self.ctx = Some(ctx);
        self.pipelines = Some(pipelines);
        self.arena = Some(arena);
        self.state = Some(state);
        Ok(())
    }

    fn fill_model_weights(&self, arena: &MetalBufferArena, state: &MetalState) -> Result<()> {
        let half_bytes = std::mem::size_of::<f16>();
        let hidden = METAL_HIDDEN;
        let vocab = METAL_VOCAB;

        let final_norm: Vec<f16> = (0..hidden).map(|_| f16::from_f32(1.0)).collect();
        let mut lm_head = Vec::with_capacity(vocab * hidden);
        for v in 0..vocab {
            for d in 0..hidden {
                lm_head.push(if d == v {
                    f16::from_f32(1.0)
                } else {
                    f16::from_f32(0.0)
                });
            }
        }
        let residual_zero = vec![f16::from_f32(0.0); hidden];

        unsafe {
            let dst = arena.host_ptr(&state.final_norm);
            std::ptr::copy_nonoverlapping(
                final_norm.as_ptr() as *const u8,
                dst,
                final_norm.len() * half_bytes,
            );
            let lm_head_ptr = arena.host_ptr(&state.lm_head);
            std::ptr::copy_nonoverlapping(
                lm_head.as_ptr() as *const u8,
                lm_head_ptr,
                lm_head.len() * half_bytes,
            );
            let residual_ptr = arena.host_ptr(&state.residual);
            std::ptr::copy_nonoverlapping(
                residual_zero.as_ptr() as *const u8,
                residual_ptr,
                hidden * half_bytes,
            );
        }

        Ok(())
    }

    fn enqueue_rollout(&mut self, handoff: &HandoffCapsule) -> Result<()> {
        let ctx_ref = self.ctx.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "runtime-metal-backend",
                },
                ctx("enqueue_rollout"),
            )
        })?;
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "runtime-metal-backend",
                },
                ctx("enqueue_rollout"),
            )
        })?;
        let arena = self.arena.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "runtime-metal-backend",
                },
                ctx("enqueue_rollout"),
            )
        })?;
        let state = self.state.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "runtime-metal-backend",
                },
                ctx("enqueue_rollout"),
            )
        })?;

        if handoff.num_sequences() > METAL_MAX_TOKENS {
            return Err(RvllmError::apple(
                AppleError::ShapeBucketMissing {
                    seqs: handoff.num_sequences() as u32,
                    tokens: METAL_MAX_TOKENS as u32,
                },
                ctx("enqueue_rollout"),
            ));
        }

        let num_tokens = handoff.num_sequences();
        let half_bytes = std::mem::size_of::<f16>();

        let mut residual = vec![f16::from_f32(0.0); num_tokens * METAL_HIDDEN];
        for (seq, token) in handoff.tokens_flat.iter().enumerate() {
            let lane = (token.raw() as usize) % METAL_HIDDEN;
            residual[seq * METAL_HIDDEN + lane] = f16::from_f32(1.0);
        }

        unsafe {
            let dst = arena.host_ptr(&state.residual);
            let dst_slice = std::slice::from_raw_parts_mut(
                dst as *mut u8,
                num_tokens * METAL_HIDDEN * half_bytes,
            );
            let src = std::slice::from_raw_parts(
                residual.as_ptr() as *const u8,
                residual.len() * half_bytes,
            );
            dst_slice.fill(0);
            dst_slice.copy_from_slice(src);
        }

        unsafe {
            metal_finalize_logits_blocking(
                ctx_ref,
                pipelines,
                arena,
                num_tokens as u32,
                METAL_HIDDEN as u32,
                METAL_VOCAB as u32,
                METAL_EPS,
                METAL_SOFTCAP,
                state.residual.offset,
                state.final_norm.offset,
                state.lm_head.offset,
                state.logits.offset,
                state.normed_hidden.offset,
                state.sampled.offset,
            )?;
        }

        let sampled = unsafe {
            let sampled_ptr = arena.host_ptr(&state.sampled) as *const i32;
            std::slice::from_raw_parts(sampled_ptr, num_tokens)
        };
        let mut outputs = Vec::with_capacity(num_tokens);
        for (idx, &req_id) in handoff.req_ids.iter().enumerate() {
            let token = TokenId(sampled[idx] as u32);
            outputs.push(StepToken {
                req_id,
                token_id: token,
                finished: false,
            });
        }
        self.pending = Some(outputs);
        Ok(())
    }
}

#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
impl AppleBackend for RuntimeMetalBackend {
    fn prepare(&mut self, plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
        plan.validate()?;
        self.prepared = true;
        self.next_step_id = 0;
        self.last_ticket = None;
        self.pending = None;
        self.initialize_metal_resources()?;
        Ok(())
    }

    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        handoff.validate()?;
        self.pending = Some(Vec::new());
        Ok(self.next_ticket(AppleLaunchKind::Prefill, None))
    }

    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: Option<RolloutBucket>,
    ) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_rollout")?;
        handoff.validate()?;
        self.enqueue_rollout(handoff)?;
        Ok(self.next_ticket(AppleLaunchKind::Rollout, bucket))
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        match self.last_ticket {
            Some(expected) if expected == ticket.step_id => {
                Ok(self.pending.take().unwrap_or_default())
            }
            Some(_) => Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "runtime-metal-backend",
                    op: "collect_stale_ticket",
                },
                ctx("collect"),
            )),
            None => Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "runtime-metal-backend",
                },
                ctx("collect"),
            )),
        }
    }
}

#[cfg(test)]
mod platform_policy_tests {
    use super::ModelMetalPlatformPolicy;

    #[test]
    fn real_backend_and_shipping_asset_policy_match_target() {
        let policy = ModelMetalPlatformPolicy::current();
        assert_eq!(
            policy.real_model_backend,
            cfg!(all(
                feature = "apple",
                any(target_os = "macos", target_os = "ios")
            ))
        );
        assert_eq!(
            policy.environment_configuration,
            cfg!(all(feature = "apple", target_os = "macos"))
        );
        assert_eq!(
            policy.development_source_compilation,
            cfg!(all(feature = "apple", target_os = "macos"))
        );
        assert_eq!(
            policy.requires_explicit_precompiled_metallib,
            cfg!(all(feature = "apple", target_os = "ios"))
        );
        assert!(
            !policy.requires_explicit_precompiled_metallib
                || (!policy.environment_configuration && !policy.development_source_compilation),
            "shipping iOS must not consult env vars or compile source"
        );
    }
}

#[cfg(test)]
#[path = "apple_metal_backend_tests.rs"]
mod tests;
