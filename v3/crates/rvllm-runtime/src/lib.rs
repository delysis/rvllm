//! rvllm-runtime: Engine + scheduler + layer_exec per specs 07, 09.
//!
//! The public API surface for v3 callers:
//! - `Engine::new()` → init
//! - `engine.step_launch()` → returns `PendingStep<'_>`
//! - `engine.step_collect(ticket)` → waits DtoH, returns per-request
//!   outputs
//!
//! One codepath. No sync vs pipelined duality. Graph replay is a
//! transparent implementation detail.

#[cfg(feature = "apple")]
pub mod ane_prefill;
#[cfg(feature = "apple")]
pub mod apple_bridge;
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
pub mod apple_continuous_worker;
#[cfg(target_os = "macos")]
pub mod apple_measurement;
#[cfg(feature = "apple")]
pub mod apple_metal_backend;
pub mod bring_up;
pub mod engine;
pub mod gemma4_bring_up;
pub mod gemma4_layer_exec;
#[cfg(all(
    feature = "macos-private-ane-research",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub mod gemma_ane_decode;
#[cfg(all(
    feature = "macos-private-ane-research",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub mod gemma_disaggregated_worker;
pub mod gemma_head_ranking;
pub mod layer_exec;
pub mod paged_kv;
pub mod paged_prompt_cache;
pub mod persistent_prompt_cache;
pub mod prompt_cache;
pub mod request_api;
pub mod sched_state;
pub mod scheduler;
pub mod text_generation;

#[cfg(feature = "apple")]
pub use apple_bridge::{
    handoff_from_decode_plan, handoff_from_prefill_plan, rollout_bucket_for_decode,
};
#[cfg(all(feature = "apple", any(target_os = "macos", target_os = "ios")))]
pub use apple_continuous_worker::{
    create_development_apple_metal_worker,
    create_development_apple_metal_worker_with_persistent_host, create_embedded_apple_metal_worker,
    create_embedded_apple_metal_worker_with_persistent_host, AppleMetalWorkerHealth,
    DevelopmentMetalWorkerConfig,
};
#[cfg(feature = "apple")]
pub use apple_metal_backend::RuntimeMetalBackend;
pub use bring_up::{Bringup, EnginePaths, FusedModules, PplResult};
pub use engine::{Engine, PendingStep, StepOutput, SubmittedStep};
pub use layer_exec::{forward, LayerDims};
pub use paged_kv::{
    CowPageCopy, CowTailError, CowTailResult, KvChainHandle, KvChainId, KvChainView, KvPageLease,
    PagedKvConfig, PagedKvError, PagedKvPool, PagedKvStats, APPLE_KV_ABI_V1, APPLE_KV_PAGE_SIZE,
};
pub use paged_prompt_cache::{
    AttachedCacheTier, CachePrefixAttachment, KvPageIo, KvPageIoError, PagedPromptCache,
    PagedPromptCacheError, UnavailableKvPageIo,
};
pub use persistent_prompt_cache::{
    PersistentCacheError, PersistentCacheHostConfig, PersistentCacheKey, PersistentCacheRecord,
    PersistentLookup, PersistentPromptCache, PersistentPromptCacheConfig, PersistentStoreOutcome,
    RestoreCostEstimate, PERSISTENT_PROMPT_CACHE_FORMAT_VERSION,
};
pub use prompt_cache::{
    reusable_prompt_tokens, CacheIdentity, CacheLookup, CacheNamespace, MemoryPressure,
    PromptCache, PromptCacheConfig, PromptCacheMetrics, PROMPT_CACHE_PAGE_TOKENS,
};
pub use request_api::{
    ActiveRequest, AppleEngineConfig, BackendFallback, BackendKind, BackendPolicy, BackendReport,
    CachePolicy, CacheTier, ContinuousInferenceWorker, ContinuousStepOutput, EngineHandle,
    FinishReason, GenerateRequest, GenerationOutcome, InferenceError, InferenceWorker,
    MemoryProfile, RequestHandle, ThermalState, TokenEmitter, TokenEvent, WorkloadProfile,
};
pub use sched_state::{ReqState, Request};
pub use scheduler::{bucket_for, BatchPlan, Scheduler, DECODE_BUCKETS};
