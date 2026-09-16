//! Shared continuous Metal worker for macOS server and embedded Apple hosts.
//!
//! This module owns accelerator scheduling, request-owned paged KV, exact T1/T2
//! prompt caching, cancellation, and memory pressure. HTTP and C/Swift adapters
//! stay outside the runtime and only construct this worker.

use crate::apple_metal_backend::{MetalModelCapacity, ModelMetalBackend};
use crate::paged_prompt_cache::{PagedPromptCacheError, PreparedWarmRestore};
use crate::persistent_prompt_cache::PersistentPublishGate;
use crate::text_generation::IncrementalTextDecoder;
use crate::{
    reusable_prompt_tokens, ActiveRequest, AppleEngineConfig, AttachedCacheTier, BackendFallback,
    BackendKind, BackendPolicy, BackendReport, BatchPlan, CacheIdentity, CacheNamespace,
    CachePolicy, CachePrefixAttachment, CacheTier, ContinuousInferenceWorker, ContinuousStepOutput,
    Engine, FinishReason, GenerateRequest, GenerationOutcome, InferenceError, KvChainHandle,
    KvPageIo, KvPageIoError, MemoryPressure, PagedKvConfig, PagedPromptCache, PersistentCacheError,
    PersistentCacheHostConfig, PersistentLookup, PersistentPromptCache, PromptCacheConfig, Request,
    RestoreCostEstimate, Scheduler, SubmittedStep, APPLE_KV_PAGE_SIZE, PROMPT_CACHE_PAGE_TOKENS,
};
use rvllm_apple::{
    AppleBackend, AppleLaunchTicket, AppleModelPackage, AppleRuntimePlan, HandoffCapsule,
    RolloutBucket, StepToken,
};
use rvllm_core::{ReqId, TokenId};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

const METAL_PREFILL_TOKEN_BUDGET: usize = 128;
const METAL_PIPELINE_DEPTH: usize = 3;
static NEVER_CANCELLED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug)]
pub struct AppleMetalWorkerHealth {
    pub prepare_ms: f64,
    pub arena_bytes: usize,
    pub debug_sync: bool,
    pub max_batch_tokens: usize,
    pub physical_kv_pages: u32,
    pub kv_page_size: u32,
    pub capacity: MetalModelCapacity,
}

#[derive(Clone, Debug)]
pub struct DevelopmentMetalWorkerConfig {
    pub model_dir: PathBuf,
    pub max_supported_total_tokens: usize,
    pub rollout_tokens: usize,
    pub cache_namespace: String,
    pub enable_memory_cache: bool,
}

pub fn create_embedded_apple_metal_worker(
    engine_config: &AppleEngineConfig,
) -> Result<(Box<dyn ContinuousInferenceWorker>, AppleMetalWorkerHealth), InferenceError> {
    if engine_config.cache_policy == CachePolicy::PersistentEncrypted {
        return Err(InferenceError::InvalidConfig {
            field: "persistent_cache",
            reason: "encrypted persistence requires the explicit host-material creator",
        });
    }
    create_embedded_apple_metal_worker_inner(engine_config, None)
}

pub fn create_embedded_apple_metal_worker_with_persistent_host(
    engine_config: &AppleEngineConfig,
    host: PersistentCacheHostConfig,
) -> Result<(Box<dyn ContinuousInferenceWorker>, AppleMetalWorkerHealth), InferenceError> {
    if engine_config.cache_policy != CachePolicy::PersistentEncrypted {
        return Err(InferenceError::InvalidConfig {
            field: "cache_policy",
            reason: "persistent host material requires the encrypted persistent cache policy",
        });
    }
    create_embedded_apple_metal_worker_inner(engine_config, Some(host))
}

fn create_embedded_apple_metal_worker_inner(
    engine_config: &AppleEngineConfig,
    persistent_host: Option<PersistentCacheHostConfig>,
) -> Result<(Box<dyn ContinuousInferenceWorker>, AppleMetalWorkerHealth), InferenceError> {
    engine_config.validate()?;
    let cache_namespace = persistent_host
        .as_ref()
        .map(PersistentCacheHostConfig::tenant_namespace)
        .unwrap_or_else(|| "embedded-apple-session".to_owned());
    if matches!(engine_config.backend_policy, BackendPolicy::CoreMlOnly | BackendPolicy::MetalPrefillAneDecode) {
        return Err(InferenceError::InvalidConfig {
            field: "backend_policy",
            reason: "requested backend policy cannot be satisfied by the embedded Metal worker",
        });
    }
    let package_path =
        engine_config
            .model_package_path
            .as_ref()
            .ok_or(InferenceError::InvalidConfig {
                field: "model_package_path",
                reason: "a validated model package path is required",
            })?;
    let package = AppleModelPackage::open(package_path).map_err(|error| {
        InferenceError::worker_initialization_failed(format!(
            "validate embedded Apple model package: {error}"
        ))
    })?;
    let model_dir = package.root().to_path_buf();
    let generation_path = model_dir.join("generation_config.json");
    let authenticated_generation = package
        .manifest()
        .model_metadata_files
        .iter()
        .any(|file| file.path == Path::new("generation_config.json"));
    let eos_token_ids = rvllm_loader::generation::load_eos_token_ids_from_files(
        authenticated_generation.then_some(generation_path.as_path()),
        &model_dir.join("config.json"),
    )
    .map_err(InferenceError::worker_initialization_failed)?;
    let backend =
        ModelMetalBackend::from_model_package_path(model_dir.clone()).map_err(|error| {
            InferenceError::worker_initialization_failed(format!(
                "open packaged Metal backend: {error}"
            ))
        })?;
    let (worker, health) = AppleMetalContinuousWorker::prepare(
        engine_config,
        model_dir,
        eos_token_ids,
        backend,
        persistent_host,
        None,
        1,
        &cache_namespace,
        true,
    )
    .map_err(InferenceError::worker_initialization_failed)?;
    Ok((Box::new(worker), health))
}

pub fn create_development_apple_metal_worker(
    engine_config: &AppleEngineConfig,
    worker_config: &DevelopmentMetalWorkerConfig,
) -> Result<(Box<dyn ContinuousInferenceWorker>, AppleMetalWorkerHealth), InferenceError> {
    if engine_config.cache_policy == CachePolicy::PersistentEncrypted {
        return Err(InferenceError::InvalidConfig {
            field: "persistent_cache",
            reason: "encrypted persistence requires the explicit host-material creator",
        });
    }
    create_development_apple_metal_worker_inner(engine_config, worker_config, None)
}

pub fn create_development_apple_metal_worker_with_persistent_host(
    engine_config: &AppleEngineConfig,
    worker_config: &DevelopmentMetalWorkerConfig,
    host: PersistentCacheHostConfig,
) -> Result<(Box<dyn ContinuousInferenceWorker>, AppleMetalWorkerHealth), InferenceError> {
    if engine_config.cache_policy != CachePolicy::PersistentEncrypted {
        return Err(InferenceError::InvalidConfig {
            field: "cache_policy",
            reason: "persistent host material requires the encrypted persistent cache policy",
        });
    }
    create_development_apple_metal_worker_inner(engine_config, worker_config, Some(host))
}

fn create_development_apple_metal_worker_inner(
    engine_config: &AppleEngineConfig,
    worker_config: &DevelopmentMetalWorkerConfig,
    persistent_host: Option<PersistentCacheHostConfig>,
) -> Result<(Box<dyn ContinuousInferenceWorker>, AppleMetalWorkerHealth), InferenceError> {
    engine_config.validate()?;
    let cache_namespace = persistent_host
        .as_ref()
        .map(PersistentCacheHostConfig::tenant_namespace)
        .unwrap_or_else(|| worker_config.cache_namespace.clone());
    if matches!(engine_config.backend_policy, BackendPolicy::CoreMlOnly | BackendPolicy::MetalPrefillAneDecode) {
        return Err(InferenceError::InvalidConfig {
            field: "backend_policy",
            reason: "requested backend policy cannot be satisfied by a Metal development worker",
        });
    }
    let backend = ModelMetalBackend::new(worker_config.model_dir.clone());
    let (worker, health) = AppleMetalContinuousWorker::prepare(
        engine_config,
        worker_config.model_dir.clone(),
        rvllm_loader::generation::load_eos_token_ids(&worker_config.model_dir)
            .map_err(InferenceError::worker_initialization_failed)?,
        backend,
        persistent_host,
        Some(worker_config.max_supported_total_tokens),
        worker_config.rollout_tokens,
        &cache_namespace,
        worker_config.enable_memory_cache,
    )
    .map_err(InferenceError::worker_initialization_failed)?;
    Ok((Box::new(worker), health))
}

struct AppleMetalContinuousWorker {
    tokenizer: tokenizers::Tokenizer,
    eos_token_ids: Vec<u32>,
    engine: Engine,
    page_io: SharedModelMetalBackend,
    hot_cache: Option<PagedPromptCache>,
    warm_cache: Option<PagedPromptCache>,
    persistent_io: Option<PersistentIoWorker>,
    cache_identity: CacheIdentity,
    kv_page_bytes: usize,
    prefill_ns_per_token: Option<f64>,
    warm_copy_bytes_per_ns: Option<f64>,
    persistent_io_ns: Option<f64>,
    requests: HashMap<ReqId, RequestState>,
    maintenance_queue: VecDeque<ReqId>,
    submitted: VecDeque<SubmittedContinuousStep>,
    ready_outputs: VecDeque<ContinuousStepOutput>,
    deferred_error: Option<InferenceError>,
    max_supported_total_tokens: usize,
    resident_memory_bytes: u64,
    fallback_from_coreml: bool,
    pressure_admission: PressureAdmission,
}

struct SubmittedContinuousStep {
    submitted: SubmittedStep,
    submitted_at: Instant,
    prefill: bool,
    scheduled_ids: Vec<ReqId>,
    prefill_context_lens: Vec<u32>,
    prefill_tokens: u32,
}

struct RequestState {
    prompt_tokens: Vec<TokenId>,
    prompt_chain: Option<KvChainHandle>,
    priority: u8,
    maintenance: CacheMaintenance,
    decoder: IncrementalTextDecoder,
    prefill_time: Duration,
    decode_time: Duration,
    queue_time: Option<Duration>,
    max_batch_size: u32,
    terminal: Option<FinishReason>,
    backpressured: bool,
    cache_attachment: Option<CachePrefixAttachment>,
    cache_tier: CacheTier,
    matched_cache_tokens: u32,
    reserved_private_pages: u32,
    cache_enabled: bool,
    persistent_cache_enabled: bool,
    cache_promoted: bool,
    deferred_kv_release: bool,
    emitted_tokens: u32,
    max_output_tokens: u32,
}

enum CacheMaintenance {
    Active,
    Restoring {
        prepared: PreparedWarmRestore,
        source: RestoreSource,
        admitted_at: Instant,
    },
    PersistentLookup {
        admitted_at: Instant,
    },
    PersistentWaiting {
        admitted_at: Instant,
        job_id: u64,
        epoch: u64,
        cancellation: Arc<AtomicBool>,
    },
    /// Memory pressure invalidated an owned prepared restore before page I/O.
    Recompute,
    /// Terminal request KV remains owned until the submission ring is idle.
    Promoting,
    /// Memory pressure cancelled an optional terminal capture; ownership still
    /// resolves at the same idle maintenance barrier.
    SkipPromotion,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum RestoreSource {
    Warm,
    Persistent,
}

/// Warning pressure allows only zero-copy attachment to T1 pages that survived
/// eviction. Critical pressure disables every new cache attachment as well as
/// optional cache population and restore work. Only an explicit host-provided
/// [`MemoryPressure::Normal`] signal restores full admission.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
enum PressureAdmission {
    #[default]
    Nominal,
    Warning,
    Critical,
}

impl PressureAdmission {
    fn from_pressure(pressure: MemoryPressure) -> Self {
        match pressure {
            MemoryPressure::Normal => Self::Nominal,
            MemoryPressure::Warning => Self::Warning,
            MemoryPressure::Critical => Self::Critical,
        }
    }

    fn allows_hot_attachment(self) -> bool {
        self != Self::Critical
    }

    fn allows_optional_cache_work(self) -> bool {
        self == Self::Nominal
    }

    fn admission_limit(self, configured_maximum: usize) -> usize {
        match self {
            Self::Nominal => configured_maximum,
            Self::Warning => configured_maximum.div_ceil(2),
            Self::Critical => usize::from(configured_maximum > 0),
        }
    }

    fn active_pressure(self) -> Option<MemoryPressure> {
        match self {
            Self::Nominal => None,
            Self::Warning => Some(MemoryPressure::Warning),
            Self::Critical => Some(MemoryPressure::Critical),
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct CacheAdmissionMode {
    hot_attachment: bool,
    optional_cache_work: bool,
    persistent_cache: bool,
}

fn cache_admission_mode(
    request_policy: CachePolicy,
    pressure: PressureAdmission,
    has_hot_cache: bool,
    has_warm_cache: bool,
    has_persistent_cache: bool,
) -> CacheAdmissionMode {
    let requested = request_policy != CachePolicy::Disabled;
    let persistent_cache = requested
        && request_policy == CachePolicy::PersistentEncrypted
        && pressure.allows_optional_cache_work()
        && has_persistent_cache;
    CacheAdmissionMode {
        hot_attachment: requested && pressure.allows_hot_attachment() && has_hot_cache,
        optional_cache_work: requested
            && pressure.allows_optional_cache_work()
            && (has_hot_cache || has_warm_cache || persistent_cache),
        persistent_cache,
    }
}

fn cancel_optional_cache_work(maintenance: CacheMaintenance) -> (CacheMaintenance, bool) {
    match maintenance {
        CacheMaintenance::Restoring { .. } | CacheMaintenance::PersistentLookup { .. } => {
            (CacheMaintenance::Recompute, true)
        }
        CacheMaintenance::PersistentWaiting { cancellation, .. } => {
            cancellation.store(true, Ordering::Release);
            (CacheMaintenance::Recompute, true)
        }
        CacheMaintenance::Promoting => (CacheMaintenance::SkipPromotion, true),
        other => (other, false),
    }
}

fn throttle_existing_request_cache_parts(
    cache_enabled: &mut bool,
    persistent_cache_enabled: &mut bool,
    maintenance: &mut CacheMaintenance,
) -> bool {
    *cache_enabled = false;
    *persistent_cache_enabled = false;
    let previous = std::mem::replace(maintenance, CacheMaintenance::Active);
    let (next, actionable) = cancel_optional_cache_work(previous);
    *maintenance = next;
    actionable
}

fn throttle_existing_request_cache(state: &mut RequestState) -> bool {
    // The request's active T0 attachment and chain are deliberately untouched.
    throttle_existing_request_cache_parts(
        &mut state.cache_enabled,
        &mut state.persistent_cache_enabled,
        &mut state.maintenance,
    )
}

fn released_attachment_resweep(
    released_cache_attachment: bool,
    pressure: PressureAdmission,
) -> Option<MemoryPressure> {
    released_cache_attachment
        .then(|| pressure.active_pressure())
        .flatten()
}

impl CacheMaintenance {
    fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    fn is_awaiting_activation(&self) -> bool {
        matches!(
            self,
            Self::Restoring { .. }
                | Self::PersistentLookup { .. }
                | Self::PersistentWaiting { .. }
                | Self::Recompute
        )
    }
}

enum PersistentIoCommand {
    Lookup {
        job_id: u64,
        epoch: u64,
        identity: CacheIdentity,
        prompt: Vec<TokenId>,
        cost: RestoreCostEstimate,
        cancellation: Arc<AtomicBool>,
    },
    Store {
        job_id: u64,
        epoch: u64,
        identity: CacheIdentity,
        prompt: Vec<TokenId>,
        pages: Vec<crate::prompt_cache::WarmPage>,
        cancellation: Arc<AtomicBool>,
    },
}

enum PersistentIoResult {
    Lookup {
        job_id: u64,
        epoch: u64,
        elapsed: Duration,
        result: Result<PersistentLookup, PersistentCacheError>,
    },
    Store {
        job_id: u64,
        epoch: u64,
        elapsed: Duration,
        result: Result<crate::PersistentStoreOutcome, PersistentCacheError>,
    },
}

struct PersistentIoWorker {
    commands: SyncSender<PersistentIoCommand>,
    results: Receiver<PersistentIoResult>,
    epoch: Arc<AtomicU64>,
    publish_gate: Arc<PersistentPublishGate>,
    next_job_id: u64,
}

impl PersistentIoWorker {
    fn spawn(store: PersistentPromptCache) -> Result<Self, String> {
        let (command_tx, command_rx) = sync_channel::<PersistentIoCommand>(1);
        let (result_tx, result_rx) = sync_channel::<PersistentIoResult>(4);
        let epoch = Arc::new(AtomicU64::new(1));
        let worker_epoch = Arc::clone(&epoch);
        let publish_gate = Arc::new(PersistentPublishGate::default());
        let worker_publish_gate = Arc::clone(&publish_gate);
        thread::Builder::new()
            .name("rvllm-persistent-cache".to_owned())
            .spawn(move || {
                while let Ok(command) = command_rx.recv() {
                    match command {
                        PersistentIoCommand::Lookup {
                            job_id,
                            epoch,
                            identity,
                            prompt,
                            cost,
                            cancellation,
                        } => {
                            if worker_epoch.load(Ordering::Acquire) != epoch {
                                continue;
                            }
                            let started = Instant::now();
                            let result =
                                store.restore_prompt_cancellable(&identity, &prompt, cost, || {
                                    worker_epoch.load(Ordering::Acquire) != epoch
                                        || cancellation.load(Ordering::Acquire)
                                });
                            if matches!(
                                &result,
                                Err(PersistentCacheError::CorruptRecord
                                    | PersistentCacheError::AuthenticationFailed
                                    | PersistentCacheError::IdentityMismatch
                                    | PersistentCacheError::Io(_))
                            ) {
                                let _ =
                                    store.quarantine_prompt_cancellable(&identity, &prompt, || {
                                        worker_epoch.load(Ordering::Acquire) != epoch
                                            || cancellation.load(Ordering::Acquire)
                                    });
                            }
                            if worker_epoch.load(Ordering::Acquire) == epoch {
                                let _ = result_tx.send(PersistentIoResult::Lookup {
                                    job_id,
                                    epoch,
                                    elapsed: started.elapsed(),
                                    result,
                                });
                            }
                        }
                        PersistentIoCommand::Store {
                            job_id,
                            epoch,
                            identity,
                            prompt,
                            pages,
                            cancellation,
                        } => {
                            if worker_epoch.load(Ordering::Acquire) != epoch {
                                continue;
                            }
                            let started = Instant::now();
                            let result = store.store_prompt_cancellable_with_publish_gate(
                                &identity,
                                &prompt,
                                &pages,
                                || {
                                    worker_epoch.load(Ordering::Acquire) != epoch
                                        || cancellation.load(Ordering::Acquire)
                                },
                                Some(&worker_publish_gate),
                            );
                            if worker_epoch.load(Ordering::Acquire) == epoch {
                                let _ = result_tx.send(PersistentIoResult::Store {
                                    job_id,
                                    epoch,
                                    elapsed: started.elapsed(),
                                    result,
                                });
                            }
                        }
                    }
                }
            })
            .map_err(|error| format!("spawn persistent cache I/O worker: {error}"))?;
        Ok(Self {
            commands: command_tx,
            results: result_rx,
            epoch,
            publish_gate,
            next_job_id: 1,
        })
    }

    fn next_job(&mut self) -> (u64, u64) {
        let job_id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1).max(1);
        (job_id, self.epoch.load(Ordering::Acquire))
    }

    fn invalidate(&self) {
        self.publish_gate.invalidate(|| {
            self.epoch.fetch_add(1, Ordering::AcqRel);
        });
    }
}

#[derive(Clone)]
struct SharedModelMetalBackend(Rc<RefCell<ModelMetalBackend>>);

impl AppleBackend for SharedModelMetalBackend {
    fn prepare(&mut self, plan: &AppleRuntimePlan) -> rvllm_core::Result<()> {
        self.0.borrow_mut().prepare(plan)
    }

    fn launch_prefill(
        &mut self,
        handoff: &HandoffCapsule,
    ) -> rvllm_core::Result<AppleLaunchTicket> {
        self.0.borrow_mut().launch_prefill(handoff)
    }

    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: Option<RolloutBucket>,
    ) -> rvllm_core::Result<AppleLaunchTicket> {
        self.0.borrow_mut().launch_rollout(handoff, bucket)
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> rvllm_core::Result<Vec<StepToken>> {
        self.0.borrow_mut().collect(ticket)
    }
}

impl KvPageIo for SharedModelMetalBackend {
    fn page_bytes(&self) -> Option<usize> {
        self.0.borrow().page_bytes()
    }

    fn capture_page(&mut self, page: rvllm_core::BlockId) -> Result<Arc<[u8]>, KvPageIoError> {
        self.0.borrow_mut().capture_page(page)
    }

    fn restore_page(
        &mut self,
        page: rvllm_core::BlockId,
        bytes: &[u8],
    ) -> Result<(), KvPageIoError> {
        self.0.borrow_mut().restore_page(page, bytes)
    }

    fn copy_page(&mut self, copy: crate::CowPageCopy) -> Result<(), KvPageIoError> {
        self.0.borrow_mut().copy_page(copy)
    }
}

impl AppleMetalContinuousWorker {
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        engine_config: &AppleEngineConfig,
        model_dir: PathBuf,
        eos_token_ids: Vec<u32>,
        mut backend: ModelMetalBackend,
        persistent_host: Option<PersistentCacheHostConfig>,
        expected_max_total_tokens: Option<usize>,
        rollout_tokens: usize,
        cache_namespace: &str,
        enable_memory_cache: bool,
    ) -> Result<(Self, AppleMetalWorkerHealth), String> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("load tokenizer {}: {error}", tokenizer_path.display()))?;
        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&model_dir)
            .map_err(|error| format!("parse Gemma4 architecture: {error}"))?;
        let layout = model_layout_fingerprint(&model_dir)?;
        let plan = build_runtime_plan(model_dir.clone(), &arch, rollout_tokens, layout)?;
        let prepare_start = Instant::now();
        backend
            .prepare(&plan)
            .map_err(|error| format!("prepare Metal backend: {error}"))?;
        let capacity = backend
            .model_capacity()
            .ok_or_else(|| "prepared Metal backend did not report capacity".to_owned())?;
        if capacity.kv_page_size != APPLE_KV_PAGE_SIZE {
            return Err(format!(
                "prepared KV page size {} does not match Apple ABI {}",
                capacity.kv_page_size, APPLE_KV_PAGE_SIZE
            ));
        }
        if let Some(expected) = expected_max_total_tokens {
            if expected != capacity.max_context_tokens {
                return Err(format!(
                    "prepared context capacity {} does not match configured {expected}",
                    capacity.max_context_tokens
                ));
            }
        }
        if engine_config.maximum_concurrency > capacity.max_batch_sequences {
            return Err(format!(
                "configured concurrency {} exceeds Metal capacity {}",
                engine_config.maximum_concurrency, capacity.max_batch_sequences
            ));
        }
        if capacity.max_context_tokens == 0 || capacity.max_batch_tokens == 0 {
            return Err("Metal backend reported empty execution capacity".to_owned());
        }
        let arena_bytes = backend
            .probe_arena_stats()
            .map(|arena| arena.capacity_bytes)
            .unwrap_or(0);
        let health = AppleMetalWorkerHealth {
            prepare_ms: prepare_start.elapsed().as_secs_f64() * 1000.0,
            arena_bytes,
            debug_sync: backend.metal_debug_sync_enabled(),
            max_batch_tokens: capacity.max_batch_tokens,
            physical_kv_pages: capacity.physical_kv_pages,
            kv_page_size: capacity.kv_page_size,
            capacity: capacity.clone(),
        };

        let max_chains = u32::try_from(engine_config.maximum_concurrency)
            .map_err(|_| "Metal request concurrency must fit u32".to_owned())?;
        let mut engine = Engine::new()
            .with_paged_kv_config(
                PagedKvConfig::apple_v1(capacity.physical_kv_pages, max_chains),
                layout,
            )
            .map_err(|error| format!("configure paged KV: {error}"))?;
        // Keep the scheduler's conservative decode target until an Apple
        // launch ticket exposes accelerator completion timing. Measuring from
        // submit to host collection would fold queueing and caller
        // backpressure into the sample, so it is not a valid calibrated
        // batch-one token latency.
        engine.scheduler = Scheduler::with_config(crate::scheduler::SchedulerConfig {
            max_prefill_tokens: u32::try_from(
                METAL_PREFILL_TOKEN_BUDGET.min(capacity.max_batch_tokens),
            )
            .map_err(|_| "Metal prefill capacity exceeds u32".to_owned())?,
            max_decode_sequences: max_chains,
            ..crate::scheduler::SchedulerConfig::default()
        });
        let shared_backend = SharedModelMetalBackend(Rc::new(RefCell::new(backend)));
        let page_bytes = shared_backend
            .page_bytes()
            .ok_or_else(|| "prepared Metal backend did not expose KV page bytes".to_owned())?;
        let persistent_cache = if engine_config.cache_policy == CachePolicy::PersistentEncrypted {
            Some(
                persistent_host
                    .ok_or_else(|| "persistent cache host configuration disappeared".to_owned())?
                    .build_store()
                    .map_err(|error| format!("configure encrypted T3 prompt cache: {error}"))?,
            )
        } else if persistent_host.is_some() {
            return Err("persistent host material supplied while T3 policy is disabled".to_owned());
        } else {
            None
        };
        let persistent_enabled = persistent_cache.is_some();
        let persistent_io = persistent_cache
            .map(PersistentIoWorker::spawn)
            .transpose()?;
        let memory_cache_enabled =
            enable_memory_cache && engine_config.cache_policy != CachePolicy::Disabled;
        let hot_cache = make_cache(
            memory_cache_enabled
                .then_some(engine_config.hot_cache_bytes)
                .unwrap_or(0),
            0,
            layout,
            page_bytes,
            "T1",
        )?;
        let warm_cache = make_cache(
            0,
            if memory_cache_enabled {
                engine_config.warm_cache_bytes
            } else if persistent_enabled {
                // A one-byte metadata budget keeps the exact page-I/O bridge
                // available without retaining T2 entries.
                1
            } else {
                0
            },
            layout,
            page_bytes,
            "T2",
        )?;
        let cache_identity = cache_identity(
            &model_dir,
            layout,
            capacity.numeric_abi_fingerprint,
            cache_namespace,
        )?;
        engine.apple_backend = Some(Box::new(shared_backend.clone()));
        engine.apple_runtime_plan = Some(plan);
        Ok((
            Self {
                tokenizer,
                eos_token_ids,
                engine,
                page_io: shared_backend,
                hot_cache,
                warm_cache,
                persistent_io,
                cache_identity,
                kv_page_bytes: page_bytes,
                prefill_ns_per_token: None,
                warm_copy_bytes_per_ns: None,
                persistent_io_ns: None,
                requests: HashMap::with_capacity(engine_config.maximum_concurrency),
                maintenance_queue: VecDeque::with_capacity(engine_config.maximum_concurrency),
                submitted: VecDeque::with_capacity(METAL_PIPELINE_DEPTH),
                ready_outputs: VecDeque::with_capacity(engine_config.maximum_concurrency),
                deferred_error: None,
                max_supported_total_tokens: capacity.max_context_tokens,
                resident_memory_bytes: arena_bytes as u64,
                fallback_from_coreml: engine_config.backend_policy
                    == BackendPolicy::CoreMlPreferred,
                pressure_admission: PressureAdmission::Nominal,
            },
            health,
        ))
    }

    fn promote_completed_prompt(
        &mut self,
        request_id: ReqId,
        prompt: &[TokenId],
    ) -> Result<(), InferenceError> {
        if !self
            .requests
            .get(&request_id)
            .is_some_and(|state| state.cache_enabled && !state.cache_promoted)
        {
            return Ok(());
        }
        let chain = self
            .engine
            .scheduler
            .kv_chain_for(request_id)
            .ok_or_else(|| {
                InferenceError::worker_failed("completed prompt has no request-owned KV chain")
            })?;
        if let Some(cache) = self.hot_cache.as_mut() {
            let _ = cache.promote_hot(
                self.engine
                    .paged_kv_pool_mut()
                    .ok_or_else(|| InferenceError::worker_failed("paged KV pool disappeared"))?,
                self.cache_identity.clone(),
                prompt,
                chain,
            );
        }
        if let Some(state) = self.requests.get_mut(&request_id) {
            state.cache_promoted = true;
            state.prompt_chain = Some(chain);
        }
        Ok(())
    }

    fn capture_terminal_warm(
        &mut self,
        state: &RequestState,
        cancellation: &AtomicBool,
    ) -> Result<(), InferenceError> {
        if !state.cache_enabled
            || self.warm_cache.is_none()
            || reusable_prompt_tokens(state.prompt_tokens.len()) == 0
        {
            return Ok(());
        }
        let chain = state.prompt_chain.ok_or_else(|| {
            InferenceError::worker_failed("completed prompt has no stable KV chain for T2")
        })?;
        let start = Instant::now();
        let cache = self
            .warm_cache
            .as_mut()
            .ok_or_else(|| InferenceError::worker_failed("warm cache disappeared"))?;
        let (_, pages) = cache
            .capture_warm(
                self.engine
                    .paged_kv_pool()
                    .ok_or_else(|| InferenceError::worker_failed("paged KV pool disappeared"))?,
                &mut self.page_io,
                self.cache_identity.clone(),
                &state.prompt_tokens,
                chain,
                || cancellation.load(Ordering::Acquire),
            )
            .map_err(|error| {
                InferenceError::worker_failed(format!(
                    "capture terminal exact T2 prompt pages: {error}"
                ))
            })?;
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        let copied = pages.len().saturating_mul(self.kv_page_bytes) as f64;
        if elapsed_ns > 0.0 && copied > 0.0 {
            update_ewma(&mut self.warm_copy_bytes_per_ns, copied / elapsed_ns);
        }
        if state.persistent_cache_enabled {
            let io = self
                .persistent_io
                .as_mut()
                .ok_or_else(|| InferenceError::worker_failed("persistent cache disappeared"))?;
            let (job_id, epoch) = io.next_job();
            io.commands
                .try_send(PersistentIoCommand::Store {
                    job_id,
                    epoch,
                    identity: self.cache_identity.clone(),
                    prompt: state.prompt_tokens.clone(),
                    pages,
                    cancellation: Arc::new(AtomicBool::new(false)),
                })
                .map_err(|error| {
                    let reason = match error {
                        TrySendError::Full(_) => "persistent cache I/O queue is full",
                        TrySendError::Disconnected(_) => {
                            "persistent cache I/O worker is unavailable"
                        }
                    };
                    InferenceError::worker_failed(reason)
                })?;
        }
        Ok(())
    }

    fn activate_request_state(
        &mut self,
        request_id: ReqId,
        state: &mut RequestState,
        mut attachment: Option<CachePrefixAttachment>,
    ) -> Result<(), InferenceError> {
        let scheduled = Request::new(
            request_id,
            state.prompt_tokens.clone(),
            state.max_output_tokens,
        )
        .with_priority(u32::from(state.priority));
        let admission = if let Some(value) = attachment.as_ref() {
            self.engine
                .enqueue_request_with_cache_attachment(scheduled, value)
        } else {
            self.engine.enqueue_request(scheduled)
        };
        if let Err(error) = admission {
            if let Some(value) = attachment.take() {
                self.release_unadmitted_attachment(&value)?;
            }
            return Err(InferenceError::worker_failed(format!(
                "activate maintained request: {error}"
            )));
        }

        let deferred_kv_release = state.cache_enabled && attachment.is_none();
        if deferred_kv_release {
            if let Err(error) = self.engine.defer_paged_kv_release(request_id) {
                let _ = self.engine.cancel_request(request_id);
                return Err(InferenceError::worker_failed(format!(
                    "defer cache-eligible KV release after maintenance: {error}"
                )));
            }
        }
        state.prompt_chain = self.engine.scheduler.kv_chain_for(request_id);
        state.deferred_kv_release = deferred_kv_release;
        state.cache_attachment = attachment;
        if let Some(value) = state.cache_attachment.as_ref() {
            state.cache_tier = match value.tier {
                AttachedCacheTier::T1Hot => CacheTier::Hot,
                AttachedCacheTier::T2Warm => CacheTier::Warm,
            };
            state.matched_cache_tokens = value.matched_tokens;
            state.cache_promoted = true;
        } else {
            state.cache_tier = CacheTier::None;
            state.matched_cache_tokens = 0;
        }
        state.maintenance = CacheMaintenance::Active;
        Ok(())
    }

    fn finish_request_state(
        &mut self,
        request_id: ReqId,
        finish_reason: FinishReason,
        mut state: RequestState,
    ) -> Result<ContinuousStepOutput, InferenceError> {
        self.release_state_resources(request_id, &mut state)?;
        let mut report = BackendReport::new(BackendKind::Metal);
        report.cache_tier = state.cache_tier;
        report.matched_cache_tokens = state.matched_cache_tokens;
        report.saved_prefill_tokens = state.matched_cache_tokens;
        report.queue_time = state.queue_time.unwrap_or_default();
        report.batch_size = state.max_batch_size;
        report.prefill_time = state.prefill_time;
        report.decode_time = state.decode_time;
        report.resident_memory_bytes = self.resident_memory_bytes;
        if self.fallback_from_coreml {
            report.fallback = Some(BackendFallback {
                from: BackendKind::CoreMl,
                to: BackendKind::Metal,
                reason: Arc::from(
                    "Core ML was preferred but no promoted public Core ML route is available",
                ),
            });
        }
        Ok(ContinuousStepOutput::Finished {
            request_id,
            outcome: GenerationOutcome {
                finish_reason,
                report,
            },
        })
    }

    fn process_one_maintenance(
        &mut self,
        active: &[ActiveRequest<'_>],
    ) -> Result<Option<ContinuousStepOutput>, InferenceError> {
        debug_assert!(self.submitted.is_empty());
        let requests = &self.requests;
        let request_id = pop_next_pending(&mut self.maintenance_queue, |request_id| {
            requests
                .get(&request_id)
                .is_some_and(|state| !state.maintenance.is_active())
        });
        let Some(request_id) = request_id else {
            return Ok(None);
        };
        let cancelled = active
            .iter()
            .find(|view| view.request_id == request_id)
            .is_some_and(|view| view.cancellation_signal().load(Ordering::Acquire));
        let Some(mut state) = self.requests.remove(&request_id) else {
            return Ok(None);
        };
        let maintenance = std::mem::replace(&mut state.maintenance, CacheMaintenance::Active);
        match maintenance {
            CacheMaintenance::Active => {
                self.requests.insert(request_id, state);
                Ok(None)
            }
            CacheMaintenance::Restoring {
                prepared,
                source,
                admitted_at,
            } => {
                if cancelled {
                    self.release_state_resources(request_id, &mut state)?;
                    return Ok(None);
                }
                let still_faster = match source {
                    RestoreSource::Warm => self.warm_restore_still_faster(
                        prepared.matched_tokens(),
                        admitted_at.elapsed(),
                    ),
                    RestoreSource::Persistent => self
                        .persistent_restore_cost(state.prompt_tokens.len(), admitted_at.elapsed())
                        .is_some_and(RestoreCostEstimate::should_restore),
                };
                let attachment = if still_faster {
                    let restore_started = Instant::now();
                    let result = self
                        .warm_cache
                        .as_mut()
                        .ok_or_else(|| InferenceError::worker_failed("warm cache disappeared"))?
                        .materialize_prepared_warm(
                            self.engine.paged_kv_pool_mut().ok_or_else(|| {
                                InferenceError::worker_failed("paged KV pool disappeared")
                            })?,
                            &mut self.page_io,
                            request_id,
                            &prepared,
                            || cancelled,
                        );
                    match result {
                        Ok(attachment) => {
                            let elapsed_ns = restore_started.elapsed().as_nanos() as f64;
                            let copied =
                                prepared.page_count().saturating_mul(self.kv_page_bytes) as f64;
                            if elapsed_ns > 0.0 && copied > 0.0 {
                                update_ewma(&mut self.warm_copy_bytes_per_ns, copied / elapsed_ns);
                            }
                            attachment
                        }
                        Err(PagedPromptCacheError::Cancelled) => {
                            self.release_state_resources(request_id, &mut state)?;
                            return Ok(None);
                        }
                        Err(PagedPromptCacheError::PageIo(KvPageIoError::BackendBusy)) => {
                            tracing::warn!(
                                request_id = ?request_id,
                                "idle T2 restore still reported busy; recomputing exact prefix"
                            );
                            None
                        }
                        Err(error) => {
                            tracing::warn!(
                                request_id = ?request_id,
                                error = %error,
                                "T2 restore skipped after transactional failure; recomputing exact prefix"
                            );
                            None
                        }
                    }
                } else {
                    None
                };
                match self.activate_request_state(request_id, &mut state, attachment) {
                    Ok(()) => {
                        if source == RestoreSource::Persistent && state.cache_attachment.is_some() {
                            state.cache_tier = CacheTier::Persistent;
                        }
                        self.requests.insert(request_id, state);
                        Ok(None)
                    }
                    Err(error) => Ok(Some(ContinuousStepOutput::Failed { request_id, error })),
                }
            }
            CacheMaintenance::PersistentLookup { admitted_at } => {
                if cancelled {
                    self.release_state_resources(request_id, &mut state)?;
                    return Ok(None);
                }
                let Some(cost) =
                    self.persistent_restore_cost(state.prompt_tokens.len(), admitted_at.elapsed())
                else {
                    return match self.activate_request_state(request_id, &mut state, None) {
                        Ok(()) => {
                            self.requests.insert(request_id, state);
                            Ok(None)
                        }
                        Err(error) => Ok(Some(ContinuousStepOutput::Failed { request_id, error })),
                    };
                };
                if !cost.should_restore() {
                    return match self.activate_request_state(request_id, &mut state, None) {
                        Ok(()) => {
                            self.requests.insert(request_id, state);
                            Ok(None)
                        }
                        Err(error) => Ok(Some(ContinuousStepOutput::Failed { request_id, error })),
                    };
                }
                let io = self.persistent_io.as_mut().ok_or_else(|| {
                    InferenceError::worker_failed("persistent cache I/O worker disappeared")
                })?;
                let (job_id, epoch) = io.next_job();
                let job_cancellation = Arc::new(AtomicBool::new(false));
                match io.commands.try_send(PersistentIoCommand::Lookup {
                    job_id,
                    epoch,
                    identity: self.cache_identity.clone(),
                    prompt: state.prompt_tokens.clone(),
                    cost,
                    cancellation: Arc::clone(&job_cancellation),
                }) {
                    Ok(()) => {
                        state.maintenance = CacheMaintenance::PersistentWaiting {
                            admitted_at,
                            job_id,
                            epoch,
                            cancellation: job_cancellation,
                        };
                        self.requests.insert(request_id, state);
                        Ok(None)
                    }
                    Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                        tracing::warn!(
                            request_id = ?request_id,
                            "bounded T3 I/O queue unavailable; recomputing prompt"
                        );
                        match self.activate_request_state(request_id, &mut state, None) {
                            Ok(()) => {
                                self.requests.insert(request_id, state);
                                Ok(None)
                            }
                            Err(error) => {
                                Ok(Some(ContinuousStepOutput::Failed { request_id, error }))
                            }
                        }
                    }
                }
            }
            CacheMaintenance::PersistentWaiting { .. } => {
                self.requests.insert(request_id, state);
                Ok(None)
            }
            CacheMaintenance::Recompute => {
                if cancelled {
                    self.release_state_resources(request_id, &mut state)?;
                    return Ok(None);
                }
                match self.activate_request_state(request_id, &mut state, None) {
                    Ok(()) => {
                        self.requests.insert(request_id, state);
                        Ok(None)
                    }
                    Err(error) => Ok(Some(ContinuousStepOutput::Failed { request_id, error })),
                }
            }
            CacheMaintenance::Promoting => {
                let finish_reason = state.terminal.ok_or_else(|| {
                    InferenceError::worker_failed("pending T2 promotion is not terminal")
                })?;
                if !cancelled {
                    if let Err(error) = self.capture_terminal_warm(
                        &state,
                        active
                            .iter()
                            .find(|view| view.request_id == request_id)
                            .map_or(&NEVER_CANCELLED, |view| view.cancellation_signal()),
                    ) {
                        tracing::warn!(
                            request_id = ?request_id,
                            error = %error,
                            "terminal T2 capture explicitly skipped; request KV ownership will be released"
                        );
                    }
                }
                self.finish_request_state(request_id, finish_reason, state)
                    .map(Some)
            }
            CacheMaintenance::SkipPromotion => {
                let finish_reason = state.terminal.ok_or_else(|| {
                    InferenceError::worker_failed("skipped T2 promotion is not terminal")
                })?;
                self.finish_request_state(request_id, finish_reason, state)
                    .map(Some)
            }
        }
    }

    fn warm_restore_predicted_faster(&self, prompt_len: usize) -> bool {
        self.warm_restore_still_faster(reusable_prompt_tokens(prompt_len) as u32, Duration::ZERO)
    }

    fn warm_restore_still_faster(&self, matched_tokens: u32, waited: Duration) -> bool {
        let (Some(prefill), Some(copy)) = (self.prefill_ns_per_token, self.warm_copy_bytes_per_ns)
        else {
            return false;
        };
        warm_restore_beats_recompute(matched_tokens, self.kv_page_bytes, prefill, copy, waited)
    }

    fn persistent_restore_cost(
        &self,
        prompt_len: usize,
        waited: Duration,
    ) -> Option<RestoreCostEstimate> {
        let prefill = self.prefill_ns_per_token?;
        let copy = self.warm_copy_bytes_per_ns?;
        let persistent = self.persistent_io_ns?;
        persistent_restore_cost_estimate(
            prompt_len,
            self.kv_page_bytes,
            prefill,
            copy,
            persistent,
            waited,
        )
    }

    fn poll_persistent_io(&mut self) -> Result<(), InferenceError> {
        loop {
            let result = match self.persistent_io.as_ref() {
                Some(io) => match io.results.try_recv() {
                    Ok(result) => result,
                    Err(TryRecvError::Empty) => return Ok(()),
                    Err(TryRecvError::Disconnected) => {
                        tracing::warn!(
                            "persistent cache I/O worker disconnected; disabling T3 and recomputing pending prompts"
                        );
                        self.disable_persistent_io();
                        return Ok(());
                    }
                },
                None => return Ok(()),
            };
            match result {
                PersistentIoResult::Store {
                    job_id,
                    epoch,
                    elapsed,
                    result,
                } => {
                    let current_epoch = self
                        .persistent_io
                        .as_ref()
                        .map(|io| io.epoch.load(Ordering::Acquire))
                        .unwrap_or(0);
                    if epoch != current_epoch {
                        continue;
                    }
                    update_ewma(&mut self.persistent_io_ns, elapsed.as_nanos() as f64);
                    if let Err(error) = result {
                        tracing::warn!(
                            job_id,
                            error = %error,
                            "optional encrypted T3 store failed; generation is unaffected"
                        );
                    }
                }
                PersistentIoResult::Lookup {
                    job_id,
                    epoch,
                    elapsed,
                    result,
                } => {
                    let current_epoch = self
                        .persistent_io
                        .as_ref()
                        .map(|io| io.epoch.load(Ordering::Acquire))
                        .unwrap_or(0);
                    if epoch != current_epoch {
                        continue;
                    }
                    let request_id = self.requests.iter().find_map(|(&request_id, state)| {
                        matches!(
                            state.maintenance,
                            CacheMaintenance::PersistentWaiting {
                                job_id: waiting,
                                epoch: waiting_epoch,
                                ..
                            } if waiting == job_id && waiting_epoch == epoch
                        )
                        .then_some(request_id)
                    });
                    let Some(request_id) = request_id else {
                        continue;
                    };
                    update_ewma(&mut self.persistent_io_ns, elapsed.as_nanos() as f64);
                    let Some(state) = self.requests.get_mut(&request_id) else {
                        continue;
                    };
                    let admitted_at = match state.maintenance {
                        CacheMaintenance::PersistentWaiting { admitted_at, .. } => admitted_at,
                        _ => continue,
                    };
                    match result {
                        Ok(PersistentLookup::Hit(record)) => {
                            let prepared = self
                                .warm_cache
                                .as_ref()
                                .ok_or_else(|| {
                                    InferenceError::worker_failed("T3 restore bridge disappeared")
                                })?
                                .prepare_persistent_restore(
                                    &self.cache_identity,
                                    &state.prompt_tokens,
                                    &record,
                                );
                            match prepared {
                                Ok(prepared) => {
                                    state.maintenance = CacheMaintenance::Restoring {
                                        prepared,
                                        source: RestoreSource::Persistent,
                                        admitted_at,
                                    };
                                }
                                Err(error) => {
                                    let mut state =
                                        self.requests.remove(&request_id).ok_or_else(|| {
                                            InferenceError::worker_failed(
                                                "T3 request state disappeared",
                                            )
                                        })?;
                                    self.release_state_resources(request_id, &mut state)?;
                                    self.ready_outputs.push_back(ContinuousStepOutput::Failed {
                                        request_id,
                                        error: InferenceError::worker_failed(format!(
                                            "authenticated T3 page layout is invalid: {error}"
                                        )),
                                    });
                                    continue;
                                }
                            }
                        }
                        Ok(
                            PersistentLookup::Disabled
                            | PersistentLookup::Recompute
                            | PersistentLookup::Miss,
                        ) => {
                            state.maintenance = CacheMaintenance::Recompute;
                        }
                        Err(error) => {
                            tracing::warn!(
                                request_id = ?request_id,
                                error = %error,
                                "encrypted T3 lookup discarded; recomputing prompt"
                            );
                            state.maintenance = CacheMaintenance::Recompute;
                        }
                    }
                    if !self.maintenance_queue.contains(&request_id) {
                        self.maintenance_queue.push_back(request_id);
                    }
                }
            }
        }
    }

    fn disable_persistent_io(&mut self) {
        self.persistent_io = None;
        let mut newly_actionable = Vec::new();
        for (&request_id, state) in &mut self.requests {
            state.persistent_cache_enabled = false;
            if let CacheMaintenance::PersistentWaiting { cancellation, .. } = &state.maintenance {
                cancellation.store(true, Ordering::Release);
            }
            if matches!(
                state.maintenance,
                CacheMaintenance::PersistentLookup { .. }
                    | CacheMaintenance::PersistentWaiting { .. }
            ) {
                state.maintenance = CacheMaintenance::Recompute;
                newly_actionable.push(request_id);
            }
        }
        for request_id in newly_actionable {
            if !self.maintenance_queue.contains(&request_id) {
                self.maintenance_queue.push_back(request_id);
            }
        }
    }

    fn outstanding_reserved_pages(&self) -> Result<u32, InferenceError> {
        let pool = self
            .engine
            .paged_kv_pool()
            .ok_or_else(|| InferenceError::worker_failed("paged KV pool disappeared"))?;
        self.requests.values().try_fold(0u32, |total, state| {
            if state.maintenance.is_awaiting_activation() {
                return total
                    .checked_add(state.reserved_private_pages)
                    .ok_or_else(|| InferenceError::worker_failed("KV reservation overflow"));
            }
            let chain = state.prompt_chain.ok_or_else(|| {
                InferenceError::worker_failed("active request lost its paged KV chain")
            })?;
            let allocated = u32::try_from(
                pool.view(chain)
                    .map_err(|error| {
                        InferenceError::worker_failed(format!(
                            "inspect active paged KV reservation: {error}"
                        ))
                    })?
                    .pages
                    .len(),
            )
            .map_err(|_| InferenceError::worker_failed("allocated page count exceeds u32"))?;
            let shared_hot = if state.cache_tier == CacheTier::Hot {
                state.matched_cache_tokens / APPLE_KV_PAGE_SIZE
            } else {
                0
            };
            total
                .checked_add(
                    state
                        .reserved_private_pages
                        .saturating_sub(allocated.saturating_sub(shared_hot)),
                )
                .ok_or_else(|| InferenceError::worker_failed("KV reservation overflow"))
        })
    }

    fn ensure_kv_admission_pages(&mut self, candidate: u32) -> Result<(), InferenceError> {
        let outstanding = self.outstanding_reserved_pages()?;
        let needed = outstanding
            .checked_add(candidate)
            .ok_or_else(|| InferenceError::worker_failed("KV admission reservation overflow"))?;
        let mut free = self
            .engine
            .paged_kv_stats()
            .ok_or_else(|| InferenceError::worker_failed("paged KV pool disappeared"))?
            .free_pages;
        if free < needed {
            if let Some(cache) = self.hot_cache.as_mut() {
                cache
                    .handle_memory_pressure(
                        self.engine.paged_kv_pool_mut().ok_or_else(|| {
                            InferenceError::worker_failed("paged KV pool disappeared")
                        })?,
                        MemoryPressure::Critical,
                    )
                    .map_err(|error| {
                        InferenceError::worker_failed(format!(
                            "evict T1 for request admission: {error}"
                        ))
                    })?;
            }
            free = self
                .engine
                .paged_kv_stats()
                .ok_or_else(|| InferenceError::worker_failed("paged KV pool disappeared"))?
                .free_pages;
        }
        if free < needed {
            return Err(InferenceError::worker_failed(format!(
                "paged KV admission requires {candidate} request pages plus {outstanding} reserved pages, but only {free} are free"
            )));
        }
        Ok(())
    }

    fn purge_unpinned_cache_state(
        &mut self,
        pressure: MemoryPressure,
    ) -> Result<(), InferenceError> {
        debug_assert_ne!(pressure, MemoryPressure::Normal);
        if let Some(cache) = self.hot_cache.as_mut() {
            cache
                .handle_memory_pressure(
                    self.engine.paged_kv_pool_mut().ok_or_else(|| {
                        InferenceError::worker_failed("paged KV pool disappeared")
                    })?,
                    pressure,
                )
                .map_err(|error| {
                    InferenceError::worker_failed(format!("evict unpinned T1 cache state: {error}"))
                })?;
        }
        if let Some(cache) = self.warm_cache.as_mut() {
            cache
                .handle_memory_pressure(
                    self.engine.paged_kv_pool_mut().ok_or_else(|| {
                        InferenceError::worker_failed("paged KV pool disappeared")
                    })?,
                    pressure,
                )
                .map_err(|error| {
                    InferenceError::worker_failed(format!("evict unpinned T2 cache state: {error}"))
                })?;
        }
        Ok(())
    }

    fn release_state_resources(
        &mut self,
        request_id: ReqId,
        state: &mut RequestState,
    ) -> Result<(), InferenceError> {
        let released_cache_attachment = state.cache_attachment.is_some();
        if let Some(attachment) = state.cache_attachment.as_ref() {
            let cache = match attachment.tier {
                AttachedCacheTier::T1Hot => self.hot_cache.as_mut().ok_or_else(|| {
                    InferenceError::worker_failed("T1 attachment cache disappeared")
                })?,
                AttachedCacheTier::T2Warm => self.warm_cache.as_mut().ok_or_else(|| {
                    InferenceError::worker_failed("T2 attachment cache disappeared")
                })?,
            };
            self.engine
                .release_cache_attachment(cache, attachment)
                .map_err(|error| {
                    InferenceError::worker_failed(format!("release T0 cache attachment: {error}"))
                })?;
            state.cache_attachment = None;
        } else if state.deferred_kv_release {
            state
                .prompt_chain
                .ok_or_else(|| InferenceError::worker_failed("deferred KV chain missing"))?;
            self.engine
                .release_deferred_paged_kv(request_id)
                .map_err(|error| {
                    InferenceError::worker_failed(format!("release deferred KV: {error}"))
                })?;
        }
        state.deferred_kv_release = false;
        if let Some(pressure) =
            released_attachment_resweep(released_cache_attachment, self.pressure_admission)
        {
            // The first pressure sweep intentionally preserved this T0 pin.
            // Re-sweep immediately after release so it cannot become an
            // unpinned T1/T2 resident while pressure remains active.
            if let Err(error) = self.purge_unpinned_cache_state(pressure) {
                tracing::warn!(
                    request_id = ?request_id,
                    error = %error,
                    "post-T0 pressure re-sweep failed; completed inference remains valid"
                );
            }
        }
        Ok(())
    }

    fn submit_until_full(&mut self) -> Result<(), InferenceError> {
        if maintenance_gate(self.has_pending_maintenance(), self.submitted.len())
            != MaintenanceGate::Refill
        {
            return Ok(());
        }
        let engine = &mut self.engine;
        refill_submission_ring(
            &mut self.submitted,
            METAL_PIPELINE_DEPTH,
            || -> Result<Option<SubmittedContinuousStep>, InferenceError> {
                let submitted_at = Instant::now();
                let submitted = engine.step_submit().map_err(|error| {
                    InferenceError::worker_failed(format!("launch continuous Metal step: {error}"))
                })?;
                let (prefill, scheduled_ids, prefill_context_lens, prefill_tokens) =
                    match submitted.plan() {
                        Some(BatchPlan::Prefill {
                            req_ids,
                            context_lens,
                            cu_seqlens_q,
                            ..
                        }) => (
                            true,
                            req_ids.clone(),
                            context_lens.clone(),
                            cu_seqlens_q.last().copied().unwrap_or(0),
                        ),
                        Some(BatchPlan::Decode { req_ids, .. }) => {
                            (false, req_ids.clone(), Vec::new(), 0)
                        }
                        Some(BatchPlan::Idle) | None => {
                            engine.collect_submitted(submitted).map_err(|error| {
                                InferenceError::worker_failed(format!(
                                    "collect idle continuous Metal step: {error}"
                                ))
                            })?;
                            return Ok(None);
                        }
                    };
                Ok(Some(SubmittedContinuousStep {
                    submitted,
                    submitted_at,
                    prefill,
                    scheduled_ids,
                    prefill_context_lens,
                    prefill_tokens,
                }))
            },
        )
    }

    fn refill_after_outputs(
        &mut self,
        outputs: Vec<ContinuousStepOutput>,
    ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
        match self.submit_until_full() {
            Ok(()) => Ok(outputs),
            Err(error) if outputs.is_empty() => Err(error),
            Err(error) => {
                // Tokens already completed before the later launch failed.
                // Preserve their ordering and surface the launch failure on
                // the next worker turn.
                if self.deferred_error.is_none() {
                    self.deferred_error = Some(error);
                }
                Ok(outputs)
            }
        }
    }

    fn collect_oldest(&mut self) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
        let Some(step) = self.submitted.pop_front() else {
            return Ok(Vec::new());
        };
        let decoded = self
            .engine
            .collect_submitted(step.submitted)
            .map_err(|error| {
                InferenceError::worker_failed(format!("collect continuous Metal step: {error}"))
            })?;
        let elapsed = step.submitted_at.elapsed();
        if step.prefill && step.prefill_tokens > 0 {
            update_ewma(
                &mut self.prefill_ns_per_token,
                elapsed.as_nanos() as f64 / f64::from(step.prefill_tokens),
            );
        }
        let batch_size = u32::try_from(step.scheduled_ids.len()).unwrap_or(u32::MAX);
        for request_id in &step.scheduled_ids {
            if let Some(state) = self.requests.get_mut(request_id) {
                state.max_batch_size = state.max_batch_size.max(batch_size);
                if step.prefill {
                    state.prefill_time += elapsed;
                } else {
                    state.decode_time += elapsed;
                }
            }
        }
        if step.prefill {
            for (&request_id, &context_len) in
                step.scheduled_ids.iter().zip(&step.prefill_context_lens)
            {
                let completed = self.requests.get(&request_id).is_some_and(|state| {
                    context_len as usize == state.prompt_tokens.len()
                        && self.engine.scheduler.request_is_alive(request_id)
                });
                if completed {
                    let prompt = self
                        .requests
                        .get(&request_id)
                        .ok_or_else(|| {
                            InferenceError::worker_failed(
                                "completed prefill request state disappeared",
                            )
                        })?
                        .prompt_tokens
                        .clone();
                    self.promote_completed_prompt(request_id, &prompt)?;
                }
            }
        }
        let mut outputs = Vec::with_capacity(decoded.len());
        for output in decoded {
            let sampled = output.new_token.raw();
            let eos = self.eos_token_ids.contains(&sampled);
            let warm_capture_available = self.warm_cache.is_some();
            let (delta, queued_promotion) = {
                let state = self.requests.get_mut(&output.req_id).ok_or_else(|| {
                    InferenceError::worker_failed(
                        "Metal scheduler returned token for unknown request",
                    )
                })?;
                state.emitted_tokens = state.emitted_tokens.saturating_add(1);
                let reached_length = state.emitted_tokens >= state.max_output_tokens;
                let delta = state
                    .decoder
                    .step(&self.tokenizer, sampled)
                    .map_err(|error| {
                        InferenceError::worker_failed(format!(
                            "incrementally decode token {sampled}: {error}"
                        ))
                    })?
                    .map(Arc::<str>::from);
                if eos {
                    state.terminal = Some(FinishReason::EndOfSequence);
                } else if reached_length || output.finished {
                    state.terminal = Some(FinishReason::Length);
                }
                let queued_promotion = state.terminal.is_some()
                    && state.cache_enabled
                    && warm_capture_available
                    && !matches!(state.cache_tier, CacheTier::Warm | CacheTier::Persistent)
                    && reusable_prompt_tokens(state.prompt_tokens.len()) > 0;
                if queued_promotion {
                    state.maintenance = CacheMaintenance::Promoting;
                }
                (delta, queued_promotion)
            };
            if queued_promotion && !self.maintenance_queue.contains(&output.req_id) {
                self.maintenance_queue.push_back(output.req_id);
            }
            if eos {
                let _ = self.engine.finish_request(output.req_id);
            }
            outputs.push(ContinuousStepOutput::Token {
                request_id: output.req_id,
                token_id: TokenId(sampled),
                text: delta,
            });
        }
        Ok(outputs)
    }

    fn take_ready_outputs(&mut self, active_ids: &HashSet<ReqId>) -> Vec<ContinuousStepOutput> {
        let mut ready = Vec::new();
        for _ in 0..self.ready_outputs.len() {
            let Some(output) = self.ready_outputs.pop_front() else {
                break;
            };
            let request_id = output_request_id(&output);
            if active_ids.contains(&request_id) {
                ready.push(output);
            } else {
                self.ready_outputs.push_back(output);
            }
        }
        ready
    }

    fn submission_contains(&self, request_id: ReqId) -> bool {
        self.submitted
            .iter()
            .any(|step| step.scheduled_ids.contains(&request_id))
    }

    fn has_pending_maintenance(&self) -> bool {
        !self.maintenance_queue.is_empty()
    }
}

impl ContinuousInferenceWorker for AppleMetalContinuousWorker {
    fn admission_limit(&self, configured_maximum: usize) -> usize {
        self.pressure_admission.admission_limit(configured_maximum)
    }

    fn admit(
        &mut self,
        request_id: ReqId,
        request: &GenerateRequest,
        cancellation: &AtomicBool,
    ) -> Result<(), InferenceError> {
        validate_token_budget(
            request.prompt_tokens.len(),
            request.max_output_tokens as usize,
            self.max_supported_total_tokens,
        )
        .map_err(InferenceError::worker_failed)?;
        if !request.sampling.is_greedy() {
            return Err(InferenceError::InvalidRequest {
                field: "sampling",
                reason: "the Metal continuous backend currently supports greedy sampling only",
            });
        }
        if self.requests.contains_key(&request_id) {
            return Err(InferenceError::worker_failed(
                "duplicate continuous Metal request id",
            ));
        }
        let cache_admission = cache_admission_mode(
            request.cache_policy,
            self.pressure_admission,
            self.hot_cache.is_some(),
            self.warm_cache.is_some(),
            self.persistent_io.is_some(),
        );
        let persistent_cache_enabled = cache_admission.persistent_cache;
        let cache_enabled = cache_admission.optional_cache_work;
        let mut attachment = if cache_admission.hot_attachment {
            if let Some(cache) = self.hot_cache.as_mut() {
                cache
                    .attach(
                        self.engine.paged_kv_pool_mut().ok_or_else(|| {
                            InferenceError::worker_failed("paged KV pool disappeared")
                        })?,
                        &mut self.page_io,
                        request_id,
                        &self.cache_identity,
                        &request.prompt_tokens,
                        || cancellation.load(Ordering::Acquire),
                    )
                    .map_err(|error| {
                        InferenceError::worker_failed(format!(
                            "attach exact T1 prompt prefix: {error}"
                        ))
                    })?
            } else {
                None
            }
        } else {
            None
        };

        let prepared_warm = if attachment.is_none()
            && cache_admission.optional_cache_work
            && self.warm_cache.is_some()
            && self.warm_restore_predicted_faster(request.prompt_tokens.len())
        {
            match self
                .warm_cache
                .as_mut()
                .ok_or_else(|| InferenceError::worker_failed("warm cache disappeared"))?
                .prepare_warm_restore(&self.cache_identity, &request.prompt_tokens)
            {
                Ok(prepared) => prepared,
                Err(error) => {
                    tracing::warn!(
                        request_id = ?request_id,
                        error = %error,
                        "exact T2 lookup skipped; admitting as a cache miss"
                    );
                    None
                }
            }
        } else {
            None
        };

        let max_resident_tokens = request
            .prompt_tokens
            .len()
            .checked_add(request.max_output_tokens.saturating_sub(1) as usize)
            .ok_or_else(|| InferenceError::worker_failed("request token budget overflow"))?;
        let total_pages = max_resident_tokens.div_ceil(APPLE_KV_PAGE_SIZE as usize);
        let matched_pages = attachment.as_ref().map_or(0, |value| {
            value.matched_tokens as usize / APPLE_KV_PAGE_SIZE as usize
        });
        let needed_pages = u32::try_from(total_pages.saturating_sub(matched_pages))
            .map_err(|_| InferenceError::worker_failed("request page count exceeds u32"))?;
        if let Err(error) = self.ensure_kv_admission_pages(needed_pages) {
            if let Some(value) = attachment.take() {
                self.release_unadmitted_attachment(&value)?;
            }
            return Err(error);
        }

        let (cache_tier, matched_cache_tokens) = attachment
            .as_ref()
            .map(|value| {
                (
                    match value.tier {
                        AttachedCacheTier::T1Hot => CacheTier::Hot,
                        AttachedCacheTier::T2Warm => CacheTier::Warm,
                    },
                    value.matched_tokens,
                )
            })
            .unwrap_or((CacheTier::None, 0));
        let maintenance = if let Some(prepared) = prepared_warm {
            CacheMaintenance::Restoring {
                prepared,
                source: RestoreSource::Warm,
                admitted_at: Instant::now(),
            }
        } else if persistent_cache_enabled {
            CacheMaintenance::PersistentLookup {
                admitted_at: Instant::now(),
            }
        } else {
            CacheMaintenance::Active
        };
        let is_restoring = !maintenance.is_active();
        let mut state = RequestState {
            prompt_tokens: request.prompt_tokens.clone(),
            prompt_chain: None,
            priority: request.priority,
            maintenance,
            decoder: IncrementalTextDecoder::default(),
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            queue_time: None,
            max_batch_size: 1,
            terminal: None,
            backpressured: false,
            cache_attachment: None,
            cache_tier,
            matched_cache_tokens,
            reserved_private_pages: needed_pages,
            cache_enabled,
            persistent_cache_enabled,
            cache_promoted: false,
            deferred_kv_release: false,
            emitted_tokens: 0,
            max_output_tokens: request.max_output_tokens,
        };
        if is_restoring {
            self.requests.insert(request_id, state);
            self.maintenance_queue.push_back(request_id);
            return Ok(());
        }
        self.activate_request_state(request_id, &mut state, attachment)?;
        self.requests.insert(request_id, state);
        Ok(())
    }

    fn step(
        &mut self,
        active: &[ActiveRequest<'_>],
    ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
        let active_ids: HashSet<_> = active.iter().map(|view| view.request_id).collect();
        for view in active {
            if let Some(state) = self.requests.get_mut(&view.request_id) {
                state.queue_time.get_or_insert(view.queue_time);
            }
        }
        self.poll_persistent_io()?;
        let eligibility_changes: Vec<_> = self
            .requests
            .iter()
            .filter_map(|(&request_id, state)| {
                if state.terminal.is_some() || !state.maintenance.is_active() {
                    return None;
                }
                let backpressured = !active_ids.contains(&request_id);
                (backpressured != state.backpressured).then_some((request_id, backpressured))
            })
            .collect();
        for (request_id, backpressured) in eligibility_changes {
            self.engine
                .set_request_backpressured(request_id, backpressured)
                .map_err(|error| {
                    InferenceError::worker_failed(format!(
                        "update Metal request backpressure: {error}"
                    ))
                })?;
            if let Some(state) = self.requests.get_mut(&request_id) {
                state.backpressured = backpressured;
            }
        }
        if let Some(error) = self.deferred_error.take() {
            return Err(error);
        }
        let ready = self.take_ready_outputs(&active_ids);
        if !ready.is_empty() {
            return self.refill_after_outputs(ready);
        }

        match maintenance_gate(self.has_pending_maintenance(), self.submitted.len()) {
            MaintenanceGate::DrainSubmitted => {
                // Idle-only page I/O is never attempted while a submitted
                // step owns the Metal arena. Do not refill the reclaimed slot.
                return self.collect_oldest();
            }
            MaintenanceGate::RunIdle => {
                if let Some(output) = self.process_one_maintenance(active)? {
                    if active_ids.contains(&output_request_id(&output)) {
                        return Ok(vec![output]);
                    }
                    self.ready_outputs.push_back(output);
                }
                if self.has_pending_maintenance() {
                    return Ok(Vec::new());
                }
            }
            MaintenanceGate::Refill => {}
        }

        let terminal_ids: Vec<_> = active
            .iter()
            .filter_map(|view| {
                self.requests.get(&view.request_id).and_then(|state| {
                    (state.maintenance.is_active())
                        .then_some(state.terminal)
                        .flatten()
                        .map(|reason| (view.request_id, reason))
                })
            })
            .collect();
        if !terminal_ids.is_empty() {
            let mut finished = Vec::with_capacity(terminal_ids.len());
            for (request_id, finish_reason) in terminal_ids {
                let Some(state) = self.requests.remove(&request_id) else {
                    continue;
                };
                finished.push(self.finish_request_state(request_id, finish_reason, state)?);
            }
            return self.refill_after_outputs(finished);
        }
        self.submit_until_full()?;
        if self.submitted.is_empty() {
            return Ok(Vec::new());
        }
        let outputs = self.collect_oldest()?;
        self.refill_after_outputs(outputs)
    }

    fn abort(&mut self, request_id: ReqId) {
        if let Some(RequestState {
            maintenance: CacheMaintenance::PersistentWaiting { cancellation, .. },
            ..
        }) = self.requests.get(&request_id)
        {
            cancellation.store(true, Ordering::Release);
        }
        let _ = self.engine.cancel_request(request_id);
        self.maintenance_queue
            .retain(|pending| *pending != request_id);
        self.ready_outputs
            .retain(|output| output_request_id(output) != request_id);
        while self.submission_contains(request_id) {
            match self.collect_oldest() {
                Ok(outputs) => self.ready_outputs.extend(
                    outputs
                        .into_iter()
                        .filter(|output| output_request_id(output) != request_id),
                ),
                Err(error) => {
                    if self.deferred_error.is_none() {
                        self.deferred_error = Some(error);
                    }
                }
            }
        }
        if let Some(mut state) = self.requests.remove(&request_id) {
            if let Err(error) = self.release_state_resources(request_id, &mut state) {
                if self.deferred_error.is_none() {
                    self.deferred_error = Some(error);
                }
            }
        }
    }

    fn handle_memory_pressure(&mut self, pressure: MemoryPressure) -> Result<(), InferenceError> {
        let requested_admission = PressureAdmission::from_pressure(pressure);
        if requested_admission > self.pressure_admission {
            // Escalation is fail-safe even if a later eviction reports an
            // error. De-escalation is committed only after successful handling.
            self.pressure_admission = requested_admission;
        }
        if pressure == MemoryPressure::Normal {
            // Admission policy is snapshotted per request: work admitted while
            // throttled remains cache-passive, while subsequent requests use
            // the recovered full policy.
            self.pressure_admission = PressureAdmission::Nominal;
            return Ok(());
        }
        if let Some(io) = self.persistent_io.as_ref() {
            io.invalidate();
        }
        let mut newly_actionable = Vec::new();
        for (&request_id, state) in &mut self.requests {
            // Existing T0 attachments remain in `state.cache_attachment` and
            // retain their request-owned pages. Only optional future reuse,
            // restore, capture, and persistent publication are disabled.
            if throttle_existing_request_cache(state) {
                newly_actionable.push(request_id);
            }
        }
        for request_id in newly_actionable {
            if !self.maintenance_queue.contains(&request_id) {
                self.maintenance_queue.push_back(request_id);
            }
        }
        self.purge_unpinned_cache_state(pressure)?;
        self.pressure_admission = requested_admission;
        Ok(())
    }
}

impl AppleMetalContinuousWorker {
    fn release_unadmitted_attachment(
        &mut self,
        attachment: &CachePrefixAttachment,
    ) -> Result<(), InferenceError> {
        let cache = match attachment.tier {
            AttachedCacheTier::T1Hot => self.hot_cache.as_mut(),
            AttachedCacheTier::T2Warm => self.warm_cache.as_mut(),
        }
        .ok_or_else(|| InferenceError::worker_failed("attachment cache disappeared"))?;
        let pool = self
            .engine
            .paged_kv_pool_mut()
            .ok_or_else(|| InferenceError::worker_failed("paged KV pool disappeared"))?;
        cache.release_attachment(pool, attachment).map_err(|error| {
            InferenceError::worker_failed(format!("release unadmitted cache attachment: {error}"))
        })
    }
}

impl Drop for AppleMetalContinuousWorker {
    fn drop(&mut self) {
        while let Some(step) = self.submitted.pop_front() {
            let _ = self.engine.collect_submitted(step.submitted);
        }
    }
}

fn output_request_id(output: &ContinuousStepOutput) -> ReqId {
    match output {
        ContinuousStepOutput::Token { request_id, .. }
        | ContinuousStepOutput::Finished { request_id, .. }
        | ContinuousStepOutput::Failed { request_id, .. } => *request_id,
    }
}

fn update_ewma(current: &mut Option<f64>, observation: f64) {
    if !observation.is_finite() || observation <= 0.0 {
        return;
    }
    *current = Some(match *current {
        Some(previous) => previous * 0.8 + observation * 0.2,
        None => observation,
    });
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum MaintenanceGate {
    Refill,
    DrainSubmitted,
    RunIdle,
}

fn maintenance_gate(has_pending: bool, submitted: usize) -> MaintenanceGate {
    if !has_pending {
        MaintenanceGate::Refill
    } else if submitted > 0 {
        MaintenanceGate::DrainSubmitted
    } else {
        MaintenanceGate::RunIdle
    }
}

fn pop_next_pending<T: Copy>(
    queue: &mut VecDeque<T>,
    mut is_pending: impl FnMut(T) -> bool,
) -> Option<T> {
    while let Some(candidate) = queue.pop_front() {
        if is_pending(candidate) {
            return Some(candidate);
        }
    }
    None
}

fn warm_restore_beats_recompute(
    matched_tokens: u32,
    page_bytes: usize,
    prefill_ns_per_token: f64,
    copy_bytes_per_ns: f64,
    waited: Duration,
) -> bool {
    if matched_tokens == 0
        || matched_tokens % APPLE_KV_PAGE_SIZE != 0
        || page_bytes == 0
        || !prefill_ns_per_token.is_finite()
        || prefill_ns_per_token <= 0.0
        || !copy_bytes_per_ns.is_finite()
        || copy_bytes_per_ns <= 0.0
    {
        return false;
    }
    let pages = matched_tokens as usize / PROMPT_CACHE_PAGE_TOKENS;
    let restore_ns = pages as f64 * page_bytes as f64 / copy_bytes_per_ns * 1.25;
    let recompute_ns = matched_tokens as f64 * prefill_ns_per_token;
    restore_ns + (waited.as_nanos() as f64) < recompute_ns
}

fn persistent_restore_cost_estimate(
    prompt_len: usize,
    page_bytes: usize,
    prefill_ns_per_token: f64,
    copy_bytes_per_ns: f64,
    persistent_io_ns: f64,
    waited: Duration,
) -> Option<RestoreCostEstimate> {
    let reusable = reusable_prompt_tokens(prompt_len);
    if reusable == 0
        || page_bytes == 0
        || !prefill_ns_per_token.is_finite()
        || prefill_ns_per_token <= 0.0
        || !copy_bytes_per_ns.is_finite()
        || copy_bytes_per_ns <= 0.0
        || !persistent_io_ns.is_finite()
        || persistent_io_ns <= 0.0
    {
        return None;
    }
    let pages = reusable / PROMPT_CACHE_PAGE_TOKENS;
    let page_restore = pages as f64 * page_bytes as f64 / copy_bytes_per_ns * 1.25;
    let restore = persistent_io_ns + page_restore + waited.as_nanos() as f64;
    let recompute = reusable as f64 * prefill_ns_per_token;
    Some(RestoreCostEstimate {
        measured_restore_ns: restore.min(u64::MAX as f64) as u64,
        measured_recompute_ns: recompute.min(u64::MAX as f64) as u64,
    })
}

fn refill_submission_ring<T, E>(
    ring: &mut VecDeque<T>,
    capacity: usize,
    mut submit: impl FnMut() -> Result<Option<T>, E>,
) -> Result<(), E> {
    while ring.len() < capacity {
        let Some(submission) = submit()? else {
            break;
        };
        ring.push_back(submission);
    }
    Ok(())
}

fn make_cache(
    hot_bytes: usize,
    warm_bytes: usize,
    layout: [u8; 32],
    page_bytes: usize,
    label: &str,
) -> Result<Option<PagedPromptCache>, String> {
    if hot_bytes == 0 && warm_bytes == 0 {
        return Ok(None);
    }
    PagedPromptCache::new(
        PromptCacheConfig {
            hot_bytes,
            warm_bytes,
            protected_fraction_percent: 80,
            frequency_aging_interval: 4096,
        },
        layout,
        page_bytes,
    )
    .map(Some)
    .map_err(|error| format!("configure {label} prompt cache: {error}"))
}

fn validate_token_budget(
    prompt_len: usize,
    max_new_tokens: usize,
    max_total_tokens: usize,
) -> Result<(), String> {
    if prompt_len == 0 {
        return Err("Metal prompt must contain at least one token".to_owned());
    }
    if max_new_tokens == 0 {
        return Err("Metal max output tokens must be positive".to_owned());
    }
    let total = prompt_len
        .checked_add(max_new_tokens)
        .ok_or_else(|| "Metal token budget overflow".to_owned())?;
    if u32::try_from(total).is_err() {
        return Err(format!("Metal token positions exceed u32: {total}"));
    }
    if total > max_total_tokens {
        return Err(format!(
            "prompt + output must be <= {max_total_tokens}; got {prompt_len} + {max_new_tokens}"
        ));
    }
    Ok(())
}

fn build_runtime_plan(
    model_dir: PathBuf,
    arch: &rvllm_loader::gemma4_arch::Gemma4Arch,
    rollout_tokens: usize,
    model_layout_hash: [u8; 32],
) -> Result<AppleRuntimePlan, String> {
    Ok(AppleRuntimePlan {
        target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple GPU", 1),
        mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
        rollout_bucket: None,
        rollout_tokens: u32::try_from(rollout_tokens)
            .map_err(|_| "rollout token count exceeds u32".to_owned())?,
        private_ane_opt_in: false,
        strict_ane: false,
        ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
        ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
        ane_hidden_size: arch.hidden_size,
        ane_intermediate_size: arch.intermediate_size,
        ane_num_layers: arch.num_hidden_layers,
        model_layout_hash,
        weights_path: Some(model_dir),
    })
}

fn model_layout_fingerprint(model_dir: &Path) -> Result<[u8; 32], String> {
    let mut hasher = Sha256::new();
    hasher.update(
        b"rvllm.apple.paged-kv-layout.v2\0page-size=32\0attention-window=per-layer-exact\0",
    );
    for name in ["config.json", "model.safetensors.index.json"] {
        let path = model_dir.join(name);
        if path.is_file() {
            hasher.update(name.as_bytes());
            hasher.update(
                std::fs::read(&path)
                    .map_err(|error| format!("read model layout {}: {error}", path.display()))?,
            );
        }
    }
    let mut tensors = std::fs::read_dir(model_dir)
        .map_err(|error| format!("list model directory {}: {error}", model_dir.display()))?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            if !name.ends_with(".safetensors") {
                return None;
            }
            Some((name, entry.metadata().ok()?.len()))
        })
        .collect::<Vec<_>>();
    tensors.sort_unstable();
    for (name, bytes) in tensors {
        hasher.update(name.as_bytes());
        hasher.update(bytes.to_le_bytes());
    }
    Ok(hasher.finalize().into())
}

fn cache_identity(
    model_dir: &Path,
    kv_layout: [u8; 32],
    numeric_path: [u8; 32],
    namespace: &str,
) -> Result<CacheIdentity, String> {
    let namespace = CacheNamespace::new(namespace)
        .map_err(|error| format!("create cache namespace: {error}"))?;
    let mut model = Sha256::new();
    model.update(b"rvllm.apple.weight-identity.v1\0");
    for name in ["config.json", "model.safetensors.index.json"] {
        let path = model_dir.join(name);
        if path.is_file() {
            model.update(name.as_bytes());
            model.update(
                std::fs::read(&path)
                    .map_err(|error| format!("read cache identity {}: {error}", path.display()))?,
            );
        }
    }
    let mut weights = std::fs::read_dir(model_dir)
        .map_err(|error| format!("list model directory {}: {error}", model_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".safetensors"))
        })
        .collect::<Vec<_>>();
    weights.sort_unstable();
    for path in weights {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("non-UTF-8 weight path: {}", path.display()))?;
        let symlink = std::fs::symlink_metadata(&path)
            .map_err(|error| format!("stat weight {}: {error}", path.display()))?;
        let metadata = std::fs::metadata(&path)
            .map_err(|error| format!("stat weight target {}: {error}", path.display()))?;
        model.update(name.as_bytes());
        model.update(metadata.len().to_le_bytes());
        if symlink.file_type().is_symlink() {
            let target = std::fs::read_link(&path)
                .map_err(|error| format!("read weight link {}: {error}", path.display()))?;
            model.update(target.as_os_str().as_encoded_bytes());
        } else {
            let canonical = std::fs::canonicalize(&path)
                .map_err(|error| format!("canonicalize weight {}: {error}", path.display()))?;
            model.update(canonical.as_os_str().as_encoded_bytes());
            model.update(metadata.dev().to_le_bytes());
            model.update(metadata.ino().to_le_bytes());
            model.update(metadata.ctime().to_le_bytes());
            model.update(metadata.ctime_nsec().to_le_bytes());
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .unwrap_or(Duration::ZERO);
            model.update(modified.as_secs().to_le_bytes());
            model.update(modified.subsec_nanos().to_le_bytes());
        }
    }
    let tokenizer_path = model_dir.join("tokenizer.json");
    let mut tokenizer = Sha256::new();
    tokenizer.update(
        b"rvllm.apple.tokenizer-semantics.v1\0bos=2\0add-special=false\0chat-template=none\0",
    );
    tokenizer.update(std::fs::read(&tokenizer_path).map_err(|error| {
        format!(
            "read tokenizer identity {}: {error}",
            tokenizer_path.display()
        )
    })?);
    Ok(CacheIdentity {
        namespace,
        model: model.finalize().into(),
        tokenizer: tokenizer.finalize().into(),
        adapter: None,
        kv_layout,
        numeric_path,
        format_version: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        cache_admission_mode, cancel_optional_cache_work, create_embedded_apple_metal_worker,
        maintenance_gate, persistent_restore_cost_estimate, pop_next_pending,
        refill_submission_ring, released_attachment_resweep, throttle_existing_request_cache_parts,
        validate_token_budget, warm_restore_beats_recompute, CacheMaintenance, MaintenanceGate,
        PressureAdmission,
    };
    use crate::{AppleEngineConfig, BackendPolicy, CachePolicy, InferenceError, MemoryPressure};
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn token_budget_fails_closed() {
        assert!(validate_token_budget(0, 1, 8).is_err());
        assert!(validate_token_budget(1, 0, 8).is_err());
        assert!(validate_token_budget(7, 2, 8).is_err());
        assert!(validate_token_budget(7, 1, 8).is_ok());
    }

    #[test]
    fn pressure_admission_limits_are_level_specific_and_explicitly_recoverable() {
        assert_eq!(PressureAdmission::Nominal.admission_limit(7), 7);
        assert_eq!(PressureAdmission::Warning.admission_limit(7), 4);
        assert_eq!(PressureAdmission::Warning.admission_limit(8), 4);
        assert_eq!(PressureAdmission::Critical.admission_limit(8), 1);

        assert_eq!(
            PressureAdmission::from_pressure(MemoryPressure::Normal),
            PressureAdmission::Nominal
        );
        assert_eq!(
            PressureAdmission::from_pressure(MemoryPressure::Warning),
            PressureAdmission::Warning
        );
        assert_eq!(
            PressureAdmission::from_pressure(MemoryPressure::Critical),
            PressureAdmission::Critical
        );
    }

    #[test]
    fn new_cache_admission_degrades_from_t1_only_to_fully_disabled() {
        let warning = cache_admission_mode(
            CachePolicy::PersistentEncrypted,
            PressureAdmission::Warning,
            true,
            true,
            true,
        );
        assert!(warning.hot_attachment);
        assert!(!warning.optional_cache_work);
        assert!(!warning.persistent_cache);

        let critical = cache_admission_mode(
            CachePolicy::PersistentEncrypted,
            PressureAdmission::Critical,
            true,
            true,
            true,
        );
        assert!(!critical.hot_attachment);
        assert!(!critical.optional_cache_work);
        assert!(!critical.persistent_cache);

        let recovered = cache_admission_mode(
            CachePolicy::PersistentEncrypted,
            PressureAdmission::Nominal,
            true,
            true,
            true,
        );
        assert!(recovered.hot_attachment);
        assert!(recovered.optional_cache_work);
        assert!(recovered.persistent_cache);
    }

    #[test]
    fn pressure_keeps_active_inference_and_cancels_optional_maintenance() {
        let mut cache_enabled = true;
        let mut persistent_cache_enabled = true;
        let mut active = CacheMaintenance::Active;
        let actionable = throttle_existing_request_cache_parts(
            &mut cache_enabled,
            &mut persistent_cache_enabled,
            &mut active,
        );
        assert!(matches!(active, CacheMaintenance::Active));
        assert!(!actionable, "active T0 inference must remain runnable");
        assert!(!cache_enabled);
        assert!(!persistent_cache_enabled);

        let (promotion, actionable) = cancel_optional_cache_work(CacheMaintenance::Promoting);
        assert!(matches!(promotion, CacheMaintenance::SkipPromotion));
        assert!(actionable);

        let cancellation = Arc::new(AtomicBool::new(false));
        let (waiting, actionable) =
            cancel_optional_cache_work(CacheMaintenance::PersistentWaiting {
                admitted_at: Instant::now(),
                job_id: 7,
                epoch: 3,
                cancellation: Arc::clone(&cancellation),
            });
        assert!(matches!(waiting, CacheMaintenance::Recompute));
        assert!(actionable);
        assert!(cancellation.load(Ordering::Acquire));
    }

    #[test]
    fn released_t0_attachment_is_reswept_while_pressure_remains_active() {
        assert_eq!(
            released_attachment_resweep(true, PressureAdmission::Warning),
            Some(MemoryPressure::Warning)
        );
        assert_eq!(
            released_attachment_resweep(true, PressureAdmission::Critical),
            Some(MemoryPressure::Critical)
        );
        assert_eq!(
            released_attachment_resweep(true, PressureAdmission::Nominal),
            None
        );
        assert_eq!(
            released_attachment_resweep(false, PressureAdmission::Critical),
            None
        );
    }

    #[test]
    fn embedded_policy_and_package_requirements_fail_before_backend_creation() {
        let mut config = AppleEngineConfig::default();
        config.backend_policy = BackendPolicy::CoreMlOnly;
        assert!(matches!(
            create_embedded_apple_metal_worker(&config),
            Err(InferenceError::InvalidConfig {
                field: "backend_policy",
                ..
            })
        ));

        config.backend_policy = BackendPolicy::MetalOnly;
        config.cache_policy = CachePolicy::PersistentEncrypted;
        config.persistent_cache_consent = true;
        config.persistent_cache_bytes = 1;
        assert!(matches!(
            create_embedded_apple_metal_worker(&config),
            Err(InferenceError::InvalidConfig {
                field: "persistent_cache",
                ..
            })
        ));

        config.cache_policy = CachePolicy::MemoryOnly;
        config.persistent_cache_consent = false;
        config.persistent_cache_bytes = 0;
        assert!(matches!(
            create_embedded_apple_metal_worker(&config),
            Err(InferenceError::InvalidConfig {
                field: "model_package_path",
                ..
            })
        ));
    }

    #[test]
    fn submission_ring_refills_each_reclaimed_slot_without_reordering() {
        let mut ring = VecDeque::from([1, 2, 3]);
        let mut pending = VecDeque::from([4, 5]);

        assert_eq!(ring.pop_front(), Some(1));
        refill_submission_ring(&mut ring, 3, || Ok::<_, ()>(pending.pop_front())).unwrap();
        assert_eq!(ring, VecDeque::from([2, 3, 4]));
        assert_eq!(pending, VecDeque::from([5]));

        assert_eq!(ring.pop_front(), Some(2));
        refill_submission_ring(&mut ring, 3, || Ok::<_, ()>(pending.pop_front())).unwrap();
        assert_eq!(ring, VecDeque::from([3, 4, 5]));
        assert!(pending.is_empty());
    }

    #[test]
    fn submission_ring_stops_at_idle_and_preserves_entries_on_error() {
        let mut ring = VecDeque::from([10]);
        let mut idle_calls = 0;
        refill_submission_ring(&mut ring, 3, || {
            idle_calls += 1;
            Ok::<Option<i32>, ()>(None)
        })
        .unwrap();
        assert_eq!(idle_calls, 1);
        assert_eq!(ring, VecDeque::from([10]));

        let error = refill_submission_ring(&mut ring, 3, || Err::<Option<i32>, _>("launch"));
        assert_eq!(error, Err("launch"));
        assert_eq!(ring, VecDeque::from([10]));
    }

    #[test]
    fn pending_page_io_drains_the_ring_before_running_idle_maintenance() {
        assert_eq!(
            maintenance_gate(false, 3),
            MaintenanceGate::Refill,
            "ordinary work may keep the submission ring full"
        );
        assert_eq!(maintenance_gate(true, 3), MaintenanceGate::DrainSubmitted);
        assert_eq!(maintenance_gate(true, 1), MaintenanceGate::DrainSubmitted);
        assert_eq!(maintenance_gate(true, 0), MaintenanceGate::RunIdle);
    }

    #[test]
    fn maintenance_queue_is_fifo_and_discards_stale_ownership() {
        let mut queue = VecDeque::from([1, 2, 3, 4]);
        let live = [2, 3, 4];
        assert_eq!(
            pop_next_pending(&mut queue, |candidate| live.contains(&candidate)),
            Some(2)
        );
        assert_eq!(
            pop_next_pending(&mut queue, |candidate| live.contains(&candidate)),
            Some(3)
        );
        assert_eq!(
            pop_next_pending(&mut queue, |candidate| live.contains(&candidate)),
            Some(4)
        );
        assert_eq!(
            pop_next_pending(&mut queue, |candidate| live.contains(&candidate)),
            None
        );
    }

    #[test]
    fn queued_warm_restore_recomputes_once_wait_cost_erases_the_win() {
        let matched_tokens = 64;
        let page_bytes = 100;
        let prefill_ns_per_token = 10.0;
        let copy_bytes_per_ns = 1.0;

        assert!(warm_restore_beats_recompute(
            matched_tokens,
            page_bytes,
            prefill_ns_per_token,
            copy_bytes_per_ns,
            Duration::from_nanos(300),
        ));
        assert!(!warm_restore_beats_recompute(
            matched_tokens,
            page_bytes,
            prefill_ns_per_token,
            copy_bytes_per_ns,
            Duration::from_nanos(400),
        ));
        assert!(!warm_restore_beats_recompute(
            33,
            page_bytes,
            prefill_ns_per_token,
            copy_bytes_per_ns,
            Duration::ZERO,
        ));
        assert!(!warm_restore_beats_recompute(
            matched_tokens,
            page_bytes,
            prefill_ns_per_token,
            0.0,
            Duration::ZERO,
        ));
    }

    #[test]
    fn persistent_restore_requires_measured_io_and_strict_total_cost_win() {
        assert!(
            persistent_restore_cost_estimate(65, 100, 100.0, 10.0, 1_000.0, Duration::ZERO,)
                .unwrap()
                .should_restore()
        );
        assert!(
            !persistent_restore_cost_estimate(65, 100, 100.0, 10.0, 6_500.0, Duration::ZERO,)
                .unwrap()
                .should_restore()
        );
        assert!(
            persistent_restore_cost_estimate(65, 100, 100.0, 10.0, 0.0, Duration::ZERO,).is_none()
        );
    }
}
