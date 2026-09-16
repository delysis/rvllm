//! Public, backend-neutral asynchronous request API.
//!
//! The accelerator backend is created inside, and remains owned by, one worker
//! thread. This matters for Metal and Core ML objects whose safe lifetime and
//! command ordering are tied to their construction thread. Public handles are
//! cheap clones around bounded standard-library channels.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{
    self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError, TrySendError,
};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rvllm_core::{ReqId, TokenId};
use rvllm_sampling::SamplingParams;

use crate::persistent_prompt_cache::PersistentCacheHostConfig;
use crate::prompt_cache::MemoryPressure;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BackendPolicy {
    /// Calibrated admission chooses Metal or public Core ML.
    Automatic,
    MetalOnly,
    /// Explicit private research route; never selected by Automatic.
    MetalPrefillAneDecode,
    CoreMlPreferred,
    CoreMlOnly,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CachePolicy {
    Disabled,
    /// T0/T1 and, when configured, lossless T2 memory.
    MemoryOnly,
    /// Encrypted persistent records may be used with explicit host consent.
    PersistentEncrypted,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WorkloadProfile {
    Interactive,
    Balanced,
    Throughput,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryProfile {
    Conservative,
    Balanced,
    MaximumPerformance,
}

/// Host-facing policy and bounded-resource configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleEngineConfig {
    /// Validated model-package root supplied by an embedded Apple host.
    /// Command-line integrations may leave this unset and provide their
    /// development model path through a backend-specific factory instead.
    pub model_package_path: Option<PathBuf>,
    /// Optional resource-bundle root containing shipping metallibs and their
    /// manifests. A model package may also carry these assets directly.
    pub resource_bundle_path: Option<PathBuf>,
    pub backend_policy: BackendPolicy,
    pub cache_policy: CachePolicy,
    pub workload_profile: WorkloadProfile,
    pub memory_profile: MemoryProfile,
    pub maximum_concurrency: usize,
    pub ingress_queue_capacity: usize,
    pub event_queue_capacity: usize,
    pub hot_cache_bytes: usize,
    pub warm_cache_bytes: usize,
    pub persistent_cache_bytes: usize,
    pub persistent_cache_consent: bool,
}

impl Default for AppleEngineConfig {
    fn default() -> Self {
        Self {
            model_package_path: None,
            resource_bundle_path: None,
            backend_policy: BackendPolicy::Automatic,
            cache_policy: CachePolicy::MemoryOnly,
            workload_profile: WorkloadProfile::Balanced,
            memory_profile: MemoryProfile::Balanced,
            maximum_concurrency: if cfg!(target_os = "ios") { 4 } else { 8 },
            ingress_queue_capacity: 64,
            event_queue_capacity: 32,
            hot_cache_bytes: 256 * 1024 * 1024,
            warm_cache_bytes: if cfg!(target_os = "ios") {
                0
            } else {
                256 * 1024 * 1024
            },
            persistent_cache_bytes: 0,
            persistent_cache_consent: false,
        }
    }
}

impl AppleEngineConfig {
    pub fn validate(&self) -> Result<(), InferenceError> {
        for (field, value) in [
            ("maximum_concurrency", self.maximum_concurrency),
            ("ingress_queue_capacity", self.ingress_queue_capacity),
            ("event_queue_capacity", self.event_queue_capacity),
        ] {
            if value == 0 {
                return Err(InferenceError::InvalidConfig {
                    field,
                    reason: "must be non-zero",
                });
            }
        }
        match self.cache_policy {
            CachePolicy::PersistentEncrypted
                if !self.persistent_cache_consent || self.persistent_cache_bytes == 0 =>
            {
                return Err(InferenceError::InvalidConfig {
                    field: "persistent_cache",
                    reason: "encrypted persistence requires consent and a non-zero quota",
                });
            }
            CachePolicy::Disabled | CachePolicy::MemoryOnly
                if self.persistent_cache_consent || self.persistent_cache_bytes != 0 =>
            {
                return Err(InferenceError::InvalidConfig {
                    field: "persistent_cache",
                    reason: "consent and quota require the encrypted persistence policy",
                });
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct GenerateRequest {
    pub prompt_tokens: Vec<TokenId>,
    pub max_output_tokens: u32,
    pub sampling: SamplingParams,
    /// Larger values receive preference within the same deadline class.
    pub priority: u8,
    /// A request may narrow the engine cache policy, but a worker must never
    /// broaden it beyond [`AppleEngineConfig::cache_policy`].
    pub cache_policy: CachePolicy,
}

impl GenerateRequest {
    pub fn new(prompt_tokens: Vec<TokenId>, max_output_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            max_output_tokens,
            sampling: SamplingParams::greedy(),
            priority: 128,
            cache_policy: CachePolicy::MemoryOnly,
        }
    }

    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.prompt_tokens.is_empty() {
            return Err(InferenceError::InvalidRequest {
                field: "prompt_tokens",
                reason: "must not be empty",
            });
        }
        if self.max_output_tokens == 0 {
            return Err(InferenceError::InvalidRequest {
                field: "max_output_tokens",
                reason: "must be non-zero",
            });
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Metal,
    MetalPrefillAneDecode,
    CoreMl,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CacheTier {
    None,
    Active,
    Hot,
    Warm,
    Persistent,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ThermalState {
    Unknown,
    Nominal,
    Fair,
    Serious,
    Critical,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendFallback {
    pub from: BackendKind,
    pub to: BackendKind,
    pub reason: Arc<str>,
}

/// Per-request evidence returned with the terminal event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendReport {
    pub selected_backend: BackendKind,
    pub cache_tier: CacheTier,
    pub matched_cache_tokens: u32,
    pub saved_prefill_tokens: u32,
    /// Time from successful submission until accelerator admission. Inference,
    /// streaming backpressure, and terminal cleanup are excluded.
    pub queue_time: Duration,
    pub batch_size: u32,
    pub padding_tokens: u32,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub fallback: Option<BackendFallback>,
    pub resident_memory_bytes: u64,
    pub thermal_state: ThermalState,
    /// Optional raw phase measurements for benchmark consumers. CPU counters
    /// never imply device cycles or a clock-normalized throughput estimate.
    pub measurement: Option<serde_json::Value>,
}

impl BackendReport {
    pub fn new(selected_backend: BackendKind) -> Self {
        Self {
            selected_backend,
            cache_tier: CacheTier::None,
            matched_cache_tokens: 0,
            saved_prefill_tokens: 0,
            queue_time: Duration::ZERO,
            batch_size: 1,
            padding_tokens: 0,
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            fallback: None,
            resident_memory_bytes: 0,
            thermal_state: ThermalState::Unknown,
            measurement: None,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FinishReason {
    EndOfSequence,
    Length,
    StopToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationOutcome {
    pub finish_reason: FinishReason,
    pub report: BackendReport,
}

impl GenerationOutcome {
    pub fn length_limited(report: BackendReport) -> Self {
        Self {
            finish_reason: FinishReason::Length,
            report,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenEvent {
    Token {
        request_id: ReqId,
        index: u32,
        token_id: TokenId,
        text: Option<Arc<str>>,
    },
    Finished {
        request_id: ReqId,
        finish_reason: FinishReason,
        report: BackendReport,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InferenceError {
    InvalidConfig {
        field: &'static str,
        reason: &'static str,
    },
    InvalidRequest {
        field: &'static str,
        reason: &'static str,
    },
    QueueFull {
        capacity: usize,
    },
    WorkerInitializationFailed {
        detail: Arc<str>,
    },
    WorkerFailed {
        detail: Arc<str>,
    },
    WorkerUnavailable,
    RequestCancelled {
        request_id: ReqId,
    },
    ResponseClosed {
        request_id: ReqId,
    },
}

impl InferenceError {
    pub fn worker_initialization_failed(detail: impl Into<Arc<str>>) -> Self {
        Self::WorkerInitializationFailed {
            detail: detail.into(),
        }
    }

    pub fn worker_failed(detail: impl Into<Arc<str>>) -> Self {
        Self::WorkerFailed {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, reason } => {
                write!(f, "invalid engine config {field}: {reason}")
            }
            Self::InvalidRequest { field, reason } => {
                write!(f, "invalid generation request {field}: {reason}")
            }
            Self::QueueFull { capacity } => {
                write!(f, "inference ingress queue is full at capacity {capacity}")
            }
            Self::WorkerInitializationFailed { detail } => {
                write!(f, "inference worker initialization failed: {detail}")
            }
            Self::WorkerFailed { detail } => write!(f, "inference worker failed: {detail}"),
            Self::WorkerUnavailable => write!(f, "inference worker is unavailable"),
            Self::RequestCancelled { request_id } => {
                write!(f, "inference request {request_id} was cancelled")
            }
            Self::ResponseClosed { request_id } => {
                write!(
                    f,
                    "response stream for request {request_id} closed unexpectedly"
                )
            }
        }
    }
}

impl std::error::Error for InferenceError {}

/// Token output available to a backend while it processes one request.
pub trait TokenEmitter {
    fn request_id(&self) -> ReqId;
    fn is_cancelled(&self) -> bool;
    fn emit_token(
        &mut self,
        token_id: TokenId,
        text: Option<Arc<str>>,
    ) -> Result<(), InferenceError>;
}

/// Pluggable accelerator implementation. The value is constructed by the
/// factory on the worker thread, so this trait intentionally does not require
/// `Send` or `Sync`.
pub trait InferenceWorker: 'static {
    fn generate(
        &mut self,
        request: &GenerateRequest,
        output: &mut dyn TokenEmitter,
    ) -> Result<GenerationOutcome, InferenceError>;

    fn handle_memory_pressure(&mut self, pressure: MemoryPressure) -> Result<(), InferenceError>;
}

/// Read-only request state presented to a step-wise accelerator scheduler.
/// A worker may select any compatible subset on each call, which permits
/// decode-priority scheduling and ragged prefill without padding.
#[derive(Clone, Debug)]
pub struct ActiveRequest<'a> {
    pub request_id: ReqId,
    pub request: &'a GenerateRequest,
    pub emitted_tokens: u32,
    /// Immutable submit-to-admission latency captured by the runtime.
    pub queue_time: Duration,
    cancellation: &'a AtomicBool,
}

impl ActiveRequest<'_> {
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn cancellation_signal(&self) -> &AtomicBool {
        self.cancellation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContinuousStepOutput {
    Token {
        request_id: ReqId,
        token_id: TokenId,
        text: Option<Arc<str>>,
    },
    Finished {
        request_id: ReqId,
        outcome: GenerationOutcome,
    },
    Failed {
        request_id: ReqId,
        error: InferenceError,
    },
}

/// Accelerator contract for continuous batching. Unlike [`InferenceWorker`],
/// this interface never owns an entire request until completion. The runtime
/// retains multiple active requests and calls `step` repeatedly, while the
/// backend keeps request-owned KV keyed by `ReqId`.
pub trait ContinuousInferenceWorker: 'static {
    /// Maximum number of requests the runtime may present to `admit` under the
    /// worker's current resource state. Lower limits leave excess requests in
    /// the bounded waiting queue; they are not failed or passed to the
    /// accelerator until the worker raises the limit.
    fn admission_limit(&self, configured_maximum: usize) -> usize {
        configured_maximum
    }

    fn admit(
        &mut self,
        request_id: ReqId,
        request: &GenerateRequest,
        cancellation: &AtomicBool,
    ) -> Result<(), InferenceError>;

    fn step(
        &mut self,
        active: &[ActiveRequest<'_>],
    ) -> Result<Vec<ContinuousStepOutput>, InferenceError>;

    fn abort(&mut self, request_id: ReqId);

    fn handle_memory_pressure(&mut self, pressure: MemoryPressure) -> Result<(), InferenceError>;
}

struct RequestCommand {
    request_id: ReqId,
    request: GenerateRequest,
    submitted_at: Instant,
    events: SyncSender<Result<TokenEvent, InferenceError>>,
    cancelled: Arc<AtomicBool>,
}

enum WorkerCommand {
    Generate(RequestCommand),
}

enum ControlCommand {
    MemoryPressure {
        pressure: MemoryPressure,
        acknowledgement: SyncSender<Result<(), InferenceError>>,
    },
}

struct WorkerLifetime {
    stopping: Arc<AtomicBool>,
    owner_thread_id: thread::ThreadId,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

impl WorkerLifetime {
    fn shutdown(&self) {
        self.stopping.store(true, Ordering::Release);
        if self.owner_thread_id == thread::current().id() {
            return;
        }
        // Keep the lock through join so concurrent shutdown callers all wait
        // for owner destruction. The worker never owns this lifetime object.
        let mut thread = self
            .thread
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(owner) = thread.take() {
            let _ = owner.join();
        }
    }
}

impl Drop for WorkerLifetime {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Cloneable, bounded admission handle. Dropping the last handle cancels
/// outstanding work and waits for owner destruction at a synchronous boundary.
#[derive(Clone)]
pub struct EngineHandle {
    lifetime: Arc<WorkerLifetime>,
    commands: SyncSender<WorkerCommand>,
    controls: Sender<ControlCommand>,
    next_request_id: Arc<AtomicU64>,
    worker_alive: Arc<AtomicBool>,
    ingress_capacity: usize,
    event_capacity: usize,
    cache_policy: CachePolicy,
}

impl fmt::Debug for EngineHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineHandle")
            .field("ingress_capacity", &self.ingress_capacity)
            .field("event_capacity", &self.event_capacity)
            .field("worker_alive", &self.is_available())
            .finish_non_exhaustive()
    }
}

impl EngineHandle {
    /// Starts an accelerator worker. `factory` executes on that worker thread,
    /// and this function returns only after construction succeeds or fails.
    pub fn spawn_with_factory<F, W>(
        config: AppleEngineConfig,
        factory: F,
    ) -> Result<Self, InferenceError>
    where
        F: FnOnce(&AppleEngineConfig) -> Result<W, InferenceError> + Send + 'static,
        W: InferenceWorker,
    {
        config.validate()?;
        if config.cache_policy == CachePolicy::PersistentEncrypted {
            return Err(InferenceError::InvalidConfig {
                field: "persistent_cache",
                reason: "encrypted persistence requires explicit host material",
            });
        }
        let ingress_capacity = config.ingress_queue_capacity;
        let event_capacity = config.event_queue_capacity;
        let cache_policy = config.cache_policy;
        let (command_tx, command_rx) = mpsc::sync_channel(ingress_capacity);
        let (control_tx, control_rx) = mpsc::channel();
        let (initialized_tx, initialized_rx) = mpsc::sync_channel(1);
        let worker_alive = Arc::new(AtomicBool::new(false));
        let worker_alive_on_thread = Arc::clone(&worker_alive);
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = stopping.clone();
        let owner_thread = thread::Builder::new()
            .name("rvllm-accelerator".to_owned())
            .spawn(move || {
                let _alive_guard = WorkerAliveGuard(Arc::clone(&worker_alive_on_thread));
                let mut worker = match factory(&config) {
                    Ok(worker) => worker,
                    Err(error) => {
                        let _ = initialized_tx.send(Err(error));
                        return;
                    }
                };
                worker_alive_on_thread.store(true, Ordering::Release);
                if initialized_tx.send(Ok(())).is_err() {
                    return;
                }
                worker_loop(&mut worker, command_rx, control_rx, worker_stopping);
            })
            .map_err(|error| {
                InferenceError::worker_initialization_failed(Arc::<str>::from(error.to_string()))
            })?;
        if let Err(error) = initialized_rx
            .recv()
            .unwrap_or(Err(InferenceError::WorkerUnavailable))
        {
            let _ = owner_thread.join();
            return Err(error);
        }
        Ok(Self {
            lifetime: Arc::new(WorkerLifetime {
                stopping,
                owner_thread_id: owner_thread.thread().id(),
                thread: Mutex::new(Some(owner_thread)),
            }),
            commands: command_tx,
            controls: control_tx,
            next_request_id: Arc::new(AtomicU64::new(1)),
            worker_alive,
            ingress_capacity,
            event_capacity,
            cache_policy,
        })
    }

    /// Starts a step-wise worker capable of retaining and batching multiple
    /// active requests. The first interactive request is dispatched without a
    /// fill delay; already queued arrivals are coalesced opportunistically.
    pub fn spawn_continuous_with_factory<F, W>(
        config: AppleEngineConfig,
        factory: F,
    ) -> Result<Self, InferenceError>
    where
        F: FnOnce(&AppleEngineConfig) -> Result<W, InferenceError> + Send + 'static,
        W: ContinuousInferenceWorker,
    {
        if config.cache_policy == CachePolicy::PersistentEncrypted {
            return Err(InferenceError::InvalidConfig {
                field: "persistent_cache",
                reason: "encrypted persistence requires explicit host material",
            });
        }
        Self::spawn_continuous_inner(config, factory)
    }

    fn spawn_continuous_inner<F, W>(
        config: AppleEngineConfig,
        factory: F,
    ) -> Result<Self, InferenceError>
    where
        F: FnOnce(&AppleEngineConfig) -> Result<W, InferenceError> + Send + 'static,
        W: ContinuousInferenceWorker,
    {
        config.validate()?;
        let ingress_capacity = config.ingress_queue_capacity;
        let event_capacity = config.event_queue_capacity;
        let cache_policy = config.cache_policy;
        let (command_tx, command_rx) = mpsc::sync_channel(ingress_capacity);
        let (control_tx, control_rx) = mpsc::channel();
        let (initialized_tx, initialized_rx) = mpsc::sync_channel(1);
        let worker_alive = Arc::new(AtomicBool::new(false));
        let worker_alive_on_thread = Arc::clone(&worker_alive);
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = stopping.clone();
        let owner_thread = thread::Builder::new()
            .name("rvllm-continuous-accelerator".to_owned())
            .spawn(move || {
                let _alive_guard = WorkerAliveGuard(Arc::clone(&worker_alive_on_thread));
                let mut worker = match factory(&config) {
                    Ok(worker) => worker,
                    Err(error) => {
                        let _ = initialized_tx.send(Err(error));
                        return;
                    }
                };
                worker_alive_on_thread.store(true, Ordering::Release);
                if initialized_tx.send(Ok(())).is_err() {
                    return;
                }
                continuous_worker_loop(
                    &mut worker,
                    command_rx,
                    control_rx,
                    config.maximum_concurrency,
                    worker_stopping,
                );
            })
            .map_err(|error| {
                InferenceError::worker_initialization_failed(Arc::<str>::from(error.to_string()))
            })?;
        if let Err(error) = initialized_rx
            .recv()
            .unwrap_or(Err(InferenceError::WorkerUnavailable))
        {
            let _ = owner_thread.join();
            return Err(error);
        }
        Ok(Self {
            lifetime: Arc::new(WorkerLifetime {
                stopping,
                owner_thread_id: owner_thread.thread().id(),
                thread: Mutex::new(Some(owner_thread)),
            }),
            commands: command_tx,
            controls: control_tx,
            next_request_id: Arc::new(AtomicU64::new(1)),
            worker_alive,
            ingress_capacity,
            event_capacity,
            cache_policy,
        })
    }

    /// Starts a continuous worker with non-cloneable encrypted-cache host
    /// material transferred exactly once onto the accelerator owner thread.
    pub fn spawn_continuous_with_persistent_host_factory<F, W>(
        config: AppleEngineConfig,
        host: PersistentCacheHostConfig,
        factory: F,
    ) -> Result<Self, InferenceError>
    where
        F: FnOnce(&AppleEngineConfig, PersistentCacheHostConfig) -> Result<W, InferenceError>
            + Send
            + 'static,
        W: ContinuousInferenceWorker,
    {
        if config.cache_policy != CachePolicy::PersistentEncrypted {
            return Err(InferenceError::InvalidConfig {
                field: "cache_policy",
                reason: "persistent host material requires encrypted persistence policy",
            });
        }
        host.validate().map_err(|_| InferenceError::InvalidConfig {
            field: "persistent_cache",
            reason: "persistent host material is incomplete or invalid",
        })?;
        Self::spawn_continuous_inner(config, move |config| factory(config, host))
    }

    /// Reports whether the accelerator owner thread is still running. This
    /// becomes false during normal shutdown and stack unwinding after a panic,
    /// so health endpoints cannot keep claiming readiness for a dead worker.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.worker_alive.load(Ordering::Acquire)
    }

    /// Nonblocking admission. Saturation is explicit rather than turning an
    /// unbounded producer into memory pressure on the inference process.
    pub fn submit(&self, request: GenerateRequest) -> Result<RequestHandle, InferenceError> {
        if self.lifetime.stopping.load(Ordering::Acquire) {
            return Err(InferenceError::WorkerUnavailable);
        }
        request.validate()?;
        if !cache_policy_allows(self.cache_policy, request.cache_policy) {
            return Err(InferenceError::InvalidRequest {
                field: "cache_policy",
                reason: "request cache policy exceeds the engine policy",
            });
        }
        let request_id = ReqId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let (event_tx, event_rx) = mpsc::sync_channel(self.event_capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        let command = WorkerCommand::Generate(RequestCommand {
            request_id,
            request,
            submitted_at: Instant::now(),
            events: event_tx,
            cancelled: Arc::clone(&cancelled),
        });
        match self.commands.try_send(command) {
            Ok(()) => Ok(RequestHandle {
                request_id,
                events: event_rx,
                cancelled,
                terminal: false,
            }),
            Err(TrySendError::Full(_)) => Err(InferenceError::QueueFull {
                capacity: self.ingress_capacity,
            }),
            Err(TrySendError::Disconnected(_)) => Err(InferenceError::WorkerUnavailable),
        }
    }

    /// Sends an acknowledged lifecycle command to the accelerator worker.
    pub fn handle_memory_pressure(&self, pressure: MemoryPressure) -> Result<(), InferenceError> {
        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        let command = ControlCommand::MemoryPressure {
            pressure,
            acknowledgement: ack_tx,
        };
        match self.controls.send(command) {
            Ok(()) => ack_rx
                .recv()
                .map_err(|_| InferenceError::WorkerUnavailable)?,
            Err(_) => Err(InferenceError::WorkerUnavailable),
        }
    }

    /// Stops admission through every clone, cancels outstanding work and waits
    /// for the owner thread to destroy accelerator resources. A synchronous
    /// accelerator call must finish before shutdown can complete.
    pub fn shutdown(&self) {
        self.lifetime.shutdown();
    }
}

struct WorkerAliveGuard(Arc<AtomicBool>);

impl Drop for WorkerAliveGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn cache_policy_allows(engine: CachePolicy, request: CachePolicy) -> bool {
    let rank = |policy| match policy {
        CachePolicy::Disabled => 0,
        CachePolicy::MemoryOnly => 1,
        CachePolicy::PersistentEncrypted => 2,
    };
    rank(request) <= rank(engine)
}

/// Single-consumer token stream and cancellation handle.
pub struct RequestHandle {
    request_id: ReqId,
    events: Receiver<Result<TokenEvent, InferenceError>>,
    cancelled: Arc<AtomicBool>,
    terminal: bool,
}

impl fmt::Debug for RequestHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestHandle")
            .field("request_id", &self.request_id)
            .field("cancelled", &self.cancelled.load(Ordering::Acquire))
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

impl RequestHandle {
    pub fn id(&self) -> ReqId {
        self.request_id
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// Returns true only for the first successful cancellation request.
    pub fn cancel(&self) -> bool {
        !self.cancelled.swap(true, Ordering::AcqRel)
    }

    pub fn recv(&mut self) -> Result<TokenEvent, InferenceError> {
        if self.cancelled.load(Ordering::Acquire) {
            self.terminal = true;
            return Err(InferenceError::RequestCancelled {
                request_id: self.request_id,
            });
        }
        let received = self
            .events
            .recv()
            .map_err(|_| InferenceError::ResponseClosed {
                request_id: self.request_id,
            })?;
        let event = match received {
            Ok(event) => event,
            Err(error) => {
                self.terminal = true;
                return Err(error);
            }
        };
        if matches!(event, TokenEvent::Finished { .. }) {
            self.terminal = true;
        }
        Ok(event)
    }

    pub fn try_recv(&mut self) -> Result<Option<TokenEvent>, InferenceError> {
        if self.cancelled.load(Ordering::Acquire) {
            self.terminal = true;
            return Err(InferenceError::RequestCancelled {
                request_id: self.request_id,
            });
        }
        match self.events.try_recv() {
            Ok(Ok(event)) => {
                if matches!(event, TokenEvent::Finished { .. }) {
                    self.terminal = true;
                }
                Ok(Some(event))
            }
            Ok(Err(error)) => {
                self.terminal = true;
                Err(error)
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(InferenceError::ResponseClosed {
                request_id: self.request_id,
            }),
        }
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        if !self.terminal {
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

struct ChannelTokenEmitter {
    request_id: ReqId,
    next_index: u32,
    events: SyncSender<Result<TokenEvent, InferenceError>>,
    cancelled: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
}

impl ChannelTokenEmitter {
    fn send(&self, mut event: Result<TokenEvent, InferenceError>) -> Result<(), InferenceError> {
        loop {
            if self.is_cancelled() {
                return Err(InferenceError::RequestCancelled {
                    request_id: self.request_id,
                });
            }
            match self.events.try_send(event) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(returned)) => {
                    event = returned;
                    thread::sleep(Duration::from_millis(1));
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(InferenceError::ResponseClosed {
                        request_id: self.request_id,
                    });
                }
            }
        }
    }
}

impl TokenEmitter for ChannelTokenEmitter {
    fn request_id(&self) -> ReqId {
        self.request_id
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || self.stopping.load(Ordering::Acquire)
    }

    fn emit_token(
        &mut self,
        token_id: TokenId,
        text: Option<Arc<str>>,
    ) -> Result<(), InferenceError> {
        let index = self.next_index;
        self.send(Ok(TokenEvent::Token {
            request_id: self.request_id,
            index,
            token_id,
            text,
        }))?;
        self.next_index += 1;
        Ok(())
    }
}

const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(1);
const MAX_OPPORTUNISTIC_COALESCE: Duration = Duration::from_millis(1);

fn worker_loop<W: InferenceWorker>(
    worker: &mut W,
    commands: Receiver<WorkerCommand>,
    controls: Receiver<ControlCommand>,
    stopping: Arc<AtomicBool>,
) {
    while !stopping.load(Ordering::Acquire) {
        drain_worker_control_commands(worker, &controls);
        match commands.recv_timeout(CONTROL_POLL_INTERVAL) {
            Ok(WorkerCommand::Generate(command)) => {
                drain_worker_control_commands(worker, &controls);
                run_request(worker, command, stopping.clone());
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                drain_worker_control_commands(worker, &controls);
                return;
            }
        }
    }
}

fn drain_worker_control_commands<W: InferenceWorker>(
    worker: &mut W,
    controls: &Receiver<ControlCommand>,
) {
    while let Ok(ControlCommand::MemoryPressure {
        pressure,
        acknowledgement,
    }) = controls.try_recv()
    {
        let _ = acknowledgement.send(worker.handle_memory_pressure(pressure));
    }
}

fn run_request<W: InferenceWorker>(
    worker: &mut W,
    command: RequestCommand,
    stopping: Arc<AtomicBool>,
) {
    let RequestCommand {
        request_id,
        request,
        submitted_at,
        events,
        cancelled,
    } = command;
    let queue_time = submitted_at.elapsed();
    let mut emitter = ChannelTokenEmitter {
        request_id,
        next_index: 0,
        events,
        cancelled,
        stopping,
    };
    if emitter.is_cancelled() {
        return;
    }
    match worker.generate(&request, &mut emitter) {
        Ok(mut outcome) => {
            outcome.report.queue_time = queue_time;
            let _ = emitter.send(Ok(TokenEvent::Finished {
                request_id,
                finish_reason: outcome.finish_reason,
                report: outcome.report,
            }));
        }
        Err(error) => {
            if !matches!(error, InferenceError::RequestCancelled { .. }) {
                let _ = emitter.send(Err(error));
            }
        }
    }
}

struct ContinuousActive {
    command: RequestCommand,
    /// Time from successful submission until accelerator admission. This is
    /// captured once and must not grow with inference or response backpressure.
    queue_time: Duration,
    next_index: u32,
    pending: VecDeque<Result<TokenEvent, InferenceError>>,
    finishing: bool,
}

fn continuous_worker_loop<W: ContinuousInferenceWorker>(
    worker: &mut W,
    commands: Receiver<WorkerCommand>,
    controls: Receiver<ControlCommand>,
    maximum_concurrency: usize,
    stopping: Arc<AtomicBool>,
) {
    let mut waiting = VecDeque::new();
    let mut active: Vec<ContinuousActive> = Vec::new();

    loop {
        if stopping.load(Ordering::Acquire) {
            for state in &active {
                if !state.finishing {
                    worker.abort(state.command.request_id);
                }
            }
            return;
        }
        drain_continuous_control_commands(worker, &controls);
        if waiting.is_empty() && active.is_empty() {
            match commands.recv_timeout(CONTROL_POLL_INTERVAL) {
                Ok(WorkerCommand::Generate(command)) => waiting.push_back(command),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    drain_continuous_control_commands(worker, &controls);
                    return;
                }
            }

            // An idle interactive request is admitted before looking for more
            // generation work. This makes coalescing opportunistic rather than
            // an implicit fill delay.
            admit_waiting(
                worker,
                &controls,
                &mut waiting,
                &mut active,
                maximum_concurrency,
            );
        }

        let available_batch_slots = maximum_concurrency
            .saturating_sub(active.len())
            .saturating_sub(waiting.len());
        if available_batch_slots > 0 {
            coalesce_already_arrived(
                &commands,
                &mut waiting,
                available_batch_slots,
                MAX_OPPORTUNISTIC_COALESCE,
            );
        }

        admit_waiting(
            worker,
            &controls,
            &mut waiting,
            &mut active,
            maximum_concurrency,
        );

        if stopping.load(Ordering::Acquire) {
            continue;
        }

        for state in &mut active {
            if state.command.cancelled.load(Ordering::Acquire) && !state.finishing {
                worker.abort(state.command.request_id);
                state.pending.clear();
                state.finishing = true;
            }
            // The receiver can disconnect after the cancellation check above.
            // Abort synchronously before the finishing state is retained away,
            // otherwise request-owned KV could outlive its only stream handle.
            if flush_continuous_events(state) {
                worker.abort(state.command.request_id);
            }
        }
        active.retain(|state| !(state.finishing && state.pending.is_empty()));

        let ready_indices: Vec<_> = active
            .iter()
            .enumerate()
            .filter(|(_, state)| !state.finishing && state.pending.is_empty())
            .map(|(index, _)| index)
            .collect();
        if ready_indices.is_empty() {
            if !active.is_empty() {
                thread::sleep(Duration::from_millis(1));
            }
            continue;
        }

        let views: Vec<_> = ready_indices
            .iter()
            .map(|&index| {
                let state = &active[index];
                ActiveRequest {
                    request_id: state.command.request_id,
                    request: &state.command.request,
                    emitted_tokens: state.next_index,
                    queue_time: state.queue_time,
                    cancellation: state.command.cancelled.as_ref(),
                }
            })
            .collect();
        drain_continuous_control_commands(worker, &controls);
        let outputs = match worker.step(&views) {
            Ok(outputs) => outputs,
            Err(error) => {
                for state in &mut active {
                    worker.abort(state.command.request_id);
                    state.pending.push_back(Err(error.clone()));
                    state.finishing = true;
                }
                continue;
            }
        };
        if outputs.is_empty() {
            // Chunked prefill normally has no host-visible token, and a
            // backend may temporarily apply bounded response backpressure.
            // Avoid turning either case into a hot spin on the accelerator
            // owner thread.
            thread::sleep(Duration::from_millis(1));
        }
        let ready_ids: HashSet<_> = views.iter().map(|view| view.request_id).collect();
        drop(views);
        let mut emitted = HashSet::new();
        for output in outputs {
            let request_id = match &output {
                ContinuousStepOutput::Token { request_id, .. }
                | ContinuousStepOutput::Finished { request_id, .. }
                | ContinuousStepOutput::Failed { request_id, .. } => *request_id,
            };
            if !ready_ids.contains(&request_id) || !emitted.insert(request_id) {
                fail_all_continuous(
                    worker,
                    &mut active,
                    InferenceError::worker_failed(
                        "continuous worker returned an unknown or duplicate request id",
                    ),
                );
                break;
            }
            let Some(state) = active
                .iter_mut()
                .find(|state| state.command.request_id == request_id)
            else {
                continue;
            };
            match output {
                ContinuousStepOutput::Token { token_id, text, .. } => {
                    if state.next_index >= state.command.request.max_output_tokens {
                        worker.abort(request_id);
                        state.pending.push_back(Err(InferenceError::worker_failed(
                            "continuous worker exceeded max_output_tokens",
                        )));
                        state.finishing = true;
                    } else {
                        state.pending.push_back(Ok(TokenEvent::Token {
                            request_id,
                            index: state.next_index,
                            token_id,
                            text,
                        }));
                        state.next_index += 1;
                    }
                }
                ContinuousStepOutput::Finished { mut outcome, .. } => {
                    outcome.report.queue_time = state.queue_time;
                    state.pending.push_back(Ok(TokenEvent::Finished {
                        request_id,
                        finish_reason: outcome.finish_reason,
                        report: outcome.report,
                    }));
                    state.finishing = true;
                }
                ContinuousStepOutput::Failed { error, .. } => {
                    worker.abort(request_id);
                    state.pending.push_back(Err(error));
                    state.finishing = true;
                }
            }
        }
    }
}

fn drain_continuous_control_commands<W: ContinuousInferenceWorker>(
    worker: &mut W,
    controls: &Receiver<ControlCommand>,
) {
    while let Ok(ControlCommand::MemoryPressure {
        pressure,
        acknowledgement,
    }) = controls.try_recv()
    {
        let _ = acknowledgement.send(worker.handle_memory_pressure(pressure));
    }
}

fn coalesce_already_arrived(
    commands: &Receiver<WorkerCommand>,
    waiting: &mut VecDeque<RequestCommand>,
    limit: usize,
    budget: Duration,
) -> usize {
    let started = Instant::now();
    let mut coalesced = 0;
    while coalesced < limit && started.elapsed() < budget {
        match commands.try_recv() {
            Ok(WorkerCommand::Generate(command)) => {
                waiting.push_back(command);
                coalesced += 1;
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
        }
    }
    coalesced
}

fn admit_waiting<W: ContinuousInferenceWorker>(
    worker: &mut W,
    controls: &Receiver<ControlCommand>,
    waiting: &mut VecDeque<RequestCommand>,
    active: &mut Vec<ContinuousActive>,
    maximum_concurrency: usize,
) {
    loop {
        // Controls have priority over admission. In particular, a successful
        // Normal acknowledgement may expand the limit, while warning or
        // critical pressure takes effect before the next request is admitted.
        drain_continuous_control_commands(worker, controls);
        let admission_limit = worker
            .admission_limit(maximum_concurrency)
            .min(maximum_concurrency);
        if active.len() >= admission_limit {
            break;
        }
        let Some(command) = waiting.pop_front() else {
            break;
        };
        if command.cancelled.load(Ordering::Acquire) {
            continue;
        }
        match worker.admit(
            command.request_id,
            &command.request,
            command.cancelled.as_ref(),
        ) {
            Ok(()) => {
                let queue_time = command.submitted_at.elapsed();
                active.push(ContinuousActive {
                    command,
                    queue_time,
                    next_index: 0,
                    pending: VecDeque::new(),
                    finishing: false,
                });
            }
            Err(error) => {
                let _ = command.events.try_send(Err(error));
            }
        }
    }
}

/// Returns true when the response receiver disconnected and the accelerator
/// worker must abort the request before `state` is removed.
fn flush_continuous_events(state: &mut ContinuousActive) -> bool {
    while let Some(event) = state.pending.pop_front() {
        match state.command.events.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(event)) => {
                state.pending.push_front(event);
                break;
            }
            Err(TrySendError::Disconnected(_)) => {
                state.command.cancelled.store(true, Ordering::Release);
                state.pending.clear();
                state.finishing = true;
                return true;
            }
        }
    }
    false
}

fn fail_all_continuous<W: ContinuousInferenceWorker>(
    worker: &mut W,
    active: &mut [ContinuousActive],
    error: InferenceError,
) {
    for state in active {
        worker.abort(state.command.request_id);
        state.pending.clear();
        state.pending.push_back(Err(error.clone()));
        state.finishing = true;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn last_handle_drop_joins_a_backpressured_owner_with_live_stream() {
        struct Owner {
            entered: Sender<()>,
            dropped: Sender<thread::ThreadId>,
        }
        impl Drop for Owner {
            fn drop(&mut self) {
                self.dropped.send(thread::current().id()).unwrap();
            }
        }
        impl InferenceWorker for Owner {
            fn generate(
                &mut self,
                _: &GenerateRequest,
                output: &mut dyn TokenEmitter,
            ) -> Result<GenerationOutcome, InferenceError> {
                output.emit_token(TokenId(11), None)?;
                self.entered.send(()).unwrap();
                // The first token fills the one-slot queue. Shutdown must
                // cancel this blocked send while the stream remains alive.
                output.emit_token(TokenId(12), None)?;
                Ok(GenerationOutcome::length_limited(BackendReport::new(
                    BackendKind::Metal,
                )))
            }
            fn handle_memory_pressure(&mut self, _: MemoryPressure) -> Result<(), InferenceError> {
                Ok(())
            }
        }
        let (entered, progress) = mpsc::channel();
        let (dropped, receipt) = mpsc::channel();
        let config = AppleEngineConfig {
            event_queue_capacity: 1,
            ..AppleEngineConfig::default()
        };
        let engine =
            EngineHandle::spawn_with_factory(config, move |_| Ok(Owner { entered, dropped }))
                .unwrap();
        let clone = engine.clone();
        let stream = engine.submit(request(&[1])).unwrap();
        progress.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(engine);
        assert!(matches!(receipt.try_recv(), Err(TryRecvError::Empty)));
        drop(clone);
        // No timing wait: returning from final handle drop proves the owner
        // destructor already completed on its construction thread.
        assert_ne!(receipt.try_recv().unwrap(), thread::current().id());
        drop(stream);
    }

    #[test]
    fn explicit_shutdown_stops_admission_through_all_clones() {
        let engine =
            EngineHandle::spawn_continuous_with_factory(AppleEngineConfig::default(), |_| {
                Ok(NoopContinuousWorker)
            })
            .unwrap();
        let clone = engine.clone();
        engine.shutdown();
        assert!(!clone.is_available());
        assert!(matches!(
            clone.submit(request(&[1])),
            Err(InferenceError::WorkerUnavailable)
        ));
        clone.shutdown();
    }
    use super::*;
    use crate::{PersistentCacheHostConfig, PersistentCacheKey};
    use std::sync::{Condvar, Mutex};

    fn request(tokens: &[u32]) -> GenerateRequest {
        GenerateRequest::new(tokens.iter().copied().map(TokenId).collect(), 4)
    }

    struct NoopContinuousWorker;

    impl ContinuousInferenceWorker for NoopContinuousWorker {
        fn admit(
            &mut self,
            _request_id: ReqId,
            _request: &GenerateRequest,
            _cancellation: &AtomicBool,
        ) -> Result<(), InferenceError> {
            Ok(())
        }

        fn step(
            &mut self,
            _active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            Ok(Vec::new())
        }

        fn abort(&mut self, _request_id: ReqId) {}

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn nonpersistent_policy_rejects_persistent_cache_capability() {
        for cache_policy in [CachePolicy::Disabled, CachePolicy::MemoryOnly] {
            let mut consent_without_quota = AppleEngineConfig {
                cache_policy,
                persistent_cache_consent: true,
                ..AppleEngineConfig::default()
            };
            assert!(matches!(
                consent_without_quota.validate(),
                Err(InferenceError::InvalidConfig {
                    field: "persistent_cache",
                    ..
                })
            ));

            consent_without_quota.persistent_cache_consent = false;
            consent_without_quota.persistent_cache_bytes = 1;
            assert!(matches!(
                consent_without_quota.validate(),
                Err(InferenceError::InvalidConfig {
                    field: "persistent_cache",
                    ..
                })
            ));
        }

        assert!(AppleEngineConfig::default().validate().is_ok());
    }

    #[test]
    fn persistent_policy_requires_separate_host_material_factory() {
        let temp = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut config = AppleEngineConfig::default();
        config.cache_policy = CachePolicy::PersistentEncrypted;
        config.persistent_cache_consent = true;
        config.persistent_cache_bytes = 1024 * 1024;

        assert!(matches!(
            EngineHandle::spawn_with_factory(config.clone(), |_| Ok(EchoWorker)),
            Err(InferenceError::InvalidConfig {
                field: "persistent_cache",
                ..
            })
        ));
        assert!(matches!(
            EngineHandle::spawn_continuous_with_factory(config.clone(), |_| {
                Ok(NoopContinuousWorker)
            }),
            Err(InferenceError::InvalidConfig {
                field: "persistent_cache",
                ..
            })
        ));

        let host = PersistentCacheHostConfig {
            consent: true,
            root: std::fs::canonicalize(temp.path()).unwrap(),
            quota_bytes: 1024 * 1024,
            max_record_bytes: 512 * 1024,
            cache_namespace: "tenant-a".to_owned(),
            key: Some(PersistentCacheKey::new([7; 32])),
        };
        let engine = EngineHandle::spawn_continuous_with_persistent_host_factory(
            config,
            host,
            |_config, host| {
                assert_eq!(host.cache_namespace, "tenant-a");
                Ok(NoopContinuousWorker)
            },
        )
        .unwrap();
        assert!(engine.is_available());
    }

    #[test]
    fn requests_can_only_narrow_the_engine_cache_policy() {
        assert!(cache_policy_allows(
            CachePolicy::PersistentEncrypted,
            CachePolicy::MemoryOnly
        ));
        assert!(cache_policy_allows(
            CachePolicy::MemoryOnly,
            CachePolicy::Disabled
        ));
        assert!(!cache_policy_allows(
            CachePolicy::MemoryOnly,
            CachePolicy::PersistentEncrypted
        ));
        assert!(!cache_policy_allows(
            CachePolicy::Disabled,
            CachePolicy::MemoryOnly
        ));
    }

    #[test]
    fn response_disconnect_requires_worker_abort_before_active_state_is_dropped() {
        let (events, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut state = ContinuousActive {
            command: RequestCommand {
                request_id: ReqId(77),
                request: request(&[1]),
                submitted_at: Instant::now(),
                events,
                cancelled: Arc::clone(&cancelled),
            },
            queue_time: Duration::ZERO,
            next_index: 1,
            pending: VecDeque::from([Ok(TokenEvent::Token {
                request_id: ReqId(77),
                index: 0,
                token_id: TokenId(9),
                text: None,
            })]),
            finishing: false,
        };

        assert!(flush_continuous_events(&mut state));
        assert!(cancelled.load(Ordering::Acquire));
        assert!(state.finishing);
        assert!(state.pending.is_empty());
    }

    struct EchoWorker;

    impl InferenceWorker for EchoWorker {
        fn generate(
            &mut self,
            request: &GenerateRequest,
            output: &mut dyn TokenEmitter,
        ) -> Result<GenerationOutcome, InferenceError> {
            for token in &request.prompt_tokens {
                output.emit_token(*token, None)?;
            }
            Ok(GenerationOutcome::length_limited(BackendReport::new(
                BackendKind::Metal,
            )))
        }

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn streaming_tokens_are_ordered_and_finish_is_terminal() {
        let engine =
            EngineHandle::spawn_with_factory(AppleEngineConfig::default(), |_| Ok(EchoWorker))
                .unwrap_or_else(|error| panic!("worker failed to start: {error}"));
        let mut handle = engine
            .submit(request(&[9, 7, 5]))
            .unwrap_or_else(|error| panic!("submit failed: {error}"));
        for (index, expected) in [9, 7, 5].into_iter().enumerate() {
            assert_eq!(
                handle.recv(),
                Ok(TokenEvent::Token {
                    request_id: handle.id(),
                    index: index as u32,
                    token_id: TokenId(expected),
                    text: None,
                })
            );
        }
        assert!(matches!(
            handle.recv(),
            Ok(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                ..
            })
        ));
        assert!(handle.is_terminal());
    }

    struct GateWorker {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }

    impl InferenceWorker for GateWorker {
        fn generate(
            &mut self,
            _request: &GenerateRequest,
            _output: &mut dyn TokenEmitter,
        ) -> Result<GenerationOutcome, InferenceError> {
            let (lock, wake) = &*self.gate;
            let mut open = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            while !*open {
                open = wake
                    .wait(open)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            Ok(GenerationOutcome::length_limited(BackendReport::new(
                BackendKind::Metal,
            )))
        }

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn bounded_ingress_reports_queue_full_without_blocking() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_gate = Arc::clone(&gate);
        let mut config = AppleEngineConfig::default();
        config.ingress_queue_capacity = 1;
        let engine =
            EngineHandle::spawn_with_factory(config, move |_| Ok(GateWorker { gate: worker_gate }))
                .unwrap_or_else(|error| panic!("worker failed to start: {error}"));

        let first = engine
            .submit(request(&[1]))
            .unwrap_or_else(|error| panic!("first submit failed: {error}"));
        // Wait until the worker has consumed the first command and is blocked.
        for _ in 0..100 {
            if engine.submit(request(&[2])).is_ok() {
                let error = engine.submit(request(&[3])).unwrap_err();
                assert_eq!(error, InferenceError::QueueFull { capacity: 1 });
                let (lock, wake) = &*gate;
                *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
                wake.notify_all();
                drop(first);
                return;
            }
            thread::sleep(Duration::from_millis(1));
        }
        panic!("worker did not consume the first request");
    }

    struct CancellationWorker {
        observed: Arc<AtomicBool>,
        started: SyncSender<()>,
    }

    impl InferenceWorker for CancellationWorker {
        fn generate(
            &mut self,
            _request: &GenerateRequest,
            output: &mut dyn TokenEmitter,
        ) -> Result<GenerationOutcome, InferenceError> {
            let _ = self.started.send(());
            while !output.is_cancelled() {
                thread::yield_now();
            }
            self.observed.store(true, Ordering::Release);
            Err(InferenceError::RequestCancelled {
                request_id: output.request_id(),
            })
        }

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn cancellation_reaches_a_running_worker() {
        let observed = Arc::new(AtomicBool::new(false));
        let worker_observed = Arc::clone(&observed);
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let engine = EngineHandle::spawn_with_factory(AppleEngineConfig::default(), move |_| {
            Ok(CancellationWorker {
                observed: worker_observed,
                started: started_tx,
            })
        })
        .unwrap_or_else(|error| panic!("worker failed to start: {error}"));
        let mut handle = engine
            .submit(request(&[1]))
            .unwrap_or_else(|error| panic!("submit failed: {error}"));
        assert_eq!(started_rx.recv(), Ok(()));
        assert!(handle.cancel());
        assert!(!handle.cancel());
        assert_eq!(
            handle.recv(),
            Err(InferenceError::RequestCancelled {
                request_id: handle.id()
            })
        );
        for _ in 0..1000 {
            if observed.load(Ordering::Acquire) {
                return;
            }
            thread::yield_now();
        }
        panic!("worker did not observe cancellation");
    }

    struct PressureWorker {
        observed: SyncSender<MemoryPressure>,
    }

    impl InferenceWorker for PressureWorker {
        fn generate(
            &mut self,
            _request: &GenerateRequest,
            _output: &mut dyn TokenEmitter,
        ) -> Result<GenerationOutcome, InferenceError> {
            Ok(GenerationOutcome::length_limited(BackendReport::new(
                BackendKind::Metal,
            )))
        }

        fn handle_memory_pressure(
            &mut self,
            pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            self.observed
                .send(pressure)
                .map_err(|_| InferenceError::worker_failed("observer closed"))
        }
    }

    #[test]
    fn memory_pressure_is_acknowledged_after_worker_handles_it() {
        let (observed_tx, observed_rx) = mpsc::sync_channel(1);
        let engine = EngineHandle::spawn_with_factory(AppleEngineConfig::default(), move |_| {
            Ok(PressureWorker {
                observed: observed_tx,
            })
        })
        .unwrap_or_else(|error| panic!("worker failed to start: {error}"));
        assert_eq!(
            engine.handle_memory_pressure(MemoryPressure::Critical),
            Ok(())
        );
        assert_eq!(observed_rx.recv(), Ok(MemoryPressure::Critical));
    }

    struct PriorityPressureWorker {
        gate: Arc<(Mutex<bool>, Condvar)>,
        observations: SyncSender<&'static str>,
        blocked_first_step: bool,
    }

    impl ContinuousInferenceWorker for PriorityPressureWorker {
        fn admit(
            &mut self,
            _request_id: ReqId,
            request: &GenerateRequest,
            _cancellation: &AtomicBool,
        ) -> Result<(), InferenceError> {
            if request.prompt_tokens == [TokenId(2)] {
                let _ = self.observations.send("admit_second");
            }
            Ok(())
        }

        fn step(
            &mut self,
            active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            if !self.blocked_first_step {
                self.blocked_first_step = true;
                let _ = self.observations.send("step_started");
                let (lock, wake) = &*self.gate;
                let mut open = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                while !*open {
                    open = wake
                        .wait(open)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
            }
            Ok(active
                .iter()
                .map(|request| ContinuousStepOutput::Finished {
                    request_id: request.request_id,
                    outcome: GenerationOutcome::length_limited(BackendReport::new(
                        BackendKind::Metal,
                    )),
                })
                .collect())
        }

        fn abort(&mut self, _request_id: ReqId) {}

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            let _ = self.observations.send("pressure");
            Ok(())
        }
    }

    #[test]
    fn memory_pressure_bypasses_full_ingress_and_precedes_admission() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_gate = Arc::clone(&gate);
        let (observations_tx, observations_rx) = mpsc::sync_channel(8);
        let mut config = AppleEngineConfig::default();
        config.ingress_queue_capacity = 1;
        config.maximum_concurrency = 2;
        let engine = EngineHandle::spawn_continuous_with_factory(config, move |_| {
            Ok(PriorityPressureWorker {
                gate: worker_gate,
                observations: observations_tx,
                blocked_first_step: false,
            })
        })
        .unwrap();

        let _first = engine.submit(request(&[1])).unwrap();
        assert_eq!(observations_rx.recv(), Ok("step_started"));
        let _second = engine.submit(request(&[2])).unwrap();
        assert_eq!(
            engine.submit(request(&[3])).unwrap_err(),
            InferenceError::QueueFull { capacity: 1 }
        );

        let (acknowledgement, acknowledged) = mpsc::sync_channel(1);
        engine
            .controls
            .send(ControlCommand::MemoryPressure {
                pressure: MemoryPressure::Critical,
                acknowledgement,
            })
            .unwrap();

        let (lock, wake) = &*gate;
        *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
        wake.notify_all();

        assert_eq!(acknowledged.recv(), Ok(Ok(())));
        assert_eq!(observations_rx.recv(), Ok("pressure"));
        assert_eq!(observations_rx.recv(), Ok("admit_second"));
    }

    struct PressureAdmissionWorker {
        pressure: MemoryPressure,
        admitted: Vec<ReqId>,
    }

    impl ContinuousInferenceWorker for PressureAdmissionWorker {
        fn admission_limit(&self, configured_maximum: usize) -> usize {
            match self.pressure {
                MemoryPressure::Normal => configured_maximum,
                MemoryPressure::Warning => configured_maximum.div_ceil(2),
                MemoryPressure::Critical => 1,
            }
        }

        fn admit(
            &mut self,
            request_id: ReqId,
            _request: &GenerateRequest,
            _cancellation: &AtomicBool,
        ) -> Result<(), InferenceError> {
            self.admitted.push(request_id);
            Ok(())
        }

        fn step(
            &mut self,
            _active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            Ok(Vec::new())
        }

        fn abort(&mut self, _request_id: ReqId) {}

        fn handle_memory_pressure(
            &mut self,
            pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            self.pressure = pressure;
            Ok(())
        }
    }

    fn waiting_command(
        request_id: u64,
    ) -> (RequestCommand, Receiver<Result<TokenEvent, InferenceError>>) {
        let (events, receiver) = mpsc::sync_channel(1);
        (
            RequestCommand {
                request_id: ReqId(request_id),
                request: request(&[request_id as u32]),
                submitted_at: Instant::now(),
                events,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
            receiver,
        )
    }

    #[test]
    fn pressure_limits_leave_excess_requests_waiting_and_normal_resumes_them() {
        let mut worker = PressureAdmissionWorker {
            pressure: MemoryPressure::Normal,
            admitted: Vec::new(),
        };
        let (controls_tx, controls_rx) = mpsc::channel();
        let mut waiting = VecDeque::new();
        let mut receivers = Vec::new();
        for request_id in 1..=4 {
            let (command, receiver) = waiting_command(request_id);
            waiting.push_back(command);
            receivers.push(receiver);
        }
        let mut active = Vec::new();

        let (warning_ack, warning_rx) = mpsc::sync_channel(1);
        controls_tx
            .send(ControlCommand::MemoryPressure {
                pressure: MemoryPressure::Warning,
                acknowledgement: warning_ack,
            })
            .unwrap();
        admit_waiting(&mut worker, &controls_rx, &mut waiting, &mut active, 4);
        assert_eq!(warning_rx.recv(), Ok(Ok(())));
        assert_eq!(active.len(), 2);
        assert_eq!(waiting.len(), 2);

        let (critical_ack, critical_rx) = mpsc::sync_channel(1);
        controls_tx
            .send(ControlCommand::MemoryPressure {
                pressure: MemoryPressure::Critical,
                acknowledgement: critical_ack,
            })
            .unwrap();
        admit_waiting(&mut worker, &controls_rx, &mut waiting, &mut active, 4);
        assert_eq!(critical_rx.recv(), Ok(Ok(())));
        assert_eq!(
            active.len(),
            2,
            "critical pressure must not abort already-active requests"
        );
        assert_eq!(waiting.len(), 2);

        active.clear();
        admit_waiting(&mut worker, &controls_rx, &mut waiting, &mut active, 4);
        assert_eq!(active.len(), 1);
        assert_eq!(waiting.len(), 1);

        let (normal_ack, normal_rx) = mpsc::sync_channel(1);
        controls_tx
            .send(ControlCommand::MemoryPressure {
                pressure: MemoryPressure::Normal,
                acknowledgement: normal_ack,
            })
            .unwrap();
        admit_waiting(&mut worker, &controls_rx, &mut waiting, &mut active, 4);
        assert_eq!(normal_rx.recv(), Ok(Ok(())));
        assert_eq!(active.len(), 2);
        assert!(waiting.is_empty());
        assert_eq!(worker.admitted, [ReqId(1), ReqId(2), ReqId(3), ReqId(4)]);

        drop(receivers);
    }

    struct LivePressureAdmissionWorker {
        pressure: MemoryPressure,
        admissions: SyncSender<ReqId>,
        finish_one: Arc<AtomicBool>,
    }

    impl ContinuousInferenceWorker for LivePressureAdmissionWorker {
        fn admission_limit(&self, configured_maximum: usize) -> usize {
            match self.pressure {
                MemoryPressure::Normal => configured_maximum,
                MemoryPressure::Warning => configured_maximum.div_ceil(2),
                MemoryPressure::Critical => usize::from(configured_maximum > 0),
            }
        }

        fn admit(
            &mut self,
            request_id: ReqId,
            _request: &GenerateRequest,
            _cancellation: &AtomicBool,
        ) -> Result<(), InferenceError> {
            self.admissions
                .send(request_id)
                .map_err(|_| InferenceError::worker_failed("admission observer closed"))
        }

        fn step(
            &mut self,
            active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            if !self.finish_one.swap(false, Ordering::AcqRel) {
                return Ok(Vec::new());
            }
            Ok(active
                .first()
                .map(|request| {
                    vec![ContinuousStepOutput::Finished {
                        request_id: request.request_id,
                        outcome: GenerationOutcome::length_limited(BackendReport::new(
                            BackendKind::Metal,
                        )),
                    }]
                })
                .unwrap_or_default())
        }

        fn abort(&mut self, _request_id: ReqId) {}

        fn handle_memory_pressure(
            &mut self,
            pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            self.pressure = pressure;
            Ok(())
        }
    }

    #[test]
    fn live_worker_loop_keeps_critical_admission_throttled_until_normal_recovery() {
        let finish_one = Arc::new(AtomicBool::new(false));
        let worker_finish_one = Arc::clone(&finish_one);
        let (admissions_tx, admissions_rx) = mpsc::sync_channel(8);
        let mut config = AppleEngineConfig::default();
        config.maximum_concurrency = 4;
        config.ingress_queue_capacity = 4;
        let engine = EngineHandle::spawn_continuous_with_factory(config, move |_| {
            Ok(LivePressureAdmissionWorker {
                pressure: MemoryPressure::Normal,
                admissions: admissions_tx,
                finish_one: worker_finish_one,
            })
        })
        .unwrap();

        engine
            .handle_memory_pressure(MemoryPressure::Critical)
            .unwrap();
        let mut first = engine.submit(request(&[1])).unwrap();
        let second = engine.submit(request(&[2])).unwrap();
        let third = engine.submit(request(&[3])).unwrap();

        assert_eq!(
            admissions_rx.recv_timeout(Duration::from_secs(1)),
            Ok(first.id())
        );
        assert_eq!(
            admissions_rx.recv_timeout(Duration::from_millis(25)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "critical pressure must remain active after its acknowledgement"
        );

        finish_one.store(true, Ordering::Release);
        assert!(matches!(first.recv(), Ok(TokenEvent::Finished { .. })));
        assert_eq!(
            admissions_rx.recv_timeout(Duration::from_secs(1)),
            Ok(second.id())
        );
        assert_eq!(
            admissions_rx.recv_timeout(Duration::from_millis(25)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "finishing one request must not silently clear critical pressure"
        );

        engine
            .handle_memory_pressure(MemoryPressure::Normal)
            .unwrap();
        assert_eq!(
            admissions_rx.recv_timeout(Duration::from_secs(1)),
            Ok(third.id())
        );

        assert!(second.cancel());
        assert!(third.cancel());
    }

    #[test]
    fn opportunistic_coalescing_caps_continuous_ingress_to_available_slots() {
        let (commands_tx, commands_rx) = mpsc::sync_channel(32);
        let mut response_receivers = Vec::new();
        for request_id in 1..=16 {
            let (events, receiver) = mpsc::sync_channel(1);
            response_receivers.push(receiver);
            commands_tx
                .try_send(WorkerCommand::Generate(RequestCommand {
                    request_id: ReqId(request_id),
                    request: request(&[request_id as u32]),
                    submitted_at: Instant::now(),
                    events,
                    cancelled: Arc::new(AtomicBool::new(false)),
                }))
                .unwrap();
        }

        let mut waiting = VecDeque::new();
        let coalesced =
            coalesce_already_arrived(&commands_rx, &mut waiting, 3, MAX_OPPORTUNISTIC_COALESCE);

        assert_eq!(coalesced, 3);
        assert_eq!(waiting.len(), 3);
        assert!(matches!(
            commands_rx.try_recv(),
            Ok(WorkerCommand::Generate(_))
        ));
        drop(response_receivers);
    }

    struct FailingWorker;

    impl InferenceWorker for FailingWorker {
        fn generate(
            &mut self,
            _request: &GenerateRequest,
            _output: &mut dyn TokenEmitter,
        ) -> Result<GenerationOutcome, InferenceError> {
            Err(InferenceError::worker_failed("device lost"))
        }

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn worker_failure_is_delivered_to_the_request() {
        let engine =
            EngineHandle::spawn_with_factory(AppleEngineConfig::default(), |_| Ok(FailingWorker))
                .unwrap_or_else(|error| panic!("worker failed to start: {error}"));
        let mut handle = engine
            .submit(request(&[1]))
            .unwrap_or_else(|error| panic!("submit failed: {error}"));
        assert_eq!(
            handle.recv(),
            Err(InferenceError::worker_failed("device lost"))
        );
        assert!(handle.is_terminal());
    }

    #[test]
    fn factory_runs_on_worker_thread_and_failure_is_synchronous() {
        let caller = thread::current().id();
        let worker_thread = Arc::new(Mutex::new(None));
        let observed = Arc::clone(&worker_thread);
        let error = EngineHandle::spawn_with_factory(AppleEngineConfig::default(), move |_| {
            *observed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(thread::current().id());
            Err::<EchoWorker, _>(InferenceError::worker_initialization_failed("no device"))
        })
        .unwrap_err();
        assert_eq!(
            error,
            InferenceError::worker_initialization_failed("no device")
        );
        let actual = worker_thread
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .unwrap_or_else(|| panic!("factory did not run"));
        assert_ne!(caller, actual);
    }

    struct TwoRequestStepWorker {
        max_batch: Arc<AtomicU64>,
        admitted: HashSet<ReqId>,
    }

    struct QueueTimingWorker {
        observed_queue_time: SyncSender<Duration>,
        service_delay: Duration,
    }

    impl ContinuousInferenceWorker for QueueTimingWorker {
        fn admit(
            &mut self,
            _request_id: ReqId,
            _request: &GenerateRequest,
            _cancellation: &AtomicBool,
        ) -> Result<(), InferenceError> {
            Ok(())
        }

        fn step(
            &mut self,
            active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            let request = active
                .first()
                .unwrap_or_else(|| panic!("request was not presented after admission"));
            self.observed_queue_time
                .send(request.queue_time)
                .unwrap_or_else(|_| panic!("queue-time observer closed"));
            thread::sleep(self.service_delay);
            Ok(vec![ContinuousStepOutput::Finished {
                request_id: request.request_id,
                outcome: GenerationOutcome::length_limited(BackendReport::new(BackendKind::Metal)),
            }])
        }

        fn abort(&mut self, _request_id: ReqId) {}

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn continuous_queue_time_is_frozen_at_admission() {
        let (observed_tx, observed_rx) = mpsc::sync_channel(1);
        let service_delay = Duration::from_millis(40);
        let engine =
            EngineHandle::spawn_continuous_with_factory(AppleEngineConfig::default(), move |_| {
                Ok(QueueTimingWorker {
                    observed_queue_time: observed_tx,
                    service_delay,
                })
            })
            .unwrap_or_else(|error| panic!("worker failed to start: {error}"));
        let mut handle = engine
            .submit(request(&[41]))
            .unwrap_or_else(|error| panic!("submit failed: {error}"));
        let queue_time_at_admission = observed_rx
            .recv()
            .unwrap_or_else(|_| panic!("worker did not publish admission timing"));

        let report = match handle
            .recv()
            .unwrap_or_else(|error| panic!("request failed: {error}"))
        {
            TokenEvent::Finished { report, .. } => report,
            event => panic!("expected terminal event, got {event:?}"),
        };
        assert_eq!(report.queue_time, queue_time_at_admission);
    }

    impl ContinuousInferenceWorker for TwoRequestStepWorker {
        fn admit(
            &mut self,
            request_id: ReqId,
            _request: &GenerateRequest,
            _cancellation: &AtomicBool,
        ) -> Result<(), InferenceError> {
            self.admitted.insert(request_id);
            Ok(())
        }

        fn step(
            &mut self,
            active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            self.max_batch
                .fetch_max(active.len() as u64, Ordering::AcqRel);
            if active.len() < 2 {
                thread::yield_now();
                return Ok(Vec::new());
            }
            Ok(active
                .iter()
                .map(|request| {
                    if request.emitted_tokens == 0 {
                        ContinuousStepOutput::Token {
                            request_id: request.request_id,
                            token_id: request.request.prompt_tokens[0],
                            text: None,
                        }
                    } else {
                        ContinuousStepOutput::Finished {
                            request_id: request.request_id,
                            outcome: GenerationOutcome::length_limited(BackendReport::new(
                                BackendKind::Metal,
                            )),
                        }
                    }
                })
                .collect())
        }

        fn abort(&mut self, request_id: ReqId) {
            self.admitted.remove(&request_id);
        }

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn continuous_worker_observes_two_active_requests_and_streams_independently() {
        let max_batch = Arc::new(AtomicU64::new(0));
        let observed = Arc::clone(&max_batch);
        let mut config = AppleEngineConfig::default();
        config.maximum_concurrency = 2;
        let engine = EngineHandle::spawn_continuous_with_factory(config, move |_| {
            Ok(TwoRequestStepWorker {
                max_batch: observed,
                admitted: HashSet::new(),
            })
        })
        .unwrap();
        let mut first = engine.submit(request(&[41])).unwrap();
        let mut second = engine.submit(request(&[73])).unwrap();

        assert!(matches!(
            first.recv(),
            Ok(TokenEvent::Token {
                token_id: TokenId(41),
                index: 0,
                ..
            })
        ));
        assert!(matches!(
            second.recv(),
            Ok(TokenEvent::Token {
                token_id: TokenId(73),
                index: 0,
                ..
            })
        ));
        assert!(matches!(first.recv(), Ok(TokenEvent::Finished { .. })));
        assert!(matches!(second.recv(), Ok(TokenEvent::Finished { .. })));
        assert_eq!(max_batch.load(Ordering::Acquire), 2);
    }

    struct SchedulerBackpressureWorker {
        scheduler: crate::Scheduler,
        admitted: HashSet<ReqId>,
        terminal: HashSet<ReqId>,
        fast_steps_while_paused: Arc<AtomicU64>,
    }

    impl ContinuousInferenceWorker for SchedulerBackpressureWorker {
        fn admit(
            &mut self,
            request_id: ReqId,
            request: &GenerateRequest,
            _cancellation: &AtomicBool,
        ) -> Result<(), InferenceError> {
            self.scheduler
                .try_enqueue(crate::Request::new(
                    request_id,
                    request.prompt_tokens.clone(),
                    request.max_output_tokens,
                ))
                .map_err(|_| InferenceError::worker_failed("scheduler admission failed"))?;
            self.admitted.insert(request_id);
            Ok(())
        }

        fn step(
            &mut self,
            active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            let active_ids: HashSet<_> = active.iter().map(|view| view.request_id).collect();
            let finished: Vec<_> = self
                .terminal
                .iter()
                .copied()
                .filter(|request_id| active_ids.contains(request_id))
                .collect();
            if !finished.is_empty() {
                return Ok(finished
                    .into_iter()
                    .map(|request_id| {
                        self.terminal.remove(&request_id);
                        self.admitted.remove(&request_id);
                        ContinuousStepOutput::Finished {
                            request_id,
                            outcome: GenerationOutcome::length_limited(BackendReport::new(
                                BackendKind::Metal,
                            )),
                        }
                    })
                    .collect());
            }

            let mut paused = 0u64;
            for &request_id in &self.admitted {
                if !self.scheduler.request_is_alive(request_id) {
                    continue;
                }
                let backpressured = !active_ids.contains(&request_id);
                paused += u64::from(backpressured);
                self.scheduler
                    .set_backpressured(request_id, backpressured)
                    .map_err(|_| {
                        InferenceError::worker_failed("scheduler backpressure update failed")
                    })?;
            }

            match self.scheduler.schedule() {
                crate::BatchPlan::Idle => Ok(Vec::new()),
                crate::BatchPlan::Prefill {
                    req_ids,
                    cu_seqlens_q,
                    ..
                } => {
                    let completed = req_ids
                        .iter()
                        .copied()
                        .zip(cu_seqlens_q.windows(2).map(|span| span[1] - span[0]))
                        .collect::<Vec<_>>();
                    self.scheduler
                        .commit_prefill(&completed)
                        .map_err(|_| InferenceError::worker_failed("prefill commit failed"))?;
                    Ok(Vec::new())
                }
                crate::BatchPlan::Decode { req_ids, .. } => {
                    if paused > 0 {
                        self.fast_steps_while_paused.fetch_add(1, Ordering::AcqRel);
                    }
                    let sampled = req_ids
                        .iter()
                        .map(|request_id| {
                            let view = active
                                .iter()
                                .find(|view| view.request_id == *request_id)
                                .expect("scheduled request must be response-ready");
                            (
                                *request_id,
                                TokenId(view.request.prompt_tokens[0].raw() + view.emitted_tokens),
                                view.emitted_tokens + 1 >= view.request.max_output_tokens,
                            )
                        })
                        .collect::<Vec<_>>();
                    self.scheduler.commit_decode(
                        &sampled
                            .iter()
                            .map(|(request_id, token_id, _)| (*request_id, *token_id))
                            .collect::<Vec<_>>(),
                    );
                    Ok(sampled
                        .into_iter()
                        .map(|(request_id, token_id, terminal)| {
                            if terminal {
                                self.terminal.insert(request_id);
                            }
                            ContinuousStepOutput::Token {
                                request_id,
                                token_id,
                                text: None,
                            }
                        })
                        .collect())
                }
            }
        }

        fn abort(&mut self, request_id: ReqId) {
            self.admitted.remove(&request_id);
            self.terminal.remove(&request_id);
            self.scheduler.cancel_request(request_id);
        }

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn slow_response_pauses_independently_then_resumes_exactly() {
        let fast_steps_while_paused = Arc::new(AtomicU64::new(0));
        let observed = Arc::clone(&fast_steps_while_paused);
        let mut config = AppleEngineConfig::default();
        config.maximum_concurrency = 2;
        config.event_queue_capacity = 1;
        let engine = EngineHandle::spawn_continuous_with_factory(config, move |_| {
            Ok(SchedulerBackpressureWorker {
                scheduler: crate::Scheduler::new(),
                admitted: HashSet::new(),
                terminal: HashSet::new(),
                fast_steps_while_paused: observed,
            })
        })
        .unwrap();
        let mut slow = engine.submit(request(&[10])).unwrap();
        let mut fast = engine.submit(request(&[100])).unwrap();

        let mut fast_tokens = Vec::new();
        loop {
            match fast.recv().unwrap() {
                TokenEvent::Token {
                    index, token_id, ..
                } => fast_tokens.push((index, token_id)),
                TokenEvent::Finished { .. } => break,
            }
        }
        assert_eq!(
            fast_tokens,
            vec![
                (0, TokenId(100)),
                (1, TokenId(101)),
                (2, TokenId(102)),
                (3, TokenId(103)),
            ]
        );
        assert!(fast_steps_while_paused.load(Ordering::Acquire) > 0);

        let mut slow_tokens = Vec::new();
        loop {
            match slow.recv().unwrap() {
                TokenEvent::Token {
                    index, token_id, ..
                } => slow_tokens.push((index, token_id)),
                TokenEvent::Finished { .. } => break,
            }
        }
        assert_eq!(
            slow_tokens,
            vec![
                (0, TokenId(10)),
                (1, TokenId(11)),
                (2, TokenId(12)),
                (3, TokenId(13)),
            ]
        );
    }
}
