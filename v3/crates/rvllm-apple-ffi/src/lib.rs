//! Stable, fail-closed C boundary for Apple application hosts.
//!
//! The C surface owns no Metal or Core ML objects. ABI v2 constructs the
//! built-in continuous Metal worker from an authenticated Apple model package.
//! Rust hosts may still install an injected worker factory for focused tests
//! and custom integrations. ABI v1 has no package path and therefore remains
//! fail-closed. This crate never fabricates tokens or silently selects a
//! placeholder backend.
//!
//! # Ownership and threads
//!
//! * Engine and request pointers are opaque and must be destroyed exactly once.
//! * Engine submission and memory-pressure calls may run concurrently.
//! * One thread at a time may call `rvllm_apple_request_recv` for a request.
//! * Cancellation may run concurrently with receive. Destruction must wait for
//!   the active receive call to return.
//! * Event text is copied into caller-owned memory and is never truncated.

use std::ffi::{c_char, c_void};
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use rvllm_apple::AppleModelPackage;
use rvllm_core::TokenId;
use rvllm_runtime::{
    ActiveRequest, AppleEngineConfig, BackendKind, BackendPolicy, CachePolicy, CacheTier,
    ContinuousInferenceWorker, ContinuousStepOutput, EngineHandle, FinishReason, GenerateRequest,
    GenerationOutcome, InferenceError, InferenceWorker, MemoryPressure, MemoryProfile,
    PersistentCacheHostConfig, PersistentCacheKey, ThermalState, TokenEmitter, TokenEvent,
    WorkloadProfile,
};
use zeroize::Zeroize;

pub const RVLLM_APPLE_ABI_VERSION: u32 = 1;
pub const RVLLM_APPLE_ABI_VERSION_V2: u32 = 2;
pub const RVLLM_APPLE_ABI_VERSION_V3: u32 = 3;
const RVLLM_APPLE_PERSISTENT_CACHE_KEY_BYTES: usize = 32;
const RVLLM_APPLE_MAX_PERSISTENT_CACHE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const RVLLM_APPLE_MAX_CACHE_ROOT_BYTES: usize = 4096;
const RVLLM_APPLE_MAX_CACHE_NAMESPACE_BYTES: usize = 128;
pub const RVLLM_APPLE_ERROR_MESSAGE_CAPACITY: usize = 512;

pub type RvllmAppleStatus = i32;
pub const RVLLM_APPLE_OK: RvllmAppleStatus = 0;
pub const RVLLM_APPLE_INVALID_ARGUMENT: RvllmAppleStatus = 1;
pub const RVLLM_APPLE_BACKEND_UNAVAILABLE: RvllmAppleStatus = 2;
pub const RVLLM_APPLE_QUEUE_FULL: RvllmAppleStatus = 3;
pub const RVLLM_APPLE_CANCELLED: RvllmAppleStatus = 4;
pub const RVLLM_APPLE_BUFFER_TOO_SMALL: RvllmAppleStatus = 5;
pub const RVLLM_APPLE_END_OF_STREAM: RvllmAppleStatus = 6;
pub const RVLLM_APPLE_INTERNAL_ERROR: RvllmAppleStatus = 7;

pub const RVLLM_APPLE_EVENT_TOKEN: u32 = 1;
pub const RVLLM_APPLE_EVENT_FINISHED: u32 = 2;

#[repr(C)]
pub struct RvllmAppleError {
    pub code: RvllmAppleStatus,
    pub message: [c_char; RVLLM_APPLE_ERROR_MESSAGE_CAPACITY],
}

impl Default for RvllmAppleError {
    fn default() -> Self {
        Self {
            code: RVLLM_APPLE_OK,
            message: [0; RVLLM_APPLE_ERROR_MESSAGE_CAPACITY],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RvllmAppleEngineConfig {
    pub abi_version: u32,
    pub struct_size: u32,
    pub backend_policy: u32,
    pub cache_policy: u32,
    pub workload_profile: u32,
    pub memory_profile: u32,
    pub maximum_concurrency: u32,
    pub ingress_queue_capacity: u32,
    pub event_queue_capacity: u32,
    pub persistent_cache_consent: u8,
    pub _reserved: [u8; 7],
    pub hot_cache_bytes: u64,
    pub warm_cache_bytes: u64,
    pub persistent_cache_bytes: u64,
}

/// ABI-v2 engine configuration. Paths are borrowed UTF-8 byte ranges copied
/// during `rvllm_apple_engine_create_v2`; they need not be NUL terminated.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RvllmAppleEngineConfigV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub backend_policy: u32,
    pub cache_policy: u32,
    pub workload_profile: u32,
    pub memory_profile: u32,
    pub maximum_concurrency: u32,
    pub ingress_queue_capacity: u32,
    pub event_queue_capacity: u32,
    pub persistent_cache_consent: u8,
    pub _reserved: [u8; 7],
    pub hot_cache_bytes: u64,
    pub warm_cache_bytes: u64,
    pub persistent_cache_bytes: u64,
    pub model_package_path: *const u8,
    pub model_package_path_length: usize,
    pub resource_bundle_path: *const u8,
    pub resource_bundle_path_length: usize,
}

/// ABI-v3 engine configuration. This preserves every v2 field in order and
/// appends the opt-in encrypted persistent-cache capability.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RvllmAppleEngineConfigV3 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub backend_policy: u32,
    pub cache_policy: u32,
    pub workload_profile: u32,
    pub memory_profile: u32,
    pub maximum_concurrency: u32,
    pub ingress_queue_capacity: u32,
    pub event_queue_capacity: u32,
    pub persistent_cache_consent: u8,
    pub _reserved: [u8; 7],
    pub hot_cache_bytes: u64,
    pub warm_cache_bytes: u64,
    pub persistent_cache_bytes: u64,
    pub model_package_path: *const u8,
    pub model_package_path_length: usize,
    pub resource_bundle_path: *const u8,
    pub resource_bundle_path_length: usize,
    pub persistent_cache_root: *const u8,
    pub persistent_cache_root_length: usize,
    pub persistent_cache_key: *const u8,
    pub persistent_cache_key_length: usize,
    pub cache_namespace: *const u8,
    pub cache_namespace_length: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RvllmAppleGenerateRequest {
    pub abi_version: u32,
    pub struct_size: u32,
    pub prompt_tokens: *const u32,
    pub prompt_token_count: usize,
    pub max_output_tokens: u32,
    pub priority: u8,
    pub _reserved: [u8; 3],
    pub cache_policy: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct RvllmAppleBackendReport {
    pub selected_backend: u32,
    pub cache_tier: u32,
    pub matched_cache_tokens: u32,
    pub saved_prefill_tokens: u32,
    pub queue_time_ns: u64,
    pub batch_size: u32,
    pub padding_tokens: u32,
    pub prefill_time_ns: u64,
    pub decode_time_ns: u64,
    pub resident_memory_bytes: u64,
    pub thermal_state: u32,
    pub had_fallback: u8,
    pub _reserved: [u8; 7],
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct RvllmAppleTokenEvent {
    pub kind: u32,
    pub request_id: u64,
    pub index: u32,
    pub token_id: u32,
    pub finish_reason: u32,
    pub text_length: usize,
    pub report: RvllmAppleBackendReport,
}

type WorkerFactory = dyn Fn(&AppleEngineConfig) -> Result<Box<dyn InferenceWorker>, InferenceError>
    + Send
    + Sync
    + 'static;

type ContinuousWorkerFactory = dyn Fn(&AppleEngineConfig) -> Result<Box<dyn ContinuousInferenceWorker>, InferenceError>
    + Send
    + Sync
    + 'static;

enum InstalledWorkerFactory {
    Serial(Arc<WorkerFactory>),
    Continuous(Arc<ContinuousWorkerFactory>),
}

static WORKER_FACTORY: OnceLock<InstalledWorkerFactory> = OnceLock::new();

/// Installs the production backend factory exactly once for this process.
///
/// This is deliberately a Rust API: application-facing C and Swift code cannot
/// inject an arbitrary token producer. The closure itself executes on the
/// runtime's accelerator worker thread.
pub fn install_worker_factory<F>(factory: F) -> Result<(), &'static str>
where
    F: Fn(&AppleEngineConfig) -> Result<Box<dyn InferenceWorker>, InferenceError>
        + Send
        + Sync
        + 'static,
{
    WORKER_FACTORY
        .set(InstalledWorkerFactory::Serial(Arc::new(factory)))
        .map_err(|_| "an Apple inference worker factory is already installed")
}

/// Installs the production continuous-batching backend factory exactly once.
///
/// C and Swift callers still cannot inject token producers. A linked Rust
/// application chooses one real serial or continuous factory during process
/// initialization; engine creation fails closed when neither is installed.
pub fn install_continuous_worker_factory<F>(factory: F) -> Result<(), &'static str>
where
    F: Fn(&AppleEngineConfig) -> Result<Box<dyn ContinuousInferenceWorker>, InferenceError>
        + Send
        + Sync
        + 'static,
{
    WORKER_FACTORY
        .set(InstalledWorkerFactory::Continuous(Arc::new(factory)))
        .map_err(|_| "an Apple inference worker factory is already installed")
}

struct BoxedWorker(Box<dyn InferenceWorker>);

impl InferenceWorker for BoxedWorker {
    fn generate(
        &mut self,
        request: &GenerateRequest,
        output: &mut dyn TokenEmitter,
    ) -> Result<GenerationOutcome, InferenceError> {
        self.0.generate(request, output)
    }

    fn handle_memory_pressure(&mut self, pressure: MemoryPressure) -> Result<(), InferenceError> {
        self.0.handle_memory_pressure(pressure)
    }
}

struct BoxedContinuousWorker(Box<dyn ContinuousInferenceWorker>);

impl ContinuousInferenceWorker for BoxedContinuousWorker {
    fn admit(
        &mut self,
        request_id: rvllm_core::ReqId,
        request: &GenerateRequest,
        cancellation: &std::sync::atomic::AtomicBool,
    ) -> Result<(), InferenceError> {
        self.0.admit(request_id, request, cancellation)
    }

    fn step(
        &mut self,
        active: &[ActiveRequest<'_>],
    ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
        self.0.step(active)
    }

    fn abort(&mut self, request_id: rvllm_core::ReqId) {
        self.0.abort(request_id);
    }

    fn handle_memory_pressure(&mut self, pressure: MemoryPressure) -> Result<(), InferenceError> {
        self.0.handle_memory_pressure(pressure)
    }
}

pub struct RvllmAppleEngine {
    handle: EngineHandle,
}

pub struct RvllmAppleRequest {
    cancelled: Arc<AtomicBool>,
    events: Mutex<Receiver<Result<TokenEvent, InferenceError>>>,
    pending: Mutex<Option<TokenEvent>>,
}

fn write_error(out: *mut RvllmAppleError, code: RvllmAppleStatus, message: &str) {
    if out.is_null() {
        return;
    }
    // SAFETY: the caller supplied a non-null writable error record.
    let error = unsafe { &mut *out };
    error.code = code;
    error.message.fill(0);
    let bytes = message.as_bytes();
    let count = bytes.len().min(error.message.len().saturating_sub(1));
    for (dst, src) in error.message[..count].iter_mut().zip(&bytes[..count]) {
        *dst = *src as c_char;
    }
}

fn clear_error(out: *mut RvllmAppleError) {
    write_error(out, RVLLM_APPLE_OK, "");
}

fn status_for(error: &InferenceError) -> RvllmAppleStatus {
    match error {
        InferenceError::InvalidConfig { .. } | InferenceError::InvalidRequest { .. } => {
            RVLLM_APPLE_INVALID_ARGUMENT
        }
        InferenceError::QueueFull { .. } => RVLLM_APPLE_QUEUE_FULL,
        InferenceError::RequestCancelled { .. } => RVLLM_APPLE_CANCELLED,
        InferenceError::WorkerUnavailable | InferenceError::WorkerInitializationFailed { .. } => {
            RVLLM_APPLE_BACKEND_UNAVAILABLE
        }
        InferenceError::WorkerFailed { .. } | InferenceError::ResponseClosed { .. } => {
            RVLLM_APPLE_INTERNAL_ERROR
        }
    }
}

fn ffi_status<F>(error: *mut RvllmAppleError, f: F) -> RvllmAppleStatus
where
    F: FnOnce() -> Result<(), (RvllmAppleStatus, String)>,
{
    clear_error(error);
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => RVLLM_APPLE_OK,
        Ok(Err((code, message))) => {
            write_error(error, code, &message);
            code
        }
        Err(_) => {
            write_error(
                error,
                RVLLM_APPLE_INTERNAL_ERROR,
                "panic contained at C ABI boundary",
            );
            RVLLM_APPLE_INTERNAL_ERROR
        }
    }
}

fn duration_ns(value: Duration) -> u64 {
    u64::try_from(value.as_nanos()).unwrap_or(u64::MAX)
}

fn default_c_config() -> RvllmAppleEngineConfig {
    let config = AppleEngineConfig::default();
    RvllmAppleEngineConfig {
        abi_version: RVLLM_APPLE_ABI_VERSION,
        struct_size: size_of::<RvllmAppleEngineConfig>() as u32,
        backend_policy: 0,
        cache_policy: 1,
        workload_profile: 1,
        memory_profile: 1,
        maximum_concurrency: config.maximum_concurrency as u32,
        ingress_queue_capacity: config.ingress_queue_capacity as u32,
        event_queue_capacity: config.event_queue_capacity as u32,
        persistent_cache_consent: config.persistent_cache_consent.into(),
        _reserved: [0; 7],
        hot_cache_bytes: config.hot_cache_bytes as u64,
        warm_cache_bytes: config.warm_cache_bytes as u64,
        persistent_cache_bytes: config.persistent_cache_bytes as u64,
    }
}

fn default_c_config_v2() -> RvllmAppleEngineConfigV2 {
    let config = default_c_config();
    RvllmAppleEngineConfigV2 {
        abi_version: RVLLM_APPLE_ABI_VERSION_V2,
        struct_size: size_of::<RvllmAppleEngineConfigV2>() as u32,
        backend_policy: config.backend_policy,
        cache_policy: config.cache_policy,
        workload_profile: config.workload_profile,
        memory_profile: config.memory_profile,
        maximum_concurrency: config.maximum_concurrency,
        ingress_queue_capacity: config.ingress_queue_capacity,
        event_queue_capacity: config.event_queue_capacity,
        persistent_cache_consent: config.persistent_cache_consent,
        _reserved: [0; 7],
        hot_cache_bytes: config.hot_cache_bytes,
        warm_cache_bytes: config.warm_cache_bytes,
        persistent_cache_bytes: config.persistent_cache_bytes,
        model_package_path: ptr::null(),
        model_package_path_length: 0,
        resource_bundle_path: ptr::null(),
        resource_bundle_path_length: 0,
    }
}

fn default_c_config_v3() -> RvllmAppleEngineConfigV3 {
    let config = default_c_config_v2();
    RvllmAppleEngineConfigV3 {
        abi_version: RVLLM_APPLE_ABI_VERSION_V3,
        struct_size: size_of::<RvllmAppleEngineConfigV3>() as u32,
        backend_policy: config.backend_policy,
        cache_policy: config.cache_policy,
        workload_profile: config.workload_profile,
        memory_profile: config.memory_profile,
        maximum_concurrency: config.maximum_concurrency,
        ingress_queue_capacity: config.ingress_queue_capacity,
        event_queue_capacity: config.event_queue_capacity,
        persistent_cache_consent: config.persistent_cache_consent,
        _reserved: [0; 7],
        hot_cache_bytes: config.hot_cache_bytes,
        warm_cache_bytes: config.warm_cache_bytes,
        persistent_cache_bytes: config.persistent_cache_bytes,
        model_package_path: ptr::null(),
        model_package_path_length: 0,
        resource_bundle_path: ptr::null(),
        resource_bundle_path_length: 0,
        persistent_cache_root: ptr::null(),
        persistent_cache_root_length: 0,
        persistent_cache_key: ptr::null(),
        persistent_cache_key_length: 0,
        cache_namespace: ptr::null(),
        cache_namespace_length: 0,
    }
}

fn parse_config(config: &RvllmAppleEngineConfig) -> Result<AppleEngineConfig, String> {
    parse_config_with_persistent_capability(config, false)
}

fn parse_config_with_persistent_capability(
    config: &RvllmAppleEngineConfig,
    allow_persistent_cache: bool,
) -> Result<AppleEngineConfig, String> {
    if config.abi_version != RVLLM_APPLE_ABI_VERSION {
        return Err(format!("unsupported ABI version {}", config.abi_version));
    }
    if config.struct_size as usize != size_of::<RvllmAppleEngineConfig>() {
        return Err("engine config size does not match ABI version".to_owned());
    }
    if config._reserved != [0; 7] {
        return Err("engine config reserved bytes must be zero".to_owned());
    }
    let backend_policy = match config.backend_policy {
        0 => BackendPolicy::Automatic,
        1 => BackendPolicy::MetalOnly,
        2 => BackendPolicy::CoreMlPreferred,
        3 => BackendPolicy::CoreMlOnly,
        _ => return Err("invalid backend policy".to_owned()),
    };
    let cache_policy = parse_cache_policy(config.cache_policy)?;
    if !allow_persistent_cache
        && (cache_policy == CachePolicy::PersistentEncrypted
            || config.persistent_cache_consent != 0
            || config.persistent_cache_bytes != 0)
    {
        return Err("encrypted persistent cache requires ABI v3".to_owned());
    }
    let workload_profile = match config.workload_profile {
        0 => WorkloadProfile::Interactive,
        1 => WorkloadProfile::Balanced,
        2 => WorkloadProfile::Throughput,
        _ => return Err("invalid workload profile".to_owned()),
    };
    let memory_profile = match config.memory_profile {
        0 => MemoryProfile::Conservative,
        1 => MemoryProfile::Balanced,
        2 => MemoryProfile::MaximumPerformance,
        _ => return Err("invalid memory profile".to_owned()),
    };
    let to_usize = |name: &str, value: u64| {
        usize::try_from(value).map_err(|_| format!("{name} exceeds this platform's address space"))
    };
    Ok(AppleEngineConfig {
        model_package_path: None,
        resource_bundle_path: None,
        backend_policy,
        cache_policy,
        workload_profile,
        memory_profile,
        maximum_concurrency: config.maximum_concurrency as usize,
        ingress_queue_capacity: config.ingress_queue_capacity as usize,
        event_queue_capacity: config.event_queue_capacity as usize,
        hot_cache_bytes: to_usize("hot cache quota", config.hot_cache_bytes)?,
        warm_cache_bytes: to_usize("warm cache quota", config.warm_cache_bytes)?,
        persistent_cache_bytes: to_usize("persistent cache quota", config.persistent_cache_bytes)?,
        persistent_cache_consent: config.persistent_cache_consent != 0,
    })
}

fn borrowed_utf8_path(
    name: &'static str,
    pointer: *const u8,
    length: usize,
    required: bool,
) -> Result<Option<PathBuf>, String> {
    if length == 0 {
        if required {
            return Err(format!("{name} must not be empty"));
        }
        if !pointer.is_null() {
            return Err(format!("{name} pointer must be null when length is zero"));
        }
        return Ok(None);
    }
    if pointer.is_null() || length > isize::MAX as usize {
        return Err(format!("{name} has an invalid byte range"));
    }
    // SAFETY: the v2 ABI requires the immutable byte range to remain valid for
    // the duration of engine creation. The path is copied before returning.
    let bytes = unsafe { slice::from_raw_parts(pointer, length) };
    let value = std::str::from_utf8(bytes).map_err(|_| format!("{name} must be UTF-8"))?;
    if value.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    Ok(Some(PathBuf::from(value)))
}

fn parse_config_v2(config: &RvllmAppleEngineConfigV2) -> Result<AppleEngineConfig, String> {
    if config.abi_version != RVLLM_APPLE_ABI_VERSION_V2 {
        return Err(format!("unsupported ABI version {}", config.abi_version));
    }
    if config.struct_size as usize != size_of::<RvllmAppleEngineConfigV2>() {
        return Err("engine config v2 size does not match ABI version".to_owned());
    }
    if config._reserved != [0; 7] {
        return Err("engine config v2 reserved bytes must be zero".to_owned());
    }
    let compatibility = RvllmAppleEngineConfig {
        abi_version: RVLLM_APPLE_ABI_VERSION,
        struct_size: size_of::<RvllmAppleEngineConfig>() as u32,
        backend_policy: config.backend_policy,
        cache_policy: config.cache_policy,
        workload_profile: config.workload_profile,
        memory_profile: config.memory_profile,
        maximum_concurrency: config.maximum_concurrency,
        ingress_queue_capacity: config.ingress_queue_capacity,
        event_queue_capacity: config.event_queue_capacity,
        persistent_cache_consent: config.persistent_cache_consent,
        _reserved: [0; 7],
        hot_cache_bytes: config.hot_cache_bytes,
        warm_cache_bytes: config.warm_cache_bytes,
        persistent_cache_bytes: config.persistent_cache_bytes,
    };
    let mut parsed = parse_config(&compatibility)?;
    parsed.model_package_path = borrowed_utf8_path(
        "model package path",
        config.model_package_path,
        config.model_package_path_length,
        true,
    )?;
    parsed.resource_bundle_path = borrowed_utf8_path(
        "resource bundle path",
        config.resource_bundle_path,
        config.resource_bundle_path_length,
        false,
    )?;
    Ok(parsed)
}

struct ParsedEngineConfigV3 {
    config: AppleEngineConfig,
    persistent_cache_host: Option<PersistentCacheHostConfig>,
}

fn parse_config_v3(config: &RvllmAppleEngineConfigV3) -> Result<ParsedEngineConfigV3, String> {
    if config.abi_version != RVLLM_APPLE_ABI_VERSION_V3 {
        return Err(format!("unsupported ABI version {}", config.abi_version));
    }
    if config.struct_size as usize != size_of::<RvllmAppleEngineConfigV3>() {
        return Err("engine config v3 size does not match ABI version".to_owned());
    }
    if config._reserved != [0; 7] {
        return Err("engine config v3 reserved bytes must be zero".to_owned());
    }
    let compatibility = RvllmAppleEngineConfig {
        abi_version: RVLLM_APPLE_ABI_VERSION,
        struct_size: size_of::<RvllmAppleEngineConfig>() as u32,
        backend_policy: config.backend_policy,
        cache_policy: config.cache_policy,
        workload_profile: config.workload_profile,
        memory_profile: config.memory_profile,
        maximum_concurrency: config.maximum_concurrency,
        ingress_queue_capacity: config.ingress_queue_capacity,
        event_queue_capacity: config.event_queue_capacity,
        persistent_cache_consent: config.persistent_cache_consent,
        _reserved: [0; 7],
        hot_cache_bytes: config.hot_cache_bytes,
        warm_cache_bytes: config.warm_cache_bytes,
        persistent_cache_bytes: config.persistent_cache_bytes,
    };
    let mut parsed = parse_config_with_persistent_capability(&compatibility, true)?;
    parsed.model_package_path = borrowed_utf8_path(
        "model package path",
        config.model_package_path,
        config.model_package_path_length,
        true,
    )?;
    parsed.resource_bundle_path = borrowed_utf8_path(
        "resource bundle path",
        config.resource_bundle_path,
        config.resource_bundle_path_length,
        false,
    )?;

    if parsed.cache_policy != CachePolicy::PersistentEncrypted {
        if config.persistent_cache_consent != 0
            || config.persistent_cache_bytes != 0
            || !config.persistent_cache_root.is_null()
            || config.persistent_cache_root_length != 0
            || !config.persistent_cache_key.is_null()
            || config.persistent_cache_key_length != 0
            || !config.cache_namespace.is_null()
            || config.cache_namespace_length != 0
        {
            return Err(
                "persistent cache capability must be empty unless encrypted persistence is selected"
                    .to_owned(),
            );
        }
        return Ok(ParsedEngineConfigV3 {
            config: parsed,
            persistent_cache_host: None,
        });
    }

    if config.persistent_cache_consent != 1 {
        return Err("encrypted persistent cache requires explicit consent".to_owned());
    }
    if config.persistent_cache_bytes == 0
        || config.persistent_cache_bytes > RVLLM_APPLE_MAX_PERSISTENT_CACHE_BYTES
        || config.persistent_cache_bytes > usize::MAX as u64
        || config.persistent_cache_bytes > i64::MAX as u64
    {
        return Err("persistent cache quota is zero or exceeds the host boundary".to_owned());
    }
    if config.persistent_cache_root_length > RVLLM_APPLE_MAX_CACHE_ROOT_BYTES {
        return Err("persistent cache root exceeds the host boundary".to_owned());
    }
    let root = borrowed_utf8_path(
        "persistent cache root",
        config.persistent_cache_root,
        config.persistent_cache_root_length,
        true,
    )?
    .ok_or_else(|| "persistent cache root is required".to_owned())?;
    if !root.is_absolute()
        || root
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err("persistent cache root must be an absolute normalized path".to_owned());
    }
    if config.cache_namespace.is_null()
        || config.cache_namespace_length == 0
        || config.cache_namespace_length > RVLLM_APPLE_MAX_CACHE_NAMESPACE_BYTES
    {
        return Err("cache namespace must contain between 1 and 128 UTF-8 bytes".to_owned());
    }
    // SAFETY: the v3 namespace follows the same synchronous borrowed-range
    // contract as the key and paths and is copied before this function returns.
    let namespace_bytes =
        unsafe { slice::from_raw_parts(config.cache_namespace, config.cache_namespace_length) };
    let cache_namespace = std::str::from_utf8(namespace_bytes)
        .map_err(|_| "cache namespace must be UTF-8".to_owned())?
        .to_owned();
    if cache_namespace
        .chars()
        .any(|character| character.is_control())
    {
        return Err("cache namespace must not contain control characters".to_owned());
    }
    if config.persistent_cache_key_length != RVLLM_APPLE_PERSISTENT_CACHE_KEY_BYTES
        || config.persistent_cache_key.is_null()
    {
        return Err("persistent cache key must contain exactly 32 bytes".to_owned());
    }
    // SAFETY: v3 requires this immutable range to remain valid for the
    // duration of engine creation. Copy it immediately into the zeroing
    // runtime wrapper; the caller may clear its temporary buffer on return.
    let key_bytes = unsafe {
        slice::from_raw_parts(
            config.persistent_cache_key,
            RVLLM_APPLE_PERSISTENT_CACHE_KEY_BYTES,
        )
    };
    let mut key = [0_u8; RVLLM_APPLE_PERSISTENT_CACHE_KEY_BYTES];
    key.copy_from_slice(key_bytes);
    let quota_bytes = config.persistent_cache_bytes;
    let max_record_bytes = if cfg!(target_os = "ios") {
        quota_bytes.min(32 * 1024 * 1024)
    } else {
        quota_bytes.min(128 * 1024 * 1024)
    };
    let persistent_cache_host = PersistentCacheHostConfig {
        consent: true,
        root,
        quota_bytes,
        max_record_bytes,
        cache_namespace,
        key: Some(PersistentCacheKey::new(key)),
    };
    key.zeroize();
    persistent_cache_host
        .validate()
        .map_err(|_| "persistent cache host capability is invalid".to_owned())?;
    Ok(ParsedEngineConfigV3 {
        config: parsed,
        persistent_cache_host: Some(persistent_cache_host),
    })
}

fn parse_cache_policy(value: u32) -> Result<CachePolicy, String> {
    match value {
        0 => Ok(CachePolicy::Disabled),
        1 => Ok(CachePolicy::MemoryOnly),
        2 => Ok(CachePolicy::PersistentEncrypted),
        _ => Err("invalid cache policy".to_owned()),
    }
}

fn parse_request(request: &RvllmAppleGenerateRequest) -> Result<GenerateRequest, String> {
    if request.abi_version != RVLLM_APPLE_ABI_VERSION {
        return Err(format!("unsupported ABI version {}", request.abi_version));
    }
    if request.struct_size as usize != size_of::<RvllmAppleGenerateRequest>() {
        return Err("generate request size does not match ABI version".to_owned());
    }
    if request._reserved != [0; 3] {
        return Err("generate request reserved bytes must be zero".to_owned());
    }
    if request.prompt_tokens.is_null() || request.prompt_token_count == 0 {
        return Err("prompt token buffer must be non-null and non-empty".to_owned());
    }
    if request.prompt_token_count > isize::MAX as usize / size_of::<u32>() {
        return Err("prompt token buffer is too large".to_owned());
    }
    // SAFETY: pointer and length were validated; the caller promises this
    // immutable range remains live for the duration of this call. We copy it.
    let token_ids =
        unsafe { slice::from_raw_parts(request.prompt_tokens, request.prompt_token_count) };
    let mut parsed = GenerateRequest::new(
        token_ids.iter().copied().map(TokenId).collect(),
        request.max_output_tokens,
    );
    parsed.priority = request.priority;
    parsed.cache_policy = parse_cache_policy(request.cache_policy)?;
    parsed.validate().map_err(|error| error.to_string())?;
    Ok(parsed)
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_config_init(
    out_config: *mut RvllmAppleEngineConfig,
) -> RvllmAppleStatus {
    if out_config.is_null() {
        return RVLLM_APPLE_INVALID_ARGUMENT;
    }
    if catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: checked non-null and documented as writable by caller.
        unsafe { ptr::write(out_config, default_c_config()) };
    }))
    .is_err()
    {
        return RVLLM_APPLE_INTERNAL_ERROR;
    }
    RVLLM_APPLE_OK
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_config_v2_init(
    out_config: *mut RvllmAppleEngineConfigV2,
) -> RvllmAppleStatus {
    if out_config.is_null() {
        return RVLLM_APPLE_INVALID_ARGUMENT;
    }
    if catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: checked non-null and documented as writable by caller.
        unsafe { ptr::write(out_config, default_c_config_v2()) };
    }))
    .is_err()
    {
        return RVLLM_APPLE_INTERNAL_ERROR;
    }
    RVLLM_APPLE_OK
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_config_v3_init(
    out_config: *mut RvllmAppleEngineConfigV3,
) -> RvllmAppleStatus {
    if out_config.is_null() {
        return RVLLM_APPLE_INVALID_ARGUMENT;
    }
    if catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: checked non-null and documented as writable by caller.
        unsafe { ptr::write(out_config, default_c_config_v3()) };
    }))
    .is_err()
    {
        return RVLLM_APPLE_INTERNAL_ERROR;
    }
    RVLLM_APPLE_OK
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_generate_request_init(
    out_request: *mut RvllmAppleGenerateRequest,
) -> RvllmAppleStatus {
    if out_request.is_null() {
        return RVLLM_APPLE_INVALID_ARGUMENT;
    }
    let request = RvllmAppleGenerateRequest {
        abi_version: RVLLM_APPLE_ABI_VERSION,
        struct_size: size_of::<RvllmAppleGenerateRequest>() as u32,
        prompt_tokens: ptr::null(),
        prompt_token_count: 0,
        max_output_tokens: 1,
        priority: 128,
        _reserved: [0; 3],
        cache_policy: 1,
    };
    // SAFETY: checked non-null and documented as writable by caller.
    unsafe { ptr::write(out_request, request) };
    RVLLM_APPLE_OK
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_create(
    config: *const RvllmAppleEngineConfig,
    out_engine: *mut *mut RvllmAppleEngine,
    error: *mut RvllmAppleError,
) -> RvllmAppleStatus {
    ffi_status(error, || {
        if config.is_null() || out_engine.is_null() {
            return Err((
                RVLLM_APPLE_INVALID_ARGUMENT,
                "config and out_engine are required".to_owned(),
            ));
        }
        // Ensure failure never leaves a stale caller pointer behind.
        unsafe { ptr::write(out_engine, ptr::null_mut()) };
        let config = parse_config(unsafe { &*config })
            .map_err(|message| (RVLLM_APPLE_INVALID_ARGUMENT, message))?;
        let handle = spawn_engine(config)?;
        unsafe {
            ptr::write(
                out_engine,
                Box::into_raw(Box::new(RvllmAppleEngine { handle })),
            )
        };
        Ok(())
    })
}

fn spawn_engine(config: AppleEngineConfig) -> Result<EngineHandle, (RvllmAppleStatus, String)> {
    match WORKER_FACTORY.get() {
        Some(InstalledWorkerFactory::Continuous(factory)) => {
            let factory = Arc::clone(factory);
            EngineHandle::spawn_continuous_with_factory(config, move |config| {
                factory(config).map(BoxedContinuousWorker)
            })
        }
        Some(InstalledWorkerFactory::Serial(factory)) => {
            let factory = Arc::clone(factory);
            EngineHandle::spawn_with_factory(config, move |config| factory(config).map(BoxedWorker))
        }
        None => {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                if config.model_package_path.is_none() {
                    return Err((
                        RVLLM_APPLE_BACKEND_UNAVAILABLE,
                        "ABI v2 model package path is required for the built-in Metal worker"
                            .to_owned(),
                    ));
                }
                EngineHandle::spawn_continuous_with_factory(config, |config| {
                    rvllm_runtime::create_embedded_apple_metal_worker(config)
                        .map(|(worker, _health)| BoxedContinuousWorker(worker))
                })
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                return Err((
                    RVLLM_APPLE_BACKEND_UNAVAILABLE,
                    "the built-in Apple Metal worker is unavailable on this target".to_owned(),
                ));
            }
        }
    }
    .map_err(|cause| (status_for(&cause), cause.to_string()))
}

fn spawn_engine_with_persistent_host(
    config: AppleEngineConfig,
    host: PersistentCacheHostConfig,
) -> Result<EngineHandle, (RvllmAppleStatus, String)> {
    if WORKER_FACTORY.get().is_some() {
        return Err((
            RVLLM_APPLE_BACKEND_UNAVAILABLE,
            "encrypted persistent cache is unavailable with a legacy injected worker factory"
                .to_owned(),
        ));
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if config.model_package_path.is_none() {
            return Err((
                RVLLM_APPLE_BACKEND_UNAVAILABLE,
                "ABI v3 model package path is required for the built-in Metal worker".to_owned(),
            ));
        }
        EngineHandle::spawn_continuous_with_persistent_host_factory(config, host, |config, host| {
            rvllm_runtime::apple_continuous_worker::create_embedded_apple_metal_worker_with_persistent_host(
                config, host,
            )
            .map(|(worker, _health)| BoxedContinuousWorker(worker))
        })
        .map_err(|cause| (status_for(&cause), cause.to_string()))
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let _ = (config, host);
        Err((
            RVLLM_APPLE_BACKEND_UNAVAILABLE,
            "the built-in Apple persistent-cache worker is unavailable on this target".to_owned(),
        ))
    }
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_create_v2(
    config: *const RvllmAppleEngineConfigV2,
    out_engine: *mut *mut RvllmAppleEngine,
    error: *mut RvllmAppleError,
) -> RvllmAppleStatus {
    ffi_status(error, || {
        if config.is_null() || out_engine.is_null() {
            return Err((
                RVLLM_APPLE_INVALID_ARGUMENT,
                "config and out_engine are required".to_owned(),
            ));
        }
        // Ensure failure never leaves a stale caller pointer behind.
        unsafe { ptr::write(out_engine, ptr::null_mut()) };
        let config = parse_config_v2(unsafe { &*config })
            .map_err(|message| (RVLLM_APPLE_INVALID_ARGUMENT, message))?;
        let model_package_path = config.model_package_path.as_deref().ok_or((
            RVLLM_APPLE_INVALID_ARGUMENT,
            "model package path is required".to_owned(),
        ))?;
        AppleModelPackage::open(model_package_path).map_err(|cause| {
            (
                RVLLM_APPLE_INVALID_ARGUMENT,
                format!("invalid Apple model package: {cause}"),
            )
        })?;
        if let Some(resources) = config.resource_bundle_path.as_deref() {
            if !resources.is_dir() {
                return Err((
                    RVLLM_APPLE_INVALID_ARGUMENT,
                    format!(
                        "Apple resource bundle is not a directory: {}",
                        resources.display()
                    ),
                ));
            }
        }
        let handle = spawn_engine(config)?;
        unsafe {
            ptr::write(
                out_engine,
                Box::into_raw(Box::new(RvllmAppleEngine { handle })),
            )
        };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_create_v3(
    config: *const RvllmAppleEngineConfigV3,
    out_engine: *mut *mut RvllmAppleEngine,
    error: *mut RvllmAppleError,
) -> RvllmAppleStatus {
    ffi_status(error, || {
        if config.is_null() || out_engine.is_null() {
            return Err((
                RVLLM_APPLE_INVALID_ARGUMENT,
                "config and out_engine are required".to_owned(),
            ));
        }
        // Ensure failure never leaves a stale caller pointer behind.
        unsafe { ptr::write(out_engine, ptr::null_mut()) };
        let parsed = parse_config_v3(unsafe { &*config })
            .map_err(|message| (RVLLM_APPLE_INVALID_ARGUMENT, message))?;
        let config = parsed.config;
        let model_package_path = config.model_package_path.as_deref().ok_or((
            RVLLM_APPLE_INVALID_ARGUMENT,
            "model package path is required".to_owned(),
        ))?;
        AppleModelPackage::open(model_package_path).map_err(|cause| {
            (
                RVLLM_APPLE_INVALID_ARGUMENT,
                format!("invalid Apple model package: {cause}"),
            )
        })?;
        if let Some(resources) = config.resource_bundle_path.as_deref() {
            if !resources.is_dir() {
                return Err((
                    RVLLM_APPLE_INVALID_ARGUMENT,
                    format!(
                        "Apple resource bundle is not a directory: {}",
                        resources.display()
                    ),
                ));
            }
        }
        let handle = match parsed.persistent_cache_host {
            Some(host) => spawn_engine_with_persistent_host(config, host)?,
            None => spawn_engine(config)?,
        };
        unsafe {
            ptr::write(
                out_engine,
                Box::into_raw(Box::new(RvllmAppleEngine { handle })),
            )
        };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_destroy(engine: *mut RvllmAppleEngine) {
    if !engine.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: caller transfers the unique pointer returned by create.
            drop(unsafe { Box::from_raw(engine) });
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_submit(
    engine: *mut RvllmAppleEngine,
    request: *const RvllmAppleGenerateRequest,
    out_request: *mut *mut RvllmAppleRequest,
    error: *mut RvllmAppleError,
) -> RvllmAppleStatus {
    ffi_status(error, || {
        if engine.is_null() || request.is_null() || out_request.is_null() {
            return Err((
                RVLLM_APPLE_INVALID_ARGUMENT,
                "engine, request, and out_request are required".to_owned(),
            ));
        }
        unsafe { ptr::write(out_request, ptr::null_mut()) };
        let request = parse_request(unsafe { &*request })
            .map_err(|message| (RVLLM_APPLE_INVALID_ARGUMENT, message))?;
        let handle = unsafe { &*engine }
            .handle
            .submit(request)
            .map_err(|cause| (status_for(&cause), cause.to_string()))?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let pump_cancelled = Arc::clone(&cancelled);
        let (event_tx, event_rx) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("rvllm-apple-request".to_owned())
            .spawn(move || {
                let mut handle = handle;
                loop {
                    if pump_cancelled.load(Ordering::Acquire) {
                        handle.cancel();
                    }
                    match handle.try_recv() {
                        Ok(Some(event)) => {
                            let terminal = matches!(event, TokenEvent::Finished { .. });
                            if event_tx.send(Ok(event)).is_err() || terminal {
                                break;
                            }
                        }
                        Ok(None) => thread::sleep(Duration::from_millis(1)),
                        Err(cause) => {
                            let _ = event_tx.send(Err(cause));
                            break;
                        }
                    }
                }
            })
            .map_err(|cause| (RVLLM_APPLE_INTERNAL_ERROR, cause.to_string()))?;
        let ffi_request = RvllmAppleRequest {
            cancelled,
            events: Mutex::new(event_rx),
            pending: Mutex::new(None),
        };
        unsafe { ptr::write(out_request, Box::into_raw(Box::new(ffi_request))) };
        Ok(())
    })
}

fn backend_report(report: &rvllm_runtime::BackendReport) -> RvllmAppleBackendReport {
    RvllmAppleBackendReport {
        selected_backend: match report.selected_backend {
            BackendKind::Metal => 1,
            BackendKind::CoreMl => 2,
            BackendKind::MetalPrefillAneDecode => 3,
        },
        cache_tier: match report.cache_tier {
            CacheTier::None => 0,
            CacheTier::Active => 1,
            CacheTier::Hot => 2,
            CacheTier::Warm => 3,
            CacheTier::Persistent => 4,
        },
        matched_cache_tokens: report.matched_cache_tokens,
        saved_prefill_tokens: report.saved_prefill_tokens,
        queue_time_ns: duration_ns(report.queue_time),
        batch_size: report.batch_size,
        padding_tokens: report.padding_tokens,
        prefill_time_ns: duration_ns(report.prefill_time),
        decode_time_ns: duration_ns(report.decode_time),
        resident_memory_bytes: report.resident_memory_bytes,
        thermal_state: match report.thermal_state {
            ThermalState::Unknown => 0,
            ThermalState::Nominal => 1,
            ThermalState::Fair => 2,
            ThermalState::Serious => 3,
            ThermalState::Critical => 4,
        },
        had_fallback: report.fallback.is_some().into(),
        _reserved: [0; 7],
    }
}

fn copy_event(
    event: &TokenEvent,
    out: &mut RvllmAppleTokenEvent,
    text_buffer: *mut c_char,
    text_capacity: usize,
) -> Result<(), usize> {
    *out = RvllmAppleTokenEvent::default();
    match event {
        TokenEvent::Token {
            request_id,
            index,
            token_id,
            text,
        } => {
            out.kind = RVLLM_APPLE_EVENT_TOKEN;
            out.request_id = request_id.raw();
            out.index = *index;
            out.token_id = token_id.raw();
            let bytes = text.as_deref().map(str::as_bytes).unwrap_or_default();
            out.text_length = bytes.len();
            if bytes.len() > text_capacity || (!bytes.is_empty() && text_buffer.is_null()) {
                return Err(bytes.len());
            }
            if !bytes.is_empty() {
                // SAFETY: capacity was checked and caller provided writable storage.
                unsafe {
                    ptr::copy_nonoverlapping(
                        bytes.as_ptr().cast::<c_char>(),
                        text_buffer,
                        bytes.len(),
                    )
                };
            }
        }
        TokenEvent::Finished {
            request_id,
            finish_reason,
            report,
        } => {
            out.kind = RVLLM_APPLE_EVENT_FINISHED;
            out.request_id = request_id.raw();
            out.finish_reason = match finish_reason {
                FinishReason::EndOfSequence => 1,
                FinishReason::Length => 2,
                FinishReason::StopToken => 3,
            };
            out.report = backend_report(report);
        }
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_request_recv(
    request: *mut RvllmAppleRequest,
    out_event: *mut RvllmAppleTokenEvent,
    text_buffer: *mut c_char,
    text_capacity: usize,
    error: *mut RvllmAppleError,
) -> RvllmAppleStatus {
    ffi_status(error, || {
        if request.is_null() || out_event.is_null() {
            return Err((
                RVLLM_APPLE_INVALID_ARGUMENT,
                "request and out_event are required".to_owned(),
            ));
        }
        let request = unsafe { &*request };
        let mut pending = request.pending.lock().map_err(|_| {
            (
                RVLLM_APPLE_INTERNAL_ERROR,
                "request event lock is poisoned".to_owned(),
            )
        })?;
        let event = if let Some(event) = pending.take() {
            event
        } else {
            let events = request.events.lock().map_err(|_| {
                (
                    RVLLM_APPLE_INTERNAL_ERROR,
                    "request receiver lock is poisoned".to_owned(),
                )
            })?;
            match events.recv() {
                Ok(Ok(event)) => event,
                Ok(Err(cause)) => return Err((status_for(&cause), cause.to_string())),
                Err(_) => {
                    return Err((RVLLM_APPLE_END_OF_STREAM, "request stream ended".to_owned()))
                }
            }
        };
        match copy_event(
            &event,
            unsafe { &mut *out_event },
            text_buffer,
            text_capacity,
        ) {
            Ok(()) => Ok(()),
            Err(required) => {
                *pending = Some(event);
                Err((
                    RVLLM_APPLE_BUFFER_TOO_SMALL,
                    format!("text buffer requires {required} bytes"),
                ))
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_request_cancel(
    request: *mut RvllmAppleRequest,
) -> RvllmAppleStatus {
    if request.is_null() {
        return RVLLM_APPLE_INVALID_ARGUMENT;
    }
    unsafe { &*request }
        .cancelled
        .store(true, Ordering::Release);
    RVLLM_APPLE_OK
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_request_destroy(request: *mut RvllmAppleRequest) {
    if !request.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let request = unsafe { Box::from_raw(request) };
            request.cancelled.store(true, Ordering::Release);
            drop(request);
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rvllm_apple_engine_handle_memory_pressure(
    engine: *mut RvllmAppleEngine,
    level: u32,
    error: *mut RvllmAppleError,
) -> RvllmAppleStatus {
    ffi_status(error, || {
        if engine.is_null() {
            return Err((
                RVLLM_APPLE_INVALID_ARGUMENT,
                "engine is required".to_owned(),
            ));
        }
        let pressure = match level {
            0 => MemoryPressure::Normal,
            1 => MemoryPressure::Warning,
            2 => MemoryPressure::Critical,
            _ => {
                return Err((
                    RVLLM_APPLE_INVALID_ARGUMENT,
                    "invalid memory pressure level".to_owned(),
                ))
            }
        };
        unsafe { &*engine }
            .handle
            .handle_memory_pressure(pressure)
            .map_err(|cause| (status_for(&cause), cause.to_string()))
    })
}

#[no_mangle]
pub extern "C" fn rvllm_apple_abi_version() -> u32 {
    RVLLM_APPLE_ABI_VERSION
}

/// Returns the newest engine-configuration ABI understood by this library.
///
/// `rvllm_apple_abi_version` remains the legacy base-ABI query so existing
/// hosts do not change behavior. New hosts should use this function for
/// feature discovery and then call the matching versioned initializer/create
/// entry points.
#[no_mangle]
pub extern "C" fn rvllm_apple_latest_abi_version() -> u32 {
    RVLLM_APPLE_ABI_VERSION_V3
}

#[no_mangle]
pub extern "C" fn rvllm_apple_staticlib_anchor() -> *const c_void {
    rvllm_apple_abi_version as *const c_void
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_runtime::{BackendReport, GenerationOutcome};
    use std::collections::HashSet;

    fn message(error: &RvllmAppleError) -> String {
        let bytes: Vec<u8> = error
            .message
            .iter()
            .copied()
            .take_while(|value| *value != 0)
            .map(|value| value as u8)
            .collect();
        String::from_utf8(bytes).unwrap()
    }

    #[test]
    fn config_and_request_initializers_are_versioned() {
        assert_eq!(rvllm_apple_abi_version(), RVLLM_APPLE_ABI_VERSION);
        assert_eq!(rvllm_apple_latest_abi_version(), RVLLM_APPLE_ABI_VERSION_V3);

        let mut config = default_c_config();
        config.abi_version = 0;
        assert_eq!(
            unsafe { rvllm_apple_engine_config_init(&mut config) },
            RVLLM_APPLE_OK
        );
        assert_eq!(config.abi_version, RVLLM_APPLE_ABI_VERSION);
        assert_eq!(
            config.struct_size as usize,
            size_of::<RvllmAppleEngineConfig>()
        );

        let mut config_v2 = default_c_config_v2();
        config_v2.abi_version = 0;
        assert_eq!(
            unsafe { rvllm_apple_engine_config_v2_init(&mut config_v2) },
            RVLLM_APPLE_OK
        );
        assert_eq!(config_v2.abi_version, RVLLM_APPLE_ABI_VERSION_V2);
        assert_eq!(
            config_v2.struct_size as usize,
            size_of::<RvllmAppleEngineConfigV2>()
        );

        let mut config_v3 = default_c_config_v3();
        config_v3.abi_version = 0;
        assert_eq!(
            unsafe { rvllm_apple_engine_config_v3_init(&mut config_v3) },
            RVLLM_APPLE_OK
        );
        assert_eq!(config_v3.abi_version, RVLLM_APPLE_ABI_VERSION_V3);
        assert_eq!(
            config_v3.struct_size as usize,
            size_of::<RvllmAppleEngineConfigV3>()
        );
        assert!(config_v3.persistent_cache_root.is_null());
        assert!(config_v3.persistent_cache_key.is_null());

        let mut request = RvllmAppleGenerateRequest {
            abi_version: 0,
            struct_size: 0,
            prompt_tokens: ptr::null(),
            prompt_token_count: 0,
            max_output_tokens: 0,
            priority: 0,
            _reserved: [0; 3],
            cache_policy: 0,
        };
        assert_eq!(
            unsafe { rvllm_apple_generate_request_init(&mut request) },
            RVLLM_APPLE_OK
        );
        assert_eq!(request.abi_version, RVLLM_APPLE_ABI_VERSION);
        assert_eq!(
            request.struct_size as usize,
            size_of::<RvllmAppleGenerateRequest>()
        );
    }

    #[test]
    fn v3_layout_is_an_append_only_extension_of_v2() {
        macro_rules! assert_prefix_offset {
            ($field:ident) => {
                assert_eq!(
                    std::mem::offset_of!(RvllmAppleEngineConfigV2, $field),
                    std::mem::offset_of!(RvllmAppleEngineConfigV3, $field),
                    concat!("v3 moved the v2 field ", stringify!($field))
                );
            };
        }
        assert_prefix_offset!(abi_version);
        assert_prefix_offset!(struct_size);
        assert_prefix_offset!(backend_policy);
        assert_prefix_offset!(cache_policy);
        assert_prefix_offset!(workload_profile);
        assert_prefix_offset!(memory_profile);
        assert_prefix_offset!(maximum_concurrency);
        assert_prefix_offset!(ingress_queue_capacity);
        assert_prefix_offset!(event_queue_capacity);
        assert_prefix_offset!(persistent_cache_consent);
        assert_prefix_offset!(_reserved);
        assert_prefix_offset!(hot_cache_bytes);
        assert_prefix_offset!(warm_cache_bytes);
        assert_prefix_offset!(persistent_cache_bytes);
        assert_prefix_offset!(model_package_path);
        assert_prefix_offset!(model_package_path_length);
        assert_prefix_offset!(resource_bundle_path);
        assert_prefix_offset!(resource_bundle_path_length);
        assert_eq!(
            std::mem::offset_of!(RvllmAppleEngineConfigV3, persistent_cache_root),
            size_of::<RvllmAppleEngineConfigV2>()
        );
    }

    #[test]
    fn parsers_reject_nonzero_reserved_bytes_for_every_public_abi() {
        let mut config = default_c_config();
        config._reserved[6] = 1;
        assert!(parse_config(&config)
            .expect_err("v1 reserved bytes must fail closed")
            .contains("reserved bytes"));

        let model = b"/tmp/model";
        let mut config_v2 = default_c_config_v2();
        config_v2.model_package_path = model.as_ptr();
        config_v2.model_package_path_length = model.len();
        config_v2._reserved[0] = 1;
        assert!(parse_config_v2(&config_v2)
            .expect_err("v2 reserved bytes must fail closed")
            .contains("reserved bytes"));

        let mut config_v3 = default_c_config_v3();
        config_v3.model_package_path = model.as_ptr();
        config_v3.model_package_path_length = model.len();
        config_v3._reserved[3] = 1;
        let v3_error = match parse_config_v3(&config_v3) {
            Err(error) => error,
            Ok(_) => panic!("v3 reserved bytes must fail closed"),
        };
        assert!(v3_error.contains("reserved bytes"));

        let tokens = [7_u32];
        let request = RvllmAppleGenerateRequest {
            abi_version: RVLLM_APPLE_ABI_VERSION,
            struct_size: size_of::<RvllmAppleGenerateRequest>() as u32,
            prompt_tokens: tokens.as_ptr(),
            prompt_token_count: tokens.len(),
            max_output_tokens: 1,
            priority: 128,
            _reserved: [0, 1, 0],
            cache_policy: 1,
        };
        assert!(parse_request(&request)
            .expect_err("request reserved bytes must fail closed")
            .contains("reserved bytes"));
    }

    #[test]
    fn null_arguments_are_rejected_without_panicking() {
        assert_eq!(
            unsafe { rvllm_apple_engine_config_init(ptr::null_mut()) },
            RVLLM_APPLE_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { rvllm_apple_engine_config_v2_init(ptr::null_mut()) },
            RVLLM_APPLE_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { rvllm_apple_engine_config_v3_init(ptr::null_mut()) },
            RVLLM_APPLE_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { rvllm_apple_request_cancel(ptr::null_mut()) },
            RVLLM_APPLE_INVALID_ARGUMENT
        );
    }

    #[test]
    fn v2_config_copies_required_model_and_optional_resource_paths() {
        let model = b"/tmp/model package";
        let resources = b"/tmp/resources";
        let mut config = default_c_config_v2();
        config.model_package_path = model.as_ptr();
        config.model_package_path_length = model.len();
        config.resource_bundle_path = resources.as_ptr();
        config.resource_bundle_path_length = resources.len();

        let parsed = parse_config_v2(&config).expect("parse v2 config");
        assert_eq!(
            parsed.model_package_path.as_deref(),
            Some(std::path::Path::new("/tmp/model package"))
        );
        assert_eq!(
            parsed.resource_bundle_path.as_deref(),
            Some(std::path::Path::new("/tmp/resources"))
        );
    }

    #[test]
    fn v2_config_rejects_missing_model_path() {
        let config = default_c_config_v2();
        assert!(parse_config_v2(&config)
            .expect_err("v2 requires a model package")
            .contains("model package path"));
    }

    #[test]
    fn legacy_configs_cannot_enable_t3_without_the_v3_capability() {
        let mut config = default_c_config();
        config.cache_policy = 2;
        config.persistent_cache_consent = 1;
        config.persistent_cache_bytes = 1024;
        assert!(parse_config(&config)
            .expect_err("v1 must not enable T3")
            .contains("ABI v3"));

        let model = b"/tmp/model";
        let mut config_v2 = default_c_config_v2();
        config_v2.model_package_path = model.as_ptr();
        config_v2.model_package_path_length = model.len();
        config_v2.cache_policy = 2;
        config_v2.persistent_cache_consent = 1;
        config_v2.persistent_cache_bytes = 1024;
        assert!(parse_config_v2(&config_v2)
            .expect_err("v2 must not enable T3")
            .contains("ABI v3"));
    }

    #[test]
    fn v3_copies_exact_key_and_bounded_root_into_runtime_config() {
        let model = b"/tmp/model";
        // macOS temp_dir() may start with the /var symlink. The production
        // capability deliberately rejects symlinks in every path component.
        let temp_root = std::fs::canonicalize(std::env::temp_dir())
            .expect("resolve the fixture temporary parent");
        let root = temp_root.join(format!(
            "rvllm-ffi-t3-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir(&root).expect("create private cache root");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700))
                .expect("make cache root private");
        }
        let root_string = root.to_str().expect("UTF-8 cache root");
        let namespace = b"local-user";
        let mut source_key = [0x5a_u8; RVLLM_APPLE_PERSISTENT_CACHE_KEY_BYTES];
        let mut config = default_c_config_v3();
        config.model_package_path = model.as_ptr();
        config.model_package_path_length = model.len();
        config.cache_policy = 2;
        config.persistent_cache_consent = 1;
        config.persistent_cache_bytes = 64 * 1024 * 1024;
        config.persistent_cache_root = root_string.as_ptr();
        config.persistent_cache_root_length = root_string.len();
        config.persistent_cache_key = source_key.as_ptr();
        config.persistent_cache_key_length = source_key.len();
        config.cache_namespace = namespace.as_ptr();
        config.cache_namespace_length = namespace.len();

        let parsed = parse_config_v3(&config).expect("parse enabled v3 config");
        source_key.fill(0);
        assert_eq!(
            parsed.config.model_package_path.as_deref(),
            Some(std::path::Path::new("/tmp/model"))
        );
        let host = parsed
            .persistent_cache_host
            .expect("copied persistent host capability");
        assert_eq!(host.root, root);
        assert_eq!(host.quota_bytes, 64 * 1024 * 1024);
        assert_eq!(host.max_record_bytes, 64 * 1024 * 1024);
        assert_eq!(host.cache_namespace, "local-user");
        assert!(
            host.key.is_some(),
            "the copied capability retains owned key material after the caller clears its buffer"
        );
        drop(host);
        std::fs::remove_dir(&root).expect("remove private cache root");
    }

    #[test]
    fn v3_persistent_capability_fails_closed_on_incomplete_or_unbounded_inputs() {
        let model = b"/tmp/model";
        let root = b"/tmp/rvllm-t3";
        let namespace = b"local-user";
        let key = [0x7c_u8; RVLLM_APPLE_PERSISTENT_CACHE_KEY_BYTES];
        let mut config = default_c_config_v3();
        config.model_package_path = model.as_ptr();
        config.model_package_path_length = model.len();
        config.cache_policy = 2;
        config.persistent_cache_consent = 1;
        config.persistent_cache_bytes = 1024;
        config.persistent_cache_root = root.as_ptr();
        config.persistent_cache_root_length = root.len();
        config.persistent_cache_key = key.as_ptr();
        config.persistent_cache_key_length = key.len() - 1;
        config.cache_namespace = namespace.as_ptr();
        config.cache_namespace_length = namespace.len();
        assert!(parse_config_v3(&config).is_err());

        let relative = b"relative/cache";
        config.persistent_cache_key_length = key.len();
        config.persistent_cache_root = relative.as_ptr();
        config.persistent_cache_root_length = relative.len();
        assert!(parse_config_v3(&config).is_err());

        config.persistent_cache_root = root.as_ptr();
        config.persistent_cache_root_length = root.len();
        config.persistent_cache_bytes = RVLLM_APPLE_MAX_PERSISTENT_CACHE_BYTES + 1;
        assert!(parse_config_v3(&config).is_err());

        config.cache_policy = 1;
        config.persistent_cache_consent = 0;
        config.persistent_cache_bytes = 0;
        assert!(parse_config_v3(&config).is_err());
    }

    #[test]
    fn missing_worker_factory_fails_closed() {
        let config = default_c_config();
        let mut engine = 1usize as *mut RvllmAppleEngine;
        let mut error = RvllmAppleError::default();
        let status = unsafe { rvllm_apple_engine_create(&config, &mut engine, &mut error) };
        assert_eq!(status, RVLLM_APPLE_BACKEND_UNAVAILABLE);
        assert!(engine.is_null());
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let expected = "model package path is required";
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let expected = "the built-in Apple Metal worker is unavailable on this target";
        let detail = message(&error);
        assert!(detail.contains(expected), "unexpected backend error: {detail}");
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn built_in_worker_rejects_an_unvalidated_package_without_fallback() {
        let mut config = AppleEngineConfig::default();
        config.model_package_path = Some(std::env::temp_dir().join(format!(
            "rvllm-missing-model-package-{}",
            std::process::id()
        )));
        let (status, detail) =
            spawn_engine(config).expect_err("missing package must fail worker creation");
        assert_eq!(status, RVLLM_APPLE_BACKEND_UNAVAILABLE);
        assert!(
            detail.contains("validate embedded Apple model package"),
            "unexpected fail-closed detail: {detail}"
        );
    }

    #[test]
    fn event_text_is_lossless_and_retryable() {
        let event = TokenEvent::Token {
            request_id: rvllm_core::ReqId(7),
            index: 3,
            token_id: TokenId(42),
            text: Some(Arc::from("hello world")),
        };
        let mut out = RvllmAppleTokenEvent::default();
        assert_eq!(copy_event(&event, &mut out, ptr::null_mut(), 0), Err(11));
        assert_eq!(out.text_length, 11);
        let mut bytes = [0i8; 11];
        copy_event(&event, &mut out, bytes.as_mut_ptr(), bytes.len()).unwrap();
        assert_eq!(bytes.map(|value| value as u8), *b"hello world");
    }

    #[test]
    fn checked_in_swift_header_matches_canonical_header() {
        assert_eq!(
            include_str!("../include/rvllm_apple.h"),
            include_str!("../../../apple/AppleInference/Sources/CRvllmApple/include/rvllm_apple.h")
        );
    }

    #[test]
    fn memory_pressure_ffi_accepts_explicit_recovery_and_rejects_unknown_levels() {
        let engine = injected_engine();
        let mut error = RvllmAppleError::default();
        for level in [0, 1, 2] {
            assert_eq!(
                unsafe { rvllm_apple_engine_handle_memory_pressure(engine, level, &mut error) },
                RVLLM_APPLE_OK
            );
        }
        assert_eq!(
            unsafe { rvllm_apple_engine_handle_memory_pressure(engine, 3, &mut error) },
            RVLLM_APPLE_INVALID_ARGUMENT
        );
        assert!(message(&error).contains("invalid memory pressure level"));
        unsafe { rvllm_apple_engine_destroy(engine) };
    }

    struct InjectedTestWorker;

    impl InferenceWorker for InjectedTestWorker {
        fn generate(
            &mut self,
            _request: &GenerateRequest,
            output: &mut dyn TokenEmitter,
        ) -> Result<GenerationOutcome, InferenceError> {
            output.emit_token(TokenId(99), Some(Arc::from("héllo 👋")))?;
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

    fn injected_engine() -> *mut RvllmAppleEngine {
        let handle = EngineHandle::spawn_with_factory(AppleEngineConfig::default(), |_| {
            Ok(InjectedTestWorker)
        })
        .unwrap();
        Box::into_raw(Box::new(RvllmAppleEngine { handle }))
    }

    #[derive(Default)]
    struct InjectedContinuousWorker {
        admitted: HashSet<rvllm_core::ReqId>,
        emitted: HashSet<rvllm_core::ReqId>,
    }

    impl ContinuousInferenceWorker for InjectedContinuousWorker {
        fn admit(
            &mut self,
            request_id: rvllm_core::ReqId,
            _request: &GenerateRequest,
            _cancellation: &std::sync::atomic::AtomicBool,
        ) -> Result<(), InferenceError> {
            if !self.admitted.insert(request_id) {
                return Err(InferenceError::worker_failed("duplicate test request"));
            }
            Ok(())
        }

        fn step(
            &mut self,
            active: &[ActiveRequest<'_>],
        ) -> Result<Vec<ContinuousStepOutput>, InferenceError> {
            Ok(active
                .iter()
                .map(|request| {
                    if self.emitted.insert(request.request_id) {
                        ContinuousStepOutput::Token {
                            request_id: request.request_id,
                            token_id: TokenId(77),
                            text: Some(Arc::from("ok")),
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

        fn abort(&mut self, request_id: rvllm_core::ReqId) {
            self.admitted.remove(&request_id);
            self.emitted.remove(&request_id);
        }

        fn handle_memory_pressure(
            &mut self,
            _pressure: MemoryPressure,
        ) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    #[test]
    fn injected_continuous_worker_streams_through_the_same_c_abi() {
        let handle =
            EngineHandle::spawn_continuous_with_factory(AppleEngineConfig::default(), |_| {
                Ok(InjectedContinuousWorker::default())
            })
            .unwrap();
        let engine = Box::into_raw(Box::new(RvllmAppleEngine { handle }));
        let prompt = [17u32];
        let mut raw_request = RvllmAppleGenerateRequest {
            abi_version: 0,
            struct_size: 0,
            prompt_tokens: ptr::null(),
            prompt_token_count: 0,
            max_output_tokens: 0,
            priority: 0,
            _reserved: [0; 3],
            cache_policy: 0,
        };
        unsafe { rvllm_apple_generate_request_init(&mut raw_request) };
        raw_request.prompt_tokens = prompt.as_ptr();
        raw_request.prompt_token_count = prompt.len();
        let mut request = ptr::null_mut();
        let mut error = RvllmAppleError::default();
        assert_eq!(
            unsafe { rvllm_apple_engine_submit(engine, &raw_request, &mut request, &mut error) },
            RVLLM_APPLE_OK
        );

        let mut event = RvllmAppleTokenEvent::default();
        let mut text = [0i8; 2];
        assert_eq!(
            unsafe {
                rvllm_apple_request_recv(
                    request,
                    &mut event,
                    text.as_mut_ptr(),
                    text.len(),
                    &mut error,
                )
            },
            RVLLM_APPLE_OK
        );
        assert_eq!(event.kind, RVLLM_APPLE_EVENT_TOKEN);
        assert_eq!(event.token_id, 77);
        assert_eq!(text.map(|byte| byte as u8), *b"ok");

        assert_eq!(
            unsafe {
                rvllm_apple_request_recv(request, &mut event, ptr::null_mut(), 0, &mut error)
            },
            RVLLM_APPLE_OK
        );
        assert_eq!(event.kind, RVLLM_APPLE_EVENT_FINISHED);

        unsafe {
            rvllm_apple_request_destroy(request);
            rvllm_apple_engine_destroy(engine);
        }
    }

    #[test]
    fn injected_unit_worker_streams_through_c_abi_without_text_loss() {
        let engine = injected_engine();
        let prompt = [17u32];
        let mut raw_request = RvllmAppleGenerateRequest {
            abi_version: 0,
            struct_size: 0,
            prompt_tokens: ptr::null(),
            prompt_token_count: 0,
            max_output_tokens: 0,
            priority: 0,
            _reserved: [0; 3],
            cache_policy: 0,
        };
        unsafe { rvllm_apple_generate_request_init(&mut raw_request) };
        raw_request.prompt_tokens = prompt.as_ptr();
        raw_request.prompt_token_count = prompt.len();
        let mut request = ptr::null_mut();
        let mut error = RvllmAppleError::default();
        assert_eq!(
            unsafe { rvllm_apple_engine_submit(engine, &raw_request, &mut request, &mut error) },
            RVLLM_APPLE_OK
        );

        let mut event = RvllmAppleTokenEvent::default();
        assert_eq!(
            unsafe {
                rvllm_apple_request_recv(request, &mut event, ptr::null_mut(), 0, &mut error)
            },
            RVLLM_APPLE_BUFFER_TOO_SMALL
        );
        let mut text = vec![0i8; event.text_length];
        assert_eq!(
            unsafe {
                rvllm_apple_request_recv(
                    request,
                    &mut event,
                    text.as_mut_ptr(),
                    text.len(),
                    &mut error,
                )
            },
            RVLLM_APPLE_OK
        );
        assert_eq!(event.kind, RVLLM_APPLE_EVENT_TOKEN);
        assert_eq!(event.token_id, 99);
        assert_eq!(
            text.into_iter().map(|byte| byte as u8).collect::<Vec<_>>(),
            "héllo 👋".as_bytes()
        );

        assert_eq!(
            unsafe {
                rvllm_apple_request_recv(request, &mut event, ptr::null_mut(), 0, &mut error)
            },
            RVLLM_APPLE_OK
        );
        assert_eq!(event.kind, RVLLM_APPLE_EVENT_FINISHED);
        assert_eq!(event.finish_reason, 2);
        assert_eq!(event.report.selected_backend, 1);

        unsafe {
            rvllm_apple_request_destroy(request);
            rvllm_apple_engine_destroy(engine);
        }
    }

    struct CancellationTestWorker;

    impl InferenceWorker for CancellationTestWorker {
        fn generate(
            &mut self,
            _request: &GenerateRequest,
            output: &mut dyn TokenEmitter,
        ) -> Result<GenerationOutcome, InferenceError> {
            while !output.is_cancelled() {
                thread::sleep(Duration::from_millis(1));
            }
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
    fn cancellation_crosses_the_abi_while_receive_is_pending() {
        let handle = EngineHandle::spawn_with_factory(AppleEngineConfig::default(), |_| {
            Ok(CancellationTestWorker)
        })
        .unwrap();
        let engine = Box::into_raw(Box::new(RvllmAppleEngine { handle }));
        let prompt = [1u32];
        let raw_request = RvllmAppleGenerateRequest {
            abi_version: RVLLM_APPLE_ABI_VERSION,
            struct_size: size_of::<RvllmAppleGenerateRequest>() as u32,
            prompt_tokens: prompt.as_ptr(),
            prompt_token_count: prompt.len(),
            max_output_tokens: 1,
            priority: 128,
            _reserved: [0; 3],
            cache_policy: 1,
        };
        let mut request = ptr::null_mut();
        let mut error = RvllmAppleError::default();
        assert_eq!(
            unsafe { rvllm_apple_engine_submit(engine, &raw_request, &mut request, &mut error) },
            RVLLM_APPLE_OK
        );
        assert_eq!(
            unsafe { rvllm_apple_request_cancel(request) },
            RVLLM_APPLE_OK
        );
        let mut event = RvllmAppleTokenEvent::default();
        assert_eq!(
            unsafe {
                rvllm_apple_request_recv(request, &mut event, ptr::null_mut(), 0, &mut error)
            },
            RVLLM_APPLE_CANCELLED
        );
        assert!(message(&error).contains("cancelled"));
        unsafe {
            rvllm_apple_request_destroy(request);
            rvllm_apple_engine_destroy(engine);
        }
    }
}
