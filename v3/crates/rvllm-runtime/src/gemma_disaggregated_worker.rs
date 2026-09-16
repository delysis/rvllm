//! Serial Gemma 4 12B owner: Metal prefill, then private ANE decode.
#![forbid(unsafe_code)]

#[cfg(test)]
#[path = "gemma_disaggregated_worker_live_tests.rs"]
mod live_tests;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use rvllm_apple::{AppleBackend, AppleRuntimePlan, HandoffKind};
use rvllm_apple_metal::{MetalFloatType, MetalKernelOptions, MetalModelLimits};
use rvllm_core::{ReqId, TokenId};
use sha2::{Digest, Sha256};

use crate::ane_prefill::AneDecodeStart;
use crate::apple_bridge::handoff_from_prefill_plan_with_paged_kv;
use crate::apple_measurement::{metal_counter_capabilities, PowerMonitor};
use crate::apple_metal_backend::{ModelMetalBackend, ModelMetalOptions};
use crate::gemma_ane_decode::{AneWeightPlan, GemmaAneDecode};
use crate::prompt_cache::MemoryPressure;
use crate::request_api::{
    AppleEngineConfig, BackendKind, BackendPolicy, BackendReport, CachePolicy, EngineHandle,
    FinishReason, GenerateRequest, GenerationOutcome, InferenceError, InferenceWorker,
    TokenEmitter,
};
use crate::{BatchPlan, PagedKvConfig, PagedKvPool};

/// Native checkpoint and qualified precompiled BF16 library. Compiling missing
/// ANE entries is disabled by default; provisioning remains an explicit step.
#[derive(Clone, Debug)]
pub struct GemmaDisaggregatedConfig {
    pub model_dir: PathBuf,
    pub metallib_bf16: PathBuf,
    pub context_capacity: usize,
    pub compile_budget: usize,
    /// Optional append-only power observations; the file must not exist.
    pub power_journal: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GemmaWorkerState {
    Starting,
    Ready,
    Failed,
    ReleasedForMemoryPressure,
    Stopped,
}

#[derive(Clone, Debug)]
pub struct GemmaPreparationTimes {
    pub metal: Duration,
    pub ane: Duration,
    pub compiler_calls: usize,
    pub measurement: serde_json::Value,
}

struct Health {
    state: AtomicU8,
    preparation: OnceLock<GemmaPreparationTimes>,
}

/// Plain data shared with frontends. No private accelerator object crosses
/// the owner thread boundary, including on construction or destruction.
#[derive(Clone)]
pub struct GemmaDisaggregatedHealth(Arc<Health>);

impl GemmaDisaggregatedHealth {
    pub fn state(&self) -> GemmaWorkerState {
        match self.0.state.load(Ordering::Acquire) {
            0 => GemmaWorkerState::Starting,
            1 => GemmaWorkerState::Ready,
            3 => GemmaWorkerState::ReleasedForMemoryPressure,
            4 => GemmaWorkerState::Stopped,
            _ => GemmaWorkerState::Failed,
        }
    }

    pub fn preparation(&self) -> Option<GemmaPreparationTimes> {
        self.0.preparation.get().cloned()
    }

    fn set_state(&self, state: GemmaWorkerState) {
        self.0.state.store(state as u8, Ordering::Release);
    }
}

impl GemmaDisaggregatedConfig {
    pub fn new(model_dir: PathBuf, metallib_bf16: PathBuf) -> Self {
        Self {
            model_dir,
            metallib_bf16,
            context_capacity: 1024,
            compile_budget: 0,
            power_journal: None,
        }
    }

    fn validate(&self, engine: &AppleEngineConfig) -> Result<(), InferenceError> {
        engine.validate()?;
        if engine.backend_policy != BackendPolicy::MetalPrefillAneDecode {
            return Err(invalid_config(
                "backend_policy",
                "requires explicit Metal prefill / ANE decode policy",
            ));
        }
        if engine.maximum_concurrency != 1 {
            return Err(invalid_config(
                "maximum_concurrency",
                "disaggregated inference is serial",
            ));
        }
        if engine.cache_policy != CachePolicy::Disabled {
            return Err(invalid_config(
                "cache_policy",
                "cross-request prompt caching is not supported",
            ));
        }
        if !matches!(self.context_capacity, 64 | 1024) || self.compile_budget > 16 {
            return Err(invalid_config(
                "gemma_disaggregated",
                "requires capacity 64 or 1024 and compile budget 0..=16",
            ));
        }
        Ok(())
    }
}

fn invalid_config(field: &'static str, reason: &'static str) -> InferenceError {
    InferenceError::InvalidConfig { field, reason }
}

fn device_error(error: impl std::fmt::Display) -> InferenceError {
    InferenceError::worker_failed(error.to_string())
}

/// Prepare both devices inside `rvllm-accelerator` and return after both are
/// ready. Cache policy is independent of the ANE compiler cache. Failed or
/// pressure-released owners require a new explicit spawn; no fallback occurs.
pub fn spawn_gemma_disaggregated_worker(
    engine: AppleEngineConfig,
    config: GemmaDisaggregatedConfig,
) -> Result<(EngineHandle, GemmaDisaggregatedHealth), InferenceError> {
    config.validate(&engine)?;
    let health = GemmaDisaggregatedHealth(Arc::new(Health {
        state: AtomicU8::new(GemmaWorkerState::Starting as u8),
        preparation: OnceLock::new(),
    }));
    let owner_health = health.clone();
    let handle = EngineHandle::spawn_with_factory(engine, move |_| {
        let result = MetalAneDevice::prepare(&config);
        match result {
            Ok((device, eos, times)) => {
                let _ = owner_health.0.preparation.set(times);
                owner_health.set_state(GemmaWorkerState::Ready);
                Ok(Worker {
                    device: Some(device),
                    eos,
                    capacity: config.context_capacity,
                    vocab: 262_144,
                    health: owner_health,
                })
            }
            Err(error) => {
                owner_health.set_state(GemmaWorkerState::Failed);
                Err(InferenceError::worker_initialization_failed(
                    error.to_string(),
                ))
            }
        }
    })?;
    Ok((handle, health))
}

// The execution protocol is testable without a driver. The production device
// remains !Send/!Sync through its actual Metal and ANE owners.
trait DisaggregatedDevice: 'static {
    type Seed;
    fn prefill(
        &mut self,
        owner: ReqId,
        prompt: &[TokenId],
    ) -> Result<(Self::Seed, TokenId), InferenceError>;
    fn import(&mut self, seed: Self::Seed) -> Result<(), InferenceError>;
    fn decode(&mut self, token: TokenId) -> Result<(TokenId, usize), InferenceError>;
    fn take_measurement(&mut self) -> Option<serde_json::Value> {
        None
    }
}

struct Worker<D: DisaggregatedDevice> {
    device: Option<D>,
    eos: Vec<u32>,
    capacity: usize,
    vocab: u32,
    health: GemmaDisaggregatedHealth,
}

fn validate_request(
    request: &GenerateRequest,
    capacity: usize,
    vocab: u32,
) -> Result<(), InferenceError> {
    request.validate()?;
    let invalid = |field, reason| InferenceError::InvalidRequest { field, reason };
    if request.cache_policy != CachePolicy::Disabled {
        return Err(invalid(
            "cache_policy",
            "disaggregated requests require disabled prompt caching",
        ));
    }
    if !request.sampling.is_greedy() {
        return Err(invalid(
            "sampling",
            "disaggregated inference currently requires greedy sampling",
        ));
    }
    if request
        .prompt_tokens
        .iter()
        .any(|token| token.raw() >= vocab)
    {
        return Err(invalid(
            "prompt_tokens",
            "token is outside the model vocabulary",
        ));
    }
    if request
        .prompt_tokens
        .len()
        .checked_add(request.max_output_tokens as usize - 1)
        .map_or(true, |total| total > capacity)
    {
        return Err(invalid(
            "max_output_tokens",
            "prompt plus consumed decode inputs exceeds context capacity",
        ));
    }
    Ok(())
}

fn check_cancelled(output: &dyn TokenEmitter) -> Result<(), InferenceError> {
    if output.is_cancelled() {
        Err(InferenceError::RequestCancelled {
            request_id: output.request_id(),
        })
    } else {
        Ok(())
    }
}

impl<D: DisaggregatedDevice> Worker<D> {
    fn generate_inner(
        &mut self,
        request: &GenerateRequest,
        output: &mut dyn TokenEmitter,
    ) -> Result<GenerationOutcome, InferenceError> {
        validate_request(request, self.capacity, self.vocab)?;
        check_cancelled(output)?;
        let device = self
            .device
            .as_mut()
            .ok_or(InferenceError::WorkerUnavailable)?;
        let mut report = BackendReport::new(BackendKind::MetalPrefillAneDecode);
        let started = Instant::now();
        let (seed, mut token) = device.prefill(output.request_id(), &request.prompt_tokens)?;
        report.prefill_time = started.elapsed();
        if token.raw() >= self.vocab {
            return Err(device_error(
                "Metal returned a token outside the vocabulary",
            ));
        }
        check_cancelled(output)?;
        output.emit_token(token, None)?;
        let mut finish_reason = if self.eos.contains(&token.raw()) {
            FinishReason::EndOfSequence
        } else {
            FinishReason::Length
        };
        if finish_reason == FinishReason::EndOfSequence || request.max_output_tokens == 1 {
            return Ok(GenerationOutcome {
                finish_reason,
                report,
            });
        }
        check_cancelled(output)?;
        let started = Instant::now();
        device.import(seed)?;
        report.prefill_time += started.elapsed();
        for index in 1..request.max_output_tokens {
            check_cancelled(output)?;
            let started = Instant::now();
            let (next, position) = device.decode(token)?;
            report.decode_time += started.elapsed();
            if next.raw() >= self.vocab {
                return Err(device_error("ANE returned a token outside the vocabulary"));
            }
            if position != request.prompt_tokens.len() + index as usize - 1 {
                return Err(device_error("ANE returned an unexpected decode position"));
            }
            check_cancelled(output)?;
            token = next;
            output.emit_token(token, None)?;
            if self.eos.contains(&token.raw()) {
                finish_reason = FinishReason::EndOfSequence;
                break;
            }
        }
        Ok(GenerationOutcome {
            finish_reason,
            report,
        })
    }
}

impl<D: DisaggregatedDevice> InferenceWorker for Worker<D> {
    fn generate(
        &mut self,
        request: &GenerateRequest,
        output: &mut dyn TokenEmitter,
    ) -> Result<GenerationOutcome, InferenceError> {
        let mut result = self.generate_inner(request, output);
        if let (Ok(outcome), Some(device)) = (&mut result, &mut self.device) {
            outcome.report.measurement = device.take_measurement();
        }
        if matches!(result, Err(InferenceError::WorkerFailed { .. })) {
            self.health.set_state(GemmaWorkerState::Failed);
            self.device.take();
        }
        result
    }

    fn handle_memory_pressure(&mut self, pressure: MemoryPressure) -> Result<(), InferenceError> {
        // The serialized request API invokes this between requests. No claim
        // of interrupting a synchronous Metal/ANE operation is made.
        if pressure == MemoryPressure::Critical && self.device.is_some() {
            self.health
                .set_state(GemmaWorkerState::ReleasedForMemoryPressure);
            self.device.take();
        }
        Ok(())
    }
}

impl<D: DisaggregatedDevice> Drop for Worker<D> {
    fn drop(&mut self) {
        self.device.take();
        if self.health.state() == GemmaWorkerState::Ready {
            self.health.set_state(GemmaWorkerState::Stopped);
        }
    }
}

struct MetalAneDevice {
    metal: ModelMetalBackend,
    ane: GemmaAneDecode,
    layout_hash: [u8; 32],
    physical_kv_pages: u32,
    monitor: PowerMonitor,
    measurement: serde_json::Value,
}

impl MetalAneDevice {
    fn prepare(
        config: &GemmaDisaggregatedConfig,
    ) -> Result<(Self, Vec<u32>, GemmaPreparationTimes), InferenceError> {
        GemmaAneDecode::validate_configuration(&config.model_dir, config.context_capacity)
            .map_err(device_error)?;
        let eos = rvllm_loader::generation::load_eos_token_ids(&config.model_dir)
            .map_err(device_error)?;
        let header = std::fs::read(config.model_dir.join("config.json")).map_err(device_error)?;
        let layout_hash = Sha256::digest(header).into();
        let monitor = PowerMonitor::start(config.power_journal.as_deref()).map_err(device_error)?;
        let environment = metal_counter_capabilities();
        let plan = AppleRuntimePlan {
            target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple Silicon", 1),
            mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
            ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 3840,
            ane_intermediate_size: 15360,
            ane_num_layers: 48,
            model_layout_hash: layout_hash,
            weights_path: Some(config.model_dir.clone()),
        };
        // This plan initializes Metal only. This concrete owner implements the
        // explicit ANE route; the generic Apple fallback router is never used.
        let options = ModelMetalOptions {
            float_type: MetalFloatType::Bf16,
            limits: MetalModelLimits {
                max_context_tokens: config.context_capacity,
                max_batch_tokens: config.context_capacity,
                max_batch_sequences: 1,
            },
            kernels: MetalKernelOptions {
                prefill_mma32: true,
                prefill_simd_attention: true,
                ..MetalKernelOptions::default()
            },
        };
        let mut metal = ModelMetalBackend::with_options(
            config.model_dir.clone(),
            config.metallib_bf16.clone(),
            options,
        );
        let measured = monitor.begin();
        let started = Instant::now();
        metal.prepare(&plan).map_err(device_error)?;
        let metal_time = started.elapsed();
        let metal_measurement = measured.finish(1);
        let capacity = metal
            .model_capacity()
            .ok_or_else(|| device_error("Metal capacity unavailable"))?;
        if capacity.max_context_tokens != config.context_capacity
            || capacity.max_batch_tokens < config.context_capacity
        {
            return Err(device_error(
                "Metal prepared limits disagree with the worker",
            ));
        }
        let before = rvllm_apple::ane_linear::compile_budget_used();
        let measured = monitor.begin();
        let started = Instant::now();
        let ane = GemmaAneDecode::load_with_compile_budget(
            &config.model_dir,
            config.context_capacity,
            AneWeightPlan::StaticInt8FfnCached,
            config.compile_budget,
        )
        .map_err(device_error)?;
        let times = GemmaPreparationTimes {
            metal: metal_time,
            ane: started.elapsed(),
            compiler_calls: rvllm_apple::ane_linear::compile_budget_used().saturating_sub(before),
            measurement: serde_json::json!({"metal":metal_measurement,"ane":measured.finish(1),"environment":environment}),
        };
        Ok((
            Self {
                metal,
                ane,
                layout_hash,
                physical_kv_pages: capacity.physical_kv_pages,
                monitor,
                measurement: serde_json::Value::Null,
            },
            eos,
            times,
        ))
    }
}

impl DisaggregatedDevice for MetalAneDevice {
    type Seed = AneDecodeStart;

    fn prefill(
        &mut self,
        owner: ReqId,
        prompt: &[TokenId],
    ) -> Result<(AneDecodeStart, TokenId), InferenceError> {
        self.measurement = serde_json::json!({"steps":[]});
        let prompt_len = u32::try_from(prompt.len()).map_err(device_error)?;
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(self.physical_kv_pages, 1))
            .map_err(device_error)?;
        let chain = pool
            .allocate_chain(owner, prompt_len)
            .map_err(device_error)?;
        let batch = BatchPlan::Prefill {
            req_ids: vec![owner],
            prompt_tokens_flat: prompt.to_vec(),
            cu_seqlens_q: vec![0, prompt_len],
            query_start_positions: vec![0],
            context_lens: vec![prompt_len],
            kv_chains: vec![Some(chain)],
        };
        let handoff = handoff_from_prefill_plan_with_paged_kv(
            &batch,
            HandoffKind::MetalPrefillToMetalDecode,
            None,
            &pool,
            self.layout_hash,
        )
        .map_err(device_error)?;
        let before = self.metal.probe_perf_stats();
        let measured = self.monitor.begin();
        let start = self.metal.prefill_for_ane(&handoff).map_err(device_error)?;
        self.measurement["prefill"] = measured.finish(prompt.len());
        self.measurement["metal_gpu_execution_ms"] = start.times.gpu_execution_ms.into();
        let after = self.metal.probe_perf_stats();
        if after.prefill_steps - before.prefill_steps != 1
            || after.decode_steps != before.decode_steps
            || after.command_buffers - before.command_buffers != 1
            || start.req_id != owner
            || start.next_position() != prompt.len()
        {
            return Err(device_error(
                "Metal prefill violated request, position or command-buffer contract",
            ));
        }
        let first = start.first_token;
        Ok((start, first))
    }

    fn import(&mut self, seed: AneDecodeStart) -> Result<(), InferenceError> {
        let measured = self.monitor.begin();
        self.ane.import_prefill(&seed.cache).map_err(device_error)?;
        self.measurement["import"] = measured.finish(1);
        Ok(())
    }

    fn decode(&mut self, token: TokenId) -> Result<(TokenId, usize), InferenceError> {
        let measured = self.monitor.begin();
        let decoded = self.ane.decode(token).map_err(device_error)?;
        let measurement = measured.finish(1);
        let times = decoded.times;
        self.measurement["steps"].as_array_mut().ok_or_else(|| device_error("missing request measurement"))?.push(serde_json::json!({
            "position":decoded.position,"input_token":token.raw(),"next_token":decoded.token.raw(),
            "qkv_ms":times.qkv_ms,"attention_ms":times.attention_ms,"output_ms":times.output_ms,
            "ffn_ms":times.ffn_ms,"vocabulary_ms":times.vocabulary_ms,"host_ms":times.host_ms,
            "total_ms":times.total_ms,"measurement":measurement,
        }));
        Ok((decoded.token, decoded.position))
    }

    fn take_measurement(&mut self) -> Option<serde_json::Value> {
        Some(std::mem::take(&mut self.measurement))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_api::TokenEvent;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::mpsc;

    #[derive(Default)]
    struct State {
        calls: RefCell<Vec<&'static str>>,
        cancelled: Cell<bool>,
        cancel_prefill: Cell<bool>,
        fail_decode: Cell<bool>,
        wrong_position: Cell<bool>,
        invalid_token: Cell<bool>,
        decode_completed: RefCell<Option<mpsc::Sender<usize>>>,
    }

    struct FakeDevice {
        state: Rc<State>,
        position: usize,
    }
    impl DisaggregatedDevice for FakeDevice {
        type Seed = usize;
        fn prefill(
            &mut self,
            _: ReqId,
            prompt: &[TokenId],
        ) -> Result<(usize, TokenId), InferenceError> {
            self.state.calls.borrow_mut().push("prefill");
            if self.state.cancel_prefill.get() {
                self.state.cancelled.set(true);
            }
            Ok((
                prompt.len(),
                TokenId(if self.state.invalid_token.get() {
                    999
                } else {
                    prompt[0].raw() + 1
                }),
            ))
        }
        fn import(&mut self, seed: usize) -> Result<(), InferenceError> {
            self.state.calls.borrow_mut().push("import");
            self.position = seed;
            Ok(())
        }
        fn decode(&mut self, token: TokenId) -> Result<(TokenId, usize), InferenceError> {
            self.state.calls.borrow_mut().push("decode");
            if self.state.fail_decode.get() {
                return Err(device_error("injected device failure"));
            }
            let position = self.position;
            self.position += 1;
            if let Some(sender) = &*self.state.decode_completed.borrow() {
                let _ = sender.send(position);
            }
            Ok((
                TokenId(token.raw() + 1),
                if self.state.wrong_position.get() {
                    position + 1
                } else {
                    position
                },
            ))
        }
    }
    impl Drop for FakeDevice {
        fn drop(&mut self) {
            self.state.calls.borrow_mut().push("drop");
        }
    }

    struct Emitter {
        state: Rc<State>,
        tokens: Vec<u32>,
        cancel_after: usize,
    }
    impl TokenEmitter for Emitter {
        fn request_id(&self) -> ReqId {
            ReqId(7)
        }
        fn is_cancelled(&self) -> bool {
            self.state.cancelled.get()
        }
        fn emit_token(
            &mut self,
            token: TokenId,
            _: Option<Arc<str>>,
        ) -> Result<(), InferenceError> {
            self.tokens.push(token.raw());
            if self.tokens.len() == self.cancel_after {
                self.state.cancelled.set(true);
            }
            Ok(())
        }
    }

    fn health() -> GemmaDisaggregatedHealth {
        GemmaDisaggregatedHealth(Arc::new(Health {
            state: AtomicU8::new(GemmaWorkerState::Ready as u8),
            preparation: OnceLock::new(),
        }))
    }
    fn fixture() -> (Worker<FakeDevice>, Emitter, Rc<State>) {
        let state = Rc::new(State::default());
        (
            Worker {
                device: Some(FakeDevice {
                    state: state.clone(),
                    position: 999,
                }),
                eos: vec![25],
                capacity: 64,
                vocab: 100,
                health: health(),
            },
            Emitter {
                state: state.clone(),
                tokens: vec![],
                cancel_after: usize::MAX,
            },
            state,
        )
    }
    fn request(prompt: Vec<u32>, count: u32) -> GenerateRequest {
        let mut request = GenerateRequest::new(prompt.into_iter().map(TokenId).collect(), count);
        request.cache_policy = CachePolicy::Disabled;
        request
    }

    #[test]
    fn exact_capacity_eos_first_and_length_one_do_not_decode() {
        let (mut worker, mut emitter, state) = fixture();
        let outcome = worker
            .generate(&request(vec![10; 64], 1), &mut emitter)
            .unwrap();
        assert_eq!(outcome.finish_reason, FinishReason::Length);
        assert_eq!(*state.calls.borrow(), ["prefill"]);
        assert_eq!(emitter.tokens, [11]);
        assert!(matches!(
            worker.generate(&request(vec![10; 64], 2), &mut emitter),
            Err(InferenceError::InvalidRequest { .. })
        ));
        emitter.tokens.clear();
        state.calls.borrow_mut().clear();
        let outcome = worker
            .generate(&request(vec![24], 4), &mut emitter)
            .unwrap();
        assert_eq!(outcome.finish_reason, FinishReason::EndOfSequence);
        assert_eq!(emitter.tokens, [25]);
        assert_eq!(*state.calls.borrow(), ["prefill"]);
    }

    #[test]
    fn request_rejections_happen_before_hardware_and_keep_owner() {
        let (mut worker, mut emitter, state) = fixture();
        let mut bad_sampling = request(vec![1], 2);
        bad_sampling.sampling.temperature = 1.0;
        let mut bad_cache = request(vec![1], 2);
        bad_cache.cache_policy = CachePolicy::MemoryOnly;
        for invalid in [
            request(vec![], 1),
            request(vec![1], 0),
            request(vec![100], 1),
            bad_sampling,
            bad_cache,
        ] {
            assert!(matches!(
                worker.generate(&invalid, &mut emitter),
                Err(InferenceError::InvalidRequest { .. })
            ));
        }
        assert!(state.calls.borrow().is_empty());
        assert_eq!(worker.health.state(), GemmaWorkerState::Ready);
    }

    #[test]
    fn cancellation_before_and_after_prefill_emits_nothing() {
        let (mut worker, mut emitter, state) = fixture();
        state.cancelled.set(true);
        assert!(matches!(
            worker.generate(&request(vec![10], 4), &mut emitter),
            Err(InferenceError::RequestCancelled { .. })
        ));
        assert!(state.calls.borrow().is_empty());
        state.cancelled.set(false);
        state.cancel_prefill.set(true);
        assert!(matches!(
            worker.generate(&request(vec![10], 4), &mut emitter),
            Err(InferenceError::RequestCancelled { .. })
        ));
        assert_eq!(*state.calls.borrow(), ["prefill"]);
        assert!(emitter.tokens.is_empty());
        assert_eq!(worker.health.state(), GemmaWorkerState::Ready);
    }

    #[test]
    fn cancelled_decode_then_new_prompt_imports_fresh_position() {
        let (mut worker, mut emitter, state) = fixture();
        emitter.cancel_after = 2;
        assert!(matches!(
            worker.generate(&request(vec![10; 5], 4), &mut emitter),
            Err(InferenceError::RequestCancelled { .. })
        ));
        assert_eq!(emitter.tokens, [11, 12]);
        assert_eq!(*state.calls.borrow(), ["prefill", "import", "decode"]);
        state.cancelled.set(false);
        emitter.cancel_after = usize::MAX;
        emitter.tokens.clear();
        worker
            .generate(&request(vec![20; 2], 3), &mut emitter)
            .unwrap();
        assert_eq!(emitter.tokens, [21, 22, 23]);
        emitter.tokens.clear();
        worker
            .generate(&request(vec![10; 5], 3), &mut emitter)
            .unwrap();
        assert_eq!(emitter.tokens, [11, 12, 13]);
        assert_eq!(worker.health.state(), GemmaWorkerState::Ready);
    }

    #[test]
    fn device_failure_invalid_result_and_position_drift_poison_owner() {
        for failure in 0..3 {
            let (mut worker, mut emitter, state) = fixture();
            match failure {
                0 => state.fail_decode.set(true),
                1 => state.wrong_position.set(true),
                _ => state.invalid_token.set(true),
            }
            assert!(matches!(
                worker.generate(&request(vec![10], 3), &mut emitter),
                Err(InferenceError::WorkerFailed { .. })
            ));
            assert_eq!(worker.health.state(), GemmaWorkerState::Failed);
            assert_eq!(state.calls.borrow().last(), Some(&"drop"));
            let calls = state.calls.borrow().len();
            assert!(matches!(
                worker.generate(&request(vec![10], 3), &mut emitter),
                Err(InferenceError::WorkerUnavailable)
            ));
            assert_eq!(state.calls.borrow().len(), calls);
        }
    }

    #[test]
    fn critical_pressure_releases_once_and_never_reloads_implicitly() {
        let (mut worker, mut emitter, state) = fixture();
        worker
            .handle_memory_pressure(MemoryPressure::Warning)
            .unwrap();
        assert!(state.calls.borrow().is_empty());
        worker
            .handle_memory_pressure(MemoryPressure::Critical)
            .unwrap();
        worker
            .handle_memory_pressure(MemoryPressure::Critical)
            .unwrap();
        assert_eq!(*state.calls.borrow(), ["drop"]);
        assert_eq!(
            worker.health.state(),
            GemmaWorkerState::ReleasedForMemoryPressure
        );
        assert!(matches!(
            worker.generate(&request(vec![10], 1), &mut emitter),
            Err(InferenceError::WorkerUnavailable)
        ));
    }

    #[test]
    fn dropping_backpressured_stream_allows_the_next_queued_request() {
        let (decoded, progress) = mpsc::channel();
        let config = AppleEngineConfig {
            backend_policy: BackendPolicy::MetalPrefillAneDecode,
            cache_policy: CachePolicy::Disabled,
            maximum_concurrency: 1,
            ingress_queue_capacity: 2,
            event_queue_capacity: 1,
            ..AppleEngineConfig::default()
        };
        let engine = EngineHandle::spawn_with_factory(config, move |_| {
            let (worker, _, state) = fixture();
            *state.decode_completed.borrow_mut() = Some(decoded);
            Ok(worker)
        })
        .unwrap();
        // The first request cannot publish its terminal event into this full
        // one-slot queue. Dropping its handle must release the owner to B.
        let mut abandoned = engine.submit(request(vec![10; 5], 4)).unwrap();
        assert!(matches!(
            abandoned.recv().unwrap(),
            TokenEvent::Token {
                token_id: TokenId(11),
                ..
            }
        ));
        // Decode 12 fills the slot. Decode 13 can finish, but cannot emit until
        // a consumer reads or cancels. No sleep or scheduling assumption.
        assert_eq!(progress.recv_timeout(Duration::from_secs(5)).unwrap(), 5);
        assert_eq!(progress.recv_timeout(Duration::from_secs(5)).unwrap(), 6);
        let mut next = engine.submit(request(vec![20; 2], 3)).unwrap();
        drop(abandoned);
        let (done, receipt) = mpsc::channel();
        std::thread::spawn(move || {
            let mut tokens = Vec::new();
            loop {
                match next.recv() {
                    Ok(TokenEvent::Token { token_id, .. }) => tokens.push(token_id.raw()),
                    Ok(TokenEvent::Finished { .. }) => {
                        done.send(Ok(tokens)).unwrap();
                        break;
                    }
                    Err(error) => {
                        done.send(Err(error)).unwrap();
                        break;
                    }
                }
            }
        });
        assert_eq!(
            receipt
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap(),
            [21, 22, 23]
        );
    }

    #[test]
    fn engine_factory_confines_non_send_owner_through_shutdown() {
        struct Owner {
            thread: std::thread::ThreadId,
            dropped: mpsc::Sender<std::thread::ThreadId>,
            _not_send: Rc<()>,
        }
        impl DisaggregatedDevice for Owner {
            type Seed = usize;
            fn prefill(
                &mut self,
                _: ReqId,
                prompt: &[TokenId],
            ) -> Result<(usize, TokenId), InferenceError> {
                assert_eq!(self.thread, std::thread::current().id());
                Ok((prompt.len(), TokenId(2)))
            }
            fn import(&mut self, _: usize) -> Result<(), InferenceError> {
                unreachable!()
            }
            fn decode(&mut self, _: TokenId) -> Result<(TokenId, usize), InferenceError> {
                unreachable!()
            }
        }
        impl Drop for Owner {
            fn drop(&mut self) {
                assert_eq!(self.thread, std::thread::current().id());
                self.dropped.send(self.thread).unwrap();
            }
        }
        let (dropped, receipt) = mpsc::channel();
        let config = AppleEngineConfig {
            backend_policy: BackendPolicy::MetalPrefillAneDecode,
            cache_policy: CachePolicy::Disabled,
            maximum_concurrency: 1,
            event_queue_capacity: 1,
            ..AppleEngineConfig::default()
        };
        let engine = EngineHandle::spawn_with_factory(config, move |_| {
            Ok(Worker {
                device: Some(Owner {
                    thread: std::thread::current().id(),
                    dropped,
                    _not_send: Rc::new(()),
                }),
                eos: vec![2],
                capacity: 64,
                vocab: 100,
                health: health(),
            })
        })
        .unwrap();
        let mut stream = engine.submit(request(vec![1], 5)).unwrap();
        assert!(matches!(
            stream.recv().unwrap(),
            TokenEvent::Token {
                token_id: TokenId(2),
                ..
            }
        ));
        assert!(matches!(
            stream.recv().unwrap(),
            TokenEvent::Finished {
                finish_reason: FinishReason::EndOfSequence,
                ..
            }
        ));
        engine
            .handle_memory_pressure(MemoryPressure::Critical)
            .unwrap();
        assert_ne!(
            receipt.recv_timeout(Duration::from_secs(5)).unwrap(),
            std::thread::current().id()
        );
    }
}
