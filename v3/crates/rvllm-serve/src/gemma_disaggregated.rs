//! HTTP text adapter; device ownership remains entirely in the runtime worker.
#![forbid(unsafe_code)]

use super::{BackendStreamEvent, CompletionBackend, ServerConfig};
use rvllm_core::TokenId;
use rvllm_runtime::gemma_disaggregated_worker::{
    spawn_gemma_disaggregated_worker, GemmaDisaggregatedConfig, GemmaDisaggregatedHealth,
    GemmaWorkerState,
};
use rvllm_runtime::text_generation::{encode_gemma4_user_prompt, IncrementalTextDecoder};
use rvllm_runtime::{
    AppleEngineConfig, BackendPolicy, CachePolicy, EngineHandle, FinishReason, GenerateRequest,
    TokenEvent,
};
use serde_json::{json, Value};
use std::path::PathBuf;

pub(super) struct GemmaBackend {
    engine: EngineHandle,
    health: GemmaDisaggregatedHealth,
    tokenizer: tokenizers::Tokenizer,
    model_dir: PathBuf,
    capacity: usize,
}

impl GemmaBackend {
    pub(super) fn new(config: &ServerConfig) -> Result<Self, String> {
        let metallib = config
            .metallib_bf16
            .clone()
            .ok_or("metal-ane requires an explicit BF16 metallib")?;
        let capacity = config.max_total_tokens.unwrap_or(1024);
        let tokenizer = tokenizers::Tokenizer::from_file(config.model_dir.join("tokenizer.json"))
            .map_err(|error| error.to_string())?;
        // Qualify the exact checkpoint formatter before initializing devices.
        encode_gemma4_user_prompt(&config.model_dir, &tokenizer, "Check formatter")?;
        let mut device = GemmaDisaggregatedConfig::new(config.model_dir.clone(), metallib);
        device.context_capacity = capacity;
        device.compile_budget = config.ane_compile_budget;
        let engine_config = AppleEngineConfig {
            backend_policy: BackendPolicy::MetalPrefillAneDecode,
            cache_policy: CachePolicy::Disabled,
            maximum_concurrency: 1,
            ingress_queue_capacity: 4,
            event_queue_capacity: 16,
            ..AppleEngineConfig::default()
        };
        let (engine, health) = spawn_gemma_disaggregated_worker(engine_config, device)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            engine,
            health,
            tokenizer,
            model_dir: config.model_dir.clone(),
            capacity,
        })
    }

    fn validate_capacity(&self, prompt_length: usize, count: usize) -> Result<(), String> {
        if count == 0
            || prompt_length
                .checked_add(count - 1)
                .filter(|&total| total <= self.capacity)
                .is_none()
        {
            return Err("prompt plus consumed decode inputs exceeds the prepared context".into());
        }
        Ok(())
    }

    fn generate(
        &self,
        prompt: &str,
        count: usize,
        mut emit: impl FnMut(String) -> Result<(), String>,
    ) -> Result<Value, String> {
        let prompt_ids = encode_gemma4_user_prompt(&self.model_dir, &self.tokenizer, prompt)?;
        self.validate_capacity(prompt_ids.len(), count)?;
        let mut request = GenerateRequest::new(
            prompt_ids.iter().copied().map(TokenId).collect(),
            u32::try_from(count).map_err(|error| error.to_string())?,
        );
        request.cache_policy = CachePolicy::Disabled;
        let mut stream = self
            .engine
            .submit(request)
            .map_err(|error| error.to_string())?;
        let mut ids = Vec::with_capacity(count);
        let mut decoder = IncrementalTextDecoder::default();
        let mut streamed = String::new();
        let (reason, report) = loop {
            match stream.recv().map_err(|error| error.to_string())? {
                TokenEvent::Token { token_id, .. } => {
                    ids.push(token_id.raw());
                    if let Some(delta) = decoder
                        .step(&self.tokenizer, token_id.raw())
                        .map_err(|error| error.to_string())?
                    {
                        streamed.push_str(&delta);
                        if !delta.is_empty() {
                            emit(delta)?;
                        }
                    }
                }
                TokenEvent::Finished {
                    finish_reason,
                    report,
                    ..
                } => break (finish_reason, report),
            }
        };
        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|error| error.to_string())?;
        let tail = text
            .strip_prefix(&streamed)
            .ok_or("streamed text differs from complete decoding")?;
        if !tail.is_empty() {
            emit(tail.into())?;
        }
        Ok(
            json!({"schema":"rvllm.gemma12b_disaggregated_http.v1","model_dir":self.model_dir,
            "prompt_token_ids":prompt_ids,"generated_token_ids":ids,"generated_text":text,
            "finish_reason":match reason { FinishReason::Length => "length", _ => "stop" },
            "backend_report":{"selected_route":"metal-prefill-ane-decode","cache_tier":"none","fallback":null,
                "queue_ms":report.queue_time.as_secs_f64()*1000.0,"prefill_including_import_ms":report.prefill_time.as_secs_f64()*1000.0,
                "decode_ms":report.decode_time.as_secs_f64()*1000.0,"measurement":report.measurement}}),
        )
    }
}

impl CompletionBackend for GemmaBackend {
    fn validate_request(&self, prompt: &str, count: usize) -> Result<(), String> {
        let ids = encode_gemma4_user_prompt(&self.model_dir, &self.tokenizer, prompt)?;
        self.validate_capacity(ids.len(), count)
    }
    fn backend_name(&self) -> &'static str {
        "metal-prefill-ane-decode"
    }
    fn model_label(&self) -> String {
        self.model_dir.display().to_string()
    }
    fn shutdown(&self) {
        self.engine.shutdown();
    }
    fn health(&self) -> Value {
        let prepared = self.health.preparation();
        json!({"ready":self.engine.is_available() && self.health.state() == GemmaWorkerState::Ready,
            "state":format!("{:?}",self.health.state()),"prepared_once":true,"context_capacity":self.capacity,
            "maximum_concurrency":1,"ingress_queue_capacity":4,"response_queue_capacity":16,
            "ane_compile_budget_used":prepared.as_ref().map(|p|p.compiler_calls),
            "prepare_ms":prepared.as_ref().map(|p|(p.metal+p.ane).as_secs_f64()*1000.0),
            "prepare_measurement":prepared.map(|p|p.measurement),"cpu_or_gpu_decode_fallback":false})
    }
    fn complete(&self, prompt: &str, count: usize) -> Result<Value, String> {
        self.generate(prompt, count, |_| Ok(()))
    }
    fn stream(
        &self,
        prompt: &str,
        count: usize,
        emit: &mut dyn FnMut(BackendStreamEvent) -> Result<(), String>,
    ) -> Result<(), String> {
        let report = self.generate(prompt, count, |delta| {
            emit(BackendStreamEvent::TextDelta(delta))
        })?;
        emit(BackendStreamEvent::Finished {
            finish_reason: report["finish_reason"].as_str().unwrap_or("unknown").into(),
            report,
        })
    }
}
