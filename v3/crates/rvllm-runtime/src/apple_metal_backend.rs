use half::f16;
use rvllm_apple::{AppleBackend, AppleLaunchKind, AppleLaunchTicket, HandoffCapsule, StepToken};
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError, TokenId};
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::cell::Cell;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::cmp::max;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::path::PathBuf;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::ptr;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::time::{Duration, Instant};

#[cfg(all(feature = "apple", target_os = "macos"))]
const RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV: &str = "RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE";

#[cfg(not(all(feature = "apple", target_os = "macos")))]
#[derive(Debug, Default)]
pub struct RuntimeMetalBackend;

#[cfg(not(all(feature = "apple", target_os = "macos")))]
impl RuntimeMetalBackend {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(not(all(feature = "apple", target_os = "macos")))]
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

#[cfg(all(feature = "apple", target_os = "macos"))]
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
#[cfg(all(feature = "apple", target_os = "macos"))]
use rvllm_apple::RolloutBucket;
#[cfg(all(feature = "apple", target_os = "macos"))]
use rvllm_apple_metal::arena::{MetalBufferArena, MetalRegion};
#[cfg(all(feature = "apple", target_os = "macos"))]
use rvllm_apple_metal::{
    context::MetalContext,
    gemma4_model::{Gemma4MetalState, MetalLayerTraceState},
    kernels,
    layer_forward::{
        metal_finalize_logits_blocking, metal_finalize_logits_encoder_count, metal_forward_layer,
        metal_prepare_ple_inputs, MetalLayerDims, MetalLayerTraceScratch, MetalLayerWeights,
        MetalMetadata, MetalPhase, MetalPlePrepare, MetalScratch,
    },
    pipeline::PipelineCache,
};

#[cfg(all(feature = "apple", target_os = "macos"))]
const METAL_HIDDEN: usize = 256;
#[cfg(all(feature = "apple", target_os = "macos"))]
const METAL_VOCAB: usize = 256;
#[cfg(all(feature = "apple", target_os = "macos"))]
const METAL_MAX_TOKENS: usize = 256;
#[cfg(all(feature = "apple", target_os = "macos"))]
const METAL_EPS: f32 = 1e-5;
#[cfg(all(feature = "apple", target_os = "macos"))]
const METAL_SOFTCAP: f32 = 0.0;
#[cfg(all(feature = "apple", target_os = "macos"))]
const METAL_ARENA_BYTES: usize = 1 * 1024 * 1024;
#[cfg(all(feature = "apple", target_os = "macos"))]
pub const RVLLM_METAL_DEBUG_SYNC_ENV: &str = "RVLLM_METAL_DEBUG_SYNC";
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

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MetalProbePerfStats {
    pub prefill_steps: u64,
    pub decode_steps: u64,
    pub tokens: u64,
    pub command_buffers: u64,
    pub encoders: u64,
    pub forced_waits: u64,
    pub cpu_wall_ns: u64,
    pub last_step_tokens: u64,
    pub last_step_command_buffers: u64,
    pub last_step_encoders: u64,
    pub last_step_forced_waits: u64,
    pub last_step_cpu_wall_ns: u64,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Debug, Default)]
struct MetalProbePerfCounters {
    prefill_steps: Cell<u64>,
    decode_steps: Cell<u64>,
    tokens: Cell<u64>,
    command_buffers: Cell<u64>,
    encoders: Cell<u64>,
    forced_waits: Cell<u64>,
    cpu_wall_ns: Cell<u64>,
    last_step_tokens: Cell<u64>,
    last_step_command_buffers: Cell<u64>,
    last_step_encoders: Cell<u64>,
    last_step_forced_waits: Cell<u64>,
    last_step_cpu_wall_ns: Cell<u64>,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl MetalProbePerfCounters {
    fn clear(&self) {
        self.prefill_steps.set(0);
        self.decode_steps.set(0);
        self.tokens.set(0);
        self.command_buffers.set(0);
        self.encoders.set(0);
        self.forced_waits.set(0);
        self.cpu_wall_ns.set(0);
        self.last_step_tokens.set(0);
        self.last_step_command_buffers.set(0);
        self.last_step_encoders.set(0);
        self.last_step_forced_waits.set(0);
        self.last_step_cpu_wall_ns.set(0);
    }

    fn snapshot(&self) -> MetalProbePerfStats {
        MetalProbePerfStats {
            prefill_steps: self.prefill_steps.get(),
            decode_steps: self.decode_steps.get(),
            tokens: self.tokens.get(),
            command_buffers: self.command_buffers.get(),
            encoders: self.encoders.get(),
            forced_waits: self.forced_waits.get(),
            cpu_wall_ns: self.cpu_wall_ns.get(),
            last_step_tokens: self.last_step_tokens.get(),
            last_step_command_buffers: self.last_step_command_buffers.get(),
            last_step_encoders: self.last_step_encoders.get(),
            last_step_forced_waits: self.last_step_forced_waits.get(),
            last_step_cpu_wall_ns: self.last_step_cpu_wall_ns.get(),
        }
    }

    fn add_command_buffers(&self, count: u64) {
        self.command_buffers
            .set(self.command_buffers.get().saturating_add(count));
    }

    fn add_encoders(&self, count: u64) {
        self.encoders.set(self.encoders.get().saturating_add(count));
    }

    fn add_forced_wait(&self) {
        self.forced_waits
            .set(self.forced_waits.get().saturating_add(1));
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
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn duration_ns_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn metal_debug_sync_enabled() -> bool {
    std::env::var(RVLLM_METAL_DEBUG_SYNC_ENV).ok().as_deref() == Some("1")
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn experimental_metal_kv_int8_enabled() -> bool {
    std::env::var(RVLLM_EXPERIMENTAL_METAL_KV_INT8_ENV)
        .ok()
        .as_deref()
        == Some("1")
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
fn metal_debug_trace_layer() -> Option<usize> {
    let raw = std::env::var(RVLLM_METAL_DEBUG_TRACE_LAYER_ENV).ok()?;
    match raw.parse::<usize>() {
        Ok(layer_idx) => Some(layer_idx),
        Err(err) => {
            eprintln!(
                "metal debug trace: ignoring invalid {RVLLM_METAL_DEBUG_TRACE_LAYER_ENV}={raw:?}: {err}"
            );
            None
        }
    }
}

#[cfg(all(test, feature = "apple", target_os = "macos"))]
fn metal_debug_trace_json_path() -> Option<PathBuf> {
    std::env::var_os(RVLLM_METAL_DEBUG_TRACE_JSON_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
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
    let json = format!(
        "{{\"schema\":\"rvllm.gemma4_metal_layer_trace.v1\",\"op\":\"{op}\",\"phase\":\"{phase_name}\",\"layer\":{layer_idx},\"num_tokens\":{num_tokens},\"summaries\":{{{}}},\"claim\":\"rvLLM Metal layer debug summary only; no final logits, ANE, or production claim.\"}}\n",
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

#[cfg(all(feature = "apple", target_os = "macos"))]
fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "runtime-metal-backend",
        op,
        device: "apple-silicon",
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn model_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "model-metal-backend",
        op,
        device: "apple-silicon",
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
fn large_gemma4_probe_opted_in() -> bool {
    std::env::var(RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(all(feature = "apple", target_os = "macos"))]
#[derive(Debug)]
struct MetalState {
    residual: MetalRegion,
    final_norm: MetalRegion,
    lm_head: MetalRegion,
    logits: MetalRegion,
    normed_hidden: MetalRegion,
    sampled: MetalRegion,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
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

#[cfg(all(feature = "apple", target_os = "macos"))]
pub struct ModelMetalBackend {
    pub model_dir: PathBuf,
    pub prepared: bool,
    pub next_step_id: u64,
    pub last_ticket: Option<u64>,
    pub pending: Option<Vec<StepToken>>,
    ctx: Option<MetalContext>,
    pipelines: Option<PipelineCache>,
    arena: Option<MetalBufferArena>,
    pub state: Option<Gemma4MetalState>,
    debug_sync: bool,
    experimental_kv_int8: bool,
    perf: MetalProbePerfCounters,
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl ModelMetalBackend {
    #[must_use]
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            model_dir,
            prepared: false,
            next_step_id: 0,
            last_ticket: None,
            pending: None,
            ctx: None,
            pipelines: None,
            arena: None,
            state: None,
            debug_sync: metal_debug_sync_enabled(),
            experimental_kv_int8: experimental_metal_kv_int8_enabled(),
            perf: MetalProbePerfCounters::default(),
        }
    }

    #[must_use]
    pub fn probe_perf_stats(&self) -> MetalProbePerfStats {
        self.perf.snapshot()
    }

    #[must_use]
    pub const fn metal_debug_sync_enabled(&self) -> bool {
        self.debug_sync
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

    #[cfg(test)]
    fn debug_read_decode_logits_f32(&self, num_tokens: usize) -> Result<Vec<f32>> {
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

        let logits_ptr = unsafe { arena.host_ptr(&state.logits) as *const u16 };
        let logits_bits = unsafe { std::slice::from_raw_parts(logits_ptr, elem_count) };
        Ok(logits_bits
            .iter()
            .map(|bits| f16::from_bits(*bits).to_f32())
            .collect())
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
        Ok(bits
            .iter()
            .map(|bits| f16::from_bits(*bits).to_f32())
            .collect())
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
        let mut ctx = MetalContext::new()?;
        ctx.compile_library(kernels::KERNEL_SOURCE)?;
        let mut pipelines = PipelineCache::new();
        pipelines.compile_all(&ctx)?;

        let arena_bytes = Gemma4MetalState::required_probe_model_arena_bytes(&self.model_dir)?;
        let mut arena = MetalBufferArena::new(ctx.device(), arena_bytes)?;
        let state = Gemma4MetalState::load_probe_model(&ctx, &mut arena, &self.model_dir)?;

        self.ctx = Some(ctx);
        self.pipelines = Some(pipelines);
        self.arena = Some(arena);
        Ok(state)
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
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("enqueue_embedding_gather"),
            )
        })?;
        let arena = self.arena.as_ref().ok_or_else(|| {
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
            encoder.setBuffer_offset_atIndex(Some(buf), state.token_ids.offset, 1);
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
        cmd_buf.commit();
        self.perf.add_command_buffers(1);
        self.perf.add_encoders(1);
        if self.debug_sync {
            cmd_buf.waitUntilCompleted();
            self.perf.add_forced_wait();
        }
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
        self.perf.add_encoders(4);
        if self.debug_sync {
            self.wait_for_metal_queue("enqueue_ple_inputs")?;
        }
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
        if num_tokens == 0 || num_tokens > state.max_probe_tokens {
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

        let positions = (0..num_tokens).map(|idx| idx as i32).collect::<Vec<_>>();
        let slot_mapping = positions.clone();
        let context_lens = [num_tokens as i32];
        let block_tables = [0_i32];
        let cu_seqlens = [0_i32, num_tokens as i32];

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
        Ok(())
    }

    fn write_decode_layer_metadata(
        &self,
        state: &Gemma4MetalState,
        handoff: &HandoffCapsule,
    ) -> Result<()> {
        let num_seqs = handoff.num_sequences();
        if num_seqs == 0 || num_seqs > state.max_probe_tokens {
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
        let slot_mapping = handoff
            .positions
            .iter()
            .enumerate()
            .map(|(seq, &position)| (seq * state.max_probe_tokens + position as usize) as i32)
            .collect::<Vec<_>>();
        let block_tables = (0..num_seqs).map(|seq| seq as i32).collect::<Vec<_>>();
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

    fn enqueue_probe_layers(
        &self,
        state: &Gemma4MetalState,
        num_tokens: usize,
        phase: MetalPhase,
        op: &'static str,
    ) -> Result<()> {
        if num_tokens == 0 || num_tokens > state.max_probe_tokens {
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
        let stop_after_layer = metal_debug_stop_after_layer();
        #[cfg(test)]
        let trace_layer = metal_debug_trace_layer();
        #[cfg(test)]
        let trace_json_path = metal_debug_trace_json_path();

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
            let intermediate = one.gate_up.size / 2 / half_bytes / hidden;

            let dims = MetalLayerDims {
                layer_idx: one.layer_idx as u32,
                num_tokens: num_tokens_u32,
                hidden: state.hidden_size as u32,
                num_layers: state.num_layers as u32,
                num_heads: one.dims.num_heads as u32,
                num_kv_heads: one.dims.num_kv_heads as u32,
                head_dim: one.dims.head_dim as u32,
                intermediate: intermediate as u32,
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
                down_proj_offset: one.down_proj.offset,
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
                gate_up_out: one.gate_up_out.offset,
                activated: one.activated.offset,
                mlp_out: one.mlp_out.offset,
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
            let layer_trace_state = if trace_layer == Some(one.layer_idx) {
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

            unsafe {
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
                )?;
            }
            self.perf.add_command_buffers(1);
            self.perf
                .add_encoders(Self::estimate_layer_encoder_count(&weights));

            #[cfg(test)]
            if metal_debug_finite_layers_enabled()
                || stop_after_layer.is_some()
                || trace_layer.is_some()
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
                if trace_layer == Some(one.layer_idx) {
                    if let Some(path) = trace_json_path.as_deref() {
                        let q_dim = one.dims.q_dim;
                        let kv_dim = one.dims.kv_dim;
                        debug_write_layer_trace_json(
                            arena,
                            path,
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
                            layer_trace_state,
                        )?;
                        eprintln!(
                            "metal debug trace: wrote layer {} summary to {}",
                            one.layer_idx,
                            path.display()
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
        cmd_buf.waitUntilCompleted();
        self.perf.add_forced_wait();
        Ok(())
    }

    fn estimate_layer_encoder_count(weights: &MetalLayerWeights) -> u64 {
        let mut count = 11;
        count += weights.q_norm_offset.is_some() as u64;
        count += weights.k_norm_offset.is_some() as u64;
        count += weights.v_norm_offset.is_some() as u64;
        count += weights.post_attn_norm_offset.is_some() as u64;
        count += weights.post_ff_norm_offset.is_some() as u64;
        if weights.per_layer_inputs_offset.is_some()
            && weights.per_layer_input_gate_offset.is_some()
            && weights.per_layer_projection_offset.is_some()
            && weights.post_per_layer_input_norm_offset.is_some()
        {
            count += 4;
        }
        count
    }

    fn run_prefill_step(&mut self, handoff: &HandoffCapsule) -> Result<()> {
        let perf_before = self.perf.snapshot();
        let wall_start = Instant::now();
        let state = self.state.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_prefill"),
            )
        })?;
        if handoff.num_sequences() != 1 && state.num_layers > 0 {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_prefill_batch_size_for_layers",
                },
                model_ctx("launch_prefill"),
            ));
        }
        const MAX_DEFAULT_PROBE_LAYERS: usize = 8;
        if state.num_layers > MAX_DEFAULT_PROBE_LAYERS && !large_gemma4_probe_opted_in() {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_probe_num_layers_without_large_model_opt_in",
                },
                model_ctx("launch_prefill"),
            ));
        }

        let num_tokens = handoff.tokens_flat.len();
        if num_tokens == 0 || num_tokens > state.max_probe_tokens {
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
        let token_ids: Vec<u32> = handoff.tokens_flat.iter().map(|t| t.raw()).collect();
        unsafe {
            let dst = arena.host_ptr(&state.token_ids) as *mut u32;
            ptr::copy_nonoverlapping(token_ids.as_ptr(), dst, token_ids.len());
        }

        self.write_prefill_layer_metadata(state, handoff)?;
        self.enqueue_embedding_gather(state, num_tokens)?;
        self.enqueue_ple_inputs(state, num_tokens)?;
        self.enqueue_probe_layers(
            state,
            num_tokens,
            MetalPhase::Prefill {
                max_seqlen_q: num_tokens as u32,
                batch_size: 1,
            },
            "launch_prefill",
        )?;
        self.wait_for_metal_queue("launch_prefill")?;
        self.perf
            .finish_step(false, num_tokens as u64, perf_before, wall_start.elapsed());
        Ok(())
    }

    fn run_decode_step(&mut self, handoff: &HandoffCapsule) -> Result<()> {
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

        let state = self.state.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_rollout"),
            )
        })?;
        const MAX_DEFAULT_PROBE_LAYERS: usize = 8;
        if state.num_layers > MAX_DEFAULT_PROBE_LAYERS && !large_gemma4_probe_opted_in() {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_probe_num_layers_without_large_model_opt_in",
                },
                model_ctx("launch_rollout"),
            ));
        }

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

        let token_ids: Vec<u32> = handoff.tokens_flat.iter().map(|t| t.raw()).collect();
        unsafe {
            let dst = arena.host_ptr(&state.token_ids) as *mut u32;
            ptr::copy_nonoverlapping(token_ids.as_ptr(), dst, token_ids.len());
        }

        self.write_decode_layer_metadata(state, handoff)?;
        self.enqueue_embedding_gather(state, num_tokens)?;
        self.enqueue_ple_inputs(state, num_tokens)?;
        self.enqueue_probe_layers(state, num_tokens, MetalPhase::Decode, "launch_rollout")?;

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
        self.perf.add_encoders(metal_finalize_logits_encoder_count(
            num_tokens_u32,
            hidden_u32,
            vocab_u32,
            state.final_logit_softcap,
        ));
        self.perf.add_forced_wait();

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
        self.pending = Some(outputs);
        self.perf
            .finish_step(true, num_tokens as u64, perf_before, wall_start.elapsed());
        Ok(())
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl AppleBackend for ModelMetalBackend {
    fn prepare(&mut self, plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
        plan.validate()?;
        self.prepared = false;
        if !self.model_dir.exists() || !self.model_dir.is_dir() {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing model path",
                },
                model_ctx("prepare"),
            ));
        }
        self.next_step_id = 0;
        self.last_ticket = None;
        self.pending = None;
        self.ctx = None;
        self.pipelines = None;
        self.arena = None;
        self.state = None;
        self.debug_sync = metal_debug_sync_enabled();
        self.experimental_kv_int8 = experimental_metal_kv_int8_enabled();
        self.perf.clear();
        let state = self.initialize_model_resources()?;
        self.state = Some(state);
        self.prepared = true;
        Ok(())
    }

    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        handoff.validate()?;
        self.run_prefill_step(handoff)?;
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
        self.run_decode_step(handoff)?;
        Ok(self.next_ticket(AppleLaunchKind::Rollout, bucket))
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        self.ensure_prepared("collect")?;
        match self.last_ticket {
            Some(expected) if expected == ticket.step_id => {
                Ok(self.pending.take().unwrap_or_default())
            }
            Some(_) => Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "collect_stale_ticket",
                },
                model_ctx("collect"),
            )),
            None => Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("collect"),
            )),
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
pub type ToyMetalBackend = RuntimeMetalBackend;

#[cfg(all(feature = "apple", target_os = "macos"))]
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
        ctx.compile_library(kernels::KERNEL_SOURCE)?;

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

#[cfg(all(feature = "apple", target_os = "macos"))]
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
#[path = "apple_metal_backend_tests.rs"]
mod tests;
