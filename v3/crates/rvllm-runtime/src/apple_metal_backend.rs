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
    gemma4_model::Gemma4MetalState,
    kernels,
    layer_forward::{
        metal_finalize_logits_blocking, metal_finalize_logits_encoder_count, metal_forward_layer,
        MetalLayerDims, MetalLayerWeights, MetalMetadata, MetalPhase, MetalScratch,
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
                num_tokens: num_tokens_u32,
                hidden: state.hidden_size as u32,
                num_heads: one.dims.num_heads as u32,
                num_kv_heads: one.dims.num_kv_heads as u32,
                head_dim: one.dims.head_dim as u32,
                intermediate: intermediate as u32,
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

            unsafe {
                metal_forward_layer(
                    ctx,
                    pipelines,
                    arena,
                    &dims,
                    &weights,
                    &scratch,
                    &meta,
                    state.residual.offset,
                    phase,
                    one.kv_cache_k.offset,
                    one.kv_cache_v.offset,
                )?;
            }
            self.perf.add_command_buffers(1);
            self.perf
                .add_encoders(Self::estimate_layer_encoder_count(&weights));
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
        const MAX_SYNTHETIC_PROBE_LAYERS: usize = 8;
        if state.num_layers > MAX_SYNTHETIC_PROBE_LAYERS {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_synthetic_probe_num_layers",
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
        const MAX_SYNTHETIC_PROBE_LAYERS: usize = 8;
        if state.num_layers > MAX_SYNTHETIC_PROBE_LAYERS {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_synthetic_probe_num_layers",
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
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use rvllm_apple_metal::weight_loader::scan_safetensor_tensors;
    use serde_json::{Map, Value};
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    const FULL_NONZERO_ZERO_DIM: usize = 0;
    const FULL_NONZERO_ORIGINAL_DIM: usize = 7;
    const FULL_NONZERO_ATTENTION_DIM: usize = 9;
    const FULL_NONZERO_VALUE_DIM: usize = 11;
    const FULL_NONZERO_FFN_DIM: usize = 13;

    #[cfg(all(feature = "apple", target_os = "macos"))]
    static METAL_DEBUG_SYNC_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[derive(Clone)]
    struct SharedModelMetalBackend {
        inner: std::rc::Rc<std::cell::RefCell<ModelMetalBackend>>,
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    impl SharedModelMetalBackend {
        fn new(model_dir: std::path::PathBuf) -> Self {
            Self {
                inner: std::rc::Rc::new(std::cell::RefCell::new(ModelMetalBackend::new(model_dir))),
            }
        }

        fn debug_read_decode_logits_f32(&self, num_tokens: usize) -> Result<Vec<f32>> {
            self.inner.borrow().debug_read_decode_logits_f32(num_tokens)
        }
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    impl AppleBackend for SharedModelMetalBackend {
        fn prepare(&mut self, plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
            self.inner.borrow_mut().prepare(plan)
        }

        fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
            self.inner.borrow_mut().launch_prefill(handoff)
        }

        fn launch_rollout(
            &mut self,
            handoff: &HandoffCapsule,
            bucket: Option<rvllm_apple::RolloutBucket>,
        ) -> Result<AppleLaunchTicket> {
            self.inner.borrow_mut().launch_rollout(handoff, bucket)
        }

        fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
            self.inner.borrow_mut().collect(ticket)
        }
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    struct MetalDebugSyncEnvGuard {
        _guard: std::sync::MutexGuard<'static, ()>,
        previous: Option<std::ffi::OsString>,
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    impl MetalDebugSyncEnvGuard {
        fn new() -> Self {
            Self {
                _guard: METAL_DEBUG_SYNC_ENV_LOCK.lock().expect("lock env guard"),
                previous: std::env::var_os(RVLLM_METAL_DEBUG_SYNC_ENV),
            }
        }

        fn set_current(&self, value: Option<&str>) {
            if let Some(value) = value {
                std::env::set_var(RVLLM_METAL_DEBUG_SYNC_ENV, value);
            } else {
                std::env::remove_var(RVLLM_METAL_DEBUG_SYNC_ENV);
            }
        }
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    impl Drop for MetalDebugSyncEnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                std::env::set_var(RVLLM_METAL_DEBUG_SYNC_ENV, previous);
            } else {
                std::env::remove_var(RVLLM_METAL_DEBUG_SYNC_ENV);
            }
        }
    }

    fn temp_fixture_dir() -> std::path::PathBuf {
        static FIXTURE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        let serial = FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rvllm-metal-zero-layer-test-{}-{}-{}",
            std::process::id(),
            now,
            serial
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create fixture dir");
        dir
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<half::f16>());
        for value in values {
            let bits = half::f16::from_f32(*value).to_bits();
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    fn write_tiny_zero_layer_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let embedding = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let norm = [1.0, 1.0, 1.0, 1.0];
        let lm_head = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0,
        ];

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[4, 4],
            &mut payload,
            &mut header,
        );
        add_tensor("model.norm.weight", &norm, &[4], &mut payload, &mut header);
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[4, 4],
            &mut payload,
            &mut header,
        );

        let config = r#"{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {
    "num_hidden_layers": 0,
    "hidden_size": 4,
    "intermediate_size": 8,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "vocab_size": 4,
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }
}"#;

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    fn zero_layer_plan(model_dir: std::path::PathBuf) -> rvllm_apple::AppleRuntimePlan {
        rvllm_apple::AppleRuntimePlan {
            target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
            ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 4,
            ane_intermediate_size: 8,
            ane_num_layers: 1,
            model_layout_hash: [0u8; 32],
            weights_path: Some(model_dir),
        }
    }

    #[test]
    fn tiny_zero_layer_fixture_has_expected_files() {
        let dir = write_tiny_zero_layer_fixture();
        assert!(dir.join("config.json").is_file());
        assert!(dir.join("model.safetensors").is_file());

        let config_raw = fs::read_to_string(dir.join("config.json")).expect("read config");
        let config: Value = serde_json::from_str(&config_raw).expect("parse config");
        assert_eq!(config["architectures"][0], "Gemma4ForCausalLM");
        assert_eq!(config["text_config"]["num_hidden_layers"], 0);
        assert_eq!(config["text_config"]["vocab_size"], 4);

        #[cfg(target_os = "macos")]
        {
            let tensors = scan_safetensor_tensors(&dir).expect("read fixture tensors");
            let embed = tensors
                .get("model.embed_tokens.weight")
                .expect("embed tensor");
            let norm = tensors.get("model.norm.weight").expect("norm tensor");
            let lm_head = tensors.get("lm_head.weight").expect("lm_head tensor");
            assert_eq!(embed.shape, vec![4, 4]);
            assert_eq!(norm.shape, vec![4]);
            assert_eq!(lm_head.shape, vec![4, 4]);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(not(all(feature = "apple", target_os = "macos")))]
    #[test]
    fn model_metal_backend_non_macos_fails_closed() {
        let mut backend = RuntimeMetalBackend::new();
        let plan = rvllm_apple::AppleRuntimePlan {
            target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
            ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 4,
            ane_intermediate_size: 8,
            ane_num_layers: 1,
            model_layout_hash: [0u8; 32],
            weights_path: None,
        };
        assert!(backend.prepare(&plan).is_err());
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_zero_layer_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_zero_layer_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = zero_layer_plan(dir.clone());
        backend.prepare(&plan).expect("prepare tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn metal_probe_microbench_counters_hook_reports_decode_work() {
        let dir = write_tiny_zero_layer_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = zero_layer_plan(dir.clone());
        backend.prepare(&plan).expect("prepare tiny model");

        for req in [1_u64, 2_u64] {
            let handoff = rvllm_apple::HandoffCapsule::new(
                rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
                vec![rvllm_core::ReqId(req)],
                vec![rvllm_core::TokenId(2)],
                vec![0, 1],
                vec![0],
                vec![1],
            );
            let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
            let out = backend.collect(ticket).expect("collect");
            assert_eq!(out.len(), 1);
        }

        let stats = backend.probe_perf_stats();
        eprintln!("metal_probe_microbench_counters_hook stats: {stats:?}");
        assert_eq!(stats.decode_steps, 2);
        assert_eq!(stats.last_step_tokens, 1);
        assert!(stats.command_buffers > 0);
        assert!(stats.encoders > 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_zero_layer_model_backend_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_zero_layer_fixture();
        let plan = zero_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty(), "zero-layer prefill returns no tokens");

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));

        assert!(
            !engine.has_pending_work(),
            "request should finish after one decoded token"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_noop_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        // token 3 should be chosen if dim 7 is high
        for d in 0..hidden {
            lm_head[3 * hidden + d] = if d == 7 { 2.0 } else { 0.0 };
            lm_head[2 * hidden + d] = if d == 7 { 1.0 } else { 0.0 };
        }

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        // One layer tensors
        let ones = vec![1.0f32; hidden];
        let zeros_qkv = vec![0.0f32; 3 * hidden * hidden];
        let zeros_o = vec![0.0f32; hidden * hidden];
        let zeros_gate = vec![0.0f32; 2 * intermediate * hidden];
        let zeros_down = vec![0.0f32; hidden * intermediate];

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.qkv.weight",
            &zeros_qkv,
            &[3 * hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &zeros_o,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_up.weight",
            &zeros_gate,
            &[2 * intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &zeros_down,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    fn one_layer_plan(model_dir: std::path::PathBuf) -> rvllm_apple::AppleRuntimePlan {
        rvllm_apple::AppleRuntimePlan {
            target: rvllm_apple::AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: rvllm_apple::AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
            strict_ane: false,
            ane_compute_profile: rvllm_core::config::AneComputeProfile::AnyAvailable,
            ane_fallback_policy: rvllm_core::config::AneFallbackPolicy::AllowMetal,
            ane_hidden_size: 128,
            ane_intermediate_size: 256,
            ane_num_layers: 1,
            model_layout_hash: [0u8; 32],
            weights_path: Some(model_dir),
        }
    }

    fn two_layer_plan(model_dir: std::path::PathBuf) -> rvllm_apple::AppleRuntimePlan {
        n_layer_plan(model_dir, 2)
    }

    fn n_layer_plan(
        model_dir: std::path::PathBuf,
        num_layers: usize,
    ) -> rvllm_apple::AppleRuntimePlan {
        rvllm_apple::AppleRuntimePlan {
            ane_num_layers: num_layers,
            ..one_layer_plan(model_dir)
        }
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_two_layer_fixture(first_layer_ffn_nonzero: bool) -> std::path::PathBuf {
        write_tiny_n_layer_fixture(2, first_layer_ffn_nonzero)
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_n_layer_fixture(
        num_layers: usize,
        first_layer_ffn_nonzero: bool,
    ) -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        if first_layer_ffn_nonzero {
            lm_head[2 * hidden + 7] = 1.0;
            lm_head[3 * hidden + 9] = 4.0;
        } else {
            lm_head[2 * hidden + 7] = 1.0;
            lm_head[3 * hidden + 7] = 2.0;
        }

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let zeros_qkv = vec![0.0f32; 3 * hidden * hidden];
        let zeros_o = vec![0.0f32; hidden * hidden];

        for layer_idx in 0..num_layers {
            let mut gate_up = vec![0.0f32; 2 * intermediate * hidden];
            let mut down_proj = vec![0.0f32; hidden * intermediate];
            if first_layer_ffn_nonzero && layer_idx == 0 {
                gate_up[7] = 0.5;
                gate_up[intermediate * hidden + 7] = 0.5;
                down_proj[9 * intermediate] = 4.0;
            }

            add_tensor(
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &ones,
                &[hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.self_attn.qkv.weight"),
                &zeros_qkv,
                &[3 * hidden, hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &zeros_o,
                &[hidden, hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.mlp_norm.weight"),
                &ones,
                &[hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.mlp.gate_up.weight"),
                &gate_up,
                &[2 * intermediate, hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &down_proj,
                &[hidden, intermediate],
                &mut payload,
                &mut header,
            );
        }

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": {},
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            num_layers, hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_two_layer_sliding_global_noop_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;
        let sliding_head_dim = 128;
        let global_head_dim = 256;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 7] = 2.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let zeros_gate_up = vec![0.0f32; 2 * intermediate * hidden];
        let zeros_down = vec![0.0f32; hidden * intermediate];

        for (layer_idx, head_dim) in [(0usize, sliding_head_dim), (1usize, global_head_dim)] {
            let zeros_qkv = vec![0.0f32; 3 * head_dim * hidden];
            let zeros_o = vec![0.0f32; hidden * head_dim];
            add_tensor(
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &ones,
                &[hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.self_attn.qkv.weight"),
                &zeros_qkv,
                &[3 * head_dim, hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &zeros_o,
                &[hidden, head_dim],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.mlp_norm.weight"),
                &ones,
                &[hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.mlp.gate_up.weight"),
                &zeros_gate_up,
                &[2 * intermediate, hidden],
                &mut payload,
                &mut header,
            );
            add_tensor(
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &zeros_down,
                &[hidden, intermediate],
                &mut payload,
                &mut header,
            );
        }

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 2,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "global_head_dim": {},
    "num_global_key_value_heads": 1,
    "layer_types": ["sliding_attention", "full_attention"],
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, sliding_head_dim, global_head_dim, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_two_layer_noop_fixture() -> std::path::PathBuf {
        write_tiny_two_layer_fixture(false)
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_two_layer_first_ffn_nonzero_fixture() -> std::path::PathBuf {
        write_tiny_two_layer_fixture(true)
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_n_layer_noop_fixture(num_layers: usize) -> std::path::PathBuf {
        write_tiny_n_layer_fixture(num_layers, false)
    }

    fn rmsnorm_f32(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let hidden = input.len();
        let sum_sq = input.iter().map(|v| v * v).sum::<f32>();
        let inv_rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        input
            .iter()
            .zip(gamma.iter())
            .map(|(x, g)| x * inv_rms * g)
            .collect()
    }

    fn gemm_f32(input: &[f32], weights: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; out_dim];
        for row in 0..out_dim {
            let mut acc = 0.0f32;
            for col in 0..in_dim {
                acc += input[col] * weights[row * in_dim + col];
            }
            out[row] = acc;
        }
        out
    }

    fn gelu_tanh_f32(x: f32) -> f32 {
        let c = 0.7978845608f32;
        0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
    }

    fn cpu_reference_one_layer_ffn_nonzero_argmax() -> usize {
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let eps = 0.000001f32;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;
        let norm = vec![1.0f32; hidden];

        let mut residual = vec![0.0f32; hidden];
        let embedding_scale = (hidden as f32).sqrt();
        for dim in 0..hidden {
            residual[dim] = embedding[2 * hidden + dim] * embedding_scale;
        }

        let mlp_input = rmsnorm_f32(&residual, &norm, eps);
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        gate_proj[7] = 0.5;
        up_proj[7] = 0.5;
        down_proj[9 * intermediate] = 4.0;

        let gate = gemm_f32(&mlp_input, &gate_proj, intermediate, hidden);
        let up = gemm_f32(&mlp_input, &up_proj, intermediate, hidden);
        let mut activated = vec![0.0f32; intermediate];
        for dim in 0..intermediate {
            activated[dim] = gelu_tanh_f32(gate[dim]) * up[dim];
        }
        let mlp_out = gemm_f32(&activated, &down_proj, hidden, intermediate);
        for dim in 0..hidden {
            residual[dim] += mlp_out[dim];
        }

        let final_hidden = rmsnorm_f32(&residual, &norm, eps);
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;
        let logits = gemm_f32(&final_hidden, &lm_head, vocab, hidden);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("finite logits"))
            .map(|(idx, _)| idx)
            .expect("nonempty logits")
    }

    fn cpu_reference_one_layer_attention_nonzero_argmax() -> usize {
        let hidden = 128usize;
        let vocab = 8usize;
        let eps = 0.000001f32;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;
        let norm = vec![1.0f32; hidden];

        let mut residual = vec![0.0f32; hidden];
        let embedding_scale = (hidden as f32).sqrt();
        for dim in 0..hidden {
            residual[dim] = embedding[2 * hidden + dim] * embedding_scale;
        }

        let attn_input = rmsnorm_f32(&residual, &norm, eps);
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 2.0;
        o_proj[9 * hidden + 11] = 6.0;

        let q = gemm_f32(&attn_input, &q_proj, hidden, hidden);
        let k = gemm_f32(&attn_input, &k_proj, hidden, hidden);
        let v = gemm_f32(&attn_input, &v_proj, hidden, hidden);
        let score =
            q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
        assert!(score.is_finite());

        let attn_out = v;
        let attn_residual = gemm_f32(&attn_out, &o_proj, hidden, hidden);
        for dim in 0..hidden {
            residual[dim] += attn_residual[dim];
        }

        let final_hidden = rmsnorm_f32(&residual, &norm, eps);
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;
        let logits = gemm_f32(&final_hidden, &lm_head, vocab, hidden);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("finite logits"))
            .map(|(idx, _)| idx)
            .expect("nonempty logits")
    }

    #[test]
    fn cpu_reference_one_layer_ffn_nonzero_fixture_argmax_is_3() {
        assert_eq!(cpu_reference_one_layer_ffn_nonzero_argmax(), 3);
    }

    #[test]
    fn cpu_reference_one_layer_attention_nonzero_fixture_argmax_is_3() {
        assert_eq!(cpu_reference_one_layer_attention_nonzero_argmax(), 3);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_noop_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_one_layer_noop_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_noop_model_backend_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_one_layer_noop_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_two_layer_noop_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_two_layer_noop_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = two_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare two-layer no-op tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_two_layer_first_ffn_nonzero_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_two_layer_first_ffn_nonzero_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = two_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare two-layer first-ffn-nonzero tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_two_layer_first_ffn_nonzero_model_backend_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_two_layer_first_ffn_nonzero_fixture();
        let plan = two_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny two-layer first-ffn-nonzero model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_three_layer_noop_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_n_layer_noop_fixture(3);
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = n_layer_plan(dir.clone(), 3);
        backend
            .prepare(&plan)
            .expect("prepare three-layer no-op tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_four_layer_noop_model_backend_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_n_layer_noop_fixture(4);
        let plan = n_layer_plan(dir.clone(), 4);

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny four-layer no-op model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_two_layer_sliding_global_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_two_layer_sliding_global_noop_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = two_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare two-layer sliding/global tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_two_layer_sliding_global_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_two_layer_sliding_global_noop_fixture();
        let plan = two_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny two-layer sliding/global model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_hf_style_noop_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        // token 3 should be chosen if dim 7 is high
        for d in 0..hidden {
            lm_head[3 * hidden + d] = if d == 7 { 2.0 } else { 0.0 };
            lm_head[2 * hidden + d] = if d == 7 { 1.0 } else { 0.0 };
        }

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        // One layer tensors (HF style separate)
        let ones = vec![1.0f32; hidden];
        let zeros_qkvo = vec![0.0f32; hidden * hidden];
        let zeros_gate_up = vec![0.0f32; intermediate * hidden];
        let zeros_down = vec![0.0f32; hidden * intermediate];

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &zeros_gate_up,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &zeros_gate_up,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &zeros_down,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_ffn_nonzero_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let zeros_qkvo = vec![0.0f32; hidden * hidden];
        let zeros_down = vec![0.0f32; hidden * intermediate];
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = zeros_down;
        gate_proj[7] = 0.5;
        up_proj[7] = 0.5;
        down_proj[9 * intermediate] = 4.0;

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &gate_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &up_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_zero_layer_decode_loop_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 2] = 10.0;
        embedding[3 * hidden + 3] = 10.0;
        embedding[4 * hidden + 4] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[3 * hidden + 2] = 2.0;
        lm_head[4 * hidden + 3] = 2.0;
        lm_head[5 * hidden + 4] = 2.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 0,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden,
            hidden * 2,
            hidden,
            vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_attention_nonzero_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let zeros_gate_up = vec![0.0f32; intermediate * hidden];
        let zeros_down = vec![0.0f32; hidden * intermediate];
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 2.0;
        o_proj[9 * hidden + 11] = 6.0;

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &o_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &zeros_gate_up,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &zeros_gate_up,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &zeros_down,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    fn cpu_full_nonzero_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mean_square = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
        let scale = (mean_square + eps).sqrt().recip();
        input
            .iter()
            .zip(weight.iter())
            .map(|(v, w)| v * scale * w)
            .collect()
    }

    fn cpu_full_nonzero_matvec(
        weight: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Vec<f32> {
        assert_eq!(weight.len(), rows * cols);
        assert_eq!(input.len(), cols);
        let mut out = vec![0.0f32; rows];
        for row in 0..rows {
            let base = row * cols;
            out[row] = (0..cols).map(|col| weight[base + col] * input[col]).sum();
        }
        out
    }

    fn cpu_full_nonzero_gelu_tanh(x: f32) -> f32 {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
    }

    fn cpu_full_nonzero_argmax(values: &[f32]) -> usize {
        let mut best_idx = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (idx, value) in values.iter().enumerate() {
            if *value > best_value {
                best_idx = idx;
                best_value = *value;
            }
        }
        best_idx
    }

    fn cpu_full_nonzero_top_two(values: &[f32]) -> (usize, usize) {
        assert!(values.len() >= 2);
        let mut ranked = values.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).expect("finite logits"));
        (ranked[0].0, ranked[1].0)
    }

    fn cpu_reference_zero_layer_decode_loop_sequence() -> Vec<usize> {
        let hidden = 128usize;
        let vocab = 8usize;
        let eps = 0.000001f32;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 2] = 10.0;
        embedding[3 * hidden + 3] = 10.0;
        embedding[4 * hidden + 4] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[3 * hidden + 2] = 2.0;
        lm_head[4 * hidden + 3] = 2.0;
        lm_head[5 * hidden + 4] = 2.0;

        let mut current = 2usize;
        let mut out = Vec::new();
        for _ in 0..3 {
            let mut residual = embedding[current * hidden..(current + 1) * hidden].to_vec();
            let embed_scale = (hidden as f32).sqrt();
            for value in &mut residual {
                *value *= embed_scale;
            }

            let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
            let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
            current = cpu_full_nonzero_argmax(&logits);
            out.push(current);
        }
        out
    }

    struct CpuFullNonzeroOneLayerReference {
        residual_after_attention: Vec<f32>,
        residual: Vec<f32>,
        final_hidden: Vec<f32>,
        logits: Vec<f32>,
    }

    fn cpu_reference_one_layer_full_nonzero() -> CpuFullNonzeroOneLayerReference {
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;
        let eps = 0.000001f32;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + FULL_NONZERO_ORIGINAL_DIM] = 10.0;
        let norm = vec![1.0f32; hidden];

        let mut residual = embedding[2 * hidden..3 * hidden].to_vec();
        let embed_scale = (hidden as f32).sqrt();
        for value in &mut residual {
            *value *= embed_scale;
        }

        let attn_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        q_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.25;
        k_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.125;
        v_proj[FULL_NONZERO_VALUE_DIM * hidden + FULL_NONZERO_ORIGINAL_DIM] = 2.0;
        o_proj[FULL_NONZERO_ATTENTION_DIM * hidden + FULL_NONZERO_VALUE_DIM] = 6.0;

        let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
        let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
        let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
        let _score =
            q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
        let attn_out = v;
        let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
        for (dst, src) in residual.iter_mut().zip(projected_attn.iter()) {
            *dst += src;
        }
        let residual_after_attention = residual.clone();

        let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        gate_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.5;
        up_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.5;
        down_proj[FULL_NONZERO_FFN_DIM * intermediate] = 4.0;

        let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
        let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
        let activated = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
            .collect::<Vec<_>>();
        let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
        for (dst, src) in residual.iter_mut().zip(mlp_out.iter()) {
            *dst += src;
        }

        let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + FULL_NONZERO_ORIGINAL_DIM] = 1.0;
        lm_head[3 * hidden + FULL_NONZERO_ATTENTION_DIM] = 4.0;
        let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);

        CpuFullNonzeroOneLayerReference {
            residual_after_attention,
            residual,
            final_hidden,
            logits,
        }
    }

    fn cpu_reference_one_layer_full_nonzero_logits() -> Vec<f32> {
        cpu_reference_one_layer_full_nonzero().logits
    }

    fn cpu_reference_one_layer_full_nonzero_argmax() -> usize {
        cpu_full_nonzero_argmax(&cpu_reference_one_layer_full_nonzero_logits())
    }

    fn cpu_reference_real_hf_style_one_layer_slice_argmax() -> usize {
        cpu_reference_one_layer_full_nonzero_argmax()
    }

    fn cpu_reference_one_layer_qkv_norm_nonzero_argmax(apply_qkv_norm: bool) -> usize {
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let eps = 1e-6f32;

        let mut residual = vec![0.0f32; hidden];
        residual[7] = 10.0 * (hidden as f32).sqrt();

        let norm = vec![1.0f32; hidden];
        let mut q_norm = vec![1.0f32; hidden];
        let mut k_norm = vec![1.0f32; hidden];
        let v_norm = vec![1.0f32; hidden];
        q_norm[0] = 0.5;
        k_norm[0] = 0.25;

        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let gate_proj = vec![0.0f32; intermediate * hidden];
        let up_proj = vec![0.0f32; intermediate * hidden];
        let down_proj = vec![0.0f32; hidden * intermediate];
        let mut lm_head = vec![0.0f32; vocab * hidden];

        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 0.25;
        o_proj[9 * hidden + 11] = 0.5;
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 32.0;

        let attn_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let mut q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
        let mut k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
        let mut v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
        if apply_qkv_norm {
            q = cpu_full_nonzero_rms_norm(&q, &q_norm, eps);
            k = cpu_full_nonzero_rms_norm(&k, &k_norm, eps);
            v = cpu_full_nonzero_rms_norm(&v, &v_norm, eps);
        }

        let _single_key_score =
            q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
        let attn_out = v;
        let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
        for d in 0..hidden {
            residual[d] += projected_attn[d];
        }

        let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
        let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
        let activated = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
            .collect::<Vec<_>>();
        let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
        for d in 0..hidden {
            residual[d] += mlp_out[d];
        }

        let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
        cpu_full_nonzero_argmax(&logits)
    }

    fn cpu_reference_one_layer_extra_norms_argmax(apply_extra_norms: bool) -> usize {
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let eps = 1e-6f32;

        let mut residual = vec![0.0f32; hidden];
        residual[7] = 10.0 * (hidden as f32).sqrt();

        let norm = vec![1.0f32; hidden];
        let mut post_attn_norm = vec![1.0f32; hidden];
        let pre_ff_norm = vec![1.0f32; hidden];
        let post_ff_norm = vec![1.0f32; hidden];

        post_attn_norm[7] = 0.01;
        post_attn_norm[9] = 64.0;

        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let gate_proj = vec![0.0f32; intermediate * hidden];
        let up_proj = vec![0.0f32; intermediate * hidden];
        let down_proj = vec![0.0f32; hidden * intermediate];
        let mut lm_head = vec![0.0f32; vocab * hidden];

        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 0.25;
        o_proj[9 * hidden + 11] = 0.5;
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;

        let attn_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
        let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
        let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
        let _single_key_score =
            q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
        let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &v);
        for d in 0..hidden {
            residual[d] += projected_attn[d];
        }

        if apply_extra_norms {
            residual = cpu_full_nonzero_rms_norm(&residual, &post_attn_norm, eps);
        }

        let mlp_normed = cpu_full_nonzero_rms_norm(
            &residual,
            if apply_extra_norms {
                &pre_ff_norm
            } else {
                &norm
            },
            eps,
        );
        let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
        let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
        let activated = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
            .collect::<Vec<_>>();
        let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
        for d in 0..hidden {
            residual[d] += mlp_out[d];
        }

        if apply_extra_norms {
            residual = cpu_full_nonzero_rms_norm(&residual, &post_ff_norm, eps);
        }

        let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
        cpu_full_nonzero_argmax(&logits)
    }

    fn cpu_reference_one_layer_layer_scalar_argmax(apply_layer_scalar: bool) -> usize {
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let eps = 1e-6f32;
        let update_scale = if apply_layer_scalar { 6.0f32 } else { 1.0f32 };

        let mut residual = vec![0.0f32; hidden];
        residual[7] = 10.0 * (hidden as f32).sqrt();

        let norm = vec![1.0f32; hidden];
        let q_proj = vec![0.0f32; hidden * hidden];
        let k_proj = vec![0.0f32; hidden * hidden];
        let v_proj = vec![0.0f32; hidden * hidden];
        let o_proj = vec![0.0f32; hidden * hidden];
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        let mut lm_head = vec![0.0f32; vocab * hidden];

        gate_proj[7] = 0.5;
        up_proj[7] = 0.5;
        down_proj[9 * intermediate] = 1.0;
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 1.0;

        let attn_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
        let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
        let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
        let _single_key_score =
            q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
        let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &v);
        for d in 0..hidden {
            residual[d] += projected_attn[d] * update_scale;
        }

        let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
        let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
        let activated = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
            .collect::<Vec<_>>();
        let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
        for d in 0..hidden {
            residual[d] += mlp_out[d] * update_scale;
        }

        let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
        cpu_full_nonzero_argmax(&logits)
    }

    fn cpu_reference_one_layer_integrated_gemma_probe_argmax() -> usize {
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let eps = 0.000001f32;
        let layer_scalar = 3.0f32;
        let softcap = 6.0f32;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let mut residual = vec![0.0f32; hidden];
        let embedding_scale = (hidden as f32).sqrt();
        for dim in 0..hidden {
            residual[dim] = embedding[2 * hidden + dim] * embedding_scale;
        }

        let input_norm = vec![1.0f32; hidden];
        let mut q_norm = vec![1.0f32; hidden];
        let mut k_norm = vec![1.0f32; hidden];
        let mut v_norm = vec![1.0f32; hidden];
        let mut post_attn_norm = vec![1.0f32; hidden];
        let mut pre_ff_norm = vec![1.0f32; hidden];
        let mut post_ff_norm = vec![1.0f32; hidden];
        let final_norm = vec![1.0f32; hidden];
        q_norm[0] = 0.75;
        k_norm[0] = 0.5;
        v_norm[11] = 1.25;
        post_attn_norm[9] = 4.0;
        pre_ff_norm[9] = 1.0;
        post_ff_norm[9] = 2.0;

        let attn_normed = cpu_full_nonzero_rms_norm(&residual, &input_norm, eps);
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 0.5;
        o_proj[9 * hidden + 11] = 0.2;

        let q = cpu_full_nonzero_rms_norm(
            &cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed),
            &q_norm,
            eps,
        );
        let k = cpu_full_nonzero_rms_norm(
            &cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed),
            &k_norm,
            eps,
        );
        let v = cpu_full_nonzero_rms_norm(
            &cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed),
            &v_norm,
            eps,
        );
        let score =
            q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
        assert!(score.is_finite());

        let attn_residual = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &v);
        for dim in 0..hidden {
            residual[dim] += attn_residual[dim] * layer_scalar;
        }
        residual = cpu_full_nonzero_rms_norm(&residual, &post_attn_norm, eps);

        let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &pre_ff_norm, eps);
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        gate_proj[9] = 0.75;
        up_proj[9] = 0.75;
        down_proj[9 * intermediate] = 1.0;

        let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
        let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
        let activated = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
            .collect::<Vec<_>>();
        let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
        for dim in 0..hidden {
            residual[dim] += mlp_out[dim] * layer_scalar;
        }
        residual = cpu_full_nonzero_rms_norm(&residual, &post_ff_norm, eps);

        let final_hidden = cpu_full_nonzero_rms_norm(&residual, &final_norm, eps);
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 1.0;
        let mut logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
        for logit in &mut logits {
            *logit = softcap * (*logit / softcap).tanh();
        }
        cpu_full_nonzero_argmax(&logits)
    }

    fn cpu_reference_prompt_len_two_prefill_logits(include_first_prompt_token: bool) -> Vec<f32> {
        let hidden = 128usize;
        let vocab = 8usize;
        let eps = 0.000001f32;
        let scale = (hidden as f32).sqrt();

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;
        embedding[4 * hidden + 5] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];

        q_proj[5] = 1.0;
        k_proj[7] = 1.0;
        k_proj[5] = -1.0;
        v_proj[11 * hidden + 7] = 2.0;
        o_proj[9 * hidden + 11] = 2.0;
        lm_head[2 * hidden + 5] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;

        let token_residual = |token: usize| -> Vec<f32> {
            let mut residual = vec![0.0f32; hidden];
            for dim in 0..hidden {
                residual[dim] = embedding[token * hidden + dim] * scale;
            }
            residual
        };

        let prompt_tokens = if include_first_prompt_token {
            vec![2usize, 4usize]
        } else {
            vec![4usize]
        };

        let mut k_cache = Vec::with_capacity(prompt_tokens.len());
        let mut v_cache = Vec::with_capacity(prompt_tokens.len());
        for &token in &prompt_tokens {
            let residual = token_residual(token);
            let normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
            k_cache.push(cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &normed));
            v_cache.push(cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &normed));
        }

        let mut residual = token_residual(4);
        let normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &normed);
        let decode_k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &normed);
        let decode_v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &normed);
        let last_slot = k_cache.len() - 1;
        k_cache[last_slot] = decode_k;
        v_cache[last_slot] = decode_v;

        let mut scores = Vec::with_capacity(k_cache.len());
        for key in &k_cache {
            let score = q
                .iter()
                .zip(key.iter())
                .map(|(qv, kv)| qv * kv)
                .sum::<f32>()
                / (hidden as f32).sqrt();
            scores.push(score);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for score in &scores {
            denom += (*score - max_score).exp();
        }

        let mut attn_out = vec![0.0f32; hidden];
        for (idx, value) in v_cache.iter().enumerate() {
            let weight = (scores[idx] - max_score).exp() / denom;
            for dim in 0..hidden {
                attn_out[dim] += value[dim] * weight;
            }
        }

        let projected = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
        for dim in 0..hidden {
            residual[dim] += projected[dim];
        }

        let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden)
    }

    fn cpu_reference_prompt_len_two_prefill_argmax(include_first_prompt_token: bool) -> usize {
        cpu_full_nonzero_argmax(&cpu_reference_prompt_len_two_prefill_logits(
            include_first_prompt_token,
        ))
    }

    #[derive(Debug)]
    struct GeneratedTinyGemma4HfDecodeLoopReference {
        generated: Vec<usize>,
        logits_by_step: Vec<Vec<f32>>,
    }

    fn cpu_reference_generated_tiny_gemma4_hf_decode_loop(
        prompt: &[usize],
        steps: usize,
    ) -> GeneratedTinyGemma4HfDecodeLoopReference {
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let eps = 0.000001f32;
        let scale = (hidden as f32).sqrt();
        let softcap = 30.0f32;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;
        embedding[3 * hidden + 6] = 10.0;
        embedding[4 * hidden + 5] = 10.0;

        let norm = vec![1.0f32; hidden];
        let q_norm = vec![1.0f32; hidden];
        let k_norm = vec![1.0f32; hidden];
        let layer_scalar = vec![1.0f32; hidden];
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        let mut lm_head = vec![0.0f32; vocab * hidden];

        q_proj[5] = 0.5;
        k_proj[5] = 1.0;
        v_proj[9 * hidden + 5] = 1.0;
        o_proj[10 * hidden + 9] = 1.0;
        gate_proj[7] = 0.25;
        up_proj[7] = 0.25;
        down_proj[10 * intermediate] = 0.5;
        lm_head[2 * hidden + 10] = 0.25;
        lm_head[3 * hidden + 5] = 3.0;
        lm_head[5 * hidden + 6] = 3.0;

        let token_residual = |token: usize| -> Vec<f32> {
            let mut residual = vec![0.0f32; hidden];
            for dim in 0..hidden {
                residual[dim] = embedding[token * hidden + dim] * scale;
            }
            residual
        };

        let project_kv = |token: usize| -> (Vec<f32>, Vec<f32>) {
            let residual = token_residual(token);
            let normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
            let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &normed);
            let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &normed);
            (cpu_full_nonzero_rms_norm(&k, &k_norm, eps), v)
        };

        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for &token in prompt {
            let (k, v) = project_kv(token);
            k_cache.push(k);
            v_cache.push(v);
        }

        let mut current = *prompt.last().expect("nonempty prompt");
        let mut generated = Vec::new();
        let mut logits_by_step = Vec::new();
        for step in 0..steps {
            let position = prompt.len() - 1 + step;
            let mut residual = token_residual(current);
            let normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
            let q = cpu_full_nonzero_rms_norm(
                &cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &normed),
                &q_norm,
                eps,
            );
            let k = cpu_full_nonzero_rms_norm(
                &cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &normed),
                &k_norm,
                eps,
            );
            let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &normed);

            if position < k_cache.len() {
                k_cache[position] = k;
                v_cache[position] = v;
            } else {
                k_cache.push(k);
                v_cache.push(v);
            }

            let mut scores = Vec::with_capacity(k_cache.len());
            for key in &k_cache {
                let score = q
                    .iter()
                    .zip(key.iter())
                    .map(|(qv, kv)| qv * kv)
                    .sum::<f32>()
                    / (hidden as f32).sqrt();
                scores.push(score);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|score| (*score - max_score).exp())
                .sum::<f32>();

            let mut attn_out = vec![0.0f32; hidden];
            for (idx, value) in v_cache.iter().enumerate() {
                let weight = (scores[idx] - max_score).exp() / denom;
                for dim in 0..hidden {
                    attn_out[dim] += value[dim] * weight;
                }
            }

            let projected = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
            for dim in 0..hidden {
                residual[dim] += projected[dim] * layer_scalar[dim];
            }
            residual = cpu_full_nonzero_rms_norm(&residual, &norm, eps);

            let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
            let gate = cpu_full_nonzero_matvec(&gate_proj, intermediate, hidden, &mlp_normed);
            let up = cpu_full_nonzero_matvec(&up_proj, intermediate, hidden, &mlp_normed);
            let activated = gate
                .iter()
                .zip(up.iter())
                .map(|(g, u)| cpu_full_nonzero_gelu_tanh(*g) * u)
                .collect::<Vec<_>>();
            let mlp_out = cpu_full_nonzero_matvec(&down_proj, hidden, intermediate, &activated);
            for dim in 0..hidden {
                residual[dim] += mlp_out[dim] * layer_scalar[dim];
            }
            residual = cpu_full_nonzero_rms_norm(&residual, &norm, eps);

            let final_hidden = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
            let mut logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
            for logit in &mut logits {
                *logit = softcap * (*logit / softcap).tanh();
            }
            let next = cpu_full_nonzero_argmax(&logits);
            logits_by_step.push(logits);
            generated.push(next);
            current = next;
        }

        GeneratedTinyGemma4HfDecodeLoopReference {
            generated,
            logits_by_step,
        }
    }

    fn cpu_reference_generated_tiny_gemma4_hf_sequence(
        prompt: &[usize],
        steps: usize,
    ) -> Vec<usize> {
        cpu_reference_generated_tiny_gemma4_hf_decode_loop(prompt, steps).generated
    }

    fn cpu_reference_generated_tiny_hf_end_to_end_decode_loop(
    ) -> GeneratedTinyGemma4HfDecodeLoopReference {
        cpu_reference_generated_tiny_gemma4_hf_decode_loop(&[2, 4], 2)
    }

    fn cpu_reference_generated_tiny_hf_end_to_end_sequence() -> Vec<usize> {
        cpu_reference_generated_tiny_hf_end_to_end_decode_loop().generated
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_full_nonzero_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + FULL_NONZERO_ORIGINAL_DIM] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + FULL_NONZERO_ORIGINAL_DIM] = 1.0;
        lm_head[3 * hidden + FULL_NONZERO_ATTENTION_DIM] = 4.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        q_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.25;
        k_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.125;
        v_proj[FULL_NONZERO_VALUE_DIM * hidden + FULL_NONZERO_ORIGINAL_DIM] = 2.0;
        o_proj[FULL_NONZERO_ATTENTION_DIM * hidden + FULL_NONZERO_VALUE_DIM] = 6.0;
        gate_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.5;
        up_proj[FULL_NONZERO_ORIGINAL_DIM] = 0.5;
        down_proj[FULL_NONZERO_FFN_DIM * intermediate] = 4.0;

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &o_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &gate_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &up_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[test]
    fn cpu_reference_one_layer_full_nonzero_fixture_argmax_is_3() {
        assert_eq!(cpu_reference_one_layer_full_nonzero_argmax(), 3);
    }

    fn assert_f32_close(label: &str, got: f32, expected: f32, tolerance: f32) {
        let diff = (got - expected).abs();
        assert!(
            diff <= tolerance,
            "{label} mismatch: got={got} expected={expected} diff={diff} tol={tolerance}"
        );
    }

    fn assert_selected_logits_close(
        label: &str,
        got: &[f32],
        expected: &[f32],
        indices: &[usize],
        tolerance: f32,
    ) {
        assert_eq!(got.len(), expected.len());
        for &idx in indices {
            assert_f32_close(
                &format!("{label} logit[{idx}]"),
                got[idx],
                expected[idx],
                tolerance,
            );
        }
    }

    #[test]
    fn cpu_reference_one_layer_full_nonzero_selected_hidden_values_are_expected() {
        let reference = cpu_reference_one_layer_full_nonzero();

        assert_eq!(reference.residual_after_attention.len(), 128);
        assert_eq!(reference.residual.len(), 128);
        assert_eq!(reference.final_hidden.len(), 128);
        assert_f32_close(
            "residual zero dim",
            reference.residual[FULL_NONZERO_ZERO_DIM],
            0.0,
            0.0001,
        );
        assert_f32_close(
            "hidden zero dim",
            reference.final_hidden[FULL_NONZERO_ZERO_DIM],
            0.0,
            0.0001,
        );
        assert_f32_close(
            "residual original dim",
            reference.residual[FULL_NONZERO_ORIGINAL_DIM],
            113.137_085,
            0.0001,
        );
        assert_f32_close(
            "hidden original dim",
            reference.final_hidden[FULL_NONZERO_ORIGINAL_DIM],
            6.943_472_4,
            0.0001,
        );
        assert_f32_close(
            "residual attention dim",
            reference.residual[FULL_NONZERO_ATTENTION_DIM],
            135.764_5,
            0.0001,
        );
        assert_f32_close(
            "hidden attention dim",
            reference.final_hidden[FULL_NONZERO_ATTENTION_DIM],
            8.332_167,
            0.0001,
        );
        assert_f32_close(
            "residual ffn pre-update dim",
            reference.residual_after_attention[FULL_NONZERO_FFN_DIM],
            0.0,
            0.0001,
        );
        assert_f32_close(
            "residual ffn dim",
            reference.residual[FULL_NONZERO_FFN_DIM],
            52.453_545,
            0.0001,
        );
        assert_f32_close(
            "hidden ffn dim",
            reference.final_hidden[FULL_NONZERO_FFN_DIM],
            3.219_189_6,
            0.0001,
        );
        assert!(
            reference.final_hidden[FULL_NONZERO_ATTENTION_DIM]
                > reference.final_hidden[FULL_NONZERO_ORIGINAL_DIM]
        );
        assert!(
            reference.final_hidden[FULL_NONZERO_ORIGINAL_DIM]
                > reference.final_hidden[FULL_NONZERO_FFN_DIM]
        );
        assert!(
            reference.final_hidden[FULL_NONZERO_FFN_DIM]
                > reference.final_hidden[FULL_NONZERO_ZERO_DIM]
        );
    }

    #[test]
    fn cpu_reference_one_layer_full_nonzero_selected_logits_pick_token_3() {
        let logits = cpu_reference_one_layer_full_nonzero_logits();
        assert_eq!(logits.len(), 8);
        assert_eq!(cpu_full_nonzero_argmax(&logits), 3);
        assert_eq!(cpu_full_nonzero_top_two(&logits), (3, 2));
        assert_eq!(logits[0], 0.0);
        assert!(logits[3] > logits[2]);
        assert!(logits[2] > logits[0]);
    }

    #[test]
    fn cpu_reference_real_hf_style_one_layer_slice_argmax_is_3() {
        assert_eq!(cpu_reference_real_hf_style_one_layer_slice_argmax(), 3);
    }

    #[test]
    fn cpu_reference_one_layer_qkv_norm_nonzero_fixture_argmax_is_3() {
        assert_eq!(cpu_reference_one_layer_qkv_norm_nonzero_argmax(false), 2);
        assert_eq!(cpu_reference_one_layer_qkv_norm_nonzero_argmax(true), 3);
    }

    #[test]
    fn cpu_reference_one_layer_extra_norms_fixture_argmax_is_3() {
        assert_eq!(cpu_reference_one_layer_extra_norms_argmax(false), 2);
        assert_eq!(cpu_reference_one_layer_extra_norms_argmax(true), 3);
    }

    #[test]
    fn cpu_reference_one_layer_layer_scalar_fixture_argmax_is_3() {
        assert_eq!(cpu_reference_one_layer_layer_scalar_argmax(false), 2);
        assert_eq!(cpu_reference_one_layer_layer_scalar_argmax(true), 3);
    }

    #[test]
    fn cpu_reference_one_layer_integrated_gemma_probe_fixture_argmax_is_3() {
        assert_eq!(cpu_reference_one_layer_integrated_gemma_probe_argmax(), 3);
    }

    #[test]
    fn cpu_reference_prompt_len_two_prefill_fixture_argmax_is_3() {
        assert_eq!(cpu_reference_prompt_len_two_prefill_argmax(false), 2);
        assert_eq!(cpu_reference_prompt_len_two_prefill_argmax(true), 3);
    }

    #[test]
    fn cpu_reference_prompt_len_two_prefill_selected_logits_pick_token_3() {
        let without_first_logits = cpu_reference_prompt_len_two_prefill_logits(false);
        let include_first_logits = cpu_reference_prompt_len_two_prefill_logits(true);
        let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(&include_first_logits);
        let low_idx = 0usize;

        assert_eq!(without_first_logits.len(), 8);
        assert_eq!(include_first_logits.len(), 8);
        assert_eq!(cpu_full_nonzero_argmax(&without_first_logits), 2);
        assert_eq!(expected_idx, 3);
        assert_eq!(runner_up_idx, 2);
        assert_eq!(cpu_full_nonzero_argmax(&include_first_logits), expected_idx);
        assert_eq!(include_first_logits[low_idx], 0.0);
        assert!(include_first_logits[expected_idx] > include_first_logits[runner_up_idx]);
        assert!(include_first_logits[runner_up_idx] > include_first_logits[low_idx]);
        assert!(without_first_logits[2] > without_first_logits[3]);
        assert!(include_first_logits[3] > without_first_logits[3]);
    }

    #[test]
    fn cpu_reference_generated_tiny_hf_end_to_end_sequence_is_3_5() {
        assert_eq!(
            cpu_reference_generated_tiny_hf_end_to_end_sequence(),
            vec![3, 5]
        );
    }

    #[test]
    fn cpu_reference_decode_loop_selected_logits_sequence_is_expected() {
        let reference = cpu_reference_generated_tiny_hf_end_to_end_decode_loop();
        assert_eq!(reference.generated, vec![3, 5]);
        assert_eq!(reference.logits_by_step.len(), 2);

        let expected_selected_logits = [
            [
                (3usize, 24.286_85f32),
                (2usize, 0.281_43f32),
                (0usize, 0.0f32),
            ],
            [
                (5usize, 24.338_19f32),
                (2usize, 0.094_23f32),
                (0usize, 0.0f32),
            ],
        ];

        for (step_idx, selected_logits) in expected_selected_logits.iter().enumerate() {
            let logits = &reference.logits_by_step[step_idx];
            let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(logits);
            let low_idx = 0usize;

            assert_eq!(logits.len(), 8);
            assert_eq!(expected_idx, reference.generated[step_idx]);
            assert_eq!(runner_up_idx, 2);
            assert_eq!(logits[low_idx], 0.0);
            assert!(logits[expected_idx] > logits[runner_up_idx]);
            assert!(logits[runner_up_idx] > logits[low_idx]);

            for &(idx, expected_logit) in selected_logits {
                assert_f32_close(
                    &format!("decode step {} logit[{idx}]", step_idx + 1),
                    logits[idx],
                    expected_logit,
                    0.05,
                );
            }
        }

        let max_step_diff = reference.logits_by_step[0]
            .iter()
            .zip(reference.logits_by_step[1].iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_step_diff > 1.0,
            "decode steps should produce different logits, max_diff={max_step_diff}"
        );

        let cold_token_three = cpu_reference_generated_tiny_gemma4_hf_decode_loop(&[3], 1);
        let max_context_diff = reference.logits_by_step[1]
            .iter()
            .zip(cold_token_three.logits_by_step[0].iter())
            .map(|(persistent, cold)| (persistent - cold).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_context_diff > 0.05,
            "second decode step should depend on retained KV/context, max_diff={max_context_diff}"
        );
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_real_hf_style_one_layer_slice_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 2.0;
        o_proj[9 * hidden + 11] = 6.0;
        gate_proj[7] = 0.5;
        up_proj[7] = 0.5;
        down_proj[9 * intermediate] = 4.0;

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &o_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.post_attention_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &gate_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &up_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_extra_norms_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut post_attn_norm = vec![1.0f32; hidden];
        let pre_ff_norm = vec![1.0f32; hidden];
        let post_ff_norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];

        post_attn_norm[7] = 0.01;
        post_attn_norm[9] = 64.0;
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let gate_up = vec![0.0f32; 2 * intermediate * hidden];
        let down_proj = vec![0.0f32; hidden * intermediate];

        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 0.25;
        o_proj[9 * hidden + 11] = 0.5;

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &o_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.post_attention_layernorm.weight",
            &post_attn_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.pre_feedforward_layernorm.weight",
            &pre_ff_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.post_feedforward_layernorm.weight",
            &post_ff_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_up.weight",
            &gate_up,
            &[2 * intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_layer_scalar_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let layer_scalar = vec![6.0f32];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 1.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let zeros_qkvo = vec![0.0f32; hidden * hidden];
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];

        gate_proj[7] = 0.5;
        up_proj[7] = 0.5;
        down_proj[9 * intermediate] = 1.0;

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &zeros_qkvo,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.layer_scalar",
            &layer_scalar,
            &[1],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &gate_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &up_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_integrated_gemma_probe_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let input_norm = vec![1.0f32; hidden];
        let mut q_norm = vec![1.0f32; hidden];
        let mut k_norm = vec![1.0f32; hidden];
        let mut v_norm = vec![1.0f32; hidden];
        let mut post_attn_norm = vec![1.0f32; hidden];
        let mut pre_ff_norm = vec![1.0f32; hidden];
        let mut post_ff_norm = vec![1.0f32; hidden];
        let final_norm = vec![1.0f32; hidden];
        q_norm[0] = 0.75;
        k_norm[0] = 0.5;
        v_norm[11] = 1.25;
        post_attn_norm[9] = 4.0;
        pre_ff_norm[9] = 1.0;
        post_ff_norm[9] = 2.0;

        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 0.5;
        o_proj[9 * hidden + 11] = 0.2;

        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        gate_proj[9] = 0.75;
        up_proj[9] = 0.75;
        down_proj[9 * intermediate] = 1.0;

        let layer_scalar = vec![3.0f32];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 1.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &final_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &input_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &o_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_norm.weight",
            &q_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_norm.weight",
            &k_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_norm.weight",
            &v_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &pre_ff_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.post_attention_layernorm.weight",
            &post_attn_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.pre_feedforward_layernorm.weight",
            &pre_ff_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.post_feedforward_layernorm.weight",
            &post_ff_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.layer_scalar",
            &layer_scalar,
            &[1],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &gate_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &up_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 6.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_prompt_len_two_prefill_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;
        embedding[4 * hidden + 5] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let gate_proj = vec![0.0f32; intermediate * hidden];
        let up_proj = vec![0.0f32; intermediate * hidden];
        let down_proj = vec![0.0f32; hidden * intermediate];
        let mut lm_head = vec![0.0f32; vocab * hidden];

        q_proj[5] = 1.0;
        k_proj[7] = 1.0;
        k_proj[5] = -1.0;
        v_proj[11 * hidden + 7] = 2.0;
        o_proj[9 * hidden + 11] = 2.0;
        lm_head[2 * hidden + 5] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &o_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &gate_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &up_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_qkv_norm_nonzero_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;

        let norm = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 32.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let mut add_tensor = |name: &str,
                              data: &[f32],
                              shape: &[usize],
                              payload: &mut Vec<u8>,
                              header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            "model.embed_tokens.weight",
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.norm.weight",
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "lm_head.weight",
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let ones = vec![1.0f32; hidden];
        let mut q_norm = vec![1.0f32; hidden];
        let mut k_norm = vec![1.0f32; hidden];
        let v_norm = vec![1.0f32; hidden];
        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let gate_up = vec![0.0f32; 2 * intermediate * hidden];
        let down_proj = vec![0.0f32; hidden * intermediate];

        q_norm[0] = 0.5;
        k_norm[0] = 0.25;
        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 0.25;
        o_proj[9 * hidden + 11] = 0.5;

        add_tensor(
            "model.layers.0.input_layernorm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.q_norm.weight",
            &q_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.k_norm.weight",
            &k_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.v_norm.weight",
            &v_norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            &o_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp_norm.weight",
            &ones,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.gate_up.weight",
            &gate_up,
            &[2 * intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {},
    "vocab_size": {},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#,
            hidden, intermediate, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_generated_tiny_hf_end_to_end_fixture() -> std::path::PathBuf {
        let dir = temp_fixture_dir();
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;
        let prefix = "model.language_model";

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;
        embedding[3 * hidden + 6] = 10.0;
        embedding[4 * hidden + 5] = 10.0;

        let norm = vec![1.0f32; hidden];
        let layer_scalar = vec![1.0f32; hidden];
        let mut lm_head = vec![0.0f32; vocab * hidden];
        lm_head[2 * hidden + 10] = 0.25;
        lm_head[3 * hidden + 5] = 3.0;
        lm_head[5 * hidden + 6] = 3.0;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        let add_tensor = |name: &str,
                          data: &[f32],
                          shape: &[usize],
                          payload: &mut Vec<u8>,
                          header: &mut Map<String, Value>| {
            let start = payload.len();
            let bytes = f16_bytes(data);
            payload.extend_from_slice(&bytes);
            let end = payload.len();
            let mut meta = Map::new();
            meta.insert("dtype".to_owned(), Value::String("F16".to_string()));
            meta.insert(
                "shape".to_owned(),
                Value::Array(
                    shape
                        .iter()
                        .map(|n| Value::Number((*n as u64).into()))
                        .collect(),
                ),
            );
            meta.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![
                    Value::Number((start as u64).into()),
                    Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.to_string(), Value::Object(meta));
        };

        add_tensor(
            &format!("{prefix}.embed_tokens.weight"),
            &embedding,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.norm.weight"),
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.lm_head.weight"),
            &lm_head,
            &[vocab, hidden],
            &mut payload,
            &mut header,
        );

        let mut q_proj = vec![0.0f32; hidden * hidden];
        let mut k_proj = vec![0.0f32; hidden * hidden];
        let mut v_proj = vec![0.0f32; hidden * hidden];
        let mut o_proj = vec![0.0f32; hidden * hidden];
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];

        q_proj[5] = 0.5;
        k_proj[5] = 1.0;
        v_proj[9 * hidden + 5] = 1.0;
        o_proj[10 * hidden + 9] = 1.0;
        gate_proj[7] = 0.25;
        up_proj[7] = 0.25;
        down_proj[10 * intermediate] = 0.5;

        add_tensor(
            &format!("{prefix}.layers.0.input_layernorm.weight"),
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.self_attn.q_proj.weight"),
            &q_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.self_attn.k_proj.weight"),
            &k_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.self_attn.v_proj.weight"),
            &v_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.self_attn.o_proj.weight"),
            &o_proj,
            &[hidden, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.self_attn.q_norm.weight"),
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.self_attn.k_norm.weight"),
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.post_attention_layernorm.weight"),
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.pre_feedforward_layernorm.weight"),
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.post_feedforward_layernorm.weight"),
            &norm,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.layer_scalar"),
            &layer_scalar,
            &[hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.mlp.gate_proj.weight"),
            &gate_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.mlp.up_proj.weight"),
            &up_proj,
            &[intermediate, hidden],
            &mut payload,
            &mut header,
        );
        add_tensor(
            &format!("{prefix}.layers.0.mlp.down_proj.weight"),
            &down_proj,
            &[hidden, intermediate],
            &mut payload,
            &mut header,
        );

        let config = format!(
            r#"{{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {},
    "intermediate_size": {},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "num_global_key_value_heads": 1,
    "head_dim": {},
    "global_head_dim": {},
    "layer_types": ["full_attention"],
    "vocab_size": {},
    "max_position_embeddings": 16,
    "sliding_window": 8,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": false,
    "attention_k_eq_v": false,
    "rope_parameters": {{
      "sliding_attention": {{"rope_theta": 10000.0}},
      "full_attention": {{"rope_theta": 1000000.0}}
    }}
  }}
}}"#,
            hidden, intermediate, hidden, hidden, vocab
        );

        fs::write(dir.join("config.json"), config).expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out =
            File::create(dir.join("model.safetensors")).expect("create fixture safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_hf_style_noop_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_one_layer_hf_style_noop_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer hf-style tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_hf_style_noop_model_backend_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_one_layer_hf_style_noop_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny hf-style one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_real_hf_style_one_layer_slice_model_backend_decodes_cpu_expected_token() {
        let expected =
            rvllm_core::TokenId(cpu_reference_real_hf_style_one_layer_slice_argmax() as u32);
        let dir = write_tiny_real_hf_style_one_layer_slice_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare real-hf-style one-layer slice");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_real_hf_style_one_layer_slice_prefill_then_decode_cpu_expected_token() {
        let expected =
            rvllm_core::TokenId(cpu_reference_real_hf_style_one_layer_slice_argmax() as u32);
        let dir = write_tiny_real_hf_style_one_layer_slice_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with real-hf-style one-layer slice plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_extra_norms_model_backend_decodes_token_2_to_3() {
        let expected = rvllm_core::TokenId(cpu_reference_one_layer_extra_norms_argmax(true) as u32);
        let dir = write_tiny_one_layer_extra_norms_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer extra-norms tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_extra_norms_prefill_then_decode_token_2_to_3() {
        let expected = rvllm_core::TokenId(cpu_reference_one_layer_extra_norms_argmax(true) as u32);
        let dir = write_tiny_one_layer_extra_norms_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny extra-norms one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_layer_scalar_model_backend_decodes_token_2_to_3() {
        let expected =
            rvllm_core::TokenId(cpu_reference_one_layer_layer_scalar_argmax(true) as u32);
        let dir = write_tiny_one_layer_layer_scalar_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer layer-scalar tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_layer_scalar_prefill_then_decode_token_2_to_3() {
        let expected =
            rvllm_core::TokenId(cpu_reference_one_layer_layer_scalar_argmax(true) as u32);
        let dir = write_tiny_one_layer_layer_scalar_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny layer-scalar one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_integrated_gemma_probe_model_backend_decodes_token_2_to_3() {
        let expected =
            rvllm_core::TokenId(cpu_reference_one_layer_integrated_gemma_probe_argmax() as u32);
        let dir = write_tiny_one_layer_integrated_gemma_probe_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer integrated Gemma probe tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cpu_reference_zero_layer_decode_loop_sequence_is_3_4_5() {
        assert_eq!(
            cpu_reference_zero_layer_decode_loop_sequence(),
            vec![3, 4, 5]
        );
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn run_zero_layer_decode_once(
        dir: &std::path::Path,
    ) -> (Vec<StepToken>, MetalProbePerfStats, bool) {
        let mut backend = ModelMetalBackend::new(dir.to_path_buf());
        let plan = zero_layer_plan(dir.to_path_buf());
        backend
            .prepare(&plan)
            .expect("prepare zero-layer decode-loop tiny model");
        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );
        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect rollout");
        let stats = backend.probe_perf_stats();
        (out, stats, backend.metal_debug_sync_enabled())
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn metal_debug_sync_env_preserves_zero_layer_decode_output() {
        let env_guard = MetalDebugSyncEnvGuard::new();
        let dir = write_tiny_zero_layer_decode_loop_fixture();

        env_guard.set_current(None);
        let (normal_out, normal_stats, normal_debug_sync) = run_zero_layer_decode_once(&dir);

        env_guard.set_current(Some("1"));
        let (debug_out, debug_stats, debug_debug_sync) = run_zero_layer_decode_once(&dir);

        assert!(!normal_debug_sync);
        assert!(debug_debug_sync);
        assert_eq!(normal_out, debug_out);
        assert_eq!(debug_out[0].token_id, rvllm_core::TokenId(3));
        assert!(normal_stats.command_buffers > 0);
        assert!(normal_stats.encoders > 0);
        assert!(normal_stats.forced_waits > 0);
        assert!(debug_stats.forced_waits > normal_stats.forced_waits);
        assert_eq!(debug_stats.decode_steps, 1);
        assert_eq!(debug_stats.last_step_tokens, 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn metal_probe_perf_counters_are_populated_and_monotonic() {
        let env_guard = MetalDebugSyncEnvGuard::new();
        env_guard.set_current(None);
        let dir = write_tiny_zero_layer_decode_loop_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = zero_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare zero-layer decode-loop tiny model");

        let first = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );
        let first_ticket = backend
            .launch_rollout(&first, None)
            .expect("run first rollout");
        let first_out = backend
            .collect(first_ticket)
            .expect("collect first rollout");
        assert_eq!(first_out[0].token_id, rvllm_core::TokenId(3));
        let first_stats = backend.probe_perf_stats();
        assert_eq!(first_stats.decode_steps, 1);
        assert_eq!(first_stats.tokens, 1);
        assert_eq!(first_stats.last_step_tokens, 1);
        assert!(first_stats.command_buffers > 0);
        assert!(first_stats.encoders > 0);
        assert!(first_stats.forced_waits > 0);
        assert!(first_stats.last_step_command_buffers > 0);
        assert!(first_stats.last_step_encoders > 0);
        assert!(first_stats.last_step_forced_waits > 0);
        assert!(first_stats.last_step_cpu_wall_ns > 0);

        let second = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![first_out[0].token_id],
            vec![0, 1],
            vec![1],
            vec![2],
        );
        let second_ticket = backend
            .launch_rollout(&second, None)
            .expect("run second rollout");
        let second_out = backend
            .collect(second_ticket)
            .expect("collect second rollout");
        assert_eq!(second_out[0].token_id, rvllm_core::TokenId(4));
        let second_stats = backend.probe_perf_stats();
        assert_eq!(second_stats.decode_steps, 2);
        assert_eq!(second_stats.tokens, 2);
        assert!(second_stats.command_buffers > first_stats.command_buffers);
        assert!(second_stats.encoders > first_stats.encoders);
        assert!(second_stats.forced_waits > first_stats.forced_waits);
        assert!(second_stats.cpu_wall_ns >= first_stats.cpu_wall_ns);
        assert_eq!(second_stats.last_step_tokens, 1);
        assert!(second_stats.last_step_command_buffers > 0);
        assert!(second_stats.last_step_encoders > 0);
        assert!(second_stats.last_step_forced_waits > 0);
        assert!(second_stats.last_step_cpu_wall_ns > 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_zero_layer_decode_loop_model_backend_generates_3_4_5() {
        let expected = cpu_reference_zero_layer_decode_loop_sequence()
            .into_iter()
            .map(|token| rvllm_core::TokenId(token as u32))
            .collect::<Vec<_>>();
        let dir = write_tiny_zero_layer_decode_loop_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = zero_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare zero-layer decode-loop tiny model");

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![
                rvllm_core::TokenId(0),
                rvllm_core::TokenId(1),
                rvllm_core::TokenId(2),
            ],
            vec![0, 3],
            vec![2],
            vec![3],
        );
        let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
        let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
        assert!(prefill_out.is_empty());

        let mut current = rvllm_core::TokenId(2);
        let mut generated = Vec::new();
        for (position, context_len) in [(2, 3), (3, 4), (4, 5)] {
            let decode = rvllm_apple::HandoffCapsule::new(
                rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
                vec![rvllm_core::ReqId(1)],
                vec![current],
                vec![0, 1],
                vec![position],
                vec![context_len],
            );
            let ticket = backend.launch_rollout(&decode, None).expect("run rollout");
            let out = backend.collect(ticket).expect("collect rollout");
            assert_eq!(out.len(), 1);
            current = out[0].token_id;
            generated.push(current);
        }

        assert_eq!(generated, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_zero_layer_decode_loop_generates_3_4_5() {
        let expected = cpu_reference_zero_layer_decode_loop_sequence()
            .into_iter()
            .map(|token| rvllm_core::TokenId(token as u32))
            .collect::<Vec<_>>();
        let dir = write_tiny_zero_layer_decode_loop_fixture();
        let plan = zero_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with zero-layer decode-loop tiny model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![
                rvllm_core::TokenId(0),
                rvllm_core::TokenId(1),
                rvllm_core::TokenId(2),
            ],
            3,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let mut generated = Vec::new();
        for expected_token in &expected {
            let step = engine.step_launch().expect("launch decode");
            let out = step.collect().expect("collect decode");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
            assert_eq!(&out[0].new_token, expected_token);
            generated.push(out[0].new_token);
        }

        assert_eq!(generated, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_zero_layer_decode_batch_two_returns_independent_tokens() {
        let dir = write_tiny_zero_layer_decode_loop_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = zero_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare zero-layer decode-loop tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(3)],
            vec![0, 1, 2],
            vec![0, 1],
            vec![1, 2],
        );
        let ticket = backend
            .launch_rollout(&handoff, None)
            .expect("run batched rollout");
        let out = backend.collect(ticket).expect("collect batched rollout");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));
        assert_eq!(out[1].req_id, rvllm_core::ReqId(2));
        assert_eq!(out[1].token_id, rvllm_core::TokenId(4));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_zero_layer_decode_batch_two_returns_exact_tokens() {
        let dir = write_tiny_zero_layer_decode_loop_fixture();
        let plan = zero_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with zero-layer decode-loop tiny model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));
        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(2),
            vec![rvllm_core::TokenId(0), rvllm_core::TokenId(3)],
            1,
        ));

        let prefill = engine.step_launch().expect("launch batched prefill");
        match prefill.plan().expect("prefill plan") {
            crate::scheduler::BatchPlan::Prefill { req_ids, .. } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
            }
            other => panic!("expected Prefill, got {other:?}"),
        }
        assert!(prefill
            .collect()
            .expect("collect batched prefill")
            .is_empty());

        let decode = engine.step_launch().expect("launch batched decode");
        match decode.plan().expect("decode plan") {
            crate::scheduler::BatchPlan::Decode {
                req_ids,
                bucket,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(req_ids, &vec![rvllm_core::ReqId(1), rvllm_core::ReqId(2)]);
                assert_eq!(*bucket, 2);
                assert_eq!(positions, &vec![0, 1]);
                assert_eq!(context_lens, &vec![1, 2]);
            }
            other => panic!("expected Decode, got {other:?}"),
        }
        let out = decode.collect().expect("collect batched decode");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out[0].new_token, rvllm_core::TokenId(3));
        assert_eq!(out[1].req_id, rvllm_core::ReqId(2));
        assert_eq!(out[1].new_token, rvllm_core::TokenId(4));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_zero_layer_decode_batch_four_returns_exact_tokens() {
        let dir = write_tiny_zero_layer_decode_loop_fixture();
        let plan = zero_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with zero-layer decode-loop tiny model plan");

        for (req_id, prompt) in [
            (1, vec![rvllm_core::TokenId(2)]),
            (2, vec![rvllm_core::TokenId(0), rvllm_core::TokenId(3)]),
            (
                3,
                vec![
                    rvllm_core::TokenId(0),
                    rvllm_core::TokenId(1),
                    rvllm_core::TokenId(4),
                ],
            ),
            (4, vec![rvllm_core::TokenId(2)]),
        ] {
            engine.scheduler.enqueue(crate::sched_state::Request::new(
                rvllm_core::ReqId(req_id),
                prompt,
                1,
            ));
        }

        let prefill = engine.step_launch().expect("launch batched prefill");
        assert!(prefill
            .collect()
            .expect("collect batched prefill")
            .is_empty());

        let decode = engine.step_launch().expect("launch batched decode");
        match decode.plan().expect("decode plan") {
            crate::scheduler::BatchPlan::Decode {
                bucket,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(*bucket, 4);
                assert_eq!(positions, &vec![0, 1, 2, 0]);
                assert_eq!(context_lens, &vec![1, 2, 3, 1]);
            }
            other => panic!("expected Decode, got {other:?}"),
        }
        let out = decode.collect().expect("collect batched decode");
        let got = out
            .iter()
            .map(|step| (step.req_id, step.new_token))
            .collect::<Vec<_>>();
        assert_eq!(
            got,
            vec![
                (rvllm_core::ReqId(1), rvllm_core::TokenId(3)),
                (rvllm_core::ReqId(2), rvllm_core::TokenId(4)),
                (rvllm_core::ReqId(3), rvllm_core::TokenId(5)),
                (rvllm_core::ReqId(4), rvllm_core::TokenId(3)),
            ]
        );
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_integrated_gemma_probe_prefill_then_decode_token_2_to_3() {
        let expected =
            rvllm_core::TokenId(cpu_reference_one_layer_integrated_gemma_probe_argmax() as u32);
        let dir = write_tiny_one_layer_integrated_gemma_probe_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny integrated Gemma probe one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_prompt_len_two_model_backend_prefill_then_decode_token_2_4_to_3() {
        let expected =
            rvllm_core::TokenId(cpu_reference_prompt_len_two_prefill_argmax(true) as u32);
        let dir = write_tiny_prompt_len_two_prefill_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare prompt length two tiny model");

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
        let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
        assert!(prefill_out.is_empty());

        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(4)],
            vec![0, 1],
            vec![1],
            vec![2],
        );
        let decode_ticket = backend.launch_rollout(&decode, None).expect("run rollout");
        let out = backend.collect(decode_ticket).expect("collect rollout");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_prompt_len_two_prefill_selected_logits_match_cpu() {
        let cpu_logits = cpu_reference_prompt_len_two_prefill_logits(true);
        let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(&cpu_logits);
        let low_idx = 0usize;
        assert_eq!(expected_idx, 3);
        assert_eq!(runner_up_idx, 2);
        assert_eq!(cpu_logits[low_idx], 0.0);

        let dir = write_tiny_prompt_len_two_prefill_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare prompt length two tiny model");

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
        let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
        assert!(prefill_out.is_empty());

        let decode = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(4)],
            vec![0, 1],
            vec![1],
            vec![2],
        );
        let decode_ticket = backend.launch_rollout(&decode, None).expect("run rollout");
        let metal_logits = backend
            .debug_read_decode_logits_f32(1)
            .expect("read decode logits");
        let out = backend.collect(decode_ticket).expect("collect rollout");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(expected_idx as u32));

        const LOGIT_TOLERANCE: f32 = 0.05;
        assert_selected_logits_close(
            "prompt length two direct backend",
            &metal_logits,
            &cpu_logits,
            &[expected_idx, runner_up_idx, low_idx],
            LOGIT_TOLERANCE,
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_prompt_len_two_prefill_then_decode_token_2_4_to_3() {
        let expected =
            rvllm_core::TokenId(cpu_reference_prompt_len_two_prefill_argmax(true) as u32);
        let dir = write_tiny_prompt_len_two_prefill_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with prompt length two tiny model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_prompt_len_two_prefill_selected_logits_match_cpu() {
        let cpu_logits = cpu_reference_prompt_len_two_prefill_logits(true);
        let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(&cpu_logits);
        let low_idx = 0usize;
        assert_eq!(expected_idx, 3);
        assert_eq!(runner_up_idx, 2);
        assert_eq!(cpu_logits[low_idx], 0.0);

        let dir = write_tiny_prompt_len_two_prefill_fixture();
        let plan = one_layer_plan(dir.clone());
        let shared_backend = SharedModelMetalBackend::new(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_backend(Box::new(shared_backend.clone()))
            .with_apple_runtime_plan(plan)
            .expect("engine with shared prompt length two tiny model backend");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(expected_idx as u32));
        assert!(!engine.has_pending_work());

        let metal_logits = shared_backend
            .debug_read_decode_logits_f32(1)
            .expect("read shared backend decode logits");
        const LOGIT_TOLERANCE: f32 = 0.05;
        assert_selected_logits_close(
            "prompt length two engine backend",
            &metal_logits,
            &cpu_logits,
            &[expected_idx, runner_up_idx, low_idx],
            LOGIT_TOLERANCE,
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_qkv_norm_nonzero_model_backend_decodes_token_2_to_3() {
        let expected =
            rvllm_core::TokenId(cpu_reference_one_layer_qkv_norm_nonzero_argmax(true) as u32);
        let dir = write_tiny_one_layer_qkv_norm_nonzero_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer qkv-norm tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_qkv_norm_nonzero_prefill_then_decode_token_2_to_3() {
        let expected =
            rvllm_core::TokenId(cpu_reference_one_layer_qkv_norm_nonzero_argmax(true) as u32);
        let dir = write_tiny_one_layer_qkv_norm_nonzero_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny qkv-norm one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_generated_gemma4_hf_model_backend_one_prompt_token_matches_cpu_token() {
        let expected =
            rvllm_core::TokenId(cpu_reference_generated_tiny_gemma4_hf_sequence(&[2], 1)[0] as u32);
        let dir = write_generated_tiny_hf_end_to_end_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare generated tiny Gemma4 HF-named model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_generated_gemma4_hf_end_to_end_model_backend_matches_cpu_tokens() {
        let expected = cpu_reference_generated_tiny_hf_end_to_end_sequence()
            .into_iter()
            .map(|token| rvllm_core::TokenId(token as u32))
            .collect::<Vec<_>>();
        let dir = write_generated_tiny_hf_end_to_end_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare generated tiny Gemma4 HF-named model");

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
        let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
        assert!(prefill_out.is_empty());

        let mut current = rvllm_core::TokenId(4);
        let mut generated = Vec::new();
        for (idx, expected_token) in expected.iter().enumerate() {
            let decode = rvllm_apple::HandoffCapsule::new(
                rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
                vec![rvllm_core::ReqId(1)],
                vec![current],
                vec![0, 1],
                vec![1 + idx as u32],
                vec![2 + idx as u32],
            );
            let ticket = backend.launch_rollout(&decode, None).expect("run rollout");
            let out = backend.collect(ticket).expect("collect rollout");
            assert_eq!(out.len(), 1);
            assert_eq!(&out[0].token_id, expected_token);
            current = out[0].token_id;
            generated.push(current);
        }

        assert_eq!(generated, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_generated_gemma4_hf_end_to_end_model_backend_selected_logits_match_cpu() {
        let cpu_reference = cpu_reference_generated_tiny_hf_end_to_end_decode_loop();
        assert_eq!(cpu_reference.generated, vec![3, 5]);

        let dir = write_generated_tiny_hf_end_to_end_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare generated tiny Gemma4 HF-named model");

        let prefill = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            vec![0, 2],
            vec![1],
            vec![2],
        );
        let prefill_ticket = backend.launch_prefill(&prefill).expect("run prefill");
        let prefill_out = backend.collect(prefill_ticket).expect("collect prefill");
        assert!(prefill_out.is_empty());

        let mut current = rvllm_core::TokenId(4);
        let mut generated = Vec::new();
        for (step_idx, expected_token) in cpu_reference.generated.iter().enumerate() {
            let decode = rvllm_apple::HandoffCapsule::new(
                rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
                vec![rvllm_core::ReqId(1)],
                vec![current],
                vec![0, 1],
                vec![1 + step_idx as u32],
                vec![2 + step_idx as u32],
            );
            let ticket = backend.launch_rollout(&decode, None).expect("run rollout");
            let metal_logits = backend
                .debug_read_decode_logits_f32(1)
                .expect("read decode logits");
            let out = backend.collect(ticket).expect("collect rollout");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].token_id, rvllm_core::TokenId(*expected_token as u32));
            current = out[0].token_id;
            generated.push(current);

            let cpu_logits = &cpu_reference.logits_by_step[step_idx];
            let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(cpu_logits);
            let low_idx = 0usize;
            assert_eq!(expected_idx, *expected_token);
            assert_eq!(metal_logits.len(), cpu_logits.len());

            const LOGIT_TOLERANCE: f32 = 0.05;
            assert_selected_logits_close(
                &format!("generated tiny HF direct decode step {}", step_idx + 1),
                &metal_logits,
                cpu_logits,
                &[expected_idx, runner_up_idx, low_idx],
                LOGIT_TOLERANCE,
            );
        }

        let expected = cpu_reference
            .generated
            .iter()
            .map(|token| rvllm_core::TokenId(*token as u32))
            .collect::<Vec<_>>();
        assert_eq!(generated, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_generated_gemma4_hf_end_to_end_matches_cpu_tokens() {
        let expected = cpu_reference_generated_tiny_hf_end_to_end_sequence()
            .into_iter()
            .map(|token| rvllm_core::TokenId(token as u32))
            .collect::<Vec<_>>();
        let dir = write_generated_tiny_hf_end_to_end_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with generated tiny Gemma4 HF-named model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            expected.len() as u32,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let mut generated = Vec::new();
        for expected_token in &expected {
            let step = engine.step_launch().expect("launch decode");
            let out = step.collect().expect("collect decode");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
            assert_eq!(&out[0].new_token, expected_token);
            generated.push(out[0].new_token);
        }

        assert_eq!(generated, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_decode_loop_selected_logits_match_cpu() {
        let cpu_reference = cpu_reference_generated_tiny_hf_end_to_end_decode_loop();
        assert_eq!(cpu_reference.generated, vec![3, 5]);

        let dir = write_generated_tiny_hf_end_to_end_fixture();
        let plan = one_layer_plan(dir.clone());
        let shared_backend = SharedModelMetalBackend::new(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_backend(Box::new(shared_backend.clone()))
            .with_apple_runtime_plan(plan)
            .expect("engine with shared generated tiny Gemma4 HF-named model backend");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2), rvllm_core::TokenId(4)],
            cpu_reference.generated.len() as u32,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let mut generated = Vec::new();
        for (step_idx, expected_token) in cpu_reference.generated.iter().enumerate() {
            let step = engine.step_launch().expect("launch decode");
            let out = step.collect().expect("collect decode");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].req_id, rvllm_core::ReqId(1));
            assert_eq!(
                out[0].new_token,
                rvllm_core::TokenId(*expected_token as u32)
            );
            generated.push(out[0].new_token);

            let metal_logits = shared_backend
                .debug_read_decode_logits_f32(1)
                .expect("read shared backend decode logits");
            let cpu_logits = &cpu_reference.logits_by_step[step_idx];
            let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(cpu_logits);
            let low_idx = 0usize;
            assert_eq!(expected_idx, *expected_token);
            assert_eq!(metal_logits.len(), cpu_logits.len());

            const LOGIT_TOLERANCE: f32 = 0.05;
            assert_selected_logits_close(
                &format!("generated tiny HF engine decode step {}", step_idx + 1),
                &metal_logits,
                cpu_logits,
                &[expected_idx, runner_up_idx, low_idx],
                LOGIT_TOLERANCE,
            );
        }

        let expected = cpu_reference
            .generated
            .iter()
            .map(|token| rvllm_core::TokenId(*token as u32))
            .collect::<Vec<_>>();
        assert_eq!(generated, expected);
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_ffn_nonzero_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_one_layer_ffn_nonzero_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer ffn-nonzero tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_ffn_nonzero_model_backend_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_one_layer_ffn_nonzero_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny ffn-nonzero one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_attention_nonzero_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_one_layer_attention_nonzero_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer attention-nonzero tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_attention_nonzero_model_backend_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_one_layer_attention_nonzero_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny attention-nonzero one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_full_nonzero_model_backend_decodes_token_2_to_3() {
        let dir = write_tiny_one_layer_full_nonzero_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer full-nonzero tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(3));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_full_nonzero_model_backend_selected_logits_match_cpu() {
        let cpu_logits = cpu_reference_one_layer_full_nonzero_logits();
        let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(&cpu_logits);
        let low_idx = 0usize;
        assert_eq!(expected_idx, 3);
        assert_eq!(runner_up_idx, 2);
        assert_eq!(cpu_logits[low_idx], 0.0);

        let dir = write_tiny_one_layer_full_nonzero_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer full-nonzero tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let metal_logits = backend
            .debug_read_decode_logits_f32(1)
            .expect("read decode logits");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(expected_idx as u32));
        assert_eq!(metal_logits.len(), cpu_logits.len());

        const LOGIT_TOLERANCE: f32 = 0.05;
        for idx in [expected_idx, runner_up_idx, low_idx] {
            let diff = (metal_logits[idx] - cpu_logits[idx]).abs();
            assert!(
                diff <= LOGIT_TOLERANCE,
                "logit[{idx}] mismatch: metal={} cpu={} diff={} tol={}",
                metal_logits[idx],
                cpu_logits[idx],
                diff,
                LOGIT_TOLERANCE
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn tiny_one_layer_full_nonzero_model_backend_selected_hidden_matches_cpu() {
        let reference = cpu_reference_one_layer_full_nonzero();
        let cpu_logits = &reference.logits;
        let (expected_idx, runner_up_idx) = cpu_full_nonzero_top_two(cpu_logits);
        let low_idx = FULL_NONZERO_ZERO_DIM;
        assert_eq!(expected_idx, 3);
        assert_eq!(runner_up_idx, 2);
        assert_eq!(cpu_logits[low_idx], 0.0);

        let selected_dims = [
            FULL_NONZERO_ZERO_DIM,
            FULL_NONZERO_ORIGINAL_DIM,
            FULL_NONZERO_ATTENTION_DIM,
            FULL_NONZERO_FFN_DIM,
        ];

        let dir = write_tiny_one_layer_full_nonzero_fixture();
        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = one_layer_plan(dir.clone());
        backend
            .prepare(&plan)
            .expect("prepare one-layer full-nonzero tiny model");

        let handoff = rvllm_apple::HandoffCapsule::new(
            rvllm_apple::HandoffKind::MetalPrefillToMetalDecode,
            vec![rvllm_core::ReqId(1)],
            vec![rvllm_core::TokenId(2)],
            vec![0, 1],
            vec![0],
            vec![1],
        );

        let ticket = backend.launch_rollout(&handoff, None).expect("run rollout");
        let metal_residual = backend
            .debug_read_residual_f32(1)
            .expect("read decode residual");
        let metal_logits = backend
            .debug_read_decode_logits_f32(1)
            .expect("read decode logits");
        let out = backend.collect(ticket).expect("collect");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_id, rvllm_core::TokenId(expected_idx as u32));
        assert_eq!(metal_residual.len(), reference.residual.len());
        assert_eq!(metal_logits.len(), cpu_logits.len());

        const HIDDEN_TOLERANCE: f32 = 0.05;
        for dim in selected_dims {
            assert_f32_close(
                &format!("residual[{dim}]"),
                metal_residual[dim],
                reference.residual[dim],
                HIDDEN_TOLERANCE,
            );
        }

        const LOGIT_TOLERANCE: f32 = 0.05;
        for idx in [expected_idx, runner_up_idx, low_idx] {
            assert_f32_close(
                &format!("logit[{idx}]"),
                metal_logits[idx],
                cpu_logits[idx],
                LOGIT_TOLERANCE,
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon Metal device"]
    fn engine_one_layer_full_nonzero_model_backend_prefill_then_decode_token_2_to_3() {
        let dir = write_tiny_one_layer_full_nonzero_fixture();
        let plan = one_layer_plan(dir.clone());

        let mut engine = crate::engine::Engine::new()
            .with_apple_runtime_plan(plan)
            .expect("engine with tiny full-nonzero one-layer model plan");

        engine.scheduler.enqueue(crate::sched_state::Request::new(
            rvllm_core::ReqId(1),
            vec![rvllm_core::TokenId(2)],
            1,
        ));

        let step1 = engine.step_launch().expect("launch prefill");
        let out1 = step1.collect().expect("collect prefill");
        assert!(out1.is_empty());

        let step2 = engine.step_launch().expect("launch decode");
        let out2 = step2.collect().expect("collect decode");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].req_id, rvllm_core::ReqId(1));
        assert_eq!(out2[0].new_token, rvllm_core::TokenId(3));
        assert!(!engine.has_pending_work());

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    fn generated_tiny_gemma4_hf_fixture_uses_real_names_and_dry_run_validates() {
        let dir = write_generated_tiny_hf_end_to_end_fixture();
        let tensors =
            rvllm_apple_metal::weight_loader::scan_safetensor_tensors(&dir).expect("scan");
        let prefix = "model.language_model";

        assert!(tensors.contains_key(&format!("{prefix}.embed_tokens.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.norm.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.lm_head.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.q_proj.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.k_proj.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.v_proj.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.q_norm.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.self_attn.k_norm.weight")));
        assert!(tensors.contains_key(&format!(
            "{prefix}.layers.0.post_attention_layernorm.weight"
        )));
        assert!(tensors.contains_key(&format!(
            "{prefix}.layers.0.pre_feedforward_layernorm.weight"
        )));
        assert!(tensors.contains_key(&format!(
            "{prefix}.layers.0.post_feedforward_layernorm.weight"
        )));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.layer_scalar")));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.mlp.gate_proj.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.mlp.up_proj.weight")));
        assert!(tensors.contains_key(&format!("{prefix}.layers.0.mlp.down_proj.weight")));
        assert!(!tensors.contains_key("model.layers.0.self_attn.qkv.weight"));
        assert!(!tensors.contains_key("model.layers.0.mlp.gate_up.weight"));

        let validation = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect("generated Gemma4 fixture dry-run validates");
        assert_eq!(validation.weight_prefix, prefix);
        assert_eq!(validation.final_logit_softcap, Some(30.0));
        assert_eq!(
            validation.layers[0].attention_kind,
            rvllm_apple_metal::gemma4_model::MetalProbeLayerAttentionKind::Full
        );
        assert_eq!(validation.layers[0].layer_scalar_dim, 128);

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    fn hf_style_one_layer_fixture_has_separate_tensors() {
        let dir = write_tiny_one_layer_hf_style_noop_fixture();
        let tensors =
            rvllm_apple_metal::weight_loader::scan_safetensor_tensors(&dir).expect("scan");
        assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.self_attn.k_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.self_attn.v_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.mlp.gate_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.mlp.up_proj.weight"));
        assert!(!tensors.contains_key("model.layers.0.self_attn.qkv.weight"));
        assert!(!tensors.contains_key("model.layers.0.mlp.gate_up.weight"));
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    fn real_hf_style_one_layer_slice_fixture_has_hf_names_and_norm_alias() {
        let dir = write_tiny_real_hf_style_one_layer_slice_fixture();
        let tensors =
            rvllm_apple_metal::weight_loader::scan_safetensor_tensors(&dir).expect("scan");
        assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.self_attn.k_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.self_attn.v_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.self_attn.o_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.mlp.gate_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.mlp.up_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.mlp.down_proj.weight"));
        assert!(tensors.contains_key("model.layers.0.input_layernorm.weight"));
        assert!(tensors.contains_key("model.layers.0.post_attention_layernorm.weight"));
        assert!(!tensors.contains_key("model.layers.0.mlp_norm.weight"));
        assert!(!tensors.contains_key("model.layers.0.self_attn.qkv.weight"));
        assert!(!tensors.contains_key("model.layers.0.mlp.gate_up.weight"));
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    #[test]
    fn model_metal_backend_prepare_rejects_missing_dir() {
        let dir = std::env::temp_dir().join("rvllm-definitely-missing-model-dir");
        let _ = fs::remove_dir_all(&dir);

        let mut backend = ModelMetalBackend::new(dir.clone());
        let plan = zero_layer_plan(dir);
        let err = backend.prepare(&plan).expect_err("missing dir should fail");
        let s = format!("{err}");
        assert!(s.contains("InvalidWeightBlob") || s.contains("missing model path"));
    }
}
