use rvllm_core::{AppleCtx, AppleError, Result, RvllmError, TokenId};
use rvllm_apple::{AppleBackend, AppleLaunchKind, AppleLaunchTicket, HandoffCapsule, StepToken};
use half::f16;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::cmp::max;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::path::PathBuf;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::ptr;
#[cfg(all(feature = "apple", target_os = "macos"))]
use rvllm_loader::load::ModelArch;

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
use rvllm_apple::{RolloutBucket};
#[cfg(all(feature = "apple", target_os = "macos"))]
use objc2_metal::MTLSize;
#[cfg(all(feature = "apple", target_os = "macos"))]
use rvllm_apple_metal::{
    kernels,
    context::MetalContext,
    layer_forward::metal_finalize_logits_blocking,
    gemma4_model::Gemma4MetalState,
    pipeline::PipelineCache,
    weight_loader::{map_safetensor_to_arena, scan_safetensor_tensors},
};
#[cfg(all(feature = "apple", target_os = "macos"))]
use rvllm_apple_metal::arena::{MetalBufferArena, MetalRegion};

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

    fn resolve_weight_prefix(
        tensors: &std::collections::BTreeMap<String, rvllm_apple_metal::weight_loader::SafetensorTensorInfo>,
    ) -> String {
        if tensors.contains_key("model.embed_tokens.weight") {
            "model".to_owned()
        } else if tensors.contains_key("model.language_model.embed_tokens.weight") {
            "model.language_model".to_owned()
        } else if tensors.contains_key("language_model.model.embed_tokens.weight") {
            "language_model.model".to_owned()
        } else {
            "model".to_owned()
        }
    }

    fn region_lookup(
        refs: &mut Vec<(String, MetalRegion)>,
        name: &str,
    ) -> Result<MetalRegion> {
        let idx = refs.iter().position(|(n, _)| n == name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing mapped tensor in map",
                },
                model_ctx("resolve_regions"),
            )
        })?;
        Ok(refs.swap_remove(idx).1)
    }

    fn initialize_model_resources(&mut self) -> Result<Gemma4MetalState> {
        let arch = ModelArch::from_dir(&self.model_dir)?;
        if arch.num_hidden_layers > 0 {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_num_layers",
                },
                model_ctx("prepare"),
            ));
        }
        if arch.hidden_size == 0 || arch.vocab_size == 0 {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "model has zero hidden_size or vocab_size",
                },
                model_ctx("prepare"),
            ));
        }

        let tensors = scan_safetensor_tensors(&self.model_dir)?;
        let wprefix = Self::resolve_weight_prefix(&tensors);
        let embed_name = format!("{wprefix}.embed_tokens.weight");
        let final_norm_name = format!("{wprefix}.norm.weight");
        let prefixed_lm_head_name = format!("{wprefix}.lm_head.weight");
        let tie_embeddings =
            !tensors.contains_key("lm_head.weight") && !tensors.contains_key(&prefixed_lm_head_name);
        let lm_head_name = if tie_embeddings {
            embed_name.clone()
        } else if tensors.contains_key("lm_head.weight") {
            "lm_head.weight".to_owned()
        } else if tensors.contains_key(&prefixed_lm_head_name) {
            prefixed_lm_head_name.clone()
        } else {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing lm_head weights",
                },
                model_ctx("prepare"),
            ));
        };

        let embed_info = tensors.get(&embed_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing embed_tokens.weight",
                },
                model_ctx("prepare"),
            )
        })?;
        if embed_info.shape.len() != 2
            || embed_info.shape[0] != arch.vocab_size
            || embed_info.shape[1] != arch.hidden_size
        {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "embed_tokens weight shape mismatch",
                },
                model_ctx("prepare"),
            ));
        }

        let final_norm_info = tensors.get(&final_norm_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing final layer norm weight",
                },
                model_ctx("prepare"),
            )
        })?;
        if final_norm_info.shape.len() != 1 || final_norm_info.shape[0] != arch.hidden_size {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "final layer norm shape mismatch",
                },
                model_ctx("prepare"),
            ));
        }

        let lm_head_info = tensors.get(&lm_head_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing lm_head weight",
                },
                model_ctx("prepare"),
            )
        })?;
        if !tie_embeddings {
            if lm_head_info.shape.len() != 2
                || lm_head_info.shape[0] != arch.vocab_size
                || lm_head_info.shape[1] != arch.hidden_size
            {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "lm_head weight shape mismatch",
                    },
                    model_ctx("prepare"),
                ));
            }
        }

        let half_bytes = std::mem::size_of::<f16>();
        let i32_bytes = std::mem::size_of::<i32>();
        let embed_bytes = embed_info.nbytes;
        let final_norm_bytes = final_norm_info.nbytes;
        let lm_head_bytes = if tie_embeddings { 0 } else { lm_head_info.nbytes };
        let residual_bytes = arch
            .hidden_size
            .checked_mul(half_bytes)
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "residual buffer size overflow",
                    },
                    model_ctx("prepare"),
                )
            })?;
        let logits_bytes = arch
            .vocab_size
            .checked_mul(half_bytes)
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "logits buffer size overflow",
                    },
                    model_ctx("prepare"),
                )
            })?;
        let normed_hidden_bytes = residual_bytes;
        let sampled_bytes = i32_bytes;
        let token_ids_bytes = 4;

        let mut arena_bytes = embed_bytes
            .checked_add(final_norm_bytes)
            .and_then(|v| v.checked_add(lm_head_bytes))
            .and_then(|v| v.checked_add(residual_bytes))
            .and_then(|v| v.checked_add(logits_bytes))
            .and_then(|v| v.checked_add(normed_hidden_bytes))
            .and_then(|v| v.checked_add(sampled_bytes))
            .and_then(|v| v.checked_add(token_ids_bytes))
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "model arena byte overflow",
                    },
                    model_ctx("prepare"),
                )
            })?;
        arena_bytes = max(arena_bytes, METAL_ARENA_BYTES);

        let mut ctx = MetalContext::new()?;
        ctx.compile_library(kernels::KERNEL_SOURCE)?;
        let mut pipelines = PipelineCache::new();
        pipelines.compile_all(&ctx)?;
        let mut arena = MetalBufferArena::new(ctx.device(), arena_bytes)?;

        let mut names = vec![embed_name.as_str(), final_norm_name.as_str()];
        if lm_head_name != embed_name {
            names.push(lm_head_name.as_str());
        }
        let mut mapped_refs = map_safetensor_to_arena(
            &mut arena,
            &self.model_dir,
            &names.iter().map(|name| *name).collect::<Vec<_>>(),
        )?;
        let embedding = Self::region_lookup(&mut mapped_refs, &embed_name)?;
        let final_norm = Self::region_lookup(&mut mapped_refs, &final_norm_name)?;
        let lm_head = if tie_embeddings {
            embedding
        } else {
            Self::region_lookup(&mut mapped_refs, &lm_head_name)?
        };
        if !mapped_refs.is_empty() {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "unexpected mapped tensor entries",
                },
                model_ctx("prepare"),
            ));
        }

        let residual = arena.region(
            "metal_model_residual",
            residual_bytes,
            16,
        )?;
        let logits = arena.region("metal_model_logits", logits_bytes, 16)?;
        let normed_hidden = arena.region("metal_model_normed_hidden", normed_hidden_bytes, 16)?;
        let sampled = arena.region("metal_model_sampled", sampled_bytes, 4)?;
        let token_ids = arena.region("metal_model_token_ids", token_ids_bytes, 4)?;

        let state = Gemma4MetalState {
            hidden_size: arch.hidden_size,
            vocab_size: arch.vocab_size,
            num_layers: arch.num_hidden_layers,
            rms_norm_eps: arch.rms_norm_eps,
            final_logit_softcap: arch.final_logit_softcapping.unwrap_or(METAL_SOFTCAP),
            embedding_scale: (arch.hidden_size as f32).sqrt(),
            embedding,
            final_norm,
            lm_head,
            residual,
            logits,
            normed_hidden,
            sampled,
            token_ids,
        };

        self.ctx = Some(ctx);
        self.pipelines = Some(pipelines);
        self.arena = Some(arena);
        Ok(state)
    }

    fn enqueue_embedding_gather(
        &self,
        state: &Gemma4MetalState,
        num_tokens: usize,
    ) -> Result<()> {
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
        encoder.setComputePipelineState(pso);
        let buf = arena.buffer_retained();
        encoder.setBuffer_offset_atIndex(Some(buf), state.embedding.offset, 0);
        encoder.setBuffer_offset_atIndex(Some(buf), state.token_ids.offset, 1);
        encoder.setBuffer_offset_atIndex(Some(buf), state.residual.offset, 2);
        let hidden = u32::try_from(state.hidden_size).map_err(|_| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "hidden_size does not fit in u32",
                },
                model_ctx("embedding_gather"),
            )
        })?;
        encoder.setBytes_length_atIndex(
            ptr::NonNull::new_unchecked(&hidden as *const _ as *mut _),
            4,
            3,
        );
        let scale = state.embedding_scale;
        encoder.setBytes_length_atIndex(
            ptr::NonNull::new_unchecked(&scale as *const _ as *mut _),
            4,
            4,
        );
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
        Ok(())
    }

    fn run_decode_step(&mut self, handoff: &HandoffCapsule) -> Result<()> {
        if handoff.num_sequences() > 1 {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_batch_size",
                },
                model_ctx("launch_rollout"),
            ));
        }
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
        if state.num_layers > 0 {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_num_layers",
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

        self.enqueue_embedding_gather(state, num_tokens)?;

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

        let sampled_ptr = arena.host_ptr(&state.sampled) as *const i32;
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
        Ok(())
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl AppleBackend for ModelMetalBackend {
    fn prepare(&mut self, plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
        plan.validate()?;
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
        let state = self.initialize_model_resources()?;
        self.state = Some(state);
        self.prepared = true;
        Ok(())
    }

    fn launch_prefill(&mut self, _handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op: "unsupported_mode",
            },
            model_ctx("launch_prefill"),
        ))
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

    fn next_ticket(&mut self, kind: AppleLaunchKind, bucket: Option<RolloutBucket>) -> AppleLaunchTicket {
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
        if self.ctx.is_some() && self.pipelines.is_some() && self.arena.is_some() && self.state.is_some()
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

        let residual = arena.region("metal_decode_residual", max_tokens * hidden * half_bytes, 16)?;
        let final_norm = arena.region("metal_decode_final_norm", hidden * half_bytes, 16)?;
        let lm_head = arena.region("metal_decode_lm_head", vocab * hidden * half_bytes, 16)?;
        let logits = arena.region("metal_decode_logits", max_tokens * vocab * half_bytes, 16)?;
        let normed_hidden = arena.region("metal_decode_normed_hidden", max_tokens * hidden * half_bytes, 16)?;
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
        let ctx_ref = self
            .ctx
            .as_ref()
            .ok_or_else(|| {
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
            Some(expected) if expected == ticket.step_id => Ok(self.pending.take().unwrap_or_default()),
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
