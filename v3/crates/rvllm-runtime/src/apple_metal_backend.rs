use rvllm_core::{AppleCtx, AppleError, Result, RvllmError, TokenId};
use rvllm_apple::{AppleBackend, AppleLaunchKind, AppleLaunchTicket, HandoffCapsule, StepToken};
use half::f16;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::path::PathBuf;

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
use rvllm_apple_metal::{
    kernels,
    context::MetalContext,
    layer_forward::metal_finalize_logits_blocking,
    gemma4_model::Gemma4MetalState,
    pipeline::PipelineCache,
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
                AppleCtx {
                    backend: "model-metal-backend",
                    op,
                    device: "apple-silicon",
                },
            ))
        }
    }
}

#[cfg(all(feature = "apple", target_os = "macos"))]
impl AppleBackend for ModelMetalBackend {
    fn prepare(&mut self, _plan: &rvllm_apple::AppleRuntimePlan) -> Result<()> {
        self.prepared = true;
        self.next_step_id = 0;
        self.last_ticket = None;
        self.pending = None;
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op: "model_metal_backend_not_implemented",
            },
            AppleCtx {
                backend: "model-metal-backend",
                op: "model_metal_backend_not_implemented",
                device: "apple-silicon",
            },
        ))
    }

    fn launch_prefill(&mut self, _handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op: "model_metal_backend_not_implemented",
            },
            AppleCtx {
                backend: "model-metal-backend",
                op: "launch_prefill",
                device: "apple-silicon",
            },
        ))
    }

    fn launch_rollout(
        &mut self,
        _handoff: &HandoffCapsule,
        bucket: Option<RolloutBucket>,
    ) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_rollout")?;
        Err(RvllmError::apple(
            AppleError::FeatureNotAvailable {
                backend: "model-metal-backend",
                op: "model_metal_backend_not_implemented",
            },
            AppleCtx {
                backend: "model-metal-backend",
                op: "launch_rollout",
                device: "apple-silicon",
            },
        ))
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
                AppleCtx {
                    backend: "model-metal-backend",
                    op: "collect",
                    device: "apple-silicon",
                },
            )),
            None => Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                AppleCtx {
                    backend: "model-metal-backend",
                    op: "collect",
                    device: "apple-silicon",
                },
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
