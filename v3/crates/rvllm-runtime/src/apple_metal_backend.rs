use half::f16;
use rvllm_apple::{AppleBackend, AppleLaunchKind, AppleLaunchTicket, HandoffCapsule, StepToken};
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError, TokenId};
#[cfg(all(feature = "apple", target_os = "macos"))]
use rvllm_loader::load::ModelArch;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::cmp::max;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::path::PathBuf;
#[cfg(all(feature = "apple", target_os = "macos"))]
use std::ptr;

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
    gemma4_model::{Gemma4MetalState, MetalOneLayerState},
    kernels,
    layer_forward::{
        metal_finalize_logits_blocking, metal_forward_layer, MetalLayerDims, MetalLayerWeights,
        MetalMetadata, MetalPhase, MetalScratch,
    },
    pipeline::PipelineCache,
    weight_loader::{map_safetensor_to_arena, scan_safetensor_tensors},
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
        tensors: &std::collections::BTreeMap<
            String,
            rvllm_apple_metal::weight_loader::SafetensorTensorInfo,
        >,
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

    fn region_lookup(refs: &mut Vec<(String, MetalRegion)>, name: &str) -> Result<MetalRegion> {
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

    fn map_fused_bytes_to_arena(
        arena: &mut MetalBufferArena,
        name: &str,
        bytes: &[u8],
    ) -> Result<MetalRegion> {
        let region = arena.region(name, bytes.len(), 16)?;
        unsafe {
            arena.write_region(&region, bytes)?;
        }
        Ok(region)
    }

    fn concat_f16_tensors(
        model_dir: &std::path::Path,
        tensors: &std::collections::BTreeMap<
            String,
            rvllm_apple_metal::weight_loader::SafetensorTensorInfo,
        >,
        names: &[String],
        expected_shapes: &[Vec<usize>],
        op: &'static str,
    ) -> Result<Vec<u8>> {
        if names.len() != expected_shapes.len() {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "concat shape metadata mismatch",
                },
                model_ctx(op),
            ));
        }

        let mut out = Vec::new();
        for (name, expected_shape) in names.iter().zip(expected_shapes.iter()) {
            let info = tensors.get(name).ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "missing tensor for concat",
                    },
                    model_ctx(op),
                )
            })?;
            if info.shape != *expected_shape {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "tensor shape mismatch for concat",
                    },
                    model_ctx(op),
                ));
            }
            let bytes = rvllm_apple_metal::weight_loader::load_safetensor_f16(model_dir, name)?;
            out.extend_from_slice(&bytes);
        }
        Ok(out)
    }

    fn write_i32_region(
        arena: &MetalBufferArena,
        region: &MetalRegion,
        values: &[i32],
    ) -> Result<()> {
        unsafe {
            let dst = arena.host_ptr(region) as *mut i32;
            ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
        }
        Ok(())
    }

    fn write_f32_region(
        arena: &MetalBufferArena,
        region: &MetalRegion,
        values: &[f32],
    ) -> Result<()> {
        unsafe {
            let dst = arena.host_ptr(region) as *mut f32;
            ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
        }
        Ok(())
    }

    fn initialize_model_resources(&mut self) -> Result<Gemma4MetalState> {
        let arch = ModelArch::from_dir(&self.model_dir)?;
        if arch.num_hidden_layers > 1 {
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
        let tie_embeddings = !tensors.contains_key("lm_head.weight")
            && !tensors.contains_key(&prefixed_lm_head_name);
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

        let mut names = vec![embed_name.clone(), final_norm_name.clone()];
        if lm_head_name != embed_name {
            names.push(lm_head_name.clone());
        }

        let mut layer_weight_bytes = 0;
        let mut fused_qkv_bytes = 0;
        let mut fused_gate_up_bytes = 0;

        if arch.num_hidden_layers == 1 {
            let hidden = arch.hidden_size;
            let intermediate = arch.intermediate_size;
            let num_heads = arch.num_attention_heads;
            let num_kv_heads = arch.num_key_value_heads;
            let head_dim = arch.head_dim;
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_rows = q_dim + 2 * kv_dim;

            if num_heads != 1 || num_kv_heads != 1 || head_dim != hidden {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason:
                            "one-layer probe requires one head, one kv head, head_dim == hidden",
                    },
                    model_ctx("prepare"),
                ));
            }
            if intermediate == 0 {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "one-layer probe requires nonzero intermediate",
                    },
                    model_ctx("prepare"),
                ));
            }

            let lprefix = format!("{wprefix}.layers.0");
            let attn_norm_name = format!("{lprefix}.input_layernorm.weight");
            let o_proj_name = format!("{lprefix}.self_attn.o_proj.weight");
            let mlp_norm_name = format!("{lprefix}.mlp_norm.weight");
            let down_proj_name = format!("{lprefix}.mlp.down_proj.weight");

            let prefused_qkv_name = format!("{lprefix}.self_attn.qkv.weight");
            let q_name = format!("{lprefix}.self_attn.q_proj.weight");
            let k_name = format!("{lprefix}.self_attn.k_proj.weight");
            let v_name = format!("{lprefix}.self_attn.v_proj.weight");
            let use_prefused_qkv = tensors.contains_key(&prefused_qkv_name);

            let prefused_gate_up_name = format!("{lprefix}.mlp.gate_up.weight");
            let gate_name = format!("{lprefix}.mlp.gate_proj.weight");
            let up_name = format!("{lprefix}.mlp.up_proj.weight");
            let use_prefused_gate_up = tensors.contains_key(&prefused_gate_up_name);

            let mut add_tensor_size = |name: &str| -> Result<()> {
                let info = tensors.get(name).ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "missing layer weight",
                        },
                        model_ctx("prepare"),
                    )
                })?;
                layer_weight_bytes += info.nbytes;
                Ok(())
            };

            add_tensor_size(&attn_norm_name)?;
            add_tensor_size(&o_proj_name)?;
            add_tensor_size(&mlp_norm_name)?;
            add_tensor_size(&down_proj_name)?;

            names.push(attn_norm_name);
            names.push(o_proj_name);
            names.push(mlp_norm_name);
            names.push(down_proj_name);

            if use_prefused_qkv {
                add_tensor_size(&prefused_qkv_name)?;
                names.push(prefused_qkv_name);
            } else {
                add_tensor_size(&q_name)?;
                add_tensor_size(&k_name)?;
                add_tensor_size(&v_name)?;
                fused_qkv_bytes = qkv_rows * hidden * std::mem::size_of::<f16>();
            }

            if use_prefused_gate_up {
                add_tensor_size(&prefused_gate_up_name)?;
                names.push(prefused_gate_up_name);
            } else {
                add_tensor_size(&gate_name)?;
                add_tensor_size(&up_name)?;
                fused_gate_up_bytes = 2 * intermediate * hidden * std::mem::size_of::<f16>();
            }
        }

        let half_bytes = std::mem::size_of::<f16>();
        let i32_bytes = std::mem::size_of::<i32>();
        let f32_bytes = std::mem::size_of::<f32>();
        let embed_bytes = embed_info.nbytes;
        let final_norm_bytes = final_norm_info.nbytes;
        let lm_head_bytes = if tie_embeddings {
            0
        } else {
            lm_head_info.nbytes
        };
        let residual_bytes = arch.hidden_size.checked_mul(half_bytes).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "residual buffer size overflow",
                },
                model_ctx("prepare"),
            )
        })?;
        let logits_bytes = arch.vocab_size.checked_mul(half_bytes).ok_or_else(|| {
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

        let mut scratch_bytes = 0;
        if arch.num_hidden_layers == 1 {
            let hidden = arch.hidden_size;
            let intermediate = arch.intermediate_size;
            let num_heads = arch.num_attention_heads;
            let num_kv_heads = arch.num_key_value_heads;
            let head_dim = arch.head_dim;
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_rows = q_dim + 2 * kv_dim;

            let qkv_out_bytes = qkv_rows * half_bytes;
            let q_bytes = q_dim * half_bytes;
            let k_bytes = kv_dim * half_bytes;
            let v_bytes = kv_dim * half_bytes;
            let attn_out_bytes = q_dim * half_bytes;
            let gate_up_out_bytes = 2 * intermediate * half_bytes;
            let activated_bytes = intermediate * half_bytes;
            let mlp_out_bytes = hidden * half_bytes;

            let block_size = 1usize;
            let num_blocks_total = 1usize;
            let kv_cache_bytes = num_blocks_total * block_size * kv_dim * half_bytes * 2;

            let half_rope = head_dim / 2;
            let max_pos = 16usize;
            let rope_table_bytes = max_pos * half_rope * f32_bytes;

            scratch_bytes = qkv_out_bytes
                + q_bytes
                + k_bytes
                + v_bytes
                + attn_out_bytes
                + gate_up_out_bytes
                + activated_bytes
                + mlp_out_bytes
                + kv_cache_bytes
                + rope_table_bytes * 2 // cos + sin
                + 64; // metadata
        }

        let mut arena_bytes = embed_bytes
            .checked_add(final_norm_bytes)
            .and_then(|v| v.checked_add(lm_head_bytes))
            .and_then(|v| v.checked_add(layer_weight_bytes))
            .and_then(|v| v.checked_add(fused_qkv_bytes))
            .and_then(|v| v.checked_add(fused_gate_up_bytes))
            .and_then(|v| v.checked_add(residual_bytes))
            .and_then(|v| v.checked_add(logits_bytes))
            .and_then(|v| v.checked_add(normed_hidden_bytes))
            .and_then(|v| v.checked_add(sampled_bytes))
            .and_then(|v| v.checked_add(token_ids_bytes))
            .and_then(|v| v.checked_add(scratch_bytes))
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "model arena byte overflow",
                    },
                    model_ctx("prepare"),
                )
            })?;
        arena_bytes = arena_bytes.checked_add(64 * 1024).ok_or_else(|| {
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

        let mut mapped_refs = map_safetensor_to_arena(
            &mut arena,
            &self.model_dir,
            &names.iter().map(|name| name.as_str()).collect::<Vec<_>>(),
        )?;
        let embedding = Self::region_lookup(&mut mapped_refs, &embed_name)?;
        let final_norm = Self::region_lookup(&mut mapped_refs, &final_norm_name)?;
        let lm_head = if tie_embeddings {
            embedding.clone()
        } else {
            Self::region_lookup(&mut mapped_refs, &lm_head_name)?
        };

        let mut one_layer = None;
        if arch.num_hidden_layers == 1 {
            let hidden = arch.hidden_size;
            let intermediate = arch.intermediate_size;
            let num_heads = arch.num_attention_heads;
            let num_kv_heads = arch.num_key_value_heads;
            let head_dim = arch.head_dim;
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_rows = q_dim + 2 * kv_dim;

            let lprefix = format!("{wprefix}.layers.0");
            let attn_norm = Self::region_lookup(
                &mut mapped_refs,
                &format!("{lprefix}.input_layernorm.weight"),
            )?;
            let o_proj = Self::region_lookup(
                &mut mapped_refs,
                &format!("{lprefix}.self_attn.o_proj.weight"),
            )?;
            let mlp_norm =
                Self::region_lookup(&mut mapped_refs, &format!("{lprefix}.mlp_norm.weight"))?;
            let down_proj =
                Self::region_lookup(&mut mapped_refs, &format!("{lprefix}.mlp.down_proj.weight"))?;

            let prefused_qkv_name = format!("{lprefix}.self_attn.qkv.weight");
            let qkv = if tensors.contains_key(&prefused_qkv_name) {
                Self::region_lookup(&mut mapped_refs, &prefused_qkv_name)?
            } else {
                let bytes = Self::concat_f16_tensors(
                    &self.model_dir,
                    &tensors,
                    &[
                        format!("{lprefix}.self_attn.q_proj.weight"),
                        format!("{lprefix}.self_attn.k_proj.weight"),
                        format!("{lprefix}.self_attn.v_proj.weight"),
                    ],
                    &[
                        vec![q_dim, hidden],
                        vec![kv_dim, hidden],
                        vec![kv_dim, hidden],
                    ],
                    "fuse_qkv",
                )?;
                Self::map_fused_bytes_to_arena(&mut arena, "metal_fused_qkv", &bytes)?
            };

            let prefused_gate_up_name = format!("{lprefix}.mlp.gate_up.weight");
            let gate_up = if tensors.contains_key(&prefused_gate_up_name) {
                Self::region_lookup(&mut mapped_refs, &prefused_gate_up_name)?
            } else {
                let bytes = Self::concat_f16_tensors(
                    &self.model_dir,
                    &tensors,
                    &[
                        format!("{lprefix}.mlp.gate_proj.weight"),
                        format!("{lprefix}.mlp.up_proj.weight"),
                    ],
                    &[vec![intermediate, hidden], vec![intermediate, hidden]],
                    "fuse_gate_up",
                )?;
                Self::map_fused_bytes_to_arena(&mut arena, "metal_fused_gate_up", &bytes)?
            };

            let qkv_out = arena.region("metal_qkv_out", qkv_rows * half_bytes, 16)?;
            let q = arena.region("metal_q", q_dim * half_bytes, 16)?;
            let k = arena.region("metal_k", kv_dim * half_bytes, 16)?;
            let v = arena.region("metal_v", kv_dim * half_bytes, 16)?;
            let attn_out = arena.region("metal_attn_out", q_dim * half_bytes, 16)?;
            let gate_up_out =
                arena.region("metal_gate_up_out", 2 * intermediate * half_bytes, 16)?;
            let activated = arena.region("metal_activated", intermediate * half_bytes, 16)?;
            let mlp_out = arena.region("metal_mlp_out", hidden * half_bytes, 16)?;

            let block_size = 1u32;
            let num_blocks_total = 1u32;
            let max_blocks_per_seq = 1u32;
            let kv_cache_k = arena.region(
                "metal_kv_cache_k",
                (num_blocks_total as usize) * (block_size as usize) * kv_dim * half_bytes,
                16,
            )?;
            let kv_cache_v = arena.region(
                "metal_kv_cache_v",
                (num_blocks_total as usize) * (block_size as usize) * kv_dim * half_bytes,
                16,
            )?;

            let positions = arena.region("metal_meta_positions", 4, 4)?;
            let slot_mapping = arena.region("metal_meta_slot_mapping", 4, 4)?;
            let context_lens = arena.region("metal_meta_context_lens", 4, 4)?;
            let block_tables = arena.region("metal_meta_block_tables", 4, 4)?;

            let half_rope = head_dim / 2;
            let max_pos = 16usize;
            let cos = arena.region("metal_rope_cos", max_pos * half_rope * f32_bytes, 16)?;
            let sin = arena.region("metal_rope_sin", max_pos * half_rope * f32_bytes, 16)?;

            Self::write_i32_region(&arena, &positions, &[0])?;
            Self::write_i32_region(&arena, &slot_mapping, &[0])?;
            Self::write_i32_region(&arena, &context_lens, &[1])?;
            Self::write_i32_region(&arena, &block_tables, &[0])?;
            Self::write_f32_region(&arena, &cos, &vec![1.0; max_pos * half_rope])?;
            Self::write_f32_region(&arena, &sin, &vec![0.0; max_pos * half_rope])?;

            one_layer = Some(MetalOneLayerState {
                layer_idx: 0,
                attn_norm,
                qkv,
                o_proj,
                mlp_norm,
                gate_up,
                down_proj,
                qkv_out,
                q,
                k,
                v,
                attn_out,
                gate_up_out,
                activated,
                mlp_out,
                positions,
                slot_mapping,
                cos,
                sin,
                block_tables,
                context_lens,
                kv_cache_k,
                kv_cache_v,
                block_size,
                max_blocks_per_seq,
                num_blocks_total,
            });
        }

        if !mapped_refs.is_empty() {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "unexpected mapped tensor entries",
                },
                model_ctx("prepare"),
            ));
        }

        let residual = arena.region("metal_model_residual", residual_bytes, 16)?;
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
            one_layer,
        };

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
        cmd_buf.waitUntilCompleted();
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
        if state.num_layers > 1 {
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

        if state.num_layers == 1 {
            let one = state.one_layer.as_ref().ok_or_else(|| {
                RvllmError::apple(
                    AppleError::NotPrepared {
                        backend: "model-metal-backend",
                    },
                    model_ctx("launch_rollout"),
                )
            })?;

            let hidden = state.hidden_size;
            let half_bytes = std::mem::size_of::<f16>();
            let intermediate = one.gate_up.size / 2 / half_bytes / hidden;

            let dims = MetalLayerDims {
                num_tokens: 1,
                hidden: state.hidden_size as u32,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: state.hidden_size as u32,
                intermediate: intermediate as u32,
                block_size: one.block_size,
                max_blocks_per_seq: one.max_blocks_per_seq,
                num_blocks_total: one.num_blocks_total,
                attn_scale: 1.0 / (state.hidden_size as f32).sqrt(),
                rms_eps: state.rms_norm_eps,
                rope_dim: state.hidden_size as u32,
                softcap: state.final_logit_softcap,
            };

            let weights = MetalLayerWeights {
                attn_norm_offset: one.attn_norm.offset,
                qkv_offset: one.qkv.offset,
                qkv_bias_offset: None,
                o_proj_offset: one.o_proj.offset,
                mlp_norm_offset: one.mlp_norm.offset,
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
                cu_seqlens_offset: None,
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
                    MetalPhase::Decode,
                    one.kv_cache_k.offset,
                    one.kv_cache_v.offset,
                )?;
            }
        }

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
        let state = self.initialize_model_resources()?;
        self.state = Some(state);
        self.prepared = true;
        Ok(())
    }

    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        handoff.validate()?;

        let state = self.state.as_ref().ok_or_else(|| {
            RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "model-metal-backend",
                },
                model_ctx("launch_prefill"),
            )
        })?;

        if state.num_layers > 1 {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "prefill_requires_transformer_layers",
                },
                model_ctx("launch_prefill"),
            ));
        }

        if handoff.num_sequences() != 1 {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_batch_size",
                },
                model_ctx("launch_prefill"),
            ));
        }

        if handoff.tokens_flat.len() != 1 {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_prefill_length",
                },
                model_ctx("launch_prefill"),
            ));
        }

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

    fn temp_fixture_dir() -> std::path::PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "rvllm-metal-zero-layer-test-{}-{}",
            std::process::id(),
            now
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

    fn cpu_full_nonzero_matvec(weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
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

    fn cpu_reference_one_layer_full_nonzero_argmax() -> usize {
        let hidden = 128;
        let intermediate = 256;
        let vocab = 8;
        let eps = 0.000001f32;

        let mut embedding = vec![0.0f32; vocab * hidden];
        embedding[2 * hidden + 7] = 10.0;
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
        q_proj[7] = 0.25;
        k_proj[7] = 0.125;
        v_proj[11 * hidden + 7] = 2.0;
        o_proj[9 * hidden + 11] = 6.0;

        let q = cpu_full_nonzero_matvec(&q_proj, hidden, hidden, &attn_normed);
        let k = cpu_full_nonzero_matvec(&k_proj, hidden, hidden, &attn_normed);
        let v = cpu_full_nonzero_matvec(&v_proj, hidden, hidden, &attn_normed);
        let _score = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (hidden as f32).sqrt();
        let attn_out = v;
        let projected_attn = cpu_full_nonzero_matvec(&o_proj, hidden, hidden, &attn_out);
        for (dst, src) in residual.iter_mut().zip(projected_attn.iter()) {
            *dst += src;
        }

        let mlp_normed = cpu_full_nonzero_rms_norm(&residual, &norm, eps);
        let mut gate_proj = vec![0.0f32; intermediate * hidden];
        let mut up_proj = vec![0.0f32; intermediate * hidden];
        let mut down_proj = vec![0.0f32; hidden * intermediate];
        gate_proj[7] = 0.5;
        up_proj[7] = 0.5;
        down_proj[9 * intermediate] = 4.0;

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
        lm_head[2 * hidden + 7] = 1.0;
        lm_head[3 * hidden + 9] = 4.0;
        let logits = cpu_full_nonzero_matvec(&lm_head, vocab, hidden, &final_hidden);
        cpu_full_nonzero_argmax(&logits)
    }

    #[cfg(all(feature = "apple", target_os = "macos"))]
    fn write_tiny_one_layer_full_nonzero_fixture() -> std::path::PathBuf {
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
