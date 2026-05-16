#[cfg(target_os = "macos")]
use std::{cmp::max, collections::BTreeMap, path::Path, ptr};

#[cfg(target_os = "macos")]
use half::f16;
#[cfg(target_os = "macos")]
use rvllm_core::{AppleCtx, AppleError, LoaderCtx, LoaderError, Result, RvllmError};
#[cfg(target_os = "macos")]
use rvllm_loader::load::{LayerAttnType, ModelArch};

#[cfg(target_os = "macos")]
use crate::{
    arena::{MetalBufferArena, MetalRegion},
    context::MetalContext,
    weight_loader::{
        load_safetensor_f16, map_safetensor_to_arena, scan_safetensor_tensors, SafetensorTensorInfo,
    },
};

#[cfg(target_os = "macos")]
const PROBE_METAL_ARENA_BYTES: usize = 1024 * 1024;
#[cfg(target_os = "macos")]
const PROBE_METAL_SOFTCAP: f32 = 0.0;
#[cfg(target_os = "macos")]
const PROBE_METAL_MAX_SYNTHETIC_LAYERS: usize = 8;
#[cfg(target_os = "macos")]
const PROBE_METAL_MAX_PROMPT_TOKENS: usize = 8;

#[cfg(target_os = "macos")]
fn probe_ctx(op: &'static str) -> AppleCtx {
    AppleCtx {
        backend: "model-metal-backend",
        op,
        device: "apple-silicon",
    }
}

#[derive(Debug, Clone)]
#[cfg(target_os = "macos")]
pub struct Gemma4MetalState {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcap: f32,
    pub embedding_scale: f32,
    pub max_probe_tokens: usize,
    pub embedding: MetalRegion,
    pub final_norm: MetalRegion,
    pub lm_head: MetalRegion,
    pub residual: MetalRegion,
    pub logits: MetalRegion,
    pub normed_hidden: MetalRegion,
    pub sampled: MetalRegion,
    pub token_ids: MetalRegion,

    pub layers: Vec<MetalOneLayerState>,
}

#[derive(Debug, Clone)]
#[cfg(target_os = "macos")]
pub struct MetalOneLayerState {
    pub layer_idx: usize,
    pub dims: MetalProbeLayerDims,

    pub attn_norm: MetalRegion,
    pub qkv: MetalRegion,
    pub q_norm: Option<MetalRegion>,
    pub k_norm: Option<MetalRegion>,
    pub v_norm: Option<MetalRegion>,
    pub o_proj: MetalRegion,
    pub mlp_norm: MetalRegion,
    pub post_attn_norm: Option<MetalRegion>,
    pub pre_ff_norm: Option<MetalRegion>,
    pub post_ff_norm: Option<MetalRegion>,
    pub layer_scalar: Option<MetalRegion>,
    pub layer_scalar_dim: u32,
    pub gate_up: MetalRegion,
    pub down_proj: MetalRegion,

    pub qkv_out: MetalRegion,
    pub q: MetalRegion,
    pub k: MetalRegion,
    pub v: MetalRegion,
    pub attn_out: MetalRegion,
    pub gate_up_out: MetalRegion,
    pub activated: MetalRegion,
    pub mlp_out: MetalRegion,

    pub positions: MetalRegion,
    pub slot_mapping: MetalRegion,
    pub cos: MetalRegion,
    pub sin: MetalRegion,
    pub block_tables: MetalRegion,
    pub context_lens: MetalRegion,
    pub cu_seqlens: MetalRegion,

    pub kv_cache_k: MetalRegion,
    pub kv_cache_v: MetalRegion,

    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(target_os = "macos")]
pub enum MetalProbeLayerAttentionKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(target_os = "macos")]
pub struct MetalProbeLayerDims {
    pub attention_kind: MetalProbeLayerAttentionKind,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub qkv_rows: usize,
    pub attn_scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg(target_os = "macos")]
pub struct Gemma4DryRunValidation {
    pub weight_prefix: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub embed_tokens: String,
    pub final_norm: String,
    pub lm_head: Option<String>,
    pub final_logit_softcap: Option<f32>,
    pub layers: Vec<Gemma4DryRunLayerValidation>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg(target_os = "macos")]
pub struct Gemma4DryRunLayerValidation {
    pub layer_idx: usize,
    pub attention_kind: MetalProbeLayerAttentionKind,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: Option<String>,
    pub v_uses_k_proj: bool,
    pub input_layernorm: String,
    pub post_attention_layernorm: String,
    pub pre_feedforward_layernorm: String,
    pub post_feedforward_layernorm: String,
    pub q_norm: String,
    pub k_norm: String,
    pub layer_scalar: Option<String>,
    pub layer_scalar_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
}

#[cfg(target_os = "macos")]
impl Gemma4DryRunValidation {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        validate_gemma4_model_dir_metadata(model_dir)
    }
}

#[cfg(target_os = "macos")]
impl MetalProbeLayerDims {
    fn from_arch_layer(arch: &ModelArch, layer_idx: usize) -> Result<Self> {
        let layer_type = arch
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerAttnType::Full);
        let (attention_kind, head_dim, num_kv_heads) = match layer_type {
            LayerAttnType::SlidingAttention => (
                MetalProbeLayerAttentionKind::Sliding,
                arch.head_dim,
                arch.num_key_value_heads,
            ),
            LayerAttnType::Full => (
                MetalProbeLayerAttentionKind::Full,
                arch.global_head_dim.unwrap_or(arch.head_dim),
                arch.num_global_key_value_heads
                    .unwrap_or(arch.num_key_value_heads),
            ),
            LayerAttnType::Linear => {
                return Err(RvllmError::apple(
                    AppleError::FeatureNotAvailable {
                        backend: "model-metal-backend",
                        op: "unsupported_probe_linear_attention_layer",
                    },
                    probe_ctx("prepare"),
                ));
            }
        };
        let num_heads = arch.num_attention_heads;
        if num_heads != 1 || num_kv_heads != 1 || head_dim == 0 || head_dim % 2 != 0 {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "synthetic probe requires one query head, one kv head, and even nonzero head_dim",
                },
                probe_ctx("prepare"),
            ));
        }

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        Ok(Self {
            attention_kind,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim: head_dim,
            q_dim,
            kv_dim,
            qkv_rows: q_dim + 2 * kv_dim,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }
}

#[cfg(target_os = "macos")]
struct ProbeModelPlan {
    arch: ModelArch,
    tensors: BTreeMap<String, SafetensorTensorInfo>,
    weight_prefix: String,
    embed_name: String,
    final_norm_name: String,
    lm_head_name: String,
    tie_embeddings: bool,
    layer_names: Vec<ProbeLayerNames>,
    names: Vec<String>,
    arena_bytes: usize,
}

#[cfg(target_os = "macos")]
struct ProbeLayerNames {
    dims: MetalProbeLayerDims,
    attn_norm_name: String,
    o_proj_name: String,
    mlp_norm_name: String,
    down_proj_name: String,
    prefused_qkv_name: String,
    q_name: String,
    k_name: String,
    v_name: String,
    q_norm_name: Option<String>,
    k_norm_name: Option<String>,
    v_norm_name: Option<String>,
    post_attn_norm_name: Option<String>,
    pre_ff_norm_name: Option<String>,
    post_ff_norm_name: Option<String>,
    layer_scalar_name: Option<String>,
    layer_scalar_dim: usize,
    prefused_gate_up_name: String,
    gate_name: String,
    up_name: String,
}

#[cfg(target_os = "macos")]
impl ProbeModelPlan {
    fn new(model_dir: &Path) -> Result<Self> {
        let arch = ModelArch::from_dir(model_dir)?;
        if arch.num_hidden_layers > PROBE_METAL_MAX_SYNTHETIC_LAYERS {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "unsupported_synthetic_probe_num_layers",
                },
                probe_ctx("prepare"),
            ));
        }
        if arch.hidden_size == 0 || arch.vocab_size == 0 {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "model has zero hidden_size or vocab_size",
                },
                probe_ctx("prepare"),
            ));
        }

        let tensors = scan_safetensor_tensors(model_dir)?;
        let weight_prefix = resolve_weight_prefix(&tensors);
        let embed_name = format!("{weight_prefix}.embed_tokens.weight");
        let final_norm_name = format!("{weight_prefix}.norm.weight");
        let prefixed_lm_head_name = format!("{weight_prefix}.lm_head.weight");
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
                probe_ctx("prepare"),
            ));
        };

        let embed_info = tensors.get(&embed_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing embed_tokens.weight",
                },
                probe_ctx("prepare"),
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
                probe_ctx("prepare"),
            ));
        }

        let final_norm_info = tensors.get(&final_norm_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing final layer norm weight",
                },
                probe_ctx("prepare"),
            )
        })?;
        if final_norm_info.shape.len() != 1 || final_norm_info.shape[0] != arch.hidden_size {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "final layer norm shape mismatch",
                },
                probe_ctx("prepare"),
            ));
        }

        let lm_head_info = tensors.get(&lm_head_name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing lm_head weight",
                },
                probe_ctx("prepare"),
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
                    probe_ctx("prepare"),
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
        let mut layer_names = Vec::new();

        if arch.num_hidden_layers > 0 {
            let hidden = arch.hidden_size;
            let intermediate = arch.intermediate_size;

            if intermediate == 0 {
                return Err(RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "one-layer probe requires nonzero intermediate",
                    },
                    probe_ctx("prepare"),
                ));
            }

            let mut add_tensor_size = |name: &str| -> Result<()> {
                let info = tensors.get(name).ok_or_else(|| {
                    RvllmError::apple(
                        AppleError::InvalidWeightBlob {
                            reason: "missing layer weight",
                        },
                        probe_ctx("prepare"),
                    )
                })?;
                layer_weight_bytes += info.nbytes;
                Ok(())
            };

            for layer_idx in 0..arch.num_hidden_layers {
                let dims = MetalProbeLayerDims::from_arch_layer(&arch, layer_idx)?;
                let q_dim = dims.q_dim;
                let kv_dim = dims.kv_dim;
                let qkv_rows = dims.qkv_rows;
                let lprefix = format!("{weight_prefix}.layers.{layer_idx}");
                let attn_norm_name = resolve_tensor_alias(
                    &tensors,
                    vec![
                        format!("{lprefix}.input_layernorm.weight"),
                        format!("{lprefix}.pre_attention_layernorm.weight"),
                    ],
                    "missing attention norm weight",
                )?;
                let o_proj_name = format!("{lprefix}.self_attn.o_proj.weight");
                let mlp_norm_name = resolve_tensor_alias(
                    &tensors,
                    vec![
                        format!("{lprefix}.mlp_norm.weight"),
                        format!("{lprefix}.pre_feedforward_layernorm.weight"),
                        format!("{lprefix}.post_attention_layernorm.weight"),
                    ],
                    "missing mlp norm weight",
                )?;
                let down_proj_name = format!("{lprefix}.mlp.down_proj.weight");

                let prefused_qkv_name = format!("{lprefix}.self_attn.qkv.weight");
                let q_name = format!("{lprefix}.self_attn.q_proj.weight");
                let k_name = format!("{lprefix}.self_attn.k_proj.weight");
                let v_name = format!("{lprefix}.self_attn.v_proj.weight");
                let q_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.self_attn.q_norm.weight")],
                );
                let k_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.self_attn.k_norm.weight")],
                );
                let v_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.self_attn.v_norm.weight")],
                );
                let post_attn_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.post_attention_layernorm.weight")],
                );
                let pre_ff_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.pre_feedforward_layernorm.weight")],
                );
                let post_ff_norm_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![format!("{lprefix}.post_feedforward_layernorm.weight")],
                );
                let layer_scalar_name = resolve_optional_tensor_alias(
                    &tensors,
                    vec![
                        format!("{lprefix}.layer_scalar"),
                        format!("{lprefix}.layer_scalar.weight"),
                    ],
                );
                let use_prefused_qkv = tensors.contains_key(&prefused_qkv_name);

                let prefused_gate_up_name = format!("{lprefix}.mlp.gate_up.weight");
                let gate_name = format!("{lprefix}.mlp.gate_proj.weight");
                let up_name = format!("{lprefix}.mlp.up_proj.weight");
                let use_prefused_gate_up = tensors.contains_key(&prefused_gate_up_name);

                validate_tensor_shape(
                    &tensors,
                    &attn_norm_name,
                    &[hidden],
                    "attention norm weight shape mismatch",
                )?;
                validate_tensor_shape(
                    &tensors,
                    &o_proj_name,
                    &[hidden, q_dim],
                    "o_proj weight shape mismatch",
                )?;
                validate_tensor_shape(
                    &tensors,
                    &mlp_norm_name,
                    &[hidden],
                    "mlp norm weight shape mismatch",
                )?;
                validate_tensor_shape(
                    &tensors,
                    &down_proj_name,
                    &[hidden, intermediate],
                    "down_proj weight shape mismatch",
                )?;
                add_tensor_size(&attn_norm_name)?;
                add_tensor_size(&o_proj_name)?;
                add_tensor_size(&mlp_norm_name)?;
                add_tensor_size(&down_proj_name)?;

                names.push(attn_norm_name.clone());
                names.push(o_proj_name.clone());
                names.push(mlp_norm_name.clone());
                names.push(down_proj_name.clone());

                if use_prefused_qkv {
                    validate_tensor_shape(
                        &tensors,
                        &prefused_qkv_name,
                        &[qkv_rows, hidden],
                        "qkv weight shape mismatch",
                    )?;
                    add_tensor_size(&prefused_qkv_name)?;
                    names.push(prefused_qkv_name.clone());
                } else {
                    validate_tensor_shape(
                        &tensors,
                        &q_name,
                        &[q_dim, hidden],
                        "q_proj weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &k_name,
                        &[kv_dim, hidden],
                        "k_proj weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &v_name,
                        &[kv_dim, hidden],
                        "v_proj weight shape mismatch",
                    )?;
                    add_tensor_size(&q_name)?;
                    add_tensor_size(&k_name)?;
                    add_tensor_size(&v_name)?;
                    fused_qkv_bytes += qkv_rows * hidden * std::mem::size_of::<f16>();
                }

                validate_optional_norm_shape(
                    &tensors,
                    &q_norm_name,
                    q_dim,
                    "q_norm weight shape mismatch",
                )?;
                validate_optional_norm_shape(
                    &tensors,
                    &k_norm_name,
                    kv_dim,
                    "k_norm weight shape mismatch",
                )?;
                validate_optional_norm_shape(
                    &tensors,
                    &v_norm_name,
                    kv_dim,
                    "v_norm weight shape mismatch",
                )?;
                for norm_name in [&q_norm_name, &k_norm_name, &v_norm_name]
                    .into_iter()
                    .flatten()
                {
                    add_tensor_size(norm_name)?;
                    names.push(norm_name.clone());
                }
                validate_optional_norm_shape(
                    &tensors,
                    &post_attn_norm_name,
                    hidden,
                    "post_attention_layernorm weight shape mismatch",
                )?;
                validate_optional_norm_shape(
                    &tensors,
                    &pre_ff_norm_name,
                    hidden,
                    "pre_feedforward_layernorm weight shape mismatch",
                )?;
                validate_optional_norm_shape(
                    &tensors,
                    &post_ff_norm_name,
                    hidden,
                    "post_feedforward_layernorm weight shape mismatch",
                )?;
                for norm_name in [&post_attn_norm_name, &pre_ff_norm_name, &post_ff_norm_name]
                    .into_iter()
                    .flatten()
                {
                    if norm_name != &mlp_norm_name {
                        add_tensor_size(norm_name)?;
                        names.push(norm_name.clone());
                    }
                }
                let layer_scalar_dim =
                    validate_optional_layer_scalar_shape(&tensors, &layer_scalar_name, hidden)?;
                if let Some(layer_scalar_name) = &layer_scalar_name {
                    add_tensor_size(layer_scalar_name)?;
                    names.push(layer_scalar_name.clone());
                }

                if use_prefused_gate_up {
                    validate_tensor_shape(
                        &tensors,
                        &prefused_gate_up_name,
                        &[2 * intermediate, hidden],
                        "gate_up weight shape mismatch",
                    )?;
                    add_tensor_size(&prefused_gate_up_name)?;
                    names.push(prefused_gate_up_name.clone());
                } else {
                    validate_tensor_shape(
                        &tensors,
                        &gate_name,
                        &[intermediate, hidden],
                        "gate_proj weight shape mismatch",
                    )?;
                    validate_tensor_shape(
                        &tensors,
                        &up_name,
                        &[intermediate, hidden],
                        "up_proj weight shape mismatch",
                    )?;
                    add_tensor_size(&gate_name)?;
                    add_tensor_size(&up_name)?;
                    fused_gate_up_bytes += 2 * intermediate * hidden * std::mem::size_of::<f16>();
                }

                layer_names.push(ProbeLayerNames {
                    dims,
                    attn_norm_name,
                    o_proj_name,
                    mlp_norm_name,
                    down_proj_name,
                    prefused_qkv_name,
                    q_name,
                    k_name,
                    v_name,
                    q_norm_name,
                    k_norm_name,
                    v_norm_name,
                    post_attn_norm_name,
                    pre_ff_norm_name,
                    post_ff_norm_name,
                    layer_scalar_name,
                    layer_scalar_dim,
                    prefused_gate_up_name,
                    gate_name,
                    up_name,
                });
            }
        }

        let half_bytes = std::mem::size_of::<f16>();
        let i32_bytes = std::mem::size_of::<i32>();
        let f32_bytes = std::mem::size_of::<f32>();
        let max_probe_tokens = PROBE_METAL_MAX_PROMPT_TOKENS;
        let embed_bytes = embed_info.nbytes;
        let final_norm_bytes = final_norm_info.nbytes;
        let lm_head_bytes = if tie_embeddings {
            0
        } else {
            lm_head_info.nbytes
        };
        let residual_bytes = arch
            .hidden_size
            .checked_mul(max_probe_tokens)
            .and_then(|v| v.checked_mul(half_bytes))
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "residual buffer size overflow",
                    },
                    probe_ctx("prepare"),
                )
            })?;
        let logits_bytes = arch
            .vocab_size
            .checked_mul(max_probe_tokens)
            .and_then(|v| v.checked_mul(half_bytes))
            .ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "logits buffer size overflow",
                    },
                    probe_ctx("prepare"),
                )
            })?;
        let normed_hidden_bytes = residual_bytes;
        let sampled_bytes = max_probe_tokens * i32_bytes;
        let token_ids_bytes = max_probe_tokens * 4;

        let mut scratch_bytes = 0;
        if arch.num_hidden_layers > 0 {
            let hidden = arch.hidden_size;
            let intermediate = arch.intermediate_size;
            for layer_names in &layer_names {
                let dims = layer_names.dims;
                let qkv_out_bytes = max_probe_tokens * dims.qkv_rows * half_bytes;
                let q_bytes = max_probe_tokens * dims.q_dim * half_bytes;
                let k_bytes = max_probe_tokens * dims.kv_dim * half_bytes;
                let v_bytes = max_probe_tokens * dims.kv_dim * half_bytes;
                let attn_out_bytes = max_probe_tokens * dims.q_dim * half_bytes;
                let gate_up_out_bytes = max_probe_tokens * 2 * intermediate * half_bytes;
                let activated_bytes = max_probe_tokens * intermediate * half_bytes;
                let mlp_out_bytes = max_probe_tokens * hidden * half_bytes;

                let block_size = max_probe_tokens;
                let num_blocks_total = max_probe_tokens;
                let kv_cache_bytes = num_blocks_total * block_size * dims.kv_dim * half_bytes * 2;
                let metadata_bytes = (5 * max_probe_tokens + 1) * i32_bytes;

                let half_rope = dims.rope_dim / 2;
                let max_pos = 16usize;
                let rope_table_bytes = max_pos * half_rope * f32_bytes;

                scratch_bytes += qkv_out_bytes
                    + q_bytes
                    + k_bytes
                    + v_bytes
                    + attn_out_bytes
                    + gate_up_out_bytes
                    + activated_bytes
                    + mlp_out_bytes
                    + kv_cache_bytes
                    + rope_table_bytes * 2
                    + metadata_bytes
                    + 64;
            }
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
                    probe_ctx("prepare"),
                )
            })?;
        arena_bytes = arena_bytes.checked_add(64 * 1024).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "model arena byte overflow",
                },
                probe_ctx("prepare"),
            )
        })?;
        arena_bytes = max(arena_bytes, PROBE_METAL_ARENA_BYTES);

        Ok(Self {
            arch,
            tensors,
            weight_prefix,
            embed_name,
            final_norm_name,
            lm_head_name,
            tie_embeddings,
            layer_names,
            names,
            arena_bytes,
        })
    }
}

#[cfg(target_os = "macos")]
impl Gemma4MetalState {
    pub fn dry_run_validate_gemma4_model_dir(model_dir: &Path) -> Result<Gemma4DryRunValidation> {
        Gemma4DryRunValidation::from_model_dir(model_dir)
    }

    pub fn required_probe_model_arena_bytes(model_dir: &Path) -> Result<usize> {
        Ok(ProbeModelPlan::new(model_dir)?.arena_bytes)
    }

    pub fn load_probe_model(
        ctx: &MetalContext,
        arena: &mut MetalBufferArena,
        model_dir: &Path,
    ) -> Result<Self> {
        let _ = ctx;
        let plan = ProbeModelPlan::new(model_dir)?;
        let mut mapped_refs = map_safetensor_to_arena(
            arena,
            model_dir,
            &plan
                .names
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>(),
        )?;
        let embedding = region_lookup(&mut mapped_refs, &plan.embed_name)?;
        let final_norm = region_lookup(&mut mapped_refs, &plan.final_norm_name)?;
        let lm_head = if plan.tie_embeddings {
            embedding.clone()
        } else {
            region_lookup(&mut mapped_refs, &plan.lm_head_name)?
        };

        let half_bytes = std::mem::size_of::<f16>();
        let f32_bytes = std::mem::size_of::<f32>();
        let max_probe_tokens = PROBE_METAL_MAX_PROMPT_TOKENS;
        let residual_bytes = max_probe_tokens * plan.arch.hidden_size * half_bytes;
        let logits_bytes = max_probe_tokens * plan.arch.vocab_size * half_bytes;
        let normed_hidden_bytes = residual_bytes;
        let sampled_bytes = max_probe_tokens * std::mem::size_of::<i32>();
        let token_ids_bytes = max_probe_tokens * 4;

        let mut layers = Vec::new();
        for layer_idx in 0..plan.arch.num_hidden_layers {
            let layer_names = plan.layer_names.get(layer_idx).ok_or_else(|| {
                RvllmError::apple(
                    AppleError::InvalidWeightBlob {
                        reason: "missing layer name plan",
                    },
                    probe_ctx("prepare"),
                )
            })?;
            let hidden = plan.arch.hidden_size;
            let intermediate = plan.arch.intermediate_size;
            let dims = layer_names.dims;
            let q_dim = dims.q_dim;
            let kv_dim = dims.kv_dim;
            let qkv_rows = dims.qkv_rows;

            let attn_norm = region_lookup(&mut mapped_refs, &layer_names.attn_norm_name)?;
            let o_proj = region_lookup(&mut mapped_refs, &layer_names.o_proj_name)?;
            let mlp_norm = region_lookup(&mut mapped_refs, &layer_names.mlp_norm_name)?;
            let down_proj = region_lookup(&mut mapped_refs, &layer_names.down_proj_name)?;
            let q_norm =
                optional_region_lookup(&mut mapped_refs, layer_names.q_norm_name.as_deref())?;
            let k_norm =
                optional_region_lookup(&mut mapped_refs, layer_names.k_norm_name.as_deref())?;
            let v_norm =
                optional_region_lookup(&mut mapped_refs, layer_names.v_norm_name.as_deref())?;
            let post_attn_norm = optional_distinct_region_lookup(
                &mut mapped_refs,
                layer_names.post_attn_norm_name.as_deref(),
                &layer_names.mlp_norm_name,
            )?;
            let pre_ff_norm = optional_region_or_alias_lookup(
                &mut mapped_refs,
                layer_names.pre_ff_norm_name.as_deref(),
                &layer_names.mlp_norm_name,
                &mlp_norm,
            )?;
            let post_ff_norm = optional_distinct_region_lookup(
                &mut mapped_refs,
                layer_names.post_ff_norm_name.as_deref(),
                &layer_names.mlp_norm_name,
            )?;
            let layer_scalar =
                optional_region_lookup(&mut mapped_refs, layer_names.layer_scalar_name.as_deref())?;

            let qkv = if plan.tensors.contains_key(&layer_names.prefused_qkv_name) {
                region_lookup(&mut mapped_refs, &layer_names.prefused_qkv_name)?
            } else {
                let bytes = concat_f16_tensors(
                    model_dir,
                    &plan.tensors,
                    &[
                        layer_names.q_name.clone(),
                        layer_names.k_name.clone(),
                        layer_names.v_name.clone(),
                    ],
                    &[
                        vec![q_dim, hidden],
                        vec![kv_dim, hidden],
                        vec![kv_dim, hidden],
                    ],
                    "fuse_qkv",
                )?;
                map_fused_bytes_to_arena(arena, &format!("metal_fused_qkv_{layer_idx}"), &bytes)?
            };

            let gate_up = if plan
                .tensors
                .contains_key(&layer_names.prefused_gate_up_name)
            {
                region_lookup(&mut mapped_refs, &layer_names.prefused_gate_up_name)?
            } else {
                let bytes = concat_f16_tensors(
                    model_dir,
                    &plan.tensors,
                    &[layer_names.gate_name.clone(), layer_names.up_name.clone()],
                    &[vec![intermediate, hidden], vec![intermediate, hidden]],
                    "fuse_gate_up",
                )?;
                map_fused_bytes_to_arena(
                    arena,
                    &format!("metal_fused_gate_up_{layer_idx}"),
                    &bytes,
                )?
            };

            let qkv_out = arena.region(
                &format!("metal_layer_{layer_idx}_qkv_out"),
                max_probe_tokens * qkv_rows * half_bytes,
                16,
            )?;
            let q = arena.region(
                &format!("metal_layer_{layer_idx}_q"),
                max_probe_tokens * q_dim * half_bytes,
                16,
            )?;
            let k = arena.region(
                &format!("metal_layer_{layer_idx}_k"),
                max_probe_tokens * kv_dim * half_bytes,
                16,
            )?;
            let v = arena.region(
                &format!("metal_layer_{layer_idx}_v"),
                max_probe_tokens * kv_dim * half_bytes,
                16,
            )?;
            let attn_out = arena.region(
                &format!("metal_layer_{layer_idx}_attn_out"),
                max_probe_tokens * q_dim * half_bytes,
                16,
            )?;
            let gate_up_out = arena.region(
                &format!("metal_layer_{layer_idx}_gate_up_out"),
                max_probe_tokens * 2 * intermediate * half_bytes,
                16,
            )?;
            let activated = arena.region(
                &format!("metal_layer_{layer_idx}_activated"),
                max_probe_tokens * intermediate * half_bytes,
                16,
            )?;
            let mlp_out = arena.region(
                &format!("metal_layer_{layer_idx}_mlp_out"),
                max_probe_tokens * hidden * half_bytes,
                16,
            )?;

            let block_size = max_probe_tokens as u32;
            let num_blocks_total = max_probe_tokens as u32;
            let max_blocks_per_seq = 1u32;
            let kv_cache_k = arena.region(
                &format!("metal_layer_{layer_idx}_kv_cache_k"),
                (num_blocks_total as usize) * (block_size as usize) * kv_dim * half_bytes,
                16,
            )?;
            let kv_cache_v = arena.region(
                &format!("metal_layer_{layer_idx}_kv_cache_v"),
                (num_blocks_total as usize) * (block_size as usize) * kv_dim * half_bytes,
                16,
            )?;

            let positions = arena.region(
                &format!("metal_layer_{layer_idx}_positions"),
                max_probe_tokens * 4,
                4,
            )?;
            let slot_mapping = arena.region(
                &format!("metal_layer_{layer_idx}_slot_mapping"),
                max_probe_tokens * 4,
                4,
            )?;
            let context_lens = arena.region(
                &format!("metal_layer_{layer_idx}_context_lens"),
                max_probe_tokens * 4,
                4,
            )?;
            let block_tables = arena.region(
                &format!("metal_layer_{layer_idx}_block_tables"),
                max_probe_tokens * (max_blocks_per_seq as usize) * 4,
                4,
            )?;
            let cu_seqlens = arena.region(
                &format!("metal_layer_{layer_idx}_cu_seqlens"),
                (max_probe_tokens + 1) * 4,
                4,
            )?;

            let half_rope = dims.rope_dim / 2;
            let max_pos = 16usize;
            let cos = arena.region(
                &format!("metal_layer_{layer_idx}_rope_cos"),
                max_pos * half_rope * f32_bytes,
                16,
            )?;
            let sin = arena.region(
                &format!("metal_layer_{layer_idx}_rope_sin"),
                max_pos * half_rope * f32_bytes,
                16,
            )?;

            write_i32_region(arena, &positions, &vec![0; max_probe_tokens])?;
            write_i32_region(arena, &slot_mapping, &vec![0; max_probe_tokens])?;
            write_i32_region(arena, &context_lens, &vec![0; max_probe_tokens])?;
            write_i32_region(arena, &block_tables, &vec![0; max_probe_tokens])?;
            write_i32_region(arena, &cu_seqlens, &vec![0; max_probe_tokens + 1])?;
            write_f32_region(arena, &cos, &vec![1.0; max_pos * half_rope])?;
            write_f32_region(arena, &sin, &vec![0.0; max_pos * half_rope])?;

            layers.push(MetalOneLayerState {
                layer_idx,
                dims,
                attn_norm,
                qkv,
                q_norm,
                k_norm,
                v_norm,
                o_proj,
                mlp_norm,
                post_attn_norm,
                pre_ff_norm,
                post_ff_norm,
                layer_scalar,
                layer_scalar_dim: layer_names.layer_scalar_dim as u32,
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
                cu_seqlens,
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
                probe_ctx("prepare"),
            ));
        }

        let residual = arena.region("metal_model_residual", residual_bytes, 16)?;
        let logits = arena.region("metal_model_logits", logits_bytes, 16)?;
        let normed_hidden = arena.region("metal_model_normed_hidden", normed_hidden_bytes, 16)?;
        let sampled = arena.region("metal_model_sampled", sampled_bytes, 4)?;
        let token_ids = arena.region("metal_model_token_ids", token_ids_bytes, 4)?;

        Ok(Self {
            hidden_size: plan.arch.hidden_size,
            vocab_size: plan.arch.vocab_size,
            num_layers: plan.arch.num_hidden_layers,
            rms_norm_eps: plan.arch.rms_norm_eps,
            final_logit_softcap: plan
                .arch
                .final_logit_softcapping
                .unwrap_or(PROBE_METAL_SOFTCAP),
            embedding_scale: (plan.arch.hidden_size as f32).sqrt(),
            max_probe_tokens,
            embedding,
            final_norm,
            lm_head,
            residual,
            logits,
            normed_hidden,
            sampled,
            token_ids,
            layers,
        })
    }
}

#[cfg(target_os = "macos")]
fn validate_gemma4_model_dir_metadata(model_dir: &Path) -> Result<Gemma4DryRunValidation> {
    let arch = ModelArch::from_dir(model_dir)?;
    if arch.hidden_size == 0 || arch.vocab_size == 0 || arch.num_hidden_layers == 0 {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason:
                    "Gemma4 dry-run requires nonzero hidden_size, vocab_size, and num_hidden_layers",
            },
            probe_ctx("dry_run_validate"),
        ));
    }
    if arch.layer_types.len() != arch.num_hidden_layers {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "Gemma4 dry-run layer_types length does not match num_hidden_layers",
            },
            probe_ctx("dry_run_validate"),
        ));
    }

    let tensors = scan_safetensor_tensors(model_dir)?;
    let weight_prefix = resolve_dry_run_weight_prefix(&tensors);
    let embed_tokens = join_weight_name(&weight_prefix, "embed_tokens.weight");
    let final_norm = join_weight_name(&weight_prefix, "norm.weight");

    validate_required_shape(
        model_dir,
        &tensors,
        &embed_tokens,
        &[arch.vocab_size, arch.hidden_size],
    )?;
    validate_required_shape(model_dir, &tensors, &final_norm, &[arch.hidden_size])?;

    let prefixed_lm_head = join_weight_name(&weight_prefix, "lm_head.weight");
    let tie_word_embeddings = arch.tie_word_embeddings
        || (!tensors.contains_key("lm_head.weight") && !tensors.contains_key(&prefixed_lm_head));
    let lm_head = resolve_optional_dry_run_alias(
        &tensors,
        vec![prefixed_lm_head.clone(), "lm_head.weight".to_owned()],
    );
    if let Some(name) = &lm_head {
        validate_required_shape(
            model_dir,
            &tensors,
            name,
            &[arch.vocab_size, arch.hidden_size],
        )?;
    } else if !tie_word_embeddings {
        return Err(missing_tensor_error(model_dir, &prefixed_lm_head));
    }

    let mut layers = Vec::with_capacity(arch.num_hidden_layers);
    for layer_idx in 0..arch.num_hidden_layers {
        let layer =
            validate_gemma4_dry_run_layer(model_dir, &arch, &tensors, &weight_prefix, layer_idx)?;
        layers.push(layer);
    }

    Ok(Gemma4DryRunValidation {
        weight_prefix,
        num_layers: arch.num_hidden_layers,
        hidden_size: arch.hidden_size,
        vocab_size: arch.vocab_size,
        tie_word_embeddings,
        embed_tokens,
        final_norm,
        lm_head,
        final_logit_softcap: arch.final_logit_softcapping,
        layers,
    })
}

#[cfg(target_os = "macos")]
fn validate_gemma4_dry_run_layer(
    model_dir: &Path,
    arch: &ModelArch,
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    weight_prefix: &str,
    layer_idx: usize,
) -> Result<Gemma4DryRunLayerValidation> {
    let layer_type = arch.layer_types[layer_idx];
    let attention_kind = match layer_type {
        LayerAttnType::SlidingAttention => MetalProbeLayerAttentionKind::Sliding,
        LayerAttnType::Full => MetalProbeLayerAttentionKind::Full,
        LayerAttnType::Linear => {
            return Err(RvllmError::apple(
                AppleError::FeatureNotAvailable {
                    backend: "model-metal-backend",
                    op: "gemma4_dry_run_linear_attention_layer",
                },
                probe_ctx("dry_run_validate"),
            ))
        }
    };
    let head_dim = match layer_type {
        LayerAttnType::SlidingAttention => arch.head_dim,
        LayerAttnType::Full => arch.global_head_dim.unwrap_or(arch.head_dim),
        LayerAttnType::Linear => unreachable!(),
    };
    let num_kv_heads = match layer_type {
        LayerAttnType::SlidingAttention => arch.num_key_value_heads,
        LayerAttnType::Full => arch
            .num_global_key_value_heads
            .unwrap_or(arch.num_key_value_heads),
        LayerAttnType::Linear => unreachable!(),
    };
    let q_dim = arch.num_attention_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let rope_dim = match layer_type {
        LayerAttnType::SlidingAttention => head_dim,
        LayerAttnType::Full => {
            let partial = arch.partial_rotary_factor.unwrap_or(1.0);
            ((head_dim as f32) * partial).round() as usize
        }
        LayerAttnType::Linear => unreachable!(),
    };
    if head_dim == 0 || q_dim == 0 || kv_dim == 0 || rope_dim == 0 || rope_dim % 2 != 0 {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "Gemma4 dry-run derived invalid attention or RoPE dimensions",
            },
            probe_ctx("dry_run_validate"),
        ));
    }
    let rope_theta = match layer_type {
        LayerAttnType::SlidingAttention => arch.rope_theta,
        LayerAttnType::Full => arch.global_rope_theta.unwrap_or(arch.rope_theta),
        LayerAttnType::Linear => unreachable!(),
    };
    if rope_theta <= 0.0 {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "Gemma4 dry-run requires positive RoPE theta",
            },
            probe_ctx("dry_run_validate"),
        ));
    }

    let lprefix = join_weight_name(weight_prefix, &format!("layers.{layer_idx}"));
    let input_layernorm = resolve_required_dry_run_alias(
        model_dir,
        tensors,
        vec![
            format!("{lprefix}.input_layernorm.weight"),
            format!("{lprefix}.pre_attention_layernorm.weight"),
        ],
    )?;
    let post_attention_layernorm = resolve_required_dry_run_alias(
        model_dir,
        tensors,
        vec![format!("{lprefix}.post_attention_layernorm.weight")],
    )?;
    let pre_feedforward_layernorm = resolve_required_dry_run_alias(
        model_dir,
        tensors,
        vec![format!("{lprefix}.pre_feedforward_layernorm.weight")],
    )?;
    let post_feedforward_layernorm = resolve_required_dry_run_alias(
        model_dir,
        tensors,
        vec![format!("{lprefix}.post_feedforward_layernorm.weight")],
    )?;
    for name in [
        &input_layernorm,
        &post_attention_layernorm,
        &pre_feedforward_layernorm,
        &post_feedforward_layernorm,
    ] {
        validate_required_shape(model_dir, tensors, name, &[arch.hidden_size])?;
    }

    let q_proj = format!("{lprefix}.self_attn.q_proj.weight");
    let k_proj = format!("{lprefix}.self_attn.k_proj.weight");
    let v_proj_name = format!("{lprefix}.self_attn.v_proj.weight");
    validate_required_shape(model_dir, tensors, &q_proj, &[q_dim, arch.hidden_size])?;
    validate_required_shape(model_dir, tensors, &k_proj, &[kv_dim, arch.hidden_size])?;
    let v_proj = if tensors.contains_key(&v_proj_name) {
        validate_required_shape(
            model_dir,
            tensors,
            &v_proj_name,
            &[kv_dim, arch.hidden_size],
        )?;
        Some(v_proj_name)
    } else if arch.attention_k_eq_v {
        None
    } else {
        return Err(missing_tensor_error(model_dir, &v_proj_name));
    };

    let o_proj = format!("{lprefix}.self_attn.o_proj.weight");
    validate_required_shape(model_dir, tensors, &o_proj, &[arch.hidden_size, q_dim])?;

    let q_norm = format!("{lprefix}.self_attn.q_norm.weight");
    let k_norm = format!("{lprefix}.self_attn.k_norm.weight");
    validate_required_shape(model_dir, tensors, &q_norm, &[head_dim])?;
    validate_required_shape(model_dir, tensors, &k_norm, &[head_dim])?;

    let v_norm = format!("{lprefix}.self_attn.v_norm.weight");
    if tensors.contains_key(&v_norm) {
        validate_required_shape(model_dir, tensors, &v_norm, &[head_dim])?;
    }

    let gate_proj = format!("{lprefix}.mlp.gate_proj.weight");
    let up_proj = format!("{lprefix}.mlp.up_proj.weight");
    let down_proj = format!("{lprefix}.mlp.down_proj.weight");
    validate_required_shape(
        model_dir,
        tensors,
        &gate_proj,
        &[arch.intermediate_size, arch.hidden_size],
    )?;
    validate_required_shape(
        model_dir,
        tensors,
        &up_proj,
        &[arch.intermediate_size, arch.hidden_size],
    )?;
    validate_required_shape(
        model_dir,
        tensors,
        &down_proj,
        &[arch.hidden_size, arch.intermediate_size],
    )?;

    let layer_scalar = resolve_optional_dry_run_alias(
        tensors,
        vec![
            format!("{lprefix}.layer_scalar"),
            format!("{lprefix}.layer_scalar.weight"),
        ],
    );
    let layer_scalar_dim = match &layer_scalar {
        Some(name) => {
            let info = tensors
                .get(name)
                .ok_or_else(|| missing_tensor_error(model_dir, name))?;
            if info.shape.len() == 1 && (info.shape[0] == 1 || info.shape[0] == arch.hidden_size) {
                info.shape[0]
            } else {
                return Err(shape_mismatch_error(
                    model_dir,
                    name,
                    &[arch.hidden_size],
                    &info.shape,
                ));
            }
        }
        None => 0,
    };

    Ok(Gemma4DryRunLayerValidation {
        layer_idx,
        attention_kind,
        q_proj,
        k_proj,
        v_uses_k_proj: v_proj.is_none() && arch.attention_k_eq_v,
        v_proj,
        input_layernorm,
        post_attention_layernorm,
        pre_feedforward_layernorm,
        post_feedforward_layernorm,
        q_norm,
        k_norm,
        layer_scalar,
        layer_scalar_dim,
        rope_dim,
        rope_theta,
        sliding_window: (attention_kind == MetalProbeLayerAttentionKind::Sliding)
            .then_some(arch.sliding_window)
            .flatten(),
    })
}

#[cfg(target_os = "macos")]
fn validate_required_shape(
    model_dir: &Path,
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    name: &str,
    expected: &[usize],
) -> Result<()> {
    let info = tensors
        .get(name)
        .ok_or_else(|| missing_tensor_error(model_dir, name))?;
    if info.shape.as_slice() != expected {
        return Err(shape_mismatch_error(model_dir, name, expected, &info.shape));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn resolve_required_dry_run_alias(
    model_dir: &Path,
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    candidates: Vec<String>,
) -> Result<String> {
    candidates
        .iter()
        .find(|name| tensors.contains_key(*name))
        .cloned()
        .ok_or_else(|| missing_tensor_error(model_dir, &candidates[0]))
}

#[cfg(target_os = "macos")]
fn resolve_optional_dry_run_alias(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    candidates: Vec<String>,
) -> Option<String> {
    candidates
        .into_iter()
        .find(|name| tensors.contains_key(name))
}

#[cfg(target_os = "macos")]
fn resolve_dry_run_weight_prefix(tensors: &BTreeMap<String, SafetensorTensorInfo>) -> String {
    for prefix in ["model.language_model", "model", "language_model.model", ""] {
        if tensors.contains_key(&join_weight_name(prefix, "embed_tokens.weight")) {
            return prefix.to_owned();
        }
    }
    "model".to_owned()
}

#[cfg(target_os = "macos")]
fn join_weight_name(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_owned()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[cfg(target_os = "macos")]
fn missing_tensor_error(model_dir: &Path, name: &str) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::MissingTensor {
            name: name.to_owned(),
        },
        ctx: LoaderCtx {
            path: model_dir.to_path_buf(),
            tensor: Some(name.to_owned()),
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(target_os = "macos")]
fn shape_mismatch_error(
    model_dir: &Path,
    name: &str,
    expected: &[usize],
    got: &[usize],
) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::ShapeMismatch {
            tensor: name.to_owned(),
            expected: expected.to_vec(),
            got: got.to_vec(),
        },
        ctx: LoaderCtx {
            path: model_dir.to_path_buf(),
            tensor: Some(name.to_owned()),
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(target_os = "macos")]
fn resolve_tensor_alias(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    candidates: Vec<String>,
    missing_reason: &'static str,
) -> Result<String> {
    candidates
        .into_iter()
        .find(|name| tensors.contains_key(name))
        .ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: missing_reason,
                },
                probe_ctx("prepare"),
            )
        })
}

#[cfg(target_os = "macos")]
fn resolve_optional_tensor_alias(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    candidates: Vec<String>,
) -> Option<String> {
    candidates
        .into_iter()
        .find(|name| tensors.contains_key(name))
}

#[cfg(target_os = "macos")]
fn validate_tensor_shape(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    name: &str,
    expected: &[usize],
    reason: &'static str,
) -> Result<()> {
    let info = tensors.get(name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing layer weight",
            },
            probe_ctx("prepare"),
        )
    })?;
    if info.shape.as_slice() != expected {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob { reason },
            probe_ctx("prepare"),
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_optional_norm_shape(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    name: &Option<String>,
    expected_dim: usize,
    reason: &'static str,
) -> Result<()> {
    let Some(name) = name else {
        return Ok(());
    };
    let info = tensors.get(name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing optional norm tensor",
            },
            probe_ctx("prepare"),
        )
    })?;
    if info.shape.len() != 1 || info.shape[0] != expected_dim {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob { reason },
            probe_ctx("prepare"),
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_optional_layer_scalar_shape(
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    name: &Option<String>,
    hidden: usize,
) -> Result<usize> {
    let Some(name) = name else {
        return Ok(0);
    };
    let info = tensors.get(name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing layer_scalar tensor",
            },
            probe_ctx("prepare"),
        )
    })?;
    if info.shape.len() == 1 && (info.shape[0] == 1 || info.shape[0] == hidden) {
        Ok(info.shape[0])
    } else {
        Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "layer_scalar shape mismatch",
            },
            probe_ctx("prepare"),
        ))
    }
}

#[cfg(target_os = "macos")]
fn resolve_weight_prefix(tensors: &BTreeMap<String, SafetensorTensorInfo>) -> String {
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

#[cfg(target_os = "macos")]
fn region_lookup(refs: &mut Vec<(String, MetalRegion)>, name: &str) -> Result<MetalRegion> {
    let idx = refs.iter().position(|(n, _)| n == name).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "missing mapped tensor in map",
            },
            probe_ctx("resolve_regions"),
        )
    })?;
    Ok(refs.swap_remove(idx).1)
}

#[cfg(target_os = "macos")]
fn optional_region_lookup(
    refs: &mut Vec<(String, MetalRegion)>,
    name: Option<&str>,
) -> Result<Option<MetalRegion>> {
    match name {
        Some(name) => Ok(Some(region_lookup(refs, name)?)),
        None => Ok(None),
    }
}

#[cfg(target_os = "macos")]
fn optional_distinct_region_lookup(
    refs: &mut Vec<(String, MetalRegion)>,
    name: Option<&str>,
    alias_name: &str,
) -> Result<Option<MetalRegion>> {
    match name {
        Some(name) if name != alias_name => Ok(Some(region_lookup(refs, name)?)),
        _ => Ok(None),
    }
}

#[cfg(target_os = "macos")]
fn optional_region_or_alias_lookup(
    refs: &mut Vec<(String, MetalRegion)>,
    name: Option<&str>,
    alias_name: &str,
    alias_region: &MetalRegion,
) -> Result<Option<MetalRegion>> {
    match name {
        Some(name) if name == alias_name => Ok(Some(alias_region.clone())),
        Some(name) => Ok(Some(region_lookup(refs, name)?)),
        None => Ok(None),
    }
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn concat_f16_tensors(
    model_dir: &Path,
    tensors: &BTreeMap<String, SafetensorTensorInfo>,
    names: &[String],
    expected_shapes: &[Vec<usize>],
    op: &'static str,
) -> Result<Vec<u8>> {
    if names.len() != expected_shapes.len() {
        return Err(RvllmError::apple(
            AppleError::InvalidWeightBlob {
                reason: "concat shape metadata mismatch",
            },
            probe_ctx(op),
        ));
    }

    let mut out = Vec::new();
    for (name, expected_shape) in names.iter().zip(expected_shapes.iter()) {
        let info = tensors.get(name).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "missing tensor for concat",
                },
                probe_ctx(op),
            )
        })?;
        if info.shape != *expected_shape {
            return Err(RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "tensor shape mismatch for concat",
                },
                probe_ctx(op),
            ));
        }
        let bytes = load_safetensor_f16(model_dir, name)?;
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
fn write_i32_region(arena: &MetalBufferArena, region: &MetalRegion, values: &[i32]) -> Result<()> {
    unsafe {
        let dst = arena.host_ptr(region) as *mut i32;
        ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn write_f32_region(arena: &MetalBufferArena, region: &MetalRegion, values: &[f32]) -> Result<()> {
    unsafe {
        let dst = arena.host_ptr(region) as *mut f32;
        ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use serde_json::{Map, Value};
    use std::{
        fs::{self, File},
        io::Write,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn test_fixture_dir(name: &str) -> PathBuf {
        let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rvllm-metal-{name}-{}-{}-{id}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create fixture dir");
        dir
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect()
    }

    fn add_tensor(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        data: &[f32],
        shape: &[usize],
    ) {
        let start = payload.len();
        payload.extend_from_slice(&f16_bytes(data));
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
    }

    fn write_two_layer_sliding_global_plan_fixture() -> PathBuf {
        let dir = test_fixture_dir("sliding-global-dims");
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let sliding_head_dim = 128usize;
        let global_head_dim = 256usize;

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();
        let zeros_embed = vec![0.0f32; vocab * hidden];
        let ones_hidden = vec![1.0f32; hidden];
        let zeros_lm_head = vec![0.0f32; vocab * hidden];
        let zeros_gate_up = vec![0.0f32; 2 * intermediate * hidden];
        let zeros_down = vec![0.0f32; hidden * intermediate];

        add_tensor(
            &mut header,
            &mut payload,
            "model.embed_tokens.weight",
            &zeros_embed,
            &[vocab, hidden],
        );
        add_tensor(
            &mut header,
            &mut payload,
            "model.norm.weight",
            &ones_hidden,
            &[hidden],
        );
        add_tensor(
            &mut header,
            &mut payload,
            "lm_head.weight",
            &zeros_lm_head,
            &[vocab, hidden],
        );

        for (layer_idx, head_dim) in [(0usize, sliding_head_dim), (1usize, global_head_dim)] {
            let qkv_rows = 3 * head_dim;
            let zeros_qkv = vec![0.0f32; qkv_rows * hidden];
            let zeros_o = vec![0.0f32; hidden * head_dim];
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &ones_hidden,
                &[hidden],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.self_attn.qkv.weight"),
                &zeros_qkv,
                &[qkv_rows, hidden],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                &zeros_o,
                &[hidden, head_dim],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.mlp_norm.weight"),
                &ones_hidden,
                &[hidden],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.mlp.gate_up.weight"),
                &zeros_gate_up,
                &[2 * intermediate, hidden],
            );
            add_tensor(
                &mut header,
                &mut payload,
                &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                &zeros_down,
                &[hidden, intermediate],
            );
        }

        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {{
    "num_hidden_layers": 2,
    "hidden_size": {hidden},
    "intermediate_size": {intermediate},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {sliding_head_dim},
    "global_head_dim": {global_head_dim},
    "num_global_key_value_heads": 1,
    "layer_types": ["sliding_attention", "full_attention"],
    "vocab_size": {vocab},
    "max_position_embeddings": 16,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 0.0,
    "tie_word_embeddings": false
  }}
}}"#
            ),
        )
        .expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out = File::create(dir.join("model.safetensors")).expect("create safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[test]
    fn probe_model_plan_selects_sliding_and_global_layer_dims() {
        let dir = write_two_layer_sliding_global_plan_fixture();
        let plan = ProbeModelPlan::new(&dir).expect("build probe model plan");

        assert_eq!(plan.layer_names.len(), 2);
        let sliding = plan.layer_names[0].dims;
        assert_eq!(
            sliding.attention_kind,
            MetalProbeLayerAttentionKind::Sliding
        );
        assert_eq!(sliding.num_heads, 1);
        assert_eq!(sliding.num_kv_heads, 1);
        assert_eq!(sliding.head_dim, 128);
        assert_eq!(sliding.rope_dim, 128);
        assert_eq!(sliding.q_dim, 128);
        assert_eq!(sliding.kv_dim, 128);
        assert_eq!(sliding.qkv_rows, 384);

        let full = plan.layer_names[1].dims;
        assert_eq!(full.attention_kind, MetalProbeLayerAttentionKind::Full);
        assert_eq!(full.num_heads, 1);
        assert_eq!(full.num_kv_heads, 1);
        assert_eq!(full.head_dim, 256);
        assert_eq!(full.rope_dim, 256);
        assert_eq!(full.q_dim, 256);
        assert_eq!(full.kv_dim, 256);
        assert_eq!(full.qkv_rows, 768);

        let _ = fs::remove_dir_all(dir);
    }

    fn add_zero_tensor(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
    ) {
        let count = shape.iter().copied().product::<usize>();
        add_tensor(header, payload, name, &vec![0.0f32; count], shape);
    }

    fn write_dry_run_full_gemma_style_fixture(
        tie_embeddings: bool,
        attention_k_eq_v: bool,
        omit_lm_head: bool,
        omit_v_proj: bool,
        q_proj0_shape: Option<&[usize]>,
        q_proj1_shape: Option<&[usize]>,
        omit_q_norm0: bool,
    ) -> PathBuf {
        let dir = test_fixture_dir("dry-run-full-gemma-style");
        let hidden = 128usize;
        let intermediate = 256usize;
        let vocab = 8usize;
        let sliding_head_dim = 128usize;
        let global_head_dim = 256usize;
        let prefix = "model.language_model";

        let mut header = Map::<String, Value>::new();
        let mut payload = Vec::new();

        add_zero_tensor(
            &mut header,
            &mut payload,
            &format!("{prefix}.embed_tokens.weight"),
            &[vocab, hidden],
        );
        add_zero_tensor(
            &mut header,
            &mut payload,
            &format!("{prefix}.norm.weight"),
            &[hidden],
        );
        if !omit_lm_head {
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{prefix}.lm_head.weight"),
                &[vocab, hidden],
            );
        }

        for (layer_idx, head_dim) in [(0usize, sliding_head_dim), (1usize, global_head_dim)] {
            let q_dim = head_dim;
            let kv_dim = head_dim;
            let lprefix = format!("{prefix}.layers.{layer_idx}");
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ] {
                add_zero_tensor(
                    &mut header,
                    &mut payload,
                    &format!("{lprefix}.{suffix}"),
                    &[hidden],
                );
            }

            let q_shape = match layer_idx {
                0 => q_proj0_shape
                    .map(|shape| shape.to_vec())
                    .unwrap_or_else(|| vec![q_dim, hidden]),
                1 => q_proj1_shape
                    .map(|shape| shape.to_vec())
                    .unwrap_or_else(|| vec![q_dim, hidden]),
                _ => vec![q_dim, hidden],
            };
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.q_proj.weight"),
                &q_shape,
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.k_proj.weight"),
                &[kv_dim, hidden],
            );
            if !omit_v_proj {
                add_zero_tensor(
                    &mut header,
                    &mut payload,
                    &format!("{lprefix}.self_attn.v_proj.weight"),
                    &[kv_dim, hidden],
                );
            }
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.o_proj.weight"),
                &[hidden, q_dim],
            );
            if !(layer_idx == 0 && omit_q_norm0) {
                add_zero_tensor(
                    &mut header,
                    &mut payload,
                    &format!("{lprefix}.self_attn.q_norm.weight"),
                    &[head_dim],
                );
            }
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.k_norm.weight"),
                &[head_dim],
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.layer_scalar"),
                &[hidden],
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.mlp.gate_proj.weight"),
                &[intermediate, hidden],
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.mlp.up_proj.weight"),
                &[intermediate, hidden],
            );
            add_zero_tensor(
                &mut header,
                &mut payload,
                &format!("{lprefix}.mlp.down_proj.weight"),
                &[hidden, intermediate],
            );
        }

        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "text_config": {{
    "num_hidden_layers": 2,
    "hidden_size": {hidden},
    "intermediate_size": {intermediate},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {sliding_head_dim},
    "global_head_dim": {global_head_dim},
    "num_global_key_value_heads": 1,
    "layer_types": ["sliding_attention", "full_attention"],
    "vocab_size": {vocab},
    "max_position_embeddings": 1024,
    "sliding_window": 32,
    "rms_norm_eps": 0.000001,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": {tie_embeddings},
    "attention_k_eq_v": {attention_k_eq_v},
    "rope_parameters": {{
      "sliding_attention": {{"rope_theta": 10000.0}},
      "full_attention": {{"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}}
    }}
  }}
}}"#
            ),
        )
        .expect("write config");

        let header_json = serde_json::to_string(&header).expect("serialize fixture header");
        let mut out = File::create(dir.join("model.safetensors")).expect("create safetensors");
        out.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("write header len");
        out.write_all(header_json.as_bytes())
            .expect("write header bytes");
        out.write_all(&payload).expect("write payload");
        dir
    }

    #[test]
    fn dry_run_validates_text_config_and_model_language_model_prefix() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, false, false, false, None, None, false);
        let validation =
            Gemma4DryRunValidation::from_model_dir(&dir).expect("dry-run validates fixture");

        assert_eq!(validation.weight_prefix, "model.language_model");
        assert_eq!(validation.num_layers, 2);
        assert_eq!(validation.hidden_size, 128);
        assert_eq!(validation.vocab_size, 8);
        assert_eq!(validation.final_logit_softcap, Some(30.0));
        assert_eq!(
            validation.layers[0].attention_kind,
            MetalProbeLayerAttentionKind::Sliding
        );
        assert_eq!(validation.layers[0].sliding_window, Some(32));
        assert_eq!(
            validation.layers[1].attention_kind,
            MetalProbeLayerAttentionKind::Full
        );
        assert_eq!(validation.layers[1].rope_dim, 64);
        assert_eq!(validation.layers[1].rope_theta, 1000000.0);
        assert_eq!(validation.layers[0].layer_scalar_dim, 128);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_allows_tied_embeddings_without_lm_head() {
        let dir =
            write_dry_run_full_gemma_style_fixture(true, false, true, false, None, None, false);
        let validation = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect("tied embeddings do not require lm_head");

        assert!(validation.tie_word_embeddings);
        assert_eq!(validation.lm_head, None);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_allows_missing_v_proj_when_attention_k_eq_v() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, true, false, true, None, None, false);
        let validation = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect("attention_k_eq_v permits missing v_proj");

        assert!(validation.layers[0].v_uses_k_proj);
        assert_eq!(validation.layers[0].v_proj, None);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_missing_tensor_error_names_missing_tensor() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, false, false, true, None, None, false);
        let err = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect_err("missing v_proj must fail");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.v_proj.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_shape_mismatch_error_names_tensor_and_expected_actual_shapes() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            false,
            Some(&[127, 128]),
            None,
            false,
        );
        let err = Gemma4MetalState::dry_run_validate_gemma4_model_dir(&dir)
            .expect_err("q_proj shape mismatch must fail");
        let msg = format!("{err}");

        assert!(msg.contains("ShapeMismatch"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(msg.contains("expected: [128, 128]"));
        assert!(msg.contains("got: [127, 128]"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_missing_q_norm_error_names_missing_tensor() {
        let dir =
            write_dry_run_full_gemma_style_fixture(false, false, false, false, None, None, true);
        let err =
            Gemma4DryRunValidation::from_model_dir(&dir).expect_err("missing q_norm must fail");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_norm.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dry_run_global_layer_shape_mismatch_uses_global_head_dim() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            false,
            None,
            Some(&[255, 128]),
            false,
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("global q_proj shape mismatch must fail");
        let msg = format!("{err}");

        assert!(msg.contains("ShapeMismatch"));
        assert!(msg.contains("model.language_model.layers.1.self_attn.q_proj.weight"));
        assert!(msg.contains("expected: [256, 128]"));
        assert!(msg.contains("got: [255, 128]"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore = "set RVLLM_GEMMA4_MODEL_DIR to validate a real Gemma4 model directory"]
    fn real_gemma4_model_dir_dry_run_validates_when_env_is_set() {
        let Some(model_dir) = std::env::var_os("RVLLM_GEMMA4_MODEL_DIR") else {
            eprintln!("skipping: RVLLM_GEMMA4_MODEL_DIR is not set");
            return;
        };
        let model_dir = PathBuf::from(model_dir);
        let validation = Gemma4DryRunValidation::from_model_dir(&model_dir).unwrap_or_else(|err| {
            panic!(
                "Gemma4 dry-run validation failed for {}: {err}",
                model_dir.display()
            )
        });

        assert!(validation.num_layers > 0);
        assert!(validation.hidden_size > 0);
        assert!(validation.vocab_size > 0);
        assert_eq!(validation.layers.len(), validation.num_layers);
        eprintln!(
            "validated Gemma4 dry-run shapes: dir={} prefix={} layers={} hidden={} vocab={}",
            model_dir.display(),
            validation.weight_prefix,
            validation.num_layers,
            validation.hidden_size,
            validation.vocab_size
        );
    }
}

#[derive(Debug, Default, Clone)]
#[cfg(not(target_os = "macos"))]
pub struct Gemma4MetalState;
