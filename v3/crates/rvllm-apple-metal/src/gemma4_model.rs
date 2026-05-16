#[cfg(target_os = "macos")]
use std::{cmp::max, collections::BTreeMap, path::Path, ptr};

#[cfg(target_os = "macos")]
use half::f16;
#[cfg(target_os = "macos")]
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
#[cfg(target_os = "macos")]
use rvllm_loader::load::ModelArch;

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

    pub attn_norm: MetalRegion,
    pub qkv: MetalRegion,
    pub o_proj: MetalRegion,
    pub mlp_norm: MetalRegion,
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

    pub kv_cache_k: MetalRegion,
    pub kv_cache_v: MetalRegion,

    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
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
    names: Vec<String>,
    arena_bytes: usize,
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

        if arch.num_hidden_layers > 0 {
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
                    probe_ctx("prepare"),
                ));
            }
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
                let lprefix = format!("{weight_prefix}.layers.{layer_idx}");
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
                    fused_qkv_bytes += qkv_rows * hidden * std::mem::size_of::<f16>();
                }

                if use_prefused_gate_up {
                    add_tensor_size(&prefused_gate_up_name)?;
                    names.push(prefused_gate_up_name);
                } else {
                    add_tensor_size(&gate_name)?;
                    add_tensor_size(&up_name)?;
                    fused_gate_up_bytes += 2 * intermediate * hidden * std::mem::size_of::<f16>();
                }
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
                probe_ctx("prepare"),
            )
        })?;
        let logits_bytes = arch.vocab_size.checked_mul(half_bytes).ok_or_else(|| {
            RvllmError::apple(
                AppleError::InvalidWeightBlob {
                    reason: "logits buffer size overflow",
                },
                probe_ctx("prepare"),
            )
        })?;
        let normed_hidden_bytes = residual_bytes;
        let sampled_bytes = i32_bytes;
        let token_ids_bytes = 4;

        let mut scratch_bytes = 0;
        if arch.num_hidden_layers > 0 {
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

            let layer_scratch_bytes = qkv_out_bytes
                + q_bytes
                + k_bytes
                + v_bytes
                + attn_out_bytes
                + gate_up_out_bytes
                + activated_bytes
                + mlp_out_bytes
                + kv_cache_bytes
                + rope_table_bytes * 2
                + 64;
            scratch_bytes = arch.num_hidden_layers * layer_scratch_bytes;
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
            names,
            arena_bytes,
        })
    }
}

#[cfg(target_os = "macos")]
impl Gemma4MetalState {
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
        let residual_bytes = plan.arch.hidden_size * half_bytes;
        let logits_bytes = plan.arch.vocab_size * half_bytes;
        let normed_hidden_bytes = residual_bytes;
        let sampled_bytes = std::mem::size_of::<i32>();
        let token_ids_bytes = 4;

        let mut layers = Vec::new();
        for layer_idx in 0..plan.arch.num_hidden_layers {
            let hidden = plan.arch.hidden_size;
            let intermediate = plan.arch.intermediate_size;
            let num_heads = plan.arch.num_attention_heads;
            let num_kv_heads = plan.arch.num_key_value_heads;
            let head_dim = plan.arch.head_dim;
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_rows = q_dim + 2 * kv_dim;

            let lprefix = format!("{}.layers.{layer_idx}", plan.weight_prefix);
            let attn_norm = region_lookup(
                &mut mapped_refs,
                &format!("{lprefix}.input_layernorm.weight"),
            )?;
            let o_proj = region_lookup(
                &mut mapped_refs,
                &format!("{lprefix}.self_attn.o_proj.weight"),
            )?;
            let mlp_norm = region_lookup(&mut mapped_refs, &format!("{lprefix}.mlp_norm.weight"))?;
            let down_proj =
                region_lookup(&mut mapped_refs, &format!("{lprefix}.mlp.down_proj.weight"))?;

            let prefused_qkv_name = format!("{lprefix}.self_attn.qkv.weight");
            let qkv = if plan.tensors.contains_key(&prefused_qkv_name) {
                region_lookup(&mut mapped_refs, &prefused_qkv_name)?
            } else {
                let bytes = concat_f16_tensors(
                    model_dir,
                    &plan.tensors,
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
                map_fused_bytes_to_arena(arena, &format!("metal_fused_qkv_{layer_idx}"), &bytes)?
            };

            let prefused_gate_up_name = format!("{lprefix}.mlp.gate_up.weight");
            let gate_up = if plan.tensors.contains_key(&prefused_gate_up_name) {
                region_lookup(&mut mapped_refs, &prefused_gate_up_name)?
            } else {
                let bytes = concat_f16_tensors(
                    model_dir,
                    &plan.tensors,
                    &[
                        format!("{lprefix}.mlp.gate_proj.weight"),
                        format!("{lprefix}.mlp.up_proj.weight"),
                    ],
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
                qkv_rows * half_bytes,
                16,
            )?;
            let q = arena.region(
                &format!("metal_layer_{layer_idx}_q"),
                q_dim * half_bytes,
                16,
            )?;
            let k = arena.region(
                &format!("metal_layer_{layer_idx}_k"),
                kv_dim * half_bytes,
                16,
            )?;
            let v = arena.region(
                &format!("metal_layer_{layer_idx}_v"),
                kv_dim * half_bytes,
                16,
            )?;
            let attn_out = arena.region(
                &format!("metal_layer_{layer_idx}_attn_out"),
                q_dim * half_bytes,
                16,
            )?;
            let gate_up_out = arena.region(
                &format!("metal_layer_{layer_idx}_gate_up_out"),
                2 * intermediate * half_bytes,
                16,
            )?;
            let activated = arena.region(
                &format!("metal_layer_{layer_idx}_activated"),
                intermediate * half_bytes,
                16,
            )?;
            let mlp_out = arena.region(
                &format!("metal_layer_{layer_idx}_mlp_out"),
                hidden * half_bytes,
                16,
            )?;

            let block_size = 1u32;
            let num_blocks_total = 1u32;
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

            let positions = arena.region(&format!("metal_layer_{layer_idx}_positions"), 4, 4)?;
            let slot_mapping =
                arena.region(&format!("metal_layer_{layer_idx}_slot_mapping"), 4, 4)?;
            let context_lens =
                arena.region(&format!("metal_layer_{layer_idx}_context_lens"), 4, 4)?;
            let block_tables =
                arena.region(&format!("metal_layer_{layer_idx}_block_tables"), 4, 4)?;

            let half_rope = head_dim / 2;
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

            write_i32_region(arena, &positions, &[0])?;
            write_i32_region(arena, &slot_mapping, &[0])?;
            write_i32_region(arena, &context_lens, &[1])?;
            write_i32_region(arena, &block_tables, &[0])?;
            write_f32_region(arena, &cos, &vec![1.0; max_pos * half_rope])?;
            write_f32_region(arena, &sin, &vec![0.0; max_pos * half_rope])?;

            layers.push(MetalOneLayerState {
                layer_idx,
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

#[derive(Debug, Default, Clone)]
#[cfg(not(target_os = "macos"))]
pub struct Gemma4MetalState;
