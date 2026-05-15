#[cfg(target_os = "macos")]
use crate::arena::MetalRegion;

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

    pub one_layer: Option<MetalOneLayerState>,
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

#[derive(Debug, Default, Clone)]
#[cfg(not(target_os = "macos"))]
pub struct Gemma4MetalState;
