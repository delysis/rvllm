//! Host-platform Gemma 4 metadata/shape dry-run validation.
//!
//! This module validates config-derived Gemma 4 tensor names and shapes
//! from safetensor headers only. It does not allocate backend buffers,
//! open Metal, or run decode.

use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};

use crate::{
    load::{LayerAttnType, ModelArch},
    safetensors::ShardIndex,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4DryRunAttentionKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, PartialEq)]
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
pub struct Gemma4DryRunLayerValidation {
    pub layer_idx: usize,
    pub attention_kind: Gemma4DryRunAttentionKind,
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

#[derive(Debug, Clone)]
struct DryRunTensorInfo {
    shape: Vec<usize>,
    dtype: DType,
}

impl Gemma4DryRunValidation {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        validate_gemma4_model_dir_metadata(model_dir)
    }
}

pub fn validate_gemma4_model_dir_metadata(model_dir: &Path) -> Result<Gemma4DryRunValidation> {
    validate_gemma4_dry_run_config_identity(model_dir)?;
    let arch = ModelArch::from_dir(model_dir)?;
    if arch.hidden_size == 0 || arch.vocab_size == 0 || arch.num_hidden_layers == 0 {
        return Err(corrupt_error(
            model_dir.join("config.json"),
            "Gemma4 dry-run requires nonzero hidden_size, vocab_size, and num_hidden_layers",
        ));
    }
    if arch.layer_types.len() != arch.num_hidden_layers {
        return Err(corrupt_error(
            model_dir.join("config.json"),
            "Gemma4 dry-run layer_types length does not match num_hidden_layers",
        ));
    }

    let tensors = scan_safetensor_tensor_metadata(model_dir)?;
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
    let tie_word_embeddings = arch.tie_word_embeddings;
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
    validate_gemma4_dry_run_fp8_scales(model_dir, &tensors, lm_head.as_deref(), &layers)?;

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

fn validate_gemma4_dry_run_layer(
    model_dir: &Path,
    arch: &ModelArch,
    tensors: &BTreeMap<String, DryRunTensorInfo>,
    weight_prefix: &str,
    layer_idx: usize,
) -> Result<Gemma4DryRunLayerValidation> {
    let layer_type = arch.layer_types[layer_idx];
    let attention_kind = match layer_type {
        LayerAttnType::SlidingAttention => Gemma4DryRunAttentionKind::Sliding,
        LayerAttnType::Full => Gemma4DryRunAttentionKind::Full,
        LayerAttnType::Linear => {
            return Err(corrupt_error(
                model_dir.join("config.json"),
                "Gemma4 dry-run does not support linear attention layers",
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
    if arch.num_attention_heads == 0
        || num_kv_heads == 0
        || arch.num_attention_heads % num_kv_heads != 0
        || head_dim == 0
        || head_dim % 2 != 0
        || q_dim == 0
        || kv_dim == 0
        || rope_dim == 0
        || rope_dim % 2 != 0
    {
        return Err(corrupt_error(
            model_dir.join("config.json"),
            "Gemma4 dry-run derived invalid grouped attention or RoPE dimensions",
        ));
    }
    let rope_theta = match layer_type {
        LayerAttnType::SlidingAttention => arch.rope_theta,
        LayerAttnType::Full => arch.global_rope_theta.unwrap_or(arch.rope_theta),
        LayerAttnType::Linear => unreachable!(),
    };
    if rope_theta <= 0.0 {
        return Err(corrupt_error(
            model_dir.join("config.json"),
            "Gemma4 dry-run requires positive RoPE theta",
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
    } else if arch.attention_k_eq_v && attention_kind == Gemma4DryRunAttentionKind::Full {
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

    let layer_scalar = resolve_required_dry_run_alias(
        model_dir,
        tensors,
        vec![
            format!("{lprefix}.layer_scalar"),
            format!("{lprefix}.layer_scalar.weight"),
        ],
    )?;
    let layer_scalar_dim = {
        let info = tensors
            .get(&layer_scalar)
            .ok_or_else(|| missing_tensor_error(model_dir, &layer_scalar))?;
        if info.shape.len() == 1 && (info.shape[0] == 1 || info.shape[0] == arch.hidden_size) {
            info.shape[0]
        } else {
            return Err(shape_mismatch_error(
                model_dir,
                &layer_scalar,
                &[arch.hidden_size],
                &info.shape,
            ));
        }
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
        layer_scalar: Some(layer_scalar),
        layer_scalar_dim,
        rope_dim,
        rope_theta,
        sliding_window: (attention_kind == Gemma4DryRunAttentionKind::Sliding)
            .then_some(arch.sliding_window)
            .flatten(),
    })
}

fn validate_gemma4_dry_run_fp8_scales(
    model_dir: &Path,
    tensors: &BTreeMap<String, DryRunTensorInfo>,
    lm_head: Option<&str>,
    layers: &[Gemma4DryRunLayerValidation],
) -> Result<()> {
    let fp8_prequant = layers
        .first()
        .and_then(|layer| tensors.get(&layer.q_proj))
        .is_some_and(|info| info.dtype == DType::Fp8E4M3);
    let mut layer_linears = Vec::new();
    for layer in layers {
        let lprefix = layer
            .q_proj
            .strip_suffix(".self_attn.q_proj.weight")
            .ok_or_else(|| corrupt_error(model_dir, "Gemma4 dry-run internal q_proj name error"))?;
        layer_linears.push(layer.q_proj.clone());
        layer_linears.push(layer.k_proj.clone());
        if let Some(v_proj) = &layer.v_proj {
            layer_linears.push(v_proj.clone());
        }
        layer_linears.push(format!("{lprefix}.self_attn.o_proj.weight"));
        layer_linears.push(format!("{lprefix}.mlp.gate_proj.weight"));
        layer_linears.push(format!("{lprefix}.mlp.up_proj.weight"));
        layer_linears.push(format!("{lprefix}.mlp.down_proj.weight"));
    }

    let any_layer_fp8 = layer_linears.iter().any(|name| {
        tensors
            .get(name)
            .is_some_and(|info| info.dtype == DType::Fp8E4M3)
    });
    for name in &layer_linears {
        let info = tensors
            .get(name)
            .ok_or_else(|| missing_tensor_error(model_dir, name))?;
        if any_layer_fp8 && !fp8_prequant {
            return Err(corrupt_error(
                model_dir.join("config.json"),
                "Gemma4 dry-run FP8 prequant mode is detected from layer 0 q_proj; mixed later FP8 linears are unsupported",
            ));
        }
        if fp8_prequant && info.dtype != DType::Fp8E4M3 {
            return Err(corrupt_error(
                model_dir.join("config.json"),
                format!(
                    "Gemma4 dry-run FP8 prequant mode requires all layer linear weights to be F8_E4M3; {name} is {:?}",
                    info.dtype
                ),
            ));
        }
        if info.dtype == DType::Fp8E4M3 {
            validate_gemma4_dry_run_fp8_scale(model_dir, tensors, name, info.shape[0])?;
        }
    }

    if let Some(name) = lm_head {
        let info = tensors
            .get(name)
            .ok_or_else(|| missing_tensor_error(model_dir, name))?;
        if info.dtype == DType::Fp8E4M3 {
            validate_gemma4_dry_run_fp8_scale(model_dir, tensors, name, info.shape[0])?;
        }
    }

    Ok(())
}

fn validate_gemma4_dry_run_fp8_scale(
    model_dir: &Path,
    tensors: &BTreeMap<String, DryRunTensorInfo>,
    weight_name: &str,
    rows: usize,
) -> Result<()> {
    let scale_name = format!("{weight_name}_scale");
    let scale = tensors
        .get(&scale_name)
        .ok_or_else(|| missing_tensor_error(model_dir, &scale_name))?;
    if scale.dtype != DType::Bf16 {
        return Err(corrupt_error(
            model_dir,
            format!(
                "{scale_name}: FP8 scale tensor must be BF16, got {:?}",
                scale.dtype
            ),
        ));
    }
    let valid_per_channel = scale.shape.as_slice() == [rows]
        || (scale.shape.len() == 2 && scale.shape[0] == rows && scale.shape[1] == 1);
    let valid_blockscale = scale.shape.len() == 2
        && scale.shape[0] > 0
        && scale.shape[1] > 0
        && scale.shape[0].saturating_mul(128) >= rows;
    if !valid_per_channel && !valid_blockscale {
        return Err(shape_mismatch_error(
            model_dir,
            &scale_name,
            &[rows],
            &scale.shape,
        ));
    }
    Ok(())
}

fn validate_required_shape(
    model_dir: &Path,
    tensors: &BTreeMap<String, DryRunTensorInfo>,
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

fn resolve_required_dry_run_alias(
    model_dir: &Path,
    tensors: &BTreeMap<String, DryRunTensorInfo>,
    candidates: Vec<String>,
) -> Result<String> {
    candidates
        .iter()
        .find(|name| tensors.contains_key(*name))
        .cloned()
        .ok_or_else(|| missing_tensor_error(model_dir, &candidates[0]))
}

fn resolve_optional_dry_run_alias(
    tensors: &BTreeMap<String, DryRunTensorInfo>,
    candidates: Vec<String>,
) -> Option<String> {
    candidates
        .into_iter()
        .find(|name| tensors.contains_key(name))
}

fn resolve_dry_run_weight_prefix(tensors: &BTreeMap<String, DryRunTensorInfo>) -> String {
    for prefix in ["model.language_model", "model", "language_model.model", ""] {
        if tensors.contains_key(&join_weight_name(prefix, "embed_tokens.weight")) {
            return prefix.to_owned();
        }
    }
    "model".to_owned()
}

fn join_weight_name(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_owned()
    } else {
        format!("{prefix}.{suffix}")
    }
}

fn validate_gemma4_dry_run_config_identity(model_dir: &Path) -> Result<()> {
    let path = model_dir.join("config.json");
    let bytes = std::fs::read(&path).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: path.clone(),
        source,
    })?;
    let config: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| corrupt_error(&path, format!("config.json: {e}")))?;
    let text_config = config
        .get("text_config")
        .unwrap_or(&serde_json::Value::Null);

    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .ok_or_else(|| corrupt_error(&path, "Gemma4 dry-run requires architectures"))?;
    let is_gemma4 = architectures
        .iter()
        .filter_map(|value| value.as_str())
        .any(|name| matches!(name, "Gemma4ForConditionalGeneration" | "Gemma4ForCausalLM"));
    if !is_gemma4 {
        return Err(corrupt_error(
            &path,
            "Gemma4 dry-run requires Gemma4 architecture",
        ));
    }

    for (scope, value) in [
        ("model_type", &config),
        ("text_config.model_type", text_config),
    ] {
        if let Some(model_type) = value.get("model_type").and_then(|value| value.as_str()) {
            if !matches!(model_type, "gemma4" | "gemma4_text") {
                return Err(corrupt_error(
                    &path,
                    format!("Gemma4 dry-run requires Gemma4 model_type at {scope}"),
                ));
            }
        }
    }

    for (scope, value) in [("root", &config), ("text_config", text_config)] {
        for field in [
            "enable_moe_block",
            "num_experts",
            "top_k_experts",
            "expert_intermediate_size",
            "moe_intermediate_size",
            "num_local_experts",
            "num_experts_per_tok",
            "router_aux_loss_coef",
        ] {
            if config_value_is_truthy(value.get(field)) {
                return Err(corrupt_error(
                    &path,
                    format!(
                        "Gemma4 dry-run supports dense Gemma4 only; unsupported MoE marker {scope}.{field}"
                    ),
                ));
            }
        }
    }

    Ok(())
}

fn config_value_is_truthy(value: Option<&serde_json::Value>) -> bool {
    match value {
        Some(serde_json::Value::Bool(value)) => *value,
        Some(serde_json::Value::Number(value)) => value.as_f64().is_some_and(|n| n != 0.0),
        Some(serde_json::Value::String(value)) => !value.is_empty() && value != "0",
        Some(serde_json::Value::Array(value)) => !value.is_empty(),
        Some(serde_json::Value::Object(value)) => !value.is_empty(),
        Some(serde_json::Value::Null) | None => false,
    }
}

fn scan_safetensor_tensor_metadata(model_dir: &Path) -> Result<BTreeMap<String, DryRunTensorInfo>> {
    let index = ShardIndex::resolve(model_dir)?;
    let mut tensors = BTreeMap::new();
    for shard in index.shards {
        for (name, info) in parse_safetensor_metadata_file(&shard)? {
            if tensors.insert(name.clone(), info).is_some() {
                return Err(corrupt_error(
                    shard,
                    format!("duplicate tensor name in safetensor files: {name}"),
                ));
            }
        }
    }
    Ok(tensors)
}

fn parse_safetensor_metadata_file(path: &Path) -> Result<Vec<(String, DryRunTensorInfo)>> {
    let mut file = File::open(path).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?
        .len() as usize;

    let mut header_len = [0u8; 8];
    file.read_exact(&mut header_len)
        .map_err(|_| corrupt_error(path, "safetensor file shorter than 8-byte prefix"))?;
    let header_bytes = u64::from_le_bytes(header_len) as usize;
    let payload_start = 8usize + header_bytes;
    if payload_start > file_len {
        return Err(corrupt_error(
            path,
            format!("safetensor header claims {header_bytes} bytes but file is only {file_len}"),
        ));
    }

    let mut header = vec![0u8; header_bytes];
    file.read_exact(&mut header)
        .map_err(|_| corrupt_error(path, "safetensor header truncated"))?;
    let header_str = std::str::from_utf8(&header)
        .map_err(|_| corrupt_error(path, "safetensor header is not valid utf-8"))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| corrupt_error(path, format!("safetensor header json: {e}")))?;

    let mut out = Vec::new();
    for (name, meta) in header.into_iter() {
        if name == "__metadata__" {
            continue;
        }
        let obj = meta
            .as_object()
            .ok_or_else(|| corrupt_error(path, format!("{name}: meta not an object")))?;
        let dtype_str = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| corrupt_error(path, format!("{name}: missing dtype")))?;
        let dtype = map_dtype(dtype_str)
            .ok_or_else(|| corrupt_error(path, format!("{name}: unsupported dtype {dtype_str}")))?;
        let shape: Vec<usize> = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| corrupt_error(path, format!("{name}: missing shape")))?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| corrupt_error(path, format!("{name}: bad shape element")))?;
        let offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| corrupt_error(path, format!("{name}: missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(corrupt_error(
                path,
                format!("{name}: expected 2 offsets got {}", offsets.len()),
            ));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| corrupt_error(path, format!("{name}: bad start offset")))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| corrupt_error(path, format!("{name}: bad end offset")))?
            as usize;
        if end < start {
            return Err(corrupt_error(
                path,
                format!("{name}: end offset precedes start offset"),
            ));
        }
        let nbytes = end - start;
        let expected = dtype_bytes(dtype) * shape.iter().copied().product::<usize>();
        if expected != nbytes {
            return Err(corrupt_error(
                path,
                format!("{name}: offset range {nbytes} != dtype*shape {expected}"),
            ));
        }
        if payload_start + end > file_len {
            return Err(corrupt_error(
                path,
                format!("{name}: data offsets exceed file length"),
            ));
        }
        out.push((name, DryRunTensorInfo { shape, dtype }));
    }
    Ok(out)
}

fn map_dtype(s: &str) -> Option<DType> {
    Some(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::Bf16,
        "F8_E4M3" | "F8E4M3" => DType::Fp8E4M3,
        _ => return None,
    })
}

fn dtype_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F16 | DType::Bf16 => 2,
        DType::Fp8E4M3 => 1,
        _ => 0,
    }
}

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

fn corrupt_error(path: impl Into<PathBuf>, detail: impl Into<String>) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt {
            detail: detail.into(),
        },
        ctx: LoaderCtx {
            path: path.into(),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, Value};
    use std::{
        fs::{self, File},
        io::Write,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn test_fixture_dir(name: &str) -> PathBuf {
        let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rvllm-loader-{name}-{}-{}-{id}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create fixture dir");
        dir
    }

    fn add_zero_tensor(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
    ) {
        add_zero_tensor_with_dtype(header, payload, name, shape, "F16");
    }

    fn add_zero_tensor_with_dtype(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
        dtype: &str,
    ) {
        let start = payload.len();
        let bytes_per_elem = match dtype {
            "F32" => 4,
            "F16" | "BF16" => 2,
            "F8_E4M3" => 1,
            other => panic!("unsupported fixture dtype {other}"),
        };
        let nbytes = shape.iter().copied().product::<usize>() * bytes_per_elem;
        payload.resize(start + nbytes, 0);
        let mut meta = Map::new();
        meta.insert("dtype".to_owned(), Value::String(dtype.to_owned()));
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
                Value::Number(((start + nbytes) as u64).into()),
            ]),
        );
        header.insert(name.to_owned(), Value::Object(meta));
    }

    fn mutate_fixture_config(dir: &Path, f: impl FnOnce(&mut Value)) {
        let path = dir.join("config.json");
        let mut config: Value =
            serde_json::from_slice(&fs::read(&path).expect("read config")).expect("parse config");
        f(&mut config);
        fs::write(
            path,
            serde_json::to_vec_pretty(&config).expect("serialize config"),
        )
        .expect("write config");
    }

    fn write_dry_run_full_gemma_style_fixture(
        tie_embeddings: bool,
        attention_k_eq_v: bool,
        omit_lm_head: bool,
        omit_v_proj_layer: Option<usize>,
        q_proj0_shape: Option<&[usize]>,
        q_proj1_shape: Option<&[usize]>,
        omit_q_norm0: bool,
        layer_scalar_suffix: Option<&str>,
    ) -> PathBuf {
        let dir = test_fixture_dir("gemma4-dry-run-full-gemma-style");
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
            if omit_v_proj_layer != Some(layer_idx) {
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
            if let Some(suffix) = layer_scalar_suffix {
                add_zero_tensor(
                    &mut header,
                    &mut payload,
                    &format!("{lprefix}.{suffix}"),
                    &[hidden],
                );
            }
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

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fp8ScaleFixture {
        PerChannel,
        PerChannelColumn,
        Block,
        BadRows,
        BadBlockRows,
        F16PerChannel,
        Missing,
    }

    fn add_fp8_scale_for_fixture(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        weight_name: &str,
        rows: usize,
        scale_override: Option<(&str, Fp8ScaleFixture)>,
    ) {
        let spec = scale_override
            .filter(|(suffix, _)| weight_name.ends_with(suffix))
            .map(|(_, spec)| spec)
            .unwrap_or(Fp8ScaleFixture::PerChannel);
        if spec == Fp8ScaleFixture::Missing {
            return;
        }

        let (shape, dtype) = match spec {
            Fp8ScaleFixture::PerChannel => (vec![rows], "BF16"),
            Fp8ScaleFixture::PerChannelColumn => (vec![rows, 1], "BF16"),
            Fp8ScaleFixture::Block => (vec![1, 1], "BF16"),
            Fp8ScaleFixture::BadRows => (vec![rows + 1], "BF16"),
            Fp8ScaleFixture::BadBlockRows => (vec![0, 1], "BF16"),
            Fp8ScaleFixture::F16PerChannel => (vec![rows], "F16"),
            Fp8ScaleFixture::Missing => unreachable!(),
        };
        add_zero_tensor_with_dtype(
            header,
            payload,
            &format!("{weight_name}_scale"),
            &shape,
            dtype,
        );
    }

    fn add_fp8_fixture_linear(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
        scale_override: Option<(&str, Fp8ScaleFixture)>,
        mixed_f16_linear_suffix: Option<&str>,
    ) {
        let dtype = if mixed_f16_linear_suffix.is_some_and(|suffix| name.ends_with(suffix)) {
            "F16"
        } else {
            "F8_E4M3"
        };
        add_zero_tensor_with_dtype(header, payload, name, shape, dtype);
        if dtype == "F8_E4M3" {
            add_fp8_scale_for_fixture(header, payload, name, shape[0], scale_override);
        }
    }

    fn write_dry_run_one_layer_fp8_fixture(
        scale_override: Option<(&str, Fp8ScaleFixture)>,
        mixed_f16_linear_suffix: Option<&str>,
        omit_v_proj: bool,
        attention_k_eq_v: bool,
        tie_embeddings: bool,
        lm_head_dtype: Option<&str>,
    ) -> PathBuf {
        let dir = test_fixture_dir("gemma4-dry-run-one-layer-fp8");
        let hidden = 8usize;
        let intermediate = 16usize;
        let vocab = 8usize;
        let head_dim = 4usize;
        let prefix = "model.language_model";
        let lprefix = format!("{prefix}.layers.0");

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
        if let Some(dtype) = lm_head_dtype {
            let name = format!("{prefix}.lm_head.weight");
            add_zero_tensor_with_dtype(&mut header, &mut payload, &name, &[vocab, hidden], dtype);
            if dtype == "F8_E4M3" {
                add_fp8_scale_for_fixture(&mut header, &mut payload, &name, vocab, scale_override);
            }
        }

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

        add_fp8_fixture_linear(
            &mut header,
            &mut payload,
            &format!("{lprefix}.self_attn.q_proj.weight"),
            &[head_dim, hidden],
            scale_override,
            mixed_f16_linear_suffix,
        );
        add_fp8_fixture_linear(
            &mut header,
            &mut payload,
            &format!("{lprefix}.self_attn.k_proj.weight"),
            &[head_dim, hidden],
            scale_override,
            mixed_f16_linear_suffix,
        );
        if !omit_v_proj {
            add_fp8_fixture_linear(
                &mut header,
                &mut payload,
                &format!("{lprefix}.self_attn.v_proj.weight"),
                &[head_dim, hidden],
                scale_override,
                mixed_f16_linear_suffix,
            );
        }
        add_fp8_fixture_linear(
            &mut header,
            &mut payload,
            &format!("{lprefix}.self_attn.o_proj.weight"),
            &[hidden, head_dim],
            scale_override,
            mixed_f16_linear_suffix,
        );
        add_zero_tensor(
            &mut header,
            &mut payload,
            &format!("{lprefix}.self_attn.q_norm.weight"),
            &[head_dim],
        );
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
        add_fp8_fixture_linear(
            &mut header,
            &mut payload,
            &format!("{lprefix}.mlp.gate_proj.weight"),
            &[intermediate, hidden],
            scale_override,
            mixed_f16_linear_suffix,
        );
        add_fp8_fixture_linear(
            &mut header,
            &mut payload,
            &format!("{lprefix}.mlp.up_proj.weight"),
            &[intermediate, hidden],
            scale_override,
            mixed_f16_linear_suffix,
        );
        add_fp8_fixture_linear(
            &mut header,
            &mut payload,
            &format!("{lprefix}.mlp.down_proj.weight"),
            &[hidden, intermediate],
            scale_override,
            mixed_f16_linear_suffix,
        );

        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "text_config": {{
    "num_hidden_layers": 1,
    "hidden_size": {hidden},
    "intermediate_size": {intermediate},
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": {head_dim},
    "global_head_dim": {head_dim},
    "num_global_key_value_heads": 1,
    "layer_types": ["full_attention"],
    "vocab_size": {vocab},
    "max_position_embeddings": 64,
    "sliding_window": 16,
    "rms_norm_eps": 0.000001,
    "tie_word_embeddings": {tie_embeddings},
    "attention_k_eq_v": {attention_k_eq_v},
    "rope_parameters": {{
      "sliding_attention": {{"rope_theta": 10000.0}},
      "full_attention": {{"rope_theta": 1000000.0, "partial_rotary_factor": 1.0}}
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
    fn gemma4_dry_run_validates_text_config_and_model_language_model_prefix() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        let validation =
            Gemma4DryRunValidation::from_model_dir(&dir).expect("dry-run validates fixture");

        assert_eq!(validation.weight_prefix, "model.language_model");
        assert_eq!(validation.num_layers, 2);
        assert_eq!(validation.hidden_size, 128);
        assert_eq!(validation.vocab_size, 8);
        assert_eq!(validation.final_logit_softcap, Some(30.0));
        assert_eq!(
            validation.layers[0].attention_kind,
            Gemma4DryRunAttentionKind::Sliding
        );
        assert_eq!(validation.layers[0].sliding_window, Some(32));
        assert_eq!(
            validation.layers[1].attention_kind,
            Gemma4DryRunAttentionKind::Full
        );
        assert_eq!(validation.layers[1].rope_dim, 64);
        assert_eq!(validation.layers[1].rope_theta, 1000000.0);
        assert_eq!(validation.layers[0].layer_scalar_dim, 128);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_accepts_generated_causallm_identity() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        mutate_fixture_config(&dir, |config| {
            config["architectures"] = serde_json::json!(["Gemma4ForCausalLM"]);
        });

        Gemma4DryRunValidation::from_model_dir(&dir)
            .expect("Gemma4ForCausalLM identity should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_non_gemma4_identity() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        mutate_fixture_config(&dir, |config| {
            config["architectures"] = serde_json::json!(["LlamaForCausalLM"]);
        });
        let err =
            Gemma4DryRunValidation::from_model_dir(&dir).expect_err("non-Gemma4 identity fails");
        let msg = format!("{err}");

        assert!(msg.contains("Corrupt"));
        assert!(msg.contains("Gemma4 architecture"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_allows_dense_moe_placeholders() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        mutate_fixture_config(&dir, |config| {
            config["model_type"] = serde_json::json!("gemma4");
            config["text_config"]["model_type"] = serde_json::json!("gemma4_text");
            config["text_config"]["enable_moe_block"] = serde_json::json!(false);
            config["text_config"]["num_experts"] = Value::Null;
            config["text_config"]["top_k_experts"] = Value::Null;
            config["text_config"]["expert_intermediate_size"] = Value::Null;
        });

        Gemma4DryRunValidation::from_model_dir(&dir)
            .expect("false/null MoE placeholders should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_explicit_moe_markers() {
        for (field, value) in [
            ("enable_moe_block", serde_json::json!(true)),
            ("num_experts", serde_json::json!(128)),
            ("top_k_experts", serde_json::json!(8)),
        ] {
            let dir = write_dry_run_full_gemma_style_fixture(
                false,
                false,
                false,
                None,
                None,
                None,
                false,
                Some("layer_scalar"),
            );
            mutate_fixture_config(&dir, |config| {
                config["text_config"][field] = value;
            });
            let err =
                Gemma4DryRunValidation::from_model_dir(&dir).expect_err("MoE marker must fail");
            let msg = format!("{err}");

            assert!(msg.contains("Corrupt"));
            assert!(msg.contains("dense Gemma4 only"));
            assert!(msg.contains(field));

            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn gemma4_dry_run_accepts_layer_scalar_weight_alias() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            None,
            None,
            false,
            Some("layer_scalar.weight"),
        );
        let validation =
            Gemma4DryRunValidation::from_model_dir(&dir).expect("layer_scalar.weight validates");

        assert_eq!(
            validation.layers[0].layer_scalar.as_deref(),
            Some("model.language_model.layers.0.layer_scalar.weight")
        );
        assert_eq!(validation.layers[0].layer_scalar_dim, 128);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_missing_layer_scalar() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false, false, false, None, None, None, false, None,
        );
        let err =
            Gemma4DryRunValidation::from_model_dir(&dir).expect_err("layer_scalar is required");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.layer_scalar"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_allows_tied_embeddings_without_lm_head() {
        let dir = write_dry_run_full_gemma_style_fixture(
            true,
            false,
            true,
            None,
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        let validation = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect("tied embeddings do not require lm_head");

        assert!(validation.tie_word_embeddings);
        assert_eq!(validation.lm_head, None);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_missing_lm_head_when_embeddings_are_not_tied() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            true,
            None,
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("untied embeddings require lm_head");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.lm_head.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_allows_missing_v_proj_when_attention_k_eq_v() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            true,
            false,
            Some(1),
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        let validation = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect("attention_k_eq_v permits missing v_proj");

        assert!(validation.layers[1].v_uses_k_proj);
        assert_eq!(validation.layers[1].v_proj, None);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_missing_sliding_v_proj_when_attention_k_eq_v() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            true,
            false,
            Some(0),
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("sliding attention requires v_proj");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.v_proj.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_missing_tensor_error_names_missing_tensor() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            Some(0),
            None,
            None,
            false,
            Some("layer_scalar"),
        );
        let err =
            Gemma4DryRunValidation::from_model_dir(&dir).expect_err("missing v_proj must fail");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.v_proj.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_shape_mismatch_error_names_tensor_and_expected_actual_shapes() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            Some(&[127, 128]),
            None,
            false,
            Some("layer_scalar"),
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("q_proj shape mismatch must fail");
        let msg = format!("{err}");

        assert!(msg.contains("ShapeMismatch"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(msg.contains("expected: [128, 128]"));
        assert!(msg.contains("got: [127, 128]"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_missing_q_norm_error_names_missing_tensor() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            None,
            None,
            true,
            Some("layer_scalar"),
        );
        let err =
            Gemma4DryRunValidation::from_model_dir(&dir).expect_err("missing q_norm must fail");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_norm.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_global_layer_shape_mismatch_uses_global_head_dim() {
        let dir = write_dry_run_full_gemma_style_fixture(
            false,
            false,
            false,
            None,
            None,
            Some(&[255, 128]),
            false,
            Some("layer_scalar"),
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
    fn gemma4_dry_run_validates_fp8_linears_with_bf16_scales() {
        let dir = write_dry_run_one_layer_fp8_fixture(None, None, false, false, true, None);
        let validation = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect("FP8 linears with BF16 scales validate");

        assert_eq!(validation.num_layers, 1);
        assert_eq!(
            validation.layers[0].attention_kind,
            Gemma4DryRunAttentionKind::Full
        );
        assert_eq!(
            validation.layers[0].q_proj,
            "model.language_model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(validation.lm_head, None);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_validates_fp8_row_column_scale_shape() {
        let dir = write_dry_run_one_layer_fp8_fixture(
            Some(("q_proj.weight", Fp8ScaleFixture::PerChannelColumn)),
            None,
            false,
            false,
            true,
            None,
        );

        Gemma4DryRunValidation::from_model_dir(&dir).expect("FP8 row-column scale shape validates");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_validates_fp8_blockscale_shape() {
        let dir = write_dry_run_one_layer_fp8_fixture(
            Some(("gate_proj.weight", Fp8ScaleFixture::Block)),
            None,
            false,
            false,
            true,
            None,
        );

        Gemma4DryRunValidation::from_model_dir(&dir).expect("FP8 blockscale shape validates");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_allows_fp8_missing_global_v_proj_when_attention_k_eq_v() {
        let dir = write_dry_run_one_layer_fp8_fixture(None, None, true, true, true, None);
        let validation = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect("attention_k_eq_v full layer can reuse k_proj as v_proj");

        assert!(validation.layers[0].v_uses_k_proj);
        assert_eq!(validation.layers[0].v_proj, None);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_fp8_linear_missing_scale() {
        let dir = write_dry_run_one_layer_fp8_fixture(
            Some(("q_proj.weight", Fp8ScaleFixture::Missing)),
            None,
            false,
            false,
            true,
            None,
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("FP8 q_proj missing scale must fail");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_proj.weight_scale"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_fp8_linear_f16_scale() {
        let dir = write_dry_run_one_layer_fp8_fixture(
            Some(("q_proj.weight", Fp8ScaleFixture::F16PerChannel)),
            None,
            false,
            false,
            true,
            None,
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("FP8 q_proj F16 scale must fail");
        let msg = format!("{err}");

        assert!(msg.contains("Corrupt"));
        assert!(msg.contains("FP8 scale tensor must be BF16"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_proj.weight_scale"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_fp8_linear_bad_scale_shape() {
        let dir = write_dry_run_one_layer_fp8_fixture(
            Some(("q_proj.weight", Fp8ScaleFixture::BadRows)),
            None,
            false,
            false,
            true,
            None,
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("FP8 q_proj bad scale shape must fail");
        let msg = format!("{err}");

        assert!(msg.contains("ShapeMismatch"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.q_proj.weight_scale"));
        assert!(msg.contains("expected: [4]"));
        assert!(msg.contains("got: [5]"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_fp8_linear_bad_blockscale_shape() {
        let dir = write_dry_run_one_layer_fp8_fixture(
            Some(("gate_proj.weight", Fp8ScaleFixture::BadBlockRows)),
            None,
            false,
            false,
            true,
            None,
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("FP8 gate_proj bad blockscale shape must fail");
        let msg = format!("{err}");

        assert!(msg.contains("ShapeMismatch"));
        assert!(msg.contains("model.language_model.layers.0.mlp.gate_proj.weight_scale"));
        assert!(msg.contains("got: [0, 1]"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_mixed_fp8_and_f16_layer_linears() {
        let dir = write_dry_run_one_layer_fp8_fixture(
            None,
            Some("k_proj.weight"),
            false,
            false,
            true,
            None,
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("mixed FP8/F16 layer linears must fail");
        let msg = format!("{err}");

        assert!(msg.contains("Corrupt"));
        assert!(msg.contains("requires all layer linear weights to be F8_E4M3"));
        assert!(msg.contains("model.language_model.layers.0.self_attn.k_proj.weight"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_rejects_fp8_lm_head_missing_scale() {
        let dir = write_dry_run_one_layer_fp8_fixture(
            Some(("lm_head.weight", Fp8ScaleFixture::Missing)),
            None,
            false,
            false,
            false,
            Some("F8_E4M3"),
        );
        let err = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect_err("FP8 lm_head missing scale must fail");
        let msg = format!("{err}");

        assert!(msg.contains("MissingTensor"));
        assert!(msg.contains("model.language_model.lm_head.weight_scale"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_accepts_non_fp8_lm_head_without_scale() {
        let dir = write_dry_run_one_layer_fp8_fixture(None, None, false, false, false, Some("F16"));
        let validation = Gemma4DryRunValidation::from_model_dir(&dir)
            .expect("non-FP8 lm_head does not require scale");

        assert_eq!(
            validation.lm_head.as_deref(),
            Some("model.language_model.lm_head.weight")
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gemma4_dry_run_real_model_dir_validates_when_env_is_set() {
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
            "validated Gemma4 dry-run metadata: dir={} prefix={} layers={} hidden={} vocab={}",
            model_dir.display(),
            validation.weight_prefix,
            validation.num_layers,
            validation.hidden_size,
            validation.vocab_size
        );
    }
}
