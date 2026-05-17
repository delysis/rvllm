//! Gemma 4 model architecture parser.
//!
//! Parsed from the real google/gemma-4-31B-it config.json:
//! - `architectures: ["Gemma4ForConditionalGeneration"]`
//! - `text_config` sub-object for the language model params
//! - `head_dim: 256` (sliding), `global_head_dim: 512` (global)
//! - `num_key_value_heads: 16` (sliding), `num_global_key_value_heads: 4`
//! - `rope_parameters` is a nested object with per-type sub-configs
//! - `layer_types` array: 5 sliding + 1 global, repeating
//! - `tie_word_embeddings: true` (no separate lm_head.weight)
//! - `hidden_activation: "gelu_pytorch_tanh"`
//! - `final_logit_softcapping: 30.0`
//!
//! Actual Gemma 4 31B dimensions:
//!   hidden=5376, heads=32, layers=60, intermediate=21504, vocab=262144
//!   sliding: head_dim=256, kv_heads=16, theta=10000, full rotation
//!   global:  head_dim=512, kv_heads=4,  theta=1M, partial_rotary=0.25

use std::{io::Read, path::Path};

use rvllm_core::{
    config::{is_gemma4_hf_architecture, is_gemma4_model_type},
    LoaderCtx, LoaderError, Result, RvllmError,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Gemma4LayerType {
    SlidingAttention,
    GlobalAttention,
}

#[derive(Clone, Debug)]
pub struct Gemma4Arch {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim_sliding: usize,
    pub head_dim_global: usize,
    pub num_kv_heads_sliding: usize,
    pub num_kv_heads_global: usize,
    pub intermediate_size: usize,
    pub use_double_wide_mlp: bool,
    pub num_kv_shared_layers: usize,
    pub hidden_size_per_layer_input: usize,
    pub vocab_size_per_layer_input: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub sliding_window_size: usize,
    pub rope_theta_sliding: f32,
    pub rope_theta_global: f32,
    pub partial_rotary_factor_global: f32,
    pub logit_softcap: f32,
    pub layer_types: Vec<Gemma4LayerType>,
    pub weight_prefix: String,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
}

impl Gemma4Arch {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let bytes = std::fs::read(&p).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: p.clone(),
            source,
        })?;
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("config.json: {e}"),
                },
                ctx: LoaderCtx {
                    path: p.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?;
        validate_gemma4_config_identity(&v, &p)?;

        let tc = if v.get("text_config").is_some() {
            &v["text_config"]
        } else {
            &v
        };

        let num_hidden_layers = tc["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
        let hidden_size = tc["hidden_size"].as_u64().unwrap_or(0) as usize;
        let num_attention_heads = tc["num_attention_heads"].as_u64().unwrap_or(0) as usize;

        let head_dim_sliding = tc["head_dim"].as_u64().unwrap_or(256) as usize;
        let head_dim_global = tc["global_head_dim"].as_u64().unwrap_or(512) as usize;

        let intermediate_size = tc["intermediate_size"].as_u64().unwrap_or(0) as usize;
        let use_double_wide_mlp = tc["use_double_wide_mlp"]
            .as_bool()
            .or_else(|| v["use_double_wide_mlp"].as_bool())
            .unwrap_or(false);
        let num_kv_shared_layers = tc["num_kv_shared_layers"]
            .as_u64()
            .or_else(|| v["num_kv_shared_layers"].as_u64())
            .unwrap_or(0) as usize;
        let hidden_size_per_layer_input = tc["hidden_size_per_layer_input"]
            .as_u64()
            .or_else(|| v["hidden_size_per_layer_input"].as_u64())
            .unwrap_or(0) as usize;
        let vocab_size_per_layer_input = tc["vocab_size_per_layer_input"]
            .as_u64()
            .or_else(|| v["vocab_size_per_layer_input"].as_u64())
            .unwrap_or(0) as usize;
        let vocab_size = tc["vocab_size"]
            .as_u64()
            .or_else(|| v["vocab_size"].as_u64())
            .unwrap_or(0) as usize;
        let rms_norm_eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let max_position_embeddings =
            tc["max_position_embeddings"].as_u64().unwrap_or(262144) as usize;
        let sliding_window_size = tc["sliding_window"]
            .as_u64()
            .or_else(|| tc["sliding_window_size"].as_u64())
            .unwrap_or(1024) as usize;

        let num_kv_heads_sliding = tc["num_key_value_heads"]
            .as_u64()
            .unwrap_or(num_attention_heads as u64) as usize;
        let num_kv_heads_global = tc["num_global_key_value_heads"]
            .as_u64()
            .or_else(|| tc["num_key_value_heads_global"].as_u64())
            .unwrap_or(num_kv_heads_sliding as u64) as usize;

        // RoPE parameters -- nested per-type in Gemma 4
        let rope = &tc["rope_parameters"];
        let rope_theta_sliding = rope["sliding_attention"]["rope_theta"]
            .as_f64()
            .or_else(|| tc["rope_theta"].as_f64())
            .unwrap_or(10000.0) as f32;
        let rope_theta_global = rope["full_attention"]["rope_theta"]
            .as_f64()
            .or_else(|| tc["rope_theta_global"].as_f64())
            .unwrap_or(1_000_000.0) as f32;
        let partial_rotary_factor_global = rope["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .or_else(|| tc["partial_rotary_factor"].as_f64())
            .unwrap_or(0.25) as f32;

        let logit_softcap = tc["final_logit_softcapping"]
            .as_f64()
            .or_else(|| tc["logit_softcapping"].as_f64())
            .unwrap_or(30.0) as f32;

        let tie_word_embeddings = tc["tie_word_embeddings"]
            .as_bool()
            .or_else(|| v["tie_word_embeddings"].as_bool())
            .unwrap_or(true);
        let attention_k_eq_v = tc["attention_k_eq_v"].as_bool().unwrap_or(false);

        let layer_types = Self::parse_layer_types(tc, num_hidden_layers);
        let weight_prefix = Self::detect_weight_prefix(dir);

        if num_attention_heads == 0 || hidden_size == 0 || num_hidden_layers == 0 {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: "Gemma4 config has zero-valued required fields".into(),
                },
                ctx: LoaderCtx {
                    path: p,
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if use_double_wide_mlp
            && (num_kv_shared_layers == 0 || num_kv_shared_layers > num_hidden_layers)
        {
            return Err(RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: "Gemma4 double-wide MLP requires num_kv_shared_layers in 1..=num_hidden_layers".into(),
                },
                ctx: LoaderCtx {
                    path: p,
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        Ok(Self {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            head_dim_sliding,
            head_dim_global,
            num_kv_heads_sliding,
            num_kv_heads_global,
            intermediate_size,
            use_double_wide_mlp,
            num_kv_shared_layers,
            hidden_size_per_layer_input,
            vocab_size_per_layer_input,
            vocab_size,
            rms_norm_eps,
            max_position_embeddings,
            sliding_window_size,
            rope_theta_sliding,
            rope_theta_global,
            partial_rotary_factor_global,
            logit_softcap,
            layer_types,
            weight_prefix,
            tie_word_embeddings,
            attention_k_eq_v,
        })
    }

    fn parse_layer_types(tc: &serde_json::Value, n: usize) -> Vec<Gemma4LayerType> {
        if let Some(arr) = tc["layer_types"].as_array() {
            return arr
                .iter()
                .map(|t| match t.as_str().unwrap_or("sliding_attention") {
                    "global_attention" | "full_attention" => Gemma4LayerType::GlobalAttention,
                    _ => Gemma4LayerType::SlidingAttention,
                })
                .collect();
        }
        // Default: 5 sliding + 1 global repeating
        (0..n)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    Gemma4LayerType::GlobalAttention
                } else {
                    Gemma4LayerType::SlidingAttention
                }
            })
            .collect()
    }

    fn detect_weight_prefix(dir: &Path) -> String {
        let prefix_from_keys = |keys: Vec<String>| -> Option<String> {
            for prefix in [
                "model.language_model",
                "language_model.model",
                "language_model",
            ] {
                let prefix_dot = format!("{prefix}.");
                if keys.iter().any(|key| key.starts_with(&prefix_dot)) {
                    return Some(prefix.to_string());
                }
            }
            None
        };

        let idx_path = dir.join("model.safetensors.index.json");
        if let Ok(bytes) = std::fs::read(&idx_path) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if let Some(map) = v["weight_map"].as_object() {
                    if let Some(prefix) = prefix_from_keys(map.keys().cloned().collect()) {
                        return prefix;
                    }
                }
            }
        }
        let single_path = dir.join("model.safetensors");
        if let Ok(keys) = read_single_safetensor_header_keys(&single_path) {
            if let Some(prefix) = prefix_from_keys(keys) {
                return prefix;
            }
        }
        "model".to_string()
    }

    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.head_dim_sliding,
            Gemma4LayerType::GlobalAttention => self.head_dim_global,
        }
    }

    pub fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.num_kv_heads_sliding,
            Gemma4LayerType::GlobalAttention => self.num_kv_heads_global,
        }
    }

    pub fn rotary_dim_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            // Sliding: full rotation of head_dim_sliding (256)
            Gemma4LayerType::SlidingAttention => self.head_dim_sliding,
            // Global: partial rotation of head_dim_global (512 * 0.25 = 128)
            Gemma4LayerType::GlobalAttention => {
                let rd = (self.head_dim_global as f32 * self.partial_rotary_factor_global) as usize;
                (rd / 2) * 2 // ensure even
            }
        }
    }

    pub fn rope_theta_for_layer(&self, layer_idx: usize) -> f32 {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.rope_theta_sliding,
            Gemma4LayerType::GlobalAttention => self.rope_theta_global,
        }
    }

    pub fn q_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.num_attention_heads * self.head_dim_for_layer(layer_idx)
    }

    pub fn kv_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.num_kv_heads_for_layer(layer_idx) * self.head_dim_for_layer(layer_idx)
    }

    pub fn intermediate_size_for_layer(&self, layer_idx: usize) -> usize {
        if self.use_double_wide_mlp
            && self.num_kv_shared_layers > 0
            && layer_idx >= self.num_hidden_layers - self.num_kv_shared_layers
        {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }

    pub fn max_head_dim(&self) -> usize {
        self.head_dim_sliding.max(self.head_dim_global)
    }

    pub fn max_kv_heads(&self) -> usize {
        self.num_kv_heads_sliding.max(self.num_kv_heads_global)
    }

    pub fn max_q_dim(&self) -> usize {
        self.num_attention_heads * self.max_head_dim()
    }
}

fn read_single_safetensor_header_keys(path: &Path) -> Result<Vec<String>> {
    let mut file = std::fs::File::open(path).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    let mut header_len = [0u8; 8];
    file.read_exact(&mut header_len)
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
    let header_bytes = u64::from_le_bytes(header_len) as usize;
    let mut header = vec![0u8; header_bytes];
    file.read_exact(&mut header)
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .map_err(|e| RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("safetensor header json: {e}"),
            },
            ctx: LoaderCtx {
                path: path.to_path_buf(),
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        })?;
    Ok(header
        .into_iter()
        .map(|(key, _)| key)
        .filter(|key| key != "__metadata__")
        .collect())
}

fn validate_gemma4_config_identity(config: &serde_json::Value, path: &Path) -> Result<()> {
    let text_config = config
        .get("text_config")
        .unwrap_or(&serde_json::Value::Null);
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .ok_or_else(|| corrupt_error(path, "Gemma4 arch requires architectures"))?;
    let is_gemma4 = architectures
        .iter()
        .filter_map(|value| value.as_str())
        .any(is_gemma4_hf_architecture);
    if !is_gemma4 {
        return Err(corrupt_error(
            path,
            "Gemma4 arch requires Gemma4 architecture",
        ));
    }

    for (scope, value) in [
        ("model_type", config),
        ("text_config.model_type", text_config),
    ] {
        if let Some(model_type) = value.get("model_type").and_then(|value| value.as_str()) {
            if !is_gemma4_model_type(model_type) {
                return Err(corrupt_error(
                    path,
                    format!("Gemma4 arch requires Gemma4 model_type at {scope}"),
                ));
            }
        }
    }

    for (scope, value) in [("root", config), ("text_config", text_config)] {
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
                    path,
                    format!(
                        "Gemma4 arch supports dense Gemma4 only; unsupported MoE marker {scope}.{field}"
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

fn corrupt_error(path: &Path, detail: impl Into<String>) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt {
            detail: detail.into(),
        },
        ctx: LoaderCtx {
            path: path.to_path_buf(),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tempdir() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rvllm-loader-gemma4-arch-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_index(dir: &Path, keys: &[&str]) {
        let weight_map: serde_json::Map<String, serde_json::Value> = keys
            .iter()
            .map(|key| {
                (
                    (*key).to_string(),
                    serde_json::Value::String("model-00001-of-00001.safetensors".to_string()),
                )
            })
            .collect();
        let index = serde_json::json!({
            "metadata": {},
            "weight_map": weight_map,
        });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
    }

    fn write_single_safetensor_header(dir: &Path, keys: &[&str]) {
        let mut header = serde_json::Map::new();
        for key in keys {
            header.insert(
                (*key).to_string(),
                serde_json::json!({
                    "dtype": "F16",
                    "shape": [1],
                    "data_offsets": [0, 2],
                }),
            );
        }
        let header_json = serde_json::to_string(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(header_json.as_bytes());
        out.extend_from_slice(&[0u8, 0u8]);
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
    }

    fn write_minimal_config(dir: &Path, architecture: &str) {
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{
  "architectures": ["{architecture}"],
  "model_type": "gemma4",
  "text_config": {{
    "model_type": "gemma4_text",
    "num_hidden_layers": 1,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "global_head_dim": 32,
    "num_global_key_value_heads": 2,
    "intermediate_size": 256,
    "vocab_size": 8,
    "layer_types": ["full_attention"],
    "rms_norm_eps": 0.000001,
    "rope_parameters": {{
      "sliding_attention": {{"rope_theta": 10000.0}},
      "full_attention": {{"rope_theta": 1000000.0, "partial_rotary_factor": 1.0}}
    }}
  }}
}}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn default_layer_pattern_every_6th_global() {
        let types = Gemma4Arch::parse_layer_types(&serde_json::Value::Null, 12);
        // 0:s 1:s 2:s 3:s 4:s 5:g 6:s 7:s 8:s 9:s 10:s 11:g
        assert_eq!(types[0], Gemma4LayerType::SlidingAttention);
        assert_eq!(types[4], Gemma4LayerType::SlidingAttention);
        assert_eq!(types[5], Gemma4LayerType::GlobalAttention);
        assert_eq!(types[11], Gemma4LayerType::GlobalAttention);
    }

    #[test]
    fn parses_real_layer_types() {
        let v: serde_json::Value = serde_json::json!({
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ]
        });
        let types = Gemma4Arch::parse_layer_types(&v, 6);
        assert_eq!(types[5], Gemma4LayerType::GlobalAttention);
        assert_eq!(types[0], Gemma4LayerType::SlidingAttention);
    }

    #[test]
    fn detects_language_model_model_weight_prefix() {
        let dir = tempdir();
        write_index(
            &dir,
            &[
                "language_model.layers.0.self_attn.q_proj.weight",
                "language_model.model.embed_tokens.weight",
            ],
        );
        assert_eq!(
            Gemma4Arch::detect_weight_prefix(&dir),
            "language_model.model"
        );
    }

    #[test]
    fn detects_model_language_model_weight_prefix() {
        let dir = tempdir();
        write_index(&dir, &["model.language_model.embed_tokens.weight"]);
        assert_eq!(
            Gemma4Arch::detect_weight_prefix(&dir),
            "model.language_model"
        );
    }

    #[test]
    fn detects_language_model_weight_prefix() {
        let dir = tempdir();
        write_index(&dir, &["language_model.embed_tokens.weight"]);
        assert_eq!(Gemma4Arch::detect_weight_prefix(&dir), "language_model");
    }

    #[test]
    fn detects_model_language_model_prefix_from_single_safetensor() {
        let dir = tempdir();
        write_single_safetensor_header(&dir, &["model.language_model.embed_tokens.weight"]);
        assert_eq!(
            Gemma4Arch::detect_weight_prefix(&dir),
            "model.language_model"
        );
    }

    #[test]
    fn from_dir_accepts_gemma4_causallm_identity() {
        let dir = tempdir();
        write_minimal_config(&dir, "Gemma4ForCausalLM");
        write_single_safetensor_header(&dir, &["model.language_model.embed_tokens.weight"]);
        let arch = Gemma4Arch::from_dir(&dir).expect("Gemma4ForCausalLM identity should parse");

        assert_eq!(arch.weight_prefix, "model.language_model");
        assert_eq!(arch.q_dim_for_layer(0), 128);
    }

    #[test]
    fn from_dir_global_kv_heads_default_to_regular_kv_heads() {
        let dir = tempdir();
        write_minimal_config(&dir, "Gemma4ForConditionalGeneration");
        write_single_safetensor_header(&dir, &["model.language_model.embed_tokens.weight"]);
        let path = dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["text_config"]["num_attention_heads"] = serde_json::json!(8);
        config["text_config"]["num_key_value_heads"] = serde_json::json!(1);
        config["text_config"]["head_dim"] = serde_json::json!(256);
        config["text_config"]["global_head_dim"] = serde_json::json!(512);
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("num_global_key_value_heads");
        config["text_config"]["layer_types"] = serde_json::json!(["full_attention"]);
        std::fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let arch = Gemma4Arch::from_dir(&dir)
            .expect("missing global kv heads should fall back to regular kv heads");

        assert_eq!(arch.num_kv_heads_sliding, 1);
        assert_eq!(arch.num_kv_heads_global, 1);
        assert_eq!(arch.q_dim_for_layer(0), 4096);
        assert_eq!(arch.kv_dim_for_layer(0), 512);
    }

    #[test]
    fn from_dir_null_global_kv_heads_default_to_regular_kv_heads() {
        let dir = tempdir();
        write_minimal_config(&dir, "Gemma4ForConditionalGeneration");
        write_single_safetensor_header(&dir, &["model.language_model.embed_tokens.weight"]);
        let path = dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["text_config"]["num_attention_heads"] = serde_json::json!(8);
        config["text_config"]["num_key_value_heads"] = serde_json::json!(1);
        config["text_config"]["head_dim"] = serde_json::json!(256);
        config["text_config"]["global_head_dim"] = serde_json::json!(512);
        config["text_config"]["num_global_key_value_heads"] = serde_json::Value::Null;
        config["text_config"]["layer_types"] = serde_json::json!(["full_attention"]);
        std::fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let arch = Gemma4Arch::from_dir(&dir)
            .expect("null global kv heads should fall back to regular kv heads");

        assert_eq!(arch.num_kv_heads_sliding, 1);
        assert_eq!(arch.num_kv_heads_global, 1);
        assert_eq!(arch.q_dim_for_layer(0), 4096);
        assert_eq!(arch.kv_dim_for_layer(0), 512);
    }

    #[test]
    fn from_dir_explicit_global_kv_heads_override_regular_kv_heads() {
        let dir = tempdir();
        write_minimal_config(&dir, "Gemma4ForConditionalGeneration");
        write_single_safetensor_header(&dir, &["model.language_model.embed_tokens.weight"]);
        let path = dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["text_config"]["num_attention_heads"] = serde_json::json!(8);
        config["text_config"]["num_key_value_heads"] = serde_json::json!(1);
        config["text_config"]["num_global_key_value_heads"] = serde_json::json!(2);
        config["text_config"]["global_head_dim"] = serde_json::json!(512);
        config["text_config"]["layer_types"] = serde_json::json!(["full_attention"]);
        std::fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let arch =
            Gemma4Arch::from_dir(&dir).expect("explicit global kv heads should keep precedence");

        assert_eq!(arch.num_kv_heads_sliding, 1);
        assert_eq!(arch.num_kv_heads_global, 2);
        assert_eq!(arch.kv_dim_for_layer(0), 1024);
    }

    #[test]
    fn from_dir_parses_double_wide_mlp_tail_layers() {
        let dir = tempdir();
        write_minimal_config(&dir, "Gemma4ForConditionalGeneration");
        write_single_safetensor_header(&dir, &["model.language_model.embed_tokens.weight"]);
        let path = dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["text_config"]["num_hidden_layers"] = serde_json::json!(35);
        config["text_config"]["intermediate_size"] = serde_json::json!(6144);
        config["text_config"]["use_double_wide_mlp"] = serde_json::json!(true);
        config["text_config"]["num_kv_shared_layers"] = serde_json::json!(20);
        config["text_config"]["layer_types"] = serde_json::Value::Array(
            std::iter::repeat(serde_json::json!("sliding_attention"))
                .take(35)
                .collect(),
        );
        std::fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let arch = Gemma4Arch::from_dir(&dir).expect("double-wide MLP config should parse");

        assert_eq!(arch.intermediate_size_for_layer(14), 6144);
        assert_eq!(arch.intermediate_size_for_layer(15), 12288);
        assert_eq!(arch.intermediate_size_for_layer(34), 12288);
    }

    #[test]
    fn from_dir_rejects_invalid_double_wide_mlp_window() {
        let dir = tempdir();
        write_minimal_config(&dir, "Gemma4ForConditionalGeneration");
        let path = dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["text_config"]["use_double_wide_mlp"] = serde_json::json!(true);
        config["text_config"]["num_kv_shared_layers"] = serde_json::json!(0);
        std::fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let err = Gemma4Arch::from_dir(&dir).expect_err("invalid double-wide MLP should fail");
        let msg = format!("{err}");

        assert!(msg.contains("double-wide MLP"));
        assert!(msg.contains("num_kv_shared_layers"));
    }

    #[test]
    fn from_dir_rejects_bad_gemma4_model_type() {
        let dir = tempdir();
        write_minimal_config(&dir, "Gemma4ForConditionalGeneration");
        let path = dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["text_config"]["model_type"] = serde_json::json!("qwen2");
        std::fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
        let err = Gemma4Arch::from_dir(&dir).expect_err("bad model_type should fail");
        let msg = format!("{err}");

        assert!(msg.contains("Gemma4 model_type"));
        assert!(msg.contains("text_config.model_type"));
    }

    #[test]
    fn from_dir_allows_dense_moe_placeholders() {
        let dir = tempdir();
        write_minimal_config(&dir, "Gemma4ForConditionalGeneration");
        write_single_safetensor_header(&dir, &["model.language_model.embed_tokens.weight"]);
        let path = dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["enable_moe_block"] = serde_json::json!(false);
        config["text_config"]["num_experts"] = serde_json::Value::Null;
        config["text_config"]["top_k_experts"] = serde_json::Value::Null;
        config["text_config"]["expert_intermediate_size"] = serde_json::Value::Null;
        std::fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let arch = Gemma4Arch::from_dir(&dir).expect("false/null MoE placeholders should parse");
        assert_eq!(arch.weight_prefix, "model.language_model");
    }

    #[test]
    fn from_dir_rejects_explicit_moe_markers() {
        for (scope, field, value) in [
            ("root", "enable_moe_block", serde_json::json!(true)),
            ("text_config", "num_experts", serde_json::json!(128)),
            ("text_config", "top_k_experts", serde_json::json!(8)),
            (
                "text_config",
                "expert_intermediate_size",
                serde_json::json!(1024),
            ),
            (
                "text_config",
                "router_aux_loss_coef",
                serde_json::json!(0.01),
            ),
        ] {
            let dir = tempdir();
            write_minimal_config(&dir, "Gemma4ForConditionalGeneration");
            let path = dir.join("config.json");
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            if scope == "root" {
                config[field] = value;
            } else {
                config["text_config"][field] = value;
            }
            std::fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

            let err = Gemma4Arch::from_dir(&dir).expect_err("MoE marker should fail");
            let msg = format!("{err}");

            assert!(msg.contains("dense Gemma4 only"));
            assert!(msg.contains(scope));
            assert!(msg.contains(field));
        }
    }

    #[test]
    fn rotary_dim_sliding_is_full() {
        let arch = Gemma4Arch {
            num_hidden_layers: 6,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim_sliding: 256,
            head_dim_global: 512,
            num_kv_heads_sliding: 16,
            num_kv_heads_global: 4,
            intermediate_size: 21504,
            use_double_wide_mlp: false,
            num_kv_shared_layers: 0,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            sliding_window_size: 1024,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1000000.0,
            partial_rotary_factor_global: 0.25,
            logit_softcap: 30.0,
            layer_types: vec![Gemma4LayerType::SlidingAttention; 6],
            weight_prefix: "model".into(),
            tie_word_embeddings: true,
            attention_k_eq_v: false,
        };
        assert_eq!(arch.rotary_dim_for_layer(0), 256);
    }

    #[test]
    fn rotary_dim_global_is_partial() {
        let arch = Gemma4Arch {
            num_hidden_layers: 6,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim_sliding: 256,
            head_dim_global: 512,
            num_kv_heads_sliding: 16,
            num_kv_heads_global: 4,
            intermediate_size: 21504,
            use_double_wide_mlp: false,
            num_kv_shared_layers: 0,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            sliding_window_size: 1024,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1000000.0,
            partial_rotary_factor_global: 0.25,
            logit_softcap: 30.0,
            layer_types: vec![Gemma4LayerType::GlobalAttention; 6],
            weight_prefix: "model".into(),
            tie_word_embeddings: true,
            attention_k_eq_v: false,
        };
        // 512 * 0.25 = 128
        assert_eq!(arch.rotary_dim_for_layer(0), 128);
    }
}
