//! Model-architecture config, parsed from HF `config.json`.

use std::path::Path;

use crate::dtype::DType;
use crate::error::{ConfigError, Result, RvllmError};

use super::hf;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ModelArch {
    Qwen2,
    Llama,
    Mistral,
    Gemma2,
    Gemma4,
}

impl ModelArch {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "Qwen2ForCausalLM" => Some(ModelArch::Qwen2),
            "LlamaForCausalLM" => Some(ModelArch::Llama),
            "MistralForCausalLM" => Some(ModelArch::Mistral),
            "Gemma2ForCausalLM" => Some(ModelArch::Gemma2),
            name if is_gemma4_hf_architecture(name) => Some(ModelArch::Gemma4),
            _ => None,
        }
    }
}

pub fn is_gemma4_hf_architecture(name: &str) -> bool {
    matches!(name, "Gemma4ForConditionalGeneration" | "Gemma4ForCausalLM")
}

pub fn is_gemma4_model_type(name: &str) -> bool {
    matches!(name, "gemma4" | "gemma4_text")
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub architecture: ModelArch,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: DType,
}

impl ModelConfig {
    /// Parse an HF `config.json`. Every referenced field is required.
    pub fn load_hf(dir: &Path) -> Result<Self> {
        let file = dir.join("config.json");
        let body = std::fs::read_to_string(&file).map_err(|source| RvllmError::Io {
            err: crate::error::IoError::from(&source),
            path: file.clone(),
            source,
        })?;
        let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("config.json is not valid JSON: {e}")],
                },
                "config.json",
            )
        })?;
        Self::from_hf_value(&v, &file)
    }

    fn from_hf_value(v: &serde_json::Value, file: &Path) -> Result<Self> {
        let arch_name = hf::str_field(v, "architectures.0", file)?;
        let architecture = ModelArch::parse(&arch_name).ok_or_else(|| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "architectures[0]",
                    reason: format!("unsupported architecture: {arch_name}"),
                },
                "architectures[0]",
            )
        })?;

        // Gemma 3/4: text model fields nested under text_config.
        let tc = if v["text_config"]["hidden_size"].is_u64() {
            &v["text_config"]
        } else {
            v
        };
        if architecture == ModelArch::Gemma4 {
            validate_optional_gemma4_model_type(v, "model_type")?;
            validate_optional_gemma4_model_type(tc, "text_config.model_type")?;
        }

        let hidden_size = hf::usize_field(tc, "hidden_size", file)?;
        let num_layers = hf::usize_field(tc, "num_hidden_layers", file)?;
        let num_attention_heads = hf::usize_field(tc, "num_attention_heads", file)?;
        let num_kv_heads = hf::usize_field(tc, "num_key_value_heads", file)?;
        let intermediate_size = hf::usize_field(tc, "intermediate_size", file)?;
        let vocab_size = hf::usize_field(tc, "vocab_size", file)?;
        let max_position_embeddings = hf::usize_field(tc, "max_position_embeddings", file)?;
        let rms_norm_eps = hf::f32_field(tc, "rms_norm_eps", file)?;
        let rope_theta = tc["rope_parameters"]["sliding_attention"]["rope_theta"]
            .as_f64()
            .map(|t| t as f32)
            .map(Ok)
            .unwrap_or_else(|| hf::f32_field(tc, "rope_theta", file))?;
        let tie_word_embeddings = hf::bool_field_opt(tc, "tie_word_embeddings")
            .or_else(|| hf::bool_field_opt(v, "tie_word_embeddings"))
            .unwrap_or(false);
        let torch_dtype = match hf::str_field(v, "torch_dtype", file)
            .or_else(|_| hf::str_field(tc, "dtype", file))?
            .as_str()
        {
            "float16" => DType::F16,
            "bfloat16" => DType::Bf16,
            other => {
                return Err(RvllmError::config(
                    ConfigError::InvalidField {
                        name: "torch_dtype",
                        reason: format!("unsupported torch_dtype: {other}"),
                    },
                    "torch_dtype",
                ));
            }
        };

        if num_attention_heads == 0 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "num_attention_heads",
                    reason: "must be > 0".into(),
                },
                "num_attention_heads",
            ));
        }
        // Gemma 4 has explicit head_dim (256) that doesn't equal hidden_size/num_heads.
        let head_dim = tc["head_dim"]
            .as_u64()
            .map(|d| d as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads);
        if tc["head_dim"].as_u64().is_none() && head_dim * num_attention_heads != hidden_size {
            return Err(RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!(
                        "hidden_size {hidden_size} not divisible by num_attention_heads {num_attention_heads}"
                    )],
                },
                "hidden_size",
            ));
        }

        Ok(Self {
            architecture,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            torch_dtype,
        })
    }
}

fn validate_optional_gemma4_model_type(
    value: &serde_json::Value,
    field_name: &'static str,
) -> Result<()> {
    if let Some(model_type) = value.get("model_type").and_then(|value| value.as_str()) {
        if !is_gemma4_model_type(model_type) {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: field_name,
                    reason: format!(
                        "Gemma4 config requires model_type gemma4 or gemma4_text, got {model_type}"
                    ),
                },
                field_name,
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn gemma4_config(architecture: &str) -> serde_json::Value {
        serde_json::json!({
            "architectures": [architecture],
            "torch_dtype": "bfloat16",
            "text_config": {
                "hidden_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "intermediate_size": 256,
                "vocab_size": 8,
                "max_position_embeddings": 128,
                "rms_norm_eps": 0.000001,
                "tie_word_embeddings": false,
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_theta": 10000.0
                    }
                }
            }
        })
    }

    #[test]
    fn parses_gemma4_conditional_generation_identity() {
        let config = ModelConfig::from_hf_value(
            &gemma4_config("Gemma4ForConditionalGeneration"),
            Path::new("config.json"),
        )
        .expect("Gemma4ForConditionalGeneration should parse");

        assert_eq!(config.architecture, ModelArch::Gemma4);
        assert_eq!(config.head_dim, 32);
    }

    #[test]
    fn parses_gemma4_causallm_identity() {
        let config = ModelConfig::from_hf_value(
            &gemma4_config("Gemma4ForCausalLM"),
            Path::new("config.json"),
        )
        .expect("Gemma4ForCausalLM should parse");

        assert_eq!(config.architecture, ModelArch::Gemma4);
        assert_eq!(config.num_kv_heads, 2);
    }

    #[test]
    fn parses_gemma4_model_type_fields() {
        let mut value = gemma4_config("Gemma4ForConditionalGeneration");
        value["model_type"] = serde_json::json!("gemma4");
        value["text_config"]["model_type"] = serde_json::json!("gemma4_text");

        let config = ModelConfig::from_hf_value(&value, Path::new("config.json"))
            .expect("Gemma4 model_type fields should parse");

        assert_eq!(config.architecture, ModelArch::Gemma4);
    }

    #[test]
    fn rejects_gemma4_bad_root_model_type() {
        let mut value = gemma4_config("Gemma4ForConditionalGeneration");
        value["model_type"] = serde_json::json!("qwen2");
        let err = ModelConfig::from_hf_value(&value, Path::new("config.json"))
            .expect_err("bad root model_type should fail");
        let msg = format!("{err}");

        assert!(msg.contains("model_type"));
        assert!(msg.contains("gemma4"));
    }

    #[test]
    fn rejects_gemma4_bad_text_model_type() {
        let mut value = gemma4_config("Gemma4ForCausalLM");
        value["text_config"]["model_type"] = serde_json::json!("qwen2");
        let err = ModelConfig::from_hf_value(&value, Path::new("config.json"))
            .expect_err("bad text_config model_type should fail");
        let msg = format!("{err}");

        assert!(msg.contains("text_config.model_type"));
        assert!(msg.contains("gemma4"));
    }
}
