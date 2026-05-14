//! Weight loader: load safetensors model weights into Metal buffers.
//!
//! Handles BF16 → F16 conversion (Metal 3 / Apple9 has no native BF16
//! compute). Weights are loaded via mmap and converted in-place or
//! via the bf16_to_f16 Metal kernel.

use std::path::Path;

use crate::arena::{MetalBufferArena, MetalRegion};
use crate::context::MetalContext;
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};

fn ctx(op: &'static str) -> AppleCtx {
    AppleCtx { backend: "metal-loader", op, device: "apple-silicon" }
}

/// Describes a loaded model's weight layout in the arena.
#[derive(Clone, Debug)]
pub struct MetalModelWeights {
    pub layers: Vec<MetalLayerWeightRegions>,
    pub final_norm: MetalRegion,
    pub lm_head: MetalRegion,
    pub rope_cos: MetalRegion,
    pub rope_sin: MetalRegion,
}

#[derive(Clone, Debug)]
pub struct MetalLayerWeightRegions {
    pub attn_norm: MetalRegion,
    pub qkv: MetalRegion,
    pub qkv_bias: Option<MetalRegion>,
    pub o_proj: MetalRegion,
    pub mlp_norm: MetalRegion,
    pub gate_up: MetalRegion,
    pub down_proj: MetalRegion,
}

pub fn bf16_to_f16_cpu(bf16_data: &[u8]) -> Vec<u8> {
    let count = bf16_data.len() / 2;
    let mut f16_data = vec![0u8; count * 2];

    for i in 0..count {
        let bf16_bits = u16::from_le_bytes([bf16_data[i * 2], bf16_data[i * 2 + 1]]);
        let f32_bits = (bf16_bits as u32) << 16;
        let f32_val = f32::from_bits(f32_bits);
        let f16_val = half::f16::from_f32(f32_val);
        let f16_bits = f16_val.to_bits();
        f16_data[i * 2] = f16_bits as u8;
        f16_data[i * 2 + 1] = (f16_bits >> 8) as u8;
    }
    f16_data
}

pub fn parse_safetensors_index(
    model_dir: &Path,
) -> Result<Vec<(String, std::path::PathBuf)>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![("model.safetensors".to_owned(), single)]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(RvllmError::apple(
            AppleError::MetallibMissing { path: index_path },
            ctx("parse_index"),
        ));
    }

    let index_bytes = std::fs::read(&index_path).map_err(|e| {
        RvllmError::Io {
            err: rvllm_core::IoError::from(&e),
            path: index_path.clone(),
            source: e,
        }
    })?;

    let index: serde_json::Value = serde_json::from_slice(&index_bytes).map_err(|_| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob { reason: "invalid index json" },
            ctx("parse_index"),
        )
    })?;

    let weight_map = index.get("weight_map").and_then(|v| v.as_object()).ok_or_else(|| {
        RvllmError::apple(
            AppleError::InvalidWeightBlob { reason: "missing weight_map" },
            ctx("parse_index"),
        )
    })?;

    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (_tensor_name, file_val) in weight_map {
        if let Some(file_name) = file_val.as_str() {
            if seen.insert(file_name.to_owned()) {
                files.push((file_name.to_owned(), model_dir.join(file_name)));
            }
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_to_f16_converts_correctly() {
        let bf16_bytes = [0x80, 0x3F];
        let f16_bytes = bf16_to_f16_cpu(&bf16_bytes);
        let f16_val = half::f16::from_bits(u16::from_le_bytes([f16_bytes[0], f16_bytes[1]]));
        assert!((f16_val.to_f32() - 1.0).abs() < 0.001);
    }
}
