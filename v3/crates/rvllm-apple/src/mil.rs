use prost::Message;
use rvllm_apple_coreml_sys::specification::Model;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct FfnMilOffsets {
    pub gate: u64,
    pub up: u64,
    pub down: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct QkvMilOffsets {
    pub q: u64,
    pub k: u64,
    pub v: u64,
}

static FFN_TEMPLATE: &[u8] = include_bytes!("../templates/ffn.mlmodel");
static QKV_TEMPLATE: &[u8] = include_bytes!("../templates/qkv.mlmodel");
static PROJ_TEMPLATE: &[u8] = include_bytes!("../templates/proj.mlmodel");

fn load_template(bytes: &[u8]) -> Model {
    Model::decode(bytes).expect("Built-in CoreML template is corrupted")
}

#[must_use]
pub fn dense_1x1_conv_mil(name: &str, in_ch: usize, out_ch: usize, spatial: usize, offset: u64) -> Vec<u8> {
    let mut model = load_template(PROJ_TEMPLATE);
    // TODO: walk AST and replace sizes and weight offsets
    model.encode_to_vec()
}

#[must_use]
pub fn fused_ffn_mil(dim: usize, hidden_dim: usize, spatial: usize, offsets: FfnMilOffsets) -> Vec<u8> {
    let mut model = load_template(FFN_TEMPLATE);
    // TODO: walk AST and replace sizes and weight offsets
    model.encode_to_vec()
}

#[must_use]
pub fn fused_qkv_mil(
    dim: usize,
    q_dim: usize,
    kv_dim: usize,
    spatial: usize,
    offsets: QkvMilOffsets,
) -> Vec<u8> {
    let mut model = load_template(QKV_TEMPLATE);
    // TODO: walk AST and replace sizes and weight offsets
    model.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_templates() {
        let ffn = fused_ffn_mil(64, 128, 4, FfnMilOffsets { gate: 0, up: 0, down: 0 });
        assert!(!ffn.is_empty());
        let qkv = fused_qkv_mil(128, 128, 32, 8, QkvMilOffsets { q: 0, k: 0, v: 0 });
        assert!(!qkv.is_empty());
    }
}
