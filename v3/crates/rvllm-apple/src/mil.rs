use prost::Message;
use rvllm_apple_coreml_sys::specification::{
    mil_spec::{dimension, value, value_type, argument, tensor_value},
    model, Model,
};
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

fn patch_ast(
    model: &mut Model,
    spatial: usize,
    in_ch: usize,
    _out_ch: usize,
    offsets: &std::collections::HashMap<&str, u64>,
) {
    if let Some(model::Type::MlProgram(ref mut program)) = model.r#type {
        if let Some(func) = program.functions.get_mut("main") {
            // Update input shape
            if let Some(input) = func.inputs.first_mut() {
                if let Some(ref mut val_type) = input.r#type {
                    if let Some(value_type::Type::TensorType(ref mut tensor)) = val_type.r#type {
                        if tensor.dimensions.len() == 4 {
                            tensor.dimensions[1].dimension = Some(dimension::Dimension::Constant(dimension::ConstantDimension { size: in_ch as u64 }));
                            tensor.dimensions[3].dimension = Some(dimension::Dimension::Constant(dimension::ConstantDimension { size: spatial as u64 }));
                        }
                    }
                }
            }

            // Update operations
            for block in func.block_specializations.values_mut() {
                for op in block.operations.iter_mut() {
                    // Patch constant weight offsets
                    if op.r#type == "const" {
                        if let Some(arg) = op.inputs.get_mut("val") {
                            if let Some(binding) = arg.arguments.first_mut() {
                                if let Some(argument::binding::Binding::Value(ref mut val)) = binding.binding {
                                    if let Some(value::Value::BlobFileValue(ref mut blob)) = val.value {
                                        if let Some(op_name) = op.attributes.get("name") {
                                            if let Some(value::Value::ImmediateValue(imm)) = &op_name.value {
                                                if let Some(value::immediate_value::Value::Tensor(tensor)) = &imm.value {
                                                    if let Some(tensor_value::Value::Strings(strings)) = &tensor.value {
                                                        if let Some(name) = strings.values.first() {
                                                            let name_str: &str = name.as_str();
                                                            if let Some(offset) = offsets.get(name_str) {
                                                                blob.offset = *offset;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[must_use]
pub fn dense_1x1_conv_mil(_name: &str, in_ch: usize, out_ch: usize, spatial: usize, offset: u64) -> Vec<u8> {
    let mut model = load_template(PROJ_TEMPLATE);
    let mut offsets = std::collections::HashMap::new();
    offsets.insert("proj.weight", offset);
    patch_ast(&mut model, spatial, in_ch, out_ch, &offsets);
    model.encode_to_vec()
}

#[must_use]
pub fn fused_ffn_mil(dim: usize, _hidden_dim: usize, spatial: usize, offsets: FfnMilOffsets) -> Vec<u8> {
    let mut model = load_template(FFN_TEMPLATE);
    let mut off = std::collections::HashMap::new();
    off.insert("gate.weight", offsets.gate);
    off.insert("up.weight", offsets.up);
    off.insert("down.weight", offsets.down);
    patch_ast(&mut model, spatial, dim, dim, &off);
    model.encode_to_vec()
}

#[must_use]
pub fn fused_qkv_mil(
    dim: usize,
    q_dim: usize,
    _kv_dim: usize,
    spatial: usize,
    offsets: QkvMilOffsets,
) -> Vec<u8> {
    let mut model = load_template(QKV_TEMPLATE);
    let mut off = std::collections::HashMap::new();
    off.insert("q.weight", offsets.q);
    off.insert("k.weight", offsets.k);
    off.insert("v.weight", offsets.v);
    patch_ast(&mut model, spatial, dim, q_dim, &off);
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
