use std::collections::HashMap;
use rvllm_apple_coreml_sys::specification::{
    Model, FeatureDescription, feature_type,
    mil_spec,
};
use prost::Message;

pub(crate) static PROJ_TEMPLATE: &[u8] = include_bytes!("../templates/proj.mlmodel");
pub(crate) static FFN_TEMPLATE: &[u8] = include_bytes!("../templates/ffn.mlmodel");
pub(crate) static QKV_TEMPLATE: &[u8] = include_bytes!("../templates/qkv.mlmodel");

pub fn load_template(name: &str) -> Model {
    let bytes = match name {
        "proj.mlmodel" => PROJ_TEMPLATE,
        "ffn.mlmodel" => FFN_TEMPLATE,
        "qkv.mlmodel" => QKV_TEMPLATE,
        _ => panic!("Unknown template: {}", name),
    };
    Model::decode(bytes).expect("Failed to decode MIL template")
}

fn patch_tensor_type(tensor: &mut mil_spec::TensorType, spatial: usize, ch: usize) {
    tensor.data_type = mil_spec::DataType::Float16 as i32; // Force FP16
    if tensor.dimensions.len() == 4 {
        let is_ane_act = tensor.dimensions[0].dimension.as_ref().map_or(false, |d| matches!(d, mil_spec::dimension::Dimension::Constant(c) if c.size == 1))
            && tensor.dimensions[2].dimension.as_ref().map_or(false, |d| matches!(d, mil_spec::dimension::Dimension::Constant(c) if c.size == 1));
        
        if is_ane_act {
            tensor.dimensions[1].dimension = Some(mil_spec::dimension::Dimension::Constant(mil_spec::dimension::ConstantDimension { size: ch as u64 }));
            tensor.dimensions[3].dimension = Some(mil_spec::dimension::Dimension::Constant(mil_spec::dimension::ConstantDimension { size: spatial as u64 }));
        }
    }
}

fn patch_weight_type(tensor: &mut mil_spec::TensorType, out_ch: usize, in_ch: usize) {
    tensor.data_type = mil_spec::DataType::Float16 as i32; // Force FP16
    if tensor.dimensions.len() == 4 {
        tensor.dimensions[0].dimension = Some(mil_spec::dimension::Dimension::Constant(mil_spec::dimension::ConstantDimension { size: out_ch as u64 }));
        tensor.dimensions[1].dimension = Some(mil_spec::dimension::Dimension::Constant(mil_spec::dimension::ConstantDimension { size: in_ch as u64 }));
    }
}

fn patch_value_type(vt: &mut mil_spec::ValueType, spatial: usize, ch: usize, is_weight: bool, out_ch: usize) {
    if let Some(ref mut t) = vt.r#type {
        match t {
            mil_spec::value_type::Type::TensorType(ref mut tensor) => {
                if is_weight {
                    patch_weight_type(tensor, out_ch, ch);
                } else {
                    patch_tensor_type(tensor, spatial, ch);
                }
            }
            _ => {}
        }
    }
}

fn patch_feature_description(desc: &mut FeatureDescription, spatial: usize, ch: usize) {
    if let Some(ref mut t) = desc.r#type {
        if let Some(feature_type::Type::MultiArrayType(ref mut array)) = t.r#type {
            array.data_type = rvllm_apple_coreml_sys::specification::array_feature_type::ArrayDataType::Float16 as i32;
            if array.shape.len() == 4 {
                array.shape[1] = ch as i64;
                array.shape[3] = spatial as i64;
            }
        }
    }
}

pub fn patch_ast(
    model: &mut Model,
    _func_name: &str,
    spatial: usize,
    in_ch: usize,
    hidden_ch: usize,
    out_ch: usize,
    _offsets: &HashMap<&str, u64>,
) {
    model.specification_version = 7;
    
    if let Some(ref mut desc) = model.description {
        for input in desc.input.iter_mut() {
            patch_feature_description(input, spatial, in_ch);
        }
        for output in desc.output.iter_mut() {
            patch_feature_description(output, spatial, out_ch);
        }
    }

    let mlp = match model.r#type {
        Some(rvllm_apple_coreml_sys::specification::model::Type::MlProgram(ref mut p)) => p,
        _ => return,
    };

    let mut symbol_channels = HashMap::new();

    for func in mlp.functions.values_mut() {
        for input in func.inputs.iter_mut() {
            symbol_channels.insert(input.name.clone(), in_ch);
            if let Some(ref mut t) = input.r#type {
                patch_value_type(t, spatial, in_ch, false, 0);
            }
        }

        for block in func.block_specializations.values_mut() {
            for op in block.operations.iter_mut() {
                let name = &op.r#type;
                
                if name == "const" || name == "weight" {
                    let out_name = &op.outputs[0].name;
                    
                    if let Some(ref mut val_attr) = op.attributes.get_mut("val") {
                        match val_attr.value {
                            Some(mil_spec::value::Value::BlobFileValue(ref mut _blob)) => {
                                let (w_out, w_in) = if out_name.contains("gate") || out_name.contains("up") {
                                    (hidden_ch, in_ch)
                                } else if out_name.contains("down") {
                                    (out_ch, hidden_ch)
                                } else {
                                    (out_ch, in_ch)
                                };

                                if let Some(ref mut vt) = val_attr.r#type {
                                    patch_value_type(vt, spatial, w_in, true, w_out);
                                }
                                
                                if let Some(ref mut vt) = op.outputs[0].r#type {
                                    patch_value_type(vt, spatial, w_in, true, w_out);
                                }
                            },
                            Some(mil_spec::value::Value::ImmediateValue(ref mut imm)) => {
                                if let Some(mil_spec::value::immediate_value::Value::Tensor(ref mut tensor)) = imm.value {
                                    if let Some(mil_spec::tensor_value::Value::Strings(ref mut rs)) = tensor.value {
                                        for s in rs.values.iter_mut() {
                                            if s == "fp32" || s == "float32" {
                                                *s = "fp16".to_string();
                                            }
                                        }
                                    }
                                }
                            },
                            _ => {}
                        }
                    }
                } else if name == "conv" || name == "linear" || name == "matmul" {
                    let weight_name = op.inputs.get("weight").and_then(|a| a.arguments.first()).and_then(|b| {
                        if let Some(mil_spec::argument::binding::Binding::Name(ref n)) = b.binding {
                            Some(n)
                        } else {
                            None
                        }
                    });

                    let target_ch = if let Some(wn) = weight_name {
                        if wn.contains("gate") || wn.contains("up") {
                            hidden_ch
                        } else if wn.contains("down") {
                            out_ch
                        } else {
                            out_ch
                        }
                    } else {
                        out_ch
                    };

                    for output in op.outputs.iter_mut() {
                        symbol_channels.insert(output.name.clone(), target_ch);
                        if let Some(ref mut t) = output.r#type {
                            patch_value_type(t, spatial, target_ch, false, 0);
                        }
                    }
                } else {
                    for (attr_name, attr_val) in op.attributes.iter_mut() {
                        if attr_name == "dtype" {
                            if let Some(mil_spec::value::Value::ImmediateValue(ref mut imm)) = attr_val.value {
                                if let Some(mil_spec::value::immediate_value::Value::Tensor(ref mut tensor)) = imm.value {
                                    if let Some(mil_spec::tensor_value::Value::Strings(ref mut rs)) = tensor.value {
                                        for s in rs.values.iter_mut() {
                                            if s == "fp32" || s == "float32" {
                                                *s = "fp16".to_string();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let mut input_ch = in_ch;
                    for arg in op.inputs.values() {
                        for binding in &arg.arguments {
                            if let Some(mil_spec::argument::binding::Binding::Name(ref n)) = binding.binding {
                                if let Some(&ch) = symbol_channels.get(n) {
                                    input_ch = ch;
                                    break;
                                }
                            }
                        }
                    }

                    for output in op.outputs.iter_mut() {
                        symbol_channels.insert(output.name.clone(), input_ch);
                        if let Some(ref mut t) = output.r#type {
                            patch_value_type(t, spatial, input_ch, false, 0);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Eq, PartialEq)]
pub struct FfnMilOffsets {
    pub gate: u64,
    pub up: u64,
    pub down: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Eq, PartialEq)]
pub struct QkvMilOffsets {
    pub q: u64,
    pub k: u64,
    pub v: u64,
}

#[must_use]
pub fn dense_1x1_conv_mil(_name: &str, in_ch: usize, out_ch: usize, spatial: usize, offset: u64) -> Vec<u8> {
    let mut model = load_template("proj.mlmodel");
    let mut offsets = std::collections::HashMap::new();
    offsets.insert("proj_weight_to_fp16", offset);
    patch_ast(&mut model, "main", spatial, in_ch, in_ch, out_ch, &offsets);
    model.encode_to_vec()
}

#[must_use]
pub fn fused_ffn_mil(dim: usize, hidden_dim: usize, spatial: usize, offsets: FfnMilOffsets) -> Vec<u8> {
    let mut model = load_template("ffn.mlmodel");
    let mut off = std::collections::HashMap::new();
    off.insert("gate_weight_to_fp16", offsets.gate);
    off.insert("up_weight_to_fp16", offsets.up);
    off.insert("down_weight_to_fp16", offsets.down);
    patch_ast(&mut model, "main", spatial, dim, hidden_dim, dim, &off);
    model.encode_to_vec()
}

pub fn fused_qkv_mil(
    q_dim: usize,
    kv_dim: usize,
    _head_dim: usize,
    spatial: usize,
    offsets: QkvMilOffsets,
) -> Vec<u8> {
    let mut model = load_template("proj.mlmodel"); 
    let mut off = std::collections::HashMap::new();
    off.insert("q_weight_to_fp16", offsets.q);
    off.insert("k_weight_to_fp16", offsets.k);
    off.insert("v_weight_to_fp16", offsets.v);
    patch_ast(&mut model, "main", spatial, q_dim, q_dim, kv_dim, &off); // Dummy
    model.encode_to_vec()
}
