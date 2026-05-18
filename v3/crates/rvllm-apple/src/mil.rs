use half::f16;
use prost::Message;
use rvllm_apple_coreml_sys::specification::{
    array_feature_type, feature_type, neural_network_layer, ConvolutionLayerParams,
    FeatureDescription, Model, NeuralNetwork, NeuralNetworkLayer,
};
use std::collections::HashMap;

pub(crate) static PROJ_TEMPLATE: &[u8] = include_bytes!("../templates/proj.mlmodel");
pub(crate) static FFN_TEMPLATE: &[u8] = include_bytes!("../templates/ffn.mlmodel");
pub(crate) static QKV_TEMPLATE: &[u8] = include_bytes!("../templates/qkv.mlmodel");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeuralNetworkWeightEncoding {
    Float32,
    Float16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureRank {
    Rank3,
    Rank4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MilPatchOptions {
    pub feature_dtype: array_feature_type::ArrayDataType,
    pub feature_rank: FeatureRank,
    pub weight_encoding: NeuralNetworkWeightEncoding,
}

impl Default for MilPatchOptions {
    fn default() -> Self {
        Self {
            feature_dtype: array_feature_type::ArrayDataType::Float32,
            feature_rank: FeatureRank::Rank3,
            weight_encoding: NeuralNetworkWeightEncoding::Float32,
        }
    }
}

pub fn load_template(name: &str) -> Model {
    let bytes = match name {
        "proj.mlmodel" => PROJ_TEMPLATE,
        "ffn.mlmodel" => FFN_TEMPLATE,
        "qkv.mlmodel" => QKV_TEMPLATE,
        _ => panic!("Unknown template: {}", name),
    };
    Model::decode(bytes).expect("Failed to decode MIL template")
}

fn patch_feature_description(
    desc: &mut FeatureDescription,
    spatial: usize,
    ch: usize,
    options: MilPatchOptions,
) {
    if let Some(ref mut t) = desc.r#type {
        if let Some(feature_type::Type::MultiArrayType(ref mut array)) = t.r#type {
            array.data_type = options.feature_dtype as i32;
            array.shape = match options.feature_rank {
                FeatureRank::Rank3 => vec![ch as i64, 1, spatial as i64],
                FeatureRank::Rank4 => vec![1, ch as i64, 1, spatial as i64],
            };
        }
    }
}

pub fn patch_ast(
    model: &mut Model,
    _func_name: &str,
    spatial: usize,
    in_ch: usize,
    _hidden_ch: usize,
    out_ch: usize,
    _offsets: &HashMap<String, u64>,
) {
    patch_ast_with_options(
        model,
        _func_name,
        spatial,
        in_ch,
        _hidden_ch,
        out_ch,
        _offsets,
        MilPatchOptions::default(),
    );
}

pub fn patch_ast_with_options(
    model: &mut Model,
    _func_name: &str,
    spatial: usize,
    in_ch: usize,
    _hidden_ch: usize,
    out_ch: usize,
    _offsets: &HashMap<String, u64>,
    options: MilPatchOptions,
) {
    model.specification_version = 4; // Downgrade to NN

    if let Some(ref mut desc) = model.description {
        for input in desc.input.iter_mut() {
            patch_feature_description(input, spatial, in_ch, options);
        }
        for output in desc.output.iter_mut() {
            patch_feature_description(output, spatial, out_ch, options);
        }
    }

    let mut nn = NeuralNetwork::default();

    let mut layer = NeuralNetworkLayer::default();
    layer.name = "proj".to_string();
    layer.input.push("x".to_string());
    layer.output.push("var_13".to_string());

    let mut conv = ConvolutionLayerParams::default();
    conv.output_channels = out_ch as u64;
    conv.kernel_channels = in_ch as u64;
    conv.kernel_size.push(1);
    conv.kernel_size.push(1);
    conv.stride.push(1);
    conv.stride.push(1);
    conv.is_deconvolution = false;
    conv.has_bias = false;
    let mut weights = rvllm_apple_coreml_sys::specification::WeightParams::default();
    let weight_elements = (out_ch * in_ch) as usize;
    match options.weight_encoding {
        NeuralNetworkWeightEncoding::Float32 => {
            weights.float_value = vec![0.0; weight_elements];
        }
        NeuralNetworkWeightEncoding::Float16 => {
            let zero = f16::from_f32(0.0).to_bits().to_le_bytes();
            weights.float16_value = Vec::with_capacity(weight_elements * 2);
            for _ in 0..weight_elements {
                weights.float16_value.extend_from_slice(&zero);
            }
        }
    }
    conv.weights = Some(weights);
    conv.convolution_padding_type = Some(rvllm_apple_coreml_sys::specification::convolution_layer_params::ConvolutionPaddingType::Valid(rvllm_apple_coreml_sys::specification::ValidPadding::default()));

    layer.layer = Some(neural_network_layer::Layer::Convolution(conv));
    nn.layers.push(layer);

    model.r#type = Some(rvllm_apple_coreml_sys::specification::model::Type::NeuralNetwork(nn));
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
pub fn dense_1x1_conv_mil(
    _name: &str,
    in_ch: usize,
    out_ch: usize,
    spatial: usize,
    offset: u64,
) -> Vec<u8> {
    let mut model = load_template("proj.mlmodel");
    let mut offsets = HashMap::new();
    offsets.insert("proj_weight_to_fp16".to_string(), offset);
    patch_ast(&mut model, "main", spatial, in_ch, in_ch, out_ch, &offsets);
    model.encode_to_vec()
}

#[must_use]
pub fn fused_ffn_mil(
    dim: usize,
    hidden_dim: usize,
    spatial: usize,
    offsets: FfnMilOffsets,
) -> Vec<u8> {
    let mut model = load_template("ffn.mlmodel");
    let mut off = HashMap::new();
    off.insert("gate_weight_to_fp16".to_string(), offsets.gate);
    off.insert("up_weight_to_fp16".to_string(), offsets.up);
    off.insert("down_weight_to_fp16".to_string(), offsets.down);
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
    let mut off = HashMap::new();
    off.insert("q_weight_to_fp16".to_string(), offsets.q);
    off.insert("k_weight_to_fp16".to_string(), offsets.k);
    off.insert("v_weight_to_fp16".to_string(), offsets.v);
    patch_ast(&mut model, "main", spatial, q_dim, q_dim, kv_dim, &off);
    model.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvllm_apple_coreml_sys::specification::{
        array_feature_type, feature_type, model, neural_network_layer,
    };

    fn patched_projection(options: MilPatchOptions) -> Model {
        let mut model = load_template("proj.mlmodel");
        patch_ast_with_options(&mut model, "main", 16, 16, 16, 16, &HashMap::new(), options);
        model
    }

    #[test]
    fn default_neural_network_projection_uses_fp32_rank3_contract() {
        let model = patched_projection(MilPatchOptions::default());
        let desc = model
            .description
            .as_ref()
            .expect("model should have description");
        let input_type = desc.input[0]
            .r#type
            .as_ref()
            .expect("input should have type");
        let feature_type::Type::MultiArrayType(array) = input_type
            .r#type
            .as_ref()
            .expect("input should be multi-array")
        else {
            panic!("input should be a multi-array");
        };
        assert_eq!(
            array.data_type,
            array_feature_type::ArrayDataType::Float32 as i32
        );
        assert_eq!(array.shape, vec![16, 1, 16]);

        let Some(model::Type::NeuralNetwork(nn)) = model.r#type else {
            panic!("patched model should be a neural network");
        };
        let Some(neural_network_layer::Layer::Convolution(conv)) = &nn.layers[0].layer else {
            panic!("first layer should be a convolution");
        };
        let weights = conv.weights.as_ref().expect("conv should have weights");
        assert_eq!(weights.float_value.len(), 256);
        assert!(weights.float16_value.is_empty());
    }

    #[test]
    fn neural_network_projection_can_emit_fp16_weights_and_rank4_features() {
        let model = patched_projection(MilPatchOptions {
            feature_rank: FeatureRank::Rank4,
            weight_encoding: NeuralNetworkWeightEncoding::Float16,
            ..MilPatchOptions::default()
        });
        let desc = model
            .description
            .as_ref()
            .expect("model should have description");
        let input_type = desc.input[0]
            .r#type
            .as_ref()
            .expect("input should have type");
        let feature_type::Type::MultiArrayType(array) = input_type
            .r#type
            .as_ref()
            .expect("input should be multi-array")
        else {
            panic!("input should be a multi-array");
        };
        assert_eq!(array.shape, vec![1, 16, 1, 16]);

        let Some(model::Type::NeuralNetwork(nn)) = model.r#type else {
            panic!("patched model should be a neural network");
        };
        let Some(neural_network_layer::Layer::Convolution(conv)) = &nn.layers[0].layer else {
            panic!("first layer should be a convolution");
        };
        let weights = conv.weights.as_ref().expect("conv should have weights");
        assert!(weights.float_value.is_empty());
        assert_eq!(weights.float16_value.len(), 512);
    }
}
