use rvllm_core::{AppleCtx, AppleError, Result, RvllmError};
use serde::{Deserialize, Serialize};

use crate::weight_blob::WeightChunkDesc;

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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FfnMilWeightDescs<'a> {
    pub gate: &'a WeightChunkDesc,
    pub up: &'a WeightChunkDesc,
    pub down: &'a WeightChunkDesc,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct QkvMilWeightDescs<'a> {
    pub q: &'a WeightChunkDesc,
    pub k: &'a WeightChunkDesc,
    pub v: &'a WeightChunkDesc,
}

const MIL_HEADER: &str = "program(1.0)\n{\n";
const MIL_FOOTER: &str = "}\n";
const FP16_BYTES: usize = 2;

fn conv_preamble() -> &'static str {
    "        tensor<string, []> c_pad_type = const()[name = tensor<string, []>(\"c_pad_type\"), val = tensor<string, []>(\"valid\")];\n\
        tensor<int32, [2]> c_strides = const()[name = tensor<string, []>(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n\
        tensor<int32, [4]> c_pad = const()[name = tensor<string, []>(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
        tensor<int32, [2]> c_dilations = const()[name = tensor<string, []>(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n\
        tensor<int32, []> c_groups = const()[name = tensor<string, []>(\"c_groups\"), val = tensor<int32, []>(1)];\n"
}

#[must_use]
pub fn dense_1x1_conv_mil(
    name: &str,
    in_ch: usize,
    out_ch: usize,
    spatial: usize,
    offset: u64,
) -> String {
    format!(
        "{MIL_HEADER}    func main<ios16>(tensor<fp16, [1, {in_ch}, 1, {spatial}]> x) {{\n{}        tensor<fp16, [{out_ch}, {in_ch}, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), val = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({offset})))];\n        tensor<fp16, [1, {out_ch}, 1, {spatial}]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = tensor<string, []>(\"{name}\")];\n    }} -> (y);\n{MIL_FOOTER}",
        conv_preamble()
    )
}

#[must_use]
pub fn fused_ffn_mil(
    dim: usize,
    hidden_dim: usize,
    spatial: usize,
    offsets: FfnMilOffsets,
) -> String {
    let mut s = String::with_capacity(8192);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios16>(tensor<fp16, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(conv_preamble());
    s.push_str(&format!(
        "        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W_gate = const()[name = tensor<string, []>(\"W_gate\"), val = tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        offsets.gate
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> gate = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_gate, x = x)[name = tensor<string, []>(\"conv_gate\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W_up = const()[name = tensor<string, []>(\"W_up\"), val = tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        offsets.up
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> up = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_up, x = x)[name = tensor<string, []>(\"conv_up\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sigmoid\")];\n        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"silu\")];\n        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> gated = mul(x = silu, y = up)[name = tensor<string, []>(\"gate_mul\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [{dim}, {hidden_dim}, 1, 1]> W_down = const()[name = tensor<string, []>(\"W_down\"), val = tensor<fp16, [{dim}, {hidden_dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        offsets.down
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_down, x = gated)[name = tensor<string, []>(\"conv_down\")];\n    }} -> (y);\n"
    ));
    s.push_str(MIL_FOOTER);
    s
}

pub fn fused_ffn_mil_from_descs(
    dim: usize,
    hidden_dim: usize,
    spatial: usize,
    weights: FfnMilWeightDescs<'_>,
) -> Result<String> {
    validate_mil_weight_desc(weights.gate, &[hidden_dim, dim, 1, 1])?;
    validate_mil_weight_desc(weights.up, &[hidden_dim, dim, 1, 1])?;
    validate_mil_weight_desc(weights.down, &[dim, hidden_dim, 1, 1])?;

    let mut s = String::with_capacity(8192);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios16>(tensor<fp16, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(conv_preamble());
    s.push_str(&format!(
        "        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W_gate = const()[name = tensor<string, []>(\"{}\"), val = tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        weights.gate.name, weights.gate.data_offset
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> gate = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_gate, x = x)[name = tensor<string, []>(\"conv_gate\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [{hidden_dim}, {dim}, 1, 1]> W_up = const()[name = tensor<string, []>(\"{}\"), val = tensor<fp16, [{hidden_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        weights.up.name, weights.up.data_offset
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> up = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_up, x = x)[name = tensor<string, []>(\"conv_up\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sigmoid\")];\n        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"silu\")];\n        tensor<fp16, [1, {hidden_dim}, 1, {spatial}]> gated = mul(x = silu, y = up)[name = tensor<string, []>(\"gate_mul\")];\n"
    ));
    s.push_str(&format!(
        "        tensor<fp16, [{dim}, {hidden_dim}, 1, 1]> W_down = const()[name = tensor<string, []>(\"{}\"), val = tensor<fp16, [{dim}, {hidden_dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        weights.down.name, weights.down.data_offset
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {dim}, 1, {spatial}]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_down, x = gated)[name = tensor<string, []>(\"conv_down\")];\n    }} -> (y);\n"
    ));
    s.push_str(MIL_FOOTER);
    Ok(s)
}

#[must_use]
pub fn fused_qkv_mil(
    dim: usize,
    q_dim: usize,
    kv_dim: usize,
    spatial: usize,
    offsets: QkvMilOffsets,
) -> String {
    let mut s = String::with_capacity(8192);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios16>(tensor<fp16, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(conv_preamble());
    s.push_str(&format!(
        "        tensor<fp16, [{q_dim}, {dim}, 1, 1]> Wq = const()[name = tensor<string, []>(\"Wq\"), val = tensor<fp16, [{q_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        offsets.q
    ));
    s.push_str(&format!(
        "        tensor<fp16, [{kv_dim}, {dim}, 1, 1]> Wk = const()[name = tensor<string, []>(\"Wk\"), val = tensor<fp16, [{kv_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        offsets.k
    ));
    s.push_str(&format!(
        "        tensor<fp16, [{kv_dim}, {dim}, 1, 1]> Wv = const()[name = tensor<string, []>(\"Wv\"), val = tensor<fp16, [{kv_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        offsets.v
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {q_dim}, 1, {spatial}]> q = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wq, x = x)[name = tensor<string, []>(\"conv_q\")];\n        tensor<fp16, [1, {kv_dim}, 1, {spatial}]> k = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x)[name = tensor<string, []>(\"conv_k\")];\n        tensor<fp16, [1, {kv_dim}, 1, {spatial}]> v = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x)[name = tensor<string, []>(\"conv_v\")];\n    }} -> (q, k, v);\n"
    ));
    s.push_str(MIL_FOOTER);
    s
}

pub fn fused_qkv_mil_from_descs(
    dim: usize,
    q_dim: usize,
    kv_dim: usize,
    spatial: usize,
    weights: QkvMilWeightDescs<'_>,
) -> Result<String> {
    validate_mil_weight_desc(weights.q, &[q_dim, dim, 1, 1])?;
    validate_mil_weight_desc(weights.k, &[kv_dim, dim, 1, 1])?;
    validate_mil_weight_desc(weights.v, &[kv_dim, dim, 1, 1])?;

    let mut s = String::with_capacity(8192);
    s.push_str(MIL_HEADER);
    s.push_str(&format!(
        "    func main<ios16>(tensor<fp16, [1, {dim}, 1, {spatial}]> x) {{\n"
    ));
    s.push_str(conv_preamble());
    s.push_str(&format!(
        "        tensor<fp16, [{q_dim}, {dim}, 1, 1]> Wq = const()[name = tensor<string, []>(\"{}\"), val = tensor<fp16, [{q_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        weights.q.name, weights.q.data_offset
    ));
    s.push_str(&format!(
        "        tensor<fp16, [{kv_dim}, {dim}, 1, 1]> Wk = const()[name = tensor<string, []>(\"{}\"), val = tensor<fp16, [{kv_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        weights.k.name, weights.k.data_offset
    ));
    s.push_str(&format!(
        "        tensor<fp16, [{kv_dim}, {dim}, 1, 1]> Wv = const()[name = tensor<string, []>(\"{}\"), val = tensor<fp16, [{kv_dim}, {dim}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({})))];\n",
        weights.v.name, weights.v.data_offset
    ));
    s.push_str(&format!(
        "        tensor<fp16, [1, {q_dim}, 1, {spatial}]> q = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wq, x = x)[name = tensor<string, []>(\"conv_q\")];\n        tensor<fp16, [1, {kv_dim}, 1, {spatial}]> k = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x)[name = tensor<string, []>(\"conv_k\")];\n        tensor<fp16, [1, {kv_dim}, 1, {spatial}]> v = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x)[name = tensor<string, []>(\"conv_v\")];\n    }} -> (q, k, v);\n"
    ));
    s.push_str(MIL_FOOTER);
    Ok(s)
}

fn validate_mil_weight_desc(desc: &WeightChunkDesc, expected_shape: &[usize]) -> Result<()> {
    if desc.name.is_empty() {
        return Err(invalid_mil("weight descriptor name is empty"));
    }
    if desc
        .name
        .bytes()
        .any(|b| matches!(b, b'"' | b'\\' | b'\n' | b'\r'))
    {
        return Err(invalid_mil("weight descriptor name is not MIL string-safe"));
    }
    if desc.shape.as_slice() != expected_shape {
        return Err(invalid_mil(
            "weight descriptor shape does not match MIL shape",
        ));
    }

    let expected_elements = shape_elements(expected_shape)?;
    if desc.elements != expected_elements {
        return Err(invalid_mil(
            "weight descriptor element count does not match MIL shape",
        ));
    }
    let expected_bytes = expected_elements
        .checked_mul(FP16_BYTES)
        .ok_or_else(|| invalid_mil("weight descriptor byte count overflowed"))?;
    if desc.data_bytes != expected_bytes {
        return Err(invalid_mil(
            "weight descriptor byte count does not match FP16 shape",
        ));
    }
    Ok(())
}

fn shape_elements(shape: &[usize]) -> Result<usize> {
    let mut elements = 1usize;
    for &dim in shape {
        elements = elements
            .checked_mul(dim)
            .ok_or_else(|| invalid_mil("MIL shape element count overflowed"))?;
    }
    Ok(elements)
}

fn invalid_mil(reason: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::InvalidMil { reason },
        AppleCtx {
            backend: "private-ane",
            op: "generate_mil",
            device: "apple-silicon",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_blob::{build_weight_blob_fp16_described, AneFp16WeightSpec};

    #[test]
    fn dense_mil_contains_shape_and_offset() {
        let mil = dense_1x1_conv_mil("proj", 64, 128, 4, 128);
        assert!(mil.contains("tensor<fp16, [1, 64, 1, 4]>"));
        assert!(mil.contains("tensor<fp16, [128, 64, 1, 1]>"));
        assert!(mil.contains("offset = tensor<uint64, []>(128)"));
    }

    #[test]
    fn fused_ffn_mil_uses_exact_blob_offsets() {
        let mil = fused_ffn_mil(
            64,
            128,
            4,
            FfnMilOffsets {
                gate: 128,
                up: 16_576,
                down: 33_024,
            },
        );
        assert!(mil.contains("conv_gate"));
        assert!(mil.contains("sigmoid"));
        assert!(mil.contains("gate_mul"));
        assert!(mil.contains("conv_down"));
        assert!(mil.contains("offset = tensor<uint64, []>(16576)"));
        assert!(mil.contains("offset = tensor<uint64, []>(33024)"));
    }

    #[test]
    fn fused_qkv_mil_names_three_outputs() {
        let mil = fused_qkv_mil(
            128,
            128,
            32,
            8,
            QkvMilOffsets {
                q: 128,
                k: 1024,
                v: 2048,
            },
        );
        assert!(mil.contains("conv_q"));
        assert!(mil.contains("conv_k"));
        assert!(mil.contains("conv_v"));
        assert!(mil.contains("-> (q, k, v)"));
    }

    #[test]
    fn fused_ffn_mil_from_blob_descs_uses_names_shapes_and_offsets() {
        let dim = 4;
        let hidden_dim = 8;
        let gate = vec![1.0f32; hidden_dim * dim];
        let up = vec![2.0f32; hidden_dim * dim];
        let down = vec![3.0f32; dim * hidden_dim];
        let (_blob, desc) = match build_weight_blob_fp16_described(&[
            AneFp16WeightSpec::new(
                "layer0.mlp.gate_proj.weight",
                &[hidden_dim, dim, 1, 1],
                &gate,
            ),
            AneFp16WeightSpec::new("layer0.mlp.up_proj.weight", &[hidden_dim, dim, 1, 1], &up),
            AneFp16WeightSpec::new(
                "layer0.mlp.down_proj.weight",
                &[dim, hidden_dim, 1, 1],
                &down,
            ),
        ]) {
            Ok(out) => out,
            Err(err) => panic!("{err}"),
        };

        let mil = match fused_ffn_mil_from_descs(
            dim,
            hidden_dim,
            3,
            FfnMilWeightDescs {
                gate: &desc[0],
                up: &desc[1],
                down: &desc[2],
            },
        ) {
            Ok(out) => out,
            Err(err) => panic!("{err}"),
        };

        assert!(mil.starts_with("program(1.0)\n{\n"));
        assert!(mil.contains("tensor<fp16, [1, 4, 1, 3]> x"));
        assert!(mil.contains("name = tensor<string, []>(\"layer0.mlp.gate_proj.weight\")"));
        assert!(mil.contains("tensor<fp16, [8, 4, 1, 1]>(BLOBFILE"));
        assert!(mil.contains("offset = tensor<uint64, []>(128)"));
        assert!(mil.contains("name = tensor<string, []>(\"layer0.mlp.up_proj.weight\")"));
        assert!(mil.contains("offset = tensor<uint64, []>(256)"));
        assert!(mil.contains("name = tensor<string, []>(\"layer0.mlp.down_proj.weight\")"));
        assert!(mil.contains("tensor<fp16, [4, 8, 1, 1]>(BLOBFILE"));
        assert!(mil.contains("offset = tensor<uint64, []>(384)"));
        assert!(mil.contains("tensor<fp16, [1, 4, 1, 3]> y"));
    }

    #[test]
    fn fused_qkv_mil_from_blob_descs_uses_names_shapes_and_offsets() {
        let dim = 4;
        let q_dim = 8;
        let kv_dim = 2;
        let q = vec![1.0f32; q_dim * dim];
        let k = vec![2.0f32; kv_dim * dim];
        let v = vec![3.0f32; kv_dim * dim];
        let (_blob, desc) = match build_weight_blob_fp16_described(&[
            AneFp16WeightSpec::new("layer0.self_attn.q_proj.weight", &[q_dim, dim, 1, 1], &q),
            AneFp16WeightSpec::new("layer0.self_attn.k_proj.weight", &[kv_dim, dim, 1, 1], &k),
            AneFp16WeightSpec::new("layer0.self_attn.v_proj.weight", &[kv_dim, dim, 1, 1], &v),
        ]) {
            Ok(out) => out,
            Err(err) => panic!("{err}"),
        };

        let mil = match fused_qkv_mil_from_descs(
            dim,
            q_dim,
            kv_dim,
            5,
            QkvMilWeightDescs {
                q: &desc[0],
                k: &desc[1],
                v: &desc[2],
            },
        ) {
            Ok(out) => out,
            Err(err) => panic!("{err}"),
        };

        assert!(mil.starts_with("program(1.0)\n{\n"));
        assert!(mil.contains("tensor<fp16, [1, 4, 1, 5]> x"));
        assert!(mil.contains("name = tensor<string, []>(\"layer0.self_attn.q_proj.weight\")"));
        assert!(mil.contains("tensor<fp16, [8, 4, 1, 1]>(BLOBFILE"));
        assert!(mil.contains("offset = tensor<uint64, []>(128)"));
        assert!(mil.contains("name = tensor<string, []>(\"layer0.self_attn.k_proj.weight\")"));
        assert!(mil.contains("tensor<fp16, [2, 4, 1, 1]>(BLOBFILE"));
        assert!(mil.contains("offset = tensor<uint64, []>(256)"));
        assert!(mil.contains("name = tensor<string, []>(\"layer0.self_attn.v_proj.weight\")"));
        assert!(mil.contains("offset = tensor<uint64, []>(336)"));
        assert!(mil.contains("-> (q, k, v);"));
    }
}
