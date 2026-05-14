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

const MIL_HEADER: &str = "program(1.0)\n{\n";
const MIL_FOOTER: &str = "}\n";

fn conv_preamble() -> &'static str {
    "        tensor<string, []> c_pad_type = const()[name = tensor<string, []>(\"c_pad_type\"), val = tensor<string, []>(\"valid\")];\n\
        tensor<int32, [2]> c_strides = const()[name = tensor<string, []>(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n\
        tensor<int32, [4]> c_pad = const()[name = tensor<string, []>(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
        tensor<int32, [2]> c_dilations = const()[name = tensor<string, []>(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n\
        tensor<int32, []> c_groups = const()[name = tensor<string, []>(\"c_groups\"), val = tensor<int32, []>(1)];\n"
}

#[must_use]
pub fn dense_1x1_conv_mil(name: &str, in_ch: usize, out_ch: usize, spatial: usize, offset: u64) -> String {
    format!(
        "{MIL_HEADER}    func main<ios16>(tensor<fp16, [1, {in_ch}, 1, {spatial}]> x) {{\n{}        tensor<fp16, [{out_ch}, {in_ch}, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), val = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>({offset})))];\n        tensor<fp16, [1, {out_ch}, 1, {spatial}]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = tensor<string, []>(\"{name}\")];\n    }} -> (y);\n{MIL_FOOTER}",
        conv_preamble()
    )
}

#[must_use]
pub fn fused_ffn_mil(dim: usize, hidden_dim: usize, spatial: usize, offsets: FfnMilOffsets) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

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
            FfnMilOffsets { gate: 128, up: 16_576, down: 33_024 },
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
        let mil = fused_qkv_mil(128, 128, 32, 8, QkvMilOffsets { q: 128, k: 1024, v: 2048 });
        assert!(mil.contains("conv_q"));
        assert!(mil.contains("conv_k"));
        assert!(mil.contains("conv_v"));
        assert!(mil.contains("-> (q, k, v)"));
    }
}
