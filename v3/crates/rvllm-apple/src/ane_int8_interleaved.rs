//! Paired gate/up rows with identical INT8 values/scales and FP16 operations.
//! This is a single-I/O cached experiment, controlled against existing stacking.
#![forbid(unsafe_code)]

use super::{
    append_rows, begin_blob, convolution, descriptor, gelu_branch, graph_header, AneInt8FfnWeights,
    AneInt8MatrixView, Int8CandidateBudget, Int8CandidateSource, GELU_CONSTANTS,
};

pub const NAME: &str = "ane-int8-ffn-interleaved";

pub fn supports(shape: (usize, usize)) -> bool {
    shape == (3840, 15360)
}

pub fn build(weights: &AneInt8FfnWeights) -> Result<Int8CandidateSource, String> {
    if !supports(weights.shape()) {
        return Err("interleaved FFN requires hidden=3840, intermediate=15360".into());
    }
    build_aligned(weights)
}

fn build_aligned(weights: &AneInt8FfnWeights) -> Result<Int8CandidateSource, String> {
    let (hidden, intermediate) = weights.shape();
    if hidden == 0 || hidden % 32 != 0 || intermediate == 0 || intermediate % 32 != 0 {
        return Err("interleaved FFN requires nonempty aligned channels".into());
    }
    let bytes = weights
        .source_blob_bytes()
        .checked_sub(128)
        .ok_or("source underflow")?;
    let io = hidden.checked_mul(64).ok_or("surface overflow")?;
    let doubled = intermediate.checked_mul(2).ok_or("GU width overflow")?;
    let mut blob = begin_blob(4, bytes)?;
    let mut constants = String::new();
    let [gate, up, down] = weights.matrices();
    append_interleaved(&mut blob, &mut constants, &gate, &up)?;
    append_rows(&mut blob, &mut constants, "Wd", &down, 0..hidden)?;
    let mut mil = graph_header(hidden, &constants);
    mil.push_str(GELU_CONSTANTS);
    convolution(&mut mil, "gu", "Wgu", "x", doubled);
    mil.push_str(&format!(r#"        tensor<int32, [4]> gu_shape = const()[name = string("gu_shape"), val = tensor<int32, [4]>([1, {intermediate}, 2, 1])];
        tensor<fp16, [1, {intermediate}, 2, 1]> gu_pairs = reshape(shape = gu_shape, x = gu)[name = string("gu_pairs")];
        tensor<int32, [4]> gate_begin = const()[name = string("gate_begin"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [4]> up_begin = const()[name = string("up_begin"), val = tensor<int32, [4]>([0, 0, 1, 0])];
        tensor<int32, [4]> branch_size = const()[name = string("branch_size"), val = tensor<int32, [4]>([1, {intermediate}, 1, 1])];
        tensor<fp16, [1, {intermediate}, 1, 1]> gate0 = slice_by_size(begin = gate_begin, size = branch_size, x = gu_pairs)[name = string("gate0")];
        tensor<fp16, [1, {intermediate}, 1, 1]> up0 = slice_by_size(begin = up_begin, size = branch_size, x = gu_pairs)[name = string("up0")];
"#));
    gelu_branch(&mut mil, 0, intermediate);
    convolution(&mut mil, "y", "Wd", "gated0", hidden);
    mil.push_str("    } -> (y);\n}\n");
    if blob.len() != bytes {
        return Err("interleaved source size mismatch".into());
    }
    Ok(Int8CandidateSource {
        name: NAME,
        mil,
        blob,
        budget: Int8CandidateBudget {
            source_blob_bytes: bytes,
            convolutions: 2,
            programs: 1,
            input_surface_bytes: io,
            output_surface_bytes: io,
        },
    })
}

/// Stream rows directly into the one source blob: no full interleaved dense
/// matrix, no new scale estimation, and no second source-sized temporary.
fn append_interleaved(
    blob: &mut Vec<u8>,
    constants: &mut String,
    gate: &AneInt8MatrixView<'_>,
    up: &AneInt8MatrixView<'_>,
) -> Result<(), String> {
    let rows = gate.scales.len();
    let columns = gate.columns;
    if rows == 0
        || rows % 32 != 0
        || columns == 0
        || columns % 32 != 0
        || up.columns != columns
        || up.scales.len() != rows
        || rows.checked_mul(columns) != Some(gate.values.len())
        || up.values.len() != gate.values.len()
    {
        return Err("interleaved GU shape mismatch".into());
    }
    let doubled = rows.checked_mul(2).ok_or("interleaved row overflow")?;
    let q_bytes = doubled
        .checked_mul(columns)
        .ok_or("interleaved payload overflow")?;
    let scale_bytes = doubled
        .checked_mul(2)
        .ok_or("interleaved scales overflow")?;
    let q = descriptor(blob, 4, q_bytes)?;
    for (g, u) in gate
        .values
        .chunks_exact(columns)
        .zip(up.values.chunks_exact(columns))
    {
        blob.extend(g.iter().chain(u).map(|value| value.to_le_bytes()[0]));
    }
    let scale = descriptor(blob, 1, scale_bytes)?;
    for (g, u) in gate.scales.iter().zip(up.scales) {
        blob.extend_from_slice(&g.to_le_bytes());
        blob.extend_from_slice(&u.to_le_bytes());
    }
    constants.push_str(&format!(
        "        tensor<fp16, [{doubled}, {columns}, 1, 1]> Wgu = constexpr_affine_dequantize()[axis = int32(0), name = string(\"Wgu\"), quantized_data = tensor<int8, [{doubled}, {columns}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({q}))), scale = tensor<fp16, [{doubled}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({scale}))), zero_point = int8(0)];\n"
    ));
    Ok(())
}

#[cfg(test)]
pub(crate) fn build_for_host_test(
    weights: &AneInt8FfnWeights,
) -> Result<Int8CandidateSource, String> {
    build_aligned(weights)
}
