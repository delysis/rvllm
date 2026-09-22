//! Explicit INT8 graph-layout proposals. No compilation, evaluation or FFI.
//! Constants are sliced in output-row order, never requantized. A single logical
//! input/output retains the existing width-32 physical surface convention.
#![forbid(unsafe_code)]

use crate::ane_int8_ffn_weights::{AneInt8FfnWeights, AneInt8LinearWeights, AneInt8MatrixView};
use std::ops::Range;

pub const FFN_CHUNK4: &str = "ane-int8-ffn-chunk4";

pub const LINEAR_TILES4: &str = "ane-int8-linear-tiles4";

/// Component API for archived sliding QKV, O and vocabulary-block shapes.
/// This does not quantize weights or select INT8 for existing FP16 operators.
/// Global QKV (8704 rows in this ANE path) is deliberately not admitted.
pub fn linear_tiles4(weights: &AneInt8LinearWeights) -> Result<Int8CandidateSource, String> {
    if !linear_tiles4_shape(weights.shape()) {
        return Err("ane-int8-linear-tiles4 supports only sliding QKV, O and head-block shapes".into());
    }
    build_linear_tiles4(weights)
}

pub fn linear_tiles4_shape(shape: (usize, usize)) -> bool {
    matches!(shape, (3840, 8192) | (4096, 3840) | (8192, 3840) | (3840, 16384))
}

fn build_linear_tiles4(weights: &AneInt8LinearWeights) -> Result<Int8CandidateSource, String> {
    let (input, output) = weights.shape();
    if input == 0 || input % 32 != 0 || output == 0 || output % 128 != 0 {
        return Err("tiles4 requires nonempty aligned output chunks".into());
    }
    let bytes = weights.source_blob_bytes().checked_add(6 * 64).ok_or("tiles4 blob overflow")?;
    let mut blob = begin_blob(8, bytes)?;
    let mut constants = String::new();
    let matrix = weights.matrix();
    let chunk = output / 4;
    for part in 0..4 {
        append_rows(&mut blob, &mut constants, &format!("W{part}"), &matrix,
            part * chunk..(part + 1) * chunk)?;
    }
    if blob.len() != bytes { return Err("tiles4 source size mismatch".into()); }
    let mut mil = graph_header(input, &constants);
    for part in 0..4 {
        convolution(&mut mil, &format!("y{part}"), &format!("W{part}"), "x", chunk);
    }
    concatenate(&mut mil, "y", &["y0", "y1", "y2", "y3"], output);
    mil.push_str("    } -> (y);\n}\n");
    Ok(Int8CandidateSource {
        name: LINEAR_TILES4, mil, blob,
        budget: Int8CandidateBudget {
            source_blob_bytes: bytes, convolutions: 4, programs: 1,
            input_surface_bytes: input * 64, output_surface_bytes: output * 64,
        },
    })
}

/// Source costs, not compiled residency or measured peak memory. Internal ANE
/// allocation, scheduling and constant folding are deliberately unspecified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8CandidateBudget {
    pub source_blob_bytes: usize,
    pub convolutions: usize,
    pub programs: usize,
    pub input_surface_bytes: usize,
    pub output_surface_bytes: usize,
}

pub struct Int8CandidateSource {
    pub name: &'static str,
    pub mil: String,
    pub blob: Vec<u8>,
    pub budget: Int8CandidateBudget,
}

/// Only the archived Gemma 4 12B FFN is admitted. This does not guess dimensions
/// for larger checkpoints. Call the baseline API explicitly for other shapes.
pub fn ffn_chunk4(weights: &AneInt8FfnWeights) -> Result<Int8CandidateSource, String> {
    if !ffn_chunk4_shape(weights.shape()) {
        return Err("ane-int8-ffn-chunk4 supports only hidden=3840, intermediate=15360".into());
    }
    build_ffn_chunk4(weights)
}

pub fn ffn_chunk4_shape(shape: (usize, usize)) -> bool {
    shape == (3840, 15360)
}

// Small aligned shapes remain private, solely for host tests of the graph and
// binary serializer; they cannot be selected by an accelerator entry point.
fn build_ffn_chunk4(weights: &AneInt8FfnWeights) -> Result<Int8CandidateSource, String> {
    let (hidden, intermediate) = weights.shape();
    if hidden == 0 || intermediate == 0 || hidden % 32 != 0 || intermediate % 128 != 0 {
        return Err("chunk4 requires nonempty aligned rows".into());
    }
    let bytes = weights.source_blob_bytes().checked_add(12 * 64)
        .ok_or("chunk4 blob overflow")?;
    let mut blob = begin_blob(18, bytes)?;
    let mut constants = String::new();
    let [gate, up, down] = weights.matrices();
    let chunk = intermediate / 4;
    for part in 0..4 {
        let rows = part * chunk..(part + 1) * chunk;
        append_rows(&mut blob, &mut constants, &format!("Wg{part}"), &gate, rows.clone())?;
        append_rows(&mut blob, &mut constants, &format!("Wu{part}"), &up, rows)?;
    }
    append_rows(&mut blob, &mut constants, "Wd", &down, 0..hidden)?;
    if blob.len() != bytes { return Err("chunk4 source size mismatch".into()); }
    let mut mil = graph_header(hidden, &constants);
    mil.push_str(GELU_CONSTANTS);
    for part in 0..4 {
        convolution(&mut mil, &format!("gate{part}"), &format!("Wg{part}"), "x", chunk);
        convolution(&mut mil, &format!("up{part}"), &format!("Wu{part}"), "x", chunk);
        gelu_branch(&mut mil, part, chunk);
    }
    concatenate(&mut mil, "gated", &["gated0", "gated1", "gated2", "gated3"], intermediate);
    // One full down projection: splitting its reduction would add new FP16
    // rounding and is NOT part of this candidate.
    convolution(&mut mil, "y", "Wd", "gated", hidden);
    mil.push_str("    } -> (y);\n}\n");
    Ok(Int8CandidateSource {
        name: FFN_CHUNK4, mil, blob,
        budget: Int8CandidateBudget {
            source_blob_bytes: bytes, convolutions: 9, programs: 1,
            input_surface_bytes: hidden * 64, output_surface_bytes: hidden * 64,
        },
    })
}

fn begin_blob(descriptors: u32, bytes: usize) -> Result<Vec<u8>, String> {
    if bytes < 64 || bytes > u32::MAX as usize {
        return Err("candidate source exceeds the existing blob ABI".into());
    }
    let mut blob = Vec::new();
    blob.try_reserve_exact(bytes).map_err(|error| format!("candidate source allocation: {error}"))?;
    blob.resize(64, 0);
    blob[..4].copy_from_slice(&descriptors.to_le_bytes());
    blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
    Ok(blob)
}

fn descriptor(blob: &mut Vec<u8>, dtype: u32, bytes: usize) -> Result<usize, String> {
    let offset = blob.len();
    let end = offset.checked_add(64).and_then(|n| n.checked_add(bytes))
        .filter(|&n| n <= u32::MAX as usize && n <= blob.capacity())
        .ok_or("candidate descriptor exceeds reserved source")?;
    if offset % 64 != 0 || bytes == 0 || bytes % 64 != 0 {
        return Err("candidate descriptor or payload is not 64-byte aligned".into());
    }
    blob.resize(end - bytes, 0);
    blob[offset..offset + 4].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
    blob[offset + 4..offset + 8].copy_from_slice(&dtype.to_le_bytes());
    blob[offset + 8..offset + 16].copy_from_slice(&(bytes as u64).to_le_bytes());
    blob[offset + 16..offset + 24].copy_from_slice(&((offset + 64) as u64).to_le_bytes());
    Ok(offset)
}

fn append_rows(
    blob: &mut Vec<u8>, constants: &mut String, name: &str,
    matrix: &AneInt8MatrixView<'_>, rows: Range<usize>,
) -> Result<(), String> {
    let columns = matrix.columns;
    if rows.start >= rows.end || rows.end > matrix.scales.len()
        || columns == 0 || columns % 32 != 0 || rows.start % 32 != 0 || rows.end % 32 != 0
        || matrix.scales.len().checked_mul(columns) != Some(matrix.values.len())
    { return Err("invalid candidate constant row range".into()); }
    let count = rows.end - rows.start;
    let start = rows.start.checked_mul(columns).ok_or("row offset overflow")?;
    let end = rows.end.checked_mul(columns).ok_or("row end overflow")?;
    let scale_bytes = count.checked_mul(2).ok_or("scale length overflow")?;
    let q = descriptor(blob, 4, end - start)?;
    blob.extend(matrix.values[start..end].iter().map(|value| value.to_le_bytes()[0]));
    let scale = descriptor(blob, 1, scale_bytes)?;
    for value in &matrix.scales[rows] { blob.extend_from_slice(&value.to_le_bytes()); }
    constants.push_str(&format!(
        "        tensor<fp16, [{count}, {columns}, 1, 1]> {name} = constexpr_affine_dequantize()[axis = int32(0), name = string(\"{name}\"), quantized_data = tensor<int8, [{count}, {columns}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({q}))), scale = tensor<fp16, [{count}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({scale}))), zero_point = int8(0)];\n"
    ));
    Ok(())
}

// Preserve the local compiler target/buildInfo, not the latest upstream target.
const HEADER: &str = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-version", "9.0"}})]
{
"#;
const CONV_CONSTANTS: &str = r#"        string pad_type = const()[name = string("pad_type"), val = string("valid")];
        tensor<int32, [2]> strides = const()[name = string("strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pad = const()[name = string("pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dilations = const()[name = string("dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 groups = const()[name = string("groups"), val = int32(1)];
"#;
const GELU_CONSTANTS: &str = r#"        fp16 gelu_half = const()[name = string("gelu_half"), val = fp16(0.5)];
        fp16 gelu_one = const()[name = string("gelu_one"), val = fp16(1.0)];
        fp16 gelu_cubic = const()[name = string("gelu_cubic"), val = fp16(0.044715)];
        fp16 gelu_root = const()[name = string("gelu_root"), val = fp16(0.7978845608)];
"#;

fn graph_header(input: usize, constants: &str) -> String {
    let mut source = String::from(HEADER);
    source.push_str(&format!("    func main<ios18>(tensor<fp16, [1, {input}, 1, 1]> x) {{\n"));
    source.push_str(CONV_CONSTANTS);
    source.push_str(constants);
    source
}

fn convolution(source: &mut String, output: &str, weight: &str, input: &str, rows: usize) {
    source.push_str(&format!("        tensor<fp16, [1, {rows}, 1, 1]> {output} = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = {weight}, x = {input})[name = string(\"{output}\")];\n"));
}

fn concatenate(source: &mut String, output: &str, inputs: &[&str], rows: usize) {
    source.push_str("        int32 concat_axis = const()[name = string(\"concat_axis\"), val = int32(1)];\n");
    source.push_str("        bool concat_interleave = const()[name = string(\"concat_interleave\"), val = bool(false)];\n");
    source.push_str(&format!("        tensor<fp16, [1, {rows}, 1, 1]> {output} = concat(axis = concat_axis, interleave = concat_interleave, values = ({}))[name = string(\"{output}\")];\n", inputs.join(", ")));
}

fn gelu_branch(source: &mut String, part: usize, rows: usize) {
    // An explicit op list makes each inherited FP16 materialization visible.
    // Do not simplify or replace this order with a fused activation operation.
    let p = |name: &str| format!("{name}{part}");
    for (name, op, x, y) in [
        ("gate_squared", "mul", p("gate"), p("gate")),
        ("gate_cubed", "mul", p("gate_squared"), p("gate")),
        ("cubic_scaled", "mul", p("gate_cubed"), "gelu_cubic".into()),
        ("gelu_sum", "add", p("gate"), p("cubic_scaled")),
        ("gelu_argument", "mul", p("gelu_sum"), "gelu_root".into()),
        ("gelu_tanh", "tanh", p("gelu_argument"), String::new()),
        ("gelu_factor", "add", p("gelu_tanh"), "gelu_one".into()),
        ("gate_halved", "mul", p("gate"), "gelu_half".into()),
        ("activated", "mul", p("gate_halved"), p("gelu_factor")),
        ("gated", "mul", p("activated"), p("up")),
    ] {
        let y = if y.is_empty() { y } else { format!(", y = {y}") };
        let name = p(name);
        source.push_str(&format!("        tensor<fp16, [1, {rows}, 1, 1]> {name} = {op}(x = {x}{y})[name = string(\"{name}\")];\n"));
    }
}

#[cfg(test)]
#[path = "ane_int8_candidate_tests.rs"]
mod tests;
