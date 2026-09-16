//! Per-output INT8 weights for private ANE constant-weight FFNs and projections.
//! Activations remain FP16. This is separate from the public group-32 packed
//! weight ABI: the ANE constexpr operation requires one scale per output row.
#![forbid(unsafe_code)]

use half::f16;
use half::slice::HalfFloatSliceExt;

#[derive(PartialEq)]
struct RowInt8 {
    rows: usize,
    columns: usize,
    values: Vec<i8>,
    scales: Vec<f16>,
}

impl RowInt8 {
    // Host preparation. Preserve scalar source bytes while reducing conversion
    // and quantization instructions; this makes no device speedup assumption.
    fn quantize_vectorized(weights: &[f16], rows: usize, columns: usize) -> Result<Self, String> {
        if rows == 0
            || columns == 0
            || rows % 32 != 0
            || columns % 32 != 0
            || rows.checked_mul(columns) != Some(weights.len())
        {
            return Err("ANE INT8 weights require a nonempty, 32-aligned matrix".into());
        }
        let mut values = Vec::with_capacity(weights.len());
        let mut scales = Vec::with_capacity(rows);
        // One bounded row buffer; no model-sized FP32 copy. The half crate's
        // safe slice operation dispatches to the host's vector conversion.
        let mut row_values = vec![0.0_f32; columns];
        for (index, row) in weights.chunks_exact(columns).enumerate() {
            row.convert_to_f32_slice(&mut row_values);
            // Positive IEEE bits have the same ordering as their values. This
            // integer reduction also detects every infinity/NaN, including
            // negative and signaling NaNs, without a branch per coefficient.
            let maximum_bits = row_values
                .iter()
                .map(|value| value.to_bits() & 0x7fff_ffff)
                .max()
                .unwrap_or(0);
            if maximum_bits >= f32::INFINITY.to_bits() {
                return Err(format!("ANE INT8 weight row {index} is nonfinite"));
            }
            let maximum = f32::from_bits(maximum_bits);
            let scale = if maximum == 0.0 {
                f16::ONE
            } else {
                f16::from_f32(maximum / 127.0)
            };
            if !scale.is_finite() || scale <= f16::ZERO {
                return Err(format!("ANE INT8 row {index} scale is not representable"));
            }
            let stored_scale = scale.to_f32();
            // Preserve division and ties-to-even exactly. Reciprocal multiply
            // can change bytes, graph identity, and subsequent model output.
            values.extend(row_values.iter().map(|&weight| {
                (weight / stored_scale)
                    .round_ties_even()
                    .clamp(-127.0, 127.0) as i8
            }));
            scales.push(scale);
        }
        Ok(Self {
            rows,
            columns,
            values,
            scales,
        })
    }

    fn quantize(weights: &[f16], rows: usize, columns: usize) -> Result<Self, String> {
        if rows == 0
            || columns == 0
            || rows % 32 != 0
            || columns % 32 != 0
            || rows.checked_mul(columns) != Some(weights.len())
        {
            return Err("ANE INT8 weights require a nonempty, 32-aligned matrix".into());
        }
        let mut values = Vec::with_capacity(weights.len());
        let mut scales = Vec::with_capacity(rows);
        for (index, row) in weights.chunks_exact(columns).enumerate() {
            let maximum = row.iter().try_fold(0.0_f32, |maximum, weight| {
                if !weight.is_finite() {
                    return Err(format!("ANE INT8 weight row {index} is nonfinite"));
                }
                Ok(maximum.max(weight.to_f32().abs()))
            })?;
            let scale = if maximum == 0.0 {
                f16::ONE
            } else {
                f16::from_f32(maximum / 127.0)
            };
            if !scale.is_finite() || scale <= f16::ZERO {
                return Err(format!("ANE INT8 row {index} scale is not representable"));
            }
            // Quantize against the exact stored FP16 scale, so reconstruction
            // is authoritative and independent of an unstored FP32 value.
            for weight in row {
                let integer = (weight.to_f32() / scale.to_f32())
                    .round_ties_even()
                    .clamp(-127.0, 127.0);
                values.push(integer as i8);
            }
            scales.push(scale);
        }
        Ok(Self {
            rows,
            columns,
            values,
            scales,
        })
    }

    fn dequantized(&self) -> Vec<f16> {
        self.values
            .chunks_exact(self.columns)
            .zip(&self.scales)
            .flat_map(|(row, scale)| {
                row.iter()
                    .map(move |&value| f16::from_f32(f32::from(value) * scale.to_f32()))
            })
            .collect()
    }
}

/// Per-output INT8 constants for a single QKV, output or vocabulary projection.
/// Uses the same quantizer and stored-scale semantics as the qualified FFNs.
/// This host representation alone establishes neither ANE speed nor quality.
pub struct AneInt8LinearWeights {
    matrix: RowInt8,
}

impl AneInt8LinearWeights {
    pub fn quantize(
        weights: &[f16],
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self, String> {
        if input_channels > 65536 || output_channels > 65536 {
            return Err("ANE INT8 linear dimensions exceed 65536".into());
        }
        let bytes = weights
            .len()
            .checked_add(
                output_channels
                    .checked_mul(2)
                    .ok_or("ANE INT8 linear scale size overflow")?,
            )
            .and_then(|n| n.checked_add(3 * 64))
            .ok_or("ANE INT8 linear blob size overflow")?;
        if bytes > u32::MAX as usize {
            return Err("ANE INT8 linear blob exceeds 4 GiB".into());
        }
        Ok(Self {
            matrix: RowInt8::quantize_vectorized(weights, output_channels, input_channels)?,
        })
    }

    /// Input channels, output channels; weights are row-major [output,input].
    pub fn shape(&self) -> (usize, usize) {
        (self.matrix.columns, self.matrix.rows)
    }

    pub fn dequantized(&self) -> Vec<f16> {
        self.matrix.dequantized()
    }

    pub fn source_blob_bytes(&self) -> usize {
        3 * 64 + self.matrix.values.len() + self.matrix.scales.len() * 2
    }

    /// Constant-weight source for the existing single-convolution graph.
    /// No model loading, compilation or evaluation occurs here.
    pub fn blob_and_constants(&self) -> (Vec<u8>, String) {
        let mut blob = vec![0; 64];
        blob.reserve(self.source_blob_bytes() - 64);
        blob[..4].copy_from_slice(&2_u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let q = append_descriptor(&mut blob, 4, self.matrix.values.len());
        blob.extend(
            self.matrix
                .values
                .iter()
                .map(|&value| value.to_le_bytes()[0]),
        );
        let scales = append_descriptor(&mut blob, 1, self.matrix.scales.len() * 2);
        for scale in &self.matrix.scales {
            blob.extend_from_slice(&scale.to_le_bytes());
        }
        let (input, output) = self.shape();
        let constants = format!(
            "        tensor<fp16, [{output}, {input}, 1, 1]> W = constexpr_affine_dequantize()[axis = int32(0), name = string(\"W\"), quantized_data = tensor<int8, [{output}, {input}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({q}))), scale = tensor<fp16, [{output}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({scales}))), zero_point = int8(0)];\n"
        );
        debug_assert_eq!(blob.len(), self.source_blob_bytes());
        (blob, constants)
    }
}

/// Three immutable quantized matrices, in gate/up/down order.
#[derive(PartialEq)]
pub struct AneInt8FfnWeights {
    hidden: usize,
    intermediate: usize,
    matrices: [RowInt8; 3],
}

/// Immutable per-row affine coefficients, with zero point zero. These are
/// the exact integer and FP16 scale bytes used by source construction.
pub struct AneInt8MatrixView<'a> {
    pub columns: usize,
    pub values: &'a [i8],
    pub scales: &'a [f16],
}

impl AneInt8FfnWeights {
    pub fn quantize(
        gate: &[f16],
        up: &[f16],
        down: &[f16],
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Self::quantize_with(
            gate,
            up,
            down,
            hidden,
            intermediate,
            RowInt8::quantize_vectorized,
        )
    }

    /// Scalar reference for source-byte audits and host preparation benchmarks.
    /// This performs no ANE calls.
    pub fn quantize_scalar_reference(
        gate: &[f16],
        up: &[f16],
        down: &[f16],
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Self::quantize_with(gate, up, down, hidden, intermediate, RowInt8::quantize)
    }

    fn quantize_with(
        gate: &[f16],
        up: &[f16],
        down: &[f16],
        hidden: usize,
        intermediate: usize,
        quantize: fn(&[f16], usize, usize) -> Result<RowInt8, String>,
    ) -> Result<Self, String> {
        let count = hidden
            .checked_mul(intermediate)
            .filter(|&n| n > 0 && n == gate.len() && n == up.len() && n == down.len())
            .ok_or("ANE INT8 FFN matrix shape mismatch or overflow")?;
        let bytes = count
            .checked_mul(3)
            .and_then(|n| n.checked_add(intermediate.checked_mul(4)?))
            .and_then(|n| n.checked_add(hidden.checked_mul(2)?))
            .and_then(|n| n.checked_add(7 * 64))
            .ok_or("ANE INT8 FFN storage overflow")?;
        if bytes > u32::MAX as usize || hidden > 65536 || intermediate > 65536 {
            return Err("ANE INT8 FFN exceeds tensor or blob limits".into());
        }
        Ok(Self {
            hidden,
            intermediate,
            matrices: [
                quantize(gate, intermediate, hidden)?,
                quantize(up, intermediate, hidden)?,
                quantize(down, hidden, intermediate)?,
            ],
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.hidden, self.intermediate)
    }

    /// Gate, up, down; no copying, reconstruction or accelerator operation.
    pub fn matrices(&self) -> [AneInt8MatrixView<'_>; 3] {
        self.matrices.each_ref().map(|matrix| AneInt8MatrixView {
            columns: matrix.columns,
            values: &matrix.values,
            scales: &matrix.scales,
        })
    }

    /// Dense FP16 reconstruction from precisely the stored integer/scale pair.
    /// Used by numerical references and the dense representation control.
    pub fn dequantized(&self) -> [Vec<f16>; 3] {
        self.matrices.each_ref().map(RowInt8::dequantized)
    }

    pub fn source_blob_bytes(&self) -> usize {
        7 * 64
            + self
                .matrices
                .iter()
                .map(|m| m.values.len() + m.scales.len() * 2)
                .sum::<usize>()
    }

    /// Same integer coefficients and scales, with gate/up rows concatenated.
    /// Two matrices need four descriptors instead of six; weight bytes do not
    /// shrink. This is an experimental graph layout, not a new quantizer.
    pub fn stacked_source_blob_bytes(&self) -> Result<usize, String> {
        if self.intermediate > 32768 {
            return Err("stacked ANE gate/up exceeds 65536 output channels".into());
        }
        Ok(self.source_blob_bytes() - 2 * 64)
    }

    pub(crate) fn stacked_blob_and_constants(&self) -> Result<(Vec<u8>, String), String> {
        let bytes = self.stacked_source_blob_bytes()?;
        let mut blob = Vec::with_capacity(bytes);
        blob.resize(64, 0);
        blob[..4].copy_from_slice(&4_u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let mut constants = String::new();
        for (matrices, name, rows, columns) in [
            (
                &self.matrices[..2],
                "Wgu",
                2 * self.intermediate,
                self.hidden,
            ),
            (&self.matrices[2..], "Wd", self.hidden, self.intermediate),
        ] {
            let q = append_descriptor(&mut blob, 4, rows * columns);
            for matrix in matrices {
                blob.extend(matrix.values.iter().map(|value| value.to_le_bytes()[0]));
            }
            let scales = append_descriptor(&mut blob, 1, rows * 2);
            for matrix in matrices {
                for scale in &matrix.scales {
                    blob.extend_from_slice(&scale.to_le_bytes());
                }
            }
            constants.push_str(&format!(
                "        tensor<fp16, [{rows}, {columns}, 1, 1]> {name} = constexpr_affine_dequantize()[axis = int32(0), name = string(\"{name}\"), quantized_data = tensor<int8, [{rows}, {columns}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({q}))), scale = tensor<fp16, [{rows}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({scales}))), zero_point = int8(0)];\n"
            ));
        }
        debug_assert_eq!(blob.len(), bytes);
        Ok((blob, constants))
    }

    /// CPU audit that decodes serialized stacked bytes in gate/up/down order.
    /// Reads descriptors and payloads, rather than the original quantizer arrays.
    pub fn dequantized_stacked_reference(&self) -> Result<[Vec<f16>; 3], String> {
        let (blob, _) = self.stacked_blob_and_constants()?;
        let read = |at: usize| -> Result<usize, String> {
            let bytes: [u8; 8] = blob
                .get(at..at + 8)
                .ok_or("truncated stacked INT8 descriptor")?
                .try_into()
                .map_err(|_| "invalid stacked INT8 descriptor")?;
            usize::try_from(u64::from_le_bytes(bytes)).map_err(|e| e.to_string())
        };
        let mut offset = 64;
        let mut output: [Vec<f16>; 3] = std::array::from_fn(|_| Vec::new());
        for (group, (rows, columns)) in [
            (2 * self.intermediate, self.hidden),
            (self.hidden, self.intermediate),
        ]
        .into_iter()
        .enumerate()
        {
            let count = read(offset + 8)?;
            let data = read(offset + 16)?;
            if count != rows * columns || blob[offset + 4..offset + 8] != 4_u32.to_le_bytes() {
                return Err("stacked INT8 coefficient descriptor mismatch".into());
            }
            let coefficients = blob
                .get(data..data + count)
                .ok_or("truncated stacked INT8 coefficients")?;
            offset = data + count;
            let count = read(offset + 8)?;
            let data = read(offset + 16)?;
            if count != rows * 2 || blob[offset + 4..offset + 8] != 1_u32.to_le_bytes() {
                return Err("stacked INT8 scale descriptor mismatch".into());
            }
            let scales = blob
                .get(data..data + count)
                .ok_or("truncated stacked INT8 scales")?;
            for (row, (values, scale)) in coefficients
                .chunks_exact(columns)
                .zip(scales.chunks_exact(2))
                .enumerate()
            {
                let scale = f16::from_le_bytes([scale[0], scale[1]]).to_f32();
                let matrix = if group == 1 {
                    2
                } else {
                    usize::from(row >= self.intermediate)
                };
                output[matrix].extend(
                    values
                        .iter()
                        .map(|&q| f16::from_f32(f32::from(i8::from_le_bytes([q])) * scale)),
                );
            }
            offset = data + count;
        }
        if offset != blob.len() {
            return Err("unexpected stacked INT8 trailing bytes".into());
        }
        Ok(output)
    }

    /// Experimental palette representation of exactly the same reconstructed
    /// INT8 weights. No new quantization, activation conversion or ANE call.
    pub fn lut8_source_blob_bytes(&self) -> Result<usize, String> {
        let bytes = 7 * 64
            + self
                .matrices
                .iter()
                .map(|m| m.values.len() + m.rows * 256 * 2)
                .sum::<usize>();
        if bytes > u32::MAX as usize {
            return Err("ANE LUT8 FFN blob exceeds 4 GiB".into());
        }
        Ok(bytes)
    }

    pub(crate) fn lut8_blob_and_constants(&self) -> Result<(Vec<u8>, String), String> {
        let mut blob = vec![0; 64];
        let bytes = self.lut8_source_blob_bytes()?;
        blob.reserve(bytes - 64);
        blob[..4].copy_from_slice(&6_u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let mut constants = String::new();
        for (matrix, name) in self.matrices.iter().zip(["Wg", "Wu", "Wd"]) {
            // Apple's MILBlob BlobDataType::UInt8 is 3 (not the MIL proto enum).
            let indices = append_descriptor(&mut blob, 3, matrix.values.len());
            blob.extend(matrix.values.iter().map(|&q| (i16::from(q) + 128) as u8));
            let lut = append_descriptor(&mut blob, 1, matrix.rows * 256 * 2);
            for scale in &matrix.scales {
                // q=-128 never occurs. Keep that unused entry finite.
                blob.extend_from_slice(&f16::ZERO.to_le_bytes());
                for q in -127_i16..=127 {
                    let value = f16::from_f32(f32::from(q) * scale.to_f32());
                    if !value.is_finite() {
                        return Err("ANE LUT8 reconstruction contains a nonfinite centroid".into());
                    }
                    blob.extend_from_slice(&value.to_le_bytes());
                }
            }
            let (rows, columns) = (matrix.rows, matrix.columns);
            constants.push_str(&format!(
                "        tensor<uint8, [{rows}, {columns}, 1, 1]> {name}_idx = const()[name = string(\"{name}_idx\"), val = tensor<uint8, [{rows}, {columns}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({indices})))];\n\
                        tensor<fp16, [{rows}, 1, 1, 1, 256, 1]> {name}_lut = const()[name = string(\"{name}_lut\"), val = tensor<fp16, [{rows}, 1, 1, 1, 256, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({lut})))];\n\
                        tensor<fp16, [{rows}, {columns}, 1, 1]> {name} = constexpr_lut_to_dense(indices = {name}_idx, lut = {name}_lut)[name = string(\"{name}\")];\n"
            ));
        }
        debug_assert_eq!(blob.len(), bytes);
        Ok((blob, constants))
    }

    /// Decode the serialized bytes rather than the original integer/scale
    /// arrays, so a representation audit covers indices and row-table routing.
    pub fn dequantized_lut8_reference(&self) -> Result<[Vec<f16>; 3], String> {
        let (blob, _) = self.lut8_blob_and_constants()?;
        let read = |at: usize| -> Result<usize, String> {
            let bytes: [u8; 8] = blob
                .get(at..at + 8)
                .ok_or("truncated LUT8 descriptor")?
                .try_into()
                .map_err(|_| "invalid LUT8 descriptor")?;
            usize::try_from(u64::from_le_bytes(bytes)).map_err(|error| error.to_string())
        };
        let mut offset = 64;
        let mut reconstructed: [Vec<f16>; 3] = std::array::from_fn(|_| Vec::new());
        for (output, matrix) in reconstructed.iter_mut().zip(&self.matrices) {
            let count = read(offset + 8)?;
            let data = read(offset + 16)?;
            if count != matrix.values.len() {
                return Err("LUT8 index length mismatch".into());
            }
            let indices = blob
                .get(data..data + count)
                .ok_or("truncated LUT8 indices")?;
            offset = data + count;
            let table_bytes = read(offset + 8)?;
            let tables = read(offset + 16)?;
            if table_bytes != matrix.rows * 512 {
                return Err("LUT8 palette length mismatch".into());
            }
            let palettes = blob
                .get(tables..tables + table_bytes)
                .ok_or("truncated LUT8 palettes")?;
            output.reserve(count);
            for (position, &index) in indices.iter().enumerate() {
                let at = (position / matrix.columns * 256 + usize::from(index)) * 2;
                output.push(f16::from_le_bytes([palettes[at], palettes[at + 1]]));
            }
            offset = tables + table_bytes;
        }
        if offset != blob.len() {
            return Err("unexpected LUT8 trailing bytes".into());
        }
        Ok(reconstructed)
    }

    /// Core ML blob codes: signed INT8=4, FP16=1. Chunk offsets address the
    /// 64-byte descriptor, whose data offset addresses the following payload.
    pub(crate) fn blob_and_constants(&self) -> (Vec<u8>, String) {
        let mut blob = Vec::with_capacity(self.source_blob_bytes());
        blob.resize(64, 0);
        blob[..4].copy_from_slice(&6_u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let mut constants = String::new();
        for (matrix, name) in self.matrices.iter().zip(["Wg", "Wu", "Wd"]) {
            let q_offset = append_descriptor(&mut blob, 4, matrix.values.len());
            blob.extend(matrix.values.iter().map(|&value| value.to_le_bytes()[0]));
            let scale_offset = append_descriptor(&mut blob, 1, matrix.scales.len() * 2);
            for scale in &matrix.scales {
                blob.extend_from_slice(&scale.to_le_bytes());
            }
            let (rows, columns) = (matrix.rows, matrix.columns);
            constants.push_str(&format!(
                "        tensor<fp16, [{rows}, {columns}, 1, 1]> {name} = constexpr_affine_dequantize()[axis = int32(0), name = string(\"{name}\"), quantized_data = tensor<int8, [{rows}, {columns}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({q_offset}))), scale = tensor<fp16, [{rows}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({scale_offset}))), zero_point = int8(0)];\n"
            ));
        }
        debug_assert_eq!(blob.len(), self.source_blob_bytes());
        (blob, constants)
    }
}

fn append_descriptor(blob: &mut Vec<u8>, dtype: u32, bytes: usize) -> usize {
    let offset = blob.len();
    debug_assert_eq!(offset % 64, 0);
    blob.resize(offset + 64, 0);
    blob[offset..offset + 4].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
    blob[offset + 4..offset + 8].copy_from_slice(&dtype.to_le_bytes());
    blob[offset + 8..offset + 16].copy_from_slice(&(bytes as u64).to_le_bytes());
    blob[offset + 16..offset + 24].copy_from_slice(&((offset + 64) as u64).to_le_bytes());
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_serialization_preserves_distinct_gate_up_and_down_rows() {
        for (hidden, intermediate) in [(32, 64), (64, 32), (3840, 32)] {
            let sources: [Vec<f16>; 3] = std::array::from_fn(|matrix| {
                (0..hidden * intermediate)
                    .map(|i| {
                        let columns = if matrix == 2 { intermediate } else { hidden };
                        let scale = if (i / columns + matrix) % 2 == 0 {
                            0.001
                        } else {
                            0.125
                        };
                        f16::from_f32(
                            (((i * (matrix + 1) + matrix * 47) % 255) as i32 - 127) as f32 * scale,
                        )
                    })
                    .collect()
            });
            let weights = AneInt8FfnWeights::quantize(
                &sources[0],
                &sources[1],
                &sources[2],
                hidden,
                intermediate,
            )
            .unwrap();
            let (blob, constants) = weights.stacked_blob_and_constants().unwrap();
            assert_eq!(blob.len(), weights.source_blob_bytes() - 128);
            assert_eq!(blob[..4], 4_u32.to_le_bytes());
            assert!(constants.contains(&format!(
                "tensor<fp16, [{}, {hidden}, 1, 1]> Wgu",
                2 * intermediate
            )));
            for (actual, expected) in weights
                .dequantized_stacked_reference()
                .unwrap()
                .iter()
                .zip(weights.dequantized())
            {
                assert_eq!(actual.len(), expected.len());
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
        let zero = vec![f16::ZERO; 32 * 32768];
        let mut weights = AneInt8FfnWeights::quantize(&zero, &zero, &zero, 32, 32768).unwrap();
        assert!(weights.stacked_source_blob_bytes().is_ok());
        weights.intermediate = 32800;
        assert!(weights.stacked_blob_and_constants().is_err());
    }

    #[test]
    fn int8_linear_blob_roundtrips_nonsquare_rows_and_matches_ffn_quantization() {
        for (input, output) in [(32, 64), (64, 32), (3840, 32)] {
            let source: Vec<_> = (0..input * output)
                .map(|i| f16::from_f32(((i * 17) % 257) as f32 / 4096.0 - 0.03))
                .collect();
            let weights = AneInt8LinearWeights::quantize(&source, input, output).unwrap();
            assert_eq!(weights.shape(), (input, output));
            let (blob, constants) = weights.blob_and_constants();
            assert_eq!(blob.len(), weights.source_blob_bytes());
            assert_eq!(&blob[..4], &2_u32.to_le_bytes());
            assert!(constants.contains(&format!("scale = tensor<fp16, [{output}]>")));
            assert!(constants.contains("axis = int32(0)"));
            let read = |at| u64::from_le_bytes(blob[at..at + 8].try_into().unwrap()) as usize;
            let q = read(80);
            let count = read(72);
            assert_eq!(count, input * output);
            assert_eq!(blob[68], 4);
            let scales_descriptor = q + count;
            assert_eq!(scales_descriptor % 64, 0);
            assert_eq!(blob[scales_descriptor + 4], 1);
            let scales = read(scales_descriptor + 16);
            for (position, expected) in weights.dequantized().iter().enumerate() {
                let row = position / input;
                let scale =
                    f16::from_le_bytes([blob[scales + 2 * row], blob[scales + 2 * row + 1]]);
                let integer = i8::from_le_bytes([blob[q + position]]);
                let actual = f16::from_f32(f32::from(integer) * scale.to_f32());
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            let ffn =
                AneInt8FfnWeights::quantize(&source, &source, &source, input, output).unwrap();
            assert!(weights
                .dequantized()
                .iter()
                .zip(&ffn.dequantized()[0])
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert!(AneInt8LinearWeights::quantize(&[], 0, 32).is_err());
        assert!(AneInt8LinearWeights::quantize(&[f16::ONE; 32], 1, 32).is_err());
        assert!(AneInt8LinearWeights::quantize(&[f16::INFINITY; 1024], 32, 32).is_err());
        assert!(AneInt8LinearWeights::quantize(&[], 65568, 32).is_err());
    }

    #[test]
    fn exact_lut8_serialization_matches_every_signed_index_and_row_scale() {
        let source: Vec<_> = (0..32 * 256)
            .map(|i| {
                let integer = (i % 255) as i32 - 127;
                let scale = if i / 256 % 2 == 0 { 0.001 } else { 0.125 };
                f16::from_f32(integer as f32 * scale)
            })
            .collect();
        let weights = AneInt8FfnWeights::quantize(&source, &source, &source, 32, 256).unwrap();
        let (blob, constants) = weights.lut8_blob_and_constants().unwrap();
        assert_eq!(blob.len(), weights.lut8_source_blob_bytes().unwrap());
        assert_eq!(blob[68], 3);
        assert!(constants.contains("tensor<fp16, [256, 1, 1, 1, 256, 1]>"));
        assert!(constants.contains("tensor<fp16, [32, 1, 1, 1, 256, 1]>"));
        for (expected, actual) in weights
            .dequantized()
            .iter()
            .zip(weights.dequantized_lut8_reference().unwrap())
        {
            assert!(expected
                .iter()
                .zip(actual)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        // Enumerate all legal signed coefficients directly as well, so this
        // coverage does not depend on which bins the quantizer happens to use.
        let matrix = |rows, columns| RowInt8 {
            rows,
            columns,
            values: (0..rows * columns)
                .map(|i| ((i % 255) as i16 - 127) as i8)
                .collect(),
            scales: (0..rows)
                .map(|r| f16::from_f32(if r % 2 == 0 { 0.001 } else { 0.125 }))
                .collect(),
        };
        let weights = AneInt8FfnWeights {
            hidden: 32,
            intermediate: 256,
            matrices: [matrix(256, 32), matrix(256, 32), matrix(32, 256)],
        };
        for (expected, actual) in weights
            .dequantized()
            .iter()
            .zip(weights.dequantized_lut8_reference().unwrap())
        {
            assert!(expected
                .iter()
                .zip(actual)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        let zero = vec![f16::ZERO; 1024];
        let weights = AneInt8FfnWeights::quantize(&zero, &zero, &zero, 32, 32).unwrap();
        assert!(weights
            .dequantized_lut8_reference()
            .unwrap()
            .iter()
            .flatten()
            .all(|v| v.to_bits() == 0));
    }

    fn compare_quantizers(weights: &[f16], rows: usize, columns: usize) {
        let scalar = RowInt8::quantize(weights, rows, columns);
        let vector = RowInt8::quantize_vectorized(weights, rows, columns);
        match (scalar, vector) {
            (Ok(scalar), Ok(vector)) => {
                assert_eq!(scalar.values, vector.values);
                assert_eq!(
                    scalar
                        .scales
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    vector
                        .scales
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            (Err(scalar), Err(vector)) => assert_eq!(scalar, vector),
            _ => panic!("quantizers disagree about whether the matrix is valid"),
        }
    }

    #[test]
    fn vectorized_quantizer_covers_all_half_patterns_and_rounding_ties() {
        // Every FP16 bit pattern appears in an anchored row. Anchors prevent
        // tiny finite values from rejecting an entire test matrix by underflow.
        for start in (0..=u16::MAX as usize).step_by(32) {
            let mut weights = vec![f16::ONE; 32 * 32];
            for row in 0..32 {
                weights[row * 32] = f16::from_bits((start + row) as u16);
            }
            compare_quantizers(&weights, 32, 32);
        }
        let ties: Vec<_> = (0..1024)
            .map(|index| {
                if index % 32 == 0 {
                    f16::from_f32(127.0)
                } else {
                    f16::from_f32((index % 31) as f32 - 15.5)
                }
            })
            .collect();
        compare_quantizers(&ties, 32, 32);
        compare_quantizers(&vec![f16::ZERO; 1024], 32, 32);
        compare_quantizers(&vec![f16::from_bits(1); 1024], 32, 32);
    }

    #[test]
    fn vectorized_quantizer_matches_gemma_row_widths_and_source_blob() {
        let mut random = 7_u64;
        for columns in [3840, 15360] {
            let weights: Vec<_> = (0..32 * columns)
                .map(|_| {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    f16::from_f32(((random >> 32) as i32) as f32 * (0.02 / i32::MAX as f32))
                })
                .collect();
            compare_quantizers(&weights, 32, columns);
        }
        let weights: Vec<_> = (0..32 * 64)
            .map(|i| f16::from_f32((i % 29) as f32 / 32.0 - 0.4))
            .collect();
        let scalar =
            AneInt8FfnWeights::quantize_scalar_reference(&weights, &weights, &weights, 32, 64)
                .unwrap();
        let vector = AneInt8FfnWeights::quantize(&weights, &weights, &weights, 32, 64).unwrap();
        assert!(scalar == vector);
        assert_eq!(scalar.blob_and_constants(), vector.blob_and_constants());
    }

    #[test]
    fn stored_scales_round_ties_even_and_zero_rows_are_exact() {
        let mut weights = vec![f16::ZERO; 32 * 32];
        weights[..6].copy_from_slice(&[127.0, -127.0, 0.5, 1.5, -0.5, -1.5].map(f16::from_f32));
        let quantized = RowInt8::quantize(&weights, 32, 32).unwrap();
        assert_eq!(&quantized.values[..6], &[127, -127, 0, 2, 0, -2]);
        assert!(quantized.scales.iter().all(|&scale| scale == f16::ONE));
        assert!(quantized.dequantized()[32..]
            .iter()
            .all(|&w| w == f16::ZERO));
        weights[8] = f16::NAN;
        assert!(RowInt8::quantize(&weights, 32, 32).is_err());
        assert!(RowInt8::quantize(&[f16::from_bits(1); 1024], 32, 32).is_err());
    }

    #[test]
    fn blob_descriptors_decode_all_three_nonsquare_matrices() {
        let weights: Vec<_> = (0..32 * 64)
            .map(|i| f16::from_f32((i % 29) as f32 / 32.0 - 0.4))
            .collect();
        let quantized = AneInt8FfnWeights::quantize(&weights, &weights, &weights, 32, 64).unwrap();
        let (blob, constants) = quantized.blob_and_constants();
        let read_u64 =
            |offset| u64::from_le_bytes(blob[offset..offset + 8].try_into().unwrap()) as usize;
        let mut offset = 64;
        for matrix in &quantized.matrices {
            assert_eq!(&blob[offset..offset + 4], &0xDEAD_BEEF_u32.to_le_bytes());
            assert_eq!(blob[offset + 4], 4);
            let count = read_u64(offset + 8);
            let data = read_u64(offset + 16);
            assert_eq!(data, offset + 64);
            assert_eq!(count, matrix.values.len());
            for (&byte, &expected) in blob[data..data + count].iter().zip(&matrix.values) {
                assert_eq!(i8::from_le_bytes([byte]), expected);
            }
            offset = data + count;
            assert_eq!(blob[offset + 4], 1);
            let count = read_u64(offset + 8);
            let data = read_u64(offset + 16);
            assert_eq!(count, matrix.rows * 2);
            for (bytes, expected) in blob[data..data + count].chunks_exact(2).zip(&matrix.scales) {
                assert_eq!(f16::from_le_bytes([bytes[0], bytes[1]]), *expected);
            }
            offset = data + count;
        }
        assert_eq!(offset, blob.len());
        assert_eq!(constants.matches("constexpr_affine_dequantize").count(), 3);
        assert_eq!(constants.matches("axis = int32(0)").count(), 3);
    }
}
