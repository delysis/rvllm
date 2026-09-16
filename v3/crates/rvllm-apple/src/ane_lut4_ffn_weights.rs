//! Scalar LUT4 constants for the private ANE FFN experiment.
//! The iOS18 constexpr_lut_to_dense operand is rank-four uint4 indices with
//! a rank-six FP16 LUT. Indices are packed low nibble first (BLOB dtype 11).
//! Fit against a histogram of the exact FP16 values, without an N-by-16
//! distance matrix. Reconstruction always uses the stored FP16 codebook.

use half::f16;

struct PaletteMatrix {
    rows: usize,
    columns: usize,
    indices: Vec<u8>,
    codebook: [f16; 16],
}

impl PaletteMatrix {
    fn quantize(weights: &[f16], rows: usize, columns: usize) -> Result<Self, String> {
        let mut histogram = vec![0_usize; 65536];
        for &weight in weights {
            if !weight.is_finite() {
                return Err("ANE LUT4 weights must be finite".into());
            }
            histogram[usize::from(weight.to_bits())] += 1;
        }
        let mut values: Vec<_> = histogram
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(bits, &count)| (f64::from(f16::from_bits(bits as u16).to_f32()), count))
            .collect();
        values.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut centroids = [0.0_f64; 16];
        let mut cumulative = 0;
        let mut bin = 0;
        for (i, centroid) in centroids.iter_mut().enumerate() {
            // Validated matrices contain a multiple of 1024 elements.
            let rank = (weights.len() / 32) * (2 * i + 1);
            while cumulative + values[bin].1 <= rank {
                cumulative += values[bin].1;
                bin += 1;
            }
            *centroid = values[bin].0;
        }
        // Weighted one-dimensional Lloyd iterations. Empty bins retain their
        // old center. Sorting keeps midpoint lookup and tie-breaking stable.
        for _ in 0..50 {
            let mut sums = [0.0_f64; 16];
            let mut counts = [0_usize; 16];
            for &(value, count) in &values {
                let index = nearest(value, &centroids);
                sums[index] += value * count as f64;
                counts[index] += count;
            }
            let previous = centroids;
            for i in 0..16 {
                if counts[i] != 0 {
                    centroids[i] = sums[i] / counts[i] as f64;
                }
            }
            centroids.sort_by(f64::total_cmp);
            if centroids == previous {
                break;
            }
        }
        let codebook = centroids.map(|value| f16::from_f64(value));
        let stored = codebook.map(|value| f64::from(value.to_f32()));
        let mut lookup = vec![0_u8; 65536];
        for (bits, count) in histogram.into_iter().enumerate() {
            if count != 0 {
                lookup[bits] =
                    nearest(f64::from(f16::from_bits(bits as u16).to_f32()), &stored) as u8;
            }
        }
        let indices = weights
            .chunks_exact(2)
            .map(|pair| {
                lookup[usize::from(pair[0].to_bits())]
                    | (lookup[usize::from(pair[1].to_bits())] << 4)
            })
            .collect();
        Ok(Self {
            rows,
            columns,
            indices,
            codebook,
        })
    }

    fn dequantized(&self) -> Vec<f16> {
        self.indices
            .iter()
            .flat_map(|&byte| {
                [
                    self.codebook[usize::from(byte & 15)],
                    self.codebook[usize::from(byte >> 4)],
                ]
            })
            .collect()
    }
}

fn nearest(value: f64, codebook: &[f64; 16]) -> usize {
    // A midpoint tie selects the lower entry. Duplicate centroids are legal.
    (0..15)
        .find(|&i| value <= (codebook[i] + codebook[i + 1]) * 0.5)
        .unwrap_or(15)
}

/// Three independently fitted, immutable scalar codebooks in gate/up/down order.
/// No fallback to a different format and no full-model quality assertion.
pub struct AneLut4FfnWeights {
    hidden: usize,
    intermediate: usize,
    matrices: [PaletteMatrix; 3],
}

impl AneLut4FfnWeights {
    pub fn quantize(
        gate: &[f16],
        up: &[f16],
        down: &[f16],
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        let count = hidden
            .checked_mul(intermediate)
            .filter(|&n| n != 0 && n == gate.len() && n == up.len() && n == down.len())
            .ok_or("ANE LUT4 FFN shape mismatch or overflow")?;
        if hidden % 32 != 0 || intermediate % 32 != 0 || hidden > 65536 || intermediate > 65536 {
            return Err("ANE LUT4 FFN dimensions must be 32-aligned and at most 65536".into());
        }
        let bytes = (count / 2)
            .checked_add(192)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(64))
            .ok_or("ANE LUT4 FFN storage overflow")?;
        if bytes > u32::MAX as usize {
            return Err("ANE LUT4 FFN blob exceeds 4 GiB".into());
        }
        Ok(Self {
            hidden,
            intermediate,
            matrices: [
                PaletteMatrix::quantize(gate, intermediate, hidden)?,
                PaletteMatrix::quantize(up, intermediate, hidden)?,
                PaletteMatrix::quantize(down, hidden, intermediate)?,
            ],
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.hidden, self.intermediate)
    }

    pub fn dequantized(&self) -> [Vec<f16>; 3] {
        self.matrices.each_ref().map(PaletteMatrix::dequantized)
    }

    pub fn source_blob_bytes(&self) -> usize {
        64 + self
            .matrices
            .iter()
            .map(|m| m.indices.len() + 192)
            .sum::<usize>()
    }

    pub(crate) fn blob_and_constants(&self) -> (Vec<u8>, String) {
        let mut blob = Vec::with_capacity(self.source_blob_bytes());
        blob.resize(64, 0);
        blob[..4].copy_from_slice(&6_u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let mut constants = String::new();
        for (matrix, name) in self.matrices.iter().zip(["Wg", "Wu", "Wd"]) {
            let indices = append_descriptor(&mut blob, 11, matrix.indices.len());
            blob.extend_from_slice(&matrix.indices);
            let lut = append_descriptor(&mut blob, 1, 32);
            for value in matrix.codebook {
                blob.extend_from_slice(&value.to_le_bytes());
            }
            blob.resize(blob.len().next_multiple_of(64), 0);
            let (rows, columns) = (matrix.rows, matrix.columns);
            constants.push_str(&format!(
                "        tensor<uint4, [{rows}, {columns}, 1, 1]> {name}_idx = const()[name = string(\"{name}_idx\"), val = tensor<uint4, [{rows}, {columns}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({indices})))];\n\
                        tensor<fp16, [1, 1, 1, 1, 16, 1]> {name}_lut = const()[name = string(\"{name}_lut\"), val = tensor<fp16, [1, 1, 1, 1, 16, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({lut})))];\n\
                        tensor<fp16, [{rows}, {columns}, 1, 1]> {name} = constexpr_lut_to_dense(indices = {name}_idx, lut = {name}_lut)[name = string(\"{name}\")];\n"
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
    fn palette_reconstruction_handles_constant_and_discrete_weights() {
        for values in [
            vec![f16::from_f32(0.125); 2048],
            (0..2048)
                .map(|i| f16::from_f32((i % 16) as f32 - 7.0))
                .collect(),
        ] {
            let weights = AneLut4FfnWeights::quantize(&values, &values, &values, 32, 64).unwrap();
            assert_eq!(
                weights.dequantized(),
                [values.clone(), values.clone(), values]
            );
        }
        let bad = vec![f16::INFINITY; 1024];
        assert!(AneLut4FfnWeights::quantize(&bad, &bad, &bad, 32, 32).is_err());
        assert!(AneLut4FfnWeights::quantize(&[], &[], &[], 0, 32).is_err());
        assert!(AneLut4FfnWeights::quantize(&[f16::ONE], &[f16::ONE], &[f16::ONE], 1, 1).is_err());
    }

    #[test]
    fn stored_blob_roundtrips_nibbles_and_codebooks_at_aligned_offsets() {
        let source: Vec<_> = (0..2048)
            .map(|i| f16::from_f32(((i * 7) % 137) as f32 / 100.0 - 0.6))
            .collect();
        let weights = AneLut4FfnWeights::quantize(&source, &source, &source, 32, 64).unwrap();
        let (blob, _) = weights.blob_and_constants();
        let read = |at| u64::from_le_bytes(blob[at..at + 8].try_into().unwrap()) as usize;
        let mut offset = 64;
        for expected in weights.dequantized() {
            assert_eq!(offset % 64, 0);
            assert_eq!(blob[offset + 4], 11);
            let packed = &blob[read(offset + 16)..read(offset + 16) + read(offset + 8)];
            offset = read(offset + 16) + read(offset + 8);
            assert_eq!(offset % 64, 0);
            assert_eq!(blob[offset + 4], 1);
            assert_eq!(read(offset + 8), 32);
            let lut = &blob[read(offset + 16)..read(offset + 16) + 32];
            let decoded: Vec<_> = packed
                .iter()
                .flat_map(|&byte| [byte & 15, byte >> 4])
                .map(|index| {
                    let i = usize::from(index) * 2;
                    f16::from_le_bytes([lut[i], lut[i + 1]])
                })
                .collect();
            assert_eq!(decoded, expected);
            offset = (read(offset + 16) + 32).next_multiple_of(64);
        }
        assert_eq!(offset, blob.len());
    }
}
