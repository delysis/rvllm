use half::f16;
use rvllm_core::{AppleCtx, AppleError, Result, RvllmError, MAX_RANK};
use serde::{Deserialize, Serialize};

const GLOBAL_HEADER_BYTES: usize = 64;
const CHUNK_HEADER_BYTES: usize = 64;
const CHUNK_MAGIC: u32 = 0xDEAD_BEEF;
const FP16_BYTES: usize = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct AneFp16WeightSpec<'a> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub weights: &'a [f32],
}

impl<'a> AneFp16WeightSpec<'a> {
    #[must_use]
    pub fn new(name: &'a str, shape: &[usize], weights: &'a [f32]) -> Self {
        Self {
            name,
            shape: shape.to_vec(),
            weights,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeightChunkDesc {
    pub name: String,
    pub shape: Vec<usize>,
    pub chunk_offset: u64,
    pub data_offset: u64,
    pub data_bytes: usize,
    pub elements: usize,
}

#[must_use]
pub fn build_weight_blob_fp16(weight_sets: &[&[f32]]) -> Vec<u8> {
    let named: Vec<(&str, &[f32])> = weight_sets
        .iter()
        .enumerate()
        .map(|(i, w)| match i {
            0 => ("w0", *w),
            1 => ("w1", *w),
            2 => ("w2", *w),
            _ => ("w", *w),
        })
        .collect();
    build_weight_blob_fp16_named(&named).0
}

#[must_use]
pub fn build_weight_blob_fp16_named(
    weight_sets: &[(&str, &[f32])],
) -> (Vec<u8>, Vec<WeightChunkDesc>) {
    let specs: Vec<AneFp16WeightSpec<'_>> = weight_sets
        .iter()
        .map(|&(name, weights)| AneFp16WeightSpec::new(name, &[weights.len()], weights))
        .collect();
    build_weight_blob_fp16_validated(&specs)
}

pub fn build_weight_blob_fp16_described(
    weight_sets: &[AneFp16WeightSpec<'_>],
) -> Result<(Vec<u8>, Vec<WeightChunkDesc>)> {
    for spec in weight_sets {
        validate_weight_spec(spec)?;
    }
    let total_bytes = total_blob_bytes(weight_sets)?;
    if total_bytes > u32::MAX as usize {
        return Err(invalid_weight_blob(
            "weight blob exceeds u32-addressed BLOBFILE offsets",
        ));
    }
    Ok(build_weight_blob_fp16_validated(weight_sets))
}

fn build_weight_blob_fp16_validated(
    weight_sets: &[AneFp16WeightSpec<'_>],
) -> (Vec<u8>, Vec<WeightChunkDesc>) {
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(weight_sets.len());

    for spec in weight_sets {
        let fp16_bytes = spec.weights.len() * FP16_BYTES;
        let mut chunk = vec![0u8; CHUNK_HEADER_BYTES + fp16_bytes];
        chunk[0..4].copy_from_slice(&CHUNK_MAGIC.to_le_bytes());
        chunk[4] = 0x01;
        chunk[8..12].copy_from_slice(&(fp16_bytes as u32).to_le_bytes());
        for (i, &val) in spec.weights.iter().enumerate() {
            let bits = f16::from_f32(val).to_bits();
            let start = CHUNK_HEADER_BYTES + i * FP16_BYTES;
            chunk[start..start + FP16_BYTES].copy_from_slice(&bits.to_le_bytes());
        }
        chunks.push(chunk);
    }

    let total = GLOBAL_HEADER_BYTES + chunks.iter().map(Vec::len).sum::<usize>();
    let mut blob = vec![0u8; total];
    blob[0] = 0x01;
    blob[4] = 0x02;

    let mut offset = GLOBAL_HEADER_BYTES;
    let mut descs = Vec::with_capacity(weight_sets.len());
    for (spec, chunk) in weight_sets.iter().zip(chunks) {
        let len = chunk.len();
        blob[offset..offset + len].copy_from_slice(&chunk);
        let data_abs = (offset + CHUNK_HEADER_BYTES) as u32;
        blob[offset + 16..offset + 20].copy_from_slice(&data_abs.to_le_bytes());
        descs.push(WeightChunkDesc {
            name: spec.name.to_owned(),
            shape: spec.shape.clone(),
            chunk_offset: offset as u64,
            data_offset: data_abs as u64,
            data_bytes: spec.weights.len() * FP16_BYTES,
            elements: spec.weights.len(),
        });
        offset += len;
    }
    (blob, descs)
}

fn validate_weight_spec(spec: &AneFp16WeightSpec<'_>) -> Result<()> {
    if spec.name.is_empty() {
        return Err(invalid_weight_blob("weight descriptor name is empty"));
    }
    if spec.shape.len() > MAX_RANK {
        return Err(invalid_weight_blob(
            "weight descriptor rank exceeds MAX_RANK",
        ));
    }
    let expected_elements = shape_elements(&spec.shape)?;
    if expected_elements != spec.weights.len() {
        return Err(invalid_weight_blob(
            "weight descriptor shape does not match payload elements",
        ));
    }
    Ok(())
}

fn total_blob_bytes(weight_sets: &[AneFp16WeightSpec<'_>]) -> Result<usize> {
    let mut total = GLOBAL_HEADER_BYTES;
    for spec in weight_sets {
        let data_bytes = spec
            .weights
            .len()
            .checked_mul(FP16_BYTES)
            .ok_or_else(|| invalid_weight_blob("weight payload byte count overflowed"))?;
        total = total
            .checked_add(CHUNK_HEADER_BYTES)
            .and_then(|bytes| bytes.checked_add(data_bytes))
            .ok_or_else(|| invalid_weight_blob("weight blob byte count overflowed"))?;
    }
    Ok(total)
}

fn shape_elements(shape: &[usize]) -> Result<usize> {
    let mut elements = 1usize;
    for &dim in shape {
        elements = elements.checked_mul(dim).ok_or_else(|| {
            invalid_weight_blob("weight descriptor shape element count overflowed")
        })?;
    }
    Ok(elements)
}

fn invalid_weight_blob(reason: &'static str) -> RvllmError {
    RvllmError::apple(
        AppleError::InvalidWeightBlob { reason },
        AppleCtx {
            backend: "private-ane",
            op: "build_weight_blob_fp16_described",
            device: "apple-silicon",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le_u32_at(bytes: &[u8], idx: usize) -> u32 {
        u32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]])
    }

    #[test]
    fn weight_blob_header_offsets_and_fp16_values_are_stable() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let (blob, desc) = build_weight_blob_fp16_named(&[("gate", &w)]);
        assert_eq!(blob[0], 0x01);
        assert_eq!(blob[4], 0x02);
        assert_eq!(&blob[64..68], &0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(le_u32_at(&blob, 72), 8);
        assert_eq!(le_u32_at(&blob, 80), 128);
        assert_eq!(desc[0].name, "gate");
        assert_eq!(desc[0].chunk_offset, 64);
        assert_eq!(desc[0].data_offset, 128);
        let h = f16::from_bits(u16::from_le_bytes([blob[128], blob[129]])).to_f32();
        assert!((h - 1.0).abs() < 0.01);
    }

    #[test]
    fn multi_chunk_descriptors_match_blob_offsets() {
        let w1 = vec![1.0f32; 4];
        let w2 = vec![2.0f32; 8];
        let (blob, desc) = build_weight_blob_fp16_named(&[("gate", &w1), ("up", &w2)]);
        assert_eq!(desc.len(), 2);
        assert_eq!(desc[0].data_offset, 128);
        assert_eq!(desc[1].chunk_offset, 64 + 64 + 8);
        assert_eq!(desc[1].data_offset, desc[1].chunk_offset + 64);
        assert_eq!(blob.len(), 64 + (64 + 8) + (64 + 16));
    }

    #[test]
    fn described_blob_records_headers_names_offsets_and_shapes() {
        let gate = vec![1.0f32; 8 * 4];
        let up = vec![2.0f32; 8 * 4];
        let (blob, desc) = match build_weight_blob_fp16_described(&[
            AneFp16WeightSpec::new("layer0.mlp.gate_proj.weight", &[8, 4, 1, 1], &gate),
            AneFp16WeightSpec::new("layer0.mlp.up_proj.weight", &[8, 4, 1, 1], &up),
        ]) {
            Ok(out) => out,
            Err(err) => panic!("{err}"),
        };

        assert_eq!(blob[0], 0x01);
        assert_eq!(blob[4], 0x02);
        assert_eq!(desc[0].name, "layer0.mlp.gate_proj.weight");
        assert_eq!(desc[0].shape, vec![8, 4, 1, 1]);
        assert_eq!(desc[0].chunk_offset, 64);
        assert_eq!(desc[0].data_offset, 128);
        assert_eq!(
            &blob[desc[0].chunk_offset as usize..desc[0].chunk_offset as usize + 4],
            &0xDEAD_BEEFu32.to_le_bytes()
        );

        assert_eq!(desc[1].name, "layer0.mlp.up_proj.weight");
        assert_eq!(desc[1].shape, vec![8, 4, 1, 1]);
        assert_eq!(desc[1].chunk_offset, 64 + 64 + 64);
        assert_eq!(desc[1].data_offset, desc[1].chunk_offset + 64);
        assert_eq!(le_u32_at(&blob, desc[1].chunk_offset as usize + 8), 64);
        assert_eq!(
            le_u32_at(&blob, desc[1].chunk_offset as usize + 16),
            desc[1].data_offset as u32
        );
    }
}
