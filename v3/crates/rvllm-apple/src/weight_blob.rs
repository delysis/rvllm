use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeightChunkDesc {
    pub name: String,
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
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(weight_sets.len());

    for (_, weights) in weight_sets {
        let fp16_bytes = weights.len() * 2;
        let mut chunk = vec![0u8; 64 + fp16_bytes];
        chunk[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        chunk[4] = 0x01;
        chunk[8..12].copy_from_slice(&(fp16_bytes as u32).to_le_bytes());
        for (i, &val) in weights.iter().enumerate() {
            let bits = f16::from_f32(val).to_bits();
            chunk[64 + i * 2..64 + i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        chunks.push(chunk);
    }

    let total = 64 + chunks.iter().map(Vec::len).sum::<usize>();
    let mut blob = vec![0u8; total];
    blob[0] = 0x01;
    blob[4] = 0x02;

    let mut offset = 64usize;
    let mut descs = Vec::with_capacity(weight_sets.len());
    for ((name, weights), chunk) in weight_sets.iter().zip(chunks) {
        let len = chunk.len();
        blob[offset..offset + len].copy_from_slice(&chunk);
        let data_abs = (offset + 64) as u32;
        blob[offset + 16..offset + 20].copy_from_slice(&data_abs.to_le_bytes());
        descs.push(WeightChunkDesc {
            name: (*name).to_owned(),
            chunk_offset: offset as u64,
            data_offset: data_abs as u64,
            data_bytes: weights.len() * 2,
            elements: weights.len(),
        });
        offset += len;
    }
    (blob, descs)
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
}
