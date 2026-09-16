//! Host-only layout and MIL generation for a single-input decode attention
//! program. This module never loads a framework or dispatches to a device.
//!
//! Each channel is one (KV head, head dimension). Its spatial row holds query
//! groups in lanes 0..G, K at 32..32+T, and V at 32+T..32+2T. One extra channel
//! holds the mask. All channel rows are aligned to 64 bytes.

use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedAttentionLayout {
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    window: Option<usize>,
    spatial: usize,
    input_bytes: usize,
    output_bytes: usize,
}

impl PackedAttentionLayout {
    pub fn new(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self, String> {
        if kv_heads == 0
            || query_heads == 0
            || query_heads % kv_heads != 0
            || query_heads / kv_heads > 32
            || head_dim == 0
            || head_dim % 32 != 0
            || capacity == 0
            || capacity % 32 != 0
            || [query_heads, kv_heads, head_dim, capacity]
                .into_iter()
                .any(|n| n > 65536)
        {
            return Err(
                "packed attention requires GQA groups <= 32 and 32-aligned head dimension/context"
                    .into(),
            );
        }
        let channels = kv_heads
            .checked_mul(head_dim)
            .and_then(|n| n.checked_add(1))
            .filter(|&n| n <= 65536)
            .ok_or("packed attention channel extent exceeds 65536")?;
        let spatial = capacity
            .checked_mul(2)
            .and_then(|n| n.checked_add(32))
            .filter(|&n| n <= 65536)
            .ok_or("packed attention spatial extent exceeds 65536")?;
        let input_bytes = channels
            .checked_mul(spatial)
            .and_then(|n| n.checked_mul(2))
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("packed attention surface exceeds 4 GiB")?;
        let output_bytes = (channels - 1) * 64;
        Ok(Self {
            query_heads,
            kv_heads,
            head_dim,
            capacity,
            window: None,
            spatial,
            input_bytes,
            output_bytes,
        })
    }

    /// Bound storage and attention work to a fixed sliding window. Absolute
    /// positions map into ring slots; they must still be used unchanged for RoPE.
    pub fn sliding(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: usize,
    ) -> Result<Self, String> {
        let capacity = window
            .checked_add(31)
            .map(|n| n / 32 * 32)
            .ok_or("packed attention window is too large")?;
        let mut layout = Self::new(query_heads, kv_heads, head_dim, capacity)?;
        layout.window = Some(window);
        Ok(layout)
    }

    pub fn query_heads(self) -> usize {
        self.query_heads
    }
    pub fn kv_heads(self) -> usize {
        self.kv_heads
    }
    pub fn head_dim(self) -> usize {
        self.head_dim
    }
    pub fn capacity(self) -> usize {
        self.capacity
    }
    pub fn window(self) -> Option<usize> {
        self.window
    }
    pub fn retained_tokens(self, tokens: usize) -> Result<usize, String> {
        match self.window {
            Some(window) => Ok(tokens.min(window)),
            None if tokens <= self.capacity => Ok(tokens),
            None => Err("packed attention capacity exceeded".into()),
        }
    }
    pub fn groups(self) -> usize {
        self.query_heads / self.kv_heads
    }
    pub fn kv_width(self) -> usize {
        self.kv_heads * self.head_dim
    }
    pub fn query_width(self) -> usize {
        self.query_heads * self.head_dim
    }
    pub fn input_bytes(self) -> usize {
        self.input_bytes
    }
    pub fn output_bytes(self) -> usize {
        self.output_bytes
    }
    pub fn row_bytes(self) -> usize {
        self.spatial * 2
    }
    pub fn key_offset(self, position: usize) -> Option<usize> {
        self.cache_slot(position).map(|slot| (32 + slot) * 2)
    }
    pub fn value_offset(self, position: usize) -> Option<usize> {
        self.cache_slot(position)
            .map(|slot| (32 + self.capacity + slot) * 2)
    }
    fn cache_slot(self, position: usize) -> Option<usize> {
        if self.window.is_some() {
            Some(position % self.capacity)
        } else {
            (position < self.capacity).then_some(position)
        }
    }
    pub fn mask_offset(self) -> usize {
        self.kv_width() * self.row_bytes()
    }

    /// Locate exactly one rejected append, before any ring slot was reused.
    /// The retained prefix must be nonempty so its attention mask is defined.
    pub fn unwrapped_tail_offsets(
        self,
        tokens: usize,
        retained: usize,
    ) -> Result<(usize, usize), String> {
        if retained == 0
            || retained.checked_add(1) != Some(tokens)
            || tokens > self.window.unwrap_or(self.capacity)
        {
            return Err(
                "attention rollback requires one unwrapped append and a nonempty prefix".into(),
            );
        }
        Ok((
            self.key_offset(retained)
                .ok_or("invalid rollback key slot")?,
            self.value_offset(retained)
                .ok_or("invalid rollback value slot")?,
        ))
    }

    /// Convert query-head-major data into [KV head, dimension, group] rows.
    pub fn encode_query(self, query: &[f16], packed: &mut [u8]) -> Result<(), String> {
        if query.len() != self.query_width() || packed.len() != query.len() * 2 {
            return Err("packed attention query shape mismatch".into());
        }
        for kv in 0..self.kv_heads {
            for dim in 0..self.head_dim {
                for group in 0..self.groups() {
                    let src = (kv * self.groups() + group) * self.head_dim + dim;
                    let dst = ((kv * self.head_dim + dim) * self.groups() + group) * 2;
                    packed[dst..dst + 2].copy_from_slice(&query[src].to_le_bytes());
                }
            }
        }
        Ok(())
    }

    pub fn decode_output(self, packed: &[u8], output: &mut [f16]) -> Result<(), String> {
        if packed.len() != self.output_bytes || output.len() != self.query_width() {
            return Err("packed attention output shape mismatch".into());
        }
        for kv in 0..self.kv_heads {
            for dim in 0..self.head_dim {
                for group in 0..self.groups() {
                    let src = (kv * self.head_dim + dim) * 64 + group * 2;
                    let dst = (kv * self.groups() + group) * self.head_dim + dim;
                    output[dst] = f16::from_le_bytes([packed[src], packed[src + 1]]);
                }
            }
        }
        Ok(())
    }

    /// Pack synchronized Metal prefill data in [token, KV head, dimension]
    /// order into the same resident surface used by subsequent token appends.
    /// Sliding layouts retain only the last window, in physical ring order.
    /// `tokens` remains the absolute count, including discarded positions.
    pub fn import_cache(
        self,
        keys: &[f16],
        values: &[f16],
        tokens: usize,
    ) -> Result<Vec<u8>, String> {
        let retained = self.retained_tokens(tokens)?;
        if tokens.checked_mul(self.kv_width()) != Some(keys.len()) || values.len() != keys.len() {
            return Err("packed attention prefill shape mismatch".into());
        }
        let mut packed = vec![0; self.input_bytes];
        for token in tokens - retained..tokens {
            let key_offset = self.key_offset(token).ok_or("invalid cache position")?;
            let value_offset = self.value_offset(token).ok_or("invalid cache position")?;
            for channel in 0..self.kv_width() {
                let src = token * self.kv_width() + channel;
                let key = channel * self.row_bytes() + key_offset;
                let value = channel * self.row_bytes() + value_offset;
                packed[key..key + 2].copy_from_slice(&keys[src].to_le_bytes());
                packed[value..value + 2].copy_from_slice(&values[src].to_le_bytes());
            }
        }
        Ok(packed)
    }

    /// Pack into reusable storage. Shape errors leave the destination untouched.
    /// Clear every byte: a shorter prefix or different layer must not retain
    /// stale query, KV, padding or mask data from the preceding import.
    pub fn import_cache_into(
        self,
        keys: &[f16],
        values: &[f16],
        tokens: usize,
        packed: &mut [u8],
    ) -> Result<(), String> {
        let retained = self.retained_tokens(tokens)?;
        if tokens.checked_mul(self.kv_width()) != Some(keys.len())
            || values.len() != keys.len()
            || packed.len() != self.input_bytes
        {
            return Err("packed attention prefill shape mismatch".into());
        }
        packed.fill(0);
        // Preserve the allocating path's packing order for this experiment.
        // That path remains an independent byte-for-byte reference.
        for token in tokens - retained..tokens {
            let key_offset = self.key_offset(token).ok_or("invalid cache position")?;
            let value_offset = self.value_offset(token).ok_or("invalid cache position")?;
            for channel in 0..self.kv_width() {
                let src = token * self.kv_width() + channel;
                let row = channel * self.row_bytes();
                packed[row + key_offset..row + key_offset + 2]
                    .copy_from_slice(&keys[src].to_le_bytes());
                packed[row + value_offset..row + value_offset + 2]
                    .copy_from_slice(&values[src].to_le_bytes());
            }
        }
        Ok(())
    }

    pub fn encode_mask(self, tokens: usize, mask: &mut [u8]) -> Result<(), String> {
        let retained = self.retained_tokens(tokens)?;
        if tokens == 0 || mask.len() != self.capacity * 2 {
            return Err("packed attention mask length or window invalid".into());
        }
        for slot in mask.chunks_exact_mut(2) {
            slot.copy_from_slice(&f16::NEG_INFINITY.to_le_bytes());
        }
        for position in tokens - retained..tokens {
            let slot = self.cache_slot(position).ok_or("invalid cache position")?;
            mask[slot * 2..slot * 2 + 2].copy_from_slice(&f16::ZERO.to_le_bytes());
        }
        Ok(())
    }

    /// A single external input/output, both [1,C,1,S]. Reshape/transpose happen
    /// inside the graph. Gemma 4 normalizes Q/K and uses attention scale 1.
    /// Scale probabilities by 32 around the value reduction to preserve small
    /// contributions on M4 Max. This passed the original 0.003 error bound over
    /// 86,016 elements, including physical sliding-window wrap. Unit-RMS V has
    /// magnitude at most sqrt(head_dim), leaving ample FP16 reduction headroom.
    pub fn mil(self) -> String {
        let (kv, dim, group, capacity, spatial) = (
            self.kv_heads,
            self.head_dim,
            self.groups(),
            self.capacity,
            self.spatial,
        );
        let width = self.kv_width();
        let channels = width + 1;
        let value_begin = 32 + capacity;
        format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {channels}, 1, {spatial}]> x) {{
        tensor<int32, [4]> bq = const()[name = string("bq"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [4]> bk = const()[name = string("bk"), val = tensor<int32, [4]>([0, 0, 0, 32])];
        tensor<int32, [4]> bv = const()[name = string("bv"), val = tensor<int32, [4]>([0, 0, 0, {value_begin}])];
        tensor<int32, [4]> bm = const()[name = string("bm"), val = tensor<int32, [4]>([0, {width}, 0, 0])];
        tensor<int32, [4]> sq = const()[name = string("sq"), val = tensor<int32, [4]>([1, {width}, 1, {group}])];
        tensor<int32, [4]> skv = const()[name = string("skv"), val = tensor<int32, [4]>([1, {width}, 1, {capacity}])];
        tensor<int32, [4]> sm = const()[name = string("sm"), val = tensor<int32, [4]>([1, 1, 1, {capacity}])];
        tensor<int32, [4]> rq = const()[name = string("rq"), val = tensor<int32, [4]>([1, {kv}, {dim}, {group}])];
        tensor<int32, [4]> rkv = const()[name = string("rkv"), val = tensor<int32, [4]>([1, {kv}, {dim}, {capacity}])];
        tensor<int32, [4]> perm = const()[name = string("perm"), val = tensor<int32, [4]>([0, 1, 3, 2])];
        bool no_transpose = const()[name = string("no_transpose"), val = bool(false)];
        int32 axis = const()[name = string("axis"), val = int32(-1)];
        fp16 probability_scale = const()[name = string("probability_scale"), val = fp16(32.0)];
        fp16 attention_scale = const()[name = string("attention_scale"), val = fp16(0.03125)];
        tensor<fp16, [1, {width}, 1, {group}]> qf = slice_by_size(x = x, begin = bq, size = sq)[name = string("qf")];
        tensor<fp16, [1, {width}, 1, {capacity}]> kf = slice_by_size(x = x, begin = bk, size = skv)[name = string("kf")];
        tensor<fp16, [1, {width}, 1, {capacity}]> vf = slice_by_size(x = x, begin = bv, size = skv)[name = string("vf")];
        tensor<fp16, [1, 1, 1, {capacity}]> mask = slice_by_size(x = x, begin = bm, size = sm)[name = string("mask")];
        tensor<fp16, [1, {kv}, {dim}, {group}]> qr = reshape(x = qf, shape = rq)[name = string("qr")];
        tensor<fp16, [1, {kv}, {group}, {dim}]> q = transpose(x = qr, perm = perm)[name = string("q")];
        tensor<fp16, [1, {kv}, {dim}, {capacity}]> k = reshape(x = kf, shape = rkv)[name = string("k")];
        tensor<fp16, [1, {kv}, {dim}, {capacity}]> vr = reshape(x = vf, shape = rkv)[name = string("vr")];
        tensor<fp16, [1, {kv}, {capacity}, {dim}]> v = transpose(x = vr, perm = perm)[name = string("v")];
        tensor<fp16, [1, {kv}, {group}, {capacity}]> scores = matmul(x = q, y = k, transpose_x = no_transpose, transpose_y = no_transpose)[name = string("scores")];
        tensor<fp16, [1, {kv}, {group}, {capacity}]> masked = add(x = scores, y = mask)[name = string("masked")];
        tensor<fp16, [1, {kv}, {group}, {capacity}]> probabilities = softmax(x = masked, axis = axis)[name = string("probabilities")];
        tensor<fp16, [1, {kv}, {group}, {capacity}]> scaled_probabilities = mul(x = probabilities, y = probability_scale)[name = string("scaled_probabilities")];
        tensor<fp16, [1, {kv}, {group}, {dim}]> scaled_attention = matmul(x = scaled_probabilities, y = v, transpose_x = no_transpose, transpose_y = no_transpose)[name = string("scaled_attention")];
        tensor<fp16, [1, {kv}, {group}, {dim}]> attention = mul(x = scaled_attention, y = attention_scale)[name = string("attention")];
        tensor<fp16, [1, {kv}, {dim}, {group}]> transposed = transpose(x = attention, perm = perm)[name = string("transposed")];
        tensor<fp16, [1, {width}, 1, {group}]> y = reshape(x = transposed, shape = sq)[name = string("y")];
    }} -> (y);
}}
"#
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_import_scratch_matches_all_bytes_across_sizes_prefixes_and_wraps() {
        let sliding = PackedAttentionLayout::sliding(16, 8, 256, 1024).unwrap();
        let global = PackedAttentionLayout::new(16, 1, 512, 1024).unwrap();
        let short = PackedAttentionLayout::sliding(4, 2, 32, 35).unwrap();
        let mut scratch = vec![0xa5; sliding.input_bytes()];
        let allocation = scratch.as_ptr();
        for (layout, tokens) in [
            (sliding, 1024),
            (global, 84),
            (sliding, 1),
            (global, 0),
            (sliding, 1027),
            (short, 133),
            (sliding, 84),
            (global, 1024),
        ] {
            let keys: Vec<_> = (0..tokens * layout.kv_width())
                .map(|i| {
                    if i % 37 == 0 {
                        f16::NEG_ZERO
                    } else {
                        f16::from_f32((i % 997) as f32 / 1024.0 - 0.5)
                    }
                })
                .collect();
            let values: Vec<_> = keys.iter().map(|v| -*v).collect();
            let expected = layout.import_cache(&keys, &values, tokens).unwrap();
            scratch.fill(0xa5);
            let bytes = layout.input_bytes();
            layout
                .import_cache_into(&keys, &values, tokens, &mut scratch[..bytes])
                .unwrap();
            assert_eq!(&scratch[..bytes], expected, "{layout:?}, tokens={tokens}");
            assert!(scratch[bytes..].iter().all(|&b| b == 0xa5));
            assert_eq!(scratch.as_ptr(), allocation);
        }
    }

    #[test]
    fn kv_import_scratch_invalid_geometry_preserves_destination() {
        let layout = PackedAttentionLayout::new(16, 1, 512, 64).unwrap();
        let keys = vec![f16::ONE; layout.kv_width()];
        let mut packed = vec![0x5a; layout.input_bytes()];
        for tokens in [0, 2, 65, usize::MAX] {
            assert!(layout
                .import_cache_into(&keys, &keys, tokens, &mut packed)
                .is_err());
            assert!(packed.iter().all(|&b| b == 0x5a));
        }
        assert!(layout
            .import_cache_into(&keys, &[], 1, &mut packed)
            .is_err());
        assert!(packed.iter().all(|&b| b == 0x5a));
        let length = packed.len();
        assert!(layout
            .import_cache_into(&keys, &keys, 1, &mut packed[..length - 1])
            .is_err());
        assert!(packed.iter().all(|&b| b == 0x5a));
    }

    #[test]
    fn two_token_reference_rollback_preserves_prefix_and_replacement() {
        for layout in [
            PackedAttentionLayout::sliding(16, 8, 256, 1024).unwrap(),
            PackedAttentionLayout::new(16, 1, 512, 1024).unwrap(),
        ] {
            for tokens in [2, 86, 1024] {
                let retained = tokens - 1;
                let width = layout.kv_width();
                let keys: Vec<_> = (0..tokens * width)
                    .map(|i| f16::from_f32((i % 997 + 1) as f32 / 1024.0))
                    .collect();
                let values: Vec<_> = keys.iter().map(|v| -*v).collect();
                let mut actual = layout.import_cache(&keys, &values, tokens).unwrap();
                let (key, value) = layout.unwrapped_tail_offsets(tokens, retained).unwrap();
                for channel in 0..width {
                    let row = channel * layout.row_bytes();
                    actual[row + key..row + key + 2].fill(0);
                    actual[row + value..row + value + 2].fill(0);
                }
                // Deliberately stale host scratch, including after import.
                let mut mask = vec![0x5a; layout.capacity() * 2];
                layout.encode_mask(retained, &mut mask).unwrap();
                let offset = layout.mask_offset();
                actual[offset..offset + mask.len()].copy_from_slice(&mask);
                let mut expected = layout
                    .import_cache(
                        &keys[..retained * width],
                        &values[..retained * width],
                        retained,
                    )
                    .unwrap();
                for slot in 0..layout.capacity() {
                    let bits = if slot < retained {
                        f16::ZERO
                    } else {
                        f16::NEG_INFINITY
                    };
                    expected[offset + 2 * slot..offset + 2 * slot + 2]
                        .copy_from_slice(&bits.to_le_bytes());
                }
                assert_eq!(actual, expected);

                // A replacement appends at the rejected absolute position.
                for channel in 0..width {
                    let row = channel * layout.row_bytes();
                    actual[row + key..row + key + 2].copy_from_slice(&f16::ONE.to_le_bytes());
                    actual[row + value..row + value + 2]
                        .copy_from_slice(&f16::NEG_ONE.to_le_bytes());
                }
                let mut replaced_keys = keys[..retained * width].to_vec();
                let mut replaced_values = values[..retained * width].to_vec();
                replaced_keys.extend(vec![f16::ONE; width]);
                replaced_values.extend(vec![f16::NEG_ONE; width]);
                let mut replaced = layout
                    .import_cache(&replaced_keys, &replaced_values, tokens)
                    .unwrap();
                layout.encode_mask(tokens, &mut mask).unwrap();
                replaced[offset..offset + mask.len()].copy_from_slice(&mask);
                actual[offset..offset + mask.len()].copy_from_slice(&mask);
                assert_eq!(actual, replaced);
            }
            for (tokens, retained) in [
                (0, 0),
                (1, 0),
                (2, 2),
                (3, 1),
                (1025, 1024),
                (0, usize::MAX),
            ] {
                assert!(layout.unwrapped_tail_offsets(tokens, retained).is_err());
            }
        }
    }

    fn get(bytes: &[u8], index: usize) -> f16 {
        f16::from_le_bytes([bytes[index], bytes[index + 1]])
    }

    #[test]
    fn gemma_layout_preserves_head_identity_and_cache_positions() {
        for (kv_heads, head_dim) in [(8, 256), (1, 512)] {
            let layout = PackedAttentionLayout::new(16, kv_heads, head_dim, 64).unwrap();
            let keys: Vec<_> = (0..5 * kv_heads * head_dim)
                .map(|i| f16::from_f32((i % 997) as f32))
                .collect();
            let values: Vec<_> = keys.iter().map(|v| -*v).collect();
            let packed = layout.import_cache(&keys, &values, 5).unwrap();
            for token in 0..64 {
                for head in 0..kv_heads {
                    for dim in 0..head_dim {
                        let channel = head * head_dim + dim;
                        let index = channel * (32 + 128) * 2 + (32 + token) * 2;
                        let expected = if token < 5 {
                            keys[(token * kv_heads + head) * head_dim + dim]
                        } else {
                            f16::ZERO
                        };
                        assert_eq!(get(&packed, index), expected);
                        let expected_v = if token < 5 { -expected } else { f16::ZERO };
                        assert_eq!(get(&packed, index + 128), expected_v);
                    }
                }
            }
            let query: Vec<_> = (0..16 * head_dim)
                .map(|i| f16::from_f32((i % 1009) as f32))
                .collect();
            let mut encoded = vec![0; query.len() * 2];
            layout.encode_query(&query, &mut encoded).unwrap();
            let mut output_storage = vec![0; layout.output_bytes()];
            for channel in 0..kv_heads * head_dim {
                let start = channel * (16 / kv_heads) * 2;
                let row = &encoded[start..start + (16 / kv_heads) * 2];
                output_storage[channel * 64..channel * 64 + row.len()].copy_from_slice(row);
                for group in 0..16 / kv_heads {
                    let head = channel / head_dim * (16 / kv_heads) + group;
                    assert_eq!(
                        get(&encoded, start + group * 2),
                        query[head * head_dim + channel % head_dim]
                    );
                }
            }
            let mut decoded = vec![f16::ZERO; query.len()];
            layout.decode_output(&output_storage, &mut decoded).unwrap();
            assert_eq!(decoded, query);
        }
    }

    #[test]
    fn causal_and_sliding_masks_exclude_uninitialized_tail() {
        for window in [None, Some(1), Some(3), Some(1024)] {
            let layout = match window {
                Some(window) => PackedAttentionLayout::sliding(16, 8, 256, window),
                None => PackedAttentionLayout::new(16, 8, 256, 64),
            }
            .unwrap();
            let mut mask = vec![0; layout.capacity() * 2];
            layout.encode_mask(7, &mut mask).unwrap();
            for i in 0..layout.capacity() {
                let expected = i < 7 && window.map_or(true, |n| i >= 7_usize.saturating_sub(n));
                assert_eq!(get(&mask, i * 2) == f16::ZERO, expected);
                assert!(!get(&mask, i * 2).is_nan());
            }
            assert!(layout.encode_mask(0, &mut mask).is_err());
        }
        let layout = PackedAttentionLayout::new(16, 8, 256, 64).unwrap();
        assert!(layout.encode_mask(65, &mut vec![0; 128]).is_err());
        assert!(PackedAttentionLayout::sliding(16, 8, 256, 0).is_err());
        assert!(PackedAttentionLayout::sliding(16, 8, 256, usize::MAX).is_err());
    }

    fn reference_attention(
        query: &[f16],
        positions: &[usize],
        key: impl Fn(usize, usize) -> f32,
        value: impl Fn(usize, usize) -> f32,
    ) -> Vec<f32> {
        let scores: Vec<f32> = positions
            .iter()
            .map(|&position| {
                query
                    .iter()
                    .enumerate()
                    .map(|(dim, q)| q.to_f32() * key(position, dim))
                    .sum()
            })
            .collect();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let probabilities: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator: f32 = probabilities.iter().sum();
        let mut output = vec![0.0; query.len()];
        for (&position, probability) in positions.iter().zip(probabilities) {
            for (dim, out) in output.iter_mut().enumerate() {
                *out += probability / denominator * value(position, dim);
            }
        }
        output
    }

    #[test]
    fn sliding_ring_import_and_append_match_chronological_attention() {
        // Real 12B sliding geometry at the first wrap and after a long prefill;
        // the short window additionally exercises masked alignment padding.
        for (heads, kv, dim, window, prefill) in [
            (16, 8, 256, 1024, 1023),
            (16, 8, 256, 1024, 2053),
            (4, 2, 32, 3, 31),
        ] {
            let layout = PackedAttentionLayout::sliding(heads, kv, dim, window).unwrap();
            let width = kv * dim;
            let sample = |i: usize| f16::from_f32(((i * 13 + 7) % 101) as f32 / 128.0 - 0.375);
            let keys: Vec<_> = (0..(prefill + 4) * width).map(sample).collect();
            let values: Vec<_> = (0..keys.len()).map(|i| sample(i * 3 + 17)).collect();
            let mut packed = layout
                .import_cache(
                    &keys[..prefill * width],
                    &values[..prefill * width],
                    prefill,
                )
                .unwrap();
            let mut mask = vec![0; layout.capacity() * 2];
            for step in 0..=4 {
                let tokens = prefill + step;
                if step != 0 {
                    let position = tokens - 1;
                    // The same strided offsets consumed by the device wrapper.
                    // No previously stored token is copied or rotated on append.
                    for channel in 0..width {
                        let row = channel * layout.row_bytes();
                        let key = row + layout.key_offset(position).unwrap();
                        let value = row + layout.value_offset(position).unwrap();
                        packed[key..key + 2]
                            .copy_from_slice(&keys[position * width + channel].to_le_bytes());
                        packed[value..value + 2]
                            .copy_from_slice(&values[position * width + channel].to_le_bytes());
                    }
                }
                layout.encode_mask(tokens, &mut mask).unwrap();
                let slots: Vec<_> = (0..layout.capacity())
                    .filter(|&i| get(&mask, i * 2) == f16::ZERO)
                    .collect();
                let chronological: Vec<_> = (tokens.saturating_sub(window)..tokens).collect();
                assert_eq!(slots.len(), chronological.len());
                assert_eq!(layout.retained_tokens(tokens).unwrap(), chronological.len());
                for head in [0, heads / 2, heads - 1] {
                    let kv_head = head / (heads / kv);
                    let query: Vec<_> = (0..dim).map(|d| sample(d + head * 19 + tokens)).collect();
                    let expected = reference_attention(
                        &query,
                        &chronological,
                        |t, d| keys[t * width + kv_head * dim + d].to_f32(),
                        |t, d| values[t * width + kv_head * dim + d].to_f32(),
                    );
                    let actual = reference_attention(
                        &query,
                        &slots,
                        |slot, d| {
                            get(
                                &packed,
                                (kv_head * dim + d) * layout.row_bytes() + (32 + slot) * 2,
                            )
                            .to_f32()
                        },
                        |slot, d| {
                            get(
                                &packed,
                                (kv_head * dim + d) * layout.row_bytes()
                                    + (32 + layout.capacity() + slot) * 2,
                            )
                            .to_f32()
                        },
                    );
                    for (d, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                        assert!((got - want).abs() < 1e-5, "window={window} tokens={tokens} head={head} dim={d} got={got} expected={want}");
                    }
                }
            }
            assert_eq!(packed.len(), layout.input_bytes());
            // Absolute counters need not fit the storage extent.
            layout.encode_mask(usize::MAX, &mut mask).unwrap();
            assert_eq!(
                (0..layout.capacity())
                    .filter(|&i| get(&mask, i * 2) == f16::ZERO)
                    .count(),
                window
            );
        }
    }

    #[test]
    fn invalid_shapes_and_io_fail_without_device_access() {
        assert!(PackedAttentionLayout::new(16, 3, 256, 64).is_err());
        assert!(PackedAttentionLayout::new(16, 8, 256, 65536).is_err());
        assert!(PackedAttentionLayout::new(16, 8, 255, 64).is_err());
        let layout = PackedAttentionLayout::new(16, 1, 512, 64).unwrap();
        assert!(layout.import_cache(&[], &[], 1).is_err());
        assert!(layout.key_offset(64).is_none());
        assert!(layout.value_offset(64).is_none());
        let mil = layout.mil();
        assert!(mil.contains("func main<ios18>(tensor<fp16, [1, 513, 1, 160]> x)"));
    }
}
