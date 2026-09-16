//! One weight-independent Gemma FFN graph, with per-layer resident weight data.
//! This is host-only packing/MIL generation; no ANE execution is performed.
//!
//! Sharing this compiled program across the 48 layers avoids compiling 48
//! distinct FFNs. Each layer still owns its weights and activation surface.
//! The layout follows maderix/ANE's dynamic-weight single-input design, using
//! Gemma's GELU instead of SiLU and leaving norms/residuals to the decoder.

use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedFfnLayout {
    hidden: usize,
    intermediate: usize,
    spatial: usize,
    input_bytes: usize,
}

impl PackedFfnLayout {
    pub fn new(hidden: usize, intermediate: usize) -> Result<Self, String> {
        if hidden == 0
            || intermediate == 0
            || hidden % 32 != 0
            || intermediate % 32 != 0
            || hidden > 65536
        {
            return Err("dynamic FFN dimensions must be nonzero multiples of 32".into());
        }
        let spatial = intermediate
            .checked_mul(3)
            .and_then(|n| n.checked_add(32))
            .filter(|&n| n <= 65536)
            .ok_or("dynamic FFN packed spatial extent exceeds 65536")?;
        let input_bytes = hidden
            .checked_mul(spatial)
            .and_then(|n| n.checked_mul(2))
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("dynamic FFN surface exceeds 4 GiB")?;
        Ok(Self {
            hidden,
            intermediate,
            spatial,
            input_bytes,
        })
    }

    pub fn hidden(self) -> usize {
        self.hidden
    }
    pub fn intermediate(self) -> usize {
        self.intermediate
    }
    pub fn input_bytes(self) -> usize {
        self.input_bytes
    }
    pub fn output_bytes(self) -> usize {
        self.hidden * 64
    }
    pub fn row_bytes(self) -> usize {
        self.spatial * 2
    }

    /// Gate/up are row-major [intermediate, hidden], down [hidden, intermediate].
    /// Transpose gate/up in tiles so each source cache line serves many values.
    /// The first 32 lanes of every row remain reserved for token activations.
    pub fn pack_weights(self, gate: &[f16], up: &[f16], down: &[f16]) -> Result<Vec<u8>, String> {
        let count = self.hidden * self.intermediate;
        if [gate, up, down]
            .into_iter()
            .any(|w| w.len() != count || w.iter().any(|v| !v.is_finite()))
        {
            return Err("dynamic FFN weights have wrong shape or nonfinite values".into());
        }
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(self.input_bytes)
            .map_err(|e| format!("allocate dynamic FFN surface staging: {e}"))?;
        packed.resize(self.input_bytes, 0);
        for (matrix, weights) in [gate, up].into_iter().enumerate() {
            let offset = 32 + matrix * self.intermediate;
            for output_tile in (0..self.intermediate).step_by(32) {
                for input_tile in (0..self.hidden).step_by(32) {
                    for output in output_tile..output_tile + 32 {
                        let row = &weights[output * self.hidden + input_tile
                            ..output * self.hidden + input_tile + 32];
                        for (local, value) in row.iter().enumerate() {
                            let destination =
                                ((input_tile + local) * self.spatial + offset + output) * 2;
                            packed[destination..destination + 2]
                                .copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }
        }
        for (row, weights) in down.chunks_exact(self.intermediate).enumerate() {
            let start = (row * self.spatial + 32 + 2 * self.intermediate) * 2;
            for (slot, value) in packed[start..start + self.intermediate * 2]
                .chunks_exact_mut(2)
                .zip(weights)
            {
                slot.copy_from_slice(&value.to_le_bytes());
            }
        }
        Ok(packed)
    }

    /// The kernel has one input and one output; weights are sliced from the
    /// input and used by matmul. No weight-dependent constant enters this MIL.
    /// GELU is expanded deliberately: the native ANE GELU operation produced
    /// 1.59% activation error in the M4 Max fixture, while this explicit tanh
    /// form passed the unchanged complete-FFN numerical gate.
    pub fn mil(self) -> String {
        let (hidden, intermediate, spatial) = (self.hidden, self.intermediate, self.spatial);
        let up_begin = 32 + intermediate;
        let down_begin = 32 + 2 * intermediate;
        format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {hidden}, 1, {spatial}]> x) {{
        tensor<int32, [4]> bx = const()[name = string("bx"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [4]> bg = const()[name = string("bg"), val = tensor<int32, [4]>([0, 0, 0, 32])];
        tensor<int32, [4]> bu = const()[name = string("bu"), val = tensor<int32, [4]>([0, 0, 0, {up_begin}])];
        tensor<int32, [4]> bd = const()[name = string("bd"), val = tensor<int32, [4]>([0, 0, 0, {down_begin}])];
        tensor<int32, [4]> sx = const()[name = string("sx"), val = tensor<int32, [4]>([1, {hidden}, 1, 1])];
        tensor<int32, [4]> sw = const()[name = string("sw"), val = tensor<int32, [4]>([1, {hidden}, 1, {intermediate}])];
        tensor<int32, [4]> rx = const()[name = string("rx"), val = tensor<int32, [4]>([1, 1, 1, {hidden}])];
        tensor<int32, [4]> rw = const()[name = string("rw"), val = tensor<int32, [4]>([1, 1, {hidden}, {intermediate}])];
        tensor<int32, [4]> ro = const()[name = string("ro"), val = tensor<int32, [4]>([1, {hidden}, 1, 1])];
        bool no = const()[name = string("no"), val = bool(false)];
        bool yes = const()[name = string("yes"), val = bool(true)];
        fp16 gelu_half = const()[name = string("gelu_half"), val = fp16(0.5)];
        fp16 gelu_one = const()[name = string("gelu_one"), val = fp16(1.0)];
        fp16 gelu_cubic = const()[name = string("gelu_cubic"), val = fp16(0.044715)];
        fp16 gelu_root = const()[name = string("gelu_root"), val = fp16(0.7978845608)];
        tensor<fp16, [1, {hidden}, 1, 1]> xf = slice_by_size(x = x, begin = bx, size = sx)[name = string("xf")];
        tensor<fp16, [1, {hidden}, 1, {intermediate}]> gf = slice_by_size(x = x, begin = bg, size = sw)[name = string("gf")];
        tensor<fp16, [1, {hidden}, 1, {intermediate}]> uf = slice_by_size(x = x, begin = bu, size = sw)[name = string("uf")];
        tensor<fp16, [1, {hidden}, 1, {intermediate}]> df = slice_by_size(x = x, begin = bd, size = sw)[name = string("df")];
        tensor<fp16, [1, 1, 1, {hidden}]> xr = reshape(x = xf, shape = rx)[name = string("xr")];
        tensor<fp16, [1, 1, {hidden}, {intermediate}]> wg = reshape(x = gf, shape = rw)[name = string("wg")];
        tensor<fp16, [1, 1, {hidden}, {intermediate}]> wu = reshape(x = uf, shape = rw)[name = string("wu")];
        tensor<fp16, [1, 1, {hidden}, {intermediate}]> wd = reshape(x = df, shape = rw)[name = string("wd")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gate = matmul(x = xr, y = wg, transpose_x = no, transpose_y = no)[name = string("gate")];
        tensor<fp16, [1, 1, 1, {intermediate}]> up = matmul(x = xr, y = wu, transpose_x = no, transpose_y = no)[name = string("up")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gate_squared = mul(x = gate, y = gate)[name = string("gate_squared")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gate_cubed = mul(x = gate_squared, y = gate)[name = string("gate_cubed")];
        tensor<fp16, [1, 1, 1, {intermediate}]> cubic_scaled = mul(x = gate_cubed, y = gelu_cubic)[name = string("cubic_scaled")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gelu_sum = add(x = gate, y = cubic_scaled)[name = string("gelu_sum")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gelu_argument = mul(x = gelu_sum, y = gelu_root)[name = string("gelu_argument")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gelu_tanh = tanh(x = gelu_argument)[name = string("gelu_tanh")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gelu_factor = add(x = gelu_tanh, y = gelu_one)[name = string("gelu_factor")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gate_halved = mul(x = gate, y = gelu_half)[name = string("gate_halved")];
        tensor<fp16, [1, 1, 1, {intermediate}]> activated = mul(x = gate_halved, y = gelu_factor)[name = string("activated")];
        tensor<fp16, [1, 1, 1, {intermediate}]> gated = mul(x = activated, y = up)[name = string("gated")];
        tensor<fp16, [1, 1, 1, {hidden}]> result = matmul(x = gated, y = wd, transpose_x = no, transpose_y = yes)[name = string("result")];
        tensor<fp16, [1, {hidden}, 1, 1]> y = reshape(x = result, shape = ro)[name = string("y")];
    }} -> (y);
}}
"#
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(data: &[u8], offset: usize) -> f16 {
        f16::from_le_bytes([data[offset], data[offset + 1]])
    }

    #[test]
    fn packed_weights_preserve_all_three_matrix_orientations() {
        let (hidden, intermediate) = (64, 96);
        let layout = PackedFfnLayout::new(hidden, intermediate).unwrap();
        let gate: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32((i % 1009) as f32 / 64.0))
            .collect();
        let up: Vec<_> = gate.iter().map(|v| -*v).collect();
        let down: Vec<_> = gate.iter().rev().copied().collect();
        let packed = layout.pack_weights(&gate, &up, &down).unwrap();
        for row in 0..hidden {
            assert!(
                packed[row * layout.row_bytes()..row * layout.row_bytes() + 64]
                    .iter()
                    .all(|&b| b == 0)
            );
            for column in 0..intermediate {
                let offset = row * layout.row_bytes() + (32 + column) * 2;
                assert_eq!(get(&packed, offset), gate[column * hidden + row]);
                assert_eq!(
                    get(&packed, offset + intermediate * 2),
                    up[column * hidden + row]
                );
                assert_eq!(
                    get(&packed, offset + intermediate * 4),
                    down[row * intermediate + column]
                );
            }
        }
        assert!(layout.pack_weights(&gate[..10], &up, &down).is_err());
        assert!(PackedFfnLayout::new(3840, 65536).is_err());
    }

    #[test]
    fn gemma_12b_fits_one_surface_with_small_activation_padding() {
        let layout = PackedFfnLayout::new(3840, 15360).unwrap();
        assert_eq!(layout.input_bytes(), 3 * 3840 * 15360 * 2 + 3840 * 64);
        assert_eq!(layout.output_bytes(), 3840 * 64);
        let mil = layout.mil();
        assert!(!mil.contains("BLOBFILE"));
        assert!(mil.contains("func main<ios18>(tensor<fp16, [1, 3840, 1, 46112]> x)"));
    }
}
