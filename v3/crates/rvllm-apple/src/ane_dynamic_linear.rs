//! Shared single-input linear graphs with independent resident weight surfaces.
//! The initial use is Gemma's two output-projection shapes, so the much larger
//! FFNs can retain compiled constant weights within the program-count budget.

use half::f16;
use rvllm_apple_ane_sys::{AneInMemoryKernel, AneInMemoryProgram, AneProgramCachePolicy};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy)]
struct Layout {
    input: usize,
    output: usize,
    row_bytes: usize,
    input_bytes: usize,
}

impl Layout {
    fn new(input: usize, output: usize) -> Result<Self, String> {
        if input == 0 || output == 0 || input % 32 != 0 || output % 32 != 0 || input > 65536 {
            return Err("dynamic linear dimensions must be nonzero multiples of 32".into());
        }
        let spatial = output
            .checked_add(32)
            .filter(|&n| n <= 65536)
            .ok_or("dynamic linear spatial extent exceeds 65536")?;
        let row_bytes = spatial * 2;
        let input_bytes = input
            .checked_mul(row_bytes)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("dynamic linear input exceeds 4 GiB")?;
        Ok(Self {
            input,
            output,
            row_bytes,
            input_bytes,
        })
    }

    fn pack(self, weights: &[f16]) -> Result<Vec<u8>, String> {
        if weights.len() != self.input * self.output || weights.iter().any(|v| !v.is_finite()) {
            return Err("dynamic linear weights must be a finite [output,input] matrix".into());
        }
        let mut packed = vec![0; self.input_bytes];
        // Tile the transpose so each source cache line supplies nearby rows.
        for output_tile in (0..self.output).step_by(32) {
            for input_tile in (0..self.input).step_by(32) {
                for output in output_tile..output_tile + 32 {
                    let source = &weights
                        [output * self.input + input_tile..output * self.input + input_tile + 32];
                    for (local, value) in source.iter().enumerate() {
                        let offset = (input_tile + local) * self.row_bytes + (32 + output) * 2;
                        packed[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        Ok(packed)
    }

    fn mil(self) -> String {
        let (input, output, spatial) = (self.input, self.output, self.row_bytes / 2);
        format!(
            r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {input}, 1, {spatial}]> x) {{
        tensor<int32, [4]> bx = const()[name = string("bx"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [4]> bw = const()[name = string("bw"), val = tensor<int32, [4]>([0, 0, 0, 32])];
        tensor<int32, [4]> sx = const()[name = string("sx"), val = tensor<int32, [4]>([1, {input}, 1, 1])];
        tensor<int32, [4]> sw = const()[name = string("sw"), val = tensor<int32, [4]>([1, {input}, 1, {output}])];
        tensor<int32, [4]> rx = const()[name = string("rx"), val = tensor<int32, [4]>([1, 1, 1, {input}])];
        tensor<int32, [4]> rw = const()[name = string("rw"), val = tensor<int32, [4]>([1, 1, {input}, {output}])];
        tensor<int32, [4]> ro = const()[name = string("ro"), val = tensor<int32, [4]>([1, {output}, 1, 1])];
        bool no = const()[name = string("no"), val = bool(false)];
        tensor<fp16, [1, {input}, 1, 1]> xf = slice_by_size(x = x, begin = bx, size = sx)[name = string("xf")];
        tensor<fp16, [1, {input}, 1, {output}]> wf = slice_by_size(x = x, begin = bw, size = sw)[name = string("wf")];
        tensor<fp16, [1, 1, 1, {input}]> xr = reshape(x = xf, shape = rx)[name = string("xr")];
        tensor<fp16, [1, 1, {input}, {output}]> wr = reshape(x = wf, shape = rw)[name = string("wr")];
        tensor<fp16, [1, 1, 1, {output}]> product = matmul(x = xr, y = wr, transpose_x = no, transpose_y = no)[name = string("product")];
        tensor<fp16, [1, {output}, 1, 1]> y = reshape(x = product, shape = ro)[name = string("y")];
    }} -> (y);
}}
"#
        )
    }
}

pub struct AneDynamicLinearProgram {
    layout: Layout,
    program: AneInMemoryProgram,
}

impl AneDynamicLinearProgram {
    pub fn compile(input: usize, output: usize) -> Result<Self, String> {
        Self::compile_with_cache_policy(input, output, AneProgramCachePolicy::Compile)
    }

    pub fn compile_with_cache_policy(
        input: usize,
        output: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let layout = Layout::new(input, output)?;
        let program = AneInMemoryProgram::compile_with_cache_policy(
            &layout.mil(),
            &[],
            layout.input_bytes,
            output * 64,
            policy,
        )?;
        Ok(Self { layout, program })
    }

    pub fn cache_identity(input: usize, output: usize) -> Result<String, String> {
        let layout = Layout::new(input, output)?;
        Ok(Sha256::digest(layout.mil().as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }

    pub fn create_layer(&self, weights: &[f16]) -> Result<AneDynamicLinear, String> {
        let packed = self.layout.pack(weights)?;
        let mut kernel = self.program.create_request()?;
        kernel.write_input(&packed)?;
        Ok(AneDynamicLinear {
            kernel,
            layout: self.layout,
            input: vec![0; self.layout.input * 2],
            output: vec![0; self.layout.output * 64],
        })
    }
}

pub struct AneDynamicLinear {
    kernel: AneInMemoryKernel,
    layout: Layout,
    input: Vec<u8>,
    output: Vec<u8>,
}

impl AneDynamicLinear {
    /// Only activation bytes change between tokens; weights remain resident.
    pub fn project(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        if input.len() != self.layout.input
            || output.len() != self.layout.output
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(
                "dynamic linear input/output shape or finite-value requirement failed".into(),
            );
        }
        for (value, bytes) in input.iter().zip(self.input.chunks_exact_mut(2)) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        self.kernel
            .write_tensor_strided(0, 0, self.layout.row_bytes, 2, &self.input)?;
        self.kernel.evaluate()?;
        self.kernel.read_output(&mut self.output)?;
        for (value, bytes) in output.iter_mut().zip(self.output.chunks_exact(64)) {
            *value = f16::from_le_bytes([bytes[0], bytes[1]]);
            if !value.is_finite() {
                return Err("dynamic linear output is nonfinite".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_weight_packing_preserves_nonsquare_matrix_and_padding() {
        let layout = Layout::new(64, 96).unwrap();
        let weights: Vec<_> = (0..64 * 96)
            .map(|i| f16::from_f32((i % 1019) as f32))
            .collect();
        let packed = layout.pack(&weights).unwrap();
        for (input, row) in packed.chunks_exact(layout.row_bytes).enumerate() {
            assert!(row[..64].iter().all(|&b| b == 0));
            for (output, value) in row[64..].chunks_exact(2).enumerate() {
                assert_eq!(
                    f16::from_le_bytes([value[0], value[1]]),
                    weights[output * 64 + input]
                );
            }
        }
        assert_eq!(Layout::new(4096, 3840).unwrap().input_bytes, 31_719_424);
        assert_eq!(Layout::new(8192, 3840).unwrap().input_bytes, 63_438_848);
        assert!(AneDynamicLinearProgram::compile(0, 3840).is_err());
        assert!(AneDynamicLinearProgram::compile(8192, 65536).is_err());
        assert!(layout.pack(&weights[..3]).is_err());
    }

    #[test]
    fn qkv_cache_identities_are_weight_independent_and_shape_bound() {
        let sliding = AneDynamicLinearProgram::cache_identity(3840, 8192).unwrap();
        let global = AneDynamicLinearProgram::cache_identity(3840, 8704).unwrap();
        assert_eq!(sliding.len(), 64);
        assert_eq!(global.len(), 64);
        assert_ne!(sliding, global);
        assert_eq!(
            sliding,
            AneDynamicLinearProgram::cache_identity(3840, 8192).unwrap()
        );
        assert!(AneDynamicLinearProgram::cache_identity(0, 8192).is_err());
    }

    #[test]
    #[ignore = "executes private ANE API; one graph, two resident weight sets, four evaluations"]
    fn hardware_shared_linear_preserves_independent_weights() {
        let (input_width, output_width) = (64, 96);
        let weights: Vec<_> = (0..input_width * output_width)
            .map(|i| f16::from_f32(((i * 7 + 3) % 29) as f32 / 64.0 - 0.2))
            .collect();
        let opposite: Vec<_> = weights.iter().map(|w| -*w).collect();
        let program = AneDynamicLinearProgram::compile(input_width, output_width).unwrap();
        let mut first = program.create_layer(&weights).unwrap();
        let mut second = program.create_layer(&opposite).unwrap();
        drop(program);
        for step in 0..2 {
            let input: Vec<_> = (0..input_width)
                .map(|i| f16::from_f32(((i * 3 + step * 11) % 23) as f32 / 32.0 - 0.3))
                .collect();
            for (layer, matrix) in [(&mut first, &weights), (&mut second, &opposite)] {
                let mut actual = vec![f16::ZERO; output_width];
                layer.project(&input, &mut actual).unwrap();
                for (row, actual) in matrix.chunks_exact(input_width).zip(actual) {
                    let expected: f32 = row
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| w.to_f32() * x.to_f32())
                        .sum();
                    assert!((actual.to_f32() - expected).abs() < 0.002);
                }
            }
        }
    }
}
