//! Resident ANE linear projection for single-token decode. The spatial lanes
//! are explicit because ANE convolution I/O is channel-first, [1,C,1,S].

use crate::ane_int8_ffn_weights::{AneInt8FfnWeights, AneInt8LinearWeights};
use crate::ane_lut4_ffn_weights::AneLut4FfnWeights;
use half::f16;
pub use rvllm_apple_ane_sys::{compile_budget_used, AneProgramCachePolicy};
use rvllm_apple_ane_sys::{AneInMemoryKernel, AneInMemoryProgram};

pub struct AneLinear {
    kernel: AneInMemoryKernel,
    input: Vec<u8>,
    output: Vec<u8>,
    input_channels: usize,
    output_channels: usize,
    spatial: usize,
    logical_spatial: usize,
}

struct LinearLayout {
    logical_spatial: usize,
    physical_spatial: usize,
    input_bytes: usize,
    output_bytes: usize,
}

impl LinearLayout {
    fn new(input: usize, output: usize, spatial: usize) -> Result<Self, String> {
        // ANE surfaces align channel rows to 64 bytes, even for one token.
        let physical_spatial = spatial
            .checked_add(31)
            .map(|n| n / 32 * 32)
            .ok_or("ANE spatial stride overflow")?;
        Ok(Self {
            logical_spatial: spatial,
            physical_spatial,
            input_bytes: io_bytes(input, physical_spatial)?,
            output_bytes: io_bytes(output, physical_spatial)?,
        })
    }
}

impl AneLinear {
    pub fn compile(
        weights: &[f16],
        input_channels: usize,
        output_channels: usize,
        spatial: usize,
    ) -> Result<Self, String> {
        Self::compile_with_cache_policy(
            weights,
            input_channels,
            output_channels,
            spatial,
            AneProgramCachePolicy::Compile,
        )
    }

    pub fn compile_with_cache_policy(
        weights: &[f16],
        input_channels: usize,
        output_channels: usize,
        spatial: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let elements = input_channels
            .checked_mul(output_channels)
            .filter(|&n| n != 0 && n == weights.len())
            .ok_or("ANE linear weight shape mismatch or overflow")?;
        let weight_bytes = elements
            .checked_mul(2)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("ANE linear weight blob exceeds 4 GiB")?;
        let layout = LinearLayout::new(input_channels, output_channels, spatial)?;
        let mut blob = vec![0_u8; 128 + weight_bytes];
        blob[0..4].copy_from_slice(&1_u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
        blob[64..68].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
        blob[68..72].copy_from_slice(&1_u32.to_le_bytes());
        blob[72..80].copy_from_slice(&(weight_bytes as u64).to_le_bytes());
        blob[80..88].copy_from_slice(&128_u64.to_le_bytes());
        for (bytes, weight) in blob[128..].chunks_exact_mut(2).zip(weights) {
            bytes.copy_from_slice(&weight.to_le_bytes());
        }
        let mil = linear_mil(input_channels, output_channels, spatial);
        Self::compile_program(&mil, &blob, input_channels, output_channels, layout, policy)
    }

    /// Experimental per-output INT8 constants with FP16 activations. This
    /// preserves the single-input, single-output projection graph. Gemma
    /// quality, device execution and speed require independent qualification.
    pub fn compile_int8_with_cache_policy(
        weights: &AneInt8LinearWeights,
        spatial: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let (input_channels, output_channels) = weights.shape();
        let layout = LinearLayout::new(input_channels, output_channels, spatial)?;
        let (blob, constants) = weights.blob_and_constants();
        let mil = linear_mil_with_constants(input_channels, output_channels, spatial, &constants);
        Self::compile_program(&mil, &blob, input_channels, output_channels, layout, policy)
    }

    /// Explicit four-way output-row tiling of already-quantized INT8 weights.
    /// Component O/head use requires separately supplied INT8 constants; the
    /// runtime selector changes only the existing experimental sliding QKV path.
    pub fn compile_int8_tiles4_with_cache_policy(
        weights: &AneInt8LinearWeights,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let source = crate::ane_int8_candidates::linear_tiles4(weights)?;
        let (input, output) = weights.shape();
        let layout = LinearLayout::new(input, output, 1)?;
        tracing::debug!(
            candidate = source.name,
            source_blob_bytes = source.blob.len(),
            convolutions = source.budget.convolutions,
            "ANE candidate source"
        );
        Self::compile_program(&source.mil, &source.blob, input, output, layout, policy)
    }

    fn compile_program(
        mil: &str,
        blob: &[u8],
        input_channels: usize,
        output_channels: usize,
        layout: LinearLayout,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let kernel = AneInMemoryProgram::compile_with_cache_policy(
            mil,
            blob,
            layout.input_bytes,
            layout.output_bytes,
            policy,
        )?
        .create_request()?;
        Ok(Self {
            kernel,
            input: vec![0; layout.input_bytes],
            output: vec![0; layout.output_bytes],
            input_channels,
            output_channels,
            spatial: layout.physical_spatial,
            logical_spatial: layout.logical_spatial,
        })
    }

    /// Project one token using lane zero, with preallocated I/O. Other lanes
    /// remain zero. No model compilation or request allocation occurs here.
    pub fn project(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        if input.len() != self.input_channels || output.len() != self.output_channels {
            return Err("ANE linear input/output shape mismatch".into());
        }
        if self.logical_spatial != 1 {
            // A preceding batch may have populated other logical columns.
            self.input.fill(0);
        }
        for (channel, value) in input.iter().enumerate() {
            let offset = channel * self.spatial * 2;
            self.input[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }
        self.kernel.write_input(&self.input)?;
        self.kernel.evaluate()?;
        self.kernel.read_output(&mut self.output)?;
        for (channel, value) in output.iter_mut().enumerate() {
            let offset = channel * self.spatial * 2;
            *value = f16::from_le_bytes([self.output[offset], self.output[offset + 1]]);
        }
        Ok(())
    }

    /// Project all logical columns in token-major order. This component route
    /// supports up to eight tokens and uses the same single external I/O pair.
    /// QKV, output and vocabulary shapes need independent device qualification.
    pub fn project_batch(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        if self.input_channels.checked_mul(self.logical_spatial) != Some(input.len())
            || self.output_channels.checked_mul(self.logical_spatial) != Some(output.len())
            || !(1..=8).contains(&self.logical_spatial)
            || self.spatial != 32
        {
            return Err("ANE linear batch input/output shape mismatch or unsupported width".into());
        }
        pack_token_columns(
            input,
            self.input_channels,
            self.logical_spatial,
            &mut self.input,
        )?;
        self.kernel.write_input(&self.input)?;
        self.kernel.evaluate()?;
        self.kernel.read_output(&mut self.output)?;
        unpack_token_columns(
            &self.output,
            self.output_channels,
            self.logical_spatial,
            output,
        )
    }
}

fn io_bytes(channels: usize, spatial: usize) -> Result<usize, String> {
    channels
        .checked_mul(spatial)
        .and_then(|n| n.checked_mul(2))
        .filter(|&n| n != 0 && n <= u32::MAX as usize)
        .ok_or_else(|| "ANE linear I/O shape is empty or exceeds 4 GiB".into())
}

fn linear_mil(input: usize, output: usize, spatial: usize) -> String {
    let constants = format!(
        "        tensor<fp16, [{output}, {input}, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [{output}, {input}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
    );
    linear_mil_with_constants(input, output, spatial, &constants)
}

fn linear_mil_with_constants(
    input: usize,
    output: usize,
    spatial: usize,
    constants: &str,
) -> String {
    format!(
        r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {input}, 1, {spatial}]> x) {{
        string pad_type = const()[name = string("pad_type"), val = string("valid")];
        tensor<int32, [2]> strides = const()[name = string("strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pad = const()[name = string("pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dilations = const()[name = string("dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 groups = const()[name = string("groups"), val = int32(1)];
{constants}        tensor<fp16, [1, {output}, 1, {spatial}]> y = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = W, x = x)[name = string("projection")];
    }} -> (y);
}}
"#
    )
}

/// Gemma's three dense FFN projections plus tanh-approximate GELU and gating
/// in one resident ANE program. Input is already normalized; residual and
/// post-FFN normalization belong to the surrounding decoder block.
pub struct AneGatedFfn {
    kernel: AneInMemoryKernel,
    input: Vec<u8>,
    output: Vec<u8>,
    hidden: usize,
    logical_spatial: usize,
}

impl AneGatedFfn {
    pub fn compile(
        gate: &[f16],
        up: &[f16],
        down: &[f16],
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Self::compile_with_cache_policy(
            gate,
            up,
            down,
            hidden,
            intermediate,
            AneProgramCachePolicy::Compile,
        )
    }

    pub fn compile_with_cache_policy(
        gate: &[f16],
        up: &[f16],
        down: &[f16],
        hidden: usize,
        intermediate: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let count = hidden.checked_mul(intermediate)
            .filter(|&n| n > 0 && n == gate.len() && n == up.len() && n == down.len())
            .ok_or("ANE FFN weight shapes must be gate/up [intermediate,hidden], down [hidden,intermediate]")?;
        let bytes = count.checked_mul(2).ok_or("ANE FFN weight size overflow")?;
        let total = bytes
            .checked_add(64)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(64))
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("ANE FFN weights exceed 4 GiB")?;
        let mut blob = vec![0_u8; total];
        blob[0..4].copy_from_slice(&3_u32.to_le_bytes());
        blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let offsets = [64, 128 + bytes, 192 + 2 * bytes];
        for (offset, weights) in offsets.into_iter().zip([gate, up, down]) {
            blob[offset..offset + 4].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
            blob[offset + 4..offset + 8].copy_from_slice(&1_u32.to_le_bytes());
            blob[offset + 8..offset + 16].copy_from_slice(&(bytes as u64).to_le_bytes());
            blob[offset + 16..offset + 24].copy_from_slice(&((offset + 64) as u64).to_le_bytes());
            for (bytes, weight) in blob[offset + 64..offset + 64 + bytes]
                .chunks_exact_mut(2)
                .zip(weights)
            {
                bytes.copy_from_slice(&weight.to_le_bytes());
            }
        }
        let mil = ffn_mil(hidden, intermediate, offsets);
        Self::compile_program(&mil, &blob, hidden, policy)
    }

    /// Experimental gate/up stacking with unchanged INT8 coefficients, FP16
    /// activations, tanh GELU and one external input/output. This changes the
    /// internal projection layout; callers must qualify numerics and latency.
    pub fn compile_int8_stacked_with_cache_policy(
        weights: &AneInt8FfnWeights,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let (hidden, intermediate) = weights.shape();
        let (blob, constants) = weights.stacked_blob_and_constants()?;
        let mil = ffn_mil_stacked(hidden, intermediate, &constants);
        Self::compile_program(&mil, &blob, hidden, policy)
    }

    /// Explicit Gemma 12B output-channel chunking; still one external I/O pair.
    /// Unsupported geometry errors before reaching a private framework call.
    pub fn compile_int8_chunk4_with_cache_policy(
        weights: &AneInt8FfnWeights,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let source = crate::ane_int8_candidates::ffn_chunk4(weights)?;
        tracing::debug!(
            candidate = source.name,
            source_blob_bytes = source.blob.len(),
            convolutions = source.budget.convolutions,
            "ANE candidate source"
        );
        Self::compile_program(&source.mil, &source.blob, weights.shape().0, policy)
    }

    /// Experimental constant INT8 weights with FP16 activations and unchanged
    /// tanh GELU. Full-model quality and compressed residency require separate
    /// evidence; a smaller source blob alone establishes neither.
    pub fn compile_int8(weights: &AneInt8FfnWeights) -> Result<Self, String> {
        Self::compile_int8_with_cache_policy(weights, AneProgramCachePolicy::Compile)
    }

    pub fn compile_int8_with_cache_policy(
        weights: &AneInt8FfnWeights,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let (hidden, intermediate) = weights.shape();
        let (blob, constants) = weights.blob_and_constants();
        let mil = ffn_mil_with_constants(hidden, intermediate, &constants);
        Self::compile_program(&mil, &blob, hidden, policy)
    }

    /// Component experiment: independent token columns in one INT8 FFN graph.
    /// This does not implement causal attention, target verification or KV
    /// rollback. Logical columns differ from the surface's width-32 padding.
    pub fn compile_int8_batch_with_cache_policy(
        weights: &AneInt8FfnWeights,
        tokens: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        if !(2..=8).contains(&tokens) {
            return Err("ANE FFN batch experiment requires 2..=8 logical tokens".into());
        }
        let (hidden, intermediate) = weights.shape();
        let (blob, constants) = weights.blob_and_constants();
        let mil = ffn_mil_batched(hidden, intermediate, &constants, tokens);
        Self::compile_program_with_spatial(&mil, &blob, hidden, tokens, policy)
    }

    /// Default-off full-K output-row down tiling, with unchanged single-I/O ABI.
    pub fn compile_int8_down4_with_cache_policy(
        weights: &AneInt8FfnWeights,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let source = crate::ane_int8_candidates::ffn_down4(weights)?;
        tracing::debug!(
            candidate = source.name,
            source_blob_bytes = source.blob.len(),
            convolutions = source.budget.convolutions,
            programs = source.budget.programs,
            "ANE candidate source; compiled memory and performance unmeasured"
        );
        Self::compile_program(&source.mil, &source.blob, weights.shape().0, policy)
    }

    /// Experimental eight-bit palette storage of the exact INT8 reconstruction.
    /// This changes constant encoding only; inference still has one input and
    /// one output and uses the existing three-convolution GELU graph.
    pub fn compile_int8_palette_with_cache_policy(
        weights: &AneInt8FfnWeights,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let (hidden, intermediate) = weights.shape();
        let (blob, constants) = weights.lut8_blob_and_constants()?;
        let mil = ffn_mil_with_constants(hidden, intermediate, &constants);
        Self::compile_program(&mil, &blob, hidden, policy)
    }

    /// Experimental scalar LUT4 constants. Input/output and GELU are unchanged;
    /// callers must qualify quantization error separately from backend error.
    pub fn compile_lut4_with_cache_policy(
        weights: &AneLut4FfnWeights,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let (hidden, intermediate) = weights.shape();
        let (blob, constants) = weights.blob_and_constants();
        let mil = ffn_mil_with_constants(hidden, intermediate, &constants);
        Self::compile_program(&mil, &blob, hidden, policy)
    }

    fn compile_program(
        mil: &str,
        blob: &[u8],
        hidden: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        Self::compile_program_with_spatial(mil, blob, hidden, 1, policy)
    }

    fn compile_program_with_spatial(
        mil: &str,
        blob: &[u8],
        hidden: usize,
        logical_spatial: usize,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        if !(1..=8).contains(&logical_spatial) {
            return Err("unsupported ANE FFN logical width".into());
        }
        let io_bytes = io_bytes(hidden, 32)?;
        let kernel =
            AneInMemoryProgram::compile_with_cache_policy(mil, blob, io_bytes, io_bytes, policy)?
                .create_request()?;
        Ok(Self {
            kernel,
            input: vec![0; io_bytes],
            output: vec![0; io_bytes],
            hidden,
            logical_spatial,
        })
    }

    pub fn project(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        if self.logical_spatial != 1 || input.len() != self.hidden || output.len() != self.hidden {
            return Err("ANE FFN input/output must match hidden size".into());
        }
        for (row, value) in self.input.chunks_exact_mut(64).zip(input) {
            row[..2].copy_from_slice(&value.to_le_bytes());
        }
        self.kernel.write_input(&self.input)?;
        self.kernel.evaluate()?;
        self.kernel.read_output(&mut self.output)?;
        for (row, value) in self.output.chunks_exact(64).zip(output) {
            *value = f16::from_le_bytes([row[0], row[1]]);
        }
        Ok(())
    }

    /// Input/output are token-major [logical_spatial, hidden]. Each complete
    /// batch overwrites every logical input column; padding remains zero.
    pub fn project_batch(&mut self, input: &[f16], output: &mut [f16]) -> Result<(), String> {
        let elements = self
            .hidden
            .checked_mul(self.logical_spatial)
            .ok_or("ANE FFN batch size overflow")?;
        if input.len() != elements || output.len() != elements {
            return Err("ANE FFN batch input/output shape mismatch".into());
        }
        pack_token_columns(input, self.hidden, self.logical_spatial, &mut self.input)?;
        self.kernel.write_input(&self.input)?;
        self.kernel.evaluate()?;
        self.kernel.read_output(&mut self.output)?;
        unpack_token_columns(&self.output, self.hidden, self.logical_spatial, output)
    }
}

fn pack_token_columns(
    input: &[f16],
    hidden: usize,
    tokens: usize,
    packed: &mut [u8],
) -> Result<(), String> {
    if hidden == 0
        || !(1..=8).contains(&tokens)
        || hidden.checked_mul(tokens) != Some(input.len())
        || hidden.checked_mul(64) != Some(packed.len())
    {
        return Err("ANE packed token column shape mismatch".into());
    }
    for (channel, row) in packed.chunks_exact_mut(64).enumerate() {
        // Also erase unused lanes if a caller reuses a previously filled buffer.
        row.fill(0);
        for token in 0..tokens {
            row[2 * token..2 * token + 2]
                .copy_from_slice(&input[token * hidden + channel].to_le_bytes());
        }
    }
    Ok(())
}

fn unpack_token_columns(
    packed: &[u8],
    channels: usize,
    tokens: usize,
    output: &mut [f16],
) -> Result<(), String> {
    if channels == 0
        || !(1..=8).contains(&tokens)
        || channels.checked_mul(tokens) != Some(output.len())
        || channels.checked_mul(64) != Some(packed.len())
    {
        return Err("ANE output token column shape mismatch".into());
    }
    for (token, values) in output.chunks_exact_mut(channels).enumerate() {
        for (channel, value) in values.iter_mut().enumerate() {
            let offset = channel * 64 + token * 2;
            *value = f16::from_le_bytes([packed[offset], packed[offset + 1]]);
        }
    }
    Ok(())
}

fn ffn_mil(hidden: usize, intermediate: usize, offsets: [usize; 3]) -> String {
    let [gate, up, down] = offsets;
    let weights = format!(
        r#"        tensor<fp16, [{intermediate}, {hidden}, 1, 1]> Wg = const()[name = string("Wg"), val = tensor<fp16, [{intermediate}, {hidden}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({gate})))];
        tensor<fp16, [{intermediate}, {hidden}, 1, 1]> Wu = const()[name = string("Wu"), val = tensor<fp16, [{intermediate}, {hidden}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({up})))];
        tensor<fp16, [{hidden}, {intermediate}, 1, 1]> Wd = const()[name = string("Wd"), val = tensor<fp16, [{hidden}, {intermediate}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({down})))];"#
    );
    ffn_mil_with_constants(hidden, intermediate, &weights)
}

fn ffn_mil_with_constants(hidden: usize, intermediate: usize, weights: &str) -> String {
    ffn_mil_batched(hidden, intermediate, weights, 1)
}

fn ffn_mil_batched(hidden: usize, intermediate: usize, weights: &str, spatial: usize) -> String {
    let projections = format!(
        r#"        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gate = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wg, x = x)[name = string("gate")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> up = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wu, x = x)[name = string("up")];"#
    );
    ffn_mil_with_projections(hidden, intermediate, weights, &projections, spatial)
}

fn ffn_mil_stacked(hidden: usize, intermediate: usize, weights: &str) -> String {
    let combined = 2 * intermediate;
    let projections = format!(
        r#"        tensor<fp16, [1, {combined}, 1, 1]> gate_up = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wgu, x = x)[name = string("gate_up")];
        tensor<int32, [4]> gate_begin = const()[name = string("gate_begin"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [4]> up_begin = const()[name = string("up_begin"), val = tensor<int32, [4]>([0, {intermediate}, 0, 0])];
        tensor<int32, [4]> branch_size = const()[name = string("branch_size"), val = tensor<int32, [4]>([1, {intermediate}, 1, 1])];
        tensor<fp16, [1, {intermediate}, 1, 1]> gate = slice_by_size(x = gate_up, begin = gate_begin, size = branch_size)[name = string("gate")];
        tensor<fp16, [1, {intermediate}, 1, 1]> up = slice_by_size(x = gate_up, begin = up_begin, size = branch_size)[name = string("up")];"#
    );
    ffn_mil_with_projections(hidden, intermediate, weights, &projections, 1)
}

fn ffn_mil_with_projections(
    hidden: usize,
    intermediate: usize,
    weights: &str,
    projections: &str,
    spatial: usize,
) -> String {
    format!(
        r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {hidden}, 1, {spatial}]> x) {{
        string pad_type = const()[name = string("pad_type"), val = string("valid")];
        tensor<int32, [2]> strides = const()[name = string("strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pad = const()[name = string("pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dilations = const()[name = string("dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 groups = const()[name = string("groups"), val = int32(1)];
        fp16 gelu_half = const()[name = string("gelu_half"), val = fp16(0.5)];
        fp16 gelu_one = const()[name = string("gelu_one"), val = fp16(1.0)];
        fp16 gelu_cubic = const()[name = string("gelu_cubic"), val = fp16(0.044715)];
        fp16 gelu_root = const()[name = string("gelu_root"), val = fp16(0.7978845608)];
{weights}
{projections}
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gate_squared = mul(x = gate, y = gate)[name = string("gate_squared")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gate_cubed = mul(x = gate_squared, y = gate)[name = string("gate_cubed")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> cubic_scaled = mul(x = gate_cubed, y = gelu_cubic)[name = string("cubic_scaled")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gelu_sum = add(x = gate, y = cubic_scaled)[name = string("gelu_sum")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gelu_argument = mul(x = gelu_sum, y = gelu_root)[name = string("gelu_argument")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gelu_tanh = tanh(x = gelu_argument)[name = string("gelu_tanh")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gelu_factor = add(x = gelu_tanh, y = gelu_one)[name = string("gelu_factor")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gate_halved = mul(x = gate, y = gelu_half)[name = string("gate_halved")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> activated = mul(x = gate_halved, y = gelu_factor)[name = string("activated")];
        tensor<fp16, [1, {intermediate}, 1, {spatial}]> gated = mul(x = activated, y = up)[name = string("gated")];
        tensor<fp16, [1, {hidden}, 1, {spatial}]> y = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wd, x = gated)[name = string("down")];
    }} -> (y);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_ffn_packing_keeps_distinct_columns_and_erases_padding() {
        let input: Vec<_> = [1.0, 2.0, 3.0, -4.0, -5.0, -6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let mut packed = vec![0xff; 3 * 64];
        pack_token_columns(&input, 3, 2, &mut packed).unwrap();
        for (channel, row) in packed.chunks_exact(64).enumerate() {
            assert_eq!(&row[..2], &input[channel].to_le_bytes());
            assert_eq!(&row[2..4], &input[3 + channel].to_le_bytes());
            assert!(row[4..].iter().all(|&byte| byte == 0));
        }
        // A second request must not inherit an old column or old padding.
        pack_token_columns(&[f16::ZERO; 6], 3, 2, &mut packed).unwrap();
        assert!(packed.iter().all(|&byte| byte == 0));
        for (hidden, tokens) in [(0, 2), (3, 0), (3, 9), (usize::MAX, 2), (3, 3)] {
            assert!(pack_token_columns(&input, hidden, tokens, &mut packed).is_err());
            assert!(packed.iter().all(|&byte| byte == 0));
        }
    }

    #[test]
    fn batch_projection_reads_each_logical_column_and_rejects_bad_shapes() {
        // Populate the physical layout independently of the packing helper.
        for tokens in [1, 2, 8] {
            let mut packed = vec![0xff; 3 * 64];
            for channel in 0..3 {
                for token in 0..tokens {
                    let value = f16::from_f32((channel * 11 + token) as f32 - 4.0);
                    let offset = channel * 64 + token * 2;
                    packed[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                }
            }
            let mut output = vec![f16::NAN; 3 * tokens];
            unpack_token_columns(&packed, 3, tokens, &mut output).unwrap();
            for token in 0..tokens {
                for channel in 0..3 {
                    assert_eq!(
                        output[token * 3 + channel].to_f32(),
                        (channel * 11 + token) as f32 - 4.0
                    );
                }
            }
            let mut repacked = vec![0xff; packed.len()];
            pack_token_columns(&output, 3, tokens, &mut repacked).unwrap();
            for (actual, expected) in repacked.chunks_exact(64).zip(packed.chunks_exact(64)) {
                assert_eq!(&actual[..tokens * 2], &expected[..tokens * 2]);
                assert!(actual[tokens * 2..].iter().all(|&byte| byte == 0));
            }
        }
        let mut output = [f16::ONE; 6];
        for (channels, tokens) in [(0, 2), (3, 0), (3, 9), (usize::MAX, 2), (3, 3)] {
            assert!(unpack_token_columns(&[0; 192], channels, tokens, &mut output).is_err());
            assert_eq!(output, [f16::ONE; 6]);
        }
        assert!(unpack_token_columns(&[0; 191], 3, 2, &mut output).is_err());
        assert_eq!(output, [f16::ONE; 6]);
    }

    #[test]
    fn batch_ffn_changes_logical_width_without_changing_math_or_io_count() {
        let original = ffn_mil_with_constants(3840, 15360, "WEIGHTS");
        for tokens in [2, 3, 8] {
            let batch = ffn_mil_batched(3840, 15360, "WEIGHTS", tokens);
            let expected = original
                .replace("[1, 3840, 1, 1]", &format!("[1, 3840, 1, {tokens}]"))
                .replace("[1, 15360, 1, 1]", &format!("[1, 15360, 1, {tokens}]"));
            assert_eq!(batch, expected);
            assert_eq!(batch.matches(" = conv(").count(), 3);
            assert_eq!(batch.matches("func main<ios18>(tensor<fp16,").count(), 1);
            assert_eq!(batch.matches("} -> (y);").count(), 1);
        }
    }

    #[test]
    fn batch_ffn_rejects_unbounded_width_before_private_api_access() {
        let weights = AneInt8FfnWeights::quantize(
            &[f16::ONE; 32 * 64],
            &[f16::ONE; 32 * 64],
            &[f16::ONE; 32 * 64],
            32,
            64,
        )
        .unwrap();
        for tokens in [0, 1, 9, usize::MAX] {
            let error = AneGatedFfn::compile_int8_batch_with_cache_policy(
                &weights,
                tokens,
                AneProgramCachePolicy::RequireExisting,
            )
            .err()
            .expect("unsupported logical width must fail without device access");
            assert!(error.contains("2..=8 logical tokens"));
        }
    }

    #[test]
    fn ffn_projection_refactor_preserves_qualified_cache_identity() {
        assert_eq!(
            ffn_mil(3840, 15360, [64, 117_964_928, 235_929_792]),
            include_str!("../tests/fixtures/gemma4-fp16-ffn.mil")
        );
        let original = ffn_mil_with_constants(3840, 15360, "WEIGHTS");
        let stacked = ffn_mil_stacked(3840, 15360, "WEIGHTS");
        // The precision-sensitive activation/down tail remains byte-identical.
        let tail = "        tensor<fp16, [1, 15360, 1, 1]> gate_squared";
        assert_eq!(
            original.split_once(tail).unwrap().1,
            stacked.split_once(tail).unwrap().1
        );
        assert_eq!(stacked.matches(" = conv(").count(), 2);
        assert!(stacked.contains("[1, 30720, 1, 1]> gate_up"));
        assert!(stacked.contains("tensor<int32, [4]>([0, 15360, 0, 0])"));
    }

    #[test]
    fn projection_refactor_preserves_qualified_fp16_cache_identity() {
        // Captured from the qualified Gemma sliding-attention O projection.
        // SHA256: 92b9c5ea08b204f71ee959a31feaaa096e5dec586a58b0fce6569bcea051fb41.
        assert_eq!(
            linear_mil(4096, 3840, 1),
            include_str!("../tests/fixtures/gemma4-fp16-linear.mil")
        );
    }

    #[test]
    fn int8_projection_keeps_graph_and_rejects_invalid_io_before_compilation() {
        let weights = AneInt8LinearWeights::quantize(&vec![f16::ONE; 64 * 128], 64, 128).unwrap();
        let (_, constants) = weights.blob_and_constants();
        let compressed = linear_mil_with_constants(64, 128, 1, &constants);
        let dense = linear_mil(64, 128, 1);
        // Only the weight declaration changes; tensor I/O, convolution and
        // return topology must remain identical to the qualified single I/O.
        let without_weights = |mil: &str| {
            mil.lines()
                .filter(|line| !line.contains("]> W = "))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert_eq!(without_weights(&compressed), without_weights(&dense));
        assert!(compressed.contains("W = constexpr_affine_dequantize()"));
        assert!(compressed.contains("quantized_data = tensor<int8, [128, 64, 1, 1]>"));
        for spatial in [0, usize::MAX, u32::MAX as usize] {
            assert!(AneLinear::compile_int8_with_cache_policy(
                &weights,
                spatial,
                AneProgramCachePolicy::RequireExisting,
            )
            .is_err());
        }
        for (logical, physical) in [(1, 32), (31, 32), (32, 32), (33, 64), (64, 64)] {
            let layout = LinearLayout::new(64, 128, logical).unwrap();
            assert_eq!(layout.physical_spatial, physical);
            assert_eq!(layout.input_bytes, 64 * physical * 2);
            assert_eq!(layout.output_bytes, 128 * physical * 2);
        }
    }

    #[test]
    #[ignore = "private ANE single-I/O INT8 projection smoke; at most two compiles, six evaluations; explicit receipt"]
    fn hardware_int8_projection_matches_cpu_and_dense_reconstruction() {
        let receipt = std::path::PathBuf::from(std::env::var("RVLLM_INT8_LINEAR_RECEIPT").unwrap());
        assert!(!receipt.exists());
        assert_eq!(compile_budget_used(), 0);
        let (input_channels, output_channels) = (64, 128);
        let original: Vec<_> = (0..input_channels * output_channels)
            .map(|i| f16::from_f32(((i * 17 + 3) % 31) as f32 / 128.0 - 0.125))
            .collect();
        let weights =
            AneInt8LinearWeights::quantize(&original, input_channels, output_channels).unwrap();
        let reconstructed = weights.dequantized();
        let policy = AneProgramCachePolicy::ReuseOrCompileUpTo(2);
        let mut compressed =
            AneLinear::compile_int8_with_cache_policy(&weights, 1, policy).unwrap();
        let mut dense = AneLinear::compile_with_cache_policy(
            &reconstructed,
            input_channels,
            output_channels,
            1,
            policy,
        )
        .unwrap();
        let mut cases = Vec::new();
        let mut violations = 0;
        for step in 0..3 {
            let mut input: Vec<_> = (0..input_channels)
                .map(|i| f16::from_f32(((i * 7 + step) % 23) as f32 / 32.0 - 0.25))
                .collect();
            if step == 2 {
                input[17] = f16::from_f32(171.5);
            }
            let mut actual = vec![f16::ZERO; output_channels];
            let mut control = actual.clone();
            compressed.project(&input, &mut actual).unwrap();
            dense.project(&input, &mut control).unwrap();
            let mut maximum_cpu_error = [0.0_f32; 2];
            let mut maximum_backend_difference = 0.0_f32;
            for ((row, actual), control) in reconstructed
                .chunks_exact(input_channels)
                .zip(&actual)
                .zip(&control)
            {
                let expected: f32 = row
                    .iter()
                    .zip(&input)
                    .map(|(w, x)| w.to_f32() * x.to_f32())
                    .sum();
                let tolerance = 0.002 + 0.01 * expected.abs();
                for (index, value) in [actual, control].into_iter().enumerate() {
                    let error = (value.to_f32() - expected).abs();
                    if !value.is_finite() || error > tolerance {
                        violations += 1;
                    }
                    maximum_cpu_error[index] = maximum_cpu_error[index].max(error);
                }
                maximum_backend_difference =
                    maximum_backend_difference.max((actual.to_f32() - control.to_f32()).abs());
            }
            cases.push(serde_json::json!({"input_case":step,
                "int8_cpu_maximum_absolute_error":maximum_cpu_error[0],
                "dense_cpu_maximum_absolute_error":maximum_cpu_error[1],
                "int8_dense_maximum_absolute_difference":maximum_backend_difference}));
        }
        drop(dense);
        drop(compressed);
        let result = serde_json::json!({"schema":"rvllm.int8_linear_smoke.v1",
            "compiler_calls":compile_budget_used(),"evaluations":6,
            "input_channels":input_channels,"output_channels":output_channels,
            "input_output_count":1,"cases":cases,"violations":violations,
            "tolerance":"0.002 + 0.01 * abs(cpu_reference)",
            "claim":"Synthetic projection backend qualification only; not Gemma quality or speed."});
        std::fs::write(receipt, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
        assert_eq!(violations, 0);
        assert!(compile_budget_used() <= 2);
    }

    #[test]
    #[ignore = "private ANE single-I/O INT8 representation smoke; at most two compiles, six evaluations; explicit receipt"]
    fn hardware_exact_int8_palette_matches_affine() {
        let receipt =
            std::path::PathBuf::from(std::env::var("RVLLM_INT8_PALETTE_RECEIPT").unwrap());
        assert!(!receipt.exists());
        assert_eq!(compile_budget_used(), 0);
        let (hidden, intermediate) = (64, 128);
        let gate: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 11 + 5) % 31) as f32 / 256.0 - 0.06))
            .collect();
        let up: Vec<_> = gate.iter().rev().copied().collect();
        let down: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 7 + 3) % 47) as f32 / 256.0 - 0.09))
            .collect();
        let weights = AneInt8FfnWeights::quantize(&gate, &up, &down, hidden, intermediate).unwrap();
        let reference = weights.dequantized();
        assert!(reference
            .iter()
            .zip(weights.dequantized_lut8_reference().unwrap())
            .all(|(a, b)| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())));
        let policy = AneProgramCachePolicy::ReuseOrCompileUpTo(2);
        let mut affine = AneGatedFfn::compile_int8_with_cache_policy(&weights, policy).unwrap();
        let mut palette =
            AneGatedFfn::compile_int8_palette_with_cache_policy(&weights, policy).unwrap();
        let mut maximum = 0.0_f32;
        let mut identical = true;
        for step in 0..3 {
            let input: Vec<_> = (0..hidden)
                .map(|i| f16::from_f32(((i + step * 7) % 19) as f32 / 32.0))
                .collect();
            let mut actual = vec![f16::ZERO; hidden];
            let mut expected = actual.clone();
            affine.project(&input, &mut expected).unwrap();
            palette.project(&input, &mut actual).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!(a.is_finite() && b.is_finite());
                maximum = maximum.max((a.to_f32() - b.to_f32()).abs());
                identical &= a.to_bits() == b.to_bits();
            }
        }
        drop(palette);
        drop(affine);
        let result = serde_json::json!({"schema":"rvllm.exact_int8_palette_smoke.v1",
            "compiler_calls":compile_budget_used(),"evaluations":6,"maximum_absolute_error":maximum,
            "output_bits_identical":identical,"input_output_count":1,"hidden":hidden,"intermediate":intermediate});
        std::fs::write(receipt, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
        assert!(maximum < 0.002);
        assert!(compile_budget_used() <= 2);
    }

    #[test]
    #[ignore = "single-I/O stacked INT8 smoke: explicit journal/receipt, at most two compiles and six evaluations"]
    fn hardware_stacked_int8_matches_affine_and_cpu() {
        let receipt = std::env::var_os("RVLLM_ANE_STACKED_SMOKE_REPORT").expect("receipt required");
        assert!(!std::path::Path::new(&receipt).exists());
        assert!(std::env::var_os("RVLLM_ANE_DIAGNOSTIC_JOURNAL").is_some());
        assert_eq!(compile_budget_used(), 0);
        let (hidden, intermediate) = (64, 128);
        let gate: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 11 + 5) % 31) as f32 / 256.0 - 0.06))
            .collect();
        let up: Vec<_> = gate.iter().rev().copied().collect();
        let down: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 7 + 3) % 47) as f32 / 256.0 - 0.09))
            .collect();
        let weights = AneInt8FfnWeights::quantize(&gate, &up, &down, hidden, intermediate).unwrap();
        let [gate, up, down] = weights.dequantized();
        let policy = AneProgramCachePolicy::ReuseOrCompileUpTo(2);
        let mut affine = AneGatedFfn::compile_int8_with_cache_policy(&weights, policy).unwrap();
        let mut stacked =
            AneGatedFfn::compile_int8_stacked_with_cache_policy(&weights, policy).unwrap();
        let mut maximum_cpu_error = [0.0_f32; 2];
        let mut maximum_difference = 0.0_f32;
        for step in 0..3 {
            let input: Vec<_> = (0..hidden)
                .map(|i| f16::from_f32(((i + step * 7) % 19) as f32 / 32.0))
                .collect();
            let dot = |row: &[f16], input: &[f16]| {
                row.iter()
                    .zip(input)
                    .map(|(a, b)| a.to_f32() * b.to_f32())
                    .sum::<f32>()
            };
            let gated: Vec<_> = gate
                .chunks_exact(hidden)
                .zip(up.chunks_exact(hidden))
                .map(|(g, u)| {
                    let g = f16::from_f32(dot(g, &input)).to_f32();
                    let u = f16::from_f32(dot(u, &input)).to_f32();
                    let activated = f16::from_f32(
                        0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
                    );
                    f16::from_f32(activated.to_f32() * u)
                })
                .collect();
            let reference: Vec<_> = down
                .chunks_exact(intermediate)
                .map(|row| dot(row, &gated))
                .collect();
            let mut outputs: [Vec<f16>; 2] = std::array::from_fn(|_| vec![f16::ZERO; hidden]);
            affine.project(&input, &mut outputs[0]).unwrap();
            stacked.project(&input, &mut outputs[1]).unwrap();
            for (kind, out) in outputs.iter().enumerate() {
                for (a, b) in out.iter().zip(&reference) {
                    assert!(a.is_finite() && b.is_finite());
                    maximum_cpu_error[kind] = maximum_cpu_error[kind].max((a.to_f32() - b).abs());
                }
            }
            for (a, b) in outputs[0].iter().zip(&outputs[1]) {
                maximum_difference = maximum_difference.max((a.to_f32() - b.to_f32()).abs());
            }
        }
        drop(stacked);
        drop(affine);
        let result = serde_json::json!({"schema":"rvllm.stacked_int8_smoke.v1",
            "compiler_calls":compile_budget_used(),"evaluations":6,"models_dropped":2,
            "maximum_cpu_error_affine_stacked":maximum_cpu_error,"maximum_backend_difference":maximum_difference,
            "hidden":hidden,"intermediate":intermediate,"input_output_count":1});
        std::fs::write(receipt, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
        assert!(maximum_cpu_error.iter().all(|&error| error < 0.002));
        assert!(maximum_difference < 0.002);
        assert!(compile_budget_used() <= 2);
    }

    #[test]
    #[ignore = "executes private ANE API; two programs, six evaluations"]
    fn hardware_int8_ffn_matches_dense_reconstruction() {
        let (hidden, intermediate) = (64, 128);
        let gate: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 11 + 5) % 31) as f32 / 256.0 - 0.06))
            .collect();
        let up: Vec<_> = gate.iter().rev().copied().collect();
        let down: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 7 + 3) % 47) as f32 / 256.0 - 0.09))
            .collect();
        let weights = AneInt8FfnWeights::quantize(&gate, &up, &down, hidden, intermediate).unwrap();
        let [gate, up, down] = weights.dequantized();
        let mut compressed = AneGatedFfn::compile_int8(&weights).unwrap();
        let mut dense = AneGatedFfn::compile(&gate, &up, &down, hidden, intermediate).unwrap();
        for step in 0..3 {
            let input: Vec<_> = (0..hidden)
                .map(|i| f16::from_f32(((i + step * 7) % 19) as f32 / 32.0))
                .collect();
            let mut actual = vec![f16::ZERO; hidden];
            let mut expected = vec![f16::ZERO; hidden];
            compressed.project(&input, &mut actual).unwrap();
            dense.project(&input, &mut expected).unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!((actual.to_f32() - expected.to_f32()).abs() < 0.002);
            }
        }
    }

    #[test]
    #[ignore = "executes private ANE API; two programs, six evaluations"]
    fn hardware_lut4_ffn_matches_dense_reconstruction() {
        let (hidden, intermediate) = (64, 128);
        let gate: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 11 + 5) % 31) as f32 / 256.0 - 0.06))
            .collect();
        let up: Vec<_> = gate.iter().rev().copied().collect();
        let down: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 7 + 3) % 47) as f32 / 256.0 - 0.09))
            .collect();
        let weights = AneLut4FfnWeights::quantize(&gate, &up, &down, hidden, intermediate).unwrap();
        let [gate, up, down] = weights.dequantized();
        let mut compressed =
            AneGatedFfn::compile_lut4_with_cache_policy(&weights, AneProgramCachePolicy::Compile)
                .unwrap();
        let mut dense = AneGatedFfn::compile(&gate, &up, &down, hidden, intermediate).unwrap();
        for step in 0..3 {
            let input: Vec<_> = (0..hidden)
                .map(|i| f16::from_f32(((i + step * 7) % 19) as f32 / 32.0))
                .collect();
            let mut actual = vec![f16::ZERO; hidden];
            let mut expected = vec![f16::ZERO; hidden];
            compressed.project(&input, &mut actual).unwrap();
            dense.project(&input, &mut expected).unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!((actual.to_f32() - expected.to_f32()).abs() < 0.002);
            }
        }
    }

    #[test]
    fn invalid_shapes_fail_before_compilation() {
        assert!(AneLinear::compile(&[], 0, 16, 16).is_err());
        assert!(AneLinear::compile(&[f16::ONE], usize::MAX, 2, 16).is_err());
        assert!(AneLinear::compile(&[f16::ONE], 1, 1, 0).is_err());
        assert!(io_bytes(usize::MAX, 2).is_err());
    }

    #[test]
    #[ignore = "executes private ANE API on Apple Silicon"]
    fn hardware_projection_matches_independent_cpu_dot_products() {
        let input_channels = 64;
        let output_channels = 128;
        let weights: Vec<_> = (0..input_channels * output_channels)
            .map(|i| f16::from_f32(((i * 17 + 3) % 31) as f32 / 128.0 - 0.125))
            .collect();
        for spatial in [1, 16, 32, 64] {
            let mut linear =
                AneLinear::compile(&weights, input_channels, output_channels, spatial).unwrap();
            for step in 0..3 {
                let input: Vec<_> = (0..input_channels)
                    .map(|i| f16::from_f32(((i * 7 + step) % 23) as f32 / 32.0 - 0.25))
                    .collect();
                let mut actual = vec![f16::ZERO; output_channels];
                linear.project(&input, &mut actual).unwrap();
                for (row, actual) in weights.chunks_exact(input_channels).zip(actual) {
                    let expected: f32 = row
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| w.to_f32() * x.to_f32())
                        .sum();
                    assert!(
                        (actual.to_f32() - expected).abs() < 0.002,
                        "actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "executes private ANE API on Apple Silicon"]
    fn hardware_gemma_ffn_matches_cpu() {
        let (hidden, intermediate) = (64, 128);
        let gate: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 11 + 5) % 31) as f32 / 256.0 - 0.06))
            .collect();
        let up: Vec<_> = gate.iter().rev().copied().collect();
        let down: Vec<_> = (0..hidden * intermediate)
            .map(|i| f16::from_f32(((i * 7 + 3) % 47) as f32 / 256.0 - 0.09))
            .collect();
        let mut ffn = AneGatedFfn::compile(&gate, &up, &down, hidden, intermediate).unwrap();
        for step in 0..3 {
            let input: Vec<_> = (0..hidden)
                .map(|i| f16::from_f32(((i + step * 7) % 19) as f32 / 32.0))
                .collect();
            let dot = |row: &[f16], x: &[f16]| {
                f16::from_f32(
                    row.iter()
                        .zip(x)
                        .map(|(a, b)| a.to_f32() * b.to_f32())
                        .sum(),
                )
            };
            let intermediate_values: Vec<_> = gate
                .chunks_exact(hidden)
                .zip(up.chunks_exact(hidden))
                .map(|(g, u)| {
                    let g = dot(g, &input).to_f32();
                    let u = dot(u, &input).to_f32();
                    let activated = f16::from_f32(
                        0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
                    );
                    f16::from_f32(activated.to_f32() * u)
                })
                .collect();
            let mut output = vec![f16::ZERO; hidden];
            ffn.project(&input, &mut output).unwrap();
            for (row, actual) in down.chunks_exact(intermediate).zip(output) {
                let expected = dot(row, &intermediate_values).to_f32();
                assert!(
                    (actual.to_f32() - expected).abs() < 0.002,
                    "FFN {actual} != {expected}"
                );
            }
        }
    }
}


impl AneGatedFfn {
    /// Explicit second-wave INT8 layout experiment. The existing single-I/O
    /// owner and logical width-one projection boundary are not changed.
    pub fn compile_int8_wave2_with_cache_policy(
        weights: &AneInt8FfnWeights,
        variant: crate::ane_int8_candidates::wave2::FfnVariant,
        policy: AneProgramCachePolicy,
    ) -> Result<Self, String> {
        let source = crate::ane_int8_candidates::wave2::build(weights, variant)?;
        tracing::debug!(
            candidate = source.name,
            source_blob_bytes = source.blob.len(),
            convolutions = source.budget.convolutions,
            "ANE candidate source"
        );
        Self::compile_program(&source.mil, &source.blob, weights.shape().0, policy)
    }
}
