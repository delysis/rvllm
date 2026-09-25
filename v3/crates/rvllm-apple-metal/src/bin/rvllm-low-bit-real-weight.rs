//! Opt-in real-checkpoint referee for experimental Metal W4ABF16/W8ABF16.
//!
//! This is an operator qualification tool, not a model-quality or full-route
//! claim. Checkpoint values, activations, the native baseline, and projection
//! output stay BF16 end to end. Group-32 scales retain their established FP16
//! package encoding and dot products accumulate in FP32.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "macos")]
mod macos {
    use half::{bf16, f16};
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLSize,
    };
    use rvllm_apple::{
        dequantize_apple_low_bit_reference, quantize_apple_low_bit_reference,
        AppleLowBitTensorRole, AppleLowBitWeightFormat, PackedAppleLowBitWeights,
    };
    use rvllm_apple_metal::{
        kernels::kernel_source_for_float_type, weight_loader::scan_safetensor_tensors,
        MetalBufferArena, MetalContext, MetalFloatType, MetalLowBitProjectionOffsets,
        PipelineCache,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "rvllm.metal_low_bit_real_weight_bf16.v1";
    const DEFAULT_MS: &[usize] = &[1, 4];
    const DEFAULT_SAMPLES: usize = 5;
    const SENTINEL: u16 = 0x7e55;

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    enum CandidateSchedule {
        Scalar,
        N4,
        N8,
    }

    impl CandidateSchedule {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "scalar" => Ok(Self::Scalar),
                "n4" => Ok(Self::N4),
                "n8" => Ok(Self::N8),
                _ => Err("--candidate must be scalar, n4, or n8".to_owned()),
            }
        }

        const fn name(self) -> &'static str {
            match self {
                Self::Scalar => "scalar",
                Self::N4 => "n4",
                Self::N8 => "n8",
            }
        }
    }

    #[derive(Debug)]
    struct Args {
        model_dir: PathBuf,
        tensor: String,
        ms: Vec<usize>,
        samples: usize,
        candidate: CandidateSchedule,
    }

    pub(super) fn usage() -> &'static str {
        "usage: rvllm-low-bit-real-weight --model-dir DIR --tensor NAME \
         [--m 1,4] [--samples 5] [--candidate scalar|n4|n8]"
    }

    fn parse_args() -> Result<Args, String> {
        let mut model_dir = None;
        let mut tensor = None;
        let mut ms = DEFAULT_MS.to_vec();
        let mut samples = DEFAULT_SAMPLES;
        let mut candidate = CandidateSchedule::Scalar;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))?;
            match flag.as_str() {
                "--model-dir" => model_dir = Some(PathBuf::from(value)),
                "--tensor" => tensor = Some(value),
                "--m" => {
                    ms = value
                        .split(',')
                        .map(|part| part.parse::<usize>().map_err(|e| e.to_string()))
                        .collect::<Result<Vec<_>, _>>()?;
                    if ms.is_empty() || ms.iter().any(|&m| m == 0 || m > 16) {
                        return Err("--m values must be in 1..=16".to_owned());
                    }
                }
                "--samples" => {
                    samples = value.parse::<usize>().map_err(|e| e.to_string())?;
                    if !(3..=31).contains(&samples) {
                        return Err("--samples must be in 3..=31".to_owned());
                    }
                }
                "--candidate" => candidate = CandidateSchedule::parse(&value)?,
                _ => return Err(format!("unknown option {flag:?}")),
            }
        }
        Ok(Args {
            model_dir: model_dir.ok_or_else(|| "--model-dir is required".to_owned())?,
            tensor: tensor.ok_or_else(|| "--tensor is required".to_owned())?,
            ms,
            samples,
            candidate,
        })
    }

    fn role(name: &str) -> Result<AppleLowBitTensorRole, String> {
        let role = if name.ends_with(".self_attn.q_proj.weight") {
            AppleLowBitTensorRole::QueryProjection
        } else if name.ends_with(".self_attn.k_proj.weight") {
            AppleLowBitTensorRole::KeyProjection
        } else if name.ends_with(".self_attn.v_proj.weight") {
            AppleLowBitTensorRole::ValueProjection
        } else if name.ends_with(".self_attn.o_proj.weight") {
            AppleLowBitTensorRole::OutputProjection
        } else if name.ends_with(".mlp.gate_proj.weight") {
            AppleLowBitTensorRole::DenseGateProjection
        } else if name.ends_with(".mlp.up_proj.weight") {
            AppleLowBitTensorRole::DenseUpProjection
        } else if name.ends_with(".mlp.down_proj.weight") {
            AppleLowBitTensorRole::DenseDownProjection
        } else {
            return Err(
                "tensor is not one of the seven supported dense projection roles".to_owned(),
            );
        };
        Ok(role)
    }

    fn read_exact_range(path: &Path, offset: usize, bytes: usize) -> Result<Vec<u8>, String> {
        let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        file.seek(SeekFrom::Start(offset as u64))
            .map_err(|e| format!("seek {}: {e}", path.display()))?;
        let mut out = vec![0; bytes];
        file.read_exact(&mut out)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        Ok(out)
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn hash_file(path: &Path) -> Result<String, String> {
        let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        Ok(format!("{:x}", hash.finalize()))
    }

    fn decode_weights(
        dtype: rvllm_core::DType,
        raw: &[u8],
    ) -> Result<(Vec<f32>, Vec<bf16>), String> {
        if raw.len() % 2 != 0 {
            return Err("16-bit tensor has odd byte length".to_owned());
        }
        let f32s = raw
            .chunks_exact(2)
            .map(|pair| {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                match dtype {
                    rvllm_core::DType::F16 => Ok(f16::from_bits(bits).to_f32()),
                    rvllm_core::DType::Bf16 => Ok(bf16::from_bits(bits).to_f32()),
                    _ => Err("real-weight referee accepts only F16 or BF16 tensors"),
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(str::to_owned)?;
        let bf16s = f32s.iter().copied().map(bf16::from_f32).collect();
        Ok((f32s, bf16s))
    }

    fn activations(m: usize, k: usize) -> Vec<bf16> {
        (0..m * k)
            .map(|index| {
                let x = ((index.wrapping_mul(131).wrapping_add(17)) % 509) as f32 - 254.0;
                bf16::from_f32(x / 768.0)
            })
            .collect()
    }

    fn bf16_bytes(values: &[bf16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes
    }

    fn f16_bytes(values: &[f16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes
    }

    unsafe fn write_f16(
        arena: &MetalBufferArena,
        region: &rvllm_apple_metal::arena::MetalRegion,
        values: &[f16],
    ) {
        let destination = unsafe { arena.host_ptr(region) }.cast::<u16>();
        for (index, value) in values.iter().enumerate() {
            unsafe { destination.add(index).write(value.to_bits()) };
        }
    }

    unsafe fn write_bf16(
        arena: &MetalBufferArena,
        region: &rvllm_apple_metal::arena::MetalRegion,
        values: &[bf16],
    ) {
        let destination = unsafe { arena.host_ptr(region) }.cast::<u16>();
        for (index, value) in values.iter().enumerate() {
            unsafe { destination.add(index).write(value.to_bits()) };
        }
    }

    unsafe fn read_bf16(
        arena: &MetalBufferArena,
        region: &rvllm_apple_metal::arena::MetalRegion,
        count: usize,
    ) -> Vec<bf16> {
        let source = unsafe { arena.host_ptr(region) }.cast::<u16>();
        (0..count)
            .map(|index| bf16::from_bits(unsafe { source.add(index).read() }))
            .collect()
    }

    unsafe fn set_u32(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        value: &u32,
        index: usize,
    ) {
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(value as *const u32 as *mut _),
                4,
                index,
            )
        };
    }

    unsafe fn set_f32(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        value: &f32,
        index: usize,
    ) {
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(value as *const f32 as *mut _),
                4,
                index,
            )
        };
    }

    fn encode_native(
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        pipelines: &PipelineCache,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        a: usize,
        b: usize,
        c: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| "native encoder unavailable".to_owned())?;
        let m = u32::try_from(m).map_err(|_| "M exceeds u32")?;
        let n = u32::try_from(n).map_err(|_| "N exceeds u32")?;
        let k = u32::try_from(k).map_err(|_| "K exceeds u32")?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        unsafe {
            encoder.setComputePipelineState(
                pipelines.get("gemm_f16_vec8").map_err(|e| e.to_string())?,
            );
            encoder.setBuffer_offset_atIndex(Some(buffer), a, 0);
            encoder.setBuffer_offset_atIndex(Some(buffer), b, 1);
            encoder.setBuffer_offset_atIndex(Some(buffer), c, 2);
            set_u32(&encoder, &m, 3);
            set_u32(&encoder, &n, 4);
            set_u32(&encoder, &k, 5);
            set_f32(&encoder, &alpha, 6);
            set_f32(&encoder, &beta, 7);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: m as usize,
                    height: (n as usize).div_ceil(8),
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        Ok(())
    }

    fn submit(
        ctx: &MetalContext,
        encode: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), String>,
    ) -> Result<f64, String> {
        let command = ctx
            .queue_retained()
            .commandBuffer()
            .ok_or_else(|| "command buffer unavailable".to_owned())?;
        encode(&command)?;
        let start = Instant::now();
        command.commit();
        command.waitUntilCompleted();
        Ok(start.elapsed().as_secs_f64() * 1_000.0)
    }

    fn median(values: &[f64]) -> f64 {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    fn accuracy(actual: &[bf16], expected: &[bf16]) -> Result<Value, String> {
        let mut max_abs = 0.0_f32;
        let mut sum_delta2 = 0.0_f64;
        let mut sum_ref2 = 0.0_f64;
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let a = actual.to_f32();
            let e = expected.to_f32();
            if !a.is_finite() {
                return Err(format!("non-finite output at {index}"));
            }
            let delta = (a - e).abs();
            max_abs = max_abs.max(delta);
            sum_delta2 += f64::from(delta).powi(2);
            sum_ref2 += f64::from(e).powi(2);
            let tolerance = 0.0625 + e.abs() * 0.01;
            if delta > tolerance {
                return Err(format!("output[{index}] delta {delta} exceeds {tolerance}"));
            }
        }
        Ok(
            json!({"max_abs": max_abs, "relative_l2": (sum_delta2 / sum_ref2.max(f64::MIN_POSITIVE)).sqrt()}),
        )
    }

    fn native_reference(
        weights: &[bf16],
        input: &[bf16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<bf16> {
        let mut output = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0_f32;
                for inner in 0..k {
                    sum += input[row * k + inner].to_f32() * weights[column * k + inner].to_f32();
                }
                output.push(bf16::from_f32(sum));
            }
        }
        output
    }

    fn low_bit_reference_bf16(
        weights: &PackedAppleLowBitWeights,
        input: &[bf16],
        m: usize,
    ) -> Result<Vec<bf16>, String> {
        let n = weights.rows();
        let k = weights.k();
        if input.len() != m.checked_mul(k).ok_or("input shape overflow")? {
            return Err("input shape does not match M and K".to_owned());
        }
        let dense = dequantize_apple_low_bit_reference(weights).map_err(|e| e.to_string())?;
        let mut output = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0_f32;
                for inner in 0..k {
                    sum += input[row * k + inner].to_f32() * dense[column * k + inner].to_f32();
                }
                output.push(bf16::from_f32(sum));
            }
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_candidate(
        schedule: CandidateSchedule,
        projection: MetalLowBitProjectionOffsets,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        pipelines: &PipelineCache,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        input_offset: usize,
        output_offset: usize,
        m: usize,
        n: usize,
    ) -> Result<(), String> {
        let result = match schedule {
            CandidateSchedule::Scalar => projection.encode_strided_bf16(
                command,
                pipelines,
                buffer,
                input_offset,
                output_offset,
                m,
                n,
                0,
            ),
            CandidateSchedule::N4 => projection.encode_strided_bf16_n4(
                command,
                pipelines,
                buffer,
                input_offset,
                output_offset,
                m,
                n,
                0,
            ),
            CandidateSchedule::N8 => projection.encode_strided_bf16_n8(
                command,
                pipelines,
                buffer,
                input_offset,
                output_offset,
                m,
                n,
                0,
            ),
        };
        result.map_err(|error| error.to_string())
    }

    fn run_shape(
        ctx: &MetalContext,
        pipelines: &PipelineCache,
        role: AppleLowBitTensorRole,
        format: AppleLowBitWeightFormat,
        source_f32: &[f32],
        native_weights: &[bf16],
        m: usize,
        n: usize,
        k: usize,
        samples: usize,
        schedule: CandidateSchedule,
    ) -> Result<Value, String> {
        let packed = quantize_apple_low_bit_reference(format, n, k, source_f32)
            .map_err(|e| e.to_string())?;
        let input_values = activations(m, k);
        let expected = low_bit_reference_bf16(&packed, &input_values, m)?;
        let native_expected = native_reference(native_weights, &input_values, m, n, k);
        let scale_bytes = packed.scales().len() * 2;
        let output_count = m.checked_mul(n).ok_or("output overflow")?;
        let arena_bytes = input_values.len() * 2
            + native_weights.len() * 2
            + packed.packed_values().len()
            + scale_bytes
            + output_count * 4
            + 4096;
        let mut arena =
            MetalBufferArena::new(ctx.device(), arena_bytes).map_err(|e| e.to_string())?;
        let input = arena
            .region("real_input", input_values.len() * 2, 16)
            .map_err(|e| e.to_string())?;
        let native_w = arena
            .region("real_native_bf16", native_weights.len() * 2, 16)
            .map_err(|e| e.to_string())?;
        let values = arena
            .region("real_low_bit_values", packed.packed_values().len(), 16)
            .map_err(|e| e.to_string())?;
        let scales = arena
            .region("real_low_bit_scales", scale_bytes, 16)
            .map_err(|e| e.to_string())?;
        let native_out = arena
            .region("real_native_output", output_count * 2, 16)
            .map_err(|e| e.to_string())?;
        let low_out = arena
            .region("real_low_bit_output", output_count * 2, 16)
            .map_err(|e| e.to_string())?;
        let guard = arena
            .region("real_output_guard", 64, 16)
            .map_err(|e| e.to_string())?;
        unsafe {
            write_bf16(&arena, &input, &input_values);
            write_bf16(&arena, &native_w, native_weights);
            arena
                .write_region(&values, packed.packed_values())
                .map_err(|e| e.to_string())?;
            write_f16(&arena, &scales, packed.scales());
            write_bf16(
                &arena,
                &low_out,
                &vec![bf16::from_bits(SENTINEL); output_count],
            );
            write_bf16(&arena, &guard, &vec![bf16::from_bits(SENTINEL); 32]);
        }
        let projection = MetalLowBitProjectionOffsets::new_for_role(
            role,
            format,
            n,
            k,
            values.offset,
            values.size,
            scales.offset,
            scales.size,
        )
        .map_err(|e| e.to_string())?;
        let buffer = arena.buffer_retained();

        let before = pipelines.low_bit_dispatch_snapshot();
        submit(ctx, |command| {
            encode_candidate(
                schedule,
                projection,
                command,
                pipelines,
                buffer,
                input.offset,
                low_out.offset,
                m,
                n,
            )
        })?;
        let actual = unsafe { read_bf16(&arena, &low_out, output_count) };
        let low_bit_accuracy = accuracy(&actual, &expected)?;
        let guard_values = unsafe { read_bf16(&arena, &guard, 32) };
        if guard_values.iter().any(|value| value.to_bits() != SENTINEL) {
            return Err("output guard changed".to_owned());
        }
        unsafe {
            write_bf16(
                &arena,
                &low_out,
                &vec![bf16::from_bits(SENTINEL); output_count],
            );
        }
        submit(ctx, |command| {
            encode_candidate(
                schedule,
                projection,
                command,
                pipelines,
                buffer,
                input.offset,
                low_out.offset,
                m,
                n,
            )
        })?;
        let repeated = unsafe { read_bf16(&arena, &low_out, output_count) };
        if repeated
            .iter()
            .zip(&actual)
            .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            return Err("repeated low-bit output bits changed".to_owned());
        }
        let mut exact = [0; AppleLowBitTensorRole::COUNT];
        exact[role.index()] = 2;
        pipelines
            .low_bit_dispatch_snapshot()
            .checked_since(before)
            .map_err(str::to_owned)?
            .verify_exact(format, exact)
            .map_err(str::to_owned)?;

        // Warm both paths before the interleaved A-B-B-A blocks.
        submit(ctx, |command| {
            encode_native(
                command,
                pipelines,
                buffer,
                input.offset,
                native_w.offset,
                native_out.offset,
                m,
                n,
                k,
            )
        })?;
        let native_actual = unsafe { read_bf16(&arena, &native_out, output_count) };
        let native_accuracy = accuracy(&native_actual, &native_expected)?;
        let timing_dispatch_before = pipelines.low_bit_dispatch_snapshot();
        submit(ctx, |command| {
            encode_candidate(
                schedule,
                projection,
                command,
                pipelines,
                buffer,
                input.offset,
                low_out.offset,
                m,
                n,
            )
        })?;
        let mut native_ms = Vec::with_capacity(samples * 2);
        let mut low_ms = Vec::with_capacity(samples * 2);
        for _ in 0..samples {
            native_ms.push(submit(ctx, |command| {
                encode_native(
                    command,
                    pipelines,
                    buffer,
                    input.offset,
                    native_w.offset,
                    native_out.offset,
                    m,
                    n,
                    k,
                )
            })?);
            low_ms.push(submit(ctx, |command| {
                encode_candidate(
                    schedule,
                    projection,
                    command,
                    pipelines,
                    buffer,
                    input.offset,
                    low_out.offset,
                    m,
                    n,
                )
            })?);
            low_ms.push(submit(ctx, |command| {
                encode_candidate(
                    schedule,
                    projection,
                    command,
                    pipelines,
                    buffer,
                    input.offset,
                    low_out.offset,
                    m,
                    n,
                )
            })?);
            native_ms.push(submit(ctx, |command| {
                encode_native(
                    command,
                    pipelines,
                    buffer,
                    input.offset,
                    native_w.offset,
                    native_out.offset,
                    m,
                    n,
                    k,
                )
            })?);
        }
        let timing_dispatch = pipelines
            .low_bit_dispatch_snapshot()
            .checked_since(timing_dispatch_before)
            .map_err(str::to_owned)?;
        let mut timing_exact = [0; AppleLowBitTensorRole::COUNT];
        timing_exact[role.index()] = 1 + (samples * 2) as u64;
        timing_dispatch
            .verify_exact(format, timing_exact)
            .map_err(str::to_owned)?;
        let guard_values = unsafe { read_bf16(&arena, &guard, 32) };
        if guard_values.iter().any(|value| value.to_bits() != SENTINEL) {
            return Err("output guard changed during timing".to_owned());
        }
        let native_median = median(&native_ms);
        let low_median = median(&low_ms);
        let scale_identity = f16_bytes(packed.scales());
        let activation_identity = bf16_bytes(&input_values);
        let expected_identity = bf16_bytes(&expected);
        Ok(json!({
            "m": m, "n": n, "k": k, "accuracy": low_bit_accuracy, "native_accuracy": native_accuracy,
            "guard_unchanged": true, "repeatable_output_bits": true,
            "identity": {
                "packed_values_sha256": sha256(packed.packed_values()),
                "scales_f16le_sha256": sha256(&scale_identity),
                "activations_bf16le_sha256": sha256(&activation_identity),
                "cpu_low_bit_reference_bf16le_sha256": sha256(&expected_identity)
            },
            "dispatch": {"format": format.name(), "role": role.report_name(),
                "candidate_abi": match format { AppleLowBitWeightFormat::W4A16 => "W4ABF16", AppleLowBitWeightFormat::W8A16 => "W8ABF16" },
                "exact_correctness_dispatches_verified": 2,
                "exact_timing_dispatch_count_verified": true,
                "timing_dispatches": 1 + samples * 2},
            "timing": {"method": "ABBA wall-clock commit-to-completion", "samples_per_arm": samples * 2,
                "native_kernel": "gemm_f16_vec8 compiled as typed BF16", "native_weight_conversion": "none",
                "candidate_kernel": match schedule {
                    CandidateSchedule::Scalar => projection.experimental_bf16_kernel_name(),
                    CandidateSchedule::N4 => projection.experimental_bf16_n4_kernel_name(),
                    CandidateSchedule::N8 => projection.experimental_bf16_n8_kernel_name(),
                },
                "activation_dtype": "BF16", "output_dtype": "BF16", "scale_dtype": "F16", "accumulation_dtype": "F32",
                "native_ms": native_ms, "candidate_ms": low_ms,
                "native_median_ms": native_median, "candidate_median_ms": low_median,
                "speedup": native_median / low_median}
        }))
    }

    pub fn run() -> Result<(), String> {
        let args = parse_args()?;
        let tensors = scan_safetensor_tensors(&args.model_dir).map_err(|e| e.to_string())?;
        let info = tensors
            .get(&args.tensor)
            .ok_or_else(|| format!("tensor {:?} is absent", args.tensor))?;
        if info.shape.len() != 2 {
            return Err("tensor must be rank two".to_owned());
        }
        let role = role(&args.tensor)?;
        let raw = read_exact_range(&info.file, info.file_offset, info.nbytes)?;
        let tensor_sha256 = sha256(&raw);
        if info.dtype != rvllm_core::DType::Bf16 {
            return Err("native-BF16 referee requires a BF16 checkpoint tensor".to_owned());
        }
        let (source_f32, native_bf16) = decode_weights(info.dtype, &raw)?;
        let [n, k] = [info.shape[0], info.shape[1]];
        if source_f32.len() != n.checked_mul(k).ok_or("tensor shape overflow")? {
            return Err("tensor payload disagrees with shape".to_owned());
        }
        let mut ctx = MetalContext::new().map_err(|e| e.to_string())?;
        let generated_msl = kernel_source_for_float_type(MetalFloatType::Bf16);
        ctx.compile_library(&generated_msl)
            .map_err(|e| e.to_string())?;
        let mut pipelines = PipelineCache::new();
        let candidate_kernels = match args.candidate {
            CandidateSchedule::Scalar => [
                "experimental_projection_w4abf16_bf16",
                "experimental_projection_w8abf16_bf16",
            ],
            CandidateSchedule::N4 => [
                "experimental_projection_w4abf16_bf16_n4",
                "experimental_projection_w8abf16_bf16_n4",
            ],
            CandidateSchedule::N8 => [
                "experimental_projection_w4abf16_bf16_n8",
                "experimental_projection_w8abf16_bf16_n8",
            ],
        };
        for kernel in ["gemm_f16_vec8", candidate_kernels[0], candidate_kernels[1]] {
            pipelines.compile(&ctx, kernel).map_err(|e| e.to_string())?;
        }
        let mut cases = Vec::new();
        for format in [
            AppleLowBitWeightFormat::W4A16,
            AppleLowBitWeightFormat::W8A16,
        ] {
            for &m in &args.ms {
                cases.push(run_shape(
                    &ctx,
                    &pipelines,
                    role,
                    format,
                    &source_f32,
                    &native_bf16,
                    m,
                    n,
                    k,
                    args.samples,
                    args.candidate,
                )?);
            }
        }
        let config_path = args.model_dir.join("config.json");
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let receipt = json!({
            "schema": SCHEMA, "claim": "real checkpoint projection operator evidence only; not full-route or model-quality evidence",
            "model_dir": args.model_dir, "config_sha256": hash_file(&config_path)?,
            "tensor": args.tensor, "tensor_role": role.report_name(), "source_dtype": format!("{:?}", info.dtype),
            "shape": info.shape, "source_file": info.file, "source_file_offset": info.file_offset,
            "source_tensor_bytes": info.nbytes, "source_tensor_sha256": tensor_sha256,
            "abi": {"activation": "BF16", "output": "BF16", "scales": "F16", "accumulation": "F32"},
            "candidate_schedule": args.candidate.name(),
            "generated_msl_sha256": sha256(generated_msl.as_bytes()), "executable_sha256": hash_file(&executable)?,
            "compile_counts": {"metal_libraries": 1, "pipeline_states": 3}, "cases": cases
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn role_inference_is_exact() {
            assert_eq!(
                role("language_model.model.layers.0.self_attn.q_proj.weight").unwrap(),
                AppleLowBitTensorRole::QueryProjection
            );
            assert_eq!(
                role("language_model.model.layers.0.mlp.down_proj.weight").unwrap(),
                AppleLowBitTensorRole::DenseDownProjection
            );
            assert!(role("language_model.model.layers.0.input_layernorm.weight").is_err());
            assert!(role("x.experts.0.down_proj.weight").is_err());
        }
        #[test]
        fn generated_activations_are_finite_and_bounded() {
            let values = activations(4, 33);
            assert_eq!(values.len(), 132);
            assert!(values
                .iter()
                .all(|v| v.is_finite() && v.to_f32().abs() <= 1.0));
        }
        #[test]
        fn median_is_order_independent() {
            assert_eq!(median(&[9.0, 1.0, 5.0]), 5.0);
        }

        #[test]
        fn candidate_schedule_is_strict() {
            assert_eq!(
                CandidateSchedule::parse("scalar").unwrap(),
                CandidateSchedule::Scalar
            );
            assert_eq!(
                CandidateSchedule::parse("n4").unwrap(),
                CandidateSchedule::N4
            );
            assert_eq!(
                CandidateSchedule::parse("n8").unwrap(),
                CandidateSchedule::N8
            );
            assert!(CandidateSchedule::parse("n16").is_err());
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    if let Err(error) = macos::run() {
        eprintln!("error: {error}\n{}", macos::usage());
        std::process::exit(1);
    }
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("rvllm-low-bit-real-weight requires macOS Metal");
        std::process::exit(1);
    }
}
