//! Device correctness screen for the imported Gemma 4 Metal candidates.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "macos")]
mod macos {
    use half::bf16;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLDevice, MTLSize,
    };
    use rvllm_apple_metal::{MetalBufferArena, MetalContext};
    use serde_json::json;
    use std::{path::PathBuf, ptr::NonNull};

    type Result<T = ()> = std::result::Result<T, String>;

    fn bytes_u16(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn bf(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect()
    }
    fn read_u16(arena: &MetalBufferArena, offset: usize, count: usize) -> Vec<u16> {
        unsafe {
            let p = arena.buffer().contents().as_ptr().add(offset) as *const u16;
            std::slice::from_raw_parts(p, count).to_vec()
        }
    }
    fn set_u32(e: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &u32, index: usize) {
        unsafe {
            e.setBytes_length_atIndex(NonNull::from(value).cast(), 4, index);
        }
    }
    fn dispatch(
        ctx: &MetalContext,
        pso: &ProtocolObject<dyn MTLComputePipelineState>,
        arena: &MetalBufferArena,
        offsets: &[usize],
        constants: &[(u32, usize)],
        groups: usize,
        threads: usize,
    ) -> Result<f64> {
        let command = ctx
            .queue_retained()
            .commandBuffer()
            .ok_or("command buffer")?;
        let encoder = command.computeCommandEncoder().ok_or("compute encoder")?;
        encoder.setComputePipelineState(pso);
        for (index, &offset) in offsets.iter().enumerate() {
            unsafe { encoder.setBuffer_offset_atIndex(Some(arena.buffer()), offset, index) };
        }
        for &(value, index) in constants {
            set_u32(&encoder, &value, index);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            return Err(format!("GPU error: {error}"));
        }
        Ok((command.GPUEndTime() - command.GPUStartTime()) * 1000.0)
    }
    fn upload(arena: &mut MetalBufferArena, name: &str, data: &[u8]) -> Result<usize> {
        let region = arena
            .region(name, data.len(), 256)
            .map_err(|e| e.to_string())?;
        unsafe {
            arena
                .write_region(&region, data)
                .map_err(|e| e.to_string())?
        };
        Ok(region.offset)
    }
    fn close(actual: u16, expected: f32, tolerance: f32) -> bool {
        (bf16::from_bits(actual).to_f32() - expected).abs() <= tolerance
    }

    fn qmv(ctx: &MetalContext, q4: bool) -> Result<serde_json::Value> {
        const K: usize = 64;
        const N: usize = 17;
        let x_f: Vec<f32> = (0..K)
            .map(|i| ((i as i32 % 11) - 5) as f32 / 16.0)
            .collect();
        let x = bf(&x_f);
        let groups = K / 64;
        let mut scales = vec![0u16; N * groups];
        let mut biases = vec![0u16; N * groups];
        let mut expected = vec![0.0f32; N];
        let w_len = if q4 { N * K / 2 } else { N * K };
        let mut w = vec![0u8; w_len];
        for row in 0..N {
            let scale = 0.015625 * (1 + row % 3) as f32;
            let bias = -0.0625 + 0.015625 * (row % 5) as f32;
            scales[row] = bf16::from_f32(scale).to_bits();
            biases[row] = bf16::from_f32(bias).to_bits();
            for k in 0..K {
                let q = ((row * 7 + k * 3 + 1) % if q4 { 16 } else { 256 }) as u8;
                if q4 {
                    let p = row * K / 2 + k / 2;
                    if k % 2 == 0 {
                        w[p] |= q
                    } else {
                        w[p] |= q << 4
                    }
                } else {
                    w[row * K + k] = q;
                }
                expected[row] += x_f[k] * (scale * q as f32 + bias);
            }
        }
        let mut arena = MetalBufferArena::new(ctx.device(), 1 << 20).map_err(|e| e.to_string())?;
        let xo = upload(&mut arena, "x", &bytes_u16(&x))?;
        let wo = upload(&mut arena, "w", &w)?;
        let so = upload(&mut arena, "scales", &bytes_u16(&scales))?;
        let bo = upload(&mut arena, "biases", &bytes_u16(&biases))?;
        let yo = upload(&mut arena, "y", &bytes_u16(&vec![0x7e55; N + 8]))?;
        let name = if q4 {
            "gemma4_qmv_q4_g64_r8_sg2"
        } else {
            "gemma4_qmv_q8_g64_r8_sg2"
        };
        let pso = ctx.make_pipeline(name).map_err(|e| e.to_string())?;
        let first_ms = dispatch(
            ctx,
            &pso,
            &arena,
            &[xo, wo, so, bo, yo],
            &[(N as u32, 5), (K as u32, 6)],
            N.div_ceil(16),
            64,
        )?;
        let first = read_u16(&arena, yo, N + 8);
        let second_ms = dispatch(
            ctx,
            &pso,
            &arena,
            &[xo, wo, so, bo, yo],
            &[(N as u32, 5), (K as u32, 6)],
            N.div_ceil(16),
            64,
        )?;
        let second = read_u16(&arena, yo, N + 8);
        if first != second {
            return Err(format!("{name}: repeated output changed"));
        }
        for i in 0..N {
            if !close(first[i], expected[i], 0.125 + expected[i].abs() * 0.01) {
                return Err(format!(
                    "{name}: output {i} mismatch: {} vs {}",
                    bf16::from_bits(first[i]).to_f32(),
                    expected[i]
                ));
            }
        }
        if first[N..] != vec![0x7e55; 8] {
            return Err(format!("{name}: trailing guard changed"));
        }
        Ok(
            json!({"kernel":name,"status":"passed","shape":[N,K],"first_gpu_ms":first_ms,"second_gpu_ms":second_ms,"repeated_bit_exact":true,"guard_untouched":true}),
        )
    }

    fn attention(ctx: &MetalContext) -> Result<serde_json::Value> {
        const D: usize = 512;
        const H: usize = 16;
        const L: usize = 3;
        let qf: Vec<f32> = (0..H * D)
            .map(|i| ((i % 13) as f32 - 6.0) / 256.0)
            .collect();
        let kf: Vec<f32> = (0..L * D)
            .map(|i| ((i % 17) as f32 - 8.0) / 128.0)
            .collect();
        let vf: Vec<f32> = (0..L * D).map(|i| ((i % 19) as f32 - 9.0) / 32.0).collect();
        let mut expected = vec![0.0f32; H * D];
        for h in 0..H {
            let scores: Vec<f32> = (0..L)
                .map(|t| (0..D).map(|d| qf[h * D + d] * kf[t * D + d]).sum())
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
            let z: f32 = exps.iter().sum();
            for d in 0..D {
                expected[h * D + d] = (0..L).map(|t| exps[t] / z * vf[t * D + d]).sum();
            }
        }
        let bytes = (H * D + 2 * L * D + H * D + 64) * 2;
        let mut arena =
            MetalBufferArena::new(ctx.device(), bytes + 4096).map_err(|e| e.to_string())?;
        let qo = upload(&mut arena, "q", &bytes_u16(&bf(&qf)))?;
        let ko = upload(&mut arena, "k", &bytes_u16(&bf(&kf)))?;
        let vo = upload(&mut arena, "v", &bytes_u16(&bf(&vf)))?;
        let oo = upload(&mut arena, "out", &bytes_u16(&vec![0x7e55; H * D + 32]))?;
        let pso = ctx
            .make_pipeline("gemma4_global_d512_g16_short_unsplit")
            .map_err(|e| e.to_string())?;
        let first_ms = dispatch(
            ctx,
            &pso,
            &arena,
            &[qo, ko, vo, oo],
            &[(L as u32, 4)],
            4,
            128,
        )?;
        let first = read_u16(&arena, oo, H * D + 32);
        let second_ms = dispatch(
            ctx,
            &pso,
            &arena,
            &[qo, ko, vo, oo],
            &[(L as u32, 4)],
            4,
            128,
        )?;
        let second = read_u16(&arena, oo, H * D + 32);
        if first != second {
            return Err("attention repeated output changed".into());
        }
        for i in 0..H * D {
            if !close(first[i], expected[i], 0.01) {
                return Err(format!(
                    "attention output {i} mismatch: {} vs {}",
                    bf16::from_bits(first[i]).to_f32(),
                    expected[i]
                ));
            }
        }
        if first[H * D..] != vec![0x7e55; 32] {
            return Err("attention trailing guard changed".into());
        }
        Ok(
            json!({"kernel":"gemma4_global_d512_g16_short_unsplit","status":"passed","length":L,"first_gpu_ms":first_ms,"second_gpu_ms":second_ms,"repeated_bit_exact":true,"guard_untouched":true}),
        )
    }

    fn ffn(ctx: &MetalContext, q4: bool) -> Result<serde_json::Value> {
        const K: usize = 3840;
        const I: usize = 15360;
        let x = bytes_u16(&bf(&(0..K)
            .map(|i| ((i % 13) as f32 - 6.0) / 32.0)
            .collect::<Vec<_>>()));
        let weight_bytes = if q4 { 2 * I * K / 2 } else { 2 * I * K * 2 };
        let metadata_bytes = if q4 { 2 * I * (K / 64) * 2 } else { 0 };
        let capacity = x.len() + weight_bytes + 2 * metadata_bytes + (I + 32) * 2 + 4096;
        let x_values: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) / 32.0).collect();
        let mut expected = vec![0.0f32; I];
        let mut weight_data = vec![0u8; weight_bytes];
        let mut bias_data = vec![0u8; metadata_bytes];
        if q4 {
            let group_sum: f32 = x_values[..64].iter().sum();
            let groups = K / 64;
            for out in 0..I {
                let gate_bias = 0.0078125 * (1 + out % 3) as f32;
                let up_bias = -0.015625 * (1 + out % 2) as f32;
                bias_data[(out * groups * 2)..(out * groups * 2 + 2)]
                    .copy_from_slice(&bf16::from_f32(gate_bias).to_bits().to_le_bytes());
                let up = (I + out) * groups * 2;
                bias_data[up..up + 2]
                    .copy_from_slice(&bf16::from_f32(up_bias).to_bits().to_le_bytes());
                let gate = gate_bias * group_sum;
                let up_value = up_bias * group_sum;
                expected[out] = 0.5
                    * gate
                    * (1.0 + (0.7978845608 * (gate + 0.044715 * gate * gate * gate)).tanh())
                    * up_value;
            }
        } else {
            for out in 0..I {
                let gate_w = 0.03125 * (1 + out % 3) as f32;
                let up_w = -0.0625 * (1 + out % 2) as f32;
                let gate = (out * K) * 2;
                weight_data[gate..gate + 2]
                    .copy_from_slice(&bf16::from_f32(gate_w).to_bits().to_le_bytes());
                let up = ((I + out) * K) * 2;
                weight_data[up..up + 2]
                    .copy_from_slice(&bf16::from_f32(up_w).to_bits().to_le_bytes());
                let gate_value = x_values[0] * gate_w;
                let up_value = x_values[0] * up_w;
                expected[out] = 0.5
                    * gate_value
                    * (1.0
                        + (0.7978845608
                            * (gate_value + 0.044715 * gate_value * gate_value * gate_value))
                            .tanh())
                    * up_value;
            }
        }
        let mut arena = MetalBufferArena::new(ctx.device(), capacity).map_err(|e| e.to_string())?;
        let xo = upload(&mut arena, "x", &x)?;
        let wo = upload(&mut arena, "w", &weight_data)?;
        let (offsets, constants, name) = if q4 {
            let so = upload(&mut arena, "scales", &vec![0u8; metadata_bytes])?;
            let bo = upload(&mut arena, "biases", &bias_data)?;
            let yo = upload(&mut arena, "activated", &bytes_u16(&vec![0x7e55; I + 32]))?;
            (
                vec![xo, wo, so, bo, yo],
                vec![(K as u32, 5), (I as u32, 6)],
                ("gemma4_ffn_gateup_gelu_q4_g64_r4_sg2", yo),
            )
        } else {
            let yo = upload(&mut arena, "activated", &bytes_u16(&vec![0x7e55; I + 32]))?;
            (
                vec![xo, wo, yo],
                vec![(K as u32, 3), (I as u32, 4)],
                ("gemma4_ffn_gateup_gelu_bf16_r4_sg2", yo),
            )
        };
        let pso = ctx.make_pipeline(name.0).map_err(|e| e.to_string())?;
        let first_ms = dispatch(ctx, &pso, &arena, &offsets, &constants, I.div_ceil(8), 64)?;
        let first = read_u16(&arena, name.1, I + 32);
        let second_ms = dispatch(ctx, &pso, &arena, &offsets, &constants, I.div_ceil(8), 64)?;
        let second = read_u16(&arena, name.1, I + 32);
        if first != second {
            return Err(format!("{}: repeated output changed", name.0));
        }
        for i in 0..I {
            if !close(first[i], expected[i], 0.002 + expected[i].abs() * 0.02) {
                return Err(format!(
                    "{}: output {i} mismatch: {} vs {}",
                    name.0,
                    bf16::from_bits(first[i]).to_f32(),
                    expected[i]
                ));
            }
        }
        if first[I..] != vec![0x7e55; 32] {
            return Err(format!("{}: trailing guard changed", name.0));
        }
        Ok(
            json!({"kernel":name.0,"status":"passed","shape":[1,K,I],"sparse_nonzero_oracle":true,
            "first_gpu_ms":first_ms,"second_gpu_ms":second_ms,"repeated_bit_exact":true,"guard_untouched":true}),
        )
    }

    pub fn run() -> Result {
        let path = PathBuf::from(
            std::env::args_os()
                .nth(1)
                .ok_or("usage: rvllm-metal-next-oracle METALLIB")?,
        );
        let mut ctx = MetalContext::new().map_err(|e| e.to_string())?;
        ctx.load_metallib(&path).map_err(|e| e.to_string())?;
        let cases = vec![
            qmv(&ctx, true)?,
            qmv(&ctx, false)?,
            ffn(&ctx, true)?,
            ffn(&ctx, false)?,
            attention(&ctx)?,
        ];
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "schema":"rvllm.gemma4_metal_next_oracle.v1",
                "status":"passed",
                "device":ctx.device().name().to_string(),
                "cases":cases
            }))
            .map_err(|e| e.to_string())?
        );
        Ok(())
    }
}
#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    macos::run()
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("macOS only");
    std::process::exit(2);
}
