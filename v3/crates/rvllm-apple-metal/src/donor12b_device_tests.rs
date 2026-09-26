//! Full-shape dense synthetic operator gates. These are NOT checkpoint, token,
//! full-model, ANE, throughput, or promotion receipts. Libraries are prebuilt,
//! strict-math and source/hash-bound by the existing queue referee Setup.
use crate::arena::MetalRegion;
use crate::attention_global_decode_device_tests::{complete, write_new, Setup, TestResult};
use crate::donor12b::*;
use crate::donor12b_metal::emit_plan_for_oracle;
use crate::research::Gemma12bResearchShape;
use crate::{MetalBufferArena, MetalFloatType};
use half::{bf16, f16};
use objc2_metal::*;
use rvllm_apple::{AppleLowBitTensorRole as Role, AppleLowBitWeightFormat as Format};
use serde_json::{json, Value};
const GUARD: usize = 64;
fn b(x: f32) -> u16 {
    bf16::from_f32(x).to_bits()
}
fn f(x: u16) -> f32 {
    bf16::from_bits(x).to_f32()
}
fn x(t: usize, k: usize) -> f32 {
    f(b(((t * 7 + k * 13) % 31) as f32 / 64.0 - 15.0 / 64.0))
}
fn nw(row: usize, k: usize, salt: usize) -> f32 {
    f(b(
        ((row * 7 + k * 3 + salt) % 29) as f32 / 128.0 - 14.0 / 128.0
    ))
}
fn code(bits: u8, row: usize, k: usize, salt: usize) -> i32 {
    let maximum = if bits == 4 { 7 } else { 127 };
    ((row * 17 + k * 5 + (k / 32) * 3 + salt) % (2 * maximum + 1)) as i32 - maximum as i32
}
fn scale(row: usize, group: usize, salt: usize) -> f32 {
    f16::from_f32((1 + (row + group * 3 + salt) % 7) as f32 / 512.0).to_f32()
}
fn words<I: Iterator<Item = u16>>(values: I) -> Vec<u8> {
    values.flat_map(u16::to_le_bytes).collect()
}
fn activation(m: usize, k: usize) -> Vec<u8> {
    words((0..m * k).map(|i| b(x(i / k, i % k))))
}
fn native_weights(n: usize, k: usize, salt: usize) -> Vec<u8> {
    words((0..n * k).map(|i| b(nw(i / k, i % k, salt))))
}
fn quant(bits: u8, n: usize, k: usize, salt: usize) -> [Vec<u8>; 2] {
    let mut values = vec![0; n * k * bits as usize / 8];
    for row in 0..n {
        for col in 0..k {
            let i = row * k + col;
            let q = code(bits, row, col, salt);
            if bits == 4 {
                values[i / 2] |= ((q & 15) as u8) << (4 * (i % 2));
            } else {
                values[i] = (q & 255) as u8;
            }
        }
    }
    let scales = words(
        (0..n * k / 32).map(|i| f16::from_f32(scale(i / (k / 32), i % (k / 32), salt)).to_bits()),
    );
    [values, scales]
}
fn dot(bits: Option<u8>, row: usize, token: usize, k: usize, salt: usize) -> f64 {
    (0..k)
        .map(|col| {
            f64::from(x(token, col))
                * match bits {
                    Some(bits) => {
                        f64::from(code(bits, row, col, salt))
                            * f64::from(scale(row, col / 32, salt))
                    }
                    None => f64::from(nw(row, col, salt)),
                }
        })
        .sum()
}
fn gelu(x: f32) -> f32 {
    if x >= 5.0 {
        x
    } else if x <= -5.0 {
        0.0
    } else {
        0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())
    }
}
fn owner(setup: &Setup, global: bool, tokens: u32, decode: bool) -> Policy {
    Policy {
        selected: setup.candidate,
        dtype: Some(MetalFloatType::Bf16),
        quantized_accumulation: false,
        decode,
        model: Gemma12bResearchShape {
            tokens,
            hidden: 3840,
            intermediate: 15360,
            layers: 48,
            heads: 16,
            kv_heads: if global { 1 } else { 8 },
            head_dim: if global { 512 } else { 256 },
            attention_window: if global { 0 } else { 1024 },
            moe_experts: 0,
            moe_top_k: 0,
            moe_intermediate: 0,
            ple: 0,
        },
    }
}
struct Data {
    arena: MetalBufferArena,
    inputs: Vec<(MetalRegion, Vec<u8>)>,
    output: (MetalRegion, Vec<u8>),
}
impl Data {
    fn new(setup: &Setup, payloads: Vec<Vec<u8>>, output_bytes: usize) -> TestResult<Self> {
        let mut arena = MetalBufferArena::new(
            setup.context.device(),
            payloads.iter().map(Vec::len).sum::<usize>() + output_bytes + 16384,
        )?;
        let mut upload = |name: &str, payload: Vec<u8>| -> TestResult<(MetalRegion, Vec<u8>)> {
            let mut bytes = vec![0xa5; GUARD];
            bytes.extend(payload);
            bytes.extend([0xa5; GUARD]);
            let region = arena.region(name, bytes.len(), 64)?;
            // SAFETY: new shared arena; no outstanding commands.
            unsafe {
                arena.write_region(&region, &bytes)?;
            }
            Ok((region, bytes))
        };
        let mut inputs = Vec::new();
        for (i, payload) in payloads.into_iter().enumerate() {
            inputs.push(upload(&format!("input-{i}"), payload)?);
        }
        let output = upload("output", vec![0xff; output_bytes])?;
        drop(upload);
        Ok(Self {
            arena,
            inputs,
            output,
        })
    }
    fn input(&self, index: usize) -> usize {
        self.inputs[index].0.offset + GUARD
    }
    fn out(&self) -> usize {
        self.output.0.offset + GUARD
    }
    fn weight(&self, index: usize, bits: u8, role: Role, n: u32, k: u32) -> Weight {
        Weight {
            format: if bits == 4 {
                Format::W4A16
            } else {
                Format::W8A16
            },
            role,
            n,
            k,
            values: self.input(index),
            scales: self.input(index + 1),
        }
    }
    fn reset(&self) -> TestResult {
        // SAFETY: the referee always completes before reusing scratch.
        unsafe {
            self.arena.write_region(&self.output.0, &self.output.1)?;
        }
        Ok(())
    }
    fn read<'a>(&'a self, region: &MetalRegion) -> &'a [u8] {
        // SAFETY: caller reads only before submission or after completion.
        unsafe { std::slice::from_raw_parts(self.arena.host_ptr(region), region.size) }
    }
    fn check(&self) {
        for (region, bytes) in &self.inputs {
            assert_eq!(
                self.read(region),
                bytes.as_slice(),
                "immutable input or input guard changed"
            );
        }
        let bytes = self.read(&self.output.0);
        assert_eq!(&bytes[..GUARD], &[0xa5; GUARD]);
        assert_eq!(&bytes[bytes.len() - GUARD..], &[0xa5; GUARD]);
    }
    fn output(&self) -> &[u8] {
        let bytes = self.read(&self.output.0);
        &bytes[GUARD..bytes.len() - GUARD]
    }
}
fn execute(
    setup: &Setup,
    data: &Data,
    plan: Plan,
    label: &str,
    m: usize,
    n: usize,
    stride: usize,
    column: usize,
    fp32: bool,
    expected: &[(usize, f64)],
    untouched: bool,
) -> TestResult<Value> {
    let mut previous: Option<Vec<u8>> = None;
    let mut max_error = 0_f64;
    for _ in 0..3 {
        data.reset()?;
        let before = setup.pipelines.research_dispatch_snapshot();
        let command = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("no command buffer")?;
        assert!(emit_plan_for_oracle(
            &setup.pipelines,
            &command,
            data.arena.buffer(),
            plan
        )?);
        complete(&command)?;
        data.check();
        let delta = setup
            .pipelines
            .research_dispatch_snapshot()
            .checked_since(before)?;
        for (index, count) in delta.counts.iter().enumerate() {
            assert_eq!(*count, u64::from(index == plan.kernel as usize));
        }
        let bytes = data.output();
        if untouched {
            assert!(bytes.iter().all(|&v| v == 0xff));
        } else {
            let element = if fp32 { 4 } else { 2 };
            for t in 0..m {
                for c in 0..stride {
                    let i = t * stride + c;
                    let raw = &bytes[i * element..(i + 1) * element];
                    if c < column || c >= column + n {
                        assert!(raw.iter().all(|&v| v == 0xff), "stride padding changed");
                    } else {
                        let actual = if fp32 {
                            f32::from_le_bytes(raw.try_into()?)
                        } else {
                            f(u16::from_le_bytes(raw.try_into()?))
                        };
                        assert!(actual.is_finite(), "unwritten/nonfinite output");
                    }
                }
            }
            for &(index, reference) in expected {
                let actual = if fp32 {
                    f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into()?)
                } else {
                    f(u16::from_le_bytes(
                        bytes[index * 2..index * 2 + 2].try_into()?,
                    ))
                };
                let error = (f64::from(actual) - reference).abs();
                max_error = max_error.max(error);
                // Declared operator tolerance, not a model quality criterion.
                let tolerance = 0.0001 + if fp32 { 0.00002 } else { 0.012 } * reference.abs();
                assert!(error<=tolerance,"{label} index {index}: {actual} vs {reference}, error {error}, tolerance {tolerance}");
            }
        }
        if let Some(previous) = &previous {
            assert_eq!(
                bytes,
                previous.as_slice(),
                "repeated scratch reuse changed output"
            );
        }
        previous = Some(bytes.to_vec());
    }
    write_new(&setup.directory.join(format!("{label}.raw")), data.output())?;
    Ok(
        json!({"case":label,"kernel":plan.kernel.name(),"grid":plan.grid,"threads":plan.threads(),
        "sampled_fp64_outputs":expected.len(),"all_outputs_checked_finite":!untouched,
        "repeat_count":3,"max_sampled_absolute_error":max_error,"expected_gpu_no_write":untouched,
        "shape":[m,n],"output_stride":stride,"output_fp32":fp32,"status":"passed"}),
    )
}
fn projection(
    setup: &Setup,
    bits: Option<u8>,
    m: usize,
    n: usize,
    k: usize,
    role: Role,
    label: &str,
) -> TestResult<Value> {
    let fp32 = bits.is_none();
    let stride = n + 8;
    let column = 4;
    let mut payloads = vec![activation(m, k)];
    if let Some(bits) = bits {
        payloads.extend(quant(bits, n, k, 1));
    } else {
        payloads.push(native_weights(n, k, 1));
    }
    let data = Data::new(setup, payloads, m * stride * if fp32 { 4 } else { 2 })?;
    let request = ProjectionRequest {
        selected: setup.candidate,
        dtype: Some(MetalFloatType::Bf16),
        quantized_accumulation: false,
        shape: [m as u32, n as u32, k as u32],
        activation: data.input(0),
        native_weights: data.input(1),
        low_bit: bits.map(|bits| data.weight(1, bits, role, n as u32, k as u32)),
        output: data.out(),
        output_stride: stride as u32,
        output_column: column as u32,
        output_f32: fp32,
        arena_bytes: data.arena.capacity(),
    };
    assert!(ProjectionRequest {
        output: request.activation,
        ..request
    }
    .plan()
    .is_none());
    assert!(ProjectionRequest {
        dtype: Some(MetalFloatType::F16),
        ..request
    }
    .plan()
    .is_none());
    let mut expected = Vec::new();
    for t in 0..m {
        for row in [0, 1, 3, 7, n / 2, n - 1] {
            expected.push((t * stride + column + row, dot(bits, row, t, k, 1)));
        }
    }
    execute(
        setup,
        &data,
        request.plan().ok_or("projection refused")?,
        label,
        m,
        n,
        stride,
        column,
        fp32,
        &expected,
        false,
    )
}
fn gate(setup: &Setup, bits: Option<u8>, label: &str) -> TestResult<Value> {
    let mut payloads = vec![activation(1, 3840)];
    if let Some(bits) = bits {
        payloads.extend(quant(bits, 15360, 3840, 1));
        payloads.extend(quant(bits, 15360, 3840, 4));
    } else {
        let mut weights = native_weights(15360, 3840, 1);
        weights.extend(native_weights(15360, 3840, 4));
        payloads.push(weights);
    }
    let data = Data::new(setup, payloads, 15360 * 2)?;
    let request = GateRequest {
        policy: owner(setup, false, 1, true),
        capture_gate_up: false,
        activation: data.input(0),
        native_weights: data.input(1),
        low_bit: bits.map(|b| {
            [
                data.weight(1, b, Role::DenseGateProjection, 15360, 3840),
                data.weight(3, b, Role::DenseUpProjection, 15360, 3840),
            ]
        }),
        output: data.out(),
        arena_bytes: data.arena.capacity(),
    };
    assert!(GateRequest {
        capture_gate_up: true,
        ..request
    }
    .plan()
    .is_none());
    let expected: Vec<_> = [0, 1, 3, 7, 31, 7680, 15359]
        .into_iter()
        .map(|r| {
            let g = f(b(dot(bits, r, 0, 3840, 1) as f32));
            let u = f(b(dot(bits, r, 0, 3840, 4) as f32));
            (r, f64::from(f(b(gelu(g) * u))))
        })
        .collect();
    execute(
        setup,
        &data,
        request.plan().ok_or("gate refused")?,
        label,
        1,
        15360,
        15360,
        0,
        false,
        &expected,
        false,
    )
}
fn qkv(setup: &Setup, bits: u8, global: bool, label: &str) -> TestResult<Value> {
    let qn = if global { 8192 } else { 4096 };
    let kn = if global { 512 } else { 2048 };
    let mut payloads = vec![activation(1, 3840)];
    payloads.extend(quant(bits, qn, 3840, 1));
    payloads.extend(quant(bits, kn, 3840, 4));
    if !global {
        payloads.extend(quant(bits, kn, 3840, 7));
    }
    let data = Data::new(setup, payloads, (qn + 2 * kn) * 2)?;
    let key = data.weight(3, bits, Role::KeyProjection, kn as u32, 3840);
    let value = if global {
        Weight {
            role: Role::ValueProjection,
            ..key
        }
    } else {
        data.weight(5, bits, Role::ValueProjection, kn as u32, 3840)
    };
    let request = QkvRequest {
        policy: owner(setup, global, 1, true),
        capture_projection: false,
        skip_kv: false,
        activation: data.input(0),
        weights: [
            data.weight(1, bits, Role::QueryProjection, qn as u32, 3840),
            key,
            value,
        ],
        output: data.out(),
        arena_bytes: data.arena.capacity(),
    };
    assert!(QkvRequest {
        skip_kv: true,
        ..request
    }
    .plan()
    .is_none());
    let mut expected = Vec::new();
    for (offset, rows, salt) in [
        (0, qn, 1),
        (qn, kn, 4),
        (qn + kn, kn, if global { 4 } else { 7 }),
    ] {
        for r in [0, 1, 3, rows / 2, rows - 1] {
            expected.push((offset + r, dot(Some(bits), r, 0, 3840, salt)));
        }
    }
    let result = execute(
        setup,
        &data,
        request.plan().ok_or("QKV refused")?,
        label,
        1,
        qn + 2 * kn,
        qn + 2 * kn,
        0,
        false,
        &expected,
        false,
    )?;
    if global {
        assert_eq!(
            &data.output()[qn * 2..(qn + kn) * 2],
            &data.output()[(qn + kn) * 2..]
        );
    }
    Ok(result)
}
fn attention(setup: &Setup, global: bool, mode: u8, label: &str) -> TestResult<Value> {
    let d = if global { 512 } else { 256 };
    let kv = if global { 1 } else { 8 };
    let bs = 16;
    let blocks = 70;
    let len = 1031_i32;
    let pos: i32 = if mode == 1 { 8 } else { 1030 };
    let mut table: Vec<i32> = (0..blocks)
        .map(|i| ((i * 13 + 5) % blocks) as i32)
        .collect();
    table[3] = -1;
    if mode == 2 {
        table.fill(-1);
    }
    if mode == 3 {
        table[0] = blocks as i32;
    }
    let q: Vec<u16> = (0..16 * d)
        .map(|i| b(((i / d * 3 + i % d * 7) % 17) as f32 / 128.0 - 8.0 / 128.0))
        .collect();
    let k: Vec<u16> = (0..blocks * bs * kv * d)
        .map(|i| b((i % 23) as f32 / 64.0 - 11.0 / 64.0))
        .collect();
    let v: Vec<u16> = (0..k.len())
        .map(|i| b(((i * 7) % 29) as f32 / 32.0 - 14.0 / 32.0))
        .collect();
    let payloads = vec![
        words(q.iter().copied()),
        words(k.iter().copied()),
        words(v.iter().copied()),
        table.iter().flat_map(|x| x.to_le_bytes()).collect(),
        len.to_le_bytes().to_vec(),
        pos.to_le_bytes().to_vec(),
    ];
    let data = Data::new(setup, payloads, 16 * d * 2)?;
    let request = AttentionRequest {
        policy: owner(setup, global, 1, true),
        offsets: [
            data.input(0),
            data.input(1),
            data.input(2),
            data.out(),
            data.input(3),
            data.input(4),
            data.input(5),
        ],
        block_size: bs as u32,
        max_blocks: blocks as u32,
        num_blocks: blocks as u32,
        scale: 1.0,
        arena_bytes: data.arena.capacity(),
    };
    assert!(AttentionRequest {
        scale: 0.0625,
        ..request
    }
    .plan()
    .is_none());
    let mut expected = Vec::new();
    if mode != 3 {
        let begin = if global {
            0
        } else {
            (pos as usize + 1).saturating_sub(1024)
        };
        for head in 0..16 {
            let mut value = vec![0_f64; d];
            let mut denom = 0.0;
            for t in begin..=pos as usize {
                let page = table[t / bs];
                if page < 0 {
                    continue;
                }
                let base = ((page as usize * bs + t % bs) * kv + head / (16 / kv)) * d;
                let score: f64 = (0..d)
                    .map(|j| f64::from(f(q[head * d + j])) * f64::from(f(k[base + j])))
                    .sum();
                let p = score.exp();
                denom += p;
                for j in 0..d {
                    value[j] += p * f64::from(f(v[base + j]));
                }
            }
            for (j, value) in value.into_iter().enumerate() {
                expected.push((head * d + j, if denom > 0.0 { value / denom } else { 0.0 }));
            }
        }
    }
    execute(
        setup,
        &data,
        request.plan().ok_or("attention refused")?,
        label,
        1,
        16 * d,
        16 * d,
        0,
        false,
        &expected,
        mode == 3,
    )
}
#[test]
#[ignore = "requires Apple9/M4, a fresh report directory and identity-bound strict-math metallib"]
fn native_donor12b_operator_oracle() -> TestResult {
    let setup = Setup::new(true)?;
    if simdgroups(setup.candidate).is_none() {
        return Err("donor family required".into());
    }
    let mut cases = Vec::new();
    cases.push(projection(
        &setup,
        Some(4),
        1,
        3840,
        15360,
        Role::DenseDownProjection,
        "w4-down-m1",
    )?);
    cases.push(projection(
        &setup,
        Some(8),
        1,
        3840,
        8192,
        Role::OutputProjection,
        "w8-o-global-m1",
    )?);
    cases.push(projection(
        &setup,
        Some(4),
        9,
        15360,
        3840,
        Role::DenseGateProjection,
        "w4-gate-m9",
    )?);
    cases.push(projection(
        &setup,
        Some(8),
        9,
        3840,
        4096,
        Role::OutputProjection,
        "w8-o-local-m9",
    )?);
    for (bits, label) in [
        (Some(4), "fused-gate-w4"),
        (Some(8), "fused-gate-w8"),
        (None, "fused-gate-native"),
    ] {
        cases.push(gate(&setup, bits, label)?);
    }
    for (bits, global, label) in [
        (4, false, "qkv-w4-local"),
        (4, true, "qkv-w4-global"),
        (8, false, "qkv-w8-local"),
        (8, true, "qkv-w8-global"),
    ] {
        cases.push(qkv(&setup, bits, global, label)?);
    }
    cases.push(projection(
        &setup,
        None,
        9,
        9216,
        3840,
        Role::QueryProjection,
        "native-qkv-m9-f32",
    )?);
    cases.push(projection(
        &setup,
        None,
        1,
        3840,
        4096,
        Role::OutputProjection,
        "native-o-m1-f32",
    )?);
    for global in [false, true] {
        for mode in 0..4 {
            let label = format!(
                "attention-{}-mode{mode}",
                if global { "global" } else { "local" }
            );
            cases.push(attention(&setup, global, mode, &label)?);
        }
    }
    let snapshot = setup.pipelines.research_dispatch_snapshot();
    for kernel in setup.candidate.kernels() {
        assert!(
            snapshot.counts[*kernel as usize] > 0,
            "unexecuted family arm"
        );
    }
    let result = json!({"status":"passed","scope":"dense-synthetic-operator-only","identity":setup.identity,
        "cases":cases,"model_oracle":false,"performance_claim":false,"production_promotion":false});
    write_new(
        &setup.directory.join("donor12b-oracle.json"),
        &serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}
fn relative(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.max(b)
}
fn projection_abba(setup: &Setup, bits: u8, m: usize, k: usize, label: &str) -> TestResult<Value> {
    use crate::low_bit_metal::MetalLowBitProjectionOffsets;
    let n = 3840;
    let stride = n + 8;
    let mut payloads = vec![activation(m, k)];
    payloads.extend(quant(bits, n, k, 1));
    let data = Data::new(setup, payloads, m * stride * 2)?;
    let role = if k == 15360 {
        Role::DenseDownProjection
    } else {
        Role::OutputProjection
    };
    let w = data.weight(1, bits, role, n as u32, k as u32);
    let request = ProjectionRequest {
        selected: setup.candidate,
        dtype: Some(MetalFloatType::Bf16),
        quantized_accumulation: false,
        shape: [m as u32, n as u32, k as u32],
        activation: data.input(0),
        native_weights: 0,
        low_bit: Some(w),
        output: data.out(),
        output_stride: stride as u32,
        output_column: 4,
        output_f32: false,
        arena_bytes: data.arena.capacity(),
    };
    let plan = request.plan().ok_or("timing projection refused")?;
    let descriptor = MetalLowBitProjectionOffsets::new_for_role(
        role,
        w.format,
        n,
        k,
        w.values,
        data.inputs[1].1.len() - 2 * GUARD,
        w.scales,
        data.inputs[2].1.len() - 2 * GUARD,
    )?;
    let expected: Vec<_> = (0..m)
        .flat_map(|t| {
            [0, 1, 3, 7, n / 2, n - 1]
                .into_iter()
                .map(move |row| (t * stride + 4 + row, dot(Some(bits), row, t, k, 1)))
        })
        .collect();
    // Qualify arithmetic before any timer; baseline uses identical bytes,
    // scales, dtype, output stride, compiler flags, queue and prepared library.
    let oracle = execute(
        setup,
        &data,
        plan,
        &format!("{label}-oracle"),
        m,
        n,
        stride,
        4,
        false,
        &expected,
        false,
    )?;
    let run = |candidate: bool, repeats: usize| -> TestResult<f64> {
        data.reset()?;
        let command = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("no command buffer")?;
        for _ in 0..repeats {
            if candidate {
                assert!(emit_plan_for_oracle(
                    &setup.pipelines,
                    &command,
                    data.arena.buffer(),
                    plan
                )?);
            } else {
                descriptor.encode_strided_bf16_n4(
                    &command,
                    &setup.pipelines,
                    data.arena.buffer(),
                    data.input(0),
                    data.out(),
                    m,
                    stride,
                    4,
                )?;
            }
        }
        complete(&command)?;
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end <= start {
            return Err("missing GPU timestamps".into());
        }
        data.check();
        Ok((end - start) * 1e9 / repeats as f64)
    };
    run(false, 1)?;
    for &(index, reference) in &expected {
        let raw = &data.output()[index * 2..index * 2 + 2];
        let actual = f(u16::from_le_bytes(raw.try_into()?));
        assert!(
            (f64::from(actual) - reference).abs() <= 0.0001 + 0.012 * reference.abs(),
            "incumbent oracle failed"
        );
    }
    for _ in 0..4 {
        run(false, 8)?;
        run(true, 8)?;
    }
    let mut samples = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut a_first = Vec::new();
    let mut a_second = Vec::new();
    let mut b_first = Vec::new();
    let mut b_second = Vec::new();
    for block in 0..20 {
        let order = if block % 2 == 0 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for (position, candidate) in order.into_iter().enumerate() {
            let ns = run(candidate, 8)?;
            if candidate {
                a.push(ns);
                if position == 0 || position == 3 {
                    a_first.push(ns);
                } else {
                    a_second.push(ns);
                }
            } else {
                b.push(ns);
                if position == 0 || position == 3 {
                    b_first.push(ns);
                } else {
                    b_second.push(ns);
                }
            }
            samples.push(json!({"block":block,"position":position,"candidate":candidate,"ns_per_operator":ns}));
        }
    }
    let candidate_ns = median(&a);
    let incumbent_ns = median(&b);
    let drift_a = relative(median(&a[..20]), median(&a[20..]));
    let drift_b = relative(median(&b[..20]), median(&b[20..]));
    let order_a = relative(median(&a_first), median(&a_second));
    let order_b = relative(median(&b_first), median(&b_second));
    let stable = [drift_a, drift_b, order_a, order_b]
        .into_iter()
        .all(|v| v <= 0.05);
    Ok(
        json!({"case":label,"scope":"same-weight-hot-cache-operator-only","oracle":oracle,
        "incumbent":descriptor.experimental_bf16_n4_kernel_name(),"candidate":plan.kernel.name(),
        "candidate_median_ns":candidate_ns,"incumbent_median_ns":incumbent_ns,
        "drift_candidate":drift_a,"drift_incumbent":drift_b,
        "order_sensitivity_candidate":order_a,"order_sensitivity_incumbent":order_b,
        "five_percent_stability_gate":stable,
        "admitted_operator_speedup":if stable{Some(incumbent_ns/candidate_ns)}else{None},
        "samples":samples,"automatic_promotion":false}),
    )
}
#[test]
#[ignore = "requires sealed strict-math metallib; serial ABBA/BAAB synthetic operator trial, not full-model speed"]
fn native_donor12b_projection_abba() -> TestResult {
    let setup = Setup::new(false)?;
    if simdgroups(setup.candidate).is_none() {
        return Err("donor family required".into());
    }
    let mut trials = Vec::new();
    for (bits, m, k, label) in [
        (4, 1, 15360, "w4-down"),
        (8, 1, 8192, "w8-global-o"),
        (8, 1, 4096, "w8-local-o"),
        (8, 9, 4096, "w8-local-o-prefill9"),
    ] {
        trials.push(projection_abba(&setup, bits, m, k, label)?);
    }
    let receipt = json!({"identity":setup.identity,"trials":trials,
        "full_model_speedup":false,"production_promotion":false});
    write_new(
        &setup.directory.join("donor12b-projection-abba.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}
