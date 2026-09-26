//! Ignored native gates using the existing sealed-library/queue referee owner.
//! Synthetic full operator shapes qualify arithmetic only, never checkpoint
//! quality, production routing, or performance. No GPU gate ran in Chat.
use crate::arena::MetalRegion;
use crate::attention_global_decode_device_tests::{
    complete, control_snapshot, env_path, sha256, write_new, Setup, TestResult,
};
use crate::low_bit_metal::MetalLowBitProjectionOffsets;
use crate::research::Gemma12bResearchShape;
use crate::research_decode::{reference as cpu, GateUpRequest};
use crate::research_decode_metal::try_encode_gate_up;
use crate::{MetalBufferArena, MetalFloatType, MetalResearchCandidate};
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use rvllm_apple::{AppleLowBitTensorRole as Role, AppleLowBitWeightFormat as Format};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};
const GUARD: usize = 64;

fn words(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
struct Data {
    arena: MetalBufferArena,
    inputs: Vec<(MetalRegion, Vec<u8>)>,
    outputs: Vec<(MetalRegion, Vec<u8>)>,
    projection: Option<MetalLowBitProjectionOffsets>,
    expected: Vec<u16>,
    column: usize,
    label: String,
}
impl Data {
    fn new(setup: &Setup, k: usize) -> TestResult<Self> {
        let ffn = setup.candidate == MetalResearchCandidate::FfnBf16R4Sg2;
        let (payloads, expected, label) = if ffn {
            let mut w = vec![0; 30720 * 3840 * 2];
            for row in 0..30720 {
                for term in 0..8 {
                    let (col, bits) = cpu::sparse_term(row, term);
                    let i = (row * 3840 + col) * 2;
                    w[i..i + 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
            (
                vec![words(&cpu::sparse_activation()), w],
                cpu::sparse_gate_up(),
                "ffn-M1-K3840-I15360".to_owned(),
            )
        } else {
            let bits = if matches!(
                setup.candidate,
                MetalResearchCandidate::QmvW4G32R8Sg2 | MetalResearchCandidate::QmvW4G32R4Sg8K8
            ) {
                4
            } else {
                8
            };
            let fixture = cpu::Group32Fixture::new(bits, 3840, k);
            let expected = fixture.output();
            (
                vec![words(&fixture.x), fixture.values, words(&fixture.scales)],
                expected,
                format!("w{bits}-M1-N3840-K{k}"),
            )
        };
        let mut arena = MetalBufferArena::new(
            setup.context.device(),
            payloads.iter().map(Vec::len).sum::<usize>() + 256 * 1024,
        )?;
        let mut upload = |name: &str, payload: Vec<u8>| -> TestResult<(MetalRegion, Vec<u8>)> {
            let mut raw = vec![0xa5; GUARD];
            raw.extend(payload);
            raw.extend([0xa5; GUARD]);
            let region = arena.region(name, raw.len(), 64)?;
            // SAFETY: fresh shared allocation, no outstanding GPU commands.
            unsafe {
                arena.write_region(&region, &raw)?;
            }
            Ok((region, raw))
        };
        let mut inputs = Vec::new();
        for (i, payload) in payloads.into_iter().enumerate() {
            inputs.push(upload(&format!("decode-round-input-{i}"), payload)?);
        }
        let column = if ffn { 0 } else { 2 };
        let outputs = vec![
            upload("activated", vec![0xff; (expected.len() + 2 * column) * 2])?,
            upload(
                "incumbent-gate-up",
                vec![0xff; if ffn { 30720 * 2 } else { 64 }],
            )?,
        ];
        drop(upload);
        let projection = if ffn {
            None
        } else {
            let w4 = matches!(
                setup.candidate,
                MetalResearchCandidate::QmvW4G32R8Sg2 | MetalResearchCandidate::QmvW4G32R4Sg8K8
            );
            Some(MetalLowBitProjectionOffsets::new_for_role(
                if w4 {
                    Role::DenseDownProjection
                } else {
                    Role::OutputProjection
                },
                if w4 { Format::W4A16 } else { Format::W8A16 },
                3840,
                k,
                inputs[1].0.offset + GUARD,
                inputs[1].0.size - 2 * GUARD,
                inputs[2].0.offset + GUARD,
                inputs[2].0.size - 2 * GUARD,
            )?)
        };
        Ok(Self {
            arena,
            inputs,
            outputs,
            projection,
            expected,
            column,
            label,
        })
    }
    fn request(&self, setup: &Setup) -> GateUpRequest {
        GateUpRequest {
            selected: setup.candidate,
            dtype: Some(MetalFloatType::Bf16),
            decode: true,
            quantized_accumulation: false,
            capture_gate_up: false,
            has_low_bit_gate_or_up: false,
            model: Gemma12bResearchShape {
                tokens: 1,
                hidden: 3840,
                intermediate: 15360,
                layers: 48,
                heads: 16,
                kv_heads: 1,
                head_dim: 512,
                attention_window: 0,
                moe_experts: 0,
                moe_top_k: 0,
                moe_intermediate: 0,
                ple: 0,
            },
            offsets: [
                self.inputs[0].0.offset + GUARD,
                self.inputs[1].0.offset + GUARD,
                self.outputs[0].0.offset + GUARD,
            ],
            arena_bytes: self.arena.capacity(),
        }
    }
    fn reset(&self) -> TestResult {
        for (region, raw) in &self.outputs {
            // SAFETY: only called outside outstanding command lifetimes.
            unsafe {
                self.arena.write_region(region, raw)?;
            }
        }
        Ok(())
    }
    fn payload(&self) -> Vec<u8> {
        let region = &self.outputs[0].0;
        // SAFETY: callers have completed the command or have not submitted it.
        unsafe {
            std::slice::from_raw_parts(
                self.arena.host_ptr(region).add(GUARD),
                region.size - 2 * GUARD,
            )
            .to_vec()
        }
    }
    fn check(&self, untouched: bool) {
        for (region, raw) in &self.inputs {
            // SAFETY: invoked only after completion or before submission.
            let actual =
                unsafe { std::slice::from_raw_parts(self.arena.host_ptr(region), region.size) };
            assert_eq!(actual, raw.as_slice(), "immutable inputs and guards");
        }
        for (index, (region, raw)) in self.outputs.iter().enumerate() {
            let actual =
                unsafe { std::slice::from_raw_parts(self.arena.host_ptr(region), region.size) };
            assert_eq!(&actual[..GUARD], &raw[..GUARD]);
            assert_eq!(&actual[actual.len() - GUARD..], &raw[raw.len() - GUARD..]);
            if untouched {
                assert_eq!(actual, raw.as_slice(), "refused output changed");
            }
            if index == 0 && self.column > 0 {
                assert_eq!(
                    &actual[GUARD..GUARD + self.column * 2],
                    &raw[GUARD..GUARD + self.column * 2]
                );
                let tail = actual.len() - GUARD - self.column * 2;
                assert_eq!(
                    &actual[tail..actual.len() - GUARD],
                    &raw[tail..raw.len() - GUARD]
                );
            }
        }
    }
    fn encode(
        &self,
        setup: &Setup,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        candidate: bool,
    ) -> TestResult {
        if let Some(p) = self.projection {
            let input = self.inputs[0].0.offset + GUARD;
            let output = self.outputs[0].0.offset + GUARD;
            let stride = self.expected.len() + 2 * self.column;
            if candidate {
                if !p.try_encode_strided_bf16_decode_candidate(
                    command,
                    &setup.pipelines,
                    self.arena.buffer(),
                    input,
                    output,
                    1,
                    stride,
                    self.column,
                )? {
                    return Err("candidate fallback invalidates gate".into());
                }
            } else {
                p.encode_strided_bf16_n4(
                    command,
                    &setup.pipelines,
                    self.arena.buffer(),
                    input,
                    output,
                    1,
                    stride,
                    self.column,
                )?;
            }
        } else if candidate {
            if !try_encode_gate_up(
                &setup.pipelines,
                command,
                self.arena.buffer(),
                self.request(setup),
            )? {
                return Err("candidate fallback invalidates gate".into());
            }
        } else {
            // SAFETY: all exact incumbent spans are persistent, aligned and
            // nonaliasing. Use the actual layer-forward encoders, not a new ABI.
            let [x, w, y] = self.request(setup).offsets;
            let temp = self.outputs[1].0.offset + GUARD;
            unsafe {
                crate::layer_forward::encode_gemm(
                    command,
                    &setup.pipelines,
                    self.arena.buffer(),
                    x,
                    w,
                    temp,
                    1,
                    30720,
                    3840,
                    1.0,
                    0.0,
                )?;
                crate::layer_forward::encode_gelu_mul(
                    command,
                    &setup.pipelines,
                    self.arena.buffer(),
                    temp,
                    y,
                    1,
                    15360,
                    "decode-round-incumbent",
                )?;
            }
        }
        Ok(())
    }
    fn check_fp64(&self) -> f64 {
        let raw = self.payload();
        let mut maximum = 0.0_f64;
        for (i, &expected) in self.expected.iter().enumerate() {
            let start = 2 * (self.column + i);
            let actual = cpu::widen(u16::from_le_bytes(
                raw[start..start + 2].try_into().unwrap(),
            ));
            let expected = cpu::widen(expected);
            let error = (actual - expected).abs();
            // Two relative BF16 quanta, with a fixed tiny near-zero floor.
            // Set before device results; independent dot/activation uses FP64.
            let bound = 0.015625 * expected.abs() + 1.0e-5;
            assert!(
                actual.is_finite() && error <= bound,
                "{} output[{i}]: {actual} vs {expected}, bound {bound}",
                self.label
            );
            maximum = maximum.max(error);
        }
        maximum
    }
}
/// Deliberately malformed launch metadata. Storage is still fully allocated;
/// the kernel must return uniformly before writes. Unlike host refusal this
/// IS an encoded dispatch and must be counted as one, never as successful work.
fn guarded_bad_launch(data: &Data, setup: &Setup, k: usize, case: usize) -> TestResult {
    data.reset()?;
    let before = setup.pipelines.research_dispatch_snapshot();
    let command = setup.context.queue().commandBuffer().ok_or("no command")?;
    let encoder = command.computeCommandEncoder().ok_or("no encoder")?;
    let kernel = setup.candidate.kernels()[0];
    let ffn = data.projection.is_none();
    let grid = if ffn {
        1920
    } else {
        3840 / kernel.qmv_output_rows().ok_or("QMV output tile missing")?
    };
    encoder.setComputePipelineState(setup.pipelines.get(kernel.name())?);
    let m = if case == 1 { 2 } else { 1 };
    let wrong_k = if case == 2 { k as u32 - 1 } else { k as u32 };
    let params = if ffn {
        [m, wrong_k, 15360, 0, 0]
    } else {
        [m, 3840, wrong_k, 3844, 2]
    };
    // SAFETY: Data owns all full spans and no work is outstanding. Only scalar
    // shape/thread constants are malformed; all are checked before loads/stores.
    unsafe {
        let offsets = if ffn {
            [
                data.inputs[0].0.offset + GUARD,
                data.inputs[1].0.offset + GUARD,
                data.outputs[0].0.offset + GUARD,
                0,
            ]
        } else {
            [
                data.inputs[0].0.offset + GUARD,
                data.inputs[1].0.offset + GUARD,
                data.inputs[2].0.offset + GUARD,
                data.outputs[0].0.offset + GUARD,
            ]
        };
        let buffers = if ffn { 3 } else { 4 };
        for (i, &offset) in offsets[..buffers].iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(data.arena.buffer()), offset, i);
        }
        for (i, param) in params[..if ffn { 3 } else { 5 }].iter().enumerate() {
            encoder.setBytes_length_atIndex(std::ptr::NonNull::from(param).cast(), 4, buffers + i);
        }
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: if case == 0 { 32 } else { kernel.limits().0 },
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    setup.pipelines.record_research_dispatch(kernel);
    complete(&command)?;
    data.check(true);
    setup
        .pipelines
        .research_dispatch_snapshot()
        .checked_since(before)?
        .verify_exact(kernel, 1)?;
    Ok(())
}

fn cases(candidate: MetalResearchCandidate) -> &'static [usize] {
    match candidate {
        MetalResearchCandidate::FfnBf16R4Sg2 => &[3840],
        MetalResearchCandidate::QmvW4G32R8Sg2 => &[15360],
        MetalResearchCandidate::QmvW4G32R4Sg8K8 => &[15360],
        MetalResearchCandidate::QmvW8G32R8Sg2 => &[4096, 8192],
        MetalResearchCandidate::QmvW8G32R4Sg8K8 => &[4096, 8192],
        _ => &[],
    }
}

#[test]
#[ignore = "native source-bound full-shape operator oracle; explicit pinned library and fresh report required"]
fn decode_round_oracle() -> TestResult {
    let setup = Setup::new(true)?;
    if !setup.candidate.decode_round_operator() {
        return Err("decode operator selector required".into());
    }
    let mut reports = Vec::new();
    for &k in cases(setup.candidate) {
        let data = Data::new(&setup, k)?;
        let identity = (
            data.arena.allocated(),
            data.arena.regions().len(),
            data.arena.buffer().contents(),
        );
        let mut first = None;
        let mut max_error = 0.0_f64;
        for _ in 0..3 {
            data.reset()?;
            let before = setup.pipelines.research_dispatch_snapshot();
            let low_before = setup.pipelines.low_bit_dispatch_snapshot();
            let command = setup.context.queue().commandBuffer().ok_or("no command")?;
            data.encode(&setup, &command, true)?;
            complete(&command)?;
            setup
                .pipelines
                .research_dispatch_snapshot()
                .checked_since(before)?
                .verify_exact(setup.candidate.kernels()[0], 1)?;
            if let Some(p) = data.projection {
                let delta = setup
                    .pipelines
                    .low_bit_dispatch_snapshot()
                    .checked_since(low_before)?;
                let mut expected = [0; Role::COUNT];
                expected[p.role().index()] = 1;
                delta.verify_exact(p.format(), expected)?;
            }
            data.check(false);
            max_error = max_error.max(data.check_fp64());
            let actual = data.payload();
            if let Some(previous) = &first {
                assert_eq!(previous, &actual, "repeated use");
            }
            first = Some(actual);
            assert_eq!(
                identity,
                (
                    data.arena.allocated(),
                    data.arena.regions().len(),
                    data.arena.buffer().contents()
                )
            );
        }
        // Incumbent must also satisfy the independent predeclared oracle bound.
        let candidate_output = first.unwrap();
        data.reset()?;
        let before = setup.pipelines.research_dispatch_snapshot();
        let command = setup.context.queue().commandBuffer().ok_or("no command")?;
        data.encode(&setup, &command, false)?;
        complete(&command)?;
        data.check(false);
        let baseline_error = data.check_fp64();
        setup
            .pipelines
            .research_dispatch_snapshot()
            .checked_since(before)?
            .verify_exact(setup.candidate.kernels()[0], 0)?;
        // Actual host adapter rejection: no output mutation or encoded dispatch.
        for negative in 0..4 {
            data.reset()?;
            let before = setup.pipelines.research_dispatch_snapshot();
            let command = setup.context.queue().commandBuffer().ok_or("no command")?;
            if let Some(p) = data.projection {
                let input = data.inputs[0].0.offset + GUARD;
                let output = data.outputs[0].0.offset + GUARD;
                let result = p.try_encode_strided_bf16_decode_candidate(
                    &command,
                    &setup.pipelines,
                    data.arena.buffer(),
                    if negative == 2 { input + 1 } else { input },
                    if negative == 3 { input } else { output },
                    if negative == 0 { 2 } else { 1 },
                    3844,
                    if negative == 1 { 5 } else { 2 },
                );
                assert!(matches!(result, Ok(false) | Err(_)));
            } else {
                let mut request = data.request(&setup);
                match negative {
                    0 => request.model.tokens = 2,
                    1 => request.capture_gate_up = true,
                    2 => request.offsets[0] += 1,
                    _ => request.offsets[2] = request.offsets[0],
                }
                assert!(!try_encode_gate_up(
                    &setup.pipelines,
                    &command,
                    data.arena.buffer(),
                    request
                )?);
            }
            complete(&command)?;
            data.check(true);
            setup
                .pipelines
                .research_dispatch_snapshot()
                .checked_since(before)?
                .verify_exact(setup.candidate.kernels()[0], 0)?;
        }
        for negative in 0..3 {
            guarded_bad_launch(&data, &setup, k, negative)?;
        }
        let path = setup.directory.join(format!("{}.bf16", data.label));
        write_new(&path, &candidate_output)?;
        reports.push(
            json!({"label":data.label,"k":k,"bf16_file":path,"bf16_sha256":sha256(&path)?,
            "max_fp64_abs_error":max_error,"incumbent_max_fp64_abs_error":baseline_error,
            "reference":"independent scalar FP64 with incumbent BF16 boundaries",
            "predeclared_bound":"0.015625*abs(reference)+1e-5",
            "repeats":3,"repeated_bit_exact":true,"guards_untouched":true,
            "host_refusals_without_dispatch":4,"shader_guarded_encoded_dispatches":3,
            "exact_dispatch_accounting":true,
            "persistent_arena_unchanged":true}),
        );
    }
    write_new(
        &setup.directory.join("oracle.json"),
        &serde_json::to_vec_pretty(&json!({
        "schema":"rvllm.decode-round.oracle.v1","status":"passed","identity":setup.identity,
        "cases":reports,"checkpoint_qualified":false,"promotion":false}))?,
    )
}

#[test]
#[ignore = "operator timing only after exact matching native oracle; balanced ABBA/BAAB, no promotion"]
fn decode_round_abba() -> TestResult {
    let setup = Setup::new(false)?;
    if !setup.candidate.decode_round_operator() {
        return Err("decode operator selector required".into());
    }
    let oracle_path = env_path("ORACLE_RECEIPT")?;
    let oracle: Value = serde_json::from_slice(&std::fs::read(&oracle_path)?)?;
    if oracle["schema"] != "rvllm.decode-round.oracle.v1"
        || oracle["status"] != "passed"
        || oracle["identity"]["candidate"] != setup.identity["candidate"]
        || oracle["identity"]["core_sha256"] != setup.identity["core_sha256"]
        || oracle["identity"]["test_executable_sha256"] != setup.identity["test_executable_sha256"]
    {
        return Err("matching native oracle required".into());
    }
    // LENGTH=0 identifies a projection cell rather than inventing a KV length.
    if std::env::var("RVLLM_METAL_GLOBAL_DECODE_LENGTH")? != "0" {
        return Err("projection timing uses explicit length sentinel 0".into());
    }
    let k: usize = std::env::var("RVLLM_METAL_GLOBAL_DECODE_OPERATOR_K")?.parse()?;
    if !cases(setup.candidate).contains(&k) {
        return Err("unqualified operator K".into());
    }
    let data = Data::new(&setup, k)?;
    let case = oracle["cases"]
        .as_array()
        .ok_or("no cases")?
        .iter()
        .find(|case| case["k"].as_u64() == Some(k as u64))
        .ok_or("K not qualified")?;
    let path = PathBuf::from(case["bf16_file"].as_str().ok_or("no oracle output")?);
    if case["bf16_sha256"] != sha256(&path)? {
        return Err("oracle output drift".into());
    }
    let expected = std::fs::read(&path)?;
    let arena_identity = (
        data.arena.allocated(),
        data.arena.regions().len(),
        data.arena.buffer().contents(),
    );
    let run = |arm: char, repeats: u32| -> TestResult<(f64, f64)> {
        let before = setup.pipelines.research_dispatch_snapshot();
        let low_before = setup.pipelines.low_bit_dispatch_snapshot();
        let start = Instant::now();
        let command = setup.context.queue().commandBuffer().ok_or("no command")?;
        for _ in 0..repeats {
            data.encode(&setup, &command, arm == 'B')?;
        }
        complete(&command)?;
        let wall = start.elapsed().as_secs_f64();
        let gpu = command.GPUEndTime() - command.GPUStartTime();
        if !gpu.is_finite() || gpu <= 0.0 {
            return Err("GPU timestamps unavailable".into());
        }
        setup
            .pipelines
            .research_dispatch_snapshot()
            .checked_since(before)?
            .verify_exact(
                setup.candidate.kernels()[0],
                if arm == 'B' { u64::from(repeats) } else { 0 },
            )?;
        if let Some(p) = data.projection {
            let delta = setup
                .pipelines
                .low_bit_dispatch_snapshot()
                .checked_since(low_before)?;
            let mut expected = [0; Role::COUNT];
            expected[p.role().index()] = u64::from(repeats);
            delta.verify_exact(p.format(), expected)?;
        }
        assert_eq!(
            arena_identity,
            (
                data.arena.allocated(),
                data.arena.regions().len(),
                data.arena.buffer().contents()
            )
        );
        data.check(false);
        if arm == 'B' {
            assert_eq!(data.payload(), expected);
        } else {
            data.check_fp64();
        }
        Ok((gpu, wall))
    };
    for _ in 0..5 {
        run('A', 1)?;
        run('B', 1)?;
    }
    let mut samples = Vec::new();
    for block in 0..10 {
        let order = if block % 2 == 0 {
            ['A', 'B', 'B', 'A']
        } else {
            ['B', 'A', 'A', 'B']
        };
        for (position, arm) in order.into_iter().enumerate() {
            let before = control_snapshot()?;
            let (gpu, wall) = run(arm, 100)?;
            let after = control_snapshot()?;
            let encoders = if data.projection.is_none() {
                100 * crate::research_decode::gate_up_encoder_count(arm == 'B')
            } else {
                100
            };
            let sample = json!({"block":block,"position":position,"arm":arm.to_string(),
                "dispatches":100,"operator_iterations":100,"actual_compute_encoders":encoders,
                "gpu_seconds":gpu,"synchronized_wall_seconds":wall,
                "controls_before":before,"controls_after":after});
            write_new(
                &setup
                    .directory
                    .join(format!("sample-{block}-{position}.json")),
                &serde_json::to_vec_pretty(&sample)?,
            )?;
            samples.push(sample);
        }
    }
    let controls: Vec<f64> = samples
        .iter()
        .filter(|s| s["arm"] == "A")
        .map(|s| s["gpu_seconds"].as_f64().unwrap())
        .collect();
    let drift = controls.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        / controls.iter().copied().fold(f64::INFINITY, f64::min)
        - 1.0;
    let receipt = json!({"schema":"rvllm.global-decode.abba.v1","status":"collected",
        "identity":setup.identity,"oracle_receipt_sha256":sha256(&oracle_path)?,
        "candidate":setup.candidate.name(),"length":0,"operator_k":k,
        "baseline":if data.projection.is_some() {"rvLLM group-32 BF16 n4 (synthetic)"} else {"native BF16 Gate||Up then GELU"},
        "warmups_per_arm":5,"blocks":10,"balanced_abba_baab":true,
        "dispatches_per_sample":100,"normalization":"operator iteration; actual_compute_encoders recorded separately",
        "control_drift_fraction":drift,"control_drift_limit":0.05,"control_drift_passed":drift<=0.05,
        "source_compiles_during_samples":0,"samples":samples,"timing_eligible":false,
        "checkpoint_qualified":false,"promotion":false});
    write_new(
        &setup.directory.join("abba.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    if drift > 0.05 {
        return Err("control drift exceeded 5%; retain all samples as invalid".into());
    }
    Ok(())
}
