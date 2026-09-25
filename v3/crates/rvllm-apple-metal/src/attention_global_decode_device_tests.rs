//! Explicit, ignored Apple9 gates. Both use prebuilt, identity-bound libraries.
//! The oracle never runs in a timing interval; these are operator gates, not
//! checkpoint/full-route qualification or a kernel-game promotion.
use crate::arena::MetalRegion;
use crate::attention_global_decode::{
    physical_base,
    reference::{self, Fixture},
    round_bf16, DecodeBuffers, DecodeOutput, DecodePlan, DecodeShape, SplitDecodeBuffers,
    SplitDecodePlan, DIM, HEADS, LIVE_LENGTHS,
};
use crate::attention_global_decode_metal::{
    try_encode_global_decode, try_encode_split_global_decode, try_encode_split_global_decode_stage,
    SplitGlobalDecodeStage,
};
use crate::layer_forward::{MetalLayerDims, MetalPhase};
use crate::{
    MetalBufferArena, MetalContext, MetalFloatType, MetalKernelOptions, MetalResearchCandidate,
    PipelineCache,
};
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
const PREFIX: &str = "RVLLM_METAL_GLOBAL_DECODE_";
const GUARD: usize = 32;
const FLOAT_BYTES: usize = HEADS as usize * DIM as usize * 4;

fn write_new(path: &Path, bytes: &[u8]) -> TestResult {
    use std::io::Write;
    let mut file = std::fs::File::create_new(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn env_path(suffix: &str) -> TestResult<PathBuf> {
    let path = PathBuf::from(std::env::var(format!("{PREFIX}{suffix}"))?);
    if !path.is_absolute() {
        return Err("all gate paths must be absolute".into());
    }
    Ok(path)
}

fn sha256(path: &Path) -> TestResult<String> {
    let result = Command::new("/usr/bin/shasum")
        .args(["-a", "256", "--"])
        .arg(path)
        .output()?;
    if !result.status.success() {
        return Err("shasum failed".into());
    }
    let text = String::from_utf8(result.stdout)?;
    let digest = text.split_whitespace().next().ok_or("missing digest")?;
    if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid digest".into());
    }
    Ok(digest.to_ascii_lowercase())
}

struct Setup {
    context: MetalContext,
    pipelines: PipelineCache,
    candidate: MetalResearchCandidate,
    directory: PathBuf,
    identity: Value,
}

impl Setup {
    fn new(oracle: bool) -> TestResult<Self> {
        let candidate: MetalResearchCandidate =
            std::env::var(format!("{PREFIX}CANDIDATE"))?.parse()?;
        if candidate.global_decode_tile().is_none()
            && candidate.split_global_decode_tile().is_none()
        {
            return Err("explicit global decode candidate required".into());
        }
        let directory = env_path("REPORT_DIR")?;
        std::fs::create_dir(&directory)?; // fresh, never truncate an earlier receipt
        let source_path = env_path("SOURCE")?;
        let library_path = env_path("METALLIB")?;
        let build_path = env_path("BUILD_RECEIPT")?;
        let options = MetalKernelOptions {
            research: candidate,
            ..MetalKernelOptions::default()
        };
        let core = crate::kernels::kernel_source_with_options(MetalFloatType::Bf16, options);
        let mut expected = core.as_bytes().to_vec();
        if oracle {
            expected.push(b'\n');
            expected.extend_from_slice(include_bytes!(
                "research_shaders/global_decode_oracle.metal"
            ));
        }
        if std::fs::read(&source_path)? != expected {
            return Err("generated-source identity mismatch".into());
        }
        write_new(&directory.join("core.metal"), core.as_bytes())?;
        let core_sha = sha256(&directory.join("core.metal"))?;
        let source_sha = sha256(&source_path)?;
        let library_sha = sha256(&library_path)?;
        let build: Value = serde_json::from_slice(&std::fs::read(&build_path)?)?;
        if build["status"] != "compiled"
            || build["source_sha256"] != source_sha
            || build["metallib_sha256"] != library_sha
            || build["flags"] != json!(["-std=metal3.1", "-fno-fast-math"])
        {
            return Err("missing or mismatched strict-math compilation receipt".into());
        }
        let mut context = MetalContext::new()?;
        context.load_metallib(&library_path)?;
        let mut pipelines = PipelineCache::with_kernel_options(options);
        pipelines.compile_all_for_type(&context, MetalFloatType::Bf16)?;
        let mut kernels = Vec::new();
        for kernel in candidate.kernels() {
            let (threads, shared) = kernel.limits();
            let pso = pipelines
                .research_pso(kernel.name(), threads, shared)
                .ok_or("candidate refused by existing family/typed-PSO/resource gates")?;
            kernels.push(json!({"name":kernel.name(),"threads":threads,
                "source_threadgroup_bytes":shared,
                "actual_static_threadgroup_bytes":pso.staticThreadgroupMemoryLength(),
                "actual_execution_width":pso.threadExecutionWidth(),
                "actual_max_threads":pso.maxTotalThreadsPerThreadgroup()}));
        }
        let tile = candidate.global_decode_tile();
        let split = candidate.split_global_decode_tile();
        let (rows, panel, threads, grid, scratch_bytes) = if let Some(tile) = tile {
            (
                tile.rows,
                tile.panel,
                tile.threads,
                json!([HEADS / tile.rows, 1, 1]),
                0,
            )
        } else {
            let tile = split.unwrap();
            (
                tile.rows,
                tile.panel,
                tile.threads,
                json!([2, 16, 1]),
                16 * 16 * 514 * 4,
            )
        };
        let identity = json!({"candidate":candidate.name(), "core_sha256":core_sha,
            "source_sha256":source_sha, "metallib_sha256":library_sha,
            "build_receipt_sha256":sha256(&build_path)?,
            "test_executable_sha256":sha256(&std::env::current_exe()?)?, "kernels":kernels,
            "kernel":candidate.kernels()[0].name(),
            "rows":rows, "panel":panel, "threads":threads, "grid":grid,
            "scratch_bytes":scratch_bytes,
            "gpu_family":format!("{:?}",pipelines.gpu_family()),
            "device_name":context.device().name().to_string(), "oracle_library":oracle});
        write_new(
            &directory.join("identity.json"),
            &serde_json::to_vec_pretty(&identity)?,
        )?;
        Ok(Self {
            context,
            pipelines,
            candidate,
            directory,
            identity,
        })
    }
}

fn dims(s: DecodeShape) -> MetalLayerDims {
    MetalLayerDims {
        layer_idx: 5,
        attention_window: s.window,
        num_tokens: s.sequences,
        hidden: 3840,
        num_layers: 48,
        num_heads: s.heads,
        num_kv_heads: s.kv_heads,
        head_dim: s.head_dim,
        intermediate: 15360,
        moe_num_experts: 0,
        moe_top_k: 0,
        moe_intermediate: 0,
        ple_dim: 0,
        block_size: s.block_size,
        max_blocks_per_seq: s.max_blocks,
        num_blocks_total: s.num_blocks,
        attn_scale: s.scale,
        rms_eps: 1e-6,
        rope_dim: 128,
        softcap: 0.0,
    }
}

struct Guarded {
    arena: MetalBufferArena,
    inputs: Vec<(MetalRegion, Vec<u8>)>,
    outputs: Vec<(MetalRegion, Vec<u8>)>, // cooperative F32, serial F32, BF16, nine raw QK dots
}

impl Guarded {
    fn new(context: &MetalContext, f: &Fixture) -> TestResult<Self> {
        let u16_bytes = |values: &[u16]| {
            values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let payloads = [
            u16_bytes(&f.q),
            u16_bytes(&f.k),
            u16_bytes(&f.v),
            f.table.iter().flat_map(|x| x.to_le_bytes()).collect(),
            f.context.to_le_bytes().to_vec(),
            f.position.to_le_bytes().to_vec(),
        ];
        let capacity = payloads.iter().map(Vec::len).sum::<usize>()
            + FLOAT_BYTES * 3
            + 16 * 16 * 514 * 4
            + 8192;
        let mut arena = MetalBufferArena::new(context.device(), capacity)?;
        let mut upload = |name: &str, payload: Vec<u8>| -> TestResult<(MetalRegion, Vec<u8>)> {
            let mut raw = vec![0xa5; GUARD];
            raw.extend(payload);
            raw.extend([0xa5; GUARD]);
            let region = arena.region(name, raw.len(), 32)?;
            // SAFETY: fresh shared storage; no command buffer owns these bytes yet.
            unsafe {
                arena.write_region(&region, &raw)?;
            }
            Ok((region, raw))
        };
        let mut inputs = Vec::new();
        for (name, payload) in ["q", "k", "v", "table", "context", "position"]
            .into_iter()
            .zip(payloads)
        {
            inputs.push(upload(name, payload)?);
        }
        let mut outputs = Vec::new();
        for (name, bytes) in [
            ("fp32", FLOAT_BYTES),
            ("serial", FLOAT_BYTES),
            ("bf16", FLOAT_BYTES / 2),
            ("sampled-dots", 36),
            ("split-partials", 16 * 16 * 514 * 4),
        ] {
            outputs.push(upload(name, vec![0xff; bytes])?);
        }
        drop(upload);
        Ok(Self {
            arena,
            inputs,
            outputs,
        })
    }

    fn bindings(&self, output: usize) -> DecodeBuffers {
        let offset = |i: usize| self.inputs[i].0.offset + GUARD;
        DecodeBuffers {
            q: offset(0),
            k: offset(1),
            v: offset(2),
            output: self.outputs[output].0.offset + GUARD,
            block_tables: offset(3),
            context_lens: offset(4),
            positions: offset(5),
        }
    }

    fn split_bindings(&self, output: usize) -> SplitDecodeBuffers {
        SplitDecodeBuffers {
            common: self.bindings(output),
            partials: self.outputs[4].0.offset + GUARD,
        }
    }

    /// Only called before submission or after a successfully completed command.
    fn read(&self, region: &MetalRegion) -> Vec<u8> {
        // SAFETY: all callers meet the synchronized shared-storage requirement.
        unsafe { std::slice::from_raw_parts(self.arena.host_ptr(region), region.size).to_vec() }
    }

    fn payload(&self, index: usize) -> Vec<u8> {
        let raw = self.read(&self.outputs[index].0);
        raw[GUARD..raw.len() - GUARD].to_vec()
    }

    fn reset(&self) -> TestResult {
        for (region, raw) in &self.outputs {
            // SAFETY: called only with no outstanding GPU users.
            unsafe {
                self.arena.write_region(region, raw)?;
            }
        }
        Ok(())
    }

    fn check(&self, untouched: bool) {
        for (region, original) in &self.inputs {
            assert_eq!(&self.read(region), original, "input/cache mutation");
        }
        for (region, original) in &self.outputs {
            let bytes = self.read(region);
            assert_eq!(&bytes[..GUARD], &[0xa5; GUARD], "leading guard");
            assert_eq!(
                &bytes[bytes.len() - GUARD..],
                &[0xa5; GUARD],
                "trailing guard"
            );
            if untouched {
                assert_eq!(&bytes, original, "rejected output was touched");
            }
        }
    }
}

fn complete(command: &ProtocolObject<dyn MTLCommandBuffer>) -> TestResult {
    command.commit();
    command.waitUntilCompleted();
    if let Some(error) = command.error() {
        return Err(format!("GPU command error: {error}").into());
    }
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err("GPU command did not complete".into());
    }
    Ok(())
}

fn encode_serial(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    data: &Guarded,
    plan: DecodePlan,
) -> TestResult {
    let encoder = command
        .computeCommandEncoder()
        .ok_or("serial encoder unavailable")?;
    encoder.setComputePipelineState(pso);
    let params = plan.params();
    let panel = plan.tile.panel;
    // SAFETY: Guarded allocated exact complete spans and plan validates metadata.
    unsafe {
        for (i, offset) in data.bindings(1).offsets().into_iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(data.arena.buffer()), offset, i);
        }
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&params).cast(),
            std::mem::size_of_val(&params),
            7,
        );
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&panel).cast(), 4, 8);
        encoder.setBuffer_offset_atIndex(
            Some(data.arena.buffer()),
            data.outputs[3].0.offset + GUARD,
            9,
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 16,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

fn expect_count(
    setup: &Setup,
    before: crate::research_evidence::ResearchDispatchSnapshot,
    expected: u64,
) {
    let delta = setup
        .pipelines
        .research_dispatch_snapshot()
        .checked_since(before)
        .unwrap();
    for (slot, &count) in delta.counts.iter().enumerate() {
        assert_eq!(
            count,
            if slot == setup.candidate.kernels()[0] as usize {
                expected
            } else {
                0
            }
        );
    }
}

fn expect_family_count(
    setup: &Setup,
    before: crate::research_evidence::ResearchDispatchSnapshot,
    expected: u64,
) {
    let delta = setup
        .pipelines
        .research_dispatch_snapshot()
        .checked_since(before)
        .unwrap();
    for (slot, &count) in delta.counts.iter().enumerate() {
        assert_eq!(
            count,
            if setup
                .candidate
                .kernels()
                .iter()
                .any(|kernel| *kernel as usize == slot)
            {
                expected
            } else {
                0
            }
        );
    }
}

#[test]
#[ignore = "explicit global D512 device oracle; requires Apple9 and prebuilt strict-math oracle library"]
fn global_decode_device_oracle() -> TestResult {
    let setup = Setup::new(true)?;
    let serial = setup.context.make_pipeline("global_decode_serial_oracle")?;
    let tile = setup.candidate.global_decode_tile().unwrap();
    let mut fixtures: Vec<(String, Fixture)> = LIVE_LENGTHS
        .into_iter()
        .map(|n| (format!("L{n}"), Fixture::new(n, 32)))
        .collect();
    for n in [1, 7, 8, 9, 31, 32, 33, 257] {
        fixtures.push((format!("tail{n}"), Fixture::new(n, 7)));
    }
    let mut holes = Fixture::new(65, 7);
    holes.table[1] = -17;
    fixtures.push(("negative-page".into(), holes));
    let mut empty = Fixture::new(65, 7);
    empty.table.fill(-1);
    fixtures.push(("all-holes".into(), empty));
    let mut prefix = Fixture::new(65, 7);
    prefix.position = 30;
    for t in 31..65 {
        let base = physical_base(prefix.shape, &prefix.table, t).unwrap();
        prefix.k[base..base + 512].fill(0x7fc1);
        prefix.v[base..base + 512].fill(0x7fc1);
    }
    fixtures.push(("restored-prefix-speculative-suffix".into(), prefix.clone()));
    prefix.context = 31;
    fixtures.push(("rollback-same-visible-prefix".into(), prefix));
    let mut tied = Fixture::new(33, 7);
    tied.q.fill(0);
    fixtures.push(("equal-logits".into(), tied));
    let mut reports = Vec::new();
    let mut prefix_output: Option<Vec<u8>> = None;
    for (label, f) in &fixtures {
        let plan =
            DecodePlan::new(tile, f.shape, DecodeOutput::F32).ok_or("oracle plan rejected")?;
        let cpu = reference::output_f32(f, plan)?;
        let sampled = reference::sampled_dots(f, plan)?;
        let data = Guarded::new(&setup.context, f)?;
        let mut first: Option<Vec<u8>> = None;
        let mut max_cpu_error = 0.0_f32;
        for _ in 0..3 {
            data.reset()?;
            let before = setup.pipelines.research_dispatch_snapshot();
            let command = setup
                .context
                .queue()
                .commandBuffer()
                .ok_or("command unavailable")?;
            for (index, kind) in [(0, DecodeOutput::F32), (2, DecodeOutput::Bf16)] {
                let encoded = try_encode_global_decode(
                    &setup.pipelines,
                    &command,
                    data.arena.buffer(),
                    &dims(f.shape),
                    MetalPhase::Decode,
                    data.bindings(index),
                    kind,
                )?
                .ok_or("actual normal-route predicate refused positive fixture")?;
                assert_eq!(encoded.plan.tile, tile);
            }
            encode_serial(&command, &serial, &data, plan)?;
            complete(&command)?;
            expect_count(&setup, before, 2);
            data.check(false);
            let raw_dots = data.payload(3);
            for (slot, (head, token)) in [0_u32, 7, 15]
                .into_iter()
                .flat_map(|head| {
                    [0, (f.position as u32 + 1) / 2, f.position as u32].map(|token| (head, token))
                })
                .enumerate()
            {
                let bytes: [u8; 4] = raw_dots[slot * 4..slot * 4 + 4].try_into().unwrap();
                if let Some((_, _, cpu_dot, fp64)) = sampled
                    .iter()
                    .find(|&&(h, t, _, _)| h == head && t == token)
                {
                    let gpu_dot = f32::from_le_bytes(bytes);
                    assert_eq!(
                        gpu_dot.to_bits(),
                        cpu_dot.to_bits(),
                        "sampled GPU QK schedule"
                    );
                    let base = physical_base(f.shape, &f.table, token).unwrap();
                    let absolute: f64 = f.q[(head * DIM) as usize..((head + 1) * DIM) as usize]
                        .iter()
                        .zip(&f.k[base..base + DIM as usize])
                        .map(|(&a, &b)| {
                            (crate::attention_global_decode::widen_bf16(a) as f64
                                * crate::attention_global_decode::widen_bf16(b) as f64)
                                .abs()
                        })
                        .sum();
                    assert!(
                        (gpu_dot as f64 - *fp64).abs()
                            <= 32.0 * f32::EPSILON as f64 * absolute + 1e-30,
                        "independent sampled GPU-dot versus FP64 bound"
                    );
                } else {
                    assert_eq!(bytes, [0xff; 4], "hole sample must remain untouched");
                }
            }
            let actual = data.payload(0);
            assert_eq!(
                actual,
                data.payload(1),
                "exact FP32 serial-GPU oracle: {label}"
            );
            let mut rounded = Vec::new();
            for (bytes, &reference) in actual.chunks_exact(4).zip(&cpu) {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                assert!(value.is_finite());
                let error = (value - reference).abs();
                assert!(
                    error <= 2e-5,
                    "fixed CPU/GPU FP32 tolerance: {label}: {error}"
                );
                max_cpu_error = max_cpu_error.max(error);
                rounded.extend_from_slice(&round_bf16(value).to_le_bytes());
            }
            assert_eq!(data.payload(2), rounded, "exact once-rounded BF16: {label}");
            if let Some(previous) = &first {
                assert_eq!(&actual, previous, "repeated-use stability");
            }
            first = Some(actual);
        }
        if label == "restored-prefix-speculative-suffix" {
            prefix_output = Some(data.payload(0));
        }
        if label == "rollback-same-visible-prefix" {
            assert_eq!(prefix_output.as_ref().unwrap(), &data.payload(0));
        }
        let bf16_path = setup.directory.join(format!("{label}.bf16"));
        write_new(&bf16_path, &data.payload(2))?;
        reports.push(
            json!({"label":label,"context":f.context,"position":f.position,
            "block_size":f.shape.block_size,"repeats":3,"exact_fp32_gpu_oracle":true,
            "exact_once_rounded_bf16":true,"read_inputs_and_guard_bytes_preserved":true,
            "max_cpu_fp32_abs_error":max_cpu_error,
            "sampled_dot_source":"independent serial GPU oracle, not instrumented candidate",
            "sampled_gpu_dots_match_cpu_fp32":true,
            "sampled_gpu_dots_pass_fp64_bound":true,"sampled_fp64_dots":sampled,
            "bf16_file":bf16_path,"bf16_sha256":sha256(&bf16_path)?}),
        );
    }
    // Same adapter used by metal_forward_layer, not a test-only admission copy.
    let f = Fixture::new(33, 7);
    let data = Guarded::new(&setup.context, &f)?;
    let good = dims(f.shape);
    let mut rejected = 0_u32;
    for bad in [good; 10].into_iter().enumerate().map(|(i, mut d)| {
        match i {
            0 => d.num_heads = 8,
            1 => d.num_kv_heads = 8,
            2 => d.head_dim = 256,
            3 => d.attention_window = 1024,
            4 => d.attn_scale = 0.5,
            5 => d.hidden = 3841,
            6 => d.num_layers = 47,
            7 => d.intermediate = 15359,
            8 => d.num_tokens = 2,
            _ => d.block_size = 0,
        }
        d
    }) {
        data.reset()?;
        let before = setup.pipelines.research_dispatch_snapshot();
        let command = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("command unavailable")?;
        assert!(try_encode_global_decode(
            &setup.pipelines,
            &command,
            data.arena.buffer(),
            &bad,
            MetalPhase::Decode,
            data.bindings(0),
            DecodeOutput::F32
        )?
        .is_none());
        complete(&command)?;
        expect_count(&setup, before, 0);
        data.check(true);
        rejected += 1;
    }
    for (phase, binding) in [
        (
            MetalPhase::Prefill {
                max_seqlen_q: 1,
                batch_size: 1,
            },
            data.bindings(0),
        ),
        (
            MetalPhase::Decode,
            DecodeBuffers {
                q: data.bindings(0).q + 1,
                ..data.bindings(0)
            },
        ),
        (
            MetalPhase::Decode,
            DecodeBuffers {
                output: data.bindings(0).k,
                ..data.bindings(0)
            },
        ),
        (
            MetalPhase::Decode,
            DecodeBuffers {
                v: data.bindings(0).k,
                ..data.bindings(0)
            },
        ),
        (
            MetalPhase::Decode,
            DecodeBuffers {
                positions: usize::MAX - 3,
                ..data.bindings(0)
            },
        ),
    ] {
        let before = setup.pipelines.research_dispatch_snapshot();
        let command = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("command unavailable")?;
        assert!(try_encode_global_decode(
            &setup.pipelines,
            &command,
            data.arena.buffer(),
            &good,
            phase,
            binding,
            DecodeOutput::F32
        )?
        .is_none());
        complete(&command)?;
        expect_count(&setup, before, 0);
        data.check(true);
        rejected += 1;
    }
    for options in [
        MetalKernelOptions::default(),
        MetalKernelOptions {
            research: setup.candidate,
            ..MetalKernelOptions::default()
        },
    ] {
        let untyped = PipelineCache::with_kernel_options(options);
        let command = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("command unavailable")?;
        assert!(try_encode_global_decode(
            &untyped,
            &command,
            data.arena.buffer(),
            &good,
            MetalPhase::Decode,
            data.bindings(0),
            DecodeOutput::F32
        )?
        .is_none());
        complete(&command)?;
        data.check(true);
        rejected += 1;
        assert!(untyped
            .research_dispatch_snapshot()
            .counts
            .iter()
            .all(|&n| n == 0));
    }
    // Invalid runtime metadata IS encoded, but the shader uniformly refuses it
    // before any output write. Counters must not misclassify this as host refusal.
    let mut shader_refusals = 0;
    for case in 0..4 {
        let mut bad = f.clone();
        match case {
            0 => bad.context = 0,
            1 => bad.position = -1,
            2 => bad.position = bad.context,
            _ => bad.table[0] = bad.shape.num_blocks as i32,
        }
        let guarded = Guarded::new(&setup.context, &bad)?;
        let before = setup.pipelines.research_dispatch_snapshot();
        let command = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("command unavailable")?;
        assert!(try_encode_global_decode(
            &setup.pipelines,
            &command,
            guarded.arena.buffer(),
            &good,
            MetalPhase::Decode,
            guarded.bindings(0),
            DecodeOutput::F32
        )?
        .is_some());
        complete(&command)?;
        expect_count(&setup, before, 1);
        guarded.check(true);
        shader_refusals += 1;
    }
    let receipt = json!({"schema":"rvllm.global-decode.oracle.v1","status":"passed",
        "scope":"synthetic attention operator; not model or full-route qualification",
        "identity":setup.identity,"cases":reports,"host_rejected_dispatches":rejected,
        "encoded_metadata_refusals":shader_refusals,"timing_eligible":false});
    write_new(
        &setup.directory.join("oracle.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}

#[test]
#[ignore = "explicit bounded split-KV D512 device oracle; requires Apple9 and prebuilt strict-math library"]
fn global_decode_split_device_oracle() -> TestResult {
    let setup = Setup::new(false)?;
    let tile = setup
        .candidate
        .split_global_decode_tile()
        .ok_or("explicit split-KV candidate required")?;
    let mut fixtures: Vec<(String, Fixture)> = [
        1_u32, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
    ]
    .into_iter()
    .map(|n| (format!("L{n}"), Fixture::new(n, 32)))
    .collect();
    for (label, page) in [
        ("first-hole", 0_usize),
        ("middle-hole", 8),
        ("last-hole", 15),
    ] {
        let mut fixture = Fixture::new(4096, 256);
        // Fixture::new deliberately reserves two future logical blocks for
        // tail-read detection.  This bounded split family admits an exact
        // 4096-token logical table, so remove only those unused table entries;
        // retain the extra physical cache allocation as poisoned padding.
        fixture.shape.max_blocks = 16;
        fixture.table.truncate(16);
        fixture.table[page] = -1;
        fixtures.push((label.into(), fixture));
    }
    let mut reports = Vec::new();
    for (label, fixture) in fixtures {
        let plan = SplitDecodePlan::new(tile, fixture.shape, DecodeOutput::F32)
            .ok_or("split plan rejected oracle fixture")?;
        let reference_plan = DecodePlan::new(
            crate::attention_global_decode::DecodeTile {
                rows: tile.rows,
                panel: tile.panel,
                threads: tile.threads,
            },
            fixture.shape,
            DecodeOutput::F32,
        )
        .ok_or("reference plan rejected split fixture")?;
        let cpu = reference::output_f32(&fixture, reference_plan)?;
        let data = Guarded::new(&setup.context, &fixture)?;
        let before = setup.pipelines.research_dispatch_snapshot();
        let command = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("command unavailable")?;
        for (output_index, output_kind) in [(0, DecodeOutput::F32), (2, DecodeOutput::Bf16)] {
            let encoded = try_encode_split_global_decode(
                &setup.pipelines,
                &command,
                data.arena.buffer(),
                &dims(fixture.shape),
                MetalPhase::Decode,
                data.split_bindings(output_index),
                output_kind,
            )?
            .ok_or("normal-route split predicate refused positive fixture")?;
            assert_eq!(encoded.plan.partial_count, 16);
            assert_eq!(encoded.plan.scratch_bytes, 16 * 16 * 514 * 4);
        }
        complete(&command)?;
        expect_family_count(&setup, before, 2);
        data.check(false);
        let actual = data.payload(0);
        let mut max_cpu_error = 0.0_f32;
        let mut rounded = Vec::with_capacity(FLOAT_BYTES / 2);
        for (bytes, &reference) in actual.chunks_exact(4).zip(&cpu) {
            let value = f32::from_le_bytes(bytes.try_into().unwrap());
            assert!(value.is_finite(), "nonfinite split result: {label}");
            max_cpu_error = max_cpu_error.max((value - reference).abs());
            rounded.extend_from_slice(&round_bf16(value).to_le_bytes());
        }
        // Split reduction changes FP32 association; this is an explicit
        // numerical bound, not a false bitwise-serial-parity claim.
        assert!(
            max_cpu_error <= 5e-5,
            "split CPU error: {label}: {max_cpu_error}"
        );
        assert_eq!(
            data.payload(2),
            rounded,
            "single final BF16 rounding: {label}"
        );
        let bf16_path = setup.directory.join(format!("{label}.split.bf16"));
        write_new(&bf16_path, &data.payload(2))?;
        reports.push(json!({"label":label,"max_cpu_fp32_abs_error":max_cpu_error,
            "bitwise_serial_parity_claimed":false,"once_rounded_bf16":true,
            "partial_count":plan.partial_count,"scratch_bytes":plan.scratch_bytes,
            "bf16_file":bf16_path,"bf16_sha256":sha256(&bf16_path)?}));
    }
    let receipt = json!({"schema":"rvllm.global-decode.split-oracle.v1","status":"passed",
        "scope":"synthetic bounded split-KV attention operator; not model/full-route qualification",
        "identity":setup.identity,"cases":reports,"timing_eligible":false});
    write_new(
        &setup.directory.join("split-oracle.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}

fn encode_baseline(
    setup: &Setup,
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    data: &Guarded,
    shape: DecodeShape,
) -> TestResult {
    let encoder = command
        .computeCommandEncoder()
        .ok_or("baseline encoder unavailable")?;
    encoder.setComputePipelineState(setup.pipelines.get("attention_decode_f16")?);
    let bindings = data.bindings(2);
    let integers = [
        shape.sequences,
        shape.heads,
        shape.kv_heads,
        shape.head_dim,
        shape.block_size,
        shape.max_blocks,
    ];
    // SAFETY: benchmark admits valid dense fixture metadata and checked spans;
    // the untouched incumbent has this fixed BF16 typed ABI.
    unsafe {
        for (i, offset) in bindings.offsets()[..6].iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(data.arena.buffer()), *offset, i);
        }
        for (i, value) in integers.iter().enumerate() {
            encoder.setBytes_length_atIndex(std::ptr::NonNull::from(value).cast(), 4, i + 6);
        }
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&shape.scale).cast(), 4, 12);
        encoder.setBytes_length_atIndex(std::ptr::NonNull::from(&shape.window).cast(), 4, 13);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 16,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    Ok(())
}

fn control_snapshot() -> TestResult<Value> {
    let mut values = Vec::new();
    for args in [["-g", "batt"], ["-g", "therm"], ["-g", "custom"]] {
        let output = Command::new("/usr/bin/pmset").args(args).output()?;
        if !output.status.success() {
            return Err("power/thermal snapshot failed".into());
        }
        values.push(String::from_utf8(output.stdout)?);
    }
    Ok(json!(values))
}

#[test]
#[ignore = "explicit raw operator ABBA; requires matching passed oracle and prebuilt core-only library"]
fn global_decode_abba() -> TestResult {
    let setup = Setup::new(false)?;
    let length: u32 = std::env::var(format!("{PREFIX}LENGTH"))?.parse()?;
    if !LIVE_LENGTHS.contains(&length) {
        return Err("length outside sealed five-cell sweep".into());
    }
    let oracle_path = env_path("ORACLE_RECEIPT")?;
    let oracle: Value = serde_json::from_slice(&std::fs::read(&oracle_path)?)?;
    if oracle["schema"] != "rvllm.global-decode.oracle.v1"
        || oracle["status"] != "passed"
        || oracle["identity"]["candidate"] != setup.identity["candidate"]
        || oracle["identity"]["core_sha256"] != setup.identity["core_sha256"]
        || oracle["identity"]["test_executable_sha256"] != setup.identity["test_executable_sha256"]
    {
        return Err("matching native oracle receipt required before any benchmark".into());
    }
    let label = format!("L{length}");
    let case = oracle["cases"]
        .as_array()
        .ok_or("missing oracle cases")?
        .iter()
        .find(|case| case["label"] == label)
        .ok_or("length not qualified")?;
    let expected_path = PathBuf::from(case["bf16_file"].as_str().ok_or("missing exact output")?);
    if case["bf16_sha256"] != sha256(&expected_path)? {
        return Err("oracle output identity mismatch".into());
    }
    let expected = std::fs::read(&expected_path)?;
    let f = Fixture::new(length, 32);
    let data = Guarded::new(&setup.context, &f)?;
    let layer = dims(f.shape);
    // All allocation, file IO, environment reads, PSO construction and controls
    // are outside the encode loop. Only the existing atomic receipt hook stays.
    let run = |arm: char, repeats: u32| -> TestResult<(f64, f64)> {
        let before = setup.pipelines.research_dispatch_snapshot();
        let wall_start = Instant::now();
        let command = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("command unavailable")?;
        for _ in 0..repeats {
            if arm == 'A' {
                encode_baseline(&setup, &command, &data, f.shape)?;
            } else {
                try_encode_global_decode(
                    &setup.pipelines,
                    &command,
                    data.arena.buffer(),
                    &layer,
                    MetalPhase::Decode,
                    data.bindings(2),
                    DecodeOutput::Bf16,
                )?
                .ok_or("candidate fallback invalidates measurement")?;
            }
        }
        complete(&command)?;
        let wall = wall_start.elapsed().as_secs_f64();
        let gpu = command.GPUEndTime() - command.GPUStartTime();
        if !gpu.is_finite() || gpu <= 0.0 {
            return Err("GPU timestamps unavailable".into());
        }
        expect_count(&setup, before, if arm == 'B' { repeats as u64 } else { 0 });
        data.check(false);
        if arm == 'B' {
            assert_eq!(
                data.payload(2),
                expected,
                "timed repeated-use output changed"
            );
        } else {
            for bytes in data.payload(2).chunks_exact(2) {
                assert!(
                    crate::attention_global_decode::widen_bf16(u16::from_le_bytes(
                        bytes.try_into().unwrap()
                    ))
                    .is_finite()
                );
            }
        }
        Ok((gpu, wall))
    };
    // Five warmups per arm, then five complete ABBA blocks, no selection/deletion.
    for _ in 0..5 {
        run('A', 1)?;
        run('B', 1)?;
    }
    let mut samples = Vec::new();
    for block in 0..5 {
        for (position, arm) in ['A', 'B', 'B', 'A'].into_iter().enumerate() {
            let before = control_snapshot()?;
            let (gpu, wall) = run(arm, 100)?;
            let after = control_snapshot()?;
            let sample = json!({"block":block,"position":position,"arm":arm.to_string(),
                "dispatches":100,"gpu_seconds":gpu,"synchronized_wall_seconds":wall,
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
        "length":length,"baseline":"attention_decode_f16 (BF16 typed)",
        "candidate":setup.candidate.name(),"warmups_per_arm":5,"blocks":5,
        "dispatches_per_sample":100,"control_drift_fraction":drift,"control_drift_limit":0.05,
        "control_drift_passed":drift <= 0.05,"source_compiles_during_samples":0,
        "samples":samples,"timing_eligible":false,
        "pending":"external queue condition continuity, sealed referee and real-weight full-route gates",
        "promotion":false});
    write_new(
        &setup.directory.join("abba.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    if drift > 0.05 {
        return Err("control drift exceeded 5%; retain all samples as invalid".into());
    }
    Ok(())
}

#[test]
#[ignore = "explicit bounded split-KV ABBA v2; requires matching passed split oracle and prebuilt core-only library"]
fn global_decode_split_abba_v2() -> TestResult {
    let setup = Setup::new(false)?;
    setup
        .candidate
        .split_global_decode_tile()
        .ok_or("explicit split-KV candidate required")?;
    let length: u32 = std::env::var(format!("{PREFIX}LENGTH"))?.parse()?;
    if !LIVE_LENGTHS.contains(&length) {
        return Err("length outside sealed five-cell sweep".into());
    }
    let oracle_path = env_path("ORACLE_RECEIPT")?;
    let oracle: Value = serde_json::from_slice(&std::fs::read(&oracle_path)?)?;
    if oracle["schema"] != "rvllm.global-decode.split-oracle.v1"
        || oracle["status"] != "passed"
        || oracle["identity"] != setup.identity
        || oracle["identity"]["oracle_library"] != false
    {
        return Err("matching native split oracle receipt required before any benchmark".into());
    }
    let label = format!("L{length}");
    let case = oracle["cases"]
        .as_array()
        .ok_or("missing split oracle cases")?
        .iter()
        .find(|case| case["label"] == label)
        .ok_or("length not split-qualified")?;
    let expected_path = PathBuf::from(case["bf16_file"].as_str().ok_or("missing exact output")?);
    if case["bf16_sha256"] != sha256(&expected_path)? {
        return Err("split oracle output identity mismatch".into());
    }
    let expected = std::fs::read(&expected_path)?;
    let fixture = Fixture::new(length, 32);
    let data = Guarded::new(&setup.context, &fixture)?;
    let layer = dims(fixture.shape);
    let run = |arm: char, repeats: u32| -> TestResult<(f64, f64, f64, f64)> {
        let before = setup.pipelines.research_dispatch_snapshot();
        let wall_start = Instant::now();
        if arm == 'A' {
            let command = setup
                .context
                .queue()
                .commandBuffer()
                .ok_or("baseline command unavailable")?;
            for _ in 0..repeats {
                encode_baseline(&setup, &command, &data, fixture.shape)?;
            }
            complete(&command)?;
            let total = command.GPUEndTime() - command.GPUStartTime();
            if !total.is_finite() || total <= 0.0 {
                return Err("baseline GPU timestamps unavailable".into());
            }
            expect_family_count(&setup, before, 0);
            return Ok((0.0, 0.0, total, wall_start.elapsed().as_secs_f64()));
        }
        let partial = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("split partial command unavailable")?;
        for _ in 0..repeats {
            try_encode_split_global_decode_stage(
                &setup.pipelines,
                &partial,
                data.arena.buffer(),
                &layer,
                MetalPhase::Decode,
                data.split_bindings(2),
                DecodeOutput::Bf16,
                SplitGlobalDecodeStage::Partial,
            )?
            .ok_or("split partial fallback invalidates measurement")?;
        }
        complete(&partial)?;
        let partial_gpu = partial.GPUEndTime() - partial.GPUStartTime();
        let merge = setup
            .context
            .queue()
            .commandBuffer()
            .ok_or("split merge command unavailable")?;
        for _ in 0..repeats {
            try_encode_split_global_decode_stage(
                &setup.pipelines,
                &merge,
                data.arena.buffer(),
                &layer,
                MetalPhase::Decode,
                data.split_bindings(2),
                DecodeOutput::Bf16,
                SplitGlobalDecodeStage::Merge,
            )?
            .ok_or("split merge fallback invalidates measurement")?;
        }
        complete(&merge)?;
        let merge_gpu = merge.GPUEndTime() - merge.GPUStartTime();
        let total_gpu = partial_gpu + merge_gpu;
        if !partial_gpu.is_finite()
            || partial_gpu <= 0.0
            || !merge_gpu.is_finite()
            || merge_gpu <= 0.0
            || !total_gpu.is_finite()
        {
            return Err("split GPU timestamps unavailable".into());
        }
        expect_family_count(&setup, before, u64::from(repeats));
        data.check(false);
        assert_eq!(data.payload(2), expected, "timed split output changed");
        Ok((
            partial_gpu,
            merge_gpu,
            total_gpu,
            wall_start.elapsed().as_secs_f64(),
        ))
    };
    for _ in 0..5 {
        run('A', 1)?;
        run('B', 1)?;
    }
    let mut samples = Vec::new();
    for block in 0..5 {
        for (position, arm) in ['A', 'B', 'B', 'A'].into_iter().enumerate() {
            let before = control_snapshot()?;
            let (partial_gpu, merge_gpu, total_gpu, wall) = run(arm, 100)?;
            let after = control_snapshot()?;
            let sample = json!({"block":block,"position":position,"arm":arm.to_string(),
                "operations":100,"partial_dispatches":if arm == 'B' {100} else {0},
                "merge_dispatches":if arm == 'B' {100} else {0},
                "baseline_dispatches":if arm == 'A' {100} else {0},
                "partial_gpu_seconds":partial_gpu,"merge_gpu_seconds":merge_gpu,
                "total_gpu_seconds":total_gpu,"synchronized_wall_seconds":wall,
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
    let receipt = json!({"schema":"rvllm.global-decode.abba.v2","status":"collected",
        "identity":setup.identity,"oracle_receipt_sha256":sha256(&oracle_path)?,
        "length":length,"baseline":"attention_decode_f16 (BF16 typed)",
        "candidate":setup.candidate.name(),"warmups_per_arm":5,"blocks":5,
        "operations_per_sample":100,"candidate_dispatches_per_operation":2,
        "source_compiles_during_samples":0,"samples":samples,
        "timing_metric":"total_gpu_seconds = partial_gpu_seconds + merge_gpu_seconds",
        "conditions_are_observations_only":true,"correctness_prerequisite":"passed",
        "timing_eligible":false,"promotion":false});
    write_new(
        &setup.directory.join("abba.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}
