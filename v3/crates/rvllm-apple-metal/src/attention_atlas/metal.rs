//! The ONLY new unsafe/Metal boundary. No runtime shader compilation. Public
//! methods own immutable snapshots and synchronously retire GPU work before
//! returning, so callers cannot mutate a cache underneath an encoded dispatch.
use super::{
    experiment::{self, Build, Control, OracleReceipt, Pin, Prepared, Sample, Spec},
    plan::{self, Layout, Params, Strategy},
    reference::{self, Data},
    CacheFormat, Error, Output, Plan, Result,
};
use crate::arena::MetalRegion;
use crate::{MetalBufferArena, MetalContext};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize,
};
use serde::Serialize;
use std::{path::Path, time::Instant};

type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
fn size(x: usize, y: usize, z: usize) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: z,
    }
}
fn err(e: impl std::fmt::Display) -> Error {
    Error::new(e.to_string())
}

struct Buffers {
    arena: MetalBufferArena,
    whole: MetalRegion,
    layout: Layout,
    original: Vec<u8>,
}
impl Buffers {
    fn new(context: &MetalContext, plan: Plan, data: &Data, limit: usize) -> Result<Self> {
        let layout = Layout::new(plan, limit)?;
        layout.validate(plan, layout.capacity)?;
        let mut arena = MetalBufferArena::new(context.device(), layout.capacity).map_err(err)?;
        let whole = arena
            .region("atlas-guarded-snapshot", layout.capacity, 32)
            .map_err(err)?;
        let mut original = vec![0xa5; layout.capacity];
        for (i, payload) in data.payloads(plan)?.iter().enumerate() {
            original[layout.spans[i].clone()].copy_from_slice(payload);
        }
        for i in [5, 6, 7] {
            original[layout.spans[i].clone()].fill(0xcd);
        }
        // SAFETY: fresh allocation, no GPU work references it.
        unsafe {
            arena.write_region(&whole, &original).map_err(err)?;
        }
        Ok(Self {
            arena,
            whole,
            layout,
            original,
        })
    }
    fn offset(&self, i: usize) -> usize {
        self.layout.spans[i].start
    }
    fn reset(&self) -> Result<()> {
        // SAFETY: private callers only reset before first submission or after wait.
        for i in [5, 6, 7] {
            let r = MetalRegion {
                name: "atlas-reset".into(),
                offset: self.offset(i),
                size: self.layout.spans[i].len(),
            };
            unsafe {
                self.arena
                    .write_region(&r, &self.original[self.layout.spans[i].clone()])
                    .map_err(err)?;
            }
        }
        Ok(())
    }
    fn snapshot(&self) -> Vec<u8> {
        // SAFETY: all callers synchronously completed the command. Bounds were
        // checked by Layout and cover the exact owned shared-mode allocation.
        unsafe {
            std::slice::from_raw_parts(self.arena.host_ptr(&self.whole), self.layout.capacity)
                .to_vec()
        }
    }
    fn status(&self) -> u32 {
        // Only four status bytes are touched; do not copy the entire KV cache
        // between samples and accidentally benchmark CPU-induced cache churn.
        let r = MetalRegion {
            name: String::new(),
            offset: self.offset(7),
            size: 4,
        };
        // SAFETY: command completed and the checked status span is four bytes.
        let bytes = unsafe { std::slice::from_raw_parts(self.arena.host_ptr(&r), 4) };
        u32::from_le_bytes(bytes.try_into().unwrap())
    }
    fn outputs(&self, plan: Plan, kind: Output) -> Vec<f32> {
        let b = self.snapshot();
        let out = &b[self.layout.spans[5].clone()];
        let n = plan.shape.output_elements();
        match kind {
            Output::F32 => out[..n * 4]
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
            Output::Bf16 => out[..n * 2]
                .chunks_exact(2)
                .map(|x| reference::widen(u16::from_le_bytes(x.try_into().unwrap())))
                .collect(),
        }
    }
    fn check(&self, plan: Plan, kind: Output, untouched: bool) -> Result<()> {
        let b = self.snapshot();
        let n = plan.shape.output_elements();
        for i in 0..b.len() {
            let written = (i >= self.layout.spans[7].start && i < self.layout.spans[7].end)
                || (!untouched
                    && ((i >= self.layout.spans[5].start
                        && i < self.layout.spans[5].start
                            + n * if kind == Output::F32 { 4 } else { 2 })
                        || (plan.candidate.splits > 1 && self.layout.spans[6].contains(&i))));
            if !written && b[i] != self.original[i] {
                return Err(Error::new(format!(
                    "guard/input/unwritten-output mutation at arena byte {i}"
                )));
            }
        }
        Ok(())
    }
    fn overwrite_first_position(&self, value: i32) -> Result<()> {
        let r = MetalRegion {
            name: "negative-control".into(),
            offset: self.offset(4),
            size: 4,
        };
        // SAFETY: used only by synchronous negative-control tests, with no work
        // in flight. Never an API for mutating a production cache.
        unsafe {
            self.arena
                .write_region(&r, &value.to_le_bytes())
                .map_err(err)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Arm {
    Control,
    Candidate,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Candidate => "candidate",
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Dispatch {
    pub host_ns: u64,
    pub gpu_ns: u64,
    pub encoded: u32,
    pub completed: u32,
    pub metadata_status: u32,
}

/// Prepared operator trial owner. This does not alter normal rvLLM dispatch.
/// PSOs, arenas, and immutable constants are prepared before any measurements.
/// The safe synchronous ownership surface is deliberately narrower than the
/// normal command-batch route until integration receives its own qualification.
pub struct PreparedKernel {
    context: MetalContext,
    plan: Plan,
    control_plan: Plan,
    candidate: Pso,
    control: Pso,
    validate: Pso,
    merge: Pso,
    control_kind: Control,
    control_name: &'static str,
    candidate_buffers: Buffers,
    control_buffers: Buffers,
}
impl PreparedKernel {
    pub fn new(spec: &Spec, data: &Data, library: &Path) -> Result<Self> {
        let plan = spec.plan(Output::Bf16)?;
        data.validate(plan)?;
        let vector = plan::catalog()[0];
        let mut control_plan = Plan::new(vector, plan.shape, plan.cache, Output::Bf16)?;
        let materialized = if spec.control != Control::AtlasVector {
            let result = data.materialized(plan)?;
            control_plan.cache = CacheFormat::SeparateBf16;
            result
        } else {
            data.clone()
        };
        let a = Layout::new(plan, spec.arena_limit_bytes)?;
        let b = Layout::new(control_plan, spec.arena_limit_bytes)?;
        if a.capacity
            .checked_add(b.capacity)
            .ok_or_else(|| Error::new("two-arm arena overflow"))?
            > spec.arena_limit_bytes
        {
            return Err(Error::new(
                "combined candidate/control arenas exceed declared budget",
            ));
        }
        let mut context = MetalContext::new().map_err(err)?;
        context.load_metallib(library).map_err(err)?;
        let candidate = context.make_pipeline("atlas_candidate").map_err(err)?;
        let validate = context.make_pipeline("atlas_validate").map_err(err)?;
        let merge = context.make_pipeline("atlas_merge").map_err(err)?;
        let control_name = if let Some(name) = spec.control.global_name() {
            name
        } else {
            match spec.control {
                Control::AtlasVector => "atlas_vector",
                Control::ExistingPrefillSimd => "attention_prefill_simdgroup_f16",
                Control::ExistingDefault if plan.shape.queries == 1 && plan.shape.dim == 256 => {
                    "attention_decode_online_f16"
                }
                Control::ExistingDefault if plan.shape.queries == 1 => "attention_decode_f16",
                Control::ExistingDefault => "attention_prefill_f16",
                _ => return Err(Error::new("unknown explicit control")),
            }
        };
        let control = context.make_pipeline(control_name).map_err(err)?;
        for (pso, p) in [
            (&candidate, plan),
            (&validate, control_plan),
            (&merge, control_plan),
        ] {
            if !p.pso_fits(
                pso.threadExecutionWidth(),
                pso.maxTotalThreadsPerThreadgroup(),
                pso.staticThreadgroupMemoryLength(),
                context.max_threadgroup_memory(),
            ) {
                return Err(Error::new(
                    "queried pipeline/device limit rejects candidate; no fallback",
                ));
            }
        }
        let control_threads = if let Some(tile) = spec.control.global_tile() {
            tile.threads as usize
        } else if control_name == "attention_decode_f16" || control_name == "attention_prefill_f16"
        {
            1
        } else {
            32
        };
        if control.maxTotalThreadsPerThreadgroup() < control_threads
            || control.threadExecutionWidth() != 32
            || control.staticThreadgroupMemoryLength() > context.max_threadgroup_memory()
        {
            return Err(Error::new("incumbent pipeline is not admitted"));
        }
        let candidate_buffers = Buffers::new(&context, plan, data, spec.arena_limit_bytes)?;
        let control_buffers = Buffers::new(
            &context,
            control_plan,
            &materialized,
            spec.arena_limit_bytes,
        )?;
        Ok(Self {
            context,
            plan,
            control_plan,
            candidate,
            control,
            validate,
            merge,
            control_kind: spec.control,
            control_name,
            candidate_buffers,
            control_buffers,
        })
    }
    pub fn control_name(&self) -> &'static str {
        self.control_name
    }
    pub fn plan(&self) -> Plan {
        self.plan
    }
    pub fn device_receipt(&self) -> serde_json::Value {
        serde_json::json!({"name":self.context.device().name().to_string(),
            "max_threadgroup_memory_bytes":self.context.max_threadgroup_memory(),
            "recommended_working_set_bytes":self.context.recommended_max_working_set_size(),
            "candidate_thread_execution_width":self.candidate.threadExecutionWidth(),
            "candidate_max_threads":self.candidate.maxTotalThreadsPerThreadgroup(),
            "candidate_static_threadgroup_bytes":self.candidate.staticThreadgroupMemoryLength(),
            "candidate_arena_bytes":self.candidate_buffers.layout.capacity,
            "control_arena_bytes":self.control_buffers.layout.capacity})
    }
    fn buffers(&self, arm: Arm) -> &Buffers {
        match arm {
            Arm::Control => &self.control_buffers,
            Arm::Candidate => &self.candidate_buffers,
        }
    }
    fn arm_plan(&self, arm: Arm, output: Output) -> Plan {
        let mut p = match arm {
            Arm::Control => self.control_plan,
            Arm::Candidate => self.plan,
        };
        p.output = output;
        p
    }
    fn bind_atlas(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        pso: &ProtocolObject<dyn MTLComputePipelineState>,
        b: &Buffers,
        p: Plan,
        grid: MTLSize,
        threads: usize,
    ) -> Result<()> {
        let enc = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::new("compute encoder unavailable"))?;
        enc.setComputePipelineState(pso);
        let params = p.params();
        // SAFETY: Layout checked exact nonaliasing lengths, data was validated,
        // and this owner has exclusive host access. The shared MTLBuffer uses
        // default hazard tracking; separate ended encoders order status and
        // partial-state producers before their consumers.
        unsafe {
            for i in 0..12 {
                enc.setBuffer_offset_atIndex(Some(b.arena.buffer()), b.offset(i), i);
            }
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(&params).cast(),
                std::mem::size_of::<Params>(),
                12,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, size(threads, 1, 1));
        enc.endEncoding();
        Ok(())
    }
    fn encode_control(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        b: &Buffers,
        p: Plan,
    ) -> Result<()> {
        if self.control_kind == Control::AtlasVector {
            return self.bind_atlas(
                command,
                &self.control,
                b,
                p,
                size(p.grid[0], p.grid[1], 1),
                32,
            );
        }
        if let Some(tile) = self.control_kind.global_tile() {
            use crate::attention_global_decode::{
                DecodeBuffers, DecodeOutput, DecodePlan, DecodeShape,
            };
            let s = p.shape;
            let old = DecodePlan::new(
                tile,
                DecodeShape {
                    sequences: 1,
                    heads: 16,
                    kv_heads: 1,
                    head_dim: 512,
                    block_size: s.page_size,
                    max_blocks: s.max_blocks,
                    num_blocks: s.physical_blocks,
                    window: 0,
                    scale: 1.0,
                },
                DecodeOutput::Bf16,
            )
            .ok_or_else(|| Error::new("existing cooperative control refused plan"))?;
            let spans = DecodeBuffers {
                q: b.offset(0),
                k: b.offset(1),
                v: b.offset(2),
                output: b.offset(5),
                block_tables: b.offset(3),
                context_lens: b.offset(12),
                positions: b.offset(4),
            };
            if !old.buffers_fit(spans, b.arena.capacity()) {
                return Err(Error::new("existing cooperative bindings refused"));
            }
            let enc = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::new("existing cooperative encoder unavailable"))?;
            enc.setComputePipelineState(&self.control);
            let params = old.params();
            // SAFETY: reuse the existing exact ABI and checked span validator.
            unsafe {
                for (i, offset) in spans.offsets().into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(b.arena.buffer()), offset, i);
                }
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&params).cast(),
                    std::mem::size_of_val(&params),
                    7,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                size(old.grid[0], old.grid[1], old.grid[2]),
                size(tile.threads as usize, 1, 1),
            );
            enc.endEncoding();
            return Ok(());
        }
        let enc = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::new("incumbent encoder unavailable"))?;
        enc.setComputePipelineState(&self.control);
        let s = p.shape;
        let decode = self.control_name.starts_with("attention_decode");
        // Decode controls use the final query's actual context rather than the
        // allocation/live length, preserving invisible future suffixes.
        let binds: &[usize] = if decode {
            &[0, 1, 2, 5, 3, 12]
        } else {
            &[0, 1, 2, 5, 3, 12, 13, 4]
        };
        // SAFETY: same owned/bounds-checked spans; these are the exact existing
        // kernel argument layouts. Scale is explicit 1.0; no generic sqrt(D).
        unsafe {
            for (i, index) in binds.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(b.arena.buffer()), b.offset(*index), i);
            }
            let decode_words = [
                1,
                16,
                s.kv_heads,
                s.dim,
                s.page_size,
                s.max_blocks,
                1.0f32.to_bits(),
                s.window,
            ];
            let prefill_words = [
                s.queries,
                1,
                16,
                s.kv_heads,
                s.dim,
                s.page_size,
                s.max_blocks,
                1.0f32.to_bits(),
                s.window,
            ];
            let words: &[u32] = if decode {
                &decode_words
            } else {
                &prefill_words
            };
            for (j, word) in words.iter().enumerate() {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::from(word).cast(),
                    4,
                    binds.len() + j,
                );
            }
        }
        let t = if self.control_name == "attention_decode_f16"
            || self.control_name == "attention_prefill_f16"
        {
            1
        } else {
            32
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            if decode {
                size(16, 1, 1)
            } else {
                size(s.queries as usize, 16, 1)
            },
            size(t, 1, 1),
        );
        enc.endEncoding();
        Ok(())
    }
    /// One complete operator call, including metadata validation and any merge.
    /// Counts are local to the exact command, never qualification or promotion.
    pub fn dispatch(&mut self, arm: Arm, output: Output) -> Result<Dispatch> {
        if matches!(arm, Arm::Control)
            && self.control_kind != Control::AtlasVector
            && output != Output::Bf16
        {
            return Err(Error::new("existing controls write BF16 only"));
        }
        let p = self.arm_plan(arm, output);
        let b = self.buffers(arm);
        let start = Instant::now();
        let command = self
            .context
            .queue()
            .commandBuffer()
            .ok_or_else(|| Error::new("command buffer unavailable"))?;
        self.bind_atlas(&command, &self.validate, b, p, size(1, 1, 1), 32)?;
        match arm {
            Arm::Candidate => self.bind_atlas(
                &command,
                &self.candidate,
                b,
                p,
                size(p.grid[0], p.grid[1], p.grid[2]),
                p.candidate.threads as usize,
            )?,
            Arm::Control => self.encode_control(&command, b, p)?,
        }
        let encoded = if p.candidate.splits > 1 {
            self.bind_atlas(
                &command,
                &self.merge,
                b,
                p,
                size(p.shape.queries as usize * 16, 1, 1),
                32,
            )?;
            3
        } else {
            2
        };
        command.commit();
        command.waitUntilCompleted();
        let host_ns = u64::try_from(start.elapsed().as_nanos()).map_err(err)?;
        if let Some(e) = command.error() {
            return Err(Error::new(format!("GPU error: {e}")));
        }
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::new("GPU did not report completed"));
        }
        let seconds = command.GPUEndTime() - command.GPUStartTime();
        if !seconds.is_finite() || seconds <= 0.0 || seconds * 1e9 > u64::MAX as f64 {
            return Err(Error::new("invalid GPU timestamp interval"));
        }
        // Shared-buffer status read occurs after completion and outside measured
        // host interval. No digesting, logging or allocation is in the shader.
        Ok(Dispatch {
            host_ns,
            gpu_ns: (seconds * 1e9).round() as u64,
            encoded,
            completed: encoded,
            metadata_status: b.status(),
        })
    }
    pub fn output(&self, arm: Arm, kind: Output) -> Vec<f32> {
        self.buffers(arm).outputs(self.arm_plan(arm, kind), kind)
    }
    pub fn check(&self, arm: Arm, kind: Output) -> Result<()> {
        self.buffers(arm)
            .check(self.arm_plan(arm, kind), kind, false)
    }
    pub fn reset(&self, arm: Arm) -> Result<()> {
        self.buffers(arm).reset()
    }
    fn negative_position(&mut self, data: &Data) -> Result<()> {
        self.candidate_buffers.reset()?;
        self.candidate_buffers
            .overwrite_first_position(self.plan.shape.live_keys as i32)?;
        let d = self.dispatch(Arm::Candidate, Output::F32)?;
        // Restore expected input before guard comparison. Refusal may change
        // status ONLY; all output/partial bytes must remain poisoned.
        self.candidate_buffers
            .overwrite_first_position(data.positions[0])?;
        self.candidate_buffers.check(self.plan, Output::F32, true)?;
        if d.metadata_status != 2 {
            return Err(Error::new("negative position was not refused"));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct OracleDetail {
    scope: &'static str,
    candidate: String,
    control: String,
    fp32: reference::Accuracy,
    bf16: reference::Accuracy,
    control_accuracy: reference::Accuracy,
    repeat_bits_equal: bool,
    bf16_is_rounded_fp32: bool,
    negative_metadata_refused: bool,
    source_sha256: String,
    matrix_order_requires_separate_qualification: bool,
    full_route_qualified: bool,
    promotion_eligible: bool,
}
fn qualify(
    owner: &mut PreparedKernel,
    spec: &Spec,
    data: &Data,
    out: &Path,
    source_sha256: String,
) -> Result<bool> {
    let expected = reference::oracle(spec.plan(Output::F32)?, data, spec.max_oracle_fmas)?;
    if expected.iter().any(|x| !x.is_finite()) {
        return Err(Error::new("nonfinite reference; native dispatch refused"));
    }
    owner.reset(Arm::Candidate)?;
    let d = owner.dispatch(Arm::Candidate, Output::F32)?;
    if d.metadata_status != 0 {
        return Err(Error::new("candidate metadata unexpectedly refused"));
    }
    owner.check(Arm::Candidate, Output::F32)?;
    let f32out = owner.output(Arm::Candidate, Output::F32);
    let fp32 = reference::compare(&f32out, &expected, spec.fp32_tolerance)?;
    owner.reset(Arm::Candidate)?;
    if owner.dispatch(Arm::Candidate, Output::F32)?.metadata_status != 0 {
        return Err(Error::new("repeat metadata refusal"));
    }
    owner.check(Arm::Candidate, Output::F32)?;
    let again = owner.output(Arm::Candidate, Output::F32);
    let repeat_bits_equal = f32out
        .iter()
        .zip(&again)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    owner.reset(Arm::Candidate)?;
    if owner
        .dispatch(Arm::Candidate, Output::Bf16)?
        .metadata_status
        != 0
    {
        return Err(Error::new("BF16 metadata refusal"));
    }
    owner.check(Arm::Candidate, Output::Bf16)?;
    let bfout = owner.output(Arm::Candidate, Output::Bf16);
    let bf16 = reference::compare(&bfout, &expected, spec.bf16_tolerance)?;
    let bf16_is_rounded_fp32 = f32out
        .iter()
        .zip(&bfout)
        .all(|(f, b)| reference::widen(reference::round(*f)).to_bits() == b.to_bits());
    owner.reset(Arm::Control)?;
    if owner.dispatch(Arm::Control, Output::Bf16)?.metadata_status != 0 {
        return Err(Error::new("control metadata refusal"));
    }
    owner.check(Arm::Control, Output::Bf16)?;
    let control_accuracy = reference::compare(
        &owner.output(Arm::Control, Output::Bf16),
        &expected,
        spec.bf16_tolerance,
    )?;
    owner.negative_position(data)?;
    let passed = fp32.passed
        && bf16.passed
        && control_accuracy.passed
        && repeat_bits_equal
        && bf16_is_rounded_fp32;
    experiment::write_json(
        &out.join("oracle-detail.json"),
        &OracleDetail {
            scope: "operator_only",
            candidate: spec.candidate.name(),
            control: owner.control_name().into(),
            fp32,
            bf16,
            control_accuracy,
            repeat_bits_equal,
            bf16_is_rounded_fp32,
            negative_metadata_refused: true,
            source_sha256,
            matrix_order_requires_separate_qualification: spec.candidate.strategy
                == Strategy::Matrix,
            full_route_qualified: false,
            promotion_eligible: false,
        },
    )?;
    Ok(passed)
}

/// Explicit native operation; callers must hold the existing queue/owner lock.
/// No lock stealing, retries, process termination or thermal changes are made.
pub fn run(
    prepared_dir: &Path,
    build_dir: &Path,
    oracle_receipt: Option<&Path>,
    out: &Path,
) -> Result<()> {
    let (p, spec) = Prepared::load(prepared_dir)?;
    let build = Build::load(prepared_dir, build_dir)?;
    let receipt_pin = if let Some(path) = oracle_receipt {
        experiment::validate_oracle(path, prepared_dir, build_dir)?;
        Some(Pin::file(path)?)
    } else {
        None
    };
    experiment::fresh(out)?;
    let result = (|| {
        // Materialize and validate inputs before loading a device; full FP64
        // comparison is outside every timed interval.
        let data = spec.data()?;
        let mut owner = PreparedKernel::new(&spec, &data, &build.library.path)?;
        experiment::write_json(&out.join("device.json"), &owner.device_receipt())?;
        let passed = qualify(&mut owner, &spec, &data, out, p.source.sha256.clone())?;
        p.verify()?;
        build.library.verify()?;
        for pin in spec.input_pins() {
            pin.verify()?;
        }
        if !passed {
            return Err(Error::new(
                "predeclared numerical component gate failed; no timing",
            ));
        }
        let receipt = OracleReceipt {
            schema: "rvllm.attention_atlas.oracle.v1".into(),
            prepared: Pin::file(&prepared_dir.join("prepared.json"))?,
            build: Pin::file(&build_dir.join("build.json"))?,
            executable: p.executable.clone(),
            detail: Pin::file(&out.join("oracle-detail.json"))?,
            passed: true,
            scope: "operator_only".into(),
            full_route_qualified: false,
            promotion_eligible: false,
        };
        experiment::write_json(&out.join("oracle.json"), &receipt)?;
        if receipt_pin.is_none() {
            return Ok(());
        }
        // Repeat the complete oracle for the exact data outside timed intervals.
        // Warmups are per arm; raw timings are write-once and retained on failure.
        for _ in 0..spec.timing.warmups {
            for arm in [Arm::Control, Arm::Candidate] {
                owner.reset(arm)?;
                let d = owner.dispatch(arm, Output::Bf16)?;
                if d.metadata_status != 0 {
                    return Err(Error::new("warmup metadata failure"));
                }
                owner.check(arm, Output::Bf16)?;
            }
        }
        let mut samples = Vec::with_capacity(spec.timing.blocks as usize * 4);
        // No cache readback, host reset, or file I/O BETWEEN samples of a block.
        // Only command allocation/encoding/submit/wait belongs to host latency.
        for block in 0..spec.timing.blocks {
            let block_result = (|| -> Result<()> {
                for arm in [Arm::Control, Arm::Candidate, Arm::Candidate, Arm::Control] {
                    let d = owner.dispatch(arm, Output::Bf16)?;
                    let sample = Sample {
                        index: samples.len() as u32,
                        arm: arm.name().into(),
                        host_ns: d.host_ns,
                        gpu_ns: d.gpu_ns,
                        encoded_dispatches: d.encoded,
                        completed_dispatches: d.completed,
                    };
                    samples.push(sample);
                    if d.metadata_status != 0 {
                        return Err(Error::new("timed metadata refusal"));
                    }
                }
                Ok(())
            })();
            // Persist complete or partial blocks before propagating an error.
            experiment::write_json(
                &out.join(format!("block-{block:03}.json")),
                &samples[block as usize * 4..],
            )?;
            block_result?;
        }
        owner.check(Arm::Control, Output::Bf16)?;
        owner.check(Arm::Candidate, Output::Bf16)?;
        let score = experiment::score(&samples, &spec.timing)?;
        experiment::write_json(&out.join("score.json"), &score)?;
        p.verify()?;
        build.library.verify()?;
        receipt_pin.as_ref().unwrap().verify()?;
        for pin in spec.input_pins() {
            pin.verify()?;
        }
        if !score.drift_passed {
            return Err(Error::new(
                "ABBA baseline drift exceeded predeclared bound; inconclusive",
            ));
        }
        Ok(())
    })();
    if let Err(e) = &result {
        experiment::write_json(
            &out.join("failure.json"),
            &serde_json::json!({
        "error":e.to_string(),"passed":false,"full_route_qualified":false,"promotion_eligible":false}),
        )?;
    }
    result
}
