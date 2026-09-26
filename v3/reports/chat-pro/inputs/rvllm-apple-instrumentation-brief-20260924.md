# rvLLM Apple instrumentation implementation brief

## Objective

Implement professional-grade, opt-in instrumentation for rvLLM's Gemma 4
Metal and ANE routes. It must make an Xcode GPU capture and stage-level
diagnosis at least as usable as MLX's debug workflow without putting clocks,
allocation, I/O, locks, string formatting, command-buffer waits, or per-dispatch
dynamic bookkeeping on the production hot path when instrumentation is off.

Remain in ordinary Chat; do not switch to Work. Return a reviewable source
archive or unified patch plus a short design/evidence report. Do not claim that
host component tests qualify Apple hardware behavior.

## Exact source identity

- Repository: `delysis/rvllm`
- Branch: `astra/gemma4-load4-tiles-20260923`
- Base commit for this task: `55ec46252de8f7dfa8bfe467154c2633f61e3186`
- Workspace root inside the attached archive: `v3/`
- Use safe idiomatic Rust. Keep Objective-C/Metal unsafety in narrow modules
  with explicit safety comments and fail-closed validation.

## What already exists and must be reused

Do not build a parallel profiling stack without integrating these facilities:

1. `rvllm-runtime/src/apple_metal_backend.rs`
   - `MetalProbePerfCounters` and `MetalProbePerfStats` already record logical
     steps/tokens, library and PSO compilation, command buffers, encoder-class
     counts, forced waits, CPU wall/encode/wait time, and the completed command
     buffer's `GPUStartTime`/`GPUEndTime`.
   - GPU timestamps are read only after normal collection; preserve the
     asynchronous submission model and introduce no profiler-only wait.
2. `rvllm-runtime/src/bin/rvllm_metal_infer.rs`
   - Existing session profile output aggregates repeated real-route runs.
3. `rvllm-apple-metal/src/layer_forward.rs` and
   `rvllm-apple-metal/src/gemma4_model.rs`
   - `RVLLM_METAL_DEBUG_TRACE_LAYER` already enables expensive tensor snapshot
     copies. Trace mode deliberately disables some fusions. Keep this separate
     from performance instrumentation and state that traced execution is not
     performance representative.
   - A few research encoders already have Metal labels, but they currently
     construct `NSString` labels inline on selected dispatch paths. Replace
     that enabled-path allocation with prepared labels; consolidate rather
     than duplicating them.
4. `rvllm-apple-metal/src/kernels.rs`
   - Metal source is generated in-process and pipeline names are centrally
     known. Numeric ABI fingerprints already include shader source and selected
     options.
5. `rvllm-apple-ane-sys/src/diagnostic_journal.rs`
   - The current opt-in ANE journal synchronously `sync_all()`s every record.
     It is crash/evidence instrumentation and explicitly incompatible with
     timing. Preserve its semantics; do not use it for benchmarks.
6. `rvllm-runtime/src/bin/rvllm_ane_probe.rs`
   - Direct dynamic-FFN profiling already separates input/evaluate/output wall
     time through `project_profiled`. Integrate its taxonomy where applicable.
7. `v3/tools/apple_profile_gate.py`, the kernel-game queue/referee, and current
   receipt identity rules must remain authoritative for timing/adjudication.
8. `rvllm-apple-metal/src/research_evidence.rs` and `pipeline.rs`
   - Candidate dispatch evidence uses relaxed atomics and quiescent snapshots.
     Preserve its evidence semantics, but do not mistake selection counts for
     completion or make it the general performance event system.

The current ANE full-route implementation also takes unconditional `Instant`
timestamps around several logical stages. Preserve receipt compatibility and
measure their cost before changing them; they are host-wall observations, not
ANE device counters.

## MLX behavior worth matching

MLX does two deliberately distinct things:

- Isolated wall timing uses five warmups and 100 `mx.eval`-synchronized
  iterations with pre-materialized inputs.
- Debug builds record Metal source and label objects, while explicit
  `start_capture(path)` / `stop_capture()` brackets emit `.gputrace` for Xcode.

MLX exposes no public per-kernel numeric GPU timer or counter API. Xcode trace
analysis supplies per-kernel durations/counters; synchronized host timing is a
separate evidence class. Preserve that distinction in rvLLM.

## Required implementation

### 1. Compile-time and runtime cost boundary

Add one clearly named Cargo feature for detailed Apple instrumentation,
disabled by default in shipping builds (for example `metal-instrumentation`,
with any private ANE feature remaining separately gated). When the feature is absent, detailed
recording calls must compile to inline no-ops and add no fields to production
backend state. When the feature is present but inactive, permit at most one
cheap predictable activation check per model step or command buffer—not per
element and preferably not per encoder. Do not read environment variables in
the step loop.

Parse and validate all configuration once during backend/profiler creation.
Enabled recording must use preallocated bounded storage with static stage IDs;
no formatting, heap growth, file I/O, mutex, channel send, JSON serialization,
or wall-clock syscall in encoder dispatch. Export only after synchronization
already required by normal result collection, or on explicit profiler finish.
Overflow must be counted and reported, never silently overwrite favorable
samples or block inference.

Provide evidence for both boundaries:

- structural tests proving the disabled recorder is zero-sized/no-output and
  cannot activate;
- allocation-count tests showing inactive and steady enabled dispatch recording
  allocate zero times after setup;
- an ABBA normal-route overhead check comparing a shipping/default build with
  an instrumentation-feature build left inactive. Preserve raw trials and host
  state; do not impose an unrealistically brittle CI timing threshold. Define
  <=1% median overhead as the design target and >3% as an investigation gate.

### 2. Metal Xcode capture

Using the existing `objc2-metal` dependency, add a narrow macOS-only RAII
wrapper around `MTLCaptureManager`/`MTLCaptureDescriptor` with
`GPUTraceDocument` destination. It must:

- require an explicit `.gputrace` path and refuse any pre-existing target;
- require capture support and surface the native error;
- capture the real device or command queue used by the prepared backend;
- have explicit start/finish and a Drop safety stop without turning Drop into
  success evidence;
- never begin during model load unless explicitly requested;
- bracket a bounded requested prefill/decode window;
- record whether stop completed and hash the final trace directory outside the
  hot path;
- label this artifact generated-source/dispatch/diagnostic evidence, not valid
  comparative timing or promotion evidence.

Expose capture through an explicit `rvllm_metal_infer` CLI surface suitable for
the experiment queue. Do not make a process environment variable the sole API.

### 3. Stable Metal labels and artifact manifest

When detailed instrumentation/capture is active, label command queues, command
buffers, and every important compute encoder with stable bounded names. The
label vocabulary must include phase, layer when applicable, logical stage, and
selected implementation/pipeline without formatting strings in the hot loop.
Precompute label objects during preparation.

Use at least this logical stage taxonomy:

- embedding / per-layer embedding
- input RMSNorm
- QKV projection (or separate Q/K/V)
- Q norm, K norm, RoPE, KV write
- sliding/full attention
- O projection
- post-attention RMSNorm/residual
- pre-FFN RMSNorm
- gate/up projection and activation
- down projection
- post-FFN RMSNorm/residual
- final RMSNorm
- LM head
- argmax/sampling

Record phase (prefill/decode), layer, attention kind, tensor dimensions,
compute dtype, weight format/bit depth, candidate/pipeline function name,
threadgroup/grid shape, threadgroup memory, and relevant fusion state.

Add an explicit artifact-dump operation performed outside inference that writes
the exact generated MSL, its SHA-256, source options, pipeline/function catalog,
numeric ABI fingerprint, executable SHA, model/config hashes, and any available
metallib/library identity. Never claim a metallib was extracted if the runtime
API does not provide it; the Xcode trace may be the binary/pipeline evidence.

### 4. Numeric event stream without forced waits

Extend the existing Metal probe rather than replace it. A bounded event record
should correlate normal collection with command buffer and stage metadata.
Command-buffer GPU start/end timestamps may be recorded after completion.
Do not pretend they are per-encoder durations when multiple encoders share a
command buffer. If Metal counter sample buffers are added, make them a separate
explicit high-overhead mode with capability checks and a perturbation warning;
they are not required for the first patch.

The exported strict JSON must distinguish:

- CPU submission/encoding time;
- time blocked waiting for an incomplete command buffer;
- completed whole-command-buffer GPU duration;
- Xcode-capture-only per-encoder/kernel information;
- unmeasured metrics.

Do not use command-buffer completion callbacks for export or I/O. Append the
event from the existing owner-thread collection boundary after completion.

It must preserve ordered events, dropped-event count, capture boundaries,
compile deltas, and exact executable/source/model/workload identities.

### 5. ANE instrumentation

Do not promise private per-kernel ANE counters that the current API cannot
provide. Add a separate in-memory, bounded, opt-in timing/event recorder around
the existing compile/load/request/input/evaluate/output/handoff boundaries.
Use preassigned stage IDs and preallocated storage; export after the run. Keep
the durable fsync journal available as a separate crash-forensics mode and mark
any run using it timing-ineligible.

At minimum report compile/load, cache hit/miss, input preparation/copy,
evaluation, output copy/validation, Metal-to-ANE handoff, and fallback/route
decisions. Include graph/cache identity, weight format, layer range, token
position/bucket, compiler-call delta, execution count, and whether the metric is
host wall time or an accelerator-native duration. Unknown native duration must
be explicit `Unmeasured`, not inferred from wall time.

### 6. Tests and delivery

Required tests:

- feature-off and feature-on/inactive behavior;
- no post-setup allocation in record paths;
- bounded overflow and ordered export;
- exclusive/non-overwriting capture paths and failure cleanup;
- no added waits and unchanged output bits with instrumentation active;
- stable label/taxonomy coverage for the Gemma 4 route;
- strict duplicate-key-rejecting JSON receipts;
- macOS compile/link test for capture APIs;
- ANE fsync journal makes timing eligibility false;
- Linux/default workspace remains buildable without Apple frameworks.

Return only a coherent vertical slice that compiles and is wired to the real
Gemma 4 route. Do not submit placeholder traits, mock-only capture, a parallel
toy backend, or tests that merely inspect source strings. Include exact commands
run and clearly separate host-only validation from unrun Apple-hardware gates.

## Non-goals and invariants

- Do not change production kernel selection, scheduling, numerical boundaries,
  quantization, default routes, or promotion state.
- Do not make tensor snapshots part of performance measurement.
- Do not add a background uploader, network service, telemetry, or hosted
  fallback.
- Do not log prompts, tensor contents, weights, credentials, or model data.
- Do not weaken existing evidence gates or treat a trace as timing proof.
