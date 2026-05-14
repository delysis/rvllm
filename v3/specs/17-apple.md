# 17 — Apple Silicon backend

## Scope

Owns Metal prefill, ANE rollout contracts, Apple device policy, scheduler handoff capsules, and host-testable layout/MIL/weight-blob invariants.

## v2 / CUDA-path problems this must not repeat

- Metadata layouts must be explicit. CUDA graph issues came from silently changing offsets/layouts between prefill and decode.
- No fallback-by-accident. If ANE is requested and private APIs are unavailable, return `RvllmError::Apple`, not Metal/CPU fallback.
- No hidden hot-path allocation. Metal/ANE launch code must preallocate persistent buffers and parameter storage.
- No mixed scheduler phase. Apple backends consume the existing `BatchPlan::{Prefill, Decode}` separation.
- No private FFI in safe crates. Unsafe Objective-C/IOSurface code belongs in a future `rvllm-apple-ane-sys` crate.

## v3 contract

Public crate:

```text
crates/rvllm-apple
  device.rs       AppleAcceleratorTarget, family/tier/NAX policy seeds
  plan.rs         AppleBackendMode, RolloutBucket, AppleRuntimePlan
  handoff.rs      HandoffCapsule with req spans, positions, state handles, layout hash
  iosurface.rs    byte-addressed IOSurface tensor descriptors
  weight_blob.rs  64-byte global + 64-byte chunk FP16 ANE blobs
  mil.rs          dense, fused FFN, fused QKV MIL text generators
  metal.rs        direct-Metal prefill contract
  backend.rs      safe AppleBackend trait and host stub
```

Runtime bridge:

```rust
#[cfg(feature = "apple")]
pub fn handoff_from_prefill_plan(plan: &BatchPlan, kind: HandoffKind) -> Result<HandoffCapsule>;

#[cfg(feature = "apple")]
pub fn handoff_from_decode_plan(plan: &BatchPlan, kind: HandoffKind) -> Result<HandoffCapsule>;
```

## Private API risk

The default Apple crate is safe and host-testable. It may describe ANE rollout
plans, MIL text, IOSurface tensor layouts, and weight blobs, but it must not
link private frameworks or expose unsafe Objective-C entry points. Private ANE
FFI belongs in a future `rvllm-apple-ane-sys` crate consumed through safe
wrappers.

Private ANE execution is a research path with deployment risk:

- Apple does not publish or stabilize the private ANE interfaces this path
  would need; symbols, signing policy, entitlement behavior, compile cache
  layout, and runtime validation may change across macOS releases.
- It is not an App Store or production distribution contract. Treat it as a
  local, explicitly enabled benchmark path until legal, signing, and support
  constraints are resolved outside the engine.
- Private ANE modes require explicit user opt-in (`private_ane_opt_in`) plus
  build/runtime gates (`macOS`, `aarch64`, and a future `private-ane` feature).
- There is no implicit fallback. If private ANE was requested and the private
  path cannot be used, rvLLM must return a typed `RvllmError::Apple` such as
  `AppleError::FeatureNotAvailable`, `AppleError::PrivateApiUnavailable`, or
  `AppleError::AneUnavailable`; it must not silently continue on Metal, MLX,
  CPU, or CUDA.
- CUDA behavior is unchanged. Apple mode selection, private API probing, and
  Apple benchmark gates must not alter CUDA feature flags, kernel catalogs,
  CUDA graph behavior, or CUDA fallback policy.

## Supported modes

These are the only Apple backend modes the planner and runtime bridge may name.
They are mode contracts, not a promise that every hardware backend is already
implemented.

```rust
pub enum AppleBackendMode {
    MetalOnly,
    MlxPrototype,
    MetalPrefillMetalDecode,
    MetalPrefillAneFfnRollout,
    MetalPrefillAneRolloutExperimental,
}
```

- `MetalOnly`: direct Metal-only contract for Apple execution. It must use
  explicit metallib/pipeline bring-up, persistent parameter buffers, and no
  hot-path allocation. It never touches private ANE APIs.
- `MlxPrototype`: MLX-based prototype path for bring-up and shape validation.
  It may allocate in the hot path and must not be used to report production
  throughput or energy claims.
- `MetalPrefillMetalDecode`: non-private mode where Metal owns prefill, decode,
  and KV-cache writes through the standard scheduler handoff. This is the
  preferred supported mode for public Apple execution.
- `MetalPrefillAneFfnRollout`: Metal handles prefill, then private ANE executes
  static-bucket FFN rollout plus LM-head work. It requires `private_ane_opt_in`
  and a selected `RolloutBucket`.
- `MetalPrefillAneRolloutExperimental`: private ANE research mode for a wider
  rollout program, including fused QKV/FFN/LM-head procedures. It has the same
  opt-in and bucket requirements as `MetalPrefillAneFfnRollout`, plus additional
  parity and stability risk.

## Invariants

- `HandoffCapsule::validate()` must pass before launch.
- `cu_seqlens[0] == 0` and `cu_seqlens.last() == tokens_flat.len()`.
- `positions.len() == context_lens.len() == req_ids.len()`.
- Rollout bucket selection minimizes padding waste among static buckets.
- Direct Metal contract requires one command buffer per layer group and no hot-path allocation.
- ANE MIL weights use explicit BLOBFILE offsets from tested descriptors.
- Private ANE requires explicit opt-in and macOS/aarch64 cfg.

## Failure modes

All Apple failures should use `RvllmError::Apple` with `AppleCtx`; do not add
stringly typed errors or fallback-only log messages.

- `AppleError::MetalUnavailable`, `AppleError::MetallibMissing`, and
  `AppleError::PipelineMissing` cover explicit Metal bring-up failures.
- `AppleError::AneUnavailable` means the machine or OS cannot provide the
  requested ANE path.
- `AppleError::PrivateApiUnavailable` means a required private symbol, selector,
  service, or framework entry point is absent after the user explicitly opted in.
- `AppleError::MilCompileFailed`, `AppleError::InvalidMil`, and
  `AppleError::InvalidWeightBlob` cover generated model artifacts that cannot be
  compiled or validated.
- `AppleError::IoSurfaceFailed` covers IOSurface allocation, descriptor, or
  sharing failures.
- `AppleError::ShapeBucketMissing` means the requested rollout shape has no
  static bucket; do not choose a larger unvetted path outside `ROLLOUT_BUCKETS`.
- `AppleError::HandoffMalformed` means scheduler metadata, token spans,
  position vectors, state handles, surfaces, or layout hashes are inconsistent.
- `AppleError::NotPrepared` means a backend launch or collect was attempted
  before `prepare()`.
- `AppleError::FeatureNotAvailable` means the requested mode or operation is
  intentionally unavailable in the current build or configuration.
- `AppleError::UnsupportedDevice` is for known device-policy rejection, not a
  generic catch-all for failed probing.

## Benchmark interpretation

Apple benchmark numbers must identify the mode, device, OS, model artifact,
batch/sequence bucket, precision, prompt length, decode length, and whether
private ANE was enabled. A benchmark that does not disclose these fields is only
bring-up evidence.

- Report TTFT separately from decode tokens/sec. Metal prefill work, private ANE
  compile/cache warmup, and first IOSurface setup can dominate TTFT while steady
  rollout throughput looks healthy.
- Report p50/p95/p99 latency with decode tokens/sec. Static rollout buckets can
  improve average throughput while adding padding waste or tail latency.
- Report energy and wall power when comparing Apple modes. ANE offload can look
  slower on raw tokens/sec but still matter if joules/token improves.
- Compare `MlxPrototype` only against other prototypes. Its hot-path allocation
  and framework overhead make it unsuitable for production claims.
- Compare private ANE numbers only against the matching non-private
  `MetalPrefillMetalDecode` baseline on the same machine. If private API probing
  fails, the result is a typed failure, not a slower Metal run.
- Do not compare Apple benchmark output directly to CUDA H100/H200 throughput
  without labeling it as cross-device context. CUDA remains the reference path
  for existing GPU benchmark gates, and Apple work must not change those gates.
- Perplexity/parity must accompany throughput claims. A fast Apple path with
  invalid handoff metadata, stale layout hashes, or missing KV-cache ownership is
  a correctness failure even if decode tokens/sec is high.

## Test plan

Host tests:

- Apple device parser: M1/M2/M3/M4/M5, Base/Pro/Max/Ultra, and `M10` false-positive guard.
- Runtime bridge: prefill/decode `BatchPlan` -> well-formed `HandoffCapsule`.
- Bucket selector: minimal padding waste and missing large shapes.
- IOSurface descriptor byte size and packed single-input layout.
- Weight blob header, chunk descriptors, FP16 conversion, BLOBFILE offsets.
- MIL generator names, shapes, and offsets.
- Stub backend prepare/launch/collect lifecycle.

Hardware tests, ignored by default:

- Metal device/context and metallib load.
- Direct Metal RMSNorm/matmul/RoPE/attention parity.
- Private ANE dense projection smoke.
- Private ANE fused FFN smoke.
- ANE compile cache reload.
- End-to-end Qwen3 dense prompt/generate parity.

## Cross-cutting deps

- 02-config: Apple backend mode should eventually be validated by `RuntimeConfigBuilder`.
- 04-memory / 05-concurrency: Apple buffers/surfaces must not allocate or alias inside captured launch regions.
- 07-scheduler: Apple consumes `BatchPlan`; it must not fork scheduling logic.
- 08-metadata: layout hash should eventually use rvllm-metadata hash, not the temporary host-test hash.
- 09-layer: Metal/ANE layer contracts should mirror `LayerDims`/`LayerPhase`.
- 15-validation: parity, perplexity, golden traces, and energy regression gates.
