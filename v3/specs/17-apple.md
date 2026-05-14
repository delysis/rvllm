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

The ANE rollout path depends on private Apple behavior: unpublished symbols,
MIL procedure details, BLOBFILE offsets, IOSurface layout rules, and per-OS
compiler/cache behavior. Those contracts can change across macOS releases or
devices, and may carry distribution or entitlement risk outside local research
builds.

Private ANE use is therefore never implicit. `AppleRuntimePlan::validate()`
requires `private_ane_opt_in` for every mode where
`AppleBackendMode::requires_private_ane()` is true. If ANE was requested and a
symbol, compiler, surface, or shape is unavailable, rvLLM must return a typed
`RvllmError::Apple` such as `PrivateApiUnavailable`, `AneUnavailable`,
`MilCompileFailed`, `IoSurfaceFailed`, `ShapeBucketMissing`, or
`FeatureNotAvailable`. It must not silently reroute to Metal, MLX, CPU, or CUDA.

The safe default build remains host-testable and free of private FFI. Unsafe
Objective-C/private-ANE bindings belong in a separate sys crate, behind explicit
platform cfg and opt-in policy.

## Supported modes

```rust
pub enum AppleBackendMode {
    MetalOnly,
    MlxPrototype,
    MetalPrefillMetalDecode,
    MetalPrefillAneFfnRollout,
    MetalPrefillAneRolloutExperimental,
}
```

Mode contracts:

- `MetalOnly`: public Metal-only execution. No ANE/private API dependency.
- `MlxPrototype`: MLX-based bring-up/prototype path. Useful for shape and parity
  exploration, not for shipping performance claims.
- `MetalPrefillMetalDecode`: public Metal prefill and public Metal decode using
  the scheduler handoff capsule. This is the private-free Apple baseline.
- `MetalPrefillAneFfnRollout`: Metal prefill plus private ANE FFN/lm-head
  rollout for a selected static `RolloutBucket`. Requires explicit opt-in.
- `MetalPrefillAneRolloutExperimental`: Metal prefill plus broader private ANE
  rollout, including QKV/FFN/lm-head procedures. Requires explicit opt-in and is
  expected to fail closed while coverage is incomplete.

Mode selection is a contract, not a preference list. If the selected mode cannot
prepare or launch, initialization/benchmarking fails with the typed Apple error
instead of trying another backend.

## Invariants

- `HandoffCapsule::validate()` must pass before launch.
- `cu_seqlens[0] == 0` and `cu_seqlens.last() == tokens_flat.len()`.
- `positions.len() == context_lens.len() == req_ids.len()`.
- Rollout bucket selection minimizes padding waste among static buckets.
- Direct Metal contract requires one command buffer per layer group and no hot-path allocation.
- ANE MIL weights use explicit BLOBFILE offsets from tested descriptors.
- Private ANE requires explicit opt-in and macOS/aarch64 cfg.

## Failure modes

All Apple failures should cross crate boundaries as `RvllmError::Apple` with an
`AppleCtx` naming backend, operation, and device.

- Metal bring-up: `MetalUnavailable`, `MetallibMissing`, `PipelineMissing`.
- Private ANE bring-up/eval: `AneUnavailable`,
  `PrivateApiUnavailable { symbol }`, `MilCompileFailed`, `IoSurfaceFailed`.
- Planning and shape policy: `ShapeBucketMissing` when no static
  `RolloutBucket` covers the requested rollout.
- Scheduler handoff: `HandoffMalformed` for req/span/position/layout mismatch.
- Backend lifecycle: `NotPrepared` when launch/collect runs before prepare.
- Policy and incomplete coverage: `FeatureNotAvailable`, `UnsupportedDevice`,
  `InvalidMil`, `InvalidWeightBlob`.

These are hard failures for the selected mode. They are not signals to retry on
another Apple path or fall back to CUDA behavior.

## Benchmark interpretation

Apple benchmark numbers are mode-specific. Always report the
`AppleBackendMode`, device name/tier, macOS build, model artifact, prompt length,
output length, batch size, selected `RolloutBucket`, and whether compile/cache
warmup was included.

Interpret results by path:

- `MlxPrototype` numbers are bring-up data only. They should not be compared to
  direct Metal, private ANE, CUDA, or vLLM as throughput claims.
- `MetalOnly` and `MetalPrefillMetalDecode` are the public Apple baseline.
- Private ANE modes measure a risk-bearing experimental path. A failed private
  API check means "unsupported on this host/config", not a performance
  regression.
- Static rollout buckets introduce padding waste. Tokens/sec should be read
  alongside bucket capacity and requested `(seqs, tokens)`.
- First-run ANE compile/cache time and steady-state launch time are separate
  metrics. Do not merge them unless the benchmark labels the run as cold-start.

Because mode selection fails closed, a successful ANE benchmark is evidence that
the ANE path actually prepared and launched. If private API setup fails, the
benchmark must fail rather than publishing Metal numbers under an ANE label.

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
