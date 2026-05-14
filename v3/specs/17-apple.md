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
  mlx.rs          optional non-production MLX parity-oracle invocation scaffold
  backend.rs      safe AppleBackend trait and host stub
```

Unsafe private ANE sys crate:

```text
crates/rvllm-apple-ane-sys
  lib.rs          typed error boundary and macOS/aarch64/private-ane gate
  objc.rs         Objective-C runtime FFI symbols
  tests/smoke.rs  ignored runtime/private ANE probes
```

Runtime bridge:

```rust
#[cfg(feature = "apple")]
pub fn handoff_from_prefill_plan(plan: &BatchPlan, kind: HandoffKind) -> Result<HandoffCapsule>;

#[cfg(feature = "apple")]
pub fn handoff_from_decode_plan(plan: &BatchPlan, kind: HandoffKind) -> Result<HandoffCapsule>;
```

Backend modes:

```rust
pub enum AppleBackendMode {
    MetalOnly,
    MlxPrototype,
    MetalPrefillMetalDecode,
    MetalPrefillAneFfnRollout,
    MetalPrefillAneRolloutExperimental,
}
```

## Invariants

- `HandoffCapsule::validate()` must pass before launch.
- `cu_seqlens[0] == 0` and `cu_seqlens.last() == tokens_flat.len()`.
- `positions.len() == context_lens.len() == req_ids.len()`.
- Rollout bucket selection minimizes padding waste among static buckets.
- Direct Metal contract requires one command buffer per layer group and no hot-path allocation.
- MLX reference mode is prototype-only, feature-gated, and never implements `AppleBackend` or fallback routing.
- ANE MIL weights use explicit BLOBFILE offsets from tested descriptors.
- Private ANE requires explicit opt-in and macOS/aarch64 cfg.

## Failure modes

- `MetalUnavailable`, `MetallibMissing`, `PipelineMissing` for Metal bring-up.
- `AneUnavailable`, `PrivateApiUnavailable`, `MilCompileFailed`, `IoSurfaceFailed` for ANE bring-up/eval.
- `ShapeBucketMissing` for unsupported rollout shape.
- `HandoffMalformed` for scheduler/metadata mismatch.
- `FeatureNotAvailable` for requested experimental paths.

## Test plan

Host tests:

- Apple device parser: M1/M2/M3/M4/M5, Base/Pro/Max/Ultra, and `M10` false-positive guard.
- Runtime bridge: prefill/decode `BatchPlan` -> well-formed `HandoffCapsule`.
- Bucket selector: minimal padding waste and missing large shapes.
- IOSurface descriptor byte size and packed single-input layout.
- Weight blob header, chunk descriptors, FP16 conversion, BLOBFILE offsets.
- MIL generator names, shapes, and offsets.
- MLX parity scaffold: valid handoff cases, explicit executor requirement, and planned-only invocation.
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
