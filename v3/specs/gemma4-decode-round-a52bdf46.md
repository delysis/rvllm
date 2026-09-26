# Gemma 4 exact-source decode round

Base: `a52bdf46627219561efa6610e88ffd5c37a821d9`, as declared for the supplied source subset. Status: **implemented research source; not Rust-compiled, Metal-compiled, device-qualified, benchmarked, or promoted in this environment**. No production selector/default or checkpoint loader was changed.

## Boundaries and implementation map

| Arm / selector | Admission and implementation | Native control |
|---|---|---|
| `metal-ffn-bf16-r4-sg2` | M=1, K=3840, I=15360; native BF16 Gate||Up `[30720,3840]`; 64 threads, 1,920 groups; two SIMDgroups, four output rows each; one encoded dispatch. | Existing native Gate||Up GEMM then GELU, two encoders. |
| `metal-qmv-w4-g32-r8-sg2` | W4 **DenseDownProjection**, M=1, N=3840, K=15360; 64 threads, 240 groups; each SIMDgroup owns eight rows. | Existing native-BF16 group-32 n4 schedule. |
| `metal-qmv-w8-g32-r8-sg2` | W8 **OutputProjection**, M=1, N=3840, K=4096 or 8192; same r8/two-SIMDgroup mapping. | Existing native-BF16 group-32 n4 schedule. |
| `metal-global-d512-short-r4t128` | M=1, D=512, 16 Q heads, 1 KV head, global window, scale=1; four groups of 128 threads, 2,048 bytes shared, zero global scratch. | Existing paged native global decode control in the device referee. |

All four use the existing Apple9, typed-pipeline, queried execution-width / maximum-thread / shared-memory checks. The existing policy is deliberately not broadened to another GPU family. `MetalKernelOptions::default()` remains Off. The selector is an explicit research policy, not an automatic shape-only promotion.

### BF16 fusion

`research_decode.rs` owns a pure checked plan; `research_decode_metal.rs` binds that plan to the immutable pipeline cache and arena. `layer_forward.rs` calls this adapter at the existing FFN gate/up site. Gate and up are accumulated in FP32, **rounded to BF16 before GELU**, and the activated product is rounded once to BF16. This retains the materialized projection boundary and the incumbent GELU saturation thresholds rather than silently adopting the prototype's different arithmetic.

Wrong model/phase/type, quantized accumulation, unsupported attention shape, any low-bit gate/up descriptor, requested gate/up trace capture, absent/untyped/incompatible PSO, short/misaligned/overflowed spans, or an output alias all refuse before encoder creation. Refusal executes the existing path. An encoder creation failure is an error, not an implicit retry. The native Gate||Up storage remains authoritative; neither weights nor intermediate storage are repacked.

### Group-32 QMV

`low_bit_metal.rs::try_encode_strided_bf16_r8_sg2` reuses the checked `MetalLowBitProjectionOffsets` constructor and common dispatch ABI. Authentication is still the existing checkpoint loader's job; an offset descriptor or a synthetic oracle alone does **not** authenticate a checkpoint. The loader and its authentication code are unchanged.

W4 remains signed two's-complement nibbles in the existing low/high order; W8 remains signed bytes. Both use the same row-major group-32 **FP16 scales**, applied before FP32 FMA. There is no affine group-64 conversion, bias, new zero point, transposed package, repacker, or dense dequantized weight allocation. The additive shader source is not subjected to the generic half-to-bfloat rewrite.

This first port is explicitly BF16 activation/output only. Its role, precision and K contracts are independent checks. The layer-forward fallback for a selected target role uses the existing BF16 n4 route rather than reinterpreting BF16 operands as the legacy F16 ABI. Other selectors and default routing remain unchanged. Prefill and unsupported M values cannot select r8.

### Paged short attention

The candidate plugs into `DecodePlan`, `DecodeBuffers`, the existing global decode adapter, and existing layer-forward routing. It reads the owner-selected K/V arena and GPU metadata; it does not read live metadata on the CPU, allocate a parallel cache, perform a second newest-KV insertion, or mutate any cache state. The preceding owning-layer cache-write encoder remains responsible for making normalized/transformed newest K/V available. Shared-KV consumers continue using the existing owner's offsets.

Negative page IDs are holes. Invalid positive IDs in the visible prefix refuse uniformly before output writes. Invalid or NaN-poisoned speculative suffix data beyond position are ignored. Position-bounded visibility handles restored prefixes and rollback without clearing the cache. All-hole output is zero. The per-head reduction retains the existing fixed-64 FP32 QK association, with a broadcast of the lane-zero reduction result before online softmax/PV.

**Deliberate first-round capacity limit:** `max_blocks * block_size <= 512`, checked on the host before dispatch. This is a bound on logical capacity, not merely live length. A long-capacity cache with live L=256 falls back; L=512 with seven-token pages (capacity 518) also falls back. This prevents a host-accepted dispatch from silently doing no work because a shader-only live-length guard refused it. Supporting short prefixes inside larger logical caches needs an explicit owner-provided bounded-length admission contract and is not implemented here.

## Evidence, guards, allocation and accounting

The dispatch schema advances append-only from v3 to `rvllm.metal.research-dispatch.v4`: old 56 slots are unchanged; new slots 56–59 match the table order. Exact delta validation rejects missing, extra, reset, or overflowed counts. It measures **encoded dispatches**, not GPU completion. QMV also increments the existing format/role ledger. Malformed shader launches are counted as encoded work even when their uniform guard writes nothing; passive host refusals are not counted.

The fused path encodes one operation instead of two. `supports_research_bf16_gate` and `gate_up_encoder_count` expose the same eligibility and arithmetic for external total-encoder consumers. The external runtime aggregate-count consumer is not present in this subset and was **not patched**. Native timing records actual compute-encoder counts separately from logical operator iterations.

New admission and binding paths have no heap collections, string construction, environment reads, logging, pipeline compilation, scratch-region creation or buffer allocation. Native repeated-use tests compare arena allocation cursor, region count and buffer identity. These are source and persistent-arena invariants, not a claim that Apple's command-encoder internals allocate nothing, nor a measured global Rust allocator trace.

Pure Rust tests cover exact shape and refusal boundaries, the ledger, BF16 rounding, signed nibble/byte endpoints, group-32 boundaries/tails, FP64 references and paged visibility. `research_decode_device_tests` reuses the existing source-bound Setup and native receipt owner rather than introducing a second queue/referee engine. It contains full-shape sparse BF16 fusion, dense group-32 QMV for all three K cells, independent FP64 expectations, actual incumbents, three repeated bit-exact uses, immutable inputs, leading/trailing guards, four host refusal cases, and three malformed-launch guard cases per cell. These are implemented tests, **not passed device receipts**.

`attention_global_decode_device_tests` is extended for L=256/512, noncontiguous pages, newest owner payload, holes, poisoned future data, rollback, corrupt visible metadata, capacity refusal and persistent reuse. The tests prepare owner pages; they do not newly qualify a real model's producer/consumer cache-write ordering. The live layer/shared-owner route still needs device integration evidence.

## Existing queue and artifact hooks

`tools/decode-round/campaign-plan.json` declares five compile jobs, four correctness jobs, four operator timing cells, and attention L=256 then L=512 cells. It is an **unbound plan**, not a fake submittable job. The extended `rvllm-global-decode-jobs` generator creates real `rvllm.experiment_job.v1` manifests on the Mac, after hashing real executable/compiler/source identities. Timing manifests cannot be generated without successful unchanged-input compile/correctness results and matching native output pins.

Explicit `prepare ... CANDIDATE...` keeps the legacy implicit campaign unchanged. Projection cells use LENGTH=0 plus OPERATOR_K, not fictitious KV lengths. New cells use ten alternating ABBA/BAAB blocks, 40 retained samples, 100 complete operator iterations/sample, five warmups/arm, exact dispatch counts and a fixed 5% baseline drift bound. FFN baseline samples encode 200 kernels against 100 fused kernels. GPU timestamps, not queue wall time, determine operator timing.

The retainer validates every block/position/order/work count and preserves complete failed-drift attempts as invalid. Reports retain all ratios, distinguish both order strata, and use deterministic whole-block bootstrap diagnostics; they never mark promotion. The new family cannot use automatic advancement. L=512 is an explicit, separately reviewed request after the L=256 screen; there is no retry-on-loss or sample pruning.

Generated-source export uses the existing catalog/source binary. Compile receipts additionally retain AIR identity. The existing public artifact evidence tool captures compiler output and PSO properties. None of these prove instruction selection, registers, spills, occupancy or performance without actual compiler/device artifacts.

## Remaining work and exclusions

This patch does not include Rust/rustfmt/Clippy results, newly compiled AIR/metallib, native receipts, real-checkpoint layer-0 or logits/perplexity gates, full-route timing, a new allocator interception test, or an external runtime total-count repair. Those could not be executed or, for missing external code, safely source-bound here. The provided source archive omits required workspace crates and some existing archived JSON test fixtures; no stubs or replacement sources were fabricated.

The existing real-weight projection binary is retained unchanged; it does not gain a new `--candidate` spelling in this round. Real-weight r8 qualification should invoke the new adapter on the authenticated descriptor through a full-checkout integration gate before any timing claim. The new operator referee is synthetic and expressly emits `checkpoint_qualified:false`.

The attached September 25 M4 Max receipts qualify the earlier group-64/contiguous prototypes only. They remain unchanged provenance, not evidence that these new group-32 or paged ports work. All source/queue statuses remain proposed and unqualified.
