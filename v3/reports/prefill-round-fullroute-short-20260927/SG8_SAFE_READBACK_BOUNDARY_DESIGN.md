# SG8 same-input readback boundary: current-tree design

Status: source-only design at PR #8 commit `e6e24d1eaa147495fd3a198ea7fb28df177472e5`. No capture implementation, device job, or numerical verdict is authorized by this document. New development remains safe idiomatic Rust; this design does not import Astra's patch or add an unsafe operation.

## Evidence and limits

The Astra `sg8-first-difference-85ee8032-bundle.zip` is a review input, not an apply script. Its ZIP SHA-256 is `2f9f8b172a6cb8b65847787d3e7c98f4100ce55810a673e1c718d1a13c006a17`; its incremental patch SHA-256 is `9f49fe757b4b4e60e9300efd10d732c81f24dc1abdbfe24167648d7b3d9a13d5`. It targets base `85ee80328f097bdc9fa50ded3274594931f49819` plus a separate earlier patch, not the current PR #8 tree. Astra reported source checks only: no Rust build, tests, Metal compilation, or device run. The proposal adds an unsafe blit encoding call and an unsafe completed-buffer slice. Neither is admitted here. Its useful contributions are a complete boundary inventory, exact descriptor and identity checks, ticket ownership, logical-KV metadata, explicit unavailable states for fused interiors, route-dispatch prefixes, and bounded create-new receipts.

The current route-trace-12b-03 first *observed* one BF16 ULP difference in layer 5 attention output, with an equal stored residual. It captured Q/K/V only as hashes/samples and KV as physical prefixes. That is not the first arithmetic error. A source-level comparison of HF and Metal final stages likewise does not establish an intermediate cause.

Current-tree readback capabilities are narrower:

- `rvllm-apple-metal/src/arena.rs`: `MetalBufferArena` has unsafe `host_ptr` and `write_region`, but no safe owned read or snapshot-copy API. Its shared buffer is reused across operations.
- `rvllm-runtime/src/apple_metal_backend.rs`: `ModelGpuSubmission::finish` checks `MTLCommandBufferStatus::Completed` before reading output, and `KvPageIo::capture_page` is a safe caller-facing method after all arena-owning submissions have been collected. The implementation contains an existing audited unsafe pointer read; its return is exact native two-byte K/V page data, not active Q or a layer residual.
- `rvllm-apple-metal/src/layer_forward.rs`: the optional legacy trace changes route selection (`trace.is_none()` participates in prefill MMA and fused QKV decisions). It cannot certify route-preserving Q capture. `probe_read_decode_logits_f32` returns final logits only.

There is therefore **no current safe route-preserving producer of the full active Q or transient layer boundary bytes**. A safe contract module alone cannot make those bytes available. Do not disguise an unsafe arena read or a recomputed Q as a safe capture.

## Minimal experiment, once transport exists

Start with one prospectively declared, distinct prompt and one selected decode ordinal, layers 4–6, both selector-off and SG8, on the same checkpoint/input/teacher-forced history. This bounds the first implementation and directly tests the prior observed neighborhood. It may identify a first stored difference *within this captured span*, never globally. A full 48-layer/64-step run needs its own explicit storage and memory budget; Astra's unexecuted source estimate for P512/64 is about 16.8–16.9 GB per arm and roughly 5.23 GB for one P512 prefill frame, before unrelated model memory. Do not infer run safety from the proposed 6/32 GiB caps alone.

Record a frame with exact model, config, tokenizer, metallib, executable, source, prompt and plan SHA-256 identities; route; command-buffer ticket; phase and logical next-token ordinal; input IDs; position; context length; execution slot; layer geometry; and the actual materialized block table and slot mapping. The native BF16 dense path and selected donor SG8 dispatch must be named and observed. `generated_token_ids` are teacher-forced transport, not greedy agreement.

For each selected layer, declare descriptors before encoding, including shape, dtype, byte order, row layout, source producer, valid logical length, and expected byte count. The first capture set is input residual, normalized input, Q after RoPE, K after RoPE, normalized V, logical attended K/V, attention output, attention residual, FFN branch and output residual. Include the final step residual. Fused `packed_qkv` and `gate_up` interiors are `unavailable` unless the production route materializes them; never run a substitute kernel to fill them. Metadata must distinguish a shared-KV consumer layer from its K/V producer layer.

For attended K/V, validate the **materialized** logical-to-physical table and valid length, then gather only live rows in logical order from exact native page bytes. Reject missing, duplicate, negative, out-of-range, or stale page entries and mismatched context/slot mapping. `KvPageIo::capture_page` is suitable for post-collection page bytes but cannot by itself reconstruct the Q consumed earlier in the submitted command. The capture plan must specify which source is obtained during encoding and which can be assembled after collection.

## Safe API boundary and ownership

The desired caller-facing API is a default-off, feature-gated `CapturePlan` plus `CaptureTicket`: `begin` validates identities, descriptors and byte budgets; `record` accepts only declared producer spans in exact order; `submit` binds the immutable plan and snapshot storage to one command-buffer ticket; `collect` returns owned bytes only after that ticket reports `Completed`, or returns an error with no partial success. The API must not expose raw Metal pointers, borrowed host slices, or unchecked offsets to the referee. Captures are copied before scratch reuse; the snapshot and source buffer stay alive through collection. One live sequence/slot is required initially. Failed GPU status, wrong ticket, missing or duplicate boundary, allocation failure, offset overflow, incomplete copy, or later scratch reuse before copy must fail closed.

This is an *interface design*, not an implementation claim. Encoding a Metal blit and turning `MTLBuffer::contents()` into owned bytes currently require unsafe FFI in this repository. Under the safe-Rust-only rule, do not add those calls to the referee or hide them in a nominally safe wrapper. The concrete transport is blocked until an existing reviewed safe platform abstraction can supply a completed, owned byte snapshot with documented lifetimes and bounds. If no such abstraction exists, stop and request an explicit decision about a separately audited platform-boundary exception; do not silently relax the rule. The existing safe KV page API remains usable within its idle-state contract meanwhile.

The pinned `objc2-metal` 0.3.2 binding confirms this boundary: `MTLBuffer::contents()` yields a raw `NonNull<c_void>`, not an owned byte slice, and `MTLBlitCommandEncoder::copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size` is explicitly `unsafe` with caller obligations for synchronization, lifetimes and bounds. Neither method is a safe snapshot primitive. No current-tree `MetalBufferArena` method discharges those obligations for transient Q.

## Fail-closed referee contract

The pure Rust verifier can be built and host-tested independently of transport, with `#![forbid(unsafe_code)]`:

1. Match an immutable, create-new plan and every source/input hash; reject unknown schema fields, malformed hashes, stale binaries, wrong route, phase, layer, ticket, input token or conditioning history.
2. Require the exact declared descriptor sequence, dtype, shape, layout, native BF16 bit width, checked byte lengths, disjoint monotonic payload spans, complete payload SHA-256 hashes and no undeclared captures. Explicitly unavailable fused interiors are never treated as equal tensors.
3. Require `Completed` GPU status and collection of the owning ticket before admitting payload. Reject any missing, duplicated, truncated, over-budget or mismatched record. Keep failed receipts and raw outputs; never overwrite them or convert failure into absence.
4. Check actual per-boundary dispatch prefix and final dispatch totals for both routes, not just a final selected-kernel flag. Count any compilation/API attempts with a precisely scoped counter; an API-call counter is not proof of no driver JIT. Refuse a route-changing legacy trace or a hidden recomputation kernel.
5. Validate logical KV table, producer alias, context and attended row order. Hash raw native bytes; decode BF16 only for separately labelled numerical statistics. Compare off/on bit patterns and report the earliest **observed stored** difference in the admitted span. An independent same-input attention reference is a separate gate before attributing arithmetic cause or correctness.

Host negative tests should cover every rejection above, especially wrong command owner, zero/extra point, source span past arena capacity, overlap/overflow, missing logical page, stale conditioning, changed dispatch prefix, `Error` status, and a declared-but-unavailable Q. A source-only lexical check, host test, or successful capture does not prove production route preservation; that needs a later serial device check with exact pins and retained condition receipts. Readback timings are invalid for speed.

## Sequencing decision

First implement only the safe plan/descriptor/referee logic and host negatives against this current tree, without a concrete Q transport or device run. Audit available Metal abstractions for a genuinely safe owned snapshot primitive. If none exists, record that as the blocker. Only after the transport and route gates are independently reviewed should a fresh immutable serial experiment be predeclared on distinct input. Do not replay prior trace IDs, weaken numerical gates, or promote from one matching top token or a first stored-bit difference.
