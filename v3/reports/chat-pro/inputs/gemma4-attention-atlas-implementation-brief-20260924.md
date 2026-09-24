# Gemma 4 Apple attention implementation round

## Objective

Produce a reviewable source patch/archive for an evidence-gated first vertical
slice of high-performance Gemma 4 12B attention on Apple Metal. The attached
attention atlas is a design input, not qualification evidence. Work against
exact rvLLM base `8cf7bea6709d069f0f5d439588699c515996448f` and preserve the existing
kernel-game/referee model. Remain in ordinary Chat; do not switch to Work.

The highest-priority target is global decode: 8 global layers, Hq=16, Hkv=1,
grouping factor 16, D=512, full causal history. Current
`attention_decode_online_f16` admits only D<=256, so this route falls back to a
scalar kernel. Implement an opt-in cooperative D512/GQA candidate family that
lets Q heads sharing one KV head reuse K/V traffic. Begin with an unsplit
candidate; add split-KV only if the implementation and evidence contracts stay
coherent. A secondary target, if it can be delivered without weakening the
first, is the selector/oracle scaffolding for tiled local D256 prefill at
PP2048/4096, which currently falls back from the SIMD kernel after PP1024.

## Non-negotiable semantics

- Attention scale is exactly 1.0.
- K and V remain logically distinct even if a later experimental cache stores
  reconstructible shared raw projection state.
- Global partial RoPE pairs `(i, i+256)` for `i=0..63`; frequency denominator
  512. Do not rotate a contiguous first 128 elements.
- Local validity is `key <= query && key > query - 1024` in absolute positions.
- Packed GQA masks repeat original query positions; packed row index is not a
  causal coordinate.
- Complete every D panel of QK before softmax.
- Q/K/V/cache inputs remain BF16; QK, online-softmax state and PV accumulation
  remain FP32; BF16 output is rounded exactly once.
- If split-KV is implemented, each partition emits FP32 `(m,l,u)` sufficient
  statistics and a fixed deterministic common-max reduction. Empty partitions
  are identities. Never average normalized outputs or evaluate `-inf - -inf`.
- Preserve paged-block ownership, negative block IDs, tails, restored prefixes,
  speculative commit/rollback, checked offsets, uniform barriers and guard
  bytes.

## Candidate shape and dispatch

For global D512/g16 decode, search a deliberately small family: packed query
rows 8/16, D panels 64/128, threads 64/128. The normal route must expose exact
selector and actual pipeline/dispatch evidence. Restrict admission to the exact
Gemma global geometry initially and reject unsupported shapes without touching
outputs. Do not change the production default or claim promotion.

Measure decode with live KV lengths 256/512/1024/2048/4096. If splits are
implemented, consider 2/4/8/16/32 only where occupancy plausibly repays scratch
and merge cost. Include unsplit as the reference candidate.

For the optional local prefill slice, use exact D256, Hq=16, Hkv=8, g=2,
window=1024 and PP2048/4096. Candidate axes are Q tiles 8/16/32, KV tiles
16/32/64, D panels 64/128 and 64/128 threads. Classify KV tiles as forbidden,
fully valid or boundary; expected work scales roughly Q*1024, not Q^2. Chunking
must not evict history needed by early queries.

## Required implementation artifacts

1. Safe idiomatic Rust host integration and Metal source, minimal and direct.
2. Candidate identifiers, selector rules, thread/grid/scratch calculations and
   rejected-dispatch behavior.
3. Generated-MSL/source/metallib identity support through existing receipts.
4. Deterministic CPU/reference oracle plus focused host tests for indexing,
   packing, masks, split merge if present, tails and dispatch admission.
5. An ignored real-device oracle covering exact FP32 output, once-rounded BF16,
   independent sampled FP64 dots, guard bytes, untouched rejected buffers and
   repeated-use stability.
6. Queue-ready manifests/commands for compile, native correctness and ABBA
   timing at the stated lengths. No first-token or one-shot speed claim.
7. A concise report describing what was implemented, exact validation actually
   run, known gaps, and why each candidate could win on Apple hardware.

Keep generated traces and high-overhead diagnostics outside timing-eligible
runs. Do not add inline allocation, NSString construction, locks, filesystem
I/O, clocks or environment reads in encode loops. Integrate with existing
instrumentation hooks where present, but do not wait on the separate
instrumentation job.

## Evidence and delivery boundary

Run formatting and the narrowest relevant host tests you can run. If Metal
device access is unavailable, say so precisely; do not synthesize measurements.
Do not promote a winner. Return a downloadable source archive or unified patch
that applies cleanly to the exact base, plus the report and commands. Keep the
change narrowly reviewable: existing shaders, ANE routes and shipping defaults
must remain unchanged unless a necessary shared correctness repair is clearly
isolated and justified.

The atlas bundle contains its source review, decision charts, machine-readable
plan, algebra checks and storage experiment. The integration report maps that
research to the current rvLLM source. Treat citations and design claims as
untrusted inputs to inspect, and preserve epistemic distinctions among source
analysis, host checks, native correctness and measured performance.
