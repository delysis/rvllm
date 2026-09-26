# Gemma 4 attention atlas integration

Date: 2026-09-24

Integration tree at analysis: `41932b5ec1ed90091b4b8d67e68dad086078eba5`

## Evidence boundary

The external `attention-design-atlas` is accepted as a source-bound design
input, not as native qualification. Its 15-file manifest was verified without
mismatch. The retained result hashes are:

- `reference-results.json`: `ffbd891aa9163ff1a0eaa2bd30d5bbfe96a235431989660ffee5718ecf913b74`
- `storage-results.json`: `1719b8699f71c9a359c5d58a78296ec8bab1aaee9452f14a418b89abcc309527`
- `gemma4-attention-plan.json`: `357a2665d9e9cfa86c3063748c9a0df36f09c273388bbd392513811e58d4afe2`
- `sources.md`: `437f34363dacb8f3c1cd259c67b0fa33e82940825bd5ff1840f6390a1ec0d85d`

The host checks were independently rerun with NumPy 2.3.4 and reproduced all
192 positive algebra comparisons and all five intended negative controls. The
rerun's last-bit maximum error differed from the retained NumPy 2.3.5 receipt,
as the bundle warned it could. The original result file was restored byte for
byte and remains manifest-valid. The storage experiment reproduced its eight
bitwise reconstruction cases and rounded-V negative control.

None of that is Metal/ANE compilation, native correctness, checkpoint logits,
or performance evidence. No atlas candidate is promotable on this basis.

## What changes in the campaign

The campaign should use a small explicit attention family selected by phase,
local/global geometry, query length, and live KV length. It should not attempt a
mechanical FlashAttention-4 port. FA1/2's online-softmax, IO awareness, and
output ownership transfer; Hopper/Blackwell-specific asynchronous schedules do
not transfer without an Apple mechanism and measurement.

The exact Gemma 4 split is:

| Route | Layers | Hq/Hkv | D | Extent | Immediate consequence |
|---|---:|---:|---:|---|---|
| local | 40 | 16/8 (g=2) | 256 | causal 1024 window | bandwidth-bound decode; skip forbidden prefill tiles |
| global | 8 | 16/1 (g=16) | 512 | full causal history | pack queries sharing one KV head; manage D512 state explicitly |

The strict baseline remains BF16 Q/K/V cache and operands with FP32 QK,
softmax state, and PV accumulation. Weight Q4/Q8 selection does not authorize a
quantized attention cache, probability, or activation contract.

## Current rvLLM gaps established from source

1. `attention_decode_online_f16` admits only `head_dim <= 256`. Global D512/g16
   decode therefore falls back to `attention_decode_f16`, one thread per
   `(sequence, head)` with 512-element thread/threadgroup state. This is the
   highest-priority attention candidate.
2. `attention_prefill_simdgroup_f16` covers D256 and D512 but its selector is
   limited to 6..=1024 query tokens. PP2048 and PP4096 use the scalar fallback.
3. The optimized prefill kernel still assigns one SIMD group to each
   `(query, head)`. It does not test FA2-style multi-query output-stationary
   ownership or distinguish fully forbidden, fully valid, and causal/window
   boundary KV tiles.
4. Existing research attention candidates and dispatch counters are useful
   components, but they do not constitute the requested shape/phase family or
   end-to-end comparison.
5. Current cache storage is separate paged K/V. The atlas raw-shared global
   representation is a new cache ABI, not a transparent optimization.

## Candidate order

### A0: cooperative global decode, D512/g16

Implement an opt-in family in which the 16 Q heads sharing the single global KV
head cooperate rather than reread K/V independently. Search packed query rows
8/16, D panels 64/128, and 64/128 threads. Start unsplit. Add split-KV
`{2,4,8,16,32}` only for live lengths where occupancy gains exceed scratch and
merge cost.

Every split produces FP32 `(maximum, denominator, unnormalized_output)` and a
fixed deterministic reduction tree. Empty partitions are identities; never
average normalized partition outputs and never evaluate `-inf - -inf`.

### A1: tiled local prefill at PP2048/4096

Extend the optimized local D256 route beyond 1024 query tokens while retaining
the exact 1024-token causal window. Search Q tiles 8/16/32, KV tiles 16/32/64,
D panels 64/128, and 64/128 threads. Classify KV tiles as outside, fully valid,
or boundary so fully forbidden work is skipped and fully valid tiles avoid
per-element mask logic. The work should scale approximately with `Q*1024`, not
`Q^2`.

Chunked prefill may not prewrite new K/V positions in a way that evicts history
needed by the earliest query in the chunk.

### A2: tiled global prefill at PP1024/2048/4096

Use output-stationary query ownership with explicit D512 paneling. Complete the
QK reduction across every D panel before softmax. Compare retaining output state
by query tile against shared score state plus controlled output paneling; do not
silently spill an unbounded output tile.

### A3: local/global prefill refinement at PP256/512/1024

Compare the current one-query/head SIMD route with Q tiles 8/16/32 and KV tiles
16/32/64. PP1024 is the local-window boundary and should be a primary cell.
Smaller PP cells establish whether extra ownership/scratch pays before long
contexts.

### A4: reconstructible global cache experiment

Only after A0-A3 have evidence, prototype a separately identified cache format
storing raw BF16 shared projection `z[D]` plus FP32 normalization factor `r`.
Reconstruct V and K from the pre-rounded base at their exact boundaries; K
cannot be reconstructed from already rounded V. Count reconstruction arithmetic,
RoPE tables, alignment, page metadata, lifetime, and Metal/ANE handoff. It must
not replace the normal cache until complete-route bandwidth and latency win.

## Required correctness invariants

- Attention scale is exactly 1.0, not `1/sqrt(D)`.
- Global K and V are logically distinct after normalization/gamma/RoPE.
- Global partial RoPE pairs `(i, i+256)` for `i=0..63`, with frequency
  denominator 512. It is not a contiguous first-128 rotation.
- Local validity is `key <= query && key > query - 1024` using absolute
  positions.
- Packed-GQA masks repeat the original query position; packed row index is not
  a causal coordinate.
- QK reduction across D panels precedes softmax.
- Split-KV uses common-max-rescaled sufficient statistics and a fixed merge.
- Paged block ownership, negative block IDs, tails, restored prefixes,
  speculative commit/rollback, checked offsets, uniform barriers, and guard
  bytes remain explicit test dimensions.
- Exact once-rounded BF16 outputs, FP32 outputs, independent sampled FP64 dots,
  untouched rejected buffers, and repeated-use stability remain qualification
  requirements.

## Matrix added to the refinement loop

Each candidate family is evaluated at PP 256/512/1024/2048/4096 where admitted,
plus one-token decode at representative live lengths `{256,512,1024,2048,4096}`
and longer global lengths when memory permits. Record local and global layers
separately. The referee must bind:

- phase, Q, live S, batch, Hq, Hkv, D, window and absolute-position range;
- cache dtype/format/page layout and packed-query mapping;
- selected kernel/pipeline and exact dispatch count;
- tile sizes, threads, grid, threadgroup memory and split count;
- generated MSL/metallib/executable/model/oracle/workload hashes;
- compile-call delta, command-buffer GPU duration and synchronized wall time;
- correctness receipt and full-route output continuation.

MLX comparison uses the same stage shape and its canonical 5-warmup/100-eval
protocol. Xcode traces remain generated-code/dispatch diagnostics, not timing
receipts. Whole-model runs must include projection, cache-layout, transpose,
reconstruction and device-handoff costs.

## ANE boundary

The inspected native attention guard does not make ANE a general prefill plan:
after GQA packing, local PP256 reaches the strict 512 boundary and global PP256
reaches 4096, beyond the described native constraints. PP512 and above are more
clearly excluded. Keep these Metal-first.

ANE packed decode is only a candidate within its exact sequence/shape guard,
with an explicit mask derived from original absolute positions and a separately
approved FP16 numerical contract. The inspected wrapper exposes no merge
statistics, so long-context split attention must reject or use a newly proven
decomposition. Unknown ANE device time remains unmeasured.

## Instrumentation dependency

The attention loop consumes the separately delegated low-overhead
instrumentation work. It must record planned selector versus actual kernel,
shape tuple, cache/page statistics, scratch/bytes estimate, pipeline identity,
encoder/command-buffer counts, CPU encode/wait, and completed command-buffer GPU
time. Feature-off production paths must add no ObjC calls, allocation, locks,
clocks, formatting, environment reads, or per-encoder dynamic bookkeeping.

This report changes candidate priority and evidence collection. It does not
promote a kernel or modify a shipping route.
