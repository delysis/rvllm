# Gemma 4 Apple next trials from the production-stack audit

Date: 2026-09-24

## Decision

The next campaign is not another flat tile sweep. The existing evidence and the
production-stack audit agree that the useful unit is a sealed
phase × layer-geometry × cache-format × quantization-format × hardware family.

The immediate order is:

1. Metal global D512/GQA16 decode with exact split-KV and merge.
2. Metal local D256/GQA2 decode, normally single-pass within the 1024 window.
3. Gemma QKV preparation fusion with preserved rounding boundaries.
4. Global D512 prefill, then local D256 prefill with analytical tile skipping.
5. Separate 4-bit, 8-bit, and 16-bit linear families for decode and prefill.
6. ANE graph-boundary reduction and fixed-shape program scheduling; no more
   cosmetic FFN tiling rounds without a boundary-count argument.

The unsplit `metal-global-d512-r8p64t128` kernel remains the Metal correctness
and performance control. It is the best current rvLLM operator at context 2048
(23.399 ms), but it uses one threadgroup and is still about 15.7 times slower
than the matched exploratory MLX operator result. Down4, Interleaved, and
Chunk4 are not ANE nominees: Down4 was 1.77% slower geometrically over 405
usable paired steps, Interleaved lost timing, and Chunk4 failed the bounded
equivalence oracle.

## Trial 0: repair the D512 control

The source audit found that the current `attention_decode_f16` timing control is
equivalent operator work but a pathological implementation: it dispatches 16
threadgroups with exactly one thread per query head, keeps a 512-element Q
array and 512-element output allocation behind that thread, walks every visible
KV token serially, and rereads the shared K/V stream independently for every
head. The normal layer route selects the 32-lane
`attention_decode_online_f16` kernel only when `head_dim <= 256`; Gemma's eight
global D512 layers therefore fall through to the scalar kernel. This also
explains a material part of the end-to-end long-context collapse.

Before attributing a split-KV gain to the new algorithm, add one bounded
research-only D512 headwise SIMD control:

- one 32-lane SIMD group per query head, grid width 16;
- 16 FP32 Q/output slots per lane for D512;
- paged-cache and logical-length semantics identical to the incumbent;
- online softmax in registers and one final BF16 rounding;
- no attempt to share K/V across heads.

This is deliberately not the final GQA16 design—it rereads K/V 16 times—but it
restores ordinary GPU occupancy and establishes whether the 23.399 ms
cooperative winner is beating GQA reuse or merely beating a scalar fallback.
Keep it a distinct fixed-D512 kernel rather than widening the existing dynamic
D256 kernel and risking register-footprint regressions in 40 local layers.

Run its device oracle and ABBA cells at 256/512/1024/2048 first. Split-KV then
compares against three controls: the retained scalar production fallback, the
D512 headwise SIMD control, and the unsplit shared-K/V `r8p64t128` kernel.

## Trial 1: Metal global D512/GQA16 split-KV

### Candidate family

Implement two ownership patterns around the proven panel-64/128-thread base:

- `rows8`: two query-head groups per KV partition, preserving the unsplit
  winner's lower accumulator pressure.
- `rows16`: one cooperative GQA group per KV partition, maximizing K/V reuse.

For each, test KV partition sizes 256, 512, and 1024. Six source identities are
the upper bound for the first round. Each identity owns two kernels:

1. a partial kernel emitting per-head `(maximum, denominator, FP32 D512
   numerator)` for each nonempty partition;
2. a merge kernel combining partials in a common exponent frame and performing
   the single final BF16 rounding.

An entirely masked partition must emit the exact identity state and must never
evaluate `-inf - -inf`. Allocated cache capacity and logical visible length are
distinct sealed inputs. No K/V expansion by query-head count is allowed.

### Correctness gate before timing

Every candidate must pass the current paged-cache, prefix/rollback, negative
page, guard-byte, repeated-use, independent FP64-dot, exact serial-GPU FP32,
and once-rounded BF16 checks, extended with:

- exact single-pass versus split parity;
- contexts 1, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048,
  and 2049;
- odd partition counts;
- first, middle, and last entirely masked partitions;
- live length smaller than allocated capacity;
- positive engagement evidence for both partial and merge kernels;
- rejected dispatches leaving output and scratch guards untouched.

The sealed receipt must include both kernel identities, both dispatch grids,
scratch offset/size/alignment, partial count, logical length, page geometry,
and zero source compiles during timed work.

### Successive-halving timing

Use the shortest **discriminating** context for each partition size rather than
pretending a one-partition run tests split occupancy:

| Stage | Required candidates | Purpose |
|---:|---|---|
| 512 | partition 256, both ownership patterns; one-partition variants as merge-overhead controls | first real split and fixed-cost screen |
| 1024 | surviving partition-256 candidates plus partition 512, both ownership patterns | crossover and merge-cost screen |
| 2048 | fastest two plus every candidate within 10%; include partition 1024 at its first real split | long-context finalist |
| 4096 | deferred until a 2048 candidate is materially better | confirmation only |

At every stage compare against both the untouched `attention_decode_f16`
incumbent and `r8p64t128`. The latter is the meaningful algorithmic control;
the former is retained to expose baseline pathology. Price partial execution,
merge execution, intermediate traffic, and total synchronized GPU time
separately.

The initial advancement metric is absolute candidate GPU time. A candidate
cannot represent a workload bucket in the eventual tuning table if it regresses
any representative point in that bucket beyond the declared tolerance.

## Trial 2: Metal local D256/GQA2 decode

Build a distinct single-pass family for the 40 sliding layers. Do not reuse the
D512 schedule with smaller arrays. Screen contexts 256, 512, and the full 1024
window. The first candidates should vary query ownership and K/V tile width,
not introduce split-KV by default: the bounded window and existing head-level
parallelism may already occupy the GPU.

Only add a split local variant if measured workgroups per Apple GPU core fall
below the occupancy threshold and at least two useful partitions exist.
Analytically skip forbidden tiles, omit predicates for wholly valid tiles, and
predicate only boundary tiles.

## Trial 3: QKV preparation fusion

Treat packed QKV output through cache append as one Gemma primitive:

`packed QKV → Q/K/V RMSNorm → partial/full RoPE → cache write + Q output`.

First benchmark it independently at decode `M=1` and prefill
`M={64,256,512,1024}`. Preserve the existing FP32/BF16 boundaries explicitly;
fusion is rejected on the first numerical mismatch even if faster. The full
route must demonstrate that no CPU-visible synchronization or extra command
buffer boundary was introduced.

Do not fuse the projection GEMM itself until the boundary-only primitive is
measured. Destroying a good matrix kernel to save a cheap launch is not an
acceptable assumption.

## Trial 4: prefill and cache correctness

Global D512 prefill needs a dedicated matrix-shaped family with D panels of 64
and 128. A full D512 dot product must finish before softmax; panel-local softmax
is invalid. Compare output-stationary against score-first/output-panel
cooperation at prompt lengths 256, 512, 1024, and 2048.

Local D256 prefill must use a write-safe sliding cache. If a chunk of length
`C` is written before all of its queries consume prior history, physical
capacity must be at least `W + C - 1`, or old and new slabs must remain
separate. Sequential and chunked prefill must produce equivalent cache-visible
results before timing.

## Trial 5: 4-bit, 8-bit, and 16-bit linear families

Weight precision is independent of the attention/KV precision. Each path must
seal signedness, group size, scale axis and dtype, zero-point policy, packing,
and rounding; the label `INT8` or `4-bit` is not a numerical contract.

For both Metal and ANE, separate:

- decode/small-M GEMV or small-GEMM at `M={1,2,4,8}`;
- prefill GEMM at `M={64,256,512,1024}`;
- QKV shared-input packing/quantization;
- gate/up shared-input packing/quantization;
- down projection and vocabulary head.

The current ANE component evidence makes the priority explicit:

| Path | Current layer-0 median | Immediate next gate |
|---|---:|---|
| LUT4 | 1.530 ms | full-model quality and exact selected-format contract |
| INT8 | 3.085 ms | incumbent performance control and full-route quality |
| BF16 | 54.349 ms | graph/program audit before further kernel tuning |

LUT4's large error relative to the original BF16 computation means its speed
does not qualify a 4-bit model. It must pass a model-level quality suite before
promotion. Fidelity to a quantized CPU reference proves implementation
correctness after quantization, not semantic equivalence to BF16.

## ANE-specific next candidate

The last three ANE layout experiments say not to spend another round rearranging
the same three convolution/program boundaries. The next proposal must reduce
one of:

- program/conv count;
- repeated activation quantization or packing for gate/up and Q/K/V;
- intermediate materialization;
- fixed-shape padding or scheduling waste.

Use compiled programs for `prefill-{64,128,256,512,1024}`, `decode-1`, and a
small verification width rather than forcing one dynamic graph to cover every
phase. The first candidate should target FFN gate/up shared-input work or a
provably fused producer/consumer boundary. ANE attention remains an R&D lane:
there is no public production Gemma 4 12B ANE attention design to copy, so it
must not displace the proven linear/FFN opportunity without evidence.

## Autotuning and condition policy

The tuning key is:

`(GPU family, kernel family, phase, D, GQA, Q bucket, S bucket, weight format,
KV format, mask, query tile, KV tile, D panel, threads, split count)`.

Borrow llama.cpp's repeated anchor principle, but honor the campaign rule that
experiments do not wait for stable conditions. Run anchors before and after
candidate blocks, log all conditions and competing activity, retain every
sample, stratify when possible, and schedule later remeasurement when anchors
move. Do not pause for cooldown or discard a thermal excursion. Final promotion
still requires a clean independent confirmation, but exploration must continue
under ordinary daytime variance.

## Promotion boundary

Operator wins are not model wins. Promotion requires, for the exact source and
compiled identities:

1. adversarial device correctness and positive dispatch engagement;
2. exact full-route identity with no fallback or compile during inference;
3. no regression at every representative point in the claimed tuning bucket;
4. operation-level timing against rvLLM controls and comparable MLX kernels;
5. end-to-end prefill and sustained decode at 256/512/1024/2048;
6. model-level quality for each 4-bit and 8-bit format;
7. independent confirmation before changing a production default.

Approximate sparse attention, reconstructible raw global caches, compressed KV,
and MTP remain separate later lanes. They must not obscure the exact BF16-KV,
4/8/16-bit-weight baseline campaign.
