# Gemma 4 rvLLM/MLX comparison gap matrix — 2026-09-24

## Scope and claim boundary

This is an audit of evidence already present in this tree. It does not add a
benchmark result and it does not infer performance from an implemented route.
In particular, `load4` in the Metal candidate name means four-element vector
loads; those candidates consume BF16 weights. It does **not** mean a four-bit
weight route.

The requested comparison is Gemma 4 12B at prompt lengths 256, 512, 1024,
2048, and 4096, for prefill and decode, at model-wide 4-, 8-, and 16-bit weight
formats, both end to end and at these stages:

1. embedding;
2. QKV, separately for a sliding and a full-attention layer;
3. attention core/SDPA, separately for sliding and full attention;
4. O projection, separately for sliding and full attention;
5. FFN gate/up plus activation;
6. FFN down;
7. representative post-attention RMSNorm plus residual;
8. LM head.

The checked-in MLX stage packet has exactly that stage/length grid for BF16,
affine Q8 group-64, and affine Q4 group-64: 15 model/length jobs, each with
prefill and one-token-decode shapes and 22 cases. Each case uses five warmups
and 100 `mx.eval`-synchronized iterations. These are isolated operator
timings, not normal-route latency and not promotion evidence. See
`reports/gemma4-mlx-stage-matrix-20260924/README.md`,
`tools/mlx_gemma4_stage_bench.py`, and
`specs/kernel-game/MLX_GEMMA4_STAGE_MICROBENCH.md`.

## Executive result

The requested apples-to-apples matrix does not yet exist on the rvLLM side.
The evidence supports narrower statements:

- rvLLM has a native 16-bit Metal full route and emits end-to-end prefill and
  decode wall time, but no per-stage durations. The existing Gemma 4 12B
  load4 tournament is a BF16 first-token candidate comparison, not a Q4
  comparison and not the five-length matrix.
- rvLLM has a Metal-prefill/ANE-decode full route at context capacities 64 and
  1024. ANE decode already emits useful application-call timing buckets for
  QKV, attention, O, fused FFN, vocabulary projection, host remainder, and
  total. It does not expose ANE prefill, split FFN timings, standalone
  embedding, or standalone norm/residual timings.
- The current Metal W4A16/W8A16 package support is a tensor-level sidecar for
  selected dense `down_proj` tensors, not a model-wide Q4/Q8 route. The ANE
  LUT4 and INT8 plans likewise quantize selected FFN and, in some plans,
  sliding-QKV components while other projections remain FP16. Those hybrid
  plans must not be labeled as the MLX model-wide Q4 or Q8 lane.
- Existing correctness-qualified receipts are valuable, but their observed
  prompt lengths are principally 21 and 84 tokens, not the requested grid.
  The 84-token interleaved campaign advances nine decode steps, so it observes
  contexts 84 through 92, not a 256-token decode shape.
- The disaggregated driver accepts 2048 and 4096 only in Metal-prefill-only
  mode. ANE decode remains restricted to capacity 64 or 1024. The standalone
  Metal driver can allocate a larger explicit total-token budget, but a
  capability is not a measurement.

## Route and evidence matrix

Here, “qualified” means a correctness route has an explicit successful
qualification receipt. It does not mean that the requested timing cell is
filled.

| Backend and weight lane | Actual implemented route | Requested lengths with existing qualifying timing evidence | Correctness boundary | Disposition for MLX comparison |
| --- | --- | --- | --- | --- |
| Metal 16-bit | Normal-route Metal prefill and Metal decode; runtime reports F16 or BF16 compute/weight dtype | No complete 256/512/1024/2048/4096 matrix identified. The current load4 tournament is BF16 first-token work, not the requested length grid | The load4 component oracle requires exact FP32 QKV bits, exact once-rounded BF16 stored projection bits, FP64 sampled dot products, guard integrity, rejected-dispatch integrity, and repeat stability; full-route screens cover actual 48-layer dispatch | Implement stage timing, then queue all five lengths. Existing end-to-end fields can be used immediately, but only at freshly sealed requested shapes |
| Metal 8-bit | Hybrid W8A16 group-32 sidecars for selected dense `down_proj` tensors; remaining tensors stay native F16 | None as a model-wide Q8 lane | Sidecar delivery/execution tests qualify the selected tensor route, not Q8 Gemma 4 as a whole | Not comparable to MLX affine Q8 g64. Queue only as an explicitly named hybrid experiment until a model-wide Q8 route exists |
| Metal 4-bit | Hybrid W4A16 group-32 sidecars for selected dense `down_proj` tensors; remaining tensors stay native F16 | None as a model-wide Q4 lane | Same tensor-scoped boundary as W8 | Not comparable to MLX affine Q4 g64. Do not relabel BF16 `load4` data as Q4 |
| ANE 16-bit | Metal prefill, then 48-layer ANE decode with CPU norms/RoPE/residual/ranking and FP16 static weights | Qualified short-route receipts exist at prompt lengths 21 and 84; no five-length matrix. Capacity is 64 or 1024 | Full-route receipts prove actual KV import, ANE steps, expected tokens, zero Metal decode fallback, and compile budget where recorded | Decode-only comparison can be queued at exact contexts 256, 512, and 1024. ANE 2048/4096 and ANE prefill are unsupported by the current route |
| ANE 8-bit | Mixed plans: INT8 FFN variants; some plans also quantize sliding QKV. Other projections, including global QKV, remain FP16 | The interleaved INT8-FFN campaign uses 84-token prompts and nine steps; no requested length matrix | `static-int8-interleaved-ffn-cached` is full-route correctness-qualified. Its controlled timing did not establish a win: only two of six process pairs passed the strict phase comparator, and both valid pairs were slower | Treat as mixed-precision candidate data, not model-wide Q8. It may be compared to FP16 rvLLM for its exact route, but not to MLX Q8 as a format match |
| ANE 4-bit | `static-lut4-ffn-cached` quantizes FFN work while the rest remains FP16 | No requested length matrix | Existing route evidence is component/short-route scoped; it does not establish a model-wide Q4 model | Treat as mixed LUT4-FFN only. A true Q4 lane requires an explicit model-wide route and independent correctness gate |

The Metal package boundary is stated directly in
`specs/apple-model-package.md`: the sidecar is “tensor-level,” not model-wide,
and currently applies to selected dense non-MoE `down_proj` tensors from
uniform F16 checkpoints. The execution ledger is
`specs/apple-backend-capability-ledger.md`. ANE plan definitions and precision
selection are in `crates/rvllm-runtime/src/gemma_ane_decode.rs`.

## Current timing fields and stage crosswalk

### End-to-end and route-level fields

The standalone Metal report in
`crates/rvllm-runtime/src/bin/rvllm_metal_infer.rs` emits:

- `prepare_ms`, `prefill_ms`, `decode_ms`, and `tok_per_s`;
- `last_step_gpu_execution_ns`;
- library and pipeline compile counts;
- command-buffer, encoder, forced-wait, CPU wall/encode/wait, and coarse
  dispatch-family counts;
- actual research-candidate dispatch and reported compute/weight/router dtype;
- generated tokens and optional HF-reference comparison.

These fields support normal-route total comparisons. They do not attribute
time to QKV, attention, O, FFN, norms, embedding, or head. Encoder and dispatch
counts are work receipts, not durations.

The disaggregated route in
`crates/rvllm-runtime/src/bin/rvllm_disaggregated_infer.rs` additionally emits
Metal-prefill wall and GPU-execution totals, KV-capture cost, host non-wait and
command-buffer wait totals, compiler deltas, dispatch evidence, and one ANE
step record per generated token.

### Stage crosswalk

| MLX stage category | Metal timing currently emitted | ANE timing currently emitted | Correct apples-to-apples interpretation | Minimal missing instrumentation |
| --- | --- | --- | --- | --- |
| Embedding | No stage duration; only coarse embedding dispatch count inside whole-route totals | No separate duration; file read, FP16 decode, and embedding scale fall into `host_ms` | MLX embedding includes `embed_scale`; rvLLM must include lookup/read, dtype conversion, and scale in the named stage | Add an explicit synchronized embedding bucket on both routes; report storage dtype/bits rather than pretending tied embeddings are quantized |
| QKV sliding/full | No duration; candidate/component receipts prove specific projection dispatches and values | `qkv_ms` sums `layer.qkv.project` over all 48 layers | ANE total is directly useful for aggregate QKV, but does not separate sliding/full. Metal has no duration | Time exact normal-route projection calls, reporting layer kind and layer index; retain both per-layer and all-layer sums |
| Attention core/SDPA sliding/full | No duration | `attention_ms` sums `layer.attention.decode` over all layers | ANE bucket is application-call wall including I/O/scheduling, not a hardware counter. It is decode only | Add Metal synchronized timing around the exact attention implementation. Split ANE sums by layer kind without changing execution |
| O projection sliding/full | No duration | `output_ms` sums `layer.output.project` | Same caveat as QKV: application-call wall, aggregate over layers | Add Metal exact-call timing; split ANE accounting by layer kind |
| FFN gate/up plus activation | No duration | Not separable: `ffn_ms` wraps the fused `layer.ffn.project` | Only `MLX gate/up+activation + MLX down` may be compared to ANE `ffn_ms`; comparing either MLX substage alone would be false precision | Add Metal substage timers. For ANE, either keep the honest fused comparison or build a separately labeled diagnostic split program using the same real weights; do not call that diagnostic timing normal-route timing |
| FFN down | No duration | Included in fused `ffn_ms` | Same fused-only boundary as above | Same as above |
| Representative post-attention RMSNorm + residual | No duration | Included in `host_ms` with all other unbucketed host work | The MLX case is one representative post-attention norm+residual, not all norm/residual work. Current ANE `host_ms` is too broad | Add a narrow timer around post-attention RMSNorm plus its residual. Keep other norms/RoPE/scales separately or in an explicit residual bucket |
| LM head | No duration | `vocabulary_ms` measures tiled projection only; final norm is in `host_ms`, while softcap/ranking may have its own CPU timing | Compare projection to projection. Do not silently include final norm/ranking on one side only | Add Metal projection timer. On ANE, emit final norm, projection, softcap, and ranking separately while retaining their total |

`AneDecodeTimes` explicitly documents that its buckets cover application calls,
including I/O and scheduling waits; they are neither hardware counters nor
isolated ANE device-time measurements. Its `host_ms` is calculated as the
residual after QKV, attention, O, fused FFN, and vocabulary projection.

## Existing correctness-qualified data that can be reused

1. **BF16 Metal load4 candidates.**
   `reports/gemma4-load4-tournament-20260924.md` retains the component oracle,
   exact dispatch screens, first-token full-route screens, and controlled ABBA
   blocks. None of the seven candidates displaced `metal-mma32-load4`. This is
   high-value BF16 projection evidence, but it does not fill a model-wide Q4
   lane, a decode lane, or the requested sequence-length matrix.
2. **ANE interleaved INT8 FFN.**
   `reports/gemma4-ane-interleaved-timing-20260924.md` retains all 108 requests
   from twelve processes. Every request produced the same ten token IDs, nine
   ANE steps, zero Metal decode steps, and zero compiler calls. Only two of six
   timing pairs were admissible; both rejected a speed win. This qualifies the
   mixed route's correctness, not a Q8 format comparison.
3. **ANE short full-route receipts.**
   `reports/gemma4-ane-interleaved-full-route-queue-20260924/results/` records a
   successful 21-token, one-step qualification; the corresponding `inference`
   report has `qualification_complete: true`.
   `reports/gemma4-ane-interleaved-copy96-queue-20260924/results/` records a
   successful 84-token, nine-step qualification with the same flag. These
   receipts are usable as regression gates, not as requested-length timing
   cells.
4. **MLX stage jobs.**
   `reports/gemma4-mlx-stage-matrix-20260924/` contains the declared 15-job
   matrix. Every result must remain labeled exploratory and isolated. A
   successful stage receipt proves that the actual module executed under the
   pinned timing protocol; it does not prove normal-route performance or
   numerical agreement with rvLLM.

## Minimal work needed to fill defensible comparison cells

### 1. Freeze one shared workload identity

Create one immutable token fixture per length, with exactly 256, 512, 1024,
2048, or 4096 valid token IDs. Both implementations must consume the same
tokens, model revision, RoPE/sliding-window semantics, attention mask, KV-cache
starting position, and requested output count. Seal hashes for tokenizer,
config, tensor inventory, fixture, executable, generated Metal source or
metallib, and runner source. Record actual dtype and quantization parameters;
the label `4-bit` or `8-bit` is insufficient without scheme, group size,
scales, zero points, and the set of quantized tensors.

Use a fixed decode workload, preferably 64 emitted tokens, where the route
supports it. Report first-token latency separately from steady-state decode.
If 4096 prompt tokens plus decode exceed a backend capacity, record the cell
as unsupported rather than shortening the work.

### 2. Add a real-weight rvLLM Metal stage runner

The smallest honest implementation is a diagnostic binary that loads the
ordinary Gemma 4 model package and invokes the same normal-route encoder
functions and generated Metal entry points. Use layer 0 for sliding attention
and the first full-attention layer for full attention, matching the MLX runner.
Materialize stage inputs from the real route, then run five warmups and 100
synchronized timed iterations per case. The enclosing timer must end only
after the command buffer is complete. Emit all 22 cases for one length as one
strict-JSON receipt, including function name, grid/threadgroup geometry,
shared-memory bytes, generated-MSL/metallib hashes, actual dispatch evidence,
compiler-call deltas, and output checksum.

The stage runner must not be presented as normal-route timing. Keep the
standalone `rvllm_metal_infer` totals as the independent normal-route view.

Minimum jobs after implementation: five BF16 jobs, one per requested length,
each containing prefill and decode shapes. There are no honest model-wide Q4
or Q8 jobs to submit until those routes exist.

### 3. Extend ANE accounting without changing the route

For normal-route decode, split the existing accumulators by sliding/full layer
kind and add explicit buckets for embedding read/decode/scale, post-attention
norm+residual, remaining norm/RoPE/scale work, final norm, vocabulary
projection, softcap, and ranking. Preserve the present totals and assert that
the parts sum to `total_ms` within clock-resolution tolerance.

Keep the current fused FFN as one bucket. Its valid MLX comparator is the sum
of gate/up+activation and down. A split ANE diagnostic requires new programs;
if built, it must use the same actual weights and be labeled isolated rather
than normal-route.

Minimum supported FP16 decode jobs: three, at exact starting contexts 256,
512, and 1024. Current ANE capacity makes 2048 and 4096 explicit unsupported
cells. No ANE-prefill job should be synthesized: prefill is Metal in the
current disaggregated architecture. Mixed LUT4/INT8 plans may be run as
separate named lanes, but they do not fill model-wide Q4/Q8 cells.

### 4. Queue end-to-end matched pairs

For every supported format/length/backend cell, queue a matched MLX and rvLLM
pair with identical semantic work. Use the kernel-game process-isolation and
thermal-stratum rules. The established controlled comparison unit is four
warmup arms followed by five complete ABBA blocks, hence 24 process arms per
pair cell. Do not pool strata, discard slow observations, or retry after
latency is visible.

The immediately actionable end-to-end set is therefore:

| Comparison lane | Supported lengths | Pair cells | Minimum controlled process arms |
| --- | --- | ---: | ---: |
| MLX BF16 vs rvLLM Metal BF16 | 256, 512, 1024, 2048, 4096 | 5 | 120 |
| MLX BF16 vs rvLLM Metal-prefill/ANE-FP16-decode | 256, 512, 1024 | 3 | 72 |

Those counts are comparison arms, not stage-runner jobs. They assume one
sealed workload per cell and the existing four-warmup/five-ABBA-block policy.
The following cells must remain blocked, not queued under misleading names:

- Metal model-wide Q4 and Q8, pending real model-wide routes;
- ANE model-wide Q4 and Q8, pending real model-wide routes;
- ANE prefill at every length, pending an ANE prefill implementation;
- ANE decode at 2048 and 4096, pending capacity support.

### 5. Qualification order

For each new route or format: delivery and identity, compile/link, exact
dispatch, bounded correctness oracle, full-route multi-token/KV correctness,
zero-compile hot path, then controlled timing. Stage results and end-to-end
results should be joined only by sealed model/workload/build identities. A
fast isolated operator cannot promote a route whose full-run output is wrong,
and a correct first token cannot establish throughput.

## Source and receipt identities

Audit tree: `8edf8c8fea3b74bdadc27ce7c499eeb8e45991e1`.

| Artifact | SHA-256 |
| --- | --- |
| `crates/rvllm-runtime/src/bin/rvllm_metal_infer.rs` | `99e7270647fc821023a6a39d7010af469d1065765352edc34d9b0d703796327c` |
| `crates/rvllm-runtime/src/bin/rvllm_disaggregated_infer.rs` | `7914c9ae99b5b51be306acc45aa48d98cc81bac9c1a3c5e325945481a9c40508` |
| `crates/rvllm-runtime/src/gemma_ane_decode.rs` | `6eb2165000b36db999f0f94e42b2a5841a1dbff537185904042ce00a39c2dd2b` |
| `crates/rvllm-runtime/src/apple_metal_backend.rs` | `faa08cdab67c60b4c8eb5ccdacba7dbfc4a88091fa1e7b6adbfef93d4970dd68` |
| `reports/gemma4-load4-tournament-20260924.md` | `3be968665d2a1f23a0731c8b870efc9132eeed09acd36bc32e41f7dd8ce308cb` |
| `reports/gemma4-ane-interleaved-timing-20260924.md` | `f952b6d74fb08f742b5c5573d9e3ff1850694fb142af90a709ec07c156936f5a` |
| `reports/gemma4-mlx-stage-matrix-20260924/README.md` | `be68071107e3e4255eb58485f2705acfd0e9cf11255dbe2647ce35da697fd2b8` |
| `tools/mlx_gemma4_stage_bench.py` | `bca9006307050bbc78c6d0d4119f9de0812d9d9b7da476489245026684b38b85` |
| `specs/kernel-game/MLX_GEMMA4_STAGE_MICROBENCH.md` | `9d57129027016fc10615b4038129e997a6f423629372631456df1bdc0a379794` |

MLX timing-protocol provenance is core MLX commit
`c215b6f88cf0fee0b0895623e4046cda797ef397`,
`benchmarks/python/time_utils.py`. MLX-LM model-implementation provenance is
commit `87b7b583a697537aa68f47130b40884700b5f55f`.

## Bottom line

Current receipts can answer whether selected BF16 Metal candidates and mixed
ANE plans execute correctly and whether a small controlled candidate contest
found a win. They cannot yet answer “our best 4/8/16-bit Metal and ANE kernels
versus MLX at 256–4096 tokens” without qualification. The shortest credible
path is to fill the five BF16 Metal stage/end-to-end cells, fill the three
supported FP16 ANE decode cells, and keep every absent model-wide low-bit or
unsupported-context cell visibly blocked until the implementation exists.
