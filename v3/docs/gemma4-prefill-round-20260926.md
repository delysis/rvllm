# Gemma 4 12B BF16 prefill tournament — exact-base handoff

> Local qualification addendum, September 27: the original delivery's
> periodic raw-FP32 projection allowance rejected both the matched load4
> control and the lookahead candidate at M6. Their entire 23,040-element raw
> output was byte-identical and matched an independent CPU sequential-FP32
> calculation bit-for-bit. The historical failed receipts are retained. This
> branch's referee revision requires exact declared sequential-FP32 results
> plus the standard `gamma_K * sum_abs_products + 1e-5` bound against FP64 for
> that periodic raw-norm fixture. It does not retroactively pass the old gate,
> alter production quality policy, or establish full-model numerical parity.
> Independent correctness jobs no longer depend on unrelated screen arms;
> ordered timing jobs remain serial. See the dated local qualification report
> for actual device outcomes and sealed artifact identities.

**Base:** `delysis/rvllm`, `fb5f169cae18eb492852e6bf668130e80c347cfb`, PR #6, branch `codex/gemma4-donor12b-pr4-integration-20260926`. Source analysis began September 26, 2026. This is a reviewable, **default-off experimental implementation**, not a production promotion or a measured speedup.

## 1. Delivered implementation and acceptance boundary

The patch adds three bounded implementation choices and a combined selector: a matched load4 control, a software-lookahead GEMM, and four-query/sixteen-key paged prefill attention. They are integrated into the real layer-forward path, not merely standalone kernels. A safe-Rust referee exports source, generates independent full-output operator fixtures, checks returned bytes and identities, and generates serial queue manifests. A separate Swift Metal driver executes those operator fixtures using public Metal APIs; it is not an inference backend or an in-process bridge.

All new Rust files forbid unsafe code. Encoding additions reuse existing unsafe Metal boundary functions; there are no new unsafe Rust functions or blocks. No dependency, model-package ABI, quantization policy, safety threshold, ANE route, production default, or existing selector identity is changed. Existing candidate records remain an identical 48-record prefix. Sixteen new kernel identities append to the existing 86-slot dispatch registry; the dispatch schema becomes `rvllm.metal.research-dispatch.v6`, with 102 entries and 52 selectors including off. Consumers must understand the new schema before relying on its receipts.

**Actually executed in this environment:** eight Python catalog tests; eight supplemental Python source/geometry/algebra tests; Swift syntax parsing; compilation and execution of the Swift driver's Linux-only refusal branch; shell syntax checks. The Linux executable refuses execution with exit 2. Delivery validation separately records patch application/reversal and byte/hash checks.

**Not executed:** Rust compilation, the 21 authored Rust tests, rustfmt/Clippy, full repository CI, Apple Swift type checking, Metal compilation, Metal execution, generated-code/register inspection, normal-route checkpoint oracles, or timings against rvLLM/MLX. This environment has neither Cargo/rustc nor an Apple GPU/SDK. Source-level checks are not substitutes for those gates. The complete changed-source archive is not a complete repository checkout.

**Remaining implementation boundary:** the operator producer/referee is implemented; a new route-preserving per-layer/KV capture producer is not. Existing layer tracing changes routing and must not be represented as that producer. Full checkpoint internal/teacher-forced quality comparison remains a blocking promotion requirement. The queue generator covers native operator correctness and timing, not an automatic model-quality or MLX tournament. Existing model/MLX commands are provided below, with their limitations.

## 2. Source-bound diagnosis and the important rounding correction

The supplied donor integration report shows candidate projection counts at long prompts corresponding to decode, not accelerated prefill. Its SG8 prefill observations were approximately 15.89/30.12/59.17/131.87 seconds at M=256/512/1024/2048. They are not isolated operator timings. The reported failed timestamp attempt supplies no stage attribution; this patch does not infer a timing breakdown from zero counter samples.

The exact base exposes two concrete targets:

* The long-prefill O/down fallback `gemm_rmsnorm_f16` launches one threadgroup per token and serializes projection columns across its threads. It does not exploit a two-dimensional matrix output tile.
* Existing MMA/load4 host eligibility is capped at 1024, and selection requires the separate existing MMA permission. Existing scalar prefill attention and its large-head work are additional candidates for independent measurement.

These are source observations, not evidence that a particular stage accounts for a measured percentage of total time. For orientation, a local layer's QKV/O/FFN projection work is 224,133,120 multiply-accumulates per prompt token; a global layer's is 243,793,920. Across 40 local and 8 global layers this is 10,915,676,160 MACs/token. The FFN portion is 8,493,465,600 MACs/token, about 77.8% of these **projection operation counts**, not 77.8% of latency. This motivates starting with gate/up and down, while timing QKV/O/attention independently.

### O/down normalization is not uniformly a BF16 materialization boundary

Read exact-base `src/kernels.rs::gemm_rmsnorm_f16` (Git blob `06a544fd58627472799308c88ed4a0a36d70e344`). The normal fused kernel stores its projected values in **FP32 threadgroup memory**, computes RMS from those FP32 values, multiplies by gamma, and only then converts to BF16. Replacing this with BF16 GEMM followed by RMSNorm silently introduces an extra rounding boundary.

For large prefill, this patch instead performs:

`BF16 A/W -> FP32 tiled projection -> FP32-input RMSNorm -> BF16 normalized output`.

The raw projection uses a distinct entry point, not a mislabelled QKV dispatch. It reuses the existing FP32 `scratch.qkv_out` allocation after its projection/norm/RoPE/cache consumers have finished. It checks raw/output/gamma separation, input/output separation, full extents, alignment, shape, precision and both pipeline resource contracts before encoding. Refusal retains the original fused kernel. No raw-norm refusal is silently converted into the older BF16 split-norm path.

**Short-prefill exception:** at M≤32, the exact normal route already selects a materialized BF16 O/down projection for these shapes. The new full route retains that boundary; raw-norm replacement is used only at M≥33. Direct raw-norm component fixtures at M=6 and 17 still test the large-prefill operator against its fused FP32 mathematical contract; they do not claim to reproduce the normal short-prefill routing. The host test `prefill_round_short_norm_retains_the_existing_bf16_boundary` records this distinction.

The raw normalization uses the same 256-lane sum-of-squares reduction topology as the fused kernel. Matrix multiplication uses SIMD matrix accumulation rather than scalar dots, so arithmetic may differ even though storage boundaries agree. **No bitwise equality with the normal scalar reduction is claimed.**

### Other preserved boundaries

QKV is written as FP32 into the existing four-byte projection scratch. The existing `qkv_projected_rmsnorm_rope_cache_f16` then remains responsible for headwise norms, BF16 storage conversions, partial RoPE, transformed K/V and slot writes. Q/K are not prematurely BF16-rounded before their headwise norm. Gate/up is BF16-materialized before the existing GELU/multiply kernel; activation and residual/scalar boundaries are unchanged. Authenticated low-bit projections keep their existing separate branches; none of these candidates reads packed W4/W8 storage as BF16.

The new attention kernel consumes the existing separate, transformed, paged BF16 K and V caches. It neither recomputes RoPE nor writes/repacks KV. Existing command-buffer ordering and cache producer ownership remain in force.

## 3. Selectors, shapes, identities and coverage

The immutable owner captures the selector using the existing `RVLLM_METAL_RESEARCH` path. No new environment reads occur during encoding. These four selectors are independent of SG8: **decode retains the incumbent path**, and no decode operation can populate the new prefill slots. There is no automatic best-arm selection.

| Selector | Replaced prefill work | New slots, in its declared kernel order |
|---|---|---|
| `metal-prefill-load4-control` | Matched 32×64×32 GEMM and large-prefill raw O/down normalization | GEMM 86, QKV 87, raw projection 96, raw norm 97 |
| `metal-prefill-pipeline32x64` | Same geometry and accumulation sequence, with next-panel lookahead | 88, 89, 98, 99 |
| `metal-prefill-q4k16` | Paged D256/D512 attention only | D256 90, D512 91 |
| `metal-prefill-pipeline32x64-q4k16` | Pipeline projection plus Q4K16 attention | GEMM 92, QKV 93, D256 94, D512 95, raw projection 100, raw norm 101 |

The old slots 0–85, names and ownership are unchanged. The checked catalog lists every exact entry symbol and resource budget. Actual dispatch counters are incremented only after encoding; a selector setting alone is not proof of participation.

**Model gate:** native BF16, quantized accumulation disabled, Apple9 or Apple10 reported by the existing owner, hidden 3840, intermediate 15360, 48 layers, 16 query heads, no MoE or per-layer embedding branch, and the supported KV/head-dimensional tuples. Missing functions, unexpected SIMD width, inadequate queried limits, incompatible phase, shape, alignment, span or aliasing reject the candidate. Existing selectors retain their old bounds.

| Operation | M | N / K | Storage contract |
|---|---|---|---|
| Local QKV | 6–2048 | 8192 / 3840 | BF16 inputs, FP32 output |
| Global QKV | 6–2048 | 9216 / 3840 | BF16 inputs, FP32 output |
| Combined gate/up | 6–2048 | 30720 / 3840 | BF16 output, then existing GELU/multiply |
| Plain local/global O | 6–2048 | 3840 / 4096 or 8192 | BF16 output |
| Plain down | 6–2048 | 3840 / 15360 | BF16 output |
| O/down feeding postnorm | Full route 33–2048; raw component screen 6–2048 | Same O/down matrices | FP32 raw output, then BF16 normalized output |
| Local attention | 6–2048, batch 1 | QH16 / KVH8 / D256 / W1024 | BF16 Q/K/V/output; FP32 scores, probabilities and running state |
| Global attention | 6–2048, batch 1 | QH16 / KVH1 / D512 / W0 | Same storage contract |

Projection alpha must equal 1 and beta 0. A/B vector loads require eight-byte alignment. Output alignment follows its two/four-byte type. M tails are guarded; N and K are exact supported shapes, not arbitrary padded matrices. Attention requires scale 1, trace off, and checked logical capacity between M and 4096. Capacity, not only current live length, is bounded. Live positions, context and page IDs are checked in the shader; negative page IDs mean holes, while an invalid visible nonnegative page produces NaN without being dereferenced.

**Large-prefill actual coverage:** a single untraced, dense, one-chunk M=256/512/1024/2048 prefill through all 48 layers should record 48 new gate/up GEMMs, 48 QKV projections, 96 raw O/down projections and 96 raw norms under either projection selector. Attention should record 40 D256 and 8 D512 dispatches. Combined should record both sets. New counters during subsequent decode must be zero. These expectations do not apply unchanged to M≤32, trace mode, low-bit substitutions, alternate model branches or externally chunked prompts.

The historical encoder estimator carries the expanded projection permission and can overestimate the successful research path when a later buffer/PSO guard falls back. It is **not** an independent actual-encoder counter. Use new per-kernel dispatch counts and device command capture for qualification, not this estimate.

## 4. Kernel design and resource hypotheses

### Matched projection control versus pipeline

Both variants compute C=A×Bᵀ with a 32-row, 64-column, K32 tile and 128 threads. Four SIMDgroups are arranged 2×2; each owns a 16×32 output region, represented by eight 8×8 FP32 accumulator fragments. Input staging uses contiguous four-element loads and no checkpoint repacking.

The control stages the current tile directly. The pipeline carries the next A/B panel in private vector arrays while the current panel is consumed. There is no asynchronous-copy API or undocumented hardware assumption: source ordering only creates an opportunity for the compiler to overlap loads with arithmetic. Both arms retain identical K/8 multiply-accumulate ordering. Their direct projection outputs must match bitwise before attributing a difference to scheduling.

The A/B shared panel consumes 6144 bytes. After the final reader barrier the same aligned allocation holds an 8192-byte FP32 output tile. This is **8192 bytes total**, not the sum of both lifetimes. Vector staging, fragment ownership and M-tail output coverage are host-tested. Barriers are uniform across the workgroup, including partial M tiles.

Source-level output state is 16 FP32 values per lane. Lookahead additionally retains 24 BF16 source values per lane, plus matrix operands, loop/address state and temporaries. These are **not physical register counts**. Whether the compiler eliminates control-arm private arrays, hoists loads, emits vector memory instructions, spills state or changes occupancy is unknown until generated-code/device inspection.

### Attention Q4K16

Each 128-thread workgroup handles four temporal query positions for one query head. One SIMDgroup owns one query and walks the causal extent in 16-key tiles. Its dot products reduce across lanes; per-tile FP32 softmax updates running maximum, denominator and output numerator. K and V reuse one BF16 shared panel with barriers separating their lifetimes. Scores/probabilities are never cast to BF16, and output is rounded once.

The union of four absolute causal windows determines the panel range; each query retains its own mask. The newest token is included. A speculative suffix in the cache is excluded. Missing pages do not contribute to the denominator. An all-hole visible interval produces zero. Invalid live metadata or an out-of-range visible physical page produces conspicuous NaNs instead of a plausible result. The final partial four-query group remains barrier-safe.

| Resource | Projection | D256 attention | D512 attention |
|---|---:|---:|---:|
| Threads / SIMDgroups | 128 / 4 | 128 / 4 | 128 / 4 |
| Source shared budget | 8192 B | 8512 B | 16704 B |
| Source output state per lane | 16 FP32 values | 8 FP32 values | 16 FP32 values |
| Resident query state per lane | Operand fragments; compiler-dependent | 8 FP32 values | 16 FP32 values |
| New global partial/output allocation | None | None | None |

These budgets are source requirements; the driver also queries compiler-reported static threadgroup memory, maximum threads and execution width. The existing owner checks its device memory limit. These checks establish eligibility, **not occupancy**. No register-file size, active-group count, cache hit rate or performance ratio is asserted.

| Threadgroups per layer/operator | M256 | M512 | M1024 | M2048 |
|---|---:|---:|---:|---:|
| O or down projection | 480 | 960 | 1920 | 3840 |
| Gate/up projection | 3840 | 7680 | 15360 | 30720 |
| Local QKV | 1024 | 2048 | 4096 | 8192 |
| Global QKV | 1152 | 2304 | 4608 | 9216 |
| Q4K16 attention, either head dimension | 1024 | 2048 | 4096 | 8192 |

The raw O/down projection requires 30 MiB at M2048 inside the already reserved QKV scratch. The existing gate/up and activation allocations remain 120 MiB and 60 MiB at that M. No reduced-memory claim is made. Arena-span checks remain authoritative; do not infer an allocation from a table entry.

**Why no further speculative kernel:** no split-K projection, TensorOps/NAX dependence, ANE offload, precision-lowered softmax or matrix-fragment manipulation is introduced. The small family isolates output tiling/lookahead and cooperative attention first. A failed or spill-heavy lookahead arm can be rejected without discarding the matched tiled control or attention candidate.

## 5. MLX comparison: actual code eligibility, not kernel-name inference

Source-analysis pin: official MLX commit **`a2a09fd56ccf064121f489528d339de40d2ba8b3`**, dated September 26, 2026. This is a new immutable analysis pin, **not an authenticated identity for the historical MLX-LM timing observations**. The supplied report does not establish that historical binary/source relationship.

Relevant exact primary sources:

- `mlx/backend/metal/matmul.cpp`, blob `5fc2c193613963e128119aa4bfb23254a9c4ec40`: [pinned source](https://github.com/ml-explore/mlx/blob/a2a09fd56ccf064121f489528d339de40d2ba8b3/mlx/backend/metal/matmul.cpp).
- `mlx/backend/metal/scaled_dot_product_attention.cpp`, blob `4a28891bf6fa8f264d1fc7032b911a8bf6c154ff`: [pinned source](https://github.com/ml-explore/mlx/blob/a2a09fd56ccf064121f489528d339de40d2ba8b3/mlx/backend/metal/scaled_dot_product_attention.cpp).
- `mlx/backend/metal/device.cpp`, blob `2f25f894e494a4328886148bb6e3332d9f9b298a`: [pinned source](https://github.com/ml-explore/mlx/blob/a2a09fd56ccf064121f489528d339de40d2ba8b3/mlx/backend/metal/device.cpp).

MLX Steel matmul chooses geometry by architecture class, dtype, transpose state and problem size. Kernel names encode transpose/input-output type/BM/BN/BK/WM/WN, while function constants specialize batch handling, output-source/axpby use and alignment. Its regular grid exposes both output dimensions. The non-NAX policy includes 64×64×16 and 64×32×32 choices; split-K has explicit small-output/large-K/device thresholds. This supports testing output tiling and specialization, but does not establish that MLX's selected geometry should be copied unchanged into rvLLM's storage/stride ABI.

NAX eligibility is itself conditional. The pinned `is_nax_available` returns false under `MLX_METAL_NO_NAX`; otherwise it checks OS availability at 26.2 and an architecture-generation threshold of 18 for suffix `p`, 17 otherwise. Matmul additionally gates complex and FP32/TF32 cases. Do not infer the M4 Max's internal architecture-generation value or an ANE execution path from the marketing name. Record actual runtime eligibility and generated kernel identities.

For D512 with query length >8, MLX's fused attention eligibility additionally requires the NAX/dtype path. Its normal fallback decision imposes query length≥1024, causal attention without an array mask, and at least 1024 query blocks where blocks=B×H×ceil(M/32). With B1/H16, the four requested lengths yield 128/256/512/1024 blocks. Thus only M2048 passes that block threshold, **conditional on all other gates**. Existence of a D512 fused source is not evidence that M256/M512/M1024, or this machine, actually dispatches it.

The existing exact-base `tools/mlx_gemma4_stage_bench.py` is a useful independent operator-orientation tool. It does not produce a normal-route stage breakdown and does not use the same synthetic fixtures as this referee:

```sh
python3 tools/mlx_gemma4_stage_bench.py \
  --mlx-source "$MLX_SOURCE" --mlx-lm-source "$MLX_LM_SOURCE" \
  --model "$MODEL" --weight-bits 16 --lengths 256,512,1024,2048 \
  --plan-only
# Omit --plan-only only for a separate queue-owned measurement job.
```

Its QKV category stops before headwise normalization/RoPE; FFN and attention categories have their own precomputed inputs. Their values must not be divided by this referee's synthetic times and labelled a framework speedup.

A fair MLX gate requires the same checkpoint bytes, config/tokenizer, prompt IDs/BOS/chat template, live cache length, attention masks, dtype/rounding contract and work boundary. Record MLX and MLX-LM commits, loaded extension hash, compiler/math options, runtime architecture and actual selected pipelines. Compare identical operator input bytes separately from whole-model throughput. Ordinary MLX BF16 linear output is not automatically a reference for rvLLM's FP32 QKV or fused raw-norm boundary. Match first-token accounting and prefill chunk size. Whole-model decode comparisons require the same generated length (for example 64), not two tokens versus 64. Interleave processes ABBA-BAAB with both frameworks resident only when explicitly intended and matched. These comparisons are specified, **not measured or automatically produced here**.

## 6. Referee, independent oracles and negative controls

The new binary is `rvllm-prefill-round`, automatically discovered under the existing Metal crate's binary layout. Its sources are `src/bin/rvllm-prefill-round.rs` and `src/bin/prefill_round/{fixtures,queue}.rs`. It never runs from the inference hot path.

The correctness ladder is **6 → 17 → 64 → 256 → 512 → 1024 → 2048**. A screen runs every required fixture at each rung before entering the next. One invocation owns one arm and one operator role. A failure retains its output and progress and is never silently retried. A timing sample requires a previously completed screen whose source, driver, library, referee and cell identities still match.

Projection fixtures use exact real matrix dimensions and full output coverage. Structured rank-two inputs have an independent FP64 closed form for every output element; they catch layout, tiling, column and tail mistakes without a huge CPU GEMM. Additional M6/M17 periodic dense BF16 fixtures exercise signs, exponent variation, cancellation and long K. They use independent FP64 dot products and absolute-product sums. These are **structured/periodic synthetic weights**, not a full-rank random or real-checkpoint qualification.

Pure structured projections require exact stored results. Periodic FP32 comparisons use `1e-5 + 2e-6*sum_abs_products`, with an additional `abs(reference)/256` allowance for BF16 output. Normalization/GELU are evaluated with their explicitly declared storage boundaries. Intermediate GEMM materializations, not only final normalized values, are checked. Tolerances are new tournament-local synthetic screen rules; they are not changes to existing safety/quality thresholds and are insufficient to bless real checkpoints.

Attention fixtures use restored absolute prefixes, permuted physical pages, independently stored BF16 V, speculative suffixes, negative holes, all holes, newest-token spikes and short dense-Q cases. They compare every output against independent FP64 stable-softmax values; the BF16 screen allowance is `5e-5 + abs(reference)/256`. Malformed visible pages and metadata must poison the output; bad launch geometry must leave output untouched. Each buffer has 32-byte guards, and read-only inputs are checked before and after execution.

The scalar incumbent has no negative-page safety check, so the referee deliberately does **not** send holes or malformed pages to it. The incumbent SIMD arm gets valid holes but no invalid positive-page test. The candidate must still pass its independent hole/invalid-metadata tests. Unsafe incumbent behavior is not an oracle requirement.

Supported roles are `qkv-local`, `qkv-global`, `gate-up`, `gate-gelu`, `o-local`, `o-global`, `down`, `o-local-norm`, `o-global-norm`, `down-norm`, `attention-local`, `attention-global`.

### Control arm semantics

`normal` uses scalar GEMM for plain matrices, the actual fused FP32 GEMM/RMSNorm for `*-norm`, and batch8 FP32 QKV for the QKV component. The latter is **not** the whole fused normal QKV/norm/RoPE/cache route. `mma32` invokes the existing matrix component; its shader can be tested at M2048 even though its normal host selector remains capped at 1024. `metal-mma32-load4` retains its shader's M≤1024 restriction; at M2048 the referee uses two separately offset 1024-row operator chunks sharing the same weights. That is a chunked control, not a newly qualified native-M2048 load4 route. Legacy MMA/load4 `*-norm` includes BF16 materialization before RMSNorm and has a separately modelled precision contract. Do not use it to approve the new raw-norm arithmetic.

### Source binding, compilation and timing

`source-all` uses the crate's actual BF16 source generator for all eight arms and appends a small identity entry containing SHA256 of the source body. The driver reads that identity and verifies it **before any workload kernel**. The referee also checks the generated source, source manifest, metallib and driver hashes, then rechecks them after execution. This is ordinary artifact integrity under a trusted compiler/host, not adversarial attestation of machine code.

The driver loads the metallib and creates pipelines before timed work. It checks actual command-buffer completion, queries pipeline/device limits, performs first-result readback, two warmups, nine timed operator chains, and final readback. All first/last mutable outputs must repeat bitwise; input and guard bytes must remain intact. Driver API call counters wrap library loading/pipeline creation; source compilation is absent by construction. They do **not** claim to observe internal driver compilation. They are also not new counters for the separate production inference executable.

GPU duration comes from each independently timed operator command buffer. Zero, nonfinite or nonpositive duration invalidates timing. CPU encode, wait and wall times are retained as distinct values; no failed stage-counter samples are transformed into a stage estimate. Guard checking, hashing, compilation, allocation, oracle evaluation and first readback are outside the nine measured intervals.

The referee uses a create-new exclusive lock and owns/reaps its one native child. It flags a 600-second overrun but **does not kill a Metal owner and release the lock while work may remain active**. It waits for safe child exit, records failure, and does not retry. Hardware hangs cannot be recovered by this user-space protocol. The referee lock must differ from the outer queue's accelerator lock; otherwise the child would conflict with its parent.

## 7. Exact local procedure

### A. Apply and inspect

Use a fresh worktree at the exact base, not a moving branch tip. Do not overwrite another active worktree's uncommitted SG8 diagnostic work.

```sh
git worktree add --detach /ABS/rvllm-prefill26 fb5f169cae18eb492852e6bf668130e80c347cfb
cd /ABS/rvllm-prefill26
git rev-parse HEAD
# Verify the delivery's PREIMAGES.json and checksums first.
git apply --check /ABS/DELIVERY/rvllm-prefill-fb5f169c.patch
git apply /ABS/DELIVERY/rvllm-prefill-fb5f169c.patch
git diff --check
cd v3
bash tools/prefill-round/host-gates.sh
```

Run rustfmt/Clippy as additional review gates using the repository toolchain. Formatting was not run in the delivery environment. No host check should select an ignored hardware test on a non-Apple machine. The existing full catalog/source-export CI runner must also pass on the complete repository; only its affected count fixtures were updated here.

### B. Freeze and build strict artifacts

```sh
cargo build --offline --locked --release -p rvllm-apple-metal --bin rvllm-prefill-round
REFEREE="$(pwd)/target/release/rvllm-prefill-round"
ARTIFACTS=/ABS/NEW/prefill26-strict
bash tools/prefill-round/build-apple.sh "$REFEREE" "$ARTIFACTS"
DRIVER="$ARTIFACTS/MetalArm"
```

The build script exports all eight sources, records toolchain versions, compiles the Swift driver, then uses `metal -std=metal3.1 -fno-fast-math` and `metallib`. Keep generated Metal, AIR, metallibs, compile/link logs, Swift source/binary, toolchain record and BUILD-SHA256SUMS. Any compile failure stops the build; do not change compiler math flags merely to get a result.

Before speed work inspect generated code or an Xcode GPU capture: verify matrix instructions, actual private spills, load width, barriers, source-vs-reported shared allocation and dispatched grids. Reject unverified occupancy claims. Profile one operator arm at a time; profiling and timing are different runs.

### C. Short screens, then long screens

The following direct invocation is useful for a single manual smoke **only while holding the shared accelerator owner through the existing queue**:

```sh
"$REFEREE" screen metal-prefill-pipeline32x64 down-norm 64 \
  "$DRIVER" "$ARTIFACTS" /ABS/NEW/down-short /ABS/locks/prefill-referee.lock
```

Prefer generating native queue jobs. Copy `screen-config.example.json` to a local config; replace every `/ABS/...` with real absolute paths, set a unique campaign/root, and keep a distinct referee lock. The generator computes actual input hashes; example configs are not runnable pre-sealed manifests.

```sh
"$REFEREE" screen-jobs /ABS/screen-config.json
cargo build --offline --locked -p rvllm-runtime \
  --features macos-private-ane-research --bin rvllm_experiment_queue
QUEUE_BIN="$(pwd)/target/debug/rvllm_experiment_queue"
for manifest in /ABS/NEW_CAMPAIGN/jobs/*.json; do
  "$QUEUE_BIN" submit /ABS/queue "$manifest"
done
"$QUEUE_BIN" run /ABS/queue /ABS/locks/shared-accelerator.lock 600
"$QUEUE_BIN" status /ABS/queue
```

The generated schema is the existing `rvllm.experiment_job.v1`: pinned executable/cwd/argv/env, hashed inputs, observed condition policy, `stable_seconds:0`, and bounded job declarations. Independent correctness arms have no cross-arm `after` dependency; the existing queue serializes accelerator ownership. ABBA/BAAB timing jobs do have serial `after` dependencies to preserve their declared order. No shell is invoked by the Rust referee. The shown shell loop only submits immutable manifests. Run the queue according to its existing ownership procedure; no task was queued or run by the original delivery.

Full-size fixture/readback retention can consume several GiB per arm/role. The example 16 GiB free-space floor is not a campaign-capacity estimate; budget disk explicitly and submit small groups rather than an unbounded all-role sweep. Never delete inputs while pinned jobs are pending.

Start with down-norm and gate-gelu for normal/matched-control/pipeline, and both attention dimensions. Complete all remaining QKV/O/down roles and old MMA/load4 controls before combined full-route experiments. After a limit64 screen passes, submit a new root/campaign with limit256. Repeat for 512/1024/2048; each screen rechecks the shorter rungs before advancing. Do not edit or relabel earlier receipts. For routine development one can screen to a chosen maximum in one invocation, but every preceding rung remains mandatory internally.

### D. One-arm timing jobs and adjudication

Use `timing-config.example.json` with both completed screen paths, the target M, role and frozen artifacts. For the first scheduling test use A=`metal-prefill-load4-control`, B=`metal-prefill-pipeline32x64`.

```sh
"$REFEREE" timing-jobs /ABS/timing-config.json
for manifest in /ABS/NEW_TIMING_CAMPAIGN/jobs/*.json; do
  "$QUEUE_BIN" submit /ABS/queue "$manifest"
done
"$QUEUE_BIN" run /ABS/queue /ABS/locks/shared-accelerator.lock 600
"$REFEREE" adjudicate /ABS/NEW_TIMING_CAMPAIGN/adjudication.json \
  /ABS/NEW_TIMING_CAMPAIGN/adjudicated.json
```

This produces **eight distinct queue jobs in ABBA-BAAB order**, each executing one arm for one operator/M, with explicit dependencies. It is not one mixed-arm process hidden behind a single queue receipt. A sample requires a completed matching correctness screen. The standalone `compare` convenience command is not the queue protocol and should not replace these one-arm manifests.

Adjudication checks order, complete receipt coverage, unchanged source/driver/referee/library identity, raw output hashes and equal read-only fixture bytes. Matched new projection schedules must agree bitwise. Each process contributes the median of its nine GPU samples; report all eight process medians, not a pooled best sample. A new local 5% max/min within-arm drift gate rejects unstable comparisons while retaining every observation. The descriptive ratio is A/B, not an automatic promotion.

**External conditions are deliberately not self-certified:** adjudication outputs `conditions_qualified:false`. Review all eight outer queue receipts, power/thermal observations, competing processes, lock participation and overruns before qualifying a timing result. Missing AC/power observations or different strata remain ineligible. Existing queue conditions observe, rather than guarantee, clocks/stability.

### E. Actual normal-route prefill

Build an Apple-feature inference binary from this exact patched full tree, freeze/hash it, and set `INFER` to that absolute executable. Use the already supported native CLI, not the old donor wrapper: that wrapper's selector whitelist excludes these new selectors.

Select exactly one existing JSONL prompt case of the desired length from `reports/gemma4-rvllm-good-enough-matrix-20260924/prompts.jsonl`. Verify the actual token-ID count before launch. The fifth case is 4096 and is outside this tournament. Avoid concatenating contexts into a supposedly single-M timing arm.

```sh
# MODEL, INFER, ONE_PROMPT_JSONL and REPORT are absolute, pinned local paths.
# Run as one explicit native queue command with these environment values.
/usr/bin/env -u RVLLM_METAL_PREFILL_GEMM -u RVLLM_METAL_PREFILL_ATTENTION \
  -u RVLLM_METAL_QKV_PREFILL -u RVLLM_METAL_BF16_ACCUM \
  RVLLM_METAL_RESEARCH=metal-prefill-pipeline32x64-q4k16 \
  RVLLM_METAL_METALLIB_BF16="$ARTIFACTS/metal-prefill-pipeline32x64-q4k16.metallib" \
  "$INFER" --model-dir "$MODEL" --prompts-jsonl "$ONE_PROMPT_JSONL" \
  --session-backend direct --max-new-tokens 2 --max-total-tokens 2050 \
  --large-model-opt-in --report "$REPORT" --json
```

For a queue manifest put the executable directly in `command.executable`, the CLI arguments in `args`, and the two selectors in `env`; the queue clears inherited environment already. Do not use `/usr/bin/env` as the pinned executable instead of pinning INFER. Two decode tokens here are only a bounded initial prefill/continuation screen. Later use equal 64-token continuations and increase the supported total-token limit accordingly; do not interpret two-token decode as sustained throughput.

Compare projection-only, attention-only and combined against selector-off with the same binary/checkpoint and an explicitly documented metallib choice. Confirm the new counts in section 3 **during prefill**, zero new decode counts, and reported inference compilation deltas. Add independent actual-API instrumentation or capture for the production executable before claiming independently verified zero compile calls there; this driver only supplies that accounting for its operator arms.

## 8. Falsifiable acceptance, rejection and promotion

**Source/compile admission:** all 21 new Rust tests, full Metal-crate all-target check, Apple runtime check, full catalog export contract, strict compilation of all source arms, Swift Apple type checking, source identity preflight, and queried pipeline/device limits must pass. Missing PSOs are not a successful experiment. Preserve every failure and exact artifact identity.

**Numerical admission:** complete independent synthetic outputs, guards and read-only inputs; correct short/long storage boundaries; tail coverage; hole/newest/prefix/suffix behavior; invalid-metadata/geometry behavior; bitwise within-arm repeats; and bitwise matched-control/pipeline arithmetic. A toleranced comparison without complete element coverage is not admission. A mismatch stops advancement at its first failing rung. Do not loosen a tolerance to rescue a speed lead.

**Operator timing:** after correctness, run paired schedules against matched control and normal/MMA/load4 components at all four target Ms, with explicit rounding/scope differences. Require valid positive timers, no API compilation in measured operator windows, no overruns, ≤5% within-arm drift, and compatible outer queue conditions. A provisional candidate must beat the best semantically appropriate control in both order blocks; use a predeclared practical threshold such as ≥10% lower median operator time in two independent campaigns, and no >5% regression at another target M. This is a proposed experimental advancement rule, not a production safety policy change. Reject lookahead if it spills/regresses or loses bitwise matched-control equality; the matched tiled control can still advance independently.

**Full-route promotion remains blocked** until actual prefill dispatch coverage and route-preserving internal comparisons establish QKV FP32 values, postnorms, RoPE outputs, transformed logical K/V (including guards, holes and newest writes), attention outputs, FFN activation and residuals against qualified reference behavior on real checkpoint activations. Teacher-force identical continuations before comparing logits; a two-token match or free-running divergence cannot localize the first arithmetic error. Include restored-prefix, chunked-prefill, sliding rollover, global long-context and boundary positions. Use held-out next-token loss/logit-distribution criteria fixed before measurement, not argmax agreement alone. The separate SG8 decode divergence remains a different experiment and is not resolved by this patch.

**End-to-end performance:** only after the quality gates, compare prefill-to-first-token and complete prefill intervals with identical work definitions, plus matching sustained decode workloads, in interleaved rvLLM/MLX trials. Report load/preparation/compile costs separately. Do not sum hot-cache operator medians to claim a full-model speedup. Keep production defaults off until correctness, coverage, repeatable full-route improvement and matched framework comparisons are recorded and reviewed.

**Rollback:** reverse this patch on its own clean worktree. Existing selectors and shaders remain intact; no model files or persistent caches were rewritten by the delivery. Keep all trial artifacts and rejected receipts as immutable evidence.
