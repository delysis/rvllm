# PR27: wider projection tiles and FP32 matrix prefill attention

**Implementation base:** `delysis/rvllm` commit
`c009497bee8125d456e4aee26f0b35836d79c231`, branch
`codex/gemma4-prefill-round-20260927`. This is an incremental patch against
that revision, **not** the earlier `fb5f169c` packet and not PR #6's other tip.
All candidates remain explicitly selected research routes. No production
default, quantization policy, numerical acceptance threshold, original shader,
RoPE implementation, KV writer, or checkpoint file is changed.

## 1. What the experiment actually established

The relevant results are the committed `prefill-round-fullroute-short-20260927`
README and MECHANISM report, not the older donor decode report. Read at the
base above, the new combined PR26 route genuinely replaced all 48 layers'
projections and attention. It produced the same complete 64-token trajectories
as control on the two varied prompts, and session receipts reported zero
inference-case library/pipeline-state compilation. This is not fallback-only
success; neither is it an independent full-tensor correctness proof.

The important correction to the headline is **workload position**. M304 as
second case repeatedly measured about 640 ms, with ~19x control/candidate
ratios passing the prescribed ABBA/BAAB drift gate. Moving that prompt first
raised candidate prefill to about 2315 ms and changed the ratio to ~5.95x.
M101 moved the other way: second-case prefill was about 262 ms. The standalone
M304 trial retained candidate samples 2191.719, 2226.328, 2023.257, 2071.590 ms;
its 7.686% candidate drift failed the 5% criterion. Nothing here retroactively
qualifies that result or proposes a favorable retry.

The standalone candidate mean, 2128.223 ms, versus MLX's inferred 659.799 ms,
is the useful ~3.226x planning gap. It is **not** evidence that a particular
GEMM is 3.226x slower: prompt position changes these numbers materially, MLX's
first-token-derived timer differs from rvLLM's launch/collect timer, processes
were not interleaved, and MLX had a stale condition observation. The native
prefill-only selector also leaves decode on the incumbent route; its slow
decode must not be used to assess these new prefill kernels or compared as if
SG8 were concurrently enabled.

MECHANISM reports isolated PR26 lookahead/control down-norm ratios of
1.25x/1.34x at M256/M512 and Q4K16/scalar-attention ratios of 7.27x/6.04x.
Those motivate the next arms, but provide no per-kernel normal-route stage
breakdown. The old zero timestamp samples remain unusable. New operator
measurements use complete command-buffer GPU duration, with wall/encode/wait
fields retained separately; a zero/nonfinite GPU timer rejects timing.

Distribution differences remain real despite matching generated IDs. The
committed M304 one-step top-logit and teacher-forced diagnostics do not certify
either Metal route against a full numerical reference. The c009497b commit
records submission of the separate immutable full-vector-v3 reference jobs,
not their completion. **Do not replay those jobs, overwrite their receipts,
or treat their absent terminal results as approval for this patch.**

## 2. What is implemented

Four new exact selectors append to the catalog:

| Selector | Projections | Attention | New dispatch slots |
|---|---|---|---|
| `metal-prefill-wide64` | 64x64x32, four 32x32 SG output tiles | unchanged | 102–105 |
| `metal-prefill-wide128` | 32x128x32, four 32x32 SG output tiles | unchanged | 106–109 |
| `metal-prefill-mma8k32` | unchanged | Q8/K32, FP32 QK and PV | 110–111 |
| `metal-prefill-wide64-mma8k32` | wide64 | Q8/K32 | 112–117 |

The catalog has 56 candidates and 118 dispatch identities, schema v7. Slots
0–101 and every old candidate's source/resource/shape description are
unchanged. The four PR26 family entries remain individually selectable;
there is no renaming of their bodies or assignment of new work to old slots.
`family.json` and the complete runtime catalog describe the actual names.

The existing `layer_forward.rs` already calls the prefill planner, obtains
PSO limits from the selected entry, uses its projection geometry and attention
grid, and records the actual entry. Extending those pure planners therefore
integrates the candidates into the normal native prefill path without adding
Metal FFI or changing the large runtime file. Existing no-trace/prefill/model,
native-BF16, Apple9/10, PSO/resource, bounds and alias gates still apply. The
new query vector loads additionally require an eight-byte-aligned Q offset.
The ordinary decode path cannot enter these new planners.

### Exact shape and rounding table

| Operation | N | K | Device result before downstream consumer |
|---|---:|---:|---|
| Local packed QKV | 8192 | 3840 | FP32, unchanged projected norm/RoPE/cache consumer |
| Global packed QKV | 9216 | 3840 | FP32, same contract |
| Packed gate/up | 30720 | 3840 | BF16, then existing GELU/multiply |
| Local O | 3840 | 4096 | FP32 before norm at M>32; existing BF16 materialization at M<=32 |
| Global O | 3840 | 8192 | same |
| Dense down | 3840 | 15360 | same |
| Sliding attention | heads16, KV8, D256 | window1024 | BF16 only at final output |
| Global attention | heads16, KV1, D512 | window0 | BF16 only at final output |

Runtime M is 6–2048 inclusive, including guarded M tails. The target timing
points remain 256/512/1024/2048; short fixtures do not claim short-prompt
performance. Projection alpha must be 1 and beta 0. Attention is one prefill
sequence, scale1, exact layer tuple, maximum logical context capacity4096.
Unsupported requests retain existing fallback behavior and fail a claimed
coverage gate even when the model returns successfully.

The large O/down norm route remains `BF16 operands -> FP32 raw projection ->
FP32-input RMSNorm -> BF16`. It reuses the existing dead QKV projection scratch
only where the existing owner permits it. It does not insert an extra BF16
round before norm. M<=32 preserves its different incumbent materialization
boundary. QKV is never sent through the BF16 output overload. RoPE, sliding
window origins, shared-KV ownership, original weight strides and slot mapping
are not reinvented here.

## 3. Projection design

PR26 exposes 2048 outputs per threadgroup, a K32 panel, 16 FP32 accumulator
values per lane, and an FP32 shared-output epilogue. PR27 exposes 4096 outputs,
32 accumulator values per lane, K32 panels with stride40, native-storage
vector-four lookahead, and direct FP32-fragment stores. Each four-SG group
owns either a 2x2 or a 1x4 grid of 32x32 subtiles. The K/8 contraction sequence
and alpha/+0 epilogue are retained. BF16 operands are not narrowed to FP16.

For wide64, doubling prompt-row reuse reduces weight reloads per output in the
source schedule. Wide128 instead doubles output-column reuse of each input
row. These are alternative hypotheses, not an automatic heuristic that calls
either a winner. Both eliminate the old shared FP32 output store/reload and
its epilogue barrier. Padding K32 to40 is a bank-layout hypothesis. It must
not be reported as measured conflict removal.

Direct fragment access follows the **actual MLX Steel** coordinate formula
in `BaseMMAFrag<float,8,8>`, pinned to MLX
`a2a09fd56ccf064121f489528d339de40d2ba8b3`, file
`mlx/backend/metal/kernels/steel/gemm/mma.h`, blob
`1d9e0a7820f1eee9276f7ff23cf96bcd5b8bdae5`. The MIT attribution is included.
This is a source-observed implementation detail, not a portable Metal ABI
promise. Every new operator process must first pass a complete 128-element
matrix-load/product layout probe; failure prevents workload dispatch. The
native model selector itself does not execute a runtime probe—component
qualification precedes any explicit full-route experiment.

MLX's pinned matmul implementation specializes layout, tile geometry,
architecture and alignment through generated names/function constants. Its
NAX routes have explicit availability checks, not universal M4 eligibility.
This patch does not assume TensorOps, NAX, ANE execution or undocumented
hardware throughput. The design-source revision is not asserted to be the
installed binary that produced the retained MLX timing receipts.

## 4. Attention design

PR26 Q4K16 already performs tilewise online softmax; it does **not** rescale
once per individual key. The new improvement hypothesis is replacing its
per-key dot-product reductions and scalar PV accumulation with two matrix
products. One group handles eight causal queries and32 keys. Four SGs first
split score columns (all eight queries by eight keys), then split output-D
columns for PV. Native BF16 Q is staged once. K/V are widened directly into
FP32 fragments using the verified coordinate map; no full D512 K/V shared
tile is required. Scores, probabilities, max, denominator and output state
remain FP32. There is no BF16 probability approximation.

The new source performs a single output rescale per 32-key block. Every row
has its own absolute position and sliding lower bound. The union range is
only a loading envelope; it does not replace per-row causal/window masks.
Negative pages are absent tokens. Nonnegative pages beyond physical capacity
are never dereferenced and poison affected visible rows. Bad live metadata
produces conspicuous NaNs. All-hole rows produce zero. Speculative future
positions are excluded. Read/write overlap and K/V allocation aliasing are
rejected by the existing host planner.

Matrix multiplication adds a subtle masked-value hazard: `0 * NaN` can poison
otherwise unrelated causal rows. The implementation sanitizes nonfinite V
fragment elements, gathers a bad-key bitset, and poisons only rows assigning
nonzero probability to those keys. A dedicated device fixture puts a bad V
value inside the shared tile: the first three queries must remain finite and
the last three must be poisoned. Another fixture uses dense Q and independently
varying V channels, with a complete FP64 reference rather than the old
four-coefficient V pattern alone.

This attention path changes reduction order. Exact token agreement, and even
passing all synthetic cases, does not establish checkpoint-wide quality.
Neither numerical tolerance nor the old precision gates are relaxed here.

## 5. Source resources and occupancy hypotheses

| Member / component | Threads | Source shared bytes | FP32 output state/lane | Next-panel native values/lane |
|---|---:|---:|---:|---:|
| wide64 projection |128|10240|32|32|
| wide128 projection |128|12800|32|40|
| Q8K32 D256 |128|5392|16|not applicable|
| Q8K32 D512 |128|9488|32|not applicable|
| raw norm, unchanged algorithm |256|1024|not a matrix fragment|not applicable|
| layout probe, diagnostic only |32|512|two matrices checked|not applicable|

These are **source allocations/state counts**, not register usage, spills,
occupancy, SIMD utilization or machine-code instruction counts. Aligned
float4 shared backing explicitly supports the vector-four native loads.
Queried PSO static memory, maximum threads, SIMD width and device limits are
recorded and checked separately by the driver.

Both projection tiles expose the same group counts at aligned target M:

| Role | M256 | M512 | M1024 | M2048 |
|---|---:|---:|---:|---:|
| O/down N3840 |240|480|960|1920|
| QKV N8192 |512|1024|2048|4096|
| QKV N9216 |576|1152|2304|4608|
| Gate/up N30720 |1920|3840|7680|15360|
| Attention,16 heads |512|1024|2048|4096|

Larger tiles halve aligned group counts versus the old32x64 projection or
Q4 attention. That may hurt small-shape occupancy or increase register spills;
it is an explicit rejection risk. Wide128 has a larger staged footprint;
D512 matrix attention doubles output state per lane versus Q4. Source-level
reuse does not guarantee speed. No split-K/global-partial scratch or additional
model memory allocation is introduced.

## 6. Referee and queue implementation

New `rvllm-prefill-next` is a separate safe-Rust binary. It reuses the exact
base's deterministic fixture builder and raw-output verifier, including the
repaired ordered-FP32 long-dot reference, without copying or weakening that
verifier. It adapts launch geometry to the new planner and adds layout,
BF16-range, dense-V and mixed masked-NaN fixtures.

The matching new `tools/prefill-next/MetalArm.swift` authenticates compiled
source identity, then verifies the fragment layout/product **inside the same
Metal process before any new workload kernel**. Both probes precede all
operator timing. Correctness, first/last mutable buffers, read-only hashes,
guards, dispatches and PSO resources are retained. Twenty fixed warmups and
nine measured whole-operator command buffers are predeclared for every arm,
including controls. API counters cover this driver's calls; they are not a
claim to observe Metal's internal compiler.

`screen` advances through6,17,64,256,512,1024,2048 only after prior cases pass.
Short custom cases do not disappear at longer admission. `sample` reconstructs
all required earlier fixtures, verifies sealed inputs, compiled-body/library/
driver/referee identities, re-reads outputs and raw receipts, and repeats
numerical verification. A JSON `status: pass` alone is insufficient. Normal
scalar attention is never fed unsupported negative-page fixtures.

The generator emits the existing `rvllm.experiment_job.v1` schema, with real
local artifact pins and dependency order. Eight timing jobs each own exactly
one arm in **ABBA–BAAB** order. Each is an `exploratory_timing` job with explicit
conditions, zero stability dwell, no retry and fresh output paths. The referee
has a separate create-new ownership lock, not the worker's accelerator lock.
It waits for a Metal child to exit safely rather than killing device work on
an overrun or releasing ownership prematurely.

Adjudication retains all72 measured GPU observations, all eight process
medians, both order-specific ratios, inter-process and all-sample drift.
Nonpositive/invalid timers fail. Both drift definitions must stay within5%.
Matched projection schedules require identical final output hashes and,
where present, identical complete FP32 raw-intermediate hashes. The report
always retains `conditions_qualified:false` until the eight outer queue
receipts are reviewed, and `production_promotion:false`. Stable but slower
candidates can pass timing validity; the ratios, not that status, decide speed.

The examples are **configuration templates**, not submitted manifests or
fabricated receipts. Replace `/ABS/...` paths before invoking generators.
The existing reference-v3 jobs are unrelated and must remain untouched.
Retained synthetic inputs can consume tens of GB across a full campaign;
use a dedicated artifact volume and budget capacity before submission.

## 7. Exact local procedure

Use a fresh worktree rather than a dirty active experiment tree:

```sh
git worktree add --detach /ABS/rvllm-prefill27 c009497bee8125d456e4aee26f0b35836d79c231
cd /ABS/rvllm-prefill27
git apply --check /ABS/rvllm-prefill27-c009497b.patch
git apply /ABS/rvllm-prefill27-c009497b.patch
cd v3
# Format before freezing/building; formatting was not run in the delivery host.
rustfmt --edition 2021 --config skip_children=true $(cat tools/prefill-next/rustfmt.paths)
bash tools/prefill-next/host-gates.sh
# Required on Apple as well: type-check the real native integration.
cargo check --offline --locked -p rvllm-runtime --features apple --lib
cargo build --offline --locked --release -p rvllm-apple-metal --bin rvllm-prefill-next
bash tools/prefill-next/build-apple.sh \
  /ABS/rvllm-prefill27/v3/target/release/rvllm-prefill-next /ABS/NEW-pr27-strict-build
```

The builder uses the real source exporter, `metal -std=metal3.1
-fno-fast-math`, `metallib`, and a native Swift driver. It records toolchain,
source, AIR, metallib, executable and build hashes. Compilation is not an
arithmetic gate. Do not time concurrent Cargo/Metal builds.

Start the focused candidate/control correctness jobs:

```sh
REF=/ABS/rvllm-prefill27/v3/target/release/rvllm-prefill-next
# Copy and fill the provided screen-config.example.json to a NEW absolute file.
"$REF" screen-jobs /ABS/pr27-screen-config.json
QUEUE=/ABS/EXISTING/rvllm_experiment_queue
for job in /ABS/NEW_SCREEN_CAMPAIGN/jobs/*.json; do
  "$QUEUE" submit /ABS/serial-queue "$job"
done
```

Use the existing serial worker; do not start a competing worker. After both
required arm/role screens reach the target rung:

```sh
"$REF" timing-jobs /ABS/pr27-timing-config.json
for job in /ABS/NEW_TIMING_CAMPAIGN/jobs/*.json; do
  "$QUEUE" submit /ABS/serial-queue "$job"
done
# Only after all eight terminal receipts exist:
"$REF" adjudicate /ABS/NEW_TIMING_CAMPAIGN/adjudication.json /ABS/pr27-result.json
```

Direct commands supported for a single queue-owned child are:

```sh
"$REF" screen metal-prefill-wide64 down-norm 256 \
  /ABS/NEW-pr27-strict-build/MetalArm /ABS/NEW-pr27-strict-build \
  /ABS/NEW-screen /ABS/pr27-inner.lock
"$REF" sample metal-prefill-wide64 down-norm 256 /ABS/NEW-screen/screen.json \
  /ABS/NEW-pr27-strict-build/MetalArm /ABS/NEW-pr27-strict-build \
  /ABS/NEW-sample /ABS/pr27-inner.lock
```

Advance to512,1024,2048 with new predeclared jobs and identities only after
shorter correctness; do not alter an already submitted job. Prioritize
PR26-pipeline versus wide64/wide128 on down-norm and gate-gelu, then QKV and O,
and PR26-Q4K16 versus MMA8K32 on both attention kinds. All original normal,
MMA32, load4 and SIMD attention operator controls remain exportable.

The frozen old full-route ABBA helper hardcodes its original choices and is
**not represented as automatically compatible with PR27**. For a new native
session build, use the existing `RVLLM_METAL_RESEARCH` selector and the newly
built corresponding BF16 metallib. For one untraced, single-chunk prefill
M>32, expected combined counts are48 GEMM,48 QKV,96 raw,96 norm,40 local,8
global under the **new** names. `prefill_round::expected_prefill_dispatches`
exports that exact ledger. A session invocation must use `--prompts-jsonl`
with `--report`, not the formerly invalid `--prompt`/`--report` combination.
Do not put `--top-logits` on a session timing command. Preserve your current
model, token-ID, executable, library and queue pins. A route-preserving
full-model numerical producer is not added by this patch.

## 8. Falsifiable decisions and fair MLX comparison

Reject immediately on strict compile/PSO failure, failed layout preflight,
nonfinite output in a finite fixture, guard/read-only corruption, visibility
or newest-KV error, missing actual dispatch, changed workload, undocumented
source drift, or a relaxed numerical gate. Keep the failed artifacts. A new
revision may have a new attempt; do not rewrite the failed attempt.

For a projection performance lead, require passing complete FP32 QKV and
raw-intermediate checks, bitwise matched projection outputs on the sealed
fixtures, >1.05 control/candidate in **both** order blocks, both5% drift gates,
qualified outer conditions, and one separately predeclared confirmation.
Attention has an arithmetic-changing reference gate, not a bitwise scalar
promise. Require its complete-output tests and masked-NaN case before timing,
then the same timing requirements. If all new members lose, keep PR26; there
is no automatic wide64 selection hidden in the patch.

For full-route acceptance, use actual checkpoint activations/weights and the
separate route-preserving full-vector reference work. Require per-layer/QKV,
postnorm, residual and KV checks, exact IDs for fixed regression trajectories,
and a predeclared broader teacher-forced/held-out quality criterion. Neither
old matching trajectories nor the short structured oracle is enough. Do not
invent a new looser threshold in response to a failure.

Benchmark M256/512/1024/2048 individually, with exact prompt IDs and equal work.
Report **fresh-process first use** separately from **resident same-process
prefill with reset logical KV state**; reset caches through the real owner,
not ad-hoc metadata writes. Do not move the favorable second case into a
shape-general chart. Full-route cold/resident instrumentation is a separate
measurement task, not supplied as an imaginary API here.

MLX comparisons must pin the measured wheel/build and actual generated kernel
path, BF16 checkpoint/tokenizer, layouts, output dtype and numerical
boundaries. BF16-output MLX linear is not the same operator as the FP32 QKV
contract; do not quietly insert casts or label a different FP32-input GEMM
as equivalent. Align whether prefill includes the final head/first token and
whether decode counts that token. Record preparation/page residency separately.
Run both frameworks through the same serial queue with predeclared ordering
and identical condition policy. Never infer a stage sum from the failed
zero-counter route. Only aligned boundaries permit a strict framework verdict.

## 9. Validation performed for this delivery

Executed on the Linux delivery host: **14 independent Python host algebra /
source-contract tests**, **8 catalog tests**, Swift syntax parsing, compilation
and execution of the driver's non-Metal branch (correct refusal, exit2), shell
syntax checks, and exact-preimage patch application/reversal plus postimage
hash validation. The host tests are not a GPU simulator or proof of Metal
rounding. The immutable preimage subset was verified against GitHub blob IDs.
This was not a full repository checkout build.

**Not executed:** rustc/Cargo/rustfmt/Clippy, the18 newly authored Rust test
functions (or the shared fixture tests), full-repository CI, Apple Swift type
checking, strict Metal compilation, PSO/device probes, native checkpoint runs,
generated GPU code/register/spill inspection or any performance trial. Rust
and Apple toolchains/device were not available in this container. Treat this
as a reviewable implementation with explicit local gates, not a compiled or
speed-qualified release. `VALIDATION.json` in the bundle separates these facts.

## 10. Source anchors

All rvLLM paths below are read at the base in this document:

- `v3/reports/prefill-round-fullroute-short-20260927/{README.md,MECHANISM.md}`:
  full-route, position sensitivity, standalone/MLX and numerical observations.
- `v3/crates/rvllm-apple-metal/src/layer_forward.rs`:
  blob `257031020c76286da2d4c50f63ab25f8dc4401f3`, dynamic prefill adapter.
- `.../prefill_round.rs`: blob `8889dfeba29ec0d79f21354f199210ba9f134b31`.
- `.../research_shaders/prefill_projection_common.metal`:
  blob `fdec1b6b138c6899864e5b58d8f1f0dbd2e36f20`.
- `.../research_shaders/prefill_attention_common.metal`:
  blob `512865e49aa2027c746f71dd6d6c97c1c80bca5e`.
- `.../bin/prefill_round/fixtures.rs`:
  blob `0b0b7cc4b04daf90c52a0a90d96e9ced3c5d45d3`, reused by the new referee.
- `v3/tools/prefill-round/MetalArm.swift`:
  blob `bc8a6fc97174f3f441bf21098736f4f8641252cc`, basis of the new probe driver.
- MLX Steel `mma.h` and MIT license are pinned in `MLX-NOTICE.txt`.

Rollback is reversing this patch in the same base worktree, or leaving its
selectors off. Never delete completed experiment artifacts to roll back code.
