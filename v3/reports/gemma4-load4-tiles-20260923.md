# Gemma 4 load4 tile batch: implementation and local-agent handoff

Base: `d391eea9acaca47653e19bb7d6e58e81dbd51b7a`, branch
`codex/gemma4-kernel-candidates`. This is a candidate-only continuation of
`gemma4-kernel-campaign-20260923.md`, not a production promotion.

## What is submitted

Seven independently selectable Metal projection candidates, fourteen new entry
points. Each candidate has a BF16-stored GEMM entry and an FP32-output QKV entry.
The existing ten candidates, their shader sources, the `off` default, all ANE
code/cache plans, decoder orchestration, and the production encoder are unchanged.

| Selector suffix (prefix `metal-load4-`) | M x N x K tile | SIMD groups M x N | Threads | Source shared bytes | Prompt tokens |
| --- | --- | --- | ---: | ---: | --- |
| `m16n32k64` | 16 x 32 x 64 | 1 x 2 | 64 | 6,144 | 6-1,024 |
| `m16n64k64` | 16 x 64 x 64 | 1 x 4 | 128 | 10,240 | 6-1,024 |
| `m32n32k64` | 32 x 32 x 64 | 2 x 2 | 128 | 8,192 | 6-1,024 |
| `m32n64k32` | 32 x 64 x 32 | 2 x 2 | 128 | 8,192 | 6-1,024 |
| `m32n64k64` | 32 x 64 x 64 | 2 x 2 | 128 | 12,288 | 6-1,024 |
| `m32n64k128` | 32 x 64 x 128 | 2 x 2 | 128 | 24,576 | 6-1,024 |
| `m64n64k64` | 64 x 64 x 64 | 2 x 2 | 128 | 16,384 | 64-1,024 |

The common implementation is `research_shaders/load4_tiled_common.metal`; the
seven small leaf files instantiate actual, separately named kernel entry points.
The Rust source exporter includes the common implementation before the selected
leaf. The delivery manifest hashes the common source as well as every leaf.

### Why these are useful experiments

The prior exploratory `metal-mma32-load4` result (about 1269 ms versus 1840 ms
control) motivates this batch; it does not qualify any speedup, including that
candidate's own speedup.

The new family keeps four-element native-storage loads and ascending K/8 MMA
accumulation. K=64 and K=128 reduce the two-per-block K-loop barriers by factors
of two and four relative to K=32. Wider N tiles reuse each loaded activation
across more output columns. Wider M tiles reuse each loaded weight across more
prompt rows. The M=16 variants reduce padded arithmetic for short prompts, and
the 16x32 variant uses two rather than four SIMD groups.

Operand staging and FP32 output staging reuse one allocation, with a full
threadgroup barrier after the final operand read and before output stores.
The source budget is `max(2*(BM+BN)*BK, 4*BM*BN)`, not the sum of simultaneously
reserved input and output arrays. In particular, 32x32x64 and 32x64x32 both fit
an 8192-byte source budget. Resource limits are still queried from each compiled
pipeline and the device. Register pressure, occupancy, code size and compiler
lowering can offset the expected savings; the 64x64 and K=128 variants are
experiments, not asserted winners.

Suggested first controlled candidates: `metal-load4-m32n32k64`, then
`metal-load4-m32n64k32`, then `metal-load4-m32n64k64`. Screen all seven before
allocating expensive timing repetitions.

## Preserved admission and evidence boundaries

Runtime admission still requires the full supported Gemma 12B model/prefill
identity, native BF16, the existing trace policy, identity projection scale,
role-specific dimensions, sufficient spans, disjoint writes, and queried PSO
limits. Both input pointers must be eight-byte aligned. Only M tails are
masked; N and K must meet the tile divisibility and model-role constraints.
The shader also rejects unsupported shapes, projection scale, threadgroup
sizes, and grid coordinates uniformly before entering barriers.

The append-only dispatch registry grows from 17 to 31 slots. Existing slot
indices, names, and the v3 receipt format are retained. Consumers must bind the
catalog and executable to the receipt; an old 17-slot receipt is not evidence
for a new executable. The catalog contains 18 selections including `off`.
Both dtype exports therefore produce 36 compile/link arms. This count is not
36 new candidate kernels: the submission adds fourteen entry points.

No FP16 operand conversion, split-K, reduced-precision accumulator, fusion,
additional rounding, tolerance relaxation, or automatic routing policy is
introduced. The BF16 runtime source is obtained through the existing source
exporter. F16 export remains compile-only compatibility, not runtime admission.

## Verification status at submission

Executed in the authoring environment: independent host calculations for all
seven profiles, checking vector staging ownership, complete/nonoverlapping
fragment ownership, M-tail coverage, ascending K/8 fragment order for K=3840,
4096, 8192 and 15360, N divisibility, thread counts and scratch bounds. All seven
profiles passed those calculations. These are index/resource calculations,
not execution or compilation of the Rust or Metal implementation.

Not executed here: rustfmt, Cargo tests/build/Clippy, the Python delivery-contract
suite, Apple Metal compilation, native component qualification, first-token
screens, or controlled timing. The authoring environment has neither a Rust
installation nor an Apple accelerator. Do not transfer the base campaign's
passing test counts to this commit. The branch is an unqualified draft.

Added portable Rust tests cover the exact new shape sets, wrong-role rejection,
input alignment, scale/alias/span checks, staging and fragment bijections,
resource budgets, accumulation order, and BF16 source export. Existing generic
catalog and dispatch-family tests now include all seven new selections.

The delivery-contract tests retain their fake-tool scope. Counts and final-arm
checks follow the expanded catalog; new cases require the shared source to be
present and make a mutation of it invalidate the delivery receipt. They do not
claim to compile Metal or execute inference.

## Ordered local-agent work

1. Review the diff against the pinned base. Apply packet-scoped rustfmt to the
   changed Rust files; record any formatting/build repair as a new commit before
   pinning native receipts. Do not format unrelated workspace files or relax a
   shader guard, numerical gate, or evidence requirement to obtain a pass.
2. Run the delivery-contract suite and the existing build-only delivery gate.
   Preserve the ordinary policy/catalog/projection/dispatch tests, nonzero test
   counts, exporter and CLI hashes, all 36 Metal exports/compiles/links, and final
   source-integrity checks. A failed compile is a compile failure, not a timing
   result. No ANE cache preparation is needed for this Metal-only batch.
3. Run the new ignored device oracle once per candidate, serially, with a fresh
   absolute report directory for each attempt. Retain failed attempts.
4. Run the pinned full-model exact first-token and complete-family screens for
   all seven. Any fallback-only result remains unexercised, not a pass. Require
   both GEMM and QKV dispatch families and no foreign-family counters.
5. Only qualify timings after the component and full-model screens pass. Use
   controlled ABBA/BAAB blocks against the unchanged `metal-mma32-load4`, and
   separately the `off` control, in one hardware/power/thermal stratum. Preserve
   raw paired samples, exact tokens and dispatch evidence. Do not promote a
   winner on one exploratory sample or on a microkernel win alone.

From `v3`, the ordinary host/build steps are:

```sh
python3 tools/test_gemma4_candidate_delivery.py
bash tools/check_gemma4_candidate_delivery.sh "$BUILD_RECEIPT_DIR" "$CARGO_TARGET_DIR"
```

Both variables must identify the local campaign's intended absolute paths;
`BUILD_RECEIPT_DIR` must be fresh. Use the established model and power-policy
settings from the base campaign. Start receipts only from a clean, pinned tree.

The new exact test name and environment contract are:

```sh
RVLLM_METAL_QKV_MODEL_DIR="$MODEL_DIR" \
RVLLM_METAL_LOAD4_TILE_CANDIDATE=metal-load4-m32n32k64 \
RVLLM_METAL_LOAD4_REPORT_DIR="$FRESH_ABSOLUTE_COMPONENT_DIR" \
cargo test --offline --locked --release --target aarch64-apple-darwin \
  -p rvllm-apple-metal --lib \
  load4_tile_tests::native_load4_tile_preserves_projection_contracts \
  -- --ignored --exact --nocapture --test-threads=1
```

The candidate must be one of the seven new selectors. The test never sweeps
candidates implicitly. It uses the actual exported production-candidate source,
not a test-only arithmetic clone, and dispatches the 64-thread profile correctly.
It qualifies four GEMM role cases and two QKV role cases, plus ten guard-refusal
arms across eight fixtures. Real checkpoint weights and synthetic BF16 inputs
include values outside FP16 range. Positive paths require exact FP32 baseline
bits or exactly once-rounded BF16 baseline bits; sampled independent FP64 dots
retain the incumbent oracle's tolerance. Poison bytes, surrounding canaries,
and repeated-use byte identity are checked. Source, queried PSO resources, and
raw outputs are saved before numerical checks; `qualification.json` is written
only after every case succeeds. This is component evidence, not a runtime
selection or timing receipt.

Keep the failed native captures and all raw timing samples in the local evidence
store, not in Git. Return the exact tested commit and build/source hashes, each
candidate's component/screen/timing status, and any compiler/resource failure.
Leave all production defaults and ANE plans unchanged.
