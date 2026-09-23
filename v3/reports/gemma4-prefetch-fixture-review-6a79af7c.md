# Prefetch fixture repair and the next local execution phase

Base: `6a79af7c81900572ee6a6f39a46280fc1f35e865`.
This is a test/diagnostic repair, not a new kernel or a performance result.
The original failed attempt and all successful component receipts remain history.

## Why the old fixture necessarily encounters poison

Its first synthetic case is M=63, N=67, K=35. Both unchanged prefetch entry
points reject that shape before their first barrier and before writing output.
The fixture initialized every output payload byte to 0xff, then required every
FP32 output to be finite. An untouched 0xffffffff is a NaN. The source therefore
contains a deterministic fixture/entry-point contract mismatch sufficient to
explain the reported failure; the original /tmp raw log and tensors were not
available to the remote reviewer, so this is not a retrospective device pass.

Simply deleting the synthetic case is insufficient: the old fixture also calls
both prefetch entry points for every real projection. The FP32 entry admits only
QKV shapes, and the storage entry admits only gate/up, output and down shapes.
This patch retains all cases and dispatches, distinguishing numerical outputs
from explicitly expected guard refusals. No shader, shape guard, production
selector, numerical threshold, private feature gate or ignored marker changes.

## Corrected component contract

Six cases still issue 24 single-command dispatches: 12 baseline controls,
5 numerical prefetch dispatches (2 QKV and 3 GEMM), and 7 refusal controls.
Refusals must leave every payload byte unchanged and preserve both canaries;
they cannot count toward numerical coverage. Positive QKV outputs retain exact
FP32 parity plus the original relative-L2 and sampled FP64-dot gates. Positive
GEMM outputs must match the once-rounded qualified baseline FP32 result exactly.
The unavailable, wrong-role FP32 entry is NOT treated as a numerical reference.
All other candidates retain their paired-output arithmetic checks.

A prefetch fixture now requires a fresh absolute RVLLM_METAL_MMA_TILE_REPORT.
Its companion <report>.artifacts directory retains the emitted Metal source,
per-case/per-entry labels, and every guarded output before oracle assertions.
A capture.json is explicitly unqualified; only the final successful v3 report
establishes component-contract completion. Old v2 failures are not rewritten.
A complete run writes 397,999,068 guarded-output bytes (about 380 MiB), plus
source and JSON. Reserve this headroom in addition to the existing disk floor.

Six new Rust unit tests are in the ordinary public/native projection filter.
Eight new Python source/guard tests are in the recurring Python runner.
The complete local inventories become 66 public Rust tests and 116 Python tests;
the 22 Metal source/compile arms and 32-path formatting scope are unchanged.
The remote reviewer did NOT execute Rust, rustfmt, Metal, ANE or inference.

## Local execution: do not block unrelated work on a long reference

1. Review/apply this repair; format only the two changed Rust files already in
   gemma4_candidate_rustfmt.paths. Run the ordinary host/compiler gates. Do not
   re-run all previously passing accelerator fixtures merely to add receipts.
2. Recheck actual boot, ownership/process policy, disk and power/thermal state.
   The previous 18 GiB observation is not a permanent admission. Preserve STOP
   state; never stop unrelated services or change OS power settings. Preserve
   binaries needed for a pinned run before a future cargo clean; any rebuild
   requires new executable/library pins, never reuse old mutable-target hashes.
3. Independently of this prefetch repair or a >=64-token reference, inspect the
   unchanged baseline cache. Use only the existing bounded preparation mechanism
   for missing programs, followed by a separate strict inspection requiring 162
   visits, zero inspection compiler calls and valid driver lifecycle evidence.
   Do not treat a failed inspection as permission for unbounded retries.
4. Run complete baseline then short-MMA continuations using unchanged existing
   references. The 21-token chat-capital reference contains only TWO expected
   output IDs. Its full completion is not a nine-decode-step timing workload.
   The tracked gemma4-12b-hf-capital-steps16.json contains a SIX-token prompt and
   SIXTEEN expected outputs; verify its model/source pins and use it unchanged
   for a longer rollout. It does not require a >=64-token prompt.
5. After successful compile gates and renewed local admission, select only the
   corrected prefetch fixture with a new job/output identity. This is a diagnosed
   fixture repair, not a blind retry. First list the exact fixture without running
   it and require a single match. Then run the one reviewed ignored fixture.
   Its unchanged shader still needs real numerical qualification. Stop on any
   numerical failure and retain the new raw outputs before returning to Astra.
6. FFN input-pin preparation is host-only and does not itself need an ANE cache.
   It does need real, independently captured baseline FFN activations. Actual
   FFN comparisons need separately inspected cached control/candidate programs.
7. Only GQA/temporal/long-tile positive full-prefill screens need the missing
   >=64-token reference. Keep RMS and packed32 blocked by their specific missing
   evidence. No such blocker should halt unrelated, fully specified baseline work.
8. Timing waits for original tensor/driver/full-continuation gates and eligible
   sampled controls. Keep strata separate. Do not pad an early-EOS reference to
   match a timing manifest or substitute its results into the old ABBA workload.
   Preserve failed results; no automatic winner/default/promotion.

### Correct fixture selection (read/list first, no tests executed by listing)

From v3, using the local owner's existing target cache and offline dependencies:

```sh
cargo test --offline --locked --release -j 2 --target aarch64-apple-darwin \
  -p rvllm-apple-metal --lib \
  layer_forward::prefill_mma_tile_tests::native_prefetch_preserves_existing_oracle_gates \
  -- --ignored --exact --list
```

Only after one exact matching test is listed, source/compiler checks pass, and
local hardware admission is satisfied: set RVLLM_METAL_QKV_MODEL_DIR to the
already-present pinned model, RVLLM_METAL_MMA_TILE_REPORT to a fresh absolute
path, and RVLLM_METAL_MMA_TILE_QUALIFY_ONLY=1. Use the same command with
`-- --ignored --exact --nocapture` instead of the listing arguments. Never run
all ignored tests. Require 6 cases, 24 commands, numerical counts GEMM=3/QKV=2,
7 unchanged-poison refusal controls, and the original oracle gates.

The researcher has performed none of these local hardware/cache commands.
Commit reviewed code and append the actual local results to root handoff.md;
return exact pins, failure case/kernel, raw artifact paths and gate outcomes.

## Source references (accessed for this review)

- Native shader, same blob 6f5282d2b7a6e3be0c957a468a4311fc5869a7b6:
  https://github.com/delysis/rvllm/blob/6a79af7c81900572ee6a6f39a46280fc1f35e865/v3/crates/rvllm-apple-metal/src/research_shaders/mma32_prefetch.metal
- Native fixture, blob 146c8ae7947960049cafc866862377cb347b79a9:
  https://github.com/delysis/rvllm/blob/6a79af7c81900572ee6a6f39a46280fc1f35e865/v3/crates/rvllm-apple-metal/src/prefill_mma_tile_tests.rs
- Reported local results:
  https://github.com/delysis/rvllm/blob/6a79af7c81900572ee6a6f39a46280fc1f35e865/v3/reports/gemma4-hardware-qualification-execution-20260923.md
- Full-path test filters and --list/--ignored/--exact semantics:
  https://doc.rust-lang.org/rustc/tests/
