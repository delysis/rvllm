# Q4K16 short real-weight route triage (2026-09-27)

See `MECHANISM.md` for a source-and-dispatch account of the prospective
prefill gain and the compiler/timing claims it does **not** establish.

This is a default-off BF16 Gemma 4 12B-it *route and correctness screen*, not
a speed qualification or a production selection. The serial experiment queue
ran both arms against the original `google/gemma-4-12B-it` safetensor, with the
same six token IDs `[2, 818, 5279, 529, 7001, 563]` and two generated steps.
The manifests pin the executable, normal or Q4K16 metallib, model config,
23,919,549,408-byte safetensor, tokenizer, and prompt file by SHA-256.

| Arm / immutable queue ID | Result | Actual research dispatch | Generated IDs | One-case prefill / decode |
| --- | --- | --- | --- | --- |
| `prefill26-fullroute-short-off-session-v2-20260927` | succeeded, queue-eligible, no violations | none | `9079, 236761` | 1854.398 / 245.968 ms |
| `prefill26-fullroute-short-q4k16-session-v2-20260927` | succeeded, queue-eligible, no violations | D256: 40; D512: 8 | `9079, 236761` | 2017.267 / 246.742 ms |

Both arms reported zero library and pipeline-state compiles *inside the case*;
their separate preparation phases compiled one library and 58/60 pipeline
states respectively. The Q4K16 dispatch count covers all 40 sliding and eight
global layers, so this is not a fallback-only success. The equal first two IDs
are weak correctness triage, not tensor, logit, or checkpoint-quality evidence.
The unpaired timings are descriptive only; they neither support nor refute a
speed win. These short runs do not measure Q4K16's isolated M256/M512 operator
advantage under a normal long prefill.

The original `prefill26-fullroute-short-off-20260927` and
`prefill26-fullroute-short-q4k16-20260927` jobs both failed *before inference*:
their single-prompt invocation combined `--prompt` with the session-only
`--report` option. Their failed reports, stderr, input manifests, conditions,
and quarantine records are retained unchanged. The new v2 IDs use a pinned
`--prompts-jsonl` input and omit `--top-logits`, which is single-prompt only.
No failed job was replayed or rewritten.

The complete compact queue archive is `queue-receipts.tar.gz`, SHA-256
`3fd91f93ded859ab1d18fd6d10378ebd065d543eb46a4bd19a790b079fb42b22`.
The frozen executable SHA-256 is
`7b6487471816cfe71999bf9d54631589105dcf88cb4f569e924c28c048cc1e20`;
normal/Q4K16 metallib SHA-256 values are respectively
`bb88c9667b4b7ac3758760b602abe40cbe72cc907bf3c40e2f2a2d33b0bd8df2`
and `83ef6c0fbc06dbd545164a03cc3e19fcb262daef51127fc60991b314708b9a1b`.
The large model and compiled artifacts remain local; they are not in Git.

Next: compare longer, varied prompts and continuations with exact dispatch,
independent numerical/logit evidence, and an explicit counterbalanced timing
protocol. Only then consider any full-route speed or quality claim.

The next correctness-only pair completed:
`prefill26-fullroute-varied-off-20260927` and
`prefill26-fullroute-varied-q4k16-20260927`. It uses the pinned
`varied-prompts.jsonl` (SHA-256
`cba2a6830503e5e95e37f1c677197dd3d93a26e0dcf33c501be973ec0cedc483`),
which contains two non-repetitive prompts of 101 and 304 actual tokens. Both
queue jobs succeeded with zero violations and eligible sampled conditions in
the same AC / power-mode-2 / thermal-state-0 stratum. Each produced 64 tokens
per prompt, with the **entire 128-token generated-ID sequence exactly equal**
between control and candidate. The Q4K16 arm actually dispatched D256 40 times
and D512 eight times in *each* case; the control dispatched no research kernel.
Both cases in both arms reported zero library and pipeline-state compiles during
inference. This is stronger route-preserving continuation evidence than the
two-token screen, but it is neither an independent tensor/logit oracle nor a
checkpoint-wide quality gate.

| Prompt | Off prefill / decode | Q4K16 prefill / decode |
| --- | --- | --- |
| 101-token ledger | 5988.558 / 13808.087 ms | 5836.678 / 13685.334 ms |
| 304-token observatory | 12234.214 / 23306.755 ms | 11674.683 / 23184.596 ms |

These are one unpaired process per arm, with different library/pipeline
preparations and no ABBA drift check. They are descriptive only, not a speed
result. The compact raw `varied-queue-receipts.tar.gz` archive has SHA-256
`ce72fba7cf1ddd1ce2fea45cb2f64b7f35a1f340dc63dfe67737e4815ddf5624`.
It retains both manifests, queue reports, full stdout/stderr and condition
journals. Neither the model nor compiled binary is included.

Next: obtain independent tensor/logit/reference quality evidence and run a
predeclared counterbalanced, same-workload full-route timing comparison. The
isolated Q4K16 operator gains cannot be promoted from these route screens.

The projection-only correctness job
`prefill26-fullroute-varied-pipeline32x64-20260927` also completed on the
*same* varied input. It succeeded in the queue with eligible sampled conditions
and no violations. Both 64-token generated trajectories again matched the
control exactly. Each case actually dispatched 48 pipeline GEMMs, 48 pipeline
QKV projections, 96 raw-norm projections and 96 normalization kernels;
there were zero inference-case library or pipeline-state compiles. The
dependent
`prefill26-fullroute-varied-pipeline32x64-q4k16-20260927` tests the combined
projection and attention selection. They are distinct selectable metallibs
with pinned hashes.

The projection-only job reported prefill 1859.332 ms at M101 and 1192.767 ms
at M304. Relative to the earlier unpaired control observations, those are
descriptively ~3.22x and ~10.26x faster. This is a **prospective full-route
prefill gain**, not an accepted speedup: the processes were not interleaved,
the M304 candidate's second case was faster than its first, and no repeat-drift
or independent-confirmation gate has run. The raw queue archive
`pipeline-queue-receipt.tar.gz` has SHA-256
`44ce3edb10c802d19e288f377bb07ad2727a9210ff0b576c06a37431bdba09c5`.

The combined job also queue-succeeded, with eligible sampled conditions, zero
violations, zero inference-case compiles, and **exact agreement across all 128
generated IDs**. Each case dispatched the four projection slots above plus
40 combined D256 and eight combined D512 prefill-attention calls. Its
descriptive unpaired prefill was 1877.087 ms at M101 and 647.637 ms at M304,
compared with the same earlier off observations of 5988.558 and 12234.214 ms.
Those are roughly 3.19x and 18.89x *orientation*, not accepted speedups:
second-case/cache effects are material and these jobs were not interleaved.
The complete `combined-queue-receipt.tar.gz` archive has SHA-256
`612b834f2dcefa6b3db993374181fd66b8722aed476e479166b2cb3ca3125ebc`.

The separate `profile-off-a`, `profile-pipeline`, `profile-off-b`,
`profile-combined`, `profile-off-c` manifests predeclare two bracketed
control/candidate/control diagnostic series, sharing the middle control. Each
process retains **all three** built-in profile samples for both varied prompts.
They have dependency order and distinct report paths. This is a variance
screen, **not** the required within-job ABBA/BAAB qualification: its arms are
not interleaved and no favorable sample may be selected. The 5% drift and
independent confirmation gates remain necessary before any speed verdict.
The safe-Rust `rvllm-prefill-route-profile-summary` offline referee takes the
five completed queue-result directories in that order. It requires all 30
case samples, exact prompt and output IDs, expected actual dispatches, zero
inference compilation and identical input/executable identities, then reports
every timing observation, sampled power stratum and within-arm/bracketing
control drift. It refuses a missing, failed, fallback or changed-work receipt.
Mixed or stale condition samples remain visible but cannot pass its diagnostic
eligibility flag; even a favorable report remains non-ABBA evidence.

The separate immutable job `prefill26-fullroute-abba-combined-01-20260927`
is queued **after** the last five-arm profile. Its feature-gated safe-Rust
driver `rvllm-prefill-route-abba` pins the original BF16 inference executable
and both metallibs, uses the same two varied prompts and 64-token
continuations, and fixes two warmups per arm followed by ABBA and BAAB
measurement blocks in one serial queue job. Each child process has a retained
stdout/stderr and validated receipt. The driver rejects changed prompt or
generated IDs, inference-time compilation, wrong research dispatch, missing
samples, and overwritten output. It reports both order-specific ratios and
all-sample 5% drift, but does **not** make a promotion decision: the queue's
condition envelope, independent confirmation, and tensor/logit/reference
quality checks remain separate. Its config and queue manifest are
`abba-combined-01-config.json` and `abba-combined-01-job.json`; the queue pins
the 23.9-GB model file rather than copying it into Git. No result was present
when this job was submitted.

Two same-checkpoint MLX-LM jobs are also queued through the **same serial
referee**: `prefill26-mlx-it-bf16-m101-g64-20260927` and
`prefill26-mlx-it-bf16-m304-g64-20260927`. Their one-case source files
`mlx-source-m101.json` and `mlx-source-m304.json` retain the exact prompt IDs
from the original passing off-route report (SHA-256
`39d4c2f5a425aefb533a5c7bc80d0d8dcd86e15506e7f27e363b9aa883e69d78`);
the two token-ID arrays were checked element-for-element against that report.
The jobs pin the original 12B-it safetensor, the previously qualified MLX-LM
source and benchmark script from PR #6, and run one warmup plus three full
64-token trials per prompt. M101 depends on the final repeat-profile control,
not on success of the new ABBA arm; M304 depends on M101. The MLX prompt
phase is inferred from first-token throughput and therefore **does not share
rvLLM's prefill timing boundary**. Any resulting ratio will be labeled
planning-grade until those boundaries are reconciled. No MLX result was
present when these jobs were submitted.
