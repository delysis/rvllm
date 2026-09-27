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

## Five-arm repeat-profile adjudication

The final control completed, and the frozen
`rvllm-prefill-route-profile-summary` accepted **all 30 cases** from
`off-a → pipeline → off-b → combined → off-c`. Each of the five queue jobs
succeeded with no violations, eligible sampled conditions in the same AC /
power-mode-2 / thermal-state-0 stratum, exact shared model/executable/prompt
pins, exact 64-token outputs, required actual candidate dispatch, and zero
inference-case compiles. The machine was not held for a stable thermal state.
Every prefill and decode sample remains in `profile-summary-01.json` and the
five complete raw queue result directories are in
`profile-queue-results-01.tar.gz` (25 source files). Their SHA-256 values are
respectively `5551ea7892ec9ab0fdce467c8382b8238d2d64c44f3c5293e4c342e4787a5fb9`
and `2f0e8d6f72b46a7924e1fb6813866cd619e69583c6c2f9b7cb477ea470022051`.

| Prompt | Projection-only: bracketed control / candidate median | Combined: bracketed control / candidate median | Five-percent profile drift |
| --- | ---: | ---: | --- |
| 101 tokens | 6086.894 / 2264.970 ms; descriptive 2.687× | 6082.425 / 1851.792 ms; descriptive 3.285× | **Fail both**: projection's maximum within-arm drift 40.125%; combined's 8.510% |
| 304 tokens | 12247.118 / 1194.070 ms; descriptive 10.257× | 12278.616 / 641.769 ms; descriptive 19.132× | Pass both: maximum within-arm drift 1.847% / 2.623%; bracket-control drift 0.133% / 0.180% |

The 101-token failures are preserved and no sample or arm was removed. The
304-token profile is a strong, condition-observed **prospective** signal, but
the arms ran in separate processes. Its drift pass is **not** within-job
ABBA/BAAB speed qualification, independent confirmation, checkpoint-quality
acceptance, or promotion. The separately queued counterbalanced full-route
job was the next speed gate; the exact-token same-checkpoint MLX jobs will
subsequently show where the new route stands relative to MLX, with their
different timing boundary explicitly labeled.

## Counterbalanced combined-route full-route screen

The immutable `prefill26-fullroute-abba-combined-01-20260927` queue job
completed successfully (exit 0, no violations, eligible sampled conditions).
The sampled comparison stratum was AC power, power mode 2, thermal state 0,
and low-power mode off; no thermal stability was required or inferred. The
feature-gated Rust driver ran two fixed warmups per arm, then four measured
control and four measured combined-route child processes in ABBA and BAAB
order. It retained all 12 children's stdout, stderr, and validated receipts.
Every child used the pinned original 12B-it weights, executable and prompts,
produced the expected 64 generated IDs for each prompt, showed the expected
candidate dispatch (or no research dispatch for control), and reported zero
inference-case compilation. These checks establish route and output-ID
agreement, **not** an independent numerical or checkpoint-quality oracle.

| Prompt | Control prefill samples (ms) | Combined prefill samples (ms) | ABBA / BAAB control-to-candidate ratio | Five-percent within-arm drift |
| --- | --- | --- | --- | --- |
| 101 tokens | 6113.187, 6076.298, 6123.950, 6246.407 | 1696.250, 1767.662, 1809.402, 1824.129 | 3.519× / 3.405× | **Fail**: control 2.179%, candidate 7.539% |
| 304 tokens | 12306.610, 12372.684, 12269.974, 12300.971 | 640.003, 640.364, 640.604, 639.076 | 19.275× / 19.201× | **Pass**: control 0.537%, candidate 0.145% |

The 304-token prefill result is a strong, queue-eligible **single-job
counterbalanced speed screen**; the 101-token result remains drift-inconclusive
despite its apparent gain. The order-specific ratios were independently
recomputed from the retained arrays. Neither result is an independently
confirmed winner, production promotion, checkpoint-wide quality verdict, or
MLX comparison. The 5% gate was not relaxed or applied selectively. Power
sampling cannot establish fixed GPU clocks or absence of every transient
competitor.

The raw driver summary `abba-combined-01-raw/summary.json` has SHA-256
`5173d5f7a3f75623461963a19c8249747caea7e91e6745c24bd57e8fe150108a`.
The complete outer queue receipt (`job.json`, `report.json`,
`conditions.jsonl`, and trial stdout/stderr) is
`abba-combined-01-queue-result.tar.gz`, SHA-256
`66f81029be7f408a3fe94b03a5cfaffc2a2e16221ba697db964c233e88adc385`.
All 12 inner child receipts, the config, and summary are in
`abba-combined-01-children.tar.gz`, SHA-256
`100f5ddef36ff87230793dbe3fdbfd7c3df5726e1bec96205fe0fe7a107cc958`.
The large checkpoint and compiled binaries remain local, with identities
recorded in the receipts rather than copied into Git.

Next: an independently run, predeclared confirmation at the same shapes;
route-preserving tensor/logit/reference and checkpoint-quality checks; then
same-checkpoint MLX orientation with its different prompt timing boundary
made explicit. Do not rerun or rewrite this completed job.

An independent fixed-protocol confirmation was submitted as the new immutable
queue ID `prefill26-fullroute-abba-combined-02-20260927`. Its separate
`abba-combined-02-config.json` (SHA-256
`4713fc37bf86d33cc766e9a6ec233b2a5505b724bb7c1c8cff78467a4195e6a1`)
uses the same pinned executable, weights, prompt file, metallibs, warmups,
ABBA/BAAB ordering and 5% gate, but a fresh output path; its job depends on
the completed first run. The serial queue accepted the manifest. This is a
predeclared confirmation attempt, **not** confirmation evidence until its
terminal receipt is read. The previously submitted MLX jobs remain in the
same serial queue; none of the completed jobs were replayed.
