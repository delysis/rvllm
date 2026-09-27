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

## Same-checkpoint MLX comparison: 101-token prompt

The pinned `prefill26-mlx-it-bf16-m101-g64-20260927` job succeeded and
retained one warmup plus all three 64-token MLX-LM trials. It used the same
original 12B-it safetensor (SHA-256 `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`)
and exact 101 prompt IDs as the rvLLM counterbalanced run. All three MLX
trials produced the **same full 64 generated IDs** as every validated rvLLM
control and combined-route child at this prompt. That is cross-framework
output-ID agreement, not tensor/logit equivalence or a quality certificate.

| Framework / route | Prompt-phase observation | Decode observation |
| --- | ---: | ---: |
| MLX-LM BF16, three-trial mean | 371.752 ms inferred from first-token throughput | 18.726 generated tok/s |
| rvLLM combined Metal, four measured ABBA/BAAB children | 1774.361 ms reported prefill | 4.646 tok/s computed from 64 tokens / mean reported decode time |

The descriptive ratios are **4.773× MLX advantage in prompt phase** and
**4.031× in decode** at M101. They are planning-grade, **not** a strict paired
cross-framework speed verdict: the jobs were not interleaved, MLX derives its
prompt phase from first-token throughput while rvLLM reports prefill
separately, and decode denominators may differ by the first generated step.
The queue reported exit 0 and no kernel failure, but marked the MLX condition
sample ineligible solely because one power-observer sample was stale
(age 2.584 s); observed controls were AC / power mode 2 / thermal state 0,
with no sampled competitor. No thermal wait or favorable rerun was used.

The complete MLX trial JSON is `mlx-it-m101-g64-report.json`, SHA-256
`91da1b613bf0e3d0f8452866b565bd4fc08ee1b64d48c26b2a4cfb002b671252`.
Its immutable outer queue receipt, including the freshness violation,
conditions journal, manifest and stdout/stderr, is
`mlx-it-m101-g64-queue-result.tar.gz`, SHA-256
`f18e2a2fd860039ecaf6b801df703e5e047df1d0f5db2c53fd09ac6b3ea6b68a`.
The dependent M304 MLX job and independent rvLLM confirmation were still
queued when this M101 result was archived; neither result is inferred here.

## Prompt-order confound probe (submitted, not scored)

The M304 prompt was the **second** case in every child of the first
counterbalanced run. That position can benefit from within-process model,
command-buffer, or cache warmup, and the remarkably low M304 candidate time
must not automatically be generalized to an M304-first workload. To isolate
this confound, the safe-Rust ABBA driver now accepts only the two explicitly
sealed expected-length orders `[101, 304]` and `[304, 101]`. Its legacy
default remains `[101, 304]`, so the frozen first and second jobs retain their
original work. The receipt still rejects changed token counts, output IDs,
dispatch, inference compilation, and an overwritten output directory. Focused
driver tests passed 3/3, including acceptance of the reversed valid case and
rejection of a mismatched declared order; the release binary built locally.

The new immutable queue job
`prefill26-fullroute-abba-combined-03-reversed-20260927` was accepted with a
fresh output path. Its reversed prompt JSONL SHA-256 is
`152ba16b32f0b5f9b73f61678e78f5d5af7dc978f0c4613e920c24c92142e992`,
its config SHA-256 is
`5c57c25dd31ab6f623936a6aed6f5b02f8d7cca23fcc735359370a94c7a0aa88`,
and its frozen driver executable SHA-256 is
`5112cbdd5265c8af0a9b9ed252d34dbf9c7148d00e7008de682909ed7ab6ac36`.
The driver source SHA-256 is
`6932185e815a27ed4e3ad2c19cea7530612c30588a4b012fb0739f91e07342d9`.
The same original weights, inference executable, metallibs, 64-token
continuation, fixed warmups, ABBA/BAAB measurement order, 5% drift threshold,
and queue condition observation remain in force. This is a predeclared
**order-sensitivity test**, not evidence of a speed result until its terminal
receipt and all 12 children are inspected. It does not replay any completed
job or alter a production default.

## Independent same-order ABBA confirmation

The separately submitted immutable job
`prefill26-fullroute-abba-combined-02-20260927` finished successfully (exit 0,
no violations, eligible sampled AC / power-mode-2 / thermal-state-0 stratum).
It used the same pinned configuration except for a fresh output path and
retained all 12 children, including four fixed warmups. All measured children
matched the expected 64 generated IDs in both cases, actual route dispatch,
and zero inference-case compilation. An independent calculation from the
retained eight measured child arrays reproduced the order-specific ratios.

| Prompt | Control prefill samples (ms) | Combined prefill samples (ms) | ABBA / BAAB ratio | Five-percent within-arm drift |
| --- | --- | --- | --- | --- |
| 101 tokens, first case | 6023.265, 5960.802, 6225.040, 5888.634 | 1899.539, 1688.483, 1886.816, 1799.092 | 3.340× / 3.286× | **Fail**: control 3.350%, candidate 11.111% |
| 304 tokens, second case | 12296.057, 12337.055, 12296.130, 12477.744 | 654.034, 639.720, 655.545, 635.646 | 19.040× / 19.187× | **Pass**: control 1.478%, candidate 2.811% |

Thus the M304-second speed screen replicated in a separate queue job with
both ABBA and BAAB blocks passing drift; the M101-first speed gate failed
again, and its unfavorable observations remain intact. The repeat strengthens
the **fixed-workload-position** result, not an arbitrary M304 prompt claim:
both jobs put M304 second, so within-process prompt-order/cache effects remain
unresolved. The queued reversed-order job is the predeclared probe of that
limit. None of these runs supplies an independent tensor/logit reference,
checkpoint-wide quality gate, matched MLX timing boundary, or production
promotion.

The second raw driver summary SHA-256 is
`d328e5b48bb85bfded1e2f4da64a6094baa63adf5e027d2234653298fa30251b`.
The complete queue receipt `abba-combined-02-queue-result.tar.gz` has SHA-256
`c39e6f120df73754f4e1c1d9d5a59b0e9d2f3ae21da26e1745ab5f4d175dbd91`.
All 12 inner child receipts, config and summary are in
`abba-combined-02-children.tar.gz`, SHA-256
`9618e8457991d135ce18544076df7899513f84d02401101fe7fbc2cb15a75ccf`.
No completed job was replayed or modified.

## Reversed prompt order: position materially changes the ratio

The predeclared `prefill26-fullroute-abba-combined-03-reversed-20260927`
probe completed successfully (exit 0, no violations, eligible sampled AC /
power-mode-2 / thermal-state-0 stratum). It ran **M304 first and M101
second**, with the same model/inference/metallib identities, fixed four
warmups, ABBA/BAAB arm order, 64-token continuations and 5% drift gate. All
12 children retained the expected full generated-ID trajectories, actual
control/candidate dispatch, and zero inference-case compiles. Independent
recalculation from all eight measured children reproduced both ratios.

| Prompt position | Control prefill samples (ms) | Combined prefill samples (ms) | ABBA / BAAB ratio | Five-percent within-arm drift |
| --- | --- | --- | --- | --- |
| M304 **first** | 13679.860, 13699.610, 13855.205, 13813.848 | 2278.586, 2328.255, 2335.224, 2316.758 | 5.943× / 5.948× | **Pass**: control 1.282%, candidate 2.486% |
| M101 **second** | 4569.423, 4571.355, 4570.747, 4588.049 | 261.863, 262.639, 261.959, 261.870 | 17.428× / 17.484× | **Pass**: control 0.408%, candidate 0.297% |

This **changes the interpretation** of the two same-order M304 results:
their ~19× ratio is repeatable for M304 *as the second case*, but not a
shape-general M304 speedup. With M304 first, the observed gain is ~5.95×.
Conversely, M101 moved from a noisy ~3.3–3.5× first-case signal to a stable
~17.5× second-case signal. This is strong evidence for a prompt-position
interaction, especially in the candidate route. It does **not** identify its
mechanism: within-process cache state, residency, scheduling, or another
warmup effect remain hypotheses, not measured per-kernel attribution. The
control also changes with position, so a simple candidate-only cache story
would overstate the evidence. Future comparisons must specify workload order
or measure each prompt in isolation; the reversed run does not retroactively
pass either first-case M101 drift failure.

The reversed raw summary SHA-256 is
`32a49c766f04be3f1dac11bc936e45659710fcd39304da284f02d29d6dd958d2`.
The outer queue receipt `abba-combined-03-reversed-queue-result.tar.gz` has
SHA-256 `9a97dda3027ea9c0e217e4fdd28aba3e956a1e82d18e79837658ab790eb077cc`.
All 12 inner children, config and summary are retained in
`abba-combined-03-reversed-children.tar.gz`, SHA-256
`4651f706264617ef4453c143f6bad5285868a58b4ab710f75214c3331aff82ea`.
This remains a research route, **not** a production promotion or an
independent tensor/logit/reference quality gate.

## Standalone M304 ABBA follow-up (submitted, not scored)

The reversed-order result motivates a direct single-prompt comparison: MLX's
M304 harness runs that prompt alone, whereas the original rvLLM M304 was
second in a two-prompt process. The safe-Rust ABBA driver now also admits only
the sealed singleton workloads `[101]` and `[304]`, in addition to the two
existing orders. Its default remains `[101, 304]`. The output validator
requires the exact declared case count and length, as well as the prior
route, generated-ID, dispatch, zero-inference-compile and fresh-output checks.
Focused tests passed 3/3, including singleton acceptance and rejection of a
wrong length or extra case, and the release binary built locally.

The new immutable serial-queue job
`prefill26-fullroute-abba-combined-04-m304-only-20260927` was accepted with
one pinned M304 prompt, a fresh output path, the same model/inference/
metallib identities, fixed warmups, ABBA/BAAB order and 5% drift gate. Its
prompt JSONL SHA-256 is
`72e42ab11ef82d9a1779c2d93c9f30c3bbc39bc9f5ea478cc293ec18a4b68406`,
config SHA-256 is
`a6b52c4cdffc4c34e1337fc0d7422599a95080f79c4f364efba8467f6c89e782`,
frozen driver executable SHA-256 is
`731958aa478f463d943986046a4a731a52f86cf112ab399c926a8908aedfbb3b`,
and source SHA-256 is
`a45eb2b8d55323b633bcdc6c5e2dd4b2d3f2586b2d0466a752a7a22736f98c43`.
No speed or quality result is asserted before the terminal queue receipt.
This job does not replay or rewrite any completed run.

## Same-checkpoint MLX comparison: 304-token prompt

The pinned `prefill26-mlx-it-bf16-m304-g64-20260927` job succeeded with one
warmup and all three 64-token MLX-LM trials. It used the same original
12B-it safetensor and exact 304 prompt token IDs as the rvLLM reversed-order
M304-first case. All three MLX trials produced the **same full 64 generated
IDs** as the rvLLM control and combined-route children at that prompt. This
is cross-framework output-ID agreement, not a tensor/logit oracle.

| Framework / workload position | Prompt-phase observation | Decode observation |
| --- | ---: | ---: |
| MLX-LM BF16, M304 alone, three-trial mean | 659.799 ms inferred from first-token throughput | 18.489 generated tok/s |
| rvLLM combined Metal, M304 **first** in two-prompt ABBA, four measured children | 2314.706 ms reported prefill | 2.743 tok/s computed from 64 tokens / mean reported decode time |

The descriptive MLX advantage is **3.508× in prompt phase** and **6.739×
in decode** against the closest available M304-first rvLLM data. These are
planning-grade, not a strict paired verdict: MLX runs the prompt alone,
rvLLM's M304 was first of two cases in each process, MLX infers prompt phase
from first-token throughput, rvLLM reports prefill separately, and decode
denominators may differ at the first generated step. The queued singleton
rvLLM M304 ABBA job will remove the two-case position difference, but not
the timer-boundary or process-interleaving differences.

The MLX queue job exited 0 with no kernel failure but was marked sampled-
condition **ineligible solely for one stale power-observer sample** (age
3.480 s). Its observed controls were AC / power mode 2 / thermal state 0,
with no sampled competitor; no thermal dwell or favorable rerun occurred.
The complete MLX JSON `mlx-it-m304-g64-report.json` has SHA-256
`0fd61d1eb42268047ea753f57304ec6e1870fe05e4eb7c9657681fd55b14d621`.
The immutable outer queue receipt, including its freshness violation,
conditions, manifest and stdout/stderr, is
`mlx-it-m304-g64-queue-result.tar.gz`, SHA-256
`df993c5b4ea10326418b031227de7d5e72437a4f0d84156213e9a96020134abc`.

## Standalone M304 result: output agreement, timing drift failure

The immutable singleton job
`prefill26-fullroute-abba-combined-04-m304-only-20260927` completed successfully
(exit 0, no queue violations, eligible sampled AC / power-mode-2 /
thermal-state-0 stratum). All 12 children, including the four fixed
warmups, ran exactly one 304-token prompt, dispatched the expected control
or combined candidate route, compiled no library or pipeline state inside
the inference case, and produced the **same full 64 generated IDs** as one
another and the three standalone MLX trials. The ABBA/BAAB ratios were
independently recomputed from every retained measured child.

| Control prefill samples (ms) | Combined prefill samples (ms) | ABBA / BAAB ratio | Five-percent within-arm drift |
| --- | --- | --- | --- |
| 13727.966, 13977.460, 13823.613, 13837.877 | 2191.719, 2226.328, 2023.257, 2071.590 | 6.271× / 6.755× | **Fail**: control 1.817%, candidate 7.686% |

The candidate's first-to-later measured samples moved enough to reject the
predeclared 5% gate. This is an **inconclusive standalone speed screen**,
not a qualified 6× result, despite its direction and exact generated IDs.
No sample was dropped, the threshold was not relaxed, and this completed job
will not be replayed. It is also consistent with the earlier observation
that later work in a process can be faster, but does not identify a cache or
kernel mechanism.

For planning only, the mean combined standalone prefill was 2128.223 ms
versus MLX's inferred 659.799 ms (~3.226× MLX advantage), while rvLLM
computed 2.748 tok/s from 64 tokens / mean reported decode time versus MLX
18.489 tok/s (~6.727×). The rvLLM timing drift failure, MLX's stale power
sample, noninterleaved processes, different prompt-phase timing boundaries,
and possible first-generated-step denominator difference prevent a strict
cross-framework verdict. The new standalone ratio is close to the earlier
M304-first two-case planning orientation, but does not independently qualify
that cross-framework comparison.

The raw singleton summary SHA-256 is
`2da4fa49909b70a09c67afa3212b984d0f7c8264c34e653db57b9878b36bfc32`.
Its complete outer queue receipt is
`abba-combined-04-m304-only-queue-result.tar.gz`, SHA-256
`fb5f7d16714317871ae89ab2710cff56dad0e882b3244a33f0e5fb40be7749e9`.
All 12 inner child receipts, config and summary are retained in
`abba-combined-04-m304-only-children.tar.gz`, SHA-256
`ca2b7077514d187ded02ffa59eb68ec5ccc2c69f6bf59d6334c494e0c064b23c`.
The next accepted speed campaign would need a *predeclared* variance-robust
protocol and independent numerical/logit/reference quality, not a favorable
resample of this job.

## One-step logit diagnostic: same token, measurable distribution difference

Two immutable correctness jobs, `prefill26-logits-m304-off-g1-20260927`
and `prefill26-logits-m304-combined-g1-20260927`, run the identical 304-token
original-12B-it prompt through the normal and combined BF16 Metal routes.
They pin the original model, tokenizer, inference executable, and respective
metallibs; the candidate depends on the control's queue success. Each requests
one generated token and the existing diagnostic top 256 logits. Neither is a
timing job, and neither changes the kernel-selection path to collect a trace.

The new safe-Rust `rvllm-prefill-logit-compare` checks the terminal queue
receipts, identities, actual dispatch, prompt and generated IDs, and finite,
unique, sorted top-logit entries before writing a create-new comparison.
It records both jobs' sampled-condition eligibility rather than rejecting a
run solely for a stale observer. Its focused tests passed locally. The hook
reads logits **after the single decode step**, so this is a limited
distribution diagnostic, not the prefill-boundary logits, a full-vocabulary
comparison, a first-internal-difference trace, an independent reference, or
a checkpoint-wide quality gate.

Both jobs finished successfully (exit 0, unchanged pinned files, no queue
violations), with eligible sampled AC / power-mode-2 / thermal-state-0
conditions. They used the same 304 prompt token IDs, BF16 compute and BF16
weights, and generated the same one token ID, `107`. The control had no
research dispatch; the candidate actually dispatched combined projection
GEMM 48, QKV 48, raw projection 96, raw norm 96, D256 attention 40, and D512
attention 8. Each report totals one library compile, with 58/64
pipeline-state compiles respectively; the single-prompt report does not
separate preparation from inference compile calls. These correctness jobs
are **not timing evidence**: their unpaired prefill times were 13,929.570
and 2,197.848 ms.

After the one decode step, both top-ranked logit IDs were `107`, but their
values were 14.625 (control) and 14.8125 (candidate). Of each arm's top 256
IDs, 248 overlapped; the ranked lists were not identical, eight control IDs
were absent from the candidate top 256, and the largest rank shift among
common IDs was 29. Over only those 248 common IDs, the maximum absolute logit
difference was 0.296875 and the mean absolute difference was 0.071037. This
proves a measurable route-dependent output difference for this one-step
probe; it does **not** establish whether either route is numerically wrong,
what its earliest internal difference is, or whether checkpoint quality
changes materially. In particular, one matching generated ID does not make
the two output distributions equivalent.

The final create-new comparator output is `logits-m304-g1-comparison-v2.json`,
SHA-256 `d22515efbb30de6adf9bfb792a52346961a1235fb5a6a25ae6d68b58d10c9cb8`.
An earlier derived output, `logits-m304-g1-comparison.json` (SHA-256
`f97448dab0aad9e9462047a795f088436898eb75a2d78c3cf64eb44ea42db737`),
is retained for audit but its claim incorrectly suggested preparation
compilation had been separately measured. The v2 comparator corrects that
claim and records the unseparated compile totals; its numerical comparison
is unchanged. Neither output is a queue job or a new measurement.
The complete compact outer queue receipts for both immutable jobs are in
`logits-m304-g1-queue-results.tar.gz`, SHA-256
`3423d6504e3403a280b3aa6f317353845664613beea820a87a27072a5fbf6067`.
No job was replayed. The next numerical gate needs a route-preserving
internal-tensor/reference comparison and broader checkpoint-specific prompts,
not promotion from token agreement or this truncated top-logit comparison.

The existing original-12B-it CPU/Hugging Face 16-step reference for the short
six-token capital prompt was checked in two new, independent correctness
jobs: `prefill26-hf-capital16-off-20260927` and
`prefill26-hf-capital16-combined-20260927`. Each pinned the same checked-in
reference, executable and checkpoint, with its respective metallib and
selector. Both queue jobs succeeded with exit 0, unchanged pinned files,
eligible sampled AC / power-mode-2 / thermal-state-0 conditions, and no
violations. The exact prompt IDs were `[2, 818, 5279, 529, 7001, 563]`.
Both routes generated all **16 IDs exactly as the independently recorded
CPU/HF reference**, and both reported `hf_reference.matched=true` with no
mismatches.

The control research-dispatch ledger was empty. The combined selector's
actual ledger recorded tiled GEMM 144, QKV 48, D256 attention 40 and D512
attention 8, without overflow. It did **not** dispatch combined raw
projection or raw norm on this tiny prompt; thus the result does not
independently validate those important M101/M304 prefill components. The
checked-in reference comparison asserts token IDs only, not logits or
internal tensors. This is a useful independent short-route token check, not
long-context or checkpoint-wide numerical quality, and not speed evidence.
Neither job was replayed or held for thermal stability.

The complete compact outer receipts, including both manifests, conditions,
stdout/stderr and terminal reports, are in
`hf-capital16-queue-results.tar.gz`, SHA-256
`85bf379b904ee11a5e98eb98d597f98b11de9db739d312b331d8dfd8a2cd3a69`.

## Long-prompt teacher-forced loss probe: complete diagnostic

The default-off `metal-quality-research` CLI feature adds a **safe-Rust**
teacher-forced diagnostic to the ordinary single-prompt Metal route. It
performs normal prefill and decode selection, reads the full-vocabulary logits
after each decode collect using the existing backend probe, scores a fixed
target token, and feeds that target into the next step. The probe adds a GPU
readback and synchronization at every step; **none of its timings is speed
evidence**. It reports each greedy sampled ID separately from each
teacher-fed ID, token rank/logit/NLL and aggregate mean NLL/perplexity. It
does not pass trace scratch into kernel selection. Its CLI is unavailable in
ordinary builds. Focused feature-enabled tests passed 14/14, and the
ordinary binary tests passed 11/11; the feature-enabled Apple release build
passed. The frozen diagnostic executable SHA-256 is
`6d79ae67f0e7fde8500e1b5b7d901940edab3d9c3aae484e83fd166950d5bfec`.

Two independent immutable serial-queue correctness jobs were submitted:
`prefill26-teacher-m304-off-20260927` and dependent
`prefill26-teacher-m304-combined-20260927`. They pin the original 12B-it
checkpoint, executable, respective metallibs, exact M304 prompt JSONL, and
the same-checkpoint MLX report. The 16 target IDs are the first 16 IDs from
that MLX run, sealed in `teacher-m304-mlx16.json` (SHA-256
`9984b53cc02946d912048a860a69fe9c95f1e279a66487c4612ab7762f97c53b`).
Both immutable jobs finished successfully with exit code zero, unchanged
pinned files, eligible sampled conditions, and no queue violations. They
produced the same 304 prompt IDs and 16 teacher-fed IDs. The control research
dispatch ledger was empty. The combined route actually dispatched tiled GEMM
48, QKV 48, raw projection 96, raw norm 96, D256 attention 40, and D512
attention 8, without overflow. This closes the tiny-prompt *dispatch coverage*
gap, but not the independent numerical-reference or checkpoint-quality gate.

Every MLX-derived target was rank one and was the greedy sampled ID in **both**
rvLLM routes at all 16 steps. Summed target NLL was 2.1273205155 for control
and 2.0495501052 for combined (candidate minus control −0.0777704104);
mean NLL was 0.1329575322 versus 0.1280968816, and exp(mean NLL) was
1.14220149 versus 1.13666312. The candidate had a lower NLL on this one
fixed, model-generated trajectory, but individual deltas had both signs:
step 5 was +0.0366221 and step 15 was −0.1027260. The complete 16-row
comparison, including target IDs, both greedy IDs, ranks, and both NLLs,
is in `teacher-m304-comparison.json` (SHA-256
`53726586f682d7368658afa80695372026e6681c3c8e14cfd30bb8404aefed8f`).
The complete outer queue receipts (manifest, terminal report, conditions,
stdout and stderr for both jobs) are in `teacher-m304-queue-results.tar.gz`
(SHA-256 `e4697ceab2e22734767207f9f3c04403073cc9771ae06089a20bbb2b54429ab7`).
The reported total compile counters were one library and 58/64 pipeline states
for control/candidate; this single-prompt report does not separate preparation
from inference compilation. No job was replayed or held for thermal stability.

The trajectory comes from same-checkpoint **MLX-generated tokens**, not
held-out truth. Agreement that they remain rank one is useful, and the
route-dependent distribution difference is measurable, but neither arm is
independently certified as numerically correct. The per-step readbacks also
invalidate every timing field in these jobs as a speed measure. No promotion
follows from this diagnostic; next use an independent same-checkpoint
reference on the long prompt and broader held-out continuation quality.

The first independent CPU attempt, immutable job
`prefill26-hf-m304-one-step-20260927`, failed before model loading or
inference. Its pinned Python environment has Transformers 5.8.1, which did not
recognize the checkpoint's `gemma4_unified` model type. The terminal queue
report records exit 1, unchanged input pins and no queue violation; the job
was quarantined, not replayed. The complete failed result directory,
including its manifest, conditions and stderr, is preserved in
`hf-m304-one-step-failed-queue-result.tar.gz` (SHA-256
`a332466773428eda3e53359f28bdb5b9761886e9bca0c667f38f1ac50501a80f`).
This is a reference-environment failure, not a Metal result or CPU numerical
comparison.

A separate installed Python environment reports Transformers 5.14.1 and
recognizes this checkpoint as `Gemma4UnifiedConfig`; its
`AutoModelForCausalLM` mapping resolves to
`Gemma4UnifiedForConditionalGeneration` without loading weights. New immutable
job `prefill26-hf-m304-one-step-compatible-20260927` uses the **unchanged**
reference script and exact same 304 prompt IDs, checkpoint, one-step full
logits and top-16 request. Its manifest
`hf-m304-one-step-compatible-job.json` (SHA-256
`7af03b53d77cd57ee47824c1e60b1990f1fe289d98dc79731b60bc9e2b57cd55`)
pins that Python executable and the installed Gemma4 implementation files in
addition to the original model and fixture. It finished successfully with
exit zero and unchanged pinned files. The queue marked its condition sample
ineligible: 13 not-ready observations had power-observer ages 2.507–2.821 s,
although sampled controls were AC, power mode 2, thermal state 0 and no
competing process. This affects any timing claim, not the one-step numerical
diagnostic. No job was replayed or held for stable conditions.

The CPU run returned all **262,144** finite BF16-derived logits after the
exact same 304 prompt IDs and chose token **107**, as did both rvLLM routes.
For token 107, max-shifted full-vocabulary log-sum-exp gives CPU target NLL
**0.46601746**; rvLLM control was **0.68074801** and combined was
**0.65161510**. Their NLL excesses relative to this CPU implementation are
**0.21473055** and **0.18559764**, respectively. Raw target logits were
14.9375 CPU, 14.625 control and 14.8125 combined. The CPU top three IDs
were 107, 108, 106 (the latter two tied at 13.6875); both Metal routes
ranked 106 ahead of 108. All CPU top-16 IDs occurred within each earlier
Metal top-256 diagnostic. Thus the combined route was closer in this one
target NLL, but **neither Metal route reproduced the CPU distribution**. A
shared difference is visible; this does not localize a first incorrect
arithmetic step or establish that the candidate is numerically better in
general.

The complete CPU terminal queue directory and full-vocabulary reference are
in `hf-m304-one-step-compatible-queue-and-reference.tar.gz` (SHA-256
`ef18f3a70e16abb36ddea043c46f1ac715002f4801a456824090d0755210fab2`).
The uncompressed reference has SHA-256
`e3bdfd440109606d0780d8bc82a5d6d25eb6059dd301d4a42ec02f09d35cc2ef`.
`hf-m304-first-step-comparison.json` (SHA-256
`71ad4c7db5dd68be8241416bdb3944e22e46a045e2f0a2d5d1ddc1b315097d70`)
records exact source receipts, NLLs, ranks and limitations. The CPU
implementation is independent, but uses Transformers 5.14.1 rather than the
checkpoint config's 5.10.0 development version; it is not a certified exact
arithmetic oracle. The CPU final full-prompt forward and Metal post-decode
readback predict the same next-token position, but their internal execution
boundaries are not proved identical. This is one position, not held-out or
checkpoint-wide quality, and none of these readback jobs supplies speed
evidence. Next localize the internal difference and evaluate predeclared
held-out targets before any promotion.
