# Q4K16 short real-weight route triage (2026-09-27)

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

Two additional correctness-only jobs are queued on the *same* varied input:
`prefill26-fullroute-varied-pipeline32x64-20260927` tests the projection
pipeline alone, and the dependent
`prefill26-fullroute-varied-pipeline32x64-q4k16-20260927` tests the combined
projection and attention selection. They are distinct selectable metallibs
with pinned hashes. No result or speed claim is attached to those jobs yet.
