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

The next correctness-only pair is queued, not adjudicated:
`prefill26-fullroute-varied-off-20260927` and
`prefill26-fullroute-varied-q4k16-20260927`. It uses the pinned
`varied-prompts.jsonl` (SHA-256
`cba2a6830503e5e95e37f1c677197dd3d93a26e0dcf33c501be973ec0cedc483`),
which contains two non-repetitive prompts with up to 64 generated tokens each.
The candidate job depends on successful completion of the control. Neither
job is a timing qualification; no result is claimed before queue receipts.
