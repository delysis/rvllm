# Distinct-input same-boundary numerical diagnostic, v2 (pre-run)

This is a **new** correctness-only trial, not a retry or reinterpretation of
the six MMLU reference jobs. Those jobs and the frozen strict referee retain
their original failed eligibility verdict. No v2 job has been submitted yet.

The immutable proposed source is `numerical-reference-distinct-v2-source.json`,
SHA-256 `2f212a4b1c316e49b52b78fc80e33a62c35772f0028e4da7d52ee7268c3d008a`.
Both passages and target continuations were authored by Codex before any model
scoring of these exact inputs. They are synthetic numerical diagnostics, **not**
natural held-out text or a quality benchmark. The original 12B-it tokenizer
SHA-256 is `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
The existing safe-Rust tokenizer fixture tool verified exact-prefix tokenization:

| Case | Prompt tokens including BOS | Continuation tokens | First target ID |
| --- | ---: | ---: | ---: |
| observatory-clock-v2 | 213 | 15 | 80444 |
| library-ledger-v2 | 203 | 31 | 506 |

The only scored target in this trial is the first continuation token. Its
identity follows the frozen source and tokenizer, not any model output. These
prompt lengths must actually dispatch the combined raw projection/norm and Q4
attention kernels; a fallback invalidates that arm. No continuation or
checkpoint-wide quality inference is allowed from one position per case.

For each case, create three **fresh** immutable jobs in dependent serial order:
HF full-prompt, Metal control prefill-final, Metal combined prefill-final. Use
new IDs `prefill26-prefill-ref-distinct-v2-{observatory,library}-{hf,off,combined}-20260927`.
The generator must pin the source, tokenizer, exact prompt-token file, original
12B-it config and safetensor, unchanged HF script/runtime, frozen safe-Rust
Metal probe executable/source, normal/combined metallibs, job generator, and
each command's output path. Refuse existing manifests and output directories.
The six jobs may be submitted only after the generator and a separate
fail-closed referee have passed focused host tests and the manifests have been
reviewed. Do not modify the old generator, manifests, referee, or receipts.

The HF full-prompt final-row output and Metal `prefill_last_step` refer to the
same *logical* next-token position. Metal `teacher_forced.steps[0]` follows a
replay of the final prompt token through one-token decode and is **not** the
comparison boundary. The Metal probe records only target logit, target rank,
target NLL, and sampled greedy ID; it does not serialize full vocabulary logits
or top margin. The HF full vector must be retained as a separately pinned
output artifact as well as its queue receipt. CPU Transformers version and BF16
math can differ from Metal; a discrepancy is not automatically a shader bug.

This v2 trial prospectively separates **numerical validity** from **timing
eligibility**. The queue's `correctness` purpose already allows a successful
receipt with sampled-condition ineligibility. The numerical referee must
require terminal success, exit zero, unchanged pins, no overrun, exact source
and token IDs, finite scores, expected candidate dispatch, and complete raw
receipt/output preservation. It must retain and report *every* sampled
condition and violation, including stale power observations. It may still
report numerical observations when the **only** condition violation is power
observer freshness; it must label them numerical-only/unqualified for timing.
Any other violation must fail closed pending explicit diagnosis. The old MMLU
strict referee must not be weakened or retroactively applied under this rule.

Report both cases, both Metal routes, all four observable score fields and
their deltas from HF, plus exact hashes and condition strata. There is no
post-hoc threshold, favorable-case selection, timing claim, arithmetic-cause
claim, or promotion. All probe timings are invalid because readback and an
M-row LM head add synchronization and work.

## Generator checkpoint (no queue submission)

The new safe-Rust `rvllm_gemma4_distinct_reference_job_gen` is separate from
the frozen v1 generator. Its source SHA-256 is
`defd8dcdf4e70768178dba27a5ad30d4e4d02d7f791376df48acc12c3f5843dc`;
two focused tests and a host build passed. It generated fresh, unopened
`numerical-reference-distinct-v2-queue/` artifacts without altering v1:

| Artifact | SHA-256 |
| --- | --- |
| observatory prompt JSONL | `6f5b0bf37acd7b0cc2f60c547c2f413bffc7f491b30fcc673686d8cd561d822d` |
| library prompt JSONL | `9d6febddc587c769cb0562735758d010d6b15afc3c76e689670633a167cf014d` |
| observatory HF job | `e39d87c30533067c35065ae745469ce48aa49a53a2b550fab8d89193fbec8382` |
| observatory control job | `c30da8b58dcc01c377707475985a74f3f732576dc8b55b887efe2f255b628079` |
| observatory combined job | `d85cc3b0f6fdf8a06a194dd683060bc897bba94b774bde61a2fc6da2d8589b4e` |
| library HF job | `e5addab1cab7c4bac3cbab4b2df467cc58a44332545a2fb382967af5d4564390` |
| library control job | `79b6d2e5fd858058e520c55a70d7ec668f79f2f535d193166e771a85dc1c491c` |
| library combined job | `361c7bd1b9b3c0caf7a0f470aa3249f9225319f24cfc5e4b8231d734a5020ccf` |

All six generated manifests have `purpose=correctness`, distinct immutable
IDs, the expected serial dependency chain, unchanged original checkpoint and
reference/Metal executable pins, correct new prompt/first-target IDs, and
`--teacher-prefill-last-logits` on Metal arms. The old MMLU source, dataset,
prompt and generator pins are absent.

The separate safe-Rust numerical referee
`rvllm_gemma4_distinct_reference_summary` has source SHA-256
`477ec2399afd35f91258ef397a3ba9f039106df941f85e1a791480880897c023`.
It retains hashes for every outer queue receipt and the separately written HF
full-vocabulary JSON, checks the complete condition journal against the
reported violations, and never reclassifies queue timing eligibility. A
pre-launch source audit tightened its power/CPU-control checks to match the
queue's readiness predicate; absent control fields cannot pass by default. Its
four focused tests and host build passed: exact six-manifest hashes, a stale
power observation accepted only as numerical-only, competitor/freshness
negative cases, rejection of missing combined dispatch, and HF tie ordering.
No v2 manifests have been submitted yet; review of the final referee and
the existing serial queue state precedes submission.
