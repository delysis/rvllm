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
