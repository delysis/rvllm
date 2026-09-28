# Prospective balanced natural-question W4 diagnostic

This protocol freezes a larger **diagnostic** input set before any inference
job is submitted. No device result, threshold, benchmark score or promotion
follows from this document. The source is the locally cached `cais/mmlu`
`all/test` Arrow file at revision
`c30699e8356da336a370243923dbaf21066bb9fe`, SHA-256
`58769d67ced092719390e496f1d4bbf8b99c43b77bba39d054b9a09aecb63cab`.
The [revision-pinned dataset page](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)
identifies the source and MIT license. The questions and choices are dataset
text; the instruction wrapper and answer-repetition continuation are
Codex-authored. The test split may have appeared in model training.

Selection is deterministic and independent of all model outputs. Number rows
in their Arrow order from zero. Exclude previously used rows 2443 and 3185.
Keep rows with a nonempty question and four nonempty choices, with Python
Unicode-character length of question plus choices from 700 through 1800 and
the labeled choice no longer than 350 characters. Sort eligible rows by the
lexicographic lowercase SHA-256 hex digest of UTF-8
`rvllm-mmlu-fulltext-w4-v2:<decimal row index>`. Traverse this order,
accepting the first 16 rows with unique subjects and no more than four rows
per labeled answer A/B/C/D. The result has four of each label and 16 distinct
subjects. Neither question content nor candidate output was used to adjust
selection. The selected global indices, in sealed order, are 10978, 12398,
3441, 5622, 6075, 5793, 13310, 2512, 9802, 12568, 9551, 841, 10376,
1227, 10089 and 3236.

The create-new source file `mmlu-balanced16-fulltext-w4-source-v1.json`
has SHA-256
`c99f34ae929484d019630ada0589ff9fbd7149e86b6109efd513ebf977cd1c22`.
Every question, choice and labeled answer was checked character-for-character against
the corresponding cached Arrow row. Each prompt uses the same fixed
multiple-choice wrapper as the earlier two-row MMLU diagnostic and ends at
`Answer:`. The forced continuation is a space, the labeled letter, a period,
a space, and the complete labeled choice text. The existing `#![forbid(unsafe_code)]`
Rust tokenizer fixture builder verified exact prompt-prefix tokenization
against original tokenizer SHA-256
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
The sealed token fixture `mmlu-balanced16-fulltext-w4-tokens-v1.json`
has SHA-256
`ff019f8554630f2c8c6ca5165728109299e17a66d6cc2a8fc5bc8325fbab2b15`:
16 prompts total 3872 tokens, 341 forced targets, largest prompt 381 and
largest continuation 43 tokens. All 16 cases meet the builder's 128–1024
prompt and 1–64 target bounds. If any source/token pin changes, abort rather
than silently replacing a row.

The source-bound safe-Rust generator and separate fail-closed referee are
stored in `v3/crates/rvllm-runtime/src/bin/`. The authoritative create-new
`mmlu-balanced16-w4-queue/` directory has 32 immutable manifests and 16
single-prompt JSONL files. The referee freezes the SHA-256 of each manifest
in case/route order and its host tests check all 32 files. Sealing these
files is host preparation, **not numerical evidence**. Before submitting
each wave, independently review its manifests and output paths, check the
live serial queue and disk budget, and retain the
separate unsubmitted `mmlu-balanced16-w4-queue-dry-run/` directory.

The frozen
teacher CLI requires **one JSON prompt and one target list per process**;
it cannot score four cases in one job. Without a separately implemented and
tested safe-Rust multi-case teacher route, this trial requires 32 fresh
jobs, paired BF16 donor then W4 for each of the 16 cases, never simultaneous
accelerator work. Two prior clean W4 teacher *process measurements* took
408–457 seconds each, and their BF16 controls 33–36 seconds each. Those
figures exclude the queue's input/output hash verification. The prior two
complete pairs took about 27 and 25 minutes respectively from submission
to terminal receipt, so a naive 16-pair end-to-end extrapolation is closer
to seven hours before variable case length and queue gaps; it is not an ETA.
The queue source starts each job's 7200-second condition-wait clock only
after its dependencies become ready. Staging waves is therefore for
bounded evidence review and resource control, **not** a workaround for a
32-job dependency chain expiring while its predecessors run.
The manifests define fixed waves of four case-pairs, beginning at q00, q04,
q08 and q12, with no dependency between waves. Submit each wave only after
the previous wave is terminal; no completed ID is replayed. If the new
generator/referee or resource review cannot make this safe, stop before
submission rather than changing the selected cases or weakening the gate.
Each arm must pin
the original model/config/tokenizer, this source and token fixture, package
assets, exact executable/metallibs, prompt/target IDs and all required
queue conditions. No thermal-stability wait. Preserve terminal failures and
unfavorable positions; never replay a completed ID or overwrite an output.

The referee must require clean terminal receipts, unchanged pins, complete
condition journals, exact 16 cases and all 341 teacher positions, actual
named W4 projection/gate/QKV **decode** dispatch and donor SG8 attention,
and no W8 dispatch. It must report each target's NLL/rank/sampled greedy ID,
per-case and aggregate W4-minus-BF16 NLL with both signs, and all greedy or
rank changes. `generated_token_ids` are forced transport, not sampled model
agreement. Teacher readback invalidates timing; the frozen CLI does not
separately attribute generic low-bit prefill kernels. The BF16 donor is not
an independent numerical oracle. Even 16 balanced MMLU questions are not a
representative benchmark, a contamination-free holdout, or a calibrated
quality gate. Do not call the result MMLU accuracy or checkpoint-wide W4
quality, and do not promote the kernel from it.
