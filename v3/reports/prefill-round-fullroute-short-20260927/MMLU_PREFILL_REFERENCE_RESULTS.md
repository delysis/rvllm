# Same-boundary reference trial: receipts retained, strict seal rejected

The six immutable serial-queue jobs declared in
`MMLU_PREFILL_REFERENCE_PROTOCOL.md` all terminated successfully with exit
code 0 and unchanged pinned files. The four Metal control/combined arms had
eligible sampled conditions; both CPU/HF arms were **ineligible** solely
because the queue power observer's sample exceeded its 2500 ms freshness
limit. No job was replayed, no thermal-stability wait was introduced, and
the queue's threshold was not changed.

| Arm | Terminal result | Sampled eligibility | Preserved condition violation |
| --- | --- | --- | --- |
| logic HF | success, exit 0 | false | one not-ready power sample, age 2634.174 ms |
| logic Metal control | success, exit 0 | true | none |
| logic Metal combined | success, exit 0 | true | none |
| computer-science HF | success, exit 0 | false | three not-ready power samples, ages 2523.426, 2540.614 and 2525.264 ms |
| computer-science Metal control | success, exit 0 | true | none |
| computer-science Metal combined | success, exit 0 | true | none |

The ineligible observations still sampled AC power, `pmset` mode 2, low-power
mode off and nominal thermal state 0, with no sampled competing process.
Those observations do not erase the recorded freshness failures. The
combined Metal routes did actually dispatch tiled GEMM, QKV, raw projection,
raw norm and Q4 D256/D512 prefill attention; the frozen receipts retain
the exact counts. All six complete outer result directories, including
`job.json`, `report.json`, `conditions.jsonl`, `trial.stdout` and
`trial.stderr` for each, are preserved in
`mmlu-prefill-reference-queue-results-v1.tar.gz` (30 files; SHA-256
`db76ef52bbd61116e2ed8a8c85a9d08440f89fa4506b09c849bf6dfbc1f728db`).
The HF stdout records only the output-file location and generated token.
The two separately written, full-vocabulary HF JSON files are retained in
the supplemental `mmlu-prefill-reference-hf-outputs-v1.tar.gz` (SHA-256
`e3f1287a55e00e322aa531c091d39042244e069dbb9bac3dfdbdb327977657be`).
The formal-logic JSON SHA-256 is
`38bf3e5e4520ce627b9a20db7c432ff856e440b9534205c6d09c4d3d944242a0`;
the computer-science JSON SHA-256 is
`30e4b96624d582641e685e81c339408ef2095a4919622ceaa2af639b0b14a688`.
Each records the exact 230/254 prompt IDs, one selected first target,
generated token 107 and 262144 full logits. This supplemental archive fixes
an evidence-packaging omission without modifying the original queue archive
or either HF output. The frozen Metal probe records only target logit,
target rank, target NLL and sampled greedy ID.

The original fail-closed safe-Rust
`rvllm_gemma4_prefill_reference_summary` source (SHA-256
`fde5adc88a9ca0f9fdb7a7f80862015c6926184e8403c01a696a64bbc28648be`)
had a one-character typo in the hard-coded SHA for the **tracked, unchanged**
logic-HF manifest. Its first executable invocation therefore rejected with
`prefill26-mmlu-prefill-ref-logic-hf-v1-20260927: frozen manifest changed`
before inspecting the queue receipts. The actual tracked manifest SHA-256
is `144965b1c3dc4967b3d828c66b404921e94631f269760e21aa6512c6d3fab4db`.
The referee source was repaired to that exact hash and gained a focused
test checking all six frozen manifest hashes; all three focused tests and a
targeted build passed. The repaired source SHA-256 is
`22932696b257f4a2d869f88e8f2307e1c35013e301ec56be71c5dc9f53f10142`.
No job, manifest or receipt changed in this repair.

The repaired, otherwise unchanged strict referee then rejected with
`prefill26-mmlu-prefill-ref-logic-hf-v1-20260927: queue receipt is not clean and terminal`.
Its policy requires an empty violation list. It produced **no** accepted
summary JSON. The computer-science HF receipt independently has the same
kind of disqualifying violation, as shown above. We preserve this result
instead of weakening the rule, selecting only eligible arms, or resubmitting
a completed job to chase a fresh observer sample. Accordingly, this trial
has **no sealed HF-versus-Metal numerical-parity verdict**.

The intended comparison is narrow even with clean conditions: CPU/HF
full-prompt step zero and Metal's final prefill row address the same logical
next-token position; Metal teacher decode step zero replays the final prompt
token and is a different boundary. CPU Transformers 5.14.1 differs from the
checkpoint's recorded development version, BF16 arithmetic may differ, and
the Metal probe does not expose the full vocabulary or top-token margin.
Probe synchronization makes all timing unusable. Two Codex-wrapped MMLU
questions are not an MMLU score, checkpoint-wide quality assessment, speed
qualification or promotion.
