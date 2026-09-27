# Two-question natural-source teacher-forced diagnostic

The four immutable serial-queue jobs declared in
`MMLU_NATURAL_PROTOCOL.md` all succeeded with exit code 0, unchanged pinned
files, eligible sampled conditions and no violations. Each observed stratum
was AC power, `pmset` mode 2, low-power mode off and nominal thermal state 0.
Neither thermal stability nor an idle machine was required. The control
research-dispatch ledger was empty in both cases. Each combined arm actually
dispatched GEMM 48, QKV 48, raw-norm projection 96, raw norm 96, Q4 D256
attention 40 and Q4 D512 attention 8; neither ledger overflowed. This is a
full-route *execution* observation, not operator-level numerical proof.

The safe-Rust `rvllm_gemma4_mmlu_teacher_summary` referee, source SHA-256
`15e9134fa442b68d35227cb39a337c53957d7b019b7e402b5f9a6cbf590be1fd`,
accepted all four terminal receipts and exact source, dataset, tokenizer,
model, executable, prompt and metallib pins. It checked every forced target,
teacher step, NLL aggregate and candidate dispatch. Four focused tests and a
build passed. An independent sum directly from every raw `trial.stdout` step
matched the four reported NLL totals. The complete all-position comparison
is `mmlu-natural-teacher-summary-v1.json`, SHA-256
`b99f902aa25fba2bf597f77907f821d18e2ee4fcbb1366dc92332dabd4c90877`.
The complete 20-file outer queue archive (job, report, conditions, stdout and
stderr for each arm) is `mmlu-natural-queue-results-v1.tar.gz`, SHA-256
`e1d44d0d82ecd83de72f15327d51a161b653e78e962c59f57ed27154a3f9903c`.
The summary also seals each of the 20 receipt hashes separately.

| Source row | Prompt / forced target tokens | Control NLL sum | Combined NLL sum | Combined minus control | Target-rank differences | Sampled-greedy differences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| formal_logic 2443 | 230 / 24 | 25.976044 | 9.486550 | -16.489494 | 2 | 1, at zero-based 17 |
| high_school_computer_science 3185 | 254 / 55 | 100.691900 | 92.996100 | -7.695800 | 7 | 1, at zero-based 39 |

The large formal-logic sum change is concentrated at target position 17:
control NLL 16.421412, target rank 171 and sampled ID 531 versus combined
NLL 0.013323, rank 1 and sampled ID 1091 (the forced target). That one
position contributes -16.408089 of the -16.489494 total. At the *first*
target position the combined route is slightly worse: NLL difference
+0.200650 in formal logic and +0.061937 in computer science. The computer
science greedy difference is at position 39: sampled IDs 236764 versus
236761, while the forced target is 563, ranked 8 versus 7. These unfavorable
and divergent observations are retained alongside every other position in
the JSON; no result was selected out of the two-case fixture.

`generated_token_ids` contains the supplied teacher-forcing targets, **not**
the model's greedy trajectory. Only `teacher_forced.steps[].sampled_token_id`
is a sampled-greedy observation. The target text repeats labeled choices in
a Codex-authored wrapper around two independently sourced MMLU questions;
unknown training contamination, only two selected rows, and no calibrated
threshold make this neither a MMLU score nor a checkpoint-quality verdict.
The control and candidate disagree in two greedy positions, so even their
forced-target NLL improvement cannot be called numerical equivalence.
Neither is an independent reference for the other. The existing CPU/HF
full-prompt step-zero script does not match this Metal teacher hook's
post-replay boundary without further proof. Per-step readback synchronizes
the route: **none of these timings are speed evidence**. No kernel promotion
follows from this diagnostic.
