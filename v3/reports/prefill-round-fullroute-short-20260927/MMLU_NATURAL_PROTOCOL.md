# Independently sourced continuation probe: frozen before model scoring

This protocol selects two questions from the locally cached `cais/mmlu`
`all/test` Arrow file at revision
`c30699e8356da336a370243923dbaf21066bb9fe`, SHA-256
`58769d67ced092719390e496f1d4bbf8b99c43b77bba39d054b9a09aecb63cab`.
The [revision-pinned dataset page](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)
labels it MIT-licensed; the [original project license](https://github.com/hendrycks/test/blob/master/LICENSE)
credits Dan Hendrycks (2020). This fixture includes only the two cited rows.
The fixed subjects are `formal_logic` and `high_school_computer_science`.
Within each subject, select its **first** test row whose question plus four
choice strings total at least 900 characters and whose labeled answer
choice has at least 100 characters. The resulting global row indices are
2443 and 3185. This rule and the two subjects were fixed without inspecting
Gemma/rvLLM/MLX scores or candidate outputs. A read-only extraction check
confirmed that each question and all four choices in
`mmlu-natural-source-v1.json` exactly match its Arrow row, and that its
continuation contains the row's labeled answer choice (D and C,
respectively). The prompt wrapper and answer-repetition format were written
by Codex; the questions and choice text came from the cached dataset.

The existing safe-Rust tokenizer fixture tool verified exact prompt-prefix
tokenization with the original 12B-it tokenizer SHA-256
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
The source file SHA-256 is
`a1d115405e86ab51bf275473cc04f452f185db2c18f286b5173d2234d4b993b4`.
The formal-logic case has 230 prompt and 24 continuation tokens; computer
science has 254 prompt and 55 continuation tokens, including explicit BOS
in each prompt. Both lengths exercise the candidate's long-prompt raw
projection/norm path if it actually dispatches. These are natural
multiple-choice questions, **not** a representative natural-prose sample;
the dataset may have appeared in model training. Two rows cannot establish
checkpoint-wide quality or MMLU benchmark accuracy.

Before any run, seal original checkpoint, source, tokenizer, fixture
generator, executable, normal/combined metallibs, prompt JSONL, target IDs,
and serial queue IDs. Run control and candidate for each case, with no
thermal-stability gate; preserve failed and unfavorable receipts. Teacher
forcing writes supplied targets into `generated_token_ids`; only
`teacher_forced.steps[].sampled_token_id` measures greedy behavior. Require
actual combined GEMM/QKV/raw projection/raw norm/Q4 dispatch, exact token
identities, and full per-step NLL/rank/greedy comparison. Report both
candidate-minus-control NLL sums and every rank or greedy difference,
including worse positions, without a calibrated acceptance threshold. All
readback timing is invalid. Independently seal a same-checkpoint CPU/HF or
MLX reference at the same next-token/logit boundary; neither Metal route
is an independent oracle for the other. No promotion follows from this
two-row probe by itself.

The safe-Rust `rvllm_gemma4_teacher_job_gen` emits this exact four-job
chain to `mmlu-natural-v1-queue/`, refusing existing output and source,
tokenizer or Arrow hash changes. Its source SHA-256 is
`e2e68b4854880743ef1f7d2e626e9347ae30715c7bfeae4777aef5a0c094198b`;
two focused tests and an actual fixture-generation run passed. The queue
order is `prefill26-mmlu-natural-logic-off-v1-20260927` →
`prefill26-mmlu-natural-logic-combined-v1-20260927` →
`prefill26-mmlu-natural-cs-off-v1-20260927` →
`prefill26-mmlu-natural-cs-combined-v1-20260927`. The four job-manifest
SHA-256 values in that order are `4a32900cfe775763996088b23e6580a4b12f26dbbf78b819391d9d8899c7ed49`,
`566c98a2d1bfb6526555d3b792b2b86f3136f6b305e2b3fd7c31a45153a9bdd2`,
`9e14b98bc1f9b27e23668411e4cf721f43b64644bf8131748d31d4a859ffe433`,
and `0bffa293b5ecf8ff656e37d31a8ba2c5d9cc85d7e91bd9eb9efd81dca1d5c510`.
The one-line prompt fixture hashes are
`a3739872c640c5efaf71ed10c6f41469dc6b6c7869253e951cb2553fff8ae0b5`
for logic and
`9ba52ea797d5342a2f9ee33dd4f07c61e608fecd7060eabe34c836df36110427`
for computer science. The generator's earlier local dry-run directories
`mmlu-natural-v1-generated/` and `mmlu-natural-v1-manifests/` were never
submitted; only `mmlu-natural-v1-queue/` is authoritative.

The offline `rvllm_gemma4_mmlu_teacher_summary` referee was added at
`190a8f91` after submission, without changing any job or its pinned input.
It pins the exact source, tokenizer, dataset, executable, model, job generator,
prompt fixtures and metallibs; refuses nonterminal or dirty queue receipts;
requires the actual combined dispatch counts; and retains every teacher-step
NLL, rank and sampled greedy ID. Its four focused tests passed. It must not
score any arm until its terminal queue receipt exists.

The existing CPU/HF full-logits script is **not yet a same-boundary oracle**
for these Metal teacher steps. That script scores the final position of a
full-prompt forward pass at step zero. The Metal teacher hook reads logits
after a one-token decode that replays the last prompt token, following the
ordinary prefill. The prior M304 boundary audit found different target scores
even within Metal between final-prefill and post-replay readback, and an
M-dependent LM-head route. A future independent reference must first prove
which residual/KV and LM-head boundary it matches; merely running the existing
HF script on the same prompt would not establish numerical parity.
