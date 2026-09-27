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
