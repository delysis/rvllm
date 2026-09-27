# Two-question, same-token-boundary numerical reference

The source audit in `SAME_BOUNDARY_REFERENCE_PLAN.md` fixes the comparison:
Hugging Face full-prompt step zero versus the **last prefill row** from Metal,
not Metal's subsequent last-token replay/decode step. These are the two
previously frozen `cais/mmlu` question prompts in `MMLU_NATURAL_PROTOCOL.md`
(230 and 254 prompt IDs), with their first continuation target IDs. The
question text is independently sourced, but its wrapper and answer-repeat
continuation were written by Codex. This is not an MMLU score, held-out
quality acceptance, or a general checkpoint reference.

The safe-Rust `rvllm_gemma4_prefill_reference_job_gen` seals the exact MMLU
source/tokenizer and prior immutable job templates, then emits six fresh
serial jobs into `mmlu-prefill-reference-v1-queue/`: HF, Metal control, Metal
combined for formal logic, then the same order for computer science. It
refuses an existing output directory. Each command pins the original BF16
12B-it config, 22 GiB safetensor and tokenizer. HF uses the unchanged
`dump_gemma4_hf_reference_logits.py` script and installed compatible
Transformers 5.14.1 environment, with one full-prompt step and all 262144
logits. Metal uses the separately frozen default-off prefill-final probe
executable SHA-256
`23a662ad72dcc4357458d8d8720e0a228b357c3940fbc8ad15b1bf1d2de45463`,
with actual control/combined dispatch required. Its probe may alter later
work, so **no timing** from these jobs is usable.

Before any numerical claim, require clean terminal queue receipts, unchanged
input pins, and exact prompt/first-target IDs. Compare target logit/rank/NLL
and greedy ID for HF versus each Metal arm at the prefill-final boundary. Keep
both favorable and unfavorable positions and the raw receipts, including
conditions and any failure. CPU Transformers differs from the checkpoint's
recorded development version; HF/Metal kernels and output rounding also
differ. Matching a first token does not imply distribution parity. This
two-position trial does not validate Metal teacher decode, long-continuation
quality, W4/W8, speed or promotion. Those gates remain separate.

**Protocol correction before results:** The frozen Metal probe serializes only
`target_logit`, `target_rank`, `negative_log_likelihood` and `sampled_token_id`,
not its full 262144-logit vector. The HF job retains its full vector, but
the originally requested HF-versus-Metal all-vocabulary differences and
top-token margins are **not measurable** in these six jobs. Do not infer or
report them. A separately frozen diagnostic executable and fresh immutable
job IDs are required for that stronger test; this trial is limited to the
four per-target/greedy fields above. This correction does not turn the
limited comparison into quality acceptance.
