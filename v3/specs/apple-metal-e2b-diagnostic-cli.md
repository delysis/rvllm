# Apple Metal E2B Diagnostic CLI

This workflow is for diagnostic Apple Metal E2B probes only. It defaults to
raw token IDs, and has an optional tokenizer-backed text prompt/text decode
mode for diagnostics. It is not production inference, ANE execution, broad
tokenizer coverage, or an optimization claim.

## Setup

Set the local checkpoint path:

```bash
export RVLLM_GEMMA4_MODEL_DIR=/Users/george/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/9d53598892698e981fc42f78b0f8c005cecd63ca
test -d "$RVLLM_GEMMA4_MODEL_DIR"
```

Use the HF reference environment when generating artifacts:

```bash
export RVLLM_HF_REF_PYTHON=/tmp/rvllm-gemma4-hf-ref-venv/bin/python
```

Real E2B Metal probes require an explicit opt-in:

```bash
export RVLLM_METAL_ALLOW_LARGE_GEMMA4_PROBE=1
```

Passing `--large-model-opt-in` to the CLI is equivalent for that process.

## Generate One Reference Artifact

Reference artifacts are generated outside the repo:

```bash
"$RVLLM_HF_REF_PYTHON" scripts/dump_gemma4_hf_reference_logits.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 1 \
  --selected-token-ids 0,1,2,3,4,5 \
  --top-k 16 \
  --output /tmp/gemma4-e2b-hf-reference-logits.json
```

Do not commit `/tmp` artifacts.

## Run Raw-Token CLI

```bash
cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 1 \
  --top-k 16 \
  --large-model-opt-in
```

The output includes sampled token IDs, per-step top-k logits, timing, arena
bytes, command-buffer and encoder counters, forced waits, debug sync state, and
the diagnostic-only claim.

## Run Text Diagnostic CLI

Text prompts are a diagnostic convenience around the same probe path. By
default the CLI tokenizes text with `tokenizer.json` from the model directory
and prepends BOS token ID `2`. Use `--no-bos` only when the prompt text already
accounts for the desired BOS handling.

```bash
cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-text "Hello" \
  --decode-steps 1 \
  --top-k 16 \
  --large-model-opt-in \
  --decode-text
```

The report still includes `prompt_token_ids`. With `--decode-text`, text mode
also reports `tokenizer_json`, `prompt_text`, `sampled_text`, and `output_text`.
This is not a tokenizer/text production workflow or a quality claim.

## Run Bounded Text Inference CLI

For a production-facing text-in/text-out command shape, use
`rvllm_metal_infer`. This path does not read debug logits by default; it runs
prefill plus greedy decode, stops on EOS or `--max-new-tokens`, and reports the
current Metal counters. It is still bounded by the current E2B Metal probe arena:
the default prompt-plus-generated cap is `16`, and `--max-total-tokens N` or
`RVLLM_METAL_MAX_PROBE_TOKENS=N` can explicitly raise the cap up to `64` for
bounded diagnostics.

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --max-total-tokens 16 \
  --large-model-opt-in \
  --json
```

The command reports schema `rvllm.apple_metal_text_infer.v1` and keeps an
explicit acceptance boundary in `claim`. If `--hf-reference <JSON>` is supplied,
it compares the tokenizer-derived prompt IDs, requested decode step count, and
generated token IDs with the existing HF artifact and reports
`hf_reference.matched`. Passing this command is workflow evidence, not complete
production readiness, broad correctness, long-context support beyond the
explicit cap, or a performance claim.

Generate a text reference artifact outside the repo before running the
reference-backed text command:

```bash
"$RVLLM_HF_REF_PYTHON" scripts/dump_gemma4_hf_reference_logits.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-text "Hello" \
  --decode-steps 1 \
  --top-k 16 \
  --output /tmp/gemma4-e2b-hf-text-infer-hello-step1.json
```

Then run the bounded text inference CLI against it:

```bash
cargo run -p rvllm-runtime --features apple --bin rvllm_metal_infer -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --large-model-opt-in \
  --hf-reference /tmp/gemma4-e2b-hf-text-infer-hello-step1.json \
  --json
```

Do not promote tokenizer/text decoding evidence unless `hf_reference.matched`
is true. The current `"Hello"`, `"Once upon a time"`, and `"The capital of
France is"` reference-backed smokes are bounded positive checks after correcting
Metal layer-scalar ordering, but they are still narrow prompt coverage inside
the probe arena, not production serving or broad tokenizer/text coverage.

## Generate Text Reference Suite Manifest

```bash
python3 scripts/dump_gemma4_e2b_hf_text_reference_suite.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt "Hello" \
  --prompt "Once upon a time" \
  --prompt "The capital of France is" \
  --decode-steps 1 \
  --top-k 16 \
  --output-dir /tmp/rvllm-e2b-text-reference-suite
```

Use `--dry-run` first to write the manifest and print the HF commands without
loading Transformers. The generated artifacts stay outside the repo.

## Run Text Inference Suite Runner

```bash
python3 tools/run_apple_metal_text_infer_suite.py \
  --manifest /tmp/rvllm-e2b-text-reference-suite/gemma4-e2b-hf-text-reference-suite-manifest.json \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR"
```

Use `--dry-run` first to print the `rvllm_metal_infer` commands without running
Metal. The runner writes `/tmp/rvllm-e2b-text-infer-suite-report.json` by
default and fails if any case does not report `hf_reference.matched: true`.

## Run Reference-Backed CLI

```bash
cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 1 \
  --top-k 16 \
  --large-model-opt-in \
  --hf-reference /tmp/gemma4-e2b-hf-reference-logits.json
```

For automation, request one JSON object:

```bash
cargo run -p rvllm-runtime --features apple --bin probe_apple_metal_decode -- \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --prompt-token-ids 2,4 \
  --decode-steps 1 \
  --top-k 16 \
  --large-model-opt-in \
  --hf-reference /tmp/gemma4-e2b-hf-reference-logits.json \
  --json
```

`hf_reference.matched` reports whether selected logits, top-k entries, and the
sampled token match the supplied artifact under the CLI comparison tolerance.

## Generate Reference Suite Manifest

```bash
python3 scripts/dump_gemma4_e2b_hf_reference_suite.py \
  "$RVLLM_GEMMA4_MODEL_DIR" \
  --case 2,4 \
  --case 2,17 \
  --case 2,17,42,4 \
  --decode-steps 1 \
  --top-k 16 \
  --full-logits \
  --output-dir /tmp/rvllm-e2b-reference-suite
```

The manifest records the commands and artifact paths needed to regenerate the
suite.

## Run CLI Suite Runner

```bash
python3 tools/run_apple_metal_e2b_cli_suite.py \
  --manifest /tmp/rvllm-e2b-reference-suite/gemma4-e2b-hf-reference-suite-manifest.json \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --cargo-bin probe_apple_metal_decode
```

Use `--dry-run` first to print the cargo commands without running Metal:

```bash
python3 tools/run_apple_metal_e2b_cli_suite.py \
  --manifest /tmp/rvllm-e2b-reference-suite/gemma4-e2b-hf-reference-suite-manifest.json \
  --model-dir "$RVLLM_GEMMA4_MODEL_DIR" \
  --cargo-bin probe_apple_metal_decode \
  --dry-run
```

The runner writes `/tmp/rvllm-e2b-cli-suite-report.json` by default.

## Interpret Output

The CLI and suite runner are diagnostic evidence. A passing comparison means
the bounded raw-token prompt and decode steps matched the supplied HF artifact
for the reported fields. Text diagnostic output means the CLI loaded
`tokenizer.json`, encoded the supplied text, and decoded the sampled/output
token IDs for inspection. The bounded text inference CLI proves a text-in/text-
out command shape without debug-logit reads, but still has a configurable probe
arena cap (`16` by default, `64` maximum in this diagnostic build). Neither path
implies complete production serving, broad prompt coverage, long-context decode,
batching coverage, ANE execution, throughput readiness, or production
acceptance.

## Slow-Test Budget

Every real E2B Metal run can take minutes. Before each run, record:

```text
Hypothesis:
Exact command:
Artifact used:
Expected pass/fail signal:
What I will do if it fails:
```

Run one slow E2B command at a time. Do not run selected/full-vocab parity or
reference-backed suites as a general health check; run them for a concrete CLI,
reporting, or acceptance boundary.
