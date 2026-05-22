# Apple Metal E2B Diagnostic CLI

This workflow is for diagnostic raw-token Apple Metal E2B probes only. It is
not production inference, ANE execution, tokenizer coverage, or an optimization
claim.

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
for the reported fields. It does not imply text decoding, broad prompt
coverage, long-context decode, batching coverage, ANE execution, throughput
readiness, or production acceptance.

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
