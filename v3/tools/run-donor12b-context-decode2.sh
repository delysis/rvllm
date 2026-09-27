#!/bin/sh
# Bounded long-context two-token screen; one queue job per route arm.
set -eu
[ "$#" -eq 5 ] || { echo 'usage: run-donor12b-context-decode2.sh INFER SELECTOR METALLIB REPORT PROMPT_LINE' >&2; exit 64; }
infer=$1
selector=$2
metallib=$3
report=$4
prompt_line=$5
case "$selector" in off|metal-donor12b-sg8) ;; *) echo 'unexpected selector' >&2; exit 64;; esac
case "$prompt_line" in 3) max_total=1026 ;; 4) max_total=2050 ;; *) echo 'prompt line must be 3 or 4' >&2; exit 64;; esac
root=/Users/george/.codex/worktrees/rvllm-donor12b-pr4-integration-20260926/v3
prompts=$root/reports/gemma4-rvllm-good-enough-matrix-20260924/prompts.jsonl
model=/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
selected=$(mktemp /tmp/rvllm-donor-context-decode2.XXXXXX)
trap 'rm -f -- "$selected"' EXIT
sed -n "${prompt_line}p" "$prompts" > "$selected"
[ "$(wc -l < "$selected")" -eq 1 ] || { echo 'expected exactly one prompt case' >&2; exit 1; }
RVLLM_METAL_RESEARCH="$selector" RVLLM_METAL_METALLIB_BF16="$metallib" \
  "$infer" --model-dir "$model" --prompts-jsonl "$selected" \
  --session-backend direct --max-new-tokens 2 --max-total-tokens "$max_total" \
  --large-model-opt-in --report "$report" --json
