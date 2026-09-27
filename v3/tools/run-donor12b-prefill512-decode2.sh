#!/bin/sh
# One real-weight 512-token prefill and two-token continuation, queue-owned.
set -eu
[ "$#" -eq 4 ] || { echo 'usage: run-donor12b-prefill512-decode2.sh INFER SELECTOR METALLIB REPORT' >&2; exit 64; }
infer=$1
selector=$2
metallib=$3
report=$4
case "$selector" in off|metal-donor12b-sg8) ;; *) echo 'unexpected selector' >&2; exit 64;; esac
root=/Users/george/.codex/worktrees/rvllm-donor12b-pr4-integration-20260926/v3
prompts=$root/reports/gemma4-rvllm-good-enough-matrix-20260924/prompts.jsonl
model=/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
selected=$(mktemp /tmp/rvllm-donor-prefill512.XXXXXX)
trap 'rm -f -- "$selected"' EXIT
sed -n '2p' "$prompts" > "$selected"
[ "$(wc -l < "$selected")" -eq 1 ] || { echo 'expected exactly one 512-token case' >&2; exit 1; }
RVLLM_METAL_RESEARCH="$selector" RVLLM_METAL_METALLIB_BF16="$metallib" \
  "$infer" --model-dir "$model" --prompts-jsonl "$selected" \
  --session-backend direct --max-new-tokens 2 --max-total-tokens 514 \
  --large-model-opt-in --report "$report" --json
