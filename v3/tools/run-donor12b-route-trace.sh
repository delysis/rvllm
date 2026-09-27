#!/bin/sh
# Queue-owned two-token real-weight SG8 on/off diagnostic at fixed 512-token prompt.
set -eu
[ "$#" -eq 4 ] || { echo 'usage: run-donor12b-route-trace.sh INFER METALLIB REPORT PROMPTS' >&2; exit 64; }
infer=$1
metallib=$2
report=$3
prompts=$4
model=/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
selected=$(mktemp /tmp/rvllm-donor12b-route-trace.XXXXXX)
trap 'rm -f -- "$selected"' EXIT
sed -n '2p' "$prompts" > "$selected"
[ "$(wc -l < "$selected")" -eq 1 ] || { echo 'expected one 512-token prompt case' >&2; exit 1; }
[ "$(jq -r .name "$selected")" = 'pp512-decode2' ] || { echo 'unexpected prompt case' >&2; exit 1; }
RVLLM_METAL_RESEARCH=metal-donor12b-sg8 RVLLM_METAL_METALLIB_BF16="$metallib" \
  "$infer" --model-dir "$model" --prompts-jsonl "$selected" \
  --session-backend direct --max-new-tokens 2 --max-total-tokens 514 \
  --large-model-opt-in --report "$report" --json
