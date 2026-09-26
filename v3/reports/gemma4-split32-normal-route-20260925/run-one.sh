#!/bin/zsh
set -euo pipefail

if (( $# != 4 )); then
  print -u2 'usage: run-one.sh LENGTH CANDIDATE METALLIB OUTPUT'
  exit 64
fi

length=$1
candidate=$2
metallib=$3
output=$4
if [[ $candidate == *split-coopkey* ]]; then
  dispatch_marker=split_coopkey_r8k8p64t128s32
else
  dispatch_marker=atlas_mma_r8k32p64t128
fi
root=/Users/george/.codex/worktrees/rvllm-pr4-20260924/v3
model=/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
source_prompts=$root/reports/gemma4-rvllm-good-enough-matrix-20260924/prompts.jsonl
case $length in
  256) line=1 ;;
  512) line=2 ;;
  1024) line=3 ;;
  2048) line=4 ;;
  *) print -u2 'length must be 256, 512, 1024, or 2048'; exit 64 ;;
esac

tmp=$(mktemp -d /tmp/rvllm-split32-normal.XXXXXX)
trap 'rm -rf -- "$tmp"' EXIT
sed -n "${line}p" "$source_prompts" > "$tmp/prompts.jsonl"
sed -n "${line}p" "$source_prompts" | jq -c '.name += "-repeat"' >> "$tmp/prompts.jsonl"

RVLLM_METAL_RESEARCH=$candidate \
RVLLM_METAL_METALLIB_BF16=$metallib \
  $root/target/release/rvllm_metal_infer \
  --model-dir "$model" \
  --prompts-jsonl "$tmp/prompts.jsonl" \
  --session-backend direct \
  --max-new-tokens 2 \
  --max-total-tokens $((length + 2)) \
  --large-model-opt-in \
  --profile-samples 3 \
  --report "$output" \
  --profile-report "${output%.json}.profile.json" \
  --json

jq -e --arg dispatch_marker "$dispatch_marker" '
  (.cases | length) == 2
  and (.cases[0].generated_token_ids == .cases[1].generated_token_ids)
  and ([.cases[].library_compiles] | all(. == 0))
  and ([.cases[].pipeline_state_compiles] | all(. == 0))
  and ([.cases[].research_dispatch.counts
        | to_entries
        | map(select(.key | contains($dispatch_marker)))
        | map(.value)
        | add] | all(. != null and . > 0))
' "$output" >/dev/null
