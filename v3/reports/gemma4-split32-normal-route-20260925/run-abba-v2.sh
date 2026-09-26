#!/bin/zsh
set -euo pipefail

if (( $# != 2 )); then
  print -u2 'usage: run-abba-v2.sh LENGTH OUTPUT_DIR'
  exit 64
fi

length=$1
output_dir=$2
root=/Users/george/.codex/worktrees/rvllm-pr4-20260924/v3
model=/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
source_prompts=$root/reports/gemma4-rvllm-good-enough-matrix-20260924/prompts.jsonl
control=metal-global-d512-atlas_mma_r8k32p64t128
candidate=metal-global-d512-split-coopkey_r8k8p64t128s32
control_lib=$root/reports/gemma4-split32-normal-route-20260925/artifacts/current-matrix.metallib
candidate_lib=$root/reports/gemma4-split32-normal-route-20260925/artifacts/split32.metallib

case $length in
  256) line=1 ;;
  512) line=2 ;;
  1024) line=3 ;;
  2048) line=4 ;;
  *) print -u2 'length must be 256, 512, 1024, or 2048'; exit 64 ;;
esac

mkdir -p "${output_dir:h}"
mkdir "$output_dir"
tmp=$(mktemp -d /tmp/rvllm-split32-abba-v2.XXXXXX)
trap 'rm -rf -- "$tmp"' EXIT

# The first case in every fresh process is a predeclared warmup. It is retained
# in the raw report but excluded from the timing estimate. Cases 1 and 2 are
# identical measured repetitions.
sed -n "${line}p" "$source_prompts" | jq -c '.name += "-warmup"' > "$tmp/prompts.jsonl"
sed -n "${line}p" "$source_prompts" | jq -c '.name += "-measure-1"' >> "$tmp/prompts.jsonl"
sed -n "${line}p" "$source_prompts" | jq -c '.name += "-measure-2"' >> "$tmp/prompts.jsonl"

arms=(control candidate candidate control)
for index in {1..4}; do
  arm=${arms[$index]}
  if [[ $arm == control ]]; then
    selector=$control
    library=$control_lib
    marker=atlas_mma_r8k32p64t128
  else
    selector=$candidate
    library=$candidate_lib
    marker=split_coopkey_r8k8p64t128s32
  fi
  report=$output_dir/${index}-${arm}.json
  RVLLM_METAL_RESEARCH=$selector \
  RVLLM_METAL_METALLIB_BF16=$library \
    $root/target/release/rvllm_metal_infer \
    --model-dir "$model" \
    --prompts-jsonl "$tmp/prompts.jsonl" \
    --session-backend direct \
    --max-new-tokens 2 \
    --max-total-tokens $((length + 2)) \
    --large-model-opt-in \
    --profile-samples 1 \
    --report "$report" \
    --profile-report "${report%.json}.profile.json" \
    --json

  jq -e --arg marker "$marker" '
    (.cases | length) == 3
    and (.cases[0].generated_token_ids == .cases[1].generated_token_ids)
    and (.cases[1].generated_token_ids == .cases[2].generated_token_ids)
    and ([.cases[].library_compiles] | all(. == 0))
    and ([.cases[].pipeline_state_compiles] | all(. == 0))
    and ([.cases[].research_dispatch.counts
          | to_entries
          | map(select(.key | contains($marker)))
          | map(.value)
          | add] | all(. != null and . > 0))
  ' "$report" >/dev/null
done

jq -n \
  --argjson length "$length" \
  --slurpfile a1 "$output_dir/1-control.json" \
  --slurpfile b1 "$output_dir/2-candidate.json" \
  --slurpfile b2 "$output_dir/3-candidate.json" \
  --slurpfile a2 "$output_dir/4-control.json" '
  def warmup($run): [$run[0].cases[0].decode_ms];
  def measured($run): [$run[0].cases[1].decode_ms, $run[0].cases[2].decode_ms];
  def tokens($run): [$run[0].cases[].generated_token_ids];
  def median4: sort | (.[1] + .[2]) / 2;
  {
    schema: "rvllm.gemma4.split32-normal-route-abba.v2",
    length: $length,
    order: ["control", "candidate", "candidate", "control"],
    exclusion_policy: {
      declared_before_run: true,
      excluded_case_index: 0,
      reason: "fresh-process warmup",
      measured_case_indices: [1, 2]
    },
    control_warmup_decode_ms: (warmup($a1) + warmup($a2)),
    candidate_warmup_decode_ms: (warmup($b1) + warmup($b2)),
    control_measured_decode_ms: (measured($a1) + measured($a2)),
    candidate_measured_decode_ms: (measured($b1) + measured($b2)),
    tokens_identical:
      ((tokens($a1) + tokens($b1) + tokens($b2) + tokens($a2)) | unique | length == 1)
  }
  | .control_median_ms = (.control_measured_decode_ms | median4)
  | .candidate_median_ms = (.candidate_measured_decode_ms | median4)
  | .speedup = (.control_median_ms / .candidate_median_ms)
' > "$output_dir/summary.json"

jq -e '
  .tokens_identical == true
  and (.control_warmup_decode_ms | length) == 2
  and (.candidate_warmup_decode_ms | length) == 2
  and (.control_measured_decode_ms | length) == 4
  and (.candidate_measured_decode_ms | length) == 4
  and .speedup > 0
' "$output_dir/summary.json" >/dev/null
