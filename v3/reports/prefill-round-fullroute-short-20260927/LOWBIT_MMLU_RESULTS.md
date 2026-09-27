# One-layer W4/W8 teacher-forced diagnostic

All six immutable serial-queue jobs declared in `LOWBIT_MMLU_PROTOCOL.md`
terminated successfully with exit code 0, unchanged pinned files, eligible
sampled conditions and no violations. Each sampled stratum was AC power,
`pmset` mode 2, low-power mode off and nominal thermal state 0. No thermal
stability wait was imposed. The packages retain the original BF16 12B-it
checkpoint except for the authenticated group-32 W4 or W8 sidecar replacing
**layer 0 down projection only**. The native arm uses the same donor SG8 BF16
Metal library as the packages.

The frozen safe-Rust `rvllm_gemma4_lowbit_teacher_summary` referee (source
SHA-256 `029e512b169403343c0303a98c04d0a812c529c2c0d893b8346f835eed76b5e4`)
accepted all six terminal receipts, frozen manifests, source and tokenizer
hashes, exact prompt and forced-target IDs, every teacher step and NLL
aggregate, and actual named low-bit **decode** dispatch. Independent sums of
all raw `trial.stdout` step NLL values matched the six summary totals. The
complete 79-position comparison is `lowbit-mmlu-teacher-summary-v1.json`,
SHA-256 `c8f3993f3b87b9b7d7b344aeabc463eb3ac1a1ba6a47837bef1fa62f505f36a1`.
The complete 30-file outer queue archive (job, report, conditions, stdout,
stderr for each arm) is `lowbit-mmlu-queue-results-v1.tar.gz`, SHA-256
`2ff408f2e3e6f63608f94e413c75f2fde338d4b469c228eaf1fbf0119a7098a2`.
The JSON separately seals every receipt hash. The source JSON SHA-256 is
`a1d115405e86ab51bf275473cc04f452f185db2c18f286b5173d2234d4b993b4`;
tokenizer SHA-256 is
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
The protocol and frozen manifests carry the model, executable, package,
metallib and dataset identities.

| Source row | Prompt / targets | Native NLL | W4 NLL (delta) | W8 NLL (delta) | W4 / W8 target-rank changes | W4 / W8 sampled-greedy changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| formal_logic 2443 | 230 / 24 | 9.357760 | 10.025114 (+0.667354) | 9.563938 (+0.206177) | 1 / 1 | 0 / 0 |
| high_school_computer_science 3185 | 254 / 55 | 92.678512 | 95.448410 (+2.769898) | 101.381070 (+8.702558) | 11 / 7 | 5 / 2 |

Both low-bit arms increased selected-target total NLL versus the new native
SG8 control in both cases, but changes have both signs across individual
positions. In the logic case, zero-based target position 0 contributes
`+0.534205` W4 and `+0.254299` W8 NLL, while position 17 improves by
`-0.053509` W4 and `-0.044581` W8. In the computer-science case, W4's
largest increase is `+4.855556` at position 48 and its largest improvement
is `-2.099957` at position 40; W8's corresponding extremes are `+5.387046`
at position 18 and `-0.911924` at position 51. W4 sampled-greedy IDs differ
from native at positions 1, 18, 39, 42 and 50; W8 differs at 39 and 40.
Every per-position target, rank, NLL, delta and sampled-greedy ID is retained
in the summary JSON, without selecting favorable positions.

The named SG8 W4 and W8 decode kernels each dispatched 24 times for logic
and 55 times for computer science, once per forced target. Their native
projection count is lower by the same amount than the BF16 control. This
proves the sidecar did not silently fall back to the native decode route.
The frozen CLI does **not** separately expose generic low-bit prefill
dispatch; no low-bit prefill kernel attribution follows from this ledger.

`generated_token_ids` contains supplied teacher-forcing targets, **not** the
model's sampled trajectory; sampled IDs are
`teacher_forced.steps[].sampled_token_id`. The two MMLU questions were
preselected by a fixed rule, but the wrapper and answer-repetition text are
Codex-authored and training contamination is unknown. This is neither an
MMLU score nor whole-checkpoint W4/W8 quality, independent numerical
reference, calibrated acceptance, or promotion. Per-step logit readback
changes scheduling, so **none of the timing fields are speed evidence**.
The unfavorable selected-target losses and sampled-token changes are
preserved, not repaired or resampled.
