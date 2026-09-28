# Balanced16 W4 first-wave terminal receipts

This is a complete first-wave receipt audit, **not** an admitted numerical result. The eight original serial jobs `prefill26-mmlu-balanced16-w4-q{00..03}-{native,w4}-v1-20260928` are terminal. No job was replayed or overwritten, and no later wave was submitted. Source selection, prompt/token fixture, package, and frozen referee are specified in `MMLU_BALANCED16_FULLTEXT_W4_PROTOCOL.md`; the live disk-guard amendment and the old q01-W4 disqualification are recorded separately in `MMLU_BALANCED16_W4_LIVE_GUARD_AMENDMENT.md` and `MMLU_BALANCED16_W4_WAVE1_DISQUALIFICATION.md`.

| Case/arm | Queue status | Exit | Inputs unchanged | Guard | Journal samples | Unready / activity-unsampled | Minimum free bytes |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| q00 native | succeeded | 0 | yes | 64 GiB | 27 | 0 / 0 | 107,943,297,024 |
| q00 W4 | succeeded | 0 | yes | 64 GiB | 313 | 0 / 0 | 106,675,728,384 |
| q01 native | succeeded | 0 | yes | 64 GiB | 26 | 0 / 0 | 95,252,774,912 |
| q01 W4 | succeeded | 0 | yes | 64 GiB | 367 | 332 / 332 | 49,231,007,744 |
| q02 native | succeeded | 0 | yes | 32 GiB | 28 | 0 / 0 | 53,428,568,064 |
| q02 W4 | succeeded | 0 | yes | 32 GiB | 333 | 0 / 0 | 53,286,162,432 |
| q03 native | succeeded | 0 | yes | 32 GiB | 26 | 0 / 0 | 53,169,246,208 |
| q03 W4 | succeeded | 0 | yes | 32 GiB | 337 | 0 / 0 | 52,949,893,120 |

The q01-W4 report has `sampled_conditions_eligible=false` and 32 recorded disk violations. The old 64 GiB guard caused 332 journal samples to skip activity/process sampling. Lowering the guard afterwards cannot reconstruct those missing observations. The other seven reports have `sampled_conditions_eligible=true` and empty violations. All eight `trial.stdout` files contain the expected number of teacher-forced positions for their respective selected cases (19, 6, 13, and 11 for q00–q03 in both arms), and their research dispatch ledgers report no overflow. This audit does not adjudicate logits, NLL, rank, greedy IDs, or named kernel dispatch.

The 40 original outer receipt files (`job.json`, `report.json`, `conditions.jsonl`, `trial.stdout`, `trial.stderr` for each arm) are copied byte-for-byte into `mmlu-balanced16-w4-wave1-eight-receipts.tar.gz`, SHA-256 `0a3f5bbd95423d0040f92e1b074d3c99cbb056a0c71e3edee158b7f462ba1581`. The queue originals remain in place. The archive was listed and contains exactly 40 named receipt files. The q01-W4 five-file archive previously pushed separately remains unchanged.

The predeclared 32-arm referee requires every arm to have complete, eligible sampled conditions, so this v1 cohort cannot be admitted even if later arms succeeded. Stop before submitting q04–q15; do not weaken the referee, relabel q01-W4, replay a completed case, select replacements using observed outputs, or infer checkpoint quality or speed from this partial wave. Any further numerical study requires a prospectively sealed distinct cohort and fresh immutable queue IDs.
