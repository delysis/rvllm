# E2B multi-case teacher qualification: one-time queue submission

At 2026-10-01 06:22 UTC, the existing serial rvLLM experiment queue accepted these three new immutable jobs once, in dependency order:

1. `e2b-multicase-teacher-qual-v1-20261001-observatory-single`
2. `e2b-multicase-teacher-qual-v1-20261001-kitchen-single`
3. `e2b-multicase-teacher-qual-v1-20261001-two-case-batch`

The queue is `/Users/george/.codex/worktrees/rvllm-pr4-20260924/v3/reports/gemma4-global-decode-local-20260924/queue`. Its pre-submission state was `waiting` with only two obsolete blocked dependencies; the worker PID was live. No matching job, result or quarantine ID existed. The exact frozen executable SHA-256 was `149d447d70227edf265c52c721534531bb69a9c31cda44b5512566054db30d06`, all three manifest hashes matched `E2B_MULTICASE_TEACHER_QUALIFICATION_PREPARED.md`, and free disk was 172 GiB against a 32 GiB floor. No thermal-stability wait was imposed.

Do not resubmit, replay or overwrite these jobs. Do not score partial results. After all three terminate, use the frozen safe-Rust referee with a fresh output path and the result directories in the listed order. Preserve rejection, failed receipts and every queue condition observation; no retry, favorable selection or gate weakening. Teacher timing is invalid. This synthetic E2B route qualification cannot establish 12B W4 quality or performance.
