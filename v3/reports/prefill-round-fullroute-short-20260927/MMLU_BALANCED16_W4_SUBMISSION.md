# Balanced-16 W4 first-wave submission

At 2026-09-28 04:14 UTC, the existing serial experiment queue accepted
exactly eight fresh jobs, q00–q03 native BF16 then full-text W4 in each pair.
Their authoritative manifests are in `mmlu-balanced16-w4-queue/`, with exact
SHA-256 values frozen in `rvllm_gemma4_mmlu_balanced16_w4_summary.rs`.
The dependency chain is q00 native → W4 → q01 native → W4 → q02 native → W4
→ q03 native → W4. The q00 native job was starting in the first read-only
status snapshot. No q04–q15 job has been submitted.

Before submission, the queue had no matching job/result/quarantine ID and
was waiting only on obsolete blocked dependencies; data-volume free space
was 100 GiB, above each manifest's 64 GiB guard. All eight submissions
returned accepted status. The manifests specify zero thermal-stability
dwell, a 1800-second run limit and 7200-second wait limit per job. They pin
the sealed 16-case source, exact-prefix token fixture, original checkpoint
and tokenizer, cached Arrow source, authenticated 328-sidecar W4 package,
BF16 SG8 donor and frozen executable. The generator's focused tests and
Apple build passed; the separate referee's four focused tests and build
passed, including all 32 authoritative manifest hashes.

The prior W4 and BF16 teacher *process* measurements are not end-to-end
queue durations: complete earlier pairs took roughly 25–27 minutes after
package/checkpoint pin verification. This was recognized after submission;
the eight immutable manifests were not changed. The queue starts a job's
condition-wait clock only when its dependencies are ready. Preserve and
review the first wave before deciding whether to submit a smaller fraction
of each later four-pair wave at a time.

No trial receipt or numerical result existed at submission. Do not resubmit
these IDs, overwrite their outputs, score a partial set, or submit the next
wave until all eight are terminal and their conditions and failures are
preserved. Even a complete 16-case diagnostic is not an MMLU accuracy score,
contamination-free quality verdict, prefill attribution, timing or promotion.
