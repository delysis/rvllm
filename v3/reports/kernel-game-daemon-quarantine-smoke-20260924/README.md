# Experiment-daemon terminal-failure continuation smoke

Date: 2026-09-24

Commit `7f67a728` changes only daemon mode: after a terminal failed or incomplete
attempt, the exact manifest is moved to `quarantined-jobs/`, a hash-bound
receipt is written, and independent later jobs continue.  One-shot `run` mode
retains its fail-stop behavior.  No result is deleted or replayed, and failed
dependencies are quarantined rather than executed.

The live LaunchAgent was rebuilt and restarted, then received two independent
jobs in lexical order:

1. `/usr/bin/false`, pinned by SHA-256, exited 1 and produced a failed report.
2. `/usr/bin/true`, pinned by SHA-256, exited 0 and produced a succeeded report.

The daemon automatically quarantined the first manifest, immediately executed
the second, and returned to `idle` under the same PID.  No manual manifest move
or daemon restart occurred between the two trials.

Evidence SHA-256:

- failed report: `2bf86b144784278dddcf208b597826c29e267e58656f5ad92247e6985a6ea375`
- succeeding report: `4f49fd7abae0167004015f83c6249c0fc5d10efe50d11b518e9b042c09fa6b2c`
- quarantine receipt: `f42afccb915eeaef1ebbe586074e04267150a6fd26e7c151fb1cae9f3670aad0`

The focused queue test suite also passed 25/25, including manifest/report
identity preservation and duplicate-quarantine refusal.
