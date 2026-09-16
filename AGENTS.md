# Development and paused research

Use safe, idiomatic Rust for development. Keep platform FFI isolated in the
existing platform boundary crates; do not introduce unsafe code into the
experiment harness or decoder orchestration.

Read [v3/HANDOFF.md](v3/HANDOFF.md) before continuing Apple inference work.
The user paused this campaign on 2026-09-16. Do not restart its workers,
clear STOP markers, run ignored accelerator tests, or launch pending trials
unless the user explicitly resumes it. Ordinary host checks are allowed.

The optimization target is Gemma 4 12B or larger, Metal prefill and ANE decode,
with INT8 as the active ANE path. Preserve the multi-I/O panic quarantine and
the separate power/thermal measurement strata described in the handoff.
