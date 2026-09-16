# Pause checkpoint validation

Base commit: `144cdd01dd51bb7d0c2c9d942e50cefc25d3ba5f`.
Destination: the user's fork, `https://github.com/delysis/rvllm`, branch `main`.
The final Git commit identifies the complete source checkpoint; the historical
receipts retain their original executable and source hashes.

Host: macOS/aarch64. Toolchain: rustc 1.92.0
(`ded5c06cf`, 2025-12-08), Cargo 1.92.0 (`344c4567c`, 2025-10-21).
Release checks use `--offline --locked -j 2`, explicit rustc/rustdoc/Cargo
paths, `CARGO_HOME=/Users/george/.cargo` and the existing
`/Users/george/Downloads/rvllm/v3/target`. This does not certify the declared
Rust 1.80 minimum; CI uses stable Rust.

## Checks

| Check | Result / receipt |
| --- | --- |
| Latest shared-probe queue host suite | **13 passed**, zero failures; `../gemma4-12b-evidence-20260914/kv-import-scratch-20260916/queue-v7-tests.stdout` |
| KV scratch filtered host suite | **2 passed**, zero failures; two captured/device fixtures ignored; `kv-import-host-tests.stdout` |
| Apple release-symbol checker unit tests | **8 passed**, zero failures; direct `python3 -m unittest tools/test_check_apple_release_symbols.py` |
| Locked workspace metadata | Passed; `workspace-metadata.json` |
| Broader Apple/runtime private-feature host suite | **303 passed**, zero failures: Apple 114 passed / 26 ignored; runtime 189 passed / 105 ignored; `apple-runtime-host-tests-host-boundaries.stdout` |
| Publication content check | No high-confidence credential-pattern findings or binary candidates in selected contents; `publication-scan.json`. This is a bounded pattern check, not a security audit. |
| Archived evidence integrity | All **2,677** selected artifacts match their inventory SHA-256; `archive-integrity.json` |
| Staged whitespace check | Passed for active code/docs; immutable historical stdout/stderr retain exact original bytes under archive attributes |

The broader suite initially exposed an old unignored
`ane::tests::test_hardware_ane_compilation_integration` test. The private API
opt-in check returned `PrivateApiUnavailable` before any ANE call. The first
attempt had 114 passed, one failed and 25 ignored; its raw logs are retained
as `apple-runtime-host-tests.{stdout,stderr}`. The fix marks that hardware
fixture explicitly ignored, matching the other accelerator tests. The check
does not enable the opt-in, compile a graph, or retry the hardware experiment.
The second attempt is retained separately as
`apple-runtime-host-tests-final.{stdout,stderr}`: Apple passed **114 tests**
with 26 ignored. Runtime passed 188, ignored 104, and exposed two older
toy-Metal tests that expected shader assets during an ordinary host run.
Both failed with `MetallibMissing` before evaluation. The actual Metal
execution fixture is now explicitly ignored; the route-policy test invokes
backend selection directly, retaining both opt-in assertions without preparing
a device. This changes only test boundaries, not production backend behavior.
The final check is `apple-runtime-host-tests-host-boundaries.{stdout,stderr}`.

Review also found two non-macOS engine tests still expecting the removed
synthetic backend fallback. They now assert immediate, typed unsupported-backend
errors. These branches are excluded on this macOS host and require Linux CI
validation; the macOS test results do not certify them.

The full command for all broad attempts is:

```sh
cargo test --offline --locked --release -j 2 \
  -p rvllm-apple -p rvllm-runtime \
  --features macos-private-ane-research --lib
```

Warnings remain in the older code. No full workspace, iOS packaging, private
symbol scan of final shipping binaries, or live model acceptance was rerun
for this pause checkpoint. Existing raw hardware receipts are historical
evidence, not new acceptance. The latest queue normal executable still needs
building after resumption; host tests compiled and exercised its current source.

## Publication and stopped state

The fork's main branch matched the base commit at fetch. Source, specifications,
reports, first-party experiment source snapshots, immutable job manifests and
selected raw receipts are included. No weights, tensor payloads, frozen
executables or build caches are published. `archive-selection.json` and
`local-artifacts.jsonl` preserve the selection and original local hashes.

The queue STOP marker remains present. Its last worker was joined with exit
zero and there is no active trial. No matching recurring automation was found.
The research subagent is inactive; there is no experiment babysitter to stop.
No accelerator work was launched while preparing this checkpoint.

Read [the handoff](../../HANDOFF.md) before resuming. GitHub CI status should
be checked against the pushed commit; local host results do not imply CI passed.
