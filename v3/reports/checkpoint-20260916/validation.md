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

## CI follow-up

The initial checkpoint is `74735053d96c9e0c6a0f16c8307db9e4ebab47f4`.
[Its first CI run](https://github.com/delysis/rvllm/actions/runs/35133416842)
passed the GB10 job but exposed two Linux portability defects: the diagnostic
prefill CLI unconditionally imported the Apple-only `ModelMetalBackend`, and
portable Core ML artifact-cache tests used `tempfile` while it was declared
only as a macOS dependency. The follow-up gates the CLI's device implementation
to macOS, returns an explicit unsupported-platform error elsewhere, and adds
the existing `tempfile` version as a platform-independent dev dependency.
These fixes do not change the macOS inference implementation or resume trials.
New CI must establish the Linux check/test result; the original failed run is
preserved rather than rerun under its old identity.

The follow-up passes `cargo check --offline --locked -j 2 -p rvllm-runtime
--features apple --bin rvllm_prefill_handoff` on macOS. The exact CLI source
also compiles to metadata with the installed `wasm32-unknown-unknown` target,
which exercises its non-macOS branch without external backend dependencies;
that is a platform-gate check, not a Linux workspace test. Raw logs are
`prefill-platform-check.*` and `prefill-non-macos-check.*`. The initial CI defect
identities are saved in `ci-initial-defects.json`. No accelerator test ran.

CI on portability commit `12ef5efe82966fe292b1f2b0ba5973f6fe801429` passed
the workspace compile check, GB10 job, and the complete Apple shipping job:
iPhone/simulator checks, Swift build, XCFramework packaging/scan, platform
metallibs and shipping private-symbol scan. See `ci-portability-results.json`.
The Linux test job reached the FFI suite and exposed two test/packaging details:
the Swift C header copy lacked an explanatory comment present in the canonical
header, and the missing-worker test expected a macOS-only diagnostic on Linux.
The follow-up synchronizes the header bytes and asserts the appropriate
platform error while retaining the backend-unavailable status and null-handle
checks. There is no ABI layout or production runtime behavior change.

The local FFI rerun additionally reproduced a fixture-only macOS failure:
`temp_dir()` began with the `/var` symlink, which the persistent-cache capability
correctly rejects. The fixture now resolves its temporary parent before
creating its private directory; production path checks are unchanged. The
original failed log remains `ffi-host-tests.*`. The final host-only command
`cargo test --offline --locked -j 2 -p rvllm-apple-ffi --lib` passes **17/17**
tests (`ffi-host-tests-final.*`), and `cmp` confirms the checked-in headers are
identical. Linux CI must validate its platform-specific error assertion.
