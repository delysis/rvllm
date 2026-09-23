# Local owner: integrate the unified cffb22da continuation

## Source transaction

This packet targets **cffb22dabcb57b5f3ecee05acd4ab810d1ce6077**. Do not reapply
either older `faa75336` series. The first wave and the staged second-wave files
are already in the base. Preserve local work, historical reports, STOP/attempted
queue state, shipping gates and the multi-I/O quarantine. No model/cache/native
artifact in this packet is an admitted experiment.

Inspect `git status --short` and `git rev-parse HEAD`. A moved or dirty source head
requires reconciliation, not a reset. Review `verify_packet.py`, then run packet verification in the extracted
packet directory with Python bytecode output disabled. `--checkout` is a readonly exact-preimage check; `--replay`
uses only a NEW isolated directory and never compiles source or creates commits.

```sh
python3 -B verify_packet.py --checkout /Users/george/Downloads/rvllm
python3 -B verify_packet.py --replay /absolute/NEW-replay-directory
```

Review the complete series and apply all mboxes in order using the local owner's
normal `git am` procedure. Partial application is not a buildable state. Do not
copy `preimages/` or `changed-files/` wholesale over the repository. Then format
only the **32 paths** in `v3/tools/gemma4_candidate_rustfmt.paths`, using the
already-established stdin/nonrecursive rustfmt method. No `cargo fmt --all`.
Formatting was not executed by the researcher.

## Host and compiler gates; no accelerator invocation

From `v3`, with the existing reviewed Python environment/NumPy dependency and
locked Cargo dependency cache:

```sh
python3 tools/run_gemma4_python_checks.py --require-rustfmt
python3 tools/gemma4_unified_proposals.py
python3 tools/run_gemma4_candidate_ci.py --output /absolute/NEW-public-host-output
bash tools/check_gemma4_candidate_delivery.sh /absolute/NEW-native-gate-output /absolute/EXISTING-cargo-target
```

Expected scope, not preclaimed results: 108 Python tests; 60 named public Rust
tests; 22 MSL exports; twelve native host-test filters; 32 formatted source paths;
ten pinned shader inputs; 22 Metal 3.1 compile/link arms. The native gate must
say only `compiled-only; no accelerator acceptance`. Preserve failures. It does
not run ignored tests, inference, cache preparation or power controls.

Check GitHub Actions on the exact resulting commit, not the ancestor. The old
9d run failed on a spelling assertion (already fixed in cff); the cff run failed
Rust compilation on the missing research module. A Python pass is not evidence
that the public Rust inventory ran. The new catalog and missing-module repairs
must pass the real Rust and native gates before this source phase is closed.
Commit/push the reviewed code, tests and appended handoff under the local owner's
authority. Do not close this job by committing only another routing document.

## Later model work: permission, pins and quiet ownership first

The following are exact fixture names/recipes for a **later explicitly admitted
local phase**, not instructions to run them as part of integration. Recheck the
16 GiB free-disk floor, single hardware owner, boot, cache and power/process
conditions. Do not stop unrelated services, change power settings, kill an
accelerator child on timeout, clear a STOP marker or invent process exemptions.

### Matrix component qualification

Use the existing native test executable or the normal locked native Cargo test
path. Compile first through the delivery gate. The full names are:

```
prefill_mma_tile_tests::native_short_tile_preserves_existing_oracle_gates
prefill_mma_tile_tests::native_prefetch_preserves_existing_oracle_gates
prefill_mma_tile_tests::native_fp32_operands_preserve_existing_oracle_gates
prefill_mma_tile_tests::native_vector_loads_preserve_existing_oracle_gates
prefill_mma_tile_tests::native_bf16_tile64_checks_both_output_abis_and_operand_paths
```

Set `RVLLM_METAL_QKV_MODEL_DIR` to the already-present pinned model,
`RVLLM_METAL_MMA_TILE_REPORT` to a fresh absolute report path, and
`RVLLM_METAL_MMA_TILE_QUALIFY_ONLY=1`. Run **one exact test** with `--ignored
--exact --nocapture`, never all ignored tests. Each new selected matrix adapter
has six projection cases and 24 single-command component dispatches. The existing
long-tile comparison includes its independent FP32-operand controls and has 36
qualification dispatches. These are component workloads, not ANE decode steps.
Actual receipts must agree with the selected fixture; no incidental timing loop
is an accepted campaign.

Example command shape (substitute exactly one fixture name from above):

```sh
cargo test --offline --locked --release -j 2 --target aarch64-apple-darwin   -p rvllm-apple-metal --lib   prefill_mma_tile_tests::native_short_tile_preserves_existing_oracle_gates   -- --ignored --exact --nocapture
```

The original tensor gates remain. Prefetch/load4 also require FP32 bit identity.
A passing component test is not full prefill/continuation or performance evidence.

### FFN input pins: host-only preparation, separately identified

A new fixture computes the pin manifest from the actual model and captured FFN
inputs; do not implement a second quantizer or invent matrix hashes. Fill the
`experiments/templates/ffn-pin-request.template.json` in the packet with one supported candidate, a layer 0–47 and
1–3 distinct actual captured activation paths. Each must contain 3840 finite
little-endian FP16 values. Their origin must be established by the existing
baseline capture/journal, not merely by this file's SHA-256.

Set these explicit absolute paths:

```
RVLLM_ANE_FFN_ORACLE_MODEL_DIR
RVLLM_ANE_FFN_ORACLE_PIN_REQUEST
RVLLM_ANE_FFN_ORACLE_PIN_OUTPUT
```

Then the separately selected host-only command is:

```sh
cargo test --offline --locked --release -j 2 --target aarch64-apple-darwin   -p rvllm-runtime --features macos-private-ane-research --lib   gemma_ane_decode::component_oracles::host_prepare_ffn_component_pins   -- --ignored --exact --nocapture
```

This fixture loads/quantizes one layer on the host, hashes all coefficients/scales
and samples, validates the result and writes a fresh `manifest.json` plus
`host-pin-receipt.json`. It never creates ANE owners, compiles ANE programs or
evaluates them. It is ignored to avoid implicit large model reads, **not because
it is an accelerator trial**. The manifest digest appears in its host receipt.

### Cached-only ANE FFN comparisons

Only after separate cache provisioning/strict inspection and admission, set:

```
RVLLM_ANE_FFN_ORACLE_MODEL_DIR
RVLLM_ANE_FFN_ORACLE_MANIFEST
RVLLM_ANE_FFN_ORACLE_MANIFEST_SHA256
RVLLM_ANE_FFN_ORACLE_OUTPUT
RVLLM_ANE_DIAGNOSTIC_JOURNAL
```

The output directory must not exist; journal identity/lifecycle is a separate
obligation. Choose the fixture matching the pinned manifest's candidate:

```
gemma_ane_decode::component_oracles::native_chunk4_matches_plain_cached_ffn
gemma_ane_decode::component_oracles::native_down4_matches_plain_cached_ffn
gemma_ane_decode::component_oracles::native_interleaved_matches_stacked_cached_ffn
```

Use the same Cargo command shape as the host fixture, changing only the exact
fixture name. The candidate and proper control must already be cached. Interleaved
uses stacked INT8 as its control. Each fixture permits 1–3 activations, two cached
programs and at most six evaluations, with no timing/retry/compile fallback. It
writes synced events and both output tensors before comparing. Require matching
outputs AND driver lifecycle evidence; a standalone fixture exit code is not a
promotion receipt.

The canonical matrix digest is SHA256 of little-endian u64 columns, little-endian
u64 row count, ordered i8 coefficients and ordered little-endian f16 scales. The
host preparer uses the existing loader/quantizer and verifies config consistency
before and after loading. Keep the checkout/model files quiescent.

### Prefill/full-route and timing

Follow the existing `v3/HANDOFF.md` sequence. Short-MMA does not have a separate
ANE cache: its missing cache is the unchanged baseline decode cache. Provision
and strictly inspect that baseline separately, requiring all 162 visits, zero
inspection compiler calls and driver lifecycle validation before continuation.
Use original independently pinned references, not repeated/padded six-token
prompts. New prefill receipts are dispatch v3 and require complete entry-point
families, not merely one matching counter. Old v1/v2 receipts are not silently
reinterpreted. Keep original tensor/driver and full-reference gates.

The new RMS selector remains default-off and its experiment remains blocked until
a direct native component/tensor adapter exercises it. Old layer tracing can
force the fallback, so tracing a baseline is not its oracle. The existing GQA and
temporal attention candidates also retain their original tensor acceptance work.
Do not promote any of them using first-token or compiler results alone.

The five JSON proposals specify independent controls and ABBA work counts; their
local pins/strata remain unset. They are not active queue manifests. Admit fresh
jobs only through the existing local process; do not edit attempted/completed
manifests. Require 5% control drift rejection and independent confirmation. Do
not pool AC/battery/Fair/nominal or normalize accelerator time with CPU cycles.

## Offline captured-tensor comparison

Use the existing external-policy template already staged with Wave2. Populate it
from the unchanged accepted oracle, not a new tolerance chosen for a candidate.
The tool checks every scalar and its actual bytes:

```sh
python3 tools/gemma4_tensor_audit.py /absolute/PINNED-tensor-manifest.json /absolute/NEW-tensor-audit.json
```

A pass applies to those tensor pairs and that policy only. It does not validate
capture provenance, per-layer dispatch, full continuation, driver behavior or
performance. No file is overwritten, including a failed audit receipt.

## Integration details that must not be lost

This series supersedes the unfinished `/mnt/data/rvllm_unified/work` review
copies, not the already-committed cff base. There is no new `research_wave2`
module: one catalog/enum now owns both waves. Preserve the first ten dispatch
slots and use v3 receipt names, not hardcoded five-/ten-/twelve-slot arrays.

The prior staged ignored fixtures named `native_wave2_fp32_operands_...` and
`native_wave2_vector_loads_...` are now the shared-source
`native_fp32_operands_...` and `native_vector_loads_...` fixtures listed above.
They remain ignored. No numerical threshold was relaxed. Update only a NEW
local job to the canonical exact name; preserve attempted historical manifests.

The packet's `experiments/README.md` maps existing experiments and deduplicates
Down4. Five canonical specifications are installed under
`v3/reports/proposals/gemma4-unified-cffb22da/`. `gemma4_unified_proposals.py`
validates these frozen templates only; it never makes them executable.
The host CI now requires the tensor-auditor and proposal tests, not just their
files' presence. The actual current Python inventory is 108 tests, superseding
the unreproduced 115-test count in the earlier unfinished review report.

A local compiler failure is a failed integration gate, not permission to remove
the test or widen the numerical/shape contract. Return its exact error and
source head. After successful integration, native arithmetic/performance remains
an independently admitted task; the report explicitly lists those unrun gates.
