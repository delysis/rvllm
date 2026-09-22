# Campaign readiness continuation — 2026-09-21

Base: `d4f5c324d60875228733132ee78cfe61fd8c9402`.
PR: https://github.com/delysis/rvllm/pull/2

## Implemented corrections

The previous queue selected the first ready job before examining later
results. A later failed/incomplete attempt could therefore be missed before
launch, including after restart. The new scheduling pass first inspects every
manifest and result, then selects among valid pending jobs. Orphan results,
filename/ID aliases, missing/cyclic dependencies and changed attempted
manifests fail closed. Existing attempts are neither deleted nor replayed.

Successful results are checked for schema and job identity, successful exit,
unchanged pins, no overrun/signal, matching purpose and required validator
success. A missing report inside an existing attempt directory is not pending.
The initial regression failed on the unchanged implementation in CI run
35671711604: zero passed, one failed, zero ignored. Its exact pre-fix queue
blob was checked before adding the test-only module. That failed receipt is
retained; it is not a failed accelerator experiment.

The old idle-server exception independently recognized a llama-server PID and
queried a supplied port, allowing an unrelated idle server to exempt a busy
PID. A rootless `lsof` check now ties that PID to the actual IPv4 loopback or
wildcard listener before `/slots` can grant an exemption. Unsupported or
unavailable ownership checks block readiness. A real loopback socket test
covers the native command; parser tests cover wrong PID, port and address.
This remains sampled cooperative observation, not hostile-process attestation.

An active timing child is disqualified after an activity-observation gap over
2.5 seconds even when the independently sampled power controls look unchanged.
Preparation results cannot claim timing eligibility; raw observed eligibility
is retained separately. No tolerances, private API boundaries, power controls,
backend arithmetic or optimized defaults changed.

## Reusable preparation commands

The normal queue binary now provides an exclusive-lock `audit` and a fresh-
directory `qualify-host` command. The former verifies all pending pins without
running a workload. The latter uses only pinned stock true/touch executables
and the supplied job's conditions. It exercises the actual observer and worker
loop with an unready first job, a ready independent job and a dependent stop.
It never executes the supplied campaign command. The queue specification gives
the complete contracts and limits.

For the recorded campaign, after building/freezing a worker under the shared
hardware ownership discipline, the target-machine commands from `v3` are:

```sh
campaign="$PWD/reports/gemma4-12b-evidence-20260914/experiment-queue-20260916"
queue="$campaign/baseline-isolated-v7"
lock="$campaign/hardware.lock"
# WORKER must name the exact newly built/frozen absolute executable path.
"$WORKER" audit "$queue" "$lock"
"$WORKER" qualify-host "$campaign/host-qualification-NEW-ID" "$lock" \
  "$queue/jobs/10-ac-A-ane.json"
```

The AC job is an example for that stratum, not an instruction to alter power.
Use the intended battery/Fair/nominal conditions as recorded. A blocked or
failed qualification is retained; do not relax the freshness/quiet/disk gates
or replace an attempted directory. Inspect and revise never-started stale
manifests under new IDs before starting any campaign.

## Evidence boundary

The complete tracked source was obtained through a read-only CI archive and
reconstructed locally to its exact Git tree, including the submodule gitlink.
The authoring container is Linux without a Rust compiler or target M4 access.
A pinned rustfmt artifact enables local syntax/format checks; Rust builds and
behavioral tests run in GitHub Actions. Exact-head outcomes and original logs
are recorded in the PR and delivered validation bundle, not inferred here.
All new regressions are included in the normal non-ignored macOS queue suite;
Linux retains the separate eight-test std-only prelaunch path. The archived
campaign test relocates only pending cwd/disk probes and preserves all seven
historical result/manifest byte pairs. It does not replay any of the 29 pending
jobs or relabel historical measurements as current acceptance.

No target worker has been frozen or live-qualified by this Chat session.
Matched native-kit ABBA results, S2 timing, a new provenance-complete S1 capture,
the CPU-oracle discrepancy, captured KV bytes, device continuation and any
conditional S2/assistant integration remain governed by the handoff. None can
be certified from hosted host tests or absent local model/capture assets.
The source-only bandwidth/fusion proposals do not justify a default change.
