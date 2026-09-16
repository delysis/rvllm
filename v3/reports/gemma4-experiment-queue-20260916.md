# Conditional experiments and native-kit baseline

The Rust queue is implemented in `rvllm-runtime` as `rvllm_experiment_queue`.
It runs independently, accepts new disk-backed jobs while active, and selects
jobs whose dependencies and sampled launch conditions have remained stable.
An unready job does not block an independent ready job. Executables and input
files are pinned, execution uses argument arrays without a shell, and one
cooperative file lock serializes accelerator work. Raw child logs, observer
journals, activity checks, receipts and failed attempts are preserved.

See [the queue specification](../specs/apple-experiment-queue.md) for commands,
manifest fields, shutdown behavior and measurement boundaries. An overrun is
reported while the child finishes safely; there is no forced kill or implicit
repair/retry. Preparation success is separate from performance eligibility.

## Verification

Ten host tests pass, including power freshness/restrictions, stable-window
reset, ownership locks, changed artifact rejection, dependency failures and
strict idle-slot parsing. The live non-accelerator smoke exercised:

- A successful child, then deliberate exit 1, with the next job never launched.
- New submissions accepted while the worker was already running.
- A second worker refused without replacing the active worker's state.
- Restart refusing the failed attempt instead of replaying it.
- An independent Fair AC job running while earlier battery and nominal jobs
  remained pending; the existing localhost llama server was checked idle.
- A STOP request ending a waiting worker without launching its pending jobs.

Evidence is under `gemma4-12b-evidence-20260914/experiment-queue-20260916/`.
These smoke jobs used only system `sleep`/`true`/`false`, not ANE or inference.
The original queue snapshots and subsequent revisions remain separate.

## Why cache maintenance precedes timing

The latest manual strict load failed before decoding even after 67 baseline
and 38 stacked cache misses were repaired. Read-only macOS logs record active
CacheDelete purge handling by `aned`; graph IDs are private, so this does not
attribute a particular deletion. Eleven GiB free is not a proved safe floor.
Conservative cleanup then removed 21,365,410,375 bytes of obsolete Cargo
archives under held profile locks, leaving approximately 31 GiB free at that
observation. Sources, model assets and frozen executables were preserved.

This headroom change justifies one separately recorded, bounded repair. The
queue will stop if subsequent strict inference fails; successful compilation
alone is not proof of retained availability. Earlier cache misses, failed
timing attempts and purge logs remain in the stacked experiment directories.

## Baseline campaign

The [native-kit baseline runner](gemma4-llama-baseline-trial-spec-20260916.md)
is an isolated safe Rust program using native-kit's current llama.cpp pin.
Its release build and two host tests pass. The runner retains the model and
context, clears KV between requests, verifies exact prompt token IDs, and
records separate synchronized prefill/decode times with actual work counts.
The complete 6,975,879,296-byte Google QAT model has now been hashed locally:
`93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.

`native-campaign-plan.json` was written before inference. After native load/work
qualification and separate ANE cache preparation, its timing order is ANE /
native-kit / native-kit / ANE. Each process runs two warmups and seven measured
requests with 84 prompt tokens, ten outputs and nine continuation evaluations.
The exploratory stratum is AC, low-power on, power mode 1 and stable Fair.
Actual IDs, counts and all power/activity evidence must pass before ratios are
reported. Outer ANE median decode drift above 5% makes the block inconclusive.

The weights differ: ANE uses the standard checkpoint with INT8 FFNs and FP16
other ANE matrices; the available native-kit baseline uses Google's native-QAT
Q4_0 GGUF with Q6_K embeddings. This is a practical backend comparison, not an
isolated hardware or quantization experiment, and it does not introduce four-bit
ANE weights. The existing idle E4B critic service remains untouched and is an
explicit, sampled ambient condition on both sides. The baseline covers the
pinned backend, not the complete native-kit frontend/request layer.

The revised `native-comparison-fair-v2` queue has completed the native
qualification: all nine requests match the exact 84 prompt IDs and ten
reference output IDs, with nine decode evaluations and matching internal work
counters. Placement logs report 49/49 layers offloaded to Metal. Its concurrent
Cargo activity correctly disqualifies timing; no speed ratio is inferred.
The separate bounded cache preparation then visited all 162 entries, repaired
17 misses (8 QKV, 6 output, 2 FFN, 1 head/attention), and validated every load
and unload with zero inference evaluations. Strict inference remains the next
retention test. Preparation permits benign nominal/Fair thermals; timing still
requires an exact stratum.

Before any timing attempt, the idle machine returned to nominal. The Fair
queue was stopped with no timing results, and
`native-campaign-nominal-amendment.json` declares the identical ABBA campaign
with exact thermal state 0. Completed preparation is referenced rather than
repeated. Nominal and Fair results are never pooled.

Launches were paused again before the first timing attempt when unrelated
builds brought free space below the 16 GiB floor. Three verified inactive
August Cargo incremental trees were removed while holding their profile
locks, after every entry passed a seven-day minimum age and non-symlink
regular-file/directory check. Logical file sizes totaled 25,268,438,767 bytes;
the measured available-space increase was 14,396,436,480 bytes (APFS logical
sizes are not reclaimed-space measurements). Free space moved from
16,965,816,320 to 31,362,252,800 bytes. Source, models, binaries and current
builds were preserved. The same nominal queue then resumed with no inference
retry or repeated cache repair. Inspection, removals, helper source and `df`
receipts are alongside the queue evidence.

During these waits, the explicit
[sliding-QKV INT8 candidate](gemma4-sliding-qkv-int8-candidate-20260916.md)
was implemented and host-checked. It leaves global QKV FP16 and remains
unqualified for full-model quality/performance; the frozen comparison binary
is unchanged.

The first nominal timing process then loaded all 162 programs with zero
compilation and completed all nine requests with the exact expected 90 output
IDs (81 continuation evaluations). Power stayed nominal AC/LP-on/mode1, but
external Cargo/rustc work began during execution. The queue marked the timing
ineligible and halted before either native timing process. This establishes
fresh cache availability and repeated correctness, not an accepted ratio.

`native-campaign-quiet-plan.json` predeclares a revised launch gate: two quiet
minutes, with independent nominal AC/LP-on/mode1 and battery/LP-off/mode0 ABBA
chains. Each is bounded; results remain separate, and failures are never
automatically replayed. No cache repair is repeated. Queue revision v3 checks
only pin syntax/file presence on submission, avoiding model-sized hash scans
while another job runs; full hashes remain mandatory before and after each
execution. The offline analyzer has ten passing tests and independent 5%
drift gates for decode, prefill/import and phase sum.

The revised release worker is running with eight model jobs and two dependent
offline analysis jobs. Each completed AC/battery ABBA chain is analyzed
automatically; JSON appears in its `14-ac-analysis/trial.stdout` or
`24-battery-analysis/trial.stdout`. Analyzer success alone does not establish
every metric: inspect its explicit drift eligibility and null ratios.
The ten queue host tests pass after the submission change. All eight model
jobs were staged in less than one second of combined tool-reported execution
time, rather than scanning the 7 GB GGUF on every submission. Frozen worker
SHA-256 is `8c678b4de49eae7d1795628d3d91c77bd3bb9fc1e3ecd31bd099b10e321f59da`.

The last historical nominal battery ANE result is 6.08594 decode steps/s.
It must not be divided into a differently powered llama.cpp result to claim
a matched speed difference. This report currently records preparation and
staging; actual campaign acceptance and numeric results are still pending.

Two component jobs were added to the already running queue without restarting
it: `30-batch-ffn-qualification` and dependent `31-batch-ffn-timing-ac`.
The [two-token INT8 FFN probe](gemma4-int8-ffn-logical-batch-20260916.md)
has two passing numerical-gate/lane-order host tests and a successful release
build. Its frozen, signed binary is
`b89c68308f5e10b636e606ab2315225a3be42bee2379d46737809a1c7a0c5bf0`.
Qualification permits unrelated CPU builds and records no timing; it retains
the accelerator lock, one-compilation budget and driver journal. Timing
requires its successful qualification, zero compilation, no journal and the
same two-minute nominal AC quiet gate. Submitted manifests, source snapshots,
build logs and hashes are under `int8-ffn-batch-probe-20260916/`.

The live qualification succeeded: the existing S=1 graph loaded from cache,
one S=2 graph compiled, sixteen evaluations completed, and both programs
unloaded. All 84,480 batched FP16 output values match their S=1 references
exactly, with CPU error tolerances, swapped lanes, zero isolation and repeated
use also passing. Three input vectors are synthetic and one is a captured
layer-0 FFN input. No timing phases ran. A separate transient `llama-server`
appeared during the preparation, making sampled timing conditions ineligible;
preparation correctness remains successful. This does not certify exclusive
access against programs outside the cooperative queue. The timing job remains
pending and no full-model speedup is established.

## Quiet-window continuity correction

Review found that v3 retained other jobs' quiet-window start times while a
trial ran, and did not bound gaps between readiness observations. Matching
controls before and after that interval could incorrectly satisfy a quiet
window. The worker was stopped cleanly before any timing job began; the sole
completed job was the successful FFN correctness qualification.

Revision v4 resets every pending quiet window after a trial and rejects
observation gaps above 2.5 seconds. Eleven host tests pass. A live system-only
test allowed a one-second child to bypass an earlier five-second gate; after
the short child ended, the earlier job waited another 5,394.36 ms before
starting. Both jobs succeeded, with no accelerator calls. The same immutable
campaign resumed under v4, preserving and skipping its completed qualification.
Worker SHA-256 is
`cdb3b76059d5dfbea47d270a2210a4bac8d94878c16fbb360c351c24d25e736b`.
Source, tests and smoke receipts are beside the earlier revisions.

An additional preparation job, `32-live-ffn-inputs`, uses a frozen diagnostic
decoder to verify the existing 21/84/652-token references and capture actual
FFN inputs from their first two ANE steps, with zero compilation. See
[the projection/input report](gemma4-batched-projections-and-live-inputs-20260916.md).
It makes no performance claim and runs under the same serial accelerator lock.

That capture succeeded with 240 verified input vectors and all 15 reference
outputs, zero compilation and 162 normal unloads. A subsequent component job
`33-batch-live-inputs` hit the existing serial FFN's CPU-reference tolerance
on one newly captured input; it performed no batched evaluations and unloaded
both programs. The v4 worker correctly halted. The failure and subsequent
offline precision diagnosis are preserved in the projection/input report.

After reviewing that numerical failure, only the still-unstarted native-kit
comparison jobs were staged in `baseline-only-v4`. Backend binaries, work
counts, power strata and quiet windows are unchanged; no cache repair or failed
component retry is included. The offline analyzer revision v3 additionally
rejects the newly introduced FFN-input capture flag/payload and passes all ten
host tests. Its frozen SHA-256 is
`68d3b64f899dc4cf0e5dea8f08fdb517e3841d24dfe28b8fa41840ef9c67cd67`.
That worker subsequently completed the eight-call baseline output capture;
accepted timing remained pending.

## Complete output capture and scheduler follow-up

The new baseline S1 diagnostic captured 30,720 finite values with one cache
hit, zero compilation and one unload. Exactly one value failed the unchanged
CPU tolerance. The subsequent CPU-only analysis exposed a JSON numeric-mirror
validation defect before accelerator execution; its failed job and corrected
five-test regression receipt are preserved. See
[the numerical report](gemma4-ffn-oracle-discrepancy-20260916.md).

Untouched native comparison jobs continue in `baseline-only-v5`, with the same
backend pins, work counts, 16 GiB hardware disk floor, power strata and quiet
windows. CPU-only job `38-s1-cpu-comparison` uses a 4 GiB disk floor because it
does not stage ANE sources or execute a device. Its never-started predecessor
37 was withdrawn while the worker was stopped; the original manifest and
withdrawal receipt remain in `withdrawn/`. No failed job is replayed.

Revision v5 of the worker skips expensive process/server probes when the
current power stratum or disk headroom already makes a job ineligible. The
earlier worker repeatedly observed CPU job 38 as ready for over 100 seconds,
but redundant scans for the blocked AC/battery timing jobs could push its
inter-observation interval above the unchanged 2.5-second limit. Skipped
activity checks are explicitly marked `activity_sampled=false` and always
produce `ready=false`. Ready results still require all existing checks.
The eleven host tests pass; fresh quiet windows remain mandatory after a trial.
Frozen worker SHA-256 is
`d7b168a93bcb946ca99397cfb8f50f6c45e68e8c0b36d5d9caf9beaa4272e359`.
On resumption the previously starved CPU job started. No accelerator cache
repair or model timing was needed to exercise this scheduling correction.
Job 38 then completed with zero accelerator/compiler calls; the corrected
worker remains active for the original AC and battery baseline chains.
At the final observation, hardware jobs remained below their 16 GiB free-disk
floor. Skipped process checks in that state make no claim that the system is
otherwise idle. No accepted native-kit slowdown ratio exists yet.

## Additional queued work and disk headroom

The v5 worker accepted and completed new CPU-only job `39-s1-hybrid-controls`
while native-kit timing remained gated. It used stored INT8 coefficients and
two hybrid projection controls; see the numerical report. Its successful
preparation is not a backend timing result. This exercises submission while
the worker is already waiting, without a babysitting agent or worker restart.

Job `15-ac-int8-head` is staged after `14-ac-analysis`. It times the already
qualified 16,384-row INT8 vocabulary tile with strict existing-cache loading,
zero compiler calls and no driver journal. Its frozen binary SHA-256 is
`63f6859d8a42f94082370464815462629817465b84ce60ecd3d4c41da4b0ba00`.
It cannot delay the native-kit AC comparison through its dependency chain.
It does not qualify the complete vocabulary head or model by itself.

Conservative cleanup under exclusive Cargo profile locks removed obsolete
compiler objects older than 48 hours from `native-platform-full-horizon` and
incremental trees whose every descendant was older than seven days from five
inactive worktrees. Sources, model assets, executables and evidence were
preserved. Observed filesystem availability increased by 8,749,953,024 bytes
(about 8.15 GiB); concurrent writes and APFS mean this is not the sum of logical
deleted file lengths. Candidate inventories, Rust helpers, lock/age checks and
before/after receipts are under `int8-next-controls-20260916/`.

The hardware disk gate was clear at the subsequent observation. Cargo/rustc
activity, including ongoing experiment development, still blocked timing.
The original AC/battery workload and power gates remain unchanged, and no
accepted native-kit slowdown ratio has been produced.

The next live preparation, `40-metal-bk64-qualification`, also completed under
the same worker, passing the new reduction-tile component checks. Conditional
timing `16-ac-metal-bk64` depends on that pass and follows job 15. See
[the Metal report](gemma4-metal-bk64-experiment-20260916.md).

Jobs 50–53 stage the first fresh AC ABBA block for the already qualified
stacked INT8 FFN full-model candidate after job 16. The original frozen decoder,
metallib and checked all-layer receipt remain pinned; candidate B changes only
the explicit FFN layout plan. Both adjacent A/B pairs have pinned comparator
validators. No compiler or driver journal is enabled. This is the first block
of the previously specified campaign, not automatic promotion: inspect outer
baseline drift and complete repeated blocks before accepting a small win.

At the observation after job 40, AC timing conditions were ready with no named
competing process. The worker was accumulating a fresh quiet window; this
readiness observation alone is not a completed timing result.

A subsequent 45-sample observation found a transient unapproved `llama-server`
(PID 34083) during an otherwise ready nominal-AC interval. The worker correctly
reported `ready=false`; this restarts the quiet window. Its later absence does
not erase that interruption. Read-only snapshots are preserved in
`int8-next-controls-20260916/quiet-window-observations.jsonl`. No server was
stopped or added to the idle-server exception by this task.

## Cache-audit v6 and continued asynchronous work

`baseline-only-v5` halted on job 45's cache miss before evaluation. All timing
jobs there remained unstarted. The new `cache-audit-v6` worker inspected the
original frozen baseline, finding 118/162 graphs. After compiler-cache cleanup,
one bounded recovery restored exactly the 44 misses; a subsequent independent
zero-compile inspection found all 162. See the updated
[batch-capture report](gemma4-int8-batch-capture-20260916.md).

The unchanged native-kit workload is now submitted in this new queue, with
the successful cache-survival receipt pinned. Nominal AC and battery are still
separate. Explicit additional Fair-state ABBA blocks are staged for battery
(jobs 30–34) and AC (40–44), using the same analyzer's already tested
`exploratory_stable_fair` classification. Fair is never pooled with nominal.
The laptop changed from battery/low-power-off/mode-0 to AC/low-power-on/mode-1;
jobs wait for their own recorded controls. Other Cargo/rustc work has repeatedly
prevented a complete quiet launch window. No accepted native-kit ratio exists.

While those timing jobs waited, the queue completed a bounded reference
transaction development cycle: job 60 compiled and passed three host tests;
job 61 passed the live full-model S1 acceptance/rejection/failure fixture with
3,960 evaluations, zero compilation and all 162 successful unload returns.
See the [transaction report](gemma4-two-token-transaction-design-20260916.md).
These are correctness results, not timing observations. The worker remains the
sole hardware-lock owner; its lock spans the worker lifetime, so a second queue
worker must not be launched alongside it using another lock path.

Job 62 subsequently retained nine completed serial FFN outputs, then halted on
the confirmed missing S2 cache entry, with no compilation or S2 execution.
The worker exited nonzero and its handle was reaped. The twenty untouched
native-kit timing/analysis manifests were submitted to `baseline-isolated-v7`;
their references to completed cache evidence still point to the original v6
receipts. No successful result was copied or represented as a new execution.
Only the benchmark worker now owns the common hardware lock. Candidate cache
restoration and broader batch qualification remain separate pending work.

## Bounded S2 recovery and return to native timing

V7 was stopped with no timing attempt started while v8 held the same hardware
lock for jobs 63–66: host tests, build, one known S2 graph restoration and broad
S1/S2 output capture. All four jobs succeeded. The capture used zero compiler
calls and matched all 145,920 batched output values against serial execution;
the unchanged legacy CPU diagnostic still fails on its known shared coordinate.

V8 was stopped and reaped with exit zero. V7 resumed after preserving its STOP
marker as `STOP-before-s2-diagnostics`; no existing job or result was rewritten.
New timing jobs 70/71 use the original CPU-qualified S2 timing binary and input
subset, with pinned original and broader qualification receipts. They follow
the nominal AC/battery native comparison analyses respectively. They cannot
produce Fair timing. Native Fair ABBA blocks remain separately exploratory.
No accepted native-kit slowdown ratio exists at this resumption.

The same live v7 worker then accepted jobs 72/73 without a restart and completed
the explicit Fair S2 probe's fourteen host tests and release build. Seven frozen
parser checks passed. Jobs 74/75 stage AC/battery Fair component timing after
the corresponding native comparison analyses. These use an explicit raw-sample
reader and keep nominal eligibility false; the original nominal jobs remain
unchanged. See the [Fair S2 report](gemma4-int8-s2-fair-timing-20260916.md).

The same worker accepted and completed layer-major host job 77 and live job 78.
The live preparation matched all 96 layer states and transaction continuations,
with 3,728 evaluations, zero compilation and all 162 unloads. Its seven observed
competing-process samples remain recorded and exclude timing claims. See the
[layer-major report](gemma4-two-token-layer-major-20260916.md).

The first independent short-lead S2 pilot, job 04, completed with eligible
sampled controls and no contention. All six adjacent comparisons favored S2,
but all three blocks failed the predeclared repeat-drift gate. Its result is
`inconclusive_repeat_drift`; the longer predeclared pilots and native-kit chains
remain untouched. No accepted native-kit slowdown ratio exists.

## Buffered report reads

After adding jobs 79/80, even preparation launch windows repeatedly reset while
the old worker spent appreciable time reading completed reports. Three JSON
read sites passed raw files directly to serde_json. They now use BufReader;
the scheduling, observation-gap, contention and power gates are unchanged.
The eleven queue host tests pass. The old worker stopped cleanly with no active
child and exit zero; its STOP marker is preserved as `STOP-before-buffered-reports`.

Host status-read observations changed from 3.64 seconds to 0.30 seconds, with
identical job summaries. These are two host observations, not a controlled
inference speedup. Frozen worker v6 has SHA-256
`bc02bf13c917874a066cd0de66b17fd81461ae612c12f07b355817eed39aa210`.
It resumed the same v7 queue and shared hardware lock with all results and
pending manifests retained. Source pins, test/build logs and status observations
are in `int8-s2-interleaved-20260916/queue-v6-build-receipt.json` and adjacent files.

## Shared observations and user-requested pause

After buffered reads, repeated process/idle-server probes for separate waiting
jobs still consumed enough time under host load to reset ready quiet windows.
The latest source shares raw observations within a single queue pass, while
applying every job's own policy and child exclusions independently. The oldest
observation includes probe duration and expires after 2.5 seconds; no data are
shared across passes, and prelaunch/active-child probes remain fresh. Thirteen
queue host tests pass, including policy separation and expiration boundaries.
The normal worker executable has not yet been rebuilt or live-qualified.

The user then requested a pause. Worker v6 (PID 30722) was already stopped and
joined with exit zero, with no active child. `baseline-isolated-v7/STOP` remains
present. Seven jobs completed and 29 remain pending; job 81 never started.
Direct host-only checkpoint tests do not alter those queue results. No worker
was restarted. See [the handoff](../HANDOFF.md) before any resumption.
