# Gemma 4 12B cache recovery and cancellation

The default INT8 decoder needs 162 compiled programs. Cache entries can
disappear despite an unchanged signing identifier and unchanged model IDs.
This change gives the CLI a one-command inspection/preparation workflow with
fresh serial processes, explicit compile bounds, graph-boundary cancellation,
and independently checked per-part receipts. Inference retains its default
zero compiler budget. No kernel, weight precision or private selector changes.

## Implementation and checks

`--inspect-ane-cache all-int8` visits all programs with zero compiles and
evaluations. `--prepare-ane-cache all-int8` uses four serial child processes
of the current executable: QKV (48), output (48), INT8 FFN (48), then
head/attention (18). Each visit loads and immediately drops one graph. The
parent validates model identity, capacity, program names, compilation bounds,
and matching per-model load/unload events before continuing. It rejects failed
driver events and any inference evaluation.

SIGINT, SIGTERM and SIGHUP set a cancellation flag. The parent writes a marker
for its child and waits for exit; the child checks between graph operations.
No synchronous private-framework operation is interrupted or automatically
retried. A cancelled part keeps its journal but contributes no unverified
partial counts to the batch aggregate. An inspection can complete with missing
entries, so callers also check `all_programs_available_at_visit`.

Focused host checks: 12 library tests and three batch validator tests passed;
one unrelated hardware test remained ignored. The earlier CLI output-budget
test also passed. Cancellation is checked before checkpoint access. Validator
tests reject incomplete compiles, unload failures, missing unload returns,
incorrect model identities, duplicate names, and wrong work metadata.

## Live receipts

Evidence directory: `gemma4-12b-evidence-20260914/cache-batch-recovery-20260916/`.
Frozen executable SHA-256:
`b2a5fcd1c7f567f98431b018e43b90fdf690e82ef12ad202c09cd6b0dafe3ae1`.
Signing identifier: `rvllm_disaggregated_infer-9d1f7275eb5937c9`.
Checkpoint: Google Gemma 4 12B IT,
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; capacity 1024.
Source snapshots, build/test logs, signature and replay arguments accompany it.

The complete first inspection visited 162 graphs and found 66 cached: 19 QKV,
20 output, 23 INT8 FFN and four head/attention. All 66 loads were successfully
unloaded, with zero compiler calls or evaluations.

The cancellation run received SIGTERM on batch parent PID 63206. It completed
the QKV/output parts, then stopped after FFN layer eight. Its journals contain
105 descriptors and 43 completed loads with 43 matching unload returns, zero
compiles, zero evaluations, and no failure events. No head/attention child
started. The parent exited nonzero with `status: cancelled`, retaining the
39 verified loads from completed parts and excluding the four FFN loads from
its aggregate. An independent journal audit accounts for all 43 loads.

No owned source-staging directory remained after inspection/cancellation.
Boot identity remained `1789488066` / `242846`.

The complete preparation then restored exactly the 96 missing graphs:
29 QKV, 28 output, 25 INT8 FFN and 14 head/attention. All 162 visits completed
with successful unload returns, and the parent validated all four parts.
There were zero inference evaluations. This establishes recovery at each
visit; the subsequent zero-compile worker run checked simultaneous residency
and numerical behavior separately.

The same frozen executable then used `--runtime-worker true` with references
for capital (21 prompt tokens), copy (84), recall (652), then capital again
(21). All 17 output token IDs matched the independent reference sequences,
including the repeated short request after the long one. The worker performed
13 ANE decode steps and zero Metal decode steps, with no fallback. It loaded
all 162 programs from cache, made zero compiler calls and completed 2,704 ANE
evaluations. At normal exit it returned successfully from all 162 unloads.
The independent journal audit found no failed events or unmatched lifecycle;
no owned staging directory remained, and boot identity was unchanged.

Receipts: `worker-four/report.json`, `worker-lifecycle-audit.json`,
`worker-driver-phases.jsonl`, `staging-remains-final.txt`, `boot-after.txt`.
This is a journaled correctness/lifecycle run, not a throughput comparison.
It establishes retention through this immediate recovery-to-inference
sequence, not indefinite cache persistence. The HTTP adapter was not rebuilt
or requalified in this cache-CLI change.

## Disk and timing boundaries

Free disk space was 17 GiB before repair. The conservative existing Rust
cleanup utility verified two inactive Cargo target directories, held their
debug/release Cargo locks, and removed only regular `.rlib`/`.rmeta` archives
at least 24 hours old. It reclaimed 15,736,535,548 bytes (10,331 files), with a
per-file receipt; free space rose to 32 GiB. No source, binaries, model assets,
or ANE daemon cache was removed. This does not establish why ANE entries were
missing.

The machine reported AC power, low-power mode enabled and Fair thermal state
before these checks. They qualify recovery and lifecycle only. Neither their
wall times nor process CPU cycles establish an accelerator speedup.

The [cache-eviction research addendum](gemma4-ane-cache-eviction-addendum-20260916.md)
distinguishes Apple's documented Core ML eviction behavior from extracted
private maintenance symbols. Neither establishes a numeric quota or the cause
of these losses on macOS 24G84. The 62/66 retained-entry observations do not
justify a 64-entry-cap assumption or a new retention selector.
