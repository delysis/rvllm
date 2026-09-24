# Kernel game standard v1

## Submission

A runnable submission MUST be immutable and content-addressed. It MUST bind:

- task ID and candidate revision;
- candidate class, candidate selector, and independent control;
- source tree, executable, generated MSL, and metallib identities;
- model, reference, workload, and oracle identities;
- exact required positive dispatch counts.

A null or missing required identity is proposal-only. Candidate and control MUST
not be identical. Unknown or duplicate transport fields MUST be rejected.

## Evidence

Evidence is monotonic. Source/build success does not imply component, dispatch,
route, timing, or promotion success.

Correctness and safety are prerequisites. A route MUST NOT receive a speed score
unless the exact sealed executable/artifacts ran, reference continuation matched,
no forbidden compiler call occurred, expected work completed, all required
candidate dispatches occurred, and no foreign candidate dispatch occurred.

Blocked infrastructure is not a candidate failure. Failed and inconclusive
attempts remain durable history and are never overwritten or silently retried.

## Timing

Timing MUST use a predeclared plan. The current helper supports repeated ABBA
blocks. Every measured sample MUST:

- follow the declared arm order;
- remain in one AC/battery, low-power, pmset, and thermal stratum;
- have finite positive duration;
- perform zero compilation.

Control drift above the declared ceiling invalidates the campaign. All samples
are retained. The practical-effect threshold is policy supplied by the task; it
is not inferred after seeing results.

Independent confirmation is required before a result can be marked promotable.

## Queue ownership

The existing experiment queue remains the single hardware-owner mechanism.
Kernel-game binding is optional for historical jobs. When present, the queue
MUST verify the sealed submission and its pinned artifacts before launch.

The queue MUST NOT clear STOP, change OS power settings, kill unrelated
processes, repair caches implicitly, overwrite result directories, or replay a
failed/incomplete attempt in place.

## Route receipts

Normal full-route receipts MUST carry their own immutable build/workload evidence
rather than relying on an operator to join unrelated files later. At minimum:

- executable and metallib path/hash;
- generated MSL hash;
- model-config and reference/workload hashes;
- requested candidate and actual dispatch receipt;
- compiler counter before/after/delta;
- output-token and decode-step counts;
- command completion and reference-match state.

These fields bind evidence; they do not themselves assert tensor qualification or
performance acceptance.

## Promotion

The game may emit a promotable recommendation only after route qualification,
valid timing, practical improvement, and independent confirmation. Promotion is
a separate explicit reviewed source change. The game never merges or changes
production defaults itself.
