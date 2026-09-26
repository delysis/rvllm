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
- remain in one AC/battery, low-power, and pmset stratum;
- record thermal state for every sample; a campaign MAY admit multiple known
  thermal states only when comparisons remain local to complete ABBA blocks,
  each block reports its observed thermal sequence, and blocks are never pooled
  as if they shared one thermal stratum;
- have finite positive duration;
- perform zero compilation.

Control drift above the declared ceiling invalidates its ABBA block. A campaign
that predeclares a single thermal stratum is invalidated by a stratum change;
  an any-known-thermal campaign instead keeps complete locally matched blocks as
separate strata and aggregates their paired candidate/control effects. All
samples are retained. The practical-effect threshold and cross-stratum
aggregation rule are policy supplied by the task; neither is inferred after
seeing results.

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

## Generated-code and pipeline-resource companion evidence

A Metal submission MAY attach an
`rvllm.metal_artifact_evidence.v1` companion report. The report is relevant to a
kernel-game receipt only when its source and metallib SHA-256 identities exactly
match the receipt's sealed generated-source and metallib identities. The report
itself MUST also be retained by hash; a path is not an identity.

The repository collector compiles the supplied source with the resolved public
Xcode Metal toolchain and records:

- collector, source, AIR, metallib, and compiler-tool path/hash identities;
- compiler versions, SDK identity, and exact compile/link argument vectors;
- hashed raw `metal-objdump` build-table and disassembly captures, including a
  failed-tool status when the installed public tool cannot produce them;
- live public `MTLDevice` properties and an observation identity; and
- public `MTLComputePipelineState` execution width, maximum thread count, and
  static threadgroup-memory size for every requested entry point.

Compilation, link, metallib loading, or any requested PSO failure is fatal. The
collector does not use private Apple APIs. Public Metal tooling does not expose
a supported register count, register residency, or occupancy value, and its raw
disassembly is not treated as a stable semantic ISA contract. Consequently the
report MUST leave those claims unavailable and SIMD-matrix or low-bit-unpack
machine lowering unverified unless a separately reviewed, version-pinned
semantic verifier is supplied. Source spelling, a successful compile, or an MLX
trace is not such proof. Pipeline metadata also does not prove that a timed run
used the PSO; that remains the route receipt's dispatch responsibility.

## Promotion

The game may emit a promotable recommendation only after route qualification,
valid timing, practical improvement, and independent confirmation. Promotion is
a separate explicit reviewed source change. The game never merges or changes
production defaults itself.
