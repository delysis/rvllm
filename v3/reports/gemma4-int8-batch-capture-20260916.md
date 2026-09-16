# Two-token INT8 FFN: complete diagnostic capture

The broader batch qualification stopped on the serial S1 CPU-reference gate
before any S2 evaluation. The earlier arithmetic diagnosis retained that
failure and did not establish a superior oracle. This new diagnostic separates
data collection from qualification, so the actual batched outputs can be
examined without changing the original gate.

`--capture-batch-outputs true` requires the two-token INT8 comparison, strict
existing-cache loading, zero compilation and a driver journal. Other layout,
CPU-only, restoration and S1-capture modes are rejected in combination. The
original qualification and timing implementation is unchanged.

Eight identified sources plus a zero vector require nine serial calls and
nineteen two-token calls. Every source occupies both lanes; the sequence checks
swapped lanes, zero isolation and repetition after changing inputs. Complete
FP16 bits, hashes, all legacy tolerance failures and all S1/S2 bit differences
are retained. Successful finite collection explicitly makes no numerical,
timing, accepted-draft-token or full-model claim.

Review found no graph, packing or indexing blocker. It identified a retention
gap if an S2 cache load failed after completing the serial outputs. The revised
implementation writes an exclusive partial-results JSONL journal, flushing
each completed record before the next device call. Its identity record binds
the exact affine coefficients and every input. Every partial record declares
`collection_complete=false`; the ordinary complete report remains separate.
This protects against ordinary later errors, not power-loss durability. The
driver journal remains authoritative for successful unloads; Rust ownership
release alone is not an unload receipt.

## Host work and queued artifacts

Job 41 ran eight release host tests successfully through the experiment worker;
no device calls occurred. Its fixed build environment required a first release
test dependency build. All pinned files were unchanged before/after execution.
The worker stopped cleanly afterward. Never-started build job 42 was preserved
under `withdrawn/` and replaced after the journal review fix; no failed attempt
was replayed.

The journal's standalone test verifies that a completed record survives a
later error and that an existing file cannot be overwritten. It passes. The
first standalone compile invocation lacked the module wrapper required for
`pub(super)` visibility; the corrected wrapper is retained with its receipt.

Jobs 43 and 44 completed the revised full probe tests (nine passed) and binary
build under the same worker, preventing overlap with native-kit hardware timing. They used the
observed battery / low-power-off / mode-0 preparation conditions. Source pins,
old and revised snapshots, manifests and the journal test are under
`gemma4-12b-evidence-20260914/int8-batch-capture-20260916/`.

The frozen capture executable has SHA-256
`04cc6b66de2d07b5f3853fa1ac4369fafb292842f448fcbb012380a3888ed56b`.
Job 45 stopped on an S1 cache miss, before any load, evaluation or compilation.
Its partial journal contains only the affine/input identity record. The failed
attempt remains in the halted `baseline-only-v5` queue; it was not replayed.

Job 46 used the older frozen full-model executable to inspect all 162 baseline
graphs. It found 118: 29 QKV, 36 output, 40 INT8 FFN and 13 head/attention.
FFN layer zero was absent under the same original client and descriptor too.
This rules out the new capture executable as the sole explanation. The boot
identity stayed unchanged; the private cache's eviction cause remains unknown.

After conservative compiler-artifact cleanup, job 47 restored exactly the 44
observed misses (19/12/8/5). All 162 load/unload visits completed and no graph
was evaluated. Job 48 then independently visited all 162 again, with all hits,
zero compilation, zero evaluation and matching successful unloads. These are
cache-availability receipts, not inference or speed qualification. The S2 graph
is outside this 162-program baseline inventory and remains unverified here.

Evidence is in `experiment-queue-20260916/cache-audit-v6/`. The nominal native-kit
ABBA comparison is resubmitted there against the surviving cache, with a
separate predeclared battery/Fair stratum. No timing attempt had begun when
the cache checks completed. The complete S1/S2 capture is still outstanding.

Job 62 later completed all nine S1 evaluations, unloaded S1 successfully, then
stopped on the separate S2 graph's cache miss before its load/evaluation. There
were zero compiler calls. The partial journal preserved the identity plus all
nine serial records, proving the intended ordinary-error retention behavior.
Every one of the first eight output bit vectors matches job 35 exactly; the
zero vector also remains zero. The same source-7/coordinate-3627 legacy CPU
failure repeats (ANE -0.5537109375, reference -0.5212233066558838). No tolerance
or numerical gate changed. This establishes reproducibility of that discrepancy
after cache recovery, not an explanation or a superior oracle.

The failed attempt halted `cache-audit-v6` and remains untouched. All its timing
jobs were still unstarted. They were copied as new submissions to the isolated
`baseline-isolated-v7` queue, preserving artifact pins and power/thermal controls
and updating only analysis-result paths and externalized prerequisites. The
worker was confirmed terminal before the new worker acquired the same hardware
lock. The S2 miss now justifies one reviewed, bounded graph restoration; it does
not authorize automatic repeated cache repair or any new layout.

## Disk receipt

The safe Rust cleanup helper held both Cargo profile locks for two inactive
targets, checked Cargo cache identity and a 48-hour age floor, and removed only
regular `.rlib`, `.rmeta` and `.o` files. It removed 19,169 compiler artifacts
with 7,260,782,048 logical bytes. Available blocks rose from 17,160,952 to
24,037,484 KiB, about 6.6 GiB recovered. Source, executables, model assets and
ANE cache were preserved. The removal inventory and both `df` receipts are in
`int8-batch-capture-20260916/`. Concurrent system activity means the block delta
is an observed change, not exact per-file physical allocation attribution.

## Complete S2 capture after bounded restoration

The separate v8 campaign restored only the already qualified missing S2 graph
and completed capture job 66 with zero compiler calls. All 145,920 batched
FP16 output values match S1 exactly, including swap, zero-isolation and repeated
cases. The legacy CPU discrepancy remains identical in both paths and its gate
is still false. Independent raw-array and driver lifecycle audits are retained
under `int8-s2-restoration-20260916/`; see the
[restoration report](gemma4-int8-s2-restoration-20260916.md).
The v8 worker exited cleanly and v7 native-kit timing resumed. This completes
the broad component equivalence capture; timing and full-model S2 remain open.
