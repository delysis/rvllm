# Split-32 normal-route qualification

This is an explicitly selected research route. Shipping defaults are unchanged.
The candidate uses 32 dynamic visible-prefix partitions and a complete FP32
sufficient-statistic merge for Gemma 4 global decode at exactly 16 query heads,
one KV head, D512, BF16 cache/output and scale 1.0.

`run-one.sh` executes two identical captured-token cases through the normal
Gemma model route. It rejects missing candidate dispatch, compilation during
the measured inference cases, or non-repeatable output tokens. The native
operator oracle separately covers independent FP64 comparison, BF16 rounding,
holes, tails, newest-K/V visibility, arena guards, and repeated use.

The eight job manifests are deliberately staged here rather than inserted into
the live queue. `stable_seconds` is zero, thermal state is not gated, and the
quiet-process list is empty. The existing v1 queue schema requires either AC or
battery as a source predicate; these manifests retain AC as that prerequisite.

`smoke-L256-summary.json` preserves the compact evidence from a completed real
Gemma 4 12B BF16 normal-route smoke. The raw reports were intentionally not
checked in; their hashes and ephemeral paths are recorded. Preparation compiled
one library and 54 pipelines, while both inference cases recorded zero compiler
calls and 16 partial plus 16 merge dispatches each. The smoke is route evidence,
not a speed comparison; the staged candidate/control jobs provide that next
measurement.
