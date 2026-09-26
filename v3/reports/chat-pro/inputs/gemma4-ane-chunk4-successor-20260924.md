# Gemma 4 ANE INT8 FFN successor design and implementation

Work in ordinary Chat and remain there; do not switch to Work.

## Objective

Produce a reviewable source patch for a new Apple Neural Engine INT8 Gemma 4
12B FFN candidate that preserves or improves the intended Chunk4 locality
benefit while satisfying the existing prospective bounded-equivalence oracle.
Do not relax, reinterpret, or replace the oracle.  The old Chunk4 candidate and
its failed receipts must remain intact under their existing identity.

## New evidence

The existing `ane-int8-ffn-chunk4` candidate was evaluated against the plain
stacked INT8 control on three pinned real layer-0 FFN inputs.  Both control and
candidate were bit-exact and finite across two repeats each.  Chunk4 nevertheless
failed the four-ULP material-value gate on every input:

| input position | differing lanes | max absolute | max material ULP | relative L2 | material ULP violations |
|---:|---:|---:|---:|---:|---:|
| 21 | 302 / 3840 | 0.015625 | 13 | 0.000085943 | 9 |
| 84 | 190 / 3840 | 0.015625 | 12 | 0.000077627 | 4 |
| 652 | 154 / 3840 | 0.031250 | 8 | 0.000061110 | 4 |

Aggregate relative L2 was 0.000069272.  All finite, hybrid absolute/spacing,
maximum-absolute, near-zero, and relative-L2 gates passed.  The only numerical
failure was localized material-value ULP error.  Two ANE compiler calls were
used and all twelve program evaluations completed.

The complete immutable evidence is attached.  Treat it as authoritative.

## Required analysis

Trace the exact numerical and graph-structural difference between the plain
INT8 control and Chunk4 implementation.  Explain which reassociation,
partial-result conversion, concatenation, reduction order, or CoreML/ANE graph
choice plausibly produces the localized 8-13 ULP material errors.  Distinguish
claims established directly by source from hypotheses about proprietary ANE
lowering.

Design the smallest coherent successor that restores the control's numerical
boundary without merely reverting to the control graph.  Preserve safe,
idiomatic Rust and current private-ANE lifecycle/cache invariants.  A new
candidate must have a new explicit identity and selector; do not mutate the
meaning of `ane-int8-ffn-chunk4` or production defaults.

## Deliverable

Return a downloadable ZIP containing:

1. a unified patch relative to the attached source tree;
2. complete edited source files;
3. focused deterministic unit tests for graph construction, layout, identity,
   resource bounds, and rejected shapes;
4. any necessary additive ignored device-oracle entrypoint using the existing
   `rvllm.ane.ffn-layout-equivalence.v1` comparator unchanged;
5. a concise `DESIGN.md` covering root-cause evidence, expected performance
   mechanism, resource/lifecycle bounds, exact qualification commands, and
   remaining uncertainty.

Do not claim correctness or speed without device evidence.  Do not change
shipping selectors, defaults, cache semantics, the historical exact oracle, or
the prospective equivalence thresholds.  Avoid broad refactors.  The patch
must compile on macOS and must not add unsafe Rust.
