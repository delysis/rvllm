# Gemma 4 ANE dynamic-QKV vertical slice

This default-off candidate replaces 48 constant-weight QKV programs with two
weight-independent single-input/single-output programs: `[3840,8192]` for the
40 sliding layers and `[3840,8704]` for the eight global layers. Each layer
keeps an independent request and resident packed weight surface. QKV still
uses one evaluation per layer; this is a compile/load graph-count experiment,
not an evaluation-count reduction.

The candidate is combined with the already qualified all-sliding fused
attention/output route. Inference is cache-only and fails on any missing graph.
The correctness referee compares every post-layer residual bit and final token
signature against the constant-QKV baseline, records both MIL SHA-256 cache
identities, requires zero compiler calls, and accounts for every QKV and
attention/output evaluation. Timing is gated on correctness and uses fresh
decoder instances with reset-identical state in both counterbalanced orders.

The manifests are prepared but have not been submitted or executed. The sealed
test executable is `target/release/rvllm-runtime-tests-dynamic-qkv`, SHA-256
`dde807197d6d7971ae3135441de31ef348c86e5188fa7332dbb02d02cf110580`.
No accelerator, queue, correctness, performance, or promotion claim is made.

## Referee correction

The original `01`--`04` packet compares the static/separate route against the
combined dynamic-QKV/all-sliding-fused route. Preserve those receipts as
combined-route evidence, but do not attribute their timing delta to QKV.

The `*-v2.json` packet fixes the causal comparison: both arms use the already
qualified `all-sliding-fused-cached` attention/output route, and only the QKV
weight/program plan changes. Use v2 for dynamic-QKV adjudication. Its sealed
executable is `target/release/rvllm-runtime-tests-dynamic-qkv-v2`, SHA-256
`23ce49d33106ad7fea1dba47031606076c37cf250270d4c3e7a43b3236f0f52a`.
