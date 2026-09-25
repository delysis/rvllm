# Output projection + FFN boundary slice

This default-off slice targets the remaining FFN-heavy decode phase by replacing
the output-projection and FFN evaluations with one single-input/single-output
graph. It reuses the exact row-INT8 gate/up/down coefficients and FP16 scales.
The packed input is `[attended, residual]`; the sole output is the raw FFN
branch. Post-FFN norm, residual, and layer scaling remain outside the boundary.

Runtime routing is deliberately deferred. The graph crosses two host FP32
RMSNorms and a residual materialization. No ANE evaluation API is exposed until
the component oracle establishes those reduction and rounding semantics. The
first queue stage permits exactly one compile and zero evaluations. The second
stage is reserved for a future bounded component-correctness probe after the
compile receipt fixes the accepted MIL dialect; it is fail-closed and is not a
16-token/full-route or timing claim. ABBA/BAAB manifests are intentionally not
provided because same-route timing before component correctness would be test
theater.

Host qualification:

```text
cargo test -p rvllm-apple ane_output_ffn --lib
cargo test -p rvllm-apple output_ffn_blob_preserves --lib
```

Evidence boundary: pooled measurements motivating this work were approximately
FFN 562 ms, QKV 473 ms, and vocabulary 168 ms. Stacked FFN was neutral;
interleaved and down4 lost. Current QKV is already one evaluation per layer.

## Compile adjudication

The sealed clean-tree probe made exactly one compile attempt and zero
evaluations. ANEC rejected the graph with `ANECCompile() FAILED`; the driver
journal records `compile_requested`, `descriptor_created`, `compile_begin`, and
`compile_failed`. The graph therefore remains compile-blocked and default-off.
No reload, correctness, timing, or full-route claim follows. The next iteration
must isolate the unsupported MIL dialect boundary (the new explicit RMS
`reduce_sum`/`rsqrt` sequence is the leading hypothesis) with separately sealed
compile probes before any evaluation API is added.
