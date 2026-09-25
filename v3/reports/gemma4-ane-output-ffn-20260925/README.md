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

The first isolation arm lowered every nested expression to explicit SSA values;
ANEC still rejected it. Nested-expression syntax is therefore ruled out as the
sole cause. The next bounded probes must test `reduce_sum` and `rsqrt`
independently in otherwise minimal known-good graphs.

Those minimal probes are now decisive: `reduce_sum` compiled, while `rsqrt`
failed. The full graph's current blocker is therefore the ANEC `rsqrt` dialect,
not reduction support. The next arm should replace it with a separately probed
inverse-square-root formulation (for example `pow(x, -0.5)`) before rebuilding
the full graph; numerical equivalence still requires device evidence.

The sealed `pow(x, -0.5)` dialect probe compiled successfully. The full graph
now uses that formulation in place of `rsqrt`; this establishes dialect
acceptance only, not RMS numerical equivalence.

The rebuilt full graph still failed compilation. Therefore `rsqrt` was a real
dialect defect but not the only blocker. The next isolation boundary is the
mixed FP16 output projection plus INT8 FFN graph without either RMS sequence;
only after that compiles should reduction-plus-power be reintroduced around the
known-good core.

That mixed core compiled successfully: FP16 `Wo`, residual add, the unchanged
row-INT8 gate/up/down constants, GELU and final projection are accepted together
in one graph. The remaining compile blocker is therefore inside the RMS
composition, not the mixed weight blob or the output-projection/FFN fusion.
Next isolate `reduce_sum -> pow` composition and scalar broadcast back onto the
hidden tensor before restoring both norms.

Both RMS composition probes compiled: `reduce_sum -> pow(-0.5)` is accepted,
and the resulting scalar broadcasts back across the hidden tensor through
`mul`. The remaining candidates are interaction with the mixed convolution
graph or the presence of two RMS sequences in one program; the next smallest
probe is two sequential RMS blocks without convolutions.

Two sequential RMS blocks also compiled. The full failure is therefore an
interaction with the mixed graph or its learned-gamma multiplications, not a
simple ANEC limit on repeated reductions or inverse square roots. The next arm
adds two learned gamma tensors to the accepted dual-RMS probe before testing
each norm boundary around the mixed convolution core.
