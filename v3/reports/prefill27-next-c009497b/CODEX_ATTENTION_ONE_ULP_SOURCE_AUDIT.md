# Source audit of the retained long-context attention differences

This is a source-only follow-up to the immutable M=1024 and M=2048
component screens. It adds **no device result** and does not identify the
first differing arithmetic instruction.

The old Q4K16 kernel calls `pr26_attention` in
`research_shaders/prefill_attention_common.metal`. Its QK score uses a
per-lane FP32 partial sum over D/32 terms followed by `simd_sum`
(lines 71–75). It updates its running softmax over 16-key tiles
(lines 82–89), then accumulates the weighted V values with scalar FP32
additions (lines 99–106). The new MMA8K32 kernel calls `pr27_attention` in
`research_shaders/prefill27_attention.metal`. It forms QK scores with
`simdgroup_multiply_accumulate` over eight-wide dimension steps
(lines 61–76), updates softmax over 32-key tiles (lines 90–98), and
accumulates P×V with matrix instructions (lines 111–131). Both normalize
their FP32 output and convert to half at the end.

These source differences provide several plausible rounding locations:
QK reduction, tile-wise max/denominator updates, and P×V accumulation.
The retained BF16 output files show the first **observed stored** differences:
one adjacent encoding at 128 elements of M=1024 holes, and at 512/512/256
elements of M=2048 structured/newest/holes respectively. They do not
record the intermediate FP32 scores, probabilities, denominators or
accumulators. Thus the output localization cannot select among the
plausible locations or prove a defect. Both kernels passed the retained
independent FP64-oracle gate, and the packet explicitly did not promise
cross-arm bitwise equivalence for this arithmetic-changing candidate.

The synthetic screen inputs retain Q/K/V and block-table bytes locally,
so a future read-only host analysis can check logical input identity and
compute a separate high-precision reference. It still cannot infer the
hardware matrix instruction's internal FP32 reduction order from source
alone. A first-device-arithmetic comparison would require a separate,
prospectively declared, default-off instrumented experiment with fresh
immutable queue IDs and outputs; instrumentation itself must be treated as
a changed route. Do not replay these completed screens or use the rejected
timing trials as acceptance evidence.
