# Corrected fused ANE attention plus output projection: component oracle

## Result

The corrected layer-0 sliding-attention plus real Gemma 4 `o_proj` graph passed its bounded component oracle on the M4 Max. This qualifies the exact component identity for comparative timing; it does not qualify production routing, full-model correctness, checkpoint quality, or promotion.

- MIL SHA-256: `cbc3acbf8ac20506c3592997b76aa0f0695d39a598bd93ae15326988a8840317`
- Weight blob SHA-256: `68a3a71d77dba7eeefe425c56cc93695f718ddf3530150395a19c5bae859b160`
- One external input and one external output
- Twelve accelerator evaluations
- Zero compiler calls in the successful oracle run; the exact graph was a cache hit
- Zero numerical, repeatability, or guard violations
- Repeated results were bit-identical for every case
- Maximum absolute error over all 3,840 outputs: `0.0014337189495563507`

## Coverage

Token counts `1`, `31`, `32`, `33`, `1024`, and `1025` cover the first-token case, both sides of the 32-lane boundary, the full sliding window, and physical ring wrap. Each token used the same per-KV-head value vector, making the attention result independent of query/key scores. The independent FP32 reference duplicated that vector into the correct query-head order and applied the actual checkpoint projection matrix. This specifically exposes the head/group flattening error found in the rejected prototype.

Every case ran twice through one persistent request. Output bits matched exactly across repetitions, and 64-byte sentinel regions surrounding the packed input remained unchanged.

## Evidence boundary

The first oracle wrapper produced numerically passing evidence but exited nonzero because it required exactly one compiler call even on a legitimate cache hit. That failed receipt remains preserved. The fresh-ID v2 job changed only this gate to `compiler_calls <= 1` and succeeded through the experiment queue.

The next admissible step is an alternating comparison of the complete separate attention-plus-output pair against this fused component, including input packing, both baseline accelerator evaluations, the fused evaluation, output reads, and matching work counts. Only a stable timing win can justify full-route integration work.
