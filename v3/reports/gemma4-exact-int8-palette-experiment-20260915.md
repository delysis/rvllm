# Exact INT8 weight representation experiment

The active optimization scope is eight-bit weights. Four-bit work was explicitly
stopped by the user; the grouped four-bit additions were removed from live code.

The current FFN uses signed INT8 coefficients and one stored FP16 scale per
output row. The candidate keeps those coefficients and scales authoritative,
but serializes each coefficient as the unsigned index `q + 128`, with a 256-entry
FP16 table per row. Entries 1–255 use exactly the current FP32 multiplication and
FP16 rounding; unused entry zero is finite zero. This changes representation,
not precision, fitting, activations, GELU or model quality.

The [Apple MIL operator contract](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py)
permits uint8 indices and a rank-six `[N,1,1,1,256,1]` scalar LUT for
rank-four `[N,K,1,1]` weights. The exact [MILBlob dtype definition](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/mlmodel/src/MILBlob/Blob/BlobDataType.hpp)
assigns UInt8 code 3, distinct from the MIL protobuf enum. These contracts do
not prove this private ANE compiler will accept or accelerate the graph.

## Verified host result

The source byte decoder reads descriptor offsets, unsigned indices and the
serialized row tables. On actual layer-zero gate/up/down weights, all
176,947,200 reconstructed FP16 coefficients are bit-identical to affine INT8.
The reconstruction SHA is
`9ec1b72edfad27068add34669a960d971f6a52cf982db9170e451cafe158bfe5`.
The four CPU reference inputs (three synthetic and one independently captured
real activation, maximum magnitude 171.5) produce identical error reports.
Five INT8 host tests pass, including nonsquare row/table routing, signed index
mapping, zero rows, rounding and existing quantizer/source-blob parity.

Source blob size is 194,642,368 bytes versus 177,016,768 for affine INT8.
This is a 9.96% source increase; compressed residency, device traffic and speed
are not established. The ANE may reject or lower the many row tables poorly.

Evidence: `gemma4-12b-evidence-20260914/exact-int8-palette-host-quality/` contains
the raw result, source, test log, binary SHA and pinned dtype header. The earlier
`-old-binary-preflight/` invocation was rejected during argument parsing and is
excluded. That host-only audit made zero accelerator/compiler calls. The signed HTTP
binary and its 6.09-token/s measured baseline remain unchanged.

## Device qualification and timing status

The 64-by-128 single-input/output smoke test passed three inputs with two
compiler calls, six evaluations and two successful unloads. Maximum difference
from affine INT8 was 0.0001183. This exposed compiler-rounding differences despite
identical reconstructed constants.

The actual layer-zero FFN then passed all four inputs against the dense CPU
reference and dense ANE reconstruction control. Palette outputs matched the dense
ANE control bit for bit, but differed from affine INT8. On the real activation,
relative L2 backend errors versus dense CPU were 0.05148% for affine INT8 and
0.03180% for palette/dense ANE. Synthetic-input errors were 0.0726–0.0862% affine
and 0.2525–0.3074% palette/dense. All passed the existing absolute/relative gate.
No full-model continuation quality is inferred from this one layer.

The durable journal records one existing affine cache hit, exactly two new
compiles, three loads, 12 completed evaluations and three successful unloads.
All caller staging paths were absent afterward and boot identity was unchanged.
Evidence: `exact-int8-palette-smoke/` and
`exact-int8-palette-real-qualification/` in the evidence directory.
The signed real-FFN probe SHA was
`593d88d28d0aaad4df3b548337a1509e3ecb50eddab2556201ba43cdf3552c41`.

An unjournaled strict-cache process then ran 12 resident ABBA/BAAB trials,
128 evaluations per trial, with zero compilation. **All six pairs were rejected**
because sampled thermal state was Fair throughout (battery, low-power off,
power mode zero). Raw median call durations were 1.684 ms affine and 7.289 ms
palette. These unfavorable observations do not constitute a qualified slowdown
ratio or speedup. No candidate was promoted. A later idle counter probe still
reported Fair thermal state; no repeat accelerator run was started.
Evidence: `exact-int8-palette-real-abba/`.

The probe source now checks power eligibility before warmup/timing and stops
collecting timing trials when eligibility is lost. That additional guard passes
compilation; the preceding frozen binary used post-measurement rejection.
The baseline full-model INT8 path remains unchanged.

## Experimental command contract

The new path requires `--mode int8 --int8-lut-storage true --compare true`
and explicit `--cache require` or `--cache reuse`. It permits at most two compiler
calls via `--ane-compile-budget 0..=2`; the existing affine model always requires
its cache. A diagnostic journal selects correctness/lifecycle qualification and
suppresses timing. Without that journal, eligible power conditions enable the
resident paired timing. This surface is an experiment, not a promoted decoder.

## Criteria before any further expansion

Use the same real layer-zero FFN and qualified single input/output three-conv
graph. Compare affine INT8 with palette-encoded INT8 and exact dense FP16
reconstruction. First validate outputs on the captured input, separating backend
error from the unchanged quantization error. Limit any new compilation to the
candidate and necessary one-layer controls; preserve the full model cache.

Only if numerics pass, run repeated resident ABBA measurements with power-mode,
thermal and actual process CPU counters. Disable durable per-evaluation logging
for timing; qualify driver cleanup separately. A speedup must beat affine INT8
under matching eligible controls before any full-model provisioning. A smaller
or larger serialized blob is not performance evidence. No four-bit variant is
part of this experiment.
