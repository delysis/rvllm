# Raw-Metal donor intake: Gemma 4 12B

This is a trial plan, not an assertion that the E2B donor's performance
transfers to 12B. Source: `john-rocky/coreai-model-zoo` commit
`a2a664e84ee4807cf4f6441944bd0ac6d047224d`, especially the *application*
`apps/CoreAIChat/Resources/g4msl/` shaders and `Gemma4MetalEngine.swift`.
Pinned shader: <https://github.com/john-rocky/coreai-model-zoo/blob/a2a664e84ee4807cf4f6441944bd0ac6d047224d/apps/CoreAIChat/Resources/g4msl/gemma4_matvec.metal.txt>.
The `conversion/gemma4_raw_metal/msl/` signatures are a different version;
neither set is a drop-in replacement for rvLLM's Gemma 4 12B route.

## Decision for this round

The first donor-inspired arm is `metal-qmv-w4-g32-r4-sg8-k8`. It changes the
decode work assignment (four output rows/SIMD group, eight SIMD groups,
eight adjacent K values/lane) but retains rvLLM's authenticated signed W4,
group-32 FP16 scales and BF16 output. It is **not** the donor's affine group-64
quantizer. The pinned v02 campaign must pass compile, native oracle, and
paired timing in that order; neither a successful compile nor the donor's
historical E2B throughput is a speed result for this arm.

If the v02 operator result is promising, extend the same schedule to the W8
output-projection shapes as an independent candidate, then screen dense real
weights and the production-selected full route. Generated MSL, compiler
output, and dispatch evidence should accompany any arithmetic/occupancy
claim. If the flat-layout schedule is limited by four separated row loads,
trial an *exact packed-word permutation* `[row,word] -> [row/4,word,row%4]`.
Require a bit-for-bit inverse check and account for any duplicate resident
storage and preparation time; do not silently requantize or change scales.

## Subsequent arms, separately gated

1. Fuse low-bit Gate/Up plus activation where the incumbent intermediate
   BF16 rounding is reproduced in registers before GELU and multiplication.
   Check dense-weight numerical output, not only a sparse reference, and time
   the complete projection-plus-activation span against its unfused route.
2. Keep the existing paged attention ABI. The current short global candidate
   has passed page-hole, rollback, speculative-suffix, and newest-owner
   negatives at the operator level; that is **not** full-route selection.
   Compare donor-style cooperative scans to the existing single-group and
   split-plus-merge candidates, including the merge and newest-K/V handling.
   Sliding and global heads need different query-to-KV mappings.
3. Treat M=8 prefill as its own weight-reuse experiment. Compare it at
   256/512/1024/2048-token prompts to existing GEMM/attention controls;
   decode-kernel repetition is not a competitive prefill baseline.
4. Treat GPU-resident token chaining as a route experiment only after the
   component gates. Measure transfers and synchronization as well as kernels;
   preserve EOS, cancellation, committed-prefix, and speculative-KV
   semantics. Metal-only and Metal/ANE routes need separate comparison.

The donor's reported complete-path E2B tokens/s, the analytical fraction of
12B matrix weights in FFN, and an operator-only speedup have three distinct
denominators. None establishes a 12B end-to-end win or an MLX-relative win.
Every promotion still requires the project's sealed queue receipts, exact
dispatch/full-route evidence, correctness, paired sampling, and independent
confirmation. Temperature and competing processes are recorded covariates,
not reasons to wait for a pristine machine.
