# BF16 12B-it prefill mechanism audit

This is a source-and-dispatch explanation of a prospective full-route gain,
not a machine-code, per-kernel GPU-time, numerical-quality, or promotion
finding. The observations use original `google/gemma-4-12B-it` BF16 weights,
two prompts of 101 and 304 tokens, and 64 generated tokens each. The paired
ABBA/BAAB route job and same-prompt MLX jobs are queued separately.

## What the current receipts establish

The normal route, projection-only route, attention-only route, and combined
route all emitted the same 128 generated token IDs across the two prompts.
Every selectable route recorded its intended per-case dispatch counts and
zero inside-case library/pipeline-state compiles. The combined route recorded
48 tiled GEMMs, 48 tiled QKV projections, 96 FP32 raw projections plus 96
normalization calls, and 40 D256 plus eight D512 tiled attention calls per
case. This rules out a fallback-only explanation for the observed gain. It
does **not** establish internal tensor/logit agreement or independent quality.

The complete separate-process repeat profiles showed these unselected
prefill samples (milliseconds):

| Arm | M101: all three samples | M304: all three samples |
| --- | --- | --- |
| Control A | 5885.212, 6070.294, 6221.680 | 12243.136, 12245.789, 12248.446 |
| Projection-only | 1781.855, 2496.823, 2264.970 | 1195.633, 1190.606, 1194.070 |
| Control B | 6042.635, 6317.738, 6103.494 | 12241.509, 12262.068, 12467.582 |
| Combined | 1851.792, 1874.072, 1694.209 | 640.614, 657.418, 641.769 |
| Control C | 6208.468, 5997.220, 6061.356 | 12284.107, 12273.125, 12313.707 |

All five jobs were queue-eligible in sampled AC/power-mode-2/thermal-0
conditions, with no violations. The M101 control-A, projection, and combined
series fail the 5% within-arm drift rule. The M304 profiles pass that drift
rule and descriptively favor projection by 10.257× and combined by 19.132×
against their bracketed controls. These profiles are **not interleaved** and
cannot be converted into strict speed verdicts by selecting their stable
M304 rows. Within-job ABBA/BAAB remains necessary. Full-case counters also
include decode: both arms used 65
command buffers/case, while the projection route used 144 more encoders than
control. The gain is therefore not evidenced as fewer command buffers or
fewer kernel launches. No named normal-route per-kernel GPU-time receipt is
available from these profiles.

## Source-level explanation, with uncertainty

The default generated MSL's `gemm_rmsnorm_f16` assigns one threadgroup to a
row and loops over all K for each output column, accumulating in FP32 before
RMSNorm. Its exact source is `crates/rvllm-apple-metal/src/kernels.rs` and the
locally exported `normal.metal` SHA-256 is
`eed915e3ce27236d341d5e1785e540910c76092d848c1c0014c742fba6b1329b`.
The source structure exposes little reuse of the activation row across output
columns or of the weight tile across input rows; it is a credible structural
bottleneck, consistent with prior ordinary-dispatch ledgers and the large
projection-only full-route gain. This is a source-level inference, not a
measured allocation of GPU time to that particular function.

The candidate's `prefill_projection_common.metal` tiles 32 rows × 64 outputs
with 128 threads. It stages BF16 operands in 8 KiB shared memory, uses
`simdgroup_multiply_accumulate` with FP32 accumulators, requests the next K32
panel before finishing the current one, and reuses the same shared allocation
for operand and FP32 output lifetimes. The source explicitly says the
lookahead values are *not* evidence of physical register residency or real
overlap; only generated ISA/resource data could establish those. The locally
exported combined MSL SHA-256 is
`084b1bc3d7828aabab7fb1aef28925a7567876780289b860831613e23c84cede`.
The M256/M512 isolated down-norm tests showed a 1.25×/1.34× advantage for
lookahead versus load4 control, but that narrower result should not be
confused with the much larger full-route gain over scalar-K fused projection.

For the fused default projection-plus-RMSNorm role, the selected route writes
an FP32 raw projection and then normalizes into BF16. That preserves the
default's *unrounded-FP32-before-normalization* boundary; it adds 96 raw
projection and 96 norm dispatches per case instead of claiming a BF16-rounded
intermediate is equivalent. Exact generated IDs alone do not prove these
intermediate tensors match. The separate normalization and larger scratch
budget are costs to measure, not free optimizations.

The candidate's `prefill_attention_common.metal` assigns four causal queries
to a threadgroup, one per SIMD group, scans 16 keys per tile, maintains an
online FP32 softmax, and reuses one K/V shared panel with explicit barriers.
At M256/M512 its complete isolated attention operator was about 7.27×/6.04×
faster than the scalar control under the referee's drift gate. In a full route,
Q4K16-only was descriptively only slightly faster than default, while the
combined route at M304 appeared faster than projection-only. That pattern is
consistent with attention becoming more visible after the projection
bottleneck is removed; process order and cache effects preclude assigning an
incremental causal speedup yet. M101 does not show the same clear incremental
attention signal.

The compiler admitted these generated MSL sources and native component
oracles passed, but no GPU ISA/register/spill capture or normal-route
per-function GPU timing has been validated. Do not infer achieved memory
coalescing, register residency, pipeline overlap, or occupancy from the
source alone. The next decision should use the complete five-arm referee,
counterbalanced full-route timing, exact-token MLX comparison with labeled
timing boundaries, and route-preserving tensor/logit/reference checks.
