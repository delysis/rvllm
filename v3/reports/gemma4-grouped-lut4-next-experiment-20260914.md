# Gemma 4 12B: one grouped-LUT4 experiment

**Closed on 2026-09-15:** the user explicitly ended four-bit work for this
project. The following is historical research, not an active recommendation.
The grouped prototype was removed; CPU-only results are archived in
`gemma4-12b-evidence-20260914/grouped-lut4-host-quality/decision.md`.

2026-09-14. Source-only review; no hardware, builds, cache access, or product changes. Current performance/quality figures below are supplied by the parent qualification: INT8 approximately 5.2 tok/s, FFNs approximately 90 of 188 ms/token; global scalar LUT4 failed full-model quality with 44.6% final-layer error.

**Recommendation: fit scalar LUT4 independently for each 16 consecutive output rows, first for one real FFN, and reject it on captured-input CPU quality before provisioning hardware if improvement is insufficient.** Keep the existing three convolutions, single input/output, FP16 activation boundaries, and explicit tanh GELU. This is a small, reversible format experiment, not evidence that four-bit Gemma quality is solved. Apple recommends grouped-channel sizes 8 or 16 as an accuracy/speed balance, while warning that more codebooks can reduce runtime performance. Its published table is principally A16/iOS17, not our M4/macOS15.6. [Apple performance guidance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)

## What to reuse and change

The current [quantizer](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_lut4_ffn_weights.rs:17) fits **three** independent global codebooks, one per matrix. Its exact-FP16 histogram, deterministic weighted Lloyd iterations, final FP16 centroid rounding, and reassignment against those stored centroids are appropriate. Change the fitting domain and table selection, not those numerical contracts. A group covers `G * columns` contiguous weights in row-major `[output,input,1,1]` storage. Decode coefficient `(r,c)` with `codebooks[r/G][index(r,c)]`; low nibble still precedes high nibble.

For `G=16`, the existing `main<ios18>` [FFN graph](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:280) needs only these constant-layout changes:

| Matrix | Indices | FP16 LUT |
|---|---|---|
| Gate, up | `uint4[15360,3840,1,1]` | `[960,1,1,1,16,1]` |
| Down | `uint4[3840,15360,1,1]` | `[240,1,1,1,16,1]` |

Each `constexpr_lut_to_dense(indices=..., lut=...)` still produces the original FP16 weight shape directly for `conv`. Omit `vector_axis`: the final LUT dimension remains **1**. The output-row group size is not the centroid vector dimension. The public equivalent is `OpPalettizerConfig(mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=16, channel_axis=0, cluster_dim=1, enable_per_channel_scale=False)`. This is an encoding reference, not a proposed Python dependency. [Pinned configuration](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/optimize/coreml/_config.py#L668)

Apple's actual compression pass splits consecutive output channels and constructs precisely this rank-six LUT; its decompressor repeats the appropriate table over each group. The MIL validator requires each index dimension to be divisible by its corresponding leading LUT dimension, and palette length to equal `2^index_bits`. This grouped representation is available from iOS18/macOS15; it does not require a newer application deployment target. [Grouping and reshape](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/optimize/coreml/_quantization_passes.py#L917), [reconstruction](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/optimize/_utils.py#L332), [MIL validation](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py#L302)

## Storage and compiler limits

All three matrices together contain 176,947,200 coefficients and 34,560 output rows. Calculations below retain the current 64-byte blob header, six 64-byte descriptors, FP16 tables, and alignment; these are **source blob sizes**, not measured compiled weights, residency, or traffic.

| Encoding | LUT/scale payload bytes | Complete source blob bytes |
|---|---:|---:|
| Current global LUT4 | 96, plus 96 padding | 88,474,240 |
| Group-16 LUT4 | 69,120 | 88,543,168 |
| Group-8 LUT4 | 138,240 | 88,612,288 |
| Per-row LUT4 | 1,105,920 | 89,579,968 |
| Qualified row INT8 | 69,120 | 177,016,768 |
| Exact-INT8 row LUT8 control | 17,694,720 | 194,642,368 |

Group-16 has 2,160 codebooks versus three today. Source overhead is negligible, but compiled table packing is consequential. A pinned reverse-engineering account describes separate palette, activation-table, output-channel-group, scale, bias, and packed-index sections; palette and activation tables share a target-specific on-chip budget. It distinguishes unsupported M1 multi-codebook packing from later implementations. This is decompilation-derived evidence, **not** a supported Apple compiler ABI or proof of this M4 geometry. No inspected source establishes our macOS15.6 M4 table-size ceiling or efficient lowering for these exact grouped FFNs. Do not reinterpret its internal “vector” terminology as the MIL `cluster_dim`. [Descriptor and budget](https://github.com/sbryngelson/ane-guide/blob/9eb1afe95cdc766d7bec4d1d02d4918ef6268ece/part-7-toolchain/25-compression-internals.md#L32), [generation differences](https://github.com/sbryngelson/ane-guide/blob/9eb1afe95cdc766d7bec4d1d02d4918ef6268ece/part-7-toolchain/25-compression-internals.md#L227)

## Fitting and qualification gates

Independent groups can adapt to different channel distributions. They do not guarantee substantially better quality if distributions are similar. To guarantee no **weight-MSE** regression against today's fit, include the stored global codebook as a candidate for every group, try deterministic group-local fitting, round and reassign, and retain the candidate with lowest error against the original FP16 weights. This is a proposed fitting safeguard. Weight MSE still does not bound `GELU(gate) * up` or accumulated decoder error.

1. **CPU rejection gate:** reconstruct solely from serialized nibbles and stored FP16 group tables. Check row/group boundaries and zero/constant groups. Use the existing captured `--input-fp16` inputs, including the large real activation, and compare original, current INT8, global LUT4, and group-16 dense reconstructions. Report gate, up, gated product, and final output relative L2/max error. Require lower final-output error than rejected global LUT4 on every captured input before spending on a compiled candidate. A result still far above qualified INT8 is insufficient evidence to provision all 48 FFNs.
2. **One-FFN backend gate:** compare compressed group-16 against a dense FP16 graph built from its **exact** reconstructed weights, plus the original control. Preserve the [existing CPU FFN rounding model](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/bin/rvllm_ane_int8_probe.rs:113), backend tolerance `0.01 + 0.02*abs(reference)`, and bit-identity recording. The probe already separates quantization error from backend error and [records these controls](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/bin/rvllm_ane_int8_probe.rs:343). Passing reconstruction accuracy is not passing original-model quality.
3. **Only after quality:** paired resident rotating-order timing against current INT8 and dense reconstruction under comparable load. Keep one-layer artifacts scoped; no full-model cache expansion yet. Previous LUT4's 1.24× over FP16 does not establish a gain over INT8. Promotion subsequently requires the existing seven-case/31-ID/24-step suite and CPU continuation checks using the same BF16-prefill imported-KV boundary, followed by broader quality coverage. Preserve the qualified INT8 fallback.

Per-row LUT4 is a useful later quality ceiling but creates sixteen times as many codebooks as group-16. Do not combine grouping, per-channel scaling, vector centroids, and graph fusion in this first probe: each adds a separate interpretation or lowering question.

## Why defer LUT8 and outlier residuals

**LUT8-exact-INT8 is a clean runtime-format control, not a quality improvement.** For row `r`, map stored signed `q` to unsigned `q+128`; store entry `j` as `f16(f32(j-128) * stored_scale[r].to_f32())`, matching [current reconstruction](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_int8_ffn_weights.rs:59). Index zero is unused because the quantizer clamps to `[-127,127]`; give that unused entry a finite value. Use uint8 indices and LUT `[N,1,1,1,256,1]`. Require byte-identical dense reconstructed weights before comparing affine-INT8 and LUT8 timing. This isolates encoding at unchanged quality, but its many tables and 10% larger source than INT8 may be counterproductive; public dtype legality does not prove efficient private lowering.

**Mixed outlier restoration has more unresolved costs.** Two convolution branches plus addition fit a single-input graph, but introduce separate FP16 projection rounding before tanh GELU; `conv(Wlut,x)+conv(Wres,x)` is not the same numerical control as `conv(Wlut+Wres,x)`. Adding reconstructed constants may instead fold into dense weights. No inspected implementation establishes an additive constexpr residual format that preserves streaming compression here. A one-bit sparse mask alone costs 22,118,400 bytes per FFN, before residual values and metadata. Defer this until grouped LUT4 establishes whether the simpler distribution correction is enough.

Reviewed local SHA256: LUT4 quantizer `e08ddfbea17b53f4e4093fe01a77d60d7c99fd6b4240e6db6e9406f3485a8f92`; probe `abcb4764170c7c954d9ea548b2957def699bdfd157bc65ab4cfa9c12dc6c6835`. Public CoreMLTools source pin: `181cbd341fa394ea53fd6895a2e0a46aaec27cb1`.
