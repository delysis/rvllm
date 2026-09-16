# Gemma 4 12B: the next ANE bandwidth experiment

Source-only follow-up, 2026-09-14. No hardware execution, builds, cache access, or implementation changes. The parent is separately qualifying the 162-program all-static cached decoder; this report does not claim that qualification has finished.

**Recommendation: test native LUT4 weight storage on one real FFN, with the existing three-convolution graph and a dense reconstruction control.** This offers a plausible reduction in the dominant weight traffic. Spatial replication, stacked gate/up, and additional block fusion may improve scheduling, but none inherently reduces the 353,894,400 FP16 source-weight bytes in each FFN.

## What the bandwidth numbers actually mean

The parent's M4 Max component medians of 3.3–4.2 ms correspond to **84.3–107.2 GB/s of effective FP16 source weights**, excluding headers. This is neither a physical DMA measurement nor an established hardware ceiling. The INT8 result—3.671 ms against 4.246 ms original and 4.324 ms dense reconstruction—is a modest runtime benefit, not evidence of a twofold bandwidth improvement. It also adds quantization error: 1.183% output relative L2 against original weights, versus 0.276% FP16 backend error. These are local observations supplied by the parent, not measurements performed here.

Useful primary comparisons have materially different boundaries:

| Source and exact workload | Reported measurement | Interpretation |
|---|---|---|
| [ANEForge M4 MacBook Air, 16 GB, macOS 26.2](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/bench/results/rooflines/roofline-apple-m4-Mac16_12-d4109cd41a75-6779bad2.json): random FP16 4096×4096 GEMV, batch 1 | 1.448 ms; approximately 23.18 GB/s of source weights. Activation-stream result: 7.67 GB/s | Lower than our current FFN; cannot be used as an M4 ceiling. Recorded executable source revision is `23ef10a6c370b5a5aa9099429d8686e60ada0533`. |
| [ANEForge M5 Pro MacBook Pro, 48 GB, macOS 26.5.1](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/bench/results/rooflines/roofline-apple-m5-pro-Mac17_8-c38210cfc8ea-a154b89e.json): same GEMV dimensions/precision | 0.3003 ms; approximately 111.72 GB/s of source weights. Activation-stream result: 24.00 GB/s | The often cited 24 GB/s measures activation streaming, not a universal weight-transfer limit. Recorded source is dirty `55a1c6afdad3e1f33cc76eed19a9d55ae3c3344f`; this is M5 Pro, not M4 Max. |
| [M3 Max field-guide dispatch probe](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/83e73eb94051dc67431829d0f3d3eb67db0b8e9b/tests/test_dispatch_scaling.cpp#L241), macOS 26.3.1; constant 1×1 convolution, square dimensions through 3072, logical width 32 | Author's fit: approximately 119 μs + bytes / 78 GB/s; bytes include weights and both surfaces | Useful evidence that fixed dispatch and streaming are separate costs; not an M4 measurement. |

ANEForge's [measurement code](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/bench/roofline_analysis.py#L296) times full `net(x)` calls using the minimum of 20 repetitions after five warmups. Its [linear API emits matmul](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/graph.py#L187); this is not rvllm's private-API three-convolution FFN. No inspected primary result establishes a physical M4 Max ANE weight-bandwidth ceiling for this graph.

## Which changes could help

**Spatial padding and replication:** rvllm already uses physical width-32 storage for a logical `[1,3840,1,1]` input in [AneGatedFfn](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:217). Changing the MIL tensor to width 32 is a different experiment from satisfying its physical row stride. Repeating the same token cannot amortize weights over 32 useful results; independent requests or speculative candidates can, but that targets aggregate throughput or requires an effective drafter. The [W-lane probe](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/83e73eb94051dc67431829d0f3d3eb67db0b8e9b/tests/test_w_lane_batching.cpp#L37) explicitly packs four independent samples. Its M3 Max [spatial-convolution investigation](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/83e73eb94051dc67431829d0f3d3eb67db0b8e9b/tests/README.md#L85) reports a legal kW=15 packed matvec at 0.547 ms versus 0.318 ms for baseline 1×1 convolution. This is a caution, not an M4 prohibition.

The newer M4 author's [geometry account](https://maderix.github.io/articles/inside-the-m4-ane-part-4/#:~:text=Geometry%20matters) reports gains from moving a **64-channel, large-spatial** convolution chain into 1024 channels using SpaceToDepth. Our 3840/15360-channel, one-token FFN has the opposite geometry. That report does not demonstrate a semantics-preserving Gemma matvec speedup; its exact geometry-probe code was not located in this bounded review.

**Stacked gate/up:** [h3's pinned implementation](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_block.m#L605) gives a concrete single-projection/channel-slice pattern. Gemma would use `Wgu[30720,3840,1,1]`, split output channels at 15360, then preserve the existing tanh GELU and down projection. This saves neither source-weight bytes nor a host dispatch: the [current FFN](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:262) already has one graph/evaluation. It may alter tiling or intermediate scheduling, so it is a cheap secondary experiment. h3's whole block has two external inputs and SiLU; that interface and activation do not transfer.

**Whole-block fusion:** potential benefits are fewer submissions and intermediate stores. It also moves Gemma's precision-sensitive norms and residual boundaries. It cannot retain 354 MB of FFN weights across autoregressive tokens merely by fusing the graph. The M4 author reports an approximately 90 μs dispatch floor; even eliminating 48 such submissions saves only 4.32 ms under that external model, versus roughly 158–202 ms for 48 current FFNs. This calculation is illustrative, not our measured dispatch cost. [Source](https://maderix.github.io/articles/inside-the-m4-ane-part-4/#:~:text=Each%20evaluation%20adds)

**LUT4:** there is primary M4-family performance evidence beyond M5 extrapolation. A [pinned ResNet experiment](https://gist.github.com/dessatel/92fd9f1e754a184c27c0d240fbbaa889/aa895244d057c53291f0f527718cbdbd1a60c64c#file-resnet50-m1-m4-md) reports M4 iPad Pro 16 GB, iPadOS 18.1 beta, Core ML Tools 8.0b1/Xcode 16b4, batch 1: LUT4(FP16) prediction 0.90 versus FP16 1.23, and longest convolution 60 versus 114. The table omits units, so use ratios: **1.37× and 1.90×**, respectively. The accompanying code uses a randomly initialized ResNet and uniform four-bit palettization. It supplies neither Gemma quality evidence nor an audited lowered-weight representation, and uses public Core ML rather than our private API.

Apple [documents](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html) that runtime benefit depends on just-in-time decompression; additional codebook groups can cost performance. Smaller source blobs do not prove that path. The parent's byte-identical MIL/staging investigation specifically rules out treating owned `net.plist`/`data` sizes as lowered-program measurements.

## One bounded next experiment

Use **one real layer, three independent scalar LUT4 codebooks, unchanged input/output and activation graph**. Adapt ANEForge's pinned [MIL emitter](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_compile.py#L135) and [blob encoding](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_blob.py#L28):

```text
indices: tensor<uint4,[N,K,1,1]>       # gate/up: N=15360,K=3840
lut:     tensor<fp16,[1,1,1,1,16,1]>  # down:    N=3840,K=15360
W = constexpr_lut_to_dense(indices=indices, lut=lut)
y = conv(x=x, weight=W, ...)         # same convolution attributes
```

Use BLOBFILE dtype code 11 for indices; pack `index[2j] | index[2j+1]<<4`. Store the 16 centroid values as FP16; feed the reconstructed rank-four constant directly to convolution. Avoid a reshape/add-zero bridge. The calculated source payload is **88,473,696 bytes** including three codebooks, excluding headers—approximately one quarter of FP16, not a predicted compiled size. ANEForge can silently fall back from LUT4 to INT8 when its weight-error gate fails; an explicit probe must identify the emitted format rather than accept a fallback result.

Compare original FP16, LUT4, and **dense FP16 reconstructed from those exact stored codebooks and indices**. Keep the accepted tanh-GELU operation order; inspect backend error against reconstruction separately from quantization error against original. Check gate/up, gated product, down output and relevant activation ranges using real imported-boundary activations. Do not promote based on weight cosine or one token. Interleave bounded warm timing samples under comparable load; require a repeatable improvement over the reconstruction control before expanding to model-level quality work. Do not add spatial replication or gate/up stacking in this probe.

Without actual DMA/representation evidence, a warm speedup establishes a runtime advantage **consistent with compressed weight traffic**, not the location of decompression. Preserve source, staging, and daemon-cache size distinctions and existing model identities; this proposal requires no new cache inspection.

The full-model upside remains limited by the rest of decode. As an illustration, if FFNs total 168 ms/token, a hypothetical fourfold FFN speedup saves 126 ms: a 2 tok/s baseline becomes 2.67 tok/s; a 3 tok/s baseline becomes 4.82 tok/s. These are Amdahl calculations, not forecasts. A meaningful LUT4 result could beat the present range; dispatch fusion alone is unlikely to do so materially.
