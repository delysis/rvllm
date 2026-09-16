# Gemma 4 12B: static FFN fusion and weight compression

Source-only follow-up, 2026-09-14. No hardware execution, builds, product changes, or duplicate decoder qualification. Earlier reports are preserved.

**Prioritized next experiment:** compare the existing static FP16 FFN with a same-graph, per-output-channel **INT8-weight/FP16-activation (W8A16)** variant on one real layer. Keep gate/up separate initially and preserve the exact accepted tanh GELU/down expression. Record compiled artifacts and resident memory as well as warm evaluation latency; a smaller input weight blob is not proof of compressed runtime traffic. A stacked FP16 gate/up convolution is the cheaper structural experiment, but it cannot reduce the dominant weight bytes and the current FFN already has one host dispatch.

The parent reports seven prompts/31 IDs and 24 decode steps passing on the 115-program baseline. Its new M4 Max layer measurements are static FFN **3.333 ms** versus dynamic **54.504 ms**, with identical measured error; dynamic O **1.261/2.270 ms** sliding/global versus static **0.436/0.739 ms**. These are the parent's current local observations, not measurements made by this research or upstream performance claims. Full-model qualification of the 116-program candidate is being performed separately.

## 1. Gate/up fusion: actual source and exact Gemma shape

Current [AneGatedFfn::ffn_mil](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:181) already compiles two constant-weight convolutions, explicit tanh GELU, gate multiplication and down convolution into **one graph and one evaluate call**. Merging the two initial convolutions would change internal scheduling and input reuse, not the host dispatch count.

A concrete M4 precedent exists in maderix/h3.c-ane, pinned `6538b226efe5c4fff905d299dcc4426db2f44de4`: [h3_ane_block.m:605–620](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_block.m#L605) performs one `fc1` projection, then channel-slices gate and up. Its FFN width is 14,336, so the combined projection has 28,672 output channels; [weight construction](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_block.m#L690) confirms that dimension. Its [projection emitter](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_block.m#L259) feeds constexpr-dequantized weights to convolution. The full h3 block uses two external inputs and SiLU; **only this internal projection/slicing pattern transfers**, not its full interface or activation.

Gemma adaptation, still unexecuted:

```text
Wgu = row_concat(Wgate, Wup)         # constant [30720,3840,1,1]
gu = conv(x, Wgu)                   # x:[1,3840,1,1], gu:[1,30720,1,1]
gate = slice_by_size(gu, [0,0,0,0],     [1,15360,1,1])
up   = slice_by_size(gu, [0,15360,0,0], [1,15360,1,1])
y = conv(tanh_GELU(gate) * up, Wdown) # Wdown:[3840,15360,1,1]
```

Use the current MIL `const` begin/size tensors, unit strides/dilations, zero padding, `valid`, groups=1, and one external FP16 input/output. ANEForge [dimension documentation](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/docs/capabilities.md#L43) distinguishes a 65,536 channel limit from smaller spatial/contraction limits. Thus 30,720 output channels is not automatically invalid; this does not prove this complete Gemma graph compiles or improves latency.

Gate/up still contain **235,929,600 FP16 weight bytes**. Their shared logical input is only 7,680 bytes. Fusion may improve tiling or scheduling, but total FFN MACs and its 353,894,400 weight bytes are unchanged. A wider convolution may also change compiler partitioning or intermediate storage. Compare the actual compiled programs and output numerics; do not promise a twofold speedup from two convolutions becoming one.

## 2. Exact compressed MIL encodings

ANEForge pin `caeef8edf13b9ec7a3338826daaa27f99e1663d1` was rechecked against current upstream. Its [weight emitter](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_compile.py#L113) provides concrete encodings. It uses e5rt, whereas rvllm uses `_ANEInMemoryModel`; the encoding is a candidate to validate, not an interface guarantee.

**INT8, one scale per output row.** For gate/up use `N=15360,K=3840`; for down use `N=3840,K=15360`. The following is the emitted form with symbolic dimensions and offsets:

```text
tensor<fp16,[N,K,1,1]> W = constexpr_affine_dequantize()[
  axis=int32(0), name=string("W"),
  quantized_data=tensor<int8,[N,K,1,1]>(BLOBFILE(
    path=string("@model_path/weights/weight.bin"), offset=uint64(Q_OFFSET))),
  scale=tensor<fp16,[N]>(BLOBFILE(
    path=string("@model_path/weights/weight.bin"), offset=uint64(SCALE_OFFSET))),
  zero_point=int8(0)];
```

Feed `W` directly to the unchanged convolution; keep activations FP16. h3's [same-private-API implementation](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/h3_ane_linear.m#L105) uses this form and explicitly notes that scales must be rank-one `[N]`, rather than `[N,1,1,1]`.

ANEForge's [quantizer/blob writer](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_blob.py#L10) uses row scale `max(abs(row))/127`, signed values in −127…127, and FP16 stored scales. Build the reference from the **stored FP16 scale**, not only the higher-precision scale used while quantizing. Handle zero rows and require usable finite scales. Its BLOBFILE dtype codes are FP16=1, INT8=4, UINT4=11, with a 64-byte file header and 64-byte descriptors. These are blob-format codes, not MIL datatype-enum values; the current all-FP16 Rust blob writer cannot simply retain code 1 for every payload.

**Four-bit scalar LUT, one codebook per weight tensor.** For a rank-four convolutional weight, keep rank-four indices and rank-six LUT:

```text
tensor<uint4,[N,K,1,1]> wi = const()[name=string("wi"),
  val=tensor<uint4,[N,K,1,1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"),
    offset=uint64(INDEX_OFFSET)))];
tensor<fp16,[1,1,1,1,16,1]> lut = const()[name=string("lut"),
  val=tensor<fp16,[1,1,1,1,16,1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"),
    offset=uint64(LUT_OFFSET)))];
tensor<fp16,[N,K,1,1]> W = constexpr_lut_to_dense(indices=wi, lut=lut)
  [name=string("W")];
```

Pack two indices per byte, low nibble first; reconstruct as `FP16_codebook[index]`. The [emitter](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_compile.py#L135) deliberately avoids reshaping constexpr outputs, which its path cannot route as weights. This is LUT palettization, not a signed INT4 arithmetic promise. Its `compress_atol` gate is **relative weight reconstruction norm**, not Gemma activation/logit accuracy. Explicit `compress="int4"` can fall back to INT8; inspect emitted MIL to know which format actually ran.

Do not begin with blockwise compression: that same emitter [inserts an add-zero bridge to dense FP16](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_compile.py#L154). A capability-table claim alone does not prove this precise lowering keeps weights compressed at execution.

## 3. What the runtime-compression evidence establishes

| Evidence | What is established | What remains unproved for rvllm |
|---|---|---|
| h3 [M4 report](https://github.com/maderix/h3.c-ane/blob/6538b226efe5c4fff905d299dcc4426db2f44de4/README.md) and source | Author reports full INT8 DiT blocks through `_ANEInMemoryModel`, roughly 19 GB compiled cache for 50 blocks/roughly 19B parameters, and INT8 remaining compressed; actual constexpr/conv encoding is inspectable | No matched FP16/INT8 Gemma FFN trace, exact artifact inventory, or proof of rvllm's post-load representation |
| ANEForge [compression benchmark JSON](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/bench/results/compress_speedup_bench.json) | For random 4096×4096, B=1, reported FP16/INT8/LUT4 medians 0.309/0.309/0.195 ms; source weight files 33,554,560/16,785,600/8,388,832 bytes | JSON lacks hardware/OS metadata. These are not M4 measurements or compiled-artifact sizes |
| Author's [compression analysis](https://github.com/sbryngelson/ane-guide/blob/9eb1afe95cdc766d7bec4d1d02d4918ef6268ece/part-2-reaching/07-weights-and-compression.md#L69) | Reports INT8 folding on M1, native INT8 streaming on M2, and 1.6–1.8× gains for compressed forms on M5; distinguishes measured family evidence from inferred intermediate-family gates | Does not establish our M4 Max/macOS 15.6 graph behavior |
| Apple's [palettization guide](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html) | Latency benefit requires just-in-time decompression; supported ANE paths can benefit; additional LUT groups can cost speed | Does not guarantee private-MIL lowering or Gemma quality |

The ANEForge [benchmark source](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/bench/compress_speedup_bench.py#L17) explicitly measures `build_dir/weights.bin`, with 30 warmups and median-of-five blocks of 200 calls. LUT4 uses a loose `compress_atol=0.5`. It measures calls including host I/O and does not audit the compiled weight stream. Its width-dependent LUT speedup is useful evidence of a runtime effect on the tested machine, but neither a DMA counter nor an M4 identification.

The same author gives stronger representation evidence for **M1 sparse weights**, not M4 INT8/LUT4: [compiled program unchanged at 2,416 bytes while the weight payload changes](https://github.com/sbryngelson/ane-guide/blob/9eb1afe95cdc766d7bec4d1d02d4918ef6268ece/part-7-toolchain/25-compression-internals.md#L199), alongside improved bandwidth. That suggests the appropriate forensic comparison. Treat its stream-gate disassembly account as the author's analysis, not independently verified hardware behavior here.

No inspected primary artifact gives an exact, audited **compiled M4 Gemma FFN** size. Calculated source payloads for our three matrices, excluding headers/alignment, are:

| Storage | One FFN | All 48 FFNs |
|---|---:|---:|
| FP16 | 353,894,400 B | 16,986,931,200 B |
| INT8 + FP16 per-output scales | 177,016,320 B | 8,496,783,360 B |
| LUT4 + three 32-byte codebooks | 88,473,696 B | 4,246,737,408 B |

These are prospective payload savings, not compiled/resident bytes or predicted speedups. A small artifact may still expand at load; a warm speedup alone may come from changed arithmetic. Capture source, compiled `data`/weight sections, model-load resident/wired-memory deltas, and warm evaluation separately. Compare the compressed graph with a dense FP16 graph containing the **same dequantized weights**, so quantization-induced zeros and coefficient changes do not confound the format comparison. Record internal program changes if observable. Without direct traffic evidence, describe the result as “consistent with compressed runtime bandwidth,” not a proven DMA mechanism.

## 4. Gemma numerical and experiment order

1. **INT8 W8A16 encoding probe, then one real FFN.** Preserve the current separate gate/up convolutions and exact tanh expression; compare original FP16, compressed INT8, and dense dequantized INT8 weights. Keep one input/output and the same 116-program count. Quantization may fail existing error gates; do not relax them silently.
2. **Stacked FP16 gate/up control.** Use the 30,720-channel shape above. This is the cheapest fusion change and can run as an independent comparison, but accept it only for measured benefit. Per-row INT8 scales concatenate naturally if fusion later wins.
3. **LUT4 after INT8 representation is understood.** Preserve independent gate/up/down codebooks first. One codebook after stacking gate/up changes the quantization problem and confounds fusion with quality. Grouped palettes may improve accuracy but require a separately validated encoding and performance comparison.

Quantization error enters both `GELU(gate)` and `up`, then their product, then the down accumulation. Weight cosine alone is insufficient: inspect gate/up errors and ranges, post-GELU product, down output, post-FFN norm/residual, logits and top-token margin. Preserve Gemma's `0.5*g*(1+tanh(sqrt(2/pi)*(g+0.044715*g^3)))` operation order and accepted FP16 rounding. Do not substitute SiLU or native GELU, and do not scale across GELU using a homogeneity assumption. h3's power-of-two bracket around down projection is a separate linear-stage numerical device, not a transferrable GELU rewrite.

The older maderix [W8A8 benchmark](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/ane_int8_bench.m#L49) quantizes/dequantizes intermediate activations and generates random weights separately from its FP16 baseline. It is a throughput experiment, not W8A16 equivalence or Gemma calibration evidence. Defer activation INT8 until weight-only quality and bandwidth are understood. Leave host norms/RoPE, imported BF16-prefill KV, and the current full-decoder qualification unchanged during these layer experiments.
