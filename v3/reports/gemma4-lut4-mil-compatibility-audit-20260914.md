# Gemma FFN LUT4: MIL compatibility and fitting audit

Source-only audit, 2026-09-14. No builds, hardware execution, cache access or product changes. The immediate experiment remains one real FFN, separate gate/up/down projections, unchanged tanh GELU, and an exact dense reconstruction control.

**The proposed signature matches rvllm's existing opset.** It requires the iOS18/macOS15 operator revision, not Core ML Tools 9.0. One implementation correction matters: **align every BLOBFILE descriptor and payload to 64 bytes, including the entry after a 32-byte codebook.** ANEForge's simplistic concatenating blob writer should not be copied verbatim here.

## Deployment target and exact signature

Apple registers this revision of `constexpr_lut_to_dense` at `_IOS18_TARGET`; its deployment enum explicitly aliases [macOS15 and iOS18](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/converters/mil/_deployment_compatibility.py#L19). It is already present in the [Core ML Tools 8.0 release source](https://github.com/apple/coremltools/blob/7b1337140c44f3fbc0c48edaa677f4c8ecca9dad/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py#L165). Therefore the existing [rvllm FFN declaration](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple/src/ane_linear.rs:262), `program(1.3)` with `func main<ios18>`, has the appropriate declared target. Its `coremltools-version:9.0` build-info string is not the operator's minimum deployment requirement; changing metadata is unnecessary for this probe. This establishes format compatibility, not proof that the installed private compiler routes the entire graph to ANE.

Apple's [validator and type inference](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py#L290) establish:

| Item | Immediate probe |
|---|---|
| Gate/up indices | `tensor<uint4,[15360,3840,1,1]>` |
| Down indices | `tensor<uint4,[3840,15360,1,1]>` |
| Each independent LUT | `tensor<fp16,[1,1,1,1,16,1]>` |
| Operation | `constexpr_lut_to_dense(indices=wi, lut=lut)` |
| Result | FP16, same shape as its indices |
| `vector_axis` | Omit: final LUT dimension is 1, meaning scalar palettization |

The LUT rank must be indices rank + 2. The penultimate dimension must match `2^index_bitwidth`; 16 therefore requires `uint4`, not `uint8` or signed `int4`. Every indices dimension must be divisible by the corresponding leading LUT dimension. The prose near the top of Apple's definition reverses this divisibility wording; the executable validator and examples settle it. All leading LUT dimensions of 1 satisfy the condition.

Do not mix revisions: the [iOS16 operator](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/converters/mil/mil/ops/defs/iOS16/constexpr_ops.py#L175) uses a flat packed **uint8** vector, a one-dimensional LUT, and explicit **uint32 shape** input. Apple's [conversion utility](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/optimize/_utils.py#L668) unpacks that older logical representation before creating the new ND uint4 tensor. The new tensor is still packed when serialized; declare logical weight dimensions, not the packed byte count.

ANEForge's [pinned emitter](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_compile.py#L135) uses precisely this ND/scalar layout and feeds the reconstructed constant directly to convolution. Avoid reshaping the constexpr output or adding a runtime multiply/add bridge: those change routing. Its [program writer](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_compile.py#L883) also emits `program(1.3)`/`main<ios18>`, although through its distinct e5rt path. It can reject a LUT fit and fall back to INT8; rvllm's explicit experiment must report LUT4 or fail, rather than silently substitute a format.

## Blob details that can invalidate an otherwise correct graph

Apple's [8.0 blob enum](https://github.com/apple/coremltools/blob/7b1337140c44f3fbc0c48edaa677f4c8ecca9dad/mlmodel/src/MILBlob/Blob/BlobDataType.hpp#L16) confirms `Float16=1`, `UInt4=11`. These are BLOBFILE storage codes, not MIL datatype enum values. Apple's [storage format](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/mlmodel/src/MILBlob/Blob/StorageFormat.hpp#L16) and [writer](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/mlmodel/src/MILBlob/Blob/StorageWriter.cpp#L103) specify 64-byte headers/descriptors and aligned descriptor/payload starts. `BLOBFILE(offset=...)` addresses the descriptor; its internal offset addresses the payload. Keep `sizeInBytes` equal to actual payload length, excluding alignment padding.

For each matrix, 58,982,400 logical indices occupy **29,491,200 bytes**; its 16 FP16 entries occupy **32 bytes**. Pad before the next descriptor after a codebook. The metadata's `padding_size_in_bits` at byte offset 24 is zero for these even counts; a generic odd uint4 count requires four unused bits. This is distinct from padding between blob entries.

Packing is row-major, least-significant nibble first: `packed[j] = index[2j] | (index[2j+1] << 4)`. Apple provides [little-endian bit packing](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/optimize/_utils.py#L756); ANEForge uses the equivalent [nibble expression](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_blob.py#L28). A host fixture with indices `[0,1,14,15]` must encode `[0x10,0xfe]`. Reconstruct every control coefficient from the exact stored FP16 codebook bytes and decoded index, never from an unrounded fitting centroid.

## Fitting guidance within the existing probe

Apple's [algorithm guide](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-algos.html) describes ordinary k-means as weight-only fitting and warns that low-bit quality can degrade substantially; grouped-channel fitting can recover accuracy. Its [configuration](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/optimize/coreml/_config.py#L668) distinguishes per-tensor, grouped-channel, vector palettes, and per-channel normalization. Those are different algorithms/encodings, not equivalent names for LUT4.

For now, fit gate, up and down separately with scalar 16-centroid k-means. Use deterministic initialization, finite-value checks, defined empty-cluster/tie behavior, and wide accumulators. After converting centroids to FP16, assign the final indices against those stored centroids. Preserve all 16 slots even if centroids coincide. Do not add per-row normalization unless its inverse is represented in the graph; that would widen the immediate experiment.

An implementation suggestion derived from the one-dimensional objective: histogram the finite FP16 coefficient values with occurrence counts, then perform weighted Lloyd updates on that histogram. This represents the full tensor's squared-error objective exactly while avoiding an `elements × 16` distance matrix. It is not a claim that this optimizer finds the global optimum or optimizes Gemma output quality. Keep reported weight error separate from backend error against dense reconstruction and activation error against original weights; the accepted tanh GELU remains unchanged.

For a later grouped-channel experiment only, `G` output rows per group yields LUT shape `[N/G,1,1,1,16,1]`, with `G` dividing `N`. Thus G=32 would give 480 codebooks for gate/up and 120 for down. More codebooks may improve weight fit, but Apple's [performance guidance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html) warns of runtime tradeoffs. Do not introduce grouping or sensitivity calibration before the immediate per-tensor format and dense-control comparison is complete.

**Confidence:** high for target/signature/blob requirements; moderate for private-path routing based on external emitter precedent; unproved for current macOS15.6 ANE lowering, speed, or Gemma quality until the parent's bounded numerical probe completes.
