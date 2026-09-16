# Bounded 32×64 Metal prefill matrix experiment

2026-09-14. Source-only addendum for M4 Max/macOS 15, BF16 Gemma 4 12B prefill feeding ANE decode. No builds, hardware operations, cache access, or implementation edits. The parent reports the SIMD-attention continuation gate passed; its full-prefill timing attribution remains underway.

**First compare the qualified 32×32×32 kernel with a native-BF16 32×64×32 kernel using the same scalar loaders and output staging.** Then, optionally, change only the candidate's operand staging/matrix dtype to FP32. Keep vector loading, padding, and direct output stores separate so a result identifies a useful change.

## Exact initial design

The current [tile helper](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:509) uses row-major activations `A[M,K]`, weights `B[N,K]`, four SIMD groups, four FP32 accumulator fragments/group, BF16 operand tiles, and two threadgroup barriers per K32 iteration. Its [two output wrappers](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:547) preserve distinct FP32 and BF16 output ABIs.

For the standalone candidate:

- Launch `ceil(M/32) × ceil(N/64)` threadgroups, each `(128,1,1)`. Define `mr=group.x*32`, `nc=group.y*64`, `sm=(sg/2)*16`, `sn=(sg%2)*32`.
- Stage `bfloat at[32*32]`, `bt[64*32]`; retain B in `[output_channel,K]` order. Independently loop A indices `<1024` and B indices `<2048`, starting at `tid`, stepping by 128. Convert index to `(row,index%32)`; guard device reads with the corresponding M/N bound and `kb+k<K`, otherwise write zero. No thread returns before a barrier.
- Each group owns 16×32 outputs: eight `simdgroup_float8x8` accumulators `c[i][j]`, `i=0..1`, `j=0..3`. For each `kk=0,8,16,24`, load two native `simdgroup_matrix<bfloat,8,8>` A fragments at `(sm+8*i)*32+kk`, and four B fragments at `(sn+8*j)*32+kk`, **B loads transposed**, stride 32. Perform all eight `c[i][j] += a[i]*b[j]` updates. Preserve the barrier after staging and after all groups finish reading.
- Store fragments to `threadgroup float ct[32*64]` at `(sm+8*i)*64+sn+8*j`, stride 64. Barrier; distribute indices `<2048` across 128 threads. Store only `mr+index/64<M && nc+index%64<N`, with output address `row*N+col`. Preserve current alpha/beta evaluation and do not read prior C when beta is zero.
- QKV output stays FP32 until head RMS; gate/up rounds to BF16 before GELU; O/down round to BF16 before their existing separate RMS. Do not add an activation or normalization fusion to this experiment.

This halves groups along N and doubles activation reuse across columns. Across two former tiles, staged operand elements decrease from 4096 to 3072, a 25% reduction; **weight reuse across prompt rows is unchanged**, and physical DRAM savings depend on caches. Matrix arithmetic count is unchanged. These are geometry calculations, not speed predictions.

## Operand controls and resource budget

| Variant | Operand staging / matrix fragments | Output staging | Static shared bytes |
|---|---|---|---:|
| Current 32×32 | BF16 / BF16, stride 32 | FP32 32×32 | 8,192 |
| Candidate 32×64 | BF16 / BF16, stride 32 | FP32 32×64 | 14,336 |
| Optional dtype control | FP32 / FP32, stride 32 | FP32 32×64 | 20,480 |
| Later padding control | BF16 / BF16, stride 40 | FP32 32×64 | 15,872 |

For the FP32 control, **device buffers remain BF16**: stage `float(A[...])`/`float(B[...])`, use `simdgroup_float8x8` for A/B and retain FP32 C. BF16→FP32 is lossless for finite values; no FP16 conversion belongs anywhere. This compares complete operand paths, including staging bandwidth and occupancy; it does not isolate matrix-instruction latency. An additional 32×32 FP32 variant is only needed if separating geometry/dtype interaction becomes useful.

Eight accumulator fragments logically hold 16 FP32 values/lane, versus eight in the current tile. Six operand fragments add 12 logical operand values/lane when simultaneously live. Actual registers, packing, spills, and resident groups are compiler/hardware results, not established by these counts. Record pipeline static shared memory and limits with the component receipt; do not infer occupancy from source alone.

## What pinned upstream code establishes

**llama.cpp `7cf1c54a96d4e950ffa614b94babf762803a8de7`:** its non-tensor matrix path uses 64 output channels ×32 tokens ×32 K, eight FP32 accumulators/group, and native BF16 operand fragments for BF16 weights. It rearranges staging into contiguous 8×8 subtiles, uses vector loads, directly stores complete FP32 output tiles, and reuses shared storage for bounded tail stores. Its activation input is FP32 converted to BF16; that input ABI is different from ours. These mechanisms support the proposed geometry, not a claim of a measured M4 advantage. Ignore the newer tensor branch for this M4 prototype. [Tile/loader/MMA/store implementation](https://github.com/ggml-org/llama.cpp/blob/7cf1c54a96d4e950ffa614b94babf762803a8de7/ggml/src/ggml-metal/kernels/mul_mm.metal#L127), [BF16 instantiation](https://github.com/ggml-org/llama.cpp/blob/7cf1c54a96d4e950ffa614b94babf762803a8de7/ggml/src/ggml-metal/kernels/mul_mm.metal#L705).

**MLX `d9add9d11f3154111a4c85f267ec2fd307ecd18e`:** conventional `BlockMMA` places A, B, and C in accumulator-type fragments (normally FP32); its element loader casts staged input values into those fragments. Thus BF16 staging does not imply BF16 matrix operands. It also stores/casts fragment elements directly with safe boundary variants. This supplies a later BF16-staging→FP32-fragment alternative without doubling shared input storage, but adapting its explicit lane mapping is more involved than the simple FP32-staging control. [Fragment types/load/MMA](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/mma.h#L411), [lane mapping and cast](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/mma.h#L29).

MLX pads operand rows by **16 bytes**, meaning eight BF16 elements, and separates safe from unchecked block loads. Its loader copies contiguous per-thread vectors. These are concrete follow-ups, not evidence that rvllm currently suffers bank conflicts. [Padding](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/gemm.h#L34), [vector/safe loaders](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/loader.h#L38).

**No inspected primary source establishes an M4 BF16-versus-FP32 SIMD-matrix throughput ratio or that native BF16 operands cause our remaining prefill time.** MLX's implementation choice and llama's opposite choice justify measurement, not a diagnosis. Compiler lowering, conversion, fragment loads, synchronization, and resource pressure remain competing explanations.

## Optional follow-ups and bounded acceptance

After an initial winner, vectorize four contiguous BF16 elements per load: vector index `v=tid+128*j`, row `v/8`, K offset `4*(v%8)`; A has 256 vectors and B 512. Exact model K and base offsets permit aligned loads, but retain scalar zero-filled handling for synthetic K tails. Padding changes both staging writes and every fragment-load stride to 40; keep logical K at 32. Do not combine vectorization and padding in the first comparison.

Defer direct stores initially. Complete FP32 QKV fragments can later store directly for alpha=1/beta=0, following llama; BF16 output requires an explicit conversion path. Merely adding a direct-store branch while retaining statically allocated `ct` does not establish reduced shared-memory reservation.

Use current [guarded FP32/BF16 and FP64-dot harness](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/prefill_mma_tests.rs:143): one synthetic M/N/K-tail fixture, then M84 and M652 actual weights for sliding/global QKV `(N,K)=(8192,3840)/(9216,3840)`, gate/up `(30720,3840)`, O `(3840,4096)/(3840,8192)`, and down `(3840,15360)`. Include sampled columns 31/32/63/64 and final column, rows across 15/16/31/32 and the final row, finite BF16 inputs beyond FP16 range, fresh poisoned guarded outputs, and both output ABIs. Keep current numerical gates; compare each candidate to both MMA32 and sampled FP64 dots. Matched rotating-order GPU timings should use the same resident buffers and output ABI, separately from CPU oracle time. Promote only after real activation/continuation qualification. No dispatch, encoder-count, or residency change is necessary for the component.
