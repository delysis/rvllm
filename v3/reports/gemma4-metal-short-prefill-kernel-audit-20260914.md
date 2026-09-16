# Gemma 4 12B short-prefill kernel audit

Source-only audit, 2026-09-14. No hardware, builds, tests, cache operations, or product changes were performed. **Next experiment: replace the serial QKV projection inside the fused prefill shader with an eight-row cooperative projection into FP32 scratch, followed by the existing normalization/RoPE/cache arithmetic.** This is a bounded way to remove a demonstrated unfavorable mapping; its latency benefit and numerical acceptance remain unmeasured.

The qualified phased run reports 21.338 s for seven prefill/sample/export operations and 25.426 s for Metal preparation, separately from 2.760 ANE decode tokens/s. Approximately 3 s per prefill is therefore an application-stage observation, not isolated GPU projection time. Current unrelated contention further prevents attributing it to one kernel. See the [qualified observations](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-metal-ane-progress-20260914.md:1018).

## What the current source establishes

| Operation | Current behavior | Implication |
|---|---|---|
| Gate/up, O, down at 20–32 prompt rows | Shape-limited `gemm_f16_batch8`, eight input rows per weight stream; separate RMSNorm for O/down | The earlier serial-projection problem is already addressed for these shapes. Do not propose raising the old nineteen-row cutoff again. |
| QKV with head norms | Bypasses GEMM dispatch and enters the fused projection/norm/RoPE/cache shader | The batch8 improvement does **not** cover this substantial projection. |
| Prefill attention | One thread per query/head, one thread per threadgroup; serial head-dimension loops and `float out_vals[512]` | Poor SIMD utilization is explicit; register spilling and actual time share require profiling. |
| Submission and output head | Normal prefill uses one ordered command buffer; vocabulary projection samples only the final prompt row | Per-layer host synchronization and a vocabulary projection for every prompt row are not supported explanations for this path. |

Local pointers: [projection eligibility](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:24), [batch8 kernel](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:368), [normal QKV selection](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:796), [attention launch](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:1369), [attention implementation](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:1266), [final-row sampling](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_metal_backend.rs:4085), and [prefill command encoding](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_metal_backend.rs:4215).

QKV's [inner loop](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:802) assigns a projected channel to a thread and walks all 3,840 reduction elements serially. At a common loop iteration, adjacent threads read weight addresses 3,840 elements/7,680 bytes apart. Each prompt token repeats the weight traversal. Cache behavior can reduce physical traffic, so these are **source-level access and dependency facts**, not measured DRAM bytes or a speedup estimate. Fusing the inexpensive epilogue has retained an unfavorable matrix mapping.

Current dimensions, using the already packed `[Q;K;V]` weights:

| Layers | A | B | FP32 projection result |
|---|---|---|---|
| 40 sliding | `[M,3840]` BF16 | `[8192,3840]` BF16 | `[M,8192]` |
| 8 global | `[M,3840]` BF16 | `[9216,3840]` BF16 | `[M,9216]` |

Global packing includes the repeated K rows used for V; preserve that layout for this experiment. The [loader](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/gemma4_model.rs:2506) already handles it. Each packed matrix is respectively 62,914,560 or 70,778,880 source bytes. With eight-row reuse, M=21–28 needs three or four source traversals rather than 21–28; that count is not a DRAM guarantee.

## What the pinned upstream implementations actually do

**MLX**, pin `d9add9d11f3154111a4c85f267ec2fd307ecd18e`: its wide GEMV route rejects more than three passes of at most five vectors, so M=21–28 falls through to GEMM. The non-NAX matrix path uses tiled threadgroup loading and SIMD-group multiply-accumulate; partial row tiles are supported. Split-K is selected only under explicit output-grid and K-size conditions: it is not universally useful for our wide gate/up or QKV. Sources: [wide-GEMV conditions and routing](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/matmul.cpp#L1375-L1387), [GEMM routing](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/matmul.cpp#L1564-L1615), [split-K conditions](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/matmul.cpp#L959-L973), [tile loading](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/gemm.h#L97-L120), [matrix primitive](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/mma.h#L181-L206).

**llama.cpp**, inspected current pin `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b` (2026-09-14): its [matrix eligibility](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/ggml/src/ggml-metal/ggml-metal-common.cpp#L10-L17) selects matrix multiplication beyond eight activation columns with K≥64 and supported layouts/device. Its conventional Metal kernel stages reusable tiles and performs [SIMD-group matrix accumulation](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/ggml/src/ggml-metal/kernels/mul_mm.metal#L287-L313); a [BF16 specialization](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/ggml/src/ggml-metal/kernels/mul_mm.metal#L770-L772) is conditionally compiled. Its actual [Gemma 4 graph](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/src/models/gemma4.cpp#L207-L229) performs the optional packed QKV matrix multiplication before head normalization and RoPE. It does not require folding the entire block into a serial projection kernel.

These are implementation precedents, not matched Gemma 12B/M4 performance results. The conventional SIMD-group path is the relevant starting point; M5/NAX performance claims do not establish M4 results. The local [experimental matrix kernel](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:407) is not a ready substitute: it has one 8×8 tile, stages FP32 values every eight K elements, returns without computing partial M tiles, and is not production-selected. Promoting it directly would leave tails at M=21–28 uncomputed.

## One bounded candidate and its decision gate

Adapt the established batch8 mapping **only for the two QKV shapes at M=20–32**, initially retaining cooperative FP32 accumulation rather than simultaneously introducing native matrix instructions. Use BF16 A/B and FP32 output, K-lane stride 32, eight output channels per 256-thread group, and grid `(ceil(M/8),ceil(N/8))`. Bounds-check partial prompt tiles. One reusable scratch region of at most `32×9216×4 = 1,179,648` bytes suffices. Follow with one head-wise norm/RoPE/cache epilogue. This isolates the effect of coalesced K access and cross-token reuse at lower implementation cost than a complete Steel-style GEMM port.

**Do not simply call the existing BF16-output GEMM.** Today [projection sums feed RMSNorm directly in FP32](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:802), then normalized values are rounded to BF16 before RoPE. An intermediate BF16 projection store adds a new numerical boundary. Preserve actual gamma, epsilon, FP32 statistics, BF16 normalized-value rounding, proportional/global RoPE, V handling and cache offsets. Parallel reduction still changes addition order, so byte identity is not promised. Keep the existing fingerprint/cache-ABI discipline for any promoted reduction change.

When hardware validation is authorized and contention is controlled: compare baseline QKV fusion against **projection plus epilogue combined** on captured real sliding/global inputs at M=21 and 28. Record GPU time, encode/wall time and all Q/K/V outputs. Then qualify the seven first-token/continuation cases and imported KV against the established mixed-precision oracle. Report prefill GPU work, final head and host export separately, with preparation excluded. Retain the candidate only if the full prefill stage improves without violating the established numerical gates. If QKV is a small measured fraction, the one-thread attention launch is the next profiling target; no additional kernel rewrite is justified by this audit alone.

## Local provenance

Checkout base: `144cdd01dd51bb7d0c2c9d942e50cefc25d3ba5f`; inspected files include uncommitted work. SHA-256 at inspection:

| File | SHA-256 |
|---|---|
| `rvllm-apple-metal/src/kernels.rs` | `fba5a0893fab0f42f02dbc34f523e7e76f2cdb7b8211dfba4c75698777f48df7` |
| `rvllm-apple-metal/src/layer_forward.rs` | `296a65e7a4b82755acf6bb7e85a1375a97ccf27638e64d2649af1685c8bf00bc` |
| `rvllm-runtime/src/apple_metal_backend.rs` | `01492a8ff71e09b4b5b8a9dec2b560f2768386580c0930975760c1505abd06d2` |
| `rvllm-apple-metal/src/gemma4_model.rs` | `f9950eddc6b20049187ae8f8eb7922fc9b5e432500303d5f705835fc85e095a0` |

All four paths are under `v3/crates`. Source inspection establishes neither a new benchmark nor full numerical acceptance of the proposed candidate.
