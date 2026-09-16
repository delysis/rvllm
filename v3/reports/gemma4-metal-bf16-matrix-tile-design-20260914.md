# BF16 matrix tiles for Gemma 4 12B short prefill

Source-only design and candidate review, 2026-09-14. **The isolated BM32/BN32/BK32, 128-thread candidate implements the recommended arithmetic design, with no source-level stop issue found. Keep that tested layout for initial integration; BM16 is a later bounded occupancy comparison.** The immediate integration requirement is a BF16-output specialization for gate/up/O/down, retaining FP32 output only for QKV. No hardware, builds, implementation changes, or cache operations were performed by this review; the local measurements below were produced by the parent task and read from its evidence.

The current [batch8 kernel](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:368) shares a weight across eight tokens but still computes scalar products and reductions. A matrix tile changes the dominant arithmetic and increases reuse across output channels. The local [8×8 experiment](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:458) is not a suitable drop-in: it stages FP32 operands every eight K elements, computes only one fragment, and drops partial prompt tiles. Approximately 3 s per application prefill does not identify a GPU kernel's measured share. The recommendation is source-grounded, not a promised speedup.

## M4 and macOS 15 support

Apple documents `simdgroup_bfloat8x8` from **Metal 3.1**, with 8×8 fragment dimensions and uniform SIMD-group control flow. M4 is Apple9; SIMD-scoped matrix multiplication is available from Apple7. These are conventional Metal matrix operations, not the M5/NAX or Metal 4 tensor path. Sources: [MSL specification, 2026-06-04, §2.4 and §6.8, pp.38/212–213](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), [feature tables, 2026-05-21, pp.2/4](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf). The installed [Apple SDK declarations](/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.2.sdk/System/Library/Frameworks/Metal.framework/Headers/MTLLibrary.h:218) mark MSL3.1 available on macOS14 and MSL3.2 on macOS15.

For mixed operand/accumulator types, implementation evidence is more explicit than the PDF's simplified same-type signature. Current llama.cpp instantiates **BF16 A/B fragments and FP32 C/D**, guarded by `GGML_METAL_HAS_BF16`; [its header disables BF16 below MSL3.1](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/ggml/src/ggml-metal/kernels/common.h#L31-L37). This is not merely a recent API assumption: a [2025-07-25 implementation](https://github.com/ggml-org/llama.cpp/blob/793c0d7f46384001738c337d7afa46b45ae32745/ggml/src/ggml-metal/ggml-metal.metal#L7083-L7111) already uses a BF16 matrix operand with a FP32 operand/accumulator, instantiated at line7523. These sources support trying the public mixed-type API on macOS15. They do not establish its throughput, emitted ISA or our exact numerical behavior on this machine.

Use `<metal_simdgroup_matrix>` and BF16 matrix operands explicitly. In rvllm's shared F16/BF16 source, prefer `simdgroup_matrix<half,8,8>` for operands: [the generator](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:2411) replaces the standalone `half` token with `bfloat`. It **does not** turn the identifier `simdgroup_half8x8` into `simdgroup_bfloat8x8`. Accumulators and QKV output remain float. Require MSL≥3.1 for this variant; record compiler language/math settings. [Current compilation](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/context.rs:132) supplies default options, so changing global math options would be a separate numerical change.

## What to adapt from upstream

| Pin and source | Useful implementation detail |
|---|---|
| MLX `d9add9d11f3154111a4c85f267ec2fd307ecd18e`: [GEMM tile loader/loop](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/gemm.h#L38-L120), [bounded loading](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/loader.h#L82-L126) | Cooperative operand staging, leading-dimension padding, explicit producer/consumer barriers, zero-filled partial tiles. |
| Same MLX pin: [BlockMMA](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/mma.h#L451-L531), [fragment loading](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/mma.h#L57-L65) | The conventional path converts stored BF16 values into **FP32 fragments**, because operand tiles use `AccumType`. Copy its tiling principles without mislabeling that arithmetic as BF16 fragments. Its lane-coordinate/`thread_elements()` machinery need not be copied; public matrix loads/stores avoid depending on an unspecified element-to-lane mapping. |
| llama.cpp `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b`: [32-wide K tiling](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/ggml/src/ggml-metal/kernels/mul_mm.metal#L160-L180), [matrix multiplication](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/ggml/src/ggml-metal/kernels/mul_mm.metal#L287-L313), [bounded stores](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/ggml/src/ggml-metal/kernels/mul_mm.metal#L317-L355), [BF16 instantiation](https://github.com/ggml-org/llama.cpp/blob/96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b/ggml/src/ggml-metal/kernels/mul_mm.metal#L770-L772) | Closest mixed-type MMA precedent. Its names/orientation differ: its full tile spans 64 output channels ×32 tokens. Adapt addressing deliberately rather than copying those strides. |

## Exact tile and memory contract

All operations use row-major BF16 `A[M,K]`, existing row-major BF16 `B[N,K]`, and row-major `C[M,N]`:

| Projection | N | K | Stored output boundary |
|---|---:|---:|---|
| Sliding QKV | 8192 | 3840 | FP32, then existing head norm/RoPE/cache epilogue |
| Global QKV | 9216 | 3840 | FP32, same epilogue; keep duplicated K-for-V rows |
| Gate/up | 30720 | 3840 | BF16 before the existing tanh-GELU × up |
| Sliding/global O | 3840 | 4096 / 8192 | BF16 before existing post-attention RMSNorm |
| Down | 3840 | 15360 | BF16 before existing post-FFN RMSNorm |

All N/K are multiples of32, so only **M needs a tail** within the proposed M20–32 gate. Keep alpha=1/beta=0 for the first variant. No model-weight transpose, quantization, head projection change or full-model temporary is needed.

The conservative design below uses `As[BM][40]` and `Bt[32][40]` in BF16: 32 active K entries plus eight padding entries per row. `Bt` keeps the source `[N,K]` orientation. Use `Ct[BM][32]` in float for bounded output conversion/stores. Total explicit threadgroup memory is **9,216 bytes for BM32** or **5,888 bytes for BM16**. Padding follows MLX's 16-byte leading-dimension allowance; its speed benefit is unmeasured here. **The subsequently implemented probe uses stride32 and 8,192 bytes for BM32, and passed; keep that simpler tested layout for initial integration.** The pseudocode's strides become32 for that version. Shared arrays should be aligned for matrix loads.

Dispatch `(ceil(M/BM),ceil(N/32),1)` groups of128 threads, verifying a SIMD width of32. For SIMD-group `s=0..3`, assign `sm=(s/2)*(BM/2)`, `sn=(s%2)*16`. Each group computes `(BM/2)×16`: four FP32 8×8 accumulators for BM32, two for BM16. The following is design pseudocode, not compiled source:

```text
zero every FP32 accumulator
for k0 in 0..K step 32:
    all 128 threads cooperatively fill As and Bt, K contiguous
    As[r,k] = (m0+r<M) ? A[(m0+r)*K+k0+k] : bfloat(0)
    Bt[n,k] = B[(n0+n)*K+k0+k]  // N aligned by eligibility
    threadgroup_barrier(mem_threadgroup)
    for kk in {0,8,16,24}:       // only this small loop needs unrolling
        a[i] = simdgroup_load(As+(sm+8*i)*40+kk, 40, origin=0, transpose=false)
        b[j] = simdgroup_load(Bt+(sn+8*j)*40+kk, 40, origin=0, transpose=true)
        C[i,j] = simdgroup_multiply_accumulate(a[i], b[j], C[i,j])
    threadgroup_barrier(mem_threadgroup)  // before ANY group overwrites tiles
simdgroup_store every C[i,j] into disjoint float Ct subtiles, stride32
threadgroup_barrier(mem_threadgroup)
all 128 threads copy valid rows from Ct to C[(m0+r)*N+n0+n]
    store float for QKV; round once to bfloat for the other projections
```

Both barriers in the K loop are necessary for cross-SIMD-group reuse. Every lane must execute every matrix operation uniformly, including lanes associated with padded rows. Do not return for individual tail rows, and do not call `simdgroup_store` directly into a partial device tile. `elements_per_row` is measured in **elements**, not bytes. Transposing the B fragment on load converts `[output,K]` into the mathematical `[K,output]` operand.

BM32 reads one weight tile across all prompt rows; BM16 rereads it for two row tiles but doubles the grid and reduces accumulator pressure. At N3840 these give120 versus240 threadgroups. Both have32 padded row slots for M21–28. That is a useful bounded comparison; it does not predict the winner. Start without split-K, asynchronous copies, double buffering, atomics or activation fusion.

## Review of the implemented probe and recorded result

Read [prefill_mma_tests.rs](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/prefill_mma_tests.rs:11), SHA-256 `dde941e68f7b0cba34816bd51d302cf1119d2210daf5be97e4f5ae80b6b04c2d`. Its four SIMD groups own disjoint16×16 output quadrants, all M/N/K tails are zero-filled, both K-loop barriers are present, and all final float stores are masked. B loads correctly request transposition. No bounds, ownership, synchronization or accumulator-type defect was found in this bounded review. Its direct `bfloat` fragment spelling avoids the shared-source type-rewriting trap.

The parent's [recorded component summary](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/metal-native-bf16-mma-component/summary.json) reports:

| Actual layer0 projection | M | Batch8 FP32-output median | MMA32 FP32-output median | Ratio |
|---|---:|---:|---:|---:|
| Sliding QKV | 21 | 1.310 ms | 0.520 ms | 2.52× |
| Gate/up | 27 | 6.519 ms | 2.064 ms | 3.16× |
| O | 28 | 1.004 ms | 0.514 ms | 1.95× |
| Down | 32 | 6.700 ms | 2.778 ms | 2.41× |

The M21/N37/K35 tail fixture was exact but slower, supporting narrow shape gating. Maximum large-shape relative L2 was `1.3811e-6`; the test includes BF16 inputs outside FP16 range, output guards and sampled independent FP64 dot products. It uses controlled activations and **FP32 output for every shape**. These records demonstrate working public BF16-fragment/FP32-accumulator operations on the task's M4/macOS15 environment; they do not measure emitted ISA, production BF16 stores, global-layer geometries or full-model quality. The [run receipt](/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/metal-native-bf16-mma-component/run-result.json) reports success; the inspected test source takes eight timing samples per variant with alternating order.

**Integration hazard:** the probe's float output cannot be bound directly to the existing two-byte gate/up/O/down regions. A typed BF16 store variant must round each valid float result once to BF16; QKV alone uses four-byte output. Reusing the float ABI for those buffers would overrun them. Before generic dispatch, verify SIMD width32 and group128 against the selected pipeline, and share the exact shape/dtype eligibility with telemetry. Obvious upstream tuning differences—larger N tiles, vectorized loads, direct full-tile stores, padded strides—are optional performance experiments, not blockers. First qualify this already measured tile with correct output types.

## Numerical and acceptance boundaries

Never narrow BF16 operands to FP16 to access matrix operations. Gate/up must retain its BF16 output before [tanh-GELU multiplication](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:1530); O/down must retain their BF16 projection output before [separate RMSNorm](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:2626). QKV must retain FP32 output. Retain default FP32 accumulation, exclude the quantized-accumulator diagnostic mode, and leave all norm/activation/RoPE math unchanged. FP32 accumulation type does not promise the previous scalar addition order or bit identity. Fingerprint any promoted arithmetic change.

For the next authorized integration check, validate the BF16-output gate/up variant on **captured BF16 pre-FFN-normalized activations**, including its unchanged activation consumer. Gate/up is about52% of packed projection arithmetic. Check the other exact shapes and global geometry, using FP32 raw-output comparison plus the existing QKV epilogue for QKV, then perform the established untraced seven-case full-model and imported-KV qualification. Keep only shapes that improve complete prefill time without failing those numerical gates. BM16 or padding can wait until those gates establish whether more tuning is needed. Component speedups alone do not establish a faster full prefill.

Local source SHA-256 at inspection: `kernels.rs` `79abd2d2d3026a4ac8b4b696323aea31114d146e7fcad4fe825291ae3744de35`; `layer_forward.rs` `d4808bcee0b201490b642609f50e495e2555186f68408b781ee5474113ada6f3`; `context.rs` `4af052a9d572153ebb3882ef0f737aee28c770617b1360190967f138bcd75d1f`. These files are under `v3/crates/rvllm-apple-metal/src` and include uncommitted work.
