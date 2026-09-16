# Gemma 4 12B Metal prefill: reduction-depth experiment

The test-only candidate keeps the existing 32-by-32 output tile, 128 threads,
native BF16 operands and four FP32 accumulator fragments per SIMD group.
Only the reduction tile grows from 32 to 64. This halves reduction-loop
barriers, while declared shared storage grows from 8 to 12 KiB. Lower
occupancy could outweigh the barrier saving. Production routing is unchanged.

For the 84-token workload, the packed gate/up projection has N=30,720 and
K=3,840; down has N=3,840 and K=15,360. Their K-loop counts change from
120 to 60 and 480 to 240. Three output-row tiles cover 96 rows, including
12 padded rows. Across 48 layers these FFN projections account for 77.8% of
the packed projection FLOPs. These are source-derived counts, not measured
time attribution.

This follows the research agent's bounded source audit. MLX's pinned
[GEMM loop](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/metal/kernels/steel/gemm/gemm.h#L72)
separates reduction depth from accumulator geometry; it does not prove BK64
is faster here. The earlier wider-output-tile and FP32-staging experiments
already lost, so neither is included in this comparison.

## Component gate

The existing guarded Metal test harness now supports a separate BK64 test:
`layer_forward::prefill_mma_tile_tests::native_bf16_bk64_checks_precision_and_m84_ffn`.
New Rust orchestration reuses the existing Metal FFI boundaries; no additional
unsafe Rust block was introduced.

It covers both real-weight M84 FFN shapes plus synthetic M63/N67/K65 tails.
Inputs are synthetic BF16, including values beyond FP16 range. Four variants
cross BK32/BK64 with FP32/BF16 output. Acceptance requires finite values,
complete FP32 bit parity, exact BF16 rounding including signed zero, sampled
independent FP64 dots, and intact output guards. Every output and guard is
checked again after repeated dispatches. This is component coverage, not
actual decoder-activation or end-to-end model acceptance.

Qualification-only mode omits timing trials. Timing mode uses three ABBA/BAAB
blocks per output ABI, six samples per variant, retaining order and raw GPU
timestamps plus host wall time. Independent queue controls determine whether
those measurements are eligible. A screening win still needs actual-activation
and full-prefill checks before any routing change.

Source review found no blocking indexing or dispatch defect. Its two evidence
improvements, post-timing guard/output checks and signed-zero-sensitive BF16
comparison, are implemented. Build and frozen-artifact receipts will be
recorded below before queue submission.

## Executed qualification and staged timing

Release compilation succeeded; a final no-run build confirmed the frozen
source. Probe binary SHA-256:
`ac41de10b18a68164934c3f569c63c4c742538caf472e3870c00654284ea5711`.
The source snapshot, logs and manifests are in
`gemma4-12b-evidence-20260914/int8-next-controls-20260916/`.

Queued preparation `40-metal-bk64-qualification` succeeded on this M4 Max.
All three shapes passed complete FP32 bit parity, exact BF16 rounding,
sampled FP64 tolerances and guard checks. Twelve public Metal commands ran,
with no timing trials and no ANE access. Runtime pipeline metadata confirms
8,192 bytes of static shared memory for BK32 and 12,288 for BK64; both report
SIMD width 32. The boot identity remained unchanged.

Qualification report:
`experiment-queue-20260916/baseline-only-v5/results/40-metal-bk64-qualification/metal.json`,
SHA-256 `a8cb92eb52e549c47279f69d9c17cb326bccfa777b9ae3b657fb5ff6ef359ab6`.
M84 gate/up and down used actual checkpoint weights with synthetic inputs;
these are not actual full-model activations.

Timing job `16-ac-metal-bk64` requires this qualification and follows the
native-kit AC analysis and INT8-head timing. It retains the 120-second quiet
window, nominal thermals, AC/low-power/mode-1 stratum and 16 GiB disk floor.
No speedup has been established and no production dispatch changed.
