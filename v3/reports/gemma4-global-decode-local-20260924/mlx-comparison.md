# Gemma 4 global D512 decode: rvLLM versus MLX

Date: 2026-09-24

## Claim boundary

This is an exploratory operator-level comparison. The rvLLM arm is one native
Metal dispatch of the correctness-qualified BF16 `r16p128t128` research kernel.
The MLX arm is the synchronized isolated `attention_core_sdpa/full_attention`
decode case from the pinned BF16 MLX-LM model: five warmups, 100 measured calls,
and `mx.eval` in every call. Both represent one Gemma 4 global-attention layer
at the stated live context, but they are not the same implementation or a
sealed cross-framework referee. Host/runtime variance remains in the MLX
measurement. This evidence can prioritize designs; it cannot promote one.

| Context | rvLLM candidate GPU ms | MLX synchronized ms | rvLLM / MLX |
|---:|---:|---:|---:|
| 256 | 4.824 | 0.325 | 14.86x |
| 512 | 9.009 | 0.240 | 37.46x |
| 1024 | 30.178 | 40.048 | 0.75x |
| 2048 | 39.593 | 1.489 | 26.59x |

The MLX 1024 observation is an obvious cross-job outlier and is retained, not
interpreted as rvLLM superiority. The 256 and especially 2048 cells establish
the useful directional result: even the best current rvLLM candidate remains
far behind MLX for the matched operator family.

## Where the rvLLM kernel spends its budget

The dispatch receipt reports grid `[1,1,1]`, 128 threads, four SIMD groups,
and 20,000 bytes of static threadgroup memory. The generated MSL specialization
packs all 16 query heads into that single threadgroup. It advances through the
context eight keys at a time and, for every block:

1. stages K in four D=128 panels with barriers;
2. performs fixed-order QK reductions for all heads and keys;
3. runs online-softmax exponentials only on SIMD leader lanes;
4. reuses the same allocation to stage V in four panels with barriers; and
5. updates the complete FP32 output state before advancing to the next block.

For H=16 and D=512, QK plus probability-times-V is approximately
`4 * H * context * D` floating-point operations. K and V together contain
approximately `2 * context * D * sizeof(BF16)` input bytes. These lower-bound
rates expose the structural problem:

| Context | Approx. FLOPs | Approx. K/V bytes | Candidate GFLOP/s | Candidate GB/s |
|---:|---:|---:|---:|---:|
| 256 | 8.39M | 0.52 MB | 1.74 | 0.11 |
| 512 | 16.78M | 1.05 MB | 1.86 | 0.12 |
| 1024 | 33.55M | 2.10 MB | 1.11 | 0.07 |
| 2048 | 67.11M | 4.19 MB | 1.69 | 0.11 |

The byte estimate excludes small Q, output, page-table and parameter traffic,
so it is not a hardware-counter result. Those omissions are far too small to
change the conclusion: the kernel uses only one threadgroup and achieves
neither meaningful arithmetic throughput nor meaningful memory bandwidth.
Time is structurally dominated by serialized context traversal, repeated
threadgroup barriers, fixed reductions, leader-lane exponentials, and the lack
of enough independent threadgroups to occupy the GPU.

## Design consequence

Do not spend the next round tuning another unsplit row/panel/thread tuple as if
it could become the final design. Retain this family as a correctness baseline.
The performance family must split context/KV work across multiple threadgroups,
emit numerically sealed partial online-softmax state, and merge those partials
with an independently checked reduction. The tournament must separately price
the partial kernel, merge kernel, intermediate traffic and additional dispatch.

MLX's Q4/Q8/BF16 isolated-stage matrix also shows that attention is not the
whole model budget: FFN remains the largest weighted family, with QKV and O
projection material. A split-KV attention win therefore needs matched rvLLM
projection, FFN, normalization, embedding and LM-head measurements before it
can explain or close end-to-end throughput.

## Evidence

- rvLLM timing source: `r16p128t128-checkpoint.md` and the corresponding
  `queue/results/*/native/abba.json` receipts.
- rvLLM generated MSL:
  `campaign/metal-global-d512-r16p128t128-core.metal`.
- MLX source: `../gemma4-mlx-long-context-queue-20260924-v6/run/results/`
  `mlx-gemma4-bf16-stage-pp*/trial.stdout`.
- MLX protocol and trace boundary: `../gemma4-mlx-stage-and-trace-evidence-20260924.md`.

The 7.2 GB MLX `.gputrace` is valid generated-source and dispatch evidence, but
`xctrace export` rejects GPU captures with `Document Missing Template Error`.
Per-dispatch GPU counters and durations have therefore not been batch-exported;
the synchronized MLX operator timings above are the current numeric baseline.
