# Gemma 4 12B current-MLX reference, 2026-09-24

This is a local reference measurement and implementation audit, not an rvLLM
candidate qualification.  It records current MLX-LM BF16 plus locally derived
affine 4-bit and 8-bit models on the same M4 Max host used by the kernel game.
It must not be ratioed directly against the existing llama.cpp Q4_0
measurements: even the nominally 4-bit stored representations and benchmark
harnesses differ.

## Sealed identities

- MLX-LM source: `87b7b583a697537aa68f47130b40884700b5f55f`
- `mlx-lm`: `0.30.3.dev268+g87b7b583a`
- `mlx`: `0.32.2`
- model: `mlx-community/gemma-4-12B-bf16`
- model snapshot: `e61ef6842dc9407c3ceba8800b7200c0b07f816f`
- `config.json` SHA-256:
  `95ac51f934f85c9243f970e6077ad5ff6056b50de7bb4c4be866205f4a7901b6`
- `model.safetensors.index.json` SHA-256:
  `5a2037525ab516767d2a213bf7cb74f7d05940bbccb64e1de281f93b40953577`
- local Q4: MLX affine, group size 64, reported 4.501 bits/weight, 6.3 GB;
  config/index SHA-256 `1ad49b8a789471a952702d7e1aac290313a3dd2af27e47de4054ccbfb496b5b4`
  and `0352c33d9baee674195c874b42687e0afa0fb68b42f5c2e1a8a2fff44b125b7a`
- local Q8: MLX affine, group size 64, reported 8.500 bits/weight, 12 GB;
  config/index SHA-256 `2b419be9d2f003d34a38f32efe88bc7dc7435d5735f39dbbd53635af1953b29c`
  and `ce4b07209d9468b021934541e3ccd5979d2e34675ab6740c8edfbdf5137d4d98`
- host state after the samples: AC power, battery charging at 60%; `pmset`
  reported no recorded thermal or performance warning level.

The benchmark command was `mlx_lm.benchmark` with batch size one, one generated
token, five trials, one second between trials, and `MLX_METAL_PREWARM=1`.  Each
prompt length ran in a fresh process and included the tool's own warmup.

## BF16 prefill observations

| Prompt tokens | Trial prompt tok/s | Mean | Median | Range | Peak memory reported (GB) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 21 | 20.514, 17.076, 19.557, 7.639, 6.857 | 14.329 | 17.076 | 6.857–20.514 | 23.921 |
| 84 | 44.532, 43.128, 62.565, 48.068, 50.427 | 49.744 | 48.068 | 43.128–62.565 | 23.944 |
| 652 | 132.600, 142.000, 133.183, 127.258, 129.564 | 132.921 | 132.600 | 127.258–142.000 | 24.420 |

The 21-token process slowed sharply in its fourth and fifth measured trials.
The other two lengths did not reproduce that pattern.  These samples establish
an observed distribution, not a stable causal ranking.  A future framework
comparison needs interleaved, precision-matched trials under the kernel-game
referee rather than selecting a favorable MLX trial.

The reported single-token `generation_tps` values ranged from roughly 10,700 to
40,900 tok/s.  Those values are not physically meaningful end-to-end decode
throughput; the numerator is one token and the timed region is dominated by the
benchmark's synchronization/timer boundary.  They are deliberately excluded
from performance claims.

## Quantized prefill observations

Both models were produced from the sealed BF16 snapshot by the pinned
`mlx_lm.convert`.  These measurements establish execution and observed timing;
they do not establish model-quality equivalence to the source checkpoint or to
rvLLM's eventual quantizers.

| Weight format | Prompt tokens | Trial prompt tok/s | Mean | Median | Range | Peak memory reported (GB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| affine Q4 g64 | 21 | 39.617, 50.090, 12.072, 12.011, 25.548 | 27.868 | 25.548 | 12.011–50.090 | 6.835 |
| affine Q4 g64 | 84 | 70.062, 67.002, 61.275, 73.114, 67.995 | 67.889 | 67.995 | 61.275–73.114 | 6.951 |
| affine Q4 g64 | 652 | 132.829, 136.526, 119.199, 152.087, 174.222 | 142.973 | 136.526 | 119.199–174.222 | 7.617 |
| affine Q8 g64 | 21 | 14.618, 22.981, 130.765, 135.968, 25.655 | 65.997 | 25.655 | 14.618–135.968 | 12.776 |
| affine Q8 g64 | 84 | 64.937, 58.221, 63.727, 27.572, 35.016 | 49.895 | 58.221 | 27.572–64.937 | 12.818 |
| affine Q8 g64 | 652 | 145.304, 60.706, 117.642, 153.604, 161.908 | 127.833 | 145.304 | 60.706–161.908 | 13.545 |

Q4 was comparatively coherent at 84 tokens, but its 21- and 652-token samples
still varied materially.  Q8 was extremely variable at every length, including
a nearly 10x span at 21 tokens.  The arithmetic means are therefore especially
misleading.  These are useful framework-dispatch observations, not
promotion-quality timings; the queue must compare candidates in interleaved
blocks with drift rejection.

For context only, the separately recorded llama.cpp Q4_0 means were 49.215,
114.544, and 153.351 prompt tok/s at 21, 84, and 652 tokens.  Precision and
harness mismatch make those numbers descriptive strata, not evidence that one
framework or kernel is faster by their quotient.

## Actual Metal specialization evidence

A real 84-token, one-trial run was launched under Instruments' `Metal System
Trace`.  The trace target exited successfully and recorded an M4 Max device.
The traced trial reported 36.598 prompt tok/s and 23.942 GB peak memory; tracing
overhead means it is not part of the timing summary above.

The shader table contains 56 distinct compute functions for the MLX process.
Important observed specializations include:

- prefill projection GEMMs:
  `steel_gemm_fused_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2_*`;
- another dense orientation:
  `steel_gemm_fused_nn_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2_*`;
- split-K projection and accumulation:
  `steel_gemm_splitk_nt_bfloat16_float32_bm32_bn32_bk16_wm2_wn2_*`,
  `steel_gemm_splitk_nt_bfloat16_float32_bm16_bn32_bk16_wm2_wn2_*`, and
  `steel_gemm_splitk_accum_bfloat16_float32`;
- attention and normalization:
  `block_softmax_precise_bfloat16`, `rmsbfloat16`,
  `sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks`, and
  `looped_logsumexp_bfloat16`;
- position and sampling work:
  `rope_bfloat16_`, `rope_single_bfloat16_`, `rope_freqs_bfloat16_`,
  `rope_single_freqs_bfloat16_`, `argmax_bfloat16`, and BF16 gathers;
- one-token paths also resident in the process:
  `gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0` and
  `gemv_bfloat16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`.

The trace also records live Metal shader compiler activity.  Function names and
tile parameters prove which specializations were resident; they do not expose
the compiler's final machine ISA or attribute per-layer duration by themselves.
The raw local trace and XML exports are under
`v3/reports/gemma4-reference-audit-20260924/mlx/`.  They are intentionally not
treated as generated-MSL source: MLX ships template Metal sources and creates
these named specializations, while Instruments reports the compiled function
identity rather than reconstructing source text.

Separate real Q4 and Q8 traces at 84 tokens show that MLX does not merely
dequantize the whole model and reuse BF16 GEMM.  Q4 selected
`affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true`,
`affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0`,
`affine_qmv_bfloat16_t_gs_64_b_4_batch_0`, and
`affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0`, with an observed
`affine_dequantize_bfloat16_t_gs_64_b_4` helper.  Q8 selected the corresponding
`b_8` split-K/QMM kernels and `affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0`,
plus `affine_dequantize_bfloat16_t_gs_64_b_8`.  Both retained BF16 attention,
normalization, RoPE, and some ordinary Steel GEMMs for non-quantized or
ineligible operations.  The traced Q4 and Q8 trials reported 59.795 and 51.310
prompt tok/s respectively; tracing overhead excludes them from the tables.

The installed pinned MLX Metal template `quantized.h` has SHA-256
`2a007016da606afe569adb9adcc05e00f14558ad2e094bcb4f8974beb53c316f`.
Inspection of that exact source explains the observed names:

- the affine QMM defaults to a 32x32x32 block, two-by-two SIMD-group MMA
  arrangement, and 128 threads;
- a `QuantizedBlockLoader` cooperatively reads packed bytes plus the group's
  scale and bias, dequantizes directly into a padded BF16 threadgroup tile, and
  feeds the existing Steel `BlockMMA`; it does not materialize a full decoded
  weight matrix in device memory;
- the split-K entry offsets packed weights, scale/bias groups, activations, and
  partial outputs before invoking the same QMM implementation on a K partition;
- each K block has a barrier before cooperative loads and another before MMA;
  only the final result store adds the third barrier outside the loop;
- the fast QMV uses two SIMD groups, four output rows per SIMD group, two packed
  words per lane for 4/8-bit cases, FP32 partial accumulators, `simd_sum`, and
  one result store from lane zero.

These details identify actionable hypotheses for rvLLM: fuse group decoding
with tile staging, keep separate prefill QMM and decode QMV dispatch, test a
32-element K tile before deeper K blocks, and consider bounded split-K for the
large projections.  They do **not** establish that MLX's exact layout or
barrier schedule is optimal for rvLLM's storage format, numerical contract, or
M4 Max workload.

## Consequences for the kernel campaign

The most relevant current-MLX BF16 prefill baseline uses 64x64x16 fused tiles,
with split-K 32x32x16 and 16x32x16 variants where its dispatcher chooses them;
its affine quantized QMM instead defaults to 32x32x32 and fused threadgroup
dequantization.
That is materially different from the rejected rvLLM load4 batch, whose K tiles
were 32–128 and whose best measured candidates still lost to rvLLM's existing
`metal-mma32-load4`.  This is a design clue, not permission to copy a tile or a
claim that tile geometry alone explains the framework gap.

Next framework work should capture per-function GPU duration or counter samples
and separate warmup/compiler work from steady execution.  Kernel-game promotion
still requires exact route and
dispatch evidence, independent numerical oracles, zero unexpected compile calls,
thermal-stratified ABBA timing, and independent confirmation.
