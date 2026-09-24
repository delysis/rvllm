# Gemma 4 12B current-MLX reference, 2026-09-24

This is a local reference measurement and implementation audit, not an rvLLM
candidate qualification.  It records the current MLX-LM BF16 model on the same
M4 Max host used by the kernel game.  It must not be ratioed directly against
the existing llama.cpp Q4_0 measurements: the stored precision, implementation,
and benchmark harness differ.

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
- host state after the samples: AC power, battery charging at 60%; `pmset`
  reported no recorded thermal or performance warning level.

The benchmark command was `mlx_lm.benchmark` with batch size one, one generated
token, five trials, one second between trials, and `MLX_METAL_PREWARM=1`.  Each
prompt length ran in a fresh process and included the tool's own warmup.

## Prefill observations

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

## Consequences for the kernel campaign

The most relevant current-MLX prefill baseline uses 64x64x16 fused BF16 tiles,
with split-K 32x32x16 and 16x32x16 variants where its dispatcher chooses them.
That is materially different from the rejected rvLLM load4 batch, whose K tiles
were 32–128 and whose best measured candidates still lost to rvLLM's existing
`metal-mma32-load4`.  This is a design clue, not permission to copy a tile or a
claim that tile geometry alone explains the framework gap.

Next framework work should capture per-function GPU duration or counter samples,
separate warmup/compiler work from steady execution, and repeat with matched
4-bit and 8-bit models.  Kernel-game promotion still requires exact route and
dispatch evidence, independent numerical oracles, zero unexpected compile calls,
thermal-stratified ABBA timing, and independent confirmation.
