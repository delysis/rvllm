# Gemma 4 12B Apple kernel comparison: MLX, llama.cpp, and rvLLM

Date: 2026-09-24.  This is a source-and-trace comparison, not a claim that the
three frameworks have precision-matched end-to-end speed.  It separates actual
observed GPU functions from source-supported interpretation and from proposed
rvLLM experiments.

## Evidence identity and limits

- llama.cpp timing and trace executable: commit
  `4f13cb742476d81a6b42a2aa5996e82a478c2481`, build 9191.
- current llama.cpp source inspected for drift: commit
  `8212c7802455255460ab8e18fc34754560031b34`.
- llama model: Google Gemma 4 12B QAT Q4_0 GGUF snapshot
  `29d097773436b69ff9feafd636ab4cf873786537`, 6,960,054,464 bytes.
- llama configuration: all 99 requested layers on Metal, 2,048 batch, 512
  microbatch, FP16 K/V, flash attention disabled.
- MLX-LM: `87b7b583a697537aa68f47130b40884700b5f55f`, MLX 0.32.2.
- MLX models: sealed BF16 source snapshot and locally derived affine Q4/Q8
  group-64 models documented in `gemma4-mlx-reference-20260924.md`.
- rvLLM comparison point: branch commit `837aed47`, with BF16
  `metal-mma32-load4` incumbent and rejected load4 tile batch.

The llama trace is a real 21-token Metal System Trace.  Its shader table contains
19 distinct functions for the target process.  The MLX traces are real 84-token
BF16/Q4/Q8 runs.  Shader Timeline was disabled by the standard template, so the
traces prove residency and execution context but do not provide defensible
per-function GPU duration.  Kernel names below are observed unless explicitly
marked source-only.

## Component map

| Gemma stage | llama.cpp Q4_0 trace | MLX trace | rvLLM present state / implication |
| --- | --- | --- | --- |
| Embedding / row access | `kernel_set_rows_f16_i64`, `kernel_get_rows_f32` | BF16 `gather_front*` | Not a demonstrated campaign bottleneck; retain route accounting so row work cannot disappear from totals. |
| Input and post norms | `kernel_rms_norm_f32_4`, `kernel_rms_norm_mul_f32_4`, `kernel_rms_norm_mul_add_f32_4` | `rmsbfloat16` plus fused elementwise kernels | llama fuses scale/add variants; MLX uses BF16 RMS and graph fusion. rvLLM has SIMD RMS candidates, but prior aggregate screens did not establish a winner. Isolated timing plus full-route confirmation is needed. |
| QKV, O, gate/up, down prefill projections | `kernel_mul_mm_q4_0_f32`; `kernel_mul_mm_f16_f32` also executed for non-Q4 operands | Q4/Q8 `affine_qmm_t_*`, split-K variants; BF16 `steel_gemm_fused_*` and split-K | rvLLM's measured BF16 incumbent is `metal-mma32-load4`. It has no qualified direct packed Q4/Q8 Metal projection equivalent yet. The strongest concrete next experiment is fused packed decode into a 32x32x32 padded threadgroup tile, separate from BF16 storage. |
| Single-token / narrow projection | `kernel_mul_mv_q4_0_f32`; `kernel_mul_mv_q6_K_f32` for a differently quantized tensor | Q4 ordinary and fast affine QMV; Q8 fast affine QMV; BF16 GEMV variants | Prefill GEMM and decode GEMV require separate selectors and evidence cells. A prefill win cannot qualify decode, nor can an isolated QMV stand in for the full ANE decoder. |
| Q/K normalization and RoPE | RMS kernels above; `kernel_rope_neox_f32` | `rope_bfloat16_`, `rope_freqs_bfloat16_`, and single-token variants | Arithmetic precision/order differs. rvLLM must preserve its exact oracle boundaries; copying a lower-precision RoPE is not an authorized speed trade. |
| Attention scores / masking / softmax | `kernel_mul_mm_f16_f32`, `kernel_soft_max_f32_4`, `kernel_pad_f32`, copies; flash attention was disabled | `block_softmax_precise_bfloat16`, BF16 Steel NN GEMM, fused select/mask; `sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks` on the one-token path | llama's trace represents an explicit non-flash decomposition. MLX retains a block softmax for prefill and a vector SDPA specialization for decode. rvLLM SIMD attention exists, but comparison must split sliding/global heads and prompt lengths. |
| FFN nonlinearity and gate product | `kernel_geglu_f32`, `kernel_bin_fuse_f32_f32_f32` | a generated fused broadcast/multiply/tanh/multiply kernel consistent with approximate GELU and gating | Function identity shows both avoid a purely scalar host path. The opaque MLX generated name does not by itself prove fewer dispatches per layer; encoder and graph evidence must be counted. |
| Residual / copies / unary work | fused norm-add and binary-fuse kernels plus `kernel_cpy_f32_f32`, `kernel_unary_f32_f32_4` | generated fused elementwise kernels and BF16 copies | Dispatch count and memory traffic matter alongside GEMM. Any rvLLM packed candidate must show full-route work counts, not only its projection kernel latency. |
| Output head / selection | llama-bench prefill-only trace does not exercise a comparable sampling step; Q6_K MV is present but cannot be attributed solely from the name | `looped_logsumexp_bfloat16`, `argmax_bfloat16`, gathers in the one-token benchmark | This row is not comparable in the captured workloads. A matched decode campaign must include logits and sampling or explicitly exclude both. |

## Projection implementation details

### llama.cpp

The executed Q4_0 prefill function is instantiated from `mul_mm.metal` as a
SIMD-group matrix kernel with packed `block_q4_0` input, `dequantize_q4_0`, FP16
matrix operands/fragments, and FP32 output.  The executed decode function comes
from the separate `mul_mv.metal` family.  This division is important: llama does
not force its matrix tile to serve the narrow-vector regime.

The current Gemma 4 graph constructs a fused QKV tensor when present, applies
per-head Q/K RMS normalization and RoPE, then builds attention and the output
projection.  Its dense FFN has separate gate/up/down weights with a fused graph
builder for the activation/gate product.  Those graph facts explain why the
trace includes Q4 matrix/vector work, FP16 matrix work, fused norms, RoPE,
softmax, GEGLU, and binary fusion.  They do not assign elapsed time to each
function.

### MLX

MLX affine QMM uses packed bytes and per-group scale/bias.  A cooperative
`QuantizedBlockLoader` decodes directly into a padded BF16 threadgroup tile and
feeds a 32x32x32, 128-thread Steel MMA.  Its split-K entry partitions packed
weights and activations before invoking the same implementation.  Its fast QMV
instead consumes packed values directly in two SIMD groups, computes four rows
per SIMD group, accumulates in FP32, and reduces with `simd_sum`.

The observed BF16 path uses 64x64x16 fused Steel tiles and smaller split-K
variants.  The observed quantized path uses 32x32x32.  This is evidence that
MLX's dispatcher changes kernel family and K depth with representation; it is
not evidence that one universal tile is optimal.

### rvLLM

The current BF16 incumbent cooperatively loads four elements at a time and uses
MMA32.  Seven larger/deeper load4 variants all passed component and route
correctness; three variance-robust comparisons lost to the incumbent and the
others were not advanced.  Consequently, "increase K depth" is contradicted as
a general BF16 optimization on this host.  MLX's quantized 32-deep tile remains
a distinct hypothesis because packed decoding changes memory traffic and shared
tile construction.

rvLLM still lacks qualified Metal Q4/Q8 full-route cells and has ANE INT8
research paths rather than six completed precision/backend cells.  Existing ANE
cache hits, first-token equality, or static-INT8 execution cannot be promoted to
professional-grade Q4/Q8/16 coverage without representation-specific numerical
oracles, stable zero-compile cache evidence, repeated-use checks, and controlled
timing.

## Speed observations without false attribution

Five-sample prompt means were:

| Framework / representation | 21 tokens | 84 tokens | 652 tokens |
| --- | ---: | ---: | ---: |
| llama.cpp Google QAT Q4_0 | 49.215 | 114.544 | 153.351 |
| MLX affine Q4 g64 | 27.868 | 67.889 | 142.973 |
| MLX affine Q8 g64 | 65.997 | 49.895 | 127.833 |
| MLX BF16 | 14.329 | 49.744 | 132.921 |

These are tok/s from different harnesses.  MLX Q8 showed extreme bimodal drift,
and MLX's single-token generation metric was invalid at timer granularity.
Therefore neither means nor their quotients establish kernel speed.  They do
show that representation and prompt length change the observed regime, and that
the tournament must use interleaved blocks rather than framework-by-framework
runs.

## Next evidence needed

1. Add Q4 and Q8 packed Metal candidates with separate QMM and QMV selectors;
   first test fused group decode into 32x32x32 staging and a bounded split-K arm.
2. Seal a representation-level CPU/FP64 oracle for packed values, scale/bias,
   tail groups, output rounding, guard bytes, and repeated use before timing.
3. Build matched framework workloads from identical token IDs and explicitly
   equal sampling/exclusion rules.  Interleave whole framework runs or treat
   them as descriptive strata only.
4. Capture GPU counter or shader-timeline evidence capable of per-function
   duration; the current standard Metal trace cannot localize aggregate time.
5. Keep ANE Q4, Q8, and 16-bit cells separate.  ANE "16-bit" needs an explicit
   representation and compiler-lowering contract rather than inference from a
   BF16 Metal path or an INT8 ANE cache.

No row in this report promotes a kernel.  Promotion remains subject to sealed
identities, actual dispatch and full-route work evidence, independent numerical
confirmation, zero unexpected compilation, thermal-stratified ABBA timing, and
independent confirmation.
