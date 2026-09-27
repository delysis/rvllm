# Real-weight BF16 projection operator screen

The existing ignored `native_bf16_mma_checks_tails_precision_and_real_projection_time`
test is submitted once through the serial experiment queue as
`g4-donor-bf16-prefill-mma-role-extended-01`. Its executable, Gemma 4 12B
BF16 safetensor blob, and configuration are pinned by SHA-256 in `job.json`.
The queue records sampled conditions without requiring a stable thermal
state. The job succeeded with eligible sampled conditions (AC, power mode 2,
thermal state 0) and exit code zero. Pipeline compilation occurred before
the timed invocations, not as part of their command-buffer GPU intervals.

The extended arm checks guarded output buffers, BF16 round-once output,
independently sampled FP64 dot products, production/prototype FP32 equality,
and real layer-0 projection weights. After warmup it alternates eight
baseline/candidate pairs, recording command-buffer GPU start/end and host
wall times for each invocation. The principal shapes are:

| Role | M × N × K | Baseline | Candidate | Output |
| --- | --- | --- | --- | --- |
| Sliding QKV | 6 and 84 × 8192 × 3840 | `qkv_project_f32_batch8` | `qkv_project_f32_mma32` | FP32 |
| Gate/up | 230 × 30720 × 3840 | `gemm_f16_batch8` | `gemm_f16_mma32` | BF16 |
| O projection | 650 × 3840 × 4096 | `gemm_f16_batch8` | `gemm_f16_mma32` | BF16 |
| FFN down | 1024 × 3840 × 15360 | `gemm_f16_batch8` | `gemm_f16_mma32` | BF16 |

Two small synthetic tail shapes also exercise bounds. These are isolated
projection operators, not a complete 512-token prefill, not MLX comparisons,
and **not** a default-fused `gemm_rmsnorm_f16` comparison. They cannot by
themselves explain the full default-versus-MMA prefill gap or clear its
arithmetic/quality mismatch. A subsequent matched fused projection plus
RMSNorm experiment remains necessary, with storage-rounding boundaries
explicitly controlled.

## Measured operator result

The table uses the median of eight individually recorded GPU command-buffer
durations per path, in milliseconds. Speedup divides the baseline median by
the candidate median; values below 1 mean the MMA candidate was slower.

| Role / M | Baseline GPU ms | Candidate GPU ms | Baseline / candidate |
| --- | ---: | ---: | ---: |
| Synthetic tail / 6 | 0.0124 | 0.0252 | 0.49× |
| Synthetic tail / 63 | 0.0187 | 0.0326 | 0.57× |
| Sliding QKV / 6 | 0.4807 | 0.4276 | 1.12× |
| Sliding QKV / 84 | 5.0539 | 1.1554 | 4.37× |
| Gate/up / 230 | 44.7093 | 10.4938 | 4.26× |
| O projection / 650 | 20.6642 | 3.7655 | 5.49× |
| FFN down / 1024 | 132.8327 | 24.2063 | 5.49× |

The test also passed production/prototype equality, once-rounded BF16
outputs, guard bytes, finite values, and sampled independent FP64 dots at
every shape. Reported relative L2 differences between the incumbent and MMA
FP32 projections were at most `1.38e-6`. This is a single exploratory
operator campaign, not independent confirmation or a full-model quality
gate. The strong size dependence is consistent with tiled matrix-work
parallelism being valuable for prefill, but does not by itself identify how
much of the normal-route default's ~21 s is spent in any named kernel. In
particular, `gemm_f16_batch8` is not the default fused
`gemm_rmsnorm_f16`, and this test does not time attention or MLX.

The complete raw `role-timing.json` SHA-256 is
`2d58ba6c1cacf6494bbe800e99882627f65dfd231c9e4a6c4232ff9035090faa`;
the immutable queue receipt archive SHA-256 is
`14e05ef1caee196f802ad392a7b0756197d3d418ed882fc73086211a70a9d73f`.

The already-implemented pinned MLX-LM BF16 stage harness was submitted next
as `g4-donor-mlx-bf16-stage-512-01` with a zero-second stability gate. It
measures 22 isolated prefill/decode stage cases at length 512 using upstream
MLX's synchronized timing-loop protocol. Its receipt is pending. Even when
complete, its loaded conversion, operator shapes, and synchronization
boundaries must be compared explicitly before any rvLLM/MLX role ratio is
called matched.
