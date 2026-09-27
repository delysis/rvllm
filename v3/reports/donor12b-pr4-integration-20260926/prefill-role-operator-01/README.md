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

## MLX BF16 stage orientation at length 512

The already-implemented pinned MLX-LM stage harness completed as
`g4-donor-mlx-bf16-stage-512-01`, with a zero-second stability gate. Its
sampled conditions were eligible (AC, power mode 2, thermal state 0); the
test exited zero and measured all 22 planned cases. It uses MLX-LM source
`87b7b583`, MLX timing-protocol source `c215b6f8`, five warmups and 100
`mx.eval`-synchronized iterations per isolated case. Inputs are materialized
before timing. The following values are the mean milliseconds per iteration,
not a normal autoregressive-route profile:

| MLX stage | Prefill M=512, ms | Decode M=1, ms |
| --- | ---: | ---: |
| Embedding and scale | 0.443 | 0.317 |
| Sliding QKV projection | 2.979 | 0.427 |
| Full-attention QKV projection | 3.002 | 0.373 |
| Sliding SDPA core | 1.593 | 0.180 |
| Full SDPA core | 1.947 | 0.287 |
| Sliding O projection | 1.524 | 0.264 |
| Full O projection | 2.882 | 0.373 |
| Gate/up and activation | 9.890 | 0.748 |
| FFN down projection | 5.772 | 0.477 |
| Representative RMSNorm and residual | 0.343 | 0.166 |
| Tied LM head and logit softcap | 80.452 | 5.102 |

The prefill LM-head case projects all 512 rows and must not be counted as
normal-route prefill cost, where only the final position is needed. The
representative layer is 0 for sliding attention and 5 for full attention.
The output is explicitly labeled
`measured_microbenchmark_not_normal_route` in the raw JSON.

**No rvLLM/MLX per-role speedup follows from this table.** The Metal
projection screen above timed GPU command-buffer intervals at M=6, 84, 230,
650, or 1024, while the MLX wall timer includes `mx.eval` at M=512. MLX
gate/up includes activation whereas the Metal gate/up operator does not.
The MLX-LM model is a BF16 conversion of the **base** 12B checkpoint, not the
rvLLM 12B-it checkpoint; three corresponding layer-0 tensor-byte hashes
are different (see `../mlx-exact-prompt-01/CHECKPOINT_IDENTITY.md`). These are useful role-scale diagnostics and
show what to match next, not a normalized candidate ranking. In particular,
the large Metal advantage over its *own* batch8 comparator cannot erase the
separately measured exploratory full-route MLX advantage (~5.20x prompt
phase, ~1.40x decode for an exact-token 512+64 sequence); that framework
comparison has additional timing-boundary and generated-output differences.

The raw stage JSON SHA-256 is
`0a1620b7cbcae5966f8ac109357e315cee9275813c77f605fb0b7902499c490b`;
the complete immutable queue receipt archive SHA-256 is
`f618e32951e5e736ea0cd56dc7baeaf9579feac398ef9aa99347df262ba0248d`.
The real-weight M=512 Metal arm `g4-donor-bf16-prefill-mma-role-m512-01`
succeeded with eligible sampled conditions, zero exit status, production-
prototype equality, BF16 round-once output, guarded buffers, and sampled
independent FP64 checks. The values below are the median of eight GPU
command-buffer intervals per path on the original 12B-it layer-0 weights:

| Role at M=512 | rvLLM batch8 GPU ms | rvLLM MMA32 GPU ms | Batch8 / MMA32 |
| --- | ---: | ---: | ---: |
| Sliding QKV, N=8192 K=3840 | 27.904 | 6.291 | 4.44× |
| Gate/up, N=30720 K=3840 | 92.791 | 21.820 | 4.25× |
| O projection, N=3840 K=4096 | 14.810 | 2.830 | 5.23× |
| FFN down, N=3840 K=15360 | 58.488 | 12.622 | 4.63× |

The largest production-versus-batch8 FP32 relative L2 was `1.38e-6`.
This is a useful same-shape **rvLLM operator** result, not a default fused
projection+RMSNorm comparison or an rvLLM/MLX speedup. It improves shape
alignment with the MLX M=512 table but does not make MLX wall and Metal GPU
intervals equivalent, nor match fusion, activation, the base-versus-it
checkpoint, or output-rounding boundaries. A normal-route stage trace or
bounded fused-operator timing is still required to assign the default
prefill's time by role. The raw timing JSON SHA-256 is
`106bf44ef443598ceb2b57f757ca7dd1ced29de1f6e08ccb08a59b8bbc05bfe3`;
the complete queue receipt archive SHA-256 is
`bef73766d5653fdb0bc2f1ecc6a062182f7e6619c91d99c97ed7f32e6764db48`.

The same stage harness was also submitted as
`g4-donor-mlx-it-bf16-stage-512-01` against the original 12B-it snapshot,
after the M=512 Metal arm. It **failed before loading MLX weights** because
the harness's identity collector assumed a sharded
`model.safetensors.index.json`, whereas this original 12B-it checkpoint has
a single `model.safetensors`. This is a benchmark-precondition failure, not
evidence that MLX-LM cannot load the checkpoint or that any kernel is
incorrect. The immutable failed queue receipt archive SHA-256 is
`db43607a52382d32e3ab3f2a6ee024c778b86ab71fff021b3b110ee9e6772dfd`.
The dependent exact-token `-01` job has not run and remains preserved.

The harness now accepts either a sharded index or a single safetensor,
records the selected layout and file hash, and rejects directories with
neither. Its focused Python suite passed 10/10 tests. The corrected source
SHA-256 is `c9454ee6e66216debfe8f7c561e24c588c8f01064f278b5049b1141e2edb6070`.
The corrected stage arm `g4-donor-mlx-it-bf16-stage-512-02` and dependent
exact-token `-02` arm use new IDs in the same serial queue. The stage arm
successfully loaded the original 12B-it safetensor and measured all 22 cases
with zero exit status. Its queue report marked sampled conditions ineligible
only because one power-observer sample exceeded the freshness budget by about
0.51 s; no competing process or thermal violation was recorded. It is an
exploratory stage microbenchmark, not a qualified timing job. Successful
same-checkpoint output equivalence remains unproved, and the dependent
exact-token run is pending.

| 12B-it MLX-LM isolated stage at M=512 | Mean wall ms per `mx.eval` |
| --- | ---: |
| Sliding QKV projections | 2.802 |
| Sliding attention core | 0.776 |
| Sliding O projection | 1.438 |
| Gate/up projections plus activation | 9.222 |
| FFN down projection | 4.695 |

Five warmups preceded 100 synchronized iterations. These timings use the
*same checkpoint bytes* as the Metal M=512 arm, but the timer boundaries are
still different: the Metal table reports GPU command-buffer intervals and
MLX reports wall time around `mx.eval`. Gate/up includes activation in MLX
but not the Metal projection screen; MLX's `attention_k_eq_v` QKV path does
not perform the same three-projection work as the Metal QKV probe. O and
down projections are closer role/shape matches. Their measured Metal MMA32
GPU intervals of 2.830 ms and 12.622 ms versus MLX wall intervals of 1.438
ms and 4.695 ms point to a roughly 2–3× remaining isolated projection gap.
That is a diagnostic direction, not a qualified cross-framework speedup or
an allocation of normal-route prefill time. The stage JSON SHA-256 is
`06ce6a9b7085673be2c2d24fac9b7aa92f6b0ea241e3ed71d250de64ea44516a`;
the complete queue archive SHA-256 is
`431c73ebd35d6857296e45d27a699bdcb647172cced63d2a8e5f9351ae13f432`.
