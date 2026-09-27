# Gemma 4 BF16 prefill: fused-route dispatch follow-up

This is a two-arm, real-weight, 512-token prefill diagnostic on the exact PR #6 tree. The manifests (`off.json`, `mma.json`), normal-route reports, and complete immutable queue receipts (`queue-results.tar.gz`) are retained here. The queue ran the arms serially in the same recorded AC / power-mode-2 / thermal-state-0 stratum. Neither arm had an inference-phase library or pipeline compile, dispatch-counter overflow, or a queue eligibility violation. The first two generated token IDs matched: `[236770, 236770]`.

| Prefill route | Wall | Command-buffer wait | CPU encode | Encoders | Ordinary prefill dispatches |
| --- | ---: | ---: | ---: | ---: | --- |
| Default | 21,334.225 ms | 21,325.805 ms | 8.406 ms | 481 | `attention_prefill_f16` 48; `gemm_f16` 48; `gemm_rmsnorm_f16` 96; `qkv_headwise_rmsnorm_rope_cache_f16` 48 |
| MMA-only opt-in | 4,391.698 ms | 4,375.593 ms | 16.105 ms | 625 | `attention_prefill_f16` 48; `gemm_f16_mma32` 144; `qkv_project_f32_mma32` 48; `qkv_projected_rmsnorm_rope_cache_f16` 48 |

The default/MMA prefill wall ratio is **4.8578×**, exploratory: one sequential sample per arm, not an ABBA/BAAB qualification. The phase boundary is `before_prefill_launch_to_after_prefill_collect`; the ledger covers only core GEMM, fused QKV, and prefill attention dispatches, and is compiled out without the opt-in `metal-route-diagnostics` feature. The wait dominates both totals, but is **not** per-kernel GPU time.

The source route explains the count change. Across 48 layers the default route uses `gemm_rmsnorm_f16` for O projection and FFN down projection (96 launches), plus `gemm_f16` for the gate/up projection (48). The MMA route uses three `gemm_f16_mma32` launches per layer, with normalization performed separately, and changes fused QKV to FP32 MMA projection plus postprocessing. The unchanged scalar prefill attention count is 48. In the checked-in Metal source, `gemm_rmsnorm_f16` has a scalar K loop and one threadgroup per output row/token, whereas `gemm_f16_mma32` is SIMD-group tiled. This is a strong structural explanation for the slow default prefill route and consistent with the large observed wait reduction; it does not establish the share of device time attributable to each role, generated ISA behavior, or numerical acceptability of the replacement.

There is also a concrete arithmetic-boundary difference to test. The fused
`gemm_rmsnorm_f16` kernel accumulates each projection in FP32, computes the
RMS from those unrounded accumulators, and rounds only the final normalized
output. The separate MMA route writes a dtype-rounded projection to storage
before `rmsnorm_f16` reads it. The BF16 source variant substitutes `bfloat`
and `bf16_sat` for the typed kernel. This difference is a plausible cause of
the later continuation mismatch, not proof of its first location or of which
route better matches the checkpoint reference. A route-preserving internal
tensor oracle is required.

Earlier 512-token/64-token continuations first diverged at generated index 5 with MMA-only. Two matching IDs here are insufficient for a quality or promotion claim. Next: capture comparable per-role GPU timing/generated-code evidence, locate the first internal numerical difference and establish checkpoint-specific logit/quality gates, then run paired full-route and matched MLX comparisons. Do not promote this switch on these receipts alone.

SHA-256: `queue-results.tar.gz` `1902aef9bc04173de12fe4822a1610d43d1504f3953cfa122b441e17395a1f0e`; `off-report.json` `4634523c7d5be5419be7ab672b5a69fe763fdd522e5292cc93f0c774c0793a9b`; `mma-report.json` `7a99634e679017c4b4370298092221930d2f01075ccd18fa9a783b3b5c898df7`.
