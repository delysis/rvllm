# BF16 512-token ordinary prefill route ledger — 2026-09-26

## Four-arm queue screen

An opt-in `metal-route-diagnostics` build recorded named GEMM, QKV-fused,
and prefill-attention encodes after `endEncoding()`. Normal builds compile the
ledger out. All four real Gemma 4 12B IT BF16 jobs succeeded in the existing
serial queue, with zero condition violations, unchanged pinned inputs, and the
same sampled AC/power-mode-2/thermal-0 stratum. The queue did not wait for a
stable thermal state. This sequential single sample per arm is **exploratory**,
not ABBA/BAAB timing qualification or a per-kernel GPU profile.

| Prefill opt-ins | Prefill wall ms | Command-buffer wait ms | CPU encode ms | Encoders | Named encodes per 48 layers |
| --- | ---: | ---: | ---: | ---: | --- |
| Neither | 22,145.591 | 22,134.239 | 11.339 | 481 | scalar attention 48; general GEMM 48; fused QKV+norm+RoPE+cache 48 |
| MMA32 only | 4,954.753 | 4,923.260 | 31.461 | 625 | scalar attention 48; BF16 MMA GEMM 144; FP32 QKV MMA 48; projected QKV postprocess 48 |
| SIMD attention only | 21,430.165 | 21,409.845 | 20.297 | 481 | SIMD prefill attention 48; general GEMM 48; fused QKV+norm+RoPE+cache 48 |
| Both | 3,830.379 | 3,802.881 | 27.464 | 625 | SIMD prefill attention 48; BF16 MMA GEMM 144; FP32 QKV MMA 48; projected QKV postprocess 48 |

All four returned generated IDs `236770, 236770` and reported zero inference
library/pipeline compiles in the bounded prefill phase. The `ordinary_dispatch`
subreceipt covers **only** the named core GEMM, fused QKV, and prefill-attention
sites; absent names are not evidence of absent work. The research-candidate
dispatch delta was empty in prefill, while the donor candidate's full-case
counts belong to decode. Completion was checked by collecting the command
buffer before the phase snapshot.

The default/MMA-only prefill ratio is 4.47x; default/SIMD-only is 1.03x;
default/both is 5.78x. The earlier queue-owned paired both-on/default-off
screen found 5.85x, consistent in direction, but these four new arms were not
counterbalanced. The source's `encode_gemm_rmsnorm` policy uses one
threadgroup per token for the fused default projection+norm kernel, while the
MMA option materializes projection then normalizes it in a separate encoder.
The 48 FP32 QKV MMA and 144 BF16 MMA encodes in the latter arm are consistent
with QKV plus O, gate/up, and down projection substitution. That is a
source-and-route explanation, **not** an allocation of the saved milliseconds
to individual shaders. The fused `gemm_rmsnorm_f16` dispatch was not included
in this v1 ledger; explicitly count it in a subsequent diagnostic before
calling that precise fallback route proven.

For orientation only, the both-on arm is approximately 133.7 prompt tok/s
(`512 / 3.830379 s`) versus an older same-host MLX BF16 512-token result of
173.051 prompt tok/s, about 1.29x short. Those runs did not match exact prompt
IDs, checkpoint files, timing order, or model-quality criteria. The opt-in
routes previously diverged from the default over longer continuations (first
generated difference at index 5 MMA-only, 57 SIMD-only, 9 both). Matching two
IDs here does not authorize promotion or imply a numerical bug in any arm.

## Provenance and next test

- Source: PR #6 commit `d1d642d46d7074570cc2b90772b53afcd3bbf172` for the
  diagnostic ledger, committed manifests at `6da37350`.
- Release executable SHA-256:
  `62105fcd948d33626cbaf20526982b40d626df76a15dea401fcc0881a2229c4e`.
- Complete queue result directories, including job manifests, stdout/stderr,
  conditions, and referee reports: `queue-results.tar.gz`, SHA-256
  `805843ad711046d84cb34a15730045756a9ada68ca85307d4211efa94fe85a76`.
- Normal-route outputs: `off-report.json`, `mma-report.json`,
  `attention-report.json`, and `both-report.json`. The queue manifests seal
  the executable, wrapper, prompt JSONL, metallib, and checkpoint config.

The next diagnostic should count the fused `gemm_rmsnorm_f16` encode and
separate O and down projection shapes/roles. A validated GPU trace or bounded
operator timings are still needed for per-role milliseconds. In parallel,
first-internal-difference/logit/reference work must resolve long-continuation
quality before promotion, and MLX needs exact-work paired comparison.
