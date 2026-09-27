# Gemma 4 12B BF16 512-token prefill ABBA screen

The existing serial queue executed **off/on/on/off** for the opt-in prefill
MMA32 plus SIMD-attention route, using the same real BF16 Gemma 4 12B model
path, same 512-token prompt case, same SG8 donor decode selector/component
mask, same executable/metallib, and two generated tokens per arm. Here
`off` means both explicit prefill options absent; `on` sets
`RVLLM_METAL_PREFILL_GEMM=mma32` and
`RVLLM_METAL_PREFILL_ATTENTION=simdgroup`. It does **not** mean donor SG8
decode was disabled. The four queue manifests in the evidence archive fix
the exact environment and input hashes. All four queue jobs succeeded, all sampled condition strata were
eligible, `stable_seconds` was zero, all returned `[236770, 236770]`, and
each reported zero inference library/pipeline compiles.

| Order | Arm | Prefill ms | Two-token decode ms |
| ---: | --- | ---: | ---: |
| 1 | off A | 22,217.25 | 157.66 |
| 2 | on B | 3,811.49 | 144.23 |
| 3 | on C | 3,865.66 | 147.02 |
| 4 | off D | 22,695.23 | 156.48 |

The off/on mean-prefill ratio is **5.85x** (22,456.24 / 3,838.57 ms),
with 2.15% end-to-end off drift and 1.42% on drift. This is a bounded
same-checkpoint, same-prompt, same-executable *total prefill* comparison,
not per-operation GPU time or an independent qualification replicate.
The donor decode ledger stayed fixed at 16 global attention, 80 local
attention, 96 fused gate, and 288 native projection dispatches in every
arm; the prefill switch therefore did not erase the donor decode route.
The broad layer-body encoder count changes 1632 → 1776 as the selected
prefill policy materializes GEMM/RMSNorm separately. Those counts are
accounting estimates, not an actual named-kernel dispatch trace.

This identifies the default-off prefill projection policy as a major
actionable bottleneck. The independent 256-token screen in
`../component-screen-01/` found MMA-only 3.13 seconds, SIMD-attention-only
14.26 seconds, and both 2.46 seconds versus 11.56–12.83 seconds for
all-on SG8 with default prefill. That points to MMA32 as the dominant
switch, with a possible additional interaction from SIMD attention.
Neither screen has valid per-stage GPU timestamps on this host, so exact
QKV/FFN/attention/host allocation remains unknown. The generated BF16
Metal source uses native `simdgroup_matrix<bfloat>` with FP32 accumulation;
it does not introduce a BF16→FP16 adaptation boundary.

For orientation only, the retained non-interleaved MLX BF16 512 prompt
baseline is 173.051 tokens/s, or about 2.96 seconds for 512 tokens. The
3.84-second on-arm average is approximately **1.30x slower** on total
prefill, a substantial improvement over the prior ~7.6–8.3x gap. The
MLX model-file/prompt-token identities and host conditions are not matched,
so this is not established MLX parity. The two-token match is not a
64-step, teacher-forced-logit, per-layer, or checkpoint-quality gate.

The queue-owned executable SHA-256 is
`bf149a5bfd2835fe8668008988f24b69f0331526750cff5712340112e9ec3dc5`;
the metallib SHA-256 is
`21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06`.
The executable predates a later diagnostic numeric-ABI mask fingerprint
repair, and this screen used direct sessions only. It is not evidence for
imported/shared KV handoff. The complete four queue receipts, including
manifests and backend reports, are in
`queue-results.tar.gz` (SHA-256
`7802bc1b4ab718496d3e1c5fe89512cd85e23bb9db8f76d00a72597db3c0fe86`).
No data was pruned, retried, or promoted.
