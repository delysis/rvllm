# 512-token prefill route isolation with 64-token continuations

Four queue arms used the same real BF16 Gemma 4 12B model path, 512-token
prompt, donor SG8 decode route, executable and metallib. Only the two
default-off prefill switches varied: `RVLLM_METAL_PREFILL_GEMM=mma32` and
`RVLLM_METAL_PREFILL_ATTENTION=simdgroup`. These are separate **diagnostic
arms**, not a paired timing qualification. The full queue receipts seal the
exact commands and inputs. Every job succeeded in the same sampled
power/thermal stratum, was queue-eligible, and reported zero inference
library/pipeline compiles. All four logged identical donor decode dispatch
counts: 512 global attention, 2560 local attention, 3072 gates, and 9216
projections.

| Prefill route | Prefill ms | Decode ms, 64 tokens | First generated-token difference vs default | Total differing positions / 64 |
| --- | ---: | ---: | ---: | ---: |
| default | 22034.50 | 4688.85 | — | 0 |
| MMA32 only | 4420.87 | 4681.50 | index 5 (sixth token) | 16 |
| SIMD attention only | 21483.26 | 4814.51 | index 57 (58th token) | 3 |
| both | 3541.77 | 4646.48 | index 9 (tenth token) | 14 |

MMA32 alone is approximately **4.98x** faster in total prefill than the
default route on this arm. SIMD attention alone gives only about **1.03x**;
both together give **6.22x**. Thus the MMA switch is the dominant observed
prefill-speed change. The 512-token ABBA screen separately found a 5.85x
off/on effect with low repeat drift. Source policy predicts that default
M512 BF16 projections use the general GEMM/QKV routes whereas the explicit
MMA32 option admits M<=1024 native-BF16 matrix projections; this is not yet a
named-kernel dispatch trace or per-stage GPU timing attribution.

The numerical result is a **promotion blocker**, not proof that MMA32 or
SIMD attention is arithmetically wrong. All routes agree on the first two
generated IDs, but each opt-in route eventually diverges under greedy
continuation. A small early logit difference can alter later generated
tokens; the first internal differing layer/tensor, top-logit margins, and an
independent reference comparison remain unknown. The token sets are notably
concentrated around IDs 236770 and 236761, so positional differences alone
cannot quantify quality. No prefill option should become the production
default until route-preserving layer/logit capture and checkpoint-specific
quality gates explain this divergence.

The queue-owned executable SHA-256 is
`bf149a5bfd2835fe8668008988f24b69f0331526750cff5712340112e9ec3dc5`;
the metallib SHA-256 is
`21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06`.
It predates the later diagnostic numeric-ABI mask fingerprint repair and
these are direct sessions only. Complete unpruned queue receipts, including
conditions, manifests, stdout/stderr, backend reports, and all 64 token IDs,
are in `queue-results.tar.gz` (SHA-256
`7d7f177bc7cf5a240c5223b7669873f4abd6bd1a0fee681e02d368d51381eed4`).
