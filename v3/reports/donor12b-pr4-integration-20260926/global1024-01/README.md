# SG8 global-attention isolation at 1024-token context

The existing serial experiment queue executed an off/on/on/off comparison on
one real BF16 Gemma 4 12B model path and the same 1024-token prompt. All arms
selected donor projections, fused gates, and local attention. The only
component switch was donor **global** attention; both prefill MMA32 and SIMD
attention were enabled in every arm to make the 1024-token setup tractable.
Each arm generated two tokens. The queue manifests seal the exact commands,
inputs, executable, and metallib. The paired controls had the same sampled
power/thermal stratum, were all queue-eligible, and reported no inference
library or pipeline compiles.

| Order | Donor global attention | Decode ms, 2 tokens | Prefill ms | Generated IDs |
| ---: | --- | ---: | ---: | --- |
| A | off | 1268.82 | 6558.84 | 236770, 236770 |
| B | on | 161.58 | 6628.50 | 236770, 236770 |
| C | on | 168.57 | 6723.02 | 236770, 236770 |
| D | off | 1271.07 | 6639.81 | 236770, 236770 |

Mean two-token decode is 1269.94 ms off and 165.08 ms on: **7.69x** in this
bounded, matched-route screen. The off pair differs by 0.18% and the on pair
by 4.23%; the queue observed the same stratum across arms but cannot prove
identical GPU clocks. Dispatch evidence changes only the donor global-attention
count: 0 off versus 16 on. Each arm records 80 donor local-attention, 96
donor gate, and 288 donor projection dispatches. The same two output IDs show
only short-continuation agreement, not logit or long-continuation equivalence.

This is strong *route-level* evidence that global attention is the main source
of the donor decode advantage at a long context. The source-level mechanism is
also concrete: the incumbent online decode path excludes Gemma's global
head_dim=512 (`head_dim <= 256`), leaving a `attention_decode_f16` route with
one thread per head and a serial KV/head-dimension loop; donor SG8 maps a
global head across 8 SIMD groups/256 threads, parallelizes the KV scan, then
merges within the threadgroup. This experiment does not isolate arithmetic,
occupancy, memory traffic, scheduling, or exact GPU time inside those kernels.
In particular, 7.69x is a full two-token route ratio, **not** a per-kernel
microbenchmark or a comparison to MLX. The separate 512/64-token prefill
study shows why two-token matching must not be promoted as a quality gate.

The queue-owned executable SHA-256 is
`bf149a5bfd2835fe8668008988f24b69f0331526750cff5712340112e9ec3dc5`;
the metallib SHA-256 is
`21d81a66d31075e9307b64b7cb8eb0eea1f1dee74c0865ff3d0fa8125579dd06`.
The executable predates the later diagnostic numeric-ABI mask fingerprint
repair; these are direct-session measurements, not imported/shared KV
qualification. Complete unpruned queue receipts, including conditions,
manifests, stdout/stderr, and backend reports, are in `queue-results.tar.gz`
(SHA-256 `e9a02b7e7f27578fb1dd0963dc7a2392b3546d414b06248a2a5921588afdade1`).
