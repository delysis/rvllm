# All-sliding ANE full-route, eight-token confirmation

The first independent longer batch, `g4-ane-all-sliding-confirm-t8-abba-20260926`,
completed in the persistent serial queue. Its [raw backend receipt](ane-all-sliding-confirm-t8-abba-receipt.json)
has SHA-256 `07249369d053427bec3c207d6ab5c1287cfacce4ac33d54f35eced1fd7898451`;
the queue runner report has SHA-256
`2c1a0e9fddbb09afdd2ca8c35889b9b659e8845496df5a429e89abc2b9faa9d9`.
The reverse-order `g4-ane-all-sliding-confirm-t8-baab-20260926` was still
running when this note was written. Do not promote from this single order.

The timed scope is eight dependent tokens from embedding through vocabulary
ranking, divided by token count. Each route has six measured observations,
48 tokens total. The baseline used separate attention and output projection;
the candidate used the cached fused path on all sliding layers, retaining
separate global attention/output. Both routes used the static-int8 FFN cache.
Outputs matched exactly, and compiler calls remained zero after load and
through warmup/timing.

| First order only | Baseline | Candidate |
|---|---:|---:|
| Median ms/token | 2275.856 | 1844.005 |
| Baseline/candidate | \- | 1.2342x |
| Within-route range drift | 11.86% | 8.14% |

The following are medians of **per-observation, per-token phase times**, not
one additive median observation. The candidate's fused operation replaces
most separate sliding attention and output work; the residual attention and
output entries largely represent the unfused global layers.

| Phase, ms/token | Baseline | Candidate |
|---|---:|---:|
| Attention | 475.4 | 77.6 |
| Output projection | 494.4 | 82.9 |
| Fused attention + output | 0 | 414.3 |
| QKV | 503.3 | 494.0 |
| FFN | 595.2 | 580.4 |
| Vocabulary | 183.3 | 176.8 |
| Host | 16.5 | 16.0 |

The phase accounting supports a narrow hypothesis: the fusion removes a
substantial part of the separate attention/output cost, while QKV and FFN
remain large in this route. It does **not** isolate hardware execution from
all host, command, cache, or contention effects. Neither route's range drift
meets the referee's 5% gate, and the reversed order is pending. The result is
an unqualified but directionally useful full-route measurement, not an MLX
comparison or ANE promotion.
