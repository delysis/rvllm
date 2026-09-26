# All-sliding ANE full-route, eight-token confirmation

Both independent longer batches, `g4-ane-all-sliding-confirm-t8-{abba,baab}-20260926`,
completed in the persistent serial queue. The first [raw backend receipt](ane-all-sliding-confirm-t8-abba-receipt.json)
has SHA-256 `07249369d053427bec3c207d6ab5c1287cfacce4ac33d54f35eced1fd7898451`;
the queue runner report has SHA-256
`2c1a0e9fddbb09afdd2ca8c35889b9b659e8845496df5a429e89abc2b9faa9d9`.
The reverse-order [raw backend receipt](ane-all-sliding-confirm-t8-baab-receipt.json)
has SHA-256 `872ff06297d58b30c11d59e1fea1018f65fc0db29db2fd756361941d3de4aedc`;
its queue runner report has SHA-256
`4d204e483317671e23945b2b18d6290a1850a5e0ea0c8953d8d8ed018152ba15`.

The timed scope is eight dependent tokens from embedding through vocabulary
ranking, divided by token count. Each route has six measured observations,
48 tokens total. The baseline used separate attention and output projection;
the candidate used the cached fused path on all sliding layers, retaining
separate global attention/output. Both routes used the static-int8 FFN cache.
In both orders, outputs matched exactly and compiler calls remained zero
after load and through warmup/timing.

| Order | Baseline ms/token | Candidate ms/token | Speed ratio | Baseline drift | Candidate drift |
|---|---:|---:|---:|---:|---:|
| ABBA-first | 2275.856 | 1844.005 | 1.2342x | 11.86% | 8.14% |
| BAAB-first | 2246.590 | 1854.070 | 1.2117x | 5.85% | 3.09% |

The following are medians of **per-observation, per-token phase times**, not
one additive median observation. The candidate's fused operation replaces
most separate sliding attention and output work; the residual attention and
output entries largely represent the unfused global layers.

| Phase, ms/token | ABBA baseline | ABBA candidate | BAAB baseline | BAAB candidate |
|---|---:|---:|---:|---:|
| Attention | 475.4 | 77.6 | 478.1 | 77.3 |
| Output projection | 494.4 | 82.9 | 481.9 | 82.5 |
| Fused attention + output | 0 | 414.3 | 0 | 413.6 |
| QKV | 503.3 | 494.0 | 509.3 | 498.1 |
| FFN | 595.2 | 580.4 | 582.3 | 583.6 |
| Vocabulary | 183.3 | 176.8 | 179.2 | 176.5 |
| Host | 16.5 | 16.0 | 17.7 | 17.4 |

The phase accounting supports a narrow hypothesis: the fusion removes a
substantial part of the separate attention/output cost, while QKV and FFN
remain large in this route. It does **not** isolate hardware execution from
all host, command, cache, or contention effects. The direction reproduces
under both starting orders, but three of the four within-route range-drift
checks exceed the referee's 5% gate. This is a **prospective**, not qualified,
full-route improvement; it is not an MLX comparison or ANE promotion. The
next ANE optimization work should target QKV and FFN without discarding this
fusion arm, while later controlled confirmation handles the variance gate.
