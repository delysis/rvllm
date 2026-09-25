# Gemma 4 ANE dynamic-QKV adjudication

## Decision

Reject `dynamic-qkv-static-int8-ffn-cached` as a decode-latency candidate. Keep
the implementation default-off as evidence that two weight-independent graphs
can replace 48 constant-QKV programs exactly; evaluate it separately only if
cold-start, cache footprint, or memory pressure becomes the objective.

The causal v2 referee held `all-sliding-fused-cached` attention/output constant
on both arms. The original v1 receipts changed both QKV and attention routes and
are preserved only as combined-route evidence; they are not used below.

## Correctness and route

The v2 exact-cache referee passed 16 dependent tokens with bit-identical
residuals after all 48 layers and identical final token/top-five logit bits.
Both arms executed 640 fused sliding-layer attention/output evaluations, 128
separate global attention evaluations, 128 separate global output evaluations,
and 768 QKV evaluations. Only the candidate selected dynamic QKV on all 48
layers. Compiler calls during inference were zero.

Correctness receipt SHA-256:
`704c496c8d0719bed5c56e0a29abf936e00291f545b2943f0eba01d331f64899`.

## Timing

Each ordering measured six observations and 24 dependent tokens per arm, with
fresh exact-cache decoder construction, one untimed warmup, and identical state
reimport outside the timed interval.

| Ordering | Static QKV, ms/token | Dynamic QKV, ms/token | Static / dynamic | Result |
| --- | ---: | ---: | ---: | --- |
| ABBA/BAAB/ABBA | 1795.439 | 1860.403 | 0.9651x | dynamic 3.62% slower |
| BAAB/ABBA/BAAB | 1770.986 | 1889.020 | 0.9375x | dynamic 6.67% slower |

Both campaigns retained exact outputs and zero compiler calls. Endpoint drift
was 2.72%/3.90% for baseline/candidate in ABBA and 14.72%/23.41% in reverse
BAAB. The noisier reverse campaign still agrees on the direction, so dynamic
QKV fails advancement rather than remaining inconclusive.

Timing receipt SHA-256 values:

- ABBA: `00a06860c3692d6989be9a8420643ed3f0b3cc12e25f5e5cb1ade20d2b16ca5b`;
- reverse BAAB: `c7b6cd19f92ab8503455c2f8494c0fec1bde2481a58a17b84df135d24dfc65b2`.

## Boundary

This does not show that reducing compiled-program count has no value. It shows
that, once loaded, the present dynamic-weight QKV evaluation path is slower
than the constant-weight cached path on this workload. Startup time, cache
storage, and memory-pressure behavior were not timed and must not be inferred.
