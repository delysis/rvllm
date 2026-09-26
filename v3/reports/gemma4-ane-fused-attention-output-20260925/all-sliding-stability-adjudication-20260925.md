# Gemma 4 ANE all-sliding stability adjudication

## Result

`all-sliding-fused-cached` passed the extended full-route stability gate and is
the current ANE decode attention/output prospective winner. It remains
default-off: this experiment does not establish checkpoint quality, long-
context behavior beyond the imported snapshot, global-layer fusion, or a
shipping selector.

The sealed release test executable was
`target/release/rvllm-runtime-tests-fused-all-sliding-stability-v1`, SHA-256
`3fb96206ff310bebb5cf9a307bef4111140e7b98f3bf68754de22a67b206d6f0`.
All jobs ran through the shared experiment queue with `stable_seconds: 0`.
Changing host conditions were recorded and never used as an admission gate.

## Correctness

The exact-cache referee evaluated 16 dependent decode tokens from the same
sealed imported state on both routes. Candidate and baseline had bit-identical
residuals after every one of 48 layers and identical final token/top-five logit
bit signatures for all 16 tokens.

- baseline: 768 separate attention and 768 separate output evaluations;
- candidate: 640 fused evaluations across the 40 sliding layers, plus 128
  separate attention and 128 separate output evaluations across 8 global
  layers;
- compiler calls during inference: zero;
- correctness receipt SHA-256:
  `035b33c23828b2788c816fa08c63b2846583756f098f0b47cd6a95f0fa23d9a1`.

## Timing

Each timing observation used a fresh exact-cache decoder, one untimed warmup,
an identical state reimport outside the timed interval, and four timed
dependent tokens. Each ordering measured six observations and 24 timed tokens
per route. Decoder construction/cache load was excluded; the complete decode
from embedding through vocabulary ranking was included.

| Ordering | Baseline median, ms/token | Candidate median, ms/token | Baseline / candidate | Baseline endpoint drift | Candidate endpoint drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABBA/BAAB/ABBA | 2130.431 | 1752.507 | 1.2156x | 2.28% | 17.57% |
| BAAB/ABBA/BAAB | 2132.803 | 1767.261 | 1.2068x | 9.97% | 6.06% |

Across both runs, the descriptive pooled medians were 2130.475 ms/token for
baseline and 1763.193 ms/token for candidate, or 1.2083x. This pooled value is
descriptive, not a third independent confirmation.

Pooled per-token phase medians show where the time remains:

| Phase | Baseline, ms | Candidate, ms |
| --- | ---: | ---: |
| QKV | 473.053 | 473.160 |
| separate attention | 457.841 | 74.152 |
| separate output | 455.492 | 79.368 |
| fused attention plus output | 0 | 395.710 |
| FFN | 567.725 | 562.478 |
| vocabulary | 168.905 | 168.266 |
| host | 10.700 | 10.303 |
| total | 2130.467 | 1763.187 |

The relevant comparison is baseline attention plus output (913.333 ms) versus
candidate remaining separate attention/output plus fused work (549.230 ms), a
descriptive 1.663x phase reduction. QKV and FFN are now the two largest
remaining measured phases and did not materially move, as expected.

Both timing receipts report exact output sequences and zero compiler calls.
Receipt SHA-256 values are:

- ABBA: `ac035b490a8e9de6863c1121c7933e3c5387cc6ccc5f5173bd7272851db473ae`;
- reverse BAAB: `734ef8532ba034624b5f7c789cbcaaeaaa243e0a549107ab6a2fe1247e70ef29`.

## Adjudication

The candidate advances beyond the earlier two-token/single-token screen: its
correctness now spans a 16-token dependent trajectory, and its full-route gain
survives two opposite-order, multi-token campaigns. Thermal and competing-
process variation remain visible, especially the first campaign's 17.57%
candidate endpoint drift, but do not reverse the result.

The next ANE work should target QKV and FFN graph/evaluation reduction while
leaving the eight global layers on the separately qualified route. Production
selection still requires an explicit selector review, broader prompt/state
coverage, and checkpoint-level quality evidence; none is inferred here.
