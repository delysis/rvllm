# Gemma 4 ANE all-sliding fused full-route adjudication

Date: 2026-09-25

## Decision

`all-sliding-fused-cached` is a **confirmed prospective full-route winner**.
It remains default-off pending longer dependent-token stability and a clean
production-selector review, but it should advance ahead of the layer-0-only arm.

| Campaign | Baseline median | Candidate median | Baseline / candidate | Baseline drift | Candidate drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABBA/BAAB/ABBA | 2070.572 ms/token | 1724.747 ms/token | 1.2005x | 7.30% | 7.11% |
| BAAB/ABBA/BAAB | 2150.135 ms/token | 1821.671 ms/token | 1.1803x | 13.05% | 12.35% |

Both independently ordered campaigns place every candidate median below its
baseline median. Across all 12 observations per route, pooled medians were
2099.148 ms/token baseline and 1767.457 ms/token candidate: a descriptive
1.1877x ratio, or 15.8% lower latency. The independently ordered campaign
ratios, not the pooled ratio, are the primary evidence.

## Sealed implementation and workload

- Frozen executable:
  `target/release/rvllm-runtime-tests-fused-all-sliding-v1`
- Executable SHA-256:
  `9677fa373fafaf4daca7723e3808ff6307c1b8062594b54bed8d73e9e8cf7001`
- Weight plan: `static-int8-ffn-cached`
- Candidate plan: `all-sliding-fused-cached`
- Candidate geometry: 40 sliding layers fused; 8 global layers remain on the
  separate attention and output-projection route.
- Each timing observation used a fresh exact-cache decoder, one equal untimed
  warmup, and an identical sealed prefill snapshot imported outside timing.
- Timed scope: one complete token from embedding through vocabulary ranking.
- Both campaigns ran on AC power, power mode 2, sampled thermal state 0. These
  are recorded strata, not stability admission gates.

Receipt SHA-256 identities:

- two-token correctness:
  `6f1e5a277519894c4a92c090b8b9ec5dca8a999ec70e52a3c9423461312840aa`
- ABBA timing:
  `81cc403ad0964f4455205c4de44f7207f990f92454088c1b8ce62a3d18bedc63`
- reverse-order BAAB timing:
  `e209f68602ff221dfd333f483c56e72bfdef83a11f4dd92dba43b70125f26d92`

## Correctness and dispatch evidence

The exact-cache two-dependent-token referee proved:

- exact FP16 residual bits after every one of 48 layers for both tokens;
- exact final token and top-five score bits;
- tokens 3730 at position 84 and 37423 at position 85 on both routes;
- baseline: 0 fused, 96 separate attention and 96 separate output evaluations;
- candidate: 80 fused, 16 separate attention and 16 separate output evaluations;
- exact fused layer set:
  0-4, 6-10, 12-16, 18-22, 24-28, 30-34, 36-40 and 42-46;
- zero compiler calls during strict load and inference.

Each timing campaign additionally recorded 480 fused evaluations and 96 each
of separate attention/output evaluations, exact output signatures in all arms,
and zero compiler calls.

## Phase attribution

Pooled median phase times across both campaigns:

| Phase | Baseline | Candidate |
| --- | ---: | ---: |
| QKV | 472.638 ms | 478.760 ms |
| Separate attention | 445.744 ms | 73.634 ms |
| Separate output projection | 459.848 ms | 77.085 ms |
| Fused sliding attention + output | 0 ms | 396.613 ms |
| FFN | 555.148 ms | 564.797 ms |
| Vocabulary | 168.554 ms | 172.011 ms |
| Host remainder | 9.448 ms | 9.168 ms |

The comparable attention-plus-output portion falls from approximately
905.593 ms to 547.332 ms at the pooled medians, about 1.65x. These internal
categories are diagnostic rather than independently randomized microbenchmarks;
complete wall time remains authoritative.

## Epistemic boundary and next gates

This establishes a large, repeatable one-token full-route effect and exact
two-token arithmetic. It does not yet establish production readiness, long-run
KV stability, throughput under a device-resident token loop, or quality for a
different checkpoint/quantization package.

Advance with:

1. a 10- to 32-token dependent decode referee comparing every final token and
   sampled intermediate residuals, with exact route counts and zero compilation;
2. repeated complete-route timing over bounded multi-token batches so load and
   snapshot import remain excluded while steady-state behavior is measured;
3. a selector review proving the default remains `Separate` and missing fused
   graphs fail closed without fallback;
4. investigation of the now-largest FFN phase and QKV, plus global-layer
   attention/output as a separate geometry rather than extending the sliding
   graph by assumption.

The one-layer arm remains useful integration evidence but is superseded as the
performance candidate: it was full-route timing-inconclusive, whereas this
all-sliding arm wins in both orderings by 18-20% in baseline/candidate ratio.

