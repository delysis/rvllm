# Gemma 4 ANE layer-0 fused full-route timing adjudication

Date: 2026-09-25

## Decision

The default-off layer-0 fused attention-to-output route is **correctness-qualified
but full-route timing-inconclusive**. Do not promote it from these runs.

The component evidence remains strong: fused layer-0 attention plus output
projection was about 1.96x faster than the separate pair in two independently
ordered component campaigns. In the complete 48-layer decode route, however,
replacing only one pair did not produce a repeatable end-to-end improvement:

| Campaign | Baseline median | Candidate median | Baseline / candidate | Baseline range drift | Candidate range drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABBA/BAAB/ABBA | 2107.921 ms/token | 2053.065 ms/token | 1.0267x | 18.13% | 4.29% |
| BAAB/ABBA/BAAB | 2065.441 ms/token | 2076.822 ms/token | 0.9945x | 7.25% | 3.67% |

Across all 12 observations per arm, the median was 2071.096 ms/token for the
baseline and 2069.254 ms/token for the candidate, a nominal 0.09% difference.
That pooled number is descriptive only: it is far below observed variation and
must not be treated as a win.

## What ran

- Frozen executable:
  `target/release/rvllm-runtime-tests-fused-full-route-timing-v3`
- Executable SHA-256:
  `cf651702b1bbaaaf815dd360d4a0a18ac913b8e52d6bc6215b0a9070fc47e2c8`
- Exact ignored test:
  `gemma_ane_decode::tests::hardware_layer0_fused_attention_output_full_route_abba_timing`
- Workload: one complete decode token from the captured prefill state through
  vocabulary ranking, six measured observations per route per campaign.
- Each observation used a fresh exact-cache decoder, one equal untimed warmup,
  and a fresh import of the identical sealed prefill snapshot. Decoder loading,
  snapshot import and warmup were outside the timed interval.
- The single-live-decoder design is required because the private runtime grants
  a deterministic model-directory lease and rejects two concurrent instances
  of the same executable/model identity.
- Both campaigns ran on AC power, power mode 2, sampled thermal state 0. These
  observations describe conditions; they are not a stability admission gate.

Receipt identities:

- ABBA receipt SHA-256:
  `04d7d270874ff9081ec235574eda70f7f7d95469dffb3fafcdf852733f8e88da`
- BAAB receipt SHA-256:
  `74368a1e1762fee4cf47424e125e865afd73fa2eb69caa7a474b358c358d2a8c`

## Correctness and route evidence

Both campaigns passed all of the following:

- exact predicted token and exact top-five score bits in every warmup and timed
  observation;
- baseline route: 0 fused evaluations, 576 separate attention evaluations and
  576 separate output evaluations;
- candidate route: 12 fused layer-0 evaluations, 564 separate attention
  evaluations and 564 separate output evaluations;
- zero compiler calls during all exact-cache loads, warmups and measurements;
- successful queue completion with no thermal-stability waiting.

The prior v1 timing manifests are invalid timing evidence because their binary
did not contain the feature-gated test and therefore reported `running 0 tests`.
The first real v2 timing attempt is retained as a failed referee receipt: it
proved that two simultaneous decoders collide on the private runtime's model
directory. Neither failure is kernel-performance evidence.

## Where the complete route spends time

Pooled median internal phase times across the two campaigns were:

| Phase | Baseline | Layer-0 fused candidate |
| --- | ---: | ---: |
| QKV | 462.538 ms | 456.512 ms |
| Attention (remaining separate layers) | 443.974 ms | 439.305 ms |
| Output projection (remaining separate layers) | 449.397 ms | 433.561 ms |
| Fused layer-0 attention + output | 0 ms | 9.488 ms |
| FFN | 552.615 ms | 550.821 ms |
| Vocabulary | 164.682 ms | 167.889 ms |
| Host remainder | 9.741 ms | 9.638 ms |

These categories are useful attribution, not independent microbenchmarks: ANE
driver scheduling, cache state and measurement boundaries can shift time among
adjacent phases. The complete wall interval is the authoritative end-to-end
quantity.

## Consequence for the next tournament

The one-layer experiment establishes route correctness and the production
integration boundary, but it is intentionally too small to validate a model-
level speedup. The next ANE candidate should fuse attention plus output for all
eligible sliding-attention layers behind the same default-off selector, while
retaining separate handling for global layers. Qualification order:

1. provision the exact executable in bounded stages under the 48-call process
   budget;
2. exact two-token and longer dependent-token correctness, including newest
   K/V visibility and zero inference compilation;
3. route evidence proving the expected fused and separate evaluation counts;
4. component timing for representative sliding and global layers;
5. complete-route ABBA and reverse-order confirmation using fresh state per
   observation;
6. only then consider a production default change.

FFN is the largest single median phase, while QKV, attention and output are all
material. Therefore the fused-attention expansion and the INT8/LUT4/native-
low-bit FFN tournament should proceed in parallel; the present result does not
justify narrowing the campaign to attention alone.

