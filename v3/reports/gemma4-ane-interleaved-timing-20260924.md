# Gemma 4 ANE interleaved FFN timing screen

Date: 2026-09-24

## Verdict

`static-int8-interleaved-ffn-cached` is **timing-inconclusive** against the
shipping `static-int8-ffn-cached` control.  It is component-correct under the
separate bounded oracle and every timing arm completed through ANE without a
CPU/GPU decode fallback or a compiler call, but this timing sequence does not
establish a speedup.

The robust center of the six per-arm mean FFN times is 153.99 ms/token for the
interleaved candidate and 154.86 ms/token for the stacked control: a nominal
0.56% advantage, far below the observed run-to-run variation.  The candidate
must not be promoted from this evidence.

## Experiment

The preserved queue results are under
`reports/gemma4-ane-interleaved-timing-queue-20260924/run/results/`.
The sequence was:

1. A1 stacked, B1 interleaved, B2 interleaved, A2 stacked
2. B3 interleaved, A3 stacked, A4 stacked, B4 interleaved
3. A5 stacked, B5 interleaved, B6 interleaved, A6 stacked

Each arm contains nine fixed prompt cases and nine measured decode steps per
case (81 token observations).  All twelve queue reports succeeded.  Every
inference report records `ane_execution_verified=true`,
`cpu_or_gpu_decode_fallback=false`, and `ane_compile_budget_used=0`.  Sampled
conditions were AC power, low-power mode enabled, and nominal thermal state 0.
Changing conditions were recorded; the queue did not wait for a stable host.

The table reports arithmetic means across the 81 decode observations in each
arm.  Times are milliseconds per generated token.

| Arm | Plan | Total | FFN | Attention | QKV | Output | Host |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | stacked | 200.680 | 95.175 | 19.378 | 35.322 | 23.611 | 6.221 |
| B1 | interleaved | 228.158 | 110.194 | 22.296 | 38.673 | 26.715 | 7.923 |
| B2 | interleaved | 307.805 | 141.562 | 32.453 | 53.788 | 35.639 | 15.875 |
| A2 | stacked | 461.605 | 182.901 | 61.881 | 82.698 | 68.017 | 25.343 |
| B3 | interleaved | 354.271 | 153.881 | 42.642 | 64.649 | 44.878 | 14.909 |
| A3 | stacked | 365.582 | 144.516 | 45.838 | 65.671 | 48.774 | 24.359 |
| A4 | stacked | 488.512 | 185.191 | 67.986 | 90.051 | 68.501 | 29.066 |
| B4 | interleaved | 406.089 | 165.459 | 52.801 | 77.450 | 54.471 | 19.768 |
| A5 | stacked | 371.189 | 165.211 | 41.863 | 65.628 | 46.449 | 16.965 |
| B5 | interleaved | 343.632 | 154.102 | 39.405 | 62.663 | 44.269 | 12.256 |
| B6 | interleaved | 354.061 | 158.148 | 40.172 | 63.594 | 43.581 | 15.780 |
| A6 | stacked | 221.425 | 104.705 | 21.304 | 39.652 | 26.482 | 6.668 |

## Drift-aware interpretation

The control's per-arm mean total time ranges from 200.680 to 488.512 ms/token;
its FFN time ranges from 95.175 to 185.191 ms/token.  This is not a small-noise
campaign.  It also demonstrates why total-token time is not a sound estimator
of a weight-layout change confined to the FFN.

For each four-arm ABBA block, the geometric candidate/control ratios are:

| Block | Total ratio | FFN ratio | Apparent result |
|---|---:|---:|---|
| 1 | 0.871 | 0.947 | candidate faster |
| 2 | 0.898 | 0.975 | candidate faster |
| 3 | 1.217 | 1.187 | candidate slower |

The sign reversal is decisive evidence of an unresolved time/order effect.
Taking the median of the six arm means per plan reduces that sensitivity:

| Plan | Median arm-mean total | Median arm-mean FFN |
|---|---:|---:|
| stacked control | 368.386 | 154.864 |
| interleaved candidate | 348.846 | 153.991 |
| candidate/control | 0.947 | 0.994 |

Only the FFN comparison is causally close to the candidate change, and its
nominal 0.56% advantage is not distinguishable from this campaign's variance.

## Next gate

Retain the candidate for correctness and future profiling, but do not promote
it.  A useful next timing pass should randomize or counterbalance individual
prompt-level A/B pairs, record the FFN program identity per step, and use enough
repetitions to bound the paired FFN ratio.  Whole-token timing remains a
secondary regression check.  The historical exact Chunk4 oracle failure is
separate evidence and is not weakened or overwritten by this report.
