# Global D512 decode successive-halving result

Date: 2026-09-24

Campaign: `g4decode20260924v1`

Device: Apple M4 Max / Apple9

This is retained exploratory operator evidence, not promotion evidence. Every
candidate first passed its compiled device oracle. Each timing cell contains
five warmups per arm and five complete ABBA blocks with 100 dispatches per
sample. Absolute B-arm GPU time determines tournament advancement. The 5%
baseline-control drift gate remains authoritative for performance claims and
was not relaxed; drift-invalid cells are explicitly retained as inconclusive.

## Tournament

| Stage | Candidate | Candidate ms/dispatch | Control drift | Advanced |
| ---: | --- | ---: | ---: | :---: |
| 256 | `metal-global-d512-r8p128t128` | 4.525 | 103.82% | yes |
| 256 | `metal-global-d512-r16p128t128` | 4.824 | 0.26% | yes |
| 256 | `metal-global-d512-r8p64t128` | 4.918 | 112.76% | yes |
| 256 | `metal-global-d512-r8p64t64` | 6.938 | 137.72% | no |
| 256 | `metal-global-d512-r16p64t128` | 7.065 | 94.74% | no |
| 256 | `metal-global-d512-r8p128t64` | 8.342 | 102.12% | no |
| 256 | `metal-global-d512-r16p64t64` | 14.018 | 108.90% | no |
| 256 | `metal-global-d512-r16p128t64` | 15.213 | 105.12% | no |
| 512 | `metal-global-d512-r8p128t128` | 7.840 | 90.89% | yes |
| 512 | `metal-global-d512-r8p64t128` | 8.293 | 68.26% | yes |
| 512 | `metal-global-d512-r16p128t128` | 9.009 | 4.92% | yes |
| 1024 | `metal-global-d512-r8p128t128` | 15.108 | 53.57% | yes |
| 1024 | `metal-global-d512-r8p64t128` | 16.473 | 68.25% | yes |
| 1024 | `metal-global-d512-r16p128t128` | 30.178 | 125.11% | no |
| 2048 | `metal-global-d512-r8p64t128` | 23.399 | 58.52% | finalist |
| 2048 | `metal-global-d512-r8p128t128` | 28.377 | 14.45% | no |

The 256/512/1024 rule retains the fastest two plus every candidate within 10%
of second place. The 2048 rule retains the fastest plus every candidate within
5% of it. Results that existed from the earlier Cartesian schedule but were no
longer in the active cohort were not used to change advancement decisions.

At 2048, `r8p64t128` is 1.692x faster than the previous
`r16p128t128` leader (23.399 ms versus 39.593 ms), a 40.9% time reduction.
It remains approximately 15.7x slower than the matched 1.489 ms isolated MLX
BF16 attention operator measurement. The absolute improvement is useful, but
the remaining gap and one-threadgroup `[1,1,1]` launch topology mean another
unsplit tile sweep is not the appropriate next round. The next design round
should be split-KV/multi-threadgroup attention with an explicit reduction.

Context 4096 remains deferred. Context 2048 is already sufficient to expose
the architectural gap, and the user requested economizing until a materially
better finalist exists.

## Receipt pins for the two L2048 finalists

Each line is `native/abba.json`, immutable `job.json`, then queue `report.json`.

- `r8p128t128`: `5a3fb06592d9e38113e08a5ee00c4bf2faf29c6bfc1512fe2876c8b11f06ffa6`, `14ecfff649f5a12a656285921439b680b9578359553d9ed7a88ad479d028c610`, `3891a263de6f7878237b4fd345f2c33ab450b20b89176d385c3ebed67b2e98fe`
- `r8p64t128`: `864b43715cd34babdbd511c5c97f2fe01c103b534b6f1fee96f3709c861a8d03`, `82fc976d6253f03cf0140cbb96bf26baca552b931bb37f1ba6bec230463cfb8d`, `bb1a0e400d36e044f8672c61567fe2119adb61c14dad79718b8426f24183cfe9`

The full receipts, all samples, conditions journals, generated Metal source,
compiled identity, and queue reports remain under `queue/results/`. No sample
was deleted or retried because it was unfavorable.

## Automation boundary

Commit `499f6a31` adds queue-owned successive-halving for new campaigns. A
dependency-bound preparation job validates exact stage receipts, writes an
immutable advancement receipt, and submits only selected next-stage jobs
through the existing queue. This historical campaign predates the required
pinned queue-runner identity, so its pre-generated manifests were advanced
manually under the same checked-in policy. New campaigns require no observing
agent in the advancement loop.
