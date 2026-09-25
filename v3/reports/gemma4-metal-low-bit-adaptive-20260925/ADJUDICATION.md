# Gemma 4 Metal low-bit vector adjudication

Scope: projection-operator evidence only. All 14 vector jobs validated. The table uses the nine-block confirmation receipt; `dense x` is dense-BF16 median divided by candidate median. Values from the earlier direct N4/N8 campaign are shown only where that campaign contains the same cell. Timings from different jobs are not paired evidence, so cross-campaign absolute comparisons are diagnostic rather than promotion evidence.

| role | fmt | M | vector ms | dense ms | dense x | prior direct evidence | disposition |
|---|---|---:|---:|---:|---:|---|---|
| Q | W4 | 1 | 0.1695 | 0.2743 | 1.619 | none | vector win |
| Q | W4 | 4 | 0.2365 | 0.5859 | 2.477 | none | vector win |
| Q | W8 | 1 | 0.2452 | 0.2745 | 1.120 | none | tentative vector win |
| Q | W8 | 4 | 0.3682 | 0.5333 | 1.448 | none | vector win |
| K | W4 | 1 | 0.1507 | 0.1760 | 1.168 | none | vector win |
| K | W4 | 4 | 0.1935 | 0.3270 | 1.690 | none | vector win |
| K | W8 | 1 | 0.2496 | 0.1987 | 0.796 | none | loss; try N4 selector |
| K | W8 | 4 | 0.2474 | 0.3465 | 1.400 | N4 beat N8 in both orders (0.6019/0.6011 vs 0.6698/0.7122 ms) | vector wins dense, but N4 remains plausible |
| V | W4 | 1 | 0.1442 | 0.1653 | 1.146 | none | vector win |
| V | W4 | 4 | 0.1758 | 0.3098 | 1.762 | N4/N8 had no 5% winner | vector win |
| V | W8 | 1 | 0.2212 | 0.1671 | 0.755 | none | loss; try N4 selector |
| V | W8 | 4 | 0.2195 | 0.2995 | 1.364 | none | vector win |
| O | W4 | 1 | 1.1743 | 1.3246 | 1.128 | N4 stable winner over N8 (0.4610/0.3949 vs 0.6893/0.6009 ms) | keep N4 incumbent; vector not competitive cross-campaign |
| O | W4 | 4 | 0.5527 | 1.0752 | 1.945 | N4 stable winner over N8 (0.9724/1.0100 vs 1.1381/1.1487 ms) | needs paired N4/vector before selection |
| O | W8 | 1 | 0.9617 | 0.5074 | 0.528 | none | severe loss; try N4 selector |
| O | W8 | 4 | 0.4721 | 0.6997 | 1.482 | none | vector win, but screen was unstable |
| gate | W4 | 1 | 0.2688 | 0.7239 | 2.693 | none | vector win |
| gate | W4 | 4 | 0.7945 | 3.6767 | 4.628 | none | vector win |
| gate | W8 | 1 | 0.3906 | 0.6587 | 1.686 | none | vector win |
| gate | W8 | 4 | 1.0384 | 2.2804 | 2.196 | N4/N8 cross-order drift | vector win |
| up | W4 | 1 | 0.2330 | 0.6726 | 2.887 | none | vector win |
| up | W4 | 4 | 0.6534 | 2.2964 | 3.515 | none | vector win |
| up | W8 | 1 | 0.3728 | 0.6697 | 1.797 | N8 stable winner over N4 (0.9155/0.9740 vs 1.0354/1.0528 ms) | keep N8 incumbent pending paired comparison |
| up | W8 | 4 | 1.0165 | 2.2645 | 2.228 | N4/N8 cross-order drift | vector win |
| down | W4 | 1 | 2.9692 | 1.8275 | 0.616 | N4 faster than N8 but absolute medians drifted (1.0968/0.5676 vs 1.9136/0.9031 ms) | severe loss; use N4 selector and remeasure |
| down | W4 | 4 | 2.4320 | 13.0809 | 5.379 | N4/N8 cross-order drift | apparent win, unstable across jobs |
| down | W8 | 1 | 2.1794 | 1.7321 | 0.795 | N4 faster than N8 but absolute medians drifted (0.5364/0.7225 vs 0.9994/1.4356 ms) | loss; use N4 selector and remeasure |
| down | W8 | 4 | 2.5987 | 3.5343 | 1.360 | N4/N8 cross-order drift | apparent vector win, unstable across jobs |

## Evidence-driven next arm

The common failure is W8 vector-K4 at M=1: reducing the K-loop iterations did not compensate for its extra per-thread scalar work, and K/V/O/down all lose to dense. W4 down M=1 shows the same long-K failure despite packed-byte reuse. The smallest defensible next candidate is therefore a **default-off selector**, not another shader: choose the existing N4 schedule for W8 M=1 K/V/O/down and W4 M=1 down; retain the vector schedule elsewhere. This also preserves the established N4 W4-output and N8 W8-up incumbents as explicit evidence constraints rather than overwriting them.

Ten sealed jobs cover only the five plausible recovery cells, in both ABBA and reverse BAAB order. Each job checks an independent BF16 CPU reference, guard bytes, rejected alias/no-dispatch behavior, exact repeated output bits, and exact dispatch counts. Changing thermal/process conditions are recorded and never gate execution.

## Adaptive-selector confirmation

All ten sealed jobs completed successfully under the persistent queue. Every arm passed its independent BF16 reference, guard, rejected-dispatch, repeated-bit and exact-dispatch checks. The timings below compare the selected N4 kernel against the typed-BF16 native control in the same receipt; `native x` is native median divided by candidate median.

| role | fmt | order | adaptive ms | native ms | native x | disposition |
|---|---|---|---:|---:|---:|---|
| K | W8 | ABBA | 0.3666 | 0.3720 | 1.015 | inconclusive |
| K | W8 | BAAB | 0.6502 | 0.5786 | 0.890 | reject selector change |
| V | W8 | ABBA | 0.4610 | 0.4575 | 0.992 | inconclusive |
| V | W8 | BAAB | 0.5125 | 0.5845 | 1.140 | order-sensitive; no promotion |
| O | W8 | ABBA | 0.5382 | 0.8353 | 1.552 | prospective winner |
| O | W8 | BAAB | 0.3695 | 0.5918 | 1.601 | prospective winner |
| down | W8 | ABBA | 1.6340 | 2.5726 | 1.574 | prospective winner |
| down | W8 | BAAB | 1.3104 | 2.0835 | 1.590 | prospective winner |
| down | W4 | ABBA | 1.1110 | 2.1336 | 1.920 | prospective winner |
| down | W4 | BAAB | 1.0649 | 1.8578 | 1.745 | prospective winner |

The evidence supports N4 as a prospective M=1 selector for W8 output and W8/W4 down projections. It does not support changing W8 K or V selection: K reverses into a loss, while V straddles parity and changes materially by order. This remains operator evidence, so the three winners require production-route dispatch proof and end-to-end confirmation before any shipping default changes.

## Limitations

- Confirmation versus screen absolute medians drift by more than 20% in several cells, especially down projections and W8 O M=4. Those cells are not stable winners.
- The CPU numerical gate is bounded BF16 error against a sequential FP32 reference; exactness refers to dispatch accounting and repeated device output bits, not bit equality to a differently associated CPU reduction.
- No full-route, checkpoint-quality, generated-ISA/resource, MLX, or shipping-selector conclusion follows from these operator receipts.
