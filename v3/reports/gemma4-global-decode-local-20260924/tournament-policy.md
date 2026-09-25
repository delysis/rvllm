# Global D512 decode successive-halving policy

Date: 2026-09-24

This policy replaces the original full Cartesian timing schedule. Correctness,
compile and identity gates still apply to every candidate before any timing.
Timing observations are never deleted, retried because they look unfavorable,
or described as promotion evidence.

## Stages

1. Screen every correctness-qualified candidate at context 256.
2. Rank by absolute candidate mean GPU milliseconds per dispatch. Advance the
   fastest two, plus any candidate within 10% of the second-fastest time.
3. Repeat the same rule at 512, then 1024.
4. Measure 2048 only for the surviving set. Select the fastest candidate plus
   any candidate within 5% for independent confirmation and full-route work.
5. Reserve context 4096 for those finalists; it is not a first-round screen.

Absolute candidate time determines exploratory advancement because every
candidate performs the same sealed operation on the same device. ABBA baseline
ratios and the 5% control-drift gate remain authoritative for performance
qualification. A candidate can advance from a drift-inconclusive exploratory
cell, but it cannot be promoted from that cell.

Ties deliberately widen rather than narrow the next stage. If a receipt has
the wrong identity, incomplete samples, failed correctness, mutated guards, or
non-equivalent work, it is invalid and cannot advance regardless of speed.

The generator now emits only the L256 screen by default. Later jobs require an
explicit length and explicit candidate names, for example:

```text
rvllm-global-decode-jobs timing-jobs ROOT 512 \
  metal-global-d512-r16p128t128 metal-global-d512-r16p64t128
```

This makes advancement a reviewable input rather than an accidental lexical
property of a pre-generated Cartesian queue.
