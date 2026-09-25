# Gemma 4 ANE Chunk4 bounded-equivalence qualification

Date: 2026-09-24

## Verdict

`ane-int8-ffn-chunk4` is **correctness-rejected** by the prospective
`rvllm.ane.ffn-layout-equivalence.v1` contract.  It must not advance to timing
or promotion.  This result preserves rather than replaces the earlier
bit-exact-oracle failure.

The run used three pinned real Gemma 4 12B layer-0 FFN inputs.  Plain INT8 and
Chunk4 were each evaluated twice per input.  Both implementations were
bit-exact and finite across their own repeats, so the rejection is not caused
by nondeterminism.  Chunk4 exceeded the predeclared four-ULP bound for material
values on every sample.

## Results

| Input position | Differing lanes | Max absolute | Max material ULP | Relative L2 | Result |
|---:|---:|---:|---:|---:|---|
| 21 | 302 / 3840 | 0.015625 | 13 | 0.000085943 | reject |
| 84 | 190 / 3840 | 0.015625 | 12 | 0.000077627 | reject |
| 652 | 154 / 3840 | 0.031250 | 8 | 0.000061110 | reject |
| aggregate | 646 / 11520 | 0.031250 | 13 | 0.000069272 | reject |

All samples passed finite-output, hybrid absolute/spacing, maximum-absolute,
near-zero ULP/absolute, and relative-L2 checks.  Material-value ULP violations
were 9, 4, and 4 respectively.  The contract requires every gate; passing the
global norm does not waive localized material errors.

The run performed twelve ANE program evaluations, completed all samples before
returning the numerical verdict, and used two compiler calls under the declared
`ReuseOrCompileUpTo(2)` budget.  Queue conditions were AC power, performance
power mode, and nominal thermal state 0.  Conditions were logged without a
stability wait and do not change the correctness verdict.

## Infrastructure incident retained

The preceding q3 job was terminated after 165 seconds blocked in `open(2)` on
a pinned sample under `~/Downloads` when run by the LaunchAgent.  Its queue
receipt remains failed and unreplayed.  Q4 copied the exact same three inputs,
verified their SHA-256 identities, and pinned them inside the authorized
worktree.  Q4 then reached ANE and produced the complete numerical result above.

## Sealed evidence

- Oracle event stream SHA-256: `40a144e2f59a96cf3d9cc624a3ac92581139d30303acd421e1e3cffb7d10e4fc`
- Driver journal SHA-256: `be1ba6343e5a36930cef7944bdc07e511c428c2ec5f5a82ae8cd5bff31038ddc`
- Queue report SHA-256: `b85131b8ed4f647ea0444bfd89fbfff0e94548055a453bea3ab9c37a8e5baacf`
- Test executable SHA-256: `83beabe949b11f67f5e5212d71193b88c2b01dca2040e5debe415593ed904fbb`
- Input manifest SHA-256: `dd6cd2579c5f7bea28d34bde61253146cca3f7c2870562aa0f8d15aaa0477122`

Raw FP16 outputs for both repeats of both layouts are retained under `oracle/`.
The queue report, condition journal, and ANE driver journal are retained beside
this report.

## Next action

Retire the unchanged Chunk4 layout.  A successor must change numerical
grouping or accumulation behavior, receive a new candidate identity, and pass
the same prospective contract before timing.  The stacked INT8 control remains
the qualified baseline.
