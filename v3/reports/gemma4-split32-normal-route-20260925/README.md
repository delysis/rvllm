# Split-32 normal-route qualification

This is an explicitly selected research route. Shipping defaults are unchanged.
The candidate uses 32 dynamic visible-prefix partitions and a complete FP32
sufficient-statistic merge for Gemma 4 global decode at exactly 16 query heads,
one KV head, D512, BF16 cache/output and scale 1.0.

`run-one.sh` executes two identical captured-token cases through the normal
Gemma model route. It rejects missing candidate dispatch, compilation during
the measured inference cases, or non-repeatable output tokens. The native
operator oracle separately covers independent FP64 comparison, BF16 rounding,
holes, tails, newest-K/V visibility, arena guards, and repeated use.

The eight job manifests are deliberately staged here rather than inserted into
the live queue. `stable_seconds` is zero, thermal state is not gated, and the
quiet-process list is empty. The existing v1 queue schema requires either AC or
battery as a source predicate; these manifests retain AC as that prerequisite.

`smoke-L256-summary.json` preserves the compact evidence from a completed real
Gemma 4 12B BF16 normal-route smoke. The raw reports were intentionally not
checked in; their hashes and ephemeral paths are recorded. Preparation compiled
one library and 54 pipelines, while both inference cases recorded zero compiler
calls and 16 partial plus 16 merge dispatches each. The smoke is route evidence,
not a speed comparison; the staged candidate/control jobs provide that next
measurement.

## Completed normal-route campaign

The staged campaign was executed through the existing experiment queue in
shortest-context-first order. The L512, L1024, and L2048 pairs were submitted
only after the preceding pair retained exact output and a plausible candidate
signal. Every arm exited successfully, left its pinned inputs unchanged,
compiled no library or pipeline during inference, and produced identical token
IDs across the control, candidate, and repeated cases. Each control case
recorded exactly 16 split-matrix research dispatches. Each candidate case
recorded exactly 16 split-32 partial and 16 split-32 merge dispatches.

| Context | Control decode ms | Split-32 decode ms | Session speedups | Profile-median speedup | Decision |
|---:|---:|---:|---:|---:|---|
| 256 | 745.359 / 729.342 | 648.565 / 391.863 | 1.149x / 1.861x | 0.967x | contradictory |
| 512 | 993.260 / 886.426 | 978.889 / 869.883 | 1.015x / 1.019x | 1.017x | below 5% margin |
| 1024 | 1330.547 / 1248.162 | 1806.287 / 1080.298 | 0.737x / 1.155x | 1.113x | direction reversal |
| 2048 | 779.656 / 784.985 | 981.366 / 1149.849 | 0.794x / 0.683x | 1.140x | contradictory |

Disposition: **not promotable**. Split-32 is a useful research arm and remains
correctness-qualified, but it is not a production selector candidate on this
evidence. The separately executed session cases and three-sample profile
medians disagree in direction at L256 and L2048, while the L1024 session
repeats disagree with each other. This exposes material cross-process and run
order variance. The split-matrix route remains the stable qualified
implementation; a subsequent full-route comparison must counterbalance route
order within one queue job before making a speed claim.

The L1024 control queue receipt logged one missing activity observation and
therefore marked `sampled_conditions_eligible=false`; it still completed as an
exploratory timing job, as required by the campaign policy. All other arms had
eligible sampled conditions. This condition observation does not rescue the
candidate: the clean L2048 session cases regress while their companion profile
median improves, so neither may be selected as the favorable truth. Conditions
were recorded rather than used as a thermal-stability wait gate.

Raw session and profile JSON is retained under `results/`. Exact queue reports
and condition journals are retained under `queue-receipts/`; no failed or
unfavorable observation was discarded.

`run-abba.sh` and `jobs/abba-L256.json` define the corrective experiment:
control/candidate/candidate/control inside one queue job, two identical cases
per process, one profile sample, exact route checks, and a summary computed
from all four observations per arm. It is a fresh experiment, not a
reinterpretation of the evidence above.
