# Stacked INT8 FFN: exploratory comparison in stable Fair conditions

This is separate from nominal-state qualification. The user wants comparisons
that account for changing laptop power/processor state. Apple defines Fair as
minimally elevated thermals with no required corrective action; it is not a
measured clock or a fixed throttling factor. The earlier blanket performance
block was too restrictive for exploration. Sources:
[Apple thermal guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/RespondToThermalStateChanges.html),
[Apple GPU profiling guidance](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance).

The nominal-only recorder and default comparison gate remain unchanged.
The explicit `--stable-fair-exploratory` reader reconstructs stability from
raw phase samples/endpoints, requires thermal state exactly 1 with matching
power controls, and rejects transitions, stale/gapped samples, reported
restrictions, or observer errors. It never rewrites nominal eligibility or
uses CPU cycles to normalize device time. Five comparator tests passed;
the code also checks KV import measurements along with prefill and decode.

The method uses homogeneous sampled operating strata and counterbalanced
blocks, following [NIST's blocking principle](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm).
The block order is predetermined, not randomized. Unobserved clocks and
other clients' queues remain uncontrolled, so the result is a conditional
observation, not universal device performance or a nominal-state estimate.

## Predeclared experiment

Evidence directory: `gemma4-12b-evidence-20260914/int8-stacked-fair-20260916/`.
`plan.json` was written before the first timing invocation. It pins the same
frozen inference binary used by the successful 624-comparison check:
`aa015ca92c735e36c1da17d258031a4c743823a4b29feacb5ba46cbbc9f3c605`.
Model revision, Metal library and INT8 weights are unchanged. A is the original
FFN layout; B is stacked. Both use Metal prefill and ANE decode, and strict
zero-compile loading. No driver journal or layer/seed export is enabled.

The fixed copy workload has 84 input tokens and ten expected output IDs,
including EOS, with nine ANE steps. Each process executes two warmups and
seven measured requests. The planned order is ABBA / BAAB / ABBA. Statistics
use process medians, then six adjacent process pairs; individual requests are
not independent replications. Repeated outer-plan median decode drift above
5% makes that block inconclusive and stops the sequence. Any cache/accelerator
error or ineligible sampled state also stops timing without automatic repair
or retry. All partial/failed attempts remain evidence.

The fresh preflight observed battery power, low-power off, power mode 0 and
Fair thermals. No power setting was changed. Other power modes and nominal
results will not be pooled with this stratum. The sum of prefill, KV import and
decode phase times excludes streaming/report I/O and is labeled a phase sum,
not complete client-observed latency. No default promotion follows automatically
from this exploratory experiment.

## Initial attempt and separate maintenance

The initial A process exited before decoding because a compiled ANE graph
was absent. It made no inference progress and supplies no timing result.
The frozen executable and Metal library hashes still match the qualified
ones. The timing sequence was stopped, and its logs/power observations are
preserved as `trial-00-A.*` and `trial-00-A/`.

Bounded cache preparation completed separately under `maintenance/`. The
default batch repaired 67 missing graphs; stacked FFN preparation repaired 38.
All 210 visited graphs loaded and returned successful unloads, with zero
evaluations, no staging residue and unchanged boot. Maintenance costs are
not part of a warm-throughput claim.

The post-maintenance power probe observed AC, low-power mode on, power mode 1
and nominal thermals. Consequently this Fair sequence was not restarted and
has no measured result. A separate, predeclared nominal-state sequence lives
under `int8-stacked-ac-low-power-20260916/`; no data are pooled across these
power states. No OS power setting was changed by this work.
