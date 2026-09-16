# Explicit Fair-state two-token FFN timing

The S2 probe now accepts `--exploratory-fair true` only for the existing
two-token INT8 comparison, strict existing-cache loading, zero compilation
and no driver journal. It refuses the flag with capture, restoration, CPU-only,
stacked or palette modes. Default nominal timing and the shared measurement
module are unchanged.

The new reader requires exactly Fair, known AC/battery and power mode, stable
sampled controls and host endpoints, no reported processor restriction, no
observer error, and no sampling gap over 2.5 seconds. It retains the raw phase's
`sampled_controls_eligible=false` and null nominal comparison stratum. Its own
classification is `exploratory_stable_fair`; it neither pools strata nor
normalizes device time with CPU counters.

The probe also refuses timing if qualification differs bitwise from serial.
Each timed trial checks its final output against the serial qualification
outside the timing interval. It retains the original 128 repetitions of two
useful outputs, with three ABBA/BAAB/ABBA blocks. The complete summary checks
all twelve trials' work counts, ordering, output checks and common stratum.
Repeat drift above 5% for either variant in any block labels the comparison
inconclusive. The ratio remains FFN component wall time; it measures neither
draft acceptance nor full-model throughput.

Job 72 passed all fourteen release probe tests through the experiment worker,
including five tests for Fair eligibility, transitions, gaps, malformed data,
work counts, ordering, cross-block strata and baseline/candidate drift.
Job 73 also completed successfully, with unchanged source pins and no overrun.
The frozen signed executable has SHA-256
`112ad00a44aa860b37846f2bebfc6ff6bd09a5a20bc52dbc243039be506e2ba0`.
Seven parser checks passed before any model/device access: the valid combination
reached a deliberately missing model path; absent batch, capture, restoration,
compilation, driver journaling and CPU-only combinations were refused. Neither
the sentinel report nor journal was created.

Jobs 74/75 now stage strict-cache AC/battery Fair trials after their respective
native-kit Fair comparison analyses. They pin the frozen build receipt, original
CPU-qualified source subset and broader S1/S2 bit-equivalence receipt. The
original nominal jobs 70/71 remain separate and unchanged. The worker is live;
other Cargo/rustc activity still prevents its 120-second quiet launch window.
No Fair S2 device timing result or accepted native-kit ratio exists yet.

Manifests and evidence are under
`gemma4-12b-evidence-20260914/int8-s2-fair-timing-20260916/` and the
`baseline-isolated-v7` queue. Source pins cover the changed probe files and
their existing comparison dependencies; they are not a full transitive
toolchain attestation. Production inference, ANE precision and graph shapes
remain unchanged.

## Short pilot protocol

Before any FFN timing attempt started, job 04 added a separate 30-second quiet
lead-in for the AC/Fair component pilot. The original 120-second jobs 05/06
and post-baseline repetitions remain unchanged. Transient external servers
and compilers observed in `pilot-wait-observations.jsonl` repeatedly reset the
long window. A two-minute lead-in is not itself evidence of constant clocks;
the short protocol instead supplies explicitly conditional component evidence
with the same eight warmups, twelve counterbalanced trials, continuous sampled
contention/controls checks and predeclared 5% repeat-drift ceiling.

This changes only the quiet lead-in, not the numerical gate, cache policy,
power stratum, work counts, runtime observations or drift rule. Its result must
retain the 30-second protocol label. It is not pooled into the separately
predeclared native-kit comparison or represented as fixed-clock performance.

## Completed short pilot: promising, drift-inconclusive

Job 04 completed with unchanged files, no compiler calls, no queue violations
and twelve eligible AC/low-power-on/mode-1/Fair trials. Qualification remained
bit-identical, with sixteen qualification and 2,328 timing evaluations. All
twelve queue observations were ready and reported no competing process.

The three block ratios of two serial calls over one S2 call were 1.8313,
1.7889 and 1.7786; all six adjacent pair ratios favored S2 (1.5754–2.1113).
However, repeat drift for serial/S2 was respectively 4.87%/19.01%,
12.69%/51.03% and 16.52%/2.90%. Every block fails the predeclared 5% ceiling.
The result is therefore `inconclusive_repeat_drift`, not an accepted speedup.
Stable coarse power controls demonstrably did not establish stable latency.
No clock, contention or cache mechanism is inferred from that fact.

An independent read of the raw trials reproduced the ratios, work counts,
phase-control checks and all three drift failures. See
`04-independent-timing-audit.json`. The original longer-lead pilots and
post-baseline repetitions were declared before this result and remain queued;
this attempt is not retried or discarded. Full-model S2 provisioning and draft
integration still require a reproducible target-side saving.

Probe report SHA-256:
`380bfef807fedd0d697e1b7997f54a62fbdde12c24b991ffce8b98c80d87c2e0`.
Queue report SHA-256:
`eecc7a05e70da525973ad0f29123b0a3f0d96e4b4ccaf2b2687838e028e26710`.
