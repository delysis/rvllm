# Apple inference performance measurements

Gemma 4 12B comparisons separate Metal prefill, KV conversion/import, ANE decode,
and preparation. Decode units count actual ANE steps, excluding Metal's first
sampled token. Keep the exact prompt and generated token IDs, context capacity,
weight plan, resident backends, model/config identity, executable and metallib
hashes with every result.

`rvllm_disaggregated_infer --output-dir PATH` records a power-observations JSONL
file, even when model preparation later fails. Each measured phase includes:

- Process CPU cycle/instruction deltas from `proc_pid_rusage(RUSAGE_INFO_V4)`;
  cycles and instructions per work unit, and cycles per instruction. These are
  hardware-backed process accounting, not wall time multiplied by nominal GHz.
  All process threads count, including the observer. Child processes, ANE daemon
  CPU, GPU and ANE execution do not count. Zero/unavailable or reset counters
  produce null metrics rather than fabricated zero work.
- Current AC/battery supply, `pmset` power mode, Foundation Low Power Mode and
  thermal state, plus any CPU/scheduler limits reported by `pmset`. Raw command
  responses are retained. Absence of a recorded limit means unknown, not 100%.
- Wall time and, for Metal prefill, the completed command buffer's
  `GPUEndTime - GPUStartTime`. The latter excludes host encoding and explicit
  waiting, but still reflects device frequency and contention. It is not cycles.

The observer samples approximately once per second. It runs three `pmset`
commands on its own thread; no subprocess runs on the token path. Phase endpoints
also read Foundation state and process counters. In-memory history is bounded to
4096 observations; older/poorly covered phases are ineligible. The JSONL stream
retains the observations on disk. Sampling and counter reads have overhead; use
the same instrumentation in both variants. Changes shorter than a sample interval
can be missed. Nominal thermal state does not prove fixed clocks.

`rvllm_power_probe` checks counter availability without model weights or ANE calls.
On the M4 Max / macOS 15.6 checked on 2026-09-15, process cycles and instructions
worked, and the public Metal API advertised only the `timestamp` counter set with
`GPUTimestamp`. Neither GPU core cycles nor ANE cycles are available through this
qualified path. Do not treat the timestamp's timebase ticks as core cycles.
`powermetrics` frequency/power sampling required administrator privileges on this
machine and was unavailable to the agent; no frequency series is synthesized.

The [pinned ANE timing audit](../reports/gemma4-ane-device-timing-research-20260915.md)
found no demonstrated request-level ANE cycle, frequency or device-time recipe
for this target. Upstream zero counters, unsupported masks and an ABI-conflicting
example do not justify changing the qualified request path. Aggregate engine
residency is not request-active cycles. Retain baseline drift observations
within each alternating block; matching sampled controls alone cannot establish
that clocks or another client's queueing remained constant.

For comparisons:

1. Run the same fixed workload and verify identical outputs. Preserve cold load,
   cache repair and warm execution as distinct measurements. Turn off diagnostic
   per-layer captures and the ANE fsync journal for timing runs.
2. Interleave baseline/candidate runs in ABBA order, reverse the initial order
   across repetitions, and collect at least five valid pairs in each power mode
   of interest. Avoid overlapping builds, model loads and other benchmarks.
3. `rvllm_compare_disaggregated BASELINE_REPORT CANDIDATE_REPORT` rejects missing
   observations, stale samples, sampled transitions, non-nominal thermal state,
   reported CPU restrictions, mismatched power/processor strata, or unlike work.
   A returned ratio is one paired observation, not a statistical speedup claim.
4. Report medians, spread and paired ratios separately for AC and battery/power
   modes. Preserve raw wall time as the user-facing metric. CPU cycles and retired
   instructions explain host work and stalls; they do not correct GPU/ANE time.
   Frequency changes can change memory stalls in cycles as well, so even CPU
   cycles are not universally invariant under DVFS or P/E-core migration.

For the original versus stacked INT8 FFN layout, supply the completed
full-model bit-parity receipt explicitly:

```sh
rvllm_compare_disaggregated baseline/report.json candidate/report.json \
  --stacked-qualification checked/report.json
```

This is a restricted exception to matching `ane_weight_plan`. It accepts only
`static-int8-ffn-cached` versus `static-int8-stacked-ffn-cached`, in either
order. The checked receipt must identify the same checkpoint path/config,
machine/OS, capacity and execution arrangement; match every reference; and
record a comparison in all 48 layers for every decode step. Every timed
prompt/output sequence must occur in that qualification. The comparator
records the qualification file's SHA-256 and labels the layout comparison.
It does not relax the power, work, capture, journal or zero-compile gates.
The checked plan's duplicate work cannot serve as a timing candidate.

Use ordinary text input for these explicit CLI timing runs, since reference
mode also exports prefill KV seed files. Retain identical frozen executable
and Metal library hashes in the experiment manifest. Use the same qualified
user text and output limit for both plans; token IDs remain in their result
receipts. Run repeated warm requests after loading each plan, alternate plan
order across processes, and preserve startup outside warm measurements.

Earlier reports lacking power observations remain historical observations and
correctness evidence. They cannot establish a causal speedup against this series.

## Separately labeled Fair-state exploration

Apple describes Fair as minimally elevated thermals without a required
corrective action; it is not a clock measurement or a fixed throttling factor.
See [Apple's thermal guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/RespondToThermalStateChanges.html).
Requiring nominal state for every exploratory observation was unnecessarily
restrictive. The original nominal-only recorder and comparison gate remain
unchanged; Fair exploration uses a distinct offline analysis path:

```sh
rvllm_compare_disaggregated baseline/report.json candidate/report.json \
  --stacked-qualification checked/report.json --stable-fair-exploratory
```

This reader recomputes stability from raw power observations and endpoints.
It requires thermal state exactly 1 throughout each phase, known/matching
power controls, fresh samples without gaps or transitions, and no reported
CPU/scheduler restrictions. Serious/Critical states remain excluded. It does
not rewrite the original `sampled_controls_eligible: false`, invent clocks,
rescale device time, or pool Fair observations with nominal measurements.
Every result is labeled `exploratory-stable-fair`.

Use counterbalanced blocks, retain bracketing baseline drift, and summarize
processes before pairing; individual requests within one process are not
independent experimental replications. This follows the experimental-design
principle of comparing treatments within observed nuisance-factor strata;
it does not control unobserved clocks or other clients' queues. See
[NIST on blocking](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm).
Any improvement claim is conditional on the measured stratum, never an
estimate of nominal-state or universal speedup.

Source semantics: [Apple XNU Recount](https://github.com/apple-oss-distributions/xnu/blob/main/doc/observability/recount.md),
[Metal GPU start time](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime),
[supported Metal counter sets](https://developer.apple.com/documentation/metal/confirming-which-counters-and-counter-sets-a-gpu-supports).
