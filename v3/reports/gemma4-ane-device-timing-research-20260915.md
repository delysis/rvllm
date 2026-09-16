# ANE execution timing and DVFS measurement: source audit

Date: 2026-09-15. Target: rvllm's existing one-input/one-output `_ANEInMemoryModel` path, M4 Max, macOS 15.6 / 24G84, ANE 8.600.2. Research only: no hardware, selectors, builds, cache access, power changes, or product edits.

**Recommendation:** retain synchronous evaluate-call wall time as the qualified ANE metric. No inspected implementation establishes reliable request-level device time, active cycles, or clock frequency on this target. Current primary evidence is stronger than an unfinished API recipe: a maintained runtime explicitly reports zero counters and deprecates its enabling option. Do not spend the next FFN experiment on a guessed stats mask or treat a zero result as zero device time.

## What the sources actually establish

| Source and immutable revision | Evidence | Limit for rvllm |
|---|---|---|
| tmc/apple `872b4b8c75cb24617d606e0484c1db1e2948f01e` | `CompileOptions.PerfStatsMask` is deprecated. Maintainer reports three evaluations on 2026-08-15 returning zero `HwExecutionTime`, `PerfStats`, and `PerfStatsArray` at masks `0xF` and all bits, including package-backed models. | Negative upstream observation, not a local M4 result; exact test hardware is not specified in this comment. Wider-mask semantics are explicitly unverified, but the `0xF` observation remains. [types.go:96–114](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/x/ane/types.go#L96-L114). |
| maderix/ANE `d91c9845c0784dec7753048954fc6d0e8411fe29` | M5/macOS 26.3 probe discovers stats properties, but `alloc/init` returns nil. Its mask-enabling explanation is a hypothesis. | Neither valid populated stats nor a verified frequency measurement. The later recipe and QoS/fixed-frequency inference exceed the demonstrated evidence. [results:27–59](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/training/m5result.md#L27-L59). |
| PMetal `5aa77a08a7d1789256ce11d062db5296eae21417` | `evaluate_with_stats` exposes `hw_execution_time_ns`. It builds a stats object and rereads it after evaluation. | Factory ABI conflict, silent zero fallback, and no mask setter in the inspected runtime; its latency probe prints timing without asserting a positive hardware value. See below. |
| ANEForge `caeef8edf13b9ec7a3338826daaa27f99e1663d1` | Optimizer measures `time.perf_counter()` around the invocation. | Host elapsed time, not a cycle/frequency counter. Its distinct e5rt route adds no established measurement capability here. [optimizer:125–139](https://github.com/sbryngelson/ANEForge/blob/caeef8edf13b9ec7a3338826daaa27f99e1663d1/aneforge/_optimize.py#L125-L139). |

The maderix probe uses one input/output, but ignores compile/load return values and passes an `alloc/init` object to `perfStats:`; it is discovery code, not a robust instrumentation implementation. [test_perf_stats.m:154–182](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/training/test_perf_stats.m#L154-L182).

## Callable names are insufficient evidence

The tmc generated bindings expose `hwExecutionTime` as `uint64`, raw `perfCounterData`/`pStatsRawData`, `performanceCounters`, `stringForPerfCounter:`, and `driverMaskForANEFMask:`. The factory `statsWithHardwareExecutionNS:` accepts a **scalar u64** in these bindings; creating a value from supplied nanoseconds is not itself measurement. Raw buffers lack a qualified M4 counter schema. [stats binding:125–187](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/private/appleneuralengine/ane_performance_stats.gen.go#L125-L187).

PMetal instead passes an **NSNumber pointer** to that factory, then reads the original object. This conflicts with the binding ABI and needs exact installed-framework signature evidence before adoption. Its missing-class/factory cases return default zero. Its own comment says a mask is needed, but the runtime never sets one. The ignored benchmark accepts and prints zero without a hardware-time validity assertion. These issues prevent treating the advertised API as working proof. [factory/readback:870–914](https://github.com/Epistates/pmetal/blob/5aa77a08a7d1789256ce11d062db5296eae21417/crates/pmetal-metal/src/ane/runtime.rs#L870-L914), [probe:1438–1488](https://github.com/Epistates/pmetal/blob/5aa77a08a7d1789256ce11d062db5296eae21417/crates/pmetal-metal/src/ane/runtime.rs#L1438-L1488).

tmc's implementation is more careful: scalar-zero factory, then request `PerfStatsArray`/`PerfStats` readback, plus raw-data collection. Nevertheless, its current high-level documentation reports the counters unavailable. Older indexed package documentation promising full package-model counters is stale relative to this pin. [eval.go](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/x/ane/telemetry/eval.go), [current doc.go:77–92](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/x/ane/doc.go#L77-L92).

## Why counters may be blocked, and what aggregate telemetry could mean

Bryngelson's primary reverse-engineering account reports daemon clearing of the stats mask for third-party clients and load rejection when a forced mask requires an absent profiling section. It separates M1-measured failures, decompiled structures, and M5-measured daemon behavior. This supports the negative recommendation; it does not independently prove the exact M4/24G84 mechanism. No hook, internal profiling mode, or entitlement workaround is proposed. [arXiv:2606.22283v1, §33.3](https://arxiv.org/html/2606.22283v1#Ch33.S3).

That study describes whole-engine energy, DRAM bytes and clock-state/frequency-point residency, including `SoC Stats / Events / SOC0_ANE_F1,F2`, and a 24 MHz residency timebase. These are aggregate state durations, not ANE active compute cycles or per-request timestamps. Its firmware MMIO helper is not a callable ordinary-client API; root signposts are a separate route. Evidence is M1/M5, not this laptop. [§33.2–33.3](https://arxiv.org/html/2606.22283v1#Ch33.S2).

Portability is unresolved: tmc reports that on macOS 26.6.1, unprivileged code could enumerate DRAM channels but could not sample them, explicitly failing to reproduce the paper's access claim. Its telemetry documentation also references `StartDRAM` and `x/powersample`, absent from the inspected pinned tree; that prose is not a ready implementation to transplant. [telemetry/doc.go:16–26](https://github.com/tmc/apple/blob/872b4b8c75cb24617d606e0484c1db1e2948f01e/x/ane/telemetry/doc.go#L16-L26).

Apple's public `MLComputePlan.Cost.weight` is a **predicted fraction of total model work**, between zero and one. It is neither elapsed time nor measured cycles. [Apple documentation](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n/cost).

## Concrete decision for the FFN work

rvllm currently passes null `perfStats` with exactly one input/output at [in_memory.rs:479–497](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:479), then synchronously evaluates at [587–603](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-ane-sys/src/in_memory.rs:587). Stats metadata would not inherently add a graph input, but no reliable setup is established; preserve this qualified ABI.

Use paired, rotating-order baseline/candidate measurements with the same resident graphs, input, repetition count, and separate I/O/evaluate scopes. Bracket each candidate block with the unchanged baseline; retain distributions and reject drift rather than deriving an ANE clock from CPU cycles, power, thermal category, or nominal TOPS. This controls some drift; it cannot normalize DVFS or remove other clients' queueing.

Even valid hardware nanoseconds would remain clock- and memory-frequency-sensitive. DVFS normalization needs independently calibrated active-state durations/frequencies and relevant memory-state information over the same interval. Neither a thermal label nor aggregate watts supplies these. Keep the current acceptance gate; if a separately labeled Fair-state exploratory comparison is conducted later, compare paired trials within that stratum and do not pool them as nominal-state acceptance. No new hardware experiment is justified by this audit alone.
