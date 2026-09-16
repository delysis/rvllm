# Stacked INT8 FFN performance comparison: prepared, thermally blocked

The [full-model bit-parity check](gemma4-stacked-int8-full-model-20260916.md)
passed, but the candidate still has no qualified speedup result. The default
Gemma 4 12B INT8 FFN is unchanged.

The safe Rust comparator now accepts original versus stacked INT8 FFNs only
with `--stacked-qualification CHECKED_REPORT`. The supplied receipt must cover
the same model path/config, machine/OS, capacity, execution arrangement and
exact prompt/output sequences, with all references passing and all 48 layers
checked on every decode step. The report digest is retained in comparison
output. Other plan differences, including four-bit or the duplicate-work
checked plan, remain rejected. Same-plan comparisons also verify model path,
zero actual compiler calls, and absence of fallback.

Power/work gates are unchanged. Three focused tests passed, covering successful
qualified comparison in both directions, rejection of incomplete qualification,
changed model/work, power transitions, thermal ineligibility, captures, driver
journals and actual compilation. A release invocation against the real
journaled candidate receipt correctly failed before producing timing ratios.
Synthetic unit-test timings are not hardware benchmark results.

Evidence directory: `gemma4-12b-evidence-20260914/int8-stacked-comparison-20260916/`.
Frozen comparator SHA-256:
`3ee6dc804307d8c743e5c6ea27bafa00825f037a3ccc6ce8caec1a4e50de2edd`.
Source, build/test logs, qualification/input digests and `run-plan.json` are
preserved. The planned experiment uses one frozen inference executable,
ordinary user text (no KV seed export), two warmups plus seven measured
requests per process, and three alternating ABBA/BAAB blocks. Process medians
are the units for six paired comparisons; within-process requests are not
treated as independent replications. All raw measurements and baseline drift
must remain visible. This plan has not been executed.

## Blocking condition

`three-cycle-power-audit.json` joins authoritative readings from cache recovery,
full-model qualification and this comparison setup. All three report AC power,
low-power mode enabled, Fair thermal state (1), and ineligible comparisons.
The final reading followed 45 seconds without accelerator work from this agent.
Other Cargo builds observed earlier were absent from the final process snapshot;
this does not establish an otherwise idle laptop or fixed clocks.

No ANE inference/compilation ran during comparison setup. No power setting was
changed. The remaining promotion gate needs an external state change: nominal
thermals and stable, matching controls through the measurement sequence.
Available CPU counters cannot normalize away ANE/GPU throttling. The goal is
blocked on that condition after three consecutive affected work cycles, rather
than complete. Resume with a fresh power preflight and cache-only inference;
stop on cache/accelerator errors or ineligible measurements, without an
automatic repair/retry loop inside the benchmark.
