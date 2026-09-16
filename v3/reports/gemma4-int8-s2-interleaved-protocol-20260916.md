# Interleaved S1/S2 FFN timing protocol

This new experiment responds to job 04's inconclusive repeat drift. It does not
replace that result or modify the queued original pilots. It uses the same
INT8 weights, strict cached S1/S2 graphs, CPU-qualified inputs, numerical
checks, eight warmup pairs and zero compiler calls. No driver journal or new
private API is permitted.

The work count stays at 2,304 measured device evaluations: three blocks, each
with 32 counterbalanced cycles, four slots per cycle and four repetitions per
slot. A serial repetition executes two S1 calls; a batch repetition executes
one S2 call. Both produce two useful outputs. Cycles alternate ABBA/BAAB, with
the starting order reversed in block 1. This places the variants milliseconds
apart instead of the original 128-repetition bursts.

Each slot records monotonic start/end/duration and verifies its final output
bit for bit after the slot timer stops. Every slot is retained. One existing
PowerMonitor phase spans each whole block; there are no per-slot process CPU
counter reads, power-command launches or journal writes. Packing, synchronous
evaluation and output copy remain timed. Power is sampled asynchronously.

Predeclared gates: exactly three complete blocks; exact order/work/numerical
checks; ordered nonoverlapping positive finite intervals within each block;
same valid nominal or explicitly exploratory Fair controls across all blocks;
at most 5% aggregate first-versus-second occurrence drift within cycles for
each variant; and at most 5% first-half-versus-second-half drift for each
variant within each block. Both drift checks use max/min minus one. No slot
is discarded, winsorized, selected or retried. Report every block's ratio of
total serial time to total S2 time and their median. Failed gates yield an
inconclusive result, never an accepted component speedup. Queue contention
eligibility is an additional requirement.

This estimates local paired wall time for one FFN, not accelerator cycles,
physical bandwidth, speculative acceptance or whole-model throughput. AC and
battery controls stay separate. Existing original and native-kit jobs remain
frozen. The first AC and battery Fair trials use a 30-second quiet lead and
are separate experiments from the original 120-second pilots.
