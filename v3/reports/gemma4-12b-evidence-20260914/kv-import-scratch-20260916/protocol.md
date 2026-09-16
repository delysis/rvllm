# KV import scratch experiment

Compare the existing allocating packing path with one scratch vector per
48-layer import, reused across layers and dropped between requests. Keep the
packing loop order, zero initialization, FP16 bytes, program cache identities
and surface-copy path unchanged. Existing decoder import remains the default.

Host qualification compares all packed bytes against the existing path on the
hash-checked 84-token Metal snapshot. Synthetic checks cover full/short/empty
prefixes, signed zero, both Gemma geometries, sliding wrap, smaller layouts,
out-of-slice preservation and malformed input rejection before writes.

The CPU-only timing fixture retains twelve slots in three ABBA/BAAB/ABBA blocks,
four complete 48-layer imports per slot, following one unmeasured request of
each variant. Allocation, zeroing, packing and deallocation are timed. Snapshot
loading, byte qualification, JSON output and hashing are outside timing.
Black-box barriers preserve both variants' output memory. No device calls or
surface copies occur. PowerMonitor records raw wall time and host CPU counters.
No single slot is selected, removed or retried.

Independent acceptance requires exact counts/order; equal valid power controls
throughout and across blocks; eligible queue contention observations; and at
most 5% repeat drift for each variant within every block. Fair data remain
explicitly exploratory, separate from nominal and from other power sources.
Report all block ratios for wall time and available host cycles/instructions.
Host cycles describe this CPU-only phase and never normalize GPU/ANE time.
Unqualified or drifting data remain inconclusive. One AC Fair and one battery
Fair timing job may be staged with a 30-second quiet lead; no automatic retry.

Device qualification uses the existing 162 cache-only programs, no compilation,
and a driver journal. Original and scratch imports must match three greedy
continuations at both 84 and 21 imported tokens, including a shorter import
after the longer one. Invalid snapshots must preserve all frontiers. Expected
device work is 2,496 evaluations, 208 requests and 162 successful unloads.
This is correctness only. A complete handoff measurement remains necessary
before claiming an end-to-end speedup or changing the default import path.
