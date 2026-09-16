# Shared KV import scratch candidate

At the pause checkpoint this candidate is implemented as an explicit alternate
path, with the original allocating import still the default. Final host checks
are recorded in [checkpoint validation](checkpoint-20260916/validation.md).
Captured-byte, device-continuation and timing qualification remain pending;
there is no speed claim. The original design rationale follows.

`GemmaAneDecode::import_prefill`
calls `AneAttention::import_cache` for all 48 layers. Each call allocates a new
zeroed full-capacity packed surface via `PackedAttentionLayout::import_cache`,
then copies that buffer into the existing owned input surface. The buffer is
dropped at the end of each call. This repeats allocation/deallocation for large
buffers even when the imported prefix is short.

The bounded candidate is a shared scratch vector reused across layers within
one complete import. Add an allocation-free layout packing method into an
exactly sized mutable byte slice; preserve the current allocating wrapper as
the independent test control. An attention import wrapper can resize a caller
owned scratch vector, pack, then invoke the existing `write_input`. The decoder
creates one vector outside its layer loop. No new surface access, selector,
graph, cache identity or arithmetic is required. Initial decoder preparation
and ordinary decode remain unchanged.

Validate all geometry before writing. Clear the entire used buffer on every
import, including query, unused KV positions and mask storage; simply copying
the active prefix would leak stale state. Preserve absolute sliding positions,
ring order and all existing import failure/poisoning behavior. Check mixed
sliding/global sizes, shorter imports after longer imports, zero initialization,
capacity endpoints, wrapped sliding prefixes and invalid lengths. Compare
every packed byte against the allocating control before any device trial.

Then compare allocation versus reuse on the actual captured 48-layer snapshot
as a CPU-only packing experiment under the queue's matched controls. Host CPU
cycles/instructions are relevant to this host-only work, unlike ANE/GPU time.
Finally qualify full imported-KV greedy continuation using the existing cached
single-I/O graphs, and measure complete Metal-to-ANE handoff. Allocation or
page-fault savings are hypotheses until these measurements establish them.
