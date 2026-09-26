# Gemma 4 ANE fused attention plus output projection

This packet stages one bounded layer-0 private-compiler probe for the
default-off, single-input/single-output fused attention plus `o_proj` graph.
The command permits exactly one compiler attempt and performs zero accelerator
evaluations. It preserves a fresh receipt and driver journal whether the
private compiler accepts or rejects the source.

This is compile-source evidence only. It is not inference, correctness,
cache-reuse, timing, route, or promotion evidence. A successful result may
advance only to a fresh-process compile-budget-zero load and component oracle.

The frozen local test executable is intentionally not committed. Its SHA-256
is sealed in the queue manifest; rebuilding requires a new job identity.
