# Gemma 4 ANE Down4 qualification — 2026-09-24

Tree: `1d286658f8e8afbcde2f87a504b7fd4e1f622e8a` plus the bounded-oracle
change committed with this report. Hardware: the local Apple Silicon host.
Model configuration SHA-256:
`478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9`.

## Result

`ane-int8-ffn-down4` is now component-correct on three independently captured
real layer-0 FFN activations from the 21-, 84-, and 652-token routes. Its output
was finite and bit-for-bit identical to the ordinary INT8 FFN control for every
sample:

| sample | input SHA-256 | shared control/candidate output SHA-256 |
|---|---|---|
| 21-token route | `20f3ffd3aa4bf8b388d90e54c781bbe03eb229d6af4abe46a0ee4b7b2badae86` | `db462c07a217f5788fc99f9bf0f454858ab6042646b38238427f095d677d5c5c` |
| 84-token route | `1780cf3e9f1d111badcac5fcc2e5940ee4db71308e9d09eb1568c8ccf14e7c52` | `d60d4ca6cf81723614c4535f2ccc0770ce7b9cbbb8642697d415f0c17f9316d2` |
| 652-token route | `8a8337024f125bf66e89642311bf98121b4155d021b16c6915620e1d4098a941` | `2a8980034792ae6e0261ccea01051a96ac0def43543f7c06a0918bc08e666d9f` |

The same invocation compiled, loaded, evaluated, unloaded, and returned from
both programs successfully. It used exactly two compiler calls, matching the
declared upper bound for one control plus one candidate. The driver journal has
40 lifecycle events and no missing terminal unload return.

Separately, the normal inference executable loaded all 48 layer-specific Down4
programs through the serialized queue. That pass reported zero compiler calls,
48 prepared programs, zero evaluations, and 57.421 seconds of preparation. The
different test executable could not reuse those entries, demonstrating that a
successful cache-preparation receipt is scoped to its client identity and must
not be treated as portable cache proof.

## Queue and environment

Both accelerator operations ran as `preparation` jobs in the existing safe-Rust
experiment queue. The queue required AC power, low-power mode enabled, pmset
power mode 1, and at least 16 GiB free space; it admitted any thermal state.
The component job remained in sampled thermal state 0 and its outer wall time
was 10.411 seconds. That duration includes host loading and compilation and is
not a kernel timing result.

Pinned executable identities:

- full 48-layer cache path:
  `729418c460acb56e2a04acbd66677ff71fa9a9441e7a626e1c18360ec4c34907`
- bounded component oracle:
  `48cf0aff8259140ed5086186700fbbffa998b16a122950453b11452c0e6a2882`

The component manifest SHA-256 is
`f9c57ab1defd9d026b1fd2382ae94492ccaac83f849c902580458817941ef470`.
It pins the three sample files and canonical hashes of the gate, up, and down
INT8 coefficient/FP16-scale matrices.

## Evidence boundary

This establishes source-to-device component correctness, bounded compilation,
and complete lifecycle for Down4 at layer 0. It does not establish a speedup,
whole-model token parity, cache portability, physical ANE weight compression,
or promotion. The next Down4 gates are an exact-tree full-route cached inference
repeat followed by matched, counterbalanced timing against ordinary INT8.
Subsequent bounded passes independently rejected Chunk4 on bit-exact correctness
and qualified Interleaved at this same component boundary; those results are
recorded separately and do not widen this Down4 claim.

Raw concise receipts are in
`v3/reports/gemma4-ane-down4-qualification-20260924/`. Large queue power logs,
model weights, and duplicate output tensors are intentionally not committed.
