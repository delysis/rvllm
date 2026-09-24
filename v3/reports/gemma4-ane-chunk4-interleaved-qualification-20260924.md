# Gemma 4 ANE Chunk4 and Interleaved qualification — 2026-09-24

Tree under test: `a8e4755e` plus the two bounded-oracle test entry points
committed with this report. Hardware: the local Apple Silicon host. Model
configuration SHA-256:
`478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9`.

## Adjudication

| candidate | control | real layer-0 inputs | result | disposition |
|---|---|---:|---|---|
| `ane-int8-ffn-chunk4` | plain INT8 | 1 of 3 reached | output 45 differed by one FP16 ULP: `1.5117188` (`0x3e0c`) vs `1.5126953` (`0x3e0d`) | rejected for exact qualification |
| `ane-int8-ffn-interleaved` | stacked INT8 | 3 of 3 | every finite FP16 output bit-exact | component-correct, not performance-qualified |

Chunk4 failed on the first independently captured activation, from the 21-token
route. The complete control output SHA-256 was
`db462c07a217f5788fc99f9bf0f454858ab6042646b38238427f095d677d5c5c`;
the candidate output SHA-256 was
`f92ee6fbd5cc4d108b2381c334f840d38e6d8b110bc868403a4bd41545bb6f0b`.
This is a real numerical failure under the existing bit-exact contract, not a
timing loss or infrastructure block. Both programs still completed unload and
unload-returned lifecycle events.

Interleaved matched its stacked-INT8 control on all three inputs:

| route | input SHA-256 | shared output SHA-256 |
|---|---|---|
| 21 token | `20f3ffd3aa4bf8b388d90e54c781bbe03eb229d6af4abe46a0ee4b7b2badae86` | `db462c07a217f5788fc99f9bf0f454858ab6042646b38238427f095d677d5c5c` |
| 84 token | `1780cf3e9f1d111badcac5fcc2e5940ee4db71308e9d09eb1568c8ccf14e7c52` | `d60d4ca6cf81723614c4535f2ccc0770ce7b9cbbb8642697d415f0c17f9316d2` |
| 652 token | `8a8337024f125bf66e89642311bf98121b4155d021b16c6915620e1d4098a941` | `2a8980034792ae6e0261ccea01051a96ac0def43543f7c06a0918bc08e666d9f` |

Its invocation made exactly two compiler calls, six ANE evaluations, and 40
driver events: two complete compile/load lifecycles, six evaluate begin/complete
pairs, and two unload begin/complete/returned triples. Manifest SHA-256 was
`303c9e621e555c266cf0d79bb2164bb361a565e12694ec14032da60856771b98`.

## Queue integrity and rejected zero-test attempts

The first submitted Chunk4 and Interleaved jobs used a test executable compiled
without `macos-private-ane-research`. They exited zero after running **zero**
tests (`108 filtered out`). Those queue receipts are preserved under their
original IDs and are explicitly rejected as qualification evidence. Fresh job
IDs, fresh output paths, and an executable that listed both exact ignored tests
were required before rerun.

The valid executable SHA-256 was
`8d092db9da70071ab4ca29ccdbebb17263d4f957ce3e365470c530e630fd616a`.
Valid jobs were:

- `gemma4-ane-chunk4-bounded-oracle-q2-20260924` — failed as expected on the
  numerical mismatch; thermal state 0, AC, low-power mode, pmset mode 1.
- `gemma4-ane-interleaved-bounded-oracle-q3-20260924` — succeeded; same sampled
  stratum. Its outer 9.725-second queue wall includes setup and compilation and
  is not kernel timing.

Raw event/journal SHA-256 identities are:

| receipt | SHA-256 |
|---|---|
| Chunk4 events | `4d5b26edee3a0ad2a1206a18097db1d0a1fba3342328d9c4206684dee8bf9a66` |
| Chunk4 driver journal | `acf1657c9984894d048458bc432d7f72ec2b33c74aa2da44a1e0714e3cc0d14f` |
| Interleaved events | `825398dd3dd9e635cc2d814186dad4bb4fdecb77b523fb7d1cc7ce9aec9764e1` |
| Interleaved driver journal | `678917b3c2e1824114b1d19d30e5429e4c44d1418c1b666e37c59392de8fd3f0` |

Large queue power logs, duplicate output tensors, and model weights remain local
and are intentionally not committed.

## Evidence boundary and next round

This rejects current Chunk4 under the game’s exact-output rule and establishes
source-to-device layer-0 component correctness for Interleaved. It does not
establish whole-model parity, speed, cache portability, compression, or
promotion. Interleaved should next receive exact-tree full-route qualification
and only then warm counterbalanced timing. Chunk4 needs a numerical-ordering fix
or a separately justified contract change; the tolerance must not be silently
widened to admit it.
