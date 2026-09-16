# Baseline INT8 FFN versus CPU reference

The broader two-token qualification stopped in its **single-token baseline**,
before running the two-token candidate. This is a new reference-accuracy
finding, not evidence of an S=2 regression or a driver fault. No tolerance,
production arithmetic or qualification gate has been changed.

## Reproduction evidence

The actual ANE decoder captured layer-0 FFN input at absolute position 653 of
the 652-token recall reference. Input SHA-256 is
`f0ddc34dd2065ce6f2c97aaeab5b01830ecbe6fa618080413770552161be6f34`.
The preceding full-model capture run matched all 15 expected output IDs across
21/84/652-token references, with zero compilation, 2,496 evaluations and 162
unloads. All 240 captured vectors were independently hash-checked.

Queue `native-comparison-quiet`, job `33-batch-live-inputs`, then loaded both
existing S=1/S=2 layer-0 FFN graphs with zero compilation. Its eighth S=1 call
failed at output 3627: ANE -0.5537109375 versus CPU -0.5212233067, exceeding the
unchanged `0.01 + 0.02*abs(reference)` tolerance. Both programs unloaded; S=2
had zero evaluations. The queue halted and that failed job was not replayed.

## Offline arithmetic controls

The legacy reference rounds gate/up projections and activated/gated outputs,
but keeps most GELU operations and constants in FP32. MIL declares FP16
constants and per-node results. Two additional controls round each declared
FP16 node and final output, using either FP32 or FP64 dot accumulation.

| Reference at output 3627 | Value |
|---|---:|
| Existing CPU reference | -0.5212233067 |
| Explicit FP16 nodes, FP32 dot | -0.5224609375 |
| Explicit FP16 nodes, FP64 dot | -0.5224609375 |
| Exact INT8 times stored scale retained through FP64 dot, explicit FP16 nodes | -0.55029296875 |
| Observed S=1 ANE | -0.5537109375 |

The first two controls rule out literal GELU-node rounding or FP32-to-FP64
accumulation changes alone as an explanation at this coordinate. The last
control reconstructs the source INT8 integers independently from the original
FP16 checkpoint rows with the exact stored FP16 scales, but avoids rounding
each integer-times-scale coefficient to FP16 before the dot. It is closer to
the ANE observation; it does **not** establish how the compiler lowers this
graph. A complete device output vector and intermediate/reduction evidence
are still missing.

Apple's [typed-execution contract](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/docs-guides/source/typed-execution.md#L76-L78)
defines tensor types as **minimum precision**: an FP16-typed operation may
execute at FP32. Explicit per-node FP16 rounding is therefore a diagnostic
control, not an authoritative execution oracle. Apple's
[host affine decompressor](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/coremltools/optimize/_utils.py#L200-L223)
does cast materialized weights to the scale's dtype, while its
[runtime compression overview](https://github.com/apple/coremltools/blob/181cbd341fa394ea53fd6895a2e0a46aaec27cb1/docs-guides/source/opt-overview.md#L80-L94)
permits on-the-fly decompression and fused kernels without specifying their
rounding boundaries. These public Core ML sources do not establish our private
M4 ANE compiler's lowering.

With one scale per output row, retaining `q*s` through the sum and applying
`s` after the sum are algebraically equivalent in exact arithmetic. Agreement
with this diagnostic would not distinguish those implementations. The
unrounded diagnostic remains seven FP16 output spacings from the observed
value at coordinate 3627.

Earlier [palette qualification](gemma4-exact-int8-palette-experiment-20260915.md)
already found affine INT8 and dense-FP16-reconstruction outputs differed, while
palette and dense outputs matched. That supports investigating representation
and fusion boundaries rather than assuming reconstructed FP16 coefficients
uniquely determine all backend arithmetic. It does not prove the new hypothesis.

CPU-only evidence is under `gemma4-12b-evidence-20260914/ffn-oracle-diagnostic-20260916/`.
The initial three-reference binary hash is
`06732a4f1e8feae14377d0594686a7a61b64a5689981ea675e5e3d16983bb9db`;
the additional unrounded-coefficient diagnostic hash is
`d65f01527919c93ed3dd69fe9fecdab6bf604dac63369b9b9fcd5f08754f0f54`.
Both reports retain complete output vectors and source snapshots. They made
zero accelerator calls. The first CPU invocation had a mistyped checkpoint
directory and failed before weight loading; the corrected invocation completed.

## Complete S1 output capture

Queue `baseline-only-v4`, job `35-s1-output-diagnostic`, collected all eight
inputs (three synthetic, five actual decoder inputs). Its driver journal records
one cache hit, eight completed evaluations, one unload and no compiler call.
All **30,720** finite outputs are retained as FP16 bits, byte hashes and FP32
values. Exactly one value failed the unchanged legacy tolerance: input 7,
coordinate 3627, reproducing the previous result. The legacy gate remains failed.
Job success means complete diagnostic collection, not numerical qualification.

The capture binary is `c3376fd68e957aba8d155730bb4c4ed107d8e60ded6fd8c880b57e70503b2842`.
The complete `probe.json` hash is
`bc00633961ca9f88bb1ff5f139f5a5c22091de7b8710cd940e8eb0d6ffce2363`.
The report is at
`gemma4-12b-evidence-20260914/experiment-queue-20260916/baseline-only-v4/results/35-s1-output-diagnostic/`.

The first offline comparison, job 36, rejected a JSON numeric-mirror identity
check before accelerator execution. A host fixture spanning FP16 exponents
reproduced the bug: direct JSON-number equality compared parsed FP64 values to
an FP32 field. The corrected reader checks exact FP32 bits after typed
deserialization, while independently checking the authoritative FP16 bits and
byte hash. It still rejects changed weights, input identity, bits, numeric
values, shape and nonfinite output. The original failure is preserved, not
replayed. All five probe tests pass after the fix.

Corrected CPU-only comparison job `38-s1-cpu-comparison` completed in
`baseline-only-v5`, using binary
`16ebde9fa99ae7a5ea71cabe4150a2475509d610feae363e52a49b2c5d0209b3`.
It requires the exact captured input identities and reconstructed-weight hash,
and records full-vector statistics and all tolerance failures for each
arithmetic hypothesis. It does not promote any hypothesis to an ANE oracle.
Independent source review identified a future provenance improvement: the
standalone capture gate binds the rounded FP16 reconstruction, while the
unrounded diagnostic re-derives INT8 integers and stored scales from the
original checkpoint. Different affine encodings can round to the same FP16
coefficients. This run uses the unchanged checkpoint and quantizer; it does
not establish that the reconstruction hash alone identifies affine arithmetic.
Future reusable captures should retain and validate the exact INT8 source-blob
hash as well, and the unrounded reference should consume or verify those
coefficient bytes before comparison.
Synthetic inputs also need their actual byte hashes in future captures: the
current report binds their pattern identifiers and this frozen generator,
which alone would not detect a subsequent generator change.
Never-started job 37 was explicitly withdrawn while the worker was stopped
to give this CPU-only computation a 4 GiB disk floor. Its original manifest
and withdrawal receipt are preserved; hardware jobs retain the 16 GiB floor.

## Full-vector arithmetic comparison

Job 38 completed without accelerator or compiler calls. It consumed the exact
capture report above and compared all 30,720 output coordinates against four
references. CPU report SHA-256 is
`554e720211ff5ebbdc162ed866e69105298c69b8a9d180abb34e178430dd3008`.
Raw per-coordinate failures and complete reference output vectors are retained
in `baseline-only-v5/results/38-s1-cpu-comparison/cpu.json`.

| Reference | Coordinates outside the unchanged tolerance |
|---|---:|
| Legacy reconstructed-FP16 weights, FP32 GELU | 1 |
| Explicit FP16 nodes, FP32 dot | 1 |
| Explicit FP16 nodes, FP64 dot | 2 |
| Unrounded INT8-times-scale, FP64 dot, explicit FP16 nodes | 0 |

The unrounded variant eliminates the specific rejection, but has **larger
relative L2 deviation on five of eight inputs** than the legacy reference.
Its maximum absolute error across all inputs is 0.125, versus 0.09454346 for
the legacy reference. For the previously failing input its relative L2 is
0.000451694, versus 0.000460987 for legacy: a small aggregate improvement,
despite the much closer individual coordinate.

Explicit FP64 accumulation also introduces another tolerance failure on the
capital input at coordinate 952 (ANE 0.083984375 versus reference
0.07171630859375). These mixed results do not establish a superior ANE oracle
or isolate the compiler's dequantization/fusion behavior. The legacy gate is
still failed, and no production arithmetic, tolerance or S=2 qualification
requirement has been changed. CPU job timing is not used for a speed claim.

## Remaining gates

Compare every captured coordinate with the independent CPU variants and
distinguish successful data collection from numerical qualification. Do not
promote a new oracle or relax a gate from one coordinate's agreement.
Keep the original S=2 and full-model speedup gates open.

The next offline controls can isolate where affine precision matters: compare
unrounded gate/up with rounded down, and rounded gate/up with unrounded down,
holding FP64 accumulation and explicit FP16 GELU fixed. Use the failing input
and a passing input. These controls distinguish projection sensitivity, not
post-accumulation scaling from higher-precision reconstruction. Retain the
unchanged legacy failure even if another reference fits better.

The untouched native-kit comparison jobs now run in `baseline-only-v5`; their
frozen backends, work counts and power conditions are unchanged. They are not
retries of the failed component experiment. No accepted slowdown ratio exists
yet.

Disk headroom recovery removed 4,105,481,540 bytes of this project's obsolete
debug incremental trees under an exclusive Cargo profile lock. Every deleted
entry was more than two hours old; no links were followed. Sources, models,
linked executables and experiment receipts were preserved. Available space
rose from about 13 to 17 GiB at that observation; this is not a lasting disk
capacity guarantee. Inspection/removal receipts and safe Rust helper are in
the diagnostic evidence directory.

## Stored-affine hybrid controls

Job `39-s1-hybrid-controls` completed offline with zero accelerator/compiler
calls, comparing all 30,720 coordinates. It reads the actual stored INT8
integers and FP16 scales rather than re-deriving them from the checkpoint.
All four prior reference-output arrays are identical to job 38.

| Affine reconstruction, FP64 dots and explicit FP16 nodes | Tolerance failures | Input 7, coordinate 3627 |
|---|---:|---:|
| Rounded gate/up and down | 2 | -0.5224609375 |
| Unrounded gate/up, rounded down | 0 | -0.54638671875 |
| Rounded gate/up, unrounded down | 1 | -0.5263671875 |
| Unrounded gate/up and down | 0 | -0.55029296875 |
| Captured ANE | — | -0.5537109375 |

This isolates greater sensitivity to gate/up reconstruction at the troublesome
coordinate. It does not identify compiler fusion or the correct device oracle.
The gate/up-only hybrid still has larger relative L2 deviation than legacy on
six of eight inputs. The legacy failure and all tolerances remain unchanged.

New v2 capture code binds all synthetic and real input bytes and the exact
framed gate/up/down integer-and-scale bytes. Its reader rejects mismatches.
No v2 hardware capture has run. Job 39 explicitly opts into the older pinned
capture and records `capture_exact_affine_and_input_identity=false`; the new
affine hash cannot retroactively certify the old capture. Six host tests pass,
including identity rejection and a hand-calculated stored-affine dot product.

Frozen hybrid probe SHA-256:
`951e1b86ef747df75a3a9828849453c476802a960e4710ba4fd8d42fc89a562b`.
CPU report SHA-256:
`13daf6b111531fe2c774ad28b8fb2ab23033bd691559ea2bb0e6a27ed396376b`.
Current affine coefficient hash:
`2078a4984159884e9a0e7fb035d26293e0f961240c0ce44310e0aaa8e51613e9`.
Manifest, frozen sources and build/test logs are under
`gemma4-12b-evidence-20260914/int8-next-controls-20260916/`.
