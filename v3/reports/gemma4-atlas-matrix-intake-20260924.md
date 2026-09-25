# Gemma 4 Atlas SIMD-matrix intake

Date: 2026-09-24

Source packet: `/Users/george/Downloads/rvllm-attention-atlas-2bc5a535`

Packet patch SHA-256: `32389c8f968efc88319b9873ce039387e64be70dc414a68e36fc35e18ae06ac6`

Port commit: `ebd84dfa`

Bounded-oracle commit: `62494e08`

Additional one-axis probe commit: `e6d899d5`

## Intake boundary

The supplied packet is byte-identical to the previously audited Atlas packet and
remains pinned to `2bc5a535b113c0bcc9b1f5e15ab747005e2e3c60`. It was not reapplied to the
newer live tree. Instead, four high-information FP32 SIMD-matrix schedules were
ported to the current BF16 K/V ABI and existing global-decode referee:

- `metal-global-d512-atlas_mma_r16k16p64t128`
- `metal-global-d512-atlas_mma_r16k32p64t128`
- `metal-global-d512-atlas_mma_r16k16p128t128`
- `metal-global-d512-atlas_mma_r8k32p64t128`

The packet's parallel runner, raw-z cache ABI, fixed-count split variants, and
non-executable ANE descriptors were not imported. Existing cooperative shader
source and immutable job IDs remain unchanged; matrix jobs carry an explicit
`-mma` discriminator.

## Exact-contract screen

Focused Rust geometry/catalog/queue tests passed (13, 3, and 8 tests). All eight
identity-bound core/oracle build jobs compiled and linked with Metal 3.1 and
`-fno-fast-math`, with unchanged inputs.

All four Apple9 native oracle jobs then failed at the L256 exact serial-FP32
comparison. This is retained as a real result rather than rewritten as a pass.
Because SIMD-matrix accumulation deliberately changes floating-point association,
the exact contract was followed by a distinct bounded contract rather than being
weakened in place.

## Independent bounded matrix oracle

Commit `62494e08` added a separate fail-closed matrix oracle using an independent
scalar FP64 reference. It requires maximum absolute error at most `5e-4`, relative
L2 error at most `1e-4`, exact once-rounded BF16 output, three-run repeatability,
input/guard preservation, rejected-dispatch no-write behavior, and exact source,
metallib, executable and workload identity. Serial-FP32 exactness remains a
diagnostic field.

All four candidates passed this native Apple9 oracle. Across the campaign, maximum
FP64 absolute error was `4.24622e-6`; maximum relative L2 error was
`1.87599e-6`. Serial-FP32 maximum differences were between `4.35114e-6` and
`4.52995e-6` and were correctly reported as non-exact.

## Exploratory successive-halving timing

Mean candidate GPU milliseconds per dispatch are below. Every sample was retained.
The boolean is the predeclared 5% control-drift gate; failure makes a cell
exploratory/inconclusive for promotion, but does not erase its scheduling signal.

| Context | R16/K16/P64/T128 | R16/K32/P64/T128 | R16/K16/P128/T128 | R8/K32/P64/T128 |
|---:|---:|---:|---:|---:|
| 256 | 1.177 (drift pass) | **1.066** (drift pass) | 1.165 (drift pass) | 1.070 (drift fail) |
| 512 | eliminated | 4.390 (drift fail) | 4.431 (drift fail) | **3.810** (drift fail) |
| 1024 | eliminated | 8.072 (drift fail) | 9.018 (drift fail) | **6.293** (drift fail) |
| 2048 | eliminated | 17.259 (drift fail) | eliminated | **14.605** (drift fail) |

R8/K32/P64/T128 is therefore the prospective matrix-family leader through 2048.
It improves the prior unsplit R8/P64/T128 result from `23.399 ms` to `14.605 ms`
(`1.60x` faster), but remains approximately `9.81x` slower than the retained
exploratory MLX BF16 full-attention operator datum of `1.489 ms` at 2048. This is
not promotion evidence: every 512-and-longer matrix cell failed the control-drift
gate, there is no independent confirmation campaign, and the comparison is not a
sealed cross-framework referee.

The campaign auto-submitted its deferred 4096 survivor confirmation after L2048;
that result is intentionally not used here. The user-requested short screen is
already decisive for iteration.

Immutable raw results are under
`reports/gemma4-global-decode-local-20260924/queue/results/g4d512atlasmma20260924v1-*-mma-*`.
The identity-bound campaign inputs and manifests are under
`reports/gemma4-global-d512-atlas-mma-20260924-v1/`.

## Next probes and gate

Commit `e6d899d5` adds two one-axis probes without changing defaults:

- R16/K16/P64/T64 isolates thread-count effects.
- R16/K64/P64/T128 isolates a larger KV tile.

R32 variants from the packet were rejected: global Q=1 exposes only 16 packed
rows, so the current exact grid would launch zero groups and a rounded-up grid
would access rows 16..31 without a separately designed masked schedule.

The two admitted probes must pass strict compilation and the same bounded native
matrix oracle before L256 timing. No matrix candidate is promotable until a
separate confirmation campaign passes the drift gate and the full inference route
is correctness-qualified. The 2048 gap to MLX also says the next architectural
round must reduce serialized context traversal and barrier/launch cost rather than
merely polish another unsplit tile.

## 2026-09-25 probe and split-matrix results

Both one-axis probes passed strict Metal compilation and the bounded Apple9 native
oracle. R16/K16/P64/T64 measured `1.973 ms` at L256 and was eliminated. The
R16/K64/P64/T128 probe measured `0.906 ms` at L256 with the drift gate passing,
about `15.3%` faster than the prior `1.070 ms` short-context leader. It remained
promising at L512 (`3.378 ms`, about `11.3%` faster than `3.810 ms`) but that cell
failed drift. At L1024 it measured `6.687 ms`, slower than the R8/K32 leader's
`6.293 ms`, and was eliminated before L2048. K64 is therefore a short-context
specialist, not the new overall leader.

Commit `06b36a9e` adds the bounded
`metal-global-d512-split-mma_r8k32s256t128` candidate: matrix QK/PV within sixteen
fixed 256-token partitions followed by a sealed stable-state merge. All sixteen
tail, partition-boundary and page-hole cases passed the new native oracle with
three-run repeatability. Worst observed error was `4.23132e-6` absolute and
`1.87520e-6` relative L2, with exact once-rounded BF16 and preserved guards.
Its L256 partial-plus-merge time was `1.834 ms` (`1.807 ms` partial, `0.027 ms`
merge), slower than the unsplit matrix leaders, so it was screened out before
longer contexts. This does not answer whether a coarser or context-adaptive split
wins at long context; it rejects this exact fixed-S256 schedule at the first gate.

The first split timing receipt also exposed a referee defect: bitwise equality was
required between a serialized total and the sum of separately serialized floating
components. Commit `10f046da` replaces that transport-fragile equality with an
eight-ULP-scale consistency bound while retaining rejection of material mismatch;
focused positive and negative scorer tests pass.
