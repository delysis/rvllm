# Gemma 4 Atlas SIMD-matrix intake

Date: 2026-09-24

Source packet: `/Users/george/Downloads/rvllm-attention-atlas-2bc5a535`

Packet patch SHA-256: `32389c8f968efc88319b9873ce039387e64be70dc414a68e36fc35e18ae06ac6`

Port commit: `ebd84dfa`

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

## Gates and result

Focused Rust geometry/catalog/queue tests passed (13, 3, and 8 tests). All eight
identity-bound core/oracle build jobs compiled and linked with Metal 3.1 and
`-fno-fast-math`, with unchanged inputs.

All four Apple9 native oracle jobs then failed at the L256 exact serial-FP32
comparison. This is retained as a real gate failure. It does not establish that
the matrix results exceed an acceptable model-level error budget because the
matrix family deliberately changes floating-point association, but it does mean
the candidates are not qualified for timing under the current exact contract.

| candidate | strict build | Apple9 exact serial-FP32 oracle | timing eligibility |
|---|---|---|---|
| R16/K16/P64/T128 | passed | failed at L256 | no |
| R16/K32/P64/T128 | passed | failed at L256 | no |
| R16/K16/P128/T128 | passed | failed at L256 | no |
| R8/K32/P64/T128 | passed | failed at L256 | no |

Immutable raw results are under
`reports/gemma4-global-decode-local-20260924/queue/results/g4d512atlasmma20260924v1-*-mma-*`.
The identity-bound campaign inputs and manifests are under
`reports/gemma4-global-d512-atlas-mma-20260924-v1/`.

## Next gate

Do not time or promote these candidates yet. Add a separate matrix numerical
contract using an independent FP64 reference, explicit absolute/relative and
once-rounded BF16 bounds, adversarial page/cache cases, repeatability, guard
bytes, and rejected-dispatch no-write checks. Keep the exact serial-FP32 result
in the receipt as a diagnostic rather than rewriting it as a pass. Only matrix
candidates that pass that independent bounded oracle may enter the L256 screen.
