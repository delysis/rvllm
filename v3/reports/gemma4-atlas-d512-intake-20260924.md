# Gemma 4 Atlas D512 intake

Date: 2026-09-24

Source packet: `/Users/george/Downloads/rvllm-attention-atlas-2bc5a535`

Packet base: `2bc5a535b113c0bcc9b1f5e15ab747005e2e3c60`

Port commit: `8ac57344`

## Intake boundary

The packet's checksums and syntax verified, but its exact-base guard correctly
rejected the live tree, which was eight commits ahead and had newer R1,
shared-KV, split-KV, evidence, and queue machinery. The packet was therefore
not applied. Four controlled schedules were manually ported to the current
BF16 K/V ABI and current catalog/referee:

- `metal-global-d512-atlas_r16k16p64t128`
- `metal-global-d512-atlas_r16k32p64t128`
- `metal-global-d512-atlas_tile_r16k16p64t128`
- `metal-global-d512-atlas_tile_r16k32p64t128`

All hold rows, panel, threads, cache ABI, and model geometry fixed. They vary
only key tile width (16 or 32) and per-key versus per-tile online-softmax
association. Atlas's parallel runner/catalog and raw-z cache ABI were not
imported.

## Gates

All four complete sources compile as Metal 3.1 with `-fno-fast-math`.

The two per-key candidates passed the existing exact serial-FP32 native oracle
on Apple M4 Max / Apple9, including once-rounded BF16 and adversarial page/cache
cases. Both per-tile candidates failed the exact serial-FP32 gate at L256.
That failure is retained: their changed reduction association may be
numerically acceptable, but they require a separate independent-reference
bounded-error oracle before timing eligibility.

The exact-oracle-qualified per-key candidates were screened at L256. Host
control drift exceeded the existing 5% validity threshold, so these are
exploratory retained measurements, not valid promotion evidence:

| candidate | mean ms | sample range ms | current comparison |
|---|---:|---:|---|
| Atlas per-key K16 | 8.215 | 8.014–8.533 | slower than 3.059 ms R1 headwise control |
| Atlas per-key K32 | 8.313 | 7.821–8.769 | slower than 3.059 ms R1 headwise control |

The candidate arms themselves were relatively compact while the baseline arm
varied widely (about 35–68 ms), but neither Atlas mean is close enough to the
current L256 leaders to justify advancement. The per-key variants are therefore
correctness-qualified non-survivors. The per-tile variants remain
correctness-inconclusive and unbenchmarked.

## Related current leader

The separately implemented split-KV R8/S256 candidate passed its bounded
native oracle and measured 5.943 ms total at L512 (5.903 ms partial, 0.039 ms
merge), versus 8.293 ms for the prior shared-KV leader. That is an exploratory
1.40x prospective improvement, not a promotion result.

