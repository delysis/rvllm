# Gemma 4 layer-0 fused ANE attention + output projection: independent timing adjudication

Date: 2026-09-25

## Decision

The fused component is a **confirmed prospective winner** and should advance to
a default-off production-route experiment. It is not yet eligible for shipping
selection or a full-model performance claim.

Two independently ordered 12-arm campaigns agree closely on the median effect:

| campaign | sequence | separate attention + o_proj | fused | baseline / fused |
| --- | --- | ---: | ---: | ---: |
| first | ABBA / BAAB / ABBA | 19.9215 ms/token | 10.1311 ms/token | 1.9664x |
| confirmation | BAAB / ABBA / BAAB | 19.8824 ms/token | 10.1533 ms/token | 1.9582x |
| pooled arm medians | both | 19.9119 ms/token | 10.1533 ms/token | 1.9611x |

The first and confirmation median ratios differ by only 0.4%. Across all 24
arms, the slowest fused observation (10.7051 ms/token) is still faster than the
fastest separate observation (18.5931 ms/token). This makes order reversal an
implausible explanation for the observed advantage.

## Scope held equal

Both arms use the same real layer-0 weights, persistent KV surfaces, newest-Q/K/V
and mask writes, token sequence, warmup, repetitions and output reads. The
control performs attention evaluation/read, an intermediate write, and output
projection evaluation/read. The candidate performs one fused evaluation/read.
Both receipts report zero compiler calls during timing and bit-identical warmup
outputs between arms.

The sealed fused MIL identity is
`cbc3acbf8ac20506c3592997b76aa0f0695d39a598bd93ae15326988a8840317`;
the real o_proj weight blob identity is
`68a3a71d77dba7eeefe425c56cc93695f718ddf3530150395a19c5bae859b160`.

## Variance and epistemic boundary

This is deliberately not presented as a pristine-clock result. The first run's
range drift was 9.25% for the separate arm and 10.86% for fused. The confirmation
was 41.72% and 6.89%, respectively; its separate-arm range was inflated by one
27.6583 ms/token observation while its median stayed within 0.2% of the first
run. The queue logged AC power, performance power mode, nominal thermal state,
and changing host conditions rather than waiting for stability.

The agreement of order-reversed medians, complete separation of the observed
ranges, exact warmup equality and zero timing compilation justify advancement.
They do not prove the gain survives production routing, all layers, full model
decode, or end-to-end token generation.

## Required next gate

Add a default-off full-route selector that replaces the separate attention and
o_proj route for a sliding-attention layer while preserving persistent state
and incremental writes. Its receipt must prove exact selected dispatch, newest
K/V visibility, first-token and repeated multi-token correctness, cache
identity, evaluation counts, fallback absence and zero inference compilation.
Only then run independently ordered full-route timing and compare end-to-end
decode contribution.

## Sealed evidence

- First timing receipt SHA-256:
  `981b1b0f05ea25210fd5fb557c9e3dc4af501ac58ef18cfdf3fa6a58d7d17b2b`
- First queue report SHA-256:
  `191d90cf297e60fcfe24a6ebaa9601eda14756adeced3ffda5c0ebec6f7cec5f`
- Confirmation timing receipt SHA-256:
  `4154bfb85e4249ac8b7b80f2edb88b93c151abe1aa4b576168d7d98342e6acaa`
- Confirmation queue report SHA-256:
  `6869329721d009adbad7e16846f340c406fa15450a7511f7d4f35f9f301ac764`

The complete queue directories retain the submitted manifests, host-condition
journals, driver journals, stdout and stderr. Failed predecessor attempts remain
preserved separately.
