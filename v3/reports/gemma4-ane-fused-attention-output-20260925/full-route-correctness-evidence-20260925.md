# Gemma 4 layer-0 fused ANE attention + output: full-route correctness evidence

Date: 2026-09-25

## Result

The default-off `Layer0FusedCached` route passed a production-shaped two-token
full-decoder referee against the existing separate attention and output route.
It is now correctness-qualified for full-route timing, but it is not selected
by default and has no full-model speed or promotion claim yet.

For both dependent decode tokens:

- every layer-0 FP16 residual bit matched;
- final sampled token, position, top-five token IDs and FP32 score bits matched;
- the candidate executed exactly two fused layer-0 evaluations;
- the candidate executed 94 separate attention and 94 separate output
  evaluations for the remaining 47 layers across two tokens;
- the baseline executed exactly 96 separate attention and 96 separate output
  evaluations and no fused evaluations;
- compiler-call count remained zero from strict load through both inference
  routes.

The dependent token chain was `3730` at position 84 followed by `37423` at
position 85 in both routes. The sealed prefill snapshot SHA-256 was
`0b096c565cc841ee625081d23b1752af0590c25c8f48554451e97e1fb7140ee8`.

## Bounded preparation

ANE's process compiler budget is 48, so setup was deliberately divided into
five fresh queue jobs using the identical frozen executable SHA-256
`39e5ba9349c351d3cdc3679ff79dbed04f92df4b7f85a997381fe6063b37084a`:

| stage | entries | compiler calls | evaluations |
| --- | ---: | ---: | ---: |
| QKV | 48 | 0 (cache hits) | 0 |
| output projections | 48 | 47 | 0 |
| INT8 FFNs | 48 | 47 | 0 |
| vocabulary + attention | 18 | 17 | 0 |
| fused layer-0 attention + o_proj | 1 | 0 (cache hit) | 0 |

The sixth job loaded every program with strict cache-only policy and performed
the referee. No preparation job was conflated with inference timing.

## Preserved failures and infrastructure repair

The first queue attempt never launched because the LaunchAgent security context
blocked while opening a newly frozen executable; it was retained as a
prelaunch infrastructure failure. The second launched and correctly exposed a
process-local cache miss. The third correctly exposed the 48-compile process
budget when an invalid monolithic provisioning strategy attempted a 49th
compile. Those jobs remain quarantined with their raw evidence rather than
being overwritten or retried in place.

The successful v4 campaign moved the frozen executable into the daemon's build
tree and expressed bounded provisioning as explicit queue dependencies. The
persistent LaunchAgent was restored after the supervised campaign and is
running again.

## Claim boundary and next gate

This proves exact routing, dependent newest-K/V behavior, exact decoder output
for two tokens and zero compilation during inference for the layer-0 fused
route. It does not prove all-layer fusion, longer-generation stability,
end-to-end speedup, checkpoint-quality acceptance, or shipping readiness.

The next gate is an independently ordered full-route timing comparison using
the now-qualified route, followed by a longer multi-token stability run. The
component-level 1.96x gain remains the performance hypothesis.

## Sealed receipts

- Full-route receipt SHA-256:
  `88142a8c5f26f0377259b5560d60fcd0d645bd3d932f8700e17f07c2421d61a2`
- Full-route queue report SHA-256:
  `e50ea21ec850ef9daf1f5fba67339714491bef3856f6efd93f2cff914235ae28`
- QKV provisioning receipt SHA-256:
  `262a722a23abe49516470de229f83dce504e2422986a6fad9ed45dff77cc9aea`
- Output provisioning receipt SHA-256:
  `4a4e7f0c0ff1f466954ee14d39ce999558c61d11fd6e73e56a59d4260d126a5d`
- FFN provisioning receipt SHA-256:
  `6c7e4eb28601b6b760d7de25d654396c3699eb7647e61059003515db01bc9b6b`
- Vocabulary/attention provisioning receipt SHA-256:
  `7055a75915d9f12d3b94ab2b7aadd2108da3ce174fecaaa018a110594a497995`
- Fused-program provisioning receipt SHA-256:
  `d559773b73c00e04890b9bfe9b71eb999514ae17ccdc8eb4f217732773ba6263`
