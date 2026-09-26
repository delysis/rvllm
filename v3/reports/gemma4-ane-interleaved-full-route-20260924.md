# Gemma 4 ANE Interleaved full-route qualification — 2026-09-24

Commit under test: `77a0b376ed3b00d2db87727a54d1cca5f2ae3fab`.
Hardware: the local Apple Silicon host. Candidate:
`static-int8-interleaved-ffn-cached`; control semantics: stacked INT8 FFN.

## Result

The exact-tree normal inference executable completed the pinned Gemma 4 12B
capital-of-France route and generated `[50429, 106]` (`Paris`). The route report
states all of the following:

- `all_references_match: true`
- `qualification_complete: true`
- `inference_complete: true`
- `ane_execution_verified: true`
- `cpu_or_gpu_decode_fallback: false`
- `ane_cache_policy: require-existing`
- `ane_compile_budget: 0`, `ane_compile_budget_used: 0`
- 162 ANE programs loaded and one ANE decode step executed
- executable SHA-256
  `729418c460acb56e2a04acbd66677ff71fa9a9441e7a626e1c18360ec4c34907`
- BF16-off metallib SHA-256
  `6efa8fbed39f6d36fb83fe823da1a5d0733c22970b96b431439be27a07617705`
- reference SHA-256
  `57a3fee12427f25c35f62fead004130ee56e3c30994b7e9196335964a2b249cd`
- model configuration SHA-256
  `478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9`

The route report SHA-256 is
`ca7711c974783a3d483bc0fe8d5a479fc828c48c97b4a4a9ed66f85af5a55d33`.
The full-route driver journal SHA-256 is
`c648e016d8da724fc76dad17e54c7efbbee95e154bdef554ff015c275ac54e0b`.
It contains no `compile_begin` or `compile_completed` event. Its 162
`compile_requested` events are strict cache lookups, all followed by cache hits.

## Cache preparation

The queue first prepared the 48 layer-specific Interleaved FFN programs through
the same normal inference executable. That job used exactly 48 bounded compiler
calls, performed zero inference evaluations, and succeeded. Its driver journal
SHA-256 is
`51c539a7a477392fa22535c4ab2bd8d27e0e96fa3437d2d35644ab45ee51f818`.
Preparation ran at sampled thermal state 1. Cache availability is scoped to the
client identity and is not a self-contained artifact claim.

## Timing boundary

The single decode step reported 2,525.011 ms, including 538.267 ms QKV,
492.495 ms attention, 532.009 ms output, 780.435 ms FFN, 176.460 ms vocabulary,
and 5.346 ms host work. The Metal prefill completed in 2,302.998 ms for 21 input
tokens. These are diagnostic phase observations only: the run occurred at
thermal state 1, was not paired or counterbalanced, and the measurement object
was ineligible for a fixed-stratum performance claim. No speedup is claimed.

## Next gate

Interleaved is eligible for a sealed, warm, counterbalanced comparison against
the ordinary INT8 FFN plan. The comparison must retain exact generated IDs,
zero compiler calls, equal work counts, raw per-request samples, matched sampled
power/thermal strata, predeclared drift handling, and independent confirmation.
It must not reuse this one-shot diagnostic latency as a timing observation.

The queue job manifests are under
`v3/reports/gemma4-ane-interleaved-full-route-queue-20260924/jobs/`. Large raw
queue power logs, layer-state tensors, journals, and model weights remain local
and are intentionally not committed.
