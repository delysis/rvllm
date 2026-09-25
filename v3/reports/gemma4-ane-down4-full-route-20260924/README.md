# Gemma 4 ANE Down4 full-route qualification

Date: 2026-09-24

## Verdict

`static-int8-down4-ffn-cached` passed the pinned normal-route correctness gate
and may advance to timing. This is not a speedup or promotion claim.

The exact inference executable generated reference tokens `[50429, 106]` for
the pinned capital-of-France workload. The route reported:

- `inference_complete=true` and `qualification_complete=true`;
- one requested and completed reference with `matches_reference=true`;
- actual ANE execution with one ANE decode step and zero Metal decode steps;
- 162 loaded ANE programs;
- zero compiler calls against a zero-compile budget; and
- no CPU or GPU decode fallback.

The separately reported component oracle remains the numerical prerequisite:
Down4 was finite, deterministic, and bit-exact with ordinary INT8 on three
independently captured real layer-0 activations. The full route adds model-wide
delivery and token-parity evidence; it does not prove that every intermediate
layer output is bit-exact.

## Conditions and retained failures

The successful `q3` job admitted any low-power mode, pmset mode, and thermal
state. Those conditions were observed and retained rather than used as a wait
gate. The outer queue accepted the completed run.

The never-started first manifest, which inherited stale low-power admission
requirements, is preserved under `deferred-jobs`. The `q2` run is separately
quarantined: Metal prefill completed, but the process stopped before ANE
preparation because an environment value incorrectly used the literal
`{output}` placeholder. Neither attempt is replayed or converted into kernel
evidence.

## Evidence pins

- inference executable SHA-256:
  `729418c460acb56e2a04acbd66677ff71fa9a9441e7a626e1c18360ec4c34907`
- successful sealed job SHA-256:
  `2c811db307cc77da046e3deca25318939d8f716e2e03a094e63f272a2a2af9e0`
- queue result SHA-256:
  `df914b41b50d6ea14fb39caaa7ef7f4c4f58afbec9379e807ad7c46486826017`
- inference report SHA-256:
  `7f7f152a5e48692532258902a8ecbf8aab5239ce98bf20682f2c0ec723fd9a94`
- diagnostic lifecycle journal SHA-256:
  `2adc64bfa155a90cec331761737839aedc1286ea275063a96e7acd64d9656234`

The raw queue result, inference report, and 708 KiB diagnostic journal remain
in the queue evidence directory and are hash-bound here. They are not needed
in Git to reproduce the concise route-level verdict.

## Next gate

The candidate is now in a twelve-arm ABBA/BAAB/ABBA process-level timing
campaign against `static-int8-ffn-cached`. Each arm contains nine identical
prompt cases and 81 measured decode steps. Timing may inform nomination only
when work identity, zero compilation, actual ANE execution, and the prospective
Down4 route/component qualifications are all retained.
