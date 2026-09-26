# Fused ANE attention plus output projection: compile-source evidence

## Outcome

The layer-0 fused attention plus output-projection program compiled and loaded successfully through the private ANE driver on the exact queued executable. This is compile-source evidence only. It does not establish numerical correctness, repeated-use stability, full-route selection, latency, throughput, or promotion.

## Identity and boundary

- Queue job: `gemma4-ane-fused-attention-output-compile-layer0-envfix-20260925`
- Queue result: `succeeded`, exit code `0`, no queue violations
- Test executable SHA-256: `da3f09893d0ef5789e73d3094cf0c11f8f36ff158fa5377c5c97133deb680f75`
- MIL SHA-256: `e5afd7e966aec93d555218310514be09cf1cec8edd03196aefd92f5cde442f9b`
- Weight blob SHA-256: `68a3a71d77dba7eeefe425c56cc93695f718ddf3530150395a19c5bae859b160`
- External inputs: `1`
- External outputs: `1`
- Input bytes: `8523840`
- Output bytes: `245760`
- Compiler calls: `1` of an allowed `1`
- Accelerator evaluations: `0`

The driver journal records `compile_requested`, descriptor creation, a cache miss, `compile_begin`, `compile_completed`, `load_completed`, removal of source data and weights, and a clean unload. The result was collected while the queue logged the host conditions without a stability wait gate.

## Preserved receipts

The authoritative machine-readable evidence is checked in under:

`reports/gemma4-global-decode-local-20260924/queue/results/gemma4-ane-fused-attention-output-compile-layer0-envfix-20260925/`

Key file SHA-256 values:

- `compile-source-receipt.json`: `49a8eb6b3baa7f235458c385bd0e3f65d9ebfce1e8c9b71280a8a454317d2776`
- `ane-driver-journal.jsonl`: `c17912ece26818be7469f54840637f0d9bd6876db1edaac1544065527b8c152e`
- `report.json`: `073b865f769f80dafb5441d0531418afeda19ac1d94e8d22d8480083587d21ac`
- `trial.stdout`: `76fdc0a3530dd56aeec4c684ce2dbdd25fb6a97e4043e586e4497acd4e720e16`
- `trial.stderr`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Next gate

Reload the exact compiled cache identity with a compiler-call budget of zero, then run a bounded component oracle against an independent CPU reference with guard bytes, holes/boundaries, newest-K/V coverage, and repeated-use checks. Timing is admissible only after those gates pass.
