# Gemma 4 ANE interleaved exact-workload qualification — 2026-09-24

## Result

The `static-int8-interleaved-ffn-cached` route completed the 84-token copy96 workload and generated the exact ten-token CPU-reference continuation:

```text
[818, 3730, 37423, 38167, 1024, 506, 12010, 8858, 236761, 106]
The blue fox jumps over the quiet river.
```

The route report records `matches_reference=true`, nine ANE decode steps, zero Metal decode steps, actual ANE execution, no CPU/GPU fallback, 162 loaded cached programs, and zero compilation against a zero-compile budget. The queue sampled thermal state 0 throughout the outer process.

This closes the correctness and route-delivery prerequisite for comparing the original stacked INT8 FFN layout with the interleaved layout on the same workload. It does not establish a speedup.

## Sealed evidence

- CPU reference: `/Users/george/Downloads/rvllm/v3/reports/gemma4-12b-evidence-20260914/long-context-cpu-oracle/copy96.json`
- CPU-reference SHA-256: `fbb7f516813a1c2977fd011a3f81f91e67fe7d30b0edb16e780f37f81fbec16a`
- Queue source: `reports/gemma4-ane-interleaved-copy96-queue-20260924/jobs/00-copy96-qualification.json`
- Raw route report: `reports/gemma4-ane-interleaved-copy96-queue-20260924/results/gemma4-ane-interleaved-copy96-qualification-20260924/inference/report.json`
- Raw route-report SHA-256: `c31c518dfa35b6bef47982f08b9a1525d92447765ebd279756db2f6c4cef6628`
- Diagnostic journal SHA-256 recorded at collection: `53099c52d588440abe0e0fd04b0ce032a944a35a4128d7722463b9c5ecd92034`

The raw queue result directory is intentionally not committed because its one-second power journal is large and mechanically reproducible. The source manifest and hashes above make the local receipt auditable without pretending the repository contains that raw stream.

## Timing boundary

Diagnostic journaling was enabled to prove the actual dispatch route and compile count. Its per-step observations varied from roughly 2.38 to 8.24 seconds and are observer-perturbed diagnostic measurements, not benchmark samples. No performance claim may be derived from them.

The next gate is a full-process alternating timing campaign over this exact 84-input-token, ten-output-token workload. Every process attempt is retained. Comparisons are made only inside matching sampled strata; thermally elevated but stable Fair-mode pairs may be reported as exploratory and may not be promoted as nominal evidence.
