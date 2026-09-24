# Gemma 4 ANE interleaved timing campaign — 2026-09-24

## Disposition

`static-int8-interleaved-ffn-cached` remains correctness-qualified but is not a performance nominee. All 108 requests in the predeclared twelve-process sequence completed with the same ten generated token IDs, nine ANE steps, zero Metal decode steps and zero compiler calls. Only two of six process pairs passed the strict phase-observation comparator. Both valid pairs failed to show an interleaved decode win.

No default or production selector changed.

## Process medians

Each process executed nine identical 84-token prompts. The first two requests were warmup; the table reports medians over the remaining seven. `decode9` is the sum of nine measured decode-step walls for one request. Values are milliseconds.

| process | layout | prefill | decode9 | FFN9 |
| --- | --- | ---: | ---: | ---: |
| A1 | stacked | 1442.44 | 1768.28 | 819.27 |
| B1 | interleaved | 1690.96 | 1866.14 | 925.03 |
| B2 | interleaved | 2872.06 | 2687.23 | 1269.20 |
| A2 | stacked | 3030.53 | 4141.47 | 1633.93 |
| B3 | interleaved | 3507.77 | 2892.12 | 1309.20 |
| A3 | stacked | 2602.56 | 4305.20 | 1578.05 |
| A4 | stacked | 3506.78 | 4579.04 | 1613.45 |
| B4 | interleaved | 3476.58 | 3488.31 | 1484.63 |
| A5 | stacked | 2959.11 | 3252.84 | 1427.38 |
| B5 | interleaved | 3217.41 | 2857.01 | 1331.82 |
| B6 | interleaved | 2257.59 | 3511.42 | 1569.63 |
| A6 | stacked | 2191.76 | 1953.49 | 923.51 |

Outer-process conditions were accepted by the queue for every process. The phase comparator accepted pairs A1/B1 and A6/B6 and rejected the other four for missing, stale, changing or thermally limited phase observations. The accepted process-median decode ratios, interleaved divided by stacked, were approximately 1.055 and 1.797. Those are two retained observations, not a pooled estimate. They do not support promotion.

## Contamination and exclusions

A local `cargo test` of the new resident-queue mode overlapped A4. The submitted manifest did not list `cargo` and `rustc` as quiet-process exclusions, so the queue did not record that interference as a violation. A4 is explicitly contaminated. Pair A4/B4 was independently rejected by the strict comparator and contributes no performance evidence. It is retained in the raw queue; it was not deleted or retried.

The other three rejected pairs are likewise retained and not converted into favorable manual comparisons. Correct generated tokens do not waive timing eligibility.

## Evidence and next action

- Ordered manifests: `reports/gemma4-ane-interleaved-timing-queue-20260924/manifests/`
- Raw append-only results: `reports/gemma4-ane-interleaved-timing-queue-20260924/run/results/`
- Full-route qualification report SHA-256: `c31c518dfa35b6bef47982f08b9a1525d92447765ebd279756db2f6c4cef6628`
- Component oracle events SHA-256: `825398dd3dd9e635cc2d814186dad4bb4fdecb77b523fb7d1cc7ce9aec9764e1`
- Comparator executable SHA-256: `0b366bcbcabc5b47a948c574c25ee094d8c5b210b4f2177f04776d485965c161`

The evidence should drive a new implementation round rather than additional sampling of the unchanged interleaved layout. Subsequent timing manifests must include build/compiler processes in the activity exclusion set while still allowing any known thermal state and stratifying results locally.

The experiment queue now has a tested `daemon` mode and is running as user LaunchAgent `com.delysis.rvllm-kernel-game`. New reviewed manifests can be submitted without manually restarting the referee; STOP and no-retry semantics remain unchanged.
