# Gemma 4 global D512 decode: `r16p128t64` checkpoint

Date: 2026-09-24

Integration commit at completion: `a0f50367`
Device: Apple M4 Max, GPU family Apple9

This is exploratory raw-operator evidence, not promotion evidence. The
candidate passed the native device oracle before timing. Every cell contains
five warmups per arm followed by five complete ABBA blocks with 100 dispatches
per sample; all 20 measured samples are retained.

| Context | Baseline GPU ms | Candidate GPU ms | Ratio of means | Median paired ratio | Control drift | Versus `r16p128t128` |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 32.642 | 15.213 | 2.146x | 2.071x | 105.119% | 3.15x slower |
| 512 | 51.586 | 24.392 | 2.115x | 2.332x | 68.893% | 2.71x slower |
| 1024 | 67.921 | 38.928 | 1.745x | 1.756x | 9.555% | 1.29x slower |
| 2048 | 270.932 | 128.115 | 2.115x | 2.195x | 86.477% | 3.24x slower |
| 4096 | 524.283 | 242.190 | 2.165x | 2.167x | 34.338% | 3.02x slower |

Every cell fails the 5% baseline-control drift gate, so none is promotion
evidence. This does not prevent a tournament decision: the candidate is slower
than `r16p128t128` at every context, by a large factor in four of five cells,
and shares the same one-threadgroup structural bottleneck. Retire it from
further timing. Do not spend independent-confirmation or full-route budget on
this configuration.

## Receipt identities

Each row lists SHA-256 for `native/abba.json`, `job.json`, and queue
`report.json`, respectively.

- L256: `49db8bfd974791c0d96cdf8aece33045c0c08762755de8848f2cc731cc8e044d`, `52e8fb23d71d53ef3306d5a13604bda557f81faf6cc154dc3cc2a011a81f7c3b`, `78f0b2c1d3509e756b52f8529660f738e342493cf74e7aa23e8f21fd23a2e8e7`
- L512: `8026533255cbf6ff23ec3772f5a3e8d9d0fb0fdb39d2be95c59b9f1c0bdca8a1`, `16258b34cfde88df080f54f2eae08d504e9a86722321f7f3b8c42068bba0ad5d`, `d73b70f63430e87a37ffe8c727937a039cccca507984dbbb2902c8d6fb753883`
- L1024: `ab862c9c2b79f8d6da7a7023e23ddd1197a7768021aff832b12fc8e533a0efb3`, `27b37041fbb7009edf51506d8339a8ffc496ff35cc48aebe1ca7b823cfb12dab`, `771ef4c5194f227fdd3b0bf3513845f164421f6190f24ad4a2270261e433864e`
- L2048: `825695209f627e6c0b62bd4441da81b79721c587befedfdb33c7e4d099abf85d`, `c8e68a498f5c4eb2a7b851244651a94a34c9995ffd0a4904c86326ec236fd948`, `40808494b8d9addb19ad4376aba433581f845ed93ec573af2e83cc73a467a7af`
- L4096: `da2c4f921b10cfdb6a32920778ac215104bdb56ffcad010b6cfc0eb81750dedf`, `5a253512b210b349e79ff2b30f722906341f1d4d28cdffd6c41bd072da431bb3`, `2225666378564c01acd806e79afc40d914bed8a596c7e45399a9ecec9bcad511`
