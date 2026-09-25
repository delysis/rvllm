# Gemma 4 global D512 decode: `r16p128t128` checkpoint

Date: 2026-09-24

Source integration commit: `a2c1b383`

Summarizer commit: `bdf05641`
Device: Apple M4 Max, GPU family Apple9

This is exploratory raw-operator evidence, not promotion evidence. The candidate passed the native device oracle before timing. Every timing cell contains five warmups per arm followed by five complete ABBA blocks with 100 dispatches per sample; all 20 measured samples are retained. The strict 5% baseline-control drift gate is applied without deleting observations.

| Context | Baseline GPU ms/dispatch | Candidate GPU ms/dispatch | Ratio of means | Median paired-block ratio | Control drift | Classification |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 17.545 | 4.824 | 3.637x | 3.640x | 0.265% | exploratory stable |
| 512 | 32.290 | 9.009 | 3.584x | 3.571x | 4.923% | exploratory stable |
| 1024 | 124.331 | 30.178 | 4.120x | 4.082x | 125.113% | inconclusive: control drift |
| 2048 | 140.082 | 39.593 | 3.538x | 3.538x | 0.953% | exploratory stable |
| 4096 | 283.899 | 80.108 | 3.544x | 3.536x | 5.838% | inconclusive: control drift |

The 256, 512, and 2048 cells pass the drift gate. The 4096 cell misses narrowly and remains directional only. The 1024 baseline is pathologically variable; its candidate/baseline ratio is not admissible for promotion. The close agreement between ratio-of-means and the median paired-block ratio outside that cell suggests a real approximately 3.5–3.6x operator improvement, but full-route real-weight qualification and independent confirmation remain mandatory.

The baseline is `attention_decode_f16 (BF16 typed)`. Its unexpectedly high absolute cost must be audited for equivalent work before attributing the whole ratio to candidate quality.

## Receipt identities

Each row lists SHA-256 for `native/abba.json`, `job.json`, and queue `report.json`, respectively.

- L256: `e86809e7ca8c1d7de4f11427503736dd8d18726687816a98544efb1fe60a3b44`, `1298f4a511cb36202f4d3426bea4c2ea122225308b098da435d52e3f37c31e3d`, `411407365c626845380f0951128a34b642e7c58f87d8525a774023d7fce40937`
- L512: `b1b73a0ecc0a14988705fe3717370da3d6e0c9e65a6d5957cc9d67dff7447fad`, `bfce9d50b8b7270a73a539c61155deb9b68cb61caef76cef9c54f01f61d0a718`, `7baa9c7da7470a5b6c15ba1b7b0256dd6899a947b0ea52cfb4df347e7cbfab5e`
- L1024: `18dd01802950088120bdbe5a1eb4aec2771f8b5adbaf32cb55856a3e2887dc45`, `3211bbd0f0e665743a4c0fe3960147f9e2e4ba0fd1fd455c8bf08d91de803244`, `a99e04794f186f24ebe5ee0e31913d28c3129853f543f4981c65da9ab2d71463`
- L2048: `b505cf8aea2a28137dd9f406b4a82f2a3ab12175e5bddee798994ead14be7a3f`, `4b7aeebdd8140d2568a9279ecbd49026240adcf9e6fef43517152ec6ce7fc642`, `749bedb7f3751d337203816847b95427aba6aea26827a13975648f5e218fb72a`
- L4096: `9333b80a35f67702ab12eefc787fb7db084a56a12091ca16af7a9dab87b97b5e`, `590fe4d0dc0844e6730453ca3f6a82c6d68aa192aafda0cc64f617a069bedeea`, `e78282c62c7b034ab66914537ec9c408d464c690b04e6c863650c5b4dfab89d8`

Recompute the complete-sample summary with:

```sh
target/release/rvllm-global-decode-report /absolute/path/to/queue
```
