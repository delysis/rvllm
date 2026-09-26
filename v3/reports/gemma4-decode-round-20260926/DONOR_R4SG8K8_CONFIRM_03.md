# Donor-schedule W4/W8 operator confirmation, corrected launch identity

Campaign [`g4-donor-r4sg8k8-confirm-03`](g4-donor-r4sg8k8-confirm-03/campaign.json)
is the independent release-build rerun of the first donor-schedule screens.
The persistent serial device queue ran both strict-math Metal compilations,
both native oracles, and all three paired timings through its own two
continuation jobs. Both native oracles passed on Apple M4 Max (Apple9) at
their full synthetic operator shapes. Unlike the first campaigns, each
oracle's launch identity matches the actual shader and host dispatch:
32 output rows per threadgroup, 256 threads, and grid `[120,1,1]`.

The [W4 oracle](g4-donor-r4sg8k8-confirm-03/w4-oracle-receipt.json)
SHA-256 is `9fc9b024fac4c2d0cbbe4d379085e49330c7c02c2cee029cd3c452fc9ca9df23`;
the [W8 oracle](g4-donor-r4sg8k8-confirm-03/w8-oracle-receipt.json)
SHA-256 is `58842e29aa21ea270bfbbc043b7303a46ebb9411a5946e6c108c0725d10ec876`.
W4 covered `M1,N3840,K15360`; W8 covered `M1,N3840,K4096` and `K8192`.
Each case matched the independent scalar-FP64 bound, repeated bit-exactly
three times, kept guards and the persistent arena untouched, and exercised
four no-dispatch host refusals plus three encoded shader negatives.
The strict-math metallib SHA-256 values are W4
`5bbda735656a5502a3c7f7394f0692f8f74ce53ffa1aad089a1a674fd14747af`
and W8 `e8f3cf094d2b7ace1022fc98067620ee613645d66ced90e2fca14565ddceb777`.
These are synthetic operator checks, not real checkpoint-weight, full-route,
or model-quality qualification.

Each timing collected 40 GPU-timestamp samples in ten balanced ABBA/BAAB
blocks, with 100 actual operator iterations per sample. A is the rvLLM
group-32 BF16 N4 control; B is the donor-schedule candidate. The ratios below
are medians of the ten within-block A/B ratios. All medians are milliseconds
per operator iteration, not full-model latency or a comparison with MLX.

| Candidate | K | A median ms | B median ms | Paired ratio median | Ratio range | Control drift |
|---|---:|---:|---:|---:|---:|---:|
| W4 r4/sg8/k8 | 15360 | 0.241354 | 0.094493 | 2.558x | 2.492–2.774x | 17.94% |
| W8 r4/sg8/k8 | 4096 | 0.096845 | 0.029666 | 2.956x | 2.422–3.973x | 102.52% |
| W8 r4/sg8/k8 | 8192 | 0.156686 | 0.061590 | 2.636x | 2.453–2.984x | 46.66% |

Raw timing receipts: [W4 K15360](g4-donor-r4sg8k8-confirm-03/w4-timing-k15360-receipt.json)
SHA-256 `f782f650580b227f93d55144fa1b5ae19d320685a560a9b1720abd9691881d54`;
[W8 K4096](g4-donor-r4sg8k8-confirm-03/w8-timing-k4096-receipt.json)
SHA-256 `28778db789357698b2d252256ff5f61715061ba222d0042e0e45949bf1afded7`;
[W8 K8192](g4-donor-r4sg8k8-confirm-03/w8-timing-k8192-receipt.json)
SHA-256 `6efe96e7098fdfbc511407dec9f9d0330886fb64b665b2c020bd7fa060a4709c`.
All three receipts correctly record `timing_eligible:false` and
`promotion:false`: their control drift exceeds the referee's 5% limit.
The measured direction is a useful research lead, not a qualified speedup.
The next stage is an independent timing campaign under recorded conditions,
then real-weight/full-route and quality gates before any dispatch promotion.
