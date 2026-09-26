# Gemma 4 ANE stacked-FFN tournament

Date: 2026-09-25

The exact-tree 12-arm `ABBA/BAAB/ABBA` sequence completed. The strict summary
is `gemma4-ane-stacked-baseline-exact-v2-20260925.json`, SHA-256
`1c51a4e726a5ff0fea31254db0995a9f80ffb9d33f747b0d663ed88400730a47`.

All arms used the same sealed executable, model/config, BF16 metallib, prompt,
and generated MSL. Each arm completed nine requests with nine measured ANE
decode steps per request, produced the same ten tokens and text, used zero of a
zero compile budget, and verified actual ANE execution. The summary retains all
972 step observations and all condition strata.

The final arm's process exited successfully with intact pins, no timeout,
signal, stop, validation failure, or recorded condition violation. The queue
nevertheless labeled it `rejected` solely because its outer timing-condition
eligibility was false. The summarizer admits only this exact clean
conditions-only form as exploratory evidence and does not upgrade it to a
promotion-qualified queue success.

## Result

| Route | Attention mean | FFN mean | Host mean | Total mean |
| --- | ---: | ---: | ---: | ---: |
| baseline INT8 | 30.469 ms | 138.110 ms | 15.052 ms | 300.710 ms |
| stacked INT8 FFN | 33.053 ms | 138.126 ms | 16.441 ms | 318.920 ms |

The three complete block-level FFN effects, expressed as stacked minus
baseline, were `-17.525`, `+9.203`, and `+8.371` ms. Their mean is effectively
zero (`+0.016` ms), and two of three blocks favor baseline. Total-step effects
were `-33.817`, `+39.615`, and `+48.833` ms; their mean favors baseline by
`18.210` ms.

The stacked graph therefore is **not a performance winner** in this campaign.
Its first block looked favorable, but the counterbalanced continuation reversed
that result. This is exactly why all arms and failed/ineligible observations
were retained. The candidate remains correctness-qualified and zero-compile,
but it should not replace the baseline route on timing evidence.

This comparison is exploratory full-process timing, not a pristine-clock
promotion run. Conditions were observed and reported rather than used as a
stability wait gate, as required by the campaign policy.
