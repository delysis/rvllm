# Gemma 4 direct N4-vs-N8 Metal results

All 22 jobs and 11 cells were validated. This is projection-operator evidence only: it is not a full-route, model-quality, shipping-selector, or promotion claim. Raw dispatch-order timings, queue measurements, condition observations, and evidence hashes remain in `summary.json`.

Campaign disposition: **mixed_or_inconclusive_operator_results_not_promotion**.

| Role | Format | M | ABBA N4/N8 | BAAB N4/N8 | N4 drift | N8 drift | Ratio drift | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| down | w4a16 | 1 | 1.745x | 1.591x | 93.2% | 111.9% | 9.7% | inconclusive_cross_order_drift |
| down | w4a16 | 4 | 1.175x | 1.243x | 72.0% | 81.9% | 5.8% | inconclusive_cross_order_drift |
| down | w8a16 | 1 | 1.863x | 1.987x | 34.7% | 43.6% | 6.7% | inconclusive_cross_order_drift |
| down | w8a16 | 4 | 1.132x | 1.041x | 76.7% | 92.3% | 8.8% | inconclusive_cross_order_drift |
| gate | w8a16 | 4 | 1.022x | 1.165x | 53.9% | 35.1% | 13.9% | inconclusive_cross_order_drift |
| k | w8a16 | 4 | 1.113x | 1.185x | 0.1% | 6.3% | 6.5% | n4 stable winner |
| o | w4a16 | 1 | 1.495x | 1.522x | 16.7% | 14.7% | 1.8% | n4 stable winner |
| o | w4a16 | 4 | 1.170x | 1.137x | 3.9% | 0.9% | 2.9% | n4 stable winner |
| up | w8a16 | 1 | 0.884x | 0.925x | 1.7% | 6.4% | 4.6% | n8 stable winner |
| up | w8a16 | 4 | 1.174x | 1.270x | 76.8% | 91.3% | 8.2% | inconclusive_cross_order_drift |
| v | w4a16 | 4 | 1.021x | 1.107x | 16.7% | 7.5% | 8.5% | stable_no_5_percent_winner |

A winner requires at least 1.05x in both ABBA and BAAB and no more than 20% ABBA/BAAB drift in the N4 median, N8 median, or reciprocal timing ratio. Failed and inconclusive cells remain represented; no favorable subset is selected.
