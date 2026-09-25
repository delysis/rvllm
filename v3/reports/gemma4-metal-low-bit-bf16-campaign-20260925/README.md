# Gemma 4 Metal native-BF16 low-bit campaign

This is real-checkpoint projection-operator evidence, not full-route, model-quality, or MLX-comparison evidence. Changing host conditions are retained and never used as a wait gate.

Campaign disposition: **not_promotable_repeat_instability_or_regression**. Stable operator wins: **3/28**.

| Role | Format | M | Screen | Confirm | Candidate drift | Native drift | Speedup drift | Disposition |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| q | w4a16 | 1 | 1.593x | 0.927x | 401.9% | 192.2% | 71.7% | inconclusive_repeat_instability |
| q | w4a16 | 4 | 1.616x | 1.316x | 776.8% | 613.8% | 22.8% | inconclusive_repeat_instability |
| q | w8a16 | 1 | 1.614x | 1.175x | 458.0% | 306.3% | 37.4% | inconclusive_repeat_instability |
| q | w8a16 | 4 | 2.387x | 0.749x | 462.2% | 76.4% | 218.8% | inconclusive_repeat_instability |
| k | w4a16 | 1 | 1.135x | 1.452x | 683.3% | 512.4% | 27.9% | inconclusive_repeat_instability |
| k | w4a16 | 4 | 1.160x | 1.744x | 28.9% | 16.7% | 50.3% | inconclusive_repeat_instability |
| k | w8a16 | 1 | 1.328x | 1.559x | 12.4% | 32.0% | 17.4% | inconclusive_repeat_instability |
| k | w8a16 | 4 | 1.713x | 1.560x | 486.3% | 543.4% | 9.8% | inconclusive_repeat_instability |
| v | w4a16 | 1 | 1.291x | 1.150x | 1.2% | 13.7% | 12.3% | stable_operator_win |
| v | w4a16 | 4 | 1.642x | 1.349x | 27.8% | 4.9% | 21.8% | inconclusive_repeat_instability |
| v | w8a16 | 1 | 1.209x | 1.136x | 34.4% | 26.3% | 6.4% | inconclusive_repeat_instability |
| v | w8a16 | 4 | 2.854x | 1.579x | 11.6% | 101.7% | 80.7% | inconclusive_repeat_instability |
| o | w4a16 | 1 | 1.703x | 1.523x | 37.2% | 22.7% | 11.9% | inconclusive_repeat_instability |
| o | w4a16 | 4 | 1.879x | 1.266x | 1293.8% | 838.9% | 48.4% | inconclusive_repeat_instability |
| o | w8a16 | 1 | 3.098x | 0.702x | 437.8% | 21.9% | 341.3% | inconclusive_repeat_instability |
| o | w8a16 | 4 | 2.200x | 1.644x | 195.0% | 120.3% | 33.9% | inconclusive_repeat_instability |
| gate | w4a16 | 1 | 2.210x | 1.753x | 5.6% | 19.5% | 26.1% | inconclusive_repeat_instability |
| gate | w4a16 | 4 | 1.260x | 1.727x | 658.1% | 939.3% | 37.1% | inconclusive_repeat_instability |
| gate | w8a16 | 1 | 1.337x | 0.912x | 754.2% | 482.5% | 46.6% | inconclusive_repeat_instability |
| gate | w8a16 | 4 | 2.936x | 2.872x | 23.5% | 20.8% | 2.2% | inconclusive_repeat_instability |
| up | w4a16 | 1 | 1.844x | 2.065x | 30.2% | 16.3% | 12.0% | inconclusive_repeat_instability |
| up | w4a16 | 4 | 2.220x | 2.221x | 0.6% | 0.7% | 0.1% | stable_operator_win |
| up | w8a16 | 1 | 2.365x | 2.730x | 264.8% | 321.1% | 15.4% | inconclusive_repeat_instability |
| up | w8a16 | 4 | 0.729x | 1.254x | 7.1% | 84.2% | 72.0% | inconclusive_repeat_instability |
| down | w4a16 | 1 | 1.880x | 1.945x | 17.6% | 13.7% | 3.4% | stable_operator_win |
| down | w4a16 | 4 | 3.027x | 2.409x | 124.2% | 181.7% | 25.6% | inconclusive_repeat_instability |
| down | w8a16 | 1 | 0.592x | 1.875x | 785.5% | 179.4% | 217.0% | inconclusive_repeat_instability |
| down | w8a16 | 4 | 2.763x | 3.444x | 2.6% | 27.8% | 24.6% | inconclusive_repeat_instability |

Promotion is refused unless candidate median, native median, and speedup agree within 20%, and both repetitions beat native by at least 5%. Even passing rows require full-route and checkpoint-quality gates before shipping promotion. The companion JSON retains every timing sample, identity hash, accuracy result, and condition observation.
