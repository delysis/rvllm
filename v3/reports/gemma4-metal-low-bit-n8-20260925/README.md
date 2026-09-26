# Gemma 4 Metal native-BF16 low-bit N8 candidate

This default-off research arm computes eight adjacent output channels per SIMD
group for W4 and W8. It doubles the accumulator footprint of N4 to reuse each
BF16 activation load across eight dot products. FP16 group-32 scales, FP32
accumulation, explicit output tails and one BF16 output rounding boundary are
unchanged. The dispatch grid is `ceil(N / 8) x M`.

The strict real-weight referee selects this arm only with `--candidate n8`.
Scalar and N4 remain separately selectable; shipping defaults are unchanged.
Rust compilation and generated BF16 Metal 3.1 compilation/linking pass.

All fourteen seven-role screen and independent-confirmation jobs completed.
All 56 real-weight correctness cases passed exact candidate schedule/kernel
identity, guard, bitwise-repeat, dispatch-count and BF16 ABI checks. Conditions
were recorded but never used as a stability wait gate.

The strict repeat policy requires both repetitions to beat native BF16 by at
least 1.05x and candidate/native/speedup median drift to stay within 20%.
Only four of 28 role/format/M cells satisfy it:

| Role | Format | M | screen | confirmation |
| --- | --- | ---: | ---: | ---: |
| K | W8 | 4 | 1.2726x | 1.2493x |
| O | W4 | 1 | 1.0799x | 1.1403x |
| O | W4 | 4 | 1.6437x | 1.6619x |
| Down | W8 | 4 | 1.6883x | 1.6069x |

This is not a promotable campaign-wide winner. It establishes four partial
operator wins. N4 and N8 were each compared independently with native BF16;
there is no counterbalanced N4-versus-N8 result and therefore no justified
production selector between them. Across both schedules the union contains
11 stable cells out of 28; N8 contributes three cells not already won by N4.
See `summary.json` for the compact adjudication and `queue-receipts/` for the
unaltered queue evidence.
