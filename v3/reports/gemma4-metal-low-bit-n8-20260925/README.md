# Gemma 4 Metal native-BF16 low-bit N8 candidate

This default-off research arm computes eight adjacent output channels per SIMD
group for W4 and W8. It doubles the accumulator footprint of N4 to reuse each
BF16 activation load across eight dot products. FP16 group-32 scales, FP32
accumulation, explicit output tails and one BF16 output rounding boundary are
unchanged. The dispatch grid is `ceil(N / 8) x M`.

The strict real-weight referee selects this arm only with `--candidate n8`.
Scalar and N4 remain separately selectable; shipping defaults are unchanged.
Rust compilation and generated BF16 Metal 3.1 compilation/linking pass. Real
Gemma correctness and speed remain unmeasured until the queued Q projection
screen completes.
