# Gemma 4 Metal native-BF16 low-bit N4 candidate

This default-off research arm computes four adjacent output channels per SIMD
group for both W4 and W8. Each activation is loaded once and reused across four
FP32 dot products. FP16 group-32 scales, FP32 accumulation, BF16 activation and
the single BF16 output rounding boundary remain unchanged. The dispatch grid is
`ceil(N / 4) x M`; explicit tail checks prevent reads from padded weight rows.

The strict real-weight referee selects it only with `--candidate n4`. The
existing scalar candidate remains `--candidate scalar` and is still the
default. Shipping routes and defaults are unchanged.

Current evidence:

- Rust formatting and the feature-gated release build pass.
- Strict schedule parsing and distinct ABI/kernel-name tests pass.
- Generated BF16 MSL compiles and links under Metal 3.1.

This is compiler and host-contract evidence only. Real Gemma weight
correctness, guards, repeatability, timing, seven-role coverage, full-route
performance and checkpoint-quality acceptance remain required before any
promotion claim.
