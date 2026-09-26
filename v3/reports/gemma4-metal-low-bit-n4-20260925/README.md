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

The initial implementation checkpoint contained compiler and host-contract
evidence only. The first real-weight screen below adds one-role operator
evidence; seven-role coverage, independent timing confirmation, full-route
performance and checkpoint-quality acceptance remain required before any
promotion claim.

## First real-weight screen

The layer-0 Q projection passed W4/W8 at M=1 and M=4 against the independent
CPU quantized reference, including guards, bitwise repeatability, exact route
counts and strict `n4` kernel identity. Screen speedups against the native BF16
GEMM were W4 M1 **1.142x**, W4 M4 **1.405x**, W8 M1 **0.802x**, and W8 M4
**1.097x**. These are single-screen observations, not stable winners. The
remaining six roles are staged for the same shortest-first screen.

## Seven-role adjudication

All 14 screen/confirmation jobs and all 56 correctness cases succeeded with
strict schedule/kernel identity, guards, bitwise repeatability and exact
dispatch accounting. Under the existing 20% drift and 1.05x-in-both-runs
policy, **8/28** timing cells are stable operator wins:

- V W4 M4: 1.421x / 1.341x.
- Gate W8 M4: 3.056x / 3.097x.
- Up W8 M1: 1.963x / 1.876x; M4: 3.057x / 3.181x.
- Down W4 M1: 1.760x / 1.764x; M4: 2.482x / 2.688x.
- Down W8 M1: 1.707x / 1.598x; M4: 3.188x / 2.898x.

The other 20 cells are unstable or fail the repeated speed margin. Campaign
disposition: **not promotable; partial operator wins**. In particular, the N4
schedule is shape-sensitive and cannot be applied indiscriminately to every
projection role. The stable rows justify route-specific full-model experiments
and a second materially different schedule for the remaining roles.
