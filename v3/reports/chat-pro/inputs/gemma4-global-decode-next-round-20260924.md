# Gemma 4 global-attention decode: next implementation round

Implement the next evidence-driven Metal global-attention decode round for rvLLM. Remain in ordinary Chat; do not switch to Work.

## Exact integration point

- Repository: `delysis/rvllm`
- Base commit: `499f6a31`
- Current candidates: eight unsplit cooperative D512/GQA-16 decode kernels spanning rows `{8,16}`, panel `{64,128}`, threads `{64,128}`.
- Current route is opt-in research only; production defaults must remain unchanged.
- The checked-in queue/referee contracts, exact dispatch evidence, source/metallib identities, and native correctness oracle are authoritative. Do not weaken them.

## Native evidence now available

All eight candidates compiled in both core and oracle forms (16/16 arms) and passed the real-device oracle on an Apple M4 Max. Each oracle covers 18 positive/tail/rollback cases plus refusal and mutation controls, exact FP32 serial-GPU comparison, once-rounded BF16 output, independent sampled FP64 dot bounds, guards, and repeated-use stability.

The exploratory ABBA tournament now screens lengths 256/512/1024/2048. Each cell uses five warmups per arm and five complete ABBA blocks, 100 dispatches per sample, retaining every observation. Length 4096 is deferred until a candidate survives the cheaper screen. The queue logs changing conditions but no longer waits for a stable-condition dwell interval.

The complete successive-halving result is:

| Context | `r8p128t128` | `r8p64t128` | prior `r16p128t128` | Survivor set |
|---:|---:|---:|---:|---|
| 256 | 4.525 ms | 4.918 ms | 4.824 ms | all three |
| 512 | 7.840 ms | 8.293 ms | 9.009 ms | all three |
| 1024 | 15.108 ms | 16.473 ms | 30.178 ms | two 8-row kernels |
| 2048 | 28.377 ms | **23.399 ms** | 39.593 ms | `r8p64t128` only |

`r8p64t128` is the unsplit-family winner: 1.692x faster than the prior leader
at 2048, reducing candidate time by 40.9%. It is still exploratory rather than
promotable because its baseline control drift was 58.5%. Every observation was
retained. Only the prior leader's 256 and 512 cells and its 2048 cell passed the
strict 5% drift gate. The incumbent baseline itself is unexpectedly slow and
variable, so audit the baseline implementation and avoid attributing all gain
to candidate quality without first-principles accounting.

The winner is nevertheless nowhere near MLX. Its dispatch grid is still exactly
one 128-thread threadgroup for all 16 query heads. The matched isolated MLX BF16
`attention_core_sdpa/full_attention` decode case takes 1.489 ms, so the 23.399
ms winner remains about 15.7x slower for this operator. Generated MSL still
shows serial context blocks, barrier-separated K and V panels, leader-lane
precise exponentials, and a single threadgroup retaining all online
softmax/output state. Treat the unsplit family as correctness scaffolding.

Separately, the current whole-model rvLLM BF16 Metal route falls behind sustained MLX decode increasingly with context: approximately 3.33x at 256, 4.82x at 512, 7.78x at 1024, 16.35x at 2048, and 31.12x at 4096. At 4096 its last decode step spent about 4.296 seconds in GPU/command-buffer wait. Those whole-route measurements are exploratory and used only two rvLLM decode steps versus 64 sustained MLX steps, but they strongly localize the long-context global-attention route as a major problem.

## Requested implementation

Deliver a reviewable source patch/archive on top of `499f6a31` implementing the highest-value next vertical slice, not merely prose. Implement a mathematically sound split-KV global D512/GQA-16 decode design; another unsplit row/panel/thread tuple is not responsive to the measured bottleneck.

Requirements:

1. Add a bounded candidate family with explicit resource budgets and exact catalog identities. Avoid combinatorial explosion; justify each candidate from occupancy, memory traffic, reductions, and expected crossover length.
2. Preserve FP32 accumulation, BF16 output rounding boundaries, paged-KV semantics, GQA head mapping, rollback/prefix behavior, guard safety, and refusal without mutation.
3. Integrate with the existing native oracle and kernel-game campaign generator. Add any split/reduction intermediate buffers to the sealed workload and route evidence; do not hide allocation or compilation inside timed regions.
4. First-round benchmark design must cover decode contexts 256/512/1024/2048, use ABBA or a more robust paired design, retain invalid/inconclusive cells, and never select favorable samples. Reserve 4096 for survivors. Promotion still requires the strict drift gate and independent confirmation.
5. Audit why `attention_decode_f16` costs roughly 124–140 ms per dispatch at 1024–2048 in this fixture. State whether the baseline is doing equivalent work and fix the comparison if it is not. Distinguish candidate improvement from baseline pathology.
6. Keep production defaults unchanged. All new paths remain research opt-in until independently qualified.
7. Provide safe idiomatic Rust, Metal source, focused tests, exact commands, and a concise handoff describing unverified boundaries. Do not claim speedups you have not measured locally.

Return downloadable source files or a patch suitable for exact-base application. Include a manifest of changed files and expected compile/oracle/timing commands.
