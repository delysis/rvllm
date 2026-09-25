# Gemma 4 attention Atlas intake and first native screen

Date: 2026-09-25

## Evidence boundary

The packet `/Users/george/Downloads/rvllm-attention-atlas-2bc5a535` was verified
against its declared base `2bc5a535b113c0bcc9b1f5e15ab747005e2e3c60` before
porting. Its packet hashes and post-application bytes matched its manifest. The
packet's original Linux evidence was not treated as native qualification.

The current-tree port is default-off behind `attention-atlas-research`. The only
semantic compatibility edit was to initialize the current `DecodeTile` fields for
the packet's eight legacy controls as K8, per-key, non-matrix controls. Queue
requests use `stable_seconds=0`; thermal, low-power, and power-mode requirements
are unset and observed conditions are retained.

The suite contains 158 configurations and generates 776 workload specifications.
Its ANE module is layout/refusal machinery only: it does not execute an ANE
attention graph and is not evidence for ANE speed or correctness.

## Component evidence

- `cargo check -p rvllm-apple-metal --features attention-atlas-research --bin rvllm-attention-atlas`: pass.
- `cargo test -p rvllm-apple-metal --features attention-atlas-research attention_atlas --lib`: 21 passed.
- Release runner SHA-256: `b430ae8d98676e05d730c7efa73718732d3fcc37caab9be413cdf152de2feac3`.
- Generated catalog: 158 configurations.
- Generated campaign: 776 specifications plus `campaign.json`.

## First global D512 native screen

Workload: Q1, 1,025 live keys, one global KV head, D512, BF16 cache, mixed
synthetic fixture. Each candidate first passed native compilation and its exact
operator-only FP64/BF16/repeat/guard oracle. Timing is seven ABBA blocks (14
measured invocations per arm); no sample was removed or retried.

| Candidate | Candidate GPU median | Scalar control median | Control / candidate | Outer-control drift |
| --- | ---: | ---: | ---: | ---: |
| cooperative per-key R8/K8/P64/T128, split 8 | **1.451 ms** | 20.147 ms | **13.88x** | 0.014% |
| cooperative per-key R8/K8/P64/T128, split 4 | 2.813 ms | 20.206 ms | 7.18x | 0.243% |
| cooperative per-key R8/K8/P64/T128, split 2 | 5.620 ms | 20.236 ms | 3.60x | 0.307% |
| cooperative per-key R8/K8/P64/T128, split 1 | 11.057 ms | 20.119 ms | 1.82x | 0.099% |
| cooperative per-key R8/K8/P64/T128, split 16 | **0.811 ms** | 20.183 ms | **24.90x** | 0.381% |
| cooperative per-key R8/K8/P64/T128, split 32 | **0.508 ms** | 20.143 ms | **39.66x** | 0.249% |
| matrix per-tile R8/K64/P64/T128 | 5.140 ms | 20.203 ms | 3.93x | 0.605% |
| matrix per-tile R8/K32/P64/T128 | 5.234 ms | 20.177 ms | 3.86x | 0.139% |
| matrix per-tile R8/K16/P64/T128 | 5.724 ms | 20.193 ms | 3.53x | 0.129% |
| cooperative per-tile R8/K32/P64/T128 | 10.279 ms | 20.162 ms | 1.96x | 0.105% |

These are warm operator-only measurements. They exclude projection, cache
production, output projection, sampling, token delivery, and full-model scheduling.
They therefore identify a prospective attention winner, not a promotable runtime
winner.

The split-4, split-8, split-16, split-32, and matrix-K64 arms also passed the Q7/129-key verification
fixture. For split-8, FP32 max-abs was `4.91e-8`, FP32 relative L2 was `1.23e-7`,
BF16 max-abs was `4.88e-4`, BF16 relative L2 was `1.67e-3`, repeated output bits
matched, BF16 equaled RNE(candidate FP32), and rejected metadata left output
untouched. Split-4 and matrix-K64 passed the same checks.

## Lessons checked against the hand-tuned reference

The referenced Hugging Face card was inspected, and its public source was checked
at `john-rocky/coreai-model-zoo` commit
`49b8484cdd1b121592d6272b6202cdcb42a5093a` rather than relying only on the prose
summary. Its `flash_sdpa_rope_occ` implementation confirms a distinct topology:
one threadgroup per head with G cooperating SIMD groups, FP32 online-softmax
partials, and a threadgroup merge. It reports a 16 KiB partial-output array for
D512/G8. That topology is not identical to Atlas's multi-threadgroup split plus
merge and should remain a separate experimental arm.

The reference also handles the just-produced K/V row without an inter-threadgroup
barrier: the owning subgroup consumes its local normalized/rotated K/V while one
head group writes the cache. That is a useful fusion pattern, but it must not be
ported until rvLLM has an explicit single-writer cache contract and an oracle that
detects stale-row races.

The actionable design rules are:

1. Complete the split-factor sweep (1, 2, 4, 8, 16, 32) at the same workload, then
   advance the best stable factor to 2,048 keys. Split and merge remain inside the
   measured interval.
2. Add a one-threadgroup/multi-SIMD-group D512 arm so the three structures are
   compared directly: one SIMD group/head, cooperating SIMD groups/head, and
   multi-threadgroup split plus merge.
3. Keep decode and prefill fusion policies separate. The reference's fused decode
   path is evidence for a hypothesis; its reported wide-prefill regression is
   evidence against treating dispatch-count reduction as success.
4. Add full-token-loop accounting before promotion: accelerator execution,
   encode/submit, synchronization, sampling, delivery, and state/cache updates.
5. Preserve arithmetic classes. Literal-operation equivalence, reduction-order
   tolerance, and quantized model-quality equivalence are separate gates.
6. For low-bit projections, profile unpack/decode instructions and scale/codebook
   loads by role and stream size. Do not infer bandwidth limitation from packed
   bytes alone.

## Current disposition

At 2,048 live keys, split-16 passed the full operator oracle and measured 2.948 ms;
split-32 passed the same oracle and measured **0.983 ms**. Their matched scalar
controls measured 260.656 ms and 242.384 ms respectively, an unexpectedly large
change from the 1,025-key control. The candidate scaling is plausible, but the
control discontinuity means the ratios (88.4x and 246.6x) are not yet suitable as
cross-harness speedup claims. The absolute candidate medians and within-run drift
(1.02% and 1.92%) are retained as exploratory evidence.

`atlas-coop-key-r8-k8-p64-t128-s32` is the prospective leader from this first
operator screen. It is not production-selected, full-route-qualified, or
independently confirmed. The next required work is an apples-to-apples comparison
against the current rvLLM matrix leader inside one harness, followed by captured
model tensors, normal-route integration, and full-token-loop timing.
