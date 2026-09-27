# SG8 mechanism and prefill audit (2026-09-26)

This is a source-and-receipt audit of the **BF16 Gemma 4 12B** donor SG8
route, not a promotion or a claim of causal attribution. It distinguishes
observed route timing, dispatch evidence, source-level mechanisms, and tests
that remain necessary. The accompanying `sg8-referee-bundle-fb5f169c.zip`
reproduces the already received diagnostic patch (patch SHA-256
`6916b7c4e8b48a73004001bdb3c6bed2390ca7d98a193e7e3fad1caf82492d20`);
it is **not** a second independent result. Its first-internal-difference
producer and actual Metal API compile-call counter are not implemented.

## What is measured

The exact integration record and raw queue receipts are in this directory's
`README.md` and `queue-results/`. On the real BF16 model, the paired 256-token,
two-decode-token SG8/off/off/SG8 screen measured 333.15/1237.18/1237.64/
337.98 ms decode. The 512-token, 64-token exploratory screen measured SG8
4,685.89 ms and selector-off 55,590.12 ms decode, but the free-running
continuations diverged and those are not a promotable timing pair. The 64-step
256-token run reproduced the first token-ID difference at zero-based generated
index 63; an earlier 512-token run first differed at 36. Matching the first
two tokens is a narrow smoke check, not numerical equivalence.

The 256–2048-token two-step receipts record, per case, 16 global-attention,
80 local-attention, 96 native-gate, and 288 native-projection donor dispatches.
Those are the two decode steps: they do not show a donor projection doing the
large prompt matrix. The widening selector-off/SG8 decode gap as context grows
is consistent with an attention advantage, but may also reflect other route
and host interactions. No attention-versus-projection fraction has yet been
measured on the real full route.

The four *synthetic same-weight* hot-cache W4/W8 projection cells measure
2.1–2.9x SG8/SG4 speedups over the rvLLM native-BF16 N4 experimental
projection. These neither exercise the BF16 full-route weights nor compare
against MLX. They establish a promising low-bit operator mechanism, not the
cause of the BF16 full-route improvement.

## Source-bound mechanism hypotheses

| Component | Donor source behavior | Plausible advantage; not yet isolated |
| --- | --- | --- |
| Low-bit M1 projection | Eight consecutive codes and activations per lane, one scale per group-of-32, four adjacent output rows per SIMD group, 256-thread group; packed W4/W8 word loads and explicit sign extension. | Fewer scale/index/loop operations, more contiguous loads, and greater group occupancy than the incumbent N4 kernel's one `k` per lane per 32-step iteration. The source alone does not establish generated instruction counts, occupancy, or each mechanism's share of the 2.1–2.9x. |
| Native-BF16 M1 projection | Four rows per SIMD group, eight `K` positions per lane, 256-wide K blocks, FP32 partials and `simd_sum`, true FP32 QKV output when requested. | More activation reuse and coalescing than a scalar/short-vector fallback. The changed reduction order is also a candidate source of numerical differences. |
| Native gate/up | One dispatch computes both halves with shared activation loads, rounds each FP32 projection to BF16 before GELU and multiply, then writes activation. | Avoided projection dispatch/materialization and shared X reads; must compare against an otherwise identical unfused route. |
| Decode attention | One threadgroup per head; local uses 16 SIMD groups and global 8, with strided independent online-softmax scans and one within-group sufficient-statistics merge. | More parallelism across a growing KV context, without an extra merge launch. Needs complete-attention A/B timing and per-context correctness, especially newest K/V. |
| Low-bit M8 batch | Weights are dequantized once for eight tokens, two rows per SIMD group. | Amortizes weight work at small prompt microbatches; no evidence of benefit for a 512-token prefill. |

One source-level contrast is now unusually clear. The incumbent
`supports_attention_decode_online` requires `head_dim <= 256`; Gemma 4 12B's
global layers have D512, so selector-off reaches `attention_decode_f16` with
**one thread per head** (`tpg.width = 1`). That shader serially loops over
visible KV tokens and all 512 head elements for its dot and output update.
The donor global attention uses eight SIMD groups (256 threads) per head and
strides the token scan across them before a within-threadgroup merge. This
is a concrete architecture-level reason for expecting a large, context-growing
global-attention advantage. The current full-route component screen below
tests whether that mechanism actually dominates; source inspection alone
still cannot assign an exact number of milliseconds to it.

Sources: `v3/crates/rvllm-apple-metal/src/research_shaders/donor12b_common.metal`,
`donor12b_sg8.metal`, `donor12b.rs`, `donor12b_metal.rs`,
`layer_forward.rs`, and the incumbent N4 kernels in `kernels.rs`.
The shader is compiled with Metal 3.1 and `-fno-fast-math`; generated MSL and
metallib identities are sealed in the existing report. Source-level claims
about register residency, vector load instructions, or occupancy require
generated-code/device resource inspection; shader syntax is not proof.

## Why the 512-token prefill is still slow: known boundary, unknown division

At 512 prompt tokens the observed prefill was 22,553.54 ms SG8 versus
22,707.15 ms selector-off in one sequential 64-token run. Same-host MLX
recorded 173.051 prompt tokens/s in an eligible seven-trial run, yielding
an exploratory roughly 7.6x rvLLM/MLX prefill gap. The model package,
prompt-token values, continuations, and interleaving were not matched, so
this is a bottleneck signal, not a qualified cross-framework ratio.

The donor projection planner accepts only `1 <= M <= 128` and the donor
attention planner accepts decode only. Thus a full 512-row prompt cannot use
the donor projection/attention primitive, and the full-route dispatch ledger
shows only decode donor projections. The central GEMM dispatcher can choose
MMA32 only when `prefill_mma32` is enabled and the shape is eligible; otherwise
it falls through batch8, vector, tiled16, or general GEMM. The long-decode
wrapper uses `rvllm_metal_infer` and does not set a prefill variable; that
binary's `ModelMetalBackend` reads `MetalKernelOptions::from_development_environment`,
which defaults prefill MMA32 and SIMD attention **off**. The distinct
`rvllm_disaggregated_infer` CLI opts both on by default, but it did not produce
these receipts. Thus the current long-decode screen did not exercise those
experimental prefill arms. The actual chosen fallback prefill kernels and per-operation GPU
durations were not captured in these receipts. It is therefore justified to
say *the donor did not fix prefill*, but **not** yet justified to assign the
22-second prefill to a particular GEMM, FFN, attention, queue wait, or host
cost. The opt-in stage diagnostic returned all-zero GPU timestamp spans on
this host and correctly marked them unsupported.

## Experiments needed to identify cause, without corrupting the comparison

1. Add a cold-path, immutable *route mask* for diagnostic binaries only:
   projection, gate, local attention, and global attention independently
   selectable, all-other-operations identical. Parse it before model execution;
   record mask, actual per-family dispatch, executable/metallib/source hashes,
   zero inference compile calls, prompt IDs, and output IDs. Reject any arm
   whose intended route did not dispatch. Compare all-off, each single-on,
   leave-one-out, and all-on in queue-controlled ABBA/BAAB blocks at 256 and
   2048 contexts. Interactions mean single-on effects must not be summed into
   an invented percentage. Count both launches for any split+merge arm.
2. For the low-bit operator, run same-weight M1/M8 N4-versus-donor arms with
   source and generated-Metal inspection: instruction mix, register/threadgroup
   limits, packed loads, scale loads, occupancy, and resource spills. Test one
   schedule feature at a time only after the route-mask result prioritizes it.
3. Localize the first BF16 numerical difference with **production-route**
   snapshots of final prefill state, then decode-step/layer input, projected
   QKV, updated K/V, attention output, gate/up, down, residual, norm and direct
   sampled logits. Compare SG8/off in the same process/weights/prompt, first
   at decode step 0, then binary-search layers/steps as needed. Preserve the
   real BF16 storage boundary and a route that does not disable fused gate.
   Token index 63 is not itself a page-rollover proof: the report identifies
   decode step 63 at position 318, slot 30 of a 32-token page.
4. For prefill, capture the *prefill* dispatch ledger before decode overwrites
   the last-step ledger; record actual GEMM/attention/FFN kernel selection and
   counts by layer/role. Since device timestamps are unsupported here, use
   queue-owned bounded stage isolation or a validated GPU trace, with
   identical prompt/model/buffer work and controls, to separate projection,
   FFN, attention and host/command-buffer cost. Compare 256/512/1024/2048
   prompt lengths and an independently checked MLX operation baseline. Enable
   the separately qualified prefill-MMA arm only as an explicit distinct arm.

Do not promote SG8 on two-token agreement or the exploratory 64-token speed
gap. Preserve its speed lead as an engineering opportunity while the internal
first-difference and checkpoint-quality gates remain open. Likewise, do not
"fix" donor reduction arithmetic from the source hypothesis alone: first
capture the earliest differing internal value and its reference tolerance.
