# Opt-in QKV prefill implementation review

Read-only review, 2026-09-14. **No finding requires stopping untraced, default-accumulation BF16 full-model qualification.** Two non-blocking defects and one qualification caveat follow. No builds, hardware execution, cache operations, or implementation edits were performed. The reported approximately 2.8× component speedup comes from the parent task's actual-weight/controlled-input run, not a measurement made by this review.

## Actionable findings

1. **P2 — reject or disable the candidate for the quantized-accumulator diagnostic mode.** [Eligibility](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:28) permits `RVLLM_METAL_BF16_ACCUM=quantized`, but the [source rewriting](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:2464) quantizes the old scalar QKV `acc += ...` expression and does not match the new [array accumulator](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/kernels.rs:431). Consequently the candidate's QKV semantics silently differ from the selected diagnostic policy. This does not affect the intended default FP32-accumulation run. Prefer retaining the old path or rejecting this combination over widening this experiment to new accumulation semantics.

2. **P2 — encoder telemetry omits the added projection.** The new [projection plus epilogue dispatch](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:823) emits two encoders. [The runtime estimate](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_metal_backend.rs:4056) still counts the original fused QKV path, with no candidate bonus. A selected 48-layer prefill therefore undercounts by 48. Update accounting from the same effective dispatch decision; this affects evidence and diagnosis rather than generated values. Do not use the present estimate as proof that the candidate executed.

3. **Qualification caveat — tracing selects the fallback.** The candidate depends on [the existing `trace.is_none()` fusion gate](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:682). Capturing traced layer intermediates can therefore produce correct fallback results while the candidate environment flag is set. Qualify without layer tracing, or record actual candidate dispatches and use a capture mechanism that leaves that path active. Ordinary KV export after an untraced prefill can inspect candidate results. This is inherited behavior, not a new arithmetic defect.

## Verified source properties

| Area | Finding |
|---|---|
| Capacity and alignment | [Planning](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/gemma4_model.rs:1596) and [allocation](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/gemma4_model.rs:2360) both reserve four bytes per packed QKV element for H=3840/48 layers. Regions have 16-byte alignment. Additional execution slots [copy the enlarged region size](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/gemma4_model.rs:2857). |
| Scratch separation | [Validation](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:248) uses four-byte projection lengths for the candidate and two-byte planar Q/K/V lengths. For model-created regions, capacity and disjointness are consistent. The offset-only validator is not a general proof of arbitrary externally supplied buffer capacities. |
| Projection indexing | `[M,K] × [N,K]ᵀ → [M,N]` uses N=8192 sliding or 9216 global. The kernel reads only `k<K`, `col<N`, and `row_base+row<M`; all lanes participate in `simd_sum`. `ceil(M/8)` groups handle M tails without skipped rows. |
| Epilogue indexing | Packed stride `(num_q_heads+2*num_kv_heads)*head_dim` equals the projection's N. Q uses token stride `q_dim`; K/V use `kv_dim`. Global repeated K-for-V rows remain unchanged. |
| Precision | Projection stores `float`, and the epilogue's A pointer remains `float` after BF16 source generation. No intermediate BF16 projection conversion was added. FP32 head RMS statistics consume the projected values; normalization rounds to BF16 before RoPE, and output rounds to BF16 afterward. |
| RoPE/cache | Direct textual comparison found the epilogue arithmetic identical to the original shader except for loading the projection from FP32 scratch. Head-half pairing, proportional-RoPE untouched channels, actual gamma, V gamma policy, negative-slot suppression and physical cache addressing are preserved. |
| Selection | Explicit opt-in, Apple9/10, M=20–32, H3840/I15360/48 layers/16 Q heads and the two intended KV geometries; prefill phase only. Existing norm/RoPE validity gates also apply. The opt-in is not restricted to BF16, so BF16 acceptance alone should not be presented as FP16 acceptance. |

The maximum packed scratch requirement within the gate is `32×9216×4 = 1,179,648` bytes per execution slot, reused across layers. Changing reduction order still changes rounding even though the FP32-before-RMS boundary is preserved; byte identity is not implied. The runtime fingerprint includes the opt-in and numeric ABI is seven in this inspection.

## Validation boundary

The [component test](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/qkv_prefill_tests.rs:13) compares complete paths with actual QKV/gamma weights, scaled synthetic BF16 activations, both geometries at M21/M28, guards, reversed slots, and one suppressed cache write. That is useful coverage of tails, strides and the epilogue. It calls the encoding helpers directly, so it does not exercise production eligibility, model scratch allocation, or the trace gate. Its relative-L2 threshold is a component gate, not independent full-model quality proof.

Proceed with the planned untraced full-model run while recording the active dtype, accumulation mode, candidate dispatch count, and exact executable/source identity. Before promotion, add focused host coverage of M19/20/32/33 eligibility and the case where two-byte packed scratch would fit but four-byte scratch overlaps planar Q. Actual captured model activations and the established mixed-precision first-token/continuation oracle remain the decisive numerical check. No new upstream survey is needed; the change follows the [previous source audit](/Users/george/Downloads/rvllm/v3/reports/gemma4-metal-short-prefill-kernel-audit-20260914.md).

Inspected SHA-256 values (uncommitted working tree; paths under `v3/crates`):

| File | SHA-256 |
|---|---|
| `rvllm-apple-metal/src/kernels.rs` | `79abd2d2d3026a4ac8b4b696323aea31114d146e7fcad4fe825291ae3744de35` |
| `rvllm-apple-metal/src/layer_forward.rs` | `836416539f3ecff15094d8726621b79798f5581ff04c44e95693b6607f12dd90` |
| `rvllm-apple-metal/src/gemma4_model.rs` | `ff34a60d3dd9e667ea19ba728e8c9afa5f651c0e910b57c656dcb3fba605cfe4` |
| `rvllm-apple-metal/src/qkv_prefill_tests.rs` | `e77e86f90a9909562b76da68cdf11cbc758a6eb8e640f3e2a54fc1d10f00c063` |
| `rvllm-runtime/src/apple_metal_backend.rs` | `ef603a1aadc5e6e084122478efb74ea75eea9e36c228d5ca597807f565cf47c1` |
