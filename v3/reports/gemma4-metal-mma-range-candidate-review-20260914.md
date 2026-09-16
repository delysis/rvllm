# MMA range candidate: independent source review

2026-09-14. Read-only review of `layer_forward.rs` and `apple_metal_backend.rs`, with relevant tests inspected. No builds, tests, hardware, cache access, or implementation edits.

**No blocking parameter, output-storage, or batched-decode regression found in the inspected candidate.** Continuing component and full-model qualification is reasonable. This is source-review clearance, not numerical or performance acceptance for M6–1024.

Reviewed source SHA256 at completion:

- `layer_forward.rs`: `a6cd1320ff6b31c732ad6f5c81ad8290199024e087950da3b2aaae52b71916c2`
- `apple_metal_backend.rs`: `52fd5fad3dcb0d0b7b1de8dd673d989795eaf62e394f3056e04c4d0c0bc7ed0b`

## Confirmed wiring

- [Eligibility resolution](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:67) requires an actual prefill phase, MMA opt-in, native BF16, Apple9/10, non-quantized diagnostic accumulation, M6–1024 and the specified 12B dimensions. Layer encoding additionally requires no trace at line 697. Batched **decode** therefore resolves false regardless of its M value. The unchanged batch8/GEMV predicates retain their prior behavior.
- All **12 `encode_gemm_with_output` calls** have the new final `allow_prefill_mma` argument in the correct position, following `output_f32`. There is only one `output_f32=true` call: the packed QKV projection at [line 870](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:870). Ordinary QKV/split and trace buffers retain two-byte output mode and cannot match the FP32-only QKV MMA shapes. Gate/up, O, down, and PLE call sites use two-byte output mode. The existing four-byte QKV scratch allocation and planar check remain applicable.
- Both `encode_gemm_rmsnorm` arguments and its nested GEMM forwarding preserve `(a,b,gamma,c,m,n,k,eps,op,allow)`. [Split selection](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:2704) calls the same `metal_gemm_rmsnorm_encoder_count` helper used by the runtime. Its projection stores BF16 before in-place RMSNorm, with input/output offsets unchanged.
- [Runtime estimator inputs](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_metal_backend.rs:3898) resolve the same trace/phase/dimension policy and pass it in the correct new parameter position. The subsequent QKV boolean remains separate. The [RMS bonus](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_metal_backend.rs:4100) uses the resolved boolean, while QKV's existing trace/debug/norm checks guard its separate extra encoder.
- The [auxiliary GEMM wrapper](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:5212) passes `false,false`. Consequently LM-head/finalization and model-level PLE preparation do not inherit the widened MMA policy. The matrix selector uses `allow_prefill_mma && is_prefill_mma_shape`, with unchanged output strides/grid geometry.
- [Numeric ABI is now 9](/Users/george/Downloads/rvllm/v3/crates/rvllm-runtime/src/apple_metal_backend.rs:262), so this Rust-only routing change changes the fingerprint even though the shader implementation is unchanged. Keeping the separate old batch8 M20–32 marker is correct because that fallback range did not change.

## Remaining review notes

1. **Guard completeness, nonblocking for this checkpoint:** the eligibility dimensions do not inspect `moe_num_experts`, `moe_top_k`, `moe_intermediate`, or `ple_dim`. A future/synthetic MoE or PLE architecture matching the listed H/I/layer/head geometry would resolve true, despite the helper's “unrelated architectures” comment. If the contract is strictly dense 12B, add zero checks for these fields to the common dimension predicate. The selected GEMM shape and output guards still prevent a storage-type mismatch; this is a scope guard, not an observed failure on the current model.
2. **Tests validate shape/count policy, not the complete resolver yet.** The updated pure test around [line 4411](/Users/george/Downloads/rvllm/v3/crates/rvllm-apple-metal/src/layer_forward.rs:4411) covers range, output ABI and larger-M counter decisions. Add direct coverage of prefill versus decode, trace exclusion, wrong dtype/family/architecture, and opt-out behavior when practical. The expanded component fixture dispatches pipelines directly, so passing it cannot alone prove those resolver branches.
3. The accepted candidate is intentionally **continuous M6–1024**, including M7–19 and M33–63. The updated shape tests reflect that decision. The current extended component fixture covers M6,63,84,230,650,1024, but still lacks actual global QKV/O matrices and a combined O/down→RMS oracle. Those are qualification limits, not source blockers. Trace now disables all layer MMA, so a traced run does not prove the untraced candidate path.

The previously identified long-M numerical change remains deliberate: old fused projection/RMS used FP32 unrounded projection values; the candidate uses the already qualified short-M BF16 projection→RMS boundary. Validate against that intended oracle and actual imported-KV continuation, not strict equality to the old fused fallback. No coexistence/residency performance claim is made by this review.
