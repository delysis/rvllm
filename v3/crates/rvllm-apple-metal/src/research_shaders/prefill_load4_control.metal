// Default-off prefill tournament. See prefill_round.rs and the handoff.

kernel void research_prefill_control_gemm(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 storage[512]; // 8192 bytes; operand/output lifetimes do not overlap.
    pr26_project<false, false, false>(A, B, C, M, N, K, alpha, beta,
        group, tid, sg, threads, storage);
}

kernel void research_prefill_control_qkv(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 storage[512]; // 8192 bytes; operand/output lifetimes do not overlap.
    pr26_project<false, true, false>(A, B, C, M, N, K, alpha, beta,
        group, tid, sg, threads, storage);
}


kernel void research_prefill_control_raw_norm_projection(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 storage[512]; // 8192 bytes; operand/output lifetimes do not overlap.
    pr26_project<false, true, true>(A, B, C, M, N, K, alpha, beta,
        group, tid, sg, threads, storage);
}

kernel void research_prefill_control_raw_norm(
    device const float *X [[buffer(0)]], device half *O [[buffer(1)]],
    device const half *G [[buffer(2)]], constant uint &N [[buffer(3)]],
    constant float &eps [[buffer(4)]], constant uint &M [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
    threadgroup float sums[256];
    pr26_raw_norm(X, O, G, N, eps, M, row, tid, tpg, sums);
}
