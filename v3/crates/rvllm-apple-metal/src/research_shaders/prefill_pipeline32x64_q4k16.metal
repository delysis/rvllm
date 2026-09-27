// Default-off prefill tournament. See prefill_round.rs and the handoff.

kernel void research_prefill_combined_gemm(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 storage[512]; // 8192 bytes; operand/output lifetimes do not overlap.
    pr26_project<true, false, false>(A, B, C, M, N, K, alpha, beta,
        group, tid, sg, threads, storage);
}

kernel void research_prefill_combined_qkv(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 storage[512]; // 8192 bytes; operand/output lifetimes do not overlap.
    pr26_project<true, true, false>(A, B, C, M, N, K, alpha, beta,
        group, tid, sg, threads, storage);
}

kernel void research_prefill_combined_d256(
    device const half *Q [[buffer(0)]], device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]], device half *O [[buffer(3)]],
    device const int *tables [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *cu [[buffer(6)]], device const int *positions [[buffer(7)]],
    constant uint &M [[buffer(8)]], constant uint &batch [[buffer(9)]],
    constant uint &heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]],
    constant uint &dim [[buffer(12)]], constant uint &block [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]], constant float &scale [[buffer(15)]],
    constant uint &window [[buffer(16)]], constant uint &num_blocks [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup half panel[16 * 256]; // K and V reuse these bytes, after a barrier.
    threadgroup float scores[4 * 16]; // Also holds FP32 probabilities; never BF16.
    threadgroup int pages[16];
    pr26_attention<256>(Q, K, V, O, tables, contexts, cu, positions,
        M, batch, heads, kv_heads, dim, block, max_blocks, scale, window, num_blocks,
        group, tid, sg, lane, threads, panel, scores, pages);
}

kernel void research_prefill_combined_d512(
    device const half *Q [[buffer(0)]], device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]], device half *O [[buffer(3)]],
    device const int *tables [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *cu [[buffer(6)]], device const int *positions [[buffer(7)]],
    constant uint &M [[buffer(8)]], constant uint &batch [[buffer(9)]],
    constant uint &heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]],
    constant uint &dim [[buffer(12)]], constant uint &block [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]], constant float &scale [[buffer(15)]],
    constant uint &window [[buffer(16)]], constant uint &num_blocks [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup half panel[16 * 512]; // K and V reuse these bytes, after a barrier.
    threadgroup float scores[4 * 16]; // Also holds FP32 probabilities; never BF16.
    threadgroup int pages[16];
    pr26_attention<512>(Q, K, V, O, tables, contexts, cu, positions,
        M, batch, heads, kv_heads, dim, block, max_blocks, scale, window, num_blocks,
        group, tid, sg, lane, threads, panel, scores, pages);
}


kernel void research_prefill_combined_raw_norm_projection(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 storage[512]; // 8192 bytes; operand/output lifetimes do not overlap.
    pr26_project<true, true, true>(A, B, C, M, N, K, alpha, beta,
        group, tid, sg, threads, storage);
}

kernel void research_prefill_combined_raw_norm(
    device const float *X [[buffer(0)]], device half *O [[buffer(1)]],
    device const half *G [[buffer(2)]], constant uint &N [[buffer(3)]],
    constant float &eps [[buffer(4)]], constant uint &M [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
    threadgroup float sums[256];
    pr26_raw_norm(X, O, G, N, eps, M, row, tid, tpg, sums);
}
