// Default-off PR27 member; shared implementation is sealed by the source exporter.

kernel void research_prefill27_wide64_mma8k32_gemm(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 aligned_panels[(64 + 64) * 40 / 8];
    threadgroup half *panels = (threadgroup half *)aligned_panels;
    pr27_project<64, 64, 2, 2, false, false>(
        A, B, C, M, N, K, alpha, beta, group, tid, sg, lane, threads, panels);
}

kernel void research_prefill27_wide64_mma8k32_qkv(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 aligned_panels[(64 + 64) * 40 / 8];
    threadgroup half *panels = (threadgroup half *)aligned_panels;
    pr27_project<64, 64, 2, 2, true, false>(
        A, B, C, M, N, K, alpha, beta, group, tid, sg, lane, threads, panels);
}

kernel void research_prefill27_wide64_mma8k32_raw(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 aligned_panels[(64 + 64) * 40 / 8];
    threadgroup half *panels = (threadgroup half *)aligned_panels;
    pr27_project<64, 64, 2, 2, true, true>(
        A, B, C, M, N, K, alpha, beta, group, tid, sg, lane, threads, panels);
}

kernel void research_prefill27_wide64_mma8k32_norm(
    device const float *X [[buffer(0)]], device half *O [[buffer(1)]],
    device const half *G [[buffer(2)]], constant uint &N [[buffer(3)]],
    constant float &eps [[buffer(4)]], constant uint &M [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
    threadgroup float sums[256];
    pr26_raw_norm(X, O, G, N, eps, M, row, tid, tpg, sums);
}

kernel void research_prefill27_wide64_mma8k32_d256(
    device const half *Q [[buffer(0)]], device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]], device half *O [[buffer(3)]],
    device const int *tables [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *cu [[buffer(6)]], device const int *positions [[buffer(7)]],
    constant uint &M [[buffer(8)]], constant uint &batch [[buffer(9)]],
    constant uint &heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]],
    constant uint &dim [[buffer(12)]], constant uint &block [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]], constant float &scale [[buffer(15)]],
    constant uint &window [[buffer(16)]], constant uint &num_blocks [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 aligned_q[256];
    threadgroup half *qt = (threadgroup half *)aligned_q;
    threadgroup float scores[8 * 32], stats[3 * 8];
    threadgroup int pages[32];
    threadgroup uint poison[8], flags[4];
    pr27_attention<256>(Q, K, V, O, tables, contexts, cu, positions,
        M, batch, heads, kv_heads, dim, block, max_blocks, scale, window, num_blocks,
        group, tid, sg, lane, threads, qt, scores, pages, stats, poison, flags);
}

kernel void research_prefill27_wide64_mma8k32_d512(
    device const half *Q [[buffer(0)]], device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]], device half *O [[buffer(3)]],
    device const int *tables [[buffer(4)]], device const int *contexts [[buffer(5)]],
    device const int *cu [[buffer(6)]], device const int *positions [[buffer(7)]],
    constant uint &M [[buffer(8)]], constant uint &batch [[buffer(9)]],
    constant uint &heads [[buffer(10)]], constant uint &kv_heads [[buffer(11)]],
    constant uint &dim [[buffer(12)]], constant uint &block [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]], constant float &scale [[buffer(15)]],
    constant uint &window [[buffer(16)]], constant uint &num_blocks [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float4 aligned_q[512];
    threadgroup half *qt = (threadgroup half *)aligned_q;
    threadgroup float scores[8 * 32], stats[3 * 8];
    threadgroup int pages[32];
    threadgroup uint poison[8], flags[4];
    pr27_attention<512>(Q, K, V, O, tables, contexts, cu, positions,
        M, batch, heads, kv_heads, dim, block, max_blocks, scale, window, num_blocks,
        group, tid, sg, lane, threads, qt, scores, pages, stats, poison, flags);
}
