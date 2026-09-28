// Default-off PR27 member; shared implementation is sealed by the source exporter.

kernel void research_prefill27_wide64_gemm(
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

kernel void research_prefill27_wide64_qkv(
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

kernel void research_prefill27_wide64_raw(
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

kernel void research_prefill27_wide64_norm(
    device const float *X [[buffer(0)]], device half *O [[buffer(1)]],
    device const half *G [[buffer(2)]], constant uint &N [[buffer(3)]],
    constant float &eps [[buffer(4)]], constant uint &M [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
    threadgroup float sums[256];
    pr26_raw_norm(X, O, G, N, eps, M, row, tid, tpg, sums);
}
