// metal-load4-m16n64k64: 128 threads, 10240 source scratch bytes.
kernel void research_gemm_load4_m16n64k64(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], uint3 threads [[threads_per_threadgroup]]) {
    threadgroup vec<half, 4> scratch[1280];
    load4_tiled_run<16, 64, 64, 1, 4, 6, false>(A, B, C, M, N, K, alpha, beta, group, tid, sg, threads, scratch);
}

kernel void research_qkv_load4_m16n64k64(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], uint3 threads [[threads_per_threadgroup]]) {
    threadgroup vec<half, 4> scratch[1280];
    load4_tiled_run<16, 64, 64, 1, 4, 6, true>(A, B, C, M, N, K, alpha, beta, group, tid, sg, threads, scratch);
}
