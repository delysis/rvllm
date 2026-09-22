// metal-short-mma16x64: 128 threads, four 16x16 accumulators, BK=32.
// Native storage type is substituted by the existing typed source exporter.
// Padding is address-space only; no FP16 demotion of BF16 inputs occurs.
inline void research_mma16x64_tile(device const half *A, device const half *B,
    uint M, uint N, uint K, uint2 group, ushort tid, ushort sg,
    threadgroup half *at, threadgroup half *bt, threadgroup float *ct) {
    const uint mr = group.x * 16u;
    const uint nc = group.y * 64u;
    const uint sn = uint(sg) * 16u;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint kb = 0; kb < K; kb += 32u) {
        for (uint index = uint(tid); index < 512u; index += 128u) {
            uint row = index / 32u, col = index % 32u;
            at[row * 40u + col] = mr + row < M && kb + col < K
                ? A[size_t(mr + row) * K + kb + col] : half(0.0f);
        }
        for (uint index = uint(tid); index < 2048u; index += 128u) {
            uint row = index / 32u, col = index % 32u;
            bt[row * 40u + col] = nc + row < N && kb + col < K
                ? B[size_t(nc + row) * K + kb + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> a0, a1, b0, b1;
            simdgroup_load(a0, at + kk, 40);
            simdgroup_load(a1, at + 8u * 40u + kk, 40);
            simdgroup_load(b0, bt + sn * 40u + kk, 40, ulong2(0), true);
            simdgroup_load(b1, bt + (sn + 8u) * 40u + kk, 40, ulong2(0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        // Readers in every SIMD group finish before either staging tile is reused.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ct + sn, 64);
    simdgroup_store(c01, ct + sn + 8u, 64);
    simdgroup_store(c10, ct + 8u * 64u + sn, 64);
    simdgroup_store(c11, ct + 8u * 64u + sn + 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void research_gemm_mma16x64(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]], constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half at[16 * 40];
    threadgroup half bt[64 * 40];
    threadgroup float ct[16 * 64];
    research_mma16x64_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = group.x * 16u + index / 64u;
        uint col = group.y * 64u + index % 64u;
        if (row < M && col < N) {
            size_t output = size_t(row) * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            float value = alpha * ct[index] + prior;
            C[output] = f16_sat(value);
        }
    }
}

kernel void research_qkv_mma16x64(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]], constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half at[16 * 40];
    threadgroup half bt[64 * 40];
    threadgroup float ct[16 * 64];
    research_mma16x64_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = group.x * 16u + index / 64u;
        uint col = group.y * 64u + index % 64u;
        if (row < M && col < N) {
            size_t output = size_t(row) * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            float value = alpha * ct[index] + prior;
            C[output] = value;
        }
    }
}
