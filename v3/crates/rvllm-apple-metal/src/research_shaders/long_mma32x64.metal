// metal-long-mma32x64: 32x64x32, 128 threads, scalar cooperative loads.
// BF16/FP16 device operands, FP32 accumulators. No cross-threadgroup scheduling.
inline void wave2_mma32x64_tile(device const half *A, device const half *B,
    uint M, uint N, uint K, uint2 group, ushort tid, ushort sg,
    threadgroup half *at, threadgroup half *bt, threadgroup float *ct) {
    const uint mr = group.x * 32u;
    const uint nc = group.y * 64u;
    const uint sm = uint(sg / 2u) * 16u;
    const uint sn = uint(sg % 2u) * 32u;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c02(0.0f), c03(0.0f);
    simdgroup_float8x8 c10(0.0f), c11(0.0f), c12(0.0f), c13(0.0f);
    for (uint kb = 0; kb < K; kb += 32u) {
        for (uint index = uint(tid); index < 1024u; index += 128u) {
            uint row = index / 32u, col = index % 32u;
            at[index] = mr + row < M && kb + col < K
                ? A[size_t(mr + row) * K + kb + col] : half(0.0f);
        }
        for (uint index = uint(tid); index < 2048u; index += 128u) {
            uint row = index / 32u, col = index % 32u;
            bt[index] = nc + row < N && kb + col < K
                ? B[size_t(nc + row) * K + kb + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, at + sm * 32u + kk, 32);
            simdgroup_load(a1, at + (sm + 8u) * 32u + kk, 32);
            simdgroup_load(b0, bt + (sn + 0u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(b1, bt + (sn + 8u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(b2, bt + (sn + 16u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(b3, bt + (sn + 24u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c02, a0, b2, c02);
            simdgroup_multiply_accumulate(c03, a0, b3, c03);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
            simdgroup_multiply_accumulate(c12, a1, b2, c12);
            simdgroup_multiply_accumulate(c13, a1, b3, c13);
        }
        // Both A/B tiles are reused only after every reader finishes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ct + (sm + 0u) * 64u + sn + 0u, 64);
    simdgroup_store(c01, ct + (sm + 0u) * 64u + sn + 8u, 64);
    simdgroup_store(c02, ct + (sm + 0u) * 64u + sn + 16u, 64);
    simdgroup_store(c03, ct + (sm + 0u) * 64u + sn + 24u, 64);
    simdgroup_store(c10, ct + (sm + 8u) * 64u + sn + 0u, 64);
    simdgroup_store(c11, ct + (sm + 8u) * 64u + sn + 8u, 64);
    simdgroup_store(c12, ct + (sm + 8u) * 64u + sn + 16u, 64);
    simdgroup_store(c13, ct + (sm + 8u) * 64u + sn + 24u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void wave2_gemm_mma32x64(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]], constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half at[32 * 32];
    threadgroup half bt[64 * 32];
    threadgroup float ct[32 * 64];
    wave2_mma32x64_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    for (uint index = uint(tid); index < 2048u; index += 128u) {
        uint row = group.x * 32u + index / 64u;
        uint col = group.y * 64u + index % 64u;
        if (row < M && col < N) {
            size_t output = size_t(row) * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            float value = alpha * ct[index] + prior;
            C[output] = f16_sat(value);
        }
    }
}

kernel void wave2_qkv_mma32x64(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]], constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half at[32 * 32];
    threadgroup half bt[64 * 32];
    threadgroup float ct[32 * 64];
    wave2_mma32x64_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    for (uint index = uint(tid); index < 2048u; index += 128u) {
        uint row = group.x * 32u + index / 64u;
        uint col = group.y * 64u + index % 64u;
        if (row < M && col < N) {
            size_t output = size_t(row) * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            float value = alpha * ct[index] + prior;
            C[output] = value;
        }
    }
}
