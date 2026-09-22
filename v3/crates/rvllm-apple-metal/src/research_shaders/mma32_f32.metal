// metal-mma32-f32: same 32x32x32 arithmetic graph, FP32 operand path.
// Device/storage ABI and projection rounding points are unchanged.
// Numerical/throughput control, not an assertion about instruction lowering.
inline void wave2_mma32_f32_tile(device const half *A, device const half *B,
    uint M, uint N, uint K, uint2 group, ushort tid, ushort sg,
    threadgroup float *at, threadgroup float *bt, threadgroup float *ct) {
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    const uint sm = uint(sg / 2u) * 16u;
    const uint sn = uint(sg % 2u) * 16u;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint kb = 0; kb < K; kb += 32u) {
        for (uint index = uint(tid); index < 1024u; index += 128u) {
            uint row = index / 32u;
            uint k = kb + index % 32u;
            at[index] = mr + row < M && k < K ? float(A[size_t(mr + row) * K + k]) : 0.0f;
            bt[index] = nc + row < N && k < K ? float(B[size_t(nc + row) * K + k]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
            simdgroup_load(a0, at + sm * 32u + kk, 32);
            simdgroup_load(a1, at + (sm + 8u) * 32u + kk, 32);
            simdgroup_load(b0, bt + sn * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(b1, bt + (sn + 8u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        // All four groups must finish reading before any overwrite the tiles.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ct + sm * 32u + sn, 32);
    simdgroup_store(c01, ct + sm * 32u + sn + 8u, 32);
    simdgroup_store(c10, ct + (sm + 8u) * 32u + sn, 32);
    simdgroup_store(c11, ct + (sm + 8u) * 32u + sn + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


kernel void wave2_qkv_mma32_f32(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float at[32 * 32];
    threadgroup float bt[32 * 32];
    threadgroup float ct[32 * 32];
    wave2_mma32_f32_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u;
        uint col = nc + index % 32u;
        if (row < M && col < N) {
            size_t output = size_t(row) * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            C[output] = alpha * ct[index] + prior;
        }
    }
}


kernel void wave2_gemm_mma32_f32(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float at[32 * 32];
    threadgroup float bt[32 * 32];
    threadgroup float ct[32 * 32];
    wave2_mma32_f32_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u;
        uint col = nc + index % 32u;
        if (row < M && col < N) {
            size_t output = size_t(row) * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            C[output] = f16_sat(alpha * ct[index] + prior);
        }
    }
}
