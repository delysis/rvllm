// metal-mma32-prefetch. Same 32x32x32 MMA / ascending-K order as mma32.
// The next operands are held in 16 scalar source values per thread, not a
// second threadgroup tile. Register allocation and overlap are unmeasured.
static inline void research_mma32_prefetch_tile(
    device const half *A, device const half *B, uint M, uint N, uint K,
    uint3 group, uint tid, ushort sg,
    threadgroup half *at, threadgroup half *bt, threadgroup float *ct) {
    const uint mr = group.x * 32, nc = group.y * 32;
    const uint sm = (sg / 2) * 16, sn = (sg % 2) * 16;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint i = 0; i < 8; ++i) {
        const uint index = tid + 128 * i, row = index / 32, k = index % 32;
        at[index] = mr + row < M && k < K ? A[ulong(mr + row) * K + k] : half(0);
        bt[index] = nc + row < N && k < K ? B[ulong(nc + row) * K + k] : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb = 0; kb < K; kb += 32) {
        half next_a[8], next_b[8];
        const bool more = kb + 32 < K;
        if (more) {
            for (uint i = 0; i < 8; ++i) {
                const uint index = tid + 128 * i, row = index / 32;
                const uint k = kb + 32 + index % 32;
                next_a[i] = mr + row < M && k < K ? A[ulong(mr + row) * K + k] : half(0);
                next_b[i] = nc + row < N && k < K ? B[ulong(nc + row) * K + k] : half(0);
            }
        }
        // Do not replace these ascending eight-wide reductions with split-K.
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_matrix<half, 8, 8> a0, a1, b0, b1;
            simdgroup_load(a0, at + sm * 32 + kk, 32);
            simdgroup_load(a1, at + (sm + 8) * 32 + kk, 32);
            simdgroup_load(b0, bt + sn * 32 + kk, 32, ulong2(0), true);
            simdgroup_load(b1, bt + (sn + 8) * 32 + kk, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        // Every reader completes before the tile is overwritten. 'more' is
        // uniform across the whole threadgroup, including the last iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (more) {
            for (uint i = 0; i < 8; ++i) {
                at[tid + 128 * i] = next_a[i];
                bt[tid + 128 * i] = next_b[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    simdgroup_store(c00, ct + sm * 32 + sn, 32);
    simdgroup_store(c01, ct + sm * 32 + sn + 8, 32);
    simdgroup_store(c10, ct + (sm + 8) * 32 + sn, 32);
    simdgroup_store(c11, ct + (sm + 8) * 32 + sn + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void research_gemm_mma32_prefetch(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], uint3 threads [[threads_per_threadgroup]]) {
    // Uniform guards precede every barrier and protect incidental bad launches.
    if (threads.x != 128 || threads.y != 1 || threads.z != 1 ||
        M < 6 || M > 1024 || group.z != 0 || alpha != 1.0f || beta != 0.0f) return;
    if (!((N == 30720 && K == 3840) ||
          (N == 3840 && (K == 4096 || K == 8192 || K == 15360)))) return;
    if (group.x >= (M + 31) / 32 || group.y >= (N + 31) / 32) return;
    threadgroup half at[1024], bt[1024];
    threadgroup float ct[1024];
    research_mma32_prefetch_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    for (uint i = tid; i < 1024; i += 128) {
        const uint row = group.x * 32 + i / 32, col = group.y * 32 + i % 32;
        if (row < M && col < N) {
            const ulong index = ulong(row) * N + col;
            const float prior = beta == 0.0f ? 0.0f : beta * float(C[index]);
            C[index] = f16_sat(alpha * ct[i] + prior);
        }
    }
}

kernel void research_qkv_mma32_prefetch(
    device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]], constant float &beta [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], uint3 threads [[threads_per_threadgroup]]) {
    // Uniform guards precede every barrier and protect incidental bad launches.
    if (threads.x != 128 || threads.y != 1 || threads.z != 1 ||
        M < 6 || M > 1024 || group.z != 0 || alpha != 1.0f || beta != 0.0f) return;
    if (K != 3840 || !(N == 8192 || N == 9216)) return;
    if (group.x >= (M + 31) / 32 || group.y >= (N + 31) / 32) return;
    threadgroup half at[1024], bt[1024];
    threadgroup float ct[1024];
    research_mma32_prefetch_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    for (uint i = tid; i < 1024; i += 128) {
        const uint row = group.x * 32 + i / 32, col = group.y * 32 + i % 32;
        if (row < M && col < N) {
            const ulong index = ulong(row) * N + col;
            const float prior = beta == 0.0f ? 0.0f : beta * C[index];
            C[index] = alpha * ct[i] + prior;
        }
    }
}
