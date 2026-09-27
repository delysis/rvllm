// pr26: bounded native-storage prefill GEMM, C = A[M,K] * B[N,K]^T.
// Baseline and lookahead arms have IDENTICAL 32x64x32 geometry and K/8
// accumulation order. Only the placement of next-panel device loads differs.
// No undocumented fragment/lane mapping, TensorOps, split-K, or BF16->FP16 cast.
// The existing source generator rewrites half/f16_sat to bfloat/bf16_sat.
inline void pr26_fetch(
    device const half *A, device const half *B, uint M, uint K,
    uint mr, uint nc, uint kb, ushort tid,
    thread vec<half, 4> *an, thread vec<half, 4> *bn) {
    #pragma unroll
    for (uint j = 0u; j < 2u; ++j) {
        const uint v = uint(tid) + j * 128u;
        const uint row = mr + v / 8u;
        const uint col = kb + 4u * (v % 8u);
        an[j] = row < M ? *((device const vec<half, 4> *)(A + size_t(row) * K + col))
                        : vec<half, 4>(half(0.0f));
    }
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const uint v = uint(tid) + j * 128u;
        const uint row = nc + v / 8u;
        const uint col = kb + 4u * (v % 8u);
        // N is a multiple of 64 and the group bound is checked by the caller.
        bn[j] = *((device const vec<half, 4> *)(B + size_t(row) * K + col));
    }
}

inline void pr26_store(device float *C, size_t at, float value) { C[at] = value; }
inline void pr26_store(device half *C, size_t at, float value) { C[at] = f16_sat(value); }

template <bool LOOKAHEAD, bool FP32, bool RAW_NORM, typename OUT>
inline void pr26_project(
    device const half *A, device const half *B, device OUT *C,
    uint M, uint N, uint K, float alpha, float beta,
    uint3 group, ushort tid, ushort sg, uint3 threads,
    threadgroup float4 *storage) {
    // Uniform refusal, before any load, barrier, or output write.
    if (threads.x != 128u || threads.y != 1u || threads.z != 1u) return;
    if (M < 6u || M > 2048u || alpha != 1.0f || beta != 0.0f) return;
    if (RAW_NORM) {
        if (!FP32 || N != 3840u || (K != 4096u && K != 8192u && K != 15360u)) return;
    } else if (FP32) {
        if (K != 3840u || (N != 8192u && N != 9216u)) return;
    } else {
        if (!((N == 30720u && K == 3840u) ||
              (N == 3840u && (K == 4096u || K == 8192u || K == 15360u)))) return;
    }
    if (K % 32u != 0u || N % 64u != 0u) return;
    if (group.z != 0u || group.x >= (M + 31u) / 32u || group.y >= N / 64u) return;
    const uint mr = group.x * 32u, nc = group.y * 64u;
    const uint sm = (uint(sg) / 2u) * 16u, sn = (uint(sg) % 2u) * 32u;
    threadgroup vec<half, 4> *sv = (threadgroup vec<half, 4> *)storage;
    threadgroup half *at = (threadgroup half *)storage;
    threadgroup half *bt = at + 32u * 32u;
    simdgroup_float8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0u; i < 2u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    }
    // 24 source BF16 values/lane, NOT a claim about physical registers.
    // The false specialization should remove these arrays; inspect generated code.
    vec<half, 4> an[2], bn[4];
    if (LOOKAHEAD) pr26_fetch(A, B, M, K, mr, nc, 0u, tid, an, bn);
    for (uint kb = 0u; kb < K; kb += 32u) {
        if (LOOKAHEAD) {
            #pragma unroll
            for (uint j = 0u; j < 2u; ++j) sv[uint(tid) + j * 128u] = an[j];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) sv[256u + uint(tid) + j * 128u] = bn[j];
        } else {
            for (uint v = uint(tid); v < 256u; v += 128u) {
                const uint row = mr + v / 8u, col = kb + 4u * (v % 8u);
                sv[v] = row < M
                    ? *((device const vec<half, 4> *)(A + size_t(row) * K + col))
                    : vec<half, 4>(half(0.0f));
            }
            for (uint v = uint(tid); v < 512u; v += 128u) {
                const uint row = nc + v / 8u, col = kb + 4u * (v % 8u);
                sv[256u + v] = *((device const vec<half, 4> *)(B + size_t(row) * K + col));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Request the next panel while the current panel is consumed. Whether
        // this overlaps useful work or spills is an empirical compiler question.
        if (LOOKAHEAD && kb + 32u < K)
            pr26_fetch(A, B, M, K, mr, nc, kb + 32u, tid, an, bn);
        #pragma unroll
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> a[2], b[4];
            #pragma unroll
            for (uint i = 0u; i < 2u; ++i)
                simdgroup_load(a[i], at + (sm + i * 8u) * 32u + kk, 32u);
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j)
                simdgroup_load(b[j], bt + (sn + j * 8u) * 32u + kk, 32u, ulong2(0), true);
            #pragma unroll
            for (uint i = 0u; i < 2u; ++i) {
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        // All readers finish before the next panel OR the FP32 output aliases it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *ct = (threadgroup float *)storage;
    #pragma unroll
    for (uint i = 0u; i < 2u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j)
            simdgroup_store(acc[i][j], ct + (sm + i * 8u) * 64u + sn + j * 8u, 64u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = uint(tid); index < 32u * 64u; index += 128u) {
        const uint row = mr + index / 64u, col = nc + index % 64u;
        if (row < M) {
            // Keep the incumbent alpha=1, beta=0 (+0) epilogue. FP32 QKV is
            // never routed through the BF16 store overload.
            pr26_store(C, size_t(row) * N + col, alpha * ct[index] + 0.0f);
        }
    }
}
