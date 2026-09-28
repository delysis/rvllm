// PR27: wider output tiles, padded K32 panels, next-panel lookahead, and a
// register-fragment epilogue. Compared to PR26 there is NO shared FP32 output
// tile or output store/load round trip. Native BF16 inputs are never cast to
// FP16; the existing source generator rewrites half -> bfloat as for PR26.
// Each SG owns a 32x32 output tile (32 FP32 accumulator values per lane).
template <uint BM, uint BN>
inline void pr27_fetch(device const half *A, device const half *B,
    uint M, uint K, uint mr, uint nc, uint kb, ushort tid,
    thread vec<half, 4> *next) {
    #pragma unroll
    for (uint j = 0u; j < (BM + BN) / 16u; ++j) {
        const uint v = uint(tid) + j * 128u;
        const uint r = v / 8u, d = kb + 4u * (v % 8u);
        if (r < BM) {
            next[j] = mr + r < M
                ? *((device const vec<half, 4> *)(A + size_t(mr + r) * K + d))
                : vec<half, 4>(half(0.0f));
        } else {
            next[j] = *((device const vec<half, 4> *)(B + size_t(nc + r - BM) * K + d));
        }
    }
}

template <uint BM, uint BN, uint GM, uint GN, bool FP32, bool RAW, typename OUT>
inline void pr27_project(device const half *A, device const half *B, device OUT *C,
    uint M, uint N, uint K, float alpha, float beta,
    uint3 group, ushort tid, ushort sg, ushort lane, uint3 threads,
    threadgroup half *panels) {
    static_assert(BM / GM == 32u && BN / GN == 32u && GM * GN == 4u,
                  "four independent 32x32 SG tiles required");
    if (any(threads != uint3(128, 1, 1))) return;
    if (M < 6u || M > 2048u || alpha != 1.0f || beta != 0.0f) return;
    if (RAW) {
        if (!FP32 || N != 3840u || (K != 4096u && K != 8192u && K != 15360u)) return;
    } else if (FP32) {
        if (K != 3840u || (N != 8192u && N != 9216u)) return;
    } else {
        if (!((N == 30720u && K == 3840u) ||
              (N == 3840u && (K == 4096u || K == 8192u || K == 15360u)))) return;
    }
    if (K % 32u || N % BN || group.z || group.x >= (M + BM - 1u) / BM || group.y >= N / BN) return;
    const uint mr = group.x * BM, nc = group.y * BN;
    const uint sm = (uint(sg) / GN) * 32u, sn = (uint(sg) % GN) * 32u;
    // +8 elements/row is a bank-layout hypothesis, not measured conflict data.
    threadgroup half *at = panels, *bt = panels + BM * 40u;
    simdgroup_float8x8 acc[4][4];
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    }
    // 32/40 native-storage values per lane. Actual registers/spills are UNMEASURED.
    vec<half, 4> next[(BM + BN) / 16u];
    pr27_fetch<BM, BN>(A, B, M, K, mr, nc, 0u, tid, next);
    for (uint kb = 0u; kb < K; kb += 32u) {
        #pragma unroll
        for (uint j = 0u; j < (BM + BN) / 16u; ++j) {
            const uint v = uint(tid) + j * 128u;
            *((threadgroup vec<half, 4> *)(panels + (v / 8u) * 40u + (v % 8u) * 4u)) = next[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + 32u < K) pr27_fetch<BM, BN>(A, B, M, K, mr, nc, kb + 32u, tid, next);
        #pragma unroll
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> a[4], b[4];
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                simdgroup_load(a[i], at + (sm + i * 8u) * 40u + kk, 40u);
                simdgroup_load(b[i], bt + (sn + i * 8u) * 40u + kk, 40u, ulong2(0), true);
            }
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const ushort2 xy = pr27_coord(lane);
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        const uint row = mr + sm + i * 8u + uint(xy.y);
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            const uint col = nc + sn + j * 8u + uint(xy.x);
            if (row < M) {
                #pragma unroll
                for (uint e = 0u; e < 2u; ++e)
                    pr27_store(C, size_t(row) * N + col + e,
                               alpha * acc[i][j].thread_elements()[e] + 0.0f);
            }
        }
    }
}
