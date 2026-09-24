// load4-tiled: native storage operands, FP32 accumulators, ascending K/8 MMA.
// Included before exactly one pair of explicit entry points. No split-K,
// operand conversion, relaxed math, or extra projection rounding is introduced.
// Source scratch bounds include max(operand staging, FP32 output staging).
template <uint BM, uint BN, uint BK, uint GM, uint GN>
inline void load4_tiled_accumulate(
    device const half *A, device const half *B,
    uint M, uint N, uint K, uint2 group, ushort tid, ushort sg,
    threadgroup vec<half, 4> *scratch) {
    static_assert(BM % (8u * GM) == 0u && BN % (8u * GN) == 0u,
                  "whole SIMD fragments required");
    static_assert(BK % 8u == 0u, "whole K fragments required");
    constexpr uint TM = BM / GM;
    constexpr uint TN = BN / GN;
    constexpr uint RM = TM / 8u;
    constexpr uint RN = TN / 8u;
    constexpr uint THREADS = 32u * GM * GN;
    const uint mr = group.x * BM;
    const uint nc = group.y * BN;
    const uint sm = (uint(sg) / GN) * TM;
    const uint sn = (uint(sg) % GN) * TN;
    threadgroup half *at = (threadgroup half *)scratch;
    threadgroup half *bt = at + BM * BK;
    simdgroup_float8x8 acc[RM][RN];
    #pragma unroll
    for (uint i = 0u; i < RM; ++i) {
        #pragma unroll
        for (uint j = 0u; j < RN; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    for (uint kb = 0u; kb < K; kb += BK) {
        for (uint v = uint(tid); v < BM * BK / 4u; v += THREADS) {
            const uint row = mr + v / (BK / 4u);
            const uint col = kb + 4u * (v % (BK / 4u));
            scratch[v] = row < M
                ? *((device const vec<half, 4> *)(A + size_t(row) * K + col))
                : vec<half, 4>(half(0.0f));
        }
        for (uint v = uint(tid); v < BN * BK / 4u; v += THREADS) {
            const uint row = nc + v / (BK / 4u);
            const uint col = kb + 4u * (v % (BK / 4u));
            scratch[BM * BK / 4u + v] = row < N
                ? *((device const vec<half, 4> *)(B + size_t(row) * K + col))
                : vec<half, 4>(half(0.0f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_matrix<half, 8, 8> a[RM], b[RN];
            #pragma unroll
            for (uint i = 0u; i < RM; ++i) {
                simdgroup_load(a[i], at + (sm + i * 8u) * BK + kk, BK);
            }
            #pragma unroll
            for (uint j = 0u; j < RN; ++j) {
                simdgroup_load(b[j], bt + (sn + j * 8u) * BK + kk,
                               BK, ulong2(0), true);
            }
            #pragma unroll
            for (uint i = 0u; i < RM; ++i) {
                #pragma unroll
                for (uint j = 0u; j < RN; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        // Includes the final iteration: all operand readers finish before
        // another K block or the output phase can overwrite these bytes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Operand lifetime has ended. Reuse the same aligned allocation, rather
    // than reserving simultaneous operand and output threadgroup arrays.
    threadgroup float *ct = (threadgroup float *)scratch;
    #pragma unroll
    for (uint i = 0u; i < RM; ++i) {
        #pragma unroll
        for (uint j = 0u; j < RN; ++j) {
            simdgroup_store(acc[i][j], ct + (sm + i * 8u) * BN + sn + j * 8u, BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint BM, uint BN, uint BK, uint GM, uint GN, uint MIN_M, bool FP32, typename OUT>
inline void load4_tiled_run(
    device const half *A, device const half *B, device OUT *C,
    uint M, uint N, uint K, float alpha, float beta,
    uint3 group, ushort tid, ushort sg, uint3 threads,
    threadgroup vec<half, 4> *scratch) {
    // These guards are uniform and precede every barrier. The runtime also
    // checks model/phase, native BF16, alignment, buffer spans and PSO limits.
    if (threads.x != 32u * GM * GN || threads.y != 1u || threads.z != 1u) return;
    if (M < MIN_M || M > 1024u || alpha != 1.0f || beta != 0.0f) return;
    if (FP32) {
        if (K != 3840u || (N != 8192u && N != 9216u)) return;
    } else {
        if (!((N == 30720u && K == 3840u) ||
              (N == 3840u && (K == 4096u || K == 8192u || K == 15360u)))) return;
    }
    if (K % BK != 0u || N % BN != 0u) return;
    if (group.z != 0u || group.x >= (M + BM - 1u) / BM || group.y >= N / BN) return;
    load4_tiled_accumulate<BM, BN, BK, GM, GN>(A, B, M, N, K, group.xy, tid, sg, scratch);
    threadgroup float *ct = (threadgroup float *)scratch;
    for (uint index = uint(tid); index < BM * BN; index += 32u * GM * GN) {
        const uint row = group.x * BM + index / BN;
        const uint col = group.y * BN + index % BN;
        if (row < M && col < N) {
            const size_t output = size_t(row) * N + col;
            const float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            const float value = alpha * ct[index] + prior;
            if (FP32) C[output] = value;
            else C[output] = f16_sat(value);
        }
    }
}
