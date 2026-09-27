// FP32 materialization preserves the incumbent gemm_rmsnorm precision boundary.
// Same 256-lane sum-of-squares tree; BF16 output occurs only AFTER normalization.
inline void pr26_raw_norm(device const float *X, device half *O,
    device const half *G, uint N, float eps, uint M, uint row, uint tid, uint tpg,
    threadgroup float *sums) {
    if (tpg != 256u || N != 3840u || M < 6u || M > 2048u || row >= M || eps != 1.0e-6f) return;
    float local = 0.0f;
    for (uint d = tid; d < N; d += 256u) {
        const float x = X[size_t(row) * N + d]; local += x * x;
    }
    sums[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (tid < step) sums[tid] += sums[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(sums[0] / float(N) + eps);
    for (uint d = tid; d < N; d += 256u)
        O[size_t(row) * N + d] = f16_sat(X[size_t(row) * N + d] * inv * float(G[d]));
}
