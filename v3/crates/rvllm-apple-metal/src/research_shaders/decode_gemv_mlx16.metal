// MLX-shaped Gemma 4 decode GEMV research kernel.
//
// Geometry mirrors the fast dense MLX GEMV family observed for Gemma 4:
//   - 4 SIMD groups / threadgroup (128 threads)
//   - 4 output rows / SIMD group
//   - 4 contiguous K values / lane
//   - 16 output rows / threadgroup
//
// Gemma 4 12B decode projection shapes are all naturally aligned:
// K in {3840, 4096, 8192, 15360}, N in {3840, 8192, 9216, 30720}.
// The shader retains bounded tail guards, but the research selector admits
// only aligned M=1 native-BF16 projection cells.
//
// No threadgroup scratch or barriers are required. Each SIMD group performs
// four independent FP32 reductions and rounds once at the output boundary.
kernel void research_decode_gemv_mlx16(
    device const half *A          [[buffer(0)]],
    device const half *B          [[buffer(1)]],
    device half       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort simdgroup              [[simdgroup_index_in_threadgroup]]
) {
    const uint row = gid.x;
    const uint col_base = gid.y * 16u + uint(simdgroup) * 4u;
    if (row >= M || simdgroup >= 4u) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const uint a_base = row * K;

    // 32 lanes * 4 contiguous K values = 128 K elements per iteration.
    // The four activation values are shared across four independent rows.
    for (uint k0 = uint(lane) * 4u; k0 < K; k0 += 128u) {
        const float x0 = float(A[a_base + k0 + 0u]);
        const float x1 = float(A[a_base + k0 + 1u]);
        const float x2 = float(A[a_base + k0 + 2u]);
        const float x3 = float(A[a_base + k0 + 3u]);

        const uint c0 = col_base + 0u;
        if (c0 < N) {
            const uint w = c0 * K + k0;
            acc0 += x0 * float(B[w + 0u]);
            acc0 += x1 * float(B[w + 1u]);
            acc0 += x2 * float(B[w + 2u]);
            acc0 += x3 * float(B[w + 3u]);
        }
        const uint c1 = col_base + 1u;
        if (c1 < N) {
            const uint w = c1 * K + k0;
            acc1 += x0 * float(B[w + 0u]);
            acc1 += x1 * float(B[w + 1u]);
            acc1 += x2 * float(B[w + 2u]);
            acc1 += x3 * float(B[w + 3u]);
        }
        const uint c2 = col_base + 2u;
        if (c2 < N) {
            const uint w = c2 * K + k0;
            acc2 += x0 * float(B[w + 0u]);
            acc2 += x1 * float(B[w + 1u]);
            acc2 += x2 * float(B[w + 2u]);
            acc2 += x3 * float(B[w + 3u]);
        }
        const uint c3 = col_base + 3u;
        if (c3 < N) {
            const uint w = c3 * K + k0;
            acc3 += x0 * float(B[w + 0u]);
            acc3 += x1 * float(B[w + 1u]);
            acc3 += x2 * float(B[w + 2u]);
            acc3 += x3 * float(B[w + 3u]);
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (lane == 0u) {
        const uint c0 = col_base + 0u;
        if (c0 < N) {
            const uint out = row * N + c0;
            const float prior = beta == 0.0f ? 0.0f : float(C[out]) * beta;
            C[out] = f16_sat(acc0 * alpha + prior);
        }
        const uint c1 = col_base + 1u;
        if (c1 < N) {
            const uint out = row * N + c1;
            const float prior = beta == 0.0f ? 0.0f : float(C[out]) * beta;
            C[out] = f16_sat(acc1 * alpha + prior);
        }
        const uint c2 = col_base + 2u;
        if (c2 < N) {
            const uint out = row * N + c2;
            const float prior = beta == 0.0f ? 0.0f : float(C[out]) * beta;
            C[out] = f16_sat(acc2 * alpha + prior);
        }
        const uint c3 = col_base + 3u;
        if (c3 < N) {
            const uint out = row * N + c3;
            const float prior = beta == 0.0f ? 0.0f : float(C[out]) * beta;
            C[out] = f16_sat(acc3 * alpha + prior);
        }
    }
}
