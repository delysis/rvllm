// MLX-shaped group-32 low-bit decode QMV experiments.
//
// The production package remains unchanged: symmetric W4/W8, one FP16 scale
// per 32 K values, native BF16 activations/output, FP32 accumulation.
//
// Geometry follows MLX qmv_fast rather than rvLLM's current one-SIMD schedules:
//   * 2 SIMD groups / threadgroup
//   * 4 output rows / SIMD group (8 rows / threadgroup)
//   * 2 packed uint32-equivalents / lane
//     - W4: 16 K values / lane, 512 K values / SIMD iteration
//     - W8:  8 K values / lane, 256 K values / SIMD iteration
//   * activation mini-vector loaded once and reused across four output rows
//   * one scale load per row/lane/iteration, reused across the mini-vector
//
// This source is benchmark-only/default-off until device correctness and
// paired timing gates qualify it.

kernel void research_projection_w4abf16_bf16_qmv_mlx(
    device const bfloat *A        [[buffer(0)]],
    device const uchar  *W        [[buffer(1)]],
    device const half   *scales   [[buffer(2)]],
    device bfloat       *C        [[buffer(3)]],
    constant uint       &M        [[buffer(4)]],
    constant uint       &N        [[buffer(5)]],
    constant uint       &K        [[buffer(6)]],
    constant uint       &C_stride [[buffer(7)]],
    constant uint       &C_column [[buffer(8)]],
    uint2 output                   [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort simdgroup              [[simdgroup_index_in_threadgroup]]
) {
    const uint m = output.y;
    const uint n0 = output.x * 8u + uint(simdgroup) * 4u;
    if (m >= M || n0 >= N || simdgroup >= 2u) return;

    const uint row_bytes = (K + 1u) >> 1u;
    const uint groups = (K + 31u) >> 5u;
    float4 acc = float4(0.0f);

    // MLX 4-bit qmv_fast: two 32-bit packs = 16 values per lane.
    for (uint k0 = uint(lane) * 16u; k0 < K; k0 += 512u) {
        const uint count = min(16u, K - k0);
        float xv[16];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 16u; ++j) {
            xv[j] = j < count ? float(A[m * K + k0 + j]) : 0.0f;
        }

#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < N) {
                const device uchar *wp = W + n * row_bytes + (k0 >> 1u);
                const float scale = float(scales[n * groups + (k0 >> 5u)]);
                float dot = 0.0f;
#pragma clang loop unroll(full)
                for (uint j = 0u; j < 16u; ++j) {
                    if (j < count) {
                        const uchar packed = wp[j >> 1u];
                        int q = int((j & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
                        q = q >= 8 ? q - 16 : q;
                        dot += xv[j] * float(q);
                    }
                }
                acc[row] += dot * scale;
            }
        }
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);

    if (lane == 0u) {
#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < N) C[m * C_stride + C_column + n] = bfloat(acc[row]);
        }
    }
}

kernel void research_projection_w8abf16_bf16_qmv_mlx(
    device const bfloat *A        [[buffer(0)]],
    device const char   *W        [[buffer(1)]],
    device const half   *scales   [[buffer(2)]],
    device bfloat       *C        [[buffer(3)]],
    constant uint       &M        [[buffer(4)]],
    constant uint       &N        [[buffer(5)]],
    constant uint       &K        [[buffer(6)]],
    constant uint       &C_stride [[buffer(7)]],
    constant uint       &C_column [[buffer(8)]],
    uint2 output                   [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort simdgroup              [[simdgroup_index_in_threadgroup]]
) {
    const uint m = output.y;
    const uint n0 = output.x * 8u + uint(simdgroup) * 4u;
    if (m >= M || n0 >= N || simdgroup >= 2u) return;

    const uint groups = (K + 31u) >> 5u;
    float4 acc = float4(0.0f);

    // MLX 8-bit qmv_fast: two 32-bit packs = 8 values per lane.
    for (uint k0 = uint(lane) * 8u; k0 < K; k0 += 256u) {
        const uint count = min(8u, K - k0);
        float xv[8];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) {
            xv[j] = j < count ? float(A[m * K + k0 + j]) : 0.0f;
        }

#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < N) {
                const device char *wp = W + n * K + k0;
                const float scale = float(scales[n * groups + (k0 >> 5u)]);
                float dot = 0.0f;
#pragma clang loop unroll(full)
                for (uint j = 0u; j < 8u; ++j) {
                    if (j < count) dot += xv[j] * float(wp[j]);
                }
                acc[row] += dot * scale;
            }
        }
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);

    if (lane == 0u) {
#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < N) C[m * C_stride + C_column + n] = bfloat(acc[row]);
        }
    }
}
