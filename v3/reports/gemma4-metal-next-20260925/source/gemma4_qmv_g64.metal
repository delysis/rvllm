#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Gemma 4 decode-oriented affine group-64 QMV candidates.
// Storage contract (deliberately explicit, MLX-like rather than rvLLM legacy):
//   x:       BF16 bits [K]
//   q4:      packed low/high nibbles, row-major [N, K/2]
//   q8:      uint8 row-major [N, K]
//   scale:   BF16 bits [N, K/64]
//   bias:    BF16 bits [N, K/64]
//   y:       BF16 bits [N]
// Dequantization is fused into the dot product. No materialized FP16/BF16 weights.
// One SIMD group computes 8 output rows; two SIMD groups / threadgroup => 16 rows/TG.
// Each lane owns exactly two elements of every 64-wide quantization group.

static inline float bf16_to_f32(ushort x) {
    return as_type<float>(uint(x) << 16);
}

static inline ushort f32_to_bf16_rne(float x) {
    uint u = as_type<uint>(x);
    uint lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return ushort(u >> 16);
}

template <uint ROWS>
static inline void qmv_q4_g64_rows(
    device const ushort *x,
    device const uchar *w,
    device const ushort *scales,
    device const ushort *biases,
    device ushort *y,
    uint N,
    uint K,
    uint row_base,
    ushort lane)
{
    const uint groups = K >> 6;  // K / 64
    float acc[ROWS];
    #pragma unroll
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (uint g = 0; g < groups; ++g) {
        const uint k0 = (g << 6) + (uint(lane) << 1);
        const float x0 = bf16_to_f32(x[k0]);
        const float x1 = bf16_to_f32(x[k0 + 1]);
        const float sx = simd_sum(x0 + x1);

        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint row = row_base + r;
            if (row < N) {
                const ulong packed_index = ulong(row) * ulong(K >> 1)
                                         + ulong(g) * 32ul + ulong(lane);
                const uchar p = w[packed_index];
                const float q0 = float(p & 0x0fu);
                const float q1 = float(p >> 4);
                const float qdot = simd_sum(q0 * x0 + q1 * x1);
                if (lane == 0) {
                    const ulong meta = ulong(row) * ulong(groups) + ulong(g);
                    acc[r] += bf16_to_f32(scales[meta]) * qdot
                            + bf16_to_f32(biases[meta]) * sx;
                }
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint row = row_base + r;
            if (row < N) y[row] = f32_to_bf16_rne(acc[r]);
        }
    }
}

template <uint ROWS>
static inline void qmv_q8_g64_rows(
    device const ushort *x,
    device const uchar *w,
    device const ushort *scales,
    device const ushort *biases,
    device ushort *y,
    uint N,
    uint K,
    uint row_base,
    ushort lane)
{
    const uint groups = K >> 6;
    float acc[ROWS];
    #pragma unroll
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (uint g = 0; g < groups; ++g) {
        const uint k0 = (g << 6) + (uint(lane) << 1);
        const float x0 = bf16_to_f32(x[k0]);
        const float x1 = bf16_to_f32(x[k0 + 1]);
        const float sx = simd_sum(x0 + x1);

        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint row = row_base + r;
            if (row < N) {
                const ulong wi = ulong(row) * ulong(K) + ulong(k0);
                const float q0 = float(w[wi]);
                const float q1 = float(w[wi + 1]);
                const float qdot = simd_sum(q0 * x0 + q1 * x1);
                if (lane == 0) {
                    const ulong meta = ulong(row) * ulong(groups) + ulong(g);
                    acc[r] += bf16_to_f32(scales[meta]) * qdot
                            + bf16_to_f32(biases[meta]) * sx;
                }
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint row = row_base + r;
            if (row < N) y[row] = f32_to_bf16_rne(acc[r]);
        }
    }
}

kernel void gemma4_qmv_q4_g64_r8_sg2(
    device const ushort *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device ushort *y [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (threads != 64) return;
    if ((K & 63u) != 0 || K == 0 || N == 0) return;
    const uint row_base = tg * 16u + uint(sg) * 8u;
    if (row_base >= N) return;
    qmv_q4_g64_rows<8>(x, w, scales, biases, y, N, K, row_base, lane);
}

kernel void gemma4_qmv_q8_g64_r8_sg2(
    device const ushort *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device ushort *y [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (threads != 64) return;
    if ((K & 63u) != 0 || K == 0 || N == 0) return;
    const uint row_base = tg * 16u + uint(sg) * 8u;
    if (row_base >= N) return;
    qmv_q8_g64_rows<8>(x, w, scales, biases, y, N, K, row_base, lane);
}
