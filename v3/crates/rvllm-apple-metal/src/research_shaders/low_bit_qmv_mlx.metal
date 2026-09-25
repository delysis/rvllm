// Gemma 4 group-32 low-bit decode QMV research kernels.
//
// All candidates preserve rvLLM's current package ABI exactly:
//   W4: signed two's-complement nibbles, row-major, FP16 scale / 32 K values.
//   W8: signed bytes, row-major, FP16 scale / 32 K values.
//   A/C: native BF16. Accumulation: FP32.
//
// Two launch geometries are intentionally kept separate for the device referee:
//   *_qmv_mlx   : 2 SIMD groups/TG, 4 rows/group = 8 output rows/TG.
//   *_qmv_core8 : 8 SIMD groups/TG, 4 rows/group = 32 output rows/TG.
//
// The first mirrors MLX qmv_fast's threadgroup width. The second mirrors the
// high-throughput CoreAIKit Gemma4 raw-Metal matvec family. Both use packed
// word/vector loads instead of scalar byte/nibble issue, load each activation
// mini-vector once, and reuse it across four output rows.

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

    const uint row_bytes = K >> 1u;
    const uint groups = K >> 5u;
    float4 acc = float4(0.0f);

    // MLX W4 qmv_fast: 2 uint32 packs = 16 codes per lane.
    for (uint k0 = uint(lane) * 16u; k0 < K; k0 += 512u) {
        float xv[16];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 16u; ++j) {
            xv[j] = float(A[m * K + k0 + j]);
        }

#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < N) {
                device const uint2 *wp =
                    (device const uint2 *)(W + n * row_bytes + (k0 >> 1u));
                const uint2 pk = wp[0];
                const float scale = float(scales[n * groups + (k0 >> 5u)]);
                float dot = 0.0f;
#pragma clang loop unroll(full)
                for (uint j = 0u; j < 16u; ++j) {
                    const uint p = j < 8u ? pk.x : pk.y;
                    int q = int((p >> ((j & 7u) * 4u)) & 0x0fu);
                    q = q >= 8 ? q - 16 : q;
                    dot += xv[j] * float(q);
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

    const uint groups = K >> 5u;
    float4 acc = float4(0.0f);

    for (uint k0 = uint(lane) * 8u; k0 < K; k0 += 256u) {
        float xv[8];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) xv[j] = float(A[m * K + k0 + j]);

#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < N) {
                device const char4 *wp =
                    (device const char4 *)(W + n * K + k0);
                const char4 wa = wp[0];
                const char4 wb = wp[1];
                const float scale = float(scales[n * groups + (k0 >> 5u)]);
                float dot = 0.0f;
                dot += xv[0] * float(int(wa.x));
                dot += xv[1] * float(int(wa.y));
                dot += xv[2] * float(int(wa.z));
                dot += xv[3] * float(int(wa.w));
                dot += xv[4] * float(int(wb.x));
                dot += xv[5] * float(int(wb.y));
                dot += xv[6] * float(int(wb.z));
                dot += xv[7] * float(int(wb.w));
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

// CoreAIKit-like 8-SIMD-group contender. W4 uses one uint32 (=8 codes)
// per lane, matching its proven Gemma4 matvec inner geometry.
kernel void research_projection_w4abf16_bf16_qmv_core8(
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
    const uint n0 = output.x * 32u + uint(simdgroup) * 4u;
    if (m >= M || n0 >= N || simdgroup >= 8u) return;

    const uint row_bytes = K >> 1u;
    const uint groups = K >> 5u;
    float4 acc = float4(0.0f);

    for (uint k0 = uint(lane) * 8u; k0 < K; k0 += 256u) {
        float xv[8];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) xv[j] = float(A[m * K + k0 + j]);

#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < N) {
                device const uint *wp =
                    (device const uint *)(W + n * row_bytes + (k0 >> 1u));
                const uint pk = wp[0];
                const float scale = float(scales[n * groups + (k0 >> 5u)]);
                float dot = 0.0f;
#pragma clang loop unroll(full)
                for (uint j = 0u; j < 8u; ++j) {
                    int q = int((pk >> (j * 4u)) & 0x0fu);
                    q = q >= 8 ? q - 16 : q;
                    dot += xv[j] * float(q);
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

kernel void research_projection_w8abf16_bf16_qmv_core8(
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
    const uint n0 = output.x * 32u + uint(simdgroup) * 4u;
    if (m >= M || n0 >= N || simdgroup >= 8u) return;

    const uint groups = K >> 5u;
    float4 acc = float4(0.0f);

    for (uint k0 = uint(lane) * 8u; k0 < K; k0 += 256u) {
        float xv[8];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) xv[j] = float(A[m * K + k0 + j]);

#pragma clang loop unroll(full)
        for (uint row = 0u; row < 4u; ++row) {
            const uint n = n0 + row;
            if (n < N) {
                device const char4 *wp =
                    (device const char4 *)(W + n * K + k0);
                const char4 wa = wp[0];
                const char4 wb = wp[1];
                const float scale = float(scales[n * groups + (k0 >> 5u)]);
                float dot = 0.0f;
                dot += xv[0] * float(int(wa.x));
                dot += xv[1] * float(int(wa.y));
                dot += xv[2] * float(int(wa.z));
                dot += xv[3] * float(int(wa.w));
                dot += xv[4] * float(int(wb.x));
                dot += xv[5] * float(int(wb.y));
                dot += xv[6] * float(int(wb.z));
                dot += xv[7] * float(int(wb.w));
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
