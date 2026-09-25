// Default-off cooperative native-BF16 low-bit projection candidates.
//
// Four SIMDgroups share each activation/scale tile. W4 consumes two signed
// nibbles per lane; W8 consumes four adjacent int8 weights per lane. Scales
// retain the package FP16 group-32 ABI, accumulation remains FP32 across the
// full K dimension, and output is rounded once to BF16. The tg16 schedules use
// four accumulators/lane; tg32 uses eight to trade registers for fewer groups.

kernel void experimental_projection_w4abf16_bf16_tg16k64(
    device const bfloat *A [[buffer(0)]],
    device const uchar *W [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device bfloat *C [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    constant uint &C_stride [[buffer(7)]],
    constant uint &C_column [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    uint m = tg.y;
    uint n_base = tg.x * 16u;
    if (m >= M || n_base >= N) return;

    uint row_bytes = (K + 1u) >> 1u;
    uint groups = (K + 31u) >> 5u;
    threadgroup float a_tile[64];
    threadgroup float s_tile[32];
    float4 acc = 0.0f;
    uint local_n0 = uint(sg) * 4u;

    for (uint kb = 0u; kb < K; kb += 64u) {
        if (tid < 64u) {
            uint k = kb + uint(tid);
            a_tile[tid] = k < K ? float(A[m * K + k]) : 0.0f;
        }
        if (tid < 32u) {
            uint local_n = uint(tid) >> 1u;
            uint local_g = uint(tid) & 1u;
            uint n = n_base + local_n;
            uint g = (kb >> 5u) + local_g;
            s_tile[tid] =
                (n < N && g < groups) ? float(scales[n * groups + g]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint kl = uint(lane) * 2u;
        uint k0 = kb + kl;
        if (k0 < K) {
            float a0 = a_tile[kl];
            bool has1 = k0 + 1u < K;
            float a1 = has1 ? a_tile[kl + 1u] : 0.0f;
            uint local_g = uint(lane) >> 4u;
            for (uint c = 0u; c < 4u; ++c) {
                uint local_n = local_n0 + c;
                uint n = n_base + local_n;
                if (n < N) {
                    uchar packed = W[n * row_bytes + (k0 >> 1u)];
                    int q0 = int(packed & 0x0fu);
                    int q1 = int(packed >> 4u);
                    q0 = q0 >= 8 ? q0 - 16 : q0;
                    q1 = q1 >= 8 ? q1 - 16 : q1;
                    float s = s_tile[local_n * 2u + local_g];
                    acc[c] += a0 * (float(q0) * s);
                    if (has1) acc[c] += a1 * (float(q1) * s);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float4 total;
    for (uint c = 0u; c < 4u; ++c) total[c] = simd_sum(acc[c]);
    if (lane == 0) {
        for (uint c = 0u; c < 4u; ++c) {
            uint n = n_base + local_n0 + c;
            if (n < N) C[m * C_stride + C_column + n] = bfloat(total[c]);
        }
    }
}

kernel void experimental_projection_w4abf16_bf16_tg32k64(
    device const bfloat *A [[buffer(0)]],
    device const uchar *W [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device bfloat *C [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    constant uint &C_stride [[buffer(7)]],
    constant uint &C_column [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    uint m = tg.y;
    uint n_base = tg.x * 32u;
    if (m >= M || n_base >= N) return;

    uint row_bytes = (K + 1u) >> 1u;
    uint groups = (K + 31u) >> 5u;
    threadgroup float a_tile[64];
    threadgroup float s_tile[64];
    float4 acc_lo = 0.0f;
    float4 acc_hi = 0.0f;
    uint local_n0 = uint(sg) * 8u;

    for (uint kb = 0u; kb < K; kb += 64u) {
        if (tid < 64u) {
            uint k = kb + uint(tid);
            a_tile[tid] = k < K ? float(A[m * K + k]) : 0.0f;

            uint local_n = uint(tid) >> 1u;
            uint local_g = uint(tid) & 1u;
            uint n = n_base + local_n;
            uint g = (kb >> 5u) + local_g;
            s_tile[tid] =
                (n < N && g < groups) ? float(scales[n * groups + g]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint kl = uint(lane) * 2u;
        uint k0 = kb + kl;
        if (k0 < K) {
            float a0 = a_tile[kl];
            bool has1 = k0 + 1u < K;
            float a1 = has1 ? a_tile[kl + 1u] : 0.0f;
            uint local_g = uint(lane) >> 4u;
            for (uint c = 0u; c < 8u; ++c) {
                uint local_n = local_n0 + c;
                uint n = n_base + local_n;
                if (n < N) {
                    uchar packed = W[n * row_bytes + (k0 >> 1u)];
                    int q0 = int(packed & 0x0fu);
                    int q1 = int(packed >> 4u);
                    q0 = q0 >= 8 ? q0 - 16 : q0;
                    q1 = q1 >= 8 ? q1 - 16 : q1;
                    float s = s_tile[local_n * 2u + local_g];
                    float term = a0 * (float(q0) * s);
                    if (has1) term += a1 * (float(q1) * s);
                    if (c < 4u) acc_lo[c] += term;
                    else acc_hi[c - 4u] += term;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float4 total_lo;
    float4 total_hi;
    for (uint c = 0u; c < 4u; ++c) {
        total_lo[c] = simd_sum(acc_lo[c]);
        total_hi[c] = simd_sum(acc_hi[c]);
    }
    if (lane == 0) {
        for (uint c = 0u; c < 8u; ++c) {
            uint n = n_base + local_n0 + c;
            if (n < N) {
                C[m * C_stride + C_column + n] =
                    bfloat(c < 4u ? total_lo[c] : total_hi[c - 4u]);
            }
        }
    }
}

kernel void experimental_projection_w8abf16_bf16_tg16k128(
    device const bfloat *A [[buffer(0)]],
    device const char *W [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device bfloat *C [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    constant uint &C_stride [[buffer(7)]],
    constant uint &C_column [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    uint m = tg.y;
    uint n_base = tg.x * 16u;
    if (m >= M || n_base >= N) return;

    uint groups = (K + 31u) >> 5u;
    threadgroup float a_tile[128];
    threadgroup float s_tile[64];
    float4 acc = 0.0f;
    uint local_n0 = uint(sg) * 4u;

    for (uint kb = 0u; kb < K; kb += 128u) {
        uint k = kb + uint(tid);
        a_tile[tid] = k < K ? float(A[m * K + k]) : 0.0f;
        if (tid < 64u) {
            uint local_n = uint(tid) >> 2u;
            uint local_g = uint(tid) & 3u;
            uint n = n_base + local_n;
            uint g = (kb >> 5u) + local_g;
            s_tile[tid] =
                (n < N && g < groups) ? float(scales[n * groups + g]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint kl = uint(lane) * 4u;
        uint k0 = kb + kl;
        if (k0 < K) {
            uint count = min(4u, K - k0);
            float4 av = 0.0f;
            for (uint j = 0u; j < count; ++j) av[j] = a_tile[kl + j];
            uint local_g = uint(lane) >> 3u;
            for (uint c = 0u; c < 4u; ++c) {
                uint local_n = local_n0 + c;
                uint n = n_base + local_n;
                if (n < N) {
                    float s = s_tile[local_n * 4u + local_g];
                    float sum = 0.0f;
                    if (count == 4u && (K & 3u) == 0u) {
                        char4 q = *((device const char4 *)(W + n * K + k0));
                        sum = dot(av, float4(q)) * s;
                    } else {
                        for (uint j = 0u; j < count; ++j) {
                            sum += av[j] * (float(W[n * K + k0 + j]) * s);
                        }
                    }
                    acc[c] += sum;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float4 total;
    for (uint c = 0u; c < 4u; ++c) total[c] = simd_sum(acc[c]);
    if (lane == 0) {
        for (uint c = 0u; c < 4u; ++c) {
            uint n = n_base + local_n0 + c;
            if (n < N) C[m * C_stride + C_column + n] = bfloat(total[c]);
        }
    }
}

kernel void experimental_projection_w8abf16_bf16_tg32k128(
    device const bfloat *A [[buffer(0)]],
    device const char *W [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device bfloat *C [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    constant uint &C_stride [[buffer(7)]],
    constant uint &C_column [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    uint m = tg.y;
    uint n_base = tg.x * 32u;
    if (m >= M || n_base >= N) return;

    uint groups = (K + 31u) >> 5u;
    threadgroup float a_tile[128];
    threadgroup float s_tile[128];
    float4 acc_lo = 0.0f;
    float4 acc_hi = 0.0f;
    uint local_n0 = uint(sg) * 8u;

    for (uint kb = 0u; kb < K; kb += 128u) {
        uint k = kb + uint(tid);
        a_tile[tid] = k < K ? float(A[m * K + k]) : 0.0f;

        uint local_n = uint(tid) >> 2u;
        uint local_g = uint(tid) & 3u;
        uint n = n_base + local_n;
        uint g = (kb >> 5u) + local_g;
        s_tile[tid] =
            (n < N && g < groups) ? float(scales[n * groups + g]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint kl = uint(lane) * 4u;
        uint k0 = kb + kl;
        if (k0 < K) {
            uint count = min(4u, K - k0);
            float4 av = 0.0f;
            for (uint j = 0u; j < count; ++j) av[j] = a_tile[kl + j];
            uint scale_group = uint(lane) >> 3u;
            for (uint c = 0u; c < 8u; ++c) {
                uint ln = local_n0 + c;
                uint row = n_base + ln;
                if (row < N) {
                    float s = s_tile[ln * 4u + scale_group];
                    float sum = 0.0f;
                    if (count == 4u && (K & 3u) == 0u) {
                        char4 q = *((device const char4 *)(W + row * K + k0));
                        sum = dot(av, float4(q)) * s;
                    } else {
                        for (uint j = 0u; j < count; ++j) {
                            sum += av[j] * (float(W[row * K + k0 + j]) * s);
                        }
                    }
                    if (c < 4u) acc_lo[c] += sum;
                    else acc_hi[c - 4u] += sum;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float4 total_lo;
    float4 total_hi;
    for (uint c = 0u; c < 4u; ++c) {
        total_lo[c] = simd_sum(acc_lo[c]);
        total_hi[c] = simd_sum(acc_hi[c]);
    }
    if (lane == 0) {
        for (uint c = 0u; c < 8u; ++c) {
            uint n = n_base + local_n0 + c;
            if (n < N) {
                C[m * C_stride + C_column + n] =
                    bfloat(c < 4u ? total_lo[c] : total_hi[c - 4u]);
            }
        }
    }
}
