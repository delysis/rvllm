
#include <metal_stdlib>
using namespace metal;

static inline bfloat bf16_sat(float x) {
    return bfloat(x);
}

static inline float bf16_acc(float x) {
    return float(bfloat(x));
}

static inline float gelu_tanh(float x) {
    if (x >= 5.0f) {
        return x;
    }
    if (x <= -5.0f) {
        return 0.0f;
    }
    float c = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

// ============================================================================
// RMSNorm (per-token, f16 in/out)
// ============================================================================
// Each threadgroup processes one token. Reduction across hidden dim
// uses threadgroup memory.

kernel void rmsnorm_f16(
    device const bfloat *input      [[buffer(0)]],
    device bfloat       *output     [[buffer(1)]],
    device const bfloat *gamma      [[buffer(2)]],
    constant uint     &hidden     [[buffer(3)]],
    constant float    &eps        [[buffer(4)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    // Each threadgroup = one token
    uint token = gid;
    uint base = token * hidden;

    // Phase 1: compute sum of squares
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = float(input[base + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(hidden) + eps);

    // Phase 2: normalize and apply gamma
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = float(input[base + i]);
        output[base + i] = bf16_sat(v * rms * float(gamma[i]));
    }
}

kernel void rmsnorm_headwise_f16(
    device const bfloat *input      [[buffer(0)]],
    device bfloat       *output     [[buffer(1)]],
    device const bfloat *gamma      [[buffer(2)]],
    constant uint     &head_dim   [[buffer(3)]],
    constant float    &eps        [[buffer(4)]],
    constant uint     &num_heads  [[buffer(5)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid / num_heads;
    uint head = gid % num_heads;
    uint hidden = num_heads * head_dim;
    uint base = token * hidden + head * head_dim;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = float(input[base + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = float(input[base + i]);
        output[base + i] = bf16_sat(v * rms * float(gamma[i]));
    }
}

kernel void rmsnorm_headwise_unit_f16(
    device const bfloat *input      [[buffer(0)]],
    device bfloat       *output     [[buffer(1)]],
    constant uint     &head_dim   [[buffer(2)]],
    constant float    &eps        [[buffer(3)]],
    constant uint     &num_heads  [[buffer(4)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid / num_heads;
    uint head = gid % num_heads;
    uint base = token * num_heads * head_dim + head * head_dim;
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = float(input[base + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += tg_size) {
        output[base + i] = bf16_sat(float(input[base + i]) * rms);
    }
}

// ============================================================================
// GEMM (f16, tiled)
// ============================================================================
// Simple but correct f16 GEMM: C[M,N] = A[M,K] * B[K,N]
// Weights stored in column-major (transposed): B is [N,K] in memory,
// accessed as B[n,k].
// Uses simdgroup matrix ops on Apple9+ for performance.

// ============================================================================
// Group-32 low-bit projection (W4A16 / W8A16)
// ============================================================================
// One simdgroup computes one output element. Weights are row-major [N,K]
// because each output channel is one independently-scaled row. Quantization
// groups never cross a row boundary and every group of at most 32 K-elements
// has one native FP16 scale. W4 stores the earlier K-element in the low
// nibble; both formats reserve their asymmetric minimum on the host.

kernel void projection_w4a16_f16(
    device const half  *A         [[buffer(0)]],  // [M,K]
    device const uchar *W         [[buffer(1)]],  // [N,ceil(K/2)]
    device const half  *scales    [[buffer(2)]],  // [N,ceil(K/32)]
    device half        *C         [[buffer(3)]],  // [M,N]
    constant uint      &M         [[buffer(4)]],
    constant uint      &N         [[buffer(5)]],
    constant uint      &K         [[buffer(6)]],
    constant uint      &C_stride  [[buffer(7)]],
    constant uint      &C_column  [[buffer(8)]],
    uint2 output                   [[threadgroup_position_in_grid]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
    uint n = output.x;
    uint m = output.y;
    if (m >= M || n >= N) return;

    uint packed_row_bytes = (K + 1u) >> 1u;
    uint groups_per_row = (K + 31u) >> 5u;
    float partial = 0.0f;
    for (uint k = uint(lane); k < K; k += 32u) {
        uchar packed = W[n * packed_row_bytes + (k >> 1u)];
        int q = int((k & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
        q = q >= 8 ? q - 16 : q;
        float scale = float(scales[n * groups_per_row + (k >> 5u)]);
        partial += float(A[m * K + k]) * (float(q) * scale);
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        C[m * C_stride + C_column + n] = half(clamp(total, -65504.0f, 65504.0f));
    }
}

kernel void projection_w8a16_f16(
    device const half  *A         [[buffer(0)]],  // [M,K]
    device const char  *W         [[buffer(1)]],  // [N,K]
    device const half  *scales    [[buffer(2)]],  // [N,ceil(K/32)]
    device half        *C         [[buffer(3)]],  // [M,N]
    constant uint      &M         [[buffer(4)]],
    constant uint      &N         [[buffer(5)]],
    constant uint      &K         [[buffer(6)]],
    constant uint      &C_stride  [[buffer(7)]],
    constant uint      &C_column  [[buffer(8)]],
    uint2 output                   [[threadgroup_position_in_grid]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
    uint n = output.x;
    uint m = output.y;
    if (m >= M || n >= N) return;

    uint groups_per_row = (K + 31u) >> 5u;
    float partial = 0.0f;
    for (uint k = uint(lane); k < K; k += 32u) {
        int q = int(W[n * K + k]);
        float scale = float(scales[n * groups_per_row + (k >> 5u)]);
        partial += float(A[m * K + k]) * (float(q) * scale);
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        C[m * C_stride + C_column + n] = half(clamp(total, -65504.0f, 65504.0f));
    }
}

// Experimental native-BF16 activation/output ABI.  The quantizer's group-32
// scales intentionally remain FP16: they are package metadata, not activation
// tensors, and retaining their established two-byte encoding permits an
// isolated A-dtype experiment without silently changing quantized weights.
// Accumulation and the SIMD reduction remain FP32; only the projection input
// and its single storage-rounding output boundary are BF16.
kernel void experimental_projection_w4abf16_bf16(
    device const bfloat *A         [[buffer(0)]],
    device const uchar  *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n = output.x;
    uint m = output.y;
    if (m >= M || n >= N) return;
    uint packed_row_bytes = (K + 1u) >> 1u;
    uint groups_per_row = (K + 31u) >> 5u;
    float partial = 0.0f;
    for (uint k = uint(lane); k < K; k += 32u) {
        uchar packed = W[n * packed_row_bytes + (k >> 1u)];
        int q = int((k & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
        q = q >= 8 ? q - 16 : q;
        float scale = float(scales[n * groups_per_row + (k >> 5u)]);
        partial += float(A[m * K + k]) * (float(q) * scale);
    }
    float total = simd_sum(partial);
    if (lane == 0) C[m * C_stride + C_column + n] = bfloat(total);
}

kernel void experimental_projection_w8abf16_bf16(
    device const bfloat *A         [[buffer(0)]],
    device const char   *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n = output.x;
    uint m = output.y;
    if (m >= M || n >= N) return;
    uint groups_per_row = (K + 31u) >> 5u;
    float partial = 0.0f;
    for (uint k = uint(lane); k < K; k += 32u) {
        int q = int(W[n * K + k]);
        float scale = float(scales[n * groups_per_row + (k >> 5u)]);
        partial += float(A[m * K + k]) * (float(q) * scale);
    }
    float total = simd_sum(partial);
    if (lane == 0) C[m * C_stride + C_column + n] = bfloat(total);
}

// Four-output native-BF16 research variants. One SIMD group owns four
// adjacent output channels and reuses every activation load across all four
// dot products. Output tails remain explicit; no padded weight row is read.
kernel void experimental_projection_w4abf16_bf16_n4(
    device const bfloat *A         [[buffer(0)]],
    device const uchar  *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n0 = output.x * 4u;
    uint m = output.y;
    if (m >= M || n0 >= N) return;
    uint packed_row_bytes = (K + 1u) >> 1u;
    uint groups_per_row = (K + 31u) >> 5u;
    float4 partial = float4(0.0f);
    for (uint k = uint(lane); k < K; k += 32u) {
        float activation = float(A[m * K + k]);
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                uchar packed = W[n * packed_row_bytes + (k >> 1u)];
                int q = int((k & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
                q = q >= 8 ? q - 16 : q;
                float scale = float(scales[n * groups_per_row + (k >> 5u)]);
                partial[column] += activation * (float(q) * scale);
            }
        }
    }
    float4 total = float4(
        simd_sum(partial.x), simd_sum(partial.y),
        simd_sum(partial.z), simd_sum(partial.w));
    if (lane == 0) {
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) C[m * C_stride + C_column + n] = bfloat(total[column]);
        }
    }
}

kernel void experimental_projection_w8abf16_bf16_n4(
    device const bfloat *A         [[buffer(0)]],
    device const char   *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n0 = output.x * 4u;
    uint m = output.y;
    if (m >= M || n0 >= N) return;
    uint groups_per_row = (K + 31u) >> 5u;
    float4 partial = float4(0.0f);
    for (uint k = uint(lane); k < K; k += 32u) {
        float activation = float(A[m * K + k]);
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                int q = int(W[n * K + k]);
                float scale = float(scales[n * groups_per_row + (k >> 5u)]);
                partial[column] += activation * (float(q) * scale);
            }
        }
    }
    float4 total = float4(
        simd_sum(partial.x), simd_sum(partial.y),
        simd_sum(partial.z), simd_sum(partial.w));
    if (lane == 0) {
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) C[m * C_stride + C_column + n] = bfloat(total[column]);
        }
    }
}

// Eight-output variants trade twice the accumulator footprint for additional
// activation reuse. They are a separate research schedule, not an alias or an
// automatic replacement for N4.
kernel void experimental_projection_w4abf16_bf16_n8(
    device const bfloat *A         [[buffer(0)]],
    device const uchar  *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n0 = output.x * 8u;
    uint m = output.y;
    if (m >= M || n0 >= N) return;
    uint packed_row_bytes = (K + 1u) >> 1u;
    uint groups_per_row = (K + 31u) >> 5u;
    float4 partial_lo = float4(0.0f);
    float4 partial_hi = float4(0.0f);
    for (uint k = uint(lane); k < K; k += 32u) {
        float activation = float(A[m * K + k]);
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                uchar packed = W[n * packed_row_bytes + (k >> 1u)];
                int q = int((k & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u));
                q = q >= 8 ? q - 16 : q;
                float term = activation * (float(q) * float(scales[n * groups_per_row + (k >> 5u)]));
                if (column < 4u) partial_lo[column] += term;
                else partial_hi[column - 4u] += term;
            }
        }
    }
    float4 total_lo = float4(
        simd_sum(partial_lo.x), simd_sum(partial_lo.y),
        simd_sum(partial_lo.z), simd_sum(partial_lo.w));
    float4 total_hi = float4(
        simd_sum(partial_hi.x), simd_sum(partial_hi.y),
        simd_sum(partial_hi.z), simd_sum(partial_hi.w));
    if (lane == 0) {
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                float value = column < 4u ? total_lo[column] : total_hi[column - 4u];
                C[m * C_stride + C_column + n] = bfloat(value);
            }
        }
    }
}

kernel void experimental_projection_w8abf16_bf16_n8(
    device const bfloat *A         [[buffer(0)]],
    device const char   *W         [[buffer(1)]],
    device const half   *scales    [[buffer(2)]],
    device bfloat       *C         [[buffer(3)]],
    constant uint       &M         [[buffer(4)]],
    constant uint       &N         [[buffer(5)]],
    constant uint       &K         [[buffer(6)]],
    constant uint       &C_stride  [[buffer(7)]],
    constant uint       &C_column  [[buffer(8)]],
    uint2 output                    [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    uint n0 = output.x * 8u;
    uint m = output.y;
    if (m >= M || n0 >= N) return;
    uint groups_per_row = (K + 31u) >> 5u;
    float4 partial_lo = float4(0.0f);
    float4 partial_hi = float4(0.0f);
    for (uint k = uint(lane); k < K; k += 32u) {
        float activation = float(A[m * K + k]);
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                float term = activation * (float(W[n * K + k]) * float(scales[n * groups_per_row + (k >> 5u)]));
                if (column < 4u) partial_lo[column] += term;
                else partial_hi[column - 4u] += term;
            }
        }
    }
    float4 total_lo = float4(
        simd_sum(partial_lo.x), simd_sum(partial_lo.y),
        simd_sum(partial_lo.z), simd_sum(partial_lo.w));
    float4 total_hi = float4(
        simd_sum(partial_hi.x), simd_sum(partial_hi.y),
        simd_sum(partial_hi.z), simd_sum(partial_hi.w));
    if (lane == 0) {
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                float value = column < 4u ? total_lo[column] : total_hi[column - 4u];
                C[m * C_stride + C_column + n] = bfloat(value);
            }
        }
    }
}

// W4 next-round schedule: each lane consumes both values in one packed byte.
kernel void experimental_projection_w4abf16_bf16_n4_packed2(
    device const bfloat *A [[buffer(0)]], device const uchar *W [[buffer(1)]],
    device const half *scales [[buffer(2)]], device bfloat *C [[buffer(3)]],
    constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]], constant uint &C_stride [[buffer(7)]],
    constant uint &C_column [[buffer(8)]],
    uint2 output [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint n0 = output.x * 4u, m = output.y;
    if (m >= M || n0 >= N) return;
    uint row_bytes = (K + 1u) >> 1u, groups = (K + 31u) >> 5u;
    float4 even = 0.0f, odd = 0.0f;
    for (uint k0 = uint(lane) * 2u; k0 < K; k0 += 64u) {
        float a0 = float(A[m * K + k0]);
        bool has_k1 = k0 + 1u < K;
        float a1 = has_k1 ? float(A[m * K + k0 + 1u]) : 0.0f;
        for (uint column = 0u; column < 4u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                uchar packed = W[n * row_bytes + (k0 >> 1u)];
                int q0 = int(packed & 0x0fu); q0 = q0 >= 8 ? q0 - 16 : q0;
                int q1 = int(packed >> 4u); q1 = q1 >= 8 ? q1 - 16 : q1;
                float scale = float(scales[n * groups + (k0 >> 5u)]);
                even[column] += a0 * (float(q0) * scale);
                if (has_k1) odd[column] += a1 * (float(q1) * scale);
            }
        }
    }
    float4 total;
    for (uint column = 0u; column < 4u; ++column)
        total[column] = simd_sum(even[column]) + simd_sum(odd[column]);
    if (lane == 0) for (uint column = 0u; column < 4u; ++column) {
        uint n = n0 + column;
        if (n < N) C[m * C_stride + C_column + n] = bfloat(total[column]);
    }
}

// W8 next-round schedule: four adjacent K values per lane, with one activation
// vector reused across eight output rows. The scalar tail keeps arbitrary K legal.
kernel void experimental_projection_w8abf16_bf16_n8_k4(
    device const bfloat *A [[buffer(0)]], device const char *W [[buffer(1)]],
    device const half *scales [[buffer(2)]], device bfloat *C [[buffer(3)]],
    constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]], constant uint &C_stride [[buffer(7)]],
    constant uint &C_column [[buffer(8)]],
    uint2 output [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint n0 = output.x * 8u, m = output.y;
    if (m >= M || n0 >= N) return;
    uint groups = (K + 31u) >> 5u;
    float4 lo = 0.0f, hi = 0.0f;
    for (uint k0 = uint(lane) * 4u; k0 < K; k0 += 128u) {
        uint count = min(4u, K - k0);
        float4 av = 0.0f;
        for (uint j = 0u; j < count; ++j) av[j] = float(A[m * K + k0 + j]);
        for (uint column = 0u; column < 8u; ++column) {
            uint n = n0 + column;
            if (n < N) {
                float sum = 0.0f;
                for (uint j = 0u; j < count; ++j) {
                    float scale = float(scales[n * groups + ((k0 + j) >> 5u)]);
                    sum += av[j] * (float(W[n * K + k0 + j]) * scale);
                }
                if (column < 4u) lo[column] += sum; else hi[column - 4u] += sum;
            }
        }
    }
    float4 total_lo, total_hi;
    for (uint column = 0u; column < 4u; ++column) {
        total_lo[column] = simd_sum(lo[column]);
        total_hi[column] = simd_sum(hi[column]);
    }
    if (lane == 0) for (uint column = 0u; column < 8u; ++column) {
        uint n = n0 + column;
        if (n < N) C[m * C_stride + C_column + n] = bfloat(column < 4u ? total_lo[column] : total_hi[column - 4u]);
    }
}

constant uint TILE_M = 8;
constant uint TILE_N = 8;
constant uint TILE16 = 16;
constant uint GEMV_N = 8;
constant uint GEMV_TG = 256;
constant uint GEMM_RMSNORM_MAX_N = 6144;
constant uint HEADWISE_RMSNORM_MAX_D = 512;
constant uint FINAL_ARGMAX_TILE_N = 8;

kernel void gemm_f16(
    device const bfloat *A          [[buffer(0)]],  // [M, K] row-major
    device const bfloat *B          [[buffer(1)]],  // [N, K] col-major (transposed)
    device bfloat       *C          [[buffer(2)]],  // [M, N] row-major
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    uint2 tid                     [[thread_position_in_threadgroup]],
    uint2 tg_size                 [[threads_per_threadgroup]]
) {
    uint row = gid.x * TILE_M + tid.x;
    uint col = gid.y * TILE_N + tid.y;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        float a = float(A[row * K + k]);
        float b = float(B[col * K + k]);  // B transposed: [N,K]
        acc += a * b;
    }

    uint idx = row * N + col;
    float prior = beta == 0.0f ? 0.0f : float(C[idx]) * beta;
    C[idx] = bf16_sat(acc * alpha + prior);
}

// Small/probe matrix tiled GEMM: C[M,N] = A[M,K] * B[K,N].
// B is stored transposed as [N,K]. This keeps the general kernel as the
// fallback and only gives dispatch a bounded aligned tile option.
kernel void gemm_f16_tiled16(
    device const bfloat *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device bfloat       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    uint2 tid                     [[thread_position_in_threadgroup]]
) {
    uint row = gid.x * TILE16 + tid.x;
    uint col = gid.y * TILE16 + tid.y;

    threadgroup bfloat tile_a[16][16];
    threadgroup bfloat tile_b[16][16];

    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TILE16) {
        uint a_col = k0 + tid.y;
        uint b_col = k0 + tid.x;
        tile_a[tid.x][tid.y] = (row < M && a_col < K) ? A[row * K + a_col] : bfloat(0.0);
        tile_b[tid.x][tid.y] = (col < N && b_col < K) ? B[col * K + b_col] : bfloat(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE16; kk++) {
            acc += float(tile_a[tid.x][kk]) * float(tile_b[kk][tid.y]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        uint idx = row * N + col;
        float prior = beta == 0.0f ? 0.0f : float(C[idx]) * beta;
        C[idx] = bf16_sat(acc * alpha + prior);
    }
}

// Decode/prompt microbatch GEMM: C[M,N] = A[M,K] * B[N,K]^T.
//
// The general/tiled kernels map one output element to one thread. That is
// correct but wastes most lanes for M<=16 and makes every thread serially walk
// all K. This path maps one row and eight output columns to a threadgroup, then
// reduces K cooperatively across 256 lanes.
kernel void gemm_f16_vec8(
    device const bfloat *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device bfloat       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]]
) {
    uint row = gid.x;
    uint col_base = gid.y * GEMV_N;
    if (row >= M) return;

    threadgroup float partial[8][256];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    for (uint k = tid; k < K; k += GEMV_TG) {
        float a = float(A[row * K + k]);
        uint col = col_base;
        if (col < N) acc0 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc1 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc2 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc3 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc4 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc5 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc6 += a * float(B[col * K + k]);
        col++;
        if (col < N) acc7 += a * float(B[col * K + k]);
    }

    partial[0][tid] = acc0;
    partial[1][tid] = acc1;
    partial[2][tid] = acc2;
    partial[3][tid] = acc3;
    partial[4][tid] = acc4;
    partial[5][tid] = acc5;
    partial[6][tid] = acc6;
    partial[7][tid] = acc7;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GEMV_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[0][tid] += partial[0][tid + stride];
            partial[1][tid] += partial[1][tid + stride];
            partial[2][tid] += partial[2][tid + stride];
            partial[3][tid] += partial[3][tid + stride];
            partial[4][tid] += partial[4][tid + stride];
            partial[5][tid] += partial[5][tid + stride];
            partial[6][tid] += partial[6][tid + stride];
            partial[7][tid] += partial[7][tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < GEMV_N) {
        uint col = col_base + tid;
        if (col < N) {
            uint idx = row * N + col;
            float prior = beta == 0.0f ? 0.0f : float(C[idx]) * beta;
            C[idx] = bf16_sat(partial[tid][0] * alpha + prior);
        }
    }
}

// Apple9/10 exact batch-eight projection GEMM.
//
// Eight simdgroups share the weight stream across all eight request rows.
// This is the promoted production path because its generated tokens are
// qualified against the independent Gemma reference.
kernel void gemm_f16_batch8(
    device const bfloat *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device bfloat       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort simdgroup              [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = gid.x * 8u;
    uint col = gid.y * 8u + uint(simdgroup);

    float accumulators[8] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    if (col < N) {
        for (uint k = uint(lane); k < K; k += 32u) {
            float weight = float(B[col * K + k]);
            for (uint row = 0u; row < 8u && row_base + row < M; row++) {
                accumulators[row] += float(A[(row_base + row) * K + k]) * weight;
            }
        }
    }
    for (uint row = 0u; row < 8u; row++) {
        accumulators[row] = simd_sum(accumulators[row]);
    }
    if (lane == 0 && col < N) {
        for (uint row = 0u; row < 8u && row_base + row < M; row++) {
            uint output_index = (row_base + row) * N + col;
            float prior = beta == 0.0f ? 0.0f : float(C[output_index]) * beta;
            C[output_index] = bf16_sat(accumulators[row] * alpha + prior);
        }
    }
}

// QKV prefill keeps projection sums in FP32 until head normalization.
kernel void qkv_project_f32_batch8(
    device const bfloat *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device float      *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort simdgroup              [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = gid.x * 8u;
    uint col = gid.y * 8u + uint(simdgroup);

    float accumulators[8] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    if (col < N) {
        for (uint k = uint(lane); k < K; k += 32u) {
            float weight = float(B[col * K + k]);
            for (uint row = 0u; row < 8u && row_base + row < M; row++) {
                accumulators[row] += float(A[(row_base + row) * K + k]) * weight;
            }
        }
    }
    for (uint row = 0u; row < 8u; row++) {
        accumulators[row] = simd_sum(accumulators[row]);
    }
    if (lane == 0 && col < N) {
        for (uint row = 0u; row < 8u && row_base + row < M; row++) {
            uint output_index = (row_base + row) * N + col;
            float prior = beta == 0.0f ? 0.0f : float(C[output_index]) * beta;
            C[output_index] = accumulators[row] * alpha + prior;
        }
    }
}

// Experimental Apple9/10 batch-eight projection GEMM.
//
// One SIMD group computes an 8x8 output tile with the public SIMD-group matrix
// instructions. A is already row-major [8,K]. Weights are stored as [N,K], so
// each tile is staged in threadgroup memory as the required [K,N] matrix before
// the SIMD-group load. This replaces eight scalar dot-product reductions per
// output column with the GPU's native 8x8 matrix primitive.
// It is compiled and hardware-tested, but not selected for production until
// the full-model numerical parity gate passes.
kernel void gemm_f16_simdgroup8x8(
    device const bfloat *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device bfloat       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &N          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant float    &alpha      [[buffer(6)]],
    constant float    &beta       [[buffer(7)]],
    uint2 gid                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]]
) {
    uint row_base = gid.x * 8u;
    uint col_base = gid.y * 8u;
    if (row_base + 8u > M || col_base + 8u > N) {
        return;
    }

    simdgroup_float8x8 mat_a;
    simdgroup_float8x8 mat_b;
    simdgroup_float8x8 mat_c(0.0f);
    threadgroup float activation_tile[64];
    threadgroup float weight_tile[64];
    for (uint k = 0u; k < K; k += 8u) {
        for (uint tile_index = uint(lane); tile_index < 64u; tile_index += 32u) {
            uint tile_row = tile_index / 8u;
            uint tile_col = tile_index % 8u;
            activation_tile[tile_index] =
                float(A[(row_base + tile_row) * K + k + tile_col]);
            weight_tile[tile_index] =
                float(B[(col_base + tile_col) * K + k + tile_row]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(mat_a, activation_tile, 8);
        simdgroup_load(mat_b, weight_tile, 8);
        simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
    }

    threadgroup float output_tile[64];
    simdgroup_store(mat_c, output_tile, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile_index = uint(lane); tile_index < 64u; tile_index += 32u) {
        uint row = row_base + tile_index / 8u;
        uint col = col_base + tile_index % 8u;
        uint output_index = row * N + col;
        float prior = beta == 0.0f ? 0.0f : float(C[output_index]) * beta;
        C[output_index] = bf16_sat(output_tile[tile_index] * alpha + prior);
    }
}

// Four SIMD groups share 32x32 BF16/FP16 tiles; products accumulate in FP32.
// The BF16 runtime path is opt-in and shape bounded. Both output ABIs use
// the same computation, preserving QKV FP32 until head normalization.
inline void prefill_mma32_tile(device const bfloat *A, device const bfloat *B,
    uint M, uint N, uint K, uint2 group, ushort tid, ushort sg,
    threadgroup bfloat *at, threadgroup bfloat *bt, threadgroup float *ct) {
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    const uint sm = uint(sg / 2u) * 16u;
    const uint sn = uint(sg % 2u) * 16u;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint kb = 0; kb < K; kb += 32u) {
        for (uint index = uint(tid); index < 1024u; index += 128u) {
            uint row = index / 32u;
            uint k = kb + index % 32u;
            at[index] = mr + row < M && k < K ? A[(mr + row) * K + k] : bfloat(0.0f);
            bt[index] = nc + row < N && k < K ? B[(nc + row) * K + k] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<bfloat, 8, 8> a0, a1, b0, b1;
            simdgroup_load(a0, at + sm * 32u + kk, 32);
            simdgroup_load(a1, at + (sm + 8u) * 32u + kk, 32);
            simdgroup_load(b0, bt + sn * 32u + kk, 32, ulong2(0), true);
            simdgroup_load(b1, bt + (sn + 8u) * 32u + kk, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        // All four groups must finish reading before any overwrite the tiles.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ct + sm * 32u + sn, 32);
    simdgroup_store(c01, ct + sm * 32u + sn + 8u, 32);
    simdgroup_store(c10, ct + (sm + 8u) * 32u + sn, 32);
    simdgroup_store(c11, ct + (sm + 8u) * 32u + sn + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


kernel void qkv_project_f32_mma32(
    device const bfloat *A [[buffer(0)]],
    device const bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat at[32 * 32];
    threadgroup bfloat bt[32 * 32];
    threadgroup float ct[32 * 32];
    prefill_mma32_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u;
        uint col = nc + index % 32u;
        if (row < M && col < N) {
            uint output = row * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            C[output] = alpha * ct[index] + prior;
        }
    }
}


kernel void gemm_f16_mma32(
    device const bfloat *A [[buffer(0)]],
    device const bfloat *B [[buffer(1)]],
    device bfloat *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    constant float &beta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat at[32 * 32];
    threadgroup bfloat bt[32 * 32];
    threadgroup float ct[32 * 32];
    prefill_mma32_tile(A, B, M, N, K, group, tid, sg, at, bt, ct);
    const uint mr = group.x * 32u;
    const uint nc = group.y * 32u;
    for (uint index = uint(tid); index < 1024u; index += 128u) {
        uint row = mr + index / 32u;
        uint col = nc + index % 32u;
        if (row < M && col < N) {
            uint output = row * N + col;
            float prior = beta == 0.0f ? 0.0f : beta * float(C[output]);
            C[output] = bf16_sat(alpha * ct[index] + prior);
        }
    }
}

kernel void gemm_rmsnorm_f16(
    device const bfloat *A          [[buffer(0)]],  // [M, K] row-major
    device const bfloat *B          [[buffer(1)]],  // [N, K] col-major (transposed)
    device const bfloat *gamma      [[buffer(2)]],  // [N]
    device bfloat       *C          [[buffer(3)]],  // [M, N] row-major
    constant uint     &M          [[buffer(4)]],
    constant uint     &N          [[buffer(5)]],
    constant uint     &K          [[buffer(6)]],
    constant float    &eps        [[buffer(7)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint row                      [[threadgroup_position_in_grid]]
) {
    if (row >= M) return;

    threadgroup float shared_sum[256];
    threadgroup float projected[6144];
    float local_sum = 0.0f;
    for (uint col = tid; col < N; col += tg_size) {
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[row * K + k]) * float(B[col * K + k]);
        }
        if (N <= GEMM_RMSNORM_MAX_N) {
            projected[col] = acc;
        }
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(N) + eps);
    for (uint col = tid; col < N; col += tg_size) {
        float acc = 0.0f;
        if (N <= GEMM_RMSNORM_MAX_N) {
            acc = projected[col];
        } else {
            for (uint k = 0; k < K; k++) {
                acc += float(A[row * K + k]) * float(B[col * K + k]);
            }
        }
        C[row * N + col] = bf16_sat(acc * rms * float(gamma[col]));
    }
}

kernel void gemm_headwise_rmsnorm_f16(
    device const bfloat *A          [[buffer(0)]],  // [M, K] row-major
    device const bfloat *B          [[buffer(1)]],  // [total_rows, K] col-major (transposed)
    device const bfloat *gamma      [[buffer(2)]],  // [head_dim]
    device bfloat       *C          [[buffer(3)]],  // [M, num_heads * head_dim]
    constant uint     &M          [[buffer(4)]],
    constant uint     &K          [[buffer(5)]],
    constant uint     &head_dim   [[buffer(6)]],
    constant uint     &num_heads  [[buffer(7)]],
    constant uint     &b_row_offset [[buffer(8)]],
    constant float    &eps        [[buffer(9)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid / num_heads;
    uint head = gid % num_heads;
    if (token >= M) return;

    threadgroup float shared_sum[256];
    threadgroup float projected[512];
    float local_sum = 0.0f;
    uint b_head_base = b_row_offset + head * head_dim;
    uint c_head_base = token * num_heads * head_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[token * K + k]) * float(B[row * K + k]);
        }
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            projected[d] = acc;
        }
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            acc = projected[d];
        } else {
            uint row = b_head_base + d;
            for (uint k = 0; k < K; k++) {
                acc += float(A[token * K + k]) * float(B[row * K + k]);
            }
        }
        C[c_head_base + d] = bf16_sat(acc * rms * float(gamma[d]));
    }
}

kernel void gemm_headwise_rmsnorm_unit_f16(
    device const bfloat *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device bfloat       *C          [[buffer(2)]],
    constant uint     &M          [[buffer(3)]],
    constant uint     &K          [[buffer(4)]],
    constant uint     &head_dim   [[buffer(5)]],
    constant uint     &num_heads  [[buffer(6)]],
    constant uint     &b_row_offset [[buffer(7)]],
    constant float    &eps        [[buffer(8)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid / num_heads;
    uint head = gid % num_heads;
    if (token >= M) return;

    threadgroup float shared_sum[256];
    threadgroup float projected[512];
    float local_sum = 0.0f;
    uint b_head_base = b_row_offset + head * head_dim;
    uint c_head_base = token * num_heads * head_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < K; k++) {
            acc += float(A[token * K + k]) * float(B[row * K + k]);
        }
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            projected[d] = acc;
        }
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            acc = projected[d];
        } else {
            uint row = b_head_base + d;
            for (uint k = 0; k < K; k++) {
                acc += float(A[token * K + k]) * float(B[row * K + k]);
            }
        }
        C[c_head_base + d] = bf16_sat(acc * rms);
    }
}

kernel void qkv_headwise_rmsnorm_f16(
    device const bfloat *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device const bfloat *q_gamma    [[buffer(2)]],
    device const bfloat *k_gamma    [[buffer(3)]],
    device const bfloat *v_gamma    [[buffer(4)]],
    device bfloat       *Q          [[buffer(5)]],
    device bfloat       *K_out      [[buffer(6)]],
    device bfloat       *V_out      [[buffer(7)]],
    constant uint     &M          [[buffer(8)]],
    constant uint     &hidden_k   [[buffer(9)]],
    constant uint     &head_dim   [[buffer(10)]],
    constant uint     &num_q_heads [[buffer(11)]],
    constant uint     &num_kv_heads [[buffer(12)]],
    constant uint     &q_row_offset [[buffer(13)]],
    constant uint     &k_row_offset [[buffer(14)]],
    constant uint     &v_row_offset [[buffer(15)]],
    constant float    &eps        [[buffer(16)]],
    constant uint     &v_has_gamma [[buffer(17)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    uint token = gid / total_heads;
    uint packed_head = gid % total_heads;
    if (token >= M) return;

    uint head = packed_head;
    uint b_head_base = q_row_offset + head * head_dim;
    uint c_head_base = token * num_q_heads * head_dim + head * head_dim;
    device bfloat *out = Q;
    device const bfloat *gamma = q_gamma;
    bool use_gamma = true;

    if (packed_head >= num_q_heads) {
        uint kv_packed = packed_head - num_q_heads;
        if (kv_packed < num_kv_heads) {
            head = kv_packed;
            b_head_base = k_row_offset + head * head_dim;
            c_head_base = token * num_kv_heads * head_dim + head * head_dim;
            out = K_out;
            gamma = k_gamma;
        } else {
            head = kv_packed - num_kv_heads;
            b_head_base = v_row_offset + head * head_dim;
            c_head_base = token * num_kv_heads * head_dim + head * head_dim;
            out = V_out;
            gamma = v_gamma;
            use_gamma = v_has_gamma != 0;
        }
    }

    threadgroup float shared_sum[256];
    threadgroup float projected[512];
    float local_sum = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < hidden_k; k++) {
            acc += float(A[token * hidden_k + k]) * float(B[row * hidden_k + k]);
        }
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            projected[d] = acc;
        }
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        if (head_dim <= HEADWISE_RMSNORM_MAX_D) {
            acc = projected[d];
        } else {
            uint row = b_head_base + d;
            for (uint k = 0; k < hidden_k; k++) {
                acc += float(A[token * hidden_k + k]) * float(B[row * hidden_k + k]);
            }
        }
        float scale = use_gamma ? float(gamma[d]) : 1.0f;
        out[c_head_base + d] = bf16_sat(acc * rms * scale);
    }
}

kernel void qkv_headwise_rmsnorm_rope_cache_f16(
    device const bfloat *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device const bfloat *q_gamma    [[buffer(2)]],
    device const bfloat *k_gamma    [[buffer(3)]],
    device const bfloat *v_gamma    [[buffer(4)]],
    device bfloat       *Q          [[buffer(5)]],
    device bfloat       *K_out      [[buffer(6)]],
    device bfloat       *V_out      [[buffer(7)]],
    device const float *cos_table [[buffer(8)]],
    device const float *sin_table [[buffer(9)]],
    device const int  *positions  [[buffer(10)]],
    device const int  *slot_map   [[buffer(11)]],
    device bfloat       *k_cache    [[buffer(12)]],
    device bfloat       *v_cache    [[buffer(13)]],
    constant uint     &M          [[buffer(14)]],
    constant uint     &hidden_k   [[buffer(15)]],
    constant uint     &head_dim   [[buffer(16)]],
    constant uint     &num_q_heads [[buffer(17)]],
    constant uint     &num_kv_heads [[buffer(18)]],
    constant uint     &q_row_offset [[buffer(19)]],
    constant uint     &k_row_offset [[buffer(20)]],
    constant uint     &v_row_offset [[buffer(21)]],
    constant float    &eps        [[buffer(22)]],
    constant uint     &v_has_gamma [[buffer(23)]],
    constant uint     &rope_dim   [[buffer(24)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    uint token = gid / total_heads;
    uint packed_head = gid % total_heads;
    if (token >= M || head_dim > HEADWISE_RMSNORM_MAX_D) return;

    uint head = packed_head;
    uint b_head_base = q_row_offset + head * head_dim;
    uint q_dim = num_q_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;
    uint row_head_base = head * head_dim;
    uint c_head_base = token * q_dim + row_head_base;
    device bfloat *out = Q;
    device const bfloat *gamma = q_gamma;
    bool use_gamma = true;
    bool is_k = false;
    bool is_v = false;

    if (packed_head >= num_q_heads) {
        uint kv_packed = packed_head - num_q_heads;
        if (kv_packed < num_kv_heads) {
            head = kv_packed;
            b_head_base = k_row_offset + head * head_dim;
            row_head_base = head * head_dim;
            c_head_base = token * kv_dim + row_head_base;
            out = K_out;
            gamma = k_gamma;
            is_k = true;
        } else {
            head = kv_packed - num_kv_heads;
            b_head_base = v_row_offset + head * head_dim;
            row_head_base = head * head_dim;
            c_head_base = token * kv_dim + row_head_base;
            out = V_out;
            gamma = v_gamma;
            use_gamma = v_has_gamma != 0;
            is_v = true;
        }
    }

    threadgroup float shared_sum[256];
    threadgroup float projected[512];
    threadgroup float normalized[512];
    float local_sum = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        float acc = 0.0f;
        for (uint k = 0; k < hidden_k; k++) {
            acc += float(A[token * hidden_k + k]) * float(B[row * hidden_k + k]);
        }
        projected[d] = acc;
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint d = tid; d < head_dim; d += tg_size) {
        float scale = use_gamma ? float(gamma[d]) : 1.0f;
        normalized[d] = float(bf16_sat(projected[d] * rms * scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int slot = slot_map[token];
    uint half_rope = rope_dim / 2;
    int pos = positions[token];
    for (uint d = tid; d < head_dim; d += tg_size) {
        float value = normalized[d];
        if (!is_v) {
            if (d < half_rope) {
                uint pair = d;
                float x0 = normalized[pair];
                float x1 = normalized[pair + head_dim / 2];
                float cos_val = cos_table[pos * half_rope + pair];
                float sin_val = sin_table[pos * half_rope + pair];
                value = x0 * cos_val - x1 * sin_val;
            } else if (d >= head_dim / 2 && d < head_dim / 2 + half_rope) {
                uint pair = d - head_dim / 2;
                float x0 = normalized[pair];
                float x1 = normalized[pair + head_dim / 2];
                float cos_val = cos_table[pos * half_rope + pair];
                float sin_val = sin_table[pos * half_rope + pair];
                value = x0 * sin_val + x1 * cos_val;
            }
        }
        bfloat out_value = bf16_sat(value);
        out[c_head_base + d] = out_value;
        if (slot >= 0 && is_k) {
            k_cache[uint(slot) * kv_dim + row_head_base + d] = out_value;
        } else if (slot >= 0 && is_v) {
            v_cache[uint(slot) * kv_dim + row_head_base + d] = out_value;
        }
    }
}

// Same norm, dtype rounding, RoPE and cache stores as the fused path.
// Buffer A holds packed FP32 QKV; buffer B is unused to share the host ABI.
kernel void qkv_projected_rmsnorm_rope_cache_f16(
    device const float *A          [[buffer(0)]],
    device const bfloat *B          [[buffer(1)]],
    device const bfloat *q_gamma    [[buffer(2)]],
    device const bfloat *k_gamma    [[buffer(3)]],
    device const bfloat *v_gamma    [[buffer(4)]],
    device bfloat       *Q          [[buffer(5)]],
    device bfloat       *K_out      [[buffer(6)]],
    device bfloat       *V_out      [[buffer(7)]],
    device const float *cos_table [[buffer(8)]],
    device const float *sin_table [[buffer(9)]],
    device const int  *positions  [[buffer(10)]],
    device const int  *slot_map   [[buffer(11)]],
    device bfloat       *k_cache    [[buffer(12)]],
    device bfloat       *v_cache    [[buffer(13)]],
    constant uint     &M          [[buffer(14)]],
    constant uint     &hidden_k   [[buffer(15)]],
    constant uint     &head_dim   [[buffer(16)]],
    constant uint     &num_q_heads [[buffer(17)]],
    constant uint     &num_kv_heads [[buffer(18)]],
    constant uint     &q_row_offset [[buffer(19)]],
    constant uint     &k_row_offset [[buffer(20)]],
    constant uint     &v_row_offset [[buffer(21)]],
    constant float    &eps        [[buffer(22)]],
    constant uint     &v_has_gamma [[buffer(23)]],
    constant uint     &rope_dim   [[buffer(24)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    uint token = gid / total_heads;
    uint packed_head = gid % total_heads;
    if (token >= M || head_dim > HEADWISE_RMSNORM_MAX_D) return;

    uint head = packed_head;
    uint b_head_base = q_row_offset + head * head_dim;
    uint q_dim = num_q_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;
    uint row_head_base = head * head_dim;
    uint c_head_base = token * q_dim + row_head_base;
    device bfloat *out = Q;
    device const bfloat *gamma = q_gamma;
    bool use_gamma = true;
    bool is_k = false;
    bool is_v = false;

    if (packed_head >= num_q_heads) {
        uint kv_packed = packed_head - num_q_heads;
        if (kv_packed < num_kv_heads) {
            head = kv_packed;
            b_head_base = k_row_offset + head * head_dim;
            row_head_base = head * head_dim;
            c_head_base = token * kv_dim + row_head_base;
            out = K_out;
            gamma = k_gamma;
            is_k = true;
        } else {
            head = kv_packed - num_kv_heads;
            b_head_base = v_row_offset + head * head_dim;
            row_head_base = head * head_dim;
            c_head_base = token * kv_dim + row_head_base;
            out = V_out;
            gamma = v_gamma;
            use_gamma = v_has_gamma != 0;
            is_v = true;
        }
    }

    threadgroup float shared_sum[256];
    threadgroup float projected[512];
    threadgroup float normalized[512];
    float local_sum = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint row = b_head_base + d;
        uint projection_width = (num_q_heads + 2u * num_kv_heads) * head_dim;
        float acc = A[token * projection_width + row];
        projected[d] = acc;
        local_sum += acc * acc;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);
    for (uint d = tid; d < head_dim; d += tg_size) {
        float scale = use_gamma ? float(gamma[d]) : 1.0f;
        normalized[d] = float(bf16_sat(projected[d] * rms * scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int slot = slot_map[token];
    uint half_rope = rope_dim / 2;
    int pos = positions[token];
    for (uint d = tid; d < head_dim; d += tg_size) {
        float value = normalized[d];
        if (!is_v) {
            if (d < half_rope) {
                uint pair = d;
                float x0 = normalized[pair];
                float x1 = normalized[pair + head_dim / 2];
                float cos_val = cos_table[pos * half_rope + pair];
                float sin_val = sin_table[pos * half_rope + pair];
                value = x0 * cos_val - x1 * sin_val;
            } else if (d >= head_dim / 2 && d < head_dim / 2 + half_rope) {
                uint pair = d - head_dim / 2;
                float x0 = normalized[pair];
                float x1 = normalized[pair + head_dim / 2];
                float cos_val = cos_table[pos * half_rope + pair];
                float sin_val = sin_table[pos * half_rope + pair];
                value = x0 * sin_val + x1 * cos_val;
            }
        }
        bfloat out_value = bf16_sat(value);
        out[c_head_base + d] = out_value;
        if (slot >= 0 && is_k) {
            k_cache[uint(slot) * kv_dim + row_head_base + d] = out_value;
        } else if (slot >= 0 && is_v) {
            v_cache[uint(slot) * kv_dim + row_head_base + d] = out_value;
        }
    }
}

kernel void copy_f16(
    device const bfloat *src [[buffer(0)]],
    device bfloat       *dst [[buffer(1)]],
    constant uint     &len [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    dst[gid] = src[gid];
}

kernel void copy_ple_layer_f16(
    device const bfloat *packed_ple [[buffer(0)]],
    device bfloat       *dst        [[buffer(1)]],
    constant uint     &num_tokens [[buffer(2)]],
    constant uint     &num_layers [[buffer(3)]],
    constant uint     &layer_idx  [[buffer(4)]],
    constant uint     &ple_dim    [[buffer(5)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= ple_dim || layer_idx >= num_layers) return;
    uint src_idx = token * num_layers * ple_dim + layer_idx * ple_dim + dim;
    dst[token * ple_dim + dim] = packed_ple[src_idx];
}

// ============================================================================
// Split fused QKV (interleaved) into planar Q/K/V.
// qkv stores [token][q_dim + 2*kv_dim] while kernels downstream expect
// contiguous planar Q, K, V regions.
// ============================================================================
kernel void split_qkv_f16(
    device const bfloat *qkv   [[buffer(0)]],  // [num_tokens, q_dim + 2*kv_dim]
    device bfloat *q           [[buffer(1)]],  // [num_tokens, q_dim]
    device bfloat *k           [[buffer(2)]],  // [num_tokens, kv_dim]
    device bfloat *v           [[buffer(3)]],  // [num_tokens, kv_dim]
    constant uint &num_tokens [[buffer(4)]],
    constant uint &q_dim      [[buffer(5)]],
    constant uint &kv_dim     [[buffer(6)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens) return;

    uint qkv_dim = q_dim + 2u * kv_dim;
    if (dim < q_dim) {
        q[token * q_dim + dim] = qkv[token * qkv_dim + dim];
    } else if (dim < q_dim + kv_dim) {
        uint kd = dim - q_dim;
        k[token * kv_dim + kd] = qkv[token * qkv_dim + q_dim + kd];
    } else if (dim < q_dim + 2u * kv_dim) {
        uint vd = dim - q_dim - kv_dim;
        v[token * kv_dim + vd] = qkv[token * qkv_dim + q_dim + kv_dim + vd];
    }
}

// ============================================================================
// Embedding gather (row lookup) f16
// ==========================================================================
kernel void embedding_gather_f16(
    device const bfloat *embedding   [[buffer(0)]],
    device const uint *token_ids    [[buffer(1)]],
    device bfloat       *out         [[buffer(2)]],
    constant uint     &num_tokens  [[buffer(3)]],
    constant uint     &hidden      [[buffer(4)]],
    constant uint     &vocab       [[buffer(5)]],
    constant float    &scale       [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= hidden) return;

    uint tok = token_ids[token];
    if (tok >= vocab) {
        out[token * hidden + dim] = bfloat(0.0);
        return;
    }

    out[token * hidden + dim] = bf16_sat(float(embedding[tok * hidden + dim]) * scale);
}

// ============================================================================
// Partial RoPE (Gemma 4 style: only rotate first rope_dim dims)
// ============================================================================
kernel void rope_partial_f16(
    device bfloat       *q          [[buffer(0)]],  // [num_tokens, q_dim]
    device bfloat       *k          [[buffer(1)]],  // [num_tokens, kv_dim]
    device const float *cos_table [[buffer(2)]],  // [max_pos, head_dim/2]
    device const float *sin_table [[buffer(3)]],  // [max_pos, head_dim/2]
    device const int  *positions  [[buffer(4)]],  // [num_tokens]
    constant uint     &num_tokens [[buffer(5)]],
    constant uint     &num_heads  [[buffer(6)]],
    constant uint     &num_kv_heads [[buffer(7)]],
    constant uint     &head_dim   [[buffer(8)]],
    constant uint     &rope_dim  [[buffer(9)]],   // dims to rotate (typically head_dim)
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint pair = gid.y;  // pair index within one head
    if (token >= num_tokens) return;

    int pos = positions[token];
    uint half_rope = rope_dim / 2;
    if (pair >= half_rope) return;

    float cos_val = cos_table[pos * half_rope + pair];
    float sin_val = sin_table[pos * half_rope + pair];

    // Apply to all Q heads
    uint q_dim = num_heads * head_dim;
    for (uint h = 0; h < num_heads; h++) {
        uint base = token * q_dim + h * head_dim;
        uint i0 = base + pair;
        uint i1 = base + pair + head_dim / 2;
        float x0 = float(q[i0]);
        float x1 = float(q[i1]);
        q[i0] = bf16_sat(x0 * cos_val - x1 * sin_val);
        q[i1] = bf16_sat(x0 * sin_val + x1 * cos_val);
    }

    // Apply to all KV heads
    uint kv_dim = num_kv_heads * head_dim;
    for (uint h = 0; h < num_kv_heads; h++) {
        uint base = token * kv_dim + h * head_dim;
        uint i0 = base + pair;
        uint i1 = base + pair + head_dim / 2;
        float x0 = float(k[i0]);
        float x1 = float(k[i1]);
        k[i0] = bf16_sat(x0 * cos_val - x1 * sin_val);
        k[i1] = bf16_sat(x0 * sin_val + x1 * cos_val);
    }
}

// ============================================================================
// KV Cache Write (slot-mapped)
// ============================================================================
kernel void kv_cache_write_f16(
    device const bfloat *k_src      [[buffer(0)]],  // [num_tokens, kv_dim]
    device const bfloat *v_src      [[buffer(1)]],  // [num_tokens, kv_dim]
    device bfloat       *k_cache    [[buffer(2)]],  // [num_blocks * block_size, kv_dim]
    device bfloat       *v_cache    [[buffer(3)]],
    device const int  *slot_map   [[buffer(4)]],  // [num_tokens] -> cache slot
    constant uint     &num_tokens [[buffer(5)]],
    constant uint     &kv_dim     [[buffer(6)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= kv_dim) return;

    int slot = slot_map[token];
    if (slot < 0) return;

    k_cache[uint(slot) * kv_dim + dim] = k_src[token * kv_dim + dim];
    v_cache[uint(slot) * kv_dim + dim] = v_src[token * kv_dim + dim];
}

// Experimental KV cache compression utilities. These kernels are not used by
// the normal checkpoint-native decode path; tests opt in explicitly and
// dequantize back to F16 before comparison. Quantization is symmetric per cache
// row.
kernel void experimental_kv_quantize_int8_f16(
    device const bfloat *src        [[buffer(0)]],  // [num_rows, kv_dim]
    device char       *dst        [[buffer(1)]],  // [num_rows, kv_dim]
    device float      *scales     [[buffer(2)]],  // [num_rows]
    constant uint     &num_rows   [[buffer(3)]],
    constant uint     &kv_dim     [[buffer(4)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint row                      [[threadgroup_position_in_grid]]
) {
    if (row >= num_rows) return;

    threadgroup float shared_max[256];
    float local_max = 0.0f;
    uint base = row * kv_dim;
    for (uint dim = tid; dim < kv_dim; dim += tg_size) {
        local_max = fmax(local_max, fabs(float(src[base + dim])));
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = shared_max[0] > 0.0f ? shared_max[0] / 127.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < kv_dim; dim += tg_size) {
        float q = round(float(src[base + dim]) / scale);
        q = clamp(q, -127.0f, 127.0f);
        dst[base + dim] = char(q);
    }
}

kernel void experimental_kv_dequantize_int8_f16(
    device const char  *src       [[buffer(0)]],  // [num_rows, kv_dim]
    device const float *scales    [[buffer(1)]],  // [num_rows]
    device bfloat        *dst       [[buffer(2)]],  // [num_rows, kv_dim]
    constant uint      &num_rows  [[buffer(3)]],
    constant uint      &kv_dim    [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
) {
    uint total = num_rows * kv_dim;
    if (gid >= total) return;
    uint row = gid / kv_dim;
    dst[gid] = bf16_sat(float(src[gid]) * scales[row]);
}

// ============================================================================
// Attention Decode (single Q token per sequence, paged KV)
// ============================================================================
// GQA-aware: each Q head group shares one KV head.
kernel void attention_decode_f16(
    device const bfloat *q          [[buffer(0)]],   // [num_seqs, q_dim]
    device const bfloat *k_cache    [[buffer(1)]],   // [total_blocks * block_size, kv_dim]
    device const bfloat *v_cache    [[buffer(2)]],
    device bfloat       *output     [[buffer(3)]],   // [num_seqs, q_dim]
    device const int  *block_tables [[buffer(4)]],  // [num_seqs, max_blocks_per_seq]
    device const int  *context_lens [[buffer(5)]],  // [num_seqs]
    constant uint     &num_seqs   [[buffer(6)]],
    constant uint     &num_heads  [[buffer(7)]],
    constant uint     &num_kv_heads [[buffer(8)]],
    constant uint     &head_dim   [[buffer(9)]],
    constant uint     &block_size [[buffer(10)]],
    constant uint     &max_blocks [[buffer(11)]],
    constant float    &scale      [[buffer(12)]],
    constant uint     &attention_window [[buffer(13)]], // 0 = full attention
    uint gid                      [[threadgroup_position_in_grid]]
) {
    // One thread per (seq, head) pair
    uint seq = gid / num_heads;
    uint head = gid % num_heads;
    if (seq >= num_seqs) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    int ctx_len = context_lens[seq];
    if (ctx_len <= 0) return;

    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;

    // Load Q vector for this head
    float q_shared[512]; // max head_dim
    for (uint d = 0; d < head_dim; d++) {
        q_shared[d] = float(q[seq * q_dim + head * head_dim + d]);
    }

    // Compute attention scores and weighted sum (online softmax)
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    threadgroup float out_accum[512];
    for (uint d = 0; d < head_dim; d++) {
        out_accum[d] = 0.0f;
    }

    uint attn_start = attention_window == 0
        ? 0
        : uint(ctx_len) - min(uint(ctx_len), attention_window);

    // Process each KV token in this layer's exact causal extent.
    for (uint t = attn_start; t < uint(ctx_len); t++) {
        uint block_idx = t / block_size;
        uint block_offset = t % block_size;
        int block_id = block_tables[seq * max_blocks + block_idx];
        if (block_id < 0) continue;

        // Dot product Q·K
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            uint k_idx = uint(block_id) * block_size * kv_dim
                       + block_offset * kv_dim
                       + kv_head * head_dim + d;
            score += q_shared[d] * float(k_cache[k_idx]);
        }
        // Reduce score across threads (simplified — single thread for now)
        score *= scale;

        // Online softmax update
        float old_max = max_score;
        max_score = max(max_score, score);
        float correction = exp(old_max - max_score);
        sum_exp = sum_exp * correction + exp(score - max_score);

        // Accumulate V weighted by attention
        float weight = exp(score - max_score);
        for (uint d = 0; d < head_dim; d++) {
            uint v_idx = uint(block_id) * block_size * kv_dim
                       + block_offset * kv_dim
                       + kv_head * head_dim + d;
            out_accum[d] = out_accum[d] * correction + weight * float(v_cache[v_idx]);
        }
    }

    // Write output
    float inv_sum = 1.0f / sum_exp;
    for (uint d = 0; d < head_dim; d++) {
        output[seq * q_dim + head * head_dim + d] = bf16_sat(out_accum[d] * inv_sum);
    }
}

// Single-pass decode attention for production decode shapes. One 32-lane
// simdgroup handles one (sequence, Q head). Each lane owns up to eight head
// dimensions, keeps Q and the online-softmax output accumulator in registers,
// and participates in exactly one Q.K reduction per KV token. The arbitrary
// physical block IDs in block_tables preserve request-owned/shared-prefix
// page semantics; the logical token coordinate still determines the exact
// sliding-window lower bound.
kernel void attention_decode_online_f16(
    device const bfloat *q          [[buffer(0)]],
    device const bfloat *k_cache    [[buffer(1)]],
    device const bfloat *v_cache    [[buffer(2)]],
    device bfloat       *output     [[buffer(3)]],
    device const int  *block_tables [[buffer(4)]],
    device const int  *context_lens [[buffer(5)]],
    constant uint     &num_seqs   [[buffer(6)]],
    constant uint     &num_heads  [[buffer(7)]],
    constant uint     &num_kv_heads [[buffer(8)]],
    constant uint     &head_dim   [[buffer(9)]],
    constant uint     &block_size [[buffer(10)]],
    constant uint     &max_blocks [[buffer(11)]],
    constant float    &scale      [[buffer(12)]],
    constant uint     &attention_window [[buffer(13)]],
    uint gid                      [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]]
) {
    uint seq = gid / num_heads;
    uint head = gid % num_heads;
    if (seq >= num_seqs || head_dim == 0 || head_dim > 256) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    int ctx_len_i = context_lens[seq];
    if (ctx_len_i <= 0) return;
    uint ctx_len = uint(ctx_len_i);
    uint attn_start = attention_window == 0
        ? 0
        : ctx_len - min(ctx_len, attention_window);

    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;

    float q_lane[8];
    float out_lane[8];
    for (uint slot = 0; slot < 8; slot++) {
        uint d = uint(lane) + slot * 32u;
        q_lane[slot] = d < head_dim
            ? float(q[seq * q_dim + head * head_dim + d])
            : 0.0f;
        out_lane[slot] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    for (uint t = attn_start; t < ctx_len; t++) {
        uint block_idx = t / block_size;
        uint block_offset = t % block_size;
        int block_id = block_tables[seq * max_blocks + block_idx];
        if (block_id < 0) continue;

        uint block_base = uint(block_id) * block_size * kv_dim;
        uint kv_base = block_base + block_offset * kv_dim + kv_head * head_dim;
        float partial_score = 0.0f;
        for (uint slot = 0; slot < 8; slot++) {
            uint d = uint(lane) + slot * 32u;
            if (d < head_dim) {
                partial_score += q_lane[slot] * float(k_cache[kv_base + d]);
            }
        }
        float score = simd_sum(partial_score) * scale;

        float next_max = max(max_score, score);
        float correction = exp(max_score - next_max);
        float weight = exp(score - next_max);
        sum_exp = sum_exp * correction + weight;
        for (uint slot = 0; slot < 8; slot++) {
            uint d = uint(lane) + slot * 32u;
            if (d < head_dim) {
                out_lane[slot] = out_lane[slot] * correction
                    + weight * float(v_cache[kv_base + d]);
            }
        }
        max_score = next_max;
    }

    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (uint slot = 0; slot < 8; slot++) {
        uint d = uint(lane) + slot * 32u;
        if (d < head_dim) {
            output[seq * q_dim + head * head_dim + d] = bf16_sat(out_lane[slot] * inv_sum);
        }
    }
}

// ============================================================================
// Attention Prefill (multi-token Q, causal mask, paged KV)
// ============================================================================
// Simplified prefill attention with causal masking.
kernel void attention_prefill_f16(
    device const bfloat *q          [[buffer(0)]],   // [total_q, q_dim]
    device const bfloat *k_cache    [[buffer(1)]],
    device const bfloat *v_cache    [[buffer(2)]],
    device bfloat       *output     [[buffer(3)]],   // [total_q, q_dim]
    device const int  *block_tables [[buffer(4)]],
    device const int  *context_lens [[buffer(5)]],
    device const int  *cu_seqlens [[buffer(6)]],   // [batch+1]
    device const int  *positions  [[buffer(7)]],   // absolute query positions
    constant uint     &total_q    [[buffer(8)]],
    constant uint     &batch_size [[buffer(9)]],
    constant uint     &num_heads  [[buffer(10)]],
    constant uint     &num_kv_heads [[buffer(11)]],
    constant uint     &head_dim   [[buffer(12)]],
    constant uint     &block_size [[buffer(13)]],
    constant uint     &max_blocks [[buffer(14)]],
    constant float    &scale      [[buffer(15)]],
    constant uint     &attention_window [[buffer(16)]], // 0 = full attention
    uint2 gid                     [[thread_position_in_grid]]
) {
    // Each thread handles one (q_token, head, dim_chunk)
    uint q_pos = gid.x;
    uint head = gid.y;
    if (q_pos >= total_q || head >= num_heads) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;

    // Find which sequence this q_pos belongs to
    uint seq = 0;
    for (uint s = 0; s < batch_size; s++) {
        if (int(q_pos) >= cu_seqlens[s] && int(q_pos) < cu_seqlens[s + 1]) {
            seq = s;
            break;
        }
    }
    uint seq_start = cu_seqlens[seq];
    uint q_offset = q_pos - seq_start;  // position within this sequence
    int ctx_len = context_lens[seq];

    // Compute attention for this Q position
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_vals[512]; // max head_dim — stack allocated
    for (uint d = 0; d < head_dim; d++) out_vals[d] = 0.0f;

    // Causal extent is absolute. q_offset is only chunk-local and would drop
    // an already-restored prefix during chunked/cache-assisted prefill.
    uint absolute_position = uint(max(positions[q_pos], 0));
    uint attn_len = min(uint(ctx_len), absolute_position + 1);
    uint attn_start = attention_window == 0
        ? 0
        : attn_len - min(attn_len, attention_window);

    for (uint t = attn_start; t < attn_len; t++) {
        uint block_idx = t / block_size;
        uint block_offset = t % block_size;
        int block_id = block_tables[seq * max_blocks + block_idx];

        // Q·K dot product
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float q_val = float(q[q_pos * q_dim + head * head_dim + d]);
            uint k_idx = uint(block_id) * block_size * kv_dim
                       + block_offset * kv_dim + kv_head * head_dim + d;
            score += q_val * float(k_cache[k_idx]);
        }
        score *= scale;

        float old_max = max_score;
        max_score = max(max_score, score);
        float correction = exp(old_max - max_score);
        sum_exp = sum_exp * correction + exp(score - max_score);

        float weight = exp(score - max_score);
        for (uint d = 0; d < head_dim; d++) {
            uint v_idx = uint(block_id) * block_size * kv_dim
                       + block_offset * kv_dim + kv_head * head_dim + d;
            out_vals[d] = out_vals[d] * correction + weight * float(v_cache[v_idx]);
        }
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        output[q_pos * q_dim + head * head_dim + d] = bf16_sat(out_vals[d] * inv_sum);
    }
}

// One SIMD group owns a query/head; dimensions are distributed across lanes.

kernel void attention_prefill_simdgroup_f16(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *k_cache [[buffer(1)]],
    device const bfloat *v_cache [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device const int *block_tables [[buffer(4)]],
    device const int *context_lens [[buffer(5)]],
    device const int *cu_seqlens [[buffer(6)]],
    device const int *positions [[buffer(7)]],
    constant uint &total_q [[buffer(8)]],
    constant uint &batch_size [[buffer(9)]],
    constant uint &num_heads [[buffer(10)]],
    constant uint &num_kv_heads [[buffer(11)]],
    constant uint &head_dim [[buffer(12)]],
    constant uint &block_size [[buffer(13)]],
    constant uint &max_blocks [[buffer(14)]],
    constant float &scale [[buffer(15)]],
    constant uint &attention_window [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint q_pos = group.x;
    uint head = group.y;
    if (q_pos >= total_q || head >= num_heads || (head_dim != 256 && head_dim != 512)) return;
    uint seq = batch_size;
    for (uint s = 0; s < batch_size; s++) {
        if (int(q_pos) >= cu_seqlens[s] && int(q_pos) < cu_seqlens[s + 1]) { seq = s; break; }
    }
    if (seq == batch_size || context_lens[seq] <= 0) return;
    uint kv_head = head / (num_heads / num_kv_heads);
    uint q_dim = num_heads * head_dim;
    uint kv_dim = num_kv_heads * head_dim;
    uint attn_len = min(uint(context_lens[seq]), uint(max(positions[q_pos], 0)) + 1u);
    uint attn_start = attention_window == 0 ? 0 : attn_len - min(attn_len, attention_window);
    uint slots = head_dim / 32u;
    float q_lane[16];
    float out_lane[16];
    for (uint slot = 0; slot < slots; slot++) {
        q_lane[slot] = float(q[q_pos * q_dim + head * head_dim + uint(lane) + slot * 32u]);
        out_lane[slot] = 0.0f;
    }
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    for (uint t = attn_start; t < attn_len; t++) {
        int block_id = block_tables[seq * max_blocks + t / block_size];
        if (block_id < 0) continue;
        uint kv_base = uint(block_id) * block_size * kv_dim + (t % block_size) * kv_dim + kv_head * head_dim;
        float partial_score = 0.0f;
        for (uint slot = 0; slot < slots; slot++) {
            uint d = uint(lane) + slot * 32u;
            partial_score += q_lane[slot] * float(k_cache[kv_base + d]);
        }
        float score = simd_sum(partial_score) * scale;
        float next_max = max(max_score, score);
        float correction = exp(max_score - next_max);
        float weight = exp(score - next_max);
        sum_exp = sum_exp * correction + weight;
        for (uint slot = 0; slot < slots; slot++) {
            uint d = uint(lane) + slot * 32u;
            out_lane[slot] = out_lane[slot] * correction + weight * float(v_cache[kv_base + d]);
        }
        max_score = next_max;
    }
    float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
    for (uint slot = 0; slot < slots; slot++) {
        output[q_pos * q_dim + head * head_dim + uint(lane) + slot * 32u] = bf16_sat(out_lane[slot] * inv_sum);
    }
}

// ============================================================================
// GELU(tanh) * up (fused activation for Gemma 4)
// ============================================================================
kernel void gelu_mul_f16(
    device const bfloat *gate_up    [[buffer(0)]],  // [num_tokens, 2*intermediate]
    device bfloat       *output     [[buffer(1)]],  // [num_tokens, intermediate]
    constant uint     &num_tokens [[buffer(2)]],
    constant uint     &intermediate [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= intermediate) return;

    float gate = float(gate_up[token * 2 * intermediate + dim]);
    float up = float(gate_up[token * 2 * intermediate + intermediate + dim]);

    output[token * intermediate + dim] = bf16_sat(gelu_tanh(gate) * up);
}

constant uint MOE_MAX_EXPERTS = 256;
constant uint MOE_MAX_TOP_K = 16;

kernel void moe_router_topk_f16(
    device const bfloat *hidden                 [[buffer(0)]],
    device const bfloat *router_proj           [[buffer(1)]],
    device const bfloat *router_scale          [[buffer(2)]],
    device const bfloat *router_per_expert_scale [[buffer(3)]],
    device int        *topk_indices           [[buffer(4)]],
    device float      *topk_weights           [[buffer(5)]],
    constant uint     &num_tokens             [[buffer(6)]],
    constant uint     &hidden_size            [[buffer(7)]],
    constant uint     &num_experts            [[buffer(8)]],
    constant uint     &top_k                  [[buffer(9)]],
    constant float    &eps                    [[buffer(10)]],
    constant float    &scalar_root_size       [[buffer(11)]],
    uint tid                                  [[thread_index_in_threadgroup]],
    uint tg_size                              [[threads_per_threadgroup]],
    uint token                                [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_sum[256];
    threadgroup float logits[MOE_MAX_EXPERTS];
    threadgroup float probs[MOE_MAX_EXPERTS];
    threadgroup int selected[MOE_MAX_TOP_K];
    threadgroup float selected_probs[MOE_MAX_TOP_K];

    if (token >= num_tokens || num_experts > MOE_MAX_EXPERTS || top_k > MOE_MAX_TOP_K) return;

    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float v = float(hidden[token * hidden_size + i]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared_sum[0] / float(hidden_size) + eps);

    if (tid < MOE_MAX_EXPERTS) {
        logits[tid] = -INFINITY;
        probs[tid] = 0.0f;
    }
    if (tid < MOE_MAX_TOP_K) {
        selected[tid] = -1;
        selected_probs[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_experts) {
        float acc = 0.0f;
        for (uint h = 0; h < hidden_size; h++) {
            float x = float(hidden[token * hidden_size + h]) * rms
                    * router_scale[h] * scalar_root_size;
            acc += x * router_proj[tid * hidden_size + h];
        }
        logits[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    if (tid < num_experts) {
        local_max = logits[tid];
    }
    shared_sum[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] = max(shared_sum[tid], shared_sum[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_logit = shared_sum[0];

    float local_exp = 0.0f;
    if (tid < num_experts) {
        float p = exp(logits[tid] - max_logit);
        probs[tid] = p;
        local_exp = p;
    }
    shared_sum[tid] = local_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = shared_sum[0] > 0.0f ? (1.0f / shared_sum[0]) : 0.0f;
    if (tid < num_experts) {
        probs[tid] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float selected_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            float best = -1.0f;
            int best_idx = -1;
            for (uint e = 0; e < num_experts; e++) {
                bool used = false;
                for (uint prev = 0; prev < k; prev++) {
                    used = used || (selected[prev] == int(e));
                }
                float p = used ? -1.0f : probs[e];
                if (p > best || (p == best && int(e) < best_idx)) {
                    best = p;
                    best_idx = int(e);
                }
            }
            selected[k] = best_idx;
            selected_probs[k] = max(best, 0.0f);
            selected_sum += selected_probs[k];
        }
        float inv_selected = selected_sum > 0.0f ? (1.0f / selected_sum) : 0.0f;
        for (uint k = 0; k < top_k; k++) {
            int expert = selected[k];
            float w = selected_probs[k] * inv_selected;
            if (expert >= 0) {
                w *= router_per_expert_scale[uint(expert)];
            }
            topk_indices[token * top_k + k] = expert;
            topk_weights[token * top_k + k] = w;
        }
    }
}

kernel void moe_expert_gate_up_f16(
    device const bfloat *hidden       [[buffer(0)]],
    device const bfloat *expert_gate_up [[buffer(1)]],
    device const int  *topk_indices [[buffer(2)]],
    device const float *topk_weights [[buffer(3)]],
    device bfloat       *activated    [[buffer(4)]],
    constant uint     &num_tokens   [[buffer(5)]],
    constant uint     &hidden_size  [[buffer(6)]],
    constant uint     &num_experts  [[buffer(7)]],
    constant uint     &top_k        [[buffer(8)]],
    constant uint     &intermediate [[buffer(9)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint route = gid.y;
    uint dim = gid.z;
    if (token >= num_tokens || route >= top_k || dim >= intermediate) return;
    int expert_i = topk_indices[token * top_k + route];
    uint out_idx = (token * top_k + route) * intermediate + dim;
    if (expert_i < 0 || uint(expert_i) >= num_experts) {
        activated[out_idx] = bfloat(0.0f);
        return;
    }
    uint expert = uint(expert_i);
    uint two_intermediate = 2 * intermediate;
    uint gate_base = (expert * two_intermediate + dim) * hidden_size;
    uint up_base = (expert * two_intermediate + intermediate + dim) * hidden_size;
    float gate = 0.0f;
    float up = 0.0f;
    uint hidden_base = token * hidden_size;
    for (uint h = 0; h < hidden_size; h++) {
        float x = float(hidden[hidden_base + h]);
        gate += x * float(expert_gate_up[gate_base + h]);
        up += x * float(expert_gate_up[up_base + h]);
    }
    float weight = topk_weights[token * top_k + route];
    activated[out_idx] = bf16_sat(gelu_tanh(gate) * up * weight);
}

kernel void moe_expert_down_f16(
    device const bfloat *activated    [[buffer(0)]],
    device const bfloat *expert_down  [[buffer(1)]],
    device const int  *topk_indices [[buffer(2)]],
    device bfloat       *output       [[buffer(3)]],
    constant uint     &num_tokens   [[buffer(4)]],
    constant uint     &hidden_size  [[buffer(5)]],
    constant uint     &num_experts  [[buffer(6)]],
    constant uint     &top_k        [[buffer(7)]],
    constant uint     &intermediate [[buffer(8)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint hidden_dim = gid.y;
    if (token >= num_tokens || hidden_dim >= hidden_size) return;

    float acc = 0.0f;
    for (uint route = 0; route < top_k; route++) {
        int expert_i = topk_indices[token * top_k + route];
        if (expert_i < 0 || uint(expert_i) >= num_experts) {
            continue;
        }
        uint expert = uint(expert_i);
        uint act_base = (token * top_k + route) * intermediate;
        uint down_base = (expert * hidden_size + hidden_dim) * intermediate;
        for (uint j = 0; j < intermediate; j++) {
            acc += float(activated[act_base + j]) * float(expert_down[down_base + j]);
        }
    }
    output[token * hidden_size + hidden_dim] = bf16_sat(acc);
}

kernel void ple_combine_f16(
    device bfloat       *token_ple   [[buffer(0)]],
    device const bfloat *context_ple [[buffer(1)]],
    constant uint     &num_tokens  [[buffer(2)]],
    constant uint     &stride      [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= stride) return;
    uint idx = token * stride + dim;
    token_ple[idx] = bf16_sat((float(token_ple[idx]) + float(context_ple[idx])) * 0.70710678118f);
}

kernel void ple_gelu_mul_f16(
    device bfloat       *gate        [[buffer(0)]],
    device const bfloat *packed_ple  [[buffer(1)]],
    constant uint     &num_tokens  [[buffer(2)]],
    constant uint     &num_layers  [[buffer(3)]],
    constant uint     &layer_idx   [[buffer(4)]],
    constant uint     &ple_dim     [[buffer(5)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint token = gid.x;
    uint dim = gid.y;
    if (token >= num_tokens || dim >= ple_dim || layer_idx >= num_layers) return;
    uint gate_idx = token * ple_dim + dim;
    uint ple_idx = token * num_layers * ple_dim + layer_idx * ple_dim + dim;
    gate[gate_idx] = bf16_sat(gelu_tanh(float(gate[gate_idx])) * float(packed_ple[ple_idx]));
}

// ============================================================================
// Residual Add
// ============================================================================
kernel void residual_add_f16(
    device bfloat       *residual   [[buffer(0)]],
    device const bfloat *addition   [[buffer(1)]],
    constant uint     &count      [[buffer(2)]],
    constant uint     &hidden     [[buffer(3)]],
    device const bfloat *layer_scale [[buffer(4)]],
    constant uint     &layer_scale_dim [[buffer(5)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float scale = 1.0f;
    if (layer_scale_dim == 1) {
        scale = float(layer_scale[0]);
    } else if (layer_scale_dim == hidden) {
        scale = float(layer_scale[gid % hidden]);
    }
    residual[gid] = bf16_sat(float(residual[gid]) + float(addition[gid]) * scale);
}

kernel void residual_add_then_scale_f16(
    device bfloat       *residual   [[buffer(0)]],
    device const bfloat *addition   [[buffer(1)]],
    constant uint     &count      [[buffer(2)]],
    constant uint     &hidden     [[buffer(3)]],
    device const bfloat *layer_scale [[buffer(4)]],
    constant uint     &layer_scale_dim [[buffer(5)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float scale = 1.0f;
    if (layer_scale_dim == 1) {
        scale = float(layer_scale[0]);
    } else if (layer_scale_dim == hidden) {
        scale = float(layer_scale[gid % hidden]);
    }
    bfloat updated = bf16_sat(float(residual[gid]) + float(addition[gid]));
    residual[gid] = bf16_sat(float(updated) * scale);
}

kernel void residual_add_rmsnorm_f16(
    device bfloat       *residual   [[buffer(0)]],
    device const bfloat *addition   [[buffer(1)]],
    device bfloat       *output     [[buffer(2)]],
    device const bfloat *gamma      [[buffer(3)]],
    constant uint     &hidden     [[buffer(4)]],
    constant float    &eps        [[buffer(5)]],
    device const bfloat *layer_scale [[buffer(6)]],
    constant uint     &layer_scale_dim [[buffer(7)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint token = gid;
    uint base = token * hidden;

    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        uint idx = base + i;
        float scale = 1.0f;
        if (layer_scale_dim == 1) {
            scale = float(layer_scale[0]);
        } else if (layer_scale_dim == hidden) {
            scale = float(layer_scale[i]);
        }
        bfloat updated = bf16_sat(float(residual[idx]) + float(addition[idx]) * scale);
        residual[idx] = updated;
        float v = float(updated);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(hidden) + eps);
    for (uint i = tid; i < hidden; i += tg_size) {
        uint idx = base + i;
        output[idx] = bf16_sat(float(residual[idx]) * rms * float(gamma[i]));
    }
}

kernel void layer_scale_f16(
    device bfloat       *x          [[buffer(0)]],
    device const bfloat *scale      [[buffer(1)]],
    constant uint     &count      [[buffer(2)]],
    constant uint     &hidden     [[buffer(3)]],
    constant uint     &scale_dim  [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float s = 1.0f;
    if (scale_dim == 1) {
        s = float(scale[0]);
    } else if (scale_dim == hidden) {
        s = float(scale[gid % hidden]);
    }
    x[gid] = bf16_sat(float(x[gid]) * s);
}

// ============================================================================
// Argmax (per-sequence)
// ============================================================================
static inline bool argmax_better(float candidate_val, int candidate_idx, float best_val, int best_idx) {
    return candidate_val > best_val || (candidate_val == best_val && candidate_idx < best_idx);
}

kernel void argmax_f16(
    device const bfloat *logits     [[buffer(0)]],  // [num_seqs, vocab]
    device int        *output     [[buffer(1)]],  // [num_seqs]
    constant uint     &num_seqs   [[buffer(2)]],
    constant uint     &vocab      [[buffer(3)]],
    uint gid                      [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    if (gid >= num_seqs) return;
    uint base = gid * vocab;

    threadgroup float shared_max[256];
    threadgroup int shared_idx[256];

    float local_max = -INFINITY;
    int local_idx = 0;
    for (uint i = tid; i < vocab; i += tg_size) {
        float v = float(logits[base + i]);
        if (argmax_better(v, int(i), local_max, local_idx)) {
            local_max = v;
            local_idx = int(i);
        }
    }
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && argmax_better(shared_max[tid + s], shared_idx[tid + s], shared_max[tid], shared_idx[tid])) {
            shared_max[tid] = shared_max[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[gid] = shared_idx[0];
    }
}

kernel void softcap_argmax_f16(
    device bfloat       *logits     [[buffer(0)]],  // [num_seqs, vocab]
    device int        *output     [[buffer(1)]],  // [num_seqs]
    constant uint     &num_seqs   [[buffer(2)]],
    constant uint     &vocab      [[buffer(3)]],
    constant float    &cap        [[buffer(4)]],
    uint gid                      [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    if (gid >= num_seqs) return;
    uint base = gid * vocab;

    threadgroup float shared_max[256];
    threadgroup int shared_idx[256];

    float local_max = -INFINITY;
    int local_idx = 0;
    for (uint i = tid; i < vocab; i += tg_size) {
        bfloat capped = bf16_sat(cap * tanh(float(logits[base + i]) / cap));
        logits[base + i] = capped;
        float v = float(capped);
        if (argmax_better(v, int(i), local_max, local_idx)) {
            local_max = v;
            local_idx = int(i);
        }
    }
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && argmax_better(shared_max[tid + s], shared_idx[tid + s], shared_max[tid], shared_idx[tid])) {
            shared_max[tid] = shared_max[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[gid] = shared_idx[0];
    }
}

// ============================================================================
// Fused final RMSNorm + lm_head + optional softcap + argmax
// ============================================================================
// Bounded small-vocab probe path. The unfused RMSNorm/GEMM/softcap/argmax path
// remains the fallback for wider shapes.
kernel void final_norm_lm_head_argmax_small_f16(
    device const bfloat *residual   [[buffer(0)]],
    device const bfloat *gamma      [[buffer(1)]],
    device const bfloat *lm_head    [[buffer(2)]],
    device bfloat       *logits     [[buffer(3)]],
    device int        *output     [[buffer(4)]],
    constant uint     &hidden     [[buffer(5)]],
    constant uint     &vocab      [[buffer(6)]],
    constant float    &eps        [[buffer(7)]],
    constant float    &softcap    [[buffer(8)]],
    uint token                    [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_max[256];
    threadgroup int shared_idx[256];

    uint base = token * hidden;
    float local_sum = 0.0f;
    for (uint d = tid; d < hidden; d += tg_size) {
        float v = float(residual[base + d]);
        local_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared_sum[0] / float(hidden) + eps);

    float local_max = -INFINITY;
    int local_idx = 0;
    for (uint v = tid; v < vocab; v += tg_size) {
        float acc = 0.0f;
        for (uint d = 0; d < hidden; d++) {
            float normed = float(residual[base + d]) * inv_rms * float(gamma[d]);
            acc += normed * float(lm_head[v * hidden + d]);
        }
        if (softcap > 0.0f) {
            acc = softcap * tanh(acc / softcap);
        }
        logits[token * vocab + v] = bf16_sat(acc);
        if (argmax_better(acc, int(v), local_max, local_idx)) {
            local_max = acc;
            local_idx = int(v);
        }
    }
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && argmax_better(shared_max[tid + s], shared_idx[tid + s], shared_max[tid], shared_idx[tid])) {
            shared_max[tid] = shared_max[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[token] = shared_idx[0];
    }
}

kernel void final_lm_head_argmax_tiles_f16(
    device const bfloat *normed_hidden [[buffer(0)]],
    device const bfloat *lm_head       [[buffer(1)]],
    device float      *partial_max   [[buffer(2)]],
    device int        *partial_idx   [[buffer(3)]],
    constant uint     &num_tokens    [[buffer(4)]],
    constant uint     &vocab         [[buffer(5)]],
    constant uint     &hidden        [[buffer(6)]],
    constant uint     &tile_count    [[buffer(7)]],
    constant float    &softcap       [[buffer(8)]],
    uint2 gid                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]]
) {
    uint token = gid.x;
    uint tile = gid.y;
    if (token >= num_tokens || tile >= tile_count) return;

    uint col_base = tile * FINAL_ARGMAX_TILE_N;
    threadgroup float partial[8][256];
    threadgroup float tile_vals[8];
    threadgroup int tile_indices[8];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    uint hidden_base = token * hidden;
    for (uint k = tid; k < hidden; k += GEMV_TG) {
        float a = float(normed_hidden[hidden_base + k]);
        uint col = col_base;
        if (col < vocab) acc0 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc1 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc2 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc3 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc4 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc5 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc6 += a * float(lm_head[col * hidden + k]);
        col++;
        if (col < vocab) acc7 += a * float(lm_head[col * hidden + k]);
    }

    partial[0][tid] = acc0;
    partial[1][tid] = acc1;
    partial[2][tid] = acc2;
    partial[3][tid] = acc3;
    partial[4][tid] = acc4;
    partial[5][tid] = acc5;
    partial[6][tid] = acc6;
    partial[7][tid] = acc7;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GEMV_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[0][tid] += partial[0][tid + stride];
            partial[1][tid] += partial[1][tid + stride];
            partial[2][tid] += partial[2][tid + stride];
            partial[3][tid] += partial[3][tid + stride];
            partial[4][tid] += partial[4][tid + stride];
            partial[5][tid] += partial[5][tid + stride];
            partial[6][tid] += partial[6][tid + stride];
            partial[7][tid] += partial[7][tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < FINAL_ARGMAX_TILE_N) {
        uint col = col_base + tid;
        float score = -INFINITY;
        int idx = 0;
        if (col < vocab) {
            bfloat quantized = bf16_sat(partial[tid][0]);
            score = float(quantized);
            if (softcap > 0.0f) {
                score = float(bf16_sat(softcap * tanh(score / softcap)));
            }
            idx = int(col);
        }
        tile_vals[tid] = score;
        tile_indices[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float best_val = tile_vals[0];
        int best_idx = tile_indices[0];
        for (uint i = 1; i < FINAL_ARGMAX_TILE_N; i++) {
            if (argmax_better(tile_vals[i], tile_indices[i], best_val, best_idx)) {
                best_val = tile_vals[i];
                best_idx = tile_indices[i];
            }
        }
        uint out = token * tile_count + tile;
        partial_max[out] = best_val;
        partial_idx[out] = best_idx;
    }
}

// Apple9/10 LM-head projection for 2 <= num_tokens <= 8. One threadgroup
// covers eight vocabulary columns for the entire request microbatch, sharing
// each `[vocab,hidden]` weight read across all request rows. It emits the same
// per-token/per-tile partial argmax ABI consumed by final_argmax_reduce_f32.
kernel void final_lm_head_argmax_tiles_batch8_f16(
    device const bfloat *normed_hidden [[buffer(0)]],
    device const bfloat *lm_head       [[buffer(1)]],
    device float      *partial_max   [[buffer(2)]],
    device int        *partial_idx   [[buffer(3)]],
    constant uint     &num_tokens    [[buffer(4)]],
    constant uint     &vocab         [[buffer(5)]],
    constant uint     &hidden        [[buffer(6)]],
    constant uint     &tile_count    [[buffer(7)]],
    constant float    &softcap       [[buffer(8)]],
    uint tile                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    ushort lane                       [[thread_index_in_simdgroup]],
    ushort simdgroup                  [[simdgroup_index_in_threadgroup]]
) {
    if (tile >= tile_count) return;
    uint col = tile * FINAL_ARGMAX_TILE_N + uint(simdgroup);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    if (col < vocab) {
        for (uint k = uint(lane); k < hidden; k += 32u) {
            float b = float(lm_head[col * hidden + k]);
            if (0u < num_tokens) acc0 += float(normed_hidden[0u * hidden + k]) * b;
            if (1u < num_tokens) acc1 += float(normed_hidden[1u * hidden + k]) * b;
            if (2u < num_tokens) acc2 += float(normed_hidden[2u * hidden + k]) * b;
            if (3u < num_tokens) acc3 += float(normed_hidden[3u * hidden + k]) * b;
            if (4u < num_tokens) acc4 += float(normed_hidden[4u * hidden + k]) * b;
            if (5u < num_tokens) acc5 += float(normed_hidden[5u * hidden + k]) * b;
            if (6u < num_tokens) acc6 += float(normed_hidden[6u * hidden + k]) * b;
            if (7u < num_tokens) acc7 += float(normed_hidden[7u * hidden + k]) * b;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    threadgroup float tile_vals[8][8];
    threadgroup int tile_indices[8][8];
    if (lane == 0) {
        float totals[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        for (uint token = 0; token < 8u; token++) {
            float score = -INFINITY;
            int idx = 0;
            if (token < num_tokens && col < vocab) {
                bfloat quantized = bf16_sat(totals[token]);
                score = float(quantized);
                if (softcap > 0.0f) {
                    score = float(bf16_sat(softcap * tanh(score / softcap)));
                }
                idx = int(col);
            }
            tile_vals[token][simdgroup] = score;
            tile_indices[token][simdgroup] = idx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_tokens) {
        uint token = tid;
        float best_val = tile_vals[token][0];
        int best_idx = tile_indices[token][0];
        for (uint i = 1; i < FINAL_ARGMAX_TILE_N; i++) {
            if (argmax_better(tile_vals[token][i], tile_indices[token][i], best_val, best_idx)) {
                best_val = tile_vals[token][i];
                best_idx = tile_indices[token][i];
            }
        }
        uint out = token * tile_count + tile;
        partial_max[out] = best_val;
        partial_idx[out] = best_idx;
    }
}

// Batch-four LM-head projection. Like the qualified batch-eight route, this
// shares each vocabulary weight row across the whole cohort, but keeps only
// four request accumulators live to avoid the register pressure of the M8
// kernel on a bfloat-full batch.
kernel void final_lm_head_argmax_tiles_batch4_f16(
    device const bfloat *normed_hidden [[buffer(0)]],
    device const bfloat *lm_head       [[buffer(1)]],
    device float      *partial_max   [[buffer(2)]],
    device int        *partial_idx   [[buffer(3)]],
    constant uint     &num_tokens    [[buffer(4)]],
    constant uint     &vocab         [[buffer(5)]],
    constant uint     &hidden        [[buffer(6)]],
    constant uint     &tile_count    [[buffer(7)]],
    constant float    &softcap       [[buffer(8)]],
    uint tile                         [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    ushort lane                       [[thread_index_in_simdgroup]],
    ushort simdgroup                  [[simdgroup_index_in_threadgroup]]
) {
    if (tile >= tile_count) return;
    uint col = tile * FINAL_ARGMAX_TILE_N + uint(simdgroup);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    if (col < vocab) {
        for (uint k = uint(lane); k < hidden; k += 32u) {
            float weight = float(lm_head[col * hidden + k]);
            acc0 += float(normed_hidden[0u * hidden + k]) * weight;
            acc1 += float(normed_hidden[1u * hidden + k]) * weight;
            acc2 += float(normed_hidden[2u * hidden + k]) * weight;
            acc3 += float(normed_hidden[3u * hidden + k]) * weight;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    threadgroup float tile_vals[4][8];
    threadgroup int tile_indices[4][8];
    if (lane == 0) {
        float totals[4] = {acc0, acc1, acc2, acc3};
        for (uint token = 0; token < 4u; token++) {
            float score = -INFINITY;
            int idx = 0;
            if (col < vocab) {
                bfloat quantized = bf16_sat(totals[token]);
                score = float(quantized);
                if (softcap > 0.0f) {
                    score = float(bf16_sat(softcap * tanh(score / softcap)));
                }
                idx = int(col);
            }
            tile_vals[token][simdgroup] = score;
            tile_indices[token][simdgroup] = idx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 4u) {
        uint token = tid;
        float best_val = tile_vals[token][0];
        int best_idx = tile_indices[token][0];
        for (uint i = 1; i < FINAL_ARGMAX_TILE_N; i++) {
            if (argmax_better(tile_vals[token][i], tile_indices[token][i], best_val, best_idx)) {
                best_val = tile_vals[token][i];
                best_idx = tile_indices[token][i];
            }
        }
        uint out = token * tile_count + tile;
        partial_max[out] = best_val;
        partial_idx[out] = best_idx;
    }
}

kernel void final_argmax_reduce_f32(
    device const float *partial_max [[buffer(0)]],
    device const int   *partial_idx [[buffer(1)]],
    device int         *output      [[buffer(2)]],
    constant uint      &num_tokens  [[buffer(3)]],
    constant uint      &tile_count  [[buffer(4)]],
    uint token                       [[threadgroup_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]]
) {
    if (token >= num_tokens) return;

    threadgroup float shared_max[256];
    threadgroup int shared_idx[256];

    float local_max = -INFINITY;
    int local_idx = 0;
    uint base = token * tile_count;
    for (uint tile = tid; tile < tile_count; tile += tg_size) {
        float val = partial_max[base + tile];
        int idx = partial_idx[base + tile];
        if (argmax_better(val, idx, local_max, local_idx)) {
            local_max = val;
            local_idx = idx;
        }
    }
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s && argmax_better(shared_max[tid + s], shared_idx[tid + s], shared_max[tid], shared_idx[tid])) {
            shared_max[tid] = shared_max[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[token] = shared_idx[0];
    }
}

// ============================================================================
// Logit Softcap: 30 * tanh(logits / 30) for Gemma 4
// ============================================================================
kernel void softcap_f16(
    device bfloat       *logits     [[buffer(0)]],
    constant uint     &count      [[buffer(1)]],
    constant float    &cap        [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float x = float(logits[gid]);
    logits[gid] = bf16_sat(cap * tanh(x / cap));
}

// ============================================================================
// BF16 → F16 conversion (for weight loading)
// ============================================================================
kernel void bf16_to_f16(
    device const ushort *bf16_in  [[buffer(0)]],
    device bfloat         *f16_out  [[buffer(1)]],
    constant uint       &count    [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    // BF16 to F32: shift left 16 bits (BF16 is truncated F32)
    uint bf16_bits = uint(bf16_in[gid]);
    uint f32_bits = bf16_bits << 16;
    float f32_val = as_type<float>(f32_bits);
    f16_out[gid] = bf16_sat(f32_val);
}

// Explicit storage ABI: do not rewrite half to bfloat in this source.
inline float round_decode_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16u);
}
inline ushort round_decode_rne(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort(bits >> 16u) | ushort(0x40u);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
// Fixed tree, no dependence on implementation-specific simd_sum association.
inline float round_decode_sum32(float x) {
    for (ushort delta = 16; delta > 0; delta /= 2) x += simd_shuffle_down(x, delta);
    return simd_broadcast(x, 0);
}
inline float round_decode_gelu(float x) {
    if (x >= 5.0f) return x;
    if (x <= -5.0f) return 0.0f;
    return 0.5f * x * (1.0f + tanh(0.7978845608f *
        (x + 0.044715f * x * x * x)));
}
// Donor-inspired decode schedule on rvLLM's authenticated signed W4/G32 ABI.
// Four rows per SIMDgroup, eight SIMDgroups per threadgroup, eight adjacent K
// values per lane. A packed uint and one FP16 scale feed each eight-value run.
// This does not import the donor's affine G64 storage or activation quantizer.
#pragma METAL fp math_mode(safe)
kernel void research_qmv_w4_g32_r4_sg8_k8(
    device const ushort *a [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const half *scales [[buffer(2)]], device ushort *out [[buffer(3)]],
    constant uint &m [[buffer(4)]], constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]], constant uint &stride [[buffer(7)]],
    constant uint &column [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]]) {
    if (any(threads != uint3(256,1,1)) || m != 1u || n != 3840u || k != 15360u
        || stride < n || column > stride - n || group.y != 0u || group.z != 0u
        || group.x >= 120u) return;
    uint row0 = group.x * 32u + uint(sg) * 4u;
    uint groups = k / 32u;
    uint words_per_row = k / 8u;
    device const uint *words = reinterpret_cast<device const uint *>(w);
    float accum[4] = {0,0,0,0};
    for (uint kb = 0; kb < k; kb += 256u) {
        uint k0 = kb + uint(lane) * 8u;
        float av[8];
        #pragma unroll
        for (uint j = 0; j < 8u; ++j) av[j] = round_decode_bf16(a[k0+j]);
        uint g = k0 / 32u;
        uint word = k0 / 8u;
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            uint row = row0 + r;
            uint packed = words[size_t(row)*words_per_row+word];
            float scale = float(scales[size_t(row)*groups+g]);
            float part = 0.0f;
            #pragma unroll
            for (uint j = 0; j < 8u; ++j) {
                int q = int((packed >> (j*4u)) & 15u);
                q = q >= 8 ? q - 16 : q;
                part = fma(av[j], float(q)*scale, part);
            }
            accum[r] += part;
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        float total = round_decode_sum32(accum[r]);
        if (lane == 0) out[column+row0+r] = round_decode_rne(total);
    }
}
